"""Spatio-Temporal Graph Convolutional Network (STGCN) implementation.

Architecture follows the STGCN paper:
  Yu et al. (2018) "Spatio-Temporal Graph Convolutional Networks:
  A Deep Learning Framework for Traffic Forecasting"
  https://arxiv.org/abs/1709.04875

Adapted for electricity demand forecasting across Canadian regions.

Structure: ST-Block × num_blocks → Output layer
Each ST-Block: Temporal Conv → Graph Conv → Temporal Conv → Layer Norm

The adjacency matrix A (N×N, pre-normalised) is passed once at construction
and registered as a fixed buffer (not a learnable parameter).  Swapping A
for a different graph configuration (identity, geographic, correlation,
interconnection) produces the graph ablation experiments.

Input shape:  (batch, T, N, F)
   T = lookback timesteps
   N = number of regions (graph nodes)
   F = number of features per node

Output shape: (batch, N, H)
   H = forecast horizon (24 hours)
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not installed.  STGCN unavailable.")


if TORCH_AVAILABLE:

    class TemporalConvBlock(nn.Module):
        """Gated temporal convolution (Γ-conv from the STGCN paper).

        Uses a 1-D causal convolution along the time axis.
        Output = tanh(Θ₁ ⊗ X) ⊙ σ(Θ₂ ⊗ X)
        """

        def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
            super().__init__()
            pad = kernel_size - 1  # causal: pad left only
            # 2× out_channels for gating
            self.conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels * 2,
                kernel_size=(kernel_size, 1),
                padding=(pad, 0),
            )
            self.out_channels = out_channels
            self.kernel_size = kernel_size

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B, C, T, N)
            out = self.conv(x)
            # Remove the extra padding on the right (causal)
            out = out[:, :, : x.size(2), :]
            gate, signal = out[:, : self.out_channels], out[:, self.out_channels :]
            return torch.tanh(signal) * torch.sigmoid(gate)


    class GraphConvLayer(nn.Module):
        """Spectral graph convolution using a fixed normalised adjacency matrix.

        Θ_g ⊗ (Â X) where Â is the pre-computed normalised adjacency.
        """

        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            self.fc = nn.Linear(in_channels, out_channels)

        def forward(
            self, x: torch.Tensor, adj: torch.Tensor
        ) -> torch.Tensor:
            # x:   (B, C, T, N)
            # adj: (N, N)
            B, C, T, N = x.shape
            # Reshape to (B*T, N, C) for batch matmul
            x_perm = x.permute(0, 2, 3, 1).reshape(B * T, N, C)
            # Spatial aggregation: (B*T, N, C) @ (N, N)^T = (B*T, N, C)
            x_agg = torch.matmul(adj.unsqueeze(0), x_perm)
            # Linear projection: (B*T, N, out_channels)
            x_out = self.fc(x_agg)
            # Reshape back: (B, C_out, T, N)
            x_out = x_out.reshape(B, T, N, -1).permute(0, 3, 1, 2)
            return F.relu(x_out)


    class STBlock(nn.Module):
        """Single Spatio-Temporal block: TConv → GraphConv → TConv → LayerNorm."""

        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
        ):
            super().__init__()
            self.temporal1 = TemporalConvBlock(in_channels, out_channels, kernel_size)
            self.graph_conv = GraphConvLayer(out_channels, out_channels)
            self.temporal2 = TemporalConvBlock(out_channels, out_channels, kernel_size)
            self.layer_norm = nn.LayerNorm(out_channels)

            # Residual projection if channel dims differ
            self.residual = (
                nn.Conv2d(in_channels, out_channels, kernel_size=1)
                if in_channels != out_channels
                else nn.Identity()
            )

        def forward(
            self, x: torch.Tensor, adj: torch.Tensor
        ) -> torch.Tensor:
            # x: (B, C_in, T, N)
            residual = self.residual(x)
            out = self.temporal1(x)
            out = self.graph_conv(out, adj)
            out = self.temporal2(out)
            # Add residual (truncate T if necessary)
            t = min(out.size(2), residual.size(2))
            out = out[:, :, :t, :] + residual[:, :, :t, :]
            # LayerNorm over channel dim: reshape to (B, T, N, C), norm, back
            out = out.permute(0, 2, 3, 1)      # (B, T, N, C)
            out = self.layer_norm(out)
            out = out.permute(0, 3, 1, 2)      # (B, C, T, N)
            return out


    class OutputLayer(nn.Module):
        """Final layer: temporal conv → linear over nodes → 24-hour output."""

        def __init__(self, in_channels: int, n_nodes: int, horizon: int = 24):
            super().__init__()
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1))
            self.fc = nn.Linear(in_channels, horizon)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B, C, T, N)
            out = F.relu(self.conv(x))
            # Global average over time dimension
            out = out.mean(dim=2)        # (B, C, N)
            out = out.permute(0, 2, 1)   # (B, N, C)
            out = self.fc(out)           # (B, N, horizon)
            return out


    class STGCN(nn.Module):
        """Full STGCN model.

        Parameters
        ----------
        n_nodes : Number of regions (graph nodes).
        in_features : Number of input features per node per timestep.
        horizon : Forecast horizon (24 for day-ahead hourly).
        num_blocks : Number of ST-blocks.
        channels : List of channel dimensions for each ST-block.
        kernel_size : Temporal convolution kernel size.
        dropout : Dropout probability.
        adj : Pre-normalised adjacency matrix (N×N float32 numpy array).
        """

        def __init__(
            self,
            n_nodes: int,
            in_features: int,
            horizon: int = 24,
            num_blocks: int = 3,
            channels: list[int] | None = None,
            kernel_size: int = 3,
            dropout: float = 0.1,
            adj: np.ndarray | None = None,
        ):
            super().__init__()
            if channels is None:
                channels = [64, 64, 64]
            assert len(channels) == num_blocks, "len(channels) must equal num_blocks"

            self.n_nodes = n_nodes
            self.in_features = in_features
            self.horizon = horizon

            # Input projection: in_features → channels[0]
            self.input_proj = nn.Conv2d(in_features, channels[0], kernel_size=1)

            # ST blocks
            self.st_blocks = nn.ModuleList()
            for i in range(num_blocks):
                in_c = channels[i]
                out_c = channels[i]
                self.st_blocks.append(STBlock(in_c, out_c, kernel_size))

            self.dropout = nn.Dropout(dropout)
            self.output_layer = OutputLayer(channels[-1], n_nodes, horizon)

            # Register adjacency as a buffer (moves with .to(device))
            if adj is not None:
                self.register_buffer(
                    "adj", torch.tensor(adj, dtype=torch.float32)
                )
            else:
                self.register_buffer("adj", torch.eye(n_nodes, dtype=torch.float32))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Parameters
            ----------
            x : (B, T, N, F) — batch, time, nodes, features

            Returns
            -------
            (B, N, H) — batch, nodes, horizon
            """
            # Reorder to (B, F, T, N) for Conv2d
            x = x.permute(0, 3, 1, 2)  # (B, F, T, N)
            x = self.input_proj(x)       # (B, channels[0], T, N)

            for block in self.st_blocks:
                x = block(x, self.adj)
                x = self.dropout(x)

            out = self.output_layer(x)   # (B, N, H)
            return out

        def swap_adjacency(self, adj: np.ndarray) -> None:
            """Replace the adjacency matrix buffer (for graph ablation)."""
            new_adj = torch.tensor(adj, dtype=torch.float32)
            if hasattr(self, "adj"):
                new_adj = new_adj.to(self.adj.device)
            self.register_buffer("adj", new_adj)
