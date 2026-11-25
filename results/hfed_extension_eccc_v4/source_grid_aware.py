"""Frozen graph early-fusion recurrent model and its seasonal residual extension.

Architecture source: Orji et al., Grid-Aware Spatio-Temporal Graph Neural Networks
for Multi-Horizon Load Forecasting. The graph cache is registered in checkpoints.
Graph attributes are explicit inputs, so HFED never needs invented capacities.
"""
from __future__ import annotations

import torch
from torch import nn
import numpy as np


class GridAware(nn.Module):
    def __init__(self, data, backbone="lstm"):
        super().__init__()
        from torch_geometric.nn import GATv2Conv

        x = torch.tensor(data["node_x"], dtype=torch.float32)
        edge = torch.tensor(data["edge_index"], dtype=torch.long)
        attr = torch.tensor(data["edge_features"], dtype=torch.float32)
        kwargs = dict(heads=2, concat=False, edge_dim=attr.shape[1])
        first = GATv2Conv(x.shape[1], 64, **kwargs)
        second = GATv2Conv(x.shape[1], 64, **kwargs)
        third = GATv2Conv(64, 64, **kwargs)
        projection = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 32))
        nn.init.xavier_uniform_(projection[1].weight)
        nn.init.zeros_(projection[1].bias)
        with torch.no_grad():
            embedding = projection(torch.cat([first(x, edge, attr),
                                             third(second(x, edge, attr), edge, attr)], -1))
        self.register_buffer("embedding", embedding)
        hidden = 256 if backbone == "lstm" else 128
        recurrent = nn.LSTM if backbone == "lstm" else nn.GRU
        self.temporal = recurrent(len(data["features"]) + 32, hidden, batch_first=True)
        self.out = nn.Linear(hidden, len(data.get("horizons", [1, 6, 24])))
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def encode_samples(self, seq, nodes):
        emb = self.embedding[nodes, None].expand(-1, seq.shape[1], -1)
        encoded, _ = self.temporal(torch.cat([seq, emb], -1))
        return encoded[:, -1]

    def forward_samples(self, seq, nodes):
        return self.out(self.encode_samples(seq, nodes))

    def forward(self, batch):
        seq = batch["dynamic"]
        b, t, n, f = seq.shape
        seq = seq.permute(0, 2, 1, 3).reshape(b * n, t, f)
        nodes = torch.arange(n, device=seq.device).repeat(b)
        return self.forward_samples(seq, nodes).reshape(b, n, -1)


class SeasonalGridAware(nn.Module):
    """Keep the paper's frozen GAT/GRU; learn corrections to causal seasonal load.

    Covariates are seven historical daily lags, their observation masks, and
    deterministic target calendars. Output scaling is fitted per region.
    Identity-graph control changes only the supplied graph, not model capacity.
    """
    def __init__(self, data, calendar_size=26):
        super().__init__()
        self.backbone = GridAware(data, backbone="gru")
        self.backbone.out = nn.Identity()
        self.head = nn.Sequential(nn.Linear(128 + calendar_size, 64), nn.GELU(),
                                  nn.Dropout(0.1404), nn.Linear(64, 1))

    def forward_samples(self, seq, nodes, covariates, anchor):
        context = self.backbone.encode_samples(seq, nodes)
        context = context[:, None].expand(-1, covariates.shape[1], -1)
        return anchor + self.head(torch.cat([context, covariates], -1)).squeeze(-1)

    def forward(self, batch):
        seq = batch["dynamic"]
        b, t, n, f = seq.shape
        seq = seq.permute(0, 2, 1, 3).reshape(b * n, t, f)
        nodes = torch.arange(n, device=seq.device).repeat(b)
        cov = batch["covariates"].reshape(b * n, -1, batch["covariates"].shape[-1])
        anchor = batch["anchor"].reshape(b * n, -1)
        return self.forward_samples(seq, nodes, cov, anchor).reshape(b, n, -1)


class TemporalResidual(nn.Module):
    """Own-history form of the existing identity ResidualSTGCN, for HFED retraining."""
    def __init__(self, data, calendar_size=26):
        super().__init__()
        from src.models.stgnn.residual import ResidualSTGCN
        self.core = ResidualSTGCN(np.eye(len(data["states"])), calendar_size)
        load_name = "load" if "load" in data["features"] else "demand_mw"
        self.load_index = data["features"].index(load_name)
        self.observed_index = (data["features"].index("demand_mw_observed")
                               if "demand_mw_observed" in data["features"] else None)
        for name in ["center", "scale", "feature_center", "feature_scale"]:
            self.register_buffer(name, torch.as_tensor(data[name], dtype=torch.float32))

    def forward_samples(self, seq, nodes, covariates, anchor):
        observed = ((seq[:, :, self.observed_index] * self.feature_scale[self.observed_index]
                     + self.feature_center[self.observed_index]).clamp(0, 1)
                    if self.observed_index is not None else torch.ones_like(seq[:, :, self.load_index]))
        raw = seq[:, :, self.load_index] * self.feature_scale[self.load_index] + self.feature_center[self.load_index]
        normalized = ((raw - self.center[nodes, None]) / self.scale[nodes, None]) * observed
        history = torch.stack([normalized, observed], -1)
        own = self.core.temporal(history.permute(0, 2, 1))
        context = torch.cat([own, own, self.core.region(nodes)], -1)
        context = context[:, None].expand(-1, covariates.shape[1], -1)
        return anchor + self.core.head(torch.cat([context, covariates], -1)).squeeze(-1)
