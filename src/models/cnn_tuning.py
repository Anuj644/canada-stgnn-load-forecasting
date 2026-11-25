"""Residual CNN variants and the training loop used to tune the selected HFED model.

The baseline reproduces ``TemporalResidual`` (grid_aware.py) layer for layer: a small Conv1d
encoder over the last 24 hours of demand, region embedding, seven daily lags and known
calendars, and an MLP head that predicts a correction to the latest known same-hour demand.
Variants only change the encoder inputs, width or the training weights. They keep the
serving interface (``dynamic``, ``covariates``, ``anchor``, ``nodes``), so a selected
variant can be served by the dashboard without changing how requests are built.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch
from torch import nn

WEATHER = ["temperature_c", "relative_humidity", "wind_speed"]


class ResidualCNN(nn.Module):
    def __init__(self, data, width=24, hidden=32, head=64, dropout=0.1, weather=False):
        super().__init__()
        names = data["features"]
        regions = len(data["states"])
        covariates = int(np.asarray(data["covariates"]).shape[-1]) if "covariates" in data else 26
        self.load_index = names.index("demand_mw")
        self.observed_index = names.index("demand_mw_observed")
        self.weather_index = ([names.index(name) for name in WEATHER]
                              + [names.index(f"{name}_observed") for name in WEATHER]) if weather else []
        self.temporal = nn.Sequential(
            nn.Conv1d(2 + len(self.weather_index), width, 5, padding=2), nn.GELU(),
            nn.Conv1d(width, width, 5, padding=2), nn.GELU(),
            nn.AdaptiveAvgPool1d(6), nn.Flatten(), nn.Linear(width * 6, hidden),
        )
        self.region = nn.Embedding(regions, 8)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2 + 8 + covariates, head), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(head, head // 2), nn.GELU(), nn.Linear(head // 2, 1),
        )
        for name in ["center", "scale", "feature_center", "feature_scale"]:
            self.register_buffer(name, torch.as_tensor(data[name], dtype=torch.float32))

    def forward_samples(self, seq, nodes, covariates, anchor):
        observed = (seq[:, :, self.observed_index] * self.feature_scale[self.observed_index]
                    + self.feature_center[self.observed_index]).clamp(0, 1)
        raw = seq[:, :, self.load_index] * self.feature_scale[self.load_index] + self.feature_center[self.load_index]
        normalized = ((raw - self.center[nodes, None]) / self.scale[nodes, None]) * observed
        channels = [normalized, observed]
        channels += [seq[:, :, index] for index in self.weather_index]
        own = self.temporal(torch.stack(channels, -1).permute(0, 2, 1))
        context = torch.cat([own, own, self.region(nodes)], -1)
        context = context[:, None].expand(-1, covariates.shape[1], -1)
        return anchor + self.head(torch.cat([context, covariates], -1)).squeeze(-1)


def make_tensors(data, device):
    tensors = {key: torch.tensor(data[key], device=device, dtype=torch.long if key == "nodes" else torch.float32)
               for key in ["dynamic", "covariates", "anchor", "actual", "nodes"]}
    tensors["center"] = torch.tensor(data["center"], device=device, dtype=torch.float32)
    tensors["scale"] = torch.tensor(data["scale"], device=device, dtype=torch.float32)
    return tensors


def sample_weights(tensors, train_ids, power):
    """Weight each sample by (province scale)^power, normalised to mean 1 over the training set."""
    scale = tensors["scale"][tensors["nodes"]] ** power
    return scale / scale[train_ids].mean()


def batch_loss(model, tensors, ids, weights=None):
    nodes = tensors["nodes"][ids]
    actual = tensors["actual"][ids]
    valid = actual.isfinite()
    centre, scale = tensors["center"][nodes, None], tensors["scale"][nodes, None]
    prediction = model.forward_samples(tensors["dynamic"][ids], nodes, tensors["covariates"][ids],
                                       tensors["anchor"][ids])
    target = (actual.nan_to_num() - centre) / scale
    elementwise = nn.functional.smooth_l1_loss(prediction, target, beta=0.25, reduction="none")
    weight = valid.float() if weights is None else valid.float() * weights[ids, None]
    return (elementwise * weight).sum() / valid.sum(), prediction * scale + centre, valid


@torch.no_grad()
def predict(model, tensors, ids, batch_size=512):
    model.eval()
    outputs = []
    for offset in range(0, len(ids), batch_size):
        batch = torch.as_tensor(ids[offset:offset + batch_size], device=tensors["nodes"].device)
        nodes = tensors["nodes"][batch]
        centre, scale = tensors["center"][nodes, None], tensors["scale"][nodes, None]
        outputs.append(model.forward_samples(tensors["dynamic"][batch], nodes, tensors["covariates"][batch],
                                             tensors["anchor"][batch]) * scale + centre)
    return torch.cat(outputs).cpu().numpy()


def fit(model, tensors, train_ids, validation_ids, seed, weights=None, epochs=200, patience=10, batch_size=32,
        lr=1e-3, weight_decay=0.01):
    """Train with the frozen benchmark's recipe: AdamW, plateau scheduler, validation-loss early stopping."""
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5)
    device = tensors["nodes"].device
    best, best_epoch, best_state, history = float("inf"), -1, None, []
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(train_ids)
        for offset in range(0, len(order), batch_size):
            ids = torch.as_tensor(order[offset:offset + batch_size], device=device)
            loss, _, _ = batch_loss(model, tensors, ids, weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for offset in range(0, len(validation_ids), 512):
                ids = torch.as_tensor(validation_ids[offset:offset + 512], device=device)
                loss, _, valid = batch_loss(model, tensors, ids, weights)
                total += float(loss) * int(valid.sum())
                count += int(valid.sum())
        validation_loss = total / count
        scheduler.step(validation_loss)
        history.append(validation_loss)
        if validation_loss < best:
            best, best_epoch, best_state = validation_loss, epoch, deepcopy(model.state_dict())
        if epoch - best_epoch >= patience:
            break
    model.load_state_dict(best_state)
    return {"best_epoch": best_epoch + 1, "epochs": len(history), "validation_loss": best}


def score(prediction, actual, nodes, states):
    """Pooled MAE/RMSE in MW, macro-averaged provincial MAPE, and per-province MAE."""
    valid = np.isfinite(actual)
    error = np.where(valid, prediction - actual, np.nan)
    provinces, mape = {}, []
    for index, state in enumerate(states):
        mask = (nodes == index)[:, None] & valid
        if mask.any():
            provinces[state] = float(np.abs(error[mask]).mean())
            large = mask & (np.abs(actual) >= 10)
            mape.append(float((np.abs(error[large]) / np.abs(actual[large])).mean() * 100))
    return {"mae": float(np.nanmean(np.abs(error))), "rmse": float(np.sqrt(np.nanmean(error ** 2))),
            "macro_mape": float(np.mean(mape)), "n_obs": int(valid.sum()), "by_province_mae": provinces}
