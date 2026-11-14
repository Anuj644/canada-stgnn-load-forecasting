"""Day-ahead graph convolution with observed-target masking and seasonal residuals."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

TIMEZONES = {
    "AB": "America/Edmonton",
    "BC": "America/Vancouver",
    "NB": "America/Moncton",
    "NL": "America/St_Johns",
    "NS": "America/Halifax",
    "ON": "America/Toronto",
    "PE": "America/Halifax",
    "QC": "America/Toronto",
}
KEY = ["region", "forecast_date", "target_hour"]


def build_examples(data, train, regions, *, stats=None, forecast_dates=None):
    """Build one sample per region and local issue date, with UTC-aligned neighbors."""
    demand = data.pivot(index="time_utc", columns="region", values="demand_mw")
    demand = demand.reindex(columns=regions).asfreq("h")
    if stats is None:
        stats = train.groupby("region")["demand_mw"].agg(["mean", "std"])
    stats = stats.reindex(regions)
    center = stats["mean"].to_numpy()
    scale = stats["std"].to_numpy()
    normalized = (demand.to_numpy() - center) / scale
    observed = np.isfinite(normalized)
    values = np.nan_to_num(normalized).astype("float32")
    history = np.stack([values, observed], axis=-1).astype("float32")

    # Each target has its own calendar. A 23/25-hour DST day has no unique 24-slot grid.
    lookback = np.arange(47, -1, -1)
    target_hours = np.arange(24)
    offsets = 24 * np.arange(1, 8)
    holiday = data.loc[data["is_holiday"].eq(1), ["region", "time_local"]].copy()
    holiday["date"] = pd.to_datetime(holiday["time_local"]).dt.normalize()
    rows, sequences, covariates, anchors, labels, node_ids = [], [], [], [], [], []
    for node, region in enumerate(regions):
        local = demand.index.tz_localize("UTC").tz_convert(TIMEZONES[region])
        dates = local.tz_localize(None).normalize()
        holiday_dates = holiday.loc[holiday["region"].eq(region), "date"].unique()
        day_size = pd.Series(dates).value_counts()
        # Newfoundland's top-of-hour grid is :30 in UTC. Its last fully
        # available UTC slot before local 23:00 is the :30 local slot at 22:30.
        minutes = local.minute[0]
        starts = np.flatnonzero(local.hour == 0)
        starts = starts[(starts >= 168) & (starts + 23 < len(demand))]
        for start in starts:
            date = dates[start]
            if forecast_dates is not None and date not in forecast_dates:
                continue
            if day_size[date] != 24:
                continue
            issue = start - 1 - int(minutes > 0)
            if not observed[issue - lookback, node].any():
                continue
            target = start + target_hours
            if forecast_dates is None and not observed[target, node].any():
                continue
            lag_positions = target[:, None] - offsets
            # The previous-day final hour can lie after issuance on a half-hour grid.
            known = (lag_positions <= issue) & observed[lag_positions, node]
            lags = np.where(known, values[lag_positions, node], 0)
            anchor = np.take_along_axis(lags, known.argmax(axis=1)[:, None], axis=1)[
                :, 0
            ]
            calendar = np.column_stack(
                [
                    np.sin(target_hours * np.pi / 12),
                    np.cos(target_hours * np.pi / 12),
                    np.full(24, np.sin(date.dayofyear * 2 * np.pi / 366)),
                    np.full(24, np.cos(date.dayofyear * 2 * np.pi / 366)),
                    np.broadcast_to(np.eye(7)[date.dayofweek], (24, 7)),
                    np.full(24, date in holiday_dates),
                ]
            )
            rows.append(
                pd.DataFrame(
                    {
                        "region": region,
                        "forecast_date": date,
                        "target_hour": target_hours,
                        "issued_at": demand.index[start]
                        - pd.Timedelta(hours=1, minutes=minutes),
                        "target_at": demand.index[target],
                        "actual": demand.iloc[target, node].to_numpy(),
                    }
                )
            )
            sequences.append(history[issue - lookback])
            covariates.append(np.column_stack([lags, known, calendar]))
            anchors.append(anchor)
            labels.append(normalized[target, node])
            node_ids.append(node)

    if not rows:
        raise ValueError("No complete forecast-day grid with sufficient demand history was found.")
    frame = pd.concat(rows, ignore_index=True)
    tensors = (
        torch.as_tensor(np.asarray(sequences), dtype=torch.float32),
        torch.as_tensor(np.asarray(covariates), dtype=torch.float32),
        torch.as_tensor(np.asarray(anchors), dtype=torch.float32),
        torch.as_tensor(np.asarray(node_ids), dtype=torch.long),
        torch.as_tensor(np.asarray(labels), dtype=torch.float32),
    )
    return frame, tensors, stats


class ResidualSTGCN(nn.Module):
    def __init__(self, adjacency, calendar_size):
        super().__init__()
        self.register_buffer(
            "adjacency", torch.as_tensor(adjacency, dtype=torch.float32)
        )
        self.temporal = nn.Sequential(
            nn.Conv1d(2, 24, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(24, 24, 5, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(6),
            nn.Flatten(),
            nn.Linear(144, 32),
        )
        self.region = nn.Embedding(len(adjacency), 8)
        self.head = nn.Sequential(
            nn.Linear(32 * 2 + 8 + calendar_size, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, history, calendar, anchor, node):
        batch, steps, regions, channels = history.shape
        temporal = history.permute(0, 2, 3, 1).reshape(batch * regions, channels, steps)
        encoded = self.temporal(temporal).reshape(batch, regions, -1)
        available = history[..., 1].any(dim=1)
        weights = self.adjacency[node] * available
        pooled = torch.einsum("bn,bnf->bf", weights, encoded)
        pooled = pooled / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        own = encoded[torch.arange(batch, device=node.device), node]
        context = torch.cat([own, pooled, self.region(node)], dim=-1)
        combined = torch.cat([context[:, None].expand(-1, 24, -1), calendar], dim=-1)
        return anchor + self.head(combined).squeeze(-1)


def fit_graph(adjacency, tensors, train_mask, tune_mask, device, seed, epochs=35):
    torch.manual_seed(seed)
    dataset = TensorDataset(*tensors)
    loaders = [
        DataLoader(
            torch.utils.data.Subset(dataset, np.flatnonzero(mask)), 128, shuffle=shuffle
        )
        for mask, shuffle in [(train_mask, True), (tune_mask, False)]
    ]
    model = ResidualSTGCN(adjacency, tensors[1].shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    loss_fn = nn.SmoothL1Loss(beta=0.25, reduction="sum")
    best = float("inf")
    history = []
    for epoch in range(epochs):
        losses = []
        for loader, training in zip(loaders, [True, False]):
            model.train(training)
            total = count = 0
            with torch.set_grad_enabled(training):
                for batch in loader:
                    *inputs, actual = [value.to(device) for value in batch]
                    predicted = model(*inputs)
                    valid = actual.isfinite()
                    loss = loss_fn(predicted[valid], actual[valid])
                    if training:
                        optimizer.zero_grad(set_to_none=True)
                        (loss / valid.sum()).backward()
                        optimizer.step()
                    total += loss.item()
                    count += valid.sum().item()
            losses.append(total / count)
        history.append(losses)
        if losses[1] < best:
            best = losses[1]
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
        if epoch - best_epoch >= 6:
            break
    model.load_state_dict(best_state)
    return model, history


@torch.inference_mode()
def predict_graph(model, frame, tensors, stats, mask):
    model.eval()
    loader = DataLoader(TensorDataset(*(tensor[mask] for tensor in tensors[:-1])), 256)
    device = next(model.parameters()).device
    predictions = torch.cat(
        [model(*(value.to(device) for value in batch)).cpu() for batch in loader]
    )
    output = frame.loc[np.repeat(mask, 24)].copy()
    output["predicted"] = (
        predictions.numpy().ravel() * output["region"].map(stats["std"])
        + output["region"].map(stats["mean"])
    ).clip(lower=0)
    return output.reset_index(drop=True)
