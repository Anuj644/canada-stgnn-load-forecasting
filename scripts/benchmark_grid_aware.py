"""Compare the existing residual model and a Grid-Aware reimplementation in Brazil.

This is a controlled reimplementation, not a claim of exact paper replication.
Run --audit first: the old Zenodo release does not cover the paper's dates.
Downloaded author code is retained as evidence and is never executed here.
"""
from __future__ import annotations

import argparse
import gc
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.stgnn.residual import ResidualSTGCN
from src.models.stgnn.grid_aware import GridAware

HORIZONS = np.array([1, 6, 24])
SPLITS = {
    "train": ("2017-01-01", "2019-01-01"),
    "validation": ("2019-01-01", "2019-07-01"),
    "test": ("2019-07-01", "2020-01-01"),
}


def issue_indices(index, start, end, lookback=24):
    """All histories and all endpoint targets must fall in this split."""
    positions = np.arange(len(index))
    first = np.searchsorted(index, pd.Timestamp(start))
    stop = np.searchsorted(index, pd.Timestamp(end))
    return positions[(positions >= first + lookback - 1) &
                     (positions + HORIZONS.max() < stop) & (positions >= 168)]


def seasonal_positions(issues):
    """Same target hour on each of seven earlier days, never after issuance."""
    pos = np.asarray(issues)[:, None, None] + HORIZONS[None, :, None]
    pos = pos - 24 * np.arange(1, 8)[None, None, :]
    if np.any(pos > np.asarray(issues)[:, None, None]):
        raise ValueError("Seasonal covariate is later than issuance")
    return pos


def score(actual, prediction):
    """Use the author's pooled MW metrics and MAPE denominator (no 10 MW filter)."""
    error = np.asarray(prediction, dtype=np.float64) - actual
    return {"mae": float(np.abs(error).mean()),
            "rmse": float(np.sqrt(np.square(error).mean())),
            "mape": float((np.abs(error) / np.clip(actual, 1e-8, None)).mean() * 100),
            "n_obs": int(actual.size)}


def load_data(folder):
    path = folder / "Updated_dynamic_data.csv"
    preview = pd.read_csv(path, nrows=2)
    date_col = "date_time" if "date_time" in preview else "datetime"
    load_col = "load" if "load" in preview else "value"
    static_columns = {"state_id", "total_in_capacity", "total_in_length", "total_out_capacity",
                      "total_out_length", "x", "y", "pv_pot", "onw_pot", "ofw_pot",
                      "avg_in_efficiency", "avg_out_efficiency"}
    features = [c for c in preview.select_dtypes(include="number") if c not in static_columns]
    df = pd.read_csv(path, usecols=[date_col, "state", *features],
                     dtype={**{c: "float64" for c in features}, "state": "category"})
    # Updated Zenodo CSV uses DD/MM/YYYY, including ambiguous January dates.
    df[date_col] = pd.to_datetime(df[date_col], format="%d/%m/%Y %H:%M")
    if df.duplicated([date_col, "state"]).any():
        raise ValueError("Duplicate state/timestamp keys")
    states = sorted(df.state.unique())
    index = pd.date_range(df[date_col].min(), df[date_col].max(), freq="h")
    if index.min() > pd.Timestamp("2017-01-01") or index.max() < pd.Timestamp("2019-12-31 23:00"):
        raise ValueError(f"Dataset spans {index.min()} to {index.max()}; paper requires 2017-2019")
    df = df.set_index([date_col, "state"]).reindex(pd.MultiIndex.from_product([index, states]))
    # Match the dynamic columns in the author preprocessing, excluding static graph attributes.
    for col in ["pv_value", "on_wind_value", "off_wind_value"]:
        df[col] = df[col].clip(lower=0)
    values = df[features].to_numpy().reshape(len(index), len(states), -1)
    if not np.isfinite(values).all():
        raise ValueError("Missing/nonfinite dynamic data; explicit policy required before benchmarking")
    demand = values[:, :, features.index(load_col)].copy()
    holiday_raw = (values[:, :, features.index("is_holiday")].copy() if "is_holiday" in features
                   else np.zeros_like(demand))
    del df
    gc.collect()
    train = (index >= SPLITS["train"][0]) & (index < SPLITS["train"][1])
    scaler = RobustScaler().fit(values[train].reshape(-1, len(features)))
    dynamic = scaler.transform(values.reshape(-1, len(features))).reshape(values.shape).astype("float32")
    target_scaler = RobustScaler().fit(demand[train].reshape(-1, 1))
    center, scale = demand[train].mean(0), demand[train].std(0, ddof=1)
    normalized = ((demand - center) / scale).astype("float32")
    history = np.stack([normalized, np.ones_like(normalized)], axis=-1)
    # Target-hour calendar and public holiday flag are known at forecast issuance.
    calendar = np.column_stack([
        np.sin(index.hour * np.pi / 12), np.cos(index.hour * np.pi / 12),
        np.sin(index.dayofyear * 2 * np.pi / 366),
        np.cos(index.dayofyear * 2 * np.pi / 366), np.eye(7)[index.dayofweek],
    ]).astype("float32")
    holiday = holiday_raw.astype("float32")
    adjacency = np.eye(len(states), dtype="float32")
    edges = pd.read_csv(folder / "grid_df.csv")
    mapping = {s: i for i, s in enumerate(states)}
    edge_index, edge_features = [], []
    for row in edges.itertuples(index=False):
        a, b = mapping[row.Source], mapping[row.Target]
        adjacency[a, b] = adjacency[b, a] = 1
        edge_index.extend([[a, b], [b, a]])
        edge_features.extend([[row.capacity, row.line_eff, row.line_len, row.line_carrier]] * 2)
    static = pd.read_csv(folder / "static_data.csv").set_index("state").loc[states]
    node_x = RobustScaler().fit_transform(static[["x", "y", "pv_pot", "onw_pot", "ofw_pot"]])
    return dict(index=index, states=states, demand=demand, dynamic=dynamic, history=history,
                normalized=normalized, calendar=calendar, holiday=holiday,
                center=center, scale=scale, feature_center=scaler.center_, feature_scale=scaler.scale_,
                target_center=float(target_scaler.center_[0]),
                target_scale=float(target_scaler.scale_[0]), adjacency=adjacency,
                node_x=node_x, edge_index=np.array(edge_index).T,
                edge_features=RobustScaler().fit_transform(edge_features), features=features,
                splits={s: issue_indices(index, *bounds) for s, bounds in SPLITS.items()})


class ResidualAdapter(nn.Module):
    """Reuse the repository model's actual modules; vectorize shared histories.

    Only the three paper endpoints receive a loss. Same 24h history as baseline.
    No learned dynamic gate or proposed future extension is added in this experiment.
    """
    def __init__(self, adjacency):
        super().__init__()
        self.core = ResidualSTGCN(adjacency, calendar_size=26)

    def forward(self, batch):
        history = batch["history"]
        b, t, n, c = history.shape
        encoded = self.core.temporal(history.permute(0, 2, 3, 1).reshape(b * n, c, t)).reshape(b, n, -1)
        adjacency = self.core.adjacency
        pooled = torch.einsum("nm,bmf->bnf", adjacency, encoded)
        pooled = pooled / adjacency.sum(-1)[None, :, None].clamp_min(1e-6)
        regions = self.core.region(torch.arange(n, device=history.device))[None].expand(b, -1, -1)
        context = torch.cat([encoded, pooled, regions], -1)
        context = context[:, :, None].expand(-1, -1, 3, -1)
        return batch["anchor"] + self.core.head(torch.cat([context, batch["covariates"]], -1)).squeeze(-1)


def device_data(data, device):
    keys = ["dynamic", "history", "normalized", "calendar", "holiday", "demand"]
    return {k: torch.tensor(data[k], dtype=torch.float32, device=device) for k in keys}


def make_batch(arrays, issues):
    device = arrays["history"].device
    issues = torch.as_tensor(issues, dtype=torch.long, device=device)
    hist = issues[:, None] - torch.arange(23, -1, -1, device=device)
    targets = issues[:, None] + torch.tensor(HORIZONS, device=device)[None]
    lag_idx = targets[:, :, None] - 24 * torch.arange(1, 8, device=device)[None, None]
    # B,H,7,N -> B,N,H,7; all are observed in this complete benchmark.
    lags = arrays["normalized"][lag_idx].permute(0, 3, 1, 2)
    n = lags.shape[1]
    calendar = arrays["calendar"][targets][:, None].expand(-1, n, -1, -1)
    holiday = arrays["holiday"][targets].permute(0, 2, 1)[..., None]
    return {"history": arrays["history"][hist], "dynamic": arrays["dynamic"][hist],
            "anchor": lags[..., 0],
            "covariates": torch.cat([lags, torch.ones_like(lags), calendar, holiday], -1),
            "actual": arrays["demand"][targets].permute(0, 2, 1)}


def run_model(name, data, arrays, args, seed, output):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if name == "grid_aware_lstm":
        model = GridAware(data)
        center, scale = data["target_center"], data["target_scale"]
        lr, wd = 1.19e-4, 3.21e-6
    else:
        adjacency = data["adjacency"] if name == "residual_grid" else np.eye(len(data["states"]))
        model = ResidualAdapter(adjacency)
        center = torch.tensor(data["center"], device=args.device, dtype=torch.float32)[None, :, None]
        scale = torch.tensor(data["scale"], device=args.device, dtype=torch.float32)[None, :, None]
        lr, wd = 1e-3, 0.01
    model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd) if name != "grid_aware_lstm" else torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5)
    best, best_epoch, history = float("inf"), -1, []
    started = time.time()
    for epoch in range(args.epochs):
        losses = []
        for split, training in [("train", True), ("validation", False)]:
            model.train(training)
            indices = data["splits"][split].copy()
            if training:
                np.random.shuffle(indices)
            total, count = 0.0, 0
            with torch.set_grad_enabled(training):
                for chunk in range(0, len(indices), args.batch_times):
                    batch = make_batch(arrays, indices[chunk:chunk + args.batch_times])
                    target = (batch["actual"] - center) / scale
                    pred = model(batch)
                    loss = (nn.functional.mse_loss(pred, target) if name == "grid_aware_lstm"
                            else nn.functional.smooth_l1_loss(pred, target, beta=0.25))
                    if training:
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()
                    total += loss.detach().item() * target.numel()
                    count += target.numel()
            losses.append(total / count)
        scheduler.step(losses[1])
        history.append({"epoch": epoch + 1, "train_loss": losses[0], "validation_loss": losses[1]})
        print(json.dumps({"model": name, "seed": seed, **history[-1], "seconds": round(time.time() - started, 1)}), flush=True)
        if losses[1] < best:
            best, best_epoch = losses[1], epoch
            state = deepcopy(model.state_dict())
        if epoch - best_epoch >= args.patience:
            break
    model.load_state_dict(state)
    torch.save({"state_dict": state, "history": history, "seed": seed,
                "features": data["features"], "states": data["states"],
                "scalers": {k: np.asarray(data[k]).tolist() for k in
                            ["center", "scale", "feature_center", "feature_scale", "target_center", "target_scale"]}},
               output / f"{name}_{seed}.pt")
    model.eval()
    frames = []
    with torch.inference_mode():
        for offset in range(0, len(data["splits"]["test"]), args.batch_times):
            issues = data["splits"]["test"][offset:offset + args.batch_times]
            batch = make_batch(arrays, issues)
            predictions = (model(batch) * scale + center).cpu().numpy()
            actual = batch["actual"].cpu().numpy()
            for h, horizon in enumerate(HORIZONS):
                frames.append(pd.DataFrame({
                    "issued_at": np.repeat(data["index"][issues].to_numpy(), len(data["states"])),
                    "state": np.tile(data["states"], len(issues)), "horizon": int(horizon),
                    "actual": actual[:, :, h].ravel(), "predicted": predictions[:, :, h].ravel()}))
    predictions = pd.concat(frames, ignore_index=True)
    predictions.to_parquet(output / f"predictions_{name}_{seed}.parquet", index=False)
    return {"model": name, "seed": seed, "epochs": len(history), "best_epoch": best_epoch + 1,
            "seconds": time.time() - started, "metrics": {str(h): score(g.actual.to_numpy(), g.predicted.to_numpy())
            for h, g in predictions.groupby("horizon")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/external/grid_aware")
    parser.add_argument("--output", type=Path, default=ROOT / "results/grid_aware_benchmark")
    parser.add_argument("--models", nargs="+", choices=["grid_aware_lstm", "residual_identity", "residual_grid"],
                        default=["grid_aware_lstm", "residual_identity", "residual_grid"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-times", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if (args.output / "summary.json").exists():
        raise FileExistsError("Choose a new --output; existing benchmark results must not be overwritten")
    torch.set_num_threads(4)
    data = load_data(args.data)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"status": "audit_only" if args.audit else "single_seed_pilot_not_exact_replication",
        "dataset": "https://zenodo.org/records/18595751", "states": data["states"],
        "start": str(data["index"][0]), "end": str(data["index"][-1]), "features": data["features"],
        "splits": SPLITS, "samples_per_horizon": {k: len(v) * len(data["states"]) for k, v in data["splits"].items()},
        "horizons": HORIZONS.tolist(), "lookback": 24, "configuration": vars(args).copy(),
        "environment": {"torch": torch.__version__, "python": sys.version,
                        "device": torch.cuda.get_device_name() if args.device == "cuda" else "cpu"},
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [Path(__file__), ROOT / "src/models/stgnn/residual.py"]},
        "deviations_from_author_snippet": [
            "Full calendar split end dates, with targets strictly inside splits; no one-hour boundary spill.",
            "All-node frozen embeddings cached in eval mode and checkpointed; corrected indexing.",
            "Published best LSTM dimensions; no Optuna search; configurable batched timestamps, not batch32 samples.",
            "Residual model uses existing modules with demand-only history and known target calendar; three endpoint losses.",
            "No clipping predictions; all models share exact held-out keys; no 10MW MAPE exclusion.",
            "Dataset already curated upstream; its preprocessing provenance is not independently established."],
        "published_reference_only": {"gat_lstm_24": {"mae": 64.44, "rmse": 151.10, "mape": 3.93},
                                     "gat_gru_24": {"mae": 63.70, "rmse": 146.30, "mape": 4.54}},
        "files": {p.name: hashlib.md5(p.read_bytes()).hexdigest() for p in args.data.glob("*.csv")}}
    manifest["configuration"] = {k: str(v) if isinstance(v, Path) else v for k, v in manifest["configuration"].items()}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    if args.audit:
        return
    arrays = device_data(data, args.device)
    summary = []
    # Frozen configuration: no model selection or weights are fitted on test results.
    for seed in args.seeds:
        for name in args.models:
            summary.append(run_model(name, data, arrays, args, seed, args.output))
            (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
