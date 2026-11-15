"""Residual STGCN retrained on the HFED API, for next-day forecasts from live data.

The published residual STGCN (``models/stgnn``) was trained on 2019-2023. This module
retrains the same architecture on processed 2019-2024 HFED data plus API demand from
2025 onward, and serves day-ahead forecasts from fresh API pulls. Its checkpoints live
in ``models/stgnn_live`` so the published results stay reproducible.

Chronological windows (by local forecast date):
    fit             < 2026-05-01
    early stopping  2026-05-01 .. 2026-06-30
    graph selection 2026-07-01 .. 2026-08-31
    held-out test   2026-09-01 onward
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.models.stgnn.residual import (
    TIMEZONES,
    ResidualSTGCN,
    build_examples,
    fit_graph,
    predict_graph,
)

GRAPHS = ("identity", "geographic", "interconnection", "correlation")
SEEDS = (42, 123)
WINDOWS = {
    "fit_end": "2026-05-01",
    "early_stop": ("2026-05-01", "2026-06-30"),
    "selection": ("2026-07-01", "2026-08-31"),
    "test_start": "2026-09-01",
}
LIVE_DIR = Path("models") / "stgnn_live"
# Lags reach back 7 days and the issue window adds 2, so 10 days of history covers it.
HISTORY_DAYS = 10


def add_calendar(demand, holidays):
    """Add province-local time and a holiday flag to hourly UTC demand rows."""
    parts = []
    for code, rows in demand.groupby("region"):
        rows = rows.sort_values("time_utc").copy()
        rows["time_local"] = (
            rows["time_utc"].dt.tz_localize("UTC").dt.tz_convert(TIMEZONES[code]).dt.tz_localize(None)
        )
        dates = pd.to_datetime(holidays.loc[holidays["region"].eq(code), "date"])
        rows["is_holiday"] = rows["time_local"].dt.normalize().isin(dates).astype(int)
        parts.append(rows)
    return pd.concat(parts, ignore_index=True)


def complete_grid(demand, regions, end=None):
    """Reindex every province onto one shared hourly UTC grid (missing hours stay NaN).

    ``end`` extends the grid past the latest reading so a future target day has slots.
    """
    last = demand["time_utc"].max().ceil("h")
    if end is not None:
        last = max(last, pd.Timestamp(end))
    grid = pd.date_range(demand["time_utc"].min().floor("h"), last, freq="h", name="time_utc")
    parts = []
    for code in regions:
        series = demand.loc[demand["region"].eq(code)].groupby("time_utc")["demand_mw"].mean()
        part = series.reindex(grid).reset_index()
        part["region"] = code
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def window_masks(frame):
    issue_dates = frame["forecast_date"].iloc[::24].reset_index(drop=True)
    return {
        "fit": issue_dates.lt(WINDOWS["fit_end"]).to_numpy(),
        "early_stop": issue_dates.between(*WINDOWS["early_stop"]).to_numpy(),
        "selection": issue_dates.between(*WINDOWS["selection"]).to_numpy(),
        "test": issue_dates.ge(WINDOWS["test_start"]).to_numpy(),
    }


def ensemble(models, frame, tensors, stats, mask):
    members = [predict_graph(model, frame, tensors, stats, mask) for model in models]
    forecast = members[0].copy()
    forecast["predicted"] = np.mean([member["predicted"].to_numpy() for member in members], axis=0)
    return forecast


def score(forecast):
    rows = forecast.dropna(subset=["actual"])
    error = rows["predicted"] - rows["actual"]
    return {
        "mae": float(error.abs().mean()),
        "rmse": float(np.sqrt((error ** 2).mean())),
        "mape": float((error.abs() / rows["actual"]).mean() * 100),
        "n_hours": len(rows),
    }


def load_live(root):
    """Load live metadata and the selected graph's seed models (CPU)."""
    model_dir = Path(root) / LIVE_DIR
    meta = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    models = []
    for seed in meta["seeds"]:
        state = torch.load(model_dir / f"residual_stgcn_{meta['selected_graph']}_{seed}.pt",
                           map_location="cpu", weights_only=True)
        model = ResidualSTGCN(state["adjacency"], meta["calendar_size"])
        model.load_state_dict(state)
        model.eval()
        models.append(model)
    meta["scaling"] = pd.DataFrame.from_dict(meta["scaling"], orient="index")
    return meta, models


def forecast_next_day(demand, holidays, last_observed_day, root):
    """Forecast all provinces for the day after ``last_observed_day``.

    ``demand`` holds API readings (``region``, ``time_utc``, ``demand_mw``) covering at
    least ``HISTORY_DAYS`` before the target day. Readings on the target day are used
    only as ground truth; the model never sees them.

    Returns a DataFrame with one row per province and local target hour:
    ``region``, ``target_hour``, ``target_at`` (UTC), ``predicted``, ``actual``.
    """
    meta, models = load_live(root)
    regions = meta["regions"]
    target_day = pd.Timestamp(last_observed_day) + pd.Timedelta(days=1)
    # The target day ends at most 8 hours after UTC midnight in Canada; keep a margin.
    grid = complete_grid(demand, regions, end=target_day + pd.Timedelta(days=1, hours=12))
    frame_data = add_calendar(grid, holidays)
    frame, tensors, _ = build_examples(frame_data, None, regions, stats=meta["scaling"],
                                       forecast_dates={target_day})
    mask = np.ones(len(tensors[3]), dtype=bool)
    forecast = ensemble(models, frame, tensors, meta["scaling"], mask)
    return forecast[["region", "target_hour", "target_at", "predicted", "actual"]]


def train_live(data, holidays, config, root, device, log=print):
    """Retrain all four graphs on ``data`` and save the live model and its scores."""
    from src.models.stgnn.graph_builder import build_all_graphs

    regions = sorted(data["region"].unique())
    data = add_calendar(complete_grid(data, regions), holidays)
    train = data.loc[data["time_utc"].lt(WINDOWS["fit_end"])]
    frame, tensors, scaling = build_examples(data, train, regions)
    masks = window_masks(frame)
    log("Samples per window: " + ", ".join(f"{k}={v.sum():,}" for k, v in masks.items()))
    graphs = build_all_graphs(regions, train, config["stgnn"]["geographic_max_distance_km"],
                              config["stgnn"]["correlation_threshold"])

    model_dir = Path(root) / LIVE_DIR
    model_dir.mkdir(parents=True, exist_ok=True)
    scored = masks["selection"] | masks["test"]
    results = {}
    for name in GRAPHS:
        models = []
        for seed in SEEDS:
            model, history = fit_graph(graphs[name], tensors, masks["fit"], masks["early_stop"], device, seed)
            torch.save({key: value.cpu() for key, value in model.state_dict().items()},
                       model_dir / f"residual_stgcn_{name}_{seed}.pt")
            (model_dir / f"history_{name}_seed{seed}.json").write_text(json.dumps(history), encoding="utf-8")
            models.append(model)
            log(f"  {name} seed {seed}: {len(history)} epochs")
        forecast = ensemble(models, frame, tensors, scaling, scored)
        issue = forecast["forecast_date"]
        results[name] = {
            "selection": score(forecast.loc[issue.between(*WINDOWS["selection"])]),
            "test": score(forecast.loc[issue.ge(WINDOWS["test_start"])]),
            "forecast": forecast,
        }
        log(f"  {name}: selection MAE {results[name]['selection']['mae']:.1f} MW, "
            f"test MAE {results[name]['test']['mae']:.1f} MW")

    selected = min(GRAPHS, key=lambda name: results[name]["selection"]["mae"])
    thresholds = {code: float(np.nanpercentile(train.loc[train["region"].eq(code), "demand_mw"], 90))
                  for code in regions}
    meta = {
        "regions": regions,
        "selected_graph": selected,
        "seeds": list(SEEDS),
        "calendar_size": int(tensors[1].shape[-1]),
        "scaling": scaling.to_dict(orient="index"),
        "windows": WINDOWS,
        "data_start": str(data["time_utc"].min()),
        "data_end": str(data.dropna(subset=["demand_mw"])["time_utc"].max()),
        "high_demand_threshold_mw": thresholds,
        "scores": {name: {k: v for k, v in result.items() if k != "forecast"}
                   for name, result in results.items()},
        "trained_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (model_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    results[selected]["forecast"].to_parquet(model_dir / "test_predictions.parquet", index=False)
    return meta, results, (frame, tensors, masks)
