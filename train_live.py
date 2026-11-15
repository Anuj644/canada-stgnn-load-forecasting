"""Retrain the residual STGCN on HFED API data for live next-day forecasts.

Combines the processed 2019-2024 dataset with demand pulled from the HFED API
(late 2024 to the latest published hour), retrains the four graph variants, and
compares the result with the published checkpoint on the same held-out hours.

Usage:
    python train_live.py               # fetch, retrain, score
    python train_live.py --score-only  # rescore the saved live model, no retraining
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.hfed_api import fetch_demand_history
from src.data.hfed_loader import load_config
from src.data.holiday_client import get_holiday_df_for_data_range
from src.models.stgnn.live import (
    LIVE_DIR,
    WINDOWS,
    add_calendar,
    complete_grid,
    ensemble,
    load_live,
    score,
    train_live,
)
from src.models.stgnn.residual import KEY, ResidualSTGCN, build_examples

ROOT = Path(__file__).resolve().parent
API_START = "2024-09-01"
DATA_PATH = ROOT / "data" / "processed" / "live_training_demand.parquet"


def log(message):
    print(message, flush=True)


def build_dataset(config, regions):
    log("Loading processed 2019-2024 HFED demand...")
    processed = pd.read_parquet(ROOT / "data" / "processed" / "hfed_processed.parquet",
                                columns=["region", "time_utc", "demand_mw"])
    processed = processed.loc[processed["region"].isin(regions)].dropna(subset=["demand_mw"])

    log(f"Fetching API demand {API_START} to today for {', '.join(regions)}...")
    started = time.time()
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize() + pd.Timedelta(days=1)
    api, unavailable = fetch_demand_history(regions, API_START, today, config)
    if unavailable:
        log(f"  No API readings for: {', '.join(unavailable)}")
    log(f"  {len(api):,} hourly readings in {time.time() - started:.0f}s; latest per province:")
    for code, latest in api.groupby("region")["time_utc"].max().items():
        log(f"    {code}: {latest:%Y-%m-%d %H:%M} UTC")

    # Processed rows take precedence where both sources cover an hour.
    data = (pd.concat([processed, api], ignore_index=True)
            .drop_duplicates(["region", "time_utc"], keep="first")
            .sort_values(["region", "time_utc"]).reset_index(drop=True))
    data.to_parquet(DATA_PATH, index=False)
    return data


def predict_window(data, holidays, regions, stats, models, dates):
    frame_data = add_calendar(complete_grid(data, regions), holidays)
    frame, tensors, _ = build_examples(frame_data, None, regions, stats=stats, forecast_dates=dates)
    mask = np.ones(len(tensors[3]), dtype=bool)
    return ensemble(models, frame, tensors, stats, mask), frame_data


def score_models(data, holidays, regions):
    """Score the live model, the published checkpoint and previous-day naive on the same test hours."""
    meta, live_models = load_live(ROOT)
    test_dates = set(pd.date_range(WINDOWS["test_start"], data["time_utc"].max().normalize()))
    live, frame_data = predict_window(data, holidays, regions, meta["scaling"], live_models, test_dates)

    published = json.loads((ROOT / "models" / "stgnn" / "residual_metadata.json").read_text(encoding="utf-8"))
    old_stats = pd.DataFrame.from_dict(published["scaling"], orient="index")
    old_name = published["selected_graph"].removeprefix("residual_stgcn_")
    old_models = []
    for seed in (42, 123):
        state = torch.load(ROOT / "models" / "stgnn" / f"residual_stgcn_{old_name}_{seed}.pt",
                           map_location="cpu", weights_only=True)
        model = ResidualSTGCN(state["adjacency"], meta["calendar_size"])
        model.load_state_dict(state)
        old_models.append(model)
    old, _ = predict_window(data, holidays, regions, old_stats, old_models, test_dates)

    merged = live[KEY + ["target_at", "actual", "predicted"]].merge(
        old[KEY + ["predicted"]], on=KEY, suffixes=("", "_published"))
    lookup = frame_data.set_index(["region", "time_utc"])["demand_mw"]
    previous = pd.MultiIndex.from_arrays([merged["region"], merged["target_at"] - pd.Timedelta(hours=24)])
    merged["naive_day"] = lookup.reindex(previous).to_numpy()
    merged = merged.dropna(subset=["actual", "naive_day"])

    columns = {"Live residual STGCN (retrained on API)": "predicted",
               f"Published residual STGCN ({old_name})": "predicted_published",
               "Seasonal naive, previous day": "naive_day"}
    table = {name: score(merged.assign(predicted=merged[column])) for name, column in columns.items()}
    by_region = {code: {name: score(rows.assign(predicted=rows[column]))["mape"]
                        for name, column in columns.items()}
                 for code, rows in merged.groupby("region")}
    window = f"{merged['forecast_date'].min():%Y-%m-%d} to {merged['forecast_date'].max():%Y-%m-%d}"
    summary = {"test_window": window, "n_hours": len(merged), "overall": table,
               "mape_by_region": by_region, "selected_graph": meta["selected_graph"],
               "graph_scores": meta["scores"]}
    (ROOT / "results" / "live_model_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    log(f"\nHeld-out test {window}, {len(merged):,} province-hours")
    log(f"{'Model':44}{'MAE':>9}{'RMSE':>9}{'MAPE':>8}")
    for name, metrics in table.items():
        log(f"{name:44}{metrics['mae']:9.1f}{metrics['rmse']:9.1f}{metrics['mape']:7.2f}%")
    log("\nMAPE (%) by province: live / published / naive")
    for code, values in by_region.items():
        log(f"  {code}: " + " / ".join(f"{value:.2f}" for value in values.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--score-only", action="store_true", help="Rescore the saved live model without retraining")
    args = parser.parse_args()

    config = load_config(ROOT / "config" / "config.yaml")
    regions = sorted(config["download"]["regions"])
    if args.score_only and DATA_PATH.exists():
        data = pd.read_parquet(DATA_PATH)
    else:
        data = build_dataset(config, regions)

    years = data["time_utc"].dt.year
    holidays = get_holiday_df_for_data_range(int(years.min()), int(years.max()) + 1, hfed_regions=regions,
                                             cache_path=ROOT / config["holidays"]["cache_file"])
    log(f"Holidays: {len(holidays):,} province-days for {int(years.min())}-{int(years.max()) + 1}")

    if not args.score_only:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log(f"Training on {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}...")
        meta, _, _ = train_live(data, holidays, config, ROOT, device, log=log)
        log(f"Selected graph on Jul-Aug 2026: {meta['selected_graph']}")
        log(f"Saved live model to {LIVE_DIR}/")

    log("Scoring live and published models on the same held-out hours...")
    score_models(data, holidays, regions)
    log("Saved scores to results/live_model_metrics.json")


if __name__ == "__main__":
    sys.exit(main())
