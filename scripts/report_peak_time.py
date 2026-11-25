"""Report daily peak timing and peak size for every saved HFED forecast set.

Reads the forecast files written by the benchmark (``<period>/predictions_<model>_<seed>.parquet``)
and writes four CSVs. Nothing is retrained and no existing result file is changed.
Every model is scored on the same fully observed forecast days within a period.

Usage:
    python scripts/report_peak_time.py
    python scripts/report_peak_time.py --runs results/hfed_extension_eccc_v4 results/hfed_cnn_tuning_v1
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.peak_time import KEYS, daily_peaks, peak_hour_counts, peak_time_metrics

DEFAULT_RUNS = ["results/hfed_extension_eccc_v4", "results/hfed_temporal_controls_v1"]
PERIODS = ["test2024", "test2025"]


def discover(runs, period):
    """Yield (model, seed, path) for each forecast file of ``period`` across the runs."""
    for run in runs:
        for path in sorted((Path(run) / period).glob("predictions_*.parquet")):
            model, seed = path.stem.removeprefix("predictions_").rsplit("_", 1)
            yield model, int(seed), path


def load_days(runs, period):
    days = {}
    for model, seed, path in discover(runs, period):
        forecast = pd.read_parquet(path, columns=[*KEYS, "horizon", "actual", "predicted"])
        days[model, seed] = daily_peaks(forecast)
    return days


def common_days(days):
    """Restrict every model to the forecast days that all of them scored."""
    shared = None
    for frame in days.values():
        keys = pd.MultiIndex.from_frame(frame[KEYS])
        shared = keys if shared is None else shared.intersection(keys)
    return {name: frame.loc[pd.MultiIndex.from_frame(frame[KEYS]).isin(shared)] for name, frame in days.items()}


def summarize(runs):
    overall, regional, counts = [], [], []
    for period in PERIODS:
        days = common_days(load_days(runs, period))
        for (model, seed), frame in days.items():
            base = {"period": period, "model": model, "seed": seed}
            overall.append({**base, **peak_time_metrics(frame)})
            for state, rows in frame.groupby("state"):
                regional.append({**base, "state": state, **peak_time_metrics(rows)})
            for kind in ("actual", "predicted"):
                series = peak_hour_counts(frame, f"{kind}_peak_hour")
                counts.extend({**base, "kind": kind, "hour": int(hour), "days": int(n)} for hour, n in series.items())
    return pd.DataFrame(overall), pd.DataFrame(regional), pd.DataFrame(counts)


def mean_std(by_seed):
    metrics = [c for c in by_seed.columns if c not in {"period", "model", "seed", "state", "n_days"}]
    grouped = by_seed.groupby(["period", "model"])
    table = grouped[metrics].agg(["mean", "std"])
    table.columns = [f"{name}_{stat}" for name, stat in table.columns]
    table.insert(0, "n_days", grouped["n_days"].first())
    return table.reset_index()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    parser.add_argument("--output", type=Path, default=ROOT / DEFAULT_RUNS[0])
    args = parser.parse_args()
    runs = [ROOT / run if not Path(run).is_absolute() else Path(run) for run in args.runs]
    overall, regional, counts = summarize([run for run in runs if run.exists()])
    args.output.mkdir(parents=True, exist_ok=True)
    overall.to_csv(args.output / "peak_time_metrics_by_seed.csv", index=False)
    mean_std(overall).to_csv(args.output / "peak_time_metrics_mean_std.csv", index=False)
    regional.to_csv(args.output / "peak_time_regional_by_seed.csv", index=False)
    counts.to_csv(args.output / "peak_time_hour_counts.csv", index=False)
    table = mean_std(overall)
    for period in PERIODS:
        rows = table.loc[table.period.eq(period)].sort_values("peak_hour_mae_mean")
        print(f"\n{period}: {int(rows.n_days.iloc[0])} fully observed forecast days per model")
        print(rows[["model", "peak_hour_mae_mean", "peak_hour_exact_mean", "peak_hour_within_1h_mean",
                    "peak_size_mae_mw_mean", "peak_regret_pct_mean"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
