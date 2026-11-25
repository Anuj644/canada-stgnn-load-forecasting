"""Audit matched prediction keys and compute fixed temporal baselines and paired CIs."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def metrics(actual, predicted):
    error = predicted - actual
    return {"mae": float(np.abs(error).mean()), "rmse": float(np.sqrt((error ** 2).mean())),
            "mape": float((np.abs(error) / np.clip(actual, 1e-8, None)).mean() * 100)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/external/grid_aware/Updated_dynamic_data.csv"))
    args = parser.parse_args()
    key = ["issued_at", "state", "horizon"]
    frames = {}
    for path in sorted(args.results.glob("predictions_*.parquet")):
        frame = pd.read_parquet(path)
        if frame.duplicated(key).any():
            raise ValueError(f"Duplicate forecast keys in {path}")
        frames[path.stem.removeprefix("predictions_")] = frame.set_index(key).sort_index()
    if not frames:
        raise ValueError("No predictions found")
    reference = next(iter(frames.values()))
    for frame in frames.values():
        if not frame.index.equals(reference.index):
            raise ValueError("Forecast keys differ between methods")
        np.testing.assert_allclose(frame.actual, reference.actual, atol=0.002, rtol=1e-6)
    raw = pd.read_csv(args.data, usecols=["date_time", "state", "load"])
    raw["date_time"] = pd.to_datetime(raw.date_time, format="%d/%m/%Y %H:%M")
    load = raw.set_index(["date_time", "state"]).load
    keys = reference.reset_index()[key]
    for name, lag in [("naive_day", 24), ("naive_week", 168)]:
        lookup = pd.MultiIndex.from_arrays([keys.issued_at + pd.to_timedelta(keys.horizon - lag, unit="h"), keys.state])
        frame = reference[["actual"]].copy()
        frame["predicted"] = load.reindex(lookup).to_numpy()
        if frame.predicted.isna().any():
            raise ValueError("Missing baseline history")
        frames[name] = frame
    rows = []
    for name, frame in frames.items():
        for horizon, subset in frame.groupby(level="horizon"):
            rows.append({"model": name, "horizon": int(horizon), "n_obs": len(subset),
                         **metrics(subset.actual.to_numpy(), subset.predicted.to_numpy())})
    table = pd.DataFrame(rows)
    table.to_csv(args.results / "comparison.csv", index=False)
    # A seven-day moving-block bootstrap preserves serial dependence within weeks.
    paired = []
    rng = np.random.default_rng(2026)
    identities = [name for name in frames if name.startswith("residual_identity_")]
    for identity in identities:
        seed = identity.removeprefix("residual_identity_")
        graph = f"residual_grid_{seed}"
        if graph not in frames:
            continue
        a, b = frames[graph], frames[identity]
        delta = (a.predicted - a.actual).abs() - (b.predicted - b.actual).abs()
        daily = delta.groupby([delta.index.get_level_values("issued_at").normalize(),
                               delta.index.get_level_values("horizon")]).mean().unstack()
        values = daily.to_numpy()
        boot = []
        for _ in range(2000):
            starts = rng.integers(0, len(values), size=int(np.ceil(len(values) / 7)))
            indices = ((starts[:, None] + np.arange(7)) % len(values)).ravel()[:len(values)]
            boot.append(values[indices].mean(axis=0))
        bounds = np.quantile(boot, [0.025, 0.975], axis=0)
        for i, horizon in enumerate(daily.columns):
            paired.append({"seed": seed, "horizon": int(horizon),
                           "grid_minus_identity_mae_mw": float(delta.xs(horizon, level="horizon").mean()),
                           "95pct_week_block_ci": bounds[:, i].tolist()})
    (args.results / "paired_graph_comparison.json").write_text(json.dumps(paired, indent=2), encoding="utf-8")
    print(table.to_string(index=False))
    print(json.dumps(paired, indent=2))


if __name__ == "__main__":
    main()
