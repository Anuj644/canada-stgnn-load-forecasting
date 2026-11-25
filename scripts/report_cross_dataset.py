"""Aggregate seed variability and paired week-block uncertainty from saved forecasts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def weather_regimes(cache, metadata):
    """Stratify by temperature known at issue, with training-only thresholds."""
    records = pd.read_parquet(cache / "records.parquet")
    features = metadata["features"]
    ti, mi = features.index("temperature_c"), features.index("temperature_c_observed")
    with np.load(cache / "examples.npz", allow_pickle=False) as archive:
        dynamic = archive["dynamic"]
        temperature = dynamic[:, :, ti] * archive["feature_scale"][ti] + archive["feature_center"][ti]
        observed = dynamic[:, :, mi] * archive["feature_scale"][mi] + archive["feature_center"][mi] > 0.5
        count = observed.sum(axis=1)
        means = np.divide(np.where(observed, temperature, 0).sum(axis=1), count,
                          out=np.full(len(count), np.nan), where=count >= 12)
        train_ids = archive["split_train"]
    records = records.assign(temperature_mean_at_issue=means, weather_regime="weather_unavailable")
    thresholds = {}
    for state in metadata["states"]:
        region = records.region.eq(state).to_numpy()
        train = means[train_ids[region[train_ids]]]
        if not np.isfinite(train).any():
            raise ValueError(f"No observed training temperature for {state}")
        low, high = np.nanquantile(train, [0.05, 0.95])
        thresholds[state] = {"cold_5pct_c": float(low), "hot_95pct_c": float(high)}
        records.loc[region & np.isfinite(means), "weather_regime"] = "typical"
        records.loc[region & (means <= low), "weather_regime"] = "cold_at_issue"
        records.loc[region & (means >= high), "weather_regime"] = "hot_at_issue"
    return records.rename(columns={"region": "state"})[["issued_at", "state", "weather_regime"]], thresholds


def hfed_details(results, plan):
    cache = Path(plan["cache"])
    metadata = json.loads((cache / "metadata.json").read_text(encoding="utf-8"))
    with np.load(cache / "examples.npz", allow_pickle=False) as archive:
        ids, nodes, actual = archive["split_train"], archive["nodes"], archive["actual"]
        thresholds = {state: float(np.nanpercentile(actual[ids[nodes[ids] == node]], 90))
                      for node, state in enumerate(metadata["states"])}
    regimes, weather_thresholds = weather_regimes(cache, metadata)
    rows, totals, weather_rows = [], [], []
    coverage = []
    for period in ["test2024", "test2025"]:
        for model in plan["models"]:
            for seed in plan["seeds"]:
                frame = pd.read_parquet(results / period / f"predictions_{model}_{seed}.parquet")
                classified = frame.merge(regimes, on=["issued_at", "state"], how="left", validate="many_to_one")
                if classified.weather_regime.isna().any():
                    raise ValueError("Weather diagnostic keys do not match forecasts")
                for regime, group in classified.groupby("weather_regime"):
                    error = group.predicted.to_numpy(dtype=float) - group.actual.to_numpy(dtype=float)
                    weather_rows.append({"period": period, "model": model, "seed": seed, "weather_regime": regime,
                        "n_obs": len(group), "mae": float(np.abs(error).mean()),
                        "rmse": float(np.sqrt(np.square(error).mean()))})
                if model == plan["models"][0] and seed == plan["seeds"][0]:
                    for state, g in frame.groupby("state"):
                        coverage.append({"period": period, "state": state, "n_obs": len(g),
                                         "first_issue": str(g.issued_at.min()), "last_issue": str(g.issued_at.max()),
                                         "forecast_days": g.issued_at.nunique()})
                for state, group in frame.groupby("state"):
                    y, p = group.actual.to_numpy(dtype=float), group.predicted.to_numpy(dtype=float)
                    err, valid_mape = p - y, np.abs(y) >= 10
                    peak = y >= thresholds[state]
                    rows.append({"period": period, "model": model, "seed": seed, "state": state,
                                 "n_obs": len(y), "mae": float(np.abs(err).mean()),
                                 "rmse": float(np.sqrt(np.square(err).mean())),
                                 "mape": float(np.abs(err[valid_mape] / y[valid_mape]).mean() * 100),
                                 "peak_n_obs": int(peak.sum()),
                                 "peak_mae": float(np.abs(err[peak]).mean()) if peak.any() else np.nan})
                y, p = frame.actual.to_numpy(dtype=float), frame.predicted.to_numpy(dtype=float)
                threshold = frame.state.map(thresholds).to_numpy()
                peak, flagged = y >= threshold, p >= threshold
                error = p - y
                tp = int((peak & flagged).sum())
                totals.append({"period": period, "model": model, "seed": seed,
                    "peak_mae": float(np.abs(error[peak]).mean()),
                    "peak_rmse": float(np.sqrt(np.square(error[peak]).mean())),
                    "peak_precision": tp / flagged.sum() if flagged.any() else np.nan,
                    "peak_recall": tp / peak.sum() if peak.any() else np.nan,
                    "peak_n_obs": int(peak.sum())})
    regional = pd.DataFrame(rows)
    regional.to_csv(results / "regional_metrics_by_seed.csv", index=False)
    regional.groupby(["period", "model", "state"])[["mae", "rmse", "mape", "peak_mae"]].agg(["mean", "std"]).to_csv(results / "regional_metrics_mean_std.csv")
    regional.groupby(["period", "model", "seed"])[["mae", "mape"]].mean().to_csv(results / "macro_regional_metrics.csv")
    pd.DataFrame(totals).to_csv(results / "peak_metrics_by_seed.csv", index=False)
    pd.DataFrame(coverage).to_csv(results / "regional_coverage.csv", index=False)
    pd.DataFrame(weather_rows).to_csv(results / "weather_regime_metrics_by_seed.csv", index=False)
    (results / "training_weather_thresholds.json").write_text(json.dumps({
        "definition": "Mean observed temperature in the 24h input history; at least 12 observations. Training-only per-province 5th/95th percentiles. This describes conditions at issue, not realized target-day extremes.",
        "thresholds": weather_thresholds}, indent=2), encoding="utf-8")
    (results / "training_peak_thresholds.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")


def matched_differences(folder, candidate, reference, seeds, horizon, reference_folder=None):
    deltas, reference_maes, candidate_maes = [], [], []
    index, actual_ref = None, None
    for seed in seeds:
        frames = []
        for source, name in [(folder, candidate), (reference_folder or folder, reference)]:
            frame = pd.read_parquet(source / f"predictions_{name}_{seed}.parquet")
            if horizon is not None:
                frame = frame.loc[frame.horizon.eq(horizon)]
            if frame.duplicated(["issued_at", "state", "horizon"]).any():
                raise ValueError("Duplicate forecast keys")
            frame = frame.set_index(["issued_at", "state", "horizon"]).sort_index()
            if index is None:
                index, actual_ref = frame.index, frame.actual.to_numpy(dtype=float)
            if not frame.index.equals(index):
                raise ValueError("Model forecast keys differ")
            np.testing.assert_allclose(frame.actual, actual_ref, atol=0.002, rtol=1e-6)
            frames.append(np.abs(frame.predicted.to_numpy(dtype=float) - actual_ref))
        deltas.append(frames[0] - frames[1])
        candidate_maes.append(float(frames[0].mean()))
        reference_maes.append(float(frames[1].mean()))
    delta = np.mean(deltas, axis=0)
    values = pd.Series(delta, index=index)
    daily = values.groupby(index.get_level_values("issued_at").normalize()).agg(["sum", "count"])
    sums, counts = daily["sum"].to_numpy(), daily["count"].to_numpy()
    rng = np.random.default_rng(2026)
    estimates = []
    for _ in range(4000):
        starts = rng.integers(0, len(daily), size=int(np.ceil(len(daily) / 7)))
        indices = ((starts[:, None] + np.arange(7)) % len(daily)).ravel()[:len(daily)]
        estimates.append(float(sums[indices].sum() / counts[indices].sum()))
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return {"candidate": candidate, "reference": reference, "horizon": horizon or "all24",
            "candidate_mean_mae": float(np.mean(candidate_maes)), "reference_mean_mae": float(np.mean(reference_maes)),
            "improvement_percent": float(100 * (1 - np.mean(candidate_maes) / np.mean(reference_maes))),
            "paired_mae_difference": float(delta.mean()), "week_block_95pct_ci": [float(lower), float(upper)],
            "lower_mae_each_seed": bool(np.all(np.array(candidate_maes) < reference_maes)),
            "n_observations_per_seed": len(delta)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--kind", choices=["brazil", "hfed"], required=True)
    args = parser.parse_args()
    evaluation = json.loads((args.results / "evaluation.json").read_text(encoding="utf-8"))
    plan = json.loads((args.results / "plan.json").read_text(encoding="utf-8"))
    expected = {(model, seed) for model in plan["models"] for seed in plan["seeds"]}
    for period in {row["period"] for row in evaluation}:
        observed = {(row["model"], row["seed"]) for row in evaluation if row["period"] == period}
        if observed != expected:
            raise ValueError("Evaluation does not contain every predeclared model and seed")
    rows = []
    for result in evaluation:
        if args.kind == "brazil":
            for horizon, metrics in result["metrics"].items():
                rows.append({"model": result["model"], "seed": result["seed"], "period": result["period"],
                             "horizon": int(horizon), **metrics})
        else:
            rows.append({**result, "horizon": "all24"})
    table = pd.DataFrame(rows)
    table.to_csv(args.results / "metrics_by_seed.csv", index=False)
    summary = table.groupby(["period", "model", "horizon"])[["mae", "rmse", "mape"]].agg(["mean", "std"])
    summary.columns = [f"{metric}_{statistic}" for metric, statistic in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(args.results / "metrics_mean_std.csv", index=False)
    paired = []
    for period in sorted(table.period.unique()):
        for candidate in ["seasonal_grid_aware", "seasonal_identity", "residual_identity"]:
            if candidate not in plan["models"]:
                continue
            for reference in ["grid_aware_lstm", "grid_aware_gru", *(["xgboost"] if args.kind == "hfed" else [])]:
                if reference in plan["models"]:
                    paired.append({"period": period, **matched_differences(args.results / period, candidate,
                        reference, plan["seeds"], 24 if args.kind == "brazil" else None)})
        if "seasonal_identity" in plan["models"] and "seasonal_grid_aware" in plan["models"]:
            paired.append({"period": period, **matched_differences(args.results / period, "seasonal_grid_aware",
                "seasonal_identity", plan["seeds"], 24 if args.kind == "brazil" else None)})
    (args.results / "paired_comparisons.json").write_text(json.dumps(paired, indent=2), encoding="utf-8")
    if args.kind == "hfed":
        hfed_details(args.results, plan)
    print(summary.to_string())
    print(json.dumps(paired, indent=2))


if __name__ == "__main__":
    main()
