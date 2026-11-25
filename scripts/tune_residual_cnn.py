"""Validation-only tuning of the selected residual CNN, then one frozen-split fit of the winner.

search: rolling-origin validation (Jan-Jun of 2022, 2023, 2024; train = earlier dates, purged at the
        boundary like the benchmark). One JSON per candidate/fold/seed, so the run can resume.
final:  fit the chosen candidate on the frozen split (3 seeds), write test2024/test2025 forecasts.
No test period is read during `search`.

Usage:
    python scripts/tune_residual_cnn.py search [--candidates a b] [--folds A B C] [--seeds 42 123 456]
    python scripts/tune_residual_cnn.py aggregate
    python scripts/tune_residual_cnn.py final --candidate weather
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.day_ahead_benchmark import prepared_data
from src.models.cnn_tuning import ResidualCNN, batch_loss, fit, make_tensors, predict, sample_weights, score

OUT = ROOT / "results/hfed_cnn_tuning_v1"
CACHE = ROOT / "data/processed/hfed_extension_eccc_v2"
SEEDS = [42, 123, 456]
CANDIDATES = {
    "baseline": {"model": {}, "power": 0.0},
    "weather": {"model": {"weather": True}, "power": 0.0},
    "weighted_0.5": {"model": {}, "power": 0.5},
    "weighted_1.0": {"model": {}, "power": 1.0},
    "wide": {"model": {"width": 48, "hidden": 64, "head": 128}, "power": 0.0},
    "weather_weighted_0.5": {"model": {"weather": True}, "power": 0.5},
}
FOLDS = {"A": ("2022-01-01", "2022-07-01"), "B": ("2023-01-01", "2023-07-01"), "C": None}


def fold_ids(data, name):
    if name == "C":
        return data["splits"]["train"], data["splits"]["validation"]
    start, stop = map(pd.Timestamp, FOLDS[name])
    records = data["records"]
    dates = pd.DatetimeIndex(records.forecast_date)
    ends = pd.DatetimeIndex(records.target_start) + pd.Timedelta(hours=24)
    validation = (dates >= start) & (dates < stop)
    train = (dates < start) & (ends <= records.loc[validation, "issued_at"].min())
    return np.flatnonzero(train), np.flatnonzero(validation)


def build(data, spec, seed):
    torch.manual_seed(seed)
    return ResidualCNN(data, **spec["model"]).cuda() if torch.cuda.is_available() else ResidualCNN(data, **spec["model"])


def run_one(data, tensors, name, fold, seed, spec):
    train, validation = fold_ids(data, fold)
    weights = sample_weights(tensors, torch.as_tensor(train, device=tensors["nodes"].device), spec["power"]) \
        if spec["power"] else None
    model = build(data, spec, seed)
    info = fit(model, tensors, train, validation, seed, weights)
    prediction = predict(model, tensors, validation)
    result = score(prediction, data["actual"][validation], data["nodes"][validation], data["states"])
    return {"candidate": name, "fold": fold, "seed": seed, "n_train": len(train), "n_validation": len(validation),
            **info, **result}


def search(args):
    data = prepared_data(ROOT, CACHE)
    tensors = make_tensors(data, "cuda" if torch.cuda.is_available() else "cpu")
    (OUT / "search").mkdir(parents=True, exist_ok=True)
    for name in args.candidates:
        for fold in args.folds:
            for seed in args.seeds:
                path = OUT / "search" / f"{name}_{fold}_{seed}.json"
                if path.exists():
                    continue
                start = time.time()
                result = run_one(data, tensors, name, fold, seed, CANDIDATES[name])
                result["seconds"] = round(time.time() - start, 1)
                path.write_text(json.dumps(result, indent=2), encoding="utf-8")
                print(f"{name} fold {fold} seed {seed}: MAE {result['mae']:.2f} MAPE {result['macro_mape']:.3f} "
                      f"epochs {result['epochs']} {result['seconds']}s", flush=True)


def aggregate(_):
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((OUT / "search").glob("*.json"))]
    frame = pd.DataFrame(rows).drop(columns="by_province_mae")
    per_fold = frame.groupby(["candidate", "fold"])[["mae", "macro_mape"]].mean().unstack("fold")
    per_fold.columns = [f"{m}_{f}" for m, f in per_fold.columns]
    seed_mean = frame.groupby(["candidate", "seed"])[["mae", "macro_mape"]].mean()
    summary = frame.groupby("candidate")[["mae", "macro_mape"]].mean()
    summary["mae_seed_std"] = seed_mean.groupby("candidate").mae.std()
    summary["n_runs"] = frame.groupby("candidate").size()
    table = summary.join(per_fold)
    base = table.loc["baseline"]
    table["mae_vs_baseline_pct"] = (table.mae / base.mae - 1) * 100
    table["accepted"] = (table.mae < base.mae) & (table.macro_mape <= base.macro_mape) & (table.index != "baseline")
    table.round(4).to_csv(OUT / "search_summary.csv")
    print(table.round(3).to_string())


def save_checkpoint(model, data, name, seed, info):
    """Same layout as the frozen benchmark checkpoints, so the dashboard loads it unchanged."""
    torch.save({"state_dict": model.state_dict(), "model": name, "seed": seed, "best_epoch": info["best_epoch"],
                "validation_loss": info["validation_loss"], "features": data["features"], "states": data["states"],
                "scalers": {k: data[k] for k in ["center", "scale", "feature_center", "feature_scale",
                                                 "target_center", "target_scale"]}},
               OUT / f"{name}_{seed}_best.pt")


def final(args):
    data = prepared_data(ROOT, CACHE)
    tensors = make_tensors(data, "cuda" if torch.cuda.is_available() else "cpu")
    spec = CANDIDATES[args.candidate]
    records, train, validation = data["records"], data["splits"]["train"], data["splits"]["validation"]
    weights = sample_weights(tensors, torch.as_tensor(train, device=tensors["nodes"].device), spec["power"]) \
        if spec["power"] else None
    name = f"residual_{args.candidate}"
    metrics = []
    for seed in args.seeds:
        if (OUT / f"{name}_{seed}_best.pt").exists():
            raise SystemExit("final outputs exist; refusing to overwrite")
        model = build(data, spec, seed)
        info = fit(model, tensors, train, validation, seed, weights)
        save_checkpoint(model, data, name, seed, info)
        for period in ["test2024", "test2025"]:
            ids = data["splits"][period]
            prediction = predict(model, tensors, ids)
            actual = data["actual"][ids]
            result = score(prediction, actual, data["nodes"][ids], data["states"])
            metrics.append({"model": name, "seed": seed, "period": period, **info,
                            **{k: v for k, v in result.items() if k != "by_province_mae"},
                            **{f"mae_{k}": v for k, v in result["by_province_mae"].items()}})
            rows = []
            for k, i in enumerate(ids):
                rec = records.iloc[i]
                for h in range(24):
                    target = pd.Timestamp(rec.target_start) + pd.Timedelta(hours=h)
                    rows.append((rec.issued_at, rec.region, h + 1, target, float(actual[k, h]), float(prediction[k, h])))
            out = OUT / period
            out.mkdir(exist_ok=True)
            pd.DataFrame(rows, columns=["issued_at", "state", "horizon", "target_at", "actual", "predicted"]) \
                .to_parquet(out / f"predictions_{name}_{seed}.parquet", index=False)
        print(seed, metrics[-2]["mae"], metrics[-1]["mae"], flush=True)
    pd.DataFrame(metrics).to_csv(OUT / f"final_metrics_{name}.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("search")
    s.add_argument("--candidates", nargs="+", default=list(CANDIDATES), choices=list(CANDIDATES))
    s.add_argument("--folds", nargs="+", default=list(FOLDS), choices=list(FOLDS))
    s.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    s.set_defaults(fn=search)
    sub.add_parser("aggregate").set_defaults(fn=aggregate)
    f = sub.add_parser("final")
    f.add_argument("--candidate", required=True, choices=list(CANDIDATES))
    f.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    f.set_defaults(fn=final)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
