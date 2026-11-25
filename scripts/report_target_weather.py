"""Report realized target-day extreme-temperature errors, without refitting."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.target_weather import classify_target_days

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--results", type=Path, default=ROOT / "results/hfed_extension_eccc_v4")
args = parser.parse_args()
plan = json.loads((args.results / "plan.json").read_text())
cache = Path(plan["cache"])
if hashlib.sha256((cache / "examples.npz").read_bytes()).hexdigest() != plan["cache_sha256"]:
    raise ValueError("Data cache changed")
records = pd.read_parquet(cache / "records.parquet")
metadata = json.loads((cache / "metadata.json").read_text())
with np.load(cache / "examples.npz") as archive:
    train_ids = archive["split_train"]
weather = pd.read_parquet(ROOT / "data/weather/eccc/historical_weather.parquet")
labels, thresholds = classify_target_days(records, train_ids, weather, metadata["states"])
rows = []
for period in ["test2024", "test2025"]:
    for model in plan["models"]:
        for seed in plan["seeds"]:
            frame = pd.read_parquet(args.results / period / f"predictions_{model}_{seed}.parquet")
            frame = frame.merge(labels, on=["issued_at", "state"], how="left", validate="many_to_one")
            if frame.regime.isna().any():
                raise ValueError("Unmatched diagnostic keys")
            for regime, group in frame.groupby("regime"):
                error = group.predicted - group.actual
                valid = group.actual.abs().ge(10)
                rows.append({"period": period, "model": model, "seed": seed, "regime": regime,
                    "n_obs": len(group), "forecast_days": group[["issued_at", "state"]].drop_duplicates().shape[0],
                    "mae": error.abs().mean(), "rmse": np.sqrt(np.square(error).mean()),
                    "mape": (error[valid].abs() / group.actual[valid].abs()).mean() * 100})
pd.DataFrame(rows).to_csv(args.results / "realized_weather_metrics_by_seed.csv", index=False)
(args.results / "realized_weather_thresholds.json").write_text(json.dumps({
    "definition": "Observed mean temperature over the 24 target-day hours; at least 12 observations. Cold/hot: provincial training-only 5th/95th percentiles.",
    "scope": "Retrospective diagnostic added for the proposal/dashboard after primary results were inspected. Realized target-day weather is not a forecasting input and does not select checkpoints. No additional confirmatory significance claim.",
    "thresholds": thresholds}, indent=2), encoding="utf-8")
print(json.dumps({"rows": len(rows), "status": "reported without refitting"}))
