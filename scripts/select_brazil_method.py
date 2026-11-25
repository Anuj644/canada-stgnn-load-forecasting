"""Freeze a method choice from completed Brazilian validation runs only."""
import argparse
import json
from pathlib import Path

import pandas as pd

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--results", type=Path, required=True)
args = parser.parse_args()
plan = json.loads((args.results / "plan.json").read_text(encoding="utf-8"))
fits = json.loads((args.results / "fit_summary.json").read_text(encoding="utf-8"))
expected = {(model, seed) for model in plan["models"] for seed in plan["seeds"]}
if {(row["model"], row["seed"]) for row in fits} != expected:
    raise ValueError("All frozen model/seed fits must finish before method selection")
rows = []
for fit in fits:
    history = json.loads((args.results / f"{fit['model']}_{fit['seed']}_history.json").read_text(encoding="utf-8"))
    selected = history[fit["best_epoch"] - 1]
    rows.append({"model": fit["model"], "seed": fit["seed"], "best_epoch": fit["best_epoch"],
                 "validation_24h_mae": selected["validation_mae_1_6_24"][2]})
table = pd.DataFrame(rows)
table.to_csv(args.results / "validation_selection.csv", index=False)
means = table.groupby("model").validation_24h_mae.mean()
candidates = [model for model in ["seasonal_identity", "seasonal_grid_aware", "residual_identity"] if model in means]
selected = means.loc[candidates].idxmin()
result = {"selected_method": selected, "criterion": "Lowest mean 24h validation MAE across all predeclared seeds",
          "validation_period": "2019-01-01 through 2019-06-30", "mean_validation_mae": means.to_dict(),
          "test_metrics_used": False,
          "hfed_models": ["grid_aware_lstm", "grid_aware_gru", "seasonal_identity", "seasonal_grid_aware", "residual_identity",
                          "xgboost", "naive_day_available", "naive_week"]}
path = args.results / "method_selection.json"
if path.exists():
    raise FileExistsError("Method selection already frozen")
path.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
