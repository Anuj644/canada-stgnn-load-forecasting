"""Report the proposal's demand -> weather -> calendar -> graph comparisons."""
import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.report_cross_dataset import matched_differences

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--full", type=Path, required=True)
parser.add_argument("--demand", type=Path, required=True)
parser.add_argument("--weather", type=Path, required=True)
args = parser.parse_args()
experiments = [("demand", args.demand, "seasonal_identity"),
               ("demand_weather", args.weather, "seasonal_identity"),
               ("demand_weather_calendar", args.full, "seasonal_identity"),
               ("demand_weather_calendar_graph", args.full, "seasonal_grid_aware")]
rows, plans = [], []
for label, folder, model in experiments:
    plan = json.loads((folder / "plan.json").read_text(encoding="utf-8"))
    plans.append(plan)
    evaluation = json.loads((folder / "evaluation.json").read_text(encoding="utf-8"))
    rows.extend({**row, "features": label} for row in evaluation if row["model"] == model)
if len({p["cache_sha256"] for p in plans}) != 1 or any(p["seeds"] != plans[0]["seeds"] for p in plans):
    raise ValueError("Ablations must share a data snapshot and seed set")
table = pd.DataFrame(rows)
table.to_csv(args.full / "ablation_metrics_by_seed.csv", index=False)
summary = table.groupby(["period", "features"])[["mae", "rmse", "mape"]].agg(["mean", "std"])
summary.to_csv(args.full / "ablation_metrics_mean_std.csv")
paired = []
for period in ["test2024", "test2025"]:
    for before, after in zip(experiments, experiments[1:]):
        comparison = matched_differences(after[1] / period, after[2], before[2], plans[0]["seeds"], None,
                                        reference_folder=before[1] / period)
        paired.append({"period": period, "before": before[0], "after": after[0], **comparison})
(args.full / "ablation_paired_comparisons.json").write_text(json.dumps(paired, indent=2), encoding="utf-8")
print(summary.to_string())
print(json.dumps(paired, indent=2))
