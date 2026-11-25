"""Descriptive, matched comparisons for the supplementary temporal controls."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.report_cross_dataset import matched_differences

primary = ROOT / "results/hfed_extension_eccc_v4"
controls = ROOT / "results/hfed_temporal_controls_v1"
plan = json.loads((controls / "plan.json").read_text())
if plan["cache_sha256"] != json.loads((primary / "plan.json").read_text())["cache_sha256"]:
    raise ValueError("Controls use a different data snapshot")
rows = []
for period in ["test2024", "test2025"]:
    for reference in plan["models"]:
        rows.append({"period": period, "scope": "Supplementary, after primary test inspection",
            **matched_differences(primary / period, "residual_identity", reference,
                plan["seeds"], None, reference_folder=controls / period)})
(controls / "paired_with_selected.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
print(json.dumps(rows, indent=2))
