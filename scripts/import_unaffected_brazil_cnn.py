"""Preserve the three eager CNN fits when correcting captured reference runs.

This copies no captured model. The CNN used an ordinary float LR, so the parent
plateau scheduler worked correctly. Source, configuration, stopping, and saved
optimizer checks prevent importing a different experiment by accident.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil

import torch

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", type=Path, default=ROOT / "results/grid_aware_confirm_v2")
parser.add_argument("--output", type=Path, default=ROOT / "results/grid_aware_confirm_v3")
args = parser.parse_args()
plan = json.loads((args.source / "plan.json").read_text(encoding="utf-8"))
for field, expected in {"epochs": 200, "patience": 10, "residual_batch_times": 32,
                        "seeds": [42, 123, 456], "data_md5": "f5003d072297deea746f63ff59ed4e47"}.items():
    if plan[field] != expected:
        raise ValueError(f"Unexpected source configuration: {field}")
if (args.source / "source_residual.py").read_bytes() != (ROOT / "src/models/stgnn/residual.py").read_bytes():
    raise ValueError("Residual architecture changed")
def definition(path, name):
    return ast.dump(next(node for node in ast.parse(path.read_text(encoding="utf-8")).body
                         if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name))
for name in ["ResidualAdapter", "make_batch"]:
    if definition(args.source / "source_benchmark_grid_aware.py", name) != definition(ROOT / "scripts/benchmark_grid_aware.py", name):
        raise ValueError(f"CNN/data adapter changed: {name}")
fits = json.loads((args.source / "fit_summary.json").read_text(encoding="utf-8"))
if args.output.exists() and any(args.output.iterdir()):
    raise FileExistsError("Import into a new, empty corrected-run directory")
args.output.mkdir(parents=True, exist_ok=True)
provenance = {"source": str(args.source), "reason": "Uncaptured CNN with float learning rate; unaffected by captured scheduler bug", "fits": []}
for seed in plan["seeds"]:
    key = f"residual_identity_{seed}"
    fit = next(row for row in fits if row["model"] == "residual_identity" and row["seed"] == seed)
    latest = torch.load(args.source / f"{key}_latest.pt", map_location="cpu", weights_only=False)
    best = torch.load(args.source / f"{key}_best.pt", map_location="cpu", weights_only=False)
    if any(group.get("capturable") or isinstance(group["lr"], torch.Tensor) for group in latest["optimizer"]["param_groups"]):
        raise ValueError("Source CNN used captured/tensor-LR execution")
    if fit["epochs"] - fit["best_epoch"] < plan["patience"] or best["best_epoch"] != fit["best_epoch"]:
        raise ValueError("Source CNN is not a completed stopped fit")
    if len(latest["history"]) != fit["epochs"]:
        raise ValueError("Source checkpoint history differs")
    files = {}
    for suffix in ["best.pt", "latest.pt", "history.json"]:
        source = args.source / f"{key}_{suffix}"
        files[source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
        shutil.copyfile(source, args.output / source.name)
    provenance["fits"].append({**fit, "sha256": files})
(args.output / "imported_fits.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
print(json.dumps(provenance, indent=2))
