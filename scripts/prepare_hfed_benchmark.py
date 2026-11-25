"""Prepare the HFED cache without loading PyTorch or creating a GPU context."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.day_ahead_benchmark import prepared_data

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--cache", type=Path, default=ROOT / "data/processed/hfed_extension_eccc_v2")
args = parser.parse_args()
data = prepared_data(ROOT, args.cache)
print(json.dumps({"examples": {k: len(v) for k, v in data["splits"].items()},
                  "exclusions": data["exclusions"]}, indent=2), flush=True)
