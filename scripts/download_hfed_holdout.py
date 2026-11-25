"""Cache untouched 2025 HFED demand from the project's official SDMX endpoint."""
from pathlib import Path
import json
import hashlib
import sys

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.hfed_api import _get_csv, series_url


def main():
    config = yaml.safe_load((ROOT / "config/config.yaml").read_text(encoding="utf-8"))
    output = ROOT / "data/external/hfed_holdout_2025"
    output.mkdir(parents=True, exist_ok=True)
    meta = []
    for code in config["download"]["regions"]:
        path = output / f"{code}.csv"
        url = series_url(code, [config["download"]["demand_variables"][code]])
        if not path.exists():
            payload = _get_csv(url, "2025-01-01", "2025-12-31", timeout=180)
            if payload is None:
                meta.append({"region": code, "status": "no_observations", "url": url})
                continue
            path.with_suffix(".csv.partial").write_bytes(payload)
            path.with_suffix(".csv.partial").replace(path)
        frame = pd.read_csv(path, usecols=["TIME_PERIOD", "OBS_VALUE"])
        meta.append({"region": code, "url": url, "startPeriod": "2025-01-01", "endPeriod": "2025-12-31",
                     "rows": len(frame), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        print(f"Verified {code}: {len(frame)} raw observations", flush=True)
    (output / "provenance.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
