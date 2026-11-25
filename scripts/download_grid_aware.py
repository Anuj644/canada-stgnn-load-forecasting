"""Download pinned public Grid-Aware assets and verify Zenodo checksums.

V1 dynamic_data.csv is intentionally excluded: it spans 2019-2020, whereas
the base paper needs 2017-2019. V2 supplies 2017-2020; static_data is in V1.
"""
import argparse
import hashlib
import json
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "data/external/grid_aware")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for record_id, names in [(17175783, ["static_data.csv"]),
                              (18595751, ["Updated_dynamic_data.csv", "grid_df.csv"])]:
        response = requests.get(f"https://zenodo.org/api/records/{record_id}", timeout=60)
        response.raise_for_status()
        record = response.json()
        (args.output / f"zenodo_{record_id}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        for item in record["files"]:
            if item["key"] not in names:
                continue
            target = args.output / item["key"]
            expected = item["checksum"].removeprefix("md5:")
            if target.exists():
                with target.open("rb") as handle:
                    verified = hashlib.file_digest(handle, "md5").hexdigest() == expected
                if verified:
                    print(f"Verified existing {target.name}", flush=True)
                    continue
            partial = target.with_suffix(".csv.partial")
            with requests.get(item["links"]["self"], stream=True, timeout=(30, 300)) as download:
                download.raise_for_status()
                with partial.open("wb") as handle:
                    for chunk in download.iter_content(1024 * 1024):
                        handle.write(chunk)
            with partial.open("rb") as handle:
                actual = hashlib.file_digest(handle, "md5").hexdigest()
            if actual != expected:
                raise ValueError(f"Checksum mismatch for {item['key']}")
            partial.replace(target)
            print(f"Downloaded and verified {target.name}", flush=True)


if __name__ == "__main__":
    main()
