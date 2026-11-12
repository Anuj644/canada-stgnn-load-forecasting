"""Download every available HFED region and electricity variable as CSV.

Install:  python -m pip install requests tqdm
Run:      python download_hfed.py
Preview:  python download_hfed.py --list
Subset:   python download_hfed.py --provinces ON QC --start 2026-01-01 --end 2026-01-31
Refresh:  python download_hfed.py --overwrite

With no date arguments, requests ALL available history, frequencies, variables,
and counterpart areas. No resampling, deduplication, or timezone conversion.
Results: data/hfed/all_history/<region>/{all.csv,<VARIABLE>.csv,manifest.json}.
Date-limited runs use a separate subfolder. Completed regions are skipped on
rerun; use --overwrite to fetch updates. Interrupted downloads restart their
region rather than attempting HTTP byte-range resume. Requires Python 3.10+.

Source: https://energy-information.canada.ca/en/resources/high-frequency-electricity-data
API: https://api.statcan.gc.ca/hfed-dehf/sdmx/rest/dataflow/CCEI/all/latest
"""

import argparse
import csv
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from contextlib import ExitStack
from datetime import UTC, date, datetime
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

# Raise the csv field-size limit — ON's all.csv has very wide rows that exceed
# the default 131 072-character limit.
csv.field_size_limit(10 * 1024 * 1024)  # 10 MB per field

BASE = "https://api.statcan.gc.ca/hfed-dehf/sdmx/rest"
NS = {
    "s": "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/structure",
    "c": "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/common",
}
LANG = "{http://www.w3.org/XML/1998/namespace}lang"


def make_session():
    session = requests.Session()
    retry = Retry(
        total=4,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        respect_retry_after_header=True,
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers["User-Agent"] = "HFED-CSV-downloader/1.0 (Python requests)"
    return session


def discover_regions(session, timeout):
    """Discover actual published flows; do not guess province IDs or versions."""
    with session.get(f"{BASE}/dataflow/CCEI/all/latest", timeout=(30, timeout)) as r:
        r.raise_for_status()
        root = ET.fromstring(r.content)
    regions = []
    for flow in root.findall(".//s:Dataflow", NS):
        flow_id = flow.attrib["id"]
        if not re.fullmatch(r"DF_HFED_[A-Z]{2}", flow_id):
            continue
        name = next(
            (n.text for n in flow.findall("c:Name", NS) if n.get(LANG) == "en"),
            flow_id,
        )
        regions.append(
            {
                "region": flow_id.removeprefix("DF_HFED_"),
                "name": name,
                "flow": flow_id,
                "agency": flow.attrib["agencyID"],
                "version": flow.attrib["version"],
            }
        )
    if not regions:
        raise ValueError("The catalogue returned no HFED dataflows.")
    return sorted(regions, key=lambda item: item["region"])


def download_csv(session, url, params, destination, timeout):
    """Stream to disk; retry interrupted transfers from the beginning."""
    for attempt in range(3):
        try:
            with session.get(
                url, params=params, stream=True, timeout=(30, timeout)
            ) as r:
                if r.status_code == 204:
                    return False, r.url
                # HFED also uses this exact 404 body for an empty date range.
                if r.status_code == 404 and r.text.strip() == "NoRecordsFound":
                    return False, r.url
                r.raise_for_status()
                # iter_content decompresses responses, so compressed wire sizes
                # cannot serve as a total for the bytes written to disk.
                length = r.headers.get("Content-Length", "")
                total = (
                    int(length)
                    if length.isdigit() and not r.headers.get("Content-Encoding")
                    else None
                )
                with (
                    destination.open("wb") as output,
                    tqdm(
                        total=total,
                        desc=f"{destination.parent.name} download",
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        position=1,
                        leave=False,
                        dynamic_ncols=True,
                    ) as progress,
                ):
                    for chunk in r.iter_content(chunk_size=64 * 1024):
                        if chunk:
                            output.write(chunk)
                            progress.update(len(chunk))
                return True, r.url
        except requests.HTTPError:
            raise
        except requests.RequestException:
            if attempt == 2:
                raise
            tqdm.write(f"  Transfer failed; retrying ({attempt + 1}/2)...")
            time.sleep(2 ** (attempt + 1))


def split_variables(source, folder):
    """Split on ENERGY_FLOWS, retaining every column and all other dimensions."""
    variables = {}
    with ExitStack() as stack:
        input_file = stack.enter_context(
            source.open("r", encoding="utf-8-sig", newline="")
        )
        reader = csv.DictReader(input_file)
        fields = reader.fieldnames
        if not fields or not {"ENERGY_FLOWS", "TIME_PERIOD", "OBS_VALUE"}.issubset(
            fields
        ):
            raise ValueError(
                "Expected SDMX CSV columns. The API may have returned an error page."
            )
        rows = stack.enter_context(
            tqdm(
                reader,
                desc=f"{folder.name} splitting CSV",
                unit=" rows",
                position=1,
                leave=False,
                dynamic_ncols=True,
            )
        )
        for row in rows:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(
                    "Malformed or truncated CSV row; region was not marked complete."
                )
            code = row["ENERGY_FLOWS"]
            if not re.fullmatch(r"[A-Z0-9_]+", code):
                raise ValueError(f"Unexpected variable code: {code!r}")
            if code not in variables:
                path = folder / f"{code}.csv.part"
                handle = stack.enter_context(
                    path.open("w", encoding="utf-8", newline="")
                )
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                variables[code] = {"writer": writer, "rows": 0, "file": f"{code}.csv"}
            entry = variables[code]
            entry["writer"].writerow(row)
            entry["rows"] += 1
    # Publish only after the whole response has been parsed successfully.
    for entry in variables.values():
        (folder / (entry["file"] + ".part")).replace(folder / entry["file"])
    return [
        {"variable": code, "file": item["file"], "rows": item["rows"]}
        for code, item in sorted(variables.items())
    ]


def is_complete(folder, region, params):
    path = folder / "manifest.json"
    if not path.exists():
        return False
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
        return (
            saved.get("status") == "complete"
            and saved.get("dataflow") == region
            and saved.get("parameters") == params
            and (folder / "all.csv").is_file()
            and all((folder / entry["file"]).is_file() for entry in saved["datasets"])
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def fetch_region(session, region, params, folder, args):
    if not args.overwrite and is_complete(folder, region, params):
        tqdm.write(
            f"[{region['region']}] Already complete; skipping. "
            "Use --overwrite to refresh."
        )
        return
    folder.mkdir(parents=True, exist_ok=True)
    # Invalidate any old completion marker before replacing files.
    (folder / "manifest.json").unlink(missing_ok=True)
    flow = f"{region['agency']},{region['flow']},{region['version']}"
    url = f"{BASE}/data/{flow}/all"
    tqdm.write(f"[{region['region']}] Downloading all variables and frequencies...")
    partial = folder / "all.csv.part"
    has_data, request_url = download_csv(session, url, params, partial, args.timeout)
    if not has_data:
        partial.unlink(missing_ok=True)
        tqdm.write(f"[{region['region']}] No observations for the requested range.")
        return
    datasets = split_variables(partial, folder)
    partial.replace(folder / "all.csv")
    manifest = {
        "status": "complete",
        "dataflow": region,
        "parameters": params,
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
        "url": request_url,
        "datasets": datasets,
        "total_rows": sum(item["rows"] for item in datasets),
    }
    manifest_part = folder / "manifest.json.part"
    manifest_part.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest_part.replace(folder / "manifest.json")
    tqdm.write(
        f"[{region['region']}] Saved {len(datasets)} variables, "
        f"{manifest['total_rows']:,} rows."
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parent / "data" / "hfed"
    )
    parser.add_argument(
        "--provinces",
        nargs="+",
        type=str.upper,
        help="Region codes, e.g. ON QC AB YK; default: all published regions",
    )
    parser.add_argument(
        "--start", type=date.fromisoformat, help="Optional first date, YYYY-MM-DD"
    )
    parser.add_argument(
        "--end", type=date.fromisoformat, help="Optional last date, YYYY-MM-DD"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=900,
        help="Read timeout in seconds (default: 900)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Download again even if a region completed previously",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List published regions without downloading observations",
    )
    args = parser.parse_args()
    if args.start and args.end and args.start > args.end:
        parser.error("--start must be on or before --end")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    params = {"dimensionAtObservation": "AllDimensions", "format": "csv"}
    if args.start:
        params["startPeriod"] = args.start.isoformat()
    if args.end:
        params["endPeriod"] = args.end.isoformat()
    scope = (
        "all_history"
        if not (args.start or args.end)
        else f"{args.start or 'earliest'}_to_{args.end or 'latest'}"
    )
    failures = []
    with make_session() as session:
        regions = discover_regions(session, args.timeout)
        if args.provinces:
            unknown = set(args.provinces) - {item["region"] for item in regions}
            if unknown:
                parser.error(
                    f"No published HFED flow for: {', '.join(sorted(unknown))}. "
                    "Use --list."
                )
            regions = [item for item in regions if item["region"] in args.provinces]
        if args.list:
            for item in regions:
                print(
                    f"{item['region']}: {item['name']} "
                    f"({item['flow']}, v{item['version']})"
                )
            return 0
        output = args.output.resolve() / scope
        print(f"Output: {output}", flush=True)
        for region in tqdm(
            regions, desc="Regions", unit="region", position=0, dynamic_ncols=True
        ):
            try:
                fetch_region(session, region, params, output / region["region"], args)
            except (requests.RequestException, OSError, ValueError) as exc:
                failures.append(region["region"])
                tqdm.write(f"[{region['region']}] FAILED: {exc}", file=sys.stderr)
        if failures:
            print(
                f"Incomplete regions: {', '.join(failures)}. "
                "Rerun the same command to retry.",
                file=sys.stderr,
            )
            return 1
    print("Finished.")
    return 0


if __name__ == "__main__":
    # Windows redirected consoles may otherwise fail on non-ASCII folder names.
    sys.stdout.reconfigure(errors="backslashreplace")
    sys.stderr.reconfigure(errors="backslashreplace")
    try:
        sys.exit(main())
    except (requests.RequestException, ET.ParseError, ValueError) as exc:
        print(f"Unable to read HFED catalogue: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print(
            "\nInterrupted. Rerun the same command to retry unfinished regions.",
            file=sys.stderr,
        )
        sys.exit(130)
