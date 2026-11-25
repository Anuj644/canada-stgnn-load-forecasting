"""Cache historical ERA5 weather at eight documented Canadian reference points.

Source: https://open-meteo.com/en/docs/historical-weather-api (CC BY 4.0).
These are retrospective reanalysis covariates, not archived operational forecasts
or province-wide measurements. The model must never consume target-day weather.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.stgnn.graph_builder import REGION_COORDS

REGIONS = ["AB", "BC", "NB", "NL", "NS", "ON", "PE", "QC"]
VARIABLES = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/weather/open_meteo")
    parser.add_argument("--start", default="2018-12-24")
    parser.add_argument("--end", default="2024-12-31")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=Retry(total=4, backoff_factor=2,
                  status_forcelist=[429, 500, 502, 503, 504], respect_retry_after_header=True)))
    frames, requests_made = [], []
    for region in REGIONS:
        lat, lon = REGION_COORDS[region]
        for year in range(start.year, end.year + 1):
            first, last = max(start, pd.Timestamp(year, 1, 1)), min(end, pd.Timestamp(year, 12, 31))
            params = {"latitude": lat, "longitude": lon, "start_date": str(first.date()),
                      "end_date": str(last.date()), "hourly": ",".join(VARIABLES),
                      "timezone": "UTC", "models": "era5", "wind_speed_unit": "ms"}
            path = args.output / f"{region}_{first.date()}_{last.date()}_era5.json"
            if not path.exists():
                response = session.get("https://archive-api.open-meteo.com/v1/archive", params=params,
                                       timeout=(30, 180))
                response.raise_for_status()
                payload = response.json()
                if "hourly" not in payload:
                    raise ValueError(f"Missing hourly data: {region} {year}")
                partial = path.with_suffix(".json.partial")
                partial.write_text(json.dumps(payload), encoding="utf-8")
                partial.replace(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            frame = pd.DataFrame(payload["hourly"]).rename(columns={"time": "time_utc"})
            frame["time_utc"] = pd.to_datetime(frame.time_utc)
            expected = pd.date_range(first, last + pd.Timedelta(hours=23), freq="h")
            if not pd.DatetimeIndex(frame.time_utc).equals(expected) or frame[VARIABLES].isna().any().any():
                raise ValueError(f"Incomplete weather: {region} {year}")
            frame["region"] = region
            frames.append(frame)
            requests_made.append({"region": region, "params": params, "file": path.name,
                                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                  "returned_latitude": payload["latitude"],
                                  "returned_longitude": payload["longitude"],
                                  "hourly_units": payload["hourly_units"]})
            print(f"Verified {region} {year}: {len(frame)} hourly records", flush=True)
    weather = pd.concat(frames, ignore_index=True).sort_values(["region", "time_utc"])
    if weather.duplicated(["region", "time_utc"]).any():
        raise ValueError("Duplicate weather keys")
    weather.to_parquet(args.output / "historical_weather.parquet", index=False)
    meta = {"source": "https://open-meteo.com/en/docs/historical-weather-api", "model": "ERA5",
            "license": "CC BY 4.0; credit Open-Meteo and Copernicus Climate Change Service",
            "limitations": ["One representative point per province; not spatially aggregated provincial weather.",
                            "Retrospective reanalysis; publication-time availability is not established.",
                            "Use only past weather as an experimental covariate, never realized target-day weather."],
            "requests": requests_made}
    (args.output / "provenance.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved {len(weather)} records", flush=True)


if __name__ == "__main__":
    main()
