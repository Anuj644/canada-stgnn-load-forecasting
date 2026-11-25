"""Cache official Canadian hourly station observations, selected on 2019 coverage.

Station choice uses distance to documented regional reference points and training-
period weather coverage only, never demand labels or future forecast performance.
Source: https://api.weather.gc.ca/collections/climate-hourly
"""
from __future__ import annotations

import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.stgnn.graph_builder import REGION_COORDS, _haversine_km

REGIONS = ["AB", "BC", "NB", "NL", "NS", "ON", "PE", "QC"]
BASE = "https://api.weather.gc.ca/collections"
FIELDS = ["UTC_DATE", "TEMP", "RELATIVE_HUMIDITY", "WIND_SPEED"]


def main():
    output = ROOT / "data/weather/eccc"
    output.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=Retry(total=4, backoff_factor=2,
                   status_forcelist=[429, 500, 502, 503, 504], respect_retry_after_header=True)))

    def year_data(identifier, year):
        path = output / f"{identifier}_{year}.csv"
        params = {"CLIMATE_IDENTIFIER": identifier, "UTC_YEAR": year, "limit": 10000,
                  "properties": ",".join(FIELDS), "f": "csv"}
        if not path.exists():
            response = session.get(f"{BASE}/climate-hourly/items", params=params, timeout=(30, 120))
            response.raise_for_status()
            if response.status_code == 204 or not response.content.strip():
                return None, params, path
            path.with_suffix(".csv.partial").write_bytes(response.content)
            path.with_suffix(".csv.partial").replace(path)
        frame = pd.read_csv(path)
        if not set(FIELDS).issubset(frame):
            return None, params, path
        frame["UTC_DATE"] = pd.to_datetime(frame.UTC_DATE)
        if len(frame) >= 10000 or (len(frame) and not frame.UTC_DATE.dt.year.eq(year).all()):
            raise ValueError("Unexpected API truncation or temporal filter behavior")
        return frame, params, path

    selections, sources, frames = [], [], []
    for region in REGIONS:
        lat, lon = REGION_COORDS[region]
        station_path = output / f"stations_{region}.json"
        if not station_path.exists():
            response = session.get(f"{BASE}/climate-stations/items",
                params={"bbox": f"{lon-1},{lat-1},{lon+1},{lat+1}", "limit": 1000, "f": "json"}, timeout=60)
            response.raise_for_status()
            station_path.write_text(json.dumps(response.json()), encoding="utf-8")
        stations = json.loads(station_path.read_text(encoding="utf-8"))
        if any(link.get("rel") == "next" for link in stations.get("links", [])):
            raise ValueError("Station inventory truncated; reduce bounding box or paginate")
        candidates = []
        for feature in stations["features"]:
            p = feature["properties"]
            first, last = p.get("HLY_FIRST_DATE"), p.get("HLY_LAST_DATE")
            if p.get("PROV_STATE_TERR_CODE") != region or p.get("HAS_HOURLY_DATA") != "Y":
                continue
            if not first or not last or first > "2019-01-01" or last < "2025-12-31":
                continue
            x, y = feature["geometry"]["coordinates"][:2]
            candidates.append((_haversine_km(lat, lon, y, x), p))
        candidates.sort(key=lambda item: item[0])
        selected = None
        for distance, station in candidates:
            identifier = str(station["CLIMATE_IDENTIFIER"])
            train_weather, _, _ = year_data(identifier, 2019)
            if train_weather is None or len(train_weather) < 0.8 * 8760:
                continue
            coverage = train_weather[["TEMP", "RELATIVE_HUMIDITY", "WIND_SPEED"]].notna().sum() / 8760
            if coverage.min() >= 0.8:
                selected = station
                selections.append({"region": region, "station": station, "distance_km": distance,
                                   "selection_2019_coverage": coverage.to_dict()})
                break
        if selected is None:
            raise ValueError(f"No sufficiently complete training-period station near {region}")
        identifier = str(selected["CLIMATE_IDENTIFIER"])
        for year in range(2019, 2026):
            frame, params, path = year_data(identifier, year)
            if frame is None:
                raise ValueError(f"No observations for selected station {identifier} in {year}")
            frame = frame.rename(columns={"UTC_DATE": "time_utc", "TEMP": "temperature_c",
                      "RELATIVE_HUMIDITY": "relative_humidity", "WIND_SPEED": "wind_speed"})
            frame = frame.groupby("time_utc")[["temperature_c", "relative_humidity", "wind_speed"]].mean().reset_index()
            frame["region"] = region
            frames.append(frame)
            sources.append({"region": region, "year": year, "params": params, "file": path.name,
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "n_rows": len(frame)})
            print(f"Verified {region} {year}, {selected['STATION_NAME']}: {len(frame)} records", flush=True)
    weather = pd.concat(frames, ignore_index=True).sort_values(["region", "time_utc"])
    if weather.duplicated(["region", "time_utc"]).any():
        raise ValueError("Duplicate weather keys")
    weather.to_parquet(output / "historical_weather.parquet", index=False)
    meta = {"source": f"{BASE}/climate-hourly", "selections": selections, "files": sources,
            "selection_policy": "Nearest same-province station spanning 2019-2025 with >=80% coverage of all three variables in 2019",
            "limitations": ["One station is a representative proxy, not province-wide weather.",
                            "Archived observations may be revised; exact historical publication latency is not reproduced.",
                            "Assume a one-hour reporting buffer and never interpolate from future observations."]}
    (output / "provenance.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved {len(weather)} official station observations", flush=True)


if __name__ == "__main__":
    main()
