"""Fetch forecast history and comparison readings from the HFED API."""

from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import pandas as pd
import requests

from hfed_dataset_downloader import BASE
from src.data.upload_forecast import read_observations, target_hours


def fetch_forecast_observations(region, forecast_date, config):
    targets = target_hours(region, forecast_date)
    start = targets[0] - pd.Timedelta(hours=336)
    end = targets[-1] + pd.Timedelta(hours=1)
    params = {
        "startPeriod": start.date().isoformat(),
        "endPeriod": end.date().isoformat(),
        "format": "csv",
        "dimensionAtObservation": "AllDimensions",
    }

    def fetch(code):
        url = f"{BASE}/data/CCEI,DF_HFED_{code},1.0/all"
        with requests.get(url, params=params, timeout=(5, 30)) as response:
            if response.status_code == 204 or (
                response.status_code == 404 and response.text.strip() == "NoRecordsFound"
            ):
                return None
            response.raise_for_status()
            return read_observations([BytesIO(response.content)], config)

    regions = config["download"]["regions"]
    frames = []
    unavailable = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {code: executor.submit(fetch, code) for code in regions}
        for code, future in futures.items():
            try:
                data = future.result()
            except (requests.RequestException, ValueError) as exc:
                if code == region:
                    raise ValueError(
                        f"Could not download usable HFED data for {code}. "
                        "Try again or switch to CSV upload."
                    ) from exc
                data = None
            if data is None:
                if code == region:
                    raise ValueError(f"HFED returned no demand observations for {code} in the requested range.")
                unavailable.append(code)
            else:
                frames.append(data)
    observations = pd.concat(frames, ignore_index=True)
    times = observations["time_utc"]
    history = observations.loc[times.ge(start) & times.lt(targets[0])]
    actuals = observations.loc[times.ge(targets[0]) & times.lt(end)]
    return history, actuals, unavailable
