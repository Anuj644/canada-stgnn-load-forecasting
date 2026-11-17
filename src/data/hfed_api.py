"""Fetch demand history and comparison readings from the HFED API.

Requests use a filtered SDMX key (``.CA_{code}._Z.{flows}``) so the API only
returns the province-total series that the models use. The unfiltered ``all``
key also returns every generator and price series; for Ontario that response
regularly exceeds the read timeout.
"""

from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from time import sleep

import pandas as pd
import requests

from hfed_dataset_downloader import BASE
from src.data.upload_forecast import read_observations, target_hours


def series_url(code, flows):
    """URL for province ``code`` restricted to the given energy-flow codes."""
    return f"{BASE}/data/CCEI,DF_HFED_{code},1.0/.CA_{code}._Z.{'+'.join(flows)}"


def _get_csv(url, start, end, timeout):
    params = {
        "startPeriod": pd.Timestamp(start).date().isoformat(),
        "endPeriod": pd.Timestamp(end).date().isoformat(),
        "format": "csv",
        "dimensionAtObservation": "AllDimensions",
    }
    with requests.Session() as session:
        for attempt in range(3):
            if attempt:
                sleep(0.5 * 2 ** (attempt - 1))
            try:
                with session.get(url, params=params, timeout=(10, timeout)) as response:
                    if response.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                        continue
                    if response.status_code == 204 or (
                        response.status_code == 404 and response.text.strip() == "NoRecordsFound"
                    ):
                        return None
                    response.raise_for_status()
                    return response.content
            except (requests.Timeout, requests.ConnectionError, requests.exceptions.ChunkedEncodingError):
                if attempt == 2:
                    raise


def fetch_demand(code, start, end, config, timeout=300):
    """Hourly province demand in MW, binned to top-of-hour UTC like the training data.

    Returns a DataFrame with ``region``, ``time_utc`` and ``demand_mw``, or None when
    HFED has no readings for the range.
    """
    flow = config["download"]["demand_variables"][code]
    content = _get_csv(series_url(code, [flow]), start, end, timeout)
    if content is None:
        return None
    readings = read_observations([BytesIO(content)], config)
    hourly = readings.groupby(readings["time_utc"].dt.floor("h"))["demand_mw"].mean()
    hourly = hourly.dropna()
    if hourly.empty:
        return None
    return pd.DataFrame({"region": code, "time_utc": hourly.index, "demand_mw": hourly.to_numpy()})


def fetch_demand_history(regions, start, end, config, timeout=300):
    """Fetch hourly demand for several provinces in parallel.

    Returns ``(demand, unavailable)``. The second return value maps each unavailable
    province to a message explaining why no readings could be loaded.
    """
    def fetch(code):
        try:
            frame = fetch_demand(code, start, end, config, timeout)
        except requests.HTTPError as exc:
            return None, f"HFED returned HTTP {exc.response.status_code}."
        except requests.Timeout:
            return None, "The HFED request timed out."
        except requests.RequestException:
            return None, "The dashboard could not complete the connection to HFED."
        except ValueError as exc:
            return None, f"The HFED response could not be read. {exc}"
        return frame, "HFED returned no demand readings for the requested dates." if frame is None else None

    regions = tuple(regions)
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = dict(zip(regions, executor.map(fetch, regions), strict=True))
    frames = [frame for frame, _ in results.values() if frame is not None]
    unavailable = {code: reason for code, (_, reason) in results.items() if reason is not None}
    demand = (pd.concat(frames, ignore_index=True) if frames
              else pd.DataFrame(columns=["region", "time_utc", "demand_mw"]))
    return demand, unavailable


def fetch_forecast_observations(region, forecast_date, config):
    """Two weeks of demand and generation history plus forecast-day readings."""
    targets = target_hours(region, forecast_date)
    start = targets[0] - pd.Timedelta(hours=336)
    end = targets[-1] + pd.Timedelta(hours=1)
    demand_codes = config["download"]["demand_variables"]
    generation_codes = config["download"]["generation_variables"]

    def fetch(code):
        flows = [demand_codes[code], *generation_codes.get(code, [])]
        content = _get_csv(series_url(code, flows), start, end, timeout=60)
        return None if content is None else read_observations([BytesIO(content)], config)

    regions = config["download"]["regions"]
    frames = []
    unavailable = []
    with ThreadPoolExecutor(max_workers=8) as executor:
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
