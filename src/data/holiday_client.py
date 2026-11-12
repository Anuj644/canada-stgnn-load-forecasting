"""Nager.Date API client for Canadian public holidays.

Features:
- Fetches national and province/territory-specific Canadian holidays.
- Uses the Community API v4 endpoint at nagerholidays.com.
- Caches responses locally to avoid redundant API calls.
- Returns a tidy DataFrame indexed by (date, region).
- Maps Nager.Date subdivision codes (CA-ON) to HFED region codes (ON).
- National holidays (global: true) apply to all regions.
- Provincial holidays apply to their specific subdivision(s) only.

API reference: https://github.com/nager/Nager.Date
Endpoint:      https://nagerholidays.com/api/v4/Holidays/CA/{year}
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

API_BASE = "https://date.nager.at/api/v3"   # v3 supports any year; endpoint: /PublicHolidays/{year}/{country}
COUNTRY_CODE = "CA"

# Mapping from Nager.Date subdivision code to HFED region code
SUBDIVISION_TO_REGION: dict[str, str] = {
    "CA-AB": "AB",
    "CA-BC": "BC",
    "CA-NB": "NB",
    "CA-NL": "NL",
    "CA-NS": "NS",
    "CA-ON": "ON",
    "CA-PE": "PE",
    "CA-QC": "QC",
    "CA-SK": "SK",
    "CA-YT": "YK",
    "CA-MB": "MB",   # Manitoba — not in HFED but kept for completeness
    "CA-NT": "NT",   # Northwest Territories
    "CA-NU": "NU",   # Nunavut
}

ALL_HFED_REGIONS = {"AB", "BC", "NB", "NL", "NS", "ON", "PE", "QC", "SK", "YK"}


# ---------------------------------------------------------------------------
# API fetch
# ---------------------------------------------------------------------------

def _fetch_year_from_api(
    year: int,
    session: requests.Session,
    retry_delay: float = 2.0,
    max_retries: int = 3,
) -> list[dict]:
    """Fetch holidays for one calendar year from the Nager.Date API.

    Returns the raw JSON list as Python dicts.
    """
    url = f"{API_BASE}/PublicHolidays/{year}/{COUNTRY_CODE}"
    for attempt in range(max_retries):
        try:
            resp = session.get(url, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                logger.warning("Year %d not found in Nager.Date API (404).", year)
                return []
            raise
        except requests.RequestException as exc:
            if attempt == max_retries - 1:
                raise
            logger.warning(
                "Nager.Date request failed (%s); retry %d/%d in %.1fs",
                exc,
                attempt + 1,
                max_retries,
                retry_delay,
            )
            time.sleep(retry_delay)
    return []


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------

def _load_cache(cache_path: Path) -> dict[int, list[dict]]:
    """Load local holiday cache.  Returns {year: [holiday_dicts]}."""
    if not cache_path.exists():
        return {}
    try:
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
        return {int(k): v for k, v in raw.items()}
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Could not read holiday cache %s: %s", cache_path, exc)
        return {}


def _save_cache(cache: dict[int, list[dict]], cache_path: Path) -> None:
    """Persist the holiday cache to disk (atomic write)."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(".json.part")
    tmp.write_text(
        json.dumps({str(k): v for k, v in cache.items()}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp.replace(cache_path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_holidays(
    years: list[int],
    cache_path: str | Path = "data/holidays/canada_holidays.json",
    force_refresh: bool = False,
) -> dict[int, list[dict]]:
    """Return holiday data for the requested years, using cache where possible.

    Parameters
    ----------
    years:
        List of calendar years to retrieve.
    cache_path:
        Path to the local JSON cache file.
    force_refresh:
        If True, ignore cached entries and re-fetch from the API.

    Returns
    -------
    dict mapping year → list of holiday dicts from the Nager.Date API.
    """
    cache_path = Path(cache_path)
    cache = {} if force_refresh else _load_cache(cache_path)
    missing = [y for y in years if y not in cache]

    if missing:
        session = requests.Session()
        session.headers["User-Agent"] = "canada-stgnn-holiday-client/1.0"
        for year in missing:
            logger.info("Fetching holidays for year %d from Nager.Date API.", year)
            try:
                holidays = _fetch_year_from_api(year, session)
                cache[year] = holidays
            except requests.RequestException as exc:
                logger.error(
                    "Failed to fetch holidays for %d: %s. Using empty list.", year, exc
                )
                cache[year] = []
        _save_cache(cache, cache_path)
        logger.info("Holiday cache saved to %s.", cache_path)

    return {y: cache[y] for y in years}


def build_holiday_dataframe(
    holiday_data: dict[int, list[dict]],
    hfed_regions: list[str] | None = None,
) -> pd.DataFrame:
    """Convert raw API holiday data to a tidy region-date DataFrame.

    Parameters
    ----------
    holiday_data:
        Output of fetch_holidays().
    hfed_regions:
        Regions to include.  Defaults to all 10 HFED regions.

    Returns
    -------
    DataFrame with columns:
        date        (datetime.date)
        region      (HFED two-letter code)
        is_holiday  (bool)
        holiday_name (str or "")
        is_national  (bool)
    """
    if hfed_regions is None:
        hfed_regions = sorted(ALL_HFED_REGIONS)

    rows = []
    for year, year_holidays in sorted(holiday_data.items()):
        for h in year_holidays:
            holiday_date = pd.to_datetime(h["date"]).date()
            name = h.get("localName") or h.get("name") or ""
            is_global = h.get("global", False)
            counties = h.get("counties") or []  # None → []

            # Determine which HFED regions this holiday applies to
            if is_global:
                affected_regions = set(hfed_regions)
            else:
                affected_regions = set()
                for subdivision in counties:
                    region = SUBDIVISION_TO_REGION.get(subdivision)
                    if region and region in hfed_regions:
                        affected_regions.add(region)

            for region in affected_regions:
                rows.append(
                    {
                        "date": holiday_date,
                        "region": region,
                        "is_holiday": True,
                        "holiday_name": name,
                        "is_national": is_global,
                    }
                )

    if not rows:
        return pd.DataFrame(
            columns=["date", "region", "is_holiday", "holiday_name", "is_national"]
        )

    df = pd.DataFrame(rows)
    # Drop any duplicates (same date, region, name)
    df = df.drop_duplicates(subset=["date", "region"]).sort_values(
        ["region", "date"]
    ).reset_index(drop=True)
    return df


def get_holiday_df_for_data_range(
    start_year: int,
    end_year: int,
    cache_path: str | Path = "data/holidays/canada_holidays.json",
    hfed_regions: list[str] | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """High-level helper: fetch and build the holiday DataFrame for a year range.

    Parameters
    ----------
    start_year, end_year : inclusive year range.

    Returns
    -------
    DataFrame with one row per (holiday-date, region) pair.
    """
    years = list(range(start_year, end_year + 1))
    holiday_data = fetch_holidays(
        years, cache_path=cache_path, force_refresh=force_refresh
    )
    return build_holiday_dataframe(holiday_data, hfed_regions=hfed_regions)
