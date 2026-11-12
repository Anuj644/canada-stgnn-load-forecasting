"""Resample sub-hourly HFED time series to hourly resolution.

Regions like NB, NL, PE, QC, YK publish at 5-min, 10-min, or 15-min
intervals.  This module aggregates them to clean hourly timestamps in UTC.

Rules:
- For demand: arithmetic mean of all observations within the hour.
- Require at least `min_coverage` fraction of expected sub-hourly observations.
  Hours below this threshold are set to NaN (not imputed).
- Generation variables: also mean-aggregated.
- Output index is a complete, regular hourly DatetimeIndex in UTC.
  Missing hours in the original data result in NaN rows.
- time_local and DATETIME_LOCAL_OFFSET are forward-filled from the
  first observation of each hour.

No interpolation, forward-fill, or demand imputation is performed here.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Columns that are NOT numeric observations and need special handling
_META_COLS = {"time_utc", "time_local", "region", "DATETIME_LOCAL_OFFSET"}


def detect_native_frequency(df: pd.DataFrame) -> Optional[int]:
    """Estimate native interval in minutes from the median time delta.

    Returns the median interval in whole minutes, or None if indeterminate.
    """
    if len(df) < 2:
        return None
    deltas = df["time_utc"].diff().dropna()
    if deltas.empty:
        return None
    median_delta = deltas.median()
    minutes = int(median_delta.total_seconds() / 60)
    return max(1, minutes)


def resample_to_hourly(
    df: pd.DataFrame,
    min_coverage: float = 0.5,
    native_freq_minutes: Optional[int] = None,
) -> pd.DataFrame:
    """Resample a region DataFrame to hourly UTC.

    Parameters
    ----------
    df:
        Region DataFrame as returned by hfed_loader.load_region.
        Must have a 'time_utc' column (datetime64).
    min_coverage:
        Minimum fraction of expected sub-hourly observations to form a
        valid hourly mean.  Hours below this are NaN.
    native_freq_minutes:
        If None, estimated automatically from the data.

    Returns
    -------
    DataFrame with a continuous hourly UTC DatetimeIndex from the first
    to the last hour of the input, with NaN where insufficient data.
    """
    df = df.copy()
    df = df.sort_values("time_utc").reset_index(drop=True)

    if native_freq_minutes is None:
        native_freq_minutes = detect_native_frequency(df)

    if native_freq_minutes is None or native_freq_minutes == 0:
        native_freq_minutes = 60  # fall back to treating as hourly

    # If already hourly, just normalise and return
    if native_freq_minutes >= 55:
        logger.debug("Data appears hourly (%d-min); skipping resample.", native_freq_minutes)
        return _normalise_hourly(df)

    logger.info(
        "Resampling %s from ~%d-min to hourly (min_coverage=%.0f%%)",
        df["region"].iloc[0] if "region" in df.columns else "unknown",
        native_freq_minutes,
        min_coverage * 100,
    )

    # Floor time_utc to the hour to create grouping key
    df["_hour"] = df["time_utc"].dt.floor("h")

    # Expected observations per hour given native frequency
    expected_per_hour = max(1, 60 // native_freq_minutes)

    # Identify numeric feature columns (exclude metadata)
    numeric_cols = [
        c for c in df.columns
        if c not in _META_COLS and c != "_hour" and pd.api.types.is_numeric_dtype(df[c])
    ]

    # Count actual observations per hour
    obs_counts = df.groupby("_hour")[numeric_cols[0] if numeric_cols else "demand_mw"].count()

    # Aggregate: simple mean per hour
    agg = df.groupby("_hour")[numeric_cols].mean()

    # Null out hours that do not meet the minimum coverage threshold.
    # At minimum, we always require at least 1 observation (even if that is below the
    # percentage threshold), so hours with 0 obs remain NaN while hours with ≥1 obs
    # but below the coverage fraction are also NaN-ed.
    min_obs_required = max(1, int(expected_per_hour * min_coverage))
    low_coverage_hours = obs_counts[obs_counts < min_obs_required].index
    if len(low_coverage_hours) > 0:
        logger.warning(
            "%d hours had < %d obs (%.0f%% coverage threshold) and were set to NaN.",
            len(low_coverage_hours),
            min_obs_required,
            min_coverage * 100,
        )
        agg.loc[agg.index.isin(low_coverage_hours), numeric_cols] = float("nan")

    # Reconstruct a complete hourly index
    full_index = pd.date_range(
        start=agg.index.min(),
        end=agg.index.max(),
        freq="h",
    )
    agg = agg.reindex(full_index)
    agg.index.name = "time_utc"
    agg = agg.reset_index()

    # Carry forward metadata from original
    meta = (
        df.drop_duplicates("_hour")
        .set_index("_hour")[["time_local", "region", "DATETIME_LOCAL_OFFSET"]]
    )
    agg = agg.merge(meta, left_on="time_utc", right_index=True, how="left")

    # Forward-fill time_local for the newly introduced NaN rows
    agg["time_local"] = agg["time_local"].ffill()
    agg["region"] = agg["region"].ffill()
    agg["DATETIME_LOCAL_OFFSET"] = agg["DATETIME_LOCAL_OFFSET"].ffill()

    return agg.reset_index(drop=True)


def _normalise_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure hourly data has a complete regular UTC index with no gaps."""
    df = df.copy()
    df["time_utc"] = df["time_utc"].dt.floor("h")

    # Deduplicate by taking the mean within any duplicate hours
    numeric_cols = [
        c for c in df.columns
        if c not in _META_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]
    meta_cols_present = [c for c in _META_COLS if c in df.columns and c != "time_utc"]

    # Keep one metadata row per hour
    meta = df.drop_duplicates("time_utc")[["time_utc"] + meta_cols_present]
    agg = df.groupby("time_utc")[numeric_cols].mean()

    # Build complete hourly index
    full_index = pd.date_range(
        start=agg.index.min(),
        end=agg.index.max(),
        freq="h",
    )
    agg = agg.reindex(full_index)
    agg.index.name = "time_utc"
    agg = agg.reset_index()

    # Merge metadata back
    agg = agg.merge(meta, on="time_utc", how="left")
    agg["time_local"] = agg["time_local"].ffill()
    agg["region"] = agg["region"].ffill()
    if "DATETIME_LOCAL_OFFSET" in agg.columns:
        agg["DATETIME_LOCAL_OFFSET"] = agg["DATETIME_LOCAL_OFFSET"].ffill()

    return agg.reset_index(drop=True)


def resample_all_regions(
    region_dfs: dict[str, pd.DataFrame],
    min_coverage: float = 0.5,
) -> dict[str, pd.DataFrame]:
    """Resample all loaded regions to hourly UTC."""
    result = {}
    for region, df in region_dfs.items():
        try:
            result[region] = resample_to_hourly(df, min_coverage=min_coverage)
            logger.info(
                "Region %s: %d hourly rows after resampling",
                region,
                len(result[region]),
            )
        except Exception as exc:
            logger.error("Region %s: resampling failed — %s", region, exc)
    return result
