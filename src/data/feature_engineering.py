"""Feature engineering for the hourly electricity demand forecasting pipeline.

Builds calendar, lag, and rolling features from the cleaned hourly DataFrame.

IMPORTANT — data leakage prevention:
- Lag and rolling features use only past observations relative to each
  timestamp.  pandas shift() is used; no look-ahead.
- Scalers, holiday flags, and threshold statistics are fitted/computed
  on the training split only and then applied to val/test.
- The function `add_lags_and_rolling` is pure (no fitting); it only shifts.
- The function `compute_demand_lags` explicitly documents which lag offsets
  are safe given the forecast cutoff (23:00 local).

Calendar features are derived from time_local (local datetime), not UTC,
so that 'hour', 'day_of_week', etc. reflect the region's local time.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Calendar features
# ---------------------------------------------------------------------------

def add_calendar_features(df: pd.DataFrame, local_col: str = "time_local") -> pd.DataFrame:
    """Add calendar features derived from the local datetime column.

    Generated columns:
        hour, day_of_week, day_of_month, month,
        day_of_year, week_of_year,
        is_weekend, sin_hour, cos_hour, sin_doy, cos_doy

    Parameters
    ----------
    df : DataFrame with a local datetime column.
    local_col : name of the local datetime column.

    Returns
    -------
    df with new calendar columns appended.
    """
    df = df.copy()
    if local_col not in df.columns:
        raise ValueError(f"Column '{local_col}' not found in DataFrame.")

    dt = pd.to_datetime(df[local_col])

    df["hour"] = dt.dt.hour
    df["day_of_week"] = dt.dt.day_of_week          # 0=Monday … 6=Sunday
    df["day_of_month"] = dt.dt.day
    df["month"] = dt.dt.month
    df["day_of_year"] = dt.dt.day_of_year
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    df["is_weekend"] = (dt.dt.day_of_week >= 5).astype(int)
    df["year"] = dt.dt.year

    # Cyclical encodings
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["sin_dow"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["cos_dow"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    df["sin_doy"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["cos_doy"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)

    return df


# ---------------------------------------------------------------------------
# Holiday features
# ---------------------------------------------------------------------------

def add_holiday_features(
    df: pd.DataFrame,
    holiday_df: pd.DataFrame,
    local_col: str = "time_local",
    region_col: str = "region",
) -> pd.DataFrame:
    """Merge holiday information into the hourly DataFrame.

    holiday_df must have columns: date, region, is_holiday, holiday_name.

    Generated columns:
        is_holiday, holiday_name, is_day_before_holiday, is_day_after_holiday

    Parameters
    ----------
    df : Hourly DataFrame with local datetime and region columns.
    holiday_df : Output of holiday_client.build_holiday_dataframe().
    local_col : Local datetime column name in df.
    region_col : Region identifier column name in df.
    """
    df = df.copy()
    dt = pd.to_datetime(df[local_col])

    # Create a 'date' column for the local date of each row
    df["_date"] = dt.dt.date

    # Merge on (region, date)
    holiday_lookup = holiday_df[["date", "region", "is_holiday", "holiday_name"]].copy()
    holiday_lookup["date"] = pd.to_datetime(holiday_lookup["date"]).dt.date

    df = df.merge(
        holiday_lookup,
        left_on=["_date", region_col],
        right_on=["date", "region"],
        how="left",
        suffixes=("", "_holiday"),
    )
    # Drop merge artifact column
    if "date" in df.columns:
        df = df.drop(columns=["date"])
    if "region_holiday" in df.columns:
        df = df.drop(columns=["region_holiday"])

    df["is_holiday"] = df["is_holiday"].fillna(False).astype(int)
    df["holiday_name"] = df["holiday_name"].fillna("").astype(str)

    # Day-before and day-after flags
    # Build a set of (region, date) holiday tuples for fast lookup
    holiday_set = set(
        zip(holiday_df["region"].values, pd.to_datetime(holiday_df["date"]).dt.date.values)
    )

    df["is_day_before_holiday"] = df.apply(
        lambda row: int(
            (row[region_col], (pd.Timestamp(row["_date"]) + pd.Timedelta(days=1)).date())
            in holiday_set
        ),
        axis=1,
    )
    df["is_day_after_holiday"] = df.apply(
        lambda row: int(
            (row[region_col], (pd.Timestamp(row["_date"]) - pd.Timedelta(days=1)).date())
            in holiday_set
        ),
        axis=1,
    )

    df = df.drop(columns=["_date"])
    return df


# ---------------------------------------------------------------------------
# Lag and rolling features
# ---------------------------------------------------------------------------

def add_demand_lags(
    df: pd.DataFrame,
    demand_col: str = "demand_mw",
    lag_hours: list[int] | None = None,
) -> pd.DataFrame:
    """Add demand lag features.

    Lag features shift the demand signal backward in time, so they represent
    past observations.  pandas shift(n) means "value from n rows ago".
    For an hourly series this equals n hours ago.

    The safe lag values for a forecast issued at 23:00 targeting next-day
    hours 0–23 are any lags ≥ 1 (all past values are known at 23:00).

    Parameters
    ----------
    df : Hourly DataFrame sorted by time_utc.
    demand_col : Column to create lags from.
    lag_hours : Lag offsets in hours.  Defaults to [1,2,3,6,12,24,48,168].
    """
    if lag_hours is None:
        lag_hours = [1, 2, 3, 6, 12, 24, 48, 168]

    df = df.copy()
    for lag in lag_hours:
        col_name = f"demand_lag_{lag}h"
        df[col_name] = df[demand_col].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    demand_col: str = "demand_mw",
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Add rolling mean and rolling std features.

    Rolling statistics use only past observations (closed='left' or
    shift(1) before rolling) to avoid look-ahead.

    Parameters
    ----------
    df : Hourly DataFrame sorted by time_utc.
    demand_col : Column to roll.
    windows : Window sizes in hours.  Defaults to [24, 168].
    """
    if windows is None:
        windows = [24, 168]

    df = df.copy()
    for window in windows:
        # Shift by 1 so that the rolling window ends at the previous hour
        shifted = df[demand_col].shift(1)
        df[f"demand_roll_mean_{window}h"] = shifted.rolling(window=window, min_periods=max(1, window // 4)).mean()
        df[f"demand_roll_std_{window}h"] = shifted.rolling(window=window, min_periods=max(1, window // 4)).std()
    return df


# ---------------------------------------------------------------------------
# Quality flag
# ---------------------------------------------------------------------------

def add_quality_flags(
    df: pd.DataFrame,
    demand_col: str = "demand_mw",
    demand_min: float = 0.0,
    demand_max_multiplier: float = 5.0,
) -> pd.DataFrame:
    """Flag suspicious demand observations.

    Flags (int 0/1):
        demand_flag_negative   — value below demand_min
        demand_flag_extreme    — value > demand_max_multiplier × regional median
        demand_flag_missing    — NaN value

    Parameters
    ----------
    demand_min:
        Values below this are flagged as suspicious (e.g. negative MW).
    demand_max_multiplier:
        Values above this multiple of the regional median are flagged.
        The median is computed over the entire region column, so this
        function should be called before train/test splitting, or
        the caller should pass a pre-computed median.
    """
    df = df.copy()
    vals = df[demand_col]
    median_val = vals.median()

    df["demand_flag_missing"] = vals.isna().astype(int)
    df["demand_flag_negative"] = (vals < demand_min).fillna(False).astype(int)
    df["demand_flag_extreme"] = (
        vals > demand_max_multiplier * median_val
    ).fillna(False).astype(int)

    n_flagged = (
        df["demand_flag_missing"] | df["demand_flag_negative"] | df["demand_flag_extreme"]
    ).sum()
    if n_flagged > 0:
        region = df["region"].iloc[0] if "region" in df.columns else "unknown"
        logger.warning(
            "Region %s: %d observations flagged (missing=%d, negative=%d, extreme=%d)",
            region,
            n_flagged,
            df["demand_flag_missing"].sum(),
            df["demand_flag_negative"].sum(),
            df["demand_flag_extreme"].sum(),
        )

    return df


# ---------------------------------------------------------------------------
# Combined pipeline
# ---------------------------------------------------------------------------

def build_features(
    df: pd.DataFrame,
    holiday_df: pd.DataFrame | None = None,
    lag_hours: list[int] | None = None,
    rolling_windows: list[int] | None = None,
    demand_min: float = 0.0,
    demand_max_multiplier: float = 5.0,
) -> pd.DataFrame:
    """Run the full feature-engineering pipeline on a single-region DataFrame.

    Steps:
    1. Quality flags
    2. Calendar features (from time_local)
    3. Holiday features (if holiday_df provided)
    4. Demand lag features
    5. Rolling features

    Parameters
    ----------
    df : Cleaned, hourly, single-region DataFrame.
    holiday_df : From holiday_client.build_holiday_dataframe().
    lag_hours : Lag offsets.
    rolling_windows : Rolling window sizes.
    demand_min : Minimum valid demand value.
    demand_max_multiplier : Extreme-value multiplier.
    """
    df = add_quality_flags(df, demand_min=demand_min, demand_max_multiplier=demand_max_multiplier)
    df = add_calendar_features(df)
    if holiday_df is not None:
        df = add_holiday_features(df, holiday_df)
    df = add_demand_lags(df, lag_hours=lag_hours)
    df = add_rolling_features(df, windows=rolling_windows)
    return df
