"""Tests for data loading and resampling."""
import numpy as np
import pandas as pd

from src.data.resampler import (
    detect_native_frequency,
    resample_to_hourly,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_hourly_df(n_hours: int = 48, region: str = "ON") -> pd.DataFrame:
    times = pd.date_range("2024-01-01", periods=n_hours, freq="h")
    return pd.DataFrame(
        {
            "time_utc": times,
            "time_local": times,
            "region": region,
            "demand_mw": np.linspace(10000, 15000, n_hours),
            "WIND_OUTPUT": np.random.default_rng(0).uniform(100, 500, n_hours),
            "DATETIME_LOCAL_OFFSET": 5.0,
        }
    )


def make_subhourly_df(n_obs: int = 192, freq_min: int = 15, region: str = "QC") -> pd.DataFrame:
    """15-minute data for 48 hours."""
    times = pd.date_range("2024-01-01", periods=n_obs, freq=f"{freq_min}min")
    return pd.DataFrame(
        {
            "time_utc": times,
            "time_local": times,
            "region": region,
            "demand_mw": np.linspace(5000, 8000, n_obs) + np.random.default_rng(1).normal(0, 50, n_obs),
            "DATETIME_LOCAL_OFFSET": 5.0,
        }
    )


# ---------------------------------------------------------------------------
# detect_native_frequency
# ---------------------------------------------------------------------------

def test_detect_frequency_hourly():
    df = make_hourly_df()
    freq = detect_native_frequency(df)
    assert freq == 60


def test_detect_frequency_15min():
    df = make_subhourly_df(freq_min=15)
    freq = detect_native_frequency(df)
    assert freq == 15


def test_detect_frequency_5min():
    times = pd.date_range("2024-01-01", periods=288, freq="5min")
    df = pd.DataFrame({"time_utc": times, "demand_mw": 1.0, "region": "NL", "time_local": times})
    freq = detect_native_frequency(df)
    assert freq == 5


# ---------------------------------------------------------------------------
# resample_to_hourly
# ---------------------------------------------------------------------------

def test_hourly_data_passthrough():
    df = make_hourly_df(n_hours=72)
    result = resample_to_hourly(df)
    assert len(result) == 72
    assert "demand_mw" in result.columns
    assert "time_utc" in result.columns


def test_subhourly_resamples_to_correct_count():
    df = make_subhourly_df(n_obs=192, freq_min=15)  # 48 hours of 15-min data
    result = resample_to_hourly(df, min_coverage=0.25)
    assert len(result) == 48


def test_resampled_demand_is_mean():
    """The resampled demand should be the mean of the sub-hourly values."""
    times = pd.date_range("2024-01-01", periods=4, freq="15min")
    df = pd.DataFrame(
        {
            "time_utc": times,
            "time_local": times,
            "region": "PE",
            "demand_mw": [100.0, 200.0, 300.0, 400.0],
            "DATETIME_LOCAL_OFFSET": 4.0,
        }
    )
    result = resample_to_hourly(df, min_coverage=0.25)
    assert len(result) == 1
    assert abs(result["demand_mw"].iloc[0] - 250.0) < 1e-6


def test_low_coverage_hours_are_nan():
    """Hours below min_obs_required should be NaN.

    With min_coverage=0.9 and native_freq_minutes=15 (expected_per_hour=4):
    min_obs_required = max(1, int(4 * 0.9)) = 3.
    Hour 1 gets only 1 obs (below 3) so it must be NaN.
    """
    base = pd.date_range("2024-01-01T00:00", periods=4, freq="15min")  # hour 0: 4 obs
    extra = pd.date_range("2024-01-01T01:00", periods=1, freq="15min")  # hour 1: 1 obs
    times = base.append(extra)
    df = pd.DataFrame(
        {
            "time_utc": times,
            "time_local": times,
            "region": "NB",
            "demand_mw": [1500.0, 1510.0, 1520.0, 1530.0, 1600.0],
            "DATETIME_LOCAL_OFFSET": 4.0,
        }
    )
    result = resample_to_hourly(df, min_coverage=0.9, native_freq_minutes=15)
    assert len(result) == 2
    hour1_demand = result.loc[
        result["time_utc"] == pd.Timestamp("2024-01-01T01:00"), "demand_mw"
    ].values[0]
    assert pd.isna(hour1_demand), f"Expected NaN for low-coverage hour, got {hour1_demand}"


def test_no_duplicate_hours_after_resample():
    df = make_hourly_df(n_hours=48)
    # Inject a duplicate hour
    duplicate = df.iloc[[0]].copy()
    df = pd.concat([df, duplicate], ignore_index=True)
    result = resample_to_hourly(df)
    assert result["time_utc"].duplicated().sum() == 0


def test_region_column_preserved():
    df = make_subhourly_df(region="AB")
    result = resample_to_hourly(df, min_coverage=0.25)
    assert (result["region"] == "AB").all()
