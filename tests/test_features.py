"""Tests for feature engineering."""
import numpy as np
import pandas as pd
import pytest

from src.data.feature_engineering import (
    add_calendar_features,
    add_demand_lags,
    add_rolling_features,
    add_quality_flags,
)


def make_df(n_hours: int = 200, region: str = "ON") -> pd.DataFrame:
    times = pd.date_range("2024-01-01", periods=n_hours, freq="h")
    return pd.DataFrame(
        {
            "time_utc": times,
            "time_local": times,
            "region": region,
            "demand_mw": np.linspace(10000, 15000, n_hours),
        }
    )


# ---------------------------------------------------------------------------
# Calendar features
# ---------------------------------------------------------------------------

class TestCalendarFeatures:
    def test_all_expected_columns_present(self):
        df = make_df()
        result = add_calendar_features(df)
        expected = [
            "hour", "day_of_week", "day_of_month", "month",
            "day_of_year", "week_of_year", "is_weekend",
            "sin_hour", "cos_hour", "sin_doy", "cos_doy",
        ]
        for col in expected:
            assert col in result.columns, f"Missing column: {col}"

    def test_hour_range(self):
        df = make_df(n_hours=48)
        result = add_calendar_features(df)
        assert result["hour"].between(0, 23).all()

    def test_day_of_week_range(self):
        df = make_df(n_hours=200)
        result = add_calendar_features(df)
        assert result["day_of_week"].between(0, 6).all()

    def test_is_weekend_flag(self):
        df = make_df(n_hours=200)
        result = add_calendar_features(df)
        expected_weekend = pd.to_datetime(df["time_local"]).dt.day_of_week >= 5
        assert (result["is_weekend"].astype(bool) == expected_weekend).all()

    def test_cyclic_sin_cos_range(self):
        df = make_df(n_hours=48)
        result = add_calendar_features(df)
        assert result["sin_hour"].between(-1, 1).all()
        assert result["cos_hour"].between(-1, 1).all()

    def test_sin_cos_hour_period(self):
        """sin/cos of hour should return to same values every 24 hours."""
        df = make_df(n_hours=48)
        result = add_calendar_features(df)
        h0 = result[result["hour"] == 0]["sin_hour"].values
        assert len(h0) >= 2
        assert abs(h0[0] - h0[1]) < 1e-9

    def test_missing_local_col_raises(self):
        df = make_df().drop(columns=["time_local"])
        df["other_col"] = df["time_utc"]
        with pytest.raises(ValueError, match="'time_local' not found"):
            add_calendar_features(df, local_col="time_local")


# ---------------------------------------------------------------------------
# Lag features
# ---------------------------------------------------------------------------

class TestLagFeatures:
    def test_lag_columns_created(self):
        df = make_df(n_hours=200)
        result = add_demand_lags(df, lag_hours=[1, 24, 168])
        assert "demand_lag_1h" in result.columns
        assert "demand_lag_24h" in result.columns
        assert "demand_lag_168h" in result.columns

    def test_lag_1h_is_previous_value(self):
        df = make_df(n_hours=50)
        result = add_demand_lags(df, lag_hours=[1])
        # Row 1's lag_1h should equal row 0's demand_mw
        assert abs(result["demand_lag_1h"].iloc[1] - df["demand_mw"].iloc[0]) < 1e-9

    def test_lag_introduces_nan_at_start(self):
        df = make_df(n_hours=50)
        result = add_demand_lags(df, lag_hours=[24])
        # First 24 rows should be NaN
        assert result["demand_lag_24h"].iloc[:24].isna().all()

    def test_lag_does_not_look_ahead(self):
        """Lag features must only reference past values — not future."""
        df = make_df(n_hours=50)
        result = add_demand_lags(df, lag_hours=[1])
        for i in range(1, len(result)):
            lag_val = result["demand_lag_1h"].iloc[i]
            past_val = result["demand_mw"].iloc[i - 1]
            if not (np.isnan(lag_val) or np.isnan(past_val)):
                assert abs(lag_val - past_val) < 1e-9


# ---------------------------------------------------------------------------
# Rolling features
# ---------------------------------------------------------------------------

class TestRollingFeatures:
    def test_rolling_columns_created(self):
        df = make_df(n_hours=200)
        result = add_rolling_features(df, windows=[24, 168])
        assert "demand_roll_mean_24h" in result.columns
        assert "demand_roll_std_24h" in result.columns
        assert "demand_roll_mean_168h" in result.columns

    def test_rolling_no_lookahead(self):
        """Rolling features use shift(1), so they should not contain the current value."""
        df = make_df(n_hours=50)
        df["demand_mw"] = np.arange(50, dtype=float)
        result = add_rolling_features(df, windows=[3])
        # Row 3: rolling mean of rows 0,1,2 = (0+1+2)/3 = 1.0
        expected = (0 + 1 + 2) / 3
        actual = result["demand_roll_mean_3h"].iloc[3]
        assert abs(actual - expected) < 1e-6


# ---------------------------------------------------------------------------
# Quality flags
# ---------------------------------------------------------------------------

class TestQualityFlags:
    def test_flag_missing(self):
        df = make_df(n_hours=10)
        df.loc[3, "demand_mw"] = np.nan
        result = add_quality_flags(df)
        assert result["demand_flag_missing"].iloc[3] == 1
        assert result["demand_flag_missing"].iloc[0] == 0

    def test_flag_negative(self):
        df = make_df(n_hours=10)
        df.loc[5, "demand_mw"] = -100.0
        result = add_quality_flags(df, demand_min=0.0)
        assert result["demand_flag_negative"].iloc[5] == 1

    def test_flag_extreme(self):
        df = make_df(n_hours=20)
        df["demand_mw"] = 10000.0
        df.loc[10, "demand_mw"] = 100000.0  # 10× median = extreme
        result = add_quality_flags(df, demand_max_multiplier=5.0)
        assert result["demand_flag_extreme"].iloc[10] == 1
        # Normal rows should not be flagged
        assert result["demand_flag_extreme"].iloc[0] == 0
