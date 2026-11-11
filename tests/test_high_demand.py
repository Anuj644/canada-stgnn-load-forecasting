"""Tests for the high-demand detection component."""
import numpy as np
import pandas as pd
import pytest

from src.high_demand.detector import flag_high_demand, high_demand_summary, classify_demand_level
from src.evaluation.evaluator import compute_peak_thresholds


def make_forecast_df(regions=("ON", "QC"), n_days: int = 5) -> pd.DataFrame:
    rows = []
    for region in regions:
        base_demand = {"ON": 15000, "QC": 25000}.get(region, 10000)
        for d in range(n_days):
            for h in range(24):
                rows.append({
                    "region": region,
                    "forecast_date": pd.Timestamp("2024-03-01") + pd.Timedelta(days=d),
                    "target_hour": h,
                    "predicted": base_demand + (h - 12) * 200 + d * 100,
                    "actual": base_demand + (h - 12) * 190 + d * 100,
                })
    return pd.DataFrame(rows)


def make_train_df(regions=("ON", "QC"), n_hours: int = 500) -> pd.DataFrame:
    frames = []
    for region in regions:
        base = {"ON": 15000, "QC": 25000}.get(region, 10000)
        times = pd.date_range("2022-01-01", periods=n_hours, freq="h")
        frames.append(pd.DataFrame({
            "time_utc": times,
            "region": region,
            "demand_mw": base + np.random.default_rng(42).normal(0, 1000, n_hours),
        }))
    return pd.concat(frames, ignore_index=True)


class TestClassifyDemandLevel:
    def test_above_threshold_is_high(self):
        assert classify_demand_level(16000, 15000) == "High demand"

    def test_below_threshold_is_normal(self):
        assert classify_demand_level(14000, 15000) == "Normal demand"

    def test_equal_to_threshold_is_high(self):
        assert classify_demand_level(15000, 15000) == "High demand"

    def test_nan_predicted_is_unknown(self):
        assert classify_demand_level(float("nan"), 15000) == "Unknown"


class TestFlagHighDemand:
    def setup_method(self):
        self.forecast = make_forecast_df()
        self.thresholds = {"ON": 15000, "QC": 25000}

    def test_output_has_required_columns(self):
        result = flag_high_demand(self.forecast, self.thresholds)
        required = [
            "region", "forecast_date", "target_hour",
            "predicted_demand_mw", "threshold_mw", "is_high_demand",
            "demand_level", "actual_demand_mw", "forecast_error_mw",
        ]
        for col in required:
            assert col in result.columns, f"Missing column: {col}"

    def test_is_high_demand_dtype_is_bool(self):
        result = flag_high_demand(self.forecast, self.thresholds)
        assert result["is_high_demand"].dtype == bool or result["is_high_demand"].dtype == np.bool_

    def test_threshold_applied_per_region(self):
        result = flag_high_demand(self.forecast, self.thresholds)
        for region in ["ON", "QC"]:
            thr = self.thresholds[region]
            rdf = result[result["region"] == region]
            high = rdf[rdf["is_high_demand"]]
            assert (high["predicted_demand_mw"] >= thr).all()
            normal = rdf[~rdf["is_high_demand"]]
            assert (normal["predicted_demand_mw"] < thr).all()

    def test_forecast_error_computed_correctly(self):
        result = flag_high_demand(self.forecast, self.thresholds)
        # error = actual - predicted
        expected_error = result["actual_demand_mw"] - result["predicted_demand_mw"]
        pd.testing.assert_series_equal(
            result["forecast_error_mw"].reset_index(drop=True),
            expected_error.reset_index(drop=True),
            check_names=False,
        )

    def test_no_rows_lost(self):
        result = flag_high_demand(self.forecast, self.thresholds)
        assert len(result) == len(self.forecast)


class TestPeakThresholds:
    def test_threshold_at_90th_percentile(self):
        train = make_train_df(regions=("ON",))
        thresholds = compute_peak_thresholds(train, ["ON"], percentile=90)
        # Manually compute
        expected = float(np.percentile(train[train["region"] == "ON"]["demand_mw"].dropna(), 90))
        assert abs(thresholds["ON"] - expected) < 1e-6

    def test_threshold_above_median(self):
        train = make_train_df(regions=("ON", "QC"))
        thresholds = compute_peak_thresholds(train, ["ON", "QC"], percentile=90)
        for region in ["ON", "QC"]:
            median = train[train["region"] == region]["demand_mw"].median()
            assert thresholds[region] > median, \
                f"90th percentile threshold should be above median for {region}"


class TestHighDemandSummary:
    def test_summary_one_row_per_region(self):
        forecast = make_forecast_df(regions=("ON", "QC"))
        thresholds = {"ON": 15000, "QC": 25000}
        flagged = flag_high_demand(forecast, thresholds)
        summary = high_demand_summary(flagged)
        assert set(summary["region"]) == {"ON", "QC"}

    def test_summary_pct_in_valid_range(self):
        forecast = make_forecast_df()
        thresholds = {"ON": 15000, "QC": 25000}
        flagged = flag_high_demand(forecast, thresholds)
        summary = high_demand_summary(flagged)
        assert (summary["high_demand_pct"] >= 0).all()
        assert (summary["high_demand_pct"] <= 100).all()
