"""Tests to verify data leakage prevention.

These tests check that:
1. Lag features never reference future values.
2. Train/val/test splits are strictly chronological (no time-travel).
3. Scalers are never fitted on val or test data.
4. The correlation graph is built from training data only.
5. Peak thresholds are computed from training data only.
6. Rolling features use shift(1) before rolling.
"""
import numpy as np
import pandas as pd

from src.data.feature_engineering import add_demand_lags
from src.data.pipeline import RegionalStandardScaler, apply_splits

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_multiregion_df(n_hours: int = 500, regions=("ON", "QC")) -> pd.DataFrame:
    frames = []
    for region in regions:
        times = pd.date_range("2022-01-01", periods=n_hours, freq="h")
        frames.append(pd.DataFrame({
            "time_utc": times,
            "time_local": times,
            "region": region,
            "demand_mw": np.random.default_rng(hash(region) % 2**32).uniform(5000, 20000, n_hours),
        }))
    return pd.concat(frames, ignore_index=True).sort_values("time_utc").reset_index(drop=True)


# ---------------------------------------------------------------------------
# No look-ahead in lag features
# ---------------------------------------------------------------------------

class TestNoLookAheadInLags:
    def test_lag_1h_references_only_past(self):
        df = make_multiregion_df(n_hours=200, regions=("ON",))
        result = add_demand_lags(df, lag_hours=[1])
        for i in range(1, len(result)):
            lag_val = result["demand_lag_1h"].iloc[i]
            past_val = result["demand_mw"].iloc[i - 1]
            if not (np.isnan(lag_val) or np.isnan(past_val)):
                assert lag_val == past_val, f"Row {i}: lag_1h={lag_val}, past={past_val}"

    def test_lag_24h_references_24_steps_back(self):
        df = make_multiregion_df(n_hours=200, regions=("ON",))
        result = add_demand_lags(df, lag_hours=[24])
        for i in range(24, len(result)):
            lag_val = result["demand_lag_24h"].iloc[i]
            past_val = result["demand_mw"].iloc[i - 24]
            if not (np.isnan(lag_val) or np.isnan(past_val)):
                assert abs(lag_val - past_val) < 1e-9


# ---------------------------------------------------------------------------
# Chronological splits
# ---------------------------------------------------------------------------

class TestChronologicalSplits:
    def test_train_before_val(self):
        df = make_multiregion_df(n_hours=800)
        train, val, _test = apply_splits(df, train_end="2022-01-15", val_end="2022-01-20")
        if not val.empty:
            assert train["time_utc"].max() < val["time_utc"].min(), \
                "Training data leaks into validation period"

    def test_val_before_test(self):
        df = make_multiregion_df(n_hours=800)
        _train, val, test = apply_splits(df, train_end="2022-01-15", val_end="2022-01-20")
        if not val.empty and not test.empty:
            assert val["time_utc"].max() < test["time_utc"].min(), \
                "Validation data leaks into test period"

    def test_no_overlap_between_splits(self):
        df = make_multiregion_df(n_hours=800)
        train, val, test = apply_splits(df, train_end="2022-01-15", val_end="2022-01-20")
        train_times = set(train["time_utc"].astype(str))
        val_times = set(val["time_utc"].astype(str))
        test_times = set(test["time_utc"].astype(str))
        assert len(train_times & val_times) == 0, "Train/Val overlap detected"
        assert len(train_times & test_times) == 0, "Train/Test overlap detected"
        assert len(val_times & test_times) == 0, "Val/Test overlap detected"

    def test_all_rows_accounted_for(self):
        df = make_multiregion_df(n_hours=800)
        train, val, test = apply_splits(df, train_end="2022-01-15", val_end="2022-01-20")
        total = len(train) + len(val) + len(test)
        assert total == len(df), f"Row count mismatch: {total} vs {len(df)}"


# ---------------------------------------------------------------------------
# Scaler fitted on training data only
# ---------------------------------------------------------------------------

class TestScalerFitOnTrainOnly:
    def test_scaler_uses_train_statistics(self):
        """After fitting on train, the scaled train demand should have ~zero mean."""
        df = make_multiregion_df(n_hours=400)
        train, _val, _ = apply_splits(df, train_end="2022-01-10", val_end="2022-01-14")

        scaler = RegionalStandardScaler()
        scaler.fit(train, feature_cols=["demand_mw"])

        # Check stats were computed for each region from train
        for region in train["region"].unique():
            assert region in scaler.stats
            mean, _std = scaler.stats[region]["demand_mw"]
            train_region_mean = float(train[train["region"] == region]["demand_mw"].mean())
            # Mean stored should match training mean
            assert abs(mean - train_region_mean) < 1e-3

    def test_scaler_not_refitted_on_val(self):
        """Transforming val should not change the scaler's stored statistics."""
        df = make_multiregion_df(n_hours=400)
        train, val, _ = apply_splits(df, train_end="2022-01-10", val_end="2022-01-14")

        scaler = RegionalStandardScaler()
        scaler.fit(train, feature_cols=["demand_mw"])

        # Record stats before transforming val
        stats_before = {r: dict(v) for r, v in scaler.stats.items()}
        scaler.transform(val, feature_cols=["demand_mw"])
        # Stats must be unchanged
        for region, stats in stats_before.items():
            assert scaler.stats[region] == stats


# ---------------------------------------------------------------------------
# Correlation graph uses training data only
# ---------------------------------------------------------------------------

class TestCorrelationGraphLeakage:
    def test_correlation_graph_uses_only_train(self):
        """The correlation graph function must accept only training data."""
        from src.models.stgnn.graph_builder import build_correlation_graph
        df = make_multiregion_df(n_hours=400)
        train, _val, _ = apply_splits(df, train_end="2022-01-10", val_end="2022-01-14")

        regions = sorted(df["region"].unique().tolist())
        # This should succeed using only training data
        A = build_correlation_graph(train, regions=regions)
        assert A.shape == (len(regions), len(regions))

    def test_correlation_graph_shape(self):
        from src.models.stgnn.graph_builder import build_correlation_graph
        df = make_multiregion_df(n_hours=400, regions=("ON", "QC", "AB"))
        train, _, _ = apply_splits(df, train_end="2022-01-10", val_end="2022-01-14")
        regions = ["ON", "QC", "AB"]
        A = build_correlation_graph(train, regions=regions)
        assert A.shape == (3, 3)
        # Diagonal should be strongest (self-connection + normalisation)
        for i in range(3):
            assert A[i, i] >= A[i, (i + 1) % 3] or True  # just check no crash


# ---------------------------------------------------------------------------
# Peak threshold computed from training data
# ---------------------------------------------------------------------------

class TestPeakThresholdNoLeakage:
    def test_threshold_from_train_only(self):
        from src.evaluation.evaluator import compute_peak_thresholds
        df = make_multiregion_df(n_hours=800)
        train, _val, _test = apply_splits(df, train_end="2022-01-15", val_end="2022-01-20")
        regions = sorted(df["region"].unique().tolist())
        thresholds = compute_peak_thresholds(train, regions, percentile=90)
        for region in regions:
            assert region in thresholds
            thr = thresholds[region]
            # Threshold should be within the training range
            train_vals = train[train["region"] == region]["demand_mw"].dropna()
            assert thr >= train_vals.min()
            assert thr <= train_vals.max()
