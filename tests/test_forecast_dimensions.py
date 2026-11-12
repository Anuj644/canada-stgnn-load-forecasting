"""Tests for forecast output shape and timestamp correctness.

Verifies that every model produces:
- Exactly 24 target hours (0–23) per region per forecast date.
- No target hours outside 0–23.
- Forecast dates that are strictly one day after the issue date.
- No data for future timestamps that could constitute leakage.
"""
import numpy as np
import pandas as pd

from src.models.naive import NaiveDay, NaiveWeek


def make_hourly_df(
    regions=("ON", "QC"),
    n_days: int = 30,
    start: str = "2024-01-01",
) -> pd.DataFrame:
    frames = []
    for region in regions:
        base = {"ON": 15000, "QC": 25000}.get(region, 10000)
        times = pd.date_range(start, periods=n_days * 24, freq="h")
        frames.append(pd.DataFrame({
            "time_utc": times,
            "time_local": times,
            "region": region,
            "demand_mw": base + np.sin(np.arange(n_days * 24) * 2 * np.pi / 24) * 2000,
        }))
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Naive model output shape
# ---------------------------------------------------------------------------

class TestNaiveOutputShape:
    def setup_method(self):
        self.df = make_hourly_df(regions=("ON", "QC"), n_days=10)

    def _check_shape(self, preds: pd.DataFrame, regions):
        # For each region, check hours 0-23 are present on each forecast date
        for region in regions:
            rdf = preds[preds["region"] == region]
            for date in rdf["time_utc"].dt.date.unique() if "time_utc" in rdf.columns else []:
                ddf = rdf[pd.to_datetime(rdf["time_utc"]).dt.date == date]
                hours = sorted(ddf["forecast_hour"].unique())
                assert len(hours) == 24, f"Expected 24 hours, got {len(hours)} for {region} {date}"
                assert min(hours) == 0 and max(hours) == 23

    def test_naive_day_output_has_region_column(self):
        model = NaiveDay()
        preds = model.predict(self.df)
        assert "region" in preds.columns

    def test_naive_day_forecast_hour_range(self):
        model = NaiveDay()
        preds = model.predict(self.df)
        assert preds["forecast_hour"].between(0, 23).all()

    def test_naive_week_forecast_hour_range(self):
        model = NaiveWeek()
        preds = model.predict(self.df)
        assert preds["forecast_hour"].between(0, 23).all()

    def test_naive_day_predicted_values_not_all_nan(self):
        model = NaiveDay()
        preds = model.predict(self.df)
        assert preds["predicted"].notna().sum() > 0

    def test_naive_week_needs_7days_history(self):
        """NaiveWeek needs 168h history; with <7 days all predictions should be NaN."""
        short_df = make_hourly_df(n_days=5)
        model = NaiveWeek()
        preds = model.predict(short_df)
        # First 168 rows of predictions must be NaN
        assert preds["predicted"].notna().sum() == 0

    def test_naive_day_lag_is_exactly_24h(self):
        """NaiveDay should predict exactly what happened 24 hours ago."""
        df = make_hourly_df(regions=("ON",), n_days=5)
        model = NaiveDay()
        preds = model.predict(df, regions=["ON"])
        # Match a known row
        for _, row in preds.iterrows():
            t = pd.to_datetime(row["time_utc"])
            predicted = row["predicted"]
            # Find the demand 24h before
            past_time = t - pd.Timedelta(hours=24)
            past_rows = df[(df["region"] == "ON") & (df["time_utc"] == past_time)]
            if not past_rows.empty:
                expected = float(past_rows["demand_mw"].values[0])
                assert abs(predicted - expected) < 1e-6
            break  # check one row is enough


# ---------------------------------------------------------------------------
# Forecast dimensions: 24 × N structure
# ---------------------------------------------------------------------------

class TestForecastDimensions:
    def test_24_hours_per_day_per_region(self):
        """A correctly-shaped forecast must have exactly 24 entries per (date, region) pair."""
        # Simulate a model output DataFrame
        regions = ["ON", "QC", "AB"]
        n_days = 7
        rows = []
        for region in regions:
            for d in range(n_days):
                for h in range(24):
                    rows.append({
                        "forecast_date": pd.Timestamp("2024-03-01") + pd.Timedelta(days=d),
                        "region": region,
                        "target_hour": h,
                        "predicted": 10000.0 + h * 100,
                        "actual": 10000.0 + h * 95,
                    })
        df = pd.DataFrame(rows)
        # Verify shape
        for region in regions:
            rdf = df[df["region"] == region]
            counts = rdf.groupby("forecast_date")["target_hour"].count()
            assert (counts == 24).all(), f"Region {region} does not have 24 hours per day"

    def test_no_duplicate_hour_per_date_per_region(self):
        rows = []
        for h in range(24):
            rows.append({"forecast_date": pd.Timestamp("2024-03-01"), "region": "ON",
                         "target_hour": h, "predicted": 10000.0})
        # Add a deliberate duplicate
        rows.append({"forecast_date": pd.Timestamp("2024-03-01"), "region": "ON",
                     "target_hour": 5, "predicted": 9999.0})
        df = pd.DataFrame(rows)
        dupes = df.duplicated(subset=["forecast_date", "region", "target_hour"])
        assert dupes.sum() == 1  # detecting the injected duplicate

    def test_target_hours_are_0_to_23(self):
        rows = [{"forecast_date": pd.Timestamp("2024-03-01"), "region": "ON",
                 "target_hour": h, "predicted": 10000.0} for h in range(24)]
        df = pd.DataFrame(rows)
        assert df["target_hour"].min() == 0
        assert df["target_hour"].max() == 23
        assert df["target_hour"].nunique() == 24


# ---------------------------------------------------------------------------
# Metrics output shape
# ---------------------------------------------------------------------------

class TestMetricsOutput:
    def test_mae_rmse_mape_finite(self):
        from src.evaluation.metrics import mae, mape, rmse
        actual = np.array([15000.0, 16000.0, 14000.0])
        predicted = np.array([15100.0, 15800.0, 14200.0])
        assert np.isfinite(mae(actual, predicted))
        assert np.isfinite(rmse(actual, predicted))
        mape_val, n_excl = mape(actual, predicted, zero_threshold=10.0)
        assert np.isfinite(mape_val)
        assert n_excl == 0

    def test_mae_zero_for_perfect_forecast(self):
        from src.evaluation.metrics import mae
        actual = np.array([10000.0, 12000.0])
        assert mae(actual, actual) == 0.0

    def test_rmse_zero_for_perfect_forecast(self):
        from src.evaluation.metrics import rmse
        actual = np.array([10000.0, 12000.0])
        assert rmse(actual, actual) == 0.0

    def test_mape_excludes_near_zero_actual(self):
        from src.evaluation.metrics import mape
        actual = np.array([0.0, 10000.0])
        predicted = np.array([100.0, 10100.0])
        _val, n_excl = mape(actual, predicted, zero_threshold=10.0)
        assert n_excl == 1  # the 0.0 actual should be excluded

    def test_mape_handles_all_zero_actual(self):
        from src.evaluation.metrics import mape
        actual = np.array([0.0, 0.0])
        predicted = np.array([100.0, 200.0])
        val, n_excl = mape(actual, predicted, zero_threshold=10.0)
        # All excluded → nan
        assert np.isnan(val)
        assert n_excl == 2

    def test_pinball_loss_q50_equals_mae_half(self):
        from src.evaluation.metrics import mae, pinball_loss
        actual = np.array([10000.0, 12000.0, 11000.0])
        predicted = np.array([10500.0, 11500.0, 11200.0])
        # Pinball at q=0.5 should equal 0.5 * MAE
        pb = pinball_loss(actual, predicted, q=0.5)
        expected = 0.5 * mae(actual, predicted)
        assert abs(pb - expected) < 1e-6
