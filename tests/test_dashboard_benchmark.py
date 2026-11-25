"""Causal serving and parity with the actual verified HFED predictions."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.benchmark_forecast import BenchmarkForecaster, SPECS, build_request, request_observations
from src.evaluation.target_weather import classify_target_days

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def engine():
    if not (ROOT / "data/processed/hfed_extension_eccc_v2/examples.npz").exists():
        pytest.skip("Local verified benchmark cache required")
    return BenchmarkForecaster(ROOT)


@pytest.mark.parametrize("region", ["ON", "NL"])
def test_serving_features_match_frozen_benchmark(engine, region):
    records = engine.data["records"]
    day = records.loc[records.region.eq(region) & records.forecast_date.between("2025-01-10", "2025-02-01")].forecast_date.iloc[0]
    demand, weather, holidays, _, _ = request_observations(ROOT, region, day)
    sample, _, _, _ = build_request(engine.data, region, day, demand, weather, holidays)
    expected = engine.cached_sample(region, day)
    for key in sample:
        np.testing.assert_allclose(sample[key], expected[key], atol=1e-6, rtol=1e-6, equal_nan=True, err_msg=key)


@pytest.mark.parametrize("region,date", [("ON", "2025-01-15"), ("QC", "2025-12-31")])
def test_every_served_checkpoint_matches_saved_forecasts(engine, region, date):
    day = pd.Timestamp(date)
    sample = engine.cached_sample(region, day)
    assert sample is not None
    predictions = engine.predict(sample, tuple(SPECS), (42,))
    record = engine.data["records"].loc[engine.data["records"].region.eq(region) & engine.data["records"].forecast_date.eq(day)].iloc[0]
    for key, spec in SPECS.items():
        saved = pd.read_parquet(ROOT / "results" / spec.folder / "test2025" / f"predictions_{spec.model}_42.parquet")
        saved = saved.loc[saved.state.eq(region) & saved.issued_at.eq(record.issued_at)].sort_values("horizon")
        served = predictions.loc[predictions.key.eq(key)].sort_values("hour")
        assert len(served) == 24 and len(saved) > 0
        comparable = served.set_index("hour").loc[saved.horizon.to_numpy()-1].predicted
        # GPU serving follows the exact benchmark batch shape, including the
        # final partial batch. CPU fallback has documented backend roundoff.
        tolerance = dict(rtol=1e-5, atol=.02) if engine.device == "cuda" else dict(rtol=3e-4, atol=.05)
        np.testing.assert_allclose(comparable, saved.predicted, **tolerance, err_msg=key)


def test_cpu_fallback_is_close_to_benchmark_backend(engine):
    cpu = BenchmarkForecaster(ROOT, device="cpu")
    sample = engine.cached_sample("ON", "2025-01-15")
    keys = ("cnn", "lstm", "graph")
    expected = engine.predict(sample, keys, (42,))
    actual = cpu.predict(sample, keys, (42,))
    assert len(actual) == 72 and np.isfinite(actual.predicted).all()
    np.testing.assert_allclose(actual.predicted, expected.predicted, rtol=3e-4, atol=.05)


def test_future_demand_and_weather_cannot_change_forecast_features(engine):
    demand, weather, holidays, _, _ = request_observations(ROOT, "ON", "2025-01-15")
    first, issue, _, _ = build_request(engine.data, "ON", "2025-01-15", demand, weather, holidays)
    demand = demand.copy()
    demand.loc[demand.index >= issue] = 1e8
    weather = weather.copy()
    weather.loc[weather.index >= issue] = 1e8
    second, _, _, _ = build_request(engine.data, "ON", "2025-01-15", demand, weather, holidays)
    for key in ["dynamic", "covariates", "anchor", "nodes"]:
        np.testing.assert_array_equal(first[key], second[key])
    assert not np.array_equal(first["actual"], second["actual"])


def test_forecasts_exist_without_target_actuals(engine):
    demand, weather, holidays, _, _ = request_observations(ROOT, "ON", "2025-01-15")
    demand = demand.loc[demand.index < pd.Timestamp("2025-01-15 05:00")]
    sample, _, _, _ = build_request(engine.data, "ON", "2025-01-15", demand, weather, holidays)
    assert np.isnan(sample["actual"]).all()
    forecast = engine.predict(sample, ("cnn",), (42,))
    assert len(forecast) == 24 and np.isfinite(forecast.predicted).all()


def test_dst_and_unavailable_history_are_explicit(engine):
    demand, weather, holidays, _, _ = request_observations(ROOT, "ON", "2025-01-15")
    with pytest.raises(ValueError, match="daylight-saving"):
        build_request(engine.data, "ON", "2025-03-09", demand, weather, holidays)
    with pytest.raises(ValueError, match="at least 12"):
        build_request(engine.data, "ON", "2025-01-15", demand.iloc[:2], weather, holidays)


def test_target_day_extreme_thresholds_are_training_only():
    days = pd.date_range("2023-12-28", periods=4)
    records = pd.DataFrame({"region": "ON", "issued_at": days-pd.Timedelta(hours=1), "target_start": days})
    weather = pd.DataFrame({"region": "ON", "time_utc": pd.date_range(days[0], periods=96, freq="h"),
                            "temperature_c": np.repeat([10., 20., -100., 100.], 24)})
    labels, thresholds = classify_target_days(records, np.array([0, 1]), weather, ["ON"])
    assert thresholds["ON"] == {"cold_5pct_c": 10.5, "hot_95pct_c": 19.5}
    assert labels.regime.tolist() == ["cold_target_day", "hot_target_day", "cold_target_day", "hot_target_day"]
