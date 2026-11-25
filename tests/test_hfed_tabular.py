from types import SimpleNamespace

import numpy as np

from src.models.hfed_tabular import tabular_features, predict_control, fit_control, _loaded_booster


def example_data():
    return {"dynamic": np.ones((2, 24, 20), dtype="float32"),
            "covariates": np.zeros((2, 24, 26), dtype="float32"),
            "anchor": np.ones((2, 24), dtype="float32"), "nodes": np.array([0, 1]),
            "states": ["A", "B"], "center": np.array([100., 200.]), "scale": np.array([10., 20.]),
            "features": ["demand_mw", "temperature_c", "relative_humidity", "wind_speed",
                         *[f"feature_{i}" for i in range(4, 20)]], "actual": np.zeros((2, 24))}


def test_tabular_inputs_do_not_depend_on_target_labels():
    data = example_data()
    first = tabular_features(data, np.array([0, 1]), "demand_weather_calendar")
    data["actual"][:] = 99999
    np.testing.assert_array_equal(first, tabular_features(data, np.array([0, 1]), "demand_weather_calendar"))


def test_weekly_baseline_uses_available_daily_anchor_when_week_is_missing():
    data = example_data()
    data["covariates"][0, 0, 6] = 2
    data["covariates"][0, 0, 13] = 1
    result = predict_control("naive_week", 42, data, np.array([0, 1]), SimpleNamespace())
    assert result[0, 0] == 120
    assert result[0, 1] == 110
    assert result[1, 0] == 220


def test_xgboost_checkpoint_roundtrip_and_endpoint_shape(tmp_path):
    data = example_data()
    data["actual"][0] = 110
    data["actual"][1] = 220
    data["splits"] = {"train": np.array([0]), "validation": np.array([1])}
    args = SimpleNamespace(output=tmp_path, resume=False, device="cpu", feature_set="demand_weather_calendar")
    fitted = fit_control("xgboost", 42, data, args)
    assert fitted["best_epoch"] >= 1
    first = predict_control("xgboost", 42, data, np.array([1]), args)
    _loaded_booster.cache_clear()
    second = predict_control("xgboost", 42, data, np.array([1]), args)
    assert first.shape == (1, 24)
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)
