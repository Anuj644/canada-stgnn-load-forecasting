"""Unit tests for the residual CNN variants and tuning helpers."""
from pathlib import Path

import numpy as np
import pytest
import torch

from src.models.cnn_tuning import ResidualCNN, batch_loss, fit, make_tensors, predict, sample_weights, score

FEATURES = ["demand_mw", "temperature_c", "relative_humidity", "wind_speed", "demand_mw_observed",
            "temperature_c_observed", "relative_humidity_observed", "wind_speed_observed",
            *[f"calendar_{i}" for i in range(12)]]


@pytest.fixture()
def data():
    rng = np.random.default_rng(0)
    n, regions = 40, 2
    dynamic = rng.normal(size=(n, 24, len(FEATURES))).astype("float32")
    dynamic[:, :, 4:8] = 1
    return {"features": FEATURES, "states": ["AB", "ON"], "dynamic": dynamic,
            "covariates": rng.normal(size=(n, 24, 26)).astype("float32"),
            "anchor": rng.normal(size=(n, 24)).astype("float32"),
            "actual": (1000 + 100 * rng.normal(size=(n, 24))).astype("float32"),
            "nodes": rng.integers(0, regions, n), "center": np.array([1000.0, 1000.0]),
            "scale": np.array([100.0, 200.0]), "feature_center": np.zeros(len(FEATURES)),
            "feature_scale": np.ones(len(FEATURES))}


def test_weather_variant_adds_channels_and_keeps_output_shape(data):
    base, weather = ResidualCNN(data), ResidualCNN(data, weather=True)
    assert sum(p.numel() for p in weather.parameters()) > sum(p.numel() for p in base.parameters())
    tensors = make_tensors(data, "cpu")
    ids = torch.arange(5)
    for model in (base, weather):
        out = model.forward_samples(tensors["dynamic"][ids], tensors["nodes"][ids], tensors["covariates"][ids],
                                    tensors["anchor"][ids])
        assert out.shape == (5, 24)


def test_loss_ignores_missing_targets(data):
    data["actual"][:, 5] = np.nan
    tensors = make_tensors(data, "cpu")
    loss, prediction, valid = batch_loss(ResidualCNN(data), tensors, torch.arange(8))
    assert torch.isfinite(loss) and not valid[:, 5].any()


def test_sample_weights_have_unit_mean_on_training_ids(data):
    tensors = make_tensors(data, "cpu")
    train = torch.arange(30)
    weights = sample_weights(tensors, train, 1.0)
    assert weights[train].mean().item() == pytest.approx(1.0, rel=1e-5)
    assert (sample_weights(tensors, train, 0.0) == 1).all()


def test_fit_is_deterministic_and_restores_best_state(data):
    tensors = make_tensors(data, "cpu")
    runs = []
    for _ in range(2):
        torch.manual_seed(1)
        model = ResidualCNN(data)
        info = fit(model, tensors, np.arange(30), np.arange(30, 40), seed=1, epochs=3)
        runs.append(predict(model, tensors, np.arange(30, 40)))
        assert 1 <= info["best_epoch"] <= 3
    np.testing.assert_allclose(runs[0], runs[1])


def test_score_matches_hand_computation():
    actual = np.array([[100.0, 200.0], [np.nan, 300.0]])
    prediction = np.array([[110.0, 190.0], [0.0, 330.0]])
    result = score(prediction, actual, np.array([0, 1]), ["AB", "ON"])
    assert result["mae"] == pytest.approx((10 + 10 + 30) / 3)
    assert result["by_province_mae"] == {"AB": 10.0, "ON": 30.0}
    assert result["n_obs"] == 3
    assert result["macro_mape"] == pytest.approx(np.mean([7.5, 10.0]))
