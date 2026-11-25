"""The uncertainty calculation must preserve matching keys and observation counts."""
import numpy as np
import pandas as pd
import pytest

from scripts.report_cross_dataset import matched_differences, weather_regimes


def save_forecasts(folder):
    times = pd.date_range("2019-07-01", periods=20 * 24, freq="h")
    for offset, seed in enumerate([42, 123, 456]):
        for name, error in [("candidate", 8 + offset), ("reference", 10 + offset)]:
            frame = pd.DataFrame({"issued_at": times, "state": "A", "horizon": 24,
                                  "actual": 100.0, "predicted": 100.0 + error})
            frame.to_parquet(folder / f"predictions_{name}_{seed}.parquet", index=False)


def test_paired_interval_uses_all_seeds_and_constant_improvement(tmp_path):
    save_forecasts(tmp_path)
    result = matched_differences(tmp_path, "candidate", "reference", [42, 123, 456], 24)
    assert result["paired_mae_difference"] == -2
    np.testing.assert_allclose(result["week_block_95pct_ci"], [-2, -2])
    assert result["lower_mae_each_seed"]
    assert result["n_observations_per_seed"] == 480
    assert result["candidate_mean_mae"] == 9


def test_paired_comparison_refuses_different_forecast_keys(tmp_path):
    save_forecasts(tmp_path)
    path = tmp_path / "predictions_candidate_123.parquet"
    pd.read_parquet(path).iloc[1:].to_parquet(path, index=False)
    with pytest.raises(ValueError, match="keys differ"):
        matched_differences(tmp_path, "candidate", "reference", [42, 123, 456], 24)


def test_ablation_comparison_reads_distinct_experiment_folders(tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    left.mkdir()
    right.mkdir()
    save_forecasts(left)
    save_forecasts(right)
    for seed in [42, 123, 456]:
        frame = pd.read_parquet(right / f"predictions_reference_{seed}.parquet")
        frame.to_parquet(right / f"predictions_candidate_{seed}.parquet", index=False)
    result = matched_differences(left, "candidate", "candidate", [42, 123, 456], 24, reference_folder=right)
    assert result["paired_mae_difference"] == -2


def test_weather_thresholds_use_training_only_and_decode_observation_masks(tmp_path):
    records = pd.DataFrame({"issued_at": pd.date_range("2023-12-28", periods=5), "region": "A"})
    records.to_parquet(tmp_path / "records.parquet", index=False)
    temperature = np.repeat(np.array([10., 20., -100., 100., 1000.])[:, None], 24, axis=1)
    mask = np.ones_like(temperature)
    mask[-1, 11:] = 0  # Insufficient observed weather must stay unclassified.
    dynamic = np.stack([(temperature - 10) / 2, (mask - .25) / .5], axis=-1)
    np.savez(tmp_path / "examples.npz", dynamic=dynamic, feature_center=[10, .25],
             feature_scale=[2, .5], split_train=[0, 1])
    regimes, thresholds = weather_regimes(tmp_path, {"states": ["A"],
        "features": ["temperature_c", "temperature_c_observed"]})
    assert thresholds["A"] == {"cold_5pct_c": 10.5, "hot_95pct_c": 19.5}
    assert regimes.weather_regime.tolist() == ["cold_at_issue", "hot_at_issue", "cold_at_issue",
                                               "hot_at_issue", "weather_unavailable"]
