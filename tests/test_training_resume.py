"""A resumed fit must preserve both optimizer results and model-named histories."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from scripts.confirm_grid_aware import train
from scripts.benchmark_grid_aware import device_data


def test_resume_matches_uninterrupted_training_and_preserves_history_name(tmp_path):
    rng = np.random.default_rng(101)
    normalized = rng.normal(size=(250, 2)).astype("float32")
    center, scale = np.array([100., 200.]), np.array([10., 20.])
    data = {"states": ["A", "B"], "features": ["load", "observed"],
            "center": center, "scale": scale, "feature_center": np.zeros(2), "feature_scale": np.ones(2),
            "target_center": 0., "target_scale": 1., "index": pd.date_range("2017-01-01", periods=250, freq="h"),
            "history": np.stack([normalized, np.ones_like(normalized)], -1),
            "dynamic": np.zeros((250, 2, 2), dtype="float32"), "normalized": normalized,
            "demand": normalized * scale + center, "calendar": np.zeros((250, 11), dtype="float32"),
            "holiday": np.zeros((250, 2), dtype="float32"),
            "splits": {"train": np.array([168, 169]), "validation": np.array([200])}}
    arrays = device_data(data, "cpu")
    resumed, continuous = tmp_path / "resumed", tmp_path / "continuous"
    resumed.mkdir()
    continuous.mkdir()
    args = SimpleNamespace(device="cpu", cuda_graphs=False, capture_steps=16, sample_batch=32,
                           residual_batch_times=2, epochs=1, patience=10, resume=False, output=resumed)
    prior_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        train("residual_identity", 42, data, arrays, args)
        args.epochs, args.resume = 2, True
        train("residual_identity", 42, data, arrays, args)
        args.output, args.resume = continuous, False
        train("residual_identity", 42, data, arrays, args)
    finally:
        torch.set_num_threads(prior_threads)
    history = json.loads((resumed / "residual_identity_42_history.json").read_text())
    assert [row["epoch"] for row in history] == [1, 2]
    assert not (resumed / "target_scale_history.json").exists()
    a = torch.load(resumed / "residual_identity_42_latest.pt", weights_only=False)
    b = torch.load(continuous / "residual_identity_42_latest.pt", weights_only=False)
    for key in a["state_dict"]:
        torch.testing.assert_close(a["state_dict"][key], b["state_dict"][key], rtol=0, atol=0)
