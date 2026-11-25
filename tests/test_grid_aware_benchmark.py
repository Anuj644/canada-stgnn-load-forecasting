"""Causality, split boundaries, and parity with the actual Canadian model."""
import numpy as np
import pandas as pd
import torch
import pytest

from scripts.benchmark_grid_aware import HORIZONS, GridAware, ResidualAdapter, issue_indices, seasonal_positions, score
from scripts.confirm_grid_aware import sample_batch
from src.models.stgnn.grid_aware import SeasonalGridAware, TemporalResidual
from src.models.stgnn.cuda_training import CapturedStep, CapturedSequenceStep


def test_split_contains_history_and_all_targets():
    index = pd.date_range("2018-12-01", "2019-07-02", freq="h")
    issues = issue_indices(index, "2019-01-01", "2019-07-01")
    assert len(issues)
    assert (index[issues - 23] >= pd.Timestamp("2019-01-01")).all()
    assert (index[issues + 24] < pd.Timestamp("2019-07-01")).all()
    positions = seasonal_positions(issues)
    assert (positions <= issues[:, None, None]).all()
    np.testing.assert_array_equal(issues[:, None] + HORIZONS - positions[:, :, 0], 24)


def test_vectorized_adapter_matches_existing_residual_forward():
    torch.manual_seed(5)
    b, n = 2, 4
    adjacency = np.eye(n, dtype=np.float32) + 0.25
    adapter = ResidualAdapter(adjacency).eval()
    history = torch.randn(b, 24, n, 2)
    history[..., 1] = 1
    calendar = torch.randn(b, n, 24, 26)
    anchors = torch.randn(b, n, 24)
    selected = torch.tensor(HORIZONS - 1)
    batch = {"history": history, "covariates": calendar[:, :, selected],
             "anchor": anchors[:, :, selected]}
    expected = []
    for node in range(n):
        with torch.no_grad():
            pred = adapter.core(history, calendar[:, node], anchors[:, node],
                                torch.full((b,), node))
        expected.append(pred[:, selected])
    with torch.no_grad():
        actual = adapter(batch)
    torch.testing.assert_close(actual, torch.stack(expected, dim=1))


def test_paper_mape_does_not_apply_canadian_10mw_exclusion():
    result = score(np.array([2.0, 100.0]), np.array([1.0, 90.0]))
    assert result["n_obs"] == 2
    assert result["mape"] == 30.0


def test_sample_training_and_full_state_evaluation_match():
    torch.manual_seed(9)
    b, n, f = 2, 4, 3
    data = {"features": list(range(f)), "node_x": np.random.default_rng(9).normal(size=(n, 5)),
            "edge_index": np.array([[0, 1, 2, 3], [1, 2, 3, 0]]),
            "edge_features": np.zeros((4, 4))}
    seq = torch.randn(b, 24, n, f)
    nodes = torch.arange(n).repeat(b)
    flat = seq.permute(0, 2, 1, 3).reshape(b * n, 24, f)
    for backbone in ["lstm", "gru"]:
        model = GridAware(data, backbone=backbone).eval()
        with torch.no_grad():
            expected = model({"dynamic": seq})
            actual = model.forward_samples(flat, nodes).reshape(b, n, 3)
        torch.testing.assert_close(actual, expected)
        assert "embedding" in model.state_dict()


def test_sample_batch_uses_own_history_and_exact_endpoint_targets():
    dynamic = torch.arange(250 * 4 * 3).reshape(250, 4, 3).float()
    demand = torch.arange(250 * 4).reshape(250, 4).float()
    issues, nodes = torch.tensor([170, 190]), torch.tensor([2, 1])
    seq, target = sample_batch({"dynamic": dynamic, "demand": demand}, issues, nodes)
    torch.testing.assert_close(seq[0], dynamic[147:171, 2])
    torch.testing.assert_close(target[1], demand[torch.tensor([191, 196, 214]), 1])


def test_seasonal_extension_sample_and_full_state_outputs_match():
    torch.manual_seed(13)
    b, n, f, h = 2, 3, 4, 24
    data = {"features": list(range(f)), "node_x": np.ones((n, 2)), "horizons": list(range(1, h + 1)),
            "edge_index": np.array([[0, 1, 2], [1, 2, 0]]), "edge_features": np.ones((3, 1))}
    model = SeasonalGridAware(data).eval()
    seq, cov, anchor = torch.randn(b, 24, n, f), torch.randn(b, n, h, 26), torch.randn(b, n, h)
    with torch.no_grad():
        expected = model({"dynamic": seq, "covariates": cov, "anchor": anchor})
        actual = model.forward_samples(seq.permute(0, 2, 1, 3).reshape(b * n, 24, f),
                                       torch.arange(n).repeat(b), cov.reshape(b * n, h, 26),
                                       anchor.reshape(b * n, h)).reshape(b, n, h)
    torch.testing.assert_close(expected, actual)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA optimizer replay test")
def test_captured_adam_matches_eager_including_partial_batch():
    torch.manual_seed(17)
    a = torch.nn.Linear(3, 2).cuda()
    b = torch.nn.Linear(3, 2).cuda()
    b.load_state_dict(a.state_dict())
    x, y = torch.randn(16, 3, device="cuda"), torch.randn(16, 2, device="cuda")
    opts = [torch.optim.Adam(m.parameters(), lr=torch.tensor(0.001, device="cuda"),
                             capturable=True, foreach=False) for m in [a, b]]

    def loss_builder(ids, mask):
        return ((a(x[ids]) - y[ids]).square() * mask[:, None]).sum() / (mask.sum() * 2)

    captured = CapturedStep(a, opts[0], loss_builder, 8, "cuda")
    for ids in [torch.arange(8, device="cuda"), torch.tensor([8, 9, 10], device="cuda")]:
        captured(ids)
        opts[1].zero_grad(set_to_none=True)
        torch.nn.functional.mse_loss(b(x[ids]), y[ids]).backward()
        opts[1].step()
    for actual, expected in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA sequential optimizer replay test")
def test_sequential_replay_performs_independent_minibatch_updates():
    torch.manual_seed(22)
    a, b = torch.nn.Linear(3, 2).cuda(), torch.nn.Linear(3, 2).cuda()
    b.load_state_dict(a.state_dict())
    x, y = torch.randn(32, 3, device="cuda"), torch.randn(32, 2, device="cuda")
    opts = [torch.optim.Adam(m.parameters(), lr=torch.tensor(0.001, device="cuda"),
                             capturable=True, foreach=False) for m in [a, b]]

    def loss_builder(ids, mask):
        return ((a(x[ids]) - y[ids]).square() * mask[:, None]).sum() / (mask.sum() * 2)

    captured = CapturedSequenceStep(a, opts[0], loss_builder, 4, 4, "cuda")
    captured(torch.arange(16, device="cuda"))
    captured(torch.arange(16, 32, device="cuda"))
    for start in range(0, 32, 4):
        opts[1].zero_grad(set_to_none=False)
        torch.nn.functional.mse_loss(b(x[start:start + 4]), y[start:start + 4]).backward()
        opts[1].step()
    for actual, expected in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert opts[0].state[next(a.parameters())]["step"].item() == 8


def test_temporal_port_decodes_scaled_masks_and_matches_legacy_identity():
    torch.manual_seed(31)
    b, t, n, h = 2, 24, 3, 3
    data = {"states": list(range(n)), "features": ["demand_mw", "demand_mw_observed"],
            "center": np.array([100., 200., 300.]), "scale": np.array([10., 20., 30.]),
            "feature_center": np.array([150., 1.]), "feature_scale": np.array([50., 1.])}
    normalized = torch.randn(b, t, n)
    raw = normalized * torch.tensor(data["scale"]).float() + torch.tensor(data["center"]).float()
    seq = torch.stack([(raw - 150) / 50, torch.zeros_like(raw)], -1)
    history = torch.stack([normalized, torch.ones_like(normalized)], -1)
    cov, anchor = torch.randn(b, n, h, 26), torch.randn(b, n, h)
    legacy, port = ResidualAdapter(np.eye(n)).eval(), TemporalResidual(data).eval()
    port.core.load_state_dict(legacy.core.state_dict())
    with torch.no_grad():
        expected = legacy({"history": history, "covariates": cov, "anchor": anchor})
        actual = port.forward_samples(seq.permute(0, 2, 1, 3).reshape(b * n, t, 2),
                   torch.arange(n).repeat(b), cov.reshape(b * n, h, 26), anchor.reshape(b * n, h)).reshape(b, n, h)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA recurrent optimizer replay test")
@pytest.mark.parametrize("kind", ["lstm", "gru"])
def test_sequential_recurrent_updates_match_eager(kind):
    class TinyRecurrent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            recurrent = torch.nn.LSTM if kind == "lstm" else torch.nn.GRU
            self.rnn = recurrent(3, 4, batch_first=True)
            self.head = torch.nn.Linear(4, 2)

        def forward(self, seq):
            return self.head(self.rnn(seq)[0][:, -1])

    torch.manual_seed(47)
    a, b = TinyRecurrent().cuda(), TinyRecurrent().cuda()
    b.load_state_dict(a.state_dict())
    x, y = torch.randn(32, 5, 3, device="cuda"), torch.randn(32, 2, device="cuda")
    optimizers = [torch.optim.Adam(model.parameters(), lr=torch.tensor(0.001, device="cuda"),
                                   capturable=True, foreach=False) for model in [a, b]]

    def builder(ids, mask):
        return ((a(x[ids]) - y[ids]).square() * mask[:, None]).sum() / (mask.sum() * 2)

    captured = CapturedSequenceStep(a, optimizers[0], builder, 4, 4, "cuda")
    captured(torch.arange(16, device="cuda"))
    captured(torch.arange(16, 32, device="cuda"))
    for start in range(0, 32, 4):
        optimizers[1].zero_grad(set_to_none=False)
        torch.nn.functional.mse_loss(b(x[start:start + 4]), y[start:start + 4]).backward()
        optimizers[1].step()
    for actual, expected in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
