"""Masked-label and stochastic-model parity before enabling HFED CUDA replay."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from scripts.benchmark_hfed_extension import create_model, forward_batch
from src.models.stgnn.cuda_training import CapturedSequenceStep, TensorPreservingPlateau


def examples(device):
    data = {"features": ["demand_mw", "temperature_c", "demand_mw_observed"],
            "states": ["A", "B", "C"], "horizons": list(range(1, 25)),
            "node_x": np.ones((3, 2)), "edge_index": np.array([[0, 1, 2], [1, 2, 0]]),
            "edge_features": np.ones((3, 1)), "center": np.array([10., 20., 30.]),
            "scale": np.array([3., 4., 5.]), "feature_center": np.zeros(3),
            "feature_scale": np.ones(3), "target_center": 10., "target_scale": 5.}
    tensors = {"dynamic": torch.randn(32, 24, 3, device=device),
               "covariates": torch.randn(32, 24, 26, device=device),
               "anchor": torch.randn(32, 24, device=device),
               "actual": torch.randn(32, 24, device=device) * 3 + 20,
               "nodes": torch.arange(32, device=device) % 3}
    tensors["dynamic"][:, :, 2] = 1
    tensors["actual"][::2, :7] = float("nan")
    return data, tensors


@pytest.mark.parametrize("name", ["grid_aware_gru", "residual_identity"])
def test_masked_label_loss_and_gradients_match_indexed_loss(name):
    torch.manual_seed(12)
    data, tensors = examples("cpu")
    a, center, scale = create_model(name, data, "cpu")
    b = deepcopy(a)
    a.eval()
    b.eval()
    ids = torch.arange(8)
    loss_a = forward_batch(a, tensors, ids, center, scale, name.startswith("grid_aware"), capture=True)[1]
    loss_b = forward_batch(b, tensors, ids, center, scale, name.startswith("grid_aware"))[1]
    torch.testing.assert_close(loss_a, loss_b)
    loss_a.backward()
    loss_b.backward()
    for p, q in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=1e-6, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA optimizer replay")
@pytest.mark.parametrize("name", ["grid_aware_lstm", "grid_aware_gru", "seasonal_identity", "residual_identity"])
def test_hfed_models_replay_independent_updates_with_dropout_and_missing_labels(name):
    torch.manual_seed(31)
    data, tensors = examples("cuda")
    a, center, scale = create_model(name, data, "cuda")
    b = deepcopy(a)
    for model in [a, b]:
        for module in model.modules():
            if isinstance(module, torch.nn.RNNBase):
                module.flatten_parameters()
    a.train()
    b.train()
    reference = name.startswith("grid_aware")
    optimizer = torch.optim.Adam if reference else torch.optim.AdamW
    optimizers = [optimizer(m.parameters(), lr=torch.tensor(.0003, device="cuda"),
                            capturable=True, foreach=False) for m in [a, b]]
    initial_rng = torch.cuda.get_rng_state().clone()
    def builder(ids, mask):
        return forward_batch(a, tensors, ids, center, scale, reference, capture=True)[1]
    captured = CapturedSequenceStep(a, optimizers[0], builder, 4, 2, "cuda",
                                    initial_ids=torch.arange(16, 24, device="cuda"))
    captured(torch.arange(8, device="cuda"))
    captured(torch.arange(8, 16, device="cuda"))
    torch.cuda.synchronize()
    torch.cuda.set_rng_state(initial_rng)
    for start in range(0, 16, 4):
        ids = torch.arange(start, start + 4, device="cuda")
        optimizers[1].zero_grad(set_to_none=False)
        forward_batch(b, tensors, ids, center, scale, reference)[1].backward()
        optimizers[1].step()
    for p, q in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(p, q, atol=1e-6, rtol=1e-5)


def test_plateau_preserves_tensor_storage_and_eager_float_behavior():
    a, b = torch.nn.Linear(2, 1), torch.nn.Linear(2, 1)
    tensor_lr = torch.tensor(.001)
    optimizers = [torch.optim.SGD(a.parameters(), lr=tensor_lr), torch.optim.SGD(b.parameters(), lr=.001)]
    schedulers = [TensorPreservingPlateau(o, patience=0, factor=.1) for o in optimizers]
    for scheduler in schedulers:
        scheduler.step(1.)
        scheduler.step(1.)
    assert optimizers[0].param_groups[0]["lr"] is tensor_lr
    assert float(tensor_lr) == pytest.approx(.0001)
    assert optimizers[1].param_groups[0]["lr"] == pytest.approx(.0001)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA scheduled optimizer replay")
@pytest.mark.parametrize("optimizer", [torch.optim.Adam, torch.optim.AdamW])
def test_plateau_reductions_change_captured_updates(optimizer):
    torch.manual_seed(8)
    a, b = torch.nn.Linear(3, 2).cuda(), torch.nn.Linear(3, 2).cuda()
    b.load_state_dict(a.state_dict())
    x, y = torch.randn(16, 3, device="cuda"), torch.randn(16, 2, device="cuda")
    lr = torch.tensor(.001, device="cuda")
    oa = optimizer(a.parameters(), lr=lr, weight_decay=.01, capturable=True, foreach=False)
    ob = optimizer(b.parameters(), lr=.001, weight_decay=.01, foreach=False)
    sa = TensorPreservingPlateau(oa, patience=0, factor=.1)
    sb = torch.optim.lr_scheduler.ReduceLROnPlateau(ob, patience=0, factor=.1)
    def builder(ids, mask):
        return torch.nn.functional.mse_loss(a(x[ids]), y[ids])
    captured = CapturedSequenceStep(a, oa, builder, 4, 2, "cuda")
    for _ in range(4):
        captured(torch.arange(8, device="cuda"))
        for offset in [0, 4]:
            ob.zero_grad(set_to_none=False)
            torch.nn.functional.mse_loss(b(x[offset:offset + 4]), y[offset:offset + 4]).backward()
            ob.step()
        sa.step(1.)
        sb.step(1.)
    assert oa.param_groups[0]["lr"] is lr
    assert float(lr) == pytest.approx(1e-6)
    for p, q in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(p, q, atol=1e-6, rtol=1e-5)
