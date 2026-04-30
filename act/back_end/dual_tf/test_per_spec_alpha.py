# pyright: reportMissingImports=false, reportPrivateUsage=false

from __future__ import annotations

import pytest
import torch

from act.back_end.core import Bounds
from act.back_end.dual_tf import dual_relu_backward
from act.util.device_manager import initialize_device


@pytest.fixture(scope="module", autouse=True)
def _cpu_float32() -> None:
    initialize_device("cpu", "float32")


def _make_amb_bounds(B_flat: int, D: int) -> Bounds:
    lb = torch.full((B_flat, D), -1.0)
    ub = torch.full((B_flat, D), 1.0)
    return Bounds(lb=lb, ub=ub)


def test_dual_relu_backward_per_spec_alpha_flatten_correctness() -> None:
    B, M, D = 2, 3, 4
    bounds = _make_amb_bounds(B * M, D)
    nu = torch.linspace(-1.0, 1.0, B * M * D).view(B * M, D)

    alpha_per_spec = torch.linspace(0.0, 1.0, B * M * D).view(B, M, D).clamp(0.0, 1.0)
    alpha_flat = alpha_per_spec.reshape(B * M, D)

    v_out_per_spec, contrib_per_spec = dual_relu_backward(nu, bounds, alpha=alpha_per_spec)
    v_out_flat, contrib_flat = dual_relu_backward(nu, bounds, alpha=alpha_flat)

    assert torch.allclose(v_out_per_spec, v_out_flat, atol=1e-7)
    assert torch.allclose(contrib_per_spec, contrib_flat, atol=1e-7)
    assert tuple(v_out_per_spec.shape) == (B * M, D)
    assert tuple(contrib_per_spec.shape) == (B * M,)


def test_dual_relu_backward_per_spec_alpha_independent_per_spec() -> None:
    B, M, D = 1, 2, 1
    bounds = _make_amb_bounds(B * M, D)
    nu = torch.full((B * M, D), 0.5)

    alpha = torch.tensor([[[0.0], [1.0]]])
    v_out, _ = dual_relu_backward(nu, bounds, alpha=alpha)

    assert v_out[0, 0].item() == pytest.approx(0.0)
    assert v_out[1, 0].item() == pytest.approx(0.5)


@pytest.mark.parametrize(
    "alpha_a,alpha_b",
    [(0.0, 1.0), (0.25, 0.75), (0.5, 0.5), (1.0, 0.0)],
)
def test_dual_relu_backward_per_spec_alpha_soundness(alpha_a: float, alpha_b: float) -> None:
    B, M, D = 1, 2, 1
    l_val, u_val = -2.0, 3.0
    bounds = Bounds(
        lb=torch.tensor([[l_val], [l_val]]),
        ub=torch.tensor([[u_val], [u_val]]),
    )
    alpha = torch.tensor([[[alpha_a], [alpha_b]]])

    for v_val in [-2.0, -0.5, 0.5, 2.0]:
        nu = torch.tensor([[v_val], [v_val]])
        v_out, contrib = dual_relu_backward(nu, bounds, alpha=alpha)

        for spec_idx in (0, 1):
            vo = v_out[spec_idx, 0].item()
            input_contrib = l_val * max(vo, 0.0) + u_val * min(vo, 0.0)
            dual_bound = contrib[spec_idx].item() + input_contrib
            true_min = v_val * 0.0 if v_val > 0 else v_val * u_val
            assert dual_bound <= true_min + 1e-6, (
                f"UNSOUND per-spec α={(alpha_a, alpha_b)[spec_idx]} v={v_val} "
                f"bound={dual_bound:.4f} > true={true_min:.4f}"
            )


def test_dual_relu_backward_legacy_2d_alpha_unchanged() -> None:
    B, D = 4, 3
    bounds = _make_amb_bounds(B, D)
    nu = torch.linspace(-1.5, 1.5, B * D).view(B, D)
    alpha = torch.tensor(
        [[0.1, 0.5, 0.9], [0.0, 0.5, 1.0], [0.3, 0.6, 0.2], [0.7, 0.8, 0.4]]
    )

    v_out, contrib = dual_relu_backward(nu, bounds, alpha=alpha)

    assert tuple(v_out.shape) == (B, D)
    assert tuple(contrib.shape) == (B,)
    legacy_alpha_3d_singleton = alpha.unsqueeze(1)
    nu_singleton = nu.unsqueeze(1).reshape(B * 1, D)
    bounds_singleton = Bounds(lb=bounds.lb.unsqueeze(1).reshape(B * 1, D),
                              ub=bounds.ub.unsqueeze(1).reshape(B * 1, D))
    v_out_via_per_spec, contrib_via_per_spec = dual_relu_backward(
        nu_singleton, bounds_singleton, alpha=legacy_alpha_3d_singleton,
    )
    assert torch.allclose(v_out_via_per_spec.reshape(B, D), v_out, atol=1e-7)
    assert torch.allclose(contrib_via_per_spec, contrib, atol=1e-7)
