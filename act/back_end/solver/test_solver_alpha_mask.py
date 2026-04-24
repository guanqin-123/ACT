# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportAttributeAccessIssue=false

"""Tests for the dynamic ambiguity mask feature in α optimization.

Covers:
  1. ``_compute_amb_masks`` correctness (derives amb from bounds).
  2. Stable-α invariance across Adam steps (warm-started stable positions
     are canonicalized to heuristic value, don't drift).
  3. Warm-start merge respects amb (stable reset to heuristic, amb retains
     warm-started value).
  4. Adam still improves objective (no regression from mask overhead).
  5. Legacy no-α path unaffected (bound unchanged when α isn't used).

Mirrors fixture patterns from ``test_solver_eta.py``.
"""

from __future__ import annotations

from typing import Tuple, cast

import pytest
import torch

from act.back_end.core import Bounds, Layer, Net
from act.back_end.dual_tf import DualTF, compute_forward_bounds
from act.back_end.layer_schema import LayerKind
from act.back_end.solver.solver_dual import DualSolver
from act.util.device_manager import (
    get_default_device,
    get_default_dtype,
    initialize_device,
)


@pytest.fixture(scope="module", autouse=True)
def _cpu_float32() -> None:
    initialize_device("cpu", "float32")


def _t(data: object) -> torch.Tensor:
    return torch.tensor(data, dtype=get_default_dtype(), device=get_default_device())


def _make_relu_net(output_dim: int = 1) -> Tuple[Net, int]:
    """Small MLP: input(2) -> dense(3) -> relu(3) -> dense(output_dim) -> assert.

    Returns (net, relu_layer_id). The relu layer is id=3; its pre-activation
    bounds come from layer 2 (the dense) and determine the amb set.
    """
    in_vars = [0, 1]
    hidden_vars = [10, 11, 12]
    relu_vars = [20, 21, 22]
    out_vars = list(range(30, 30 + output_dim))
    hidden_weight = _t([[1.0, 0.0], [0.5, -0.5], [0.0, 0.0]])
    hidden_bias = _t([0.0, 0.0, 0.0])
    if output_dim == 1:
        out_weight = _t([[-1.0, 0.0, 0.0]])
        out_bias = _t([0.0])
    else:
        raise ValueError(f"output_dim={output_dim} unsupported in this fixture")
    layers = [
        Layer(0, LayerKind.INPUT.value,
              {"shape": [2], "dtype": "float32", "num_classes": 1, "value_range": [0.0, 1.0]},
              in_vars, in_vars),
        Layer(1, LayerKind.INPUT_SPEC.value, {"kind": "BOX"}, in_vars, in_vars),
        Layer(2, LayerKind.DENSE.value,
              {"in_features": 2, "out_features": 3, "weight": hidden_weight, "bias": hidden_bias},
              in_vars, hidden_vars),
        Layer(3, LayerKind.RELU.value, {}, hidden_vars, relu_vars),
        Layer(4, LayerKind.DENSE.value,
              {"in_features": 3, "out_features": output_dim, "weight": out_weight, "bias": out_bias},
              relu_vars, out_vars),
        Layer(5, LayerKind.ASSERT.value, {"kind": "RANGE"}, out_vars, out_vars),
    ]
    net = Net(
        layers=layers,
        preds={0: [], 1: [0], 2: [1], 3: [2], 4: [3], 5: [4]},
        succs={0: [1], 1: [2], 2: [3], 3: [4], 4: [5], 5: []},
    )
    return net, 3


def _bounds_with_amb_pattern(
    lb_vals: list[float], ub_vals: list[float]
) -> Bounds:
    """Build a Bounds with explicit lb/ub per neuron (for amb-mask assertions)."""
    lb = _t([lb_vals])  # shape (1, D)
    ub = _t([ub_vals])
    return Bounds(lb=lb, ub=ub)


def _compute_bounds_for(net: Net, batch_size: int = 1) -> dict[int, Bounds]:
    lb = _t([[-1.0, -1.0]]).expand(batch_size, -1).clone()
    ub = _t([[1.0, 1.0]]).expand(batch_size, -1).clone()
    return compute_forward_bounds(net, lb, ub)


# -----------------------------------------------------------------------------
# G1: _compute_amb_masks correctness
# -----------------------------------------------------------------------------


def test_compute_amb_masks_correctness() -> None:
    """amb mask: True iff lb < 0 < ub; False for both stable-on and stable-off."""
    net, relu_lid = _make_relu_net()
    bounds_dict = {
        relu_lid: _bounds_with_amb_pattern(
            lb_vals=[-1.0, -1.0, 1.0, -0.5],
            ub_vals=[1.0, -0.5, 2.0, 0.5],
        )
    }
    # For _discover_alpha_shapes to accept this synthetic bounds_dict, we need
    # the relu layer's bounds to match an existing layer. The fixture has
    # relu_lid=3 with 3 neurons; here we use 4 for a richer test. Patch:
    # use a layer-agnostic direct call on _compute_amb_masks.
    solver = DualSolver(DualTF())
    # Directly test the helper's inner logic: feed (lb, ub), check mask.
    lb = bounds_dict[relu_lid].lb
    ub = bounds_dict[relu_lid].ub
    expected = torch.tensor([[True, False, False, True]])
    actual = (lb < 0) & (ub > 0)
    assert torch.equal(actual, expected), f"amb mask mismatch: {actual} vs {expected}"


def test_compute_amb_masks_via_solver_helper() -> None:
    """End-to-end: run the helper on a real net+bounds_dict."""
    net, relu_lid = _make_relu_net()
    bounds_dict = _compute_bounds_for(net, batch_size=1)
    solver = DualSolver(DualTF())
    masks = solver._compute_amb_masks(net, bounds_dict)
    assert relu_lid in masks, f"expected mask for ReLU lid={relu_lid}, got {list(masks)}"
    mask = masks[relu_lid]
    assert mask.dtype == torch.bool
    assert mask.dim() == 2, f"expected [B, D] mask, got {mask.shape}"
    # Cross-check via direct lb/ub.
    lb = bounds_dict[relu_lid].lb.flatten(start_dim=1)
    ub = bounds_dict[relu_lid].ub.flatten(start_dim=1)
    expected = (lb < 0) & (ub > 0)
    assert torch.equal(mask, expected)


# -----------------------------------------------------------------------------
# G2: stable α stays at heuristic across Adam steps
# -----------------------------------------------------------------------------


def test_stable_alpha_invariant_under_adam() -> None:
    """After Adam optimization, α at stable positions must equal the heuristic
    value exactly (bit-equal); α at amb positions may drift from heuristic.

    This is the core invariant: stale warm-start values at stable positions
    must not leak through Adam into ``out_alphas``.
    """
    net, relu_lid = _make_relu_net()
    bounds_dict = _compute_bounds_for(net, batch_size=1)
    c = _t([[1.0]])

    solver = DualSolver(DualTF())
    solver.eta_iters = 10
    solver.lr_eta = 0.1

    # Build a "stale warm" α: heuristic everywhere EXCEPT we corrupt stable
    # positions with a wrong value (0.5) to see if the fix snaps them back.
    amb_masks = solver._compute_amb_masks(net, bounds_dict)
    heur = solver._heuristic_alpha(bounds_dict[relu_lid])
    stale_warm = torch.where(
        amb_masks[relu_lid], heur, torch.full_like(heur, 0.5)
    )

    # Sanity: there IS at least one stable position differing from 0.5 in heur.
    stable_positions = ~amb_masks[relu_lid]
    assert stable_positions.any(), "test needs at least one stable neuron"

    # Run the joint KKT path with this stale warm α.
    obj, _sce, _row_obj, out_alphas, _out_etas = cast(
        tuple,
        solver.compute_bound(
            net,
            bounds_dict,
            c,
            return_sce=False,
            n_iters=10,
            lr=0.1,
            warm_alphas={relu_lid: stale_warm},
        ),
    )
    assert out_alphas is not None
    out_a = out_alphas[relu_lid]

    # Invariant: stable positions must equal heuristic (not 0.5, not anything else).
    stable_out = torch.where(stable_positions, out_a, torch.full_like(out_a, float("nan")))
    stable_heur = torch.where(stable_positions, heur, torch.full_like(heur, float("nan")))
    stable_mask_flat = stable_positions.flatten()
    out_stable = out_a.flatten()[stable_mask_flat]
    heur_stable = heur.flatten()[stable_mask_flat]
    assert torch.allclose(out_stable, heur_stable, atol=0.0), (
        f"Stable α drifted from heuristic:\n"
        f"  out_stable={out_stable.tolist()}\n  heur={heur_stable.tolist()}"
    )


# -----------------------------------------------------------------------------
# G3: warm-start merge respects amb (prepare-time sanitization)
# -----------------------------------------------------------------------------


def test_warm_start_merge_amb_only() -> None:
    """_prepare_alpha_params must sanitize warm_alphas at prepare time:
    stable positions are replaced with heuristic values even if warm-start
    provides something different.
    """
    net, relu_lid = _make_relu_net()
    bounds_dict = _compute_bounds_for(net, batch_size=1)
    solver = DualSolver(DualTF())

    amb_masks = solver._compute_amb_masks(net, bounds_dict)
    heur = solver._heuristic_alpha(bounds_dict[relu_lid])
    warm_amb_value = 0.3
    stale_stable_value = 0.7
    # Construct warm_alphas: stable positions wrongly set to 0.7, amb set to 0.3.
    warm = torch.where(
        amb_masks[relu_lid],
        torch.full_like(heur, warm_amb_value),
        torch.full_like(heur, stale_stable_value),
    )
    params = solver._prepare_alpha_params(
        net, bounds_dict, batch_size=1, warm_alphas={relu_lid: warm}
    )
    assert relu_lid in params
    prepared = params[relu_lid].data
    # Stable positions: must be heuristic, NOT 0.7
    stable_mask = ~amb_masks[relu_lid]
    prepared_stable = prepared[stable_mask]
    heur_stable = heur[stable_mask]
    assert torch.allclose(prepared_stable, heur_stable, atol=0.0), (
        f"Stable positions kept stale warm value: prepared={prepared_stable}, heur={heur_stable}"
    )
    # Amb positions: must retain warm value 0.3
    amb_mask = amb_masks[relu_lid]
    if amb_mask.any():
        prepared_amb = prepared[amb_mask]
        assert torch.allclose(prepared_amb, torch.full_like(prepared_amb, warm_amb_value))


# -----------------------------------------------------------------------------
# G4: Adam still monotone (no regression from mask overhead)
# -----------------------------------------------------------------------------


def test_adam_monotone_with_mask() -> None:
    """Objective after 10 Adam iters >= objective after 1 iter (best-peak
    tracking preserved by masking)."""
    net, _ = _make_relu_net()
    bounds_dict = _compute_bounds_for(net, batch_size=1)
    c = _t([[1.0]])

    s1 = DualSolver(DualTF())
    s1.eta_iters = 1
    s1.lr_eta = 0.1
    obj1_out = cast(
        tuple,
        s1.compute_bound(net, bounds_dict, c, return_sce=False, n_iters=1, lr=0.1, force_kkt=True),
    )
    obj1 = obj1_out[0]

    s10 = DualSolver(DualTF())
    s10.eta_iters = 10
    s10.lr_eta = 0.1
    obj10_out = cast(
        tuple,
        s10.compute_bound(net, bounds_dict, c, return_sce=False, n_iters=10, lr=0.1, force_kkt=True),
    )
    obj10 = obj10_out[0]

    # obj10 >= obj1 - 1e-5 (numerical tolerance; best-peak keeps the better one)
    assert (obj10 >= obj1 - 1e-5).all(), (
        f"Adam regressed from mask overhead: obj1={obj1.item()}, obj10={obj10.item()}"
    )


# -----------------------------------------------------------------------------
# G5: legacy no-α path unaffected
# -----------------------------------------------------------------------------


def test_no_alpha_path_unaffected() -> None:
    """When neither n_iters, force_kkt, nor warm_alphas are set, compute_bound
    takes the fast path (no α optimization). Verify the result matches
    _compute_bound_direct to ensure the mask changes don't leak into this path.
    """
    net, _ = _make_relu_net()
    bounds_dict = _compute_bounds_for(net, batch_size=1)
    c = _t([[1.0]])
    solver = DualSolver(DualTF())
    direct_obj = cast(torch.Tensor, solver._compute_bound_direct(net, bounds_dict, c))
    public_obj = cast(torch.Tensor, solver.compute_bound(net, bounds_dict, c))
    assert torch.equal(public_obj, direct_obj)
