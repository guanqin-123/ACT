# pyright: reportMissingImports=false, reportPrivateUsage=false

"""Regression: per-spec alpha must survive BaB-style warm-start.

Discovered empirically on instance 3995 (CIFAR-100 ResNet-medium):
``ValueError: DualSolver.compute_bound: warm alpha width mismatch at layer 3;
expected 14400, got 1440000``.

The handoff (HANDOFF_tier4_session4.md) claims Phase 3's
``test_dual_solver_alpha_per_spec_warm_start_roundtrip`` covered warm-start.
It does -- but only with a 1-ReLU net where
``optimize_initial_intermediate_bounds`` yields no intermediate-sid alphas
(no upstream ReLUs). The failing path requires:

1. >= 2 ReLU layers, so the pre-BaB tightener actually populates
   intermediate-sid alphas.
2. ``alpha_per_spec=True``, so ``AlphaState.per_spec`` is True.
3. A second ``evaluate_spec`` call after the first call returns FINAL_SID
   per-spec alphas, with the parent state merged BaB-style via
   ``_merge_alpha_states``.

That third condition is what the existing roundtrip test misses: it
``set_alphas(result1.out_alphas)`` directly, bypassing the BaB merge.
"""

from __future__ import annotations

import pytest
import torch

from act.back_end.bab.node import SubproblemBatch, _concat_subproblem_batches
from act.back_end.core import Bounds, Layer, Net
from act.back_end.dual_tf import DualTF, compute_forward_bounds
from act.back_end.layer_schema import LayerKind
from act.back_end.solver.alpha_state import AlphaState
from act.back_end.solver._initial_alpha_crown import (
    optimize_initial_intermediate_bounds,
)
from act.back_end.solver.solver_dual import DualSolver
from act.front_end.specs import OutKind, OutputSpec
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


def _make_two_relu_net() -> Net:
    """2-ReLU MLP: input(2) -> dense(3) -> relu -> dense(3) -> relu -> dense(2)."""
    in_vars = [0, 1]
    h1 = [10, 11, 12]
    r1 = [20, 21, 22]
    h2 = [30, 31, 32]
    r2 = [40, 41, 42]
    out_vars = [50, 51]
    layers = [
        Layer(0, LayerKind.INPUT.value,
              {"shape": (2,), "dtype": "float32", "num_classes": 1, "value_range": (0.0, 1.0)},
              in_vars, in_vars),
        Layer(1, LayerKind.INPUT_SPEC.value, {"kind": "BOX"}, in_vars, in_vars),
        Layer(2, LayerKind.DENSE.value,
              {"in_features": 2, "out_features": 3,
               "weight": _t([[1.0, 0.0], [0.5, -0.5], [0.2, 0.3]]),
               "bias": _t([0.0, 0.0, 0.1])},
              in_vars, h1),
        Layer(3, LayerKind.RELU.value, {}, h1, r1),
        Layer(4, LayerKind.DENSE.value,
              {"in_features": 3, "out_features": 3,
               "weight": _t([[0.5, -0.3, 0.2], [-0.2, 0.6, -0.1], [0.1, -0.1, 0.4]]),
               "bias": _t([0.05, -0.05, 0.0])},
              r1, h2),
        Layer(5, LayerKind.RELU.value, {}, h2, r2),
        Layer(6, LayerKind.DENSE.value,
              {"in_features": 3, "out_features": 2,
               "weight": _t([[-1.0, 0.0, 0.2], [0.0, 1.0, -0.1]]),
               "bias": _t([0.0, 0.0])},
              r2, out_vars),
        Layer(7, LayerKind.ASSERT.value, {"kind": "RANGE"}, out_vars, out_vars),
    ]
    return Net(
        layers=layers,
        preds={0: [], 1: [0], 2: [1], 3: [2], 4: [3], 5: [4], 6: [5], 7: [6]},
        succs={0: [1], 1: [2], 2: [3], 3: [4], 4: [5], 5: [6], 6: [7], 7: []},
    )


def _bounds(net: Net, B: int = 1) -> dict[int, Bounds]:
    lb = _t([[-1.0, -1.0]]).expand(B, -1).contiguous()
    ub = _t([[1.0, 1.0]]).expand(B, -1).contiguous()
    return compute_forward_bounds(net, lb, ub)


def _multi_spec() -> OutputSpec:
    return OutputSpec(kind=OutKind.RANGE, lb=_t([-1.0, -1.0]), ub=_t([1.0, 1.0]))


def test_per_spec_alpha_survives_concat_then_evaluate_spec() -> None:
    """End-to-end: pre-BaB -> evaluate -> sibling concat -> evaluate (the bug path).

    Mirrors ``_verify_bab_batched`` (bab.py:624-741) AND the BFS pop path
    that calls ``_concat_subproblem_batches`` (bounding.py:224, 293):

    1. ``optimize_initial_intermediate_bounds(per_spec=True)`` populates the
       root ``AlphaState`` with intermediate-sid alphas.
    2. First ``evaluate_spec`` adds FINAL_SID per-spec 3-D alphas.
    3. Two sibling subproblem batches are concatenated -- this is the
       producer that drops ``per_spec=True`` (node.py:531) in the bug.
    4. Second ``evaluate_spec`` on the concat result -- in the bug, raises
       ``warm alpha width mismatch at layer 3`` because the dropped flag
       routes the still-3-D FINAL_SID through the legacy expand path.
    """
    net = _make_two_relu_net()
    bounds = _bounds(net, B=1)
    spec = _multi_spec()

    new_bounds, root_alpha_state = optimize_initial_intermediate_bounds(
        net, bounds, alpha_iters=1, lr_alpha=0.5, per_spec=True,
    )
    assert root_alpha_state.per_spec is True
    intermediate_sids = [
        sid for sid in root_alpha_state.start_nodes if sid != AlphaState.FINAL_SID
    ]
    assert intermediate_sids, (
        "regression invariant: pre-BaB tightener must populate intermediate-sid "
        "alphas for the bug to reproduce; net needs >= 2 ReLU layers"
    )

    solver = DualSolver(DualTF())
    solver.eta_iters = 1
    solver.alpha_per_spec = True
    solver.set_alphas(root_alpha_state)

    result1 = solver.evaluate_spec(net, new_bounds, spec)
    assert result1.out_alphas is not None
    assert result1.out_alphas.per_spec is True

    sibling_alpha = result1.out_alphas.clone()
    lb_root, ub_root = new_bounds[0].lb, new_bounds[0].ub
    depth = torch.tensor([0], dtype=torch.long)
    sib1 = SubproblemBatch(lb=lb_root, ub=ub_root, depths=depth, alphas=result1.out_alphas)
    sib2 = SubproblemBatch(lb=lb_root, ub=ub_root, depths=depth, alphas=sibling_alpha)
    merged = _concat_subproblem_batches(sib1, sib2)
    assert merged.alphas is not None
    assert merged.alphas.per_spec is True, (
        "regression: _concat_subproblem_batches dropped per_spec=True; "
        "the next evaluate_spec will trip warm alpha width mismatch"
    )

    bounds_b2 = {lid: Bounds(lb=b.lb.repeat(2, *([1] * (b.lb.dim() - 1))),
                              ub=b.ub.repeat(2, *([1] * (b.ub.dim() - 1))))
                 for lid, b in new_bounds.items()}

    solver.set_alphas(merged.alphas)
    result2 = solver.evaluate_spec(net, bounds_b2, spec)
    assert result2.out_alphas is not None
    assert result2.out_alphas.per_spec is True
