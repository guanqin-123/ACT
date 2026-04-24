# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportUntypedFunctionDecorator=false
# ===- act/back_end/dual_tf/test_concat_end_to_end.py - CONCAT regression --===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025- ACT Team
# Licensed under AGPLv3+; distributed without warranty.
# ===---------------------------------------------------------------------===#
# End-to-end tests for CONCAT in the dual verification pipeline.
#
# Goal: prove that a Net containing a CONCAT layer runs without
# NotImplementedError through analyze() + DualSolver.compute_bound(), and
# that backward_concat correctly inverts forward_concat by splitting ν
# along concat_dim into per-predecessor slices.
#
# These tests do NOT mock dispatch_tf or any internals - they exercise the
# full real pipeline (load net -> compute_forward_bounds -> DualSolver).
# ===---------------------------------------------------------------------===#

from __future__ import annotations

import pytest
import torch

from act.back_end.core import Layer, Net
from act.back_end.dual_tf import DualTF, compute_forward_bounds
from act.back_end.dual_tf.tf_forward import backward_concat
from act.back_end.layer_schema import LayerKind
from act.back_end.solver.solver_dual import DualSolver
from act.util.device_manager import get_default_device, get_default_dtype, initialize_device


@pytest.fixture(scope="module", autouse=True)
def _cpu_float32() -> None:
    initialize_device("cpu", "float32")


def _t(data: object) -> torch.Tensor:
    return torch.tensor(data, dtype=get_default_dtype(), device=get_default_device())


def _make_mlp_concat_net() -> tuple[Net, torch.Tensor, torch.Tensor]:
    """2 parallel DENSE branches merged via CONCAT(dim=1), then DENSE -> ASSERT.

    Topology (DAG):
        INPUT(4) -> INPUT_SPEC -> DENSE_A(4->3) --.
                               '-> DENSE_B(4->2) --> CONCAT(dim=1)=5 -> DENSE_OUT(5->2) -> ASSERT
    """
    w_a = _t([[0.5, -0.2, 0.1, 0.0],
              [0.1, 0.3, -0.4, 0.2],
              [-0.3, 0.1, 0.2, 0.5]])
    b_a = _t([0.1, -0.05, 0.0])
    w_b = _t([[0.2, 0.1, -0.1, 0.3],
              [-0.2, 0.4, 0.05, -0.1]])
    b_b = _t([0.05, -0.1])
    w_out = _t([[0.3, -0.2, 0.1, 0.4, -0.3],
                [0.1, 0.2, -0.3, 0.2, 0.4]])
    b_out = _t([0.0, 0.0])

    n_in = 4
    input_ids = [0, 1, 2, 3]
    a_ids = [10, 11, 12]
    b_ids = [20, 21]
    concat_ids = [30, 31, 32, 33, 34]
    out_ids = [40, 41]

    lb = _t([[0.0, 0.0, 0.0, 0.0]])
    ub = _t([[1.0, 1.0, 1.0, 1.0]])

    layers = [
        Layer(0, LayerKind.INPUT.value,
              {"shape": [n_in], "dtype": "float32", "num_classes": 1, "value_range": [0.0, 1.0]},
              input_ids, input_ids),
        Layer(1, LayerKind.INPUT_SPEC.value,
              {"kind": "BOX", "lb": lb, "ub": ub},
              input_ids, input_ids),
        Layer(2, LayerKind.DENSE.value,
              {"in_features": n_in, "out_features": 3, "weight": w_a, "bias": b_a},
              input_ids, a_ids),
        Layer(3, LayerKind.DENSE.value,
              {"in_features": n_in, "out_features": 2, "weight": w_b, "bias": b_b},
              input_ids, b_ids),
        Layer(4, LayerKind.CONCAT.value, {"concat_dim": 1}, a_ids + b_ids, concat_ids),
        Layer(5, LayerKind.DENSE.value,
              {"in_features": 5, "out_features": 2, "weight": w_out, "bias": b_out},
              concat_ids, out_ids),
        Layer(6, LayerKind.ASSERT.value, {"kind": "RANGE"}, out_ids, out_ids),
    ]
    net = Net(
        layers=layers,
        preds={0: [], 1: [0], 2: [1], 3: [1], 4: [2, 3], 5: [4], 6: [5]},
        succs={0: [1], 1: [2, 3], 2: [4], 3: [4], 4: [5], 5: [6], 6: []},
    )
    return net, lb, ub


def test_concat_forward_bounds_preserve_batch_and_concat_width():
    net, lb, ub = _make_mlp_concat_net()
    fwd = compute_forward_bounds(net, lb, ub, post_activation=True)

    assert 4 in fwd, "CONCAT layer must have bounds in forward dict"
    concat_bounds = fwd[4]
    assert concat_bounds.lb.shape == (1, 5), (
        f"CONCAT output should be [B=1, 3+2=5]; got {tuple(concat_bounds.lb.shape)}"
    )
    assert concat_bounds.ub.shape == (1, 5)
    assert (concat_bounds.lb <= concat_bounds.ub).all(), "lb <= ub invariant violated"


def test_concat_backward_splits_nu_into_per_pred_slices():
    net, lb, ub = _make_mlp_concat_net()
    fwd = compute_forward_bounds(net, lb, ub, post_activation=True)

    nu = _t([[1.0, 2.0, 3.0, 4.0, 5.0]])
    concat_layer = net.by_id[4]
    pred_nus, contrib = backward_concat(concat_layer, nu, fwd, preds=[2, 3])

    assert len(pred_nus) == 2, "Must route ν to each of the 2 preds"
    assert pred_nus[0].shape == (1, 3), (
        f"First pred (DENSE_A, 3 outputs) should get nu[:, :3]; got {tuple(pred_nus[0].shape)}"
    )
    assert pred_nus[1].shape == (1, 2), (
        f"Second pred (DENSE_B, 2 outputs) should get nu[:, 3:]; got {tuple(pred_nus[1].shape)}"
    )
    assert torch.allclose(pred_nus[0], _t([[1.0, 2.0, 3.0]])), "Split content mismatch for pred 0"
    assert torch.allclose(pred_nus[1], _t([[4.0, 5.0]])), "Split content mismatch for pred 1"
    assert contrib.shape == (1,) and contrib.item() == 0.0, "CONCAT has no bias; contrib must be 0"


def test_concat_backward_rejects_size_mismatch():
    net, lb, ub = _make_mlp_concat_net()
    fwd = compute_forward_bounds(net, lb, ub, post_activation=True)
    concat_layer = net.by_id[4]
    bad_nu = _t([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])

    with pytest.raises(ValueError, match="backward_concat.*nu.shape"):
        backward_concat(concat_layer, bad_nu, fwd, preds=[2, 3])


def test_concat_backward_rejects_missing_concat_dim():
    net, lb, ub = _make_mlp_concat_net()
    fwd = compute_forward_bounds(net, lb, ub, post_activation=True)

    class _BadLayer:
        id = 99
        params: dict = {}

    nu = _t([[1.0, 2.0, 3.0, 4.0, 5.0]])
    with pytest.raises(ValueError, match="missing required 'concat_dim'"):
        backward_concat(_BadLayer(), nu, fwd, preds=[2, 3])


def test_concat_end_to_end_compute_bound_finite():
    net, lb, ub = _make_mlp_concat_net()
    fwd = compute_forward_bounds(net, lb, ub, post_activation=True)

    c = _t([[1.0, -1.0]])
    solver = DualSolver(DualTF(), n_iters=0)
    bound = solver.compute_bound(net, fwd, c)

    assert bound.shape == (1,), f"compute_bound output should be [B=1]; got {tuple(bound.shape)}"
    assert torch.isfinite(bound).all(), f"bound must be finite; got {bound}"
