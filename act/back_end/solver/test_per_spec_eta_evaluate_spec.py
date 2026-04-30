"""Phase 4 test: per-spec eta survives evaluate_spec output boundary.

Mirrors the Tier 4 test_per_spec_alpha_evaluate_spec.py pattern but for
EtaState. Validates:
  1. Default-off (eta_per_spec=False): out_etas is legacy 2-D EtaState.
  2. Per-spec on: out_etas has per_spec=True with val [B, M, D] and
     sign/point [B, D] (per the Tier 6 storage contract).
  3. Warm-start roundtrip: set_eta(per_spec=True) -> evaluate_spec ->
     out_etas preserves per_spec across cycles (the Tier 4 bugfix mirror).
"""
# pyright: reportMissingImports=false, reportPrivateUsage=false

from __future__ import annotations

import pytest
import torch

from act.back_end.bab.eta import EtaState
from act.back_end.core import Bounds, Layer, Net
from act.back_end.dual_tf import DualTF, compute_forward_bounds
from act.back_end.layer_schema import LayerKind
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


def _make_test_net() -> Net:
    in_vars = [0, 1]
    h = [10, 11, 12]
    r = [20, 21, 22]
    out_vars = [30, 31]
    layers = [
        Layer(0, LayerKind.INPUT.value,
              {"shape": (2,), "dtype": "float32", "num_classes": 1, "value_range": (0.0, 1.0)},
              in_vars, in_vars),
        Layer(1, LayerKind.INPUT_SPEC.value, {"kind": "BOX"}, in_vars, in_vars),
        Layer(2, LayerKind.DENSE.value,
              {"in_features": 2, "out_features": 3,
               "weight": _t([[1.0, 0.0], [0.5, -0.5], [0.2, 0.3]]),
               "bias": _t([0.0, 0.0, 0.1])},
              in_vars, h),
        Layer(3, LayerKind.RELU.value, {}, h, r),
        Layer(4, LayerKind.DENSE.value,
              {"in_features": 3, "out_features": 2,
               "weight": _t([[-1.0, 0.0, 0.2], [0.0, 1.0, -0.1]]),
               "bias": _t([0.0, 0.0])},
              r, out_vars),
        Layer(5, LayerKind.ASSERT.value, {"kind": "RANGE"}, out_vars, out_vars),
    ]
    return Net(
        layers=layers,
        preds={0: [], 1: [0], 2: [1], 3: [2], 4: [3], 5: [4]},
        succs={0: [1], 1: [2], 2: [3], 3: [4], 4: [5], 5: []},
    )


def _bounds(net: Net, B: int = 1) -> dict[int, Bounds]:
    lb = _t([[-1.0, -1.0]]).expand(B, -1).contiguous()
    ub = _t([[1.0, 1.0]]).expand(B, -1).contiguous()
    return compute_forward_bounds(net, lb, ub)


def _multi_spec() -> OutputSpec:
    return OutputSpec(kind=OutKind.RANGE, lb=_t([-1.0, -1.0]), ub=_t([1.0, 1.0]))


def _seed_eta(net: Net, B: int, *, per_spec: bool, M: int = 0) -> EtaState:
    affine_lid = 2
    width = 3
    if per_spec:
        val = {affine_lid: torch.zeros(B, M, width)}
    else:
        val = {affine_lid: torch.zeros(B, width)}
    sign = {affine_lid: torch.zeros(B, width)}
    sign[affine_lid][:, 0] = 1.0
    point = {affine_lid: torch.zeros(B, width)}
    return EtaState(val=val, sign=sign, point=point, per_spec=per_spec)


def test_dual_solver_eta_per_spec_default_off_returns_legacy_etastate() -> None:
    net = _make_test_net()
    bounds = _bounds(net, B=1)
    spec = _multi_spec()
    solver = DualSolver(DualTF())
    solver.eta_iters = 1
    assert solver.eta_per_spec is False

    eta_seed = _seed_eta(net, B=1, per_spec=False)
    solver.set_eta(eta_seed)
    result = solver.evaluate_spec(net, bounds, spec)

    assert result.out_etas is not None
    assert result.out_etas.per_spec is False
    sample_val = next(iter(result.out_etas.val.values()))
    assert sample_val.dim() == 2
    assert sample_val.shape[0] == 1


def _discover_spec_m(net: Net, bounds: dict[int, Bounds], spec: OutputSpec) -> int:
    solver = DualSolver(DualTF())
    solver.eta_iters = 0
    result = solver.evaluate_spec(net, bounds, spec)
    return int(result.margins.shape[1])


def test_dual_solver_eta_per_spec_on_returns_per_spec_etastate() -> None:
    net = _make_test_net()
    bounds = _bounds(net, B=1)
    spec = _multi_spec()
    M = _discover_spec_m(net, bounds, spec)

    solver = DualSolver(DualTF())
    solver.eta_iters = 1
    solver.eta_per_spec = True

    eta_seed = _seed_eta(net, B=1, per_spec=True, M=M)
    solver.set_eta(eta_seed)
    result = solver.evaluate_spec(net, bounds, spec)

    assert result.out_etas is not None
    assert result.out_etas.per_spec is True
    sample_val = next(iter(result.out_etas.val.values()))
    assert sample_val.dim() == 3
    assert sample_val.shape[0] == 1
    assert sample_val.shape[1] == M


def test_dual_solver_eta_per_spec_warm_start_roundtrip() -> None:
    net = _make_test_net()
    bounds = _bounds(net, B=1)
    spec = _multi_spec()
    M = _discover_spec_m(net, bounds, spec)

    solver = DualSolver(DualTF())
    solver.eta_iters = 3
    solver.eta_per_spec = True

    eta_seed = _seed_eta(net, B=1, per_spec=True, M=M)
    solver.set_eta(eta_seed)
    result1 = solver.evaluate_spec(net, bounds, spec)
    assert result1.out_etas is not None
    assert result1.out_etas.per_spec is True

    solver.set_eta(result1.out_etas)
    assert solver._eta_state is not None
    assert solver._eta_state.per_spec is True

    result2 = solver.evaluate_spec(net, bounds, spec)
    assert result2.out_etas is not None
    assert result2.out_etas.per_spec is True

    sample_val_2 = next(iter(result2.out_etas.val.values()))
    assert sample_val_2.shape == (1, M, 3)
