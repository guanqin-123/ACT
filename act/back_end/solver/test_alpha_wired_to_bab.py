from __future__ import annotations

from typing import Any, cast

import pytest
import torch

import act.back_end.bab.bab as bab_module
from act.back_end.config import BaBConfig
from act.back_end.core import Layer, Net
from act.back_end.dual_tf import DualTF, compute_forward_bounds
from act.back_end.layer_schema import LayerKind
from act.back_end.solver._backward_truncated import backward_truncated_lb
from act.back_end.solver.alpha_state import AlphaState
from act.back_end.solver.solver_dual import DualSolver
from act.back_end.solver.solver_interval import TorchLPSolver
from act.front_end.specs import OutKind, OutputSpec
from act.util.device_manager import get_default_device, get_default_dtype, initialize_device
from act.util.stats import VerifyStatus


def setup_module() -> None:
    initialize_device("cpu", "float64")


def _t(data: object) -> torch.Tensor:
    return torch.tensor(data, dtype=get_default_dtype(), device=get_default_device())


def _sample_box(lb: torch.Tensor, ub: torch.Tensor, *, count: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device=get_default_device()).manual_seed(seed)
    return lb + torch.rand((count, lb.shape[-1]), generator=gen, dtype=lb.dtype, device=lb.device) * (ub - lb)


def _make_four_layer_relu_net() -> Net:
    in_vars = [0, 1]
    z1, a1 = [10, 11], [20, 21]
    z2, a2 = [30, 31], [40, 41]
    z3, a3 = [50, 51], [60, 61]
    out = [70]
    return Net(
        layers=[
            Layer(0, LayerKind.INPUT.value, {"shape": (2,), "dtype": "float64", "num_classes": 1, "value_range": (0.0, 1.0)}, in_vars, in_vars),
            Layer(1, LayerKind.INPUT_SPEC.value, {"kind": "BOX", "lb": _t([-1.0, -1.0]), "ub": _t([1.0, 1.0])}, in_vars, in_vars),
            Layer(2, LayerKind.DENSE.value, {"in_features": 2, "out_features": 2, "weight": _t([[1.2, -0.4], [0.5, 0.9]]), "bias": _t([0.1, -0.2])}, in_vars, z1),
            Layer(3, LayerKind.RELU.value, {}, z1, a1),
            Layer(4, LayerKind.DENSE.value, {"in_features": 2, "out_features": 2, "weight": _t([[0.8, -0.3], [-0.6, 1.1]]), "bias": _t([0.05, 0.1])}, a1, z2),
            Layer(5, LayerKind.RELU.value, {}, z2, a2),
            Layer(6, LayerKind.DENSE.value, {"in_features": 2, "out_features": 2, "weight": _t([[1.0, 0.2], [-0.7, 0.6]]), "bias": _t([-0.05, 0.2])}, a2, z3),
            Layer(7, LayerKind.RELU.value, {}, z3, a3),
            Layer(8, LayerKind.DENSE.value, {"in_features": 2, "out_features": 1, "weight": _t([[0.7, -0.2]]), "bias": _t([0.0])}, a3, out),
            Layer(9, LayerKind.ASSERT.value, {"kind": "LINEAR_LE", "c": _t([1.0]), "d": _t([0.8])}, out, out),
        ],
        preds={0: [], 1: [0], 2: [1], 3: [2], 4: [3], 5: [4], 6: [5], 7: [6], 8: [7], 9: [8]},
        succs={0: [1], 1: [2], 2: [3], 3: [4], 4: [5], 5: [6], 6: [7], 7: [8], 8: [9], 9: []},
    )


def _forward_preacts(net: Net, x: torch.Tensor) -> dict[int, torch.Tensor]:
    y = x
    preacts: dict[int, torch.Tensor] = {}
    for layer in net.layers:
        kind = layer.kind.upper()
        if kind in {LayerKind.INPUT.value, LayerKind.INPUT_SPEC.value, LayerKind.ASSERT.value}:
            continue
        if kind == LayerKind.DENSE.value:
            weight = layer.get_tensor("weight")
            bias = layer.get_tensor("bias")
            assert weight is not None and bias is not None
            y = y @ weight.T + bias
            preacts[layer.id] = y
        elif kind == LayerKind.RELU.value:
            y = torch.relu(y)
        else:
            raise AssertionError(kind)
    return preacts


def test_project_alpha_for_forward_picks_next_downstream_sid() -> None:
    net = _make_four_layer_relu_net()
    state = AlphaState()
    relu_ids = [3, 5, 7]
    sid_values = [2, 4, 6]
    for relu_lid in relu_ids:
        for sid in sid_values:
            state.set(relu_lid, sid, torch.full((1, 2), relu_lid * 100 + sid, dtype=get_default_dtype(), device=get_default_device()))

    projected = bab_module.project_alpha_for_forward(net, state)
    assert set(projected) == {3, 5}
    assert torch.equal(projected[3], cast(torch.Tensor, state.get(3, 4)))
    assert torch.equal(projected[5], cast(torch.Tensor, state.get(5, 6)))
    assert 7 not in projected


def test_bab_iter1_passes_alpha_to_compute_forward_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    net = _make_four_layer_relu_net()
    solver = DualSolver(DualTF())
    seen: list[dict[int, torch.Tensor] | None] = []
    original = bab_module.compute_forward_bounds

    def wrapped_compute_forward_bounds(*args: Any, **kwargs: Any):
        alphas = kwargs.get("alphas")
        seen.append(cast(dict[int, torch.Tensor] | None, alphas))
        return original(*cast(tuple[Any, ...], args), **cast(dict[str, Any], kwargs))

    monkeypatch.setattr(bab_module, "compute_forward_bounds", wrapped_compute_forward_bounds)
    monkeypatch.setattr(bab_module, "_extract_ces", lambda *args, **kwargs: None)

    res = bab_module.verify_bab(
        net,
        TorchLPSolver(),
        config=BaBConfig(
            max_depth=2,
            max_nodes=3,
            branching_method="random",
            bounding_method="bfs",
            subproblem_batch_size=1,
            alpha_split_objective=True,
            alpha_iters=1,
            lr_alpha=0.5,
        ),
        dual_solver=solver,
    )

    assert res.status == VerifyStatus.UNKNOWN
    assert len(seen) >= 2
    assert seen[0] is None
    assert seen[1] is not None
    assert set(cast(dict[int, torch.Tensor], seen[1]).keys()) == {3, 5}


def test_bab_alpha_split_off_no_alphas_passed(monkeypatch: pytest.MonkeyPatch) -> None:
    net = _make_four_layer_relu_net()
    solver = DualSolver(DualTF())
    seen: list[dict[int, torch.Tensor] | None] = []
    original = bab_module.compute_forward_bounds

    def wrapped_compute_forward_bounds(*args: Any, **kwargs: Any):
        seen.append(cast(dict[int, torch.Tensor] | None, kwargs.get("alphas")))
        return original(*cast(tuple[Any, ...], args), **cast(dict[str, Any], kwargs))

    monkeypatch.setattr(bab_module, "compute_forward_bounds", wrapped_compute_forward_bounds)
    monkeypatch.setattr(bab_module, "_extract_ces", lambda *args, **kwargs: None)

    res = bab_module.verify_bab(
        net,
        TorchLPSolver(),
        config=BaBConfig(
            max_depth=2,
            max_nodes=3,
            branching_method="random",
            bounding_method="bfs",
            subproblem_batch_size=1,
            alpha_split_objective=False,
        ),
        dual_solver=solver,
    )

    assert res.status == VerifyStatus.UNKNOWN
    assert seen
    assert all(alpha is None for alpha in seen)


def test_alpha_intermediate_in_joint_adam_param_list(monkeypatch: pytest.MonkeyPatch) -> None:
    net = _make_four_layer_relu_net()
    bounds = compute_forward_bounds(net, _t([[-1.0, -1.0]]), _t([[1.0, 1.0]]), post_activation=False)
    warm = AlphaState.from_legacy({3: torch.full((1, 2), 0.25, dtype=get_default_dtype(), device=get_default_device())})
    intermediate = torch.full((1, 2), 0.6, dtype=get_default_dtype(), device=get_default_device())
    warm.set(3, 4, intermediate)
    captured: list[list[torch.nn.Parameter]] = []
    captured_clones: list[list[torch.Tensor]] = []
    original_adam = torch.optim.Adam

    def capture_adam(params: Any, *args: Any, **kwargs: Any):
        param_list = list(cast(list[torch.nn.Parameter], params))
        captured.append(param_list)
        captured_clones.append([p.detach().clone() for p in param_list])
        return original_adam(param_list, *args, **kwargs)

    monkeypatch.setattr(torch.optim, "Adam", capture_adam)
    solver = DualSolver(DualTF())
    solver.set_alphas(warm)
    _ = solver.compute_bound(net, bounds, _t([[1.0]]), n_iters=1, lr=0.1, force_kkt=True)

    assert captured
    assert any(param is intermediate for param in captured[0]) is False
    assert any(torch.equal(clone, intermediate) for clone in captured_clones[0])


def test_alpha_final_unchanged_when_only_intermediate_added() -> None:
    net = _make_four_layer_relu_net()
    bounds = compute_forward_bounds(net, _t([[-1.0, -1.0]]), _t([[1.0, 1.0]]), post_activation=False)
    solver = DualSolver(DualTF())
    warm_final = {3: torch.full((1, 2), 0.25, dtype=get_default_dtype(), device=get_default_device())}
    baseline = cast(tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, AlphaState | None, object], solver.compute_bound(net, bounds, _t([[1.0]]), n_iters=1, lr=0.1, force_kkt=True, warm_alphas=warm_final))
    warm = AlphaState()
    assert baseline[3] is not None
    for lid, tensor in baseline[3].for_start_node(AlphaState.FINAL_SID).items():
        warm.set(lid, AlphaState.FINAL_SID, tensor.detach().clone())
    warm.set(3, 4, torch.full((1, 2), 0.4, dtype=get_default_dtype(), device=get_default_device()))

    with_intermediate = cast(tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, AlphaState | None, object], solver.compute_bound(net, bounds, _t([[1.0]]), n_iters=1, lr=0.1, force_kkt=True, warm_alphas=warm))
    assert baseline[3] is not None and with_intermediate[3] is not None
    for lid, tensor in baseline[3].for_start_node(AlphaState.FINAL_SID).items():
        torch.testing.assert_close(with_intermediate[3].for_start_node(AlphaState.FINAL_SID)[lid], tensor)


def test_soundness_preserved_with_alpha_wired() -> None:
    net = _make_four_layer_relu_net()
    lb = _t([[-1.0, -1.0]])
    ub = _t([[1.0, 1.0]])
    bounds = compute_forward_bounds(net, lb, ub, post_activation=False)
    alpha_state = AlphaState()
    alpha_state.set(3, 4, torch.full((1, 2), 0.2, dtype=get_default_dtype(), device=get_default_device()))
    alpha_state.set(5, 6, torch.full((1, 2), 0.7, dtype=get_default_dtype(), device=get_default_device()))
    projected = bab_module.project_alpha_for_forward(net, alpha_state)
    wired_bounds = compute_forward_bounds(net, lb, ub, post_activation=False, alphas=projected)
    samples = _sample_box(lb[0], ub[0], count=100, seed=13)
    preacts = _forward_preacts(net, samples)
    for lid in [2, 4, 6]:
        actual = preacts[lid]
        assert torch.all(wired_bounds[lid].lb[0].unsqueeze(0) <= actual + 1e-7)
        assert torch.all(wired_bounds[lid].ub[0].unsqueeze(0) >= actual - 1e-7)


def test_intermediate_grad_nonzero_at_bab_depth_1() -> None:
    net = _make_four_layer_relu_net()
    bounds = compute_forward_bounds(net, _t([[-1.0, -1.0]]), _t([[1.0, 1.0]]), post_activation=False)
    solver = DualSolver(DualTF())
    solver.lambda_intermediate = 1.0
    warm = AlphaState.from_legacy({3: torch.full((1, 2), 0.25, dtype=get_default_dtype(), device=get_default_device())})
    warm.set(3, 4, torch.full((1, 2), 0.6, dtype=get_default_dtype(), device=get_default_device()))
    captured: dict[str, list[torch.nn.Parameter]] = {}
    original_adam = torch.optim.Adam

    class _NoStepAdam:
        def __init__(self, params: Any, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            captured["params"] = list(cast(list[torch.nn.Parameter], params))

        def zero_grad(self) -> None:
            for param in captured["params"]:
                param.grad = None

        def step(self) -> None:
            return None

    torch.optim.Adam = _NoStepAdam  # type: ignore[assignment]
    try:
        _ = solver.compute_bound(net, bounds, _t([[1.0]]), n_iters=1, lr=0.1, force_kkt=True, warm_alphas=warm)
    finally:
        torch.optim.Adam = original_adam  # type: ignore[assignment]

    params = captured["params"]
    assert any(param.grad is not None and param.grad.abs().sum().item() > 0 for param in params)


def test_lambda_zero_reproduces_fix1() -> None:
    net = _make_four_layer_relu_net()
    bounds = compute_forward_bounds(net, _t([[-1.0, -1.0]]), _t([[1.0, 1.0]]), post_activation=False)
    warm = AlphaState.from_legacy({3: torch.full((1, 2), 0.25, dtype=get_default_dtype(), device=get_default_device())})
    warm.set(3, 4, torch.full((1, 2), 0.4, dtype=get_default_dtype(), device=get_default_device()))

    solver_a = DualSolver(DualTF())
    solver_a.lambda_intermediate = 0.0
    out_a = cast(
        tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, AlphaState | None, object],
        solver_a.compute_bound(net, bounds, _t([[1.0]]), n_iters=1, lr=0.1, force_kkt=True, warm_alphas=warm),
    )

    solver_b = DualSolver(DualTF())
    solver_b.lambda_intermediate = 0.0
    out_b = cast(
        tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, AlphaState | None, object],
        solver_b.compute_bound(net, bounds, _t([[1.0]]), n_iters=1, lr=0.1, force_kkt=True, warm_alphas=warm),
    )

    torch.testing.assert_close(out_a[0], out_b[0])
    assert out_a[3] is not None and out_b[3] is not None
    for lid, tensor in out_a[3].for_start_node(AlphaState.FINAL_SID).items():
        torch.testing.assert_close(out_b[3].for_start_node(AlphaState.FINAL_SID)[lid], tensor)


def test_soundness_with_intermediate_loss() -> None:
    net = _make_four_layer_relu_net()
    lb = _t([[-1.0, -1.0]])
    ub = _t([[1.0, 1.0]])
    bounds = compute_forward_bounds(net, lb, ub, post_activation=False)
    solver = DualSolver(DualTF())
    solver.lambda_intermediate = 1.0
    spec = OutputSpec(kind=OutKind.LINEAR_LE, c=_t([1.0]), d=_t([0.0]))
    result = solver.evaluate_spec(net, bounds, spec, enable_grad=True)
    assert torch.isfinite(result.margins).all()
    if result.out_alphas is not None:
        lb_int = backward_truncated_lb(net, bounds, 4, result.out_alphas)
        assert torch.isfinite(lb_int).all()
