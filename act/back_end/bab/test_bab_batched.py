from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from act.back_end.bab import verify_bab
from act.back_end.bab.bab import (
    _compute_pre_act_widths,
    check_violation_at_point_batched,
)
from act.back_end.bab.branching.bounding import BFSBounding
from act.back_end.bab.eta import get_pre_activation_layer_id
from act.back_end.config import BaBConfig
from act.back_end.core import Bounds, Layer, Net
from act.back_end.layer_schema import LayerKind
from act.back_end.solver.spec_batching import build_spec_batch
from act.back_end.solver.spec_batching import SpecBatchResult
from act.back_end.solver.solver_dual import DualSolver
from act.back_end.solver.solver_interval import TorchLPSolver
from act.back_end.dual_tf import DualTF
from act.front_end.spec_creator_base import LabeledInputTensor
from act.front_end.specs import InputSpec, OutKind, OutputSpec
from act.front_end.verifiable_model import (
    InputLayer,
    InputSpecLayer,
    OutputSpecLayer,
    VerifiableModel,
)
from act.pipeline.verification.torch2act import TorchToACT
from act.util.device_manager import get_default_device, get_default_dtype, initialize_device
from act.util.stats import VerifyStatus


def _t(data: object) -> torch.Tensor:
    return torch.tensor(data, dtype=get_default_dtype(), device=get_default_device())


def _make_relu_net(hidden_bias: float, out_weight: float, out_bias: float, threshold: float) -> Net:
    in_vars = [0]
    z = [10]
    a = [20]
    out = [30]
    layers = [
        Layer(0, LayerKind.INPUT.value, {"shape": [1], "dtype": "float32", "num_classes": 1, "value_range": [0.0, 1.0]}, in_vars, in_vars),
        Layer(1, LayerKind.INPUT_SPEC.value, {"kind": "BOX", "lb": _t([-1.0]), "ub": _t([1.0])}, in_vars, in_vars),
        Layer(2, LayerKind.DENSE.value, {"in_features": 1, "out_features": 1, "weight": _t([[1.0]]), "bias": _t([hidden_bias])}, in_vars, z),
        Layer(3, LayerKind.RELU.value, {}, z, a),
        Layer(4, LayerKind.DENSE.value, {"in_features": 1, "out_features": 1, "weight": _t([[out_weight]]), "bias": _t([out_bias])}, a, out),
        Layer(5, LayerKind.ASSERT.value, {"kind": OutKind.LINEAR_LE, "c": _t([1.0]), "d": _t([threshold])}, out, out),
    ]
    return Net(
        layers=layers,
        preds={0: [], 1: [0], 2: [1], 3: [2], 4: [3], 5: [4]},
        succs={0: [1], 1: [2], 2: [3], 3: [4], 4: [5], 5: []},
    )


def _make_tanh_net() -> Net:
    in_vars = [0]
    z = [10]
    a = [20]
    out = [30]
    layers = [
        Layer(0, LayerKind.INPUT.value, {"shape": [1], "dtype": "float32", "num_classes": 1, "value_range": [0.0, 1.0]}, in_vars, in_vars),
        Layer(1, LayerKind.INPUT_SPEC.value, {"kind": "BOX", "lb": _t([-1.0]), "ub": _t([1.0])}, in_vars, in_vars),
        Layer(2, LayerKind.DENSE.value, {"in_features": 1, "out_features": 1, "weight": _t([[1.0]]), "bias": _t([0.0])}, in_vars, z),
        Layer(3, LayerKind.TANH.value, {}, z, a),
        Layer(4, LayerKind.DENSE.value, {"in_features": 1, "out_features": 1, "weight": _t([[0.5]]), "bias": _t([0.0])}, a, out),
        Layer(5, LayerKind.ASSERT.value, {"kind": OutKind.LINEAR_LE, "c": _t([1.0]), "d": _t([0.2])}, out, out),
    ]
    return Net(
        layers=layers,
        preds={0: [], 1: [0], 2: [1], 3: [2], 4: [3], 5: [4]},
        succs={0: [1], 1: [2], 2: [3], 3: [4], 4: [5], 5: []},
    )


def _batched_cfg(**kwargs) -> BaBConfig:
    return BaBConfig(
        max_depth=kwargs.pop("max_depth", 4),
        max_nodes=kwargs.pop("max_nodes", 20),
        branching_method=kwargs.pop("branching_method", "babsr"),
        bounding_method=kwargs.pop("bounding_method", "bfs"),
        subproblem_batch_size=kwargs.pop("subproblem_batch_size", 1),
        **kwargs,
    )


def setup_module() -> None:
    initialize_device("cpu", "float32")


def test_root_certified() -> None:
    net = _make_relu_net(hidden_bias=2.0, out_weight=1.0, out_bias=0.0, threshold=5.0)
    res = verify_bab(
        net,
        TorchLPSolver(),
        config=_batched_cfg(subproblem_batch_size=1, max_nodes=1, max_depth=0),
        dual_solver=DualSolver(DualTF()),
    )
    assert res.status == VerifyStatus.CERTIFIED
    assert res.metadata["nodes"] == 1


def test_root_falsified() -> None:
    net = _make_relu_net(hidden_bias=0.0, out_weight=1.0, out_bias=0.0, threshold=0.1)
    res = verify_bab(
        net,
        TorchLPSolver(),
        config=_batched_cfg(subproblem_batch_size=1, max_nodes=2, max_depth=0),
        dual_solver=DualSolver(DualTF()),
    )
    assert res.status == VerifyStatus.FALSIFIED
    assert res.counterexample is not None
    mask = check_violation_at_point_batched(net, res.counterexample.unsqueeze(0), net.layers[-1])
    assert bool(mask[0].item())


def test_one_split_certified() -> None:
    net = _make_relu_net(hidden_bias=0.0, out_weight=0.2, out_bias=0.0, threshold=0.2)
    res = verify_bab(
        net,
        TorchLPSolver(),
        config=_batched_cfg(branching_method="babsr", bounding_method="bfs", max_nodes=8, max_depth=2),
        dual_solver=DualSolver(DualTF()),
    )
    assert res.status == VerifyStatus.CERTIFIED
    assert res.metadata["nodes"] >= 1


def test_one_split_falsified() -> None:
    net = _make_relu_net(hidden_bias=0.0, out_weight=1.0, out_bias=0.0, threshold=0.75)
    res = verify_bab(
        net,
        TorchLPSolver(),
        config=_batched_cfg(branching_method="babsr", bounding_method="bfs", max_nodes=8, max_depth=2),
        dual_solver=DualSolver(DualTF()),
    )
    assert res.status == VerifyStatus.FALSIFIED
    assert res.counterexample is not None


def test_bfs_processes_level_by_level() -> None:
    batch_root = Bounds(lb=_t([-1.0]), ub=_t([1.0]))
    from act.back_end.bab.node import SubproblemBatch

    root = SubproblemBatch.from_bounds_with_eta(batch_root, {2: 1})
    sibling = SubproblemBatch.from_bounds_with_eta(Bounds(lb=_t([2.0]), ub=_t([3.0])), {2: 1})
    pool = BFSBounding()
    pool.push(root)
    pool.push(sibling)
    parent = pool.pop(1)
    left, right = parent.split_neuron_batched([(2, 0, "relu")])
    pool.push(left)
    pool.push(right)
    first_remaining = pool.pop(1)
    second_remaining = pool.pop(1)
    third_remaining = pool.pop(1)
    assert first_remaining.lb[0, 0].item() == 2.0
    assert second_remaining.depths[0].item() == 1
    assert third_remaining.depths[0].item() == 1


def test_smooth_network_reaches_termination() -> None:
    net = _make_tanh_net()
    net.layers[-1].params["d"] = _t([0.4])
    res = verify_bab(
        net,
        TorchLPSolver(),
        config=_batched_cfg(branching_method="babsr", bounding_method="bfs", max_nodes=50, max_depth=5),
        dual_solver=DualSolver(DualTF()),
    )
    assert res.status in {VerifyStatus.CERTIFIED, VerifyStatus.UNKNOWN}


def test_cnn_multiclass_batched_verify() -> None:
    torch.manual_seed(0)
    device = get_default_device()
    dtype = get_default_dtype()

    model = nn.Sequential(
        nn.Conv2d(1, 4, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(4 * 4 * 4, 5),
        nn.ReLU(),
        nn.Linear(5, 3),
    ).to(device=device, dtype=dtype)
    model.eval()

    x0 = torch.rand((1, 1, 4, 4), device=device, dtype=dtype)
    eps = torch.tensor(0.05, device=device, dtype=dtype)
    lb = (x0 - eps).clamp(0.0, 1.0)
    ub = (x0 + eps).clamp(0.0, 1.0)
    y_true = model(x0).argmax(dim=1).to(dtype=torch.int64)
    margin = torch.tensor([0.0], device=device, dtype=dtype)

    wrapped = VerifiableModel(
        input_layer=InputLayer(
            labeled_input=LabeledInputTensor(tensor=x0, label=y_true),
            shape=tuple(x0.shape),
            dtype=dtype,
            layout="CHW",
            dataset_name="unit_test",
        ),
        input_spec=InputSpecLayer(spec=InputSpec(kind="BOX", lb=lb, ub=ub)),
        model=model,
        output_spec=OutputSpecLayer(
            spec=OutputSpec(
                kind=OutKind.MARGIN_ROBUST,
                y_true=y_true,
                margin=margin,
            )
        ),
    )
    net = TorchToACT(wrapped).run()

    relu_ids = [layer.id for layer in net.layers if layer.kind == LayerKind.RELU.value]
    widths = _compute_pre_act_widths(net)
    assert len(relu_ids) == 2
    assert widths[get_pre_activation_layer_id(net, relu_ids[0])] == 4 * 4 * 4
    assert widths[get_pre_activation_layer_id(net, relu_ids[1])] == 5

    spec_batch = build_spec_batch(
        OutputSpec(kind=OutKind.MARGIN_ROBUST, y_true=y_true[:1], margin=margin[:1]),
        B=4,
        n_out=3,
        num_classes=3,
        device=device,
        dtype=dtype,
    )
    assert spec_batch.C.shape == (12, 3)
    assert spec_batch.thresholds.shape == (4, 3)
    assert spec_batch.active_mask.shape == (4, 3)
    assert torch.allclose(spec_batch.thresholds, torch.zeros((4, 3), device=device, dtype=dtype))
    assert torch.equal(spec_batch.active_mask.sum(dim=1), torch.full((4,), 2, device=device, dtype=torch.long))

    res = verify_bab(
        net,
        TorchLPSolver(),
        config=_batched_cfg(
            branching_method="babsr",
            bounding_method="bfs",
            subproblem_batch_size=4,
            max_nodes=50,
            max_depth=6,
        ),
        dual_solver=DualSolver(DualTF()),
    )
    assert res.status in {
        VerifyStatus.CERTIFIED,
        VerifyStatus.FALSIFIED,
        VerifyStatus.UNKNOWN,
    }


def test_parent_margin_monotonicity_safety(monkeypatch: pytest.MonkeyPatch) -> None:
    """If a child bound regresses below parent, verify_bab clamps to parent bound."""
    net = _make_relu_net(hidden_bias=0.0, out_weight=0.2, out_bias=0.0, threshold=0.2)
    solver = DualSolver(DualTF())
    captured_parent_margins: list[torch.Tensor] = []
    eval_calls = {"count": 0}

    def fake_evaluate_spec(
        self,
        net,
        bounds_dict,
        out_spec,
        num_classes=None,
        chunk_size=None,
        enable_grad=False,
    ) -> SpecBatchResult:
        del self, net, out_spec, num_classes, chunk_size, enable_grad
        batch_size = next(iter(bounds_dict.values())).lb.shape[0]
        value = -0.1 if eval_calls["count"] == 0 else -0.3
        eval_calls["count"] += 1
        slack = torch.full(
            (batch_size, 1),
            value,
            device=get_default_device(),
            dtype=get_default_dtype(),
        )
        return SpecBatchResult(
            margins=slack,
            slack=slack,
            active_mask=torch.ones_like(slack, dtype=torch.bool),
            certified=torch.zeros(batch_size, dtype=torch.bool, device=slack.device),
        )

    original_push = BFSBounding.push

    def capture_push(self, batch):
        if batch.parent_margins is not None and int(batch.depths.min().item()) == 1:
            captured_parent_margins.append(batch.parent_margins.detach().cpu().clone())
        return original_push(self, batch)

    monkeypatch.setattr(solver, "evaluate_spec", fake_evaluate_spec.__get__(solver, DualSolver))
    monkeypatch.setattr("act.back_end.bab.bab._extract_ces", lambda *args, **kwargs: None)
    monkeypatch.setattr(BFSBounding, "push", capture_push)

    res = verify_bab(
        net,
        TorchLPSolver(),
        config=_batched_cfg(
            branching_method="random",
            bounding_method="bfs",
            subproblem_batch_size=1,
            max_nodes=2,
            max_depth=2,
        ),
        dual_solver=solver,
    )

    assert res.status == VerifyStatus.UNKNOWN
    assert captured_parent_margins
    for parent_margin in captured_parent_margins:
        assert parent_margin.tolist() == pytest.approx([-0.1])
