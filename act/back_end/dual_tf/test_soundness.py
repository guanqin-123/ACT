# pyright: reportMissingImports=false, reportDeprecated=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportImplicitOverride=false, reportUnannotatedClassAttribute=false, reportAny=false, reportUntypedFunctionDecorator=false, reportAttributeAccessIssue=false, reportArgumentType=false, reportIndexIssue=false, reportCallIssue=false, reportOptionalMemberAccess=false, reportOptionalSubscript=false, reportUnknownArgumentType=false, reportUnusedImport=false, reportPrivateUsage=false
# ===- act/back_end/dual_tf/test_soundness.py - Dual bound soundness -----====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------===#
#
# Purpose:
#   Mathematical soundness tests for DualTF bound propagation.
#
#   A certified lower bound L on the objective must satisfy:
#                   L  <=  min_{x in [lb, ub]} c^T @ f(x)
#
#   These tests sample concrete inputs, run the network, and assert that
#   the dual lower bound never exceeds any concrete objective value.
#
#   The historical bugs fixed in this module (unsound crossing bias,
#   wrong η sign, wrong η position) would all be caught by these tests.
#
#   Ported from origin/batchBaB with API adaptations:
#     * DualSolver(DualTF()).compute_bound(net, bounds, c[B, K])
#     * DualSolver(DualTF()).compute_robust_bound(...) -> (min_slack, certified)
#     * compute_forward_bounds(net, lb, ub, post_activation=...)
#     * MLP helpers emit INPUT -> INPUT_SPEC -> DENSE/RELU* -> ASSERT
#
#   Phase 1 (eta) has landed: EtaState, expand_eta_state, set_eta/clear_eta,
#   ν-modification hook, Adam loop with eta_iters, forward bound clamping,
#   evaluate_spec wrapping. Plus the dual_relu_backward Wong-Kolter hotfix
#   for ν<0 on ambiguous ReLU neurons. End-to-end soundness tests and the
#   negative-ν unit tests are now ACTIVE.
#
#   Tests that depend on learnable alpha slopes / KKT alpha+eta joint
#   optimization / warm-start alpha+eta return/accept / the non-existent
#   compute_linear_bound method and private DualTF internals remain skipped
#   as Phase 2 follow-up.
#
# ===---------------------------------------------------------------------===#

from __future__ import annotations

from typing import Dict, Tuple, cast

import pytest
import torch

from act.back_end.core import Bounds, Layer, Net
from act.back_end.dual_tf import DualTF, compute_forward_bounds, dual_relu_backward
from act.back_end.layer_schema import LayerKind
from act.back_end.solver.solver_dual import DualSolver
from act.util.device_manager import initialize_device


@pytest.fixture(scope="module", autouse=True)
def _cpu_float32() -> None:
    """Run these tests on CPU/float32 for determinism."""
    initialize_device("cpu", "float32")


def _make_mlp_net(
    layer_dims: list[int], weights: list[torch.Tensor], biases: list[torch.Tensor]
) -> Net:
    """Build INPUT -> INPUT_SPEC -> DENSE -> RELU -> ... -> DENSE -> ASSERT.

    Wraps the MLP topology with the INPUT / INPUT_SPEC preamble and trailing
    ASSERT required by the current DualSolver.compute_bound API. Variable-id
    bookkeeping threads a fresh block per layer so validate_layer passes.
    """
    in_dim = layer_dims[0]
    in_vars = list(range(in_dim))
    layers: list[Layer] = [
        Layer(
            0,
            LayerKind.INPUT.value,
            {
                "shape": [in_dim],
                "dtype": "float32",
                "num_classes": 1,
                "value_range": [0.0, 1.0],
            },
            in_vars,
            in_vars,
        ),
        Layer(
            1,
            LayerKind.INPUT_SPEC.value,
            {"kind": "BOX"},
            in_vars,
            in_vars,
        ),
    ]
    next_var = in_dim
    prev_out_vars = in_vars

    for i, (W, b) in enumerate(zip(weights, biases)):
        dense_out = list(range(next_var, next_var + W.shape[0]))
        next_var += W.shape[0]
        layers.append(
            Layer(
                len(layers),
                LayerKind.DENSE.value,
                {
                    "in_features": W.shape[1],
                    "out_features": W.shape[0],
                    "weight": W,
                    "bias": b,
                },
                prev_out_vars,
                dense_out,
            )
        )
        prev_out_vars = dense_out

        if i < len(weights) - 1:
            relu_out = list(range(next_var, next_var + W.shape[0]))
            next_var += W.shape[0]
            layers.append(
                Layer(
                    len(layers),
                    LayerKind.RELU.value,
                    {},
                    prev_out_vars,
                    relu_out,
                )
            )
            prev_out_vars = relu_out

    layers.append(
        Layer(
            len(layers),
            LayerKind.ASSERT.value,
            {"kind": "RANGE"},
            prev_out_vars,
            prev_out_vars,
        )
    )

    preds = {i: [i - 1] if i > 0 else [] for i in range(len(layers))}
    succs = {i: [i + 1] if i < len(layers) - 1 else [] for i in range(len(layers))}
    return Net(layers=layers, preds=preds, succs=succs)


def _concrete_forward(net: Net, x: torch.Tensor) -> torch.Tensor:
    """Concrete forward pass through an MLP Net. Input x is (B, D)."""
    y = x
    for layer in net.layers:
        k = layer.kind.upper()
        if k in (LayerKind.INPUT.value, LayerKind.INPUT_SPEC.value, LayerKind.ASSERT.value):
            continue
        if k == LayerKind.DENSE.value:
            W = layer.params["weight"]
            b = layer.params["bias"]
            y = y @ W.T + b
        elif k == LayerKind.RELU.value:
            y = torch.relu(y)
        else:
            raise AssertionError(f"Unsupported layer in test helper: {k}")
    return y


def _sample_objective_min(
    net: Net, c: torch.Tensor, lb: torch.Tensor, ub: torch.Tensor, n: int = 4096
) -> float:
    """Monte-Carlo estimate of min_{x in [lb, ub]} c^T @ f(x)."""
    torch.manual_seed(0)
    x = lb + (ub - lb) * torch.rand(n, lb.shape[-1])
    corners = torch.stack([lb.expand(8, -1), ub.expand(8, -1)]).view(-1, lb.shape[-1])
    x = torch.cat([x, corners], dim=0)
    y = _concrete_forward(net, x)
    return (y @ c).min().item()


def _sample_concrete_outputs(
    net: Net, lb: torch.Tensor, ub: torch.Tensor, n: int = 4096
) -> torch.Tensor:
    torch.manual_seed(0)
    x = lb + (ub - lb) * torch.rand(n, lb.shape[-1])
    corners = torch.stack([lb.expand(8, -1), ub.expand(8, -1)]).view(-1, lb.shape[-1])
    x = torch.cat([x, corners], dim=0)
    return _concrete_forward(net, x)


def _relu_layer_ids(net: Net) -> list[int]:
    return [
        layer.id for layer in net.layers if layer.kind.upper() == LayerKind.RELU.value
    ]


def _constant_alphas(
    net: Net, bounds: Dict[int, Bounds], value: float
) -> Dict[int, torch.Tensor]:
    alphas: Dict[int, torch.Tensor] = {}
    for lid in _relu_layer_ids(net):
        alphas[lid] = torch.full_like(bounds[lid].lb, value)
    return alphas


# ---------------------------------------------------------------------------
# ReLU backward unit tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "v_val",
    [
        -2.0,
        -1.0,
        -0.3,
        0.3,
        1.0,
        2.0,
    ],
)
def test_relu_backward_single_neuron_soundness(v_val: float) -> None:
    """On a single ReLU(x), x in [-1, 1], the dual bound must <= true min."""
    bounds = Bounds(torch.tensor([[-1.0]]), torch.tensor([[1.0]]))
    nu = torch.tensor([[v_val]])
    v_out, contrib = dual_relu_backward(nu, bounds)

    l, u = -1.0, 1.0
    vo = v_out.item()
    input_contrib = l * max(vo, 0.0) + u * min(vo, 0.0)
    dual_bound = contrib.item() + input_contrib

    true_min = v_val * 0.0 if v_val > 0 else v_val * u
    assert dual_bound <= true_min + 1e-6, (
        f"UNSOUND: v={v_val} bound={dual_bound:.4f} > true={true_min:.4f}"
    )


@pytest.mark.parametrize("alpha_val", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_relu_backward_alpha_range_soundness(alpha_val: float) -> None:
    """Any α ∈ [0, 1] must yield a sound lower bound."""
    bounds = Bounds(torch.tensor([[-2.0]]), torch.tensor([[3.0]]))
    alpha = torch.tensor([[alpha_val]])
    for v_val in [-2.0, -0.5, 0.5, 2.0]:
        nu = torch.tensor([[v_val]])
        v_out, contrib = dual_relu_backward(nu, bounds, alpha=alpha)
        vo = v_out.item()
        input_contrib = -2.0 * max(vo, 0.0) + 3.0 * min(vo, 0.0)
        dual_bound = contrib.item() + input_contrib
        true_min = v_val * 0.0 if v_val > 0 else v_val * 3.0
        assert dual_bound <= true_min + 1e-6, (
            f"UNSOUND: v={v_val} alpha={alpha_val} "
            f"bound={dual_bound:.4f} > true={true_min:.4f}"
        )


def test_relu_backward_stable_on_neuron() -> None:
    """For an always-ON neuron (l >= 0), bound = v * min_x x."""
    bounds = Bounds(torch.tensor([[1.0]]), torch.tensor([[3.0]]))
    for v_val in [-1.5, -0.5, 0.5, 1.5]:
        nu = torch.tensor([[v_val]])
        v_out, contrib = dual_relu_backward(nu, bounds)
        assert torch.allclose(v_out, nu), "ON neuron slope must be 1"
        assert contrib.item() == pytest.approx(0.0), "ON neuron has no bias"


def test_relu_backward_stable_off_neuron() -> None:
    """For an always-OFF neuron (u <= 0), bound = 0."""
    bounds = Bounds(torch.tensor([[-3.0]]), torch.tensor([[-1.0]]))
    for v_val in [-1.5, -0.5, 0.5, 1.5]:
        nu = torch.tensor([[v_val]])
        v_out, contrib = dual_relu_backward(nu, bounds)
        assert torch.allclose(v_out, torch.zeros_like(nu)), "OFF neuron slope must be 0"
        assert contrib.item() == pytest.approx(0.0), "OFF neuron has no bias"


def test_relu_backward_alpha_zero_equals_off_slope() -> None:
    """alpha=0 on AMB with nu>0 must yield v_out=0 (inactive lower relaxation)."""
    from act.back_end.dual_tf.tf_mlp import dual_relu_backward
    from act.back_end.core import Bounds

    lb = torch.tensor([[-1.0]])
    ub = torch.tensor([[1.0]])
    nu = torch.tensor([[0.5]])
    alpha = torch.tensor([[0.0]])
    v_out, _ = dual_relu_backward(nu, Bounds(lb=lb, ub=ub), alpha=alpha)
    assert torch.allclose(v_out, torch.tensor([[0.0]]))


# ---------------------------------------------------------------------------
# End-to-end soundness on small MLPs
# ---------------------------------------------------------------------------


def _random_mlp(layer_sizes: list[int], seed: int) -> Net:
    g = torch.Generator().manual_seed(seed)
    weights, biases = [], []
    for i in range(len(layer_sizes) - 1):
        W = torch.randn(layer_sizes[i + 1], layer_sizes[i], generator=g) * 0.5
        b = torch.randn(layer_sizes[i + 1], generator=g) * 0.1
        weights.append(W)
        biases.append(b)
    return _make_mlp_net(layer_sizes, weights, biases)


def _stable_relu_test_net() -> Net:
    weights = [
        torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        ),
        torch.tensor(
            [
                [1.0, 2.0],
                [2.0, 1.0],
            ]
        ),
    ]
    biases = [torch.tensor([1.0, 1.0]), torch.tensor([0.0, 0.0])]
    return _make_mlp_net([2, 2, 2], weights, biases)


@pytest.mark.parametrize("seed", list(range(5)))
def test_compute_bound_soundness_2layer_mlp(seed: int) -> None:
    """2-layer MLP: dual bound must be <= true minimum (MC-estimated)."""
    net = _random_mlp([3, 6, 2], seed)

    lb = torch.tensor([[-1.0, -1.0, -1.0]])
    ub = torch.tensor([[1.0, 1.0, 1.0]])

    bounds = compute_forward_bounds(net, lb, ub, post_activation=False)

    solver = DualSolver(DualTF())
    for j in range(2):
        c = torch.zeros(2)
        c[j] = 1.0
        dual_bound = solver.compute_bound(net, bounds, c.unsqueeze(0))
        true_min = _sample_objective_min(net, c, lb[0], ub[0])
        assert dual_bound.item() <= true_min + 1e-3, (
            f"UNSOUND 2-layer seed={seed} j={j}: "
            f"bound={dual_bound.item():.4f} > true_min~={true_min:.4f}"
        )


@pytest.mark.parametrize("seed", list(range(5)))
def test_compute_bound_soundness_3layer_mlp(seed: int) -> None:
    """3-layer MLP (deeper → more relaxation): bound still sound."""
    net = _random_mlp([3, 5, 5, 2], seed)
    lb = torch.tensor([[-1.0, -1.0, -1.0]])
    ub = torch.tensor([[1.0, 1.0, 1.0]])
    bounds = compute_forward_bounds(net, lb, ub, post_activation=False)

    solver = DualSolver(DualTF())
    for j in range(2):
        c = torch.zeros(2)
        c[j] = 1.0
        dual_bound = solver.compute_bound(net, bounds, c.unsqueeze(0))
        true_min = _sample_objective_min(net, c, lb[0], ub[0])
        assert dual_bound.item() <= true_min + 1e-3, (
            f"UNSOUND 3-layer seed={seed} j={j}: "
            f"bound={dual_bound.item():.4f} > true_min~={true_min:.4f}"
        )


@pytest.mark.parametrize("seed", list(range(3)))
def test_compute_bound_soundness_mixed_sign_c(seed: int) -> None:
    """Mixed-sign c (y_true − y_other): covers both v>0 and v<0 branches."""
    net = _random_mlp([4, 6, 3], seed)
    lb = torch.full((1, 4), -0.5)
    ub = torch.full((1, 4), 0.5)
    bounds = compute_forward_bounds(net, lb, ub, post_activation=False)

    solver = DualSolver(DualTF())
    for y_true in range(3):
        for j in range(3):
            if j == y_true:
                continue
            c = torch.zeros(3)
            c[y_true] = 1.0
            c[j] = -1.0
            dual_bound = solver.compute_bound(net, bounds, c.unsqueeze(0))
            true_min = _sample_objective_min(net, c, lb[0], ub[0])
            assert dual_bound.item() <= true_min + 1e-3, (
                f"UNSOUND mixed-c seed={seed} y_true={y_true} j={j}: "
                f"bound={dual_bound.item():.4f} > true_min~={true_min:.4f}"
            )


# ---------------------------------------------------------------------------
# Batched backward = sequential backward
# ---------------------------------------------------------------------------


def test_compute_robust_bound_batched_matches_single() -> None:
    """B=N batched compute_robust_bound equals N sequential B=1 calls."""
    net = _random_mlp([4, 6, 3], seed=42)
    N = 3
    lb_single = torch.tensor([-0.5, -0.5, -0.5, -0.5])
    ub_single = torch.tensor([0.5, 0.5, 0.5, 0.5])

    bounds_batched: dict[int, Bounds] = {}
    bounds_single_list = []
    for i in range(N):
        lb = lb_single + 0.1 * i
        ub = ub_single + 0.1 * i
        b = compute_forward_bounds(
            net, lb.unsqueeze(0), ub.unsqueeze(0), post_activation=False
        )
        bounds_single_list.append(b)
        for lid, bb in b.items():
            if lid not in bounds_batched:
                bounds_batched[lid] = Bounds(bb.lb.clone(), bb.ub.clone())
            else:
                bounds_batched[lid] = Bounds(
                    torch.cat([bounds_batched[lid].lb, bb.lb], dim=0),
                    torch.cat([bounds_batched[lid].ub, bb.ub], dim=0),
                )

    solver = DualSolver(DualTF())
    margins_batched, _ = cast(
        Tuple[torch.Tensor, torch.Tensor],
        solver.compute_robust_bound(
            net, bounds_batched, y_true=0, num_classes=3
        ),
    )

    margins_sequential = []
    for b in bounds_single_list:
        m, _ = cast(
            Tuple[torch.Tensor, torch.Tensor],
            DualSolver(DualTF()).compute_robust_bound(
                net, b, y_true=0, num_classes=3
            ),
        )
        margins_sequential.append(m[0].item())

    for i in range(N):
        assert margins_batched[i].item() == pytest.approx(
            margins_sequential[i], abs=1e-4
        ), f"Batched != sequential at i={i}"


def test_forward_backward_compat_no_alphas() -> None:
    torch.manual_seed(1234)
    net = _random_mlp([3, 5, 4, 2], seed=7)
    lb = torch.tensor([[-0.8, -0.3, -1.1], [-0.2, -0.7, -0.5]])
    ub = torch.tensor([[0.4, 0.9, 0.6], [0.8, 0.1, 0.7]])

    bounds = compute_forward_bounds(net, lb, ub, post_activation=False)

    expected = {
        lid: (
            b.lb.detach().clone(),
            b.ub.detach().clone(),
        )
        for lid, b in bounds.items()
    }
    repeat = compute_forward_bounds(net, lb, ub, post_activation=False)

    for lid, (exp_lb, exp_ub) in expected.items():
        assert torch.allclose(repeat[lid].lb, exp_lb, atol=0.0, rtol=0.0)
        assert torch.allclose(repeat[lid].ub, exp_ub, atol=0.0, rtol=0.0)


def test_forward_alphas_none_bit_identical() -> None:
    """With alphas=None, compute_forward_bounds must be byte-identical."""
    weights = [
        torch.tensor(
            [
                [0.7, -0.2],
                [-0.4, 0.5],
            ]
        ),
        torch.tensor(
            [
                [0.3, -0.6],
            ]
        ),
    ]
    biases = [torch.tensor([0.1, -0.3]), torch.tensor([0.2])]
    net = _make_mlp_net([2, 2, 1], weights, biases)
    lb = torch.tensor([[-0.8, -0.1]])
    ub = torch.tensor([[0.4, 0.9]])

    without_kwarg = compute_forward_bounds(net, lb, ub, post_activation=False)
    with_none = compute_forward_bounds(net, lb, ub, post_activation=False, alphas=None)

    for lid in without_kwarg:
        assert torch.equal(without_kwarg[lid].lb, with_none[lid].lb)
        assert torch.equal(without_kwarg[lid].ub, with_none[lid].ub)


def test_compute_bound_n_iters_zero_matches_legacy_path() -> None:
    net = _stable_relu_test_net()
    lb = torch.tensor([[0.0, 0.0]])
    ub = torch.tensor([[1.0, 1.0]])
    bounds = compute_forward_bounds(net, lb, ub, post_activation=False)
    c = torch.tensor([[1.0, -0.25]])

    legacy_obj, legacy_sce = cast(
        Tuple[torch.Tensor, torch.Tensor | None],
        DualSolver(DualTF()).compute_bound(net, bounds, c, return_sce=True),
    )
    explicit_obj, explicit_sce = cast(
        Tuple[torch.Tensor, torch.Tensor | None],
        DualSolver(DualTF()).compute_bound(
            net,
            bounds,
            c,
            n_iters=0,
            return_sce=True,
        ),
    )

    assert torch.equal(explicit_obj, legacy_obj)
    assert legacy_sce is not None and explicit_sce is not None
    assert torch.equal(explicit_sce, legacy_sce)


@pytest.mark.parametrize("alpha_val", [0.0, 0.5, 1.0])
def test_forward_with_provided_alphas_is_sound(alpha_val: float) -> None:
    net = _random_mlp([3, 6, 2], seed=11)
    lb = torch.tensor([[-1.0, -0.5, -0.75]])
    ub = torch.tensor([[0.8, 1.1, 0.9]])
    base_bounds = compute_forward_bounds(net, lb, ub, post_activation=False)
    alphas = _constant_alphas(net, base_bounds, alpha_val)

    post_bounds = compute_forward_bounds(
        net,
        lb,
        ub,
        post_activation=True,
        alphas=alphas,
    )
    concrete_y = _sample_concrete_outputs(net, lb[0], ub[0])
    out_bounds = post_bounds[net.layers[-1].id]

    assert torch.all(concrete_y >= out_bounds.lb[0].unsqueeze(0) - 1e-5)
    assert torch.all(concrete_y <= out_bounds.ub[0].unsqueeze(0) + 1e-5)


def test_joint_alpha_tighter_than_same_slope() -> None:
    net = _random_mlp([3, 6, 5, 2], seed=19)
    lb = torch.full((1, 3), -1.0)
    ub = torch.full((1, 3), 1.0)
    bounds = compute_forward_bounds(net, lb, ub, post_activation=False)
    c = torch.tensor([1.0, -1.0])

    baseline_obj, baseline_sce = cast(
        Tuple[torch.Tensor, torch.Tensor | None],
        DualSolver(DualTF()).compute_bound(net, bounds, c, n_iters=0),
    )
    joint_obj, joint_sce = cast(
        Tuple[torch.Tensor, torch.Tensor | None],
        DualSolver(DualTF()).compute_bound(
            net,
            bounds,
            c,
            n_iters=30,
            force_kkt=True,
        ),
    )

    assert joint_obj.item() >= baseline_obj.item() - 1e-6
    assert baseline_sce is None and joint_sce is None


def test_joint_alpha_shared_batch_matches_independent() -> None:
    net = _random_mlp([3, 6, 5, 4], seed=23)
    lb = torch.tensor([[-1.0, -0.75, -0.5]])
    ub = torch.tensor([[0.6, 0.8, 1.0]])
    bounds = compute_forward_bounds(net, lb, ub, post_activation=False)

    baseline, _, _, _, _ = cast(
        Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            Dict[int, torch.Tensor] | None,
            object,
        ],
        DualSolver(DualTF()).compute_robust_bound(
            net, bounds, y_true=0, num_classes=4, n_iters=0
        ),
    )
    joint, _, _, _, _ = cast(
        Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            Dict[int, torch.Tensor] | None,
            object,
        ],
        DualSolver(DualTF()).compute_robust_bound(
            net, bounds, y_true=0, num_classes=4, n_iters=10
        ),
    )

    assert joint.shape == baseline.shape == (1,)
    assert joint.item() >= baseline.item() - 1e-6


def test_joint_alpha_gradient_flows() -> None:
    """Use a non-trivial MLP (input dim > 1, wider hidden layer, mixed-sign
    weights and non-zero biases) so that the joint-α dual objective depends on
    α and a true gradient flows.

    The previous 1-1-1-1 symmetric network produced an objective independent
    of α (mathematically zero gradient). This test replaces it with a small
    random MLP so autograd sees a non-zero gradient w.r.t. α.
    """
    # 3 dense layers => 2 ReLU layers. Input dim > 1 and hidden layers wider.
    net = _random_mlp([3, 6, 4, 2], seed=42)

    lb = torch.tensor([[-1.0, -0.8, -0.5]])
    ub = torch.tensor([[1.0, 0.9, 1.2]])
    base_bounds = compute_forward_bounds(net, lb, ub, post_activation=False)
    relu_lids = _relu_layer_ids(net)

    # Make α a true Parameter (expand to neuron shape) so gradients collect.
    alpha = torch.nn.Parameter(
        torch.tensor(0.5).expand_as(base_bounds[relu_lids[0]].lb).clone()
    )
    alphas = {
        relu_lids[0]: alpha,
        # keep the second ReLU α fixed to 0.5 (non-trainable) for simplicity
        relu_lids[1]: torch.full_like(base_bounds[relu_lids[1]].lb, 0.5),
    }

    fresh_bounds = compute_forward_bounds(
        net,
        lb,
        ub,
        post_activation=False,
        alphas=alphas,
    )

    dual = DualTF()
    dual._bounds_dict = fresh_bounds
    dual._alphas = alphas

    # v has same width as network output (2 here); use mixed signs so both
    # positive and negative branches contribute to the dual objective.
    v = torch.tensor([[-1.0, 0.5]])
    obj, _ = dual._backward_objective(net, v, return_sce=False)
    (-obj.sum()).backward()

    assert alpha.grad is not None
    assert alpha.grad.abs().sum().item() > 0


def test_per_class_alpha_strictly_tightens_shared() -> None:
    torch.manual_seed(17)
    net = _random_mlp([4, 12, 10, 6], seed=17)
    lb = torch.full((1, 4), -0.5)
    ub = torch.full((1, 4), 0.5)
    bounds = compute_forward_bounds(net, lb, ub, post_activation=False)
    num_classes = 6
    y_true = 0

    shared_margin, _, _, _, _ = cast(
        Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            Dict[int, torch.Tensor] | None,
            object,
        ],
        DualSolver(DualTF()).compute_robust_bound(
            net, bounds, y_true=y_true, num_classes=num_classes, n_iters=30, lr=0.1
        ),
    )

    per_class_margin, _, _, _, _ = cast(
        Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            Dict[int, torch.Tensor] | None,
            object,
        ],
        DualSolver(DualTF()).compute_robust_bound(
            net,
            bounds,
            y_true=y_true,
            num_classes=num_classes,
            n_iters=30,
            lr=0.1,
            per_class_alpha=True,
        ),
    )

    assert (per_class_margin >= shared_margin - 1e-6).all(), (
        "per-class alpha MUST never produce looser bounds than shared alpha"
    )
    strictly_tighter = (per_class_margin > shared_margin + 1e-4).any()
    assert strictly_tighter, (
        f"per-class alpha must strictly tighten at least one class pair; "
        f"shared={shared_margin.tolist()} per_class={per_class_margin.tolist()}"
    )


def test_warm_start_returns_alpha_eta() -> None:
    torch.manual_seed(2026)
    net = _random_mlp([3, 6, 5, 2], seed=31)
    N = 2
    lb = torch.tensor([[-0.5, -0.5, -0.5], [-0.3, -0.4, -0.6]])
    ub = torch.tensor([[0.5, 0.5, 0.5], [0.3, 0.4, 0.6]])
    bounds = compute_forward_bounds(net, lb, ub, post_activation=False)

    _, _, _, out_a, out_e = cast(
        Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            Dict[int, torch.Tensor] | None,
            object,
        ],
        DualSolver(DualTF()).compute_robust_bound(
            net, bounds, y_true=0, num_classes=2, n_iters=5, lr=0.1
        ),
    )
    assert out_a is not None, "warm α should be returned when KKT ran"
    for lid, a in out_a.items():
        assert a.shape[0] == N, f"out_alpha[{lid}] must have batch N, got {a.shape}"
        assert ((a >= 0) & (a <= 1)).all(), f"out_alpha[{lid}] out of [0,1]"


def test_warm_start_monotonic_improves_or_matches() -> None:
    torch.manual_seed(2026)
    net = _random_mlp([3, 6, 4, 2], seed=37)
    lb = torch.tensor([[-0.4, -0.4, -0.4], [-0.3, -0.5, -0.5]])
    ub = torch.tensor([[0.4, 0.4, 0.4], [0.3, 0.5, 0.5]])
    bounds = compute_forward_bounds(net, lb, ub, post_activation=False)

    cold_margins, _, _, out_a_cold, out_e_cold = cast(
        Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            Dict[int, torch.Tensor] | None,
            object,
        ],
        DualSolver(DualTF()).compute_robust_bound(
            net, bounds, y_true=0, num_classes=2, n_iters=20, lr=0.1
        ),
    )

    warm_margins, _, _, _, _ = cast(
        Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            Dict[int, torch.Tensor] | None,
            object,
        ],
        DualSolver(DualTF()).compute_robust_bound(
            net,
            bounds,
            y_true=0,
            num_classes=2,
            n_iters=20,
            lr=0.1,
            warm_alphas=out_a_cold,
            warm_etas=out_e_cold,
        ),
    )
    assert (warm_margins >= cold_margins - 1e-3).all(), (
        f"warm-start bound drift exceeds 1e-3 tolerance. "
        f"cold={cold_margins.tolist()} warm={warm_margins.tolist()}. "
        f"Small drift expected from Adam non-monotonicity (best-peak tracking), "
        f"but should stay within noise floor."
    )


def test_warm_start_is_sound_monte_carlo() -> None:
    torch.manual_seed(2026)
    net = _random_mlp([3, 8, 5, 3], seed=41)
    lb = torch.tensor([[-0.4, -0.4, -0.4]])
    ub = torch.tensor([[0.4, 0.4, 0.4]])
    bounds = compute_forward_bounds(net, lb, ub, post_activation=False)

    _, _, _, out_a, out_e = cast(
        Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            Dict[int, torch.Tensor] | None,
            object,
        ],
        DualSolver(DualTF()).compute_robust_bound(
            net, bounds, y_true=0, num_classes=3, n_iters=10, lr=0.1
        ),
    )

    warm_margins, _, _, _, _ = cast(
        Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            Dict[int, torch.Tensor] | None,
            object,
        ],
        DualSolver(DualTF()).compute_robust_bound(
            net,
            bounds,
            y_true=0,
            num_classes=3,
            n_iters=10,
            lr=0.1,
            warm_alphas=out_a,
            warm_etas=out_e,
        ),
    )

    n_samples = 500
    xs = lb + (ub - lb) * torch.rand(n_samples, lb.shape[1])
    with torch.no_grad():
        out = xs
        for layer in net.layers:
            k = layer.kind.upper()
            if k == LayerKind.INPUT.value:
                continue
            if k == LayerKind.DENSE.value:
                W, b = layer.params["weight"], layer.params["bias"]
                out = out @ W.T + b
            elif k == LayerKind.RELU.value:
                out = out.clamp(min=0)

        y_true = 0
        for j in range(out.shape[1]):
            if j == y_true:
                continue
            sampled_margins = out[:, y_true] - out[:, j]
            assert (sampled_margins >= warm_margins[0].item() - 1e-4).all(), (
                f"warm-start bound UNSOUND: class {j} certified {warm_margins[0].item():.4f} "
                f"but saw concrete sample {sampled_margins.min().item():.4f}"
            )


def test_compute_linear_bound_single_row() -> None:
    net = _stable_relu_test_net()
    lb = torch.tensor([[0.0, 0.0]])
    ub = torch.tensor([[1.0, 1.0]])
    bounds = compute_forward_bounds(net, lb, ub, post_activation=False)

    C = torch.tensor([[1.0, 0.0]])
    row_obj, _row_sce = cast(
        Tuple[torch.Tensor, torch.Tensor | None],
        DualSolver(DualTF()).compute_bound(net, bounds, C[0], n_iters=0),
    )
    d = torch.tensor([row_obj.item() - 0.5])

    best_lb, certified, best_rows, out_a, out_e = DualSolver(DualTF()).compute_linear_bound(
        net,
        bounds,
        C,
        d,
        n_iters=5,
        lr=0.1,
    )

    assert best_lb.shape == (1,)
    assert certified.tolist() == [True]
    assert best_rows == [0]
    assert best_lb.item() == pytest.approx(0.5, abs=1e-5)
    assert out_a is None or all(v.shape[0] == 1 for v in out_a.values())
    assert out_e is None or all(v.shape[0] == 1 for v in out_e.values())


def test_compute_linear_bound_multi_row() -> None:
    net = _stable_relu_test_net()
    lb = torch.tensor([[0.0, 0.0]])
    ub = torch.tensor([[1.0, 1.0]])
    bounds = compute_forward_bounds(net, lb, ub, post_activation=False)

    C = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    row_bounds = torch.stack(
        [DualSolver(DualTF()).compute_bound(net, bounds, row, n_iters=0)[0] for row in C]
    ).reshape(-1)
    d = torch.tensor([row_bounds[0].item() + 0.2, row_bounds[1].item() - 0.4])

    best_lb, certified, best_rows, _, _ = DualSolver(DualTF()).compute_linear_bound(
        net,
        bounds,
        C,
        d,
        n_iters=5,
        lr=0.1,
    )

    assert best_lb.shape == (1,)
    assert certified.tolist() == [True]
    assert best_rows == [1]
    assert best_lb.item() == pytest.approx(0.4, abs=1e-5)


def test_compute_linear_bound_falsifiable() -> None:
    net = _stable_relu_test_net()
    lb = torch.tensor([[0.0, 0.0]])
    ub = torch.tensor([[1.0, 1.0]])
    bounds = compute_forward_bounds(net, lb, ub, post_activation=False)

    C = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, -1.0],
        ]
    )
    row_bounds = torch.stack(
        [DualSolver(DualTF()).compute_bound(net, bounds, row, n_iters=0)[0] for row in C]
    ).reshape(-1)
    d = row_bounds + torch.tensor([1.0, 0.6, 0.2])

    best_lb, certified, best_rows, _, _ = DualSolver(DualTF()).compute_linear_bound(
        net,
        bounds,
        C,
        d,
        n_iters=5,
        lr=0.1,
    )

    expected = row_bounds - d
    assert certified.tolist() == [False]
    assert best_lb.item() <= 0.0
    assert best_rows == [int(expected.argmax().item())]
    assert best_lb.item() == pytest.approx(expected.max().item(), abs=1e-5)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
