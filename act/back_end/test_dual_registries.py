#===- act/back_end/test_dual_registries.py -----------------------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025- ACT Team
# Licensed under the GNU Affero General Public License v3.0 or later.
#===--------------------------------------------------------------------===#
#
# Purpose:
#   Registry-invariant tests for DualTF._FORWARD_REGISTRY and
#   DualTF._BACKWARD_REGISTRY. Verifies keyset equality, handler
#   callability, stub-set consistency, and the forward-handler contract.
#
#===--------------------------------------------------------------------===#

# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUntypedFunctionDecorator=false, reportPrivateUsage=false, reportUnusedFunction=false, reportOperatorIssue=false

import pytest
import torch

from act.back_end.core import Bounds, Layer, Net
from act.back_end.dual_tf.dual_tf import DualTF, _FORWARD_STUBS, _BACKWARD_STUBS
from act.back_end.dual_tf.tf_forward import LinearBound, _identity_lin, compute_forward_bounds
from act.back_end.dual_tf.tf_mlp import forward_dense
from act.back_end.layer_schema import LayerKind
from act.util.device_manager import get_default_device, get_default_dtype, initialize_device


@pytest.fixture(scope="module", autouse=True)
def _cpu_float32() -> None:
    """Run these registry tests on a stable CPU/float32 configuration so that
    tensors built with torch.device('cpu') match compute_forward_bounds()'s
    internal default device/dtype."""
    initialize_device("cpu", "float32")


# ---------------------------------------------------------------------
# Test 1: keysets match
# ---------------------------------------------------------------------

def test_keysets_match():
    """DualTF._FORWARD_REGISTRY and _BACKWARD_REGISTRY must have identical keysets."""
    fwd = set(DualTF._FORWARD_REGISTRY.keys())
    bwd = set(DualTF._BACKWARD_REGISTRY.keys())
    assert fwd == bwd, (
        f"fwd-only={fwd - bwd}, bwd-only={bwd - fwd}"
    )


# ---------------------------------------------------------------------
# Test 2: all handlers callable
# ---------------------------------------------------------------------

def test_all_handlers_callable():
    """Every value in both registries must be callable."""
    for registry_name, registry in [
        ("forward", DualTF._FORWARD_REGISTRY),
        ("backward", DualTF._BACKWARD_REGISTRY),
    ]:
        for kind, fn in registry.items():
            assert callable(fn), f"{registry_name}[{kind!r}] is not callable: {fn!r}"


# ---------------------------------------------------------------------
# Test 3: stub sets point at registered functions
# ---------------------------------------------------------------------

def test_stub_sets_point_at_registered_fns():
    """Every function in _FORWARD_STUBS must appear in _FORWARD_REGISTRY.values();
    every function in _BACKWARD_STUBS must appear in _BACKWARD_REGISTRY.values().
    No dangling stub entries."""
    fwd_fns = set(DualTF._FORWARD_REGISTRY.values())
    bwd_fns = set(DualTF._BACKWARD_REGISTRY.values())
    orphan_fwd = _FORWARD_STUBS - fwd_fns
    orphan_bwd = _BACKWARD_STUBS - bwd_fns
    assert not orphan_fwd, f"orphan forward stubs: {orphan_fwd}"
    assert not orphan_bwd, f"orphan backward stubs: {orphan_bwd}"


# ---------------------------------------------------------------------
# Test 4: stubs raise NotImplementedError
# ---------------------------------------------------------------------

def test_stubs_raise_not_implemented():
    """Every handler in the stub sets must raise NotImplementedError when its
    body executes. Forward and backward handlers have different locked
    signatures (plan §4.2 / §4.4), so we dispatch on set membership and pass
    minimal dummy arguments that let the body run to its `raise` statement.
    """
    device, dtype = get_default_device(), get_default_dtype()
    dummy_lb = torch.zeros(1, 1, device=device, dtype=dtype)
    dummy_ub = torch.zeros(1, 1, device=device, dtype=dtype)
    dummy_box = Bounds(dummy_lb, dummy_ub)
    dummy_lin = _identity_lin(1, 1, device, dtype)
    dummy_frame = (dummy_lb, dummy_ub)
    dummy_nu = torch.zeros(1, 1, device=device, dtype=dtype)
    dummy_bounds_dict: dict[int, Bounds] = {}
    dummy_preds: list[int] = []

    class _DummyLayer:
        id: int = 0
        kind: str = "STUB"
        params: dict[str, object] = {}

    for fn in _FORWARD_STUBS:
        with pytest.raises(NotImplementedError):
            fn(_DummyLayer(), [dummy_box], [dummy_lin], [dummy_frame],
               dummy_preds, False, device, dtype)

    for fn in _BACKWARD_STUBS:
        with pytest.raises(NotImplementedError):
            fn(_DummyLayer(), dummy_nu, dummy_bounds_dict, dummy_preds)


# ---------------------------------------------------------------------
# Test 5: forward handler contract via direct call (4-tuple)
# ---------------------------------------------------------------------

def test_forward_handler_contract_via_direct_call():
    """Directly invoke a REAL forward handler and verify the 4-tuple return
    contract (stored, out, lin, frame). compute_forward_bounds() returns a
    Dict[int, Bounds], not the per-handler tuple — we must call a handler
    directly to observe the tuple shape.
    """
    B, n_in, n_out = 2, 3, 4
    device, dtype = get_default_device(), get_default_dtype()

    lb = torch.zeros(B, n_in, device=device, dtype=dtype)
    ub = torch.ones(B, n_in, device=device, dtype=dtype)
    parent_box = Bounds(lb, ub)
    parent_lin = _identity_lin(B, n_in, device, dtype)
    parent_frame = (lb, ub)

    weight = torch.randn(n_out, n_in, device=device, dtype=dtype)
    bias = torch.zeros(n_out, device=device, dtype=dtype)

    L = Layer(
        id=1, kind=LayerKind.DENSE.value,
        params={
            "weight":       weight,
            "bias":         bias,
            "in_features":  n_in,
            "out_features": n_out,
        },
        in_vars=list(range(n_in)),
        out_vars=list(range(n_in, n_in + n_out)),
    )

    result = forward_dense(
        L, [parent_box], [parent_lin], [parent_frame], [0],
        False, device, dtype,
    )
    assert isinstance(result, tuple) and len(result) == 4, (
        f"forward_dense returned {type(result).__name__} "
        f"len={len(result) if hasattr(result, '__len__') else 'N/A'}"
    )
    stored, out, new_lin, new_frame = result
    assert isinstance(stored, Bounds), f"stored is {type(stored).__name__}, expected Bounds"
    assert isinstance(out, Bounds), f"out is {type(out).__name__}, expected Bounds"
    assert isinstance(new_lin, LinearBound), f"new_lin is {type(new_lin).__name__}, expected LinearBound"
    assert isinstance(new_frame, tuple) and len(new_frame) == 2, (
        f"new_frame is {type(new_frame).__name__} len="
        f"{len(new_frame) if hasattr(new_frame, '__len__') else 'N/A'}, "
        f"expected tuple of length 2"
    )
    assert stored.lb.shape[0] == B
    assert out.lb.shape == (B, n_out)


# ---------------------------------------------------------------------
# Test 6: observable compute_forward_bounds shape contract
# ---------------------------------------------------------------------

def _build_dense_relu_dense_net(n_in: int = 3, n_hid: int = 4, n_out: int = 2):
    """Minimal INPUT -> INPUT_SPEC -> DENSE -> RELU -> DENSE -> ASSERT ACT
    network for observable shape testing. The wrapper layers (INPUT, INPUT_SPEC,
    ASSERT) are required by validate_wrapper_graph() in Net.__post_init__; the
    DENSE/RELU/DENSE core is what exercises the forward registry dispatch.
    """
    device, dtype = get_default_device(), get_default_dtype()

    input_vars  = list(range(n_in))
    dense0_out  = list(range(n_in, n_in + n_hid))
    dense1_out  = list(range(n_in + n_hid, n_in + n_hid + n_out))

    W0 = torch.randn(n_hid, n_in,  device=device, dtype=dtype)
    b0 = torch.zeros(n_hid,        device=device, dtype=dtype)
    W1 = torch.randn(n_out, n_hid, device=device, dtype=dtype)
    b1 = torch.zeros(n_out,        device=device, dtype=dtype)

    layers = [
        Layer(id=0, kind=LayerKind.INPUT.value,
              params={"shape": [n_in], "dtype": "float32"},
              in_vars=input_vars,  out_vars=input_vars),
        Layer(id=1, kind=LayerKind.INPUT_SPEC.value,
              params={"kind": "BOX"},
              in_vars=input_vars,  out_vars=input_vars),
        Layer(id=2, kind=LayerKind.DENSE.value,
              params={"weight": W0, "bias": b0,
                      "in_features": n_in, "out_features": n_hid},
              in_vars=input_vars,  out_vars=dense0_out),
        Layer(id=3, kind=LayerKind.RELU.value, params={},
              in_vars=dense0_out,  out_vars=dense0_out),
        Layer(id=4, kind=LayerKind.DENSE.value,
              params={"weight": W1, "bias": b1,
                      "in_features": n_hid, "out_features": n_out},
              in_vars=dense0_out,  out_vars=dense1_out),
        Layer(id=5, kind=LayerKind.ASSERT.value,
              params={"kind": "RANGE"},
              in_vars=dense1_out,  out_vars=dense1_out),
    ]
    preds = {0: [], 1: [0], 2: [1], 3: [2], 4: [3], 5: [4]}
    succs = {0: [1], 1: [2], 2: [3], 3: [4], 4: [5], 5: []}
    return Net(layers=layers, preds=preds, succs=succs)


def test_observable_compute_forward_bounds_shape():
    """Verify the public compute_forward_bounds() contract: returns
    Dict[int, Bounds] with one entry per layer, each Bounds a batch-first
    tensor pair with lb <= ub elementwise.
    """
    B = 2
    n_in = 3
    device, dtype = get_default_device(), get_default_dtype()
    net = _build_dense_relu_dense_net(n_in=n_in)
    input_lb = torch.zeros(B, n_in, device=device, dtype=dtype)
    input_ub = torch.ones(B, n_in,  device=device, dtype=dtype)

    result = compute_forward_bounds(net, input_lb, input_ub, post_activation=True)

    assert isinstance(result, dict)
    layer_ids = {L.id for L in net.layers}
    assert set(result.keys()) == layer_ids, (
        f"missing layers: {layer_ids - set(result.keys())}, "
        f"extra: {set(result.keys()) - layer_ids}"
    )

    for lid, bounds in result.items():
        assert isinstance(bounds, Bounds), (
            f"layer {lid} value is {type(bounds).__name__}, expected Bounds"
        )
        assert bounds.lb.shape[0] == B, (
            f"layer {lid} lb batch dim = {bounds.lb.shape[0]}, expected {B}"
        )
        assert bounds.ub.shape == bounds.lb.shape, (
            f"layer {lid} ub shape {bounds.ub.shape} != lb shape {bounds.lb.shape}"
        )
        assert torch.all(bounds.lb <= bounds.ub), (
            f"layer {lid} has lb > ub in some position"
        )


# ---------------------------------------------------------------------
# Test 7: factory reports stub kinds as unsupported
# ---------------------------------------------------------------------

def test_factory_reports_stub_kinds_unsupported():
    """net_factory._get_tf_capabilities() must report kinds that are stubs
    on both sides as unsupported for DualTF, and real-handler kinds as
    supported."""
    from act.back_end.net_factory import _get_tf_capabilities
    _get_tf_capabilities.cache_clear()

    caps = _get_tf_capabilities()
    assert "dual" in caps, f"caps missing 'dual' key: {caps.keys()}"

    # Both forward and backward are stubs → must NOT be in caps.
    assert "LSTM" not in caps["dual"], f"LSTM should be stubbed on both sides but caps: {caps['dual']}"
    assert "GRU" not in caps["dual"]
    assert "ATT_SCORES" not in caps["dual"]
    assert "ATT_MIX" not in caps["dual"]
    assert "MHA_SPLIT" not in caps["dual"]
    assert "MHA_JOIN" not in caps["dual"]
    assert "MASK_ADD" not in caps["dual"]
    assert "LAYERNORM" not in caps["dual"]
    assert "GELU" not in caps["dual"]

    # Real on both sides → must be in caps.
    for kind in ("DENSE", "RELU", "CONV2D", "BIAS", "SCALE", "BN", "SIGMOID",
                 "TANH", "LRELU", "ADD", "FLATTEN", "RESHAPE", "TRANSPOSE",
                 "SQUEEZE", "UNSQUEEZE", "INPUT", "INPUT_SPEC", "ASSERT"):
        assert kind in caps["dual"], (
            f"{kind} should be supported but caps['dual']={caps['dual']}"
        )


# ---------------------------------------------------------------------
# Test 8: factory reports asymmetric real kinds as unsupported
# ---------------------------------------------------------------------

def test_factory_reports_asymmetric_real_kinds_unsupported():
    """Kinds with a real handler on only ONE side (forward or backward,
    not both) must be reported as unsupported by _get_tf_capabilities.

    MAXPOOL2D and AVGPOOL2D have real forward handlers but backward stubs —
    they should NOT appear in caps['dual'].

    CONCAT has both forward AND backward implemented (backward_concat
    landed in Stage 3 of the shape-preserving refactor), so it MUST appear
    in caps['dual'].
    """
    from act.back_end.net_factory import _get_tf_capabilities
    _get_tf_capabilities.cache_clear()

    caps = _get_tf_capabilities()
    assert "MAXPOOL2D" not in caps["dual"], (
        f"MAXPOOL2D has real forward but stub backward — should be filtered out; "
        f"caps['dual']={caps['dual']}"
    )
    assert "AVGPOOL2D" not in caps["dual"]
    assert "CONCAT" in caps["dual"], (
        f"CONCAT has real forward AND backward — should be supported; "
        f"caps['dual']={caps['dual']}"
    )
