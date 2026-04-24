# pyright: reportMissingImports=false, reportImportCycles=false
#===- act/back_end/dual_tf/dual_tf.py - Dual Transfer Function Class ----====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025- ACT Team
# Licensed under AGPLv3+; distributed without warranty.
#===---------------------------------------------------------------------===#
# DualTF: forward-only TransferFunction for dual mode. Holds alpha parameters.
# All backward kernels live as module-level dispatch functions; the actual
# certified-bound solver is act.back_end.solver.solver_dual.DualSolver.
#===---------------------------------------------------------------------===#


import importlib

import torch
from typing import Dict, Optional, Tuple, cast
from act.back_end.core import Bounds, Fact, Layer, Net, ConSet
from act.back_end.layer_schema import LayerKind
from act.back_end.transfer_functions import TransferFunction
from .tf_mlp import (
    backward_dense, backward_relu, backward_bias, backward_scale,
    backward_bn, backward_identity,
    forward_dense, forward_relu, forward_bias, forward_scale,
    forward_bn, forward_lrelu, forward_identity, forward_reshape,
)
from .tf_cnn import (
    backward_maxpool2d, backward_avgpool2d,
    forward_maxpool2d, forward_avgpool2d,
)
from .tf_cnn_patches import backward_conv2d_patches
from .tf_smooth import (
    backward_sigmoid, backward_tanh,
    forward_sigmoid, forward_tanh,
)
from .tf_rnn import forward_lstm, backward_lstm, forward_gru, backward_gru
from .tf_transformer import (
    forward_attention, backward_attention,
    forward_layernorm, backward_layernorm,
    forward_gelu, backward_gelu,
)
from .tf_forward import (
    compute_forward_bounds,
    forward_add, backward_add,
    forward_concat, backward_concat,
)


def _forward_conv2d_dispatch(*args, **kwargs):
    from act.back_end.bounds_dispatch import dispatch_conv_forward

    return dispatch_conv_forward(*args, **kwargs)


class DualTF(TransferFunction):
    """Forward-only TF for dual mode. Backward kernels are module-level dispatch
    functions; the actual solver lives at act.back_end.solver.DualSolver.

    `_BACKWARD_REGISTRY` maps layer-kind strings to callable dispatch functions
    with DAG-aware per-predecessor ν routing:

        (L: Layer, nu: Tensor[B, *shape], bounds_dict: Dict[int, Bounds],
         preds: List[int]) -> (pred_nus: List[Tensor], contrib: Tensor[B])

    Each ``pred_nus[i]`` is the ν routed to predecessor ``preds[i]``. Unary
    layers return ``[nu_out]``; ADD returns ``[nu] * len(preds)`` (identity
    skip, same ν to every predecessor). net_factory.py reads only ``.keys()``
    so callable values are fine.
    """

    _FORWARD_REGISTRY = {
        LayerKind.INPUT.value:      forward_identity,
        LayerKind.INPUT_SPEC.value: forward_identity,
        LayerKind.ASSERT.value:     forward_identity,
        LayerKind.DENSE.value:      forward_dense,
        LayerKind.BIAS.value:       forward_bias,
        LayerKind.SCALE.value:      forward_scale,
        LayerKind.BN.value:         forward_bn,
        LayerKind.RELU.value:       forward_relu,
        LayerKind.LRELU.value:      forward_lrelu,
        "LEAKY_RELU":               forward_lrelu,   # alias (not a LayerKind member)
        LayerKind.SIGMOID.value:    forward_sigmoid,
        LayerKind.TANH.value:       forward_tanh,
        LayerKind.CONV2D.value:     _forward_conv2d_dispatch,
        LayerKind.MAXPOOL2D.value:  forward_maxpool2d,
        LayerKind.AVGPOOL2D.value:  forward_avgpool2d,
        LayerKind.FLATTEN.value:    forward_reshape,
        LayerKind.RESHAPE.value:    forward_reshape,
        LayerKind.TRANSPOSE.value:  forward_identity,
        LayerKind.SQUEEZE.value:    forward_identity,
        LayerKind.UNSQUEEZE.value:  forward_identity,
        LayerKind.ADD.value:        forward_add,
        LayerKind.CONCAT.value:     forward_concat,
        LayerKind.LSTM.value:       forward_lstm,
        LayerKind.GRU.value:        forward_gru,
        LayerKind.ATT_SCORES.value: forward_attention,
        LayerKind.ATT_MIX.value:    forward_attention,
        LayerKind.MHA_SPLIT.value:  forward_attention,
        LayerKind.MHA_JOIN.value:   forward_attention,
        LayerKind.MASK_ADD.value:   forward_attention,
        LayerKind.LAYERNORM.value:  forward_layernorm,
        LayerKind.GELU.value:       forward_gelu,
    }

    _BACKWARD_REGISTRY = {
        LayerKind.INPUT.value:      backward_identity,
        LayerKind.INPUT_SPEC.value: backward_identity,
        LayerKind.ASSERT.value:     backward_identity,
        LayerKind.DENSE.value:      backward_dense,
        LayerKind.BIAS.value:       backward_bias,
        LayerKind.SCALE.value:      backward_scale,
        LayerKind.BN.value:         backward_bn,
        LayerKind.RELU.value:       backward_relu,
        LayerKind.LRELU.value:      backward_relu,
        "LEAKY_RELU":               backward_relu,   # alias (not a LayerKind member)
        LayerKind.SIGMOID.value:    backward_sigmoid,
        LayerKind.TANH.value:       backward_tanh,
        LayerKind.CONV2D.value:     backward_conv2d_patches,
        LayerKind.MAXPOOL2D.value:  backward_maxpool2d,
        LayerKind.AVGPOOL2D.value:  backward_avgpool2d,
        LayerKind.FLATTEN.value:    backward_identity,
        LayerKind.RESHAPE.value:    backward_identity,
        LayerKind.TRANSPOSE.value:  backward_identity,
        LayerKind.SQUEEZE.value:    backward_identity,
        LayerKind.UNSQUEEZE.value:  backward_identity,
        LayerKind.ADD.value:        backward_add,
        LayerKind.CONCAT.value:     backward_concat,
        LayerKind.LSTM.value:       backward_lstm,
        LayerKind.GRU.value:        backward_gru,
        LayerKind.ATT_SCORES.value: backward_attention,
        LayerKind.ATT_MIX.value:    backward_attention,
        LayerKind.MHA_SPLIT.value:  backward_attention,
        LayerKind.MHA_JOIN.value:   backward_attention,
        LayerKind.MASK_ADD.value:   backward_attention,
        LayerKind.LAYERNORM.value:  backward_layernorm,
        LayerKind.GELU.value:       backward_gelu,
    }

    def __init__(self):
        self._forward_bounds_cache: Dict[int, Bounds] = {}
        self._cache_net_id: Optional[int] = None
        self._bounds_dict: Optional[Dict[int, Bounds]] = None
        self._alphas: Optional[Dict[int, torch.Tensor]] = None

    @property
    def name(self) -> str: return "DualTF"

    def supports_layer(self, layer_kind: str) -> bool:
        return layer_kind.upper() in self._BACKWARD_REGISTRY

    def apply(self, L: Layer, input_bounds: Bounds, net: Net,
              before: Dict[int, Fact], after: Dict[int, Fact],
              alphas: Optional[Dict[int, torch.Tensor]] = None) -> Fact:
        """Return unbatched Bounds Fact for analyze()/BaB integration."""
        net_id = id(net)
        use_cache = alphas is None
        if (not use_cache or self._cache_net_id != net_id or not self._forward_bounds_cache):
            input_lb, input_ub = None, None
            for layer in net.layers:
                if layer.kind.upper() in (LayerKind.INPUT.value, LayerKind.INPUT_SPEC.value):
                    if layer.id in before:
                        input_lb = before[layer.id].bounds.lb
                        input_ub = before[layer.id].bounds.ub
                        break
                    elif "lb" in layer.params and "ub" in layer.params:
                        input_lb = cast(torch.Tensor, layer.params["lb"])
                        input_ub = cast(torch.Tensor, layer.params["ub"])
                        break
            if input_lb is None or input_ub is None:
                input_lb, input_ub = input_bounds.lb, input_bounds.ub
            computed = compute_forward_bounds(
                net, input_lb, input_ub, post_activation=True, alphas=alphas)
            if use_cache:
                self._forward_bounds_cache = computed
                self._cache_net_id = net_id
            else:
                self._forward_bounds_cache = {}
                self._cache_net_id = None
                if L.id in computed:
                    return Fact(bounds=computed[L.id], cons=ConSet())
                return Fact(bounds=input_bounds, cons=ConSet())

        if L.id in self._forward_bounds_cache:
            return Fact(bounds=self._forward_bounds_cache[L.id], cons=ConSet())
        return Fact(bounds=input_bounds, cons=ConSet())

    def clear_cache(self):
        self._forward_bounds_cache.clear()
        self._cache_net_id = None

    def _backward_objective(
        self,
        net: "Net",
        v: torch.Tensor,
        return_sce: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Convenience wrapper for gradient-flow tests."""
        if self._bounds_dict is None:
            raise ValueError("DualTF._backward_objective: _bounds_dict must be set before use")
        solver_module = importlib.import_module("act.back_end.solver.solver_dual")
        DualSolver = getattr(solver_module, "DualSolver")
        solver = DualSolver(self)
        return solver._backward_pass(
            net,
            self._bounds_dict,
            v,
            alphas=self._alphas,
            return_sce=return_sce,
        )


# Explicit stub registry: any handler whose semantics are "raise NotImplementedError"
# goes here. Membership is the ground truth for stub detection; net_factory filters
# by identity against these sets.
# To implement a stub: fill its body AND remove it from this set in the same commit.
_FORWARD_STUBS = frozenset({
    forward_lstm, forward_gru, forward_attention,
    forward_layernorm, forward_gelu,
})
_BACKWARD_STUBS = frozenset({
    backward_maxpool2d, backward_avgpool2d,
    backward_lstm, backward_gru, backward_attention,
    backward_layernorm, backward_gelu,
})

# --- registry invariant (fires once at module import) ---
assert set(DualTF._FORWARD_REGISTRY.keys()) == set(DualTF._BACKWARD_REGISTRY.keys()), (
    f"DualTF registry keyset mismatch: "
    f"forward-only={set(DualTF._FORWARD_REGISTRY) - set(DualTF._BACKWARD_REGISTRY)}, "
    f"backward-only={set(DualTF._BACKWARD_REGISTRY) - set(DualTF._FORWARD_REGISTRY)}"
)
