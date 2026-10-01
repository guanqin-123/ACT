#===- act/back_end/solver/solver_dual.py - Dual Bounds Solver ----------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025- ACT Team
# Licensed under AGPLv3+; distributed without warranty.
#===---------------------------------------------------------------------===#
# DualSolver: linear-relaxation dual certified lower-bound solver.
# STRICT batched API ([B, *shape] only). Raises ValueError on 1-D input.
# Mirrors HZSolver precedent in solver_hz.py.
#===---------------------------------------------------------------------===#
# pyright: reportMissingImports=false, reportImportCycles=false
# justification: torch C-extension stubs are absent in CI; DualSolver and verifier share result utilities during type analysis

from __future__ import annotations
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, TypeAlias, Union, cast

import torch
from act.back_end.core import Bounds, Layer, Net, get_topo_order
from act.back_end.layer_schema import LayerKind
from act.back_end.dual_tf.tf_forward import (
    ForwardFrame,
    _dual_norm_exponent,
    _intersect_boxes,
    _resolve_perturbation_norm,
    forward_frame_row_lower_bounds,
    input_block_slices,
    lp_ball_support,
)
from act.config.config import DualConfig
from act.back_end.solver.solver_base import (
    BatchLPSolution,
    SolveStatus,
    Solver,
    SolverCaps,
)
from act.front_end.specs import OutputSpec, OutKind
from act.util.device_manager import get_default_device, get_default_dtype
from act.util.stats import SpecBatchResult

if TYPE_CHECKING:
    from act.back_end.dual_tf.dual_tf import DualTF


logger = logging.getLogger(__name__)

_monotonic = time.monotonic
"""Clock of the alpha/eta time cap; tests substitute a deterministic clock."""

_MIB = 1024 * 1024
_PER_CLASS_ALPHA_MAX_MIB = 1024.0
_INCREMENTAL_ALPHA_PER_LANE_MAX_MIB = 32.0
_PER_CLASS_ALPHA_CAP_REPORTED: set[Tuple[int, int, int]] = set()
_INCREMENTAL_ALPHA_CAP_REPORTED: set[Tuple[int, int]] = set()


def _per_class_alpha_estimated_bytes(
    net: Net,
    bounds_dict: Dict[int, Bounds],
    lanes: int,
    rows: int,
    dtype: torch.dtype,
) -> int:
    """Bytes needed by one complete per-class ReLU alpha state."""
    neurons = 0
    for layer in net.layers:
        kind = layer.kind.upper() if isinstance(layer.kind, str) else layer.kind
        bounds = bounds_dict.get(layer.id)
        if kind == LayerKind.RELU.value and bounds is not None:
            neurons += int(bounds.lb[0].numel())
    itemsize = torch.empty((), dtype=dtype).element_size()
    return lanes * rows * neurons * itemsize


def _retain_incremental_alpha_state(
    state: Optional[AlphaState], lanes: int,
) -> Optional[AlphaState]:
    """Keep a warm start only when its alpha state is bounded per lane."""
    if state is None:
        return None
    if lanes <= 0:
        raise ValueError(f"incremental alpha lanes must be positive, got {lanes}")
    total_bytes = sum(
        leaf.numel() * leaf.element_size()
        for tree in state.values()
        for leaf in _alpha_tree_leaves(tree)
    )
    cap_bytes = int(_INCREMENTAL_ALPHA_PER_LANE_MAX_MIB * _MIB) * lanes
    if total_bytes <= cap_bytes:
        return state
    report_key = (lanes, total_bytes)
    if report_key not in _INCREMENTAL_ALPHA_CAP_REPORTED:
        _INCREMENTAL_ALPHA_CAP_REPORTED.add(report_key)
        logger.warning(
            "incremental alpha state %.3f MiB exceeds %.3f MiB per-lane cap "
            "for lanes=%d; discarding warm start",
            total_bytes / _MIB,
            _INCREMENTAL_ALPHA_PER_LANE_MAX_MIB,
            lanes,
        )
    return None


def _max_backward_tensor_width(
    net: Net, bounds_dict: Dict[int, Bounds], start_lid: int,
) -> int:
    """Largest per-lane width visited by a backward pass from ``start_lid``."""
    pending = [start_lid]
    visited: set[int] = set()
    max_width = 0
    while pending:
        lid = pending.pop()
        if lid in visited:
            continue
        visited.add(lid)
        bounds = bounds_dict.get(lid)
        if bounds is not None and bounds.lb.dim() >= 2:
            max_width = max(max_width, int(bounds.lb[0].numel()))
        pending.extend(net.preds.get(lid, []))
    return max_width


def _skip_refinement_for_tensor_cap(
    *,
    net: Net,
    bounds_dict: Dict[int, Bounds],
    layer_id: int,
    start_lid: int,
    lanes: int,
    rows: int,
    itemsize: int,
    max_tensor_mib: float,
    refinement_stats: Optional[Dict[str, int]],
) -> bool:
    """Log and apply the deterministic dense-backward tensor cap."""
    width = _max_backward_tensor_width(net, bounds_dict, start_lid)
    estimate_bytes = lanes * rows * width * itemsize
    estimate_mib = estimate_bytes / _MIB
    skipped = estimate_bytes > max_tensor_mib * _MIB
    logger.info(
        "Intermediate refinement layer=%d lanes=%d rows=%d width=%d "
        "estimate=%.3f MiB cap=%.3f MiB action=%s",
        layer_id,
        lanes,
        rows,
        width,
        estimate_mib,
        max_tensor_mib,
        "skipped" if skipped else "refined",
    )
    if skipped and refinement_stats is not None:
        key = "intermediate_refine_skipped_calls"
        refinement_stats[key] = refinement_stats.get(key, 0) + 1
    return skipped


def _effective_time_cap(time_cap: Optional[float], max_time: float) -> Optional[float]:
    """Combine a per-call cap with ``DualConfig.max_time`` (0 = off): the
    tighter of the two applies; ``None`` means no cap."""
    caps = [cap for cap in (time_cap, max_time if max_time > 0.0 else None) if cap is not None]
    return min(caps) if caps else None


AlphaTree: TypeAlias = Union[
    torch.Tensor,
    Dict[Any, "AlphaTree"],
    List["AlphaTree"],
    Tuple["AlphaTree", ...],
    None,
]
AlphaState: TypeAlias = Dict[int, AlphaTree]


@dataclass(frozen=True)
class DualResult:
    """Result of ``compute_certified_bound``. Fields depend on caller flags."""

    margins: torch.Tensor
    sce: Optional[Any] = None
    alpha_state: Optional[AlphaState] = None
    eta_state: Optional[Dict[int, torch.Tensor]] = None
    nu_per_layer: Optional[Dict[int, torch.Tensor]] = None
    # ``margins`` = max(dual_margins, forward_margins) on the spec layer:
    # ``dual_margins`` is the backward certificate alone (what CLIMB replays),
    # ``forward_margins`` the forward frame/box bound of the same rows.
    dual_margins: Optional[torch.Tensor] = None
    forward_margins: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class DualBatchResult:
    """Decoded result of :meth:`DualSolver.solve_spec_batch` for one K-lane batch."""

    solution: BatchLPSolution
    margins: torch.Tensor
    lower_bounds: torch.Tensor
    dual_row_slack: Optional[torch.Tensor] = None
    forward_only_certified: Optional[torch.Tensor] = None
    bounds_dict: Optional[Dict[int, Bounds]] = None
    nu_per_layer: Optional[Dict[int, torch.Tensor]] = None
    alpha_state: Optional[AlphaState] = None
    eta_state: Optional[Dict[int, torch.Tensor]] = None
    witness_input: Optional[torch.Tensor] = None
    row_slack: Optional[torch.Tensor] = None
    c_rows: Optional[torch.Tensor] = None
    thresholds: Optional[torch.Tensor] = None
    m_specs: int = 0
    reference_bounds: Optional[Dict[int, Bounds]] = None
    # Split-free bounds a split_refresh lane was refreshed from (CLIMB C3).
    base_reference_bounds: Optional[Dict[int, Bounds]] = None
    # Actual ReLU mode before optional warm-state retention/shedding.
    alpha_mode: Optional[str] = None


def expand_bounds_dict(bounds_dict: Dict[int, Bounds], M: int) -> Dict[int, Bounds]:
    """Expand each batched Bounds entry from [B, *shape] to [B*M, *shape].

    The dual solver threads ``M`` through ``compute_certified_bound`` and
    broadcasts inside the activation handlers (lazy M-broadcast), avoiding the
    M× memory blowup; this explicit expansion is for callers that need a
    materialized M-expanded bounds dict.

    repeat_interleave aligns with row b*M+j sharing sample b's bounds. All
    entries must already be batched (lb.dim() >= 2). M=1 returns the dict
    unchanged.
    """
    if M <= 0:
        raise ValueError(f"expand_bounds_dict: M must be positive, got {M}")
    if M == 1:
        return dict(bounds_dict)
    out: Dict[int, Bounds] = {}
    for lid, bounds in bounds_dict.items():
        if bounds.lb.dim() < 2:
            raise ValueError(
                f"expand_bounds_dict: layer {lid} bounds must be batched "
                f"[B, *shape], got dim={bounds.lb.dim()} shape={tuple(bounds.lb.shape)}"
            )
        out[lid] = Bounds(
            lb=bounds.lb.repeat_interleave(M, dim=0),
            ub=bounds.ub.repeat_interleave(M, dim=0),
        )
    return out


def _alpha_tree_leaves(tree: Any):
    """Yield the tensor leaves of a dual-alpha pytree.

    A per-layer alpha is a pytree whose shape is interpreted by the matching
    backward kernel: RELU is a single tensor leaf; future per-kind allocators
    may nest leaves in lists/dicts. ``None`` marks a fixed-slope kind with no
    alpha. The optimizer, projection, and keep-best clone all walk these leaves
    so they cannot drift out of agreement on the pytree shape.
    """
    if tree is None:
        return
    if isinstance(tree, torch.Tensor):
        yield tree
    elif isinstance(tree, dict):
        for value in tree.values():
            yield from _alpha_tree_leaves(value)
    elif isinstance(tree, (list, tuple)):
        for value in tree:
            yield from _alpha_tree_leaves(value)
    else:
        raise TypeError(f"unsupported alpha pytree node: {type(tree)!r}")


def _alpha_tree_map(
    tree: AlphaTree,
    transform: Callable[[torch.Tensor], torch.Tensor],
) -> AlphaTree:
    """Apply ``transform`` to every tensor leaf while preserving structure."""
    if tree is None:
        return None
    if isinstance(tree, torch.Tensor):
        return transform(tree)
    if isinstance(tree, dict):
        return {key: _alpha_tree_map(value, transform) for key, value in tree.items()}
    if isinstance(tree, list):
        return [_alpha_tree_map(value, transform) for value in tree]
    if isinstance(tree, tuple):
        return tuple(_alpha_tree_map(value, transform) for value in tree)
    raise TypeError(f"unsupported alpha pytree node: {type(tree)!r}")


def _alpha_tree_gather_lanes(tree: AlphaTree, indices: torch.Tensor) -> AlphaTree:
    """Gather leading-axis lanes from every alpha leaf."""
    return _alpha_tree_map(
        tree,
        lambda leaf: leaf.index_select(0, indices.to(leaf.device)),
    )


def _alpha_tree_repeat_lanes(tree: AlphaTree, repeats: int) -> AlphaTree:
    """Repeat each leading-axis lane consecutively in every alpha leaf."""
    if repeats < 1:
        raise ValueError(f"alpha lane repeats must be positive, got {repeats}")
    return _alpha_tree_map(
        tree,
        lambda leaf: leaf.repeat_interleave(repeats, dim=0),
    )


def _alpha_tree_zero_lanes(tree: AlphaTree, lanes: int) -> AlphaTree:
    """Build an all-zero tree with ``lanes`` and matching trailing shapes."""
    return _alpha_tree_map(
        tree,
        lambda leaf: torch.zeros(
            (lanes, *leaf.shape[1:]), dtype=leaf.dtype, device=leaf.device
        ),
    )


def _alpha_concat_views(
    left: torch.Tensor, right: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align shared/per-row ReLU state without allocating row copies."""
    if left.dim() == 2 and right.dim() == 3 and left.shape[-1] == right.shape[-1]:
        left = left.unsqueeze(1).expand(-1, right.shape[1], -1)
    elif right.dim() == 2 and left.dim() == 3 and left.shape[-1] == right.shape[-1]:
        right = right.unsqueeze(1).expand(-1, left.shape[1], -1)
    if left.dim() == right.dim() == 3 and left.shape[-1] == right.shape[-1]:
        if left.shape[1] == 1:
            left = left.expand(-1, right.shape[1], -1)
        elif right.shape[1] == 1:
            right = right.expand(-1, left.shape[1], -1)
    return left, right


def _alpha_tree_concat_lanes(
    left: AlphaTree,
    right: AlphaTree,
    n_left: int,
    n_right: int,
) -> AlphaTree:
    """Concatenate lanes recursively, zero-padding a missing subtree."""
    if left is None and right is None:
        return None
    if left is None:
        assert right is not None
        left = _alpha_tree_zero_lanes(right, n_left)
    if right is None:
        right = _alpha_tree_zero_lanes(left, n_right)
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        left, right = _alpha_concat_views(left, right)
        if left.shape[1:] != right.shape[1:]:
            raise ValueError(
                "alpha leaf trailing shape mismatch: "
                f"left {tuple(left.shape[1:])} vs right {tuple(right.shape[1:])}"
            )
        return torch.cat(
            [left, right.to(device=left.device, dtype=left.dtype)], dim=0
        )
    if isinstance(left, dict) and isinstance(right, dict):
        keys = list(left)
        keys.extend(key for key in right if key not in left)
        return {
            key: _alpha_tree_concat_lanes(
                left.get(key), right.get(key), n_left, n_right
            )
            for key in keys
        }
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        if len(left) != len(right):
            raise ValueError(
                f"alpha sequence length mismatch: left {len(left)} vs right {len(right)}"
            )
        values = [
            _alpha_tree_concat_lanes(lval, rval, n_left, n_right)
            for lval, rval in zip(left, right)
        ]
        return tuple(values) if isinstance(left, tuple) else values
    raise TypeError(
        "alpha pytree structure mismatch: "
        f"left {type(left)!r} vs right {type(right)!r}"
    )


def _alpha_spec_row_count(state: Optional[AlphaState]) -> int:
    """Return ``M`` from top-level ReLU tensors, ignoring attention pytrees."""
    if not state:
        return 1
    counts = {
        int(tree.shape[1])
        for tree in state.values()
        if isinstance(tree, torch.Tensor) and tree.dim() >= 3
    }
    if len(counts) > 1:
        raise ValueError(f"inconsistent alpha spec-row counts: {sorted(counts)}")
    return next(iter(counts), 1)


def _alpha_tree_select_spec_rows(
    state: Optional[AlphaState], keep_rows: torch.Tensor
) -> Optional[AlphaState]:
    """Select ReLU specification rows; lane-only omega leaves stay unchanged."""
    if state is None:
        return None
    return cast(
        AlphaState,
        _alpha_tree_map(
            state,
            lambda leaf: (
                leaf.index_select(1, keep_rows.to(leaf.device))
                if leaf.dim() >= 3
                else leaf
            ),
        ),
    )


def _alpha_relu_forward_view(state: Optional[AlphaState]) -> Dict[int, torch.Tensor]:
    """Return only top-level ReLU alpha tensors in forward-pass shape."""
    if not state:
        return {}
    return {
        lid: tree[:, 0, :] if tree.dim() == 3 else tree
        for lid, tree in state.items()
        if isinstance(tree, torch.Tensor)
    }


def _phase_slope_replay_alpha(
    alpha: AlphaTree,
    split_signs: torch.Tensor,
    phase_slope_signs: torch.Tensor,
) -> AlphaTree:
    """Use the source phase slope only where replay dropped a ReLU literal."""
    if not isinstance(alpha, torch.Tensor):
        raise TypeError("phase-slope ReLU replay requires a tensor alpha state")
    current = split_signs.to(device=alpha.device)
    source = phase_slope_signs.to(device=alpha.device)
    # A memory-capped certificate shares its lower slope across spec rows.
    # Expand before applying row-specific dropped-literal phase overrides.
    if (
        alpha.dim() == 2
        and current.dim() == 3
        and alpha.shape == (current.shape[0], current.shape[2])
    ):
        alpha = alpha.unsqueeze(1).expand_as(current)
    if current.shape != alpha.shape or source.shape != alpha.shape:
        raise ValueError(
            "phase-slope replay shape mismatch: "
            f"alpha={tuple(alpha.shape)}, split={tuple(current.shape)}, "
            f"source={tuple(source.shape)}"
        )
    dropped = source.ne(0) & current.eq(0)
    return torch.where(dropped, source.gt(0).to(dtype=alpha.dtype), alpha)


def _phase_slope_replay_relu(
    nu: torch.Tensor,
    bounds: Bounds,
    m_specs: int,
    split_signs: torch.Tensor,
    phase_slope_signs: torch.Tensor,
    pred_nu: torch.Tensor,
    contrib: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace dropped ReLU chord terms with the source phase-slope strip."""
    bm = nu.shape[0]
    if bm % m_specs:
        raise ValueError(f"phase-slope replay batch {bm} is not divisible by M={m_specs}")
    lanes = bm // m_specs
    values = nu.flatten(start_dim=1)
    lower = bounds.lb.flatten(start_dim=1)
    upper = bounds.ub.flatten(start_dim=1)
    width = min(values.shape[-1], lower.shape[-1])
    values = values[..., :width].view(lanes, m_specs, width)
    lower = lower[..., :width].unsqueeze(1)
    upper = upper[..., :width].unsqueeze(1)
    current = split_signs.to(device=values.device)[..., :width]
    source = phase_slope_signs.to(device=values.device)[..., :width]
    if current.shape != values.shape or source.shape != values.shape:
        raise ValueError(
            "phase-slope replay sign shape mismatch: "
            f"nu={tuple(values.shape)}, split={tuple(current.shape)}, "
            f"source={tuple(source.shape)}"
        )
    dropped = source.ne(0) & current.eq(0)
    if not bool(dropped.any()):
        return pred_nu, contrib

    denominator = (upper - lower).clamp(min=1e-12)
    chord_slope = upper / denominator
    chord_intercept = -chord_slope * lower
    phase_slope = source.gt(0).to(dtype=values.dtype)
    phase_intercept = torch.where(source.gt(0), -lower, upper)
    negative = values < 0
    old_contrib = torch.where(negative, values * chord_intercept, 0.0)
    new_contrib = torch.where(negative, values * phase_intercept, 0.0)

    corrected_nu = pred_nu.flatten(start_dim=1)[..., :width].view_as(values).clone()
    corrected_nu[dropped] = (phase_slope * values)[dropped]
    corrected_contrib = contrib + torch.where(
        dropped, new_contrib - old_contrib, 0.0
    ).sum(dim=-1).reshape(bm)
    return corrected_nu.reshape(bm, width), corrected_contrib


def _max_with_optional(value: torch.Tensor, other: Optional[torch.Tensor]) -> torch.Tensor:
    """Elementwise maximum that treats a missing second operand as no-op."""
    if other is None:
        return value
    return torch.maximum(value, other.to(device=value.device, dtype=value.dtype))


# ALL-rows kinds whose SAFE set is open: ``TOP1_ROBUST`` requires ``z_t > z_j``
# and ``MARGIN_ROBUST`` requires ``z_t - z_j > m`` (a tie already violates;
# ``OutputSpec.CLOSED_KINDS`` lists them as kinds with a closed UNSAFE set).
_OPEN_SAFE_SET_KINDS = frozenset({OutKind.TOP1_ROBUST, OutKind.MARGIN_ROBUST})


def unproven_spec_rows(
    slack: torch.Tensor, reference: torch.Tensor, out_kind: str
) -> torch.Tensor:
    """ALL-rows spec kinds: rows whose ``slack`` does not prove them.

    ``slack`` is ``lower bound - threshold`` of a row in certification form and
    ``reference`` the bound it was derived from (the row margins, broadcastable
    to ``slack``). A NaN / inf slack is never proven, so a lane with such a row
    is not certified. The boundary depends on the kind:

    - ``LINEAR_LE`` / ``RANGE`` have a CLOSED safe set (``c y <= d``,
      ``lb <= y <= ub``): ``slack == 0`` proves the row, hard zero. A band here
      would only turn boundary-touching safe properties into UNKNOWN.
    - ``TOP1_ROBUST`` / ``MARGIN_ROBUST`` have an OPEN safe set (``z_t > z_j``,
      ``z_t - z_j > m``; an exact tie is a violation): a row is proven only
      when it clears the ``escaping_spec_rows`` band, exactly like an
      UNSAFE_LINEAR escape. With a hard zero an exact tie (two identical logit
      rows: true slack exactly ``0``) certified, and rounding of a few ulp
      (``certification_tolerance``) could certify a violated property.

    Raises:
        ValueError: for the EXISTS-row kind ``UNSAFE_LINEAR`` (decided by
            ``escaping_spec_rows``, whose lane rule is ``.any()``).
    """
    if out_kind == OutKind.UNSAFE_LINEAR:
        raise ValueError(
            "unproven_spec_rows: UNSAFE_LINEAR is an EXISTS-row kind; use "
            "escaping_spec_rows"
        )
    if out_kind in _OPEN_SAFE_SET_KINDS:
        return ~escaping_spec_rows(slack, reference)
    return (slack < 0) | ~torch.isfinite(slack)


# Absolute floor of the certification band (float64 accumulation noise of a
# deep dual / forward chain, ~1e-12 observed; see certification_tolerance).
_CERT_EPS_FLOOR = 1e-11


def certification_tolerance(reference: torch.Tensor) -> torch.Tensor:
    """Band a certified bound must clear before a strict comparison counts as proof.

    ``max(100 ulp, 1e-11) * max(|reference|, 1)`` in the dtype of
    ``reference`` (the bound whose rounding is being absorbed, e.g. the row
    margin the slack was derived from).

    No bound in ACT is computed with directed rounding: the forward linear
    track is composed in centre/radius form (``_fwd_dense_lanes`` /
    ``_fwd_conv2d``: two accumulations that cancel and can land a few ulp on
    the unsound side of the exact ``W+ A_ub + W- A_lb``), the concretization
    einsums and the backward dual pass round to nearest as well, and the dual
    bound accumulates ~1e-12 over deep float64 conv chains (a 100-ulp-only band
    was seen to false-CERTIFY a netfactory draw in CI with a concrete
    counterexample). A boundary-touching property (true slack exactly 0, e.g.
    sat_relu ``Y_0 >= 1`` with ``max Y_0 == 1``) is otherwise certified by
    that rounding alone. 100 ulp is the per-neuron 'auto' bounds tolerance of
    ``act/pipeline/cli.py`` (pairwise-reduction drift ~log2(n)*eps); float32
    (100 ulp = 1.2e-5) is above the floor, float64 (2.2e-14) takes the floor.

    Remaining assumption: the accumulated rounding of a bound stays below the
    band, i.e. intermediate partial sums are within ~1e5x (float64) / ~100x
    (float32) of ``max(|reference|, 1)``. Larger internal magnitudes need a
    rigorous (directed-rounding or error-bound-carrying) propagation, which
    torch does not provide.
    """
    eps = max(100.0 * torch.finfo(reference.dtype).eps, _CERT_EPS_FLOOR)
    return eps * reference.abs().clamp(min=1.0)


def escaping_spec_rows(slack: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """EXISTS-row kind (UNSAFE_LINEAR): rows proven to escape.

    The unsafe polytope ``C y <= d`` is closed, so a row escapes only when its
    lower bound is STRICTLY above the threshold. A finite ``slack`` proves
    that only once it clears ``certification_tolerance(reference)``, where
    ``reference`` is the bound the slack was derived from (the row margins;
    broadcastable to ``slack``); a slack inside the band is rounding noise,
    not proof. Every UNSAFE_LINEAR certification path (``solve_spec_batch``,
    ``evaluate_spec``, CLIMB replay, the interval path of ``verify_once``)
    decides through this predicate.
    """
    tol = certification_tolerance(reference).to(device=slack.device, dtype=slack.dtype)
    return (slack > tol) & torch.isfinite(slack)


def _clone_alpha_tree(tree: AlphaTree) -> AlphaTree:
    """Detach-clone every leaf of an alpha pytree, preserving its structure."""
    if tree is None:
        return None
    if isinstance(tree, torch.Tensor):
        return tree.detach().clone()
    if isinstance(tree, dict):
        return {key: _clone_alpha_tree(value) for key, value in tree.items()}
    if isinstance(tree, (list, tuple)):
        return type(tree)(_clone_alpha_tree(value) for value in tree)
    raise TypeError(f"unsupported alpha pytree node: {type(tree)!r}")


# Floor on the L2 witness denominator so a zero-coefficient block yields a
# center witness instead of a divide-by-zero.
_DUAL_NORM_EPS = 1e-12


class DualSolver(Solver):
    """Dual (linear-relaxation) certified bounds solver. Strict [B, *shape] API."""

    # Kinds whose forward state in ``bounds_dict`` is the PRE-activation box
    # (``post_activation=False``); bilinear kernels reading an operand from
    # such a layer need the post-activation box instead.
    _PRE_ACTIVATION_KINDS = {
        LayerKind.RELU.value, LayerKind.LRELU.value, LayerKind.SIGMOID.value,
        LayerKind.TANH.value, LayerKind.ERF.value, LayerKind.SQRT.value,
        LayerKind.SIN.value, LayerKind.COS.value, LayerKind.QUANTIZE.value,
        LayerKind.GELU.value, LayerKind.SOFTMAX.value, LayerKind.LAYERNORM.value,
    }
    _BILINEAR_KINDS = {LayerKind.MATMUL.value, LayerKind.MUL.value}

    _AFFINE_CONTRIB_KINDS = {
        LayerKind.DENSE.value,
        LayerKind.CONV2D.value,
        LayerKind.BIAS.value,
        LayerKind.BN.value,
        LayerKind.ADD.value,
    }

    def __init__(self, n_iters: int = 0):
        # DualTF is a backward-handler registry (not a TransferFunction);
        # instantiate it internally so DualSolver is self-contained and callers
        # need no knowledge of it.
        from act.back_end.dual_tf.dual_tf import DualTF
        self.tf = DualTF()
        self.n_iters = n_iters
        self._last_bounds: Optional[Bounds] = None
        self.last_forward_bounds: Optional[Dict[int, Bounds]] = None
        self.last_alpha_iterations: int = 0
        self.last_alpha_stop_reason: Optional[str] = None
        self.last_branching_state_skipped: bool = False

    def capabilities(self) -> SolverCaps:
        return SolverCaps(supports_gpu=True, supports_csp=False, supports_dual=True)

    def solve_spec_batch(
        self,
        net: Net,
        bounds_dict: Dict[int, Bounds],
        out_spec: OutputSpec,
        *,
        optimize: bool,
        dual_config: DualConfig,
        input_shape: tuple[int, ...],
        keep_rows: Optional[torch.Tensor] = None,
        split_signs: Optional[Dict[int, torch.Tensor]] = None,
        eta: Optional[Dict[int, torch.Tensor]] = None,
        incremental_alphas: Optional[AlphaState] = None,
        incremental_etas: Optional[Dict[int, torch.Tensor]] = None,
        optimize_alpha: bool = True,
        refresh_forward: bool = True,
        return_nu: bool = False,
        reuse_bounds_for_branching: bool = False,
        forward_frame: Optional[ForwardFrame] = None,
        n_iters: Optional[int] = None,
        time_cap: Optional[float] = None,
        nu_time_cap: Optional[float] = None,
    ) -> DualBatchResult:
        """Solve and decode one prepared K-lane output-spec batch.

        Bound construction and refinement are caller policy. This method owns
        output-row encoding, dual optimization, lane status/witness decoding,
        and the optional converged-state nu pass used by neuron branching.
        Lane statuses follow the intersected bound ``max(dual, forward)``;
        ``dual_row_slack`` / ``forward_only_certified`` expose which lanes the
        backward certificate alone would certify, so CLIMB only learns from
        those.

        ``n_iters`` overrides ``dual_config.n_iters`` for this batch (BaB child
        batches); ``time_cap`` is the caller's per-call cap in seconds on the
        alpha/eta loop, combined with ``dual_config.max_time``. The loop also
        honours ``dual_config.stagnation_patience`` and, when
        ``dual_config.stop_when_verified`` is set, stops once every lane's
        keep-best bound certifies under this method's own lane rule.
        ``nu_time_cap`` (seconds from this call's start, None = never) skips
        the extra branching-state pass (a full forward pass plus a backward
        pass, heuristic-only) once it has elapsed: the lanes then return
        ``bounds_dict=None`` / ``nu_per_layer=None`` and the caller must not
        branch them. The bound itself is unaffected;
        ``last_branching_state_skipped`` records the skip.
        """
        call_started = _monotonic() if nu_time_cap is not None else 0.0
        self.last_branching_state_skipped = False
        sample_bounds = next(iter(bounds_dict.values()))
        device = sample_bounds.lb.device
        dtype = sample_bounds.lb.dtype
        if sample_bounds.lb.dim() < 2:
            raise ValueError(
                "DualSolver.solve_spec_batch: bounds_dict entries must be batched "
                f"[K, *shape]; got dim={sample_bounds.lb.dim()}"
            )
        k_actual = int(sample_bounds.lb.shape[0])

        assert_layers = [
            layer
            for layer in net.layers
            if (layer.kind.upper() if isinstance(layer.kind, str) else layer.kind)
            == LayerKind.ASSERT.value
        ]
        if not assert_layers:
            raise ValueError("DualSolver.solve_spec_batch: net has no ASSERT layer")
        assert_layer = assert_layers[0]
        assert_preds = net.preds.get(assert_layer.id, [])
        if len(assert_preds) != 1:
            raise ValueError(
                f"ASSERT layer {assert_layer.id} must have exactly 1 predecessor, "
                f"got {len(assert_preds)}"
            )
        output_bounds = bounds_dict[assert_preds[0]]
        n_out = int(output_bounds.lb.flatten(start_dim=1).shape[-1])
        encoded_spec = out_spec.encode_linear(
            B=k_actual, n_out=n_out, device=device, dtype=dtype
        )
        m_specs = int(encoded_spec["M"])

        if out_spec.kind == OutKind.UNSAFE_LINEAR:
            c_rows = cast(torch.Tensor, encoded_spec["C"]).contiguous()
            thresholds = cast(torch.Tensor, encoded_spec["thresholds"]).contiguous()
        else:
            c_rows = -cast(torch.Tensor, encoded_spec["C"]).contiguous()
            thresholds = -cast(torch.Tensor, encoded_spec["thresholds"]).contiguous()
            if keep_rows is not None:
                idx = keep_rows.to(device=device, dtype=torch.long)
                c_rows = (
                    c_rows.view(k_actual, m_specs, n_out)
                    .index_select(1, idx)
                    .reshape(k_actual * int(idx.numel()), n_out)
                    .contiguous()
                )
                thresholds = thresholds.index_select(1, idx).contiguous()
                m_specs = int(idx.numel())
        active_mask = torch.ones(
            k_actual, m_specs, dtype=torch.bool, device=device
        )

        def _lanes_certified(row_bounds: torch.Tensor) -> torch.Tensor:
            row_margins = row_bounds.view(k_actual, m_specs)
            row_slack = row_margins - thresholds
            if out_spec.kind == OutKind.UNSAFE_LINEAR:
                return (
                    escaping_spec_rows(row_slack, row_margins) & active_mask
                ).any(dim=-1)
            return ~(
                unproven_spec_rows(row_slack, row_margins, out_spec.kind) & active_mask
            ).any(dim=-1)

        def _all_lanes_certified(row_bounds: torch.Tensor) -> bool:
            return bool(_lanes_certified(row_bounds).all().item())

        stop_criterion = _all_lanes_certified if dual_config.stop_when_verified else None

        compute_certified_bound = cast(Any, self.compute_certified_bound)
        if optimize:
            dual_result = compute_certified_bound(
                net,
                bounds_dict,
                c_rows,
                M=m_specs,
                optimize=True,
                optimize_alpha=optimize_alpha,
                refresh_forward=refresh_forward,
                forward_lin_max_perturbed=dual_config.forward_lin_max_perturbed,
                n_iters=dual_config.n_iters if n_iters is None else n_iters,
                stagnation_patience=dual_config.stagnation_patience,
                stagnation_tol=dual_config.stagnation_tol,
                stop_criterion=stop_criterion,
                max_time=dual_config.max_time,
                time_cap=time_cap,
                lr_alpha=dual_config.lr_alpha,
                lr_beta=dual_config.lr_beta,
                lr_decay=dual_config.lr_decay,
                eta=eta,
                incremental_alphas=incremental_alphas,
                incremental_etas=incremental_etas,
                split_signs=split_signs,
                return_optimized=True,
                return_sce=True,
                per_class_alpha=dual_config.per_class_alpha,
                forward_frame=forward_frame,
                **({"return_nu_per_layer": True} if return_nu else {}),
            )
        else:
            dual_result = compute_certified_bound(
                net,
                bounds_dict,
                c_rows,
                M=m_specs,
                return_sce=True,
                forward_frame=forward_frame,
                **({"return_nu_per_layer": True} if return_nu else {}),
            )

        margins = dual_result.margins.view(k_actual, m_specs)
        dual_margins = (
            dual_result.dual_margins.view(k_actual, m_specs)
            if dual_result.dual_margins is not None
            else margins
        )
        sce = cast(Optional[torch.Tensor], dual_result.sce)
        slack = margins - thresholds
        dual_slack = dual_margins - thresholds
        if out_spec.kind == OutKind.UNSAFE_LINEAR:
            certified = (escaping_spec_rows(slack, margins) & active_mask).any(dim=-1)
            dual_certified = (
                escaping_spec_rows(dual_slack, dual_margins) & active_mask
            ).any(dim=-1)
            candidate_rows = torch.zeros(
                k_actual, dtype=torch.long, device=device
            )
        else:
            violations = unproven_spec_rows(slack, margins, out_spec.kind) & active_mask
            certified = ~violations.any(dim=-1)
            dual_certified = ~(
                unproven_spec_rows(dual_slack, dual_margins, out_spec.kind) & active_mask
            ).any(dim=-1)
            candidate_rows = torch.where(
                violations.any(dim=1),
                violations.to(torch.int64).argmax(dim=1),
                torch.zeros(k_actual, dtype=torch.long, device=device),
            )
        forward_only_certified = certified & ~dual_certified

        statuses = tuple(
            SolveStatus.UNSAT if bool(is_certified.item()) else SolveStatus.SAT
            for is_certified in certified
        )
        nvars = (
            max((max(layer.out_vars) for layer in net.layers if layer.out_vars), default=-1)
            + 1
        )
        x_candidate = torch.zeros(k_actual, nvars, device=device, dtype=dtype)
        input_layers = [
            layer
            for layer in net.layers
            if (layer.kind.upper() if isinstance(layer.kind, str) else layer.kind)
            == LayerKind.INPUT.value
        ]
        if len(input_layers) != 1:
            raise ValueError(
                f"Expected exactly one INPUT layer, found {len(input_layers)}."
            )
        input_ids_list = list(input_layers[0].out_vars)
        input_ids = torch.tensor(input_ids_list, device=device, dtype=torch.long)
        if sce is not None:
            sce_flat = sce.flatten(start_dim=1).to(device=device)
            row_offsets = (
                torch.arange(k_actual, device=device) * m_specs
                + candidate_rows.to(device=device)
            )
            chosen_sce = sce_flat.index_select(0, row_offsets)
            x_candidate[:, input_ids] = chosen_sce.to(device=device, dtype=dtype)
        else:
            statuses = tuple(
                SolveStatus.UNSAT if status == SolveStatus.UNSAT else SolveStatus.UNKNOWN
                for status in statuses
            )
        solution = BatchLPSolution(
            statuses=statuses,
            x=x_candidate,
            max_viol=-slack.min(dim=1).values.detach(),
        )

        branch_bounds: Optional[Dict[int, Bounds]] = None
        branch_nu: Optional[Dict[int, torch.Tensor]] = None
        skip_nu_pass = (
            nu_time_cap is not None and _monotonic() - call_started >= nu_time_cap
        )
        if return_nu and reuse_bounds_for_branching:
            branch_bounds = bounds_dict
            branch_nu = dual_result.nu_per_layer
            if branch_nu is None and skip_nu_pass:
                branch_bounds = None
                self.last_branching_state_skipped = True
            elif branch_nu is None:
                nu_pass = self.compute_certified_bound(
                    net,
                    bounds_dict,
                    c_rows,
                    M=m_specs,
                    alpha=dual_result.alpha_state,
                    eta=dual_result.eta_state if split_signs is not None else None,
                    split_signs=split_signs,
                    return_nu_per_layer=True,
                )
                branch_nu = nu_pass.nu_per_layer
        elif return_nu and skip_nu_pass:
            self.last_branching_state_skipped = True
        elif return_nu:
            branch_bounds, branch_nu = self.recompute_bounds_and_nu(
                net,
                bounds_dict,
                c_rows,
                m_specs,
                alpha_state=dual_result.alpha_state,
                eta_state=dual_result.eta_state if split_signs is not None else None,
                split_signs=split_signs,
                per_class_alpha=dual_config.per_class_alpha,
            )

        witness_input = (
            x_candidate[:, input_ids].reshape(k_actual, *input_shape).detach()
            if sce is not None
            else None
        )
        lower_bounds = -solution.max_viol
        retained_alpha_state = _retain_incremental_alpha_state(
            dual_result.alpha_state, k_actual,
        )
        relu_alphas = [
            value for value in (dual_result.alpha_state or {}).values()
            if isinstance(value, torch.Tensor)
        ]
        alpha_mode = (
            ("shared" if any(value.dim() == 2 for value in relu_alphas) else "per_row")
            if relu_alphas else None
        )
        return DualBatchResult(
            solution=solution,
            margins=margins,
            lower_bounds=lower_bounds,
            dual_row_slack=dual_slack.detach(),
            forward_only_certified=forward_only_certified.detach(),
            bounds_dict=branch_bounds,
            nu_per_layer=branch_nu,
            alpha_state=retained_alpha_state,
            alpha_mode=alpha_mode,
            eta_state=dual_result.eta_state,
            witness_input=witness_input,
            row_slack=slack.detach(),
            c_rows=c_rows.detach(),
            thresholds=thresholds.detach(),
            m_specs=m_specs,
            reference_bounds=bounds_dict,
        )

    def compute_certified_bound(
        self, net: Net, bounds_dict: Dict[int, Bounds],
        c: torch.Tensor, M: int = 1,
        return_sce: bool = False,
        enable_grad: bool = False,
        alpha: Optional[AlphaState] = None,
        eta: Optional[Dict[int, torch.Tensor]] = None,
        split_signs: Optional[Union[Dict[int, torch.Tensor], List[Dict[int, torch.Tensor]]]] = None,
        optimize: bool = False,
        n_iters: int = 50,
        lr_alpha: float = 0.1,
        lr_beta: float = 0.1,
        lr_decay: float = 0.98,
        incremental_alphas: Optional[AlphaState] = None,
        incremental_etas: Optional[Dict[int, torch.Tensor]] = None,
        return_optimized: bool = False,
        per_class_alpha: bool = True,
        return_nu_per_layer: bool = False,
        optimize_alpha: bool = True,
        refresh_forward: bool = True,
        start_lid: Optional[int] = None,
        local_phase_clamp: bool = False,
        phase_slope_signs: Optional[Dict[int, torch.Tensor]] = None,
        forward_lin_max_perturbed: Optional[int] = None,
        forward_frame: Optional[ForwardFrame] = None,
        stagnation_patience: int = 0,
        stagnation_tol: Optional[float] = None,
        stop_criterion: Optional[Callable[[torch.Tensor], bool]] = None,
        max_time: float = 0.0,
        time_cap: Optional[float] = None,
    ) -> DualResult:
        """Batched certified lower bound on c^T @ output (DAG-aware).

        Implements ``Solver.compute_certified_bound``; see base for the
        full contract. DualSolver realises this via reverse-topological
        backward propagation of a per-layer accumulator:
          nu_accum[lid] = sum over all successors s of ν routed by s's handler to lid.

        Each handler returns per-pred νs; the outer loop distributes them to preds.

        Unknown layer kind raises ValueError (no silent identity fallback for soundness).

        Lazy M-broadcast: ``c`` has shape ``[B*M, num_classes]`` packed
        sample-major (row ``b*M+j`` = sample b's j-th spec row), but
        ``bounds_dict`` entries stay at ``[B, *shape]``. Activation handlers
        (RELU/SIGMOID/TANH) view nu as ``[B, M, n]`` and broadcast bounds
        ``[B, 1, n]`` against it — mathematically equivalent to the legacy
        M-expanded path, with M× lower bounds memory.

        Caveats:
            Uses GLOBAL intermediate bounds — bounds_dict stays at [B, *shape] across
            BaB lanes within the same sample (lazy-M-broadcast design). Per-lane
            intermediate-bound tightening (a stricter dual variant) is OUT OF SCOPE
            for this solver; bounds may be looser than a per-lane refinement would
            give, but the dual lower bound remains SOUND.

        When ``optimize=True``, runs joint Adam optimization over ReLU α/η via
        ``_optimize_alpha_eta``. If ``return_optimized=True``, also returns the optimized
        α state for incremental-starting. When ``optimize=False``, executes a single-pass
        backward; with ``alpha=None`` it uses the default fixed-slope relaxation.

        eta: Per-layer η multiplier — Lagrange (KKT) multiplier for branch-split
            constraints on activation pre-activations. Keyed by activation layer id
            (RELU, LRELU, SIGMOID, TANH, GELU). η ≥ 0 invariant (enforced by clamp).
        split_signs: Per-layer split direction. {-1: inactive, +1: active, 0: unsplit}.
            Same key set as eta.
        local_phase_clamp: For fixed-parameter replay, shallow-copy and clamp only
            the current ReLU handler's bounds. Reference bounds remain immutable;
            optimization is rejected because it requires globally hardened bounds.
        phase_slope_signs: Original replay literals. Where a current split sign
            is zero but its original sign is nonzero, use alpha=1 for the active
            phase and alpha=0 for the inactive phase. Other alpha entries stay
            frozen at their stored values.
        η is applied to the TRUE pre-activation variable (immediately AFTER the
        activation handler in the reverse-topological backward loop):
        nu_pre = slope · nu_post − η · sign, so the multiplier acts on the
        affine pre-activation unscaled. Applying it before the handler would
        scale η by the relaxation slope, forcing the effective multiplier to 0
        on inactive-split (slope = 0) neurons and discarding the z ≤ 0
        constraint's input-region information. Sound for any η ≥ 0.

        forward_lin_max_perturbed: ``None`` resolves to the DualConfig default
            at call time rather than being frozen at import time.
        forward_frame: optional explicit forward frame of the spec layer. On
            the spec-layer objective (``start_lid is None``) the returned
            ``margins`` are ``max(dual, forward)`` where ``forward`` is the
            frame concretization when given, intersected with the forward box
            of the spec layer from ``bounds_dict``; both are valid lower bounds
            of the same rows, so the maximum is sound. ``dual_margins`` keeps
            the backward bound alone.
        stagnation_patience / stagnation_tol / stop_criterion / max_time /
            time_cap: early-stop and time-cap settings of the alpha/eta loop,
            forwarded to ``_optimize_alpha_eta`` when ``optimize=True`` (see
            there); ignored by the single-pass backward.
        """
        if forward_lin_max_perturbed is None:
            forward_lin_max_perturbed = DualConfig().forward_lin_max_perturbed
        if isinstance(split_signs, list):
            # KFSB path: accept K split hypotheses and return stacked margins
            # [K, N, ...], evaluating each hypothesis through the single-hypothesis
            # backward pass (no leading-K vectorization).
            if not split_signs:
                raise ValueError("split_signs list cannot be empty")
            if optimize:
                raise ValueError("list-form split_signs is only supported with optimize=False")
            if return_optimized:
                raise ValueError("return_optimized requires optimize=True and single split_signs")
            stacked_split_signs = self._stack_split_sign_hypotheses(split_signs)
            normalized_split_signs = self._unstack_split_sign_hypotheses(stacked_split_signs)
            margins = []
            sce_values = []
            for hypo in normalized_split_signs:
                result = self.compute_certified_bound(
                    net,
                    bounds_dict,
                    c,
                    M=M,
                    return_sce=return_sce,
                    enable_grad=enable_grad,
                    alpha=alpha,
                    eta=eta,
                    split_signs=hypo,
                    optimize=False,
                    n_iters=n_iters,
                    lr_alpha=lr_alpha,
                    lr_beta=lr_beta,
                    lr_decay=lr_decay,
                    incremental_alphas=incremental_alphas,
                    incremental_etas=incremental_etas,
                    return_optimized=False,
                    per_class_alpha=per_class_alpha,
                    return_nu_per_layer=False,
                    local_phase_clamp=local_phase_clamp,
                    phase_slope_signs=phase_slope_signs,
                    forward_lin_max_perturbed=forward_lin_max_perturbed,
                    forward_frame=forward_frame,
                )
                margins.append(result.margins)
                if return_sce:
                    sce_values.append(result.sce)
            stacked_sce = None
            if return_sce and sce_values and all(sce is not None for sce in sce_values):
                stacked_sce = torch.stack(cast(List[torch.Tensor], sce_values), dim=0)
            return DualResult(margins=torch.stack(margins, dim=0), sce=stacked_sce)

        if optimize and local_phase_clamp:
            raise ValueError("local_phase_clamp is only valid for fixed-parameter replay")
        # Branchers can emit a singleton spec axis when their warm alpha is
        # shared (or absent). A phase literal constrains the domain, not one
        # objective: repeat it for every actual row before flattening B*M.
        def expand_shared_rows(
            state: Optional[Dict[int, torch.Tensor]],
        ) -> Optional[Dict[int, torch.Tensor]]:
            if state is None or M == 1:
                return state
            return {
                lid: value.expand(-1, M, -1)
                if value.dim() == 3 and value.shape[1] == 1 else value
                for lid, value in state.items()
            }

        split_signs = expand_shared_rows(split_signs)
        phase_slope_signs = expand_shared_rows(phase_slope_signs)
        eta = expand_shared_rows(eta)
        incremental_etas = expand_shared_rows(incremental_etas)
        if not local_phase_clamp:
            bounds_dict = self._harden_split_bounds(bounds_dict, split_signs)

        if optimize:
            bound, sce, alpha_state, eta_state, dual_bound, fwd_bound = self._optimize_alpha_eta(
                net,
                bounds_dict,
                c,
                M=M,
                n_iters=n_iters,
                lr_alpha=lr_alpha,
                lr_beta=lr_beta,
                lr_decay=lr_decay,
                incremental_alphas=incremental_alphas,
                incremental_etas=incremental_etas,
                split_signs=split_signs,
                return_sce=return_sce,
                per_class_alpha=per_class_alpha,
                optimize_alpha=optimize_alpha,
                refresh_forward=refresh_forward,
                start_lid=start_lid,
                forward_lin_max_perturbed=forward_lin_max_perturbed,
                forward_frame=forward_frame,
                stagnation_patience=stagnation_patience,
                stagnation_tol=stagnation_tol,
                stop_criterion=stop_criterion,
                max_time=max_time,
                time_cap=time_cap,
            )
            if return_optimized:
                return DualResult(
                    margins=bound,
                    sce=sce if return_sce else None,
                    alpha_state=alpha_state if alpha_state else None,
                    eta_state=eta_state if eta_state else None,
                    dual_margins=dual_bound,
                    forward_margins=fwd_bound,
                )
            return DualResult(
                margins=bound,
                sce=sce if return_sce else None,
                dual_margins=dual_bound,
                forward_margins=fwd_bound,
            )

        if c.dim() != 2:
            raise ValueError(
                f"c must be 2-D [B*M, num_classes], got shape {tuple(c.shape)}. "
                "Use c.unsqueeze(0) for single instance.")
        if M < 1:
            raise ValueError(f"M must be >= 1, got {M}")
        if c.shape[0] % M != 0:
            raise ValueError(
                f"c batch dim {c.shape[0]} not divisible by M={M}; "
                f"expected c.shape[0] == B*M for some integer B"
            )
        with torch.set_grad_enabled(enable_grad):
            assert len(bounds_dict) > 0, "bounds_dict cannot be empty"
            device, dtype = get_default_device(), get_default_dtype()
            if c.dtype != dtype or c.device != device:
                c = c.to(device=device, dtype=dtype)
            B = c.shape[0]

            if start_lid is not None:
                # Interior start: c is a linear functional on layer
                # ``start_lid``'s output; only its ancestors are visited by
                # the backward loop (non-ancestors never enter nu_accum).
                # Used by refine_intermediate_bounds.
                output_lid = start_lid
            else:
                assert_layer = None
                for layer in net.layers:
                    k = layer.kind.upper() if isinstance(layer.kind, str) else layer.kind
                    if k == LayerKind.ASSERT.value:
                        assert_layer = layer
                        break
                if assert_layer is None:
                    raise ValueError("DualSolver.compute_certified_bound: net has no ASSERT layer")

                assert_preds = net.preds.get(assert_layer.id, [])
                if len(assert_preds) != 1:
                    raise ValueError(
                        f"DualSolver.compute_certified_bound: ASSERT layer {assert_layer.id} must have "
                        f"exactly 1 predecessor, got {len(assert_preds)}"
                    )

                output_lid = assert_preds[0]
            nu_accum: Dict[int, torch.Tensor] = {output_lid: c.clone()}
            nu_snapshot: Dict[int, torch.Tensor] = {}
            obj = torch.zeros(B, dtype=c.dtype, device=c.device)

            topo_order = get_topo_order(net, reverse=True)
            registry = self.tf._BACKWARD_REGISTRY

            for lid in topo_order:
                layer = net.by_id[lid]
                k = layer.kind.upper() if isinstance(layer.kind, str) else layer.kind

                if k in (LayerKind.INPUT.value, LayerKind.INPUT_SPEC.value, LayerKind.ASSERT.value):
                    continue

                if lid not in nu_accum:
                    continue

                nu_here = nu_accum.pop(lid)
                handler = registry.get(k)
                if handler is None:
                    raise ValueError(
                        f"DualSolver.compute_certified_bound: unknown layer kind '{k}' at layer {lid}; "
                        f"soundness requires explicit backward handler. "
                        f"Supported kinds: {sorted(registry.keys())}"
                    )

                if return_nu_per_layer and k == LayerKind.RELU.value:
                    nu_snapshot[lid] = nu_here.detach().clone()

                preds = list(net.preds.get(lid, []))
                handler_bounds = bounds_dict
                if (
                    local_phase_clamp
                    and k == LayerKind.RELU.value
                    and split_signs is not None
                    and lid in split_signs
                ):
                    handler_bounds = self._local_phase_bounds(
                        bounds_dict, lid, split_signs[lid]
                    )
                if k in self._BILINEAR_KINDS:
                    handler_bounds = self._bilinear_operand_bounds(net, handler_bounds, preds)
                layer_alpha = alpha.get(lid) if alpha is not None else None
                if (
                    phase_slope_signs is not None
                    and split_signs is not None
                    and lid in phase_slope_signs
                    and lid in split_signs
                ):
                    layer_alpha = _phase_slope_replay_alpha(
                        layer_alpha, split_signs[lid], phase_slope_signs[lid]
                    )
                if layer_alpha is None:
                    pred_nus, contrib = handler(layer, nu_here, handler_bounds, preds, M)
                else:
                    pred_nus, contrib = handler(
                        layer, nu_here, handler_bounds, preds, M, alpha=layer_alpha
                    )

                if (
                    k == LayerKind.RELU.value
                    and phase_slope_signs is not None
                    and split_signs is not None
                    and lid in phase_slope_signs
                    and lid in split_signs
                ):
                    corrected_nu, contrib = _phase_slope_replay_relu(
                        nu_here,
                        bounds_dict[lid],
                        M,
                        split_signs[lid],
                        phase_slope_signs[lid],
                        pred_nus[0],
                        contrib,
                    )
                    pred_nus = [corrected_nu, *pred_nus[1:]]

                if eta is not None and split_signs is not None and lid in eta:
                    # Split Lagrangian on the TRUE pre-activation variable:
                    # nu_pre = D nu_post - eta * sign.
                    # Applying it BEFORE the handler would scale eta by the
                    # relaxation slope D - forcing beta = 0 on inactive-split
                    # (D = 0) neurons, which silently discards the z <= 0
                    # constraint's restriction of the input region (the only
                    # channel carrying it: the y = 0 relaxation says nothing
                    # about x). Sound for any eta >= 0: on the child region
                    # sign * z >= 0, so -eta * sign * z <= 0 only lowers the
                    # minimized Lagrangian below the true child minimum.
                    eta_l = eta[lid].to(device=nu_here.device, dtype=nu_here.dtype)
                    signs_l = split_signs[lid].to(device=nu_here.device, dtype=nu_here.dtype)
                    if eta_l.dim() == 3:
                        eta_l = eta_l.reshape(-1, eta_l.shape[-1])
                    if signs_l.dim() == 3:
                        signs_l = signs_l.reshape(-1, signs_l.shape[-1])
                    pn = pred_nus[0]
                    n_clip = min(pn.shape[-1], eta_l.shape[-1])
                    pn = pn.clone()
                    pn[..., :n_clip] = (
                        pn[..., :n_clip]
                        - eta_l[..., :n_clip] * signs_l[..., :n_clip]
                    )
                    pred_nus = [pn, *pred_nus[1:]]

                if len(pred_nus) != len(preds):
                    raise ValueError(
                        f"handler {k} at layer {lid} returned {len(pred_nus)} pred_nus, "
                        f"expected {len(preds)}"
                    )
                if contrib.shape != (B,):
                    raise ValueError(
                        f"handler {k} at layer {lid} contrib shape {tuple(contrib.shape)}, "
                        f"expected ({B},)"
                    )

                if k in self._AFFINE_CONTRIB_KINDS:
                    contrib = -contrib

                obj = obj + contrib
                for pred_id, pred_nu in zip(preds, pred_nus):
                    if pred_id in nu_accum:
                        nu_accum[pred_id] = nu_accum[pred_id] + pred_nu
                    else:
                        nu_accum[pred_id] = pred_nu.clone()

            input_lid = self._find_input_layer_id(net)
            nu_final = nu_accum.get(input_lid) if input_lid is not None else None
            sce = None
            if input_lid is not None and nu_final is not None:
                input_contrib, sce = self._input_contribution_from_nu(
                    net,
                    input_lid,
                    nu_final,
                    bounds_dict,
                    M=M,
                    return_sce=return_sce,
                    enable_grad=enable_grad,
                )
                obj = obj + input_contrib

            forward_margins = None
            margins = obj
            if start_lid is None:
                forward_margins = self._forward_margin_bound(
                    bounds_dict, output_lid, c, M, forward_frame,
                )
                if forward_margins is not None:
                    margins = torch.maximum(obj, forward_margins)
            return DualResult(
                margins=margins,
                sce=sce if return_sce else None,
                nu_per_layer=nu_snapshot if return_nu_per_layer else None,
                dual_margins=obj,
                forward_margins=forward_margins,
            )

    def _bilinear_operand_bounds(
        self, net: Net, bounds_dict: Dict[int, Bounds], preds: List[int],
    ) -> Dict[int, Bounds]:
        """View of ``bounds_dict`` with post-activation boxes for bilinear operands.

        ``bounds_dict`` stores activation layers pre-activation, but a MATMUL /
        MUL operand read from e.g. a SOFTMAX layer is the activation OUTPUT
        (the probabilities, not the scores). Each such operand is mapped through
        its registered dual forward handler on its own stored box, exactly as
        the forward pass computed it; other entries are shared unchanged.
        """
        from act.back_end.dual_tf.tf_forward import _reset_forward_box

        device, dtype = get_default_device(), get_default_dtype()
        view: Optional[Dict[int, Bounds]] = None
        for pid in preds:
            pred = net.by_id[int(pid)]
            kind = pred.kind.upper() if isinstance(pred.kind, str) else pred.kind
            pre = bounds_dict.get(int(pid))
            if kind not in self._PRE_ACTIVATION_KINDS or pre is None:
                continue
            pre_box = Bounds(pre.lb.flatten(start_dim=1), pre.ub.flatten(start_dim=1))
            lin, frame = _reset_forward_box(pre_box.lb, pre_box.ub, device, dtype)
            forward = self.tf._FORWARD_REGISTRY[kind]
            _, out, _, _ = forward(
                pred, [pre_box], [lin], [frame], list(net.preds.get(int(pid), [])),
                True, device, dtype,
            )
            if view is None:
                view = dict(bounds_dict)
            view[int(pid)] = out
        return bounds_dict if view is None else view

    def _forward_margin_bound(
        self,
        bounds_dict: Dict[int, Bounds],
        output_lid: int,
        c: torch.Tensor,
        M: int,
        forward_frame: Optional[ForwardFrame],
    ) -> Optional[torch.Tensor]:
        """Forward lower bound of the spec rows ``c . y`` on layer ``output_lid``.

        Box part: ``c+ . lb + c- . ub`` on the stored forward box (lazy
        M-broadcast, ``[B, 1, n]`` against ``[B, M, n]``). Frame part: the
        explicit frame concretization when ``forward_frame`` belongs to this
        layer. Returns the tighter of the available parts, ``None`` when neither
        the box nor a matching frame exists.
        """
        BM = c.shape[0]
        B = BM // M
        bound: Optional[torch.Tensor] = None
        out_bounds = bounds_dict.get(output_lid)
        if out_bounds is not None and out_bounds.lb.dim() >= 2:
            lb = out_bounds.lb.flatten(start_dim=1).to(device=c.device, dtype=c.dtype)
            ub = out_bounds.ub.flatten(start_dim=1).to(device=c.device, dtype=c.dtype)
            n = min(lb.shape[-1], c.shape[-1])
            if lb.shape[0] == B:
                c_view = c[..., :n].view(B, M, n)
                lb_bc, ub_bc = lb[..., :n].unsqueeze(1), ub[..., :n].unsqueeze(1)
                bound = (
                    (c_view.clamp(min=0) * lb_bc).sum(dim=-1)
                    + (c_view.clamp(max=0) * ub_bc).sum(dim=-1)
                ).reshape(BM)
        if forward_frame is not None and forward_frame.lid == output_lid:
            frame_bound = forward_frame_row_lower_bounds(forward_frame, c, M).to(
                device=c.device, dtype=c.dtype
            )
            bound = frame_bound if bound is None else torch.maximum(bound, frame_bound)
        return bound

    def _stack_split_sign_hypotheses(
        self,
        split_signs: List[Dict[int, torch.Tensor]],
    ) -> Dict[int, torch.Tensor]:
        layer_ids = sorted({lid for hypo in split_signs for lid in hypo})
        stacked: Dict[int, torch.Tensor] = {}
        for lid in layer_ids:
            template = next(hypo[lid] for hypo in split_signs if lid in hypo)
            entries = [hypo.get(lid, torch.zeros_like(template)) for hypo in split_signs]
            stacked[lid] = torch.stack(entries, dim=0)
        return stacked

    def _unstack_split_sign_hypotheses(
        self,
        stacked_split_signs: Dict[int, torch.Tensor],
    ) -> List[Dict[int, torch.Tensor]]:
        first = next(iter(stacked_split_signs.values()))
        hypotheses: List[Dict[int, torch.Tensor]] = []
        for idx in range(first.shape[0]):
            hypotheses.append({lid: signs[idx] for lid, signs in stacked_split_signs.items()})
        return hypotheses

    def _init_alpha(
        self,
        layer: Layer,
        bounds_dict: Dict[int, Bounds],
        B: int,
        M: int,
        device: torch.device,
        dtype: torch.dtype,
        *,
        per_class_alpha: bool,
        optimize_alpha: bool,
        incremental_alphas: Optional[AlphaState],
        preds: Optional[List[int]] = None,
    ) -> Any:
        """Per-kind dual-alpha allocation returning a pytree of leaves, or None.

        RELU allocates an optimizable lower-envelope slope ``[B, M, n]`` (or
        ``[B, n]`` when ``per_class_alpha`` is off), warm-started from
        ``incremental_alphas`` when present. SOFTMAX and LAYERNORM are
        fixed-slope for now and allocate no alpha (``None``), so only RELU
        contributes optimizer leaves; their backward kernels interpret the
        absent alpha as the fixed relaxation. Each kind owns the shape of its
        own pytree, which the matching backward kernel reads back via
        ``alpha.get(lid)``.
        """
        k = layer.kind.upper() if isinstance(layer.kind, str) else layer.kind
        if k in (LayerKind.ATT_SCORES.value, LayerKind.ATT_MIX.value):
            return self._init_attention_alpha(
                layer, bounds_dict, device, dtype,
                optimize_alpha=optimize_alpha,
                incremental_alphas=incremental_alphas,
            )
        if k == LayerKind.MATMUL.value:
            return self._init_matmul_alpha(
                layer, bounds_dict, device, dtype, list(preds or []),
                optimize_alpha=optimize_alpha,
                incremental_alphas=incremental_alphas,
            )
        if k != LayerKind.RELU.value:
            return None
        b = bounds_dict.get(layer.id)
        if b is None:
            return None
        if incremental_alphas is not None and layer.id in incremental_alphas:
            prior_alpha = cast(torch.Tensor, incremental_alphas[layer.id])
            alpha_init = (
                prior_alpha
                .detach()
                .clone()
                .to(device=device, dtype=dtype)
                .clamp(0.0, 1.0)
            )
        else:
            lb_flat = b.lb.to(device=device, dtype=dtype).flatten(start_dim=1)
            ub_flat = b.ub.to(device=device, dtype=dtype).flatten(start_dim=1)
            n_neurons = lb_flat.shape[-1]
            denom = (ub_flat - lb_flat).clamp(min=1e-12)
            alpha_init_bn = (ub_flat / denom).clamp(0.0, 1.0).detach()
            if per_class_alpha:
                alpha_init = (
                    alpha_init_bn.unsqueeze(1)
                    .expand(B, M, n_neurons)
                    .contiguous()
                )
            else:
                alpha_init = alpha_init_bn.contiguous()
        return torch.nn.Parameter(alpha_init) if optimize_alpha else alpha_init.detach()

    def _init_attention_alpha(
        self,
        layer: Layer,
        bounds_dict: Dict[int, Bounds],
        device: torch.device,
        dtype: torch.dtype,
        *,
        optimize_alpha: bool,
        incremental_alphas: Optional[AlphaState],
    ) -> Any:
        """Allocate the bilinear-attention fusion-slope pytree for one core.

        The pytree is ``{omega_l, omega_u}`` with per-output ``[B, 1]`` slopes,
        warm-started at the rule init derived from the same local input boxes the
        backward kernel reads (so allocator, kernel, keep-best clone and the
        ``[0, 1]`` projection all agree on the shape). Returns ``None`` when the
        predecessor boxes are missing so the kernel falls back to its rule slope.
        """
        from act.back_end.dual_tf.tf_transformer import (
            attention_rule_alpha, _attention_input_boxes,
        )
        boxes = None
        try:
            boxes = _attention_input_boxes(layer, bounds_dict)
        except KeyError:
            # Missing predecessor bounds disable learned attention slopes.
            boxes = None
        if boxes is None:
            return None
        x_l, x_u, y_l, y_u, _scale, _mask = boxes
        x_l = x_l.to(device=device, dtype=dtype)
        x_u = x_u.to(device=device, dtype=dtype)
        y_l = y_l.to(device=device, dtype=dtype)
        y_u = y_u.to(device=device, dtype=dtype)
        if incremental_alphas is not None and layer.id in incremental_alphas:
            prior = cast(Dict[str, torch.Tensor], cast(object, incremental_alphas[layer.id]))
            tree = {
                "omega_l": prior["omega_l"].detach().clone().to(device=device, dtype=dtype).clamp(0.0, 1.0),
                "omega_u": prior["omega_u"].detach().clone().to(device=device, dtype=dtype).clamp(0.0, 1.0),
            }
        else:
            tree = attention_rule_alpha(x_l, x_u, y_l, y_u)
        if optimize_alpha:
            return {key: torch.nn.Parameter(val.detach().clone()) for key, val in tree.items()}
        return {key: val.detach() for key, val in tree.items()}

    def _init_matmul_alpha(
        self,
        layer: Layer,
        bounds_dict: Dict[int, Bounds],
        device: torch.device,
        dtype: torch.dtype,
        preds: List[int],
        *,
        optimize_alpha: bool,
        incremental_alphas: Optional[AlphaState],
    ) -> Any:
        """Allocate the per-element fusion-slope pytree of a batched MATMUL.

        Same contract as :meth:`_init_attention_alpha` with ``[B, G*I*J]``
        leaves (one slope pair per output element) warm-started at the rule
        init of the operand boxes; ``None`` when either operand box is missing.
        """
        from act.back_end.dual_tf.tf_transformer import _matmul_shapes, matmul_rule_alpha

        if len(preds) != 2 or any(pid not in bounds_dict for pid in preds):
            return None
        G, I, K, J = _matmul_shapes(layer.params["x_shape"], layer.params["y_shape"])
        x_box, y_box = bounds_dict[preds[0]], bounds_dict[preds[1]]
        x_l = x_box.lb.to(device=device, dtype=dtype).flatten(start_dim=1)
        x_u = x_box.ub.to(device=device, dtype=dtype).flatten(start_dim=1)
        y_l = y_box.lb.to(device=device, dtype=dtype).flatten(start_dim=1)
        y_u = y_box.ub.to(device=device, dtype=dtype).flatten(start_dim=1)
        if incremental_alphas is not None and layer.id in incremental_alphas:
            prior = cast(Dict[str, torch.Tensor], cast(object, incremental_alphas[layer.id]))
            tree = {
                key: prior[key].detach().clone().to(device=device, dtype=dtype).clamp(0.0, 1.0)
                for key in ("omega_l", "omega_u")
            }
        else:
            k_thresh = float(cast(float, layer.params.get("k_thresh", 1.0)))
            tree = matmul_rule_alpha(x_l, x_u, y_l, y_u, G, I, K, J, k_thresh)
        if optimize_alpha:
            return {key: torch.nn.Parameter(val.detach().clone()) for key, val in tree.items()}
        return {key: val.detach() for key, val in tree.items()}

    def _optimize_alpha_eta(
        self,
        net: Net,
        bounds_dict: Dict[int, Bounds],
        c: torch.Tensor,
        M: int = 1,
        n_iters: int = 50,
        lr_alpha: float = 0.1,
        lr_beta: float = 0.1,
        lr_decay: float = 0.98,
        incremental_alphas: Optional[AlphaState] = None,
        incremental_etas: Optional[Dict[int, torch.Tensor]] = None,
        split_signs: Optional[Dict[int, torch.Tensor]] = None,
        return_sce: bool = False,
        per_class_alpha: bool = True,
        optimize_alpha: bool = True,
        refresh_forward: bool = True,
        start_lid: Optional[int] = None,
        forward_lin_max_perturbed: Optional[int] = None,
        forward_frame: Optional[ForwardFrame] = None,
        stagnation_patience: int = 0,
        stagnation_tol: Optional[float] = None,
        stop_criterion: Optional[Callable[[torch.Tensor], bool]] = None,
        max_time: float = 0.0,
        time_cap: Optional[float] = None,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        AlphaState,
        Dict[int, torch.Tensor],
        torch.Tensor,
        Optional[torch.Tensor],
    ]:
        """Joint α/η optimization: iterative dual lower-bound refinement.

        Each ReLU gets a learnable lower-envelope slope α constrained to [0, 1].
        Each split layer with a nonzero split sign gets a learnable η multiplier
        constrained to η ≥ 0.

        Returns:
            ``(best_bounds, best_sce, alpha_state, eta_state, best_dual,
            best_forward)`` where ``best_bounds = max(best_dual, best_forward)``
            has shape ``[B*M]``, ``best_sce`` is optional, ``alpha_state`` maps
            each optimized layer to its tensor or attention pytree, and
            ``eta_state`` maps split layer id to optimized η. Keep-best and the
            α/η selection follow the dual bound alone, so the returned
            certificate is the one CLIMB replays; the forward bound is the
            per-iteration maximum (each iteration's forward bounds are valid
            for the same problem).

        ``forward_lin_max_perturbed=None`` and ``stagnation_tol=None`` resolve
        to the DualConfig defaults at call time rather than being frozen at
        import time.

        Early stopping (all off by default) is checked once at the end of
        every iteration, so the first iteration always completes and the
        keep-best state is returned: ``stagnation_patience`` consecutive
        iterations in which no row's keep-best dual bound gained more than
        ``stagnation_tol``; ``stop_criterion(best_bound)`` returning True on
        the current ``max(best_dual, best_forward)``; or the wall clock
        exceeding ``min(time_cap, max_time)`` (``time_cap`` is the caller's
        per-call cap in seconds, ``max_time`` the DualConfig cap, 0 = off).
        ``last_alpha_iterations`` / ``last_alpha_stop_reason`` record the
        iterations run and why the loop ended (``n_iters``, ``stagnation``,
        ``verified`` or ``time``). With ``refresh_forward`` the alpha
        independent forward prefix is computed once per call through
        :class:`ForwardPrefixCache`, which is bitwise identical to the plain
        forward pass.
        """
        if forward_lin_max_perturbed is None:
            forward_lin_max_perturbed = DualConfig().forward_lin_max_perturbed
        if stagnation_tol is None:
            stagnation_tol = DualConfig().stagnation_tol
        effective_cap = _effective_time_cap(time_cap, max_time)
        if c.dim() != 2:
            raise ValueError(
                f"c must be 2-D [B*M, n_out], got shape {tuple(c.shape)}"
            )
        if M < 1:
            raise ValueError(f"M must be >= 1, got {M}")
        BM = c.shape[0]
        if BM % M != 0:
            raise ValueError(
                f"c batch dim {BM} not divisible by M={M}; expected B*M rows"
            )
        B = BM // M
        device, dtype = c.device, c.dtype

        if per_class_alpha:
            alpha_bytes = _per_class_alpha_estimated_bytes(
                net, bounds_dict, B, M, dtype,
            )
            if alpha_bytes > _PER_CLASS_ALPHA_MAX_MIB * _MIB:
                report_key = (B, M, alpha_bytes)
                if report_key not in _PER_CLASS_ALPHA_CAP_REPORTED:
                    _PER_CLASS_ALPHA_CAP_REPORTED.add(report_key)
                    logger.warning(
                        "per-class alpha state estimate %.3f MiB exceeds %.3f MiB cap "
                        "for lanes=%d rows=%d; using shared alpha slopes",
                        alpha_bytes / _MIB,
                        _PER_CLASS_ALPHA_MAX_MIB,
                        B,
                        M,
                    )
                per_class_alpha = False
                if incremental_alphas is not None and _alpha_spec_row_count(incremental_alphas) > 1:
                    incremental_alphas = None

        input_lid = self._find_input_layer_id(net)
        if input_lid is None:
            raise ValueError("DualSolver._optimize_alpha_eta: net has no INPUT/INPUT_SPEC layer")
        by_id = getattr(net, "by_id", {layer.id: layer for layer in net.layers})
        input_layer = by_id[input_lid]
        input_bounds = bounds_dict.get(input_lid)
        if input_bounds is None:
            if "lb" not in input_layer.params or "ub" not in input_layer.params:
                raise ValueError(
                        f"DualSolver._optimize_alpha_eta: input layer {input_lid} has no bounds"
                )
            input_lb = cast(torch.Tensor, input_layer.params["lb"])
            input_ub = cast(torch.Tensor, input_layer.params["ub"])
        else:
            input_lb = input_bounds.lb
            input_ub = input_bounds.ub
        input_lb = input_lb.to(device=device, dtype=dtype)
        input_ub = input_ub.to(device=device, dtype=dtype)

        ancestor_lids: Optional[set[int]] = None
        if start_lid is not None:
            # Interior-start objectives only depend on ancestor layers; alpha
            # parameters outside that cone receive no gradient and would crash
            # the optimizer.
            ancestor_lids = {start_lid}
            stack = [start_lid]
            while stack:
                for p in net.preds.get(stack.pop(), []):
                    if p not in ancestor_lids:
                        ancestor_lids.add(p)
                        stack.append(p)

        alphas: AlphaState = {}
        for layer in net.layers:
            if ancestor_lids is not None and layer.id not in ancestor_lids:
                continue
            layer_preds = list(net.preds.get(layer.id, []))
            layer_kind = layer.kind.upper() if isinstance(layer.kind, str) else layer.kind
            alpha_bounds = (
                self._bilinear_operand_bounds(net, bounds_dict, layer_preds)
                if layer_kind in self._BILINEAR_KINDS else bounds_dict
            )
            tree = self._init_alpha(
                layer, alpha_bounds, B, M, device, dtype,
                per_class_alpha=per_class_alpha,
                optimize_alpha=optimize_alpha,
                incremental_alphas=incremental_alphas,
                preds=layer_preds,
            )
            if tree is not None:
                alphas[layer.id] = tree

        etas: Dict[int, torch.nn.Parameter] = {}
        if split_signs is not None:
            for lid, signs in split_signs.items():
                signs_init = signs.detach().to(device=device, dtype=dtype)
                if not (signs_init != 0).any():
                    continue
                if incremental_etas is not None and lid in incremental_etas:
                    eta_init = (
                        incremental_etas[lid]
                        .detach()
                        .clone()
                        .to(device=device, dtype=dtype)
                        .clamp(min=0)
                    )
                    if eta_init.shape != signs_init.shape:
                        eta_init = torch.zeros_like(signs_init)
                else:
                    eta_init = torch.zeros_like(signs_init)
                etas[lid] = torch.nn.Parameter(eta_init)

        def _fixed_point_result(result: DualResult) -> Tuple[
            torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]
        ]:
            dual = (result.dual_margins if result.dual_margins is not None else result.margins)
            return result.margins.detach(), result.sce, dual.detach(), result.forward_margins

        self.last_alpha_iterations = 0
        self.last_alpha_stop_reason = "fixed"
        if not alphas and not etas:
            result = self.compute_certified_bound(
                net, bounds_dict, c, M=M, return_sce=return_sce,
                start_lid=start_lid, forward_frame=forward_frame,
            )
            bound, sce, dual, fwd = _fixed_point_result(result)
            return bound, sce, {}, {}, dual, fwd

        param_groups: List[Dict[str, object]] = []
        alpha_params = [
            leaf
            for tree in alphas.values()
            for leaf in _alpha_tree_leaves(tree)
            if isinstance(leaf, torch.nn.Parameter)
        ]
        eta_params = list(etas.values())
        if alpha_params:
            param_groups.append({"params": alpha_params, "lr": lr_alpha})
        if eta_params:
            param_groups.append({"params": eta_params, "lr": lr_beta})
        if not param_groups:
            result = self.compute_certified_bound(
                net, bounds_dict, c, M=M, return_sce=return_sce,
                alpha=alphas if alphas else None,
                start_lid=start_lid, forward_frame=forward_frame,
            )
            bound, sce, dual, fwd = _fixed_point_result(result)
            return (
                bound,
                sce,
                {lid: _clone_alpha_tree(tree) for lid, tree in alphas.items()},
                {},
                dual,
                fwd,
            )
        optimizer = torch.optim.Adam(param_groups)
        scheduler = (
            torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)
            if lr_decay < 1.0
            else None
        )
        best_bounds = torch.full((BM,), float("-inf"), device=device, dtype=dtype)
        best_forward: Optional[torch.Tensor] = None
        best_sce: Optional[torch.Tensor] = None
        best_alpha_state: AlphaState = {
            lid: _clone_alpha_tree(tree) for lid, tree in alphas.items()
        }
        best_eta_state: Dict[int, torch.Tensor] = {
            lid: e.detach().clone() for lid, e in etas.items()
        }

        from act.back_end.dual_tf.tf_forward import (
            ForwardPrefixCache,
            compute_forward_bounds,
        )

        prefix_cache: Optional[ForwardPrefixCache] = None
        if refresh_forward and n_iters > 0:
            prefix_cache = ForwardPrefixCache(
                net,
                input_lb,
                input_ub,
                alpha_relu_ids=_alpha_relu_forward_view(alphas).keys(),
                post_activation=False,
                forward_lin_max_perturbed=forward_lin_max_perturbed,
            )
        iterations_done = 0
        stop_reason = "n_iters"
        flat_iterations = 0
        loop_started = _monotonic() if effective_cap is not None else 0.0

        with torch.enable_grad():
            for _ in range(n_iters):
                optimizer.zero_grad()
                alpha_trees = alphas
                eta_tensors = cast(Dict[int, torch.Tensor], etas)
                if refresh_forward:
                    forward_alphas = _alpha_relu_forward_view(alpha_trees)
                    fresh_bounds = compute_forward_bounds(
                        net,
                        input_lb,
                        input_ub,
                        post_activation=False,
                        alphas=forward_alphas,
                        forward_lin_max_perturbed=forward_lin_max_perturbed,
                        prefix_cache=prefix_cache,
                    )
                else:
                    # Fixed intermediate bounds (root-reuse mode): alpha/eta
                    # only enter the backward pass; sound for any valid
                    # bounds_dict.
                    fresh_bounds = bounds_dict
                result = self.compute_certified_bound(
                    net,
                    fresh_bounds,
                    c,
                    M=M,
                    return_sce=return_sce,
                    enable_grad=True,
                    alpha=alpha_trees,
                    eta=eta_tensors,
                    split_signs=split_signs,
                    start_lid=start_lid,
                    forward_frame=forward_frame,
                )
                bound_bm = (
                    result.dual_margins if result.dual_margins is not None else result.margins
                )
                sce = result.sce
                if result.forward_margins is not None:
                    fwd_detached = result.forward_margins.detach()
                    best_forward = (
                        fwd_detached if best_forward is None
                        else torch.maximum(best_forward, fwd_detached)
                    )

                if not bound_bm.requires_grad:
                    # No autograd path reaches any α/η parameter (every ReLU
                    # stable and every pool window dominant): gradient steps
                    # cannot move the bound, and backward() would raise.
                    # Return the fixed-slope bound as optimize=False would.
                    detached = bound_bm.detach()
                    improved = detached > best_bounds
                    best_bounds = torch.where(improved, detached, best_bounds)
                    if return_sce and sce is not None:
                        if best_sce is None:
                            best_sce = sce.detach().clone()
                        else:
                            best_sce[improved] = sce[improved].detach()
                    self.last_alpha_iterations = iterations_done + 1
                    self.last_alpha_stop_reason = "fixed"
                    return (
                        _max_with_optional(best_bounds, best_forward), best_sce,
                        best_alpha_state, best_eta_state, best_bounds, best_forward,
                    )

                (-bound_bm.sum()).backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                with torch.no_grad():
                    for tree in alphas.values():
                        for leaf in _alpha_tree_leaves(tree):
                            leaf.data.clamp_(0.0, 1.0)
                    for e in etas.values():
                        e.data.clamp_(min=0)

                    improved = bound_bm > best_bounds
                    if stagnation_patience > 0:
                        gained = bound_bm > best_bounds + stagnation_tol
                        flat_iterations = 0 if bool(gained.any().item()) else flat_iterations + 1
                    if improved.any():
                        best_bounds = torch.where(improved, bound_bm.detach(), best_bounds)
                        best_alpha_state = {
                            lid: _clone_alpha_tree(tree) for lid, tree in alphas.items()
                        }
                        best_eta_state = {
                            lid: e.detach().clone() for lid, e in etas.items()
                        }
                    if return_sce and sce is not None:
                        if best_sce is None:
                            best_sce = sce.detach().clone()
                        else:
                            best_sce[improved] = sce[improved].detach()

                    iterations_done += 1
                    if stop_criterion is not None and stop_criterion(
                        _max_with_optional(best_bounds, best_forward)
                    ):
                        stop_reason = "verified"
                        break
                    if stagnation_patience > 0 and flat_iterations >= stagnation_patience:
                        stop_reason = "stagnation"
                        break
                    if (
                        effective_cap is not None
                        and _monotonic() - loop_started >= effective_cap
                    ):
                        stop_reason = "time"
                        break

        self.last_alpha_iterations = iterations_done
        self.last_alpha_stop_reason = stop_reason if n_iters > 0 else "n_iters"

        if n_iters <= 0:
            result = self.compute_certified_bound(
                net,
                bounds_dict,
                c,
                M=M,
                return_sce=return_sce,
                enable_grad=False,
                alpha=alphas,
                eta=cast(Dict[int, torch.Tensor], etas),
                split_signs=split_signs,
                start_lid=start_lid,
                forward_frame=forward_frame,
            )
            best_bounds = (
                result.dual_margins if result.dual_margins is not None else result.margins
            )
            best_forward = result.forward_margins
            best_sce = result.sce

        best_dual = best_bounds.detach()
        return (
            _max_with_optional(best_dual, best_forward), best_sce,
            best_alpha_state, best_eta_state, best_dual, best_forward,
        )

    def _interval_refresh_bounds(
        self,
        net: Net,
        base: Dict[int, Bounds],
        split_signs: Dict[int, torch.Tensor],
    ) -> Optional[Dict[int, Bounds]]:
        """Re-propagate split phases with interval arithmetic and intersect base."""
        from act.back_end.dual_tf.tf_forward import _fwd_conv2d_interval

        out = dict(base)
        vals: Dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for layer in net.layers:
            k = layer.kind.upper() if isinstance(layer.kind, str) else layer.kind
            lid = layer.id
            if k == LayerKind.ASSERT.value:
                continue
            if k in (LayerKind.INPUT.value, LayerKind.INPUT_SPEC.value):
                b = out.get(lid)
                if b is None:
                    return None
                vals[lid] = (b.lb.flatten(start_dim=1), b.ub.flatten(start_dim=1))
                continue
            preds = net.preds.get(lid, [])
            refreshed = None
            try:
                if k == LayerKind.CONV2D.value:
                    plb, pub = vals[preds[0]]
                    lb, ub = _fwd_conv2d_interval(layer, plb, pub)
                    lb, ub = lb.flatten(start_dim=1), ub.flatten(start_dim=1)
                elif k == LayerKind.DENSE.value:
                    w = layer.params["weight"]
                    bias = layer.params.get("bias")
                    if not isinstance(w, torch.Tensor):
                        return None
                    plb, pub = vals[preds[0]]
                    c = (plb + pub) * 0.5
                    r = (pub - plb) * 0.5
                    m = c @ w.T
                    rho = r @ w.abs().T
                    lb = m - rho
                    ub = m + rho
                    if isinstance(bias, torch.Tensor):
                        lb, ub = lb + bias, ub + bias
                elif k == LayerKind.ADD.value:
                    (alb, aub), (blb, bub) = vals[preds[0]], vals[preds[1]]
                    lb, ub = alb + blb, aub + bub
                elif k in (LayerKind.FLATTEN.value, LayerKind.RESHAPE.value):
                    lb, ub = vals[preds[0]]
                elif k == LayerKind.RELU.value:
                    lb, ub = vals[preds[0]]
                else:
                    return None
                refreshed = (lb, ub)
            except (KeyError, IndexError, ValueError):
                # Missing/incompatible interval metadata disables this optional refresh.
                refreshed = None
            if refreshed is None:
                return None
            lb, ub = refreshed

            b = out.get(lid)
            if b is not None:
                lb = torch.maximum(lb, b.lb.flatten(start_dim=1))
                ub = torch.minimum(ub, b.ub.flatten(start_dim=1))
                ub = torch.maximum(ub, lb)
            if k == LayerKind.RELU.value:
                s = split_signs.get(lid)
                if s is not None:
                    sl = s[:, 0, :] if s.dim() == 3 else s
                    n = min(lb.shape[-1], sl.shape[-1])
                    sn = sl[..., :n].to(lb.device)
                    lb, ub = lb.clone(), ub.clone()
                    lb[..., :n] = torch.where(
                        sn > 0, lb[..., :n].clamp(min=0.0), lb[..., :n]
                    )
                    ub[..., :n] = torch.where(
                        sn < 0, ub[..., :n].clamp(max=0.0), ub[..., :n]
                    )
                    ub[..., :n] = torch.maximum(ub[..., :n], lb[..., :n])
            if b is not None:
                out[lid] = Bounds(
                    lb.view_as(b.lb).clone(), ub.view_as(b.ub).clone()
                )
            if k == LayerKind.RELU.value:
                vals[lid] = (lb.clamp(min=0.0), ub.clamp(min=0.0))
            else:
                vals[lid] = (lb, ub)
        return out

    def _harden_split_bounds(
        self,
        bounds_dict: Dict[int, Bounds],
        split_signs: Optional[Dict[int, torch.Tensor]],
    ) -> Dict[int, Bounds]:
        """Fix split ReLU phases in the relaxation itself.

        sign=+1 asserts pre-activation >= 0 (lb clamped to 0: the relaxation
        collapses to the exact identity); sign=-1 asserts <= 0 (ub clamped to
        0: exact zero). Applied alongside the eta Lagrangian term, the split
        tightens the bound through both channels. Sound: clamping encodes
        exactly the branch's split assumption.
        """
        if not split_signs or isinstance(split_signs, list):
            return bounds_dict
        out = dict(bounds_dict)
        for lid, signs in split_signs.items():
            b = out.get(lid)
            if b is None:
                continue
            s = signs[:, 0, :] if signs.dim() == 3 else signs
            if not bool((s != 0).any().item()):
                continue
            lb = b.lb.flatten(start_dim=1).clone()
            ub = b.ub.flatten(start_dim=1).clone()
            n = min(lb.shape[-1], s.shape[-1])
            s_n = s[..., :n].to(device=lb.device)
            lb[..., :n] = torch.where(s_n > 0, lb[..., :n].clamp(min=0.0), lb[..., :n])
            ub[..., :n] = torch.where(s_n < 0, ub[..., :n].clamp(max=0.0), ub[..., :n])
            # An infeasible branch (split contradicts a stable phase) yields a
            # degenerate [x, x] interval; the branch represents an empty input
            # region, so any bound for it is sound.
            ub[..., :n] = torch.maximum(ub[..., :n], lb[..., :n])
            out[lid] = Bounds(lb.view_as(b.lb), ub.view_as(b.ub))
        return out

    def _local_phase_bounds(
        self,
        bounds_dict: Dict[int, Bounds],
        lid: int,
        signs: torch.Tensor,
    ) -> Dict[int, Bounds]:
        """Clamp one ReLU entry for its handler without mutating reference bounds."""
        bounds = bounds_dict.get(lid)
        if bounds is None:
            return bounds_dict
        phase = signs[:, 0, :] if signs.dim() == 3 else signs
        if not bool((phase != 0).any().item()):
            return bounds_dict
        lb = bounds.lb.flatten(start_dim=1).clone()
        ub = bounds.ub.flatten(start_dim=1).clone()
        n = min(lb.shape[-1], phase.shape[-1])
        phase = phase[..., :n].to(device=lb.device)
        lb[..., :n] = torch.where(
            phase > 0, lb[..., :n].clamp(min=0.0), lb[..., :n]
        )
        ub[..., :n] = torch.where(
            phase < 0, ub[..., :n].clamp(max=0.0), ub[..., :n]
        )
        ub[..., :n] = torch.maximum(ub[..., :n], lb[..., :n])
        local = dict(bounds_dict)
        local[lid] = Bounds(lb.view_as(bounds.lb), ub.view_as(bounds.ub))
        return local

    def refine_intermediate_bounds(
        self,
        net: Net,
        bounds_dict: Dict[int, Bounds],
        mode: str = "auto",
        blowup_ratio: float = 10.0,
        max_rows_per_call: int = 4096,
        optimize_iters: int = 20,
        max_tensor_mib: float = 1024.0,
        refinement_stats: Optional[Dict[str, int]] = None,
        stagnation_patience: int = 0,
        stagnation_tol: Optional[float] = None,
        max_time: float = 0.0,
        time_cap: Optional[float] = None,
    ) -> Dict[int, Bounds]:
        """Metric-driven backward refinement of selected pre-activation bounds.

        Forward-mode concretization loses correlation at wide fan-in affine
        layers; a backward pass from the affected layer to the input keeps it.
        Selection is architecture-agnostic: an activation layer qualifies if it
        has unstable neurons and (mode="auto") its mean pre-activation width
        exceeds ``blowup_ratio`` x the median width of all activation layers;
        mode="all" refines every unstable activation layer. Refined bounds are
        intersected with the forward bounds (both are valid over-approximations,
        so the intersection is sound). Layers are processed in topological
        order so later refinements consume earlier ones.

        ``max_tensor_mib`` skips a selected layer as one unit when the largest
        existing row chunk would make one dense backward tensor exceed the cap;
        its current sound forward bounds are retained. ``stagnation_patience`` /
        ``stagnation_tol`` / ``max_time`` apply to every row chunk's alpha loop;
        ``time_cap`` caps the whole call: each chunk receives the remaining time
        and, once it is used up, the remaining layers keep their forward bounds.
        """
        if mode == "none":
            return bounds_dict
        if mode not in ("auto", "all", "tail"):
            raise ValueError(
                f"intermediate_refine mode must be none|auto|all|tail, got {mode!r}"
            )
        call_started = _monotonic() if time_cap is not None else 0.0

        def _remaining_cap() -> Optional[float]:
            if time_cap is None:
                return None
            return max(0.0, time_cap - (_monotonic() - call_started))

        stats = []
        for layer in net.layers:
            k = layer.kind.upper() if isinstance(layer.kind, str) else layer.kind
            if k != LayerKind.RELU.value or layer.id not in bounds_dict:
                continue
            b = bounds_dict[layer.id]
            lb, ub = b.lb.flatten(start_dim=1), b.ub.flatten(start_dim=1)
            unstable = int(((lb < 0) & (ub > 0)).sum().item())
            stats.append((layer.id, unstable, float((ub - lb).mean().item())))
        selected: List[int] = []
        if stats:
            median_width = sorted(s[2] for s in stats)[len(stats) // 2]
            threshold = max(median_width, 1e-9) * blowup_ratio
            if mode == "tail":
                unstable_lids = [lid for lid, unstable, _ in stats if unstable > 0]
                selected = unstable_lids[-2:]
            else:
                selected = [
                    lid for lid, unstable, width in stats
                    if unstable > 0 and (mode == "all" or width > threshold)
                ]
        targets = [
            (lid, net.preds[lid][0], False)
            for lid in selected if len(net.preds.get(lid, [])) == 1
        ]
        if mode == "all":
            targets += self._attention_refine_targets(net, bounds_dict)
            position = {layer.id: index for index, layer in enumerate(net.layers)}
            targets.sort(key=lambda target: position[target[0]])
        if not targets:
            return bounds_dict

        out = dict(bounds_dict)
        for layer_index, (lid, pred_lid, every_neuron) in enumerate(targets):
            if layer_index > 0 and _remaining_cap() == 0.0:
                break
            b = out[lid]
            lb0 = b.lb.flatten(start_dim=1)
            ub0 = b.ub.flatten(start_dim=1)
            if lb0.shape[0] != 1:
                continue
            n = lb0.shape[-1]
            device, dtype = lb0.device, lb0.dtype
            # Only unstable neurons need refinement: stable phases make the
            # relaxation exact regardless of bound width, so querying them
            # would spend backward rows for zero tightening.
            if every_neuron:
                amb_idx = torch.where(ub0[0] > lb0[0])[0]
            else:
                amb_idx = torch.where((lb0[0] < 0) & (ub0[0] > 0))[0]
            n_amb = int(amb_idx.numel())
            if n_amb == 0:
                continue
            dense_rows = 2 * min(n_amb, max_rows_per_call)
            if _skip_refinement_for_tensor_cap(
                net=net,
                bounds_dict=out,
                layer_id=lid,
                start_lid=pred_lid,
                lanes=1,
                rows=dense_rows,
                itemsize=lb0.element_size(),
                max_tensor_mib=max_tensor_mib,
                refinement_stats=refinement_stats,
            ):
                continue
            lb_new = torch.empty(n_amb, device=device, dtype=dtype)
            ub_new = torch.empty(n_amb, device=device, dtype=dtype)
            for s in range(0, n_amb, max_rows_per_call):
                e = min(s + max_rows_per_call, n_amb)
                eye = torch.zeros(e - s, n, device=device, dtype=dtype)
                eye[torch.arange(e - s), amb_idx[s:e]] = 1.0
                rows = torch.cat([eye, -eye], dim=0)
                res = self.compute_certified_bound(
                    net, out, rows.contiguous(), M=int(rows.shape[0]),
                    start_lid=pred_lid,
                    optimize=optimize_iters > 0 and not every_neuron,
                    n_iters=optimize_iters,
                    lr_alpha=0.25,
                    lr_decay=0.98,
                    per_class_alpha=True,
                    refresh_forward=False,
                    stagnation_patience=stagnation_patience,
                    stagnation_tol=stagnation_tol,
                    max_time=max_time,
                    time_cap=_remaining_cap(),
                )
                lb_new[s:e] = res.margins[: e - s]
                ub_new[s:e] = -res.margins[e - s:]
            lb_ref = lb0[0].clone()
            ub_ref = ub0[0].clone()
            lb_ref[amb_idx], ub_ref[amb_idx] = _intersect_boxes(lb_ref[amb_idx], ub_ref[amb_idx], lb_new, ub_new)
            ub_ref = torch.maximum(ub_ref, lb_ref)
            refined = Bounds(
                lb_ref.view_as(b.lb[0]).unsqueeze(0).clone(),
                ub_ref.view_as(b.ub[0]).unsqueeze(0).clone(),
            )
            out[lid] = refined
            if pred_lid in out and out[pred_lid].lb.shape == refined.lb.shape:
                out[pred_lid] = refined
        return out

    _ATTENTION_REFINE_ACTIVATIONS = {
        LayerKind.SOFTMAX.value, LayerKind.TANH.value,
        LayerKind.SIGMOID.value, LayerKind.GELU.value,
    }

    def _attention_refine_targets(
        self, net: Net, bounds_dict: Dict[int, Bounds],
    ) -> List[Tuple[int, int, bool]]:
        """Relaxation-input boxes of a net with a bilinear layer, as refine targets.

        Each target is ``(lid, start_lid, True)``: the box stored at ``lid`` is
        the output of ``start_lid`` and every neuron of it is refined. Smooth
        activations store their pre-activation box (``start_lid`` = the
        predecessor); a bilinear operand stores its own output
        (``start_lid == lid``). Targets take one rule-alpha backward pass (no
        alpha loop), so the root budget is kept for the presolve. Forward-mode composition through the McCormick
        planes of the value mixing loses the objective-aware plane choice of a
        backward pass, so deep attention blocks feed these relaxations with
        boxes far wider than the true range. Nets without a bilinear layer get
        no targets.
        """
        kinds = {
            layer.id: layer.kind.upper() if isinstance(layer.kind, str) else layer.kind
            for layer in net.layers
        }
        if not any(kind in self._BILINEAR_KINDS for kind in kinds.values()):
            return []
        targets: Dict[int, Tuple[int, int, bool]] = {}
        for layer in net.layers:
            lid, preds = layer.id, net.preds.get(layer.id, [])
            if kinds[lid] in self._ATTENTION_REFINE_ACTIVATIONS:
                if len(preds) == 1 and lid in bounds_dict:
                    targets[lid] = (lid, preds[0], True)
            elif kinds[lid] in self._BILINEAR_KINDS:
                for operand in preds:
                    if (
                        operand in bounds_dict
                        and net.preds.get(operand)
                        and kinds.get(operand) not in self._PRE_ACTIVATION_KINDS
                    ):
                        targets.setdefault(operand, (operand, operand, True))
        return list(targets.values())

    def refine_intermediate_bounds_batched(
        self,
        net: Net,
        bounds_dict: Dict[int, Bounds],
        split_signs: Optional[Dict[int, torch.Tensor]] = None,
        mode: str = "tail",
        rows_cap: int = 64,
        optimize_iters: int = 0,
        lane_chunk: int = 32,
        max_tensor_mib: float = 1024.0,
        refinement_stats: Optional[Dict[str, int]] = None,
        stagnation_patience: int = 0,
        stagnation_tol: Optional[float] = None,
        max_time: float = 0.0,
        time_cap: Optional[float] = None,
    ) -> Dict[int, Bounds]:
        """K-lane per-subproblem sparse refinement of pre-activation bounds.

        Batched counterpart of ``refine_intermediate_bounds`` for the BaB loop:
        ``bounds_dict`` entries are ``[K, *shape]`` (one lane per subproblem).
        Split constraints are applied by hardening the bounds FIRST (sign=+1
        clamps lb to 0, sign=-1 clamps ub to 0); the backward pass then sees
        the hardened relaxation slopes, which propagates each lane's splits
        relationally to downstream layers - the tightening that the interval
        refresh cannot provide. ``split_signs`` is NOT forwarded to
        ``compute_certified_bound`` (its eta machinery is shaped for the final
        spec's M, not the refine rows); hardening alone carries the split.

        Rows are the per-neuron one-hot +/- queries for the UNION of unstable
        neurons across lanes, capped at ``rows_cap`` by descending interval
        width. Each refined bound is intersected per lane with the existing
        bound (both are valid over-approximations: sound). Layers are visited
        in topological order so later refinements consume earlier ones.

        The tensor cap and early-stop / time-cap arguments follow
        ``refine_intermediate_bounds``: a capped layer or layers left when time
        is used up keep their current sound bounds.
        """
        if mode == "none":
            return bounds_dict
        if mode not in ("tail", "all"):
            raise ValueError(
                f"per_subproblem_refine mode must be none|tail|all, got {mode!r}"
            )
        call_started = _monotonic() if time_cap is not None else 0.0

        def _remaining_cap() -> Optional[float]:
            if time_cap is None:
                return None
            return max(0.0, time_cap - (_monotonic() - call_started))

        out = self._harden_split_bounds(bounds_dict, split_signs)

        stats: List[tuple[int, int]] = []
        for layer in net.layers:
            k = layer.kind.upper() if isinstance(layer.kind, str) else layer.kind
            if k != LayerKind.RELU.value or layer.id not in out:
                continue
            b = out[layer.id]
            lb, ub = b.lb.flatten(start_dim=1), b.ub.flatten(start_dim=1)
            n_unstable = int(((lb < 0) & (ub > 0)).any(dim=0).sum().item())
            stats.append((layer.id, n_unstable))
        unstable_lids = [lid for lid, n_unstable in stats if n_unstable > 0]
        if not unstable_lids:
            return out
        selected = unstable_lids[-2:] if mode == "tail" else unstable_lids

        for layer_index, lid in enumerate(selected):
            if layer_index > 0 and _remaining_cap() == 0.0:
                break
            preds = net.preds.get(lid, [])
            if len(preds) != 1:
                continue
            pred_lid = preds[0]
            b = out[lid]
            lb0 = b.lb.flatten(start_dim=1)
            ub0 = b.ub.flatten(start_dim=1)
            k_lanes = lb0.shape[0]
            n = lb0.shape[-1]
            device, dtype = lb0.device, lb0.dtype
            amb_union = ((lb0 < 0) & (ub0 > 0)).any(dim=0)
            amb_idx = torch.where(amb_union)[0]
            n_amb = int(amb_idx.numel())
            if n_amb == 0:
                continue
            if n_amb > rows_cap:
                width = (ub0 - lb0).amax(dim=0)[amb_idx]
                amb_idx = amb_idx[torch.topk(width, k=rows_cap).indices]
                n_amb = rows_cap
            dense_lanes = min(k_lanes, lane_chunk)
            dense_rows = 2 * n_amb
            if _skip_refinement_for_tensor_cap(
                net=net,
                bounds_dict=out,
                layer_id=lid,
                start_lid=pred_lid,
                lanes=dense_lanes,
                rows=dense_rows,
                itemsize=lb0.element_size(),
                max_tensor_mib=max_tensor_mib,
                refinement_stats=refinement_stats,
            ):
                continue
            eye = torch.zeros(n_amb, n, device=device, dtype=dtype)
            eye[torch.arange(n_amb), amb_idx] = 1.0
            rows = torch.cat([eye, -eye], dim=0)
            m_rows = int(rows.shape[0])
            margins = torch.empty(k_lanes, m_rows, device=device, dtype=dtype)
            for k0 in range(0, k_lanes, lane_chunk):
                k1 = min(k0 + lane_chunk, k_lanes)
                sub = {
                    l: Bounds(bb.lb[k0:k1], bb.ub[k0:k1]) for l, bb in out.items()
                }
                c = rows.repeat(k1 - k0, 1).contiguous()
                res = self.compute_certified_bound(
                    net, sub, c, M=m_rows,
                    start_lid=pred_lid,
                    optimize=optimize_iters > 0,
                    n_iters=optimize_iters,
                    lr_alpha=0.25,
                    lr_decay=0.98,
                    per_class_alpha=True,
                    refresh_forward=False,
                    stagnation_patience=stagnation_patience,
                    stagnation_tol=stagnation_tol,
                    max_time=max_time,
                    time_cap=_remaining_cap(),
                )
                margins[k0:k1] = res.margins.view(k1 - k0, m_rows)
            lb_new = margins[:, :n_amb]
            ub_new = -margins[:, n_amb:]
            lb_ref = lb0.clone()
            ub_ref = ub0.clone()
            lb_ref[:, amb_idx], ub_ref[:, amb_idx] = _intersect_boxes(lb_ref[:, amb_idx], ub_ref[:, amb_idx], lb_new, ub_new)
            ub_ref = torch.maximum(ub_ref, lb_ref)
            refined = Bounds(
                lb_ref.view_as(b.lb).clone(),
                ub_ref.view_as(b.ub).clone(),
            )
            out[lid] = refined
            if pred_lid in out and out[pred_lid].lb.shape == refined.lb.shape:
                out[pred_lid] = refined
        return out

    def recompute_bounds_and_nu(
        self,
        net: Net,
        bounds_dict: Dict[int, Bounds],
        c: torch.Tensor,
        M: int,
        alpha_state: Optional[AlphaState] = None,
        eta_state: Optional[Dict[int, torch.Tensor]] = None,
        split_signs: Optional[Dict[int, torch.Tensor]] = None,
        per_class_alpha: bool = True,
    ) -> Tuple[Dict[int, Bounds], Optional[Dict[int, torch.Tensor]]]:
        """Forward bounds and per-RELU ν at the converged (α, η), from one pass.

        BaBSR/FSB scoring pairs each RELU's slope/intercept (from interval bounds)
        with its backward multiplier ν. Both MUST come from the same forward pass or
        the heuristic mixes an un-optimized interval with an optimized multiplier.
        Returns ``(fresh_bounds, nu_per_layer)``; soundness is unaffected (the
        certified bound is produced separately — this output is heuristic-only).
        """
        from act.back_end.dual_tf.tf_forward import compute_forward_bounds

        device, dtype = c.device, c.dtype
        input_lid = self._find_input_layer_id(net)
        if input_lid is None:
            return bounds_dict, None
        input_bounds = bounds_dict.get(input_lid)
        if input_bounds is None:
            return bounds_dict, None
        input_lb = input_bounds.lb.to(device=device, dtype=dtype)
        input_ub = input_bounds.ub.to(device=device, dtype=dtype)

        with torch.no_grad():
            forward_alphas = _alpha_relu_forward_view(alpha_state) or None
            fresh_bounds = compute_forward_bounds(
                net, input_lb, input_ub, post_activation=False, alphas=forward_alphas,
            )
            result = self.compute_certified_bound(
                net,
                fresh_bounds,
                c,
                M=M,
                enable_grad=False,
                alpha=alpha_state if alpha_state else None,
                eta=eta_state if eta_state else None,
                split_signs=split_signs,
                return_nu_per_layer=True,
            )
        return fresh_bounds, result.nu_per_layer

    def evaluate_spec(
        self, net: Net,
        out_spec: OutputSpec,
        bounds_dict: Optional[Dict[int, Bounds]] = None,
        num_classes: Optional[int] = None,
        chunk_size: Optional[int] = None,
        enable_grad: bool = False,
        collect_bounds: bool = False,
    ) -> SpecBatchResult:
        """Dual bound evaluation for any OutputSpec — self-contained entry point.

        Refactor note: ``bounds_dict`` is optional. When omitted (the typical
        case), the solver gathers the net's INPUT_SPEC seed bounds and computes
        per-layer pre-activation forward bounds internally via
        ``compute_forward_bounds(post_activation=False)``. Callers who already
        have a bounds_dict (e.g. BaB refinement loops) may pass it explicitly to
        skip the recomputation. When ``collect_bounds`` is true, the solver
        stores the same dict on ``last_forward_bounds`` so post-verification
        soundness checks validate exactly the bounds used by the dual certificate.

        Strategy: dispatch on ``out_spec.kind`` into two branches that share
        ``compute_certified_bound`` but use opposite sign conventions and
        opposite row aggregators.

        - ALL-rows kinds (LINEAR_LE, TOP1_ROBUST, MARGIN_ROBUST, RANGE):
          ``encode_linear`` emits (C, thresholds) in UB-cert form (CERTIFIED
          iff ``UB(C @ y) < threshold``). Pass ``-C`` / ``-thresholds`` to
          ``compute_certified_bound`` and compare through the kind-aware
          ``unproven_spec_rows``: ``slack >= 0`` passes a LINEAR_LE / RANGE
          row (closed safe set), while a TOP1_ROBUST / MARGIN_ROBUST row
          (open safe set, a tie violates) passes only when its slack clears
          the ``certification_tolerance`` band. Certified iff every row
          passes (``.all()``).
        - EXISTS-row kind (UNSAFE_LINEAR): the unsafe polytope is
          ``P = {y : c_i^T y <= d_i for ALL i}``. SAFE iff for all reachable
          y, some row i satisfies ``c_i^T y > d_i`` (escape). Sound
          strengthening via quantifier swap (mirrors ``verifier.py:574-580``):
          certify SAFE iff there exists a row i with ``LB_dual(c_i^T y) > d_i``.
          ``encode_linear`` emits UNSAFE_LINEAR in LB-cert form, so pass ``+C``
          / ``+thresholds`` directly (no sign flip). Certified iff any row
          escapes (``.any()``).

        Raises:
            ValueError: if net lacks ASSERT layer, ASSERT has != 1 predecessor,
                or (when bounds_dict is supplied) the output layer's bounds are
                missing / unbatched.
        """
        forward_frame: Optional[ForwardFrame] = None
        if bounds_dict is None:
            from act.back_end.dual_tf.tf_forward import compute_forward_bounds_with_frame
            from act.back_end.verifier import (
                gather_input_spec_layers,
                seed_from_input_specs,
            )
            spec_layers = gather_input_spec_layers(net)
            seed_bounds = seed_from_input_specs(spec_layers)
            bounds_dict, forward_frame = compute_forward_bounds_with_frame(
                net, seed_bounds.lb, seed_bounds.ub,
                frame_lid=self._spec_layer_id(net),
                post_activation=False,
            )

        if collect_bounds:
            self.last_forward_bounds = bounds_dict

        sample = next(iter(bounds_dict.values()))
        device = sample.lb.device
        dtype = sample.lb.dtype
        if sample.lb.dim() < 2:
            raise ValueError(
                "DualSolver.evaluate_spec: bounds_dict entries must be batched "
                f"[B, *shape]; got dim={sample.lb.dim()}"
            )
        B = sample.lb.shape[0]

        assert_layer = None
        for layer in net.layers:
            k = layer.kind.upper() if isinstance(layer.kind, str) else layer.kind
            if k == LayerKind.ASSERT.value:
                assert_layer = layer
                break
        if assert_layer is None:
            raise ValueError("DualSolver.evaluate_spec: net has no ASSERT layer")
        assert_preds = net.preds.get(assert_layer.id, [])
        if len(assert_preds) != 1:
            raise ValueError(
                f"ASSERT layer must have exactly 1 predecessor, got {len(assert_preds)}"
            )
        output_lid = assert_preds[0]
        if output_lid not in bounds_dict:
            raise ValueError(
                f"DualSolver.evaluate_spec: bounds_dict missing output layer "
                f"{output_lid} (ASSERT predecessor); run forward analysis first."
            )
        out_bounds = bounds_dict[output_lid]
        if out_bounds.lb.dim() < 2:
            raise ValueError(
                f"DualSolver.evaluate_spec: output layer {output_lid} bounds "
                f"must be batched; got dim={out_bounds.lb.dim()}"
            )
        n_out = int(out_bounds.lb.flatten(start_dim=1).shape[-1])

        if out_spec.kind == OutKind.UNSAFE_LINEAR:
            # EXISTS-row branch. encode_linear emits LB-cert form for
            # UNSAFE_LINEAR (specs.py:179-201) — pass +C / +thresholds
            # directly. Certified iff any row escapes the unsafe polytope.
            # Slack semantics is ASYMMETRIC vs ALL-rows kinds below:
            # here a row certifies once its slack clears the
            # ``escaping_spec_rows`` band; ``min_slack`` is NOT a meaningful
            # summary (use ``slack.max(dim=-1)`` instead).
            fe_params = out_spec.encode_linear(B=B, n_out=n_out, device=device, dtype=dtype)
            C = fe_params["C"].contiguous()
            thresholds = fe_params["thresholds"].contiguous()
            N = int(fe_params["M"])
            active_mask = torch.ones(B, N, dtype=torch.bool, device=device)

            with torch.set_grad_enabled(enable_grad):
                if chunk_size is None or N <= chunk_size:
                    result = self.compute_certified_bound(
                        net, bounds_dict, C, M=N, enable_grad=enable_grad,
                        forward_frame=forward_frame,
                    )
                    margins_flat = result.margins
                else:
                    margins_flat = self._chunked_eval(
                        net, bounds_dict, C, B, N, n_out, chunk_size, enable_grad,
                        forward_frame=forward_frame,
                    )
                margins = margins_flat.view(B, N)
                slack = margins - thresholds
                certified = (
                    escaping_spec_rows(slack, margins) & active_mask
                ).any(dim=-1)

            return SpecBatchResult(
                margins=margins,
                slack=slack,
                active_mask=active_mask,
                certified=certified,
            )

        fe_params = out_spec.encode_linear(B=B, n_out=n_out, device=device, dtype=dtype)
        C_neg = -fe_params["C"].contiguous()
        thresholds_neg = -fe_params["thresholds"].contiguous()
        M = int(fe_params["M"])
        active_mask = torch.ones(B, M, dtype=torch.bool, device=device)

        with torch.set_grad_enabled(enable_grad):
            if chunk_size is None or M <= chunk_size:
                result = self.compute_certified_bound(
                    net, bounds_dict, C_neg, M=M, enable_grad=enable_grad,
                    forward_frame=forward_frame,
                )
                margins_flat = result.margins
            else:
                margins_flat = self._chunked_eval(
                    net, bounds_dict, C_neg, B, M, n_out, chunk_size, enable_grad,
                    forward_frame=forward_frame,
                )

            margins = margins_flat.view(B, M)
            slack = margins - thresholds_neg
            # margins is a SOUND LOWER bound on the true margin: certify iff
            # every active row is proven under the kind's boundary rule
            # (hard zero for LINEAR_LE / RANGE, strict + band for TOP1 /
            # MARGIN), matching bab.py; a non-finite slack never passes.
            violations = unproven_spec_rows(slack, margins, out_spec.kind) & active_mask
            certified = ~violations.any(dim=-1)

        return SpecBatchResult(
            margins=margins,
            slack=slack,
            active_mask=active_mask,
            certified=certified,
        )

    def _chunked_eval(
        self, net: Net, bounds_dict: Dict[int, Bounds],
        C_neg: torch.Tensor, B: int, M: int, n_out: int,
        chunk_size: int, enable_grad: bool,
        forward_frame: Optional[ForwardFrame] = None,
    ) -> torch.Tensor:
        """Evaluate sign-flipped C in chunks along the M dimension.

        For large M (e.g. CIFAR-100 K=100), trades time for memory by
        processing chunk_size specs per sample at a time.

        Chunked evaluation invariant: slicing the leading B*M axis at arbitrary
        chunk_size is bit-identical to unchunked evaluation, because each
        (sample, spec) row is fully independent in the dual backward pass — no
        cross-row computation exists within a chunk or across chunk boundaries.
        """
        C_view = C_neg.view(B, M, n_out)
        chunks: List[torch.Tensor] = []
        for start in range(0, M, chunk_size):
            end = min(start + chunk_size, M)
            m_chunk = end - start
            # Slice specs [start:end] for all B samples — independent rows, invariant-safe.
            C_chunk = C_view[:, start:end, :].reshape(B * m_chunk, n_out).contiguous()
            result = self.compute_certified_bound(
                net, bounds_dict, C_chunk, M=m_chunk, enable_grad=enable_grad,
                forward_frame=forward_frame,
            )
            chunks.append(result.margins.view(B, m_chunk))
        return torch.cat(chunks, dim=1).reshape(B * M)

    def compute_robust_bound(
        self, net: Net, bounds_dict: Dict[int, Bounds],
        y_true: Union[int, torch.Tensor], num_classes: int,
        margin: float = 0.0,
        return_full: bool = False,
        enable_grad: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], SpecBatchResult]:
        """Dual certified robust bound for classification (top-1 or margin).

        Unified via evaluate_spec(). Retained as a first-class API for robust
        training loops and existing verification callers.

        Args:
            net: the ACT Net with an ASSERT layer.
            bounds_dict: layer bounds from forward analysis.
            y_true: [B] true class labels, or scalar for uniform label.
            num_classes: K (output dim of network's ASSERT predecessor).
            margin: if > 0 use MARGIN_ROBUST semantics (require y_t - y_j > margin);
                    else use TOP1_ROBUST (require y_t - y_j > 0). Both are
                    strict: an exact tie violates (``OutputSpec.CLOSED_KINDS``).
            return_full: if True, return the full SpecBatchResult (has per-class
                         [B, K] margins useful for training losses). If False,
                         return legacy tuple (min_slack: Tensor[B], certified: Tensor[B] bool).
            enable_grad: if True, allow gradients to flow through the computation
                         (for robust training). Default False (inference/verification).

        Returns:
            SpecBatchResult if return_full else (Tensor[B], Tensor[B] bool).
        """
        sample = next(iter(bounds_dict.values()))
        device = sample.lb.device
        if isinstance(y_true, int):
            B = sample.lb.shape[0] if sample.lb.dim() >= 2 else 1
            y_true_t = torch.full((B,), y_true, dtype=torch.long, device=device)
        else:
            y_true_t = y_true.to(device=device, dtype=torch.long)

        kind = OutKind.MARGIN_ROBUST if margin > 0 else OutKind.TOP1_ROBUST
        out_spec = OutputSpec(
            kind=kind,
            y_true=y_true_t,
            margin=(
                torch.as_tensor([margin], device=device, dtype=sample.lb.dtype)
                if margin > 0
                else None
            ),
        )
        result = self.evaluate_spec(
            net, out_spec,
            bounds_dict=bounds_dict,
            num_classes=num_classes,
            enable_grad=enable_grad,
        )
        if return_full:
            return result
        return result.min_slack, result.certified

    @staticmethod
    def _spec_layer_id(net: Net) -> int:
        """Id of the ASSERT layer's single predecessor (the spec/output layer)."""
        for layer in net.layers:
            k = layer.kind.upper() if isinstance(layer.kind, str) else layer.kind
            if k == LayerKind.ASSERT.value:
                preds = net.preds.get(layer.id, [])
                if len(preds) != 1:
                    raise ValueError(
                        f"ASSERT layer {layer.id} must have exactly 1 predecessor, got {len(preds)}"
                    )
                return preds[0]
        raise ValueError("DualSolver: net has no ASSERT layer")

    def _find_input_layer_id(self, net: Net) -> Optional[int]:
        """Return the INPUT_SPEC layer id if present, else INPUT's id, else None."""
        input_spec_id = None
        input_id = None
        for layer in net.layers:
            k = layer.kind.upper() if isinstance(layer.kind, str) else layer.kind
            if k == LayerKind.INPUT_SPEC.value:
                input_spec_id = layer.id
            elif k == LayerKind.INPUT.value:
                input_id = layer.id
        return input_spec_id if input_spec_id is not None else input_id

    def _input_contribution_from_nu(self, net: Net, input_lid: int,
                                    nu: torch.Tensor, bounds_dict: Dict[int, Bounds],
                                    M: int = 1,
                                    return_sce: bool = False,
                                    enable_grad: bool = False
                                    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Exact dual lower bound of ``nu @ x`` over the input region.

        For a box / L_inf spec (``p_norm`` unset or ``inf``) this is the box
        concretization ``lb·[nu]_+ + ub·[nu]_-``. For a finite-p LP_EMBEDDING
        spec it is the closed-form per-perturbed-word dual norm
        ``nu·center − Σ_block ‖half_block ⊙ nu_block‖_q`` (``q`` the Hölder dual
        of ``p``), which coincides with the box result exactly at p=inf, q=1.

        Lazy M-broadcast: ``nu`` has leading dim ``B*M`` (sample-major)
        while batched ``bounds_dict[input_lid]`` is ``[B, *shape]``. The
        contribution per (b, m) reuses the same bounds for all m via
        ``[B, 1, n]`` broadcast against ``[B, M, n]``. Bit-identical to
        legacy M-expanded path.

        The unbatched (``lb.dim() < 2``) and missing-bounds (lb/ub from
        ``input_layer.params``) paths are preserved: they broadcast a single
        ``[n]`` tensor against ``[BM, n]`` nu — the same as legacy with B=BM.
        """
        with torch.set_grad_enabled(enable_grad):
            BM = nu.shape[0]
            assert BM % M == 0, (
                f"_input_contribution_from_nu: nu batch {BM} not divisible by M={M}"
            )
            B = BM // M
            input_layer = net.by_id[input_lid]

            bounds = bounds_dict.get(input_lid)
            if bounds is None:
                if "lb" in input_layer.params and "ub" in input_layer.params:
                    lb = cast(torch.Tensor, input_layer.params["lb"])
                    ub = cast(torch.Tensor, input_layer.params["ub"])
                else:
                    raise ValueError(
                        f"_input_contribution_from_nu: input layer {input_lid} has no "
                        f"bounds in bounds_dict and no lb/ub params"
                    )
            else:
                lb = bounds.lb
                ub = bounds.ub

            # A finite input Lp ball (LP_EMBEDDING) is concretized by its exact
            # per-block dual norm; p=inf (or unset) falls through to the box
            # path below, bit-identical for every current vision / box spec.
            p_norm = _resolve_perturbation_norm(input_layer.params.get("p_norm"))
            if p_norm != float("inf"):
                return self._dual_norm_contribution(
                    input_layer, lb, ub, nu, B, M,
                    q=_dual_norm_exponent(p_norm),
                    return_sce=return_sce,
                )

            orig_shape = lb.shape
            v_flat = nu.flatten(start_dim=1)                       # [BM, n_in]

            if lb.dim() < 2:
                lb_b = lb.flatten().unsqueeze(0).expand(BM, -1)
                ub_b = ub.flatten().unsqueeze(0).expand(BM, -1)
                n = min(v_flat.shape[-1], lb_b.shape[-1])
                if v_flat.shape[-1] != lb_b.shape[-1]:
                    lb_b, ub_b, v_flat = lb_b[..., :n], ub_b[..., :n], v_flat[..., :n]
                assert (lb_b <= ub_b).all(), "Invalid input bounds: lb > ub"
                contrib = ((lb_b * v_flat.clamp(min=0)).sum(dim=-1)
                           + (ub_b * v_flat.clamp(max=0)).sum(dim=-1))
                sce = None
                if return_sce:
                    sce_flat = torch.where(v_flat > 0, lb_b, ub_b)
                    if sce_flat.shape[-1] == lb.flatten().numel():
                        sce = sce_flat.view(BM, *orig_shape)
                    else:
                        sce = sce_flat
                return contrib, sce

            lb_B = lb.flatten(start_dim=1)                         # [B, n_in]
            ub_B = ub.flatten(start_dim=1)                         # [B, n_in]
            n = min(v_flat.shape[-1], lb_B.shape[-1])
            if v_flat.shape[-1] != lb_B.shape[-1]:
                lb_B = lb_B[..., :n]
                ub_B = ub_B[..., :n]
                v_flat = v_flat[..., :n]
            assert (lb_B <= ub_B).all(), "Invalid input bounds: lb > ub"

            v = v_flat.view(B, M, n)                               # [B, M, n] view
            lb_bc = lb_B.unsqueeze(1)                              # [B, 1, n]
            ub_bc = ub_B.unsqueeze(1)                              # [B, 1, n]
            contrib_BM = ((lb_bc * v.clamp(min=0)).sum(dim=-1)
                          + (ub_bc * v.clamp(max=0)).sum(dim=-1))  # [B, M]
            contrib = contrib_BM.view(BM)

            sce = None
            if return_sce:
                sce_BMn = torch.where(v > 0, lb_bc, ub_bc)         # [B, M, n]
                sce_flat = sce_BMn.view(BM, n)
                total = int(torch.tensor(orig_shape[1:]).prod().item())
                sce = sce_flat.view(BM, *orig_shape[1:]) if sce_flat.shape[-1] == total else sce_flat
            return contrib, sce

    def _dual_norm_contribution(self, input_layer: Layer,
                                lb: torch.Tensor, ub: torch.Tensor,
                                nu: torch.Tensor, B: int, M: int,
                                q: float, return_sce: bool
                                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Per-perturbed-word-block dual-norm input contribution (finite Lp).

        Evaluates the inner minimization in closed form:
        ``nu·center − Σ_block ‖half_block ⊙ nu_block‖_q`` with
        ``center = (lb+ub)/2`` and ``half = (ub−lb)/2``. ``half`` equals the
        radius ε on perturbed coordinates and 0 on degenerate (non-perturbed)
        ones, so the latter contribute exactly ``nu·center`` with no penalty.
        Each embedding block is normed independently, so word balls stay
        decoupled. The dual norm is the exact value of ``min`` over the Lp ball,
        so no second-order-cone constraint is needed (q=1↔p=inf reproduces the
        box result, which the caller already routes through the box path). The
        arithmetic is :func:`lp_ball_support`, shared with the forward frame
        concretization so that every frame w.r.t. the INPUT is bounded alike.

        Args:
            input_layer: INPUT/INPUT_SPEC layer carrying ``p_norm`` and the
                optional ``perturbed_positions`` / ``embed_dim`` block metadata.
            lb: Input lower bounds, ``[B, *shape]`` (batched) or ``[*shape]``.
            ub: Input upper bounds, matching ``lb``.
            nu: Backward coefficient ``[B*M, *shape]`` (sample-major).
            B: Number of samples (``B*M == nu.shape[0]``).
            M: Spec rows per sample (lazy-M-broadcast factor).
            q: Hölder dual exponent of the spec norm ``p``.
            return_sce: Whether to also return the worst-case input witness.

        Returns:
            ``(contrib, sce)`` with ``contrib`` shape ``[B*M]`` and ``sce`` the
            ball minimizer (``None`` when ``return_sce`` is False).
        """
        orig_shape = lb.shape
        v_flat = nu.flatten(start_dim=1)                          # [BM, n_in]
        batched = lb.dim() >= 2
        if batched:
            lb_f = lb.flatten(start_dim=1)                        # [B, n_in]
            ub_f = ub.flatten(start_dim=1)
        else:
            lb_f = lb.flatten().unsqueeze(0)                      # [1, n_in]
            ub_f = ub.flatten().unsqueeze(0)
        n = min(v_flat.shape[-1], lb_f.shape[-1])
        if v_flat.shape[-1] != lb_f.shape[-1]:
            lb_f, ub_f, v_flat = lb_f[..., :n], ub_f[..., :n], v_flat[..., :n]
        assert (lb_f <= ub_f).all(), "Invalid input bounds: lb > ub"

        center = (lb_f + ub_f) * 0.5                              # [B|1, n]
        half = (ub_f - lb_f) * 0.5
        BM = v_flat.shape[0]
        blocks = input_block_slices(input_layer.params, orig_shape, n)

        if batched:
            v = v_flat.reshape(B, M, n)                           # [B, M, n]
            block_eps = input_layer.params.get("bab_block_eps")
            dot, penalty = lp_ball_support(
                v, center.unsqueeze(1), half.unsqueeze(1), blocks, q,
                block_eps if isinstance(block_eps, torch.Tensor) else None,
            )                                                     # [B, M]
            contrib = (dot - penalty).reshape(BM)
        else:
            dot, penalty = lp_ball_support(v_flat, center, half, blocks, q)   # [BM]
            contrib = dot - penalty

        sce = None
        if return_sce:
            sce = self._dual_norm_sce(
                center, half, v_flat, blocks, q, M, orig_shape, batched
            )
        return contrib, sce

    def _dual_norm_sce(self, center: torch.Tensor, half: torch.Tensor,
                       v_flat: torch.Tensor, blocks: List[Tuple[int, int]],
                       q: float, M: int, orig_shape: torch.Size,
                       batched: bool) -> torch.Tensor:
        """Worst-case input on the per-block Lp ball that attains the bound.

        The minimizer is ``center + δ*`` where, per block, ``δ*`` is the Hölder
        witness of ``‖half ⊙ nu‖_q``: the scaled-L2 ray ``−(half²⊙nu)/‖half⊙nu‖₂``
        for q=2 and a single max-coordinate spike for q=inf. Both lie on the ball
        boundary, so the witness is a sound counterexample candidate (the box
        corner is not, as it leaves the Lp ball). Non-perturbed coordinates stay
        at center (zero width).
        """
        BM = v_flat.shape[0]
        if batched:
            center_bm = center.repeat_interleave(M, dim=0)        # [B,n] -> [BM,n]
            half_bm = half.repeat_interleave(M, dim=0)
        else:
            center_bm = center.expand(BM, -1)                     # [1,n] -> [BM,n]
            half_bm = half.expand(BM, -1)
        sce_flat = center_bm.clone()
        for s, e in blocks:
            nu_b = v_flat[:, s:e]                                  # [BM, bs]
            h_b = half_bm[:, s:e]
            if q == 2.0:
                hn = h_b * nu_b
                norm = torch.linalg.vector_norm(hn, ord=2, dim=-1, keepdim=True)
                delta = -(h_b * hn) / norm.clamp_min(_DUAL_NORM_EPS)
            elif q == float("inf"):
                idx = (h_b * nu_b).abs().argmax(dim=-1, keepdim=True)
                spike = -torch.sign(nu_b.gather(-1, idx)) * h_b.gather(-1, idx)
                delta = torch.zeros_like(h_b)
                delta.scatter_(-1, idx, spike)
            else:
                delta = torch.zeros_like(h_b)
            sce_flat[:, s:e] = center_bm[:, s:e] + delta
        if batched:
            tail = orig_shape[1:]
            total = int(torch.tensor(tail).prod().item()) if len(tail) else 1
            return sce_flat.view(BM, *tail) if sce_flat.shape[-1] == total else sce_flat
        total = int(torch.tensor(orig_shape).prod().item())
        return sce_flat.view(BM, *orig_shape) if sce_flat.shape[-1] == total else sce_flat
