#===- act/back_end/solver/spec_batching.py - Batched Spec Representation -====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025- ACT Team
# Licensed under AGPLv3+; distributed without warranty.
#===---------------------------------------------------------------------===#
# Unified batched SpecBatch abstraction for dual-bound evaluation.
# Maps any OutputSpec kind (LINEAR_LE, UNSAFE_LINEAR, TOP1_ROBUST,
# MARGIN_ROBUST, RANGE) into a single [B*M, n_out] linear-form tensor
# for one-shot backward-pass evaluation via DualSolver.compute_bound.
#===---------------------------------------------------------------------===#
# pyright: reportMissingImports=false

from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, TYPE_CHECKING, cast

from act.back_end.bab.eta import EtaState
from act.back_end.bounds_dispatch import expand_rank3, materialize_if_needed
from act.back_end.core import Bounds
from act.back_end.dual_tf.tf_forward import LinearBound
from act.back_end.solver.alpha_state import AlphaState
from act.front_end.specs import OutputSpec, OutKind

if TYPE_CHECKING:
    from act.util.stats import VerifyResult


__all__ = [
    "SpecBatch",
    "SpecBatchResult",
    "build_spec_batch",
    "expand_bounds_dict",
]


@dataclass(frozen=True)
class SpecBatch:
    """Batched linear-form specification for dual-bound evaluation.

    Each of B samples carries M linear specs. Row b*M+j of C is the coefficient
    vector c_ij; we want to check LB(c_ij^T y) >= thresholds[b, j] for all
    active_mask[b, j] == True cells.

    Attributes:
        C: [B*M, n_out] - row-major flat batch of linear coefficients.
        thresholds: [B, M] - per-cell threshold t_ij.
        active_mask: [B, M] bool - which cells participate in certification.
        B: batch size.
        M: specs per sample.
    """
    C: torch.Tensor
    thresholds: torch.Tensor
    active_mask: torch.Tensor
    B: int
    M: int

    def __post_init__(self) -> None:
        assert self.C.dim() == 2, f"C must be 2-D [B*M, n_out], got {self.C.shape}"
        assert self.C.shape[0] == self.B * self.M, (
            f"C.shape[0]={self.C.shape[0]} != B*M={self.B * self.M}")
        assert self.thresholds.shape == (self.B, self.M), (
            f"thresholds.shape={self.thresholds.shape} != ({self.B}, {self.M})")
        assert self.active_mask.shape == (self.B, self.M), (
            f"active_mask.shape={self.active_mask.shape} != ({self.B}, {self.M})")
        assert self.active_mask.dtype == torch.bool, (
            f"active_mask.dtype={self.active_mask.dtype} must be torch.bool")

    def num_rows(self) -> int:
        return self.B * self.M


@dataclass(frozen=True)
class SpecBatchResult:
    """Batched result of evaluating a SpecBatch via dual lower bounds.

    Low-level counterpart to act.util.stats.VerifyResult:
    - SpecBatchResult: batched numerical tensors for training and further
      analysis (margins, slack, certified mask).
    - VerifyResult: per-sample verdict (status enum + optional counterexample)
      for end-user verification reporting.

    Use `to_verify_results()` to convert to a list of VerifyResult when
    reporting to callers expecting that API.

    Attributes:
        margins: [B, M] - LB(c_ij^T y) per cell.
        slack: [B, M] - margins - thresholds (non-negative means that cell
               passes certification).
        active_mask: [B, M] bool - which cells participated.
        certified: [B] bool - True iff all active cells have slack >= 0.
        out_etas: Optional EtaState with batch_size == B, carrying the Adam-
               optimised η values from this evaluation. Consumed by BaB as a
               warm start for child subproblems so that η state survives
               across splits instead of being recomputed from scratch.
               None when no η was in play (root subproblem / fast path / the
               chunked evaluation path where per-chunk η is not aggregated).
        out_alphas: Optional AlphaState of Adam-optimised ReLU relaxation
               slopes, keyed by (ReLU layer id, start_node_id). Phase 1 only
               uses AlphaState.FINAL_SID = -1, but the container is typed for
               future intermediate start nodes. Same warm-start semantics as
               out_etas: BaB copies into the subproblem batch so children
               inherit the parent's tuned α instead of re-deriving from the
               default heuristic. None on the legacy / chunked path.
    """
    margins: torch.Tensor
    slack: torch.Tensor
    active_mask: torch.Tensor
    certified: torch.Tensor
    out_etas: Optional[EtaState] = None
    out_alphas: Optional[AlphaState] = None

    def __post_init__(self) -> None:
        raw_out_alphas = self.__dict__.get("out_alphas")
        if isinstance(raw_out_alphas, dict):
            object.__setattr__(self, "out_alphas", AlphaState.from_legacy(raw_out_alphas))
        assert self.margins.dim() == 2, f"margins must be [B, M], got {self.margins.shape}"
        assert self.slack.shape == self.margins.shape
        assert self.active_mask.shape == self.margins.shape
        assert self.active_mask.dtype == torch.bool
        assert self.certified.shape == (self.margins.shape[0],)
        assert self.certified.dtype == torch.bool

    @property
    def min_slack(self) -> torch.Tensor:
        """[B] - min slack over active cells, +inf for all-inactive samples.

        Provides backward-compatible value for legacy callers expecting a
        single per-sample worst-case margin.
        """
        inf_fill = torch.full_like(self.slack, float("inf"))
        masked = torch.where(self.active_mask, self.slack, inf_fill)
        return masked.min(dim=-1).values

    @property
    def worst_violation(self) -> torch.Tensor:
        """[B] - min of clamp(slack, max=0) over active cells.

        Zero if all active cells pass; negative indicates worst violation magnitude.
        Intended for differentiable certified-training losses.
        """
        inf_fill = torch.full_like(self.slack, float("inf"))
        masked = torch.where(self.active_mask, self.slack, inf_fill)
        return masked.clamp(max=0.0).min(dim=-1).values

    def to_verify_results(self) -> List["VerifyResult"]:
        """Convert per-sample to legacy VerifyResult list.

        Maps:
          certified[b] == True  -> VerifyStatus.CERTIFIED
          certified[b] == False -> VerifyStatus.UNKNOWN  (NOT FALSIFIED:
              dual is sound but not complete; negative result may reflect
              relaxation gap rather than true violation)

        counterexample is None for all (dual path doesn't extract CEs).
        metadata carries margins[b], slack[b], min_slack[b] for downstream.
        """
        from act.util.stats import VerifyResult, VerifyStatus
        results: List[VerifyResult] = []
        min_slack_vals = self.min_slack
        B = self.margins.shape[0]
        for b in range(B):
            status = (VerifyStatus.CERTIFIED if bool(self.certified[b].item())
                      else VerifyStatus.UNKNOWN)
            results.append(VerifyResult(
                status=status,
                counterexample=None,
                metadata={
                    "margins": self.margins[b].detach().cpu().tolist(),
                    "slack": self.slack[b].detach().cpu().tolist(),
                    "min_slack": float(min_slack_vals[b].item()),
                    "source": "dual_bound",
                },
            ))
        return results


def _materialize_expanded_bounds(bounds: Bounds) -> Bounds:
    lin = LinearBound(A_lb=bounds.lb, b_lb=bounds.lb, A_ub=bounds.ub, b_ub=bounds.ub)
    mat = materialize_if_needed(lin)
    return Bounds(
        lb=mat.A_lb.reshape(mat.A_lb.shape[0] * mat.A_lb.shape[1], *mat.A_lb.shape[2:]).contiguous(),
        ub=mat.A_ub.reshape(mat.A_ub.shape[0] * mat.A_ub.shape[1], *mat.A_ub.shape[2:]).contiguous(),
    )


def expand_bounds_dict(
    bounds_dict: Dict[int, Bounds],
    M: int,
    *,
    materialize: bool = False,
) -> Dict[int, Bounds]:
    """Expand each batched Bounds entry from [B, *shape] to [B*M, *shape].

    Uses repeat_interleave(M, dim=0) so row b*M+j of the expanded tensor
    equals row b of the original. This aligns with SpecBatch.C.view(B, M, n_out)
    semantics: each sample b's M specs share sample b's bounds.

    All Bounds entries MUST be batched (lb.dim() >= 2); unbatched bounds are
    rejected to prevent silent broadcasting that can mask shape-alignment bugs.

    Args:
        bounds_dict: mapping layer_id -> Bounds. All entries must carry a
            leading batch dimension.
        M: number of specs per sample (must be >= 1).

    Returns:
        New dict with same keys and expanded Bounds values.

    Raises:
        ValueError: if M <= 0 or any Bounds entry is unbatched.
    """
    if M <= 0:
        raise ValueError(f"expand_bounds_dict: M must be positive, got {M}")
    if M == 1:
        return dict(bounds_dict)
    for lid, bounds in bounds_dict.items():
        if bounds.lb.dim() < 2:
            raise ValueError(
                f"expand_bounds_dict: layer {lid} bounds must be batched "
                f"[B, *shape], got dim={bounds.lb.dim()} shape={tuple(bounds.lb.shape)}"
            )
    expanded = cast(dict[int, Bounds], expand_rank3(bounds_dict, M))
    if not materialize:
        return expanded
    return {lid: _materialize_expanded_bounds(bounds) for lid, bounds in expanded.items()}


def _build_linear_le(
    spec: OutputSpec, B: int, n_out: int,
    device: torch.device, dtype: torch.dtype,
) -> SpecBatch:
    """Property: c . y <= d.  Certified iff LB(-c . y) >= -d.

    Encoded as M=1 row per sample: c_row = -c, threshold = -d.
    """
    if spec.c is None or spec.d is None:
        raise ValueError("LINEAR_LE requires spec.c and spec.d")
    c = spec.c.flatten().to(device=device, dtype=dtype)
    if c.shape[0] != n_out:
        raise ValueError(f"LINEAR_LE c length {c.shape[0]} != n_out {n_out}")
    d_val = float(spec.d.item()) if isinstance(spec.d, torch.Tensor) else float(spec.d)

    C = (-c).unsqueeze(0).expand(B, -1).contiguous()  # [B, n_out]
    thresholds = torch.full((B, 1), -d_val, device=device, dtype=dtype)
    active_mask = torch.ones(B, 1, dtype=torch.bool, device=device)
    return SpecBatch(C=C, thresholds=thresholds, active_mask=active_mask, B=B, M=1)


def _build_unsafe_linear(
    spec: OutputSpec, B: int, n_out: int,
    device: torch.device, dtype: torch.dtype,
) -> SpecBatch:
    """Property: for all rows i, c_i . y <= d_i.  Certified iff for all i: LB(-c_i . y) >= -d_i.

    Encoded as M = N rows per sample, where N = spec.c.shape[0].
    """
    if spec.c is None or spec.d is None:
        raise ValueError("UNSAFE_LINEAR requires spec.c and spec.d")
    c_mat = spec.c.to(device=device, dtype=dtype)
    if c_mat.dim() == 1:
        c_mat = c_mat.unsqueeze(0)
    N = c_mat.shape[0]
    if c_mat.shape[1] != n_out:
        raise ValueError(f"UNSAFE_LINEAR c cols {c_mat.shape[1]} != n_out {n_out}")
    d_vec = spec.d.flatten().to(device=device, dtype=dtype)
    if d_vec.shape[0] != N:
        raise ValueError(f"UNSAFE_LINEAR d length {d_vec.shape[0]} != N {N}")

    # C: [B, N, n_out] -> [B*N, n_out]
    C = (-c_mat).unsqueeze(0).expand(B, -1, -1).reshape(B * N, n_out).contiguous()
    thresholds = (-d_vec).unsqueeze(0).expand(B, -1).contiguous()  # [B, N]
    active_mask = torch.ones(B, N, dtype=torch.bool, device=device)
    return SpecBatch(C=C, thresholds=thresholds, active_mask=active_mask, B=B, M=N)


def _build_top1_robust(
    y_true: torch.Tensor, num_classes: int, B: int, n_out: int,
    device: torch.device, dtype: torch.dtype,
) -> SpecBatch:
    """Property: y[y_true] > y[j] for all j != y_true.
    Certified iff for all j != y_true: LB(y[y_true] - y[j]) >= 0.

    Encoded as M = num_classes rows per sample:
      row (b, j) = e_{y_true[b]} - e_j.
      threshold = 0.
      active_mask[b, j] = (j != y_true[b]).
    """
    if n_out != num_classes:
        raise ValueError(f"TOP1_ROBUST requires n_out == num_classes (got {n_out} vs {num_classes})")
    y_true_t = y_true.to(device=device, dtype=torch.long).view(-1)
    if y_true_t.shape[0] != B:
        raise ValueError(f"y_true length {y_true_t.shape[0]} != B {B}")
    K = num_classes
    eye = torch.eye(K, device=device, dtype=dtype)
    # C[b, j, :] = eye[y_true[b]] - eye[j]
    C_per_sample = eye[y_true_t].unsqueeze(1) - eye.unsqueeze(0)  # [B, K, K]
    C = C_per_sample.reshape(B * K, K).contiguous()
    thresholds = torch.zeros(B, K, device=device, dtype=dtype)
    class_idx = torch.arange(K, device=device).unsqueeze(0)        # [1, K]
    active_mask = (class_idx != y_true_t.unsqueeze(1))             # [B, K]
    return SpecBatch(C=C, thresholds=thresholds, active_mask=active_mask, B=B, M=K)


def _build_margin_robust(
    spec: OutputSpec, y_true: torch.Tensor, num_classes: int, B: int, n_out: int,
    device: torch.device, dtype: torch.dtype,
) -> SpecBatch:
    """Property: y[y_true] - y[j] >= margin for all j != y_true.
    Certified iff for all j != y_true: LB(y[y_true] - y[j]) >= margin.

    Identical to TOP1_ROBUST but with threshold = margin. Requires spec.margin
    to be explicitly set; use OutKind.TOP1_ROBUST for zero-margin semantics
    rather than relying on an implicit default.
    """
    if spec.margin is None:
        raise ValueError(
            "MARGIN_ROBUST requires spec.margin; use OutKind.TOP1_ROBUST for "
            "zero-margin semantics."
        )
    base = _build_top1_robust(y_true, num_classes, B, n_out, device, dtype)
    if isinstance(spec.margin, torch.Tensor):
        margin_t = spec.margin.to(device=device, dtype=dtype).reshape(-1)
    else:
        margin_t = torch.tensor([float(spec.margin)], device=device, dtype=dtype)
    if margin_t.numel() == 1:
        thresholds = torch.full_like(base.thresholds, float(margin_t.item()))
    elif margin_t.numel() == B:
        thresholds = margin_t.unsqueeze(1).expand(B, base.M).contiguous()
    else:
        raise ValueError(f"MARGIN_ROBUST margin length {margin_t.numel()} != 1 or B {B}")
    return SpecBatch(C=base.C, thresholds=thresholds, active_mask=base.active_mask, B=B, M=base.M)


def _build_range(
    spec: OutputSpec, B: int, n_out: int,
    device: torch.device, dtype: torch.dtype,
) -> SpecBatch:
    """Property: lb_i <= y_i <= ub_i for all i.
    Certified iff for all i:
      LB(y_i) >= lb_i  (row = +e_i, threshold = lb_i)
      LB(-y_i) >= -ub_i (row = -e_i, threshold = -ub_i)

    M = 2 * n_out if both lb and ub given, n_out if only one side given.
    Rows are stacked: [+e_0, -e_0, +e_1, -e_1, ...] when both sides given.
    """
    has_lb = spec.lb is not None
    has_ub = spec.ub is not None
    if not has_lb and not has_ub:
        raise ValueError("RANGE requires at least one of spec.lb / spec.ub")

    eye = torch.eye(n_out, device=device, dtype=dtype)
    rows: List[torch.Tensor] = []
    thresh_rows: List[torch.Tensor] = []

    if has_lb:
        assert spec.lb is not None  # narrowed by has_lb
        lb_vec = spec.lb.flatten().to(device=device, dtype=dtype)
        if lb_vec.shape[0] != n_out:
            raise ValueError(f"RANGE lb length {lb_vec.shape[0]} != n_out {n_out}")
        rows.append(eye)             # [n_out, n_out] rows = +e_i
        thresh_rows.append(lb_vec)   # [n_out]
    if has_ub:
        assert spec.ub is not None  # narrowed by has_ub
        ub_vec = spec.ub.flatten().to(device=device, dtype=dtype)
        if ub_vec.shape[0] != n_out:
            raise ValueError(f"RANGE ub length {ub_vec.shape[0]} != n_out {n_out}")
        rows.append(-eye)
        thresh_rows.append(-ub_vec)

    # Stack: if both, interleave so ordering is [+e_0, -e_0, +e_1, -e_1, ...]
    if has_lb and has_ub:
        # rows[0]=eye [n_out, n_out], rows[1]=-eye [n_out, n_out]
        # Want C_per_sample[i, :] for i in 0..2*n_out: e_{i//2} if i%2==0 else -e_{i//2}
        stacked = torch.stack([rows[0], rows[1]], dim=1).reshape(2 * n_out, n_out)
        thresh_stacked = torch.stack([thresh_rows[0], thresh_rows[1]], dim=1).reshape(2 * n_out)
    else:
        stacked = rows[0]
        thresh_stacked = thresh_rows[0]
    M = 2 * n_out if (has_lb and has_ub) else n_out

    # Broadcast to B: C [B*M, n_out], thresholds [B, M]
    C = stacked.unsqueeze(0).expand(B, -1, -1).reshape(B * M, n_out).contiguous()
    thresholds = thresh_stacked.unsqueeze(0).expand(B, -1).contiguous()
    active_mask = torch.ones(B, M, dtype=torch.bool, device=device)
    return SpecBatch(C=C, thresholds=thresholds, active_mask=active_mask, B=B, M=M)


def build_spec_batch(
    out_spec: OutputSpec,
    B: int,
    n_out: int,
    num_classes: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> SpecBatch:
    """Dispatch OutputSpec.kind -> SpecBatch.

    Classification-robust kinds (TOP1_ROBUST, MARGIN_ROBUST) read y_true from
    out_spec.y_true (must be set by the caller) and require num_classes to
    be passed explicitly.

    Args:
        out_spec: the OutputSpec to encode. For TOP1/MARGIN robust kinds,
            out_spec.y_true must be populated.
        B: batch size.
        n_out: output dimension of the network's ASSERT predecessor.
        num_classes: K; required for TOP1_ROBUST / MARGIN_ROBUST. Must equal
            n_out; the assertion is checked in _build_top1_robust.
        device, dtype: target tensor device/dtype (default: CPU/float32).

    Returns:
        SpecBatch with C=[B*M, n_out], thresholds=[B, M], active_mask=[B, M].

    Raises:
        NotImplementedError: if out_spec.kind is not supported by dual path.
        ValueError: if required fields missing or shapes mismatched.
    """
    if device is None:
        from act.util.device_manager import get_default_device
        device = get_default_device()
    if dtype is None:
        from act.util.device_manager import get_default_dtype
        dtype = get_default_dtype()

    y_true_tensor = getattr(out_spec, "y_true", None)
    if y_true_tensor is not None:
        y_true = y_true_tensor.reshape(-1)
        margin = out_spec.margin
        if y_true.numel() == 1 and B > 1:
            y_true = y_true.repeat(B)
            if margin is not None:
                margin = margin.reshape(-1)
                if margin.numel() == 1:
                    margin = margin.repeat(B)
            out_spec = OutputSpec(
                kind=out_spec.kind,
                c=out_spec.c,
                d=out_spec.d,
                y_true=y_true,
                margin=margin,
                lb=out_spec.lb,
                ub=out_spec.ub,
            )

    kind = out_spec.kind
    if kind == OutKind.LINEAR_LE:
        return _build_linear_le(out_spec, B, n_out, device, dtype)
    if kind == OutKind.UNSAFE_LINEAR:
        return _build_unsafe_linear(out_spec, B, n_out, device, dtype)
    if kind == OutKind.TOP1_ROBUST:
        if out_spec.y_true is None:
            raise ValueError("TOP1_ROBUST requires OutputSpec.y_true")
        if num_classes is None:
            raise ValueError("TOP1_ROBUST requires num_classes")
        return _build_top1_robust(out_spec.y_true, num_classes, B, n_out, device, dtype)
    if kind == OutKind.MARGIN_ROBUST:
        if out_spec.y_true is None:
            raise ValueError("MARGIN_ROBUST requires OutputSpec.y_true")
        if num_classes is None:
            raise ValueError("MARGIN_ROBUST requires num_classes")
        return _build_margin_robust(
            out_spec, out_spec.y_true, num_classes, B, n_out, device, dtype
        )
    if kind == OutKind.RANGE:
        return _build_range(out_spec, B, n_out, device, dtype)
    raise NotImplementedError(
        f"Unsupported OutKind for dual path: {kind}. "
        f"Supported: LINEAR_LE, UNSAFE_LINEAR, TOP1_ROBUST, MARGIN_ROBUST, RANGE."
    )
