# ===- act/back_end/bab/node.py - Subproblem Representation ---------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------====#
#
# Purpose:
#   Tensor-native representation of BaB subproblems.
#
#   SubproblemBatch is the primary data structure — every field is a tensor
#   with leading batch dimension (N, …) so that branching, bounding, and
#   (future) batched solving operate in pure tensor arithmetic.
#
#   BabNode is retained for backward compatibility with existing callers.
#
# ===---------------------------------------------------------------------====#

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from act.back_end.bab.eta import EtaState
from act.back_end.bab.trace import BoundTrace
from act.back_end.core import Bounds
from act.back_end.solver.alpha_state import AlphaState
from act.util.device_manager import get_default_device, get_default_dtype


# ---------------------------------------------------------------------------
# Split record (per-neuron branching history entry)
# ---------------------------------------------------------------------------


@dataclass
class Split:
    """Record of a single neuron split along a BaB tree path.

    Sign convention (canonical beta-CROWN, matches :class:`EtaState`):

    * ``sign = +1`` — INACTIVE side, constraint ``z ≤ split_point``
    * ``sign = -1`` — ACTIVE   side, constraint ``z ≥ split_point``

    Attributes
    ----------
    layer_id:
        Pre-activation (affine) layer id whose output ``z`` is being split.
    neuron_idx:
        Index of the neuron within ``layer_id``'s output (column index).
    sign:
        +1 (INACTIVE) or -1 (ACTIVE).
    split_point:
        Threshold ``z`` is clamped against.  ``0.0`` for ReLU; midpoint of
        the pre-activation interval for smooth activations.
    kind:
        ``'relu'`` or ``'smooth'``.
    """

    layer_id: int
    neuron_idx: int
    sign: int  # +1 INACTIVE, -1 ACTIVE
    split_point: float
    kind: str  # 'relu' or 'smooth'


# ---------------------------------------------------------------------------
# Tensor-native batch representation (primary)
# ---------------------------------------------------------------------------


@dataclass
class SubproblemBatch:
    """Batched BaB subproblems for tensor-driven processing.

    Shape convention
    ~~~~~~~~~~~~~~~~
    * ``lb``, ``ub``:       ``(N, D)``  — input-space bounds per subproblem.
    * ``depths``:           ``(N,)``    — tree depth of each subproblem.
    * ``eta``:              per-layer ``(N, D_layer)`` split-multiplier state.
    * ``histories``:        per-row list of :class:`Split` records.
    * ``parent_margins``:   ``(N,)`` worst-spec-row margin inherited from
                            the parent (used for monotonicity checks).

    The new fields (``eta`` / ``histories`` / ``parent_margins``) are all
    ``Optional`` — root construction via :meth:`from_bounds` omits them so
    existing callers stay bit-compatible.  Batched-BaB consumers should
    construct with :meth:`from_bounds_with_eta` and use
    :meth:`split_neuron_batched` to spawn children.
    """

    lb: torch.Tensor  # (N, D)  lower bounds
    ub: torch.Tensor  # (N, D)  upper bounds
    depths: torch.Tensor  # (N,)    tree depth

    # Optional per-subproblem state for neuron-space BaB with eta penalty.
    eta: Optional[EtaState] = None
    alphas: Optional[AlphaState] = None
    histories: Optional[list[list[Split]]] = None
    parent_margins: Optional[torch.Tensor] = None
    subproblem_ids: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        raw_alphas = self.__dict__.get("alphas")
        if isinstance(raw_alphas, dict):
            self.alphas = AlphaState.from_legacy(raw_alphas)

    # -- properties ---------------------------------------------------------

    @property
    def batch_size(self) -> int:
        """Number of subproblems in this batch."""
        return self.lb.shape[0]

    @property
    def input_dim(self) -> int:
        """Dimensionality of the input space."""
        return self.lb.shape[-1]

    def __len__(self) -> int:
        return self.batch_size

    # -- constructors -------------------------------------------------------

    @staticmethod
    def from_bounds(bounds: Bounds, depth: int = 0) -> SubproblemBatch:
        """Wrap a single ``Bounds`` into a batch of size 1 (flattened to (1, D))."""
        lb = bounds.lb.detach().reshape(1, -1)  # (1, D_flat)
        ub = bounds.ub.detach().reshape(1, -1)  # (1, D_flat)
        depths = torch.tensor([depth], dtype=torch.long)
        return SubproblemBatch(lb=lb, ub=ub, depths=depths)

    @staticmethod
    def from_bounds_with_eta(
        bounds: Bounds,
        layer_widths: dict[int, int],
        depth: int = 0,
        trace: Optional[BoundTrace] = None,
    ) -> SubproblemBatch:
        """Build a root batch of size 1 with zero-initialised ``EtaState``.

        Args:
            bounds:
                Input-space bounds; flattened to shape ``(1, D_flat)``.
            layer_widths:
                ``{pre_activation_layer_id: D_layer}`` — one entry per
                splittable pre-activation layer.  Eta tensors allocate a
                row of zeros per layer (sign = 0 ⇒ no split yet).
            depth:
                Initial tree depth (default 0).

        Returns:
            Fresh ``SubproblemBatch`` with eta/histories/parent_margins
            populated: eta is all zeros, histories has one empty list,
            parent_margins is ``None``.  ``eta.fast_path_skip()`` returns
            ``True`` — downstream solvers take the zero-eta fast path.
        """
        device = get_default_device()
        dtype = get_default_dtype()

        lb = bounds.lb.detach().reshape(1, -1).to(device=device, dtype=dtype)
        ub = bounds.ub.detach().reshape(1, -1).to(device=device, dtype=dtype)
        depths = torch.tensor([depth], dtype=torch.long, device=device)

        val = {lid: torch.zeros(1, d, device=device, dtype=dtype) for lid, d in layer_widths.items()}
        sign = {lid: torch.zeros(1, d, device=device, dtype=dtype) for lid, d in layer_widths.items()}
        point = {lid: torch.zeros(1, d, device=device, dtype=dtype) for lid, d in layer_widths.items()}
        eta = EtaState(val=val, sign=sign, point=point)
        subproblem_ids = None
        if trace is not None:
            root_id = trace.new_id(depth=depth)
            subproblem_ids = torch.tensor([root_id], dtype=torch.long, device=device)

        return SubproblemBatch(
            lb=lb,
            ub=ub,
            depths=depths,
            eta=eta,
            histories=[[]],
            parent_margins=None,
            subproblem_ids=subproblem_ids,
        )

    # -- conversions --------------------------------------------------------

    def to_bounds_list(self) -> list[Bounds]:
        """Convert back to a list of ``Bounds`` for solver compatibility."""
        return [Bounds(self.lb[i], self.ub[i]) for i in range(self.batch_size)]

    # -- geometry -----------------------------------------------------------

    def widths(self) -> torch.Tensor:
        """Per-dimension widths: ``(N, D)``."""
        return self.ub - self.lb

    def total_width(self) -> torch.Tensor:
        """Sum of widths per subproblem: ``(N,)``."""
        return self.widths().sum(dim=-1)

    # -- batch indexing -----------------------------------------------------

    def select(self, idx: torch.Tensor) -> SubproblemBatch:
        """Return a new batch containing the rows selected by ``idx``.

        Args:
            idx: 1-D long tensor of row indices.

        Returns:
            ``SubproblemBatch`` of batch size ``idx.numel()``.  All
            optional fields (``eta`` / ``histories`` / ``parent_margins``)
            are forwarded if non-None, indexed in lockstep.
        """
        if idx.dim() != 1:
            raise ValueError(f"select: idx must be 1-D, got shape {tuple(idx.shape)}")

        lb = self.lb.index_select(0, idx)
        ub = self.ub.index_select(0, idx)
        depths = self.depths.index_select(0, idx)

        eta = self.eta.select(idx) if self.eta is not None else None
        alphas = self.alphas.select(idx) if self.alphas is not None else None
        histories = (
            [self.histories[row_idx] for row_idx in idx.detach().cpu().tolist()]
            if self.histories is not None
            else None
        )
        parent_margins = (
            self.parent_margins.index_select(0, idx)
            if self.parent_margins is not None
            else None
        )
        subproblem_ids = (
            self.subproblem_ids.index_select(0, idx)
            if self.subproblem_ids is not None
            else None
        )

        return SubproblemBatch(
            lb=lb,
            ub=ub,
            depths=depths,
            eta=eta,
            alphas=alphas,
            histories=histories,
            parent_margins=parent_margins,
            subproblem_ids=subproblem_ids,
        )

    # -- neuron splitting (batched) -----------------------------------------

    def split_neuron_batched(
        self,
        decisions: list[tuple[int, int, str]],
        per_row_split_points: Optional[torch.Tensor] = None,
    ) -> tuple[SubproblemBatch, SubproblemBatch]:
        """Split one neuron per row; return (left=INACTIVE, right=ACTIVE) children.

        Each child inherits:

        * ``lb``/``ub``:  cloned from the parent (input-space bounds
          are unchanged — splitting happens in neuron space via eta).
        * ``depths``:     ``parent.depths + 1``.
        * ``eta``:        deep-copy of parent.eta with the newly-split
          neuron's ``val`` reset to 0, ``sign`` set to ±1, ``point``
          set to ``split_point`` (0.0 for ReLU, caller-provided for
          smooth).
        * ``histories``:  parent's list copied with a new :class:`Split`
          appended, if parent.histories is non-None.
        * ``parent_margins``:  initialised to ``None``; the BaB loop is
          expected to populate this from the parent's ``min_slack``
          after calling ``split_neuron_batched``.

        Args:
            decisions:
                ``[(layer_id, neuron_idx, kind)]`` of length
                ``self.batch_size``.  ``kind`` is ``'relu'`` or
                ``'smooth'``.
            per_row_split_points:
                Optional shape-``(B,)`` tensor of split thresholds.
                **Required** for any row with ``kind == 'smooth'``.
                Ignored for rows with ``kind == 'relu'`` (those always
                split at ``0.0``).

        Returns:
            ``(left, right)`` — left child is the INACTIVE side
            (``sign = +1``, ``z ≤ split_point``); right is the ACTIVE
            side (``sign = -1``, ``z ≥ split_point``).

        Raises:
            AssertionError:
                If a ``'relu'`` decision targets a neuron whose
                ``eta.sign`` slot is already non-zero (ReLU neurons
                may only be split once per BaB tree path).
            ValueError:
                If ``self.eta is None``; if ``decisions`` length
                mismatches ``batch_size``; if a ``'smooth'`` decision
                is made without a corresponding entry in
                ``per_row_split_points``; or if a decision references
                a layer not present in ``eta``.
        """
        B = self.batch_size
        if len(decisions) != B:
            raise ValueError(
                f"split_neuron_batched: decisions length {len(decisions)} "
                f"!= batch size {B}"
            )
        if self.eta is None:
            raise ValueError(
                "split_neuron_batched requires self.eta to be non-None. "
                "Construct the root batch via SubproblemBatch.from_bounds_with_eta."
            )

        # Per-row smooth-kind validation + double-split check for ReLU.
        for i, (lid, nidx, kind) in enumerate(decisions):
            if kind not in ("relu", "smooth"):
                raise ValueError(
                    f"split_neuron_batched: row {i} has unknown kind={kind!r}; "
                    f"expected 'relu' or 'smooth'."
                )
            if lid not in self.eta.sign:
                raise ValueError(
                    f"split_neuron_batched: row {i} references layer_id={lid} "
                    f"not present in eta (known layer ids: {self.eta.layer_ids})."
                )
            if kind == "smooth" and per_row_split_points is None:
                raise ValueError(
                    f"split_neuron_batched: row {i} is kind='smooth' but "
                    f"per_row_split_points is None. Pass an explicit "
                    f"(B,) tensor of split points."
                )
            if kind == "relu":
                current = self.eta.sign[lid][i, nidx].item()
                if current != 0.0:
                    raise AssertionError(
                        f"split_neuron_batched: ReLU double-split detected at "
                        f"row={i}, layer_id={lid}, neuron_idx={nidx}: "
                        f"eta.sign already has non-zero value {current}. "
                        f"ReLU neurons can only be split once along a BaB path."
                    )

        # Resolve per-row split points (0.0 for ReLU, caller-provided for smooth).
        split_points: list[float] = []
        for i, (lid, nidx, kind) in enumerate(decisions):
            if kind == "relu":
                split_points.append(0.0)
            else:
                assert per_row_split_points is not None
                split_points.append(float(per_row_split_points[i].item()))

        def _build_child_eta(sign_for_new: float) -> EtaState:
            """Clone parent eta and write the per-row new-split entries."""
            assert self.eta is not None
            new_val = {lid: t.clone() for lid, t in self.eta.val.items()}
            new_sign = {lid: t.clone() for lid, t in self.eta.sign.items()}
            new_point = {lid: t.clone() for lid, t in self.eta.point.items()}
            for i, (lid, nidx, _kind) in enumerate(decisions):
                # Newly-split neuron starts at val=0 (warm-start for OTHER
                # neurons is preserved via the clone above).
                new_val[lid][i, nidx] = 0.0
                new_sign[lid][i, nidx] = sign_for_new
                new_point[lid][i, nidx] = split_points[i]
            return EtaState(val=new_val, sign=new_sign, point=new_point)

        left_eta = _build_child_eta(+1.0)  # INACTIVE: z ≤ split_point
        right_eta = _build_child_eta(-1.0)  # ACTIVE:   z ≥ split_point

        # Propagate parent's Adam-optimised α wholesale to both children.
        # Unlike η, α has no structural change at a split — it's a per-ReLU
        # relaxation slope that remains meaningful as bounds tighten. Copy
        # (not share) so each child can re-optimise independently in Adam.
        if self.alphas is not None:
            left_alphas = self.alphas.clone()
            right_alphas = self.alphas.clone()
        else:
            left_alphas = None
            right_alphas = None

        new_depths = self.depths + 1

        # Extend per-row history lists (copy parent's, append new Split).
        if self.histories is not None:
            left_histories: Optional[list[list[Split]]] = []
            right_histories: Optional[list[list[Split]]] = []
            for i, (lid, nidx, kind) in enumerate(decisions):
                parent_hist = self.histories[i]
                left_histories.append(
                    list(parent_hist)
                    + [
                        Split(
                            layer_id=lid,
                            neuron_idx=nidx,
                            sign=+1,
                            split_point=split_points[i],
                            kind=kind,
                        )
                    ]
                )
                right_histories.append(
                    list(parent_hist)
                    + [
                        Split(
                            layer_id=lid,
                            neuron_idx=nidx,
                            sign=-1,
                            split_point=split_points[i],
                            kind=kind,
                        )
                    ]
                )
        else:
            left_histories = None
            right_histories = None

        # Input-space bounds are unchanged — BaB now lives in neuron space.
        left = SubproblemBatch(
            lb=self.lb.clone(),
            ub=self.ub.clone(),
            depths=new_depths.clone(),
            eta=left_eta,
            alphas=left_alphas,
            histories=left_histories,
            parent_margins=None,
            subproblem_ids=None,
        )
        right = SubproblemBatch(
            lb=self.lb.clone(),
            ub=self.ub.clone(),
            depths=new_depths.clone(),
            eta=right_eta,
            alphas=right_alphas,
            histories=right_histories,
            parent_margins=None,
            subproblem_ids=None,
        )
        return left, right


# ---------------------------------------------------------------------------
# Batch concatenation helper (used by BFSBounding.pop across queue entries)
# ---------------------------------------------------------------------------


def _concat_subproblem_batches(*batches: SubproblemBatch) -> SubproblemBatch:
    """Concatenate multiple :class:`SubproblemBatch` into one along dim 0.

    Each field is handled per-type:

    * ``lb``, ``ub``, ``depths``  — ``torch.cat`` along dim 0.
    * ``eta`` — only populated if ALL inputs have eta; layer ids must match;
      per-layer tensors are concatenated along dim 0.
    * ``histories`` — if ANY input has histories, the result has histories:
      inputs with ``histories is None`` contribute ``[[]]*N`` empty lists
      so row counts align.
    * ``parent_margins`` — only populated if ALL inputs have parent_margins.

    Single-element input is returned unchanged.
    """
    if len(batches) == 0:
        raise ValueError("_concat_subproblem_batches: no batches given")
    if len(batches) == 1:
        return batches[0]

    lb = torch.cat([b.lb for b in batches], dim=0)
    ub = torch.cat([b.ub for b in batches], dim=0)
    depths = torch.cat([b.depths for b in batches], dim=0)

    # eta — all-or-nothing to keep layer structure consistent.
    if all(b.eta is not None for b in batches):
        eta_batches = [b.eta for b in batches if b.eta is not None]
        layer_ids = eta_batches[0].layer_ids
        for b in batches[1:]:
            assert b.eta is not None
            if b.eta.layer_ids != layer_ids:
                raise ValueError(
                    f"_concat_subproblem_batches: eta layer id mismatch "
                    f"{b.eta.layer_ids} vs {layer_ids}"
                )
        new_val = {
            lid: torch.cat([eta.val[lid] for eta in eta_batches], dim=0)
            for lid in layer_ids
        }
        new_sign = {
            lid: torch.cat([eta.sign[lid] for eta in eta_batches], dim=0)
            for lid in layer_ids
        }
        new_point = {
            lid: torch.cat([eta.point[lid] for eta in eta_batches], dim=0)
            for lid in layer_ids
        }
        eta = EtaState(val=new_val, sign=new_sign, point=new_point)
    else:
        eta = None

    # histories — upgrade None to empty lists so row counts line up.
    if any(b.histories is not None for b in batches):
        histories: Optional[list[list[Split]]] = []
        for b in batches:
            if b.histories is None:
                histories.extend([[] for _ in range(b.batch_size)])
            else:
                histories.extend(b.histories)
    else:
        histories = None

    # parent_margins — all-or-nothing, same reasoning as eta.
    if all(b.parent_margins is not None for b in batches):
        margin_batches = [b.parent_margins for b in batches if b.parent_margins is not None]
        parent_margins = torch.cat(margin_batches, dim=0)
    else:
        parent_margins = None

    if all(b.subproblem_ids is not None for b in batches):
        subproblem_id_batches = [b.subproblem_ids for b in batches if b.subproblem_ids is not None]
        subproblem_ids = torch.cat(subproblem_id_batches, dim=0)
    else:
        subproblem_ids = None

    return SubproblemBatch(
        lb=lb,
        ub=ub,
        depths=depths,
        eta=eta,
        histories=histories,
        parent_margins=parent_margins,
        subproblem_ids=subproblem_ids,
    )


# ---------------------------------------------------------------------------
# Batch splitting (input-space, tensor-native) — kept for backward compat
# ---------------------------------------------------------------------------


def split_subproblems(
    batch: SubproblemBatch,
    split_dims: torch.Tensor,
) -> tuple[SubproblemBatch, SubproblemBatch]:
    """Bisect each subproblem along the chosen input dimension.

    This is a pure tensor operation — no Python loops over the batch.

    Args:
        batch:      ``(N, D)`` subproblems.
        split_dims: ``(N,)`` long tensor — dimension to bisect per subproblem.

    Returns:
        ``(left, right)`` — two ``SubproblemBatch`` of the same shape,
        where ``left.ub[i, d] == right.lb[i, d] == midpoint``.
    """
    mid = (batch.lb + batch.ub) / 2  # (N, D)
    split_vals = mid.gather(1, split_dims.unsqueeze(1))  # (N, 1)

    # Left child: upper bound clamped at midpoint
    left_ub = batch.ub.clone()
    left_ub.scatter_(1, split_dims.unsqueeze(1), split_vals)

    # Right child: lower bound raised to midpoint
    right_lb = batch.lb.clone()
    right_lb.scatter_(1, split_dims.unsqueeze(1), split_vals)

    new_depths = batch.depths + 1

    left = SubproblemBatch(
        lb=batch.lb.clone(),
        ub=left_ub,
        depths=new_depths.clone(),
    )
    right = SubproblemBatch(
        lb=right_lb,
        ub=batch.ub.clone(),
        depths=new_depths.clone(),
    )
    return left, right


# ---------------------------------------------------------------------------
# Legacy compat
# ---------------------------------------------------------------------------


@dataclass
class BabNode:
    """Legacy single-node representation (backward compatibility).

    New code should prefer :class:`SubproblemBatch`.
    """

    box: Bounds
    depth: int
    score: float
    candidate_ce: Optional[np.ndarray] = None

    def __lt__(self, other: BabNode) -> bool:  # max-heap by score
        return self.score > other.score

    def to_batch(self) -> SubproblemBatch:
        """Upgrade to tensor batch of size 1."""
        return SubproblemBatch.from_bounds(self.box, depth=self.depth)
