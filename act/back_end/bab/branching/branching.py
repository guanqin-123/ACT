# ===- act/back_end/bab/branching/branching.py - Branching ---------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------====#
#
# Purpose:
#   Branching strategies for Branch-and-Bound.
#
#   A branching strategy decides *which dimension (or neuron) to split* for each
#   subproblem in a batch. Two paradigms are supported:
#     1. Input splitting  — bisect (or N-ary section) the input domain along a dim.
#     2. Neuron splitting  — fix an unstable ReLU's activation phase (on/off).
#
#   Joint multi-neuron splitting (``_multi_split_from_decision``): split each lane's
#   top-k BaBSR-scored neurons together into all 2^k sign combinations (one "skip"
#   instead of k greedy single splits); gains are super-additive.
#     * "Mining Verdict Boundaries for Neural Network Verification",
#       Jiawei Ren, Guanqin Zhang, Zhenya Zhang, Yulei Sui, FM 2026.
#
#   Strategies (subclasses of ``BranchingStrategy``):
#     * ``RandomBranching`` — uniform-random over eligible dims (width-weighted for
#       input splits; masked to unstable neurons for neuron splits). Baseline.
#     * ``BaBSRBranching`` — BaBSR score ``|bias_term + intercept_term|`` over candidate
#       neurons from the dual ν/bounds (relaxation intercept ``(-l)·u/(u-l)`` weighted by
#       ν, with the pre-activation-bias term). Intercept-only backup
#       ``(-l)·u/(u-l)·clamp(-ν,0)``; width-based fallback when ν/bounds are absent.
#       "Branch and Bound for Piecewise Linear Neural Network Verification",
#       Bunel et al., JMLR 2020.
#     * ``FSBBranching`` — Filtered Smart Branching; extends BaBSR by re-scoring the
#       top candidates with the dual solver and picking the best bound improvement.
#       "Improved Branch and Bound for Neural Network Verification via Lagrangian
#       Decomposition", De Palma et al., 2021.
#
#   Result types: ``BranchingScores`` (per-dim / per-neuron scores) and
#   ``SplitDecision`` (``input_axis`` / ``cut_dim`` + ``fanout`` for input splits;
#   ``layer_id`` / ``neuron_idx`` for neuron splits). Factory:
#   ``_build_branching_strategy``.
#
#   All tensor shapes follow the (N, D) batch convention for batch-parallel scoring.
#
# ===---------------------------------------------------------------------====#

from __future__ import annotations

import bisect
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from act.back_end.bab.node import SubproblemBatch
from act.back_end.core import Bounds, Layer, Net
from act.back_end.layer_schema import LayerKind
from act.front_end.specs import InKind, OutKind
from act.util.device_manager import get_default_device, get_default_dtype


# ---------------------------------------------------------------------------
# Branching result types
# ---------------------------------------------------------------------------


@dataclass
class BranchingScores:
    flat: Optional[torch.Tensor] = None
    per_layer: Optional[Dict[int, torch.Tensor]] = None
    intercept_per_layer: Optional[Dict[int, torch.Tensor]] = None
    input_fallback: Optional[torch.Tensor] = None
    """Width-masked input-axis scores kept alongside neuron scores, so a lane
    with no splittable neuron falls back to a *wide* input dimension instead of
    a hard-coded dim 0 (zero width on any sparse-perturbation benchmark)."""


@dataclass
class SplitDecision:
    kind: str
    input_axis: Optional[torch.Tensor | int] = None
    cut_dim: Optional[torch.Tensor] = None
    fanout: int = 2
    layer_id: Optional[torch.Tensor] = None
    neuron_idx: Optional[torch.Tensor] = None


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BranchingStrategy(ABC):
    """Abstract branching strategy for Branch-and-Bound.

    Lifecycle (called by the BaB engine per iteration)::

        scores     = strategy.compute_scores(batch, net, unstable_mask)
        split_dims = strategy.select(scores)
        left, right = split_subproblems(batch, split_dims)

    Subclass contract
    ~~~~~~~~~~~~~~~~~
    * ``compute_scores`` **must** return ``(N, D)`` float tensor.
    * Dimensions that must not be split should receive score ``-inf``
      (or ``0`` when ``select`` uses ``argmax``).
    * ``select`` defaults to row-wise ``argmax``; override for
      stochastic or top-k selection.
    """

    @abstractmethod
    def compute_scores(
        self,
        batch: SubproblemBatch,
        net: Net,
        unstable_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | BranchingScores:
        """Score every candidate split dimension.

        Args:
            batch:          Current subproblems ``(N, D)``.
            net:            ACT network (for layer structure / neuron info).
            unstable_mask:  ``(D,)`` or ``(N, D)`` bool tensor marking
                            neurons eligible for splitting.
                            ``None`` ⇒ all dimensions are candidates
                            (input-split mode).

        Returns:
            ``(N, D)`` float tensor — higher score = better split.
        """
        ...

    def select(self, scores: torch.Tensor | BranchingScores) -> torch.Tensor | SplitDecision:
        """Pick one split dimension per subproblem.

        Default implementation: deterministic ``argmax`` per row.
        Override for stochastic or multi-split selection.

        Args:
            scores: ``(N, D)`` branching scores.

        Returns:
            ``(N,)`` long tensor of selected dimension indices.
        """
        if isinstance(scores, BranchingScores):
            if scores.flat is None:
                raise ValueError("Base BranchingStrategy requires flat scores")
            scores = scores.flat
        return _argmax_splittable(scores)


def _argmax_splittable(scores: torch.Tensor) -> torch.Tensor:
    """Row-wise ``argmax`` that refuses a masked-out (``-inf``) winner.

    Input-axis scorers mask zero-width dimensions to ``-inf``. A lane whose best
    score is still ``-inf`` has nothing to split: bisecting a zero-width
    dimension yields two children identical to the parent, so the frontier keeps
    re-expanding them and BaB livelocks. Failing loudly is the sound option —
    silently emitting no children would let ``bab.py`` drain the pool and report
    CERTIFIED.
    """
    best = scores.argmax(dim=-1)
    best_score = scores.gather(-1, best.unsqueeze(-1)).squeeze(-1)
    if not bool(torch.isfinite(best_score).all().item()):
        raise ValueError(
            "no splittable input dimension: every candidate dimension is "
            "zero-width or masked out in at least one lane"
        )
    return best


# ---------------------------------------------------------------------------
# Random baseline
# ---------------------------------------------------------------------------


class RandomBranching(BranchingStrategy):
    """Uniform-random branching over eligible dimensions.

    Supports both paradigms:

    * **Input split** (``unstable_mask is None``):
      Random scores weighted by domain width — wider dimensions are more
      likely to be chosen, and zero-width dimensions are excluded.

    * **Neuron split** (``unstable_mask`` provided):
      Uniform-random scores masked to unstable neurons.  Stable neurons
      receive score 0 and are never selected.
    """

    def compute_scores(
        self,
        batch: SubproblemBatch,
        net: Net,
        unstable_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        N, D = batch.batch_size, batch.input_dim
        device = batch.lb.device

        if unstable_mask is not None:
            # Neuron-split mode: zero-out stable neurons
            scores = torch.rand(N, D, device=device)
            mask = unstable_mask.float()
            if mask.dim() == 1:
                mask = mask.unsqueeze(0).expand(N, -1)  # (D,) → (N, D)
            scores = scores * mask
        else:
            embedding_mask = _perturbed_embedding_input_mask(net, batch)
            widths = batch.widths()  # (N, D)
            if embedding_mask is None:
                scores = torch.rand(N, D, device=device)
            else:
                mask = embedding_mask.unsqueeze(0).expand(N, -1)
                scores = widths.masked_fill(~mask, float("-inf"))
            scores = scores.masked_fill(widths <= 0, float("-inf"))

        return scores


class InputBranching(BranchingStrategy):
    """Deterministic widest-dimension input branching.

    Scores each input dimension by its current domain width, so ``select``
    bisects the widest side of the box - the classic input-split rule for
    low-dimensional domains (e.g. ACAS Xu), where halving the dominant side
    maximizes the worst-case per-child bound tightening.
    """

    def compute_scores(
        self,
        batch: SubproblemBatch,
        net: Net,
        unstable_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        widths = batch.widths()
        scores = widths.masked_fill(widths <= 0, float("-inf"))
        embedding_mask = _perturbed_embedding_input_mask(net, batch)
        if embedding_mask is not None:
            mask = embedding_mask.unsqueeze(0).expand_as(widths)
            return scores.masked_fill(~mask, float("-inf"))
        return scores


# ---------------------------------------------------------------------------
# Score-based branching (width-weighted, optional slope-gradient scoring)
# ---------------------------------------------------------------------------


class BaBSRBranching(BranchingStrategy):
    """BaBSR neuron-split scoring (Bunel et al., 1909.06588).

    For each ambiguous ReLU (``l<0<u``) with backward coefficient ν and pre-activation
    bias ``b``, scores the estimated bound improvement of splitting it as
    ``|bias_term + intercept_term|`` where ``s = u/(u-l)``,
    ``bias_term = max(b·ν·(s-1), b·ν·s)`` and ``intercept_term = clamp(ν, max=0)·(-l)·u/(u-l)``.
    An intercept-only score (``(-l)·u/(u-l)·clamp(-ν, 0)``) is kept as the backup.
    When ν / intermediate bounds are unavailable it falls back to width-based scores
    compatible with ``RandomBranching``.
    """

    def __init__(
        self,
        decision_threshold: float = 1e-3,
        intercept_fallback_max: int = 2,
        sparsest_layer: Optional[int] = None,
    ) -> None:
        self.decision_threshold = decision_threshold
        self.intercept_fallback_max = intercept_fallback_max
        self.sparsest_layer = sparsest_layer
        self.icp_score_counter = 0

    def compute_scores(
        self,
        batch: SubproblemBatch,
        net: Net,
        unstable_mask: Optional[torch.Tensor] = None,
        *,
        bounds_dict: Optional[Dict[int, Bounds]] = None,
        nu_per_layer: Optional[Dict[int, torch.Tensor]] = None,
    ) -> BranchingScores:
        if bounds_dict is None or nu_per_layer is None:
            return BranchingScores(flat=self._baseline_scores(batch, unstable_mask, net))

        per_layer: Dict[int, torch.Tensor] = {}
        intercept_per_layer: Dict[int, torch.Tensor] = {}
        for lid, bounds in bounds_dict.items():
            if self.sparsest_layer is not None and lid != self.sparsest_layer:
                continue
            if lid not in nu_per_layer:
                continue
            lb = bounds.lb.flatten(start_dim=1)
            ub = bounds.ub.flatten(start_dim=1)
            N, n_neurons = lb.shape
            ambiguous = (lb < 0) & (ub > 0)
            nu = nu_per_layer[lid]
            if nu.dim() > 2:
                nu = nu.flatten(start_dim=1)
            if nu.shape[0] == N:
                nu_view = nu.reshape(N, 1, n_neurons)
            elif nu.shape[0] % N == 0:
                nu_view = nu.reshape(N, nu.shape[0] // N, n_neurons)
            else:
                continue

            split_mask = self._already_split_mask(batch, lid, n_neurons)
            effective = ambiguous & ~split_mask
            denom = torch.clamp(ub - lb, min=torch.finfo(lb.dtype).eps)
            slope = (ub / denom).unsqueeze(1)
            relax_intercept = ((-lb) * ub / denom).unsqueeze(1)
            preact_bias = _preact_bias_of(net, lid)
            if preact_bias.numel() != n_neurons:
                preact_bias = torch.zeros(n_neurons, dtype=lb.dtype, device=lb.device)
            preact_bias = preact_bias.reshape(1, 1, -1).to(device=nu_view.device, dtype=nu_view.dtype)
            bias_term = torch.maximum(
                preact_bias * nu_view * (slope - 1.0),
                preact_bias * nu_view * slope,
            )
            intercept_term = nu_view.clamp(max=0.0) * relax_intercept
            primary = (bias_term + intercept_term).abs().mean(dim=1)
            primary = primary.masked_fill(~effective, float("-inf"))
            intercept = ((-lb) * ub / denom) * nu_view.mean(dim=1).neg().clamp(min=0.0)
            intercept = intercept.masked_fill(~effective, float("-inf"))
            per_layer[lid] = primary
            intercept_per_layer[lid] = intercept

        if not per_layer:
            return BranchingScores(flat=self._baseline_scores(batch, unstable_mask, net))
        return BranchingScores(
            flat=None,
            per_layer=per_layer,
            intercept_per_layer=intercept_per_layer,
            input_fallback=self._baseline_scores(batch, None, net),
        )

    def _baseline_scores(
        self,
        batch: SubproblemBatch,
        unstable_mask: Optional[torch.Tensor] = None,
        net: Optional[Net] = None,
    ) -> torch.Tensor:
        widths = batch.ub - batch.lb

        embedding_mask = _perturbed_embedding_input_mask(net=net, batch=batch)
        if embedding_mask is not None:
            scores = widths.masked_fill(~embedding_mask.unsqueeze(0), float("-inf"))
        elif batch.incremental_alpha is None:
            scores = widths * torch.rand_like(widths)
        else:
            scores = widths

        if unstable_mask is not None:
            mask = unstable_mask.float()
            if mask.dim() == 1:
                mask = mask.unsqueeze(0).expand(batch.batch_size, -1)
            scores = scores * mask

        # Masked last: -inf * 0 from the unstable mask would be NaN.
        return scores.masked_fill(widths <= 0, float("-inf"))

    def _already_split_mask(
        self,
        batch: SubproblemBatch,
        lid: int,
        n_neurons: int,
    ) -> torch.Tensor:
        if batch.split_signs is None or lid not in batch.split_signs:
            return torch.zeros(
                (batch.batch_size, n_neurons),
                device=batch.lb.device,
                dtype=torch.bool,
            )
        signs = batch.split_signs[lid]
        return (signs != 0).any(dim=1)

    @staticmethod
    def _input_axis_fallback(scores: BranchingScores) -> SplitDecision:
        if scores.input_fallback is None:
            raise ValueError(
                "no splittable input dimension: neuron branching fell back to an "
                "input-axis split but no width scores were recorded"
            )
        return SplitDecision(
            kind="input_axis", input_axis=_argmax_splittable(scores.input_fallback)
        )

    def select(self, scores: torch.Tensor | BranchingScores) -> torch.Tensor | SplitDecision:
        if isinstance(scores, torch.Tensor):
            return _argmax_splittable(scores)
        if scores.flat is not None:
            return SplitDecision(kind="input_axis", input_axis=_argmax_splittable(scores.flat))
        per_layer = scores.per_layer
        if not per_layer:
            return self._input_axis_fallback(scores)

        N = next(iter(per_layer.values())).shape[0]
        device = next(iter(per_layer.values())).device
        decisions_layer = torch.zeros(N, dtype=torch.long, device=device)
        decisions_neuron = torch.zeros(N, dtype=torch.long, device=device)

        for n in range(N):
            best_lid: Optional[int] = None
            best_idx = 0
            best_val = float("-inf")
            for lid, score in per_layer.items():
                val, idx = score[n].max(dim=0)
                if float(val.item()) > best_val:
                    best_val = float(val.item())
                    best_lid = lid
                    best_idx = int(idx.item())

            if best_lid is not None and best_val > self.decision_threshold:
                decisions_layer[n] = best_lid
                decisions_neuron[n] = best_idx
                self.icp_score_counter = 0
                continue

            intercept = scores.intercept_per_layer
            if intercept is not None and self.icp_score_counter < self.intercept_fallback_max:
                ic_lid: Optional[int] = None
                ic_idx = 0
                ic_val = float("-inf")
                for lid, score in intercept.items():
                    val, idx = score[n].max(dim=0)
                    if float(val.item()) > ic_val:
                        ic_val = float(val.item())
                        ic_lid = lid
                        ic_idx = int(idx.item())
                if ic_lid is not None and ic_val > float("-inf"):
                    decisions_layer[n] = ic_lid
                    decisions_neuron[n] = ic_idx
                    self.icp_score_counter += 1
                    continue

            self.icp_score_counter = 0
            return self._input_axis_fallback(scores)

        return SplitDecision(kind="neuron", layer_id=decisions_layer, neuron_idx=decisions_neuron)


_WITNESS_GAP_TOL = 1e-6


class WitnessResidualBranching(BaBSRBranching):
    """Branch on the realized ReLU relaxation gap at the dual witness.

    For an unstable ReLU with pre-activation interval ``[l, u]`` and witness
    pre-activation ``z``, the triangle upper relaxation is
    ``u / (u - l) * (z - l)``.  The score is the realized residual magnitude
    ``Relaxed(z) - ReLU(z)`` masked to currently unstable, unsplit neurons.
    Missing witness data falls back to BaBSR so the strategy remains usable on
    paths whose solver does not expose an SCE.
    """

    def __init__(self, decision_threshold: float = 1e-12) -> None:
        super().__init__(decision_threshold=decision_threshold)
        self.fallback_count = 0
        self.different_from_babsr_count = 0

    track_babsr_disagreement: bool = False
    """Opt-in: compare each decision against BaBSR and count divergences.

    Off by default because it costs a second full BaBSR scoring pass on every
    branching step and produces a number the verifier never consumes. Set it on
    the class when studying how far this heuristic drifts from BaBSR.
    """

    def compute_scores(
        self,
        batch: SubproblemBatch,
        net: Net,
        unstable_mask: Optional[torch.Tensor] = None,
        *,
        bounds_dict: Optional[Dict[int, Bounds]] = None,
        nu_per_layer: Optional[Dict[int, torch.Tensor]] = None,
        witness_preact_per_layer: Optional[Dict[int, torch.Tensor]] = None,
    ) -> BranchingScores:
        if bounds_dict is None or witness_preact_per_layer is None:
            self.fallback_count += int(batch.batch_size)
            return super().compute_scores(
                batch,
                net,
                unstable_mask,
                bounds_dict=bounds_dict,
                nu_per_layer=nu_per_layer,
            )

        per_layer: Dict[int, torch.Tensor] = {}
        for lid, bounds in bounds_dict.items():
            z_raw = witness_preact_per_layer.get(lid)
            if z_raw is None:
                continue
            lb = bounds.lb.flatten(start_dim=1)
            ub = bounds.ub.flatten(start_dim=1)
            z = z_raw.flatten(start_dim=1).to(device=lb.device, dtype=lb.dtype)
            n_neurons = min(lb.shape[-1], ub.shape[-1], z.shape[-1])
            if n_neurons <= 0 or lb.shape[0] != batch.batch_size or z.shape[0] != batch.batch_size:
                continue
            lb_n = lb[:, :n_neurons]
            ub_n = ub[:, :n_neurons]
            z_n = z[:, :n_neurons]
            ambiguous = (lb_n < 0) & (ub_n > 0)
            split_mask = self._already_split_mask(batch, lid, n_neurons)
            effective = ambiguous & ~split_mask
            denom = torch.clamp(ub_n - lb_n, min=torch.finfo(lb_n.dtype).eps)
            relaxed = ub_n / denom * (z_n - lb_n)
            relu = z_n.clamp(min=0.0)
            # On z in [l, u] the triangle relaxation dominates ReLU, so the gap
            # is non-negative. A negative gap means the witness pre-activation
            # escaped this layer's own bounds (stale bounds, or a witness from a
            # different subproblem). clamp(min=0) neutralises it; abs() would
            # instead turn the violation into a HIGH score and steer branching
            # straight at the corrupted neuron.
            gap = relaxed - relu
            assert not bool(
                (gap[effective] < -_WITNESS_GAP_TOL).any().item()
            ), f"witness pre-activation outside layer {lid} bounds"
            score = gap.clamp(min=0.0).masked_fill(~effective, float("-inf"))
            per_layer[lid] = score

        if not per_layer or not any(bool(torch.isfinite(score).any().item()) for score in per_layer.values()):
            self.fallback_count += int(batch.batch_size)
            return super().compute_scores(
                batch,
                net,
                unstable_mask,
                bounds_dict=bounds_dict,
                nu_per_layer=nu_per_layer,
            )

        if self.track_babsr_disagreement:
            self._record_babsr_disagreement(
                batch, net, unstable_mask, bounds_dict, nu_per_layer, per_layer
            )
        return BranchingScores(
            flat=None,
            per_layer=per_layer,
            intercept_per_layer=None,
            input_fallback=self._baseline_scores(batch, None, net),
        )

    def _record_babsr_disagreement(
        self,
        batch: SubproblemBatch,
        net: Net,
        unstable_mask: Optional[torch.Tensor],
        bounds_dict: Dict[int, Bounds],
        nu_per_layer: Optional[Dict[int, torch.Tensor]],
        witness_scores: Dict[int, torch.Tensor],
    ) -> None:
        if nu_per_layer is None:
            return
        babsr = super().compute_scores(
            batch,
            net,
            unstable_mask,
            bounds_dict=bounds_dict,
            nu_per_layer=nu_per_layer,
        )
        if babsr.per_layer is None:
            return
        wr_choice, wr_valid = self._best_pairs(witness_scores)
        babsr_choice, babsr_valid = self._best_pairs(babsr.per_layer)
        if wr_choice is None or babsr_choice is None:
            return
        both_valid = wr_valid & babsr_valid
        differ = (wr_choice != babsr_choice).any(dim=1) & both_valid
        self.different_from_babsr_count += int(differ.sum().item())

    def _best_pairs(
        self, scores: Dict[int, torch.Tensor]
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """Argmax over the concatenated (layer, neuron) axis, one row per lane.

        Returns ``([K, 2]`` tensor of ``(layer_id, neuron_idx)``, ``[K]`` validity
        mask). Concatenating first keeps this to a single ``argmax`` and one host
        sync, instead of a ``.item()`` per lane per layer.
        """
        if not scores:
            return None, torch.zeros(0, dtype=torch.bool)
        layer_ids = sorted(scores)
        flat = torch.cat([scores[lid] for lid in layer_ids], dim=1)
        widths = [scores[lid].shape[1] for lid in layer_ids]
        offsets = torch.tensor(
            [0, *widths[:-1]], device=flat.device, dtype=torch.long
        ).cumsum(0)
        owner = torch.repeat_interleave(
            torch.tensor(layer_ids, device=flat.device, dtype=torch.long),
            torch.tensor(widths, device=flat.device, dtype=torch.long),
        )
        start = torch.repeat_interleave(
            offsets, torch.tensor(widths, device=flat.device, dtype=torch.long)
        )
        best_val, best_col = flat.max(dim=1)
        pairs = torch.stack([owner[best_col], best_col - start[best_col]], dim=1)
        return pairs, torch.isfinite(best_val)


def _preact_bias_of(net: Net, lid: int) -> torch.Tensor:
    layer = net.by_id[lid]
    n_neurons = len(layer.out_vars)
    preds = net.preds.get(lid, [])
    if preds:
        pred = net.by_id[preds[0]]
        bias = pred.params.get("bias")
        if isinstance(bias, torch.Tensor):
            return bias.detach().clone()
        for value in pred.params.values():
            if isinstance(value, torch.Tensor):
                return torch.zeros(n_neurons, dtype=value.dtype, device=value.device)
    dtype = get_default_dtype()
    device = get_default_device()
    for value in layer.params.values():
        if isinstance(value, torch.Tensor):
            dtype = value.dtype
            device = value.device
            break
    return torch.zeros(n_neurons, dtype=dtype, device=device)


def _perturbed_embedding_input_mask(net: Optional[Net], batch: SubproblemBatch) -> Optional[torch.Tensor]:
    """Flat input mask for LP_EMBEDDING perturbed token coordinates.

    The mask is stored on the batch after first discovery so fallback input
    branching can still reuse it when a strategy helper lacks the net handle.
    """
    cached = getattr(batch, "_perturbed_embedding_mask", None)
    if isinstance(cached, torch.Tensor):
        return cached.to(device=batch.lb.device, dtype=torch.bool)
    if net is None:
        return None
    for layer in net.layers:
        if (
            layer.kind != LayerKind.INPUT_SPEC.value
            or layer.params.get("kind") != InKind.LP_EMBEDDING
        ):
            continue
        center = layer.params.get("center")
        if not isinstance(center, torch.Tensor) or center.dim() < 2:
            continue
        token_count = int(center.shape[-2])
        embed_dim = int(center.shape[-1])
        from act.front_end.specs import normalize_position_mask

        positions_raw = layer.params.get("perturbed_positions")
        positions = positions_raw if isinstance(positions_raw, (torch.Tensor, list, tuple)) else None
        if (
            isinstance(positions, torch.Tensor)
            and positions.dtype == torch.bool
            and positions.numel() != token_count
            and positions.numel() % token_count == 0
        ):
            positions = positions.reshape(-1, token_count)[0]
        token_mask = normalize_position_mask(
            positions, token_count, device=batch.lb.device,
        )
        flat = token_mask.unsqueeze(-1).expand(token_count, embed_dim).reshape(-1)
        if flat.numel() < batch.input_dim:
            pad = torch.zeros(batch.input_dim - flat.numel(), device=batch.lb.device, dtype=torch.bool)
            flat = torch.cat([flat, pad], dim=0)
        elif flat.numel() > batch.input_dim:
            flat = flat[:batch.input_dim]
        setattr(batch, "_perturbed_embedding_mask", flat)
        return flat
    return None


def attention_nu_exposure_candidates(
    bounds_dict: Optional[Dict[int, Bounds]],
    nu_per_layer: Optional[Dict[int, torch.Tensor]],
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Deferred hook for future attention-neuron splitting.

    Attention BaBSR needs a stable exposure of catalytic attention ν and
    pre-activation intervals.  Phase P4 intentionally ships input splitting
    only, so this hook documents the integration point and returns no
    candidates until that exposure contract exists.
    """
    del bounds_dict, nu_per_layer
    return None


class FSBBranching(BaBSRBranching):
    def __init__(
        self,
        dual_solver: Any,
        branching_candidates: int = 3,
        decision_threshold: float = 1e-3,
        intercept_fallback_max: int = 2,
        sparsest_layer: Optional[int] = None,
    ) -> None:
        super().__init__(
            decision_threshold=decision_threshold,
            intercept_fallback_max=intercept_fallback_max,
            sparsest_layer=sparsest_layer,
        )
        self.dual_solver = dual_solver
        self.branching_candidates = branching_candidates

    def compute_scores(
        self,
        batch: SubproblemBatch,
        net: Net,
        unstable_mask: Optional[torch.Tensor] = None,
        *,
        bounds_dict: Optional[Dict[int, Bounds]] = None,
        nu_per_layer: Optional[Dict[int, torch.Tensor]] = None,
    ) -> BranchingScores:
        bsr = super().compute_scores(
            batch,
            net,
            unstable_mask,
            bounds_dict=bounds_dict,
            nu_per_layer=nu_per_layer,
        )
        if bsr.per_layer is None or not bsr.per_layer:
            return bsr

        hypothesis_list: List[Dict[int, torch.Tensor]] = []
        candidate_metadata: List[Tuple[int, torch.Tensor]] = []
        topk = self.branching_candidates
        for lid, score in bsr.per_layer.items():
            k = min(topk, score.shape[-1])
            if k > 0:
                top_values, top_idx = score.topk(k, dim=-1)
                for idx in range(top_idx.shape[-1]):
                    if not bool(torch.isfinite(top_values[:, idx]).any().item()):
                        continue
                    neuron_per_lane = top_idx[:, idx]
                    hypothesis_list.append(
                        self._clone_split_signs_with_hypothesis(batch, lid, neuron_per_lane, net)
                    )
                    candidate_metadata.append((lid, neuron_per_lane))
        if bsr.intercept_per_layer is not None:
            for lid, score in bsr.intercept_per_layer.items():
                k = min(topk, score.shape[-1])
                if k > 0:
                    top_values, top_idx = score.topk(k, dim=-1)
                    for idx in range(top_idx.shape[-1]):
                        if not bool(torch.isfinite(top_values[:, idx]).any().item()):
                            continue
                        neuron_per_lane = top_idx[:, idx]
                        hypothesis_list.append(
                            self._clone_split_signs_with_hypothesis(batch, lid, neuron_per_lane, net)
                        )
                        candidate_metadata.append((lid, neuron_per_lane))

        if not hypothesis_list:
            return bsr

        N = batch.batch_size
        baseline = batch.parent_margins
        try:
            result = self._evaluate_hypotheses(net, bounds_dict, hypothesis_list)
            stacked_margins = result.margins.to(device=batch.lb.device, dtype=batch.lb.dtype)
            if stacked_margins.dim() == 1:
                stacked_margins = stacked_margins.reshape(1, -1)
            if stacked_margins.dim() > 2:
                stacked_margins = stacked_margins.reshape(stacked_margins.shape[0], N, -1).mean(dim=-1)
            if stacked_margins.shape[0] != len(hypothesis_list):
                raise ValueError("KFSB solver returned wrong candidate dimension")
            if stacked_margins.shape[1] != N:
                stacked_margins = stacked_margins.reshape(len(hypothesis_list), N, -1).mean(dim=-1)
            improvements = (
                stacked_margins - baseline.to(stacked_margins.device).unsqueeze(0)
                if baseline is not None
                else stacked_margins
            )
        except Exception:
            improvements = self._evaluate_hypotheses_serial(
                net,
                bounds_dict,
                hypothesis_list,
                baseline,
                N,
            )

        if bool(torch.isneginf(improvements).all().item()):
            return bsr

        best_cand_idx = improvements.argmax(dim=0)
        final_per_layer: Dict[int, torch.Tensor] = {}
        for lane in range(N):
            ci = int(best_cand_idx[lane].item())
            lid, neuron_tensor = candidate_metadata[ci]
            neuron_idx = int(neuron_tensor[lane].item())
            source = bsr.per_layer.get(lid) if bsr.per_layer is not None else None
            if source is None and bsr.intercept_per_layer is not None:
                source = bsr.intercept_per_layer.get(lid)
            if source is None:
                continue
            if lid not in final_per_layer:
                final_per_layer[lid] = torch.full_like(source, float("-inf"))
            final_per_layer[lid][lane, neuron_idx] = improvements[ci, lane]

        return BranchingScores(
            flat=None,
            per_layer=final_per_layer,
            intercept_per_layer=bsr.intercept_per_layer,
            input_fallback=bsr.input_fallback,
        )

    def _evaluate_hypotheses(
        self,
        net: Net,
        bounds_dict: Optional[Dict[int, Bounds]],
        split_signs: List[Dict[int, torch.Tensor]],
    ) -> Any:
        if bounds_dict is None:
            raise ValueError("FSB hypothesis evaluation requires bounds_dict")
        c = self._zero_objective(net, bounds_dict)
        return self.dual_solver.compute_certified_bound(
            net,
            bounds_dict,
            c,
            M=1,
            split_signs=split_signs,
            optimize=False,
        )

    def _evaluate_hypotheses_serial(
        self,
        net: Net,
        bounds_dict: Optional[Dict[int, Bounds]],
        hypothesis_list: List[Dict[int, torch.Tensor]],
        baseline: Optional[torch.Tensor],
        N: int,
    ) -> torch.Tensor:
        improvements = torch.full(
            (len(hypothesis_list), N),
            float("-inf"),
            device=next(iter(hypothesis_list[0].values())).device,
            dtype=next(iter(hypothesis_list[0].values())).dtype,
        )
        for c_idx, hypo in enumerate(hypothesis_list):
            try:
                result = self._evaluate_hypothesis(net, bounds_dict, hypo)
                margins = result.margins.reshape(N, -1).mean(dim=-1).to(improvements)
                improvements[c_idx] = margins - baseline.to(margins.device) if baseline is not None else margins
            except Exception:
                # Recoverable: an individual hypothetical split may be incompatible with a legacy solver/mock.
                continue
        return improvements

    def _evaluate_hypothesis(
        self,
        net: Net,
        bounds_dict: Optional[Dict[int, Bounds]],
        split_signs: Dict[int, torch.Tensor],
    ) -> Any:
        if bounds_dict is None:
            raise ValueError("FSB hypothesis evaluation requires bounds_dict")
        c = self._zero_objective(net, bounds_dict)
        return self.dual_solver.compute_certified_bound(
            net,
            bounds_dict,
            c,
            M=1,
            split_signs=split_signs,
            optimize=False,
        )

    def _zero_objective(
        self,
        net: Net,
        bounds_dict: Dict[int, Bounds],
    ) -> torch.Tensor:
        assert_layers = [
            layer for layer in net.layers if layer.kind == LayerKind.ASSERT.value
        ]
        if assert_layers:
            assert_layer = assert_layers[-1]
            preds = net.preds.get(assert_layer.id, [])
            n_out = len(net.by_id[preds[0]].out_vars) if preds else 1
        else:
            n_out = 1
        sample = next(iter(bounds_dict.values()))
        B = sample.lb.shape[0]
        return torch.zeros((B, n_out), dtype=sample.lb.dtype, device=sample.lb.device)

    def _clone_split_signs_with_hypothesis(
        self,
        batch: SubproblemBatch,
        lid: int,
        neuron_idx_per_lane: torch.Tensor,
        net: Net,
    ) -> Dict[int, torch.Tensor]:
        hypo: Dict[int, torch.Tensor] = {}
        if batch.split_signs is not None:
            for key, value in batch.split_signs.items():
                hypo[key] = value.clone()

        n_neurons = _layer_neuron_count(net.by_id[lid])
        n_specs = 1
        if batch.incremental_alpha is not None and lid in batch.incremental_alpha:
            n_specs = int(batch.incremental_alpha[lid].shape[1])
        elif batch.incremental_eta is not None and lid in batch.incremental_eta:
            n_specs = int(batch.incremental_eta[lid].shape[1])

        if lid not in hypo:
            hypo[lid] = torch.zeros(
                (batch.batch_size, n_specs, n_neurons),
                device=batch.lb.device,
                dtype=batch.lb.dtype,
            )
        for lane in range(batch.batch_size):
            idx = int(neuron_idx_per_lane[lane].item())
            hypo[lid][lane, :, idx] = +1.0
        return hypo


# ---------------------------------------------------------------------------
# Strategy factory
# ---------------------------------------------------------------------------


def _build_branching_strategy(
    method: str,
    *,
    dual_solver: Any = None,
    branching_candidates: int = 3,
) -> BranchingStrategy:
    if method == "random":
        return RandomBranching()
    if method == "width":
        return InputBranching()
    if method == "babsr":
        return BaBSRBranching()
    if method == "witness_residual":
        return WitnessResidualBranching()
    if method == "fsb":
        if dual_solver is None:
            raise ValueError("FSB branching requires a dual_solver instance (inject via factory).")
        return FSBBranching(dual_solver=dual_solver, branching_candidates=branching_candidates)
    raise ValueError(f"Unknown branching method: {method!r}")


# ---------------------------------------------------------------------------
# Joint multi-neuron splitting (verdict-boundary)
#
#   "Mining Verdict Boundaries for Neural Network Verification"
#   Jiawei Ren, Guanqin Zhang, Zhenya Zhang, Yulei Sui
#   FM 2026
# ---------------------------------------------------------------------------


def _collect_neuron_candidates(
    branch_batch: SubproblemBatch,
    bounds_dict: Dict[int, Bounds],
    nu_per_layer: Dict[int, torch.Tensor],
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Per-lane BaBSR scores (area x |nu|) over all splittable neurons.

    Returns ``(scores, layer_ids, neuron_ids)``, each ``[K, C]`` over the
    concatenated candidate axis; stable or already-split neurons score -inf.
    """
    kb = branch_batch.batch_size
    device = branch_batch.lb.device
    cand_layers: List[torch.Tensor] = []
    cand_neurons: List[torch.Tensor] = []
    cand_scores: List[torch.Tensor] = []
    for lid, nut in nu_per_layer.items():
        b = bounds_dict.get(lid)
        if b is None:
            continue
        lb = b.lb.flatten(start_dim=1)
        ub = b.ub.flatten(start_dim=1)
        if lb.shape[0] != kb:
            continue
        n = min(lb.shape[-1], nut.shape[-1])
        amb = (lb[:, :n] < 0) & (ub[:, :n] > 0)
        already = branch_batch.split_signs.get(lid) if branch_batch.split_signs else None
        if already is not None:
            amb &= already[:, 0, :n].to(device) == 0
        area = (-lb[:, :n] * ub[:, :n] / (ub[:, :n] - lb[:, :n]).clamp(min=1e-12)).clamp(min=0)
        nv = nut.reshape(kb, -1, nut.shape[-1])[:, :, :n].abs().sum(dim=1)
        sc = torch.where(amb, area * nv, torch.full_like(area, float("-inf")))
        cand_scores.append(sc)
        cand_layers.append(torch.full((kb, n), lid, device=device, dtype=torch.long))
        cand_neurons.append(
            torch.arange(n, device=device, dtype=torch.long).expand(kb, n)
        )
    if not cand_scores:
        return None
    return (
        torch.cat(cand_scores, dim=1),
        torch.cat(cand_layers, dim=1),
        torch.cat(cand_neurons, dim=1),
    )


def enumerate_unstable_candidates(
    branch_batch: SubproblemBatch,
    bounds_dict: Optional[Dict[int, Bounds]],
    nu_per_layer: Optional[Dict[int, torch.Tensor]],
    *,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if bounds_dict is None or nu_per_layer is None:
        return []
    cand = _collect_neuron_candidates(branch_batch, bounds_dict, nu_per_layer)
    if cand is None:
        return []
    scores, layers, neurons = cand
    finite_mask = torch.isfinite(scores)
    total = int(finite_mask.sum().item())
    if total == 0 or (limit is not None and total > int(limit)):
        return []
    out: List[Dict[str, Any]] = []
    for lane in range(scores.shape[0]):
        flat_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        for col in torch.where(finite_mask[lane])[0].tolist():
            lid = int(layers[lane, col].item())
            nidx = int(neurons[lane, col].item())
            b = bounds_dict.get(lid)
            if b is not None and lid not in flat_cache:
                flat_cache[lid] = (b.lb.flatten(start_dim=1), b.ub.flatten(start_dim=1))
            lb = float(flat_cache[lid][0][lane, nidx].item()) if lid in flat_cache else 0.0
            ub = float(flat_cache[lid][1][lane, nidx].item()) if lid in flat_cache else 0.0
            denom = ub - lb
            area = float(max(0.0, (-lb * ub) / denom)) if denom > 1e-12 else 0.0
            score = float(scores[lane, col].item())
            out.append({
                "lane": lane, "layer_id": lid, "neuron_idx": nidx,
                "score": score, "lb": lb, "ub": ub,
                "nu": (score / area) if area > 1e-12 else None, "area": area,
            })
    return out


def _layer_neuron_count(layer: Layer) -> int:
    """Width of a layer's output block.

    Derived from the variable-id span rather than ``len(out_vars)`` so that the
    sign tensors stay aligned with the solver's flat variable indexing even if a
    layer ever declares a non-contiguous ``out_vars`` list.
    """
    span = int(layer.out_vars[-1] - layer.out_vars[0] + 1)
    assert span == len(layer.out_vars), (
        f"layer {layer.id} has non-contiguous out_vars: span {span} != "
        f"{len(layer.out_vars)} declared variables"
    )
    return span


def _index_subproblem_batch(
    batch: SubproblemBatch, lanes: torch.Tensor
) -> SubproblemBatch:
    """Restrict every field of ``batch`` to the given lane indices."""

    def _pick(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        return None if t is None else t.index_select(0, lanes.to(t.device))

    def _pick_dict(
        state: Optional[Dict[int, torch.Tensor]]
    ) -> Optional[Dict[int, torch.Tensor]]:
        if state is None:
            return None
        return {l: t.index_select(0, lanes.to(t.device)) for l, t in state.items()}

    return SubproblemBatch(
        lb=batch.lb.index_select(0, lanes.to(batch.lb.device)),
        ub=batch.ub.index_select(0, lanes.to(batch.ub.device)),
        depths=batch.depths.index_select(0, lanes.to(batch.depths.device)),
        incremental_alpha=_pick_dict(batch.incremental_alpha),
        incremental_eta=_pick_dict(batch.incremental_eta),
        split_signs=_pick_dict(batch.split_signs),
        parent_margins=_pick(batch.parent_margins),
        lower_bound=_pick(batch.lower_bound),
        node_id=_pick(batch.node_id),
        parent_id=_pick(batch.parent_id),
    )


def _concat_subproblem_batches(
    a: SubproblemBatch, b: SubproblemBatch
) -> SubproblemBatch:
    """Stack two child batches that were split at different depths.

    The per-layer state dicts may disagree on their key sets: a layer touched
    only by one group's selection exists there and not in the other. A missing
    key means "no neuron of this layer is constrained", which is exactly what an
    all-zero sign block encodes, so padding with zeros preserves each side's
    feasible region rather than widening or narrowing it.
    """

    def _cat(x: Optional[torch.Tensor], y: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if x is None or y is None:
            return None
        return torch.cat([x, y.to(device=x.device, dtype=x.dtype)], dim=0)

    def _cat_dict(
        x: Optional[Dict[int, torch.Tensor]],
        y: Optional[Dict[int, torch.Tensor]],
        n_x: int,
        n_y: int,
    ) -> Optional[Dict[int, torch.Tensor]]:
        if not x and not y:
            return None
        x = x or {}
        y = y or {}
        out: Dict[int, torch.Tensor] = {}
        for lid in sorted(set(x) | set(y)):
            present = x.get(lid)
            if present is None:
                present = y[lid]
            trailing = present.shape[1:]
            left = x.get(lid)
            if left is None:
                left = torch.zeros(
                    (n_x, *trailing), dtype=present.dtype, device=present.device
                )
            right = y.get(lid)
            if right is None:
                right = torch.zeros(
                    (n_y, *trailing), dtype=present.dtype, device=present.device
                )
            out[lid] = torch.cat([left, right.to(device=left.device)], dim=0)
        return out

    n_a, n_b = a.batch_size, b.batch_size
    return SubproblemBatch(
        lb=torch.cat([a.lb, b.lb.to(a.lb.device)], dim=0),
        ub=torch.cat([a.ub, b.ub.to(a.ub.device)], dim=0),
        depths=torch.cat([a.depths, b.depths.to(a.depths.device)], dim=0),
        incremental_alpha=_cat_dict(a.incremental_alpha, b.incremental_alpha, n_a, n_b),
        incremental_eta=_cat_dict(a.incremental_eta, b.incremental_eta, n_a, n_b),
        split_signs=_cat_dict(a.split_signs, b.split_signs, n_a, n_b),
        parent_margins=_cat(a.parent_margins, b.parent_margins),
        lower_bound=_cat(a.lower_bound, b.lower_bound),
        node_id=_cat(a.node_id, b.node_id),
        parent_id=_cat(a.parent_id, b.parent_id),
    )


def _multi_split_from_groups(
    batch: SubproblemBatch,
    net: Net,
    top_layers: torch.Tensor,
    top_neurons: torch.Tensor,
    k_eff: int,
) -> Tuple[SubproblemBatch, torch.Tensor]:
    # Soundness: the 2^k_eff sign-combination children exactly partition each
    # lane's region for ANY (top_layers, top_neurons) — BaBSR- or LLM-chosen.
    n_lanes = batch.batch_size
    n_children = 2 ** k_eff
    device = batch.lb.device
    parent_index = torch.arange(n_lanes, device=device).repeat(n_children)

    def _gather(state: Optional[Dict[int, torch.Tensor]]) -> Optional[Dict[int, torch.Tensor]]:
        if state is None:
            return None
        return {
            l: t.index_select(0, parent_index.to(t.device)) for l, t in state.items()
        }

    m_specs = 1
    if batch.incremental_alpha:
        m_specs = int(next(iter(batch.incremental_alpha.values())).shape[1])
    elif batch.split_signs:
        m_specs = int(next(iter(batch.split_signs.values())).shape[1])

    signs = _gather(batch.split_signs) or {}
    for lid_val in torch.unique(top_layers).tolist():
        lid_int = int(lid_val)
        layer = net.by_id[lid_int]
        n_neurons = _layer_neuron_count(layer)
        if lid_int not in signs:
            signs[lid_int] = torch.zeros(
                n_children * n_lanes, m_specs, n_neurons,
                device=device, dtype=batch.lb.dtype,
            )
        else:
            signs[lid_int] = signs[lid_int].clone()
        for bit in range(k_eff):
            lane_sel = torch.where(top_layers[:, bit] == lid_val)[0]
            if lane_sel.numel() == 0:
                continue
            neuron_sel = top_neurons[lane_sel, bit].to(device=device, dtype=torch.long)
            for j in range(n_children):
                sign_val = 1.0 if (j >> bit) & 1 else -1.0
                rows = j * n_lanes + lane_sel
                # Two index tensors separated by a `:` slice broadcast against
                # EACH OTHER to shape [len(rows), m_specs] — positional pairing
                # rows[i] <-> neuron_sel[i], not an outer product. Do not rewrite
                # as signs[lid_int][rows][:, :, neuron_sel] = ...: that indexes a
                # copy, so the assignment silently no-ops.
                signs[lid_int][rows, :, neuron_sel] = sign_val

    children = SubproblemBatch(
        lb=batch.lb.index_select(0, parent_index),
        ub=batch.ub.index_select(0, parent_index),
        depths=batch.depths.index_select(0, parent_index) + k_eff,
        incremental_alpha=_gather(batch.incremental_alpha),
        incremental_eta=_gather(batch.incremental_eta),
        split_signs=signs,
        parent_margins=(
            batch.parent_margins.index_select(0, parent_index)
            if batch.parent_margins is not None
            else None
        ),
        lower_bound=(
            batch.lower_bound.index_select(0, parent_index)
            if batch.lower_bound is not None
            else None
        ),
    )
    return children, parent_index


def _multi_split_from_decision(
    batch: SubproblemBatch,
    net: Net,
    bounds_dict: Optional[Dict[int, Bounds]],
    nu_per_layer: Optional[Dict[int, torch.Tensor]],
    k_levels: int,
) -> Optional[Tuple[SubproblemBatch, torch.Tensor]]:
    """Joint top-k neuron split: each lane emits all 2^k sign combinations.

    Verdict-boundary multi-neuron splitting — "Mining Verdict Boundaries for
    Neural Network Verification", Jiawei Ren, Guanqin Zhang, Zhenya Zhang,
    Yulei Sui, FM 2026. Candidates are scored by the BaBSR heuristic
    (``_collect_neuron_candidates``).

    The 2^k children exactly partition each lane's region (every selected
    neuron is constrained to >=0 or <=0 in both directions across the
    combination set), so replacing the lane by its children is sound. Joint
    splits are super-additive in bound gain versus greedy single splits.

    Lanes are grouped by how many finite candidates they actually have, so a
    single candidate-starved lane no longer collapses ``k_eff`` for the whole
    wave: lanes with >=2 candidates take the joint split, lanes with exactly one
    take a plain 2-child split, and the two groups are concatenated. Returns
    None only when some lane has no splittable neuron at all, since neuron
    branching cannot cover such a lane and the caller must fall back to the
    decision path (which can still split an input axis).
    """
    if bounds_dict is None or nu_per_layer is None:
        return None
    cand = _collect_neuron_candidates(batch, bounds_dict, nu_per_layer)
    if cand is None:
        return None
    all_scores, all_layers, all_neurons = cand
    finite_per_lane = torch.isfinite(all_scores).sum(dim=1)
    if int(finite_per_lane.min().item()) < 1:
        return None
    k_lane = finite_per_lane.clamp(max=k_levels)
    joint_mask = k_lane >= 2
    if not bool(joint_mask.any().item()):
        return None

    def _group(lanes: torch.Tensor, k: int) -> Tuple[SubproblemBatch, torch.Tensor]:
        sub = _index_subproblem_batch(batch, lanes)
        scores = all_scores.index_select(0, lanes)
        top = torch.topk(scores, k=k, dim=1).indices
        children, local_parent = _multi_split_from_groups(
            sub,
            net,
            all_layers.index_select(0, lanes).gather(1, top),
            all_neurons.index_select(0, lanes).gather(1, top),
            k,
        )
        return children, lanes.to(local_parent.device).index_select(0, local_parent)

    k_eff = min(k_levels, int(finite_per_lane[joint_mask].min().item()))
    lanes_joint = torch.where(joint_mask)[0]
    lanes_single = torch.where(~joint_mask)[0]
    joint_children, joint_parent = _group(lanes_joint, k_eff)
    if lanes_single.numel() == 0:
        return joint_children, joint_parent
    single_children, single_parent = _group(lanes_single, 1)
    return (
        _concat_subproblem_batches(joint_children, single_children),
        torch.cat([joint_parent, single_parent]),
    )


# ---------------------------------------------------------------------------
# CLIMB certificate reuse
# ---------------------------------------------------------------------------


LiteralKey = Tuple[int, int]

# Eq. 12 accounting runs in float64 regardless of the solver dtype: the
# deletion budget is a rounding-sensitive prefix sum against a slack that can
# be orders of magnitude smaller than the bound itself.
CLIMB_DTYPE = torch.float64
CLIMB_DELTA_ABS = 1e-9
CLIMB_DELTA_REL = 1e-7
CLIMB_PROPAGATE_CORE_CHUNK = 4096


class LiteralVocabulary:
    """Lexicographically ordered ``(ReLU layer, neuron)`` -> dense column map."""

    def __init__(self) -> None:
        self._keys: List[LiteralKey] = []
        self._index: Dict[LiteralKey, int] = {}
        self._widths: Dict[int, int] = {}

    def register_layer(self, layer_id: int, width: int) -> None:
        self._widths[layer_id] = max(width, self._widths.get(layer_id, 0))

    def add(self, key: LiteralKey) -> int:
        if key not in self._index:
            bisect.insort(self._keys, key)
            self._index = {existing: column for column, existing in enumerate(self._keys)}
        return self._index[key]

    def column(self, key: LiteralKey) -> int:
        return self._index[key]

    def key(self, index: int) -> LiteralKey:
        return self._keys[index]

    def width(self, layer_id: int) -> int:
        return self._widths[layer_id]

    def __len__(self) -> int:
        return len(self._keys)


@dataclass(frozen=True)
class PackedLiterals:
    indices: torch.Tensor
    signs: torch.Tensor
    mask: torch.Tensor
    vocabulary: LiteralVocabulary


@dataclass(frozen=True)
class Core:
    literals: frozenset[Tuple[LiteralKey, int]]
    sequence: int


@dataclass
class ClimbMetrics:
    enabled: bool = False
    generated_children: int = 0
    main_bound_row_passes: int = 0
    main_bound_calls: int = 0
    peak_frontier_rows: int = 0
    prebound_discharged: int = 0
    presplit_discharged: int = 0
    recheck_row_passes: int = 0
    recheck_calls: int = 0
    recheck_time_s: float = 0.0
    coarsen_time_s: float = 0.0
    propagate_time_s: float = 0.0
    cores_inserted: int = 0
    core_literals_original: int = 0
    core_literals_retained: int = 0
    library: Optional[CoreLibrary] = None

    def observe_frontier(self, pending_plus_active_rows: int) -> None:
        self.peak_frontier_rows = max(self.peak_frontier_rows, pending_plus_active_rows)

    def metadata(self) -> Dict[str, int | float]:
        if not self.enabled:
            return {}
        return {
            "climb_generated_children": self.generated_children,
            "climb_main_bound_row_passes": self.main_bound_row_passes,
            "climb_main_bound_calls": self.main_bound_calls,
            "climb_peak_frontier_rows": self.peak_frontier_rows,
            "climb_prebound_discharged": self.prebound_discharged,
            "climb_presplit_discharged": self.presplit_discharged,
            "climb_recheck_row_passes": self.recheck_row_passes,
            "climb_recheck_calls": self.recheck_calls,
            "climb_recheck_time_s": self.recheck_time_s,
            "climb_coarsen_time_s": self.coarsen_time_s,
            "climb_propagate_time_s": self.propagate_time_s,
            "climb_cores_inserted": self.cores_inserted,
            "climb_cores_subsumed": self.library.subsumed if self.library is not None else 0,
            "climb_cores_resolved": self.library.resolved if self.library is not None else 0,
            "climb_cores_evicted": self.library.evicted if self.library is not None else 0,
            "climb_core_literals_original": self.core_literals_original,
            "climb_core_literals_retained": self.core_literals_retained,
        }


def pack_split_matrix(
    split_signs: Optional[Dict[int, torch.Tensor]],
    n_rows: int,
    vocabulary: LiteralVocabulary,
    *,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, PackedLiterals]:
    """Pack split-state dictionaries into dense assignments and row literals."""
    target_device = (
        next(iter(split_signs.values())).device
        if split_signs
        else (device if device is not None else get_default_device())
    )
    keyed_rows: List[List[Tuple[LiteralKey, int]]] = [[] for _ in range(n_rows)]
    if split_signs:
        for layer_id in sorted(split_signs):
            values = split_signs[layer_id]
            if values.shape[0] != n_rows:
                raise ValueError("split_signs leading dimension mismatch")
            flattened = values.reshape(n_rows, values.shape[1], -1)
            first = flattened[:, :1, :]
            if not torch.equal(flattened, first.expand_as(flattened)):
                raise ValueError("CLIMB requires spec-invariant split signs")
            width = flattened.shape[-1]
            vocabulary.register_layer(layer_id, width)
            lane_values = first[:, 0, :].cpu()
            for neuron in torch.where((lane_values != 0).any(dim=0))[0].tolist():
                key = (layer_id, neuron)
                vocabulary.add(key)
                for lane in torch.where(lane_values[:, neuron] != 0)[0].tolist():
                    keyed_rows[lane].append((key, int(lane_values[lane, neuron].item())))
    # Columns are resolved only after every key is registered: adding a key
    # re-sorts the vocabulary, so earlier column numbers would go stale.
    row_literals = [
        [(vocabulary.column(key), sign) for key, sign in literals]
        for literals in keyed_rows
    ]

    dense = torch.zeros((n_rows, len(vocabulary)), dtype=torch.int8, device=target_device)
    width = max((len(row) for row in row_literals), default=0)
    indices = torch.zeros((n_rows, width), dtype=torch.long, device=target_device)
    signs = torch.zeros((n_rows, width), dtype=torch.int8, device=target_device)
    mask = torch.zeros((n_rows, width), dtype=torch.bool, device=target_device)
    for lane, literals in enumerate(row_literals):
        for slot, (column, sign) in enumerate(literals):
            dense[lane, column] = sign
            indices[lane, slot] = column
            signs[lane, slot] = sign
            mask[lane, slot] = True
    return dense, PackedLiterals(indices, signs, mask, vocabulary)


class CoreLibrary:
    """Query-local certified phase cores with subsumption and resolution."""

    def __init__(
        self, max_cores: int, vocabulary: Optional[LiteralVocabulary] = None
    ) -> None:
        if max_cores < 1:
            raise ValueError("max_cores must be positive")
        self.max_cores = max_cores
        self.vocabulary = vocabulary if vocabulary is not None else LiteralVocabulary()
        self.subsumed = 0
        self.resolved = 0
        self.evicted = 0
        # Cores are keyed by (layer, neuron), not by column, so they survive
        # the vocabulary re-sorting that happens whenever a new literal appears.
        self._cores: List[Core] = []
        self._next_sequence = 0

    @property
    def core_count(self) -> int:
        return len(self._cores)

    @property
    def cores(self) -> Tuple[Core, ...]:
        return tuple(self._cores)

    def _admit(self, literals: frozenset[Tuple[LiteralKey, int]]) -> int:
        """Insert unless subsumed; drop strict supersets. Returns removed count."""
        before = len(self._cores)
        self._cores = [core for core in self._cores if not literals < core.literals]
        self._cores.append(Core(literals, self._next_sequence))
        self._next_sequence += 1
        return before + 1 - len(self._cores)

    def _is_subsumed(self, literals: frozenset[Tuple[LiteralKey, int]]) -> bool:
        return any(core.literals <= literals for core in self._cores)

    def insert(self, packed: PackedLiterals) -> int:
        inserted = 0
        for lane in range(packed.mask.shape[0]):
            literals = frozenset(
                (
                    packed.vocabulary.key(int(packed.indices[lane, slot].item())),
                    int(packed.signs[lane, slot].item()),
                )
                for slot in torch.where(packed.mask[lane])[0].tolist()
            )
            if self._is_subsumed(literals):
                self.subsumed += 1
                continue
            self.subsumed += self._admit(literals)
            inserted += 1

        while (resolvent := self._find_resolvent()) is not None:
            self._admit(resolvent)
            self.resolved += 1

        ordered = sorted(self._cores, key=lambda core: (len(core.literals), core.sequence))
        self.evicted += max(0, len(ordered) - self.max_cores)
        self._cores = ordered[: self.max_cores]
        return inserted

    def _find_resolvent(self) -> Optional[frozenset[Tuple[LiteralKey, int]]]:
        """First pair differing only by one opposite pivot sign, if any."""
        snapshot = list(self._cores)
        for left_index, left in enumerate(snapshot):
            left_map = dict(left.literals)
            for right in snapshot[left_index + 1 :]:
                right_map = dict(right.literals)
                if left_map.keys() != right_map.keys():
                    continue
                pivots = [key for key in left_map if left_map[key] == -right_map[key]]
                if len(pivots) != 1:
                    continue
                resolvent = frozenset(
                    (key, sign) for key, sign in left_map.items() if key != pivots[0]
                )
                if not self._is_subsumed(resolvent):
                    return resolvent
        return None

    def tensors(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        width = max((len(core.literals) for core in self._cores), default=0)
        indices = torch.zeros((len(self._cores), width), dtype=torch.long, device=device)
        signs = torch.zeros((len(self._cores), width), dtype=torch.int8, device=device)
        mask = torch.zeros((len(self._cores), width), dtype=torch.bool, device=device)
        for row, core in enumerate(self._cores):
            for slot, (key, sign) in enumerate(sorted(core.literals)):
                indices[row, slot] = self.vocabulary.column(key)
                signs[row, slot] = sign
                mask[row, slot] = True
        return indices, signs, mask


def apply_literal_assignments(
    batch: SubproblemBatch,
    assignments: torch.Tensor,
    vocabulary: LiteralVocabulary,
) -> SubproblemBatch:
    """Write dense inferred phases back to a lossless subproblem batch."""
    signs = {key: value.clone() for key, value in (batch.split_signs or {}).items()}
    specs = 1
    if signs:
        specs = next(iter(signs.values())).shape[1]
    elif batch.incremental_alpha:
        specs = next(iter(batch.incremental_alpha.values())).shape[1]
    elif batch.incremental_eta:
        specs = next(iter(batch.incremental_eta.values())).shape[1]
    for column in range(assignments.shape[1]):
        layer_id, neuron = vocabulary.key(column)
        values = assignments[:, column]
        if not bool((values != 0).any().item()):
            continue
        if layer_id not in signs:
            signs[layer_id] = torch.zeros(
                batch.batch_size,
                specs,
                vocabulary.width(layer_id),
                dtype=batch.lb.dtype,
                device=batch.lb.device,
            )
        signs[layer_id][:, :, neuron] = values.to(signs[layer_id]).unsqueeze(1)
    batch.split_signs = signs or None
    return batch


@torch.no_grad()
def propagate(
    batches: Sequence[SubproblemBatch], library: CoreLibrary
) -> Tuple[SubproblemBatch, ...]:
    """Apply certified cores jointly to pending and active groups to fixpoint."""
    if not batches:
        return ()
    group_sizes = [batch.batch_size for batch in batches]
    for batch in batches:
        for layer_id, values in (batch.split_signs or {}).items():
            library.vocabulary.register_layer(layer_id, values.shape[-1])
            # torch.where yields ascending indices, so vocabulary ids are
            # assigned in the same order as a per-neuron range() scan.
            split_neurons = torch.where(
                (values != 0).reshape(-1, values.shape[-1]).any(dim=0)
            )[0].tolist()
            for neuron in split_neurons:
                library.vocabulary.add((layer_id, neuron))
    dense_parts = [
        pack_split_matrix(
            batch.split_signs,
            batch.batch_size,
            library.vocabulary,
            device=batch.lb.device,
        )[0]
        for batch in batches
    ]
    assignments = torch.cat(dense_parts, dim=0)
    if library.core_count == 0:
        return tuple(batches)

    core_indices, core_signs, core_mask = library.tensors(assignments.device)
    core_dense = torch.zeros(
        library.core_count, len(library.vocabulary), dtype=torch.int8, device=assignments.device
    )
    for row in range(core_indices.shape[0]):
        valid = core_mask[row]
        if bool(valid.any().item()):
            core_dense[row, core_indices[row, valid]] = core_signs[row, valid]
    active = torch.ones(assignments.shape[0], dtype=torch.bool, device=assignments.device)
    compute_dtype = get_default_dtype()
    core_abs = core_dense.abs()
    lengths = core_abs.sum(dim=1).to(compute_dtype)
    while bool(active.any().item()):
        active_rows = torch.where(active)[0]
        work = assignments[active]
        work_f = work.to(compute_dtype)
        work_abs_f = work.abs().to(compute_dtype)
        # One snapshot per round: every chunk reads the same `work`, so the
        # fixpoint is independent of chunk size (memory is O(N * chunk * W)).
        discharge_local = torch.zeros(work.shape[0], dtype=torch.bool, device=work.device)
        positive = torch.zeros_like(work, dtype=torch.bool)
        negative = torch.zeros_like(positive)
        for start in range(0, core_dense.shape[0], CLIMB_PROPAGATE_CORE_CHUNK):
            chunk = core_dense[start : start + CLIMB_PROPAGATE_CORE_CHUNK]
            chunk_lengths = lengths[start : start + CLIMB_PROPAGATE_CORE_CHUNK]
            product = work_f @ chunk.to(compute_dtype).T
            overlap = work_abs_f @ core_abs[start : start + CLIMB_PROPAGATE_CORE_CHUNK].to(compute_dtype).T
            discharge_local |= (product == chunk_lengths.unsqueeze(0)).any(dim=1)
            unit = (product == overlap) & (overlap == chunk_lengths.unsqueeze(0) - 1)
            for lane_local, core_local in torch.nonzero(unit, as_tuple=False).tolist():
                missing = (chunk[core_local] != 0) & (work[lane_local] == 0)
                if int(missing.sum().item()) != 1:
                    continue
                column = int(torch.where(missing)[0].item())
                proposed = -int(chunk[core_local, column].item())
                (positive if proposed > 0 else negative)[lane_local, column] = True
        if bool(discharge_local.any().item()):
            active[active_rows[discharge_local]] = False

        survivor_local = ~discharge_local
        if not bool(survivor_local.any().item()):
            continue
        survivor_rows = active_rows[survivor_local]
        positive = positive[survivor_local]
        negative = negative[survivor_local]
        conflict = (positive & negative).any(dim=1)
        if bool(conflict.any().item()):
            active[survivor_rows[conflict]] = False
        writable = ~conflict
        changed = False
        if bool(writable.any().item()):
            rows = survivor_rows[writable]
            proposals = positive[writable].to(torch.int8) - negative[writable].to(torch.int8)
            write = (assignments[rows] == 0) & (proposals != 0)
            if bool(write.any().item()):
                assignments[rows] = torch.where(write, proposals, assignments[rows])
                changed = True
        if not changed and not bool(discharge_local.any().item()) and not bool(conflict.any().item()):
            break

    result_batches: List[SubproblemBatch] = []
    offset = 0
    for batch, size in zip(batches, group_sizes):
        group_active = active[offset : offset + size]
        keep = torch.where(group_active)[0]
        restricted = _index_subproblem_batch(batch, keep)
        restricted_assignments = assignments[offset : offset + size].index_select(0, keep)
        result_batches.append(
            apply_literal_assignments(
                restricted, restricted_assignments, library.vocabulary
            )
        )
        offset += size
    return tuple(result_batches)


def is_certified(slack: torch.Tensor, out_kind: str) -> torch.Tensor:
    """Per-lane certification mask over ``[N, M]`` slack rows.

    Byte-consistent with the ``_dispatch_dual_solve`` lane statuses: an
    ``UNSAFE_LINEAR`` lane certifies when any finite row is strictly positive;
    every other kind certifies only when all rows are finite and non-negative.
    """
    if out_kind == OutKind.UNSAFE_LINEAR:
        return (torch.isfinite(slack) & (slack > 0)).any(dim=1)
    return torch.isfinite(slack).all(dim=1) & (slack >= 0).all(dim=1)


@torch.no_grad()
def certificate_replay(
    *,
    net: Net,
    batch: SubproblemBatch,
    reference_bounds: Dict[int, Bounds],
    c_rows: torch.Tensor,
    thresholds: torch.Tensor,
    m_specs: int,
    out_kind: str,
    literals: Optional[PackedLiterals] = None,
    with_costs: bool = True,
) -> Optional[tuple[torch.Tensor, Optional[torch.Tensor]]]:
    """Replay a frozen alpha/eta certificate directly through ``DualSolver``.

    Returns ``(slack, costs)`` — for ``UNSAFE_LINEAR`` both are gathered on the
    pinned strict row (``slack [N, 1]``, ``costs [N, 1, W]``); otherwise
    ``slack`` is ``[N, M]`` and ``costs`` is ``[N, M, W]``. ``costs`` is
    ``None`` when ``with_costs=False``. Returns ``None`` when the batch lacks
    replayable state or the solver emits no usable per-layer nu.
    """
    if with_costs and literals is None:
        raise ValueError("with_costs=True requires packed literals")
    if (
        batch.incremental_alpha is None
        or batch.incremental_eta is None
        or batch.split_signs is None
        or m_specs < 1
    ):
        return None
    from act.back_end.solver.solver_dual import DualSolver

    eta = {
        layer_id: value.detach().clone().clamp(min=0)
        for layer_id, value in batch.incremental_eta.items()
    }
    result = DualSolver().compute_certified_bound(
        net,
        reference_bounds,
        c_rows,
        M=m_specs,
        alpha=batch.incremental_alpha,
        eta=eta,
        split_signs=batch.split_signs,
        optimize=False,
        return_nu_per_layer=True,
        local_phase_clamp=True,
    )
    n_lanes = batch.batch_size
    slack = result.margins.reshape(n_lanes, m_specs).to(CLIMB_DTYPE)
    slack = slack - thresholds.reshape(n_lanes, m_specs).to(slack)
    pinned_rows: Optional[torch.Tensor] = None
    if out_kind == OutKind.UNSAFE_LINEAR:
        passing = torch.isfinite(slack) & (slack > 0)
        pinned_rows = passing.to(torch.int64).argmax(dim=1)
    if result.nu_per_layer is None:
        return None
    nu: Dict[int, torch.Tensor] = {}
    for layer_id, value in result.nu_per_layer.items():
        if value.shape[0] == n_lanes * m_specs:
            nu[layer_id] = value.reshape(n_lanes, m_specs, -1)
        elif value.shape[0] == n_lanes:
            nu[layer_id] = value.reshape(n_lanes, -1, value.shape[-1])
        else:
            return None
    costs: Optional[torch.Tensor] = None
    if with_costs:
        assert literals is not None
        costs = _literal_costs(
            literals, slack.shape, eta, nu, reference_bounds, pinned_rows
        )
    if pinned_rows is not None:
        slack = slack.gather(1, pinned_rows[:, None])
    return slack, costs


@torch.no_grad()
def _literal_costs(
    literals: PackedLiterals,
    slack_shape: torch.Size,
    eta: Dict[int, torch.Tensor],
    nu: Dict[int, torch.Tensor],
    bounds: Dict[int, Bounds],
    pinned_rows: Optional[torch.Tensor],
) -> torch.Tensor:
    """Full ReLU Eq. 12 costs ``(eta + relu(-nu)) * m`` in float64, ``[N, M, W]``."""
    n_lanes, width = literals.mask.shape
    m_specs = slack_shape[1]
    compute_dtype = CLIMB_DTYPE
    costs = torch.zeros(
        (n_lanes, m_specs, width),
        dtype=compute_dtype,
        device=literals.mask.device,
    )
    for lane in range(n_lanes):
        for slot in torch.where(literals.mask[lane])[0].tolist():
            column = int(literals.indices[lane, slot].item())
            layer_id, neuron = literals.vocabulary.key(column)
            if layer_id not in eta or layer_id not in nu or layer_id not in bounds:
                costs[lane, :, slot] = torch.inf
                continue
            lane_eta = eta[layer_id][lane, :, neuron].to(compute_dtype).clamp(min=0)
            lane_nu = nu[layer_id][lane, :, neuron].to(compute_dtype)
            layer_bounds = bounds[layer_id]
            lower = layer_bounds.lb.flatten(start_dim=1)[lane, neuron].to(compute_dtype)
            upper = layer_bounds.ub.flatten(start_dim=1)[lane, neuron].to(compute_dtype)
            sign = int(literals.signs[lane, slot].item())
            magnitude = (-lower).clamp(min=0) if sign > 0 else upper.clamp(min=0)
            costs[lane, :, slot] = (lane_eta + (-lane_nu).clamp(min=0)) * magnitude
    if pinned_rows is not None:
        rows = pinned_rows.to(device=costs.device, dtype=torch.long)
        costs = costs.gather(1, rows[:, None, None].expand(-1, 1, width))
    return costs


@torch.no_grad()
def coarsen(
    literals: PackedLiterals,
    costs: torch.Tensor,
    slack: torch.Tensor,
    *,
    theta: float,
    recheck_k: int,
    recheck: Callable[[PackedLiterals, torch.Tensor], torch.Tensor],
) -> PackedLiterals:
    """Alg. 2 vector-budget deletion plus at most ``recheck_k`` one-step rechecks.

    Per row ``m`` the budget is ``theta * slack_m - delta_m`` with
    ``delta_m = 1e-9 + 1e-7 * |slack_m|``; rows with ``slack_m <= delta_m`` never
    fund a deletion. Literals are ordered by ``(max_m cost_m / budget_m, column,
    slot)`` and the longest feasible prefix is deleted. A recheck retries the
    next literal in that same order for the first ``recheck_k`` lanes;
    ``recheck`` returns the ``BoolTensor[K]`` certification mask that confirms
    each candidate deletion.
    """
    if not 0.0 <= theta <= 1.0:
        raise ValueError("theta must be in [0, 1]")
    if recheck_k < 0:
        raise ValueError("recheck_k must be non-negative")
    n_lanes, width = literals.mask.shape
    if costs.shape[0] != n_lanes or costs.shape[2] != width:
        raise ValueError("cost tensor shape must be [N, M, W]")
    if slack.shape != costs.shape[:2]:
        raise ValueError("slack tensor shape must be [N, M]")
    retained = literals.mask.clone()
    next_in_cost_order: Dict[int, int] = {}
    for lane in range(n_lanes):
        valid_slots = torch.where(literals.mask[lane])[0]
        if valid_slots.numel() == 0:
            continue
        lane_costs = costs[lane].index_select(1, valid_slots).to(CLIMB_DTYPE)
        lane_slack = slack[lane].to(CLIMB_DTYPE)
        if not bool(torch.isfinite(lane_costs).all().item()) or not bool(
            torch.isfinite(lane_slack).all().item()
        ):
            continue
        delta = CLIMB_DELTA_ABS + CLIMB_DELTA_REL * lane_slack.abs()
        if bool((lane_slack <= delta).any().item()):
            continue
        budget = theta * lane_slack - delta
        denominator = budget.clamp(min=torch.finfo(CLIMB_DTYPE).tiny)
        ratios = (lane_costs / denominator[:, None]).amax(dim=0)
        ordered = sorted(
            range(valid_slots.numel()),
            key=lambda pos: (
                float(ratios[pos].item()),
                int(literals.indices[lane, valid_slots[pos]].item()),
                int(valid_slots[pos].item()),
            ),
        )
        running = torch.zeros_like(lane_slack)
        for position in ordered:
            candidate = running + lane_costs[:, position]
            slot = int(valid_slots[position].item())
            if not bool((candidate <= budget).all().item()):
                next_in_cost_order[lane] = slot
                break
            retained[lane, slot] = False
            running = candidate

    candidates: List[int] = []
    next_slots: List[int] = []
    if recheck_k > 0:
        for lane, slot in next_in_cost_order.items():
            candidates.append(lane)
            next_slots.append(slot)
            if len(candidates) == recheck_k:
                break
    if candidates:
        rows = torch.tensor(candidates, dtype=torch.long, device=literals.mask.device)
        candidate_mask = retained.index_select(0, rows).clone()
        for local, slot in enumerate(next_slots):
            candidate_mask[local, slot] = False
        candidate = PackedLiterals(
            literals.indices.index_select(0, rows),
            literals.signs.index_select(0, rows),
            candidate_mask,
            literals.vocabulary,
        )
        passing = recheck(candidate, rows)
        for local, passed in enumerate(passing.tolist()):
            if passed:
                retained[candidates[local], next_slots[local]] = False

    return PackedLiterals(literals.indices, literals.signs, retained, literals.vocabulary)
