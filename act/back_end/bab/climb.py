# ===- act/back_end/bab/climb.py - CLIMB Certificate Reuse ----------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------====#
#
# Purpose:
#   Query-local CLIMB certificate replay, core learning, and phase propagation.
#   ClimbSession owns one verification run's CLIMB state and BaB-loop hooks.
#
# ===---------------------------------------------------------------------====#

from __future__ import annotations

import bisect
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from act.back_end.bab.branching.bounding import TopKBounding
from act.back_end.bab.branching.branching import BranchingScores
from act.back_end.bab.node import (
    SubproblemBatch,
    _encode_literal_tensor,
    _relu_literal_layout,
    _state_spec_row_count,
    slice_bounds_dict,
)
from act.back_end.core import Bounds, Net
from act.back_end.solver.solver_base import SolveStatus
from act.back_end.solver.solver_dual import (
    DualBatchResult, certification_tolerance, unproven_spec_rows,
)
from act.config.config import (
    BaBConfig,
    CLIMB_SOLVER_TIER,
    ConfigError,
    DualConfig,
    NEURON_BRANCHING_METHODS,
    NO_REFINEMENT_MODE,
    TOP_K_BOUNDINGS,
)
from act.front_end.specs import OutKind, OutputSpec
from act.util.device_manager import get_default_device, get_default_dtype


# ---------------------------------------------------------------------------
# CLIMB certificate reuse
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LiteralCodec:
    """Immutable DIMACS-style codec for the network's ReLU phase literals."""

    layer_ids: Tuple[int, ...]
    widths: Tuple[int, ...]
    offsets: Tuple[int, ...]

    @classmethod
    def from_net(cls, net: Net) -> LiteralCodec:
        return cls(*_relu_literal_layout(net))

    def __post_init__(self) -> None:
        if len(self.layer_ids) != len(self.widths) or len(self.widths) != len(self.offsets):
            raise ValueError("literal codec tables must have equal lengths")
        if tuple(sorted(self.layer_ids)) != self.layer_ids:
            raise ValueError("literal codec layer ids must be increasing")
        expected = 0
        for width, offset in zip(self.widths, self.offsets):
            if width < 0 or offset != expected:
                raise ValueError("literal codec offsets must be contiguous")
            expected += width

    def width(self, layer_id: int) -> int:
        index = bisect.bisect_left(self.layer_ids, layer_id)
        if index == len(self.layer_ids) or self.layer_ids[index] != layer_id:
            raise KeyError(layer_id)
        return self.widths[index]

    def encode(self, layer_id: int, neuron: int, sign: int) -> int:
        index = bisect.bisect_left(self.layer_ids, layer_id)
        if index == len(self.layer_ids) or self.layer_ids[index] != layer_id:
            raise KeyError(layer_id)
        if neuron < 0 or neuron >= self.widths[index]:
            raise IndexError(neuron)
        if sign not in (-1, 1):
            raise ValueError("literal sign must be -1 or 1")
        return sign * (self.offsets[index] + neuron + 1)

    def decode(self, lit: int) -> tuple[int, int, int]:
        if lit == 0:
            raise ValueError("zero is literal padding")
        variable = abs(lit) - 1
        index = bisect.bisect_right(self.offsets, variable) - 1
        if index < 0 or variable >= self.offsets[index] + self.widths[index]:
            raise IndexError(variable)
        return self.layer_ids[index], variable - self.offsets[index], 1 if lit > 0 else -1

    def encode_tensor(
        self, layer_ids: torch.Tensor, neurons: torch.Tensor, signs: torch.Tensor
    ) -> torch.Tensor:
        return _encode_literal_tensor(self.layer_ids, self.offsets, layer_ids, neurons, signs)

    def decode_tensor(
        self, literals: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        variables = literals.abs() - 1
        ends = torch.tensor(
            tuple(offset + width for offset, width in zip(self.offsets, self.widths)),
            dtype=torch.long,
            device=literals.device,
        )
        offsets = torch.tensor(self.offsets, dtype=torch.long, device=literals.device)
        layer_ids = torch.tensor(self.layer_ids, dtype=torch.long, device=literals.device)
        indices = torch.searchsorted(ends, variables.clamp(min=0), right=True)
        flat_indices = indices.reshape(-1)
        return (
            layer_ids.index_select(0, flat_indices).reshape_as(literals),
            variables - offsets.index_select(0, flat_indices).reshape_as(literals),
            literals.sign(),
        )


def _canonical_rows(rows: torch.Tensor) -> torch.Tensor:
    """Sort exact signed literals by variable, with unique values and zero padding."""
    keys = rows.abs().masked_fill(rows == 0, torch.iinfo(torch.long).max)
    rows = rows.gather(1, keys.argsort(dim=1, stable=True))
    if rows.shape[1] > 1:
        repeated = (rows[:, 1:].abs() == rows[:, :-1].abs()) & (rows[:, 1:] != 0)
        if bool((repeated & (rows[:, 1:] != rows[:, :-1])).any()):
            raise ValueError("core contains complementary literals")
        rows[:, 1:] = rows[:, 1:].masked_fill(repeated, 0)
        rows = rows.gather(1, rows.abs().masked_fill(rows == 0, torch.iinfo(torch.long).max).argsort(dim=1, stable=True))
    width = int((rows != 0).sum(1).max()) if rows.shape[0] else 0
    return rows[:, :width]


def _subset_rows(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Exact pairwise subset test; callers bound both row dimensions."""
    if right.shape[1] == 0:
        return (left == 0).all(1)[:, None].expand(-1, right.shape[0])
    keys = right.abs().masked_fill(right == 0, torch.iinfo(torch.long).max)
    indices = torch.searchsorted(
        keys[None].expand(left.shape[0], -1, -1).contiguous(),
        left.abs()[:, None].expand(-1, right.shape[0], -1).contiguous(),
    ).clamp(max=right.shape[1] - 1)
    matched = right[None].expand(left.shape[0], -1, -1).gather(2, indices)
    return ((left[:, None] == 0) | (left[:, None] == matched)).all(2)


def _canonical_ranked(
    rows: torch.Tensor, ranks: Optional[torch.Tensor],
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Carry decision chronology as metadata beside canonical literal storage."""
    canonical = _canonical_rows(rows)
    if ranks is None:
        return canonical, None
    ranked = torch.zeros_like(canonical)
    if canonical.shape[1]:
        keys = canonical.abs().masked_fill(canonical == 0, torch.iinfo(torch.long).max)
        positions = torch.searchsorted(keys.contiguous(), rows.abs().contiguous()).clamp(max=canonical.shape[1] - 1)
        ranked.scatter_reduce_(1, positions, ranks.masked_fill(rows == 0, 0), reduce="amax")
    return canonical, ranked


def _counters_non_decreasing(
    before: tuple[int, int, int], after: tuple[int, int, int]
) -> bool:
    return all(current >= previous for previous, current in zip(before, after))


def _retained_literals_are_subset(
    original: torch.Tensor, retained: torch.Tensor
) -> bool:
    return original.shape == retained.shape and bool(
        ((retained == 0) | (retained == original)).all()
    )


def _only_fills_unassigned(initial: torch.Tensor, final: torch.Tensor) -> bool:
    return initial.shape == final.shape and bool(
        ((initial == 0) | (final == initial)).all()
    )


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
    coarsen_replay_fallbacks: int = 0
    retired_unknown_lanes: int = 0
    propagate_time_s: float = 0.0
    cores_inserted: int = 0
    merged_pairs: int = 0
    complement_calls: int = 0
    complement_row_passes: int = 0
    complement_time_s: float = 0.0
    steered_candidates: int = 0
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
            "climb_coarsen_replay_fallbacks": self.coarsen_replay_fallbacks,
            "climb_retired_unknown_lanes": self.retired_unknown_lanes,
            "climb_propagate_time_s": self.propagate_time_s,
            "climb_cores_inserted": self.cores_inserted,
            "climb_merged_pairs": self.merged_pairs,
            "climb_complement_calls": self.complement_calls,
            "climb_complement_row_passes": self.complement_row_passes,
            "climb_complement_time_s": self.complement_time_s,
            "climb_steered_candidates": self.steered_candidates,
            "climb_cores_subsumed": self.library.subsumed if self.library is not None else 0,
            "climb_cores_resolved": self.library.resolved if self.library is not None else 0,
            "climb_cores_evicted": self.library.evicted if self.library is not None else 0,
            "climb_core_literals_original": self.core_literals_original,
            "climb_core_literals_retained": self.core_literals_retained,
        }


def pack_split_matrix(
    split_signs: Optional[Dict[int, torch.Tensor]],
    n_rows: int,
    codec: LiteralCodec,
    *,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Pack split-state dictionaries into zero-padded signed literal rows."""
    target_device = (
        next(iter(split_signs.values())).device
        if split_signs
        else (device if device is not None else get_default_device())
    )
    if n_rows == 0:
        return torch.empty((0, 0), dtype=torch.long, device=target_device)
    row_parts: List[torch.Tensor] = []
    literal_parts: List[torch.Tensor] = []
    count_parts: List[torch.Tensor] = []
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
            if codec.width(layer_id) != width:
                raise ValueError("split_signs width does not match literal codec")
            lane_values = first[:, 0, :].to(target_device)
            positions = torch.nonzero(lane_values, as_tuple=False)
            counts = torch.bincount(positions[:, 0], minlength=n_rows)
            count_parts.append(counts)
            row_parts.append(positions[:, 0])
            literal_parts.append(
                codec.encode_tensor(
                    torch.full_like(positions[:, 0], layer_id),
                    positions[:, 1],
                    lane_values[positions[:, 0], positions[:, 1]].sign(),
                )
            )
    if not count_parts:
        return torch.zeros((n_rows, 0), dtype=torch.long, device=target_device)
    total_counts = torch.stack(count_parts).sum(dim=0)
    packed = torch.zeros(
        (n_rows, int(total_counts.max())), dtype=torch.long, device=target_device
    )
    layer_bases = torch.zeros(n_rows, dtype=torch.long, device=target_device)
    for rows, literals, counts in zip(row_parts, literal_parts, count_parts):
        starts = counts.cumsum(0) - counts
        local_slots = torch.arange(rows.numel(), device=target_device) - torch.repeat_interleave(
            starts, counts
        )
        packed[rows, layer_bases[rows] + local_slots] = literals
        layer_bases += counts
    return packed


class CoreLibrary:
    """One packed tensor of exact certified cores; comparisons use bounded tiles."""

    def __init__(self, max_cores: int, *, track_complements: bool = False) -> None:
        if max_cores < 1:
            raise ValueError("max_cores must be positive")
        self.max_cores = max_cores
        self.subsumed = 0
        self.resolved = 0
        self.evicted = 0
        self.generation = 0
        self._cores = torch.empty((0, 0), dtype=torch.long)
        self._ranks = torch.empty((0, 0), dtype=torch.long) if track_complements else None
        self._attempted = torch.empty(0, dtype=torch.bool)
        self.chunk = 128

    @property
    def core_count(self) -> int:
        return self._cores.shape[0]

    def _admit(
        self, literals: torch.Tensor, ranks: Optional[torch.Tensor] = None,
        *, attempted: bool = False,
    ) -> int:
        """Insert unless subsumed; drop strict supersets. Returns removed count."""
        keep = torch.ones(self.core_count, dtype=torch.bool, device=literals.device)
        for start in range(0, self.core_count, self.chunk):
            keep[start:start + self.chunk] = ~_subset_rows(literals, self._cores[start:start + self.chunk])[0]
        removed = int((~keep).sum())
        width = max(self._cores.shape[1], literals.shape[1])
        self._cores = torch.cat((
            torch.nn.functional.pad(self._cores[keep], (0, width - self._cores.shape[1])),
            torch.nn.functional.pad(literals, (0, width - literals.shape[1])),
        ))
        if self._ranks is not None:
            assert ranks is not None
            self._ranks = torch.cat((
                torch.nn.functional.pad(self._ranks[keep], (0, width - self._ranks.shape[1])),
                torch.nn.functional.pad(ranks, (0, width - ranks.shape[1])),
            ))
            self._attempted = torch.cat((self._attempted[keep], keep.new_tensor([attempted])))
        return removed

    def insert(self, packed: torch.Tensor, *, suppress_complements: bool = False) -> int:
        """Admit a whole batch, preserving chronological insertion counters."""
        if packed.shape[0] == 0:
            return 0
        counters_before = (self.subsumed, self.resolved, self.evicted)
        ranks = (torch.arange(1, packed.shape[1] + 1, device=packed.device)[None].expand_as(packed)
                 if self._ranks is not None else None)
        packed, ranks = _canonical_ranked(packed, ranks)
        old_count = self.core_count
        width = max(self._cores.shape[1], packed.shape[1])
        rows = torch.cat((
            torch.nn.functional.pad(self._cores.to(packed.device), (0, width - self._cores.shape[1])),
            torch.nn.functional.pad(packed, (0, width - packed.shape[1])),
        ))
        count = rows.shape[0]
        ids = torch.arange(count, device=packed.device)
        lengths = rows.ne(0).sum(1)
        keep = torch.ones(count, dtype=torch.bool, device=packed.device)
        admitted = torch.ones_like(keep)
        chunk = min(self.chunk, max(1, int((1048576 / max(1, width)) ** .5)))
        for start in range(0, count, chunk):
            stop = min(start + chunk, count)
            for other in range(0, count, chunk):
                end = min(other + chunk, count)
                if stop <= old_count and end <= old_count:
                    continue  # The existing library is already an antichain.
                subset = _subset_rows(rows[other:end], rows[start:stop])
                earlier = ids[other:end, None] < ids[None, start:stop]
                shorter = lengths[other:end, None] < lengths[None, start:stop]
                admitted[start:stop] &= ~(subset & earlier).any(0)
                keep[start:stop] &= ~(subset & (shorter | earlier)).any(0)
        inserted = int(admitted[old_count:].sum())
        # Every skipped incoming row or removed stored row increments the same
        # subsumption counter as sequential admission, including supersets that
        # were admitted earlier in this batch and removed by a later row.
        self.subsumed += count - int(keep.sum())
        self._cores = rows[keep]
        if self._ranks is not None:
            assert ranks is not None
            self._ranks = torch.cat((
                torch.nn.functional.pad(self._ranks.to(packed.device), (0, width - self._ranks.shape[1])),
                torch.nn.functional.pad(ranks, (0, width - ranks.shape[1])),
            ))[keep]
            self._attempted = torch.cat((
                self._attempted.to(packed.device),
                torch.full((packed.shape[0],), suppress_complements, dtype=torch.bool, device=packed.device),
            ))[keep]

        while (resolvent := self._find_resolvent()) is not None:
            self._admit(*resolvent, attempted=suppress_complements)
            self.resolved += 1

        order = (self._cores != 0).sum(1).argsort(stable=True)
        self.evicted += max(0, self.core_count - self.max_cores)
        self._cores = _canonical_rows(self._cores[order[:self.max_cores]])
        if self._ranks is not None:
            self._ranks = self._ranks[order[:self.max_cores], :self._cores.shape[1]]
            self._attempted = self._attempted[order[:self.max_cores]]
        self.generation += inserted + self.resolved - counters_before[1]
        counters_after = (self.subsumed, self.resolved, self.evicted)
        assert _counters_non_decreasing(counters_before, counters_after), (
            "CLIMB invariant violated: core-library counters decreased "
            f"(before={counters_before}, after={counters_after})"
        )
        return inserted

    def _find_resolvent(self) -> Optional[tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """First pair differing only by one opposite pivot sign, if any."""
        chunk = min(self.chunk, max(1, int((1048576 / max(1, self._cores.shape[1])) ** .5)))
        for i in range(0, self.core_count, chunk):
            left = self._cores[i:i + chunk]
            for j in range(i, self.core_count, chunk):
                right = self._cores[j:j + chunk]
                different = left[:, None] != right[None]
                pairs = ((left[:, None].abs() == right[None].abs()).all(2)
                         & (different.sum(2) == 1))
                positions = torch.nonzero(pairs)
                if positions.numel():
                    a, b = positions[0].unbind()
                    ranks = self._ranks[i + a:i + a + 1] if self._ranks is not None else None
                    return _canonical_ranked(left[a:a + 1].masked_fill(different[a, b][None], 0), ranks)
        return None

    def take_complements(self, limit: int, device: torch.device) -> torch.Tensor:
        """Flip each new core's deepest actual decision once; never recurse."""
        if self._ranks is None:
            return torch.empty((0, 0), dtype=torch.long, device=device)
        rows = torch.where(~self._attempted & (self._cores != 0).any(1))[0][:max(0, limit)]
        self._attempted[rows] = True
        candidates = self._cores[rows].clone()
        if rows.numel():
            pivot = self._ranks[rows].argmax(1)
            candidates[torch.arange(rows.numel(), device=rows.device), pivot] *= -1
        return candidates.to(device)

    def pack(self, device: torch.device) -> torch.Tensor:
        return self._cores.to(device)


def apply_literal_assignments(
    batch: SubproblemBatch,
    assignments: torch.Tensor,
    active_variables: torch.Tensor,
    codec: LiteralCodec,
) -> SubproblemBatch:
    """Decode compact inferred phases back to a lossless subproblem batch."""
    signs = {key: value.clone() for key, value in (batch.split_signs or {}).items()}
    specs = _state_spec_row_count(batch)
    if active_variables.numel() == 0:
        batch.split_signs = signs or None
        return batch
    positive_literals = active_variables + 1
    layer_ids, neurons, _ = codec.decode_tensor(positive_literals)
    for layer_id, width in zip(codec.layer_ids, codec.widths):
        columns = torch.where(layer_ids == layer_id)[0]
        if columns.numel() == 0:
            continue
        if layer_id not in signs:
            signs[layer_id] = torch.zeros(
                batch.batch_size,
                specs,
                width,
                dtype=batch.lb.dtype,
                device=batch.lb.device,
            )
        layer_values = assignments.index_select(1, columns)
        lane_indices, local_columns = torch.nonzero(layer_values, as_tuple=True)
        if lane_indices.numel() == 0:
            continue
        layer_neurons = neurons.index_select(0, columns).index_select(0, local_columns)
        values = layer_values[lane_indices, local_columns].to(signs[layer_id])
        signs[layer_id][lane_indices, :, layer_neurons] = values.unsqueeze(1)
    batch.split_signs = signs or None
    return batch


def _replace_literal_assignments(
    batch: SubproblemBatch, literals: torch.Tensor, codec: LiteralCodec,
) -> None:
    """Rebuild only the supplied conditions; implied phases are not free facts."""
    batch.split_signs = {lid: torch.zeros_like(signs)
                         for lid, signs in (batch.split_signs or {}).items()}
    variables = torch.unique(literals[literals != 0].abs() - 1, sorted=True)
    dense = torch.zeros((batch.batch_size, variables.numel()), dtype=torch.int8, device=literals.device)
    lanes, slots = torch.nonzero(literals, as_tuple=True)
    columns = torch.searchsorted(variables, literals[lanes, slots].abs() - 1)
    dense[lanes, columns] = literals[lanes, slots].sign().to(torch.int8)
    apply_literal_assignments(batch, dense, variables, codec)


@torch.no_grad()
def propagate(
    batches: Sequence[SubproblemBatch],
    library: CoreLibrary,
    codec: LiteralCodec,
    *,
    core_chunk: int,
) -> Tuple[Tuple[SubproblemBatch, ...], Tuple[torch.Tensor, ...]]:
    """Apply certified cores jointly to pending and active groups to fixpoint.

    ``core_chunk`` caps the cores per matmul chunk; it bounds memory only, the
    fixpoint is independent of it. Return the propagated groups and their
    original-row selectors so external bound/witness arrays stay aligned.
    """
    if core_chunk < 1:
        raise ValueError("core_chunk must be positive")
    if not batches:
        return (), ()
    group_sizes = [batch.batch_size for batch in batches]
    packed_batches = [
        pack_split_matrix(
            batch.split_signs,
            batch.batch_size,
            codec,
            device=batch.lb.device,
        )
        for batch in batches
    ]
    if library.core_count == 0:
        return tuple(batches), tuple(
            torch.arange(batch.batch_size, device=batch.lb.device) for batch in batches
        )
    device = batches[0].lb.device
    packed_cores = library.pack(device)
    batch_values = torch.cat(
        [packed[packed != 0] for packed in packed_batches]
    )
    core_values = packed_cores[packed_cores != 0]
    all_values = torch.cat((batch_values, core_values))
    active_variables, inverse = torch.unique(
        all_values.abs() - 1, sorted=True, return_inverse=True
    )
    assignments = torch.zeros(
        (sum(group_sizes), active_variables.numel()), dtype=torch.int8, device=device
    )
    offset = 0
    inverse_offset = 0
    for packed, size in zip(packed_batches, group_sizes):
        lanes, slots = torch.nonzero(packed, as_tuple=True)
        count = lanes.numel()
        assignments[offset + lanes, inverse[inverse_offset : inverse_offset + count]] = packed[
            lanes, slots
        ].sign().to(torch.int8)
        inverse_offset += count
        offset += size
    initial_assignments = assignments.clone()
    # Snapshot actual decisions before adding inferred phases. Runtime roots
    # already carry ordered slots; this also covers manually constructed batches.
    for batch, packed in zip(batches, packed_batches):
        if batch.decision_literals is None:
            batch.decision_literals = packed
    core_dense = torch.zeros(
        (library.core_count, active_variables.numel()), dtype=torch.int8, device=device
    )
    core_rows, core_slots = torch.nonzero(packed_cores, as_tuple=True)
    core_dense[core_rows, inverse[inverse_offset:]] = packed_cores[
        core_rows, core_slots
    ].sign().to(torch.int8)
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
        for start in range(0, core_dense.shape[0], core_chunk):
            chunk = core_dense[start : start + core_chunk]
            chunk_lengths = lengths[start : start + core_chunk]
            product = work_f @ chunk.to(compute_dtype).T
            overlap = work_abs_f @ core_abs[start : start + core_chunk].to(compute_dtype).T
            discharge_local |= (product == chunk_lengths.unsqueeze(0)).any(dim=1)
            unit = (product == overlap) & (overlap == chunk_lengths.unsqueeze(0) - 1)
            lane_local, core_local = torch.nonzero(unit, as_tuple=True)
            if lane_local.numel() != 0:
                missing = (chunk.index_select(0, core_local) != 0) & (
                    work.index_select(0, lane_local) == 0
                )
                valid = missing.sum(dim=1) == 1
                lane_local = lane_local[valid]
                core_local = core_local[valid]
                columns = missing[valid].to(torch.int64).argmax(dim=1)
                proposed = -chunk[core_local, columns]
                positive[lane_local[proposed > 0], columns[proposed > 0]] = True
                negative[lane_local[proposed < 0], columns[proposed < 0]] = True
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

    assert _only_fills_unassigned(initial_assignments, assignments), (
        "CLIMB invariant violated: propagation changed an existing split literal "
        f"(rows={assignments.shape[0]}, variables={assignments.shape[1]})"
    )
    result_batches: List[SubproblemBatch] = []
    kept_rows: List[torch.Tensor] = []
    offset = 0
    for batch, size in zip(batches, group_sizes):
        group_active = active[offset : offset + size]
        keep = torch.where(group_active)[0]
        kept_rows.append(keep)
        restricted = batch.select(keep)
        restricted_assignments = assignments[offset : offset + size].index_select(0, keep)
        propagated = apply_literal_assignments(
            restricted, restricted_assignments, active_variables, codec
        )
        assert propagated.batch_size <= size, (
            "CLIMB invariant violated: propagation increased a group batch size "
            f"(input_rows={size}, output_rows={propagated.batch_size})"
        )
        result_batches.append(propagated)
        offset += size
    return tuple(result_batches), tuple(kept_rows)


def is_certified(slack: torch.Tensor, out_kind: str) -> torch.Tensor:
    """Per-lane certification mask over ``[N, M]`` slack rows.

    Replay has already subtracted the shared band for closed unsafe sets.
    """
    proven = ~unproven_spec_rows(slack, out_kind)
    return proven.any(dim=1) if out_kind == OutKind.UNSAFE_LINEAR else proven.all(dim=1)


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
    literals: Optional[torch.Tensor] = None,
    codec: Optional[LiteralCodec] = None,
    with_costs: bool = True,
    base_reference_bounds: Optional[Dict[int, Bounds]] = None,
    refresh_mode: str = "none",
    lin_support_refinement: bool = True,
    refresh_rows_cap: int = 64,
    forward_lin_max_perturbed: Optional[int] = None,
) -> Optional[tuple[torch.Tensor, Optional[torch.Tensor]]]:
    """Replay a frozen alpha/eta certificate directly through ``DualSolver``.

    Returns ``(slack, costs)`` — for ``UNSAFE_LINEAR`` both are gathered on the
    pinned strict row (``slack [N, 1]``, ``costs [N, 1, W]``); otherwise
    ``slack`` is ``[N, M]`` and ``costs`` is ``[N, M, W]``. ``costs`` is
    ``None`` when ``with_costs=False``. Returns ``None`` when the batch lacks
    replayable state or the solver emits no usable per-layer nu.
    """
    if with_costs and (literals is None or codec is None):
        raise ValueError("with_costs=True requires packed literals")
    split_refresh = refresh_mode in ("split_refresh", "split_refresh_linear")
    if split_refresh and base_reference_bounds is None:
        raise ValueError("split-dependent replay requires base_reference_bounds")
    if (
        batch.incremental_alpha is None
        or batch.incremental_eta is None
        or batch.split_signs is None
        or m_specs < 1
    ):
        return None
    from act.back_end.solver.solver_dual import DualSolver

    n_lanes = batch.batch_size
    if c_rows.dim() != 2 or c_rows.shape[0] != n_lanes * m_specs or thresholds.shape != (n_lanes, m_specs):
        raise ValueError("replay objectives must match lane/spec dimensions")
    supports = None
    if split_refresh:
        from act.back_end.bab.bound_refresh import refresh_bounds
        assert base_reference_bounds is not None
        refreshed = refresh_bounds(
            net, base_reference_bounds, batch.split_signs,
            literals=literals if with_costs else None,
            codec=codec if with_costs else None,
            linear=refresh_mode == "split_refresh_linear",
            support_refinement=lin_support_refinement,
            rows_cap=refresh_rows_cap,
            forward_lin_max_perturbed=forward_lin_max_perturbed,
        )
        reference_bounds, supports = refreshed.bounds, refreshed.supports
    eta = {
        layer_id: value.detach().clone().clamp(min=0)
        for layer_id, value in batch.incremental_eta.items()
    }
    try:
        result = DualSolver().compute_certified_bound(
            net,
            reference_bounds,
            c_rows,
            M=m_specs,
            alpha=batch.incremental_alpha,
            eta=eta,
            split_signs=batch.split_signs,
            optimize=False,
            return_nu_per_layer=with_costs,
            local_phase_clamp=True,
        )
    except (RuntimeError, ValueError):
        return None
    assert result.dual_margins is not None
    margins = result.dual_margins.reshape(n_lanes, m_specs)
    threshold_rows = thresholds.reshape(n_lanes, m_specs).to(margins)
    slack = margins - threshold_rows
    pinned_rows: Optional[torch.Tensor] = None
    if out_kind in OutputSpec.CLOSED_KINDS:
        # The budget must remain a backward certificate, including its strict
        # rounding band. Combined forward margins cannot fund literal deletion.
        slack = slack - certification_tolerance(
            torch.maximum(margins.abs(), threshold_rows.abs())
        )
    if out_kind == OutKind.UNSAFE_LINEAR:
        passing = ~unproven_spec_rows(slack, out_kind)
        pinned_rows = passing.to(torch.int64).argmax(dim=1)
    costs: Optional[torch.Tensor] = None
    if with_costs:
        assert result.nu_per_layer is not None, (
            "certificate_replay: solver returned no nu_per_layer despite return_nu_per_layer=True"
        )
        nu: Dict[int, torch.Tensor] = {}
        for layer_id, value in result.nu_per_layer.items():
            if value.shape[0] == n_lanes * m_specs:
                nu[layer_id] = value.reshape(n_lanes, m_specs, -1)
            elif value.shape[0] == n_lanes:
                nu[layer_id] = value.reshape(n_lanes, -1, value.shape[-1])
            else:
                return None
        assert literals is not None
        assert codec is not None
        costs = _literal_costs(
            literals, codec, slack, eta, nu, reference_bounds, pinned_rows
        )
        if supports is not None:
            dependent = torch.zeros_like(literals, dtype=torch.bool)
            unknown = torch.zeros(n_lanes, dtype=torch.bool, device=literals.device)
            for layer_id, sensitivity in nu.items():
                support = supports.get(layer_id)
                if support is None:
                    unknown |= sensitivity.flatten(1).ne(0).any(1)
                    continue
                used = sensitivity.ne(0).any(1)
                unknown |= (used & ~support.known).any(1)
                dependent |= ((support.lower | support.upper) & used[:, :, None]).any(1)
            # Costs are a proposal heuristic, not a certificate. Dependency
            # removal must buy a real replay; unknown provenance never grants
            # a zero-cost deletion.
            costs += dependent[:, None, :] * (slack.abs().amax(1) + 1)[:, None, None]
            costs[unknown] = torch.inf
    if pinned_rows is not None:
        slack = slack.gather(1, pinned_rows[:, None])
    return slack, costs


@torch.no_grad()
def _literal_costs(
    literals: torch.Tensor,
    codec: LiteralCodec,
    slack: torch.Tensor,
    eta: Dict[int, torch.Tensor],
    nu: Dict[int, torch.Tensor],
    bounds: Dict[int, Bounds],
    pinned_rows: Optional[torch.Tensor],
) -> torch.Tensor:
    """Estimate Eq. 12 costs ``(eta + relu(-nu)) * m``, shaped ``[N, M, W]``.

    Retained alpha can change the relaxation after literal deletion. These
    costs select candidates; only replay certifies a changed learned core.
    """
    n_lanes, width = literals.shape
    m_specs = slack.shape[1]
    costs = torch.zeros(
        (n_lanes, m_specs, width),
        dtype=slack.dtype,
        device=literals.device,
    )
    mask = literals != 0
    if bool(mask.any()):
        layer_ids, neurons, signs = codec.decode_tensor(literals.masked_fill(~mask, 1))
        for layer_id in codec.layer_ids:
            lane, slot = torch.where(mask & (layer_ids == layer_id))
            if lane.numel() == 0:
                continue
            if layer_id not in eta or layer_id not in nu or layer_id not in bounds:
                costs[lane, :, slot] = torch.inf
                continue
            neuron = neurons[lane, slot]
            lane_eta = eta[layer_id][lane, :, neuron].clamp(min=0)
            lane_nu = nu[layer_id][lane, :, neuron]
            layer_bounds = bounds[layer_id]
            lower = layer_bounds.lb.flatten(start_dim=1)[lane, neuron]
            upper = layer_bounds.ub.flatten(start_dim=1)[lane, neuron]
            magnitude = torch.where(
                signs[lane, slot] > 0,
                (-lower).clamp(min=0),
                upper.clamp(min=0),
            )
            costs[lane, :, slot] = (
                lane_eta + (-lane_nu).clamp(min=0)
            ) * magnitude.unsqueeze(1)
    if pinned_rows is not None:
        rows = pinned_rows.to(device=costs.device, dtype=torch.long)
        costs = costs.gather(1, rows[:, None, None].expand(-1, 1, width))
    return costs


@torch.no_grad()
def coarsen(
    literals: torch.Tensor,
    costs: torch.Tensor,
    slack: torch.Tensor,
    *,
    theta: float,
    delta_abs: float,
    delta_rel: float,
    recheck_k: int,
    recheck: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Choose budget deletions plus at most ``recheck_k`` one-step rechecks.

    The caller must replay every changed result before core admission,
    including rows changed only by the cost-budget heuristic.

    Per row ``m`` the budget is ``theta * slack_m - delta_m`` with
    ``delta_m = delta_abs + delta_rel * |slack_m|``; rows with
    ``slack_m <= delta_m`` never fund a deletion. The budget arithmetic runs
    in the dtype of ``costs``/``slack``. Literals are ordered by
    ``(max_m cost_m / budget_m, column, slot)`` and the longest feasible prefix
    is deleted. A recheck retries the next literal in that same order for the
    first ``recheck_k`` lanes; ``recheck`` returns the ``BoolTensor[K]``
    certification mask that confirms each candidate deletion.
    """
    if not 0.0 <= theta <= 1.0:
        raise ValueError("theta must be in [0, 1]")
    if not (delta_abs >= 0.0 and delta_rel >= 0.0):
        raise ValueError("delta_abs and delta_rel must be non-negative")
    if recheck_k < 0:
        raise ValueError("recheck_k must be non-negative")
    n_lanes, width = literals.shape
    if costs.shape[0] != n_lanes or costs.shape[2] != width:
        raise ValueError("cost tensor shape must be [N, M, W]")
    if slack.shape != costs.shape[:2]:
        raise ValueError("slack tensor shape must be [N, M]")
    retained = literals != 0
    next_in_cost_order: Dict[int, int] = {}
    for lane in range(n_lanes):
        valid_slots = torch.where(literals[lane] != 0)[0]
        if valid_slots.numel() == 0:
            continue
        lane_costs = costs[lane].index_select(1, valid_slots)
        lane_slack = slack[lane]
        if not bool(torch.isfinite(lane_costs).all().item()) or not bool(
            torch.isfinite(lane_slack).all().item()
        ):
            continue
        delta = delta_abs + delta_rel * lane_slack.abs()
        if bool((lane_slack <= delta).any().item()):
            continue
        budget = theta * lane_slack - delta
        denominator = budget.clamp(min=torch.finfo(budget.dtype).tiny)
        ratios = (lane_costs / denominator[:, None]).amax(dim=0)
        ordered = sorted(
            range(valid_slots.numel()),
            key=lambda pos: (
                float(ratios[pos].item()),
                int(literals[lane, valid_slots[pos]].abs().item()) - 1,
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
        rows = torch.tensor(candidates, dtype=torch.long, device=literals.device)
        candidate_mask = retained.index_select(0, rows).clone()
        for local, slot in enumerate(next_slots):
            candidate_mask[local, slot] = False
        candidate = literals.index_select(0, rows).masked_fill(~candidate_mask, 0)
        passing = recheck(candidate, rows)
        for local, passed in enumerate(passing.tolist()):
            if passed:
                retained[candidates[local], next_slots[local]] = False

    return literals.masked_fill(~retained, 0)


def validate_climb_config(config: BaBConfig) -> None:
    """Fail before solving when CLIMB's soundness prerequisites are absent."""
    if not config.climb_enabled:
        return
    failures: List[str] = []
    if config.solver_tier != CLIMB_SOLVER_TIER:
        failures.append(f"solver_tier must be {CLIMB_SOLVER_TIER!r}")
    if config.bounding not in TOP_K_BOUNDINGS:
        failures.append(f"bounding must be one of {TOP_K_BOUNDINGS}")
    if config.branching_method not in NEURON_BRANCHING_METHODS:
        failures.append(f"branching_method must be one of {NEURON_BRANCHING_METHODS}")
    if config.per_subproblem_refine != NO_REFINEMENT_MODE:
        failures.append(f"per_subproblem_refine must be {NO_REFINEMENT_MODE!r}")
    if failures:
        raise ConfigError("CLIMB configuration error: " + "; ".join(failures))


@dataclass(frozen=True)
class _ReplayContext:
    batch: SubproblemBatch
    bounds: Dict[int, Bounds]
    c_lanes: torch.Tensor
    thresholds: torch.Tensor
    m_specs: int
    base_bounds: Optional[Dict[int, Bounds]] = None
    checked_candidates: Optional[torch.Tensor] = None
    checked_passing: Optional[torch.Tensor] = None


class ClimbSession:
    """Own the query-local CLIMB library, metrics, and BaB-loop hooks."""

    def __init__(
        self, config: BaBConfig, net: Net, out_kind: str,
        dual_config: Optional[DualConfig] = None,
    ) -> None:
        self._net = net
        self._out_kind = out_kind
        self._coarsening_enabled = config.climb_coarsening_enabled
        self._propagation_enabled = config.climb_propagation_enabled
        self._presplit_propagation_enabled = config.climb_presplit_propagation_enabled
        self._complement_bounding = config.climb_complement_bounding
        self._steer = config.climb_steer
        self._merge_enabled = config.climb_merge_enabled
        self._merge_generation = 0
        self._merged_parents = torch.empty(0, dtype=torch.long)
        self._theta = config.climb_theta
        self._delta_abs = config.climb_delta_abs
        self._delta_rel = config.climb_delta_rel
        self._recheck_k = config.climb_recheck_k
        self._propagate_core_chunk = config.climb_propagate_core_chunk
        self._refresh_mode = config.root_bounds_reuse
        self._lin_support_refinement = config.climb_lin_support_refinement
        self._refresh_rows_cap = config.per_subproblem_refine_rows_cap
        self._forward_lin_max_perturbed = (dual_config or DualConfig()).forward_lin_max_perturbed
        from act.back_end.verifier import gather_input_spec_layers, seed_from_input_specs
        self._root_bounds = seed_from_input_specs(gather_input_spec_layers(net))
        self.codec = LiteralCodec.from_net(net)
        self.library = CoreLibrary(config.climb_max_cores, track_complements=self._complement_bounding)
        self.metrics = ClimbMetrics(enabled=True, library=self.library)

    @classmethod
    def from_config(
        cls, config: BaBConfig, net: Net, out_kind: str,
        dual_config: Optional[DualConfig] = None,
    ) -> Optional[ClimbSession]:
        validate_climb_config(config)
        return cls(config, net, out_kind, dual_config) if config.climb_enabled else None

    def before_pop(self, pool: TopKBounding) -> None:
        self._merge(pool)
        if not self._propagation_enabled:
            self.metrics.observe_frontier(len(pool))
            return
        if self.library.core_count > 0:
            # H1: eager discharge of the whole pending pool before any lane is
            # bounded. view_all/replace_all never score, cool, or probe.
            pending_before = pool.view_all()
            propagate_started = time.perf_counter()
            (pending_after,), _ = propagate(
                [pending_before],
                self.library,
                self.codec,
                core_chunk=self._propagate_core_chunk,
            )
            self.metrics.propagate_time_s += time.perf_counter() - propagate_started
            self.metrics.prebound_discharged += (
                pending_before.batch_size - pending_after.batch_size
            )
            pool.replace_all(pending_after)
            if pool.empty:
                return
        self.metrics.observe_frontier(len(pool))

    def _merge(
        self, pool: TopKBounding, active: Optional[SubproblemBatch] = None,
    ) -> Tuple[Optional[SubproblemBatch], torch.Tensor]:
        """Merge equal-domain complementary siblings once per new proof generation.

        Only revived parents return to pending; they must be bounded again.
        A parent id can be revived once, preventing split/undo-split loops.
        """
        device = active.lb.device if active is not None else self._root_bounds.lb.device
        keep_active = torch.arange(active.batch_size if active is not None else 0, device=device)
        if not self._merge_enabled or self._merge_generation == self.library.generation:
            return active, keep_active
        self._merge_generation = self.library.generation
        n_pending = len(pool)
        joint = pool.view_all() if n_pending else active
        if joint is None:
            return active, keep_active
        if n_pending and active is not None:
            joint = joint.concat(active)
        if joint.parent_id is None or joint.node_id is None or joint.decision_literals is None:
            return active, keep_active
        phases = pack_split_matrix(joint.split_signs, joint.batch_size, self.codec, device=device)
        decisions = joint.decision_literals
        used = torch.zeros(joint.batch_size, dtype=torch.bool, device=device)
        partners = torch.full((joint.batch_size,), -1, dtype=torch.long, device=device)
        blocked = torch.isin(joint.parent_id, self._merged_parents.to(device)) | (joint.parent_id < 0)
        width = max(1, joint.input_dim, phases.shape[1], decisions.shape[1])
        chunk = min(self._propagate_core_chunk, 128, max(1, int((1048576 / width) ** .5)))
        for i in range(0, joint.batch_size, chunk):
            for j in range(i, joint.batch_size, chunk):
                a, b = phases[i:i + chunk], phases[j:j + chunk]
                diff = a[:, None] != b[None]
                da, db = decisions[i:i + chunk], decisions[j:j + chunk]
                ddiff = da[:, None] != db[None]
                pairs = ((a[:, None].abs() == b[None].abs()).all(2) & (diff.sum(2) == 1)
                         & (da[:, None].abs() == db[None].abs()).all(2) & (ddiff.sum(2) == 1))
                # The erased phase must be a real decision, not an implication.
                pairs &= (a[:, None].masked_fill(~diff, 0).abs().sum(2)
                          == da[:, None].expand(-1, db.shape[0], -1).masked_fill(~ddiff, 0).abs().sum(2))
                pairs &= (joint.parent_id[i:i + chunk, None] == joint.parent_id[None, j:j + chunk])
                pairs &= (joint.depths[i:i + chunk, None] == joint.depths[None, j:j + chunk])
                pairs &= (joint.lb[i:i + chunk, None] == joint.lb[None, j:j + chunk]).all(2)
                pairs &= (joint.ub[i:i + chunk, None] == joint.ub[None, j:j + chunk]).all(2)
                pairs &= ~blocked[i:i + chunk, None] & ~blocked[None, j:j + chunk]
                row_index = torch.arange(a.shape[0], device=device)
                while True:
                    available = pairs & ~used[i:i + a.shape[0], None] & ~used[None, j:j + b.shape[0]]
                    choice = available.to(torch.int64).argmax(1)
                    if i == j:
                        # Symmetric tile: mutual choices are disjoint undirected
                        # edges. Process all such pairs in one device operation.
                        mutual = (choice[choice] == row_index) & (row_index < choice)
                    else:
                        reverse = available.to(torch.int64).argmax(0)
                        mutual = reverse[choice] == row_index
                    selected = torch.where(available.any(1) & mutual)[0]
                    if not selected.numel():
                        break
                    left, right = selected + i, choice[selected] + j
                    used[left] = used[right] = True
                    partners[left] = right
        left = torch.where(partners >= 0)[0]
        if not left.numel():
            return active, keep_active
        right = partners[left]
        merged = joint.select(left)
        assert merged.parent_id is not None and merged.decision_literals is not None
        merged.node_id = merged.parent_id.clone()
        self._merged_parents = torch.cat((self._merged_parents.to(device), merged.node_id))
        merged.parent_id = torch.full_like(merged.node_id, -1)
        merged.depths -= 1
        merged.decision_literals.masked_fill_(decisions[left] != decisions[right], 0)
        if merged.split_signs:
            assert joint.split_signs is not None
            for lid, values in merged.split_signs.items():
                values.masked_fill_(joint.split_signs[lid][left] != joint.split_signs[lid][right], 0)
        # Clearing the entire affected pool avoids pretending zero alpha padding
        # is a cold start for the revived lanes.
        joint.incremental_alpha = joint.incremental_eta = None
        merged.incremental_alpha = merged.incremental_eta = None
        merged.lower_bound = torch.full_like(merged.depths, -torch.inf, dtype=merged.lb.dtype)
        merged.parent_margins = merged.lower_bound.clone()
        pending = joint.select(torch.where(~used[:n_pending])[0])
        pool.replace_all(pending.concat(merged), cold_rebound=True)
        keep_active = torch.where(~used[n_pending:])[0]
        active = joint.select(keep_active + n_pending) if active is not None else None
        self.metrics.merged_pairs += int(left.numel())
        return active, keep_active

    def after_pop(self, frontier_rows: int) -> None:
        self.metrics.observe_frontier(frontier_rows)

    def after_main_bound(self, k_actual: int) -> None:
        self.metrics.main_bound_calls += 1
        self.metrics.main_bound_row_passes += k_actual

    def retire_lanes_without_relu_candidate(self, retired: int) -> None:
        """Record unresolved lanes that cannot use CLIMB's neuron-only splits."""
        self.metrics.retired_unknown_lanes += retired

    def learn(
        self,
        batch: SubproblemBatch,
        statuses: Sequence[str],
        dual_solve_result: DualBatchResult,
        k_actual: int,
        *,
        allow_complements: bool = True,
    ) -> None:
        # Replay uses the immutable root-domain reference, optionally refined
        # once at the root. Budget coarsening proposes a smaller row; a final
        # replay must certify it before admission, or restore the original.
        if k_actual != batch.batch_size or len(statuses) != k_actual:
            raise ValueError("CLIMB admission lane count mismatch")
        if self._refresh_mode in ("split_refresh", "split_refresh_linear") and dual_solve_result.base_reference_bounds is None:
            raise ValueError("split-dependent learning requires base_reference_bounds")
        unsat_idx = torch.tensor(
            [i for i, status in enumerate(statuses) if status == SolveStatus.UNSAT],
            device=batch.lb.device,
            dtype=torch.long,
        )
        if dual_solve_result.forward_only_certified is not None:
            forward_only = dual_solve_result.forward_only_certified.to(unsat_idx.device)
            unsat_idx = unsat_idx[~forward_only.index_select(0, unsat_idx)]
        # A proof learned on a smaller input box cannot become a global
        # root-domain core, even when that lane still carries ReLU decisions.
        root_lb, root_ub = self._root_bounds.lb.flatten(1), self._root_bounds.ub.flatten(1)
        same_domain = (batch.lb == root_lb).all(1) & (batch.ub == root_ub).all(1)
        unsat_idx = unsat_idx[same_domain.index_select(0, unsat_idx)]
        if (
            int(unsat_idx.numel()) == 0
            or dual_solve_result.c_rows is None
            or dual_solve_result.thresholds is None
            or dual_solve_result.reference_bounds is None
            or dual_solve_result.m_specs <= 0
        ):
            return
        replay_batch = batch.select(unsat_idx)
        replay_m_specs = dual_solve_result.m_specs
        # Slice both snapshots with the same lane selector. Split-dependent
        # replay regenerates its conditioned bounds from the split-free one.
        replay_bounds = slice_bounds_dict(
            dual_solve_result.reference_bounds, unsat_idx
        )
        replay_base = (
            slice_bounds_dict(dual_solve_result.base_reference_bounds, unsat_idx)
            if dual_solve_result.base_reference_bounds is not None else None
        )
        replay_c_lanes = dual_solve_result.c_rows.reshape(
            k_actual, replay_m_specs, -1
        ).index_select(0, unsat_idx.to(dual_solve_result.c_rows.device))
        replay_c = replay_c_lanes.reshape(-1, replay_c_lanes.shape[-1])
        replay_thresholds = dual_solve_result.thresholds.index_select(
            0, unsat_idx.to(dual_solve_result.thresholds.device)
        )
        packed = pack_split_matrix(
            replay_batch.split_signs,
            replay_batch.batch_size,
            self.codec,
            device=replay_batch.lb.device,
        )
        if replay_batch.decision_literals is not None:
            packed = replay_batch.decision_literals
            _replace_literal_assignments(replay_batch, packed, self.codec)
        replay = certificate_replay(
            net=self._net,
            batch=replay_batch,
            reference_bounds=replay_bounds,
            c_rows=replay_c,
            thresholds=replay_thresholds,
            m_specs=replay_m_specs,
            out_kind=self._out_kind,
            literals=packed,
            codec=self.codec,
            with_costs=self._coarsening_enabled,
            base_reference_bounds=replay_base,
            refresh_mode=self._refresh_mode,
            lin_support_refinement=self._lin_support_refinement,
            refresh_rows_cap=self._refresh_rows_cap,
            forward_lin_max_perturbed=self._forward_lin_max_perturbed,
        )
        if replay is not None:
            slack, costs = replay
            valid = is_certified(slack, self._out_kind)
        else:
            slack = costs = valid = None
        if valid is None or not bool(valid.any().item()):
            return
        budget_slack = slack
        assert budget_slack is not None
        context = _ReplayContext(
            replay_batch, replay_bounds, replay_c_lanes, replay_thresholds, replay_m_specs,
            replay_base, packed.clone(), valid.clone(),
        )
        coarsen_started = time.perf_counter()
        if self._coarsening_enabled:
            assert costs is not None
            costs[~valid] = torch.inf
            retained = coarsen(
                packed,
                costs,
                budget_slack,
                theta=self._theta,
                delta_abs=self._delta_abs,
                delta_rel=self._delta_rel,
                recheck_k=self._recheck_k,
                recheck=lambda candidate, rows: self._instrumented_recheck(
                    context, candidate, rows
                ),
            )
        else:
            retained = packed
        changed = torch.where(valid & (retained != packed).any(dim=1))[0]
        if changed.numel():
            assert context.checked_candidates is not None and context.checked_passing is not None
            fresh = (retained.index_select(0, changed) != context.checked_candidates.index_select(0, changed)).any(1)
            pending = changed[fresh]
            if pending.numel():
                self._instrumented_recheck(context, retained.index_select(0, pending), pending)
            certified = context.checked_passing.index_select(0, changed)
            failed = changed[~certified]
            retained[failed] = packed[failed]
            self.metrics.coarsen_replay_fallbacks += int(failed.numel())
        self.metrics.coarsen_time_s += time.perf_counter() - coarsen_started
        original_literals = int((packed != 0).sum())
        retained_literals = int((retained != 0).sum())
        assert _retained_literals_are_subset(packed, retained), (
            "CLIMB invariant violated: coarsening added or changed a literal "
            f"(rows={packed.shape[0]}, width={packed.shape[1]}, "
            f"original_literals={original_literals}, retained_literals={retained_literals})"
        )
        self.metrics.core_literals_original += original_literals
        self.metrics.core_literals_retained += retained_literals
        assert (
            self.metrics.core_literals_retained
            <= self.metrics.core_literals_original
        ), (
            "CLIMB invariant violated: retained literal accounting exceeds original "
            f"(original_total={self.metrics.core_literals_original}, "
            f"retained_total={self.metrics.core_literals_retained})"
        )
        valid_rows = torch.where(valid)[0]
        valid_cores = retained.index_select(0, valid_rows)
        self.metrics.cores_inserted += self.library.insert(
            valid_cores, suppress_complements=not allow_complements,
        )

    def bound_complements(
        self, solve: Callable[[SubproblemBatch, float], DualBatchResult],
        remaining: Callable[[], float], node_budget: int,
    ) -> int:
        """Probe new cores on the original input domain using the shared bounder."""
        if not self._complement_bounding or node_budget <= 0 or remaining() <= 0:
            return 0
        literals = self.library.take_complements(node_budget, self._root_bounds.lb.device)
        count = literals.shape[0]
        if not count:
            return 0
        candidate = SubproblemBatch.from_bounds(Bounds(
            self._root_bounds.lb.expand(count, *self._root_bounds.lb.shape[1:]),
            self._root_bounds.ub.expand(count, *self._root_bounds.ub.shape[1:]),
        ))
        candidate.decision_literals = literals
        candidate.depths = (literals != 0).sum(1)
        _replace_literal_assignments(candidate, literals, self.codec)
        cap = remaining()
        if cap <= 0:
            return 0
        started = time.perf_counter()
        self.metrics.complement_calls += 1
        self.metrics.complement_row_passes += count
        try:
            result = solve(candidate, cap)
        except (TimeoutError, RuntimeError):
            return count
        finally:
            self.metrics.complement_time_s += time.perf_counter() - started
        if remaining() > 0:
            self.learn(candidate, result.lane_solution().statuses, result, count, allow_complements=False)
        return count

    def steer_scores(self, scores: BranchingScores) -> BranchingScores:
        """Bias finite branching candidates by core frequency, never add facts."""
        if not self._steer or self.library.core_count == 0:
            return scores
        for attr in ("per_layer", "intercept_per_layer", "babsr_per_layer"):
            values = getattr(scores, attr)
            if not values:
                continue
            biased = {}
            for lid, value in values.items():
                literals = self.library.pack(value.device)
                variables = literals[literals != 0].abs() - 1
                layers, neurons, _ = self.codec.decode_tensor(variables + 1)
                counts = torch.bincount(neurons[layers == lid], minlength=value.shape[1]).to(value)
                finite = torch.isfinite(value)
                scale = value.masked_fill(~finite, 0).abs().amax(1, keepdim=True).clamp(min=1)
                bias = counts[:value.shape[1]][None] / max(1, self.library.core_count)
                biased[lid] = torch.where(finite, value + scale * bias, value)
                self.metrics.steered_candidates += int((finite & (bias > 0)).sum())
            setattr(scores, attr, biased)
        return scores

    def _recheck(
        self,
        context: _ReplayContext,
        candidate: torch.Tensor,
        rows: torch.Tensor,
    ) -> torch.Tensor:
        candidate_batch = context.batch.select(rows)
        # The solver indexes split_signs[lid] for every incremental_eta layer, so a
        # layer whose literals were all coarsened away stays as an all-zero block.
        _replace_literal_assignments(candidate_batch, candidate, self.codec)
        eta_layers = set(candidate_batch.incremental_eta or {})
        sign_layers = set(candidate_batch.split_signs or {})
        missing_layers = sorted(eta_layers - sign_layers)
        assert not missing_layers, (
            "CLIMB invariant violated: recheck split_signs missing incremental_eta layer "
            f"(KeyError: {missing_layers[0]}, eta_layers={len(eta_layers)}, "
            f"split_sign_layers={len(sign_layers)})"
        )
        row_bounds = slice_bounds_dict(context.bounds, rows)
        row_c_lanes = context.c_lanes.index_select(
            0, rows.to(context.c_lanes.device)
        )
        row_c = row_c_lanes.reshape(-1, row_c_lanes.shape[-1])
        row_thresholds = context.thresholds.index_select(
            0, rows.to(context.thresholds.device)
        )
        replayed = certificate_replay(
            net=self._net,
            batch=candidate_batch,
            reference_bounds=row_bounds,
            c_rows=row_c,
            thresholds=row_thresholds,
            m_specs=context.m_specs,
            out_kind=self._out_kind,
            with_costs=False,
            base_reference_bounds=(
                slice_bounds_dict(context.base_bounds, rows)
                if context.base_bounds is not None else None
            ),
            refresh_mode=self._refresh_mode,
            lin_support_refinement=self._lin_support_refinement,
            refresh_rows_cap=self._refresh_rows_cap,
            forward_lin_max_perturbed=self._forward_lin_max_perturbed,
        )
        if replayed is None:
            return torch.zeros(
                int(rows.numel()), dtype=torch.bool, device=rows.device
            )
        replayed_slack, _ = replayed
        return is_certified(replayed_slack, self._out_kind)

    def _instrumented_recheck(
        self,
        context: _ReplayContext,
        candidate: torch.Tensor,
        rows: torch.Tensor,
    ) -> torch.Tensor:
        recheck_started = time.perf_counter()
        passing = self._recheck(context, candidate, rows)
        if context.checked_candidates is not None and context.checked_passing is not None:
            context.checked_candidates[rows] = candidate
            context.checked_passing[rows] = passing
        self.metrics.recheck_time_s += time.perf_counter() - recheck_started
        self.metrics.recheck_calls += 1
        self.metrics.recheck_row_passes += int(rows.numel())
        return passing

    def before_split(
        self, pool: TopKBounding, unresolved: SubproblemBatch
    ) -> Tuple[SubproblemBatch, torch.Tensor]:
        original_rows = torch.arange(unresolved.batch_size, device=unresolved.lb.device)
        merged_active, original_rows = self._merge(pool, unresolved)
        assert merged_active is not None
        unresolved = merged_active
        if not self._propagation_enabled or not self._presplit_propagation_enabled:
            self.metrics.observe_frontier(len(pool) + unresolved.batch_size)
            return unresolved, original_rows
        if self.library.core_count == 0:
            return unresolved, original_rows
        # H3: one joint fixpoint over (pending, active); pending removals were
        # never bounded (pre-bound), active removals are parents spared a split
        # (pre-split).
        self.metrics.observe_frontier(len(pool) + unresolved.batch_size)
        propagation_groups: List[SubproblemBatch] = []
        has_pool = len(pool) > 0
        if has_pool:
            propagation_groups.append(pool.view_all())
        propagation_groups.append(unresolved)
        presplit_before = unresolved.batch_size
        propagate_started = time.perf_counter()
        propagation, kept_rows = propagate(
            propagation_groups,
            self.library,
            self.codec,
            core_chunk=self._propagate_core_chunk,
        )
        self.metrics.propagate_time_s += time.perf_counter() - propagate_started
        if has_pool:
            pool.replace_all(propagation[0])
            self.metrics.prebound_discharged += (
                propagation_groups[0].batch_size - propagation[0].batch_size
            )
        unresolved = propagation[-1]
        self.metrics.presplit_discharged += presplit_before - unresolved.batch_size
        self.metrics.observe_frontier(len(pool) + unresolved.batch_size)
        return unresolved, original_rows.index_select(0, kept_rows[-1])

    def after_children(self, generated: int, frontier_rows: int) -> None:
        self.metrics.generated_children += generated
        self.metrics.observe_frontier(frontier_rows)

    def metadata(self) -> Dict[str, int | float]:
        return {
            **self.metrics.metadata(),
            "climb_merge_enabled": self._merge_enabled,
            "climb_complement_bounding": self._complement_bounding,
            "climb_steer": self._steer,
            "climb_propagation_enabled": self._propagation_enabled,
            "climb_presplit_propagation_enabled": self._presplit_propagation_enabled,
        }


if __name__ == "__main__":
    from tempfile import TemporaryDirectory
    from unittest.mock import patch

    from act.back_end.dual_tf.tf_forward import compute_forward_bounds
    from act.back_end.net_factory import LAYER_TESTING_SPECS, NetFactory
    from act.back_end.solver.solver_dual import DualSolver
    from act.config.config import BackendConfig, DualConfig
    from act.front_end.specs import OutputSpec
    from act.util.device_manager import initialize_device

    initialize_device("cpu", "float64")
    with TemporaryDirectory(prefix="act-climb-admission-") as directory:
        factory = NetFactory(BackendConfig.from_yaml().generation.net_factory,
                             output_dir=directory, write_manifest=False)
        net = factory.create_network("admission", LAYER_TESTING_SPECS["layer_testing_climb_admission"]())
        upper = torch.ones(1, 1, device=get_default_device(), dtype=get_default_dtype())
        lower = -upper
        bounds, frame, _ = compute_forward_bounds(net, lower, upper)
        for tile in (1, 2, 128):
            library = CoreLibrary(8)
            library.chunk = tile
            assert library.insert(torch.tensor([[2, 1, 1], [1, -2, 0]])) == 2
            assert library.resolved == 1
            assert torch.equal(library.pack(upper.device), torch.tensor([[1]], device=upper.device))
            assert library.insert(torch.tensor([[1, 3], [1, 0]])) == 0
            assert library.subsumed == 2
            assert library.insert(torch.tensor([[-1]])) == 1
            assert library.resolved == 2 and library.pack(upper.device).shape == (1, 0)
            try:
                library.insert(torch.tensor([[1, -1]]))
            except ValueError:
                pass
            else:
                raise AssertionError("complementary literals admitted")
            evict = CoreLibrary(1)
            evict.insert(torch.tensor([[1, 2], [3, 0]]))
            assert evict.evicted == 1 and torch.equal(evict.pack(upper.device), torch.tensor([[3]], device=upper.device))
            chronological = CoreLibrary(8)
            chronological.chunk = tile
            assert chronological.insert(torch.tensor([[1, 2], [1, 0], [1, 3]])) == 2
            assert chronological.subsumed == 2
            assert torch.equal(chronological.pack(upper.device), torch.tensor([[1]], device=upper.device))
        merge_config = BaBConfig(climb_enabled=True, climb_merge_enabled=True,
                                 climb_propagation_enabled=False, climb_propagate_core_chunk=1)
        empty_active = SubproblemBatch(
            lb=lower[:0], ub=upper[:0],
            depths=torch.empty(0, dtype=torch.long, device=upper.device),
        )
        empty_active.split_signs = {2: upper.new_empty(0, 2, 1)}
        empty_library = CoreLibrary(8)
        empty_library.insert(torch.tensor([[-1]]))
        survivor = SubproblemBatch.from_bounds(Bounds(lower, upper))
        (pending_after, active_after), (_, active_rows) = propagate(
            (survivor, empty_active), empty_library, LiteralCodec.from_net(net),
            core_chunk=1,
        )
        assert pending_after.batch_size == 1
        assert active_after.batch_size == active_rows.numel() == 0
        for mismatch in ("none", "box", "ancestry", "implied", "rest"):
            merge_session = ClimbSession(merge_config, net, OutKind.LINEAR_LE)
            merge_session.codec = LiteralCodec((2,), (3,), (0,))
            siblings = SubproblemBatch(
                lb=lower.expand(2, -1).clone(), ub=upper.expand(2, -1).clone(),
                depths=torch.ones(2, dtype=torch.long),
                split_signs={2: upper.new_tensor([[[-1, 1, 0], [-1, 1, 0]], [[1, 1, 0], [1, 1, 0]]])},
                incremental_alpha={2: upper.new_tensor([[.2, .3, .4], [.7, .8, .9]])},
                incremental_eta={2: upper.new_ones(2, 2, 3)},
                node_id=torch.tensor([1, 2]), parent_id=torch.tensor([0, 0]),
                decision_literals=torch.tensor([[-1, 2], [1, 2]]),
            )
            assert siblings.parent_id is not None and siblings.decision_literals is not None
            assert siblings.split_signs is not None and siblings.incremental_alpha is not None
            if mismatch == "box":
                siblings.lb[1] += .1
            elif mismatch == "ancestry":
                siblings.parent_id[1] = 4
            elif mismatch == "implied":
                siblings.decision_literals[:, 0] = 0
            elif mismatch == "rest":
                siblings.split_signs[2][1, :, 2] = 1
            pool = TopKBounding(k=1)
            pool.push(siblings)
            viewed = pool.view_all()
            assert viewed.incremental_alpha is not None
            viewed_alpha, sibling_alpha = viewed.incremental_alpha[2], siblings.incremental_alpha[2]
            assert isinstance(viewed_alpha, torch.Tensor) and isinstance(sibling_alpha, torch.Tensor)
            assert torch.equal(viewed_alpha, sibling_alpha)
            assert viewed.decision_literals is not None
            assert torch.equal(viewed.decision_literals, siblings.decision_literals)
            pool.replace_all(viewed)
            active = pool.pop()
            # No fresh proof means no gather and no immediate split undo.
            with patch.object(pool, "view_all", side_effect=AssertionError("unneeded gather")):
                merge_session.before_split(pool, active)
            merge_session.library.insert(torch.tensor([[3]]))
            active_after, selected = merge_session.before_split(pool, active)
            if mismatch == "none":
                assert active_after.batch_size == selected.numel() == 0 and len(pool) == 1
                revived = pool.pop()
                assert revived.node_id is not None and revived.parent_id is not None
                assert revived.lower_bound is not None and revived.split_signs is not None
                assert revived.decision_literals is not None
                assert revived.node_id.item() == 0 and revived.parent_id.item() == -1
                assert revived.depths.item() == 0
                assert revived.incremental_alpha is revived.incremental_eta is None
                assert bool(torch.isneginf(revived.lower_bound).all())
                assert torch.equal(revived.split_signs[2], upper.new_tensor([[[0, 1, 0], [0, 1, 0]]]))
                assert torch.equal(revived.decision_literals, torch.tensor([[0, 2]]))
                # A second proof cannot undo another split of the revived parent.
                pool.push(siblings)
                merge_session.library.insert(torch.tensor([[2]]))
                merge_session.before_pop(pool)
                assert len(pool) == 2 and merge_session.metrics.merged_pairs == 1
            else:
                assert active_after.batch_size == selected.numel() == len(pool) == 1
                assert merge_session.metrics.merged_pairs == 0
        disabled = ClimbSession(BaBConfig(climb_propagation_enabled=False), net, OutKind.LINEAR_LE)
        disabled.library.insert(torch.tensor([[1]]))
        pool = TopKBounding()
        with patch.object(pool, "view_all", side_effect=AssertionError("disabled merge gathered")):
            disabled.before_pop(pool)
        ordered = CoreLibrary(8, track_complements=True)
        ordered.insert(torch.tensor([[2, -1]]))
        assert torch.equal(ordered.take_complements(1, upper.device), torch.tensor([[1, 2]], device=upper.device))
        ordered.insert(torch.tensor([[2, -1]]))
        assert ordered.take_complements(1, upper.device).shape[0] == 0
        ordered.insert(torch.tensor([[2, 1]]), suppress_complements=True)
        assert ordered.take_complements(8, upper.device).shape[0] == 0
        assert torch.equal(ordered.pack(upper.device), torch.tensor([[2]], device=upper.device))
        derived = CoreLibrary(8, track_complements=True)
        derived.insert(torch.tensor([[2, 1, 3], [2, -1, 3]]))
        assert torch.equal(derived.take_complements(1, upper.device), torch.tensor([[2, -3]], device=upper.device))
        try:
            SubproblemBatch.from_bounds(Bounds(lower, upper)).concat(
                SubproblemBatch(lb=lower, ub=upper, depths=torch.zeros(1, dtype=torch.long),
                                decision_literals=torch.empty((1, 0), dtype=torch.long)),
            )
        except ValueError:
            pass
        else:
            raise AssertionError("mixed decision provenance silently lost")
        for total in (False, True):
            for presplit_gate in (False, True):
                gates = ClimbSession(BaBConfig(
                    climb_propagation_enabled=total,
                    climb_presplit_propagation_enabled=presplit_gate,
                ), net, OutKind.LINEAR_LE)
                gates.library.insert(torch.tensor([[-1]]))
                gated = SubproblemBatch.from_bounds(Bounds(lower, upper))
                gated.split_signs = {2: -upper[:, None]}
                pending = TopKBounding()
                pending.push(gated)
                gates.before_pop(pending)
                assert len(pending) == int(not total)
                pending = TopKBounding()
                pending.push(gated)
                remaining_batch, remaining_rows = gates.before_split(pending, gated)
                assert remaining_batch.batch_size == remaining_rows.numel() == int(not (total and presplit_gate))
                assert len(pending) == int(not (total and presplit_gate))
        steering = ClimbSession(BaBConfig(climb_steer=True), net, OutKind.LINEAR_LE)
        steering.codec = LiteralCodec((2,), (3,), (0,))
        untouched = BranchingScores(per_layer={2: upper.new_tensor([[.1, .2, -torch.inf]])})
        assert steering.steer_scores(untouched) is untouched
        steering.library.insert(torch.tensor([[1]]))
        assert untouched.per_layer is not None
        steered_tables = steering.steer_scores(untouched).per_layer
        assert steered_tables is not None
        steered = steered_tables[2]
        assert steered.argmax(1).item() == 0 and torch.isneginf(steered[0, 2])
        assert steering.metrics.steered_candidates == 1
        from unittest.mock import Mock

        probing = ClimbSession(BaBConfig(climb_complement_bounding=True), net, OutKind.LINEAR_LE)
        solve_complement = Mock(side_effect=RuntimeError("failed complement solve"))
        assert probing.bound_complements(solve_complement, lambda: 1., 1) == 0
        probing.library.insert(torch.tensor([[-1]]))
        assert probing.bound_complements(solve_complement, lambda: 0., 1) == 0
        assert probing.bound_complements(solve_complement, lambda: 1., 0) == 0
        assert solve_complement.call_count == 0
        assert probing.bound_complements(solve_complement, lambda: 1., 1) == 1
        assert probing.bound_complements(solve_complement, lambda: 1., 1) == 0
        assert solve_complement.call_count == 1 and probing.metrics.complement_row_passes == 1
        attempted_batch, cap = solve_complement.call_args.args
        assert cap == 1. and torch.equal(attempted_batch.lb, lower)
        assert torch.equal(attempted_batch.decision_literals, torch.tensor([[1]]))
        assert disabled.bound_complements(solve_complement, lambda: 1., 1) == 0
        from act.back_end.bab.bound_refresh import refresh_bounds

        for precision in ("float32", "float64"):
            initialize_device("cpu", precision)
            # B2, M2: opposite lanes share decision-slot layout but never bits.
            refresh_spec = LAYER_TESTING_SPECS["layer_testing_climb_admission"]()
            refresh_spec["layers"].insert(3, {
                "kind": "ADD", "params": {
                    "bias": torch.tensor([.125], dtype=get_default_dtype()),
                    "input_shape": [1, 1], "output_shape": [1, 1],
                },
                "preds": [2, 2],
            })
            refresh_net = factory.create_network(f"refresh_{precision}", refresh_spec)
            refresh_box = torch.ones(2, 1, dtype=get_default_dtype())
            base, _, _ = compute_forward_bounds(refresh_net, -refresh_box, refresh_box)
            phases = {2: torch.tensor([[[1.], [1.]], [[-1.], [-1.]]], dtype=get_default_dtype())}
            codec = LiteralCodec.from_net(refresh_net)
            packed = pack_split_matrix(phases, 2, codec, device=refresh_box.device)
            tracked = refresh_bounds(refresh_net, base, phases, literals=packed, codec=codec)
            plain = refresh_bounds(refresh_net, base, phases)
            assert plain.supports is None and tracked.supports is not None
            assert not refresh_bounds(refresh_net, base, phases, linear=True).linear_refined
            assert tracked.supports[2].lower.shape == (2, 1, 1)
            assert bool(tracked.supports[2].known.all())
            assert not bool(tracked.supports[0].lower.any())
            missing_decisions = refresh_bounds(
                refresh_net, base, phases, literals=packed[:, :0], codec=codec
            )
            assert missing_decisions.supports is not None
            assert not bool(missing_decisions.supports[2].known.any())
            for lid in plain.bounds:
                assert torch.equal(plain.bounds[lid].lb, tracked.bounds[lid].lb)
                assert torch.equal(plain.bounds[lid].ub, tracked.bounds[lid].ub)
            assert torch.equal(tracked.bounds[3].ub[1], refresh_box[1] * .125)
            released = refresh_bounds(refresh_net, base, {2: torch.zeros_like(phases[2])})
            assert bool((released.bounds[3].ub[1] > tracked.bounds[3].ub[1]).all())
            assert bool((released.bounds[2].lb[0] < tracked.bounds[2].lb[0]).all())
            inconsistent = phases[2].clone()
            inconsistent[:, 1] *= -1
            try:
                refresh_bounds(refresh_net, base, {2: inconsistent})
            except ValueError:
                pass
            else:
                raise AssertionError("spec-dependent phases were accepted")
            unsupported_spec = LAYER_TESTING_SPECS["layer_testing_climb_admission"]()
            unsupported_spec["layers"].insert(3, {"kind": "SIGMOID", "params": {}})
            unsupported = factory.create_network(f"unsupported_{precision}", unsupported_spec)
            unknown_base, _, _ = compute_forward_bounds(unsupported, -refresh_box, refresh_box)
            unknown = refresh_bounds(unsupported, unknown_base, phases, literals=packed, codec=codec)
            assert unknown.supports is not None
            assert all(not bool(s.known.any()) for s in unknown.supports.values())
            assert unknown.bounds[2] is unknown_base[2]
            unknown_batch = SubproblemBatch.from_bounds(Bounds(-refresh_box, refresh_box), depth=1)
            unknown_batch.split_signs = {2: -torch.ones(2, 1, 1, dtype=refresh_box.dtype)}
            unknown_batch.incremental_alpha = {2: torch.ones(2, 1, 1, dtype=refresh_box.dtype)}
            unknown_batch.incremental_eta = {2: torch.zeros(2, 1, 1, dtype=refresh_box.dtype)}
            unknown_literals = pack_split_matrix(unknown_batch.split_signs, 2, codec, device=refresh_box.device)
            unknown_replay = certificate_replay(
                net=unsupported, batch=unknown_batch, reference_bounds=unknown_base,
                base_reference_bounds=unknown_base, c_rows=refresh_box, thresholds=refresh_box * 0,
                m_specs=1, out_kind=OutKind.LINEAR_LE, literals=unknown_literals, codec=codec,
                refresh_mode="split_refresh",
            )
            assert unknown_replay is not None and unknown_replay[1] is not None
            assert bool(torch.isinf(unknown_replay[1]).all()), "unknown support cannot fund deletion"
            token_net = factory.create_network(
                f"token_{precision}", LAYER_TESTING_SPECS["layer_testing_token_wise_dense"]()
            )
            token_lb = torch.arange(12, dtype=get_default_dtype()).reshape(2, 2, 3) / 10
            token_base, _, _ = compute_forward_bounds(token_net, token_lb, token_lb + .1)
            token_refresh = refresh_bounds(token_net, token_base, {})
            weight, bias = token_net.by_id[2].params["weight"], token_net.by_id[2].params["bias"]
            exact_lo = (token_lb @ weight.clamp(min=0).T
                        + (token_lb + .1) @ weight.clamp(max=0).T + bias).flatten(1)
            assert token_refresh.bounds[2].lb.shape == (2, 4)
            assert torch.allclose(token_refresh.bounds[2].lb, exact_lo)
        initialize_device("cpu", "float64")
        for alpha in (1., 0.):
            batch = SubproblemBatch.from_bounds(Bounds(lower, upper), depth=1)
            batch.split_signs = {2: -torch.ones(1, 1, 1, device=upper.device, dtype=upper.dtype)}
            result = DualSolver().solve_spec_batch(
                net, bounds, OutputSpec(OutKind.LINEAR_LE, c=-upper[0], d=upper.new_tensor(.25)),
                optimize=True, dual_config=DualConfig(n_iters=0), input_shape=(1,),
                split_signs=batch.split_signs, incremental_alphas={2: torch.full_like(batch.split_signs[2], alpha)},
                refresh_forward=False, forward_frame=frame,
            )
            assert result.solution.statuses == (SolveStatus.UNSAT,)
            batch.incremental_alpha, batch.incremental_eta = result.alpha_state, result.eta_state
            for coarsening in (True, False):
                for propagation in (True, False):
                    config = BaBConfig(
                        climb_enabled=True, solver_tier="dual_alpha_eta",
                        branching_method="babsr", bounding="depth_bound_blend",
                        climb_theta=.5, climb_recheck_k=0,
                        climb_coarsening_enabled=coarsening,
                        climb_propagation_enabled=propagation,
                    )
                    session = ClimbSession(config, net, OutKind.LINEAR_LE)
                    with (
                        patch.object(
                            DualSolver, "compute_certified_bound", autospec=True,
                            side_effect=DualSolver.compute_certified_bound,
                        ) as solve_spy,
                        patch(f"{__name__}._literal_costs", wraps=_literal_costs) as costs_spy,
                    ):
                        session.learn(batch, result.solution.statuses, result, 1)
                    expected = torch.tensor([[-1]] if alpha == 1 or not coarsening else [[]], dtype=torch.long)
                    assert torch.equal(session.library.pack(upper.device), expected.to(upper.device))
                    assert solve_spy.call_count == 1 + int(coarsening)
                    assert costs_spy.call_count == int(coarsening)
                    assert solve_spy.call_args_list[0].kwargs["return_nu_per_layer"] == coarsening
                    assert session.metadata()["climb_recheck_calls"] == int(coarsening)
                    assert session.metadata()["climb_coarsen_replay_fallbacks"] == int(alpha) * int(coarsening)

                    pool = TopKBounding()
                    pool.push(batch)
                    session.before_pop(pool)
                    assert len(pool) == int(not propagation)
                    pool = TopKBounding()
                    pool.push(batch)
                    unresolved, kept = session.before_split(pool, batch)
                    assert len(pool) == unresolved.batch_size == kept.numel() == int(not propagation)
                    if not propagation:
                        assert unresolved is batch
                        assert session.metrics.prebound_discharged == 0
                        assert session.metrics.presplit_discharged == 0
                        assert session.metrics.propagate_time_s == 0.0

            # Disabling coarsening must still reject a failed first replay:
            # alpha=1 only certifies this reference under the inactive literal.
            if alpha == 1:
                invalid = batch.select(torch.zeros(1, device=upper.device, dtype=torch.long))
                invalid.split_signs = {2: torch.zeros_like(batch.split_signs[2])}
                invalid.decision_literals = torch.empty((1, 0), dtype=torch.long, device=upper.device)
                session = ClimbSession(config, net, OutKind.LINEAR_LE)
                with patch.object(
                    DualSolver, "compute_certified_bound", autospec=True,
                    side_effect=DualSolver.compute_certified_bound,
                ) as solve_spy:
                    session.learn(invalid, result.solution.statuses, result, 1)
                assert solve_spy.call_count == 1
                assert session.library.core_count == 0

        # A stale conditioned reference would certify deletion of the only
        # inactive literal. Rebuild from the mandatory split-free base instead.
        replay_batch = SubproblemBatch.from_bounds(Bounds(lower, upper), depth=1)
        replay_batch.split_signs = {2: -torch.ones_like(upper[:, None])}
        replay_batch.incremental_alpha = {2: torch.ones_like(upper[:, None])}
        replay_batch.incremental_eta = {2: torch.zeros_like(upper[:, None])}
        conditioned = refresh_bounds(net, bounds, replay_batch.split_signs).bounds
        replay_result = DualSolver().solve_spec_batch(
            net, conditioned, OutputSpec(OutKind.LINEAR_LE, c=-upper[0], d=upper.new_tensor(.25)),
            optimize=True, dual_config=DualConfig(n_iters=0), input_shape=(1,),
            split_signs=replay_batch.split_signs,
            incremental_alphas=replay_batch.incremental_alpha, refresh_forward=False,
        )
        from dataclasses import replace

        replay_result = replace(replay_result, base_reference_bounds=bounds)
        replay_batch.incremental_alpha, replay_batch.incremental_eta = replay_result.alpha_state, replay_result.eta_state
        implied_only = replay_batch.select(torch.tensor([0], device=upper.device))
        implied_only.decision_literals = torch.empty((1, 0), dtype=torch.long, device=upper.device)
        implied_session = ClimbSession(BaBConfig(
            root_bounds_reuse="split_refresh", climb_coarsening_enabled=False,
        ), net, OutKind.LINEAR_LE)
        implied_session.learn(implied_only, replay_result.solution.statuses, replay_result, 1)
        assert implied_session.library.core_count == 0
        objective_rows, threshold_rows = replay_result.c_rows, replay_result.thresholds
        assert objective_rows is not None and threshold_rows is not None

        def replay_test(
            candidate: SubproblemBatch,
            base_reference: Optional[Dict[int, Bounds]],
            thresholds: torch.Tensor,
            spec_count: int,
        ) -> Optional[tuple[torch.Tensor, Optional[torch.Tensor]]]:
            """Exercise replay's real proof path with explicit typed inputs."""
            assert objective_rows is not None
            return certificate_replay(
                net=net, batch=candidate, reference_bounds=conditioned,
                base_reference_bounds=base_reference, c_rows=objective_rows,
                thresholds=thresholds, m_specs=spec_count,
                out_kind=OutKind.LINEAR_LE, with_costs=False, refresh_mode="split_refresh",
            )

        full = replay_test(replay_batch, bounds, threshold_rows, 1)
        assert full is not None and bool(is_certified(full[0], OutKind.LINEAR_LE).all())
        deleted = replay_batch.select(torch.tensor([0], device=upper.device))
        deleted.split_signs = {2: torch.zeros_like(upper[:, None])}
        empty = replay_test(deleted, bounds, threshold_rows, 1)
        assert empty is not None and not bool(is_certified(empty[0], OutKind.LINEAR_LE).any())
        split_config = BaBConfig(
            climb_enabled=True, solver_tier="dual_alpha_eta",
            branching_method="babsr", bounding="depth_bound_blend",
            root_bounds_reuse="split_refresh", climb_theta=.5, climb_recheck_k=1,
        )
        validate_climb_config(split_config)
        split_session = ClimbSession(split_config, net, OutKind.LINEAR_LE)
        # The heuristic may be arbitrarily optimistic; admission cannot be.
        with patch(f"{__name__}.coarsen", side_effect=lambda packed, *args, **kwargs: torch.zeros_like(packed)):
            split_session.learn(replay_batch, replay_result.solution.statuses, replay_result, 1)
        assert torch.equal(split_session.library.pack(upper.device), torch.tensor([[-1]], device=upper.device))
        assert split_session.metrics.coarsen_replay_fallbacks == 1
        cheap = replace(replay_batch, incremental_alpha={2: torch.zeros_like(upper[:, None])})
        cheap_session = ClimbSession(split_config, net, OutKind.LINEAR_LE)
        with patch.object(DualSolver, "compute_certified_bound", autospec=True,
                          side_effect=DualSolver.compute_certified_bound) as replay_spy:
            cheap_session.learn(cheap, replay_result.solution.statuses, replay_result, 1)
        assert cheap_session.library.pack(upper.device).shape == (1, 0)
        assert replay_spy.call_count == 2, "one changed candidate must not be replayed twice"
        for bad_result, bad_batch in (
            (replace(replay_result, forward_only_certified=torch.ones(1, dtype=torch.bool)), replay_batch),
            (replay_result, replace(replay_batch, lb=replay_batch.lb / 2)),
        ):
            excluded = ClimbSession(split_config, net, OutKind.LINEAR_LE)
            with patch.object(DualSolver, "compute_certified_bound", side_effect=AssertionError("must not replay")):
                excluded.learn(bad_batch, bad_result.solution.statuses, bad_result, 1)
            assert excluded.library.core_count == 0
        failed = ClimbSession(split_config, net, OutKind.LINEAR_LE)
        with patch.object(DualSolver, "compute_certified_bound", side_effect=RuntimeError("failed solve")):
            failed.learn(replay_batch, replay_result.solution.statuses, replay_result, 1)
        assert failed.library.core_count == 0
        forward_batch = replace(replay_batch, split_signs={2: torch.zeros_like(upper[:, None])})
        forward_result = DualSolver().solve_spec_batch(
            net, bounds, OutputSpec(OutKind.LINEAR_LE, c=-upper[0], d=upper.new_tensor(.25)),
            optimize=True, dual_config=DualConfig(n_iters=0), input_shape=(1,),
            split_signs=forward_batch.split_signs,
            incremental_alphas={2: torch.ones_like(upper[:, None])},
            refresh_forward=False, forward_frame=frame,
        )
        assert forward_result.forward_only_certified is not None
        assert bool(forward_result.forward_only_certified.all())
        failed.learn(forward_batch, forward_result.solution.statuses,
                     replace(forward_result, base_reference_bounds=bounds), 1)
        assert failed.library.core_count == 0
        for bad_base, bad_thresholds, bad_count in (
            (None, threshold_rows, 1),
            (bounds, upper.expand(2, 1), 1),
            (bounds, threshold_rows, 2),
        ):
            try:
                replay_test(replay_batch, bad_base, bad_thresholds, bad_count)
            except ValueError:
                pass
            else:
                raise AssertionError("invalid replay provenance/objectives accepted")
        try:
            failed.learn(replay_batch, replay_result.solution.statuses,
                         replace(replay_result, base_reference_bounds=None), 1)
        except ValueError:
            pass
        else:
            raise AssertionError("missing base was silently replaced by conditioned bounds")

        # Independent phase enumeration/LP oracle for the linear path. The
        # numerical engine remains ACT's existing forward handlers; SciPy is
        # used only here to solve explicit two-variable phase polytopes.
        from itertools import product
        from scipy.optimize import linprog
        from act.back_end.bab.bound_refresh import _halfspace_lower
        from act.back_end.dual_tf.tf_forward import Frame

        for precision in ("float32", "float64"):
            initialize_device("cpu", precision)
            w1 = torch.tensor([[1., 0.], [1., 0.], [0., 1.]], dtype=get_default_dtype())
            b1 = torch.tensor([0., .2, 0.], dtype=get_default_dtype())
            w2 = torch.tensor([[1., -1., 0.], [0., 0., 1.]], dtype=get_default_dtype())
            b2 = torch.tensor([.3, .1], dtype=get_default_dtype())
            linear_spec = LAYER_TESTING_SPECS["layer_testing_climb_reuse"]()
            linear_spec["layers"] = linear_spec["layers"][:2] + [
                {"kind": "DENSE", "params": {"weight": w1, "bias": b1, "in_features": 2, "out_features": 3}},
                {"kind": "RELU", "params": {}},
                {"kind": "DENSE", "params": {"weight": w2, "bias": b2, "in_features": 3, "out_features": 2}},
                {"kind": "RELU", "params": {}},
                {"kind": "ASSERT", "params": {"kind": "LINEAR_LE", "c": [1., 1.], "d": 2.}},
            ]
            linear_net = factory.create_network(f"linear_{precision}", linear_spec)
            lin_box = torch.ones(2, 2, dtype=get_default_dtype())
            lin_base, _, _ = compute_forward_bounds(linear_net, -lin_box, lin_box)
            lin_codec = LiteralCodec.from_net(linear_net)
            lin_phases = {3: torch.tensor(
                [[[1., 0., -1.], [1., 0., -1.]], [[-1., 0., 1.], [-1., 0., 1.]]],
                dtype=get_default_dtype(),
            )}
            lin_packed = pack_split_matrix(lin_phases, 2, lin_codec, device=lin_box.device)
            linear_results = []
            for refined in (True, False):
                r = refresh_bounds(
                    linear_net, lin_base, lin_phases, literals=lin_packed, codec=lin_codec,
                    linear=True, support_refinement=refined, rows_cap=1,
                )
                assert r.linear_refined and r.supports is not None
                assert r.bounds[3].lb[0, 1] > .19, "same-layer halfspace must imply an active phase"
                assert r.bounds[5].lb[0, 0] > .09
                if refined:
                    assert r.supports[3].lower[0, 1].tolist() == [True, False]
                else:
                    assert bool(r.supports[3].lower[0, 1].all())
                linear_results.append(r)
            for lid in lin_base:
                assert torch.equal(linear_results[0].bounds[lid].lb, linear_results[1].bounds[lid].lb)
                assert torch.equal(linear_results[0].bounds[lid].ub, linear_results[1].bounds[lid].ub)
            dropped = {3: lin_phases[3].clone()}
            dropped[3][:, :, 0] = 0
            enlarged = refresh_bounds(linear_net, lin_base, dropped, linear=True)
            assert enlarged.bounds[3].lb[0, 1] < 0
            assert enlarged.bounds[5].lb[0, 0] < linear_results[0].bounds[5].lb[0, 0]
            wide_rows = refresh_bounds(linear_net, lin_base, lin_phases, linear=True, rows_cap=64)
            assert torch.equal(wide_rows.bounds[5].lb, linear_results[0].bounds[5].lb)
            # Unsupported conv/DAG never advertises linear refinement.
            conv_spec = LAYER_TESTING_SPECS["layer_testing_climb_admission"]()
            conv_spec["layers"][0]["params"]["shape"] = [1, 1, 2, 2]
            conv_spec["layers"].insert(2, {"kind": "CONV2D", "params": {
                "weight": torch.ones(1, 1, 1, 1, dtype=get_default_dtype()),
                "bias": torch.zeros(1, dtype=get_default_dtype()),
                "in_channels": 1, "out_channels": 1, "kernel_size": 1,
                "stride": 1, "padding": 0, "dilation": 1, "groups": 1,
                "input_shape": [1, 1, 2, 2], "output_shape": [1, 1, 2, 2],
            }})
            conv_spec["layers"][-1]["params"].update(c=[1., 1., 1., 1.], d=5.)
            conv_net = factory.create_network(f"conv_fallback_{precision}", conv_spec)
            conv_box = torch.ones(1, 1, 2, 2, dtype=get_default_dtype())
            conv_base, _, _ = compute_forward_bounds(conv_net, -conv_box, conv_box)
            assert not refresh_bounds(conv_net, conv_base, {3: torch.ones(1, 2, 4)}, linear=True).linear_refined
            feasible = infeasible = 0
            oracle_base = {lid: Bounds(box.lb[:1], box.ub[:1]) for lid, box in lin_base.items()}
            for assignments in product((-1., 1.), repeat=5):
                first = w1.new_tensor(assignments[:3])
                second = w1.new_tensor(assignments[3:])
                a2 = w2 @ (first.clamp(min=0)[:, None] * w1)
                c2 = w2 @ (first.clamp(min=0) * b1) + b2
                a_ub = torch.cat([-first[:, None] * w1, -second[:, None] * a2]).double().numpy()
                b_ub = torch.cat([first * b1, second * c2]).double().numpy()
                feasibility = linprog([0., 0.], A_ub=a_ub, b_ub=b_ub, bounds=[(-1., 1.)] * 2, method="highs")
                enum_phases = {3: first.reshape(1, 1, 3), 5: second.reshape(1, 1, 2)}
                observed = refresh_bounds(linear_net, oracle_base, enum_phases, linear=True)
                assert all(bool(torch.isfinite(box.lb).all() & torch.isfinite(box.ub).all()) for box in observed.bounds.values())
                if feasibility.status == 2:
                    infeasible += 1
                    continue
                assert feasibility.success
                feasible += 1
                for lid, a, intercept in ((3, w1, b1), (5, a2, c2)):
                    for j in range(a.shape[0]):
                        minimum = linprog(a[j].double().numpy(), A_ub=a_ub, b_ub=b_ub, bounds=[(-1., 1.)] * 2, method="highs")
                        maximum = linprog(-a[j].double().numpy(), A_ub=a_ub, b_ub=b_ub, bounds=[(-1., 1.)] * 2, method="highs")
                        assert minimum.success and maximum.success
                        tolerance = 2e-6 if precision == "float32" else 1e-10
                        assert observed.bounds[lid].lb[0, j] <= minimum.fun + float(intercept[j]) + tolerance
                        assert observed.bounds[lid].ub[0, j] >= -maximum.fun + float(intercept[j]) - tolerance
            assert feasible > 0 and infeasible > 0
            objectives = w1.new_tensor([
                [[.4, .2, -1.], [1., 1., 1.], [-1., 2., -.5]],
                [[1., -.3, .7], [-.5, 1., 2.], [1., -.5, -2.]],
            ])
            halfspace = w1.new_tensor([[1., -2., .5], [-1., .5, 2.]])
            offset = w1.new_tensor([-.3, .1])
            cube = torch.ones_like(halfspace)
            estimates = _halfspace_lower(
                objectives, w1.new_zeros(2, 3), Frame(Bounds(-cube, cube)),
                halfspace, offset,
            )
            with patch("act.back_end.bab.bound_refresh._lin_bound_lane_chunk", return_value=1):
                chunked_estimates = _halfspace_lower(
                    objectives, w1.new_zeros(2, 3), Frame(Bounds(-cube, cube)),
                    halfspace, offset,
                )
            assert torch.equal(estimates, chunked_estimates)
            with patch("act.back_end.bab.bound_refresh._lin_bound_lane_chunk", return_value=None):
                no_memory_bound = _halfspace_lower(
                    objectives, w1.new_zeros(2, 3), Frame(Bounds(-cube, cube)),
                    halfspace, offset,
                )
            assert bool(torch.isneginf(no_memory_bound).all())
            for lane in range(2):
                for row in range(3):
                    optimum = linprog(
                        objectives[lane, row].double().numpy(),
                        A_ub=-halfspace[lane:lane + 1].double().numpy(),
                        b_ub=offset[lane:lane + 1].double().numpy(),
                        bounds=[(-1., 1.)] * 3, method="highs",
                    )
                    assert optimum.success
                    error = optimum.fun - estimates[lane, row].item()
                    assert -1e-10 <= error <= (1e-4 if precision == "float32" else 1e-10)
            # Empty halfspaces cannot inject infinities or an empty root core.
            empty_bound = _halfspace_lower(
                w1[:1].unsqueeze(0), b1[:1].unsqueeze(0),
                Frame(Bounds(-lin_box[:1], lin_box[:1])),
                torch.zeros_like(lin_box[:1]), lin_box.new_tensor([-1.]),
            )
            assert bool(torch.isfinite(empty_bound).all()) and empty_bound.item() <= -1
        initialize_device("cpu", "float64")
