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
import dataclasses
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple, TypeAlias

import torch

if TYPE_CHECKING:
    from act.back_end.solver.terminal_cores import TerminalCoreLP

from act.back_end.bab import climb_tensor
from act.back_end.bab.branching.bounding import TopKBounding
from act.back_end.bab.frontier_merge import literal_set_keys, merge_frontier
from act.back_end.bab.support_tracker import (
    SupportRefreshResult,
    interval_refresh_with_supports,
)
from act.back_end.bab.node import (
    SubproblemBatch,
    _layer_neuron_count,
    slice_bounds_dict,
)
from act.back_end.core import Bounds, Net
from act.back_end.solver.solver_base import SolveStatus
from act.back_end.solver.solver_dual import (
    DualBatchResult,
    _alpha_spec_row_count,
    escaping_spec_rows,
    unproven_spec_rows,
)
from act.config.config import (
    BaBConfig,
    CLIMB_SOLVER_TIER,
    ConfigError,
    NEURON_BRANCHING_METHODS,
    NO_REFINEMENT_MODE,
    TOP_K_BOUNDINGS,
)
from act.front_end.specs import OutKind
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
        layers = sorted(
            (layer for layer in net.layers if layer.kind == "RELU"),
            key=lambda layer: layer.id,
        )
        layer_ids = tuple(layer.id for layer in layers)
        widths = tuple(_layer_neuron_count(layer) for layer in layers)
        offsets: List[int] = []
        total = 0
        for width in widths:
            offsets.append(total)
            total += width
        return cls(layer_ids, widths, tuple(offsets))

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
        table_ids = torch.tensor(self.layer_ids, dtype=torch.long, device=layer_ids.device)
        table_offsets = torch.tensor(self.offsets, dtype=torch.long, device=layer_ids.device)
        indices = torch.searchsorted(table_ids, layer_ids.to(torch.long))
        return signs.to(torch.long) * (
            table_offsets.index_select(0, indices) + neurons.to(torch.long) + 1
        )

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


Core: TypeAlias = frozenset[int]


def _core_has_no_complements(core: Core) -> bool:
    return len(core) == len({abs(literal) for literal in core})


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
    replay_time_s: float = 0.0
    insert_time_s: float = 0.0
    coarsen_time_s: float = 0.0
    propagate_time_s: float = 0.0
    incremental_pool_rounds_skipped: int = 0
    cores_inserted: int = 0
    core_literals_original: int = 0
    core_literals_retained: int = 0
    forward_certified_lanes: int = 0
    retired_unknown_lanes: int = 0
    coarsen_replay_fallbacks: int = 0
    core_height: int = 0
    active_eta_entries: int = 0
    acquisition_replay_calls: int = 0
    acquisition_replay_row_passes: int = 0
    merge_calls: int = 0
    merged_pairs: int = 0
    merge_rows_before: int = 0
    merge_rows_after: int = 0
    merge_literals_before: int = 0
    merge_literals_after: int = 0
    merge_time_s: float = 0.0
    merge_skipped_pairs: int = 0
    prebound_unit_writes: int = 0
    presplit_unit_writes: int = 0
    complement_attempts: int = 0
    complement_successes: int = 0
    kraft_sum: float = 0.0
    kraft_hist: Dict[int, int] = dataclasses.field(default_factory=dict)
    library: Optional[CoreLibrary] = None

    def add_certified(self, widths: Sequence[int], sign: int = 1) -> None:
        """Kraft coverage: each certified row/core of width ``w`` covers ``2^-w``."""
        for width in widths:
            self.kraft_sum += sign * 2.0 ** -int(width)
            self.kraft_hist[int(width)] = self.kraft_hist.get(int(width), 0) + sign

    def observe_frontier(self, pending_plus_active_rows: int) -> None:
        self.peak_frontier_rows = max(self.peak_frontier_rows, pending_plus_active_rows)

    def metadata(self) -> Dict[str, Any]:
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
            "climb_replay_time_s": self.replay_time_s,
            "climb_insert_time_s": self.insert_time_s,
            "climb_coarsen_time_s": self.coarsen_time_s,
            "climb_propagate_time_s": self.propagate_time_s,
            "climb_incremental_pool_rounds_skipped": self.incremental_pool_rounds_skipped,
            "climb_cores_inserted": self.cores_inserted,
            "climb_cores_subsumed": self.library.subsumed if self.library is not None else 0,
            "climb_cores_resolved": self.library.resolved if self.library is not None else 0,
            "climb_cores_evicted": self.library.evicted if self.library is not None else 0,
            "climb_core_literals_original": self.core_literals_original,
            "climb_core_literals_retained": self.core_literals_retained,
            "climb_forward_certified_lanes": self.forward_certified_lanes,
            "climb_retired_unknown_lanes": self.retired_unknown_lanes,
            "climb_coarsen_replay_fallbacks": self.coarsen_replay_fallbacks,
            "climb_core_height": self.core_height,
            "climb_active_eta_entries": self.active_eta_entries,
            "climb_acquisition_replay_calls": self.acquisition_replay_calls,
            "climb_acquisition_replay_row_passes": self.acquisition_replay_row_passes,
            "climb_merge_calls": self.merge_calls,
            "climb_merged_pairs": self.merged_pairs,
            "climb_merge_rows_before": self.merge_rows_before,
            "climb_merge_rows_after": self.merge_rows_after,
            "climb_merge_literals_before": self.merge_literals_before,
            "climb_merge_literals_after": self.merge_literals_after,
            "climb_merge_time_s": self.merge_time_s,
            "climb_merge_skipped_pairs": self.merge_skipped_pairs,
            "climb_prebound_unit_writes": self.prebound_unit_writes,
            "climb_presplit_unit_writes": self.presplit_unit_writes,
            "climb_unit_writes": self.prebound_unit_writes + self.presplit_unit_writes,
            "climb_cores_covering_resolved": (
                self.library.covering_resolved if self.library is not None else 0
            ),
            "climb_complement_attempts": self.complement_attempts,
            "climb_complement_successes": self.complement_successes,
            # Kraft sum over the library's non-subsumed cores (resolvents
            # included); the per-lane sum counts every certified lane, so it
            # over-counts duplicate/subsumed cores but misses resolvents.
            "climb_kraft_sum": (
                sum(2.0 ** -len(core) for core in self.library.cores)
                if self.library is not None
                else 0.0
            ),
            "climb_kraft_sum_lanes": self.kraft_sum,
            "climb_kraft_certified": sum(self.kraft_hist.values()),
            "climb_kraft_hist": {w: c for w, c in sorted(self.kraft_hist.items()) if c},
        }


def _row_literal_counts(rows: SubproblemBatch) -> torch.Tensor:
    counts = torch.zeros(rows.batch_size, dtype=torch.long, device=rows.lb.device)
    if rows.batch_size == 0:
        return counts
    for signs in (rows.split_signs or {}).values():
        counts += (signs.reshape(signs.shape[0], signs.shape[1], -1)[:, 0] != 0).sum(dim=1).to(counts.device)
    return counts


_MASK64 = (1 << 64) - 1


def _literal_hash(literal: int) -> int:
    """splitmix64 of a codec literal; row hashes are sums, so ``H(R - {d}) = H(R) - z(d)``."""
    value = (literal * 0x9E3779B97F4A7C15) & _MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
    return value ^ (value >> 31)


def _row_hash(literals: Sequence[int]) -> int:
    return sum(_literal_hash(literal) for literal in literals) & _MASK64


def _row_literal_count(rows: SubproblemBatch) -> int:
    return sum(
        int((signs.reshape(signs.shape[0], signs.shape[1], -1)[:, 0] != 0).sum())
        for signs in (rows.split_signs or {}).values()
    )


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
    """Query-local certified phase cores with subsumption and resolution."""

    def __init__(self, max_cores: int) -> None:
        if max_cores < 1:
            raise ValueError("max_cores must be positive")
        self.max_cores = max_cores
        self.subsumed = 0
        self.resolved = 0
        self.evicted = 0
        self.covering_resolved = 0
        self.revision = 0
        self.last_new_cores: Tuple[Core, ...] = ()
        self._cores: List[Core] = []
        self._pack_cache: Optional[Tuple[Tuple[Core, ...], torch.device, torch.Tensor]] = None

    @property
    def has_empty_core(self) -> bool:
        """The empty core certifies every row: the query is proved."""
        return any(not core for core in self._cores)

    @property
    def core_count(self) -> int:
        return len(self._cores)

    @property
    def cores(self) -> Tuple[Core, ...]:
        return tuple(self._cores)

    def _admit(self, literals: Core) -> int:
        """Insert unless subsumed; drop strict supersets. Returns removed count."""
        assert _core_has_no_complements(literals), (
            "CLIMB invariant violated: admitted core contains complementary literals "
            f"(core_size={len(literals)})"
        )
        before = len(self._cores)
        self._cores = [core for core in self._cores if not literals < core]
        self._cores.append(literals)
        return before + 1 - len(self._cores)

    def _is_subsumed(self, literals: Core) -> bool:
        return any(core <= literals for core in self._cores)

    def insert(self, packed: torch.Tensor) -> int:
        counters_before = (self.subsumed, self.resolved, self.evicted)
        cores_before = tuple(self._cores)
        inserted = 0
        admitted: List[Core] = []
        host_rows = packed.tolist() if climb_tensor.ENABLED else None
        for lane in range(packed.shape[0]):
            literals = (frozenset(literal for literal in host_rows[lane] if literal)
                        if host_rows is not None
                        else frozenset(packed[lane, packed[lane] != 0].tolist()))
            if self._is_subsumed(literals):
                self.subsumed += 1
                continue
            self.subsumed += self._admit(literals)
            admitted.append(literals)
            inserted += 1

        worklist = list(admitted)
        while worklist:
            core = worklist.pop(0)
            if core not in self._cores:
                continue
            while (found := self._find_resolvent(core)) is not None:
                resolvent, covering = found
                self._admit(resolvent)
                self.resolved += 1
                self.covering_resolved += int(covering)
                admitted.append(resolvent)
                worklist.append(resolvent)
                if core not in self._cores:
                    break

        self._cores.sort(key=len)
        self.evicted += max(0, len(self._cores) - self.max_cores)
        self._cores = self._cores[: self.max_cores]
        counters_after = (self.subsumed, self.resolved, self.evicted)
        assert _counters_non_decreasing(counters_before, counters_after), (
            "CLIMB invariant violated: core-library counters decreased "
            f"(before={counters_before}, after={counters_after})"
        )
        if tuple(self._cores) != cores_before:
            self.revision += 1
        kept = set(self._cores)
        self.last_new_cores = tuple(core for core in admitted if core in kept)
        return inserted

    def _find_resolvent(self, core: Core) -> Optional[Tuple[Core, bool]]:
        """Covering resolution of ``core`` against the library, if any.

        ``C1 | {l}`` and ``C2 | {-l}`` with ``C1 <= C2`` resolve to ``C2`` (the
        longer side minus its pivot), which subsumes that side. Returns the
        first unsubsumed resolvent and whether it was covering (unequal sides).
        """
        for other in list(self._cores):
            for short, long in ((core, other), (other, core)):
                if len(short) > len(long):
                    continue
                extra = short - long
                if len(extra) != 1:
                    continue
                (pivot,) = extra
                if -pivot not in long:
                    continue
                resolvent = long - {-pivot}
                if not self._is_subsumed(resolvent):
                    return resolvent, len(short) != len(long)
        return None

    def pack(self, device: torch.device) -> torch.Tensor:
        if climb_tensor.ENABLED:
            cores = tuple(self._cores)
            cached = self._pack_cache
            if cached is None or cached[1] != device or cached[0] != cores:
                cached = (cores, device, climb_tensor.pack_cores(cores, device))
                self._pack_cache = cached
            return cached[2]
        width = max((len(core) for core in self._cores), default=0)
        packed = torch.zeros((len(self._cores), width), dtype=torch.long, device=device)
        for row, core in enumerate(self._cores):
            values = sorted(core, key=lambda literal: (abs(literal), literal))
            packed[row, : len(values)] = torch.tensor(values, dtype=torch.long, device=device)
        return packed


def apply_literal_assignments(
    batch: SubproblemBatch,
    assignments: torch.Tensor,
    active_variables: torch.Tensor,
    codec: LiteralCodec,
) -> SubproblemBatch:
    """Decode compact inferred phases back to a lossless subproblem batch."""
    signs = {key: value.clone() for key, value in (batch.split_signs or {}).items()}
    specs = 1
    if signs:
        specs = next(iter(signs.values())).shape[1]
    elif batch.incremental_alpha:
        specs = _alpha_spec_row_count(batch.incremental_alpha)
    elif batch.incremental_eta:
        specs = next(iter(batch.incremental_eta.values())).shape[1]
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


@torch.no_grad()
def _propagate_with_indices(
    batches: Sequence[SubproblemBatch],
    library: CoreLibrary,
    codec: LiteralCodec,
    *,
    core_chunk: int,
) -> tuple[Tuple[SubproblemBatch, ...], Tuple[torch.Tensor, ...]]:
    """Apply certified cores jointly to pending and active groups to fixpoint.

    ``core_chunk`` caps the cores per matmul chunk; it bounds memory only, the
    fixpoint is independent of it.
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
    if climb_tensor.ENABLED:
        assignments, active = climb_tensor.propagate_dense(
            assignments, core_dense, core_chunk=core_chunk, compute_dtype=compute_dtype
        )
    while not climb_tensor.ENABLED and bool(active.any().item()):
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
    kept_indices: List[torch.Tensor] = []
    offset = 0
    for batch, size in zip(batches, group_sizes):
        group_active = active[offset : offset + size]
        keep = torch.where(group_active)[0]
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
        kept_indices.append(keep)
        offset += size
    return tuple(result_batches), tuple(kept_indices)


def propagate(
    batches: Sequence[SubproblemBatch],
    library: CoreLibrary,
    codec: LiteralCodec,
    *,
    core_chunk: int,
) -> Tuple[SubproblemBatch, ...]:
    """Apply certified cores and return each group's surviving rows."""
    propagated, _ = _propagate_with_indices(
        batches, library, codec, core_chunk=core_chunk
    )
    return propagated


def is_certified(
    slack: torch.Tensor,
    out_kind: str,
    reference: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Per-lane certification mask over ``[N, M]`` slack rows.

    Byte-consistent with the ``DualSolver.solve_spec_batch`` lane statuses: an
    ``UNSAFE_LINEAR`` lane certifies when any finite row clears the
    ``escaping_spec_rows`` band of ``reference`` (the bounds the slack was
    derived from, broadcastable to ``slack``; defaults to the slack itself,
    which is only adequate when those bounds are O(1)); every other kind
    certifies only when no row is ``unproven_spec_rows`` for that kind
    (finite and non-negative for LINEAR_LE / RANGE; finite and clearing the
    same band for TOP1_ROBUST / MARGIN_ROBUST, whose safe set is open).
    """
    reference_rows = slack if reference is None else reference
    if out_kind == OutKind.UNSAFE_LINEAR:
        return escaping_spec_rows(slack, reference_rows).any(dim=1)
    return ~unproven_spec_rows(slack, reference_rows, out_kind).any(dim=1)


def replay_reference(thresholds: torch.Tensor) -> torch.Tensor:
    """Per-lane ``[N, 1]`` magnitude reference for a replayed slack.

    A replayed ``UNSAFE_LINEAR`` slack is pinned to one row, whose margin is
    unknown to the caller; near the boundary that margin is within the band of
    its threshold, so the lane's largest ``|threshold|`` is an upper (hence
    conservative) reference for ``is_certified``.
    """
    return thresholds.abs().amax(dim=1, keepdim=True)


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
    phase_slope_signs: Optional[Dict[int, torch.Tensor]] = None,
    cost_fn: Optional[Callable[..., torch.Tensor]] = None,
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
        phase_slope_signs=phase_slope_signs,
    )
    n_lanes = batch.batch_size
    # Replay validates the backward certificate alone; the forward part of the
    # solver's intersected bound is not produced by alpha/eta/nu.
    dual_margins = result.dual_margins if result.dual_margins is not None else result.margins
    margins = dual_margins.reshape(n_lanes, m_specs)
    slack = margins - thresholds.reshape(n_lanes, m_specs).to(margins)
    pinned_rows: Optional[torch.Tensor] = None
    if out_kind == OutKind.UNSAFE_LINEAR:
        passing = escaping_spec_rows(slack, margins)
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
        assert codec is not None
        if cost_fn is not None:
            costs = cost_fn(slack, eta, nu, pinned_rows)
        else:
            costs = _literal_costs(
                literals, codec, slack, eta, nu, reference_bounds, pinned_rows
            )
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
    """Full ReLU Eq. 12 costs ``(eta + relu(-nu)) * m`` in ``slack.dtype``, ``[N, M, W]``."""
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
def _supported_literal_costs(
    literals: torch.Tensor,
    codec: LiteralCodec,
    slack: torch.Tensor,
    eta: Dict[int, torch.Tensor],
    nu: Dict[int, torch.Tensor],
    base_bounds: Dict[int, Bounds],
    reference_bounds: Dict[int, Bounds],
    tracked: Optional[SupportRefreshResult],
    pinned_rows: Optional[torch.Tensor],
) -> torch.Tensor:
    """Eq. (cost) under split-refreshed bounds: ``c_l = eta_l m_l + sum_{i: l in E_i} eps_i``.

    ``m_l`` and every ``eps_i`` are measured over the split-free intervals
    ``base_bounds``, which allow every retained subset. ``eps_i`` is
    ``max(0, -Lambda_i)`` times the largest violation, over that interval, of
    the upper line frozen on ``reference_bounds``: the chord of an unstable
    neuron, ``z`` for an active and ``0`` for an inactive split-dependent phase.
    ``E_i`` are the W1.3 endpoint supports. A missing tracker or layer, an
    unknown support, or a protected literal makes the literal undeletable.
    """
    zero_nu = {layer_id: torch.zeros_like(value) for layer_id, value in nu.items()}
    costs = _literal_costs(literals, codec, slack, eta, zero_nu, base_bounds, None)
    active = (literals != 0).unsqueeze(1)
    layers = [
        (nu.get(lid), reference_bounds.get(lid), base_bounds.get(lid),
         tracked.supports.get(lid) if tracked is not None else None)
        for lid in codec.layer_ids
    ]
    if tracked is None or any(item is None for layer in layers for item in layer):
        costs.masked_fill_(active.expand_as(costs), torch.inf)
        layers = []
    for lam, reference, base, support in layers:
        lower, upper = reference.lb.flatten(1).to(costs), reference.ub.flatten(1).to(costs)
        lower0, upper0 = base.lb.flatten(1).to(costs), base.ub.flatten(1).to(costs)
        zero = torch.zeros_like(lower)
        width = (upper - lower).clamp(min=torch.finfo(costs.dtype).tiny)
        chord = torch.maximum(upper * (lower - lower0), -lower * (upper0 - upper)) / width
        chord = torch.where((lower < 0) & (upper > 0), chord.clamp(min=0), zero)
        phase_on = torch.where(lower >= 0, (-lower0).clamp(min=0), zero)
        phase_off = torch.where(upper <= 0, upper0.clamp(min=0), zero)
        weight = (-lam.to(costs)).clamp(min=0)
        known = support.known.unsqueeze(1)
        for unit, mask in (
            (chord, support.combined),
            (phase_on, support.lower),
            (phase_off, support.upper),
        ):
            eps = weight * unit.unsqueeze(1)
            unknown = ((eps > 0) & ~known).any(dim=2, keepdim=True)
            costs.masked_fill_(unknown & active, torch.inf)
            costs += torch.einsum("bmn,bnw->bmw", eps * known.to(eps), mask.to(eps))
    if tracked is not None:
        costs.masked_fill_(tracked.protected_literals.unsqueeze(1) & active, torch.inf)
    costs = torch.nan_to_num(costs, nan=torch.inf, posinf=torch.inf)
    if pinned_rows is not None:
        rows = pinned_rows.to(device=costs.device, dtype=torch.long)
        costs = costs.gather(1, rows[:, None, None].expand(-1, 1, costs.shape[2]))
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
    """Alg. 2 vector-budget deletion plus at most ``recheck_k`` one-step rechecks.

    Per row ``m`` the budget is ``theta * slack_m - delta_m`` with
    ``delta_m = delta_abs + delta_rel * |slack_m|``; rows with
    ``slack_m <= delta_m`` never fund a deletion. The budget arithmetic runs
    in the dtype of ``costs``/``slack``. Literals are ordered by
    ``(max_m cost_m / budget_m, column, slot)`` and the longest feasible prefix
    is deleted. At ``(theta, recheck_k) == (0, 0)``, exactly-zero-cost literals
    are deleted by a linear mask instead of entering the budget/sort path. A
    recheck retries the next literal in that same order for the first
    ``recheck_k`` lanes; ``recheck`` returns the ``BoolTensor[K]`` certification
    mask that confirms each candidate deletion.
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
    if theta == 0.0 and recheck_k == 0:
        delta = delta_abs + delta_rel * slack.abs()
        eligible = (
            torch.isfinite(slack).all(dim=1)
            & torch.isfinite(costs).all(dim=(1, 2))
            & (slack > delta).all(dim=1)
        )
        zero_cost = (costs == 0).all(dim=1)
        retained &= ~(eligible[:, None] & zero_cost)
        return literals.masked_fill(~retained, 0)
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


CLIMB_ROOT_BOUNDS_REUSE_MODES: Tuple[str, ...] = ("none", "plain")
"""Root-bound reuse modes whose reference bounds depend only on the lane box.

``plain`` hands every descendant the root forward dict with INPUT / INPUT_SPEC
replaced by the lane box (bab.py ``_solve_dual_batch``); root-level
``intermediate_refine`` tightens that dict once, on the root box, before any
split exists. Both keep the replay reference free of split literals, which is
the premise of core reuse. ``per_subproblem_refine`` hardens bounds from the
lane's split signs and stays rejected; ``split_refresh`` is the one
split-dependent mode (D21), see ``CLIMB_SPLIT_REFRESH_MODE``.
"""

CLIMB_SPLIT_REFRESH_MODE: str = "split_refresh"
"""D21 (amends D7): lanes use split-refreshed bounds, so ``ClimbSession`` replays
every candidate core with bounds refreshed from the core's own literals, valid
on the whole core region, and charges literal deletions the E_i support costs.
"""


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
    from act.back_end.bab.linear_refresh import LINEAR_REFRESH_MODE

    reuse_modes = CLIMB_ROOT_BOUNDS_REUSE_MODES + (CLIMB_SPLIT_REFRESH_MODE, LINEAR_REFRESH_MODE)
    if config.root_bounds_reuse not in reuse_modes:
        failures.append(f"root_bounds_reuse must be one of {reuse_modes}")
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


class ClimbSession:
    """Own the query-local CLIMB library, metrics, and BaB-loop hooks."""

    def __init__(
        self,
        config: BaBConfig,
        net: Net,
        out_kind: str,
        budget_remaining: Optional[Callable[[], float]] = None,
    ) -> None:
        self._net = net
        self._out_kind = out_kind
        self._budget_remaining = budget_remaining
        self._coarsening_enabled = config.climb_coarsening_enabled
        self._propagation_enabled = config.climb_propagation_enabled
        self._theta = config.climb_theta
        self._delta_abs = config.climb_delta_abs
        self._delta_rel = config.climb_delta_rel
        self._recheck_k = config.climb_recheck_k
        self._propagate_core_chunk = config.climb_propagate_core_chunk
        self._refresh_mode = config.root_bounds_reuse
        self._core_local = self._refresh_mode in (CLIMB_SPLIT_REFRESH_MODE, "split_refresh_linear")
        self._merge_enabled = config.climb_merge_enabled
        self._presplit_enabled = config.climb_presplit_propagation_enabled
        self._steer = config.climb_steer
        self._seen_cores: set[frozenset[int]] = set()
        self._merge_armed = False
        self._complement_enabled = config.climb_complement_bounding
        self._terminal_cores_enabled = config.climb_terminal_cores
        self._lin_support_refinement = config.climb_lin_support_refinement
        self.complement_bounder: Optional[
            Callable[[SubproblemBatch], Optional[DualBatchResult]]
        ] = None
        self.oom_handler: Optional[Callable[[], None]] = None
        self._in_complement = False
        self._bounded_unresolved: set[bytes] = set()
        self._fresh_keys: set[bytes] = set()
        self._split_parent_hashes: set[int] = set()
        self.codec = LiteralCodec.from_net(net)
        self.library = CoreLibrary(config.climb_max_cores)
        self.metrics = ClimbMetrics(enabled=True, library=self.library)
        self._terminal_core_lp: Optional[TerminalCoreLP] = None
        self._active_library_revision: Optional[int] = None

    @classmethod
    def from_config(
        cls,
        config: BaBConfig,
        net: Net,
        out_kind: str,
        budget_remaining: Optional[Callable[[], float]] = None,
    ) -> Optional[ClimbSession]:
        """``budget_remaining`` (seconds left in the BaB budget, None =
        unbounded) lets ``learn`` skip coarsening once the budget is spent."""
        validate_climb_config(config)
        if not config.climb_enabled:
            return None
        return cls(config, net, out_kind, budget_remaining)

    def before_pop(self, pool: TopKBounding) -> None:
        if not self._propagation_enabled:
            self.metrics.observe_frontier(len(pool))
            return
        if self.library.core_count > 0:
            # The copy and Propagate leave the pool untouched until
            # apply_propagation, so a CUDA OOM here just skips the round.
            try:
                pending = pool.propagation_batch(self.library.revision)
                propagation = kept = None
                if pending is not None:
                    selected, pending_before = pending
                    # New rows see every core. A library revision marks all existing
                    # rows stale, so every row also sees every newly admitted core.
                    propagate_started = time.perf_counter()
                    propagation, kept = _propagate_with_indices(
                        [pending_before],
                        self.library,
                        self.codec,
                        core_chunk=self._propagate_core_chunk,
                    )
                oom = False
            except torch.cuda.OutOfMemoryError:
                oom = True
            if oom:
                pending = propagation = kept = None
                self._report_oom()
                self.metrics.observe_frontier(len(pool))
                return
            if pending is None:
                self.metrics.incremental_pool_rounds_skipped += 1
            else:
                assert propagation is not None and kept is not None
                self.metrics.propagate_time_s += time.perf_counter() - propagate_started
                pending_after = propagation[0]
                self.metrics.prebound_discharged += (
                    pending_before.batch_size - pending_after.batch_size
                )
                self.metrics.prebound_unit_writes += self._note_writes(
                    pending_before, kept[0], pending_after
                )
                pool.apply_propagation(
                    selected,
                    kept[0],
                    pending_after,
                    self.library.revision,
                )
                if pool.empty:
                    return
        if not self._presplit_enabled and self._merge_enabled and self._merge_armed:
            self._merge(pool, None)
        self._active_library_revision = self.library.revision
        self.metrics.observe_frontier(len(pool))

    def after_pop(self, frontier_rows: int) -> None:
        self.metrics.observe_frontier(frontier_rows)

    def after_main_bound(
        self, k_actual: int, batch: Optional[SubproblemBatch] = None
    ) -> None:
        self.metrics.main_bound_calls += 1
        self.metrics.main_bound_row_passes += k_actual
        if batch is not None and batch.split_signs:
            self.metrics.active_eta_entries += sum(
                int((signs != 0).sum()) for signs in batch.split_signs.values()
            )

    def learn(
        self,
        batch: SubproblemBatch,
        statuses: Sequence[str],
        dual_solve_result: DualBatchResult,
        k_actual: int,
        learnable: Optional[torch.Tensor] = None,
    ) -> None:
        # D22: lanes with ``learnable`` False (input box shrunk below the root
        # box) are never learned: their cores would not hold on the root box.
        # H2: replay -> coarsen -> insert. Certified lanes only; replay reads
        # the immutable forward snapshot, the theta budget coarsens the
        # validated vector, and the survivor is inserted as a core. Once the
        # BaB budget is spent after the replay, the replay-certified rows are
        # inserted whole: no coarsening rechecks.
        forward_only = getattr(dual_solve_result, "forward_only_certified", None)
        unsat_lanes = []
        learnable_rows = learnable.tolist() if learnable is not None else None
        forward_rows = forward_only.tolist() if forward_only is not None else None
        for i, status in enumerate(statuses):
            if status != SolveStatus.UNSAT:
                continue
            if learnable_rows is not None and not learnable_rows[i]:
                continue
            if forward_rows is not None and forward_rows[i]:
                # Certified by the forward bound only: no dual certificate to
                # replay or learn from; the lane is pruned as-is.
                self.metrics.forward_certified_lanes += 1
                continue
            unsat_lanes.append(i)
        certified_widths = _row_literal_counts(batch).tolist()
        self.metrics.add_certified(
            [int(certified_widths[i]) for i, status in enumerate(statuses) if status == SolveStatus.UNSAT]
        )
        unsat_idx = torch.tensor(unsat_lanes, device=batch.lb.device, dtype=torch.long)
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
        # The replay reference is the exact unhardened forward snapshot the
        # main solve consumed, sliced by certified lane; it is never recomputed
        # so Eq. 12 magnitudes and the local clamps read the same immutable bounds.
        # Invariant (D7): that snapshot depends only on the lane input box -
        # a fresh forward pass, or under root_bounds_reuse="plain" the root dict
        # (optionally refined once on the root box) with the lane box swapped
        # in - never on the lane's split literals. Split-dependent hardening
        # (split_refresh, per_subproblem_refine) is rejected by
        # validate_climb_config, so a learned core stays valid on every
        # sub-box whose signs contain it.
        replay_bounds = slice_bounds_dict(
            dual_solve_result.reference_bounds, unsat_idx
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
        base_bounds: Optional[Dict[int, Bounds]] = None
        cost_fn: Optional[Callable[..., torch.Tensor]] = None
        if self._core_local:
            # D21: the lane's refreshed bounds hold only on its own region, so
            # replay reads bounds refreshed from the row's literals over the
            # split-free base, and costs add the E_i support charges.
            split_free = (
                getattr(dual_solve_result, "base_reference_bounds", None)
                or dual_solve_result.reference_bounds
            )
            base_bounds = slice_bounds_dict(split_free, unsat_idx)
            refreshed = self._core_local_bounds(base_bounds, replay_batch.split_signs)
            if refreshed is None:
                return
            replay_bounds = refreshed
            if self._refresh_mode == CLIMB_SPLIT_REFRESH_MODE:
                tracked = interval_refresh_with_supports(
                    self._net, base_bounds, replay_batch.split_signs or {}, packed, self.codec
                )
            else:
                # Halfspace refresh depends on same-layer decisions as well.
                from act.back_end.bab.linear_refresh import conservative_supports

                tracked = conservative_supports(
                    self._net, refreshed, packed, self.codec,
                    same_layer_supports=self._lin_support_refinement,
                )

            def cost_fn(slack, eta, nu, pinned_rows):
                return _supported_literal_costs(
                    packed, self.codec, slack, eta, nu,
                    base_bounds, refreshed, tracked, pinned_rows,
                )

        replay_started = time.perf_counter()
        self.metrics.acquisition_replay_calls += 1
        self.metrics.acquisition_replay_row_passes += replay_batch.batch_size
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
            cost_fn=cost_fn,
        )
        self.metrics.replay_time_s += time.perf_counter() - replay_started
        if replay is not None:
            slack, costs = replay
            valid = is_certified(
                slack, self._out_kind, replay_reference(replay_thresholds)
            )
        else:
            slack = costs = valid = None
        if costs is None or valid is None or not bool(valid.any().item()):
            return
        budget_slack = slack
        assert budget_slack is not None
        costs[~valid] = torch.inf
        context = _ReplayContext(
            replay_batch, replay_bounds, replay_c_lanes, replay_thresholds, replay_m_specs,
            base_bounds,
        )
        coarsen_started = time.perf_counter()
        budget_spent = (
            self._budget_remaining is not None and self._budget_remaining() <= 0.0
        )
        if self._coarsening_enabled and not budget_spent:
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
        original_literals = int((packed != 0).sum())
        coarsened_literals = int((retained != 0).sum())
        assert _retained_literals_are_subset(packed, retained), (
            "CLIMB invariant violated: coarsening added or changed a literal "
            f"(rows={packed.shape[0]}, width={packed.shape[1]}, "
            f"original_literals={original_literals}, retained_literals={coarsened_literals})"
        )
        retained = self._admit_replay_certified_rows(context, packed, retained, valid)
        self.metrics.coarsen_time_s += time.perf_counter() - coarsen_started
        retained_literals = int((retained != 0).sum())
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
        if valid_cores.numel() > 0:
            self.metrics.core_height = max(
                self.metrics.core_height,
                int((valid_cores != 0).sum(dim=1).max().item()),
            )
        core_rows = valid_cores.tolist()
        fresh = {frozenset(lit for lit in row if lit) for row in core_rows}
        self.metrics.add_certified([len(core) for core in map(frozenset, core_rows)], 1)
        self.metrics.add_certified(
            [int(certified_widths[unsat_lanes[int(i)]]) for i in valid_rows.tolist()], -1
        )
        insert_started = time.perf_counter()
        self.metrics.cores_inserted += self.library.insert(valid_cores)
        self.metrics.insert_time_s += time.perf_counter() - insert_started
        self._register_new_cores(fresh)
        if self._complement_enabled and not self._in_complement and self.complement_bounder is not None:
            self._bound_complements(
                replay_batch.select(valid_rows),
                packed.index_select(0, valid_rows),
                [frozenset(lit for lit in row if lit) for row in core_rows],
            )

    def _report_oom(self) -> None:
        if self.oom_handler is not None:
            self.oom_handler()
        else:
            torch.cuda.empty_cache()

    @property
    def proved(self) -> bool:
        """An empty core (learned or resolved) certifies the whole query."""
        return self.library.has_empty_core

    def _register_new_cores(self, learned: Optional[set[Core]] = None) -> None:
        """Unseen learned cores and resolvents arm Merge (Alg. 1)."""
        candidates = set(learned or ()) | set(self.library.last_new_cores)
        if candidates - self._seen_cores:
            self._merge_armed = True
            self._seen_cores |= candidates

    def _note_writes(
        self, before: SubproblemBatch, kept: torch.Tensor, after: SubproblemBatch
    ) -> int:
        """Count Propagate unit writes; written rows become merge-fresh."""
        counts_before = _row_literal_counts(before).index_select(0, kept.to(before.lb.device))
        written = _row_literal_counts(after) - counts_before.to(after.lb.device)
        rows = torch.where(written > 0)[0]
        if self._merge_enabled and rows.numel():
            self._fresh_keys.update(literal_set_keys(after.select(rows)))
        return int(written.sum())

    def record_certified(self, rows: SubproblemBatch) -> None:
        """Kraft accounting for rows certified outside ``learn`` (terminal LP)."""
        self.metrics.add_certified(_row_literal_counts(rows).tolist())

    def learn_terminal_cores(
        self, rows: SubproblemBatch, root_bounds: Bounds,
    ) -> Dict[str, int]:
        """Admit terminal cores on the root box, then use the usual library.

        Shrunk boxes cannot justify query-global phase cores (D22). All
        deletions, including zero-cost ones, require the admission LP.
        """
        import logging

        from act.back_end.solver.terminal_cores import TerminalCoreLP
        from act.back_end.solver.terminal_lp import terminal_lp_mode

        self.record_certified(rows)
        stats: Dict[str, int] = {}
        if not self._terminal_cores_enabled or terminal_lp_mode(self._net) != "affine":
            return stats
        remaining = self._budget_remaining or (lambda: float("inf"))
        original_widths = _row_literal_counts(rows).tolist()
        learned: List[torch.Tensor] = []
        widths: List[int] = []
        source_widths: List[int] = []
        started = time.perf_counter()
        for lane in range(rows.batch_size):
            if remaining() <= 0.:
                break
            if not (
                torch.equal(rows.lb[lane].reshape(-1), root_bounds.lb.reshape(-1))
                and torch.equal(rows.ub[lane].reshape(-1), root_bounds.ub.reshape(-1))
            ):
                continue
            try:
                learner = self._terminal_core_lp
                if learner is None:
                    learner = TerminalCoreLP(self._net, root_bounds)
                    self._terminal_core_lp = learner
                signs = {key: value[lane:lane + 1] for key, value in
                         (rows.split_signs or {}).items()}
                result = learner.learn(
                    signs, theta=self._theta, delta_abs=self._delta_abs,
                    delta_rel=self._delta_rel, remaining=remaining,
                    coarsening=self._coarsening_enabled,
                )
            except (ValueError, RuntimeError, ArithmeticError) as error:
                logging.getLogger(__name__).warning("terminal core learning unavailable: %s", error)
                continue
            if result is None:
                continue
            signs, _, metadata = result
            packed = pack_split_matrix(signs, 1, self.codec, device=rows.lb.device)
            learned.append(packed)
            width = int((packed != 0).sum().item())
            widths.append(width)
            source_widths.append(original_widths[lane])
            key = f"terminal_core_width_{width}"
            stats[key] = stats.get(key, 0) + 1
            stats["terminal_core_fallbacks"] = stats.get("terminal_core_fallbacks", 0) + int(
                metadata["fallback_full_row"]
            )
        if learned:
            cores = torch.zeros(len(learned), max(core.shape[1] for core in learned),
                                dtype=torch.long, device=rows.lb.device)
            for lane, core in enumerate(learned):
                cores[lane, :core.shape[1]] = core[0]
            self.metrics.core_literals_original += sum(source_widths)
            self.metrics.core_literals_retained += sum(widths)
            self.metrics.core_height = max(self.metrics.core_height, max(widths))
            self.metrics.add_certified(source_widths, -1)
            self.metrics.add_certified(widths)
            insert_started = time.perf_counter()
            self.metrics.cores_inserted += self.library.insert(cores)
            self.metrics.insert_time_s += time.perf_counter() - insert_started
            self._register_new_cores({frozenset(lit for lit in row if lit)
                                      for row in cores.tolist()})
            stats["terminal_cores_learned"] = len(learned)
        stats["terminal_core_time_ms"] = round(1000. * (time.perf_counter() - started))
        return stats

    def _bound_complements(
        self, rows: SubproblemBatch, packed: torch.Tensor, cores: List[Core]
    ) -> None:
        """S6 speculative complement bounding, at most once per new core.

        For a new core ``C`` with deepest retained decision ``l_h`` (found by
        walking the row's split parents), bound ``(C - {l_h}) | {-l_h}`` off the
        frontier. A certified complement is learned (and covering-resolves with
        ``C``); anything else is discarded, so frontier and termination are unaffected.
        """
        new_cores = set(self.library.last_new_cores)
        lanes: List[int] = []
        literal_rows: List[List[int]] = []
        for lane, (core, row) in enumerate(zip(cores, packed.tolist())):
            if core not in new_cores:
                continue
            literals = [lit for lit in row if lit]
            row_hash = _row_hash(literals)
            pivot = None
            while pivot is None and literals:
                last = next(
                    (lit for lit in literals
                     if (row_hash - _literal_hash(lit)) & _MASK64 in self._split_parent_hashes),
                    None,
                )
                if last is None:
                    break
                if last in core:
                    pivot = last
                literals = [lit for lit in literals if lit != last]
                row_hash = (row_hash - _literal_hash(last)) & _MASK64
            if pivot is None:
                continue
            new_cores.discard(core)
            lanes.append(lane)
            literal_rows.append(sorted(core - {pivot}) + [-pivot])
        if not lanes:
            return
        batch = rows.select(torch.tensor(lanes, dtype=torch.long, device=rows.lb.device))
        width = max(len(row) for row in literal_rows)
        candidate = torch.zeros((len(lanes), width), dtype=torch.long, device=rows.lb.device)
        for index, row in enumerate(literal_rows):
            candidate[index, : len(row)] = torch.tensor(row, dtype=torch.long)
        if batch.split_signs is not None:
            batch.split_signs = {lid: torch.zeros_like(v) for lid, v in batch.split_signs.items()}
        variables = torch.unique(candidate[candidate != 0].abs() - 1, sorted=True)
        dense = torch.zeros((len(lanes), variables.numel()), dtype=torch.int8, device=candidate.device)
        lane_idx, slots = torch.nonzero(candidate, as_tuple=True)
        dense[lane_idx, torch.searchsorted(variables, candidate[lane_idx, slots].abs() - 1)] = (
            candidate[lane_idx, slots].sign().to(torch.int8)
        )
        apply_literal_assignments(batch, dense, variables, self.codec)
        if batch.incremental_eta is not None and batch.split_signs is not None:
            for lid, eta in batch.incremental_eta.items():
                signs = batch.split_signs.get(lid)
                if signs is not None and signs.shape == eta.shape:
                    eta.masked_fill_(signs == 0, 0)
        batch.lower_bound = None
        self.metrics.complement_attempts += batch.batch_size
        result = self.complement_bounder(batch)
        if result is None:
            return
        statuses = result.solution.statuses
        self.metrics.complement_successes += sum(status == SolveStatus.UNSAT for status in statuses)
        self._in_complement = True
        try:
            self.learn(batch, statuses, result, batch.batch_size)
        finally:
            self._in_complement = False

    def _admit_replay_certified_rows(
        self,
        context: _ReplayContext,
        packed: torch.Tensor,
        retained: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Restore the full certified row wherever the coarsened row fails replay.

        Dropped literals replay with their phase-slope strip (1 active, 0
        inactive), so Eq. 12's ``(eta + relu(-nu)) * m`` charge is exact. The
        replay gate remains the admission safety net: a retained row is inserted
        only if the same computation used by ``_recheck`` certifies it;
        otherwise the packed row, already replay-certified, is restored.
        """
        changed = valid & (retained != packed).any(dim=1)
        rows = torch.where(changed)[0]
        if rows.numel() == 0:
            return retained
        passing = self._recheck(context, retained.index_select(0, rows), rows)
        failed = rows[~passing.to(rows.device)]
        if failed.numel() == 0:
            return retained
        self.metrics.coarsen_replay_fallbacks += int(failed.numel())
        restored = retained.clone()
        restored[failed] = packed[failed]
        return restored

    def _recheck(
        self,
        context: _ReplayContext,
        candidate: torch.Tensor,
        rows: torch.Tensor,
    ) -> torch.Tensor:
        candidate_batch = context.batch.select(rows)
        lin_refined = self._refresh_mode == "split_refresh_linear" and self._lin_support_refinement
        if lin_refined and candidate_batch.incremental_alpha is not None:
            # Implied phases used effective lower slopes 0/1, not the dormant
            # cached alpha. Preserve those globally valid lower lines without
            # adding an implied phase constraint to the candidate region.
            source_bounds = slice_bounds_dict(context.bounds, rows)
            candidate_batch.incremental_alpha = dict(candidate_batch.incremental_alpha)
            for lid in self.codec.layer_ids:
                alpha = candidate_batch.incremental_alpha.get(lid)
                bounds = source_bounds.get(lid)
                if not isinstance(alpha, torch.Tensor) or bounds is None:
                    continue
                lower, upper = bounds.lb.flatten(1).unsqueeze(1), bounds.ub.flatten(1).unsqueeze(1)
                candidate_batch.incremental_alpha[lid] = torch.where(
                    lower >= 0, torch.ones_like(alpha),
                    torch.where(upper <= 0, torch.zeros_like(alpha), alpha),
                )
        phase_slope_signs = {
            lid: signs.detach().clone()
            for lid, signs in (candidate_batch.split_signs or {}).items()
        }
        # The solver indexes split_signs[lid] for every incremental_eta layer, so a
        # layer whose literals were all coarsened away stays as an all-zero block.
        if candidate_batch.split_signs is not None:
            candidate_batch.split_signs = {
                lid: torch.zeros_like(signs)
                for lid, signs in candidate_batch.split_signs.items()
            }
        candidate_values = candidate[candidate != 0]
        active_variables = torch.unique(candidate_values.abs() - 1, sorted=True)
        dense = torch.zeros(
            (candidate.shape[0], active_variables.numel()),
            dtype=torch.int8,
            device=candidate.device,
        )
        lanes, slots = torch.nonzero(candidate, as_tuple=True)
        columns = torch.searchsorted(active_variables, candidate[lanes, slots].abs() - 1)
        dense[lanes, columns] = candidate[lanes, slots].sign().to(torch.int8)
        apply_literal_assignments(candidate_batch, dense, active_variables, self.codec)
        eta_layers = set(candidate_batch.incremental_eta or {})
        sign_layers = set(candidate_batch.split_signs or {})
        missing_layers = sorted(eta_layers - sign_layers)
        assert not missing_layers, (
            "CLIMB invariant violated: recheck split_signs missing incremental_eta layer "
            f"(KeyError: {missing_layers[0]}, eta_layers={len(eta_layers)}, "
            f"split_sign_layers={len(sign_layers)})"
        )
        if context.base_bounds is None:
            row_bounds = slice_bounds_dict(context.bounds, rows)
        else:
            core_bounds = self._core_local_bounds(
                slice_bounds_dict(context.base_bounds, rows), candidate_batch.split_signs
            )
            if core_bounds is None:
                return torch.zeros(int(rows.numel()), dtype=torch.bool, device=rows.device)
            row_bounds = core_bounds
            if lin_refined:
                from act.back_end.bab.linear_refresh import linear_refresh_bounds

                warm_bounds = linear_refresh_bounds(
                    self._net, slice_bounds_dict(context.base_bounds, rows),
                    candidate_batch.split_signs or {}, warm_signs=phase_slope_signs,
                )
                if warm_bounds is not None:
                    row_bounds = warm_bounds
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
            phase_slope_signs=phase_slope_signs,
        )
        if replayed is None:
            return torch.zeros(
                int(rows.numel()), dtype=torch.bool, device=rows.device
            )
        replayed_slack, _ = replayed
        return is_certified(
            replayed_slack, self._out_kind, replay_reference(row_thresholds)
        )

    def _core_local_bounds(
        self,
        base: Dict[int, Bounds],
        split_signs: Optional[Dict[int, torch.Tensor]],
    ) -> Optional[Dict[int, Bounds]]:
        """Split-free ``base`` refreshed from ``split_signs`` alone (C3), same refresh as the lane solve."""
        from act.back_end.bab.linear_refresh import refresh_split_bounds
        from act.back_end.solver.solver_dual import DualSolver

        return refresh_split_bounds(
            DualSolver(), self._net, base, split_signs or {}, self._refresh_mode
        )

    def _instrumented_recheck(
        self,
        context: _ReplayContext,
        candidate: torch.Tensor,
        rows: torch.Tensor,
    ) -> torch.Tensor:
        recheck_started = time.perf_counter()
        passing = self._recheck(context, candidate, rows)
        self.metrics.recheck_time_s += time.perf_counter() - recheck_started
        self.metrics.recheck_calls += 1
        self.metrics.recheck_row_passes += int(rows.numel())
        return passing

    def before_split(
        self, pool: TopKBounding, unresolved: SubproblemBatch
    ) -> Tuple[SubproblemBatch, torch.Tensor]:
        """Pre-split Propagate, then Merge on unseen cores (Alg. 1, lines 12-15).

        ``unresolved`` rows were just bounded: their literal sets enter the
        merge-undo memo. Returns the rows to split and their positions.
        """
        if self._merge_enabled and unresolved.batch_size:
            keys = literal_set_keys(unresolved)
            self._bounded_unresolved.update(keys)
            self._fresh_keys.difference_update(keys)
        rows, positions = self._before_split(pool, unresolved)
        if self._complement_enabled and rows.batch_size:
            packed = pack_split_matrix(rows.split_signs, rows.batch_size, self.codec, device=rows.lb.device)
            self._split_parent_hashes.update(
                _row_hash([lit for lit in row if lit]) for row in packed.tolist()
            )
        return rows, positions

    def _before_split(
        self, pool: TopKBounding, unresolved: SubproblemBatch
    ) -> Tuple[SubproblemBatch, torch.Tensor]:
        positions = torch.arange(unresolved.batch_size, device=unresolved.lb.device)
        if not self._propagation_enabled:
            self.metrics.observe_frontier(len(pool) + unresolved.batch_size)
            return unresolved, positions
        if self.library.core_count == 0 or not self._presplit_enabled:
            return unresolved, positions
        self.metrics.observe_frontier(len(pool) + unresolved.batch_size)
        propagation_groups: List[SubproblemBatch] = []
        pending = pool.propagation_batch(self.library.revision)
        if pending is not None:
            propagation_groups.append(pending[1])
        active_changed = self._active_library_revision != self.library.revision
        if active_changed:
            propagation_groups.append(unresolved)
        if not propagation_groups:
            self.metrics.incremental_pool_rounds_skipped += 1
            return self._merge_active(pool, unresolved, positions)
        presplit_before = unresolved.batch_size
        propagate_started = time.perf_counter()
        propagation, kept = _propagate_with_indices(
            propagation_groups,
            self.library,
            self.codec,
            core_chunk=self._propagate_core_chunk,
        )
        self.metrics.propagate_time_s += time.perf_counter() - propagate_started
        result_index = 0
        if pending is not None:
            selected, pending_before = pending
            pending_after = propagation[result_index]
            self.metrics.presplit_unit_writes += self._note_writes(
                pending_before, kept[result_index], pending_after
            )
            pool.apply_propagation(
                selected,
                kept[result_index],
                pending_after,
                self.library.revision,
            )
            self.metrics.prebound_discharged += (
                pending_before.batch_size - pending_after.batch_size
            )
            result_index += 1
        if active_changed:
            self.metrics.presplit_unit_writes += self._note_writes(
                unresolved, kept[result_index], propagation[result_index]
            )
            unresolved = propagation[result_index]
            positions = kept[result_index].to(positions.device)
            self._active_library_revision = self.library.revision
        self.metrics.presplit_discharged += presplit_before - unresolved.batch_size
        self.metrics.observe_frontier(len(pool) + unresolved.batch_size)
        return self._merge_active(pool, unresolved, positions)

    def _merge_active(
        self, pool: TopKBounding, unresolved: SubproblemBatch, positions: torch.Tensor
    ) -> Tuple[SubproblemBatch, torch.Tensor]:
        if not (self._merge_enabled and self._merge_armed):
            return unresolved, positions
        untouched = self._merge(pool, unresolved)
        if untouched is None:
            return unresolved, positions
        untouched = untouched.to(positions.device)
        return unresolved.select(untouched), positions.index_select(0, untouched)

    def _merge(
        self, pool: TopKBounding, active: Optional[SubproblemBatch]
    ) -> Optional[torch.Tensor]:
        """Merge pending pool rows and active rows; merged rows return to the pool.

        Merged rows are pending, so they are bounded again before any split.
        Returns positions of the untouched active rows, or None if nothing merged.
        """
        self._merge_armed = False
        n_pool = len(pool)
        # Merge copies the whole frontier; the pool is untouched until
        # replace_all, so a CUDA OOM only skips this wave: the handler halves
        # the lane cap and Merge is re-armed for the next wave.
        try:
            parts = ([pool.view_all()] if n_pool else []) + (
                [active] if active is not None and active.batch_size else []
            )
            if sum(part.batch_size for part in parts) < 2:
                return None
            rows = parts[0] if len(parts) == 1 else parts[0].concat(parts[1])
            node_ids = rows.node_id
            rows.node_id = torch.arange(rows.batch_size, device=rows.lb.device)
            started = time.perf_counter()
            fresh = None
            if self._fresh_keys:
                fresh = torch.tensor([key in self._fresh_keys for key in literal_set_keys(rows)])
            result = merge_frontier(
                rows, pending=rows.node_id < n_pool,
                bounded_unresolved=self._bounded_unresolved, fresh=fresh,
            )
            oom = False
        except torch.cuda.OutOfMemoryError:
            oom = True
        if oom:
            parts = rows = None
            self._merge_armed = True
            self._report_oom()
            return None
        self.metrics.merge_time_s += time.perf_counter() - started
        self.metrics.merge_skipped_pairs += result.skipped_pairs
        self.metrics.merge_calls += 1
        self.metrics.merged_pairs += result.merged_pairs
        self.metrics.merge_rows_before += rows.batch_size
        self.metrics.merge_rows_after += result.rows.batch_size
        self.metrics.merge_literals_before += _row_literal_count(rows)
        self.metrics.merge_literals_after += _row_literal_count(result.rows)
        if result.merged_pairs == 0:
            return None
        origin = result.rows.node_id
        assert origin is not None
        pending_mask = result.pending.to(origin.device)
        requeued = result.pending_rows()
        requeued.node_id = (
            None
            if node_ids is None
            else node_ids.index_select(0, origin[pending_mask].to(node_ids.device))
        )
        pool.replace_all(requeued)
        return origin[~pending_mask] - n_pool

    def steer_scores(self, scores: Any) -> Any:
        """C5: scale BaBSR scores by ``1 + share`` of learned cores naming the neuron."""
        if not self._steer or self.library.core_count == 0 or scores.per_layer is None:
            return scores
        variables = torch.tensor(
            [abs(lit) - 1 for core in self.library.cores for lit in core], dtype=torch.long
        )
        counts = torch.bincount(variables, minlength=sum(self.codec.widths)).to(torch.float64)
        share = counts / counts.max()

        def steer(per_layer: Optional[Dict[int, torch.Tensor]]) -> Optional[Dict[int, torch.Tensor]]:
            if per_layer is None:
                return None
            steered = dict(per_layer)
            for layer_id, layer_scores in per_layer.items():
                if layer_id not in self.codec.layer_ids:
                    continue
                index = self.codec.layer_ids.index(layer_id)
                offset = self.codec.offsets[index]
                bonus = share[offset : offset + self.codec.widths[index]].to(layer_scores)
                flat = layer_scores.reshape(layer_scores.shape[0], -1)
                steered[layer_id] = (flat * (1.0 + bonus)).reshape(layer_scores.shape)
            return steered

        return dataclasses.replace(
            scores,
            per_layer=steer(scores.per_layer),
            babsr_per_layer=steer(scores.babsr_per_layer),
        )

    def reject_input_axis_split(self) -> None:
        raise ValueError(
            "CLIMB requires ReLU phase literals only: the "
            "brancher fell back to an input-axis split, which "
            "would break the shared-box premise of core reuse"
        )

    def record_terminal_failures(self, failed: int) -> None:
        """Count candidate-less lanes whose terminal LP remained UNKNOWN."""
        assert failed >= 0
        self.metrics.retired_unknown_lanes += failed

    def after_children(self, generated: int, frontier_rows: int) -> None:
        self.metrics.generated_children += generated
        self.metrics.observe_frontier(frontier_rows)

    def metadata(self) -> Dict[str, Any]:
        return self.metrics.metadata()
