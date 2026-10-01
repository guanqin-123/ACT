from __future__ import annotations

from array import array
from dataclasses import dataclass
from itertools import product
from typing import Collection, Dict, Iterable, List, Optional, Tuple

import torch

from act.back_end.bab import climb_tensor
from act.back_end.bab.node import SubproblemBatch


Literal = Tuple[int, int, int]
Pivot = Tuple[int, int]
ProjectionKey = Tuple[Tuple[Literal, ...], Pivot]
Pair = Tuple[int, int, Pivot]


@dataclass(frozen=True)
class FrontierMergeResult:
    """Merged frontier rows and their pending/active status."""

    rows: SubproblemBatch
    pending: torch.Tensor
    merged_pairs: int
    skipped_pairs: int = 0

    def pending_rows(self) -> SubproblemBatch:
        """Rows that must enter the pool and be bounded before splitting."""
        return self.rows.select(torch.where(self.pending)[0])

    def active_rows(self) -> SubproblemBatch:
        """Untouched active rows that retain their pre-merge status."""
        return self.rows.select(torch.where(~self.pending)[0])


def _literal_rows(rows: SubproblemBatch) -> List[Tuple[Literal, ...]]:
    if not rows.split_signs:
        return [tuple() for _ in range(rows.batch_size)]

    literals: List[List[Literal]] = [[] for _ in range(rows.batch_size)]
    for layer_id in sorted(rows.split_signs):
        values = rows.split_signs[layer_id]
        if values.shape[0] != rows.batch_size:
            raise ValueError("split_signs leading dimension mismatch")
        flattened = values.reshape(rows.batch_size, values.shape[1], -1)
        first = flattened[:, :1, :]
        if not torch.equal(flattened, first.expand_as(flattened)):
            raise ValueError("frontier merge requires spec-invariant split signs")
        lane_values = first[:, 0, :]
        if not bool(((lane_values == -1) | (lane_values == 0) | (lane_values == 1)).all()):
            raise ValueError("split signs must be -1, 0, or 1")
        lanes, neurons = torch.nonzero(lane_values, as_tuple=True)
        signs = lane_values[lanes, neurons].to(torch.int8)
        for lane, neuron, sign in zip(lanes.tolist(), neurons.tolist(), signs.tolist()):
            literals[lane].append((layer_id, neuron, sign))
    return [tuple(row) for row in literals]


def literal_set_key(literals: Iterable[Literal]) -> bytes:
    """Exact canonical key of a literal set: sorted packed ``(layer, neuron, sign)`` int64s."""
    packed = sorted((layer << 32) | (neuron << 1) | (sign > 0) for layer, neuron, sign in literals)
    return array("q", packed).tobytes()


def literal_set_keys(rows: SubproblemBatch) -> List[bytes]:
    if climb_tensor.ENABLED:
        return climb_tensor.literal_set_keys(rows.split_signs, rows.batch_size)
    return [literal_set_key(literals) for literals in _literal_rows(rows)]


def _discover_pairs(
    rows: SubproblemBatch,
    bounded_unresolved: Optional[Collection[bytes]] = None,
    fresh: Optional[List[bool]] = None,
) -> Tuple[List[Pair], int]:
    """Row-disjoint complementary pairs, minus undo pairs.

    An undo pair rebuilds a literal set in ``bounded_unresolved`` (already
    bounded and left unresolved) while neither side is ``fresh`` (carries a
    Propagate-written literal since it was last bounded); it is skipped.
    """
    if climb_tensor.ENABLED:
        return climb_tensor.discover_pairs(rows.split_signs, rows.batch_size, bounded_unresolved, fresh)
    buckets: Dict[ProjectionKey, Dict[int, List[int]]] = {}
    for row_index, literals in enumerate(_literal_rows(rows)):
        for position, (layer_id, neuron, sign) in enumerate(literals):
            projection = literals[:position] + literals[position + 1 :]
            key = (projection, (layer_id, neuron))
            by_sign = buckets.setdefault(key, {-1: [], 1: []})
            by_sign[sign].append(row_index)

    candidates = set()
    undo: set[Tuple[int, int, Pivot]] = set()
    for (projection, pivot), by_sign in buckets.items():
        pairs = [
            (min(left, right), max(left, right), pivot)
            for left, right in product(by_sign[-1], by_sign[1])
        ]
        if pairs and bounded_unresolved and literal_set_key(projection) in bounded_unresolved:
            undo.update(
                pair for pair in pairs
                if not (fresh is not None and (fresh[pair[0]] or fresh[pair[1]]))
            )
        candidates.update(pairs)
    disjoint: List[Pair] = []
    used: set[int] = set()
    skipped = 0
    for left, right, pivot in sorted(candidates):
        if left in used or right in used:
            continue
        if (left, right, pivot) in undo:
            skipped += 1
            continue
        disjoint.append((left, right, pivot))
        used.add(left)
        used.add(right)
    return disjoint, skipped


def _merge_pass(
    rows: SubproblemBatch,
    pending: torch.Tensor,
    pairs: List[Pair],
    fresh: Optional[List[bool]] = None,
) -> tuple[SubproblemBatch, torch.Tensor, Optional[List[bool]]]:
    paired = {index for left, right, _ in pairs for index in (left, right)}
    untouched = [index for index in range(rows.batch_size) if index not in paired]
    source = untouched + [left for left, _, _ in pairs]
    source_indices = torch.tensor(source, dtype=torch.long, device=rows.lb.device)
    merged = rows.select(source_indices)
    next_pending = pending.index_select(0, source_indices.to(pending.device)).clone()
    next_fresh = (
        None if fresh is None
        else [fresh[index] for index in untouched] + [fresh[l] or fresh[r] for l, r, _ in pairs]
    )

    first_merged = len(untouched)
    for pair_index, (left, right, (layer_id, neuron)) in enumerate(pairs):
        output = first_merged + pair_index
        merged.lb[output] = torch.minimum(rows.lb[left], rows.lb[right])
        merged.ub[output] = torch.maximum(rows.ub[left], rows.ub[right])
        merged.depths[output] = torch.clamp(
            torch.minimum(rows.depths[left], rows.depths[right]) - 1,
            min=0,
        )
        if merged.lower_bound is not None:
            assert rows.lower_bound is not None
            merged.lower_bound[output] = torch.minimum(
                rows.lower_bound[left], rows.lower_bound[right]
            )
        if merged.parent_margins is not None:
            assert rows.parent_margins is not None
            merged.parent_margins[output] = torch.minimum(
                rows.parent_margins[left], rows.parent_margins[right]
            )
        assert merged.split_signs is not None
        merged.split_signs[layer_id][output, :, neuron] = 0
        if merged.incremental_eta is not None:
            assert rows.incremental_eta is not None
            for eta_layer, eta in merged.incremental_eta.items():
                eta[output] = torch.minimum(
                    rows.incremental_eta[eta_layer][left],
                    rows.incremental_eta[eta_layer][right],
                )
                signs = merged.split_signs.get(eta_layer)
                if signs is not None and eta[output].shape == signs[output].shape:
                    eta[output].masked_fill_((signs[output] == 0).to(eta.device), 0)
        if merged.parent_id is not None:
            assert rows.parent_id is not None
            same_parent = rows.parent_id[left] == rows.parent_id[right]
            merged.parent_id[output] = torch.where(
                same_parent,
                rows.parent_id[left],
                torch.full_like(rows.parent_id[left], -1),
            )
        next_pending[output] = True
    return merged, next_pending, next_fresh


@torch.no_grad()
def merge_frontier(
    rows: SubproblemBatch,
    *,
    pending: Optional[torch.Tensor] = None,
    bounded_unresolved: Optional[Collection[bytes]] = None,
    fresh: Optional[torch.Tensor] = None,
) -> FrontierMergeResult:
    """Merge complementary frontier rows to a fixed point.

    Pair discovery hashes every row projection together with its omitted pivot,
    then greedily selects row-disjoint pairs in lexicographic order. A pair is
    replaced by its common split signs, componentwise box hull, and minimum
    priority lower bound. Newly merged rows are marked pending so callers must
    bound them again before any split; untouched rows keep their prior status.
    Pairs whose merge would rebuild a key of ``bounded_unresolved`` are skipped
    unless a side is ``fresh`` (a merged row is fresh if either side was).
    """
    if pending is None:
        pending = torch.ones(rows.batch_size, dtype=torch.bool, device=rows.lb.device)
    elif pending.shape != (rows.batch_size,):
        raise ValueError("pending mask shape must match the frontier row count")
    elif pending.dtype != torch.bool:
        raise ValueError("pending mask must have boolean dtype")
    else:
        pending = pending.to(rows.lb.device)

    current = rows
    current_pending = pending
    current_fresh = None if fresh is None else [bool(value) for value in fresh.tolist()]
    merged_pairs = 0
    skipped_pairs = 0
    while True:
        pairs, skipped = _discover_pairs(current, bounded_unresolved, current_fresh)
        skipped_pairs += skipped
        if not pairs:
            break
        current, current_pending, current_fresh = _merge_pass(
            current, current_pending, pairs, current_fresh
        )
        merged_pairs += len(pairs)
    return FrontierMergeResult(current, current_pending, merged_pairs, skipped_pairs)
