"""Device tensor kernels for CLIMB hooks: exact drop-ins for the Python references.

``ENABLED`` selects these kernels at the call sites; tests flip it to run the
Python reference paths (``frontier_merge._discover_pairs_reference`` and
``CoreLibrary`` packing) and compare outputs, order included.
"""

from __future__ import annotations

from typing import Collection, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

ENABLED = True

Pivot = Tuple[int, int]
Pair = Tuple[int, int, Pivot]

_HASH_BITS = 48
_HASH_TABLES: Dict[Tuple[int, str], torch.Tensor] = {}


def _hash_table(n_variables: int, device: torch.device) -> torch.Tensor:
    """Fixed random ``[2 * V]`` values below 2**48, so row sums of <= 2**15 literals never overflow int64."""
    key = (n_variables, str(device))
    table = _HASH_TABLES.get(key)
    if table is None:
        generator = torch.Generator().manual_seed(0x5EED)
        table = torch.randint(
            0, 1 << _HASH_BITS, (2 * n_variables,), generator=generator, dtype=torch.long, device="cpu"
        ).to(device)
        _HASH_TABLES[key] = table
    return table


def literal_entries(split_signs: Optional[Dict[int, torch.Tensor]], n_rows: int) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int
]:
    """Every split literal as ``(row, layer, neuron, sign, variable)``, ordered by ``(row, layer, neuron)``.

    Same order and the same ``ValueError`` checks as ``frontier_merge._literal_rows``;
    ``variable`` is a dense id over the sorted layers and the last item is their total width.
    """
    rows_parts: List[torch.Tensor] = []
    layer_parts: List[torch.Tensor] = []
    neuron_parts: List[torch.Tensor] = []
    sign_parts: List[torch.Tensor] = []
    variable_parts: List[torch.Tensor] = []
    offset = 0
    for layer_id, values in sorted((split_signs or {}).items()):
        if values.shape[0] != n_rows:
            raise ValueError("split_signs leading dimension mismatch")
        flattened = values.reshape(n_rows, values.shape[1], -1)
        first = flattened[:, :1, :]
        if not torch.equal(flattened, first.expand_as(flattened)):
            raise ValueError("frontier merge requires spec-invariant split signs")
        lane_values = first[:, 0, :]
        if not bool(((lane_values == -1) | (lane_values == 0) | (lane_values == 1)).all()):
            raise ValueError("split signs must be -1, 0, or 1")
        lanes, neurons = torch.nonzero(lane_values, as_tuple=True)
        rows_parts.append(lanes)
        layer_parts.append(torch.full_like(lanes, layer_id))
        neuron_parts.append(neurons)
        sign_parts.append(lane_values[lanes, neurons].to(torch.long))
        variable_parts.append(neurons + offset)
        offset += flattened.shape[-1]
    if not rows_parts:
        empty = torch.zeros(0, dtype=torch.long)
        return empty, empty, empty, empty, empty, 0
    rows = torch.cat(rows_parts)
    order = torch.sort(rows, stable=True).indices
    return (
        rows[order],
        torch.cat(layer_parts)[order],
        torch.cat(neuron_parts)[order],
        torch.cat(sign_parts)[order],
        torch.cat(variable_parts)[order],
        offset,
    )


@torch.no_grad()
def discover_pairs(
    split_signs: Optional[Dict[int, torch.Tensor]],
    n_rows: int,
    bounded_unresolved: Optional[Collection[bytes]] = None,
    fresh: Optional[List[bool]] = None,
) -> Tuple[List[Pair], int]:
    """Tensor ``frontier_merge._discover_pairs``: identical pairs, order and skip count.

    Rows pair on pivot ``v`` iff their literal sets agree except for opposite
    signs at ``v``. Entries are grouped on a hash of (row hash minus own
    literal hash, variable); every candidate is then verified exactly, so hash
    collisions can only cost time, never change the result.
    """
    rows, layers, neurons, signs, variables, n_variables = literal_entries(split_signs, n_rows)
    if rows.numel() == 0:
        return [], 0
    device = rows.device
    table = _hash_table(n_variables, device)
    literal_hash = table[2 * variables + (signs > 0).to(torch.long)]
    row_hash = torch.zeros(n_rows, dtype=torch.long, device=device).index_add_(0, rows, literal_hash)
    key = (row_hash[rows] - literal_hash) ^ (variables * 0x9E3779B1)
    by_sign = torch.sort((signs > 0).to(torch.long), stable=True).indices
    order = by_sign[torch.sort(key[by_sign], stable=True).indices]
    key = key[order]
    _, group, counts = torch.unique_consecutive(key, return_inverse=True, return_counts=True)
    starts = counts.cumsum(0) - counts
    n_negative = torch.zeros_like(counts).index_add_(0, group, (signs[order] < 0).to(torch.long))
    n_positive = counts - n_negative
    n_pairs = n_negative * n_positive
    if not bool((n_pairs > 0).any()):
        return [], 0
    pair_group = torch.repeat_interleave(torch.arange(counts.numel(), device=device), n_pairs)
    within = torch.arange(pair_group.numel(), device=device) - torch.repeat_interleave(
        n_pairs.cumsum(0) - n_pairs, n_pairs
    )
    negative_entry = order[starts[pair_group] + within // n_positive[pair_group]]
    positive_entry = order[starts[pair_group] + n_negative[pair_group] + within % n_positive[pair_group]]

    left, right = rows[negative_entry], rows[positive_entry]
    involved, local = torch.unique(torch.cat((left, right)), return_inverse=True)
    position = torch.full((n_rows,), -1, dtype=torch.long, device=device)
    position[involved] = torch.arange(involved.numel(), device=device)
    dense = torch.zeros((involved.numel(), n_variables), dtype=torch.int8, device=device)
    member = position[rows] >= 0
    dense[position[rows[member]], variables[member]] = signs[member].to(torch.int8)
    left_dense, right_dense = dense[local[: left.numel()]], dense[local[left.numel() :]]
    pivot = variables[negative_entry]
    exact = ((left_dense != right_dense).sum(dim=1) == 1) & (
        right_dense.gather(1, pivot[:, None])[:, 0] > 0
    ) & (variables[positive_entry] == pivot)
    low, high = torch.minimum(left, right)[exact], torch.maximum(left, right)[exact]
    entry = negative_entry[exact]
    ranked = torch.sort((low * n_rows + high) * max(n_variables, 1) + variables[entry]).indices
    low, high, entry = low[ranked].tolist(), high[ranked].tolist(), entry[ranked]
    pivot_layers, pivot_neurons = layers[entry].tolist(), neurons[entry].tolist()
    entry_list, entry_rows = entry.tolist(), rows[entry].tolist()

    packed = None
    row_starts: List[int] = []
    disjoint: List[Pair] = []
    used: set[int] = set()
    skipped = 0
    for index, (left_row, right_row) in enumerate(zip(low, high)):
        if left_row in used or right_row in used:
            continue
        if bounded_unresolved and not (fresh is not None and (fresh[left_row] or fresh[right_row])):
            if packed is None:
                packed = ((layers << 32) | (neurons << 1) | (signs > 0).to(torch.long)).cpu().numpy()
                row_counts = torch.bincount(rows, minlength=n_rows)
                row_starts = (row_counts.cumsum(0) - row_counts).tolist() + [int(rows.numel())]
            entry_index = entry_list[index]
            row = entry_rows[index]
            segment = packed[row_starts[row] : row_starts[row + 1]]
            cut = entry_index - row_starts[row]
            if segment[:cut].tobytes() + segment[cut + 1 :].tobytes() in bounded_unresolved:
                skipped += 1
                continue
        disjoint.append((left_row, right_row, (pivot_layers[index], pivot_neurons[index])))
        used.add(left_row)
        used.add(right_row)
    return disjoint, skipped


def pack_cores(cores: Sequence[frozenset[int]], device: torch.device) -> torch.Tensor:
    """Tensor ``CoreLibrary.pack``: rows zero-padded, literals ordered by ``(|l|, l)``."""
    width = max((len(core) for core in cores), default=0)
    if width == 0:
        return torch.zeros((len(cores), width), dtype=torch.long, device=device)
    packed = np.zeros((len(cores), width), dtype=np.int64)
    for row, core in enumerate(cores):
        packed[row, :len(core)] = sorted(core, key=lambda literal: (abs(literal), literal))
    return torch.from_numpy(packed).to(device)



def literal_set_keys(split_signs: Optional[Dict[int, torch.Tensor]], n_rows: int) -> List[bytes]:
    """Tensor ``frontier_merge.literal_set_keys``: row order is already the sorted packed order."""
    rows, layers, neurons, signs, _, _ = literal_entries(split_signs, n_rows)
    if rows.numel() == 0:
        return [b""] * n_rows
    packed = ((layers << 32) | (neurons << 1) | (signs > 0).to(torch.long)).cpu().numpy()
    keys: List[bytes] = []
    start = 0
    for end in torch.bincount(rows, minlength=n_rows).cumsum(0).tolist():
        keys.append(packed[start:end].tobytes())
        start = end
    return keys


@torch.no_grad()
def propagate_dense(
    assignments: torch.Tensor,
    core_dense: torch.Tensor,
    *,
    core_chunk: int,
    compute_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Synchronous unit closure with one convergence transfer per round.

    Every chunk sees the same assignment snapshot. A unit core has exactly
    one missing variable; masking occupied variables after the proposal
    matmuls therefore gives the same proposals as gathering unit pairs.
    Discharged/conflicting lanes keep their previous assignments.
    """
    active = torch.ones(assignments.shape[0], dtype=torch.bool, device=assignments.device)
    chunks = []
    for start in range(0, core_dense.shape[0], core_chunk):
        chunk = core_dense[start:start + core_chunk]
        absolute = chunk.abs().to(compute_dtype)
        chunks.append((chunk.to(compute_dtype).T, absolute.T,
                       absolute.sum(dim=1), (chunk < 0).to(compute_dtype),
                       (chunk > 0).to(compute_dtype)))
    while True:
        work = assignments.to(compute_dtype)
        work_abs = assignments.abs().to(compute_dtype)
        discharge = torch.zeros_like(active)
        positive = torch.zeros_like(assignments, dtype=torch.bool)
        negative = torch.zeros_like(positive)
        for signed_t, absolute_t, lengths, negative_core, positive_core in chunks:
            product = work @ signed_t
            overlap = work_abs @ absolute_t
            discharge |= (product == lengths.unsqueeze(0)).any(dim=1)
            unit = ((product == overlap) & (overlap == lengths.unsqueeze(0) - 1)).to(compute_dtype)
            positive |= (unit @ negative_core) > 0
            negative |= (unit @ positive_core) > 0
        unassigned = assignments == 0
        positive &= unassigned
        negative &= unassigned
        conflict = (positive & negative).any(dim=1)
        active &= ~(discharge | conflict)
        write = (positive | negative) & active[:, None]
        proposals = positive.to(torch.int8) - negative.to(torch.int8)
        assignments = torch.where(write, proposals, assignments)
        if not bool(write.any().item()):
            break
    return assignments, active
