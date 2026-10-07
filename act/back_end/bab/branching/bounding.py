# ===- act/back_end/bab/branching/bounding.py - Subproblem Bounding ------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------====#
#
# Purpose:
#   Subproblem pool management for Branch-and-Bound.
#
#   Strategies selectable via ``--bab-bounding`` / ``BaBConfig.bounding``:
#
#   +---------------------+---------------------+--------------------------------+-------------+
#   | value               | pool class          | order function                 | --bab-top-k |
#   +=====================+=====================+================================+=============+
#   | depth_bound_blend   | TopKBounding        | DepthLowerBoundOrder:          | honoured    |
#   |                     |                     | 0.5*norm(depth) + 0.5*urgency  |             |
#   +---------------------+---------------------+--------------------------------+-------------+
#   | greedy              | TopKBounding        | GreedyOrder: best-first on     | honoured    |
#   |                     |                     | |lb| (Oliva-Greedy)            |             |
#   +---------------------+---------------------+--------------------------------+-------------+
#   | annealed            | TopKBounding        | SAOrder: Gumbel noise with     | honoured    |
#   |                     |                     | temp = cooling_rate**step      |             |
#   |                     |                     | (Oliva-SA)                     |             |
#   +---------------------+---------------------+--------------------------------+-------------+
#   | diverse_split_signs | DiverseTopKBounding | any order, then hash/soft      | honoured    |
#   |                     |                     | repulsion over split-sign      |             |
#   |                     |                     | signatures                     |             |
#   +---------------------+---------------------+--------------------------------+-------------+
#   | random              | RandomBounding      | none — uniform sampling        | ValueError  |
#   +---------------------+---------------------+--------------------------------+-------------+
#   | mcts                | MCTSBounding        | DepthLowerBoundOrder, pinned;  | ValueError  |
#   |                     |                     | observer-only, pop is plain    |             |
#   |                     |                     | top-k until UCB1 lands         |             |
#   +---------------------+---------------------+--------------------------------+-------------+
#
#   The first four share the ``TopKBounding`` pool and so honour ``top_k``; the
#   last two do not rank by an order function, so ``top_k`` is rejected rather
#   than silently ignored. ``top_k`` caps a single ``pop`` without discarding
#   anything — unlike ``evict_to``, which drops worst-priority leaves to honour a
#   frontier cap and therefore forces a sound ``UNKNOWN``.
#
#   ``GreedyOrder`` / ``SAOrder`` implement the Oliva order-leading exploration of the
#   BaB tree — "Efficient Neural Network Verification via Order Leading Exploration of
#   Branch-and-Bound Trees", Guanqin Zhang, Kota Fukuda, Zhenya Zhang, H.M.N. Dilum
#   Bandara, Shiping Chen, Jianjun Zhao, Yulei Sui, ECOOP 2025 (arXiv:2507.17453).
#   ``MCTSBounding`` is specified in ``docs/design/mcts_bab.md``.
#
#   A bounding strategy maintains a *pool* of pending subproblems and
#   decides which ones to process next.  All data flows through
#   ``SubproblemBatch`` (tensor-native) so that:
#
#     * ``push`` and ``pop`` operate on batches, not individual nodes.
#     * Internal storage can be a single tensor block (GPU-friendly).
#     * Future batch-parallel BaB pops N subproblems at once for
#       vectorised solving.
#
# ===---------------------------------------------------------------------====#

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, Callable, Dict, List, Literal, Optional, Protocol, Sequence, Set, Tuple, cast

import torch

from act.back_end.bab.node import SubproblemBatch
from act.back_end.solver.solver_base import SolveStatus
from act.back_end.solver.solver_dual import (
    AlphaState,
    _alpha_concat_views,
    _alpha_tree_concat_lanes,
    _alpha_tree_gather_lanes,
    _alpha_tree_leaves,
    _alpha_tree_map,
)
from act.util.device_manager import get_default_dtype


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BoundingStrategy(ABC):
    """Abstract subproblem pool for Branch-and-Bound.

    Lifecycle (called by the BaB engine)::

        pool.push(root_batch)
        while not pool.empty:
            batch = pool.pop(batch_size=N)
            …solve / branch…
            pool.push(children_batch)

    Subclass contract
    ~~~~~~~~~~~~~~~~~
    * ``push`` accepts any-sized ``SubproblemBatch``.
    * ``pop(k)`` returns *at most* ``k`` subproblems; fewer if the
      pool is smaller.  Raises ``IndexError`` on empty pool.
    * ``__len__`` returns the current pool size.
    """

    @abstractmethod
    def push(self, batch: SubproblemBatch) -> None:
        """Enqueue a batch of subproblems.

        Args:
            batch: ``(N, D)`` subproblems to add to the pool.
        """
        ...

    @abstractmethod
    def pop(self, batch_size: int = 1) -> SubproblemBatch:
        """Dequeue subproblems for the next bounding iteration.

        Args:
            batch_size: Maximum number of subproblems to return.

        Returns:
            ``(M, D)`` batch where ``M <= batch_size``.

        Raises:
            IndexError: If the pool is empty.
        """
        ...

    @abstractmethod
    def evict_to(self, cap: int) -> int:
        """Drop pending subproblems until at most cap remain."""
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Number of pending subproblems."""
        ...

    @property
    def empty(self) -> bool:
        """True when no subproblems remain."""
        return len(self) == 0


# ---------------------------------------------------------------------------
# Random baseline
# ---------------------------------------------------------------------------


class RandomBounding(BoundingStrategy):
    """Uniform-random subproblem selection.

    ``pop(k)`` selects ``k`` subproblems uniformly at random from the
    pool (without replacement).

    Internal storage is fully tensor-native: three tensors ``(M, D)``,
    ``(M, D)``, ``(M,)`` for lower bounds, upper bounds, and depths
    respectively.
    """

    def __init__(self) -> None:
        self._lb: Optional[torch.Tensor] = None  # (M, D)
        self._ub: Optional[torch.Tensor] = None  # (M, D)
        self._depths: Optional[torch.Tensor] = None  # (M,)

    # -- BoundingStrategy interface -----------------------------------------

    def push(self, batch: SubproblemBatch) -> None:
        if self._lb is None:
            self._lb = batch.lb.clone()
            self._ub = batch.ub.clone()
            self._depths = batch.depths.clone()
        else:
            assert self._ub is not None and self._depths is not None
            self._lb = torch.cat([self._lb, batch.lb], dim=0)
            self._ub = torch.cat([self._ub, batch.ub], dim=0)
            self._depths = torch.cat([self._depths, batch.depths], dim=0)

    def pop(self, batch_size: int = 1) -> SubproblemBatch:
        if self.empty:
            raise IndexError("pop from empty pool")

        n = min(batch_size, len(self))
        assert self._lb is not None and self._ub is not None and self._depths is not None
        perm = torch.randperm(len(self), device=self._lb.device)
        selected = perm[:n]
        remaining = perm[n:]

        result = SubproblemBatch(
            lb=self._lb[selected],
            ub=self._ub[selected],
            depths=self._depths[selected],
        )

        if len(remaining) > 0:
            self._lb = self._lb[remaining]
            self._ub = self._ub[remaining]
            self._depths = self._depths[remaining]
        else:
            self._lb = None
            self._ub = None
            self._depths = None

        return result

    def evict_to(self, cap: int) -> int:
        total = len(self)
        if total <= cap or cap <= 0:
            return 0
        assert self._lb is not None and self._ub is not None and self._depths is not None
        self._lb = self._lb[:cap].clone()
        self._ub = self._ub[:cap].clone()
        self._depths = self._depths[:cap].clone()
        return total - cap

    def __len__(self) -> int:
        return 0 if self._lb is None else self._lb.shape[0]

    def nbytes(self) -> int:
        return sum(_tensor_nbytes(t) for t in (self._lb, self._ub, self._depths))

    @property
    def bytes_per_node(self) -> float:
        return self.nbytes() / len(self) if len(self) else 0.0

    def projected_push_nbytes(self, batch: SubproblemBatch) -> int:
        return (len(self) + batch.batch_size) * sum(
            math.prod(t.shape[1:]) * t.element_size()
            for t in (batch.lb, batch.ub, batch.depths)
        )

    def evict_all(self) -> int:
        total = len(self)
        self._lb = self._ub = self._depths = None
        return total


# ---------------------------------------------------------------------------
# Top-k priority selection (quantitative total order: depth + lower bound)
# ---------------------------------------------------------------------------


def _clone_optional_dict(
    d: Optional[Dict[int, Any]],
) -> Optional[Dict[int, Any]]:
    return cast(
        Optional[Dict[int, Any]],
        _alpha_tree_map(d, lambda leaf: leaf.clone()),
    )


def _index_optional_dict(
    d: Optional[Dict[int, Any]], idx: torch.Tensor
) -> Optional[Dict[int, Any]]:
    return cast(Optional[Dict[int, Any]], _alpha_tree_gather_lanes(d, idx))


def _merge_optional_dict(
    existing: Optional[Dict[int, Any]],
    n_existing: int,
    incoming: Optional[Dict[int, Any]],
    n_incoming: int,
) -> Optional[Dict[int, Any]]:
    # Per-key concat with key-union; a subproblem missing a key is padded with
    # zeros (e.g. split_signs keys differ per branch — a missing layer means "not
    # split", i.e. all-zero signs). Keeps the pool lossless across heterogeneous
    # incremental-state/split structures.
    return cast(
        Optional[Dict[int, Any]],
        _alpha_tree_concat_lanes(existing, incoming, n_existing, n_incoming),
    )


def _tensor_nbytes(t: Optional[torch.Tensor]) -> int:
    return 0 if t is None else t.numel() * t.element_size()


def _tree_nbytes(tree: Any) -> int:
    return sum(leaf.numel() * leaf.element_size() for leaf in _alpha_tree_leaves(tree))


def _tree_to_device(
    tree: Optional[Dict[int, Any]], device: torch.device
) -> Optional[Dict[int, Any]]:
    return cast(
        Optional[Dict[int, Any]],
        _alpha_tree_map(tree, lambda leaf: leaf.to(device)),
    )


def _concat_tree_nbytes(left: Any, right: Any, n_left: int, n_right: int) -> int:
    """Bytes ``_alpha_tree_concat_lanes(left, right, ...)`` would allocate,
    including the zero padding of a subtree present on one side only."""
    if left is None and right is None:
        return 0
    present = left if left is not None else right
    if isinstance(present, torch.Tensor):
        if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
            present, _ = _alpha_concat_views(left, right)
        return (n_left + n_right) * math.prod(present.shape[1:]) * present.element_size()
    if isinstance(present, dict):
        lhs = left if isinstance(left, dict) else {}
        rhs = right if isinstance(right, dict) else {}
        return sum(
            _concat_tree_nbytes(lhs.get(key), rhs.get(key), n_left, n_right)
            for key in set(lhs) | set(rhs)
        )
    if isinstance(present, (list, tuple)):
        lhs_seq = left if left is not None else [None] * len(present)
        rhs_seq = right if right is not None else [None] * len(present)
        return sum(
            _concat_tree_nbytes(lval, rval, n_left, n_right)
            for lval, rval in zip(lhs_seq, rhs_seq)
        )
    raise TypeError(f"unsupported alpha pytree node: {type(present)!r}")


def _compact_tree_inplace(tree: Any, idx: torch.Tensor) -> Any:
    """Keep lanes ``idx`` of every leaf, replacing leaves one at a time so
    at most one old leaf is alive next to its copy."""
    if isinstance(tree, torch.Tensor):
        return tree.index_select(0, idx.to(tree.device))
    if isinstance(tree, dict):
        for key in list(tree):
            tree[key] = _compact_tree_inplace(tree[key], idx)
        return tree
    if isinstance(tree, list):
        for pos in range(len(tree)):
            tree[pos] = _compact_tree_inplace(tree[pos], idx)
        return tree
    if isinstance(tree, tuple):
        return tuple(_compact_tree_inplace(value, idx) for value in tree)
    if tree is None:
        return None
    raise TypeError(f"unsupported alpha pytree node: {type(tree)!r}")


def _scatter_lanes(
    tree: Optional[Dict[int, Any]], positions: torch.Tensor, lanes: int
) -> Optional[Dict[int, Any]]:
    """Place ``tree``'s lanes at ``positions`` of an all-zero ``lanes``-row tree."""
    return cast(
        Optional[Dict[int, Any]],
        _alpha_tree_map(
            tree,
            lambda leaf: torch.zeros(
                (lanes, *leaf.shape[1:]), dtype=leaf.dtype, device=leaf.device
            ).index_copy_(0, positions.to(leaf.device), leaf),
        ),
    )


class OrderFunction(Protocol):
    def __call__(self, depths: torch.Tensor, lower_bound: torch.Tensor) -> torch.Tensor:
        ...


def _advance_order_schedule(order: OrderFunction) -> None:
    """Tick a stateful order's schedule once per wave.

    Scoring and scheduling must stay separate: ``pop`` skips ``_priority_scores``
    whenever the pool already fits the wave, and ``evict_to`` scores without
    consuming a wave. Folding the tick into ``__call__`` therefore made the
    annealing temperature depend on pool size and eviction pressure rather than
    on elapsed waves.
    """
    advance = getattr(order, "advance_schedule", None)
    if advance is not None:
        advance()


class DepthLowerBoundOrder:
    def __init__(self, depth_weight: float = 0.5, bound_weight: float = 0.5) -> None:
        self.depth_weight = depth_weight
        self.bound_weight = bound_weight

    def __call__(self, depths: torch.Tensor, lower_bound: torch.Tensor) -> torch.Tensor:
        dtype = lower_bound.dtype
        eps = torch.finfo(dtype).eps
        d = depths.to(dtype=dtype)
        d_norm = (d - d.min()) / (d.max() - d.min()).clamp(min=eps)
        urgency = (lower_bound.max() - lower_bound) / (
            lower_bound.max() - lower_bound.min()
        ).clamp(min=eps)
        return self.depth_weight * d_norm + self.bound_weight * urgency


class GreedyOrder(DepthLowerBoundOrder):
    """Oliva-Greedy: best-first by lower bound (``|lb|``).

    "Efficient Neural Network Verification via Order Leading Exploration of
    Branch-and-Bound Trees", Guanqin Zhang, Kota Fukuda, Zhenya Zhang,
    H.M.N. Dilum Bandara, Shiping Chen, Jianjun Zhao, Yulei Sui, ECOOP 2025.
    """

    def __init__(self) -> None:
        super().__init__(depth_weight=0.0, bound_weight=1.0)


class SAOrder:
    """Oliva-SA: temperature-annealed exploration order.

    "Efficient Neural Network Verification via Order Leading Exploration of
    Branch-and-Bound Trees", Guanqin Zhang, Kota Fukuda, Zhenya Zhang,
    H.M.N. Dilum Bandara, Shiping Chen, Jianjun Zhao, Yulei Sui, ECOOP 2025.

    ``temp = cooling_rate ** step`` cools each call, so selection explores early and
    converges to greedy (``|lb|`` best-first) as it cools.
    """

    def __init__(self, cooling_rate: float = 0.99) -> None:
        self.cooling_rate = cooling_rate
        self.step = 0

    def advance_schedule(self) -> None:
        self.step += 1

    def __call__(self, depths: torch.Tensor, lower_bound: torch.Tensor) -> torch.Tensor:
        dtype = lower_bound.dtype
        eps = torch.finfo(dtype).eps
        temp = max(self.cooling_rate ** self.step, 1e-6)
        base = (lower_bound.max() - lower_bound) / (
            lower_bound.max() - lower_bound.min()
        ).clamp(min=eps)
        u = torch.rand_like(base).clamp(min=eps, max=1.0 - eps)
        gumbel = -torch.log(-torch.log(u))
        return base / temp + gumbel


class TopKBounding(BoundingStrategy):
    """Priority pool: keep the top-k subproblems chosen by an order callable.

    The BaB tensor (batch) size is capped by compute resources, so when the pool
    holds more subproblems than the requested batch size, the next wave keeps only
    the k highest-priority ones; the rest stay pooled. Priority comes from a
    swappable order strategy (default :class:`DepthLowerBoundOrder` — a
    50/50 blend of depth and lower bound).

    ``k`` caps how many subproblems a single ``pop`` may return, independently of
    the caller's ``batch_size``. ``k = 0`` means unbounded, matching the
    ``BaBConfig.frontier_cap`` convention. Unlike ``evict_to``, capping ``pop``
    never discards a subproblem: the remainder stays pooled for later waves, so
    a smaller ``k`` only re-sorts priorities more often.

    Storage is lossless: bounds, depth, lower bound, parent margins and every
    incremental-state dict (including split_signs, which neuron-split BaB requires) are
    preserved across push/pop.

    Warm-start state (``incremental_alpha`` / ``incremental_eta``) is the only
    state the frontier memory planner may shed: ``offload_warm_state`` moves it
    to host memory (lossless) and ``drop_warm_state`` discards it for the
    lowest-priority rows, which are then popped in cold-only waves and solved
    from the solver's cold start. ``_warm`` is ``None`` until the first drop
    (every row warm); afterwards the warm-state tensors hold only the warm
    rows, in pool order.
    """

    def __init__(
        self,
        order: Optional[OrderFunction] = None,
        select_probe: Optional[Callable[[SubproblemBatch], None]] = None,
        *,
        k: int = 0,
    ) -> None:
        if k < 0:
            raise ValueError(f"top-k must be non-negative, got k={k}")
        self.order: OrderFunction = order if order is not None else DepthLowerBoundOrder()
        self.k = int(k)
        self.select_probe = select_probe
        self._lb: Optional[torch.Tensor] = None
        self._ub: Optional[torch.Tensor] = None
        self._depths: Optional[torch.Tensor] = None
        self._lower_bound: Optional[torch.Tensor] = None
        self._parent_margins: Optional[torch.Tensor] = None
        self._node_id: Optional[torch.Tensor] = None
        self._parent_id: Optional[torch.Tensor] = None
        self._incremental_alpha: Optional[AlphaState] = None
        self._incremental_eta: Optional[Dict[int, torch.Tensor]] = None
        self._split_signs: Optional[Dict[int, torch.Tensor]] = None
        self._propagation_revisions: Optional[torch.Tensor] = None
        self._warm: Optional[torch.Tensor] = None
        self._warm_offloaded = False
        self.cold_pops = 0

    def push(self, batch: SubproblemBatch) -> None:
        size_before = len(self)
        assert batch.batch_size == batch.lb.shape[0], "TOP-K BOUNDING invariant violated: batch size mismatch"
        alpha_in, eta_in, warm_in = self._incoming_warm_state(batch)
        n_new = batch.batch_size
        device, dtype = batch.lb.device, batch.lb.dtype
        lower = (
            batch.lower_bound
            if batch.lower_bound is not None
            else torch.zeros(n_new, dtype=dtype, device=device)
        )
        parent = (
            batch.parent_margins
            if batch.parent_margins is not None
            else torch.zeros(n_new, dtype=dtype, device=device)
        )
        prev_lb, prev_ub, prev_depths = self._lb, self._ub, self._depths
        prev_lower, prev_parent = self._lower_bound, self._parent_margins
        if prev_lb is None:
            self._lb = batch.lb.clone()
            self._ub = batch.ub.clone()
            self._depths = batch.depths.clone()
            self._lower_bound = lower.clone()
            self._parent_margins = parent.clone()
            self._node_id = batch.node_id.clone() if batch.node_id is not None else None
            self._parent_id = batch.parent_id.clone() if batch.parent_id is not None else None
            self._incremental_alpha = _clone_optional_dict(alpha_in)
            self._incremental_eta = _clone_optional_dict(eta_in)
            self._split_signs = _clone_optional_dict(batch.split_signs)
            self._warm = None if warm_in is None else warm_in.clone()
            self._propagation_revisions = torch.full(
                (n_new,), -1, dtype=torch.long, device=device
            )
            self._assert_push_transition(size_before, n_new)
            return

        assert prev_ub is not None and prev_depths is not None
        assert prev_lower is not None and prev_parent is not None
        assert self._propagation_revisions is not None
        assert (self._node_id is None) == (batch.node_id is None)
        assert (self._parent_id is None) == (batch.parent_id is None)
        n_old = prev_lb.shape[0]
        n_old_warm = n_old if self._warm is None else int(self._warm.sum().item())
        n_new_warm = n_new if warm_in is None else int(warm_in.sum().item())
        self._incremental_alpha = _merge_optional_dict(self._incremental_alpha, n_old_warm, alpha_in, n_new_warm)
        self._incremental_eta = _merge_optional_dict(self._incremental_eta, n_old_warm, eta_in, n_new_warm)
        self._split_signs = _merge_optional_dict(self._split_signs, n_old, batch.split_signs, n_new)
        if self._warm is not None or warm_in is not None:
            self._warm = torch.cat([
                self._warm if self._warm is not None
                else torch.ones(n_old, dtype=torch.bool, device=prev_lb.device),
                warm_in.to(prev_lb.device) if warm_in is not None
                else torch.ones(n_new, dtype=torch.bool, device=prev_lb.device),
            ])
        self._lb = torch.cat([prev_lb, batch.lb], dim=0)
        self._ub = torch.cat([prev_ub, batch.ub], dim=0)
        self._depths = torch.cat([prev_depths, batch.depths], dim=0)
        self._lower_bound = torch.cat([prev_lower, lower.to(prev_lower)], dim=0)
        self._parent_margins = torch.cat([prev_parent, parent.to(prev_parent)], dim=0)
        if self._node_id is not None:
            assert batch.node_id is not None
            self._node_id = torch.cat([self._node_id, batch.node_id.to(self._node_id.device)], dim=0)
        if self._parent_id is not None:
            assert batch.parent_id is not None
            self._parent_id = torch.cat([self._parent_id, batch.parent_id.to(self._parent_id.device)], dim=0)
        self._propagation_revisions = torch.cat(
            [
                self._propagation_revisions,
                torch.full(
                    (n_new,),
                    -1,
                    dtype=torch.long,
                    device=self._propagation_revisions.device,
                ),
            ]
        )
        self._assert_push_transition(size_before, n_new)

    def pop(self, batch_size: int = 1) -> SubproblemBatch:
        lb = self._lb
        if lb is None:
            raise IndexError("pop from empty pool")
        total = lb.shape[0]
        n = min(batch_size, total)
        if self.k > 0:
            n = min(n, self.k)
        _advance_order_schedule(self.order)
        if n >= total:
            selected = torch.arange(total, device=lb.device)
            remaining: Optional[torch.Tensor] = None
        else:
            order = torch.argsort(self._priority_scores(), descending=True)
            selected = order[:n]
            remaining = order[n:]

        selected, remaining = self._split_warm_classes(selected, remaining)
        result = self._build(selected)
        if self.select_probe is not None:
            self.select_probe(result)
        if remaining is None or remaining.numel() == 0:
            self._clear()
        else:
            self._restrict(remaining)
        self._assert_pop_transition(total, result.batch_size)
        return result

    def _split_warm_classes(
        self, selected: torch.Tensor, remaining: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Keep one warm-start class per wave: the class of the first selected
        row. Rows of the other class return to the pool for a later wave, so
        the solver never sees a batch mixing real and placeholder warm starts."""
        if self._warm is None:
            return selected, remaining
        warm = self._warm.index_select(0, selected.to(self._warm.device))
        same = warm == warm[0]
        lead_warm = bool(warm[0].item())
        if not bool(same.all().item()):
            same = same.to(selected.device)
            deferred = selected[~same]
            selected = selected[same]
            remaining = deferred if remaining is None else torch.cat([deferred, remaining])
        if not lead_warm:
            self.cold_pops += int(selected.numel())
        return selected, remaining

    # -- frontier memory accounting / shedding ------------------------------

    def warm_state_nbytes(self) -> int:
        """Bytes of the stored ``incremental_alpha`` / ``incremental_eta``."""
        return _tree_nbytes(self._incremental_alpha) + _tree_nbytes(self._incremental_eta)

    def nbytes(self) -> int:
        """Device-resident bytes of every stored field (sum of
        ``numel * element_size``); warm state offloaded to host is excluded
        (see ``offloaded_nbytes``)."""
        total = sum(
            _tensor_nbytes(t)
            for t in (
                self._lb, self._ub, self._depths, self._lower_bound,
                self._parent_margins, self._node_id, self._parent_id,
                self._propagation_revisions, self._warm,
            )
        )
        total += _tree_nbytes(self._split_signs)
        if not self._warm_offloaded:
            total += self.warm_state_nbytes()
        return total

    def offloaded_nbytes(self) -> int:
        """Host bytes of warm state moved off the device."""
        return self.warm_state_nbytes() if self._warm_offloaded else 0

    @property
    def bytes_per_node(self) -> float:
        """Mean device-resident bytes per pending node."""
        n = len(self)
        return self.nbytes() / n if n else 0.0

    @property
    def warm_offloaded(self) -> bool:
        return self._warm_offloaded

    def warm_nodes(self) -> int:
        """Pending rows whose warm-start state is stored."""
        if self._incremental_alpha is None and self._incremental_eta is None:
            return 0
        return len(self) if self._warm is None else int(self._warm.sum().item())

    def offload_warm_state(self) -> bool:
        """Move the warm-start state to host memory for the rest of the run.

        Lossless: popped rows are copied back to the frontier's device, and
        warm state pushed later is stored on the host too. Returns False when
        it was already moved."""
        if self._warm_offloaded:
            return False
        host = torch.device("cpu")
        self._incremental_alpha = _tree_to_device(self._incremental_alpha, host)
        self._incremental_eta = _tree_to_device(self._incremental_eta, host)
        self._warm_offloaded = True
        return True

    def drop_warm_state(self, count: int) -> int:
        """Discard the warm start of the ``count`` lowest-priority warm rows.

        The rows stay pending (bounds, depth, split signs and lower bound are
        kept) and are later solved from the solver's cold start, which can only
        weaken their bounds. Returns the number of rows made cold."""
        if count <= 0 or self.warm_nodes() == 0:
            return 0
        assert self._lb is not None
        warm = (
            self._warm if self._warm is not None
            else torch.ones(len(self), dtype=torch.bool, device=self._lb.device)
        )
        warm_rows = torch.where(warm)[0]
        scores = self._priority_scores().index_select(0, warm_rows.to(self._lb.device))
        coldest = torch.argsort(scores)[:count]
        keep = torch.ones(warm_rows.numel(), dtype=torch.bool, device=warm_rows.device)
        keep[coldest.to(keep.device)] = False
        kept_slots = torch.where(keep)[0]
        try:
            self._incremental_alpha = _compact_tree_inplace(self._incremental_alpha, kept_slots)
            self._incremental_eta = _compact_tree_inplace(self._incremental_eta, kept_slots)
            oom = False
        except torch.cuda.OutOfMemoryError:
            oom = True
        if oom:
            # No room even for the kept copy: every row goes cold.
            self._incremental_alpha = self._incremental_eta = None
            self._warm = torch.zeros(len(self), dtype=torch.bool, device=self._lb.device)
            return int(warm_rows.numel())
        new_warm = warm.clone()
        new_warm[warm_rows[~keep]] = False
        self._warm = new_warm
        return int(coldest.numel())

    def _incoming_warm_state(
        self, batch: SubproblemBatch
    ) -> Tuple[Optional[AlphaState], Optional[Dict[int, torch.Tensor]], Optional[torch.Tensor]]:
        """Warm-start rows of ``batch`` to store, and its warm mask (None = all warm).

        Once rows have been made cold, a batch arriving without any warm state
        (e.g. a cold wave returned after an OOM) is recorded as cold instead of
        being zero-padded into warm rows."""
        mask = batch.warm_start
        if (
            mask is None
            and self._warm is not None
            and batch.incremental_alpha is None
            and batch.incremental_eta is None
            and (self._incremental_alpha is not None or self._incremental_eta is not None)
        ):
            mask = torch.zeros(batch.batch_size, dtype=torch.bool, device=batch.lb.device)
        alpha: Optional[AlphaState] = batch.incremental_alpha
        eta = batch.incremental_eta
        if mask is not None:
            mask = mask.to(device=batch.lb.device, dtype=torch.bool)
            if not bool(mask.all().item()):
                rows = torch.where(mask)[0]
                alpha = cast(Optional[AlphaState], _index_optional_dict(alpha, rows))
                eta = _index_optional_dict(eta, rows)
        if self._warm_offloaded:
            host = torch.device("cpu")
            alpha = cast(Optional[AlphaState], _tree_to_device(alpha, host))
            eta = _tree_to_device(eta, host)
        return alpha, eta, mask

    def evict_all(self) -> int:
        """Drop every pending row (a memory allowance below one node)."""
        total = len(self)
        self._clear()
        return total

    def projected_push_nbytes(self, batch: SubproblemBatch) -> int:
        """Device-resident bytes after ``push(batch)``, computed without
        allocating: per-field concat sizes including the key-union zero padding
        of the per-layer dicts; warm state is excluded once offloaded."""
        n_old, n_new = len(self), batch.batch_size
        n = n_old + n_new
        row_bytes = sum(
            math.prod(t.shape[1:]) * t.element_size()
            for t in (batch.lb, batch.ub, batch.depths)
        )
        total = n * row_bytes + 2 * n * batch.lb.element_size() + n * 8
        for mine, theirs in ((self._node_id, batch.node_id), (self._parent_id, batch.parent_id)):
            if mine is not None or theirs is not None:
                total += n * 8
        total += _concat_tree_nbytes(self._split_signs, batch.split_signs, n_old, n_new)
        if self._warm is not None or batch.warm_start is not None:
            total += n
        if not self._warm_offloaded:
            n_old_warm = n_old if self._warm is None else int(self._warm.sum().item())
            n_new_warm = n_new if batch.warm_start is None else int(batch.warm_start.sum().item())
            for mine_tree, their_tree in (
                (self._incremental_alpha, batch.incremental_alpha),
                (self._incremental_eta, batch.incremental_eta),
            ):
                total += _concat_tree_nbytes(mine_tree, their_tree, n_old_warm, n_new_warm)
        return total

    def _gather_warm_state(
        self, idx: torch.Tensor, upload: bool = True
    ) -> Tuple[Optional[AlphaState], Optional[Dict[int, torch.Tensor]], Optional[torch.Tensor]]:
        """Warm state of rows ``idx``, on the frontier device when ``upload``
        (else on its storage device); cold rows of a mixed selection get zero
        placeholders flagged in the returned mask."""
        assert self._lb is not None
        device = self._lb.device if upload else torch.device("cpu") if self._warm_offloaded else self._lb.device
        alpha_store, eta_store = self._incremental_alpha, self._incremental_eta
        if self._warm is None:
            alpha = _index_optional_dict(alpha_store, idx)
            eta = _index_optional_dict(eta_store, idx)
            if self._warm_offloaded:
                alpha, eta = _tree_to_device(alpha, device), _tree_to_device(eta, device)
            return cast(Optional[AlphaState], alpha), eta, None
        warm = self._warm.index_select(0, idx)
        slots = (torch.cumsum(self._warm.long(), dim=0) - 1).index_select(0, idx)
        n_warm = int(warm.sum().item())
        if n_warm == 0:
            return None, None, None
        warm_pos = torch.where(warm)[0]
        alpha = _tree_to_device(_index_optional_dict(alpha_store, slots[warm_pos]), device)
        eta = _tree_to_device(_index_optional_dict(eta_store, slots[warm_pos]), device)
        if n_warm == int(idx.numel()):
            return cast(Optional[AlphaState], alpha), eta, None
        lanes = int(idx.numel())
        return (
            cast(Optional[AlphaState], _scatter_lanes(alpha, warm_pos, lanes)),
            _scatter_lanes(eta, warm_pos, lanes),
            warm,
        )

    def _assert_push_transition(self, size_before: int, added: int) -> None:
        size_after = len(self)
        assert size_after == size_before + added, (
            "TOP-K BOUNDING invariant violated: push changed pool by wrong size "
            f"(before={size_before}, batch_size={added}, after={size_after}, "
            f"expected_after={size_before + added})"
        )

    def _assert_pop_transition(self, size_before: int, popped: int) -> None:
        size_after = len(self)
        cap_ok = self.k <= 0 or popped <= self.k
        bookkeeping_ok = size_before - popped == size_after
        assert cap_ok and bookkeeping_ok, (
            "TOP-K BOUNDING invariant violated: pop cap or pool bookkeeping failed "
            f"(before={size_before}, popped={popped}, after={size_after}, "
            f"k={self.k}, expected_after={size_before - popped})"
        )

    def view_all(self) -> SubproblemBatch:
        """Return a lossless, non-destructive view of the full frontier."""
        if self._lb is None:
            raise IndexError("view of empty pool")
        indices = torch.arange(self._lb.shape[0], device=self._lb.device)
        # Offloaded warm state stays on the host: push re-offloads it anyway.
        return self._build(indices, upload=False)

    def replace_all(self, batch: SubproblemBatch) -> None:
        """Replace the full frontier without scoring or advancing schedules."""
        self._clear()
        if batch.batch_size > 0:
            self.push(batch)

    def propagation_batch(
        self, library_revision: int
    ) -> Optional[Tuple[torch.Tensor, SubproblemBatch]]:
        """Return only rows not checked against the current core library."""
        if self._lb is None:
            return None
        assert self._propagation_revisions is not None
        indices = torch.where(self._propagation_revisions != library_revision)[0]
        if indices.numel() == 0:
            return None
        # Propagation reads split signs only; the warm state is not copied.
        return indices, (self._build(indices) if self._warm is None and not self._warm_offloaded else self._build_base(indices))

    def apply_propagation(
        self,
        selected: torch.Tensor,
        kept: torch.Tensor,
        propagated: SubproblemBatch,
        library_revision: int,
    ) -> None:
        """Write inferred signs and remove discharged rows from a selected subset."""
        assert self._lb is not None and self._propagation_revisions is not None
        selected = selected.to(self._lb.device)
        kept = kept.to(selected.device)
        assert propagated.batch_size == int(kept.numel()), (
            "incremental propagation survivor index mismatch"
        )
        survivor_indices = selected.index_select(0, kept)
        if propagated.split_signs is not None:
            if self._split_signs is None:
                self._split_signs = {}
            for layer_id, incoming in propagated.split_signs.items():
                stored = self._split_signs.get(layer_id)
                if stored is None:
                    stored = torch.zeros(
                        (len(self), *incoming.shape[1:]),
                        dtype=incoming.dtype,
                        device=incoming.device,
                    )
                    self._split_signs[layer_id] = stored
                stored.index_copy_(
                    0, survivor_indices.to(stored.device), incoming.to(stored)
                )
        self._propagation_revisions[survivor_indices] = library_revision

        selected_survived = torch.zeros(
            selected.numel(), dtype=torch.bool, device=selected.device
        )
        selected_survived[kept] = True
        discharged = selected[~selected_survived]
        if discharged.numel() == 0:
            return
        retain = torch.ones(len(self), dtype=torch.bool, device=self._lb.device)
        retain[discharged] = False
        retained = torch.where(retain)[0]
        if retained.numel() == 0:
            self._clear()
        else:
            self._restrict(retained)

    def _priority_scores(self) -> torch.Tensor:
        depths_t, lb = self._depths, self._lower_bound
        assert depths_t is not None and lb is not None
        return self.order(depths_t, lb)

    def _build(self, idx: torch.Tensor, upload: bool = True) -> SubproblemBatch:
        lb = self._lb
        assert lb is not None
        idx = idx.to(lb.device)
        alpha, eta, warm = self._gather_warm_state(idx, upload)
        batch = self._build_base(idx)
        batch.incremental_alpha, batch.incremental_eta, batch.warm_start = alpha, eta, warm
        return batch

    def _build_base(self, idx: torch.Tensor) -> SubproblemBatch:
        """Rows ``idx`` of every field except the warm-start state."""
        lb, ub, depths = self._lb, self._ub, self._depths
        lower, parent = self._lower_bound, self._parent_margins
        assert lb is not None and ub is not None and depths is not None
        assert lower is not None and parent is not None
        idx = idx.to(lb.device)
        return SubproblemBatch(
            lb=lb.index_select(0, idx),
            ub=ub.index_select(0, idx),
            depths=depths.index_select(0, idx),
            split_signs=_index_optional_dict(self._split_signs, idx),
            parent_margins=parent.index_select(0, idx),
            lower_bound=lower.index_select(0, idx),
            node_id=(
                self._node_id.index_select(0, idx.to(self._node_id.device))
                if self._node_id is not None
                else None
            ),
            parent_id=(
                self._parent_id.index_select(0, idx.to(self._parent_id.device))
                if self._parent_id is not None
                else None
            ),
        )

    def _restrict(self, idx: torch.Tensor) -> None:
        if self._warm is None and not self._warm_offloaded:
            revisions = self._propagation_revisions
            assert revisions is not None
            kept = self._build(idx)
            self._lb, self._ub, self._depths = kept.lb, kept.ub, kept.depths
            self._lower_bound, self._parent_margins = kept.lower_bound, kept.parent_margins
            self._node_id, self._parent_id = kept.node_id, kept.parent_id
            self._incremental_alpha, self._incremental_eta = kept.incremental_alpha, kept.incremental_eta
            self._split_signs = kept.split_signs
            self._propagation_revisions = revisions.index_select(0, idx.to(revisions.device))
            return
        revisions = self._propagation_revisions
        assert revisions is not None
        kept = self._build_base(idx)
        self._lb, self._ub, self._depths = kept.lb, kept.ub, kept.depths
        self._lower_bound, self._parent_margins = kept.lower_bound, kept.parent_margins
        self._node_id, self._parent_id = kept.node_id, kept.parent_id
        self._split_signs = kept.split_signs
        # Warm state stays on its storage device; only kept warm rows remain.
        if self._warm is None:
            slots = idx
        else:
            warm = self._warm.index_select(0, idx.to(self._warm.device))
            slots = (torch.cumsum(self._warm.long(), dim=0) - 1).index_select(
                0, idx.to(self._warm.device)
            )[warm]
            self._warm = warm
        self._incremental_alpha = _compact_tree_inplace(self._incremental_alpha, slots)
        self._incremental_eta = _compact_tree_inplace(self._incremental_eta, slots)
        self._propagation_revisions = revisions.index_select(
            0, idx.to(revisions.device)
        )

    def _clear(self) -> None:
        self._lb = self._ub = self._depths = None
        self._lower_bound = self._parent_margins = None
        self._node_id = self._parent_id = None
        self._incremental_alpha = self._incremental_eta = self._split_signs = None
        self._propagation_revisions = None
        self._warm = None

    def evict_to(self, cap: int) -> int:
        total = len(self)
        if total <= cap or cap <= 0:
            return 0
        order = torch.argsort(self._priority_scores(), descending=True)
        self._restrict(order[:cap])
        return total - cap

    def __len__(self) -> int:
        return 0 if self._lb is None else self._lb.shape[0]


class DiverseTopKBounding(TopKBounding):
    """Top-k priority pool with optional diversity-aware scheduling.

    ``diversity_mode='hash'`` preserves the original exact split-sign
    de-duplication behaviour.  ``'soft'`` uses a value-weighted farthest-point
    selector over split-sign vectors (neuron splitting) or box centres (input
    splitting / no split signs).  This is scheduling-only: no node is pruned, no
    bound is changed, and all non-selected nodes remain in the pool for later
    waves.  ``'none'`` is an explicit off switch and is equivalent to
    :class:`TopKBounding` selection.
    """

    def __init__(
        self,
        order: Optional[OrderFunction] = None,
        select_probe: Optional[Callable[[SubproblemBatch], None]] = None,
        *,
        k: int = 0,
        diversity_mode: Literal["hash", "soft", "none"] = "hash",
        diversity_weight: float = 1.0,
    ) -> None:
        super().__init__(order, select_probe=select_probe, k=k)
        if diversity_mode not in {"hash", "soft", "none"}:
            raise ValueError(
                "diversity_mode must be one of 'hash', 'soft', or 'none', "
                f"got {diversity_mode!r}"
            )
        self.diversity_mode = diversity_mode
        self.diversity_weight = float(diversity_weight)

    def pop(self, batch_size: int = 1) -> SubproblemBatch:
        lb = self._lb
        if lb is None:
            raise IndexError("pop from empty pool")
        total = lb.shape[0]
        n = min(batch_size, total)
        if self.k > 0:
            n = min(n, self.k)
        _advance_order_schedule(self.order)
        if n >= total:
            selected = torch.arange(total, device=lb.device)
            remaining: Optional[torch.Tensor] = None
        else:
            order = torch.argsort(self._priority_scores(), descending=True)
            if self.diversity_mode == "none":
                selected = order[:n]
            elif self.diversity_mode == "soft":
                selected = self._soft_diverse_select(order, n)
            else:
                selected = self._dedup_select(order, n)
            selected_mask = torch.zeros(total, dtype=torch.bool, device=lb.device)
            selected_mask[selected] = True
            remaining = order[~selected_mask.index_select(0, order)]

        selected, remaining = self._split_warm_classes(selected, remaining)
        result = self._build(selected)
        if self.select_probe is not None:
            self.select_probe(result)
        if remaining is None or remaining.numel() == 0:
            self._clear()
        else:
            self._restrict(remaining)
        self._assert_pop_transition(total, result.batch_size)
        return result

    def _dedup_select(self, order: torch.Tensor, n: int) -> torch.Tensor:
        signatures = self._split_sign_signatures()
        if signatures is None:
            return order[:n]

        selected: List[int] = []
        selected_set: Set[int] = set()
        seen: Set[Tuple[int, ...]] = set()
        ordered_indices = [int(i) for i in order.detach().cpu().tolist()]

        for idx in ordered_indices:
            signature = signatures[idx]
            if signature in seen:
                continue
            selected.append(idx)
            selected_set.add(idx)
            seen.add(signature)
            if len(selected) == n:
                break

        if len(selected) < n:
            for idx in ordered_indices:
                if idx in selected_set:
                    continue
                selected.append(idx)
                selected_set.add(idx)
                if len(selected) == n:
                    break

        return torch.tensor(selected, dtype=torch.long, device=order.device)

    def _split_sign_signatures(self) -> Optional[List[Tuple[int, ...]]]:
        split_signs = self._split_signs
        if not split_signs:
            return None

        pieces: List[torch.Tensor] = []
        for layer_id in sorted(split_signs):
            value = split_signs[layer_id]
            if value.shape[0] == 0:
                continue
            pieces.append(value.detach().reshape(value.shape[0], -1).to(device="cpu"))
        if not pieces:
            return None

        features = torch.cat(pieces, dim=1)
        if features.shape[1] == 0:
            return None
        return [tuple(int(v) for v in row.tolist()) for row in features]

    def _soft_diverse_select(self, order: torch.Tensor, n: int) -> torch.Tensor:
        features = self._diversity_features()
        if features is None or features.shape[0] < 2:
            return order[:n]

        candidate_features = features.index_select(0, order.to(features.device))
        distances = torch.cdist(candidate_features, candidate_features, p=2)
        max_distance = distances.max().clamp(min=torch.finfo(distances.dtype).eps)
        distances = distances / max_distance

        lb = self._lb
        assert lb is not None
        priorities = self._priority_scores().index_select(0, order.to(lb.device))
        priorities = priorities.to(device=features.device, dtype=features.dtype)
        priority_span = (priorities.max() - priorities.min()).clamp(
            min=torch.finfo(priorities.dtype).eps
        )
        priorities = (priorities - priorities.min()) / priority_span

        selected_positions: List[int] = [0]
        remaining = torch.ones(order.shape[0], dtype=torch.bool, device=features.device)
        remaining[0] = False
        min_dist_to_selected = distances[0].clone()

        while len(selected_positions) < n and bool(remaining.any().item()):
            scores = priorities + self.diversity_weight * min_dist_to_selected
            scores = scores.masked_fill(~remaining, -torch.inf)
            next_pos = int(torch.argmax(scores).item())
            selected_positions.append(next_pos)
            remaining[next_pos] = False
            min_dist_to_selected = torch.minimum(
                min_dist_to_selected, distances[next_pos]
            )

        if len(selected_positions) < n:
            for pos in range(order.shape[0]):
                if pos not in selected_positions:
                    selected_positions.append(pos)
                    if len(selected_positions) == n:
                        break
        return order[torch.tensor(selected_positions, dtype=torch.long, device=order.device)]

    def _diversity_features(self) -> Optional[torch.Tensor]:
        split_features = self._split_sign_features()
        if split_features is not None:
            return split_features
        return self._box_center_features()

    def _split_sign_features(self) -> Optional[torch.Tensor]:
        split_signs = self._split_signs
        if not split_signs:
            return None
        pieces: List[torch.Tensor] = []
        for layer_id in sorted(split_signs):
            value = split_signs[layer_id]
            if value.shape[0] == 0:
                continue
            pieces.append(
                value.detach().reshape(value.shape[0], -1).to(dtype=get_default_dtype())
            )
        if not pieces:
            return None
        features = torch.cat(pieces, dim=1)
        if features.shape[1] == 0:
            return None
        return features

    def _box_center_features(self) -> Optional[torch.Tensor]:
        if self._lb is None or self._ub is None:
            return None
        centers = (
            ((self._lb + self._ub) / 2.0)
            .detach()
            .to(dtype=get_default_dtype())
        )
        if centers.ndim > 2:
            centers = centers.reshape(centers.shape[0], -1)
        if centers.shape[1] == 0:
            return None
        return centers


ROOT_PARENT = -1


def _mcts_visit_accounting_valid(
    parent: Dict[int, int], visits: Dict[int, int], n_tot: int
) -> bool:
    if n_tot < 0 or any(count < 0 for count in visits.values()):
        return False
    child_visit_sums: Dict[int, int] = {}
    for node, count in visits.items():
        parent_id = parent.get(node, ROOT_PARENT)
        if parent_id != ROOT_PARENT:
            child_visit_sums[parent_id] = child_visit_sums.get(parent_id, 0) + count
    if any(visits.get(node, 0) < total for node, total in child_visit_sums.items()):
        return False
    root_visits = sum(
        count
        for node, count in visits.items()
        if parent.get(node, ROOT_PARENT) == ROOT_PARENT
    )
    return root_visits == n_tot


def _mcts_visit_diagnostics(
    parent: Dict[int, int], visits: Dict[int, int]
) -> tuple[int, int, int]:
    child_visit_sums: Dict[int, int] = {}
    for node, count in visits.items():
        parent_id = parent.get(node, ROOT_PARENT)
        if parent_id != ROOT_PARENT:
            child_visit_sums[parent_id] = child_visit_sums.get(parent_id, 0) + count
    root_visits = sum(
        count
        for node, count in visits.items()
        if parent.get(node, ROOT_PARENT) == ROOT_PARENT
    )
    min_visit = min(visits.values(), default=0)
    max_child_excess = max(
        (total - visits.get(node, 0) for node, total in child_visit_sums.items()),
        default=0,
    )
    return root_visits, min_visit, max_child_excess


class MCTSBounding(TopKBounding):
    """MCTS side tables (``N``/``Q``) over the BaB tree, maintained as a pure observer.

    ``order`` governs **eviction priority only**; selection is UCB1 (W2). At this
    observer stage ``pop`` is plain top-k by ``order`` — no UCB1 term is applied
    yet — while ``evict_to`` keeps the lb-based ``order`` priority so a frontier
    cap always drops the least promising leaves.

    Storage is lossless: bounds, depth, lower bound, parent margins, ``node_id`` /
    ``parent_id`` provenance and every incremental-state dict (including
    split_signs) are preserved across push/pop.
    """

    def __init__(
        self,
        order: Optional[OrderFunction] = None,
        select_probe: Optional[Callable[[SubproblemBatch], None]] = None,
        *,
        exploration: float = 1.0,
        lambda_: float = 0.5,
        virtual_loss: float = 1.0,
    ) -> None:
        super().__init__(order, select_probe)
        self.exploration = float(exploration)
        self.lambda_ = float(lambda_)
        self.virtual_loss = float(virtual_loss)
        self.parent: Dict[int, int] = {}
        self.N: Dict[int, int] = {}
        self.Q: Dict[int, float] = {}
        self.n_tot = 0

    def push(self, batch: SubproblemBatch) -> None:
        if batch.node_id is None or batch.parent_id is None:
            missing = "node_id" if batch.node_id is None else "parent_id"
            raise ValueError(
                f"MCTSBounding.push requires provenance, but batch.{missing} is None"
            )
        super().push(batch)
        for nid, pid in zip(batch.node_id.tolist(), batch.parent_id.tolist()):
            self.parent[int(nid)] = int(pid)

    def observe(
        self,
        node_ids: torch.Tensor,
        lower_bounds: torch.Tensor,
        statuses: Sequence[str],
        depths: torch.Tensor,
        n_unstable: int,
    ) -> None:
        """Backpropagate one wave of solve results into ``N``/``Q``.

        Callers must invoke this only after counterexample validation, so a
        ``SAT`` status here always denotes a spurious, still-unresolved lane.
        """
        ids = node_ids.detach().cpu()
        lb = lower_bounds.detach().cpu().to(dtype=get_default_dtype())
        depth = depths.detach().cpu().to(dtype=get_default_dtype())
        blended = self.lambda_ * depth / max(n_unstable, 1) + (
            1.0 - self.lambda_
        ) * self._rank01(ids, lb)
        blended = torch.where(
            torch.isnan(lb) | torch.isnan(blended),
            torch.full_like(blended, -math.inf),
            blended,
        )

        for i, (node, status) in enumerate(zip(ids.tolist(), statuses)):
            reward = -math.inf if status == SolveStatus.UNSAT else float(blended[i])
            self.n_tot += 1
            visited = int(node)
            while visited != ROOT_PARENT:
                self.N[visited] = self.N.get(visited, 0) + 1
                visited = self.parent[visited]
            valued = int(node)
            while valued != ROOT_PARENT and self.Q.get(valued, -math.inf) < reward:
                self.Q[valued] = reward
                valued = self.parent[valued]
        assert _mcts_visit_accounting_valid(self.parent, self.N, self.n_tot), (
            "MCTS invariant violated: visit-tree accounting is inconsistent "
            f"(n_tot={self.n_tot}, root_visits="
            f"{_mcts_visit_diagnostics(self.parent, self.N)[0]}, min_visit="
            f"{_mcts_visit_diagnostics(self.parent, self.N)[1]}, "
            f"max_child_excess={_mcts_visit_diagnostics(self.parent, self.N)[2]})"
        )

    def frontier_parent_visit_histogram(self) -> Dict[int, int]:
        parent_ids = self._parent_id
        if parent_ids is None:
            return {}
        counts = Counter(self.N.get(int(pid), 0) for pid in parent_ids.tolist())
        return dict(sorted(counts.items()))

    @staticmethod
    def _rank01(node_ids: torch.Tensor, lower_bounds: torch.Tensor) -> torch.Tensor:
        n = int(lower_bounds.numel())
        if n < 2:
            return torch.zeros_like(lower_bounds)
        # Rank lexicographically on (lb, node_id): scale-free in [0, 1] and
        # invariant to the order the wave's results arrive in.
        by_id = torch.argsort(node_ids, stable=True)
        order = by_id[torch.argsort(lower_bounds[by_id], stable=True)]
        ranks = torch.empty_like(lower_bounds)
        ranks[order] = torch.arange(n, dtype=lower_bounds.dtype, device=lower_bounds.device)
        return ranks / (n - 1)
