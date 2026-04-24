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
#   Contains the abstract base class ``BoundingStrategy`` and three
#   implementations:
#
#     * ``RandomBounding`` — uniform-random pop (baseline / ablation).
#     * ``BFSBounding``    — FIFO deque; level-order traversal.
#     * ``DFSBounding``    — LIFO stack; depth-first traversal.
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

import collections
from abc import ABC, abstractmethod
from typing import Optional

import torch

from act.back_end.bab.node import SubproblemBatch, _concat_subproblem_batches


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
            self._lb = torch.cat([self._lb, batch.lb], dim=0)
            self._ub = torch.cat([self._ub, batch.ub], dim=0)
            self._depths = torch.cat([self._depths, batch.depths], dim=0)

    def pop(self, batch_size: int = 1) -> SubproblemBatch:
        if self.empty:
            raise IndexError("pop from empty pool")

        n = min(batch_size, len(self))
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

    def __len__(self) -> int:
        return 0 if self._lb is None else self._lb.shape[0]


# ---------------------------------------------------------------------------
# BFS (FIFO) pool — queue of SubproblemBatch, preserves insertion order
# ---------------------------------------------------------------------------


class BFSBounding(BoundingStrategy):
    """First-in-first-out pool using a ``collections.deque`` of batches.

    Push appends the incoming batch to the back; pop drains from the front.
    Order is preserved across batch boundaries:

    * If the front-most queue entry has ``N`` rows and ``N <= batch_size``,
      the whole front is taken and the loop continues with the next entry
      until ``batch_size`` is reached or the queue is empty.
    * If the front-most entry has ``N > batch_size``, it is split via
      :meth:`SubproblemBatch.select` — the first ``batch_size`` rows are
      returned and the remaining ``N - batch_size`` rows are pushed **back
      to the front** (``deque.appendleft``) so subsequent pops see them
      next, preserving FIFO semantics at row granularity.

    Unlike :class:`RandomBounding`, this class does not re-stack tensors
    internally — it keeps the original batch shape intact (cheaper when
    ``push`` rate ≈ ``pop`` rate).
    """

    def __init__(self) -> None:
        self._queue: collections.deque[SubproblemBatch] = collections.deque()

    # -- BoundingStrategy interface -----------------------------------------

    def push(self, batch: SubproblemBatch) -> None:
        if batch.batch_size == 0:
            return
        self._queue.append(batch)

    def pop(self, batch_size: int = 1) -> SubproblemBatch:
        if self.empty:
            raise IndexError("pop from empty pool")
        if batch_size <= 0:
            raise ValueError(f"pop: batch_size must be positive, got {batch_size}")

        pieces: list[SubproblemBatch] = []
        remaining = batch_size

        while remaining > 0 and len(self._queue) > 0:
            front = self._queue.popleft()
            n = front.batch_size

            if n <= remaining:
                pieces.append(front)
                remaining -= n
            else:
                device = front.lb.device
                head_idx = torch.arange(remaining, dtype=torch.long, device=device)
                tail_idx = torch.arange(remaining, n, dtype=torch.long, device=device)
                pieces.append(front.select(head_idx))
                # Push the unclaimed tail back to the FRONT so the next
                # pop() still sees the same row-order as the caller pushed.
                self._queue.appendleft(front.select(tail_idx))
                remaining = 0

        return _concat_subproblem_batches(*pieces)

    def __len__(self) -> int:
        return sum(b.batch_size for b in self._queue)


# ---------------------------------------------------------------------------
# DFS (LIFO) pool — stack of SubproblemBatch, most-recent batch popped first
# ---------------------------------------------------------------------------


class DFSBounding(BoundingStrategy):
    """Last-in-first-out pool using a stack of batches.

    Push appends to the top; pop drains from the top. Matches a classical
    depth-first search: when a subproblem is split into ``left`` / ``right``
    and both are pushed, the batch pushed **last** (``right``) is popped
    first, so BaB dives down the most recent branch before backtracking.

    Semantics at batch granularity
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    * Batches pop in reverse of their push order (stack LIFO).
    * **Within a single batch, rows keep their insertion order.** When
      ``pop(k)`` asks for fewer rows than the top batch holds, the first
      ``k`` rows are returned and the tail is pushed back on top of the
      stack, so a subsequent ``pop`` still sees those tail rows next.
      This pairs naturally with BaB's ``push(left); push(right)`` idiom
      where each batch is a set of siblings at the same depth.

    Unlike :class:`RandomBounding`, this class does not re-stack tensors
    internally — it keeps each pushed batch shape intact.
    """

    def __init__(self) -> None:
        self._stack: list[SubproblemBatch] = []

    # -- BoundingStrategy interface -----------------------------------------

    def push(self, batch: SubproblemBatch) -> None:
        if batch.batch_size == 0:
            return
        self._stack.append(batch)

    def pop(self, batch_size: int = 1) -> SubproblemBatch:
        if self.empty:
            raise IndexError("pop from empty pool")
        if batch_size <= 0:
            raise ValueError(f"pop: batch_size must be positive, got {batch_size}")

        pieces: list[SubproblemBatch] = []
        remaining = batch_size

        while remaining > 0 and len(self._stack) > 0:
            top = self._stack.pop()
            n = top.batch_size

            if n <= remaining:
                pieces.append(top)
                remaining -= n
            else:
                device = top.lb.device
                head_idx = torch.arange(remaining, dtype=torch.long, device=device)
                tail_idx = torch.arange(remaining, n, dtype=torch.long, device=device)
                pieces.append(top.select(head_idx))
                # Push the unclaimed tail back on top so the next pop() still
                # drains the most-recently-pushed batch's remaining rows first.
                self._stack.append(top.select(tail_idx))
                remaining = 0

        return _concat_subproblem_batches(*pieces)

    def __len__(self) -> int:
        return sum(b.batch_size for b in self._stack)
