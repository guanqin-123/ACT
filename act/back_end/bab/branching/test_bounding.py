"""Tests for act.back_end.bab.branching.bounding — BFS/DFS/RandomBounding."""
from __future__ import annotations

import pytest
import torch

from act.back_end.bab.branching.bounding import BFSBounding, DFSBounding, RandomBounding
from act.back_end.bab.eta import EtaState
from act.back_end.bab.node import Split, SubproblemBatch
from act.back_end.core import Bounds


def _make_single_row(tag: float, *, with_eta: bool = True) -> SubproblemBatch:
    """Build a 1-row batch with a unique tag-value embedded in lb/ub."""
    lb = torch.tensor([[tag, tag]])
    ub = lb + 1.0
    depths = torch.tensor([0], dtype=torch.long)
    if with_eta:
        eta = EtaState(
            val={1: torch.tensor([[tag]])},
            sign={1: torch.zeros(1, 1)},
            point={1: torch.zeros(1, 1)},
        )
        histories: list[list[Split]] = [[]]
        return SubproblemBatch(
            lb=lb, ub=ub, depths=depths, eta=eta, histories=histories
        )
    return SubproblemBatch(lb=lb, ub=ub, depths=depths)


def _make_multi_row(tags: list[float]) -> SubproblemBatch:
    """Build a batch whose i-th row carries a unique scalar tag."""
    B = len(tags)
    lb = torch.tensor([[t, t] for t in tags])
    ub = lb + 1.0
    depths = torch.zeros(B, dtype=torch.long)
    eta = EtaState(
        val={1: torch.tensor([[t] for t in tags])},
        sign={1: torch.zeros(B, 1)},
        point={1: torch.zeros(B, 1)},
    )
    histories: list[list[Split]] = [[] for _ in range(B)]
    return SubproblemBatch(
        lb=lb, ub=ub, depths=depths, eta=eta, histories=histories
    )


def test_bfs_push_pop_fifo_order():
    pool = BFSBounding()
    pool.push(_make_single_row(1.0))
    pool.push(_make_single_row(2.0))
    pool.push(_make_single_row(3.0))

    assert len(pool) == 3
    assert not pool.empty

    out1 = pool.pop(1)
    out2 = pool.pop(1)
    out3 = pool.pop(1)

    assert out1.lb[0, 0].item() == 1.0
    assert out2.lb[0, 0].item() == 2.0
    assert out3.lb[0, 0].item() == 3.0
    assert pool.empty


def test_bfs_pop_larger_than_single_batch():
    pool = BFSBounding()
    pool.push(_make_multi_row([10.0, 20.0, 30.0, 40.0]))

    assert len(pool) == 4

    first = pool.pop(2)
    assert first.batch_size == 2
    assert first.lb[:, 0].tolist() == [10.0, 20.0]
    assert len(pool) == 2

    second = pool.pop(2)
    assert second.batch_size == 2
    assert second.lb[:, 0].tolist() == [30.0, 40.0]
    assert pool.empty


def test_bfs_pop_spans_multiple_batches():
    pool = BFSBounding()
    pool.push(_make_multi_row([1.0, 2.0, 3.0]))
    pool.push(_make_multi_row([4.0, 5.0]))

    assert len(pool) == 5

    got = pool.pop(4)
    assert got.batch_size == 4
    assert got.lb[:, 0].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert got.eta is not None
    # Eta rows must follow the same 1→2→3→4 order.
    assert got.eta.val[1].squeeze(-1).tolist() == [1.0, 2.0, 3.0, 4.0]
    assert got.histories is not None and len(got.histories) == 4

    assert len(pool) == 1
    remainder = pool.pop(10)
    assert remainder.batch_size == 1
    assert remainder.lb[0, 0].item() == 5.0
    assert pool.empty


def test_bfs_empty():
    pool = BFSBounding()
    assert len(pool) == 0
    assert pool.empty
    with pytest.raises(IndexError):
        pool.pop(1)


def test_dfs_push_pop_lifo_order():
    pool = DFSBounding()
    pool.push(_make_single_row(1.0))
    pool.push(_make_single_row(2.0))
    pool.push(_make_single_row(3.0))

    assert len(pool) == 3
    assert not pool.empty

    out1 = pool.pop(1)
    out2 = pool.pop(1)
    out3 = pool.pop(1)

    assert out1.lb[0, 0].item() == 3.0
    assert out2.lb[0, 0].item() == 2.0
    assert out3.lb[0, 0].item() == 1.0
    assert pool.empty


def test_dfs_pop_larger_than_single_batch_preserves_intra_batch_order():
    pool = DFSBounding()
    pool.push(_make_multi_row([10.0, 20.0, 30.0, 40.0]))

    assert len(pool) == 4

    first = pool.pop(2)
    assert first.batch_size == 2
    assert first.lb[:, 0].tolist() == [10.0, 20.0]
    assert len(pool) == 2

    second = pool.pop(2)
    assert second.batch_size == 2
    assert second.lb[:, 0].tolist() == [30.0, 40.0]
    assert pool.empty


def test_dfs_pop_spans_multiple_batches_lifo_across():
    pool = DFSBounding()
    pool.push(_make_multi_row([1.0, 2.0, 3.0]))
    pool.push(_make_multi_row([4.0, 5.0]))

    assert len(pool) == 5

    got = pool.pop(4)
    assert got.batch_size == 4
    assert got.lb[:, 0].tolist() == [4.0, 5.0, 1.0, 2.0]
    assert got.eta is not None
    assert got.eta.val[1].squeeze(-1).tolist() == [4.0, 5.0, 1.0, 2.0]
    assert got.histories is not None and len(got.histories) == 4

    assert len(pool) == 1
    remainder = pool.pop(10)
    assert remainder.batch_size == 1
    assert remainder.lb[0, 0].item() == 3.0
    assert pool.empty


def test_dfs_empty():
    pool = DFSBounding()
    assert len(pool) == 0
    assert pool.empty
    with pytest.raises(IndexError):
        pool.pop(1)


def test_dfs_sibling_pair_dive_deep():
    """BaB idiom: push(left); push(right) - right is popped first (depth-first into right subtree)."""
    pool = DFSBounding()
    left = _make_multi_row([100.0, 200.0])
    right = _make_multi_row([300.0, 400.0])
    pool.push(left)
    pool.push(right)

    got_first = pool.pop(2)
    assert got_first.lb[:, 0].tolist() == [300.0, 400.0]

    got_second = pool.pop(2)
    assert got_second.lb[:, 0].tolist() == [100.0, 200.0]


def test_random_bounding_backward_compat():
    torch.manual_seed(0)

    pool = RandomBounding()
    bounds1 = Bounds(lb=torch.tensor([-1.0, -1.0]), ub=torch.tensor([1.0, 1.0]))
    pool.push(SubproblemBatch.from_bounds(bounds1, depth=0))
    bounds2 = Bounds(lb=torch.tensor([0.0, 0.0]), ub=torch.tensor([2.0, 2.0]))
    pool.push(SubproblemBatch.from_bounds(bounds2, depth=1))

    assert len(pool) == 2
    picked = pool.pop(1)
    assert picked.batch_size == 1
    assert len(pool) == 1
    remaining = pool.pop(1)
    assert remaining.batch_size == 1
    assert pool.empty
