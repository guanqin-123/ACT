"""Tests for act.back_end.bab.node — SubproblemBatch, Split, and helpers."""
from __future__ import annotations

import pytest
import torch

from act.back_end.bab.eta import EtaState
from act.back_end.bab.node import (
    Split,
    SubproblemBatch,
    _concat_subproblem_batches,
)
from act.back_end.core import Bounds
from act.back_end.solver.alpha_state import AlphaState


def _make_eta(B: int, widths: dict[int, int]) -> EtaState:
    val = {lid: torch.zeros(B, d) for lid, d in widths.items()}
    sign = {lid: torch.zeros(B, d) for lid, d in widths.items()}
    point = {lid: torch.zeros(B, d) for lid, d in widths.items()}
    return EtaState(val=val, sign=sign, point=point)


def test_subproblem_batch_backward_compat():
    lb = torch.tensor([[-1.0, -1.0], [0.0, 0.0]])
    ub = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    depths = torch.tensor([0, 1], dtype=torch.long)

    batch = SubproblemBatch(lb=lb, ub=ub, depths=depths)

    assert batch.batch_size == 2
    assert batch.input_dim == 2
    assert batch.eta is None
    assert batch.histories is None
    assert batch.parent_margins is None

    bounds = Bounds(lb=torch.tensor([-1.0, -1.0]), ub=torch.tensor([1.0, 1.0]))
    root = SubproblemBatch.from_bounds(bounds, depth=0)
    assert root.batch_size == 1
    assert root.eta is None
    assert root.histories is None


def test_subproblem_batch_select_tensors_and_lists():
    B, D = 4, 3
    lb = torch.arange(B * D, dtype=torch.float32).reshape(B, D)
    ub = lb + 1.0
    depths = torch.arange(B, dtype=torch.long)

    eta = _make_eta(B, {7: 5})
    for b in range(B):
        eta.sign[7][b, 0] = float(b + 10)

    histories: list[list[Split]] = [
        [Split(layer_id=7, neuron_idx=0, sign=+1, split_point=0.0, kind="relu")],
        [],
        [
            Split(layer_id=7, neuron_idx=1, sign=-1, split_point=0.0, kind="relu"),
            Split(layer_id=7, neuron_idx=2, sign=+1, split_point=0.0, kind="relu"),
        ],
        [],
    ]
    parent_margins = torch.tensor([0.1, 0.2, 0.3, 0.4])

    batch = SubproblemBatch(
        lb=lb,
        ub=ub,
        depths=depths,
        eta=eta,
        histories=histories,
        parent_margins=parent_margins,
    )

    idx = torch.tensor([0, 2], dtype=torch.long)
    picked = batch.select(idx)

    assert picked.batch_size == 2
    assert torch.equal(picked.lb, lb.index_select(0, idx))
    assert torch.equal(picked.ub, ub.index_select(0, idx))
    assert torch.equal(picked.depths, depths.index_select(0, idx))

    assert picked.eta is not None
    assert picked.eta.batch_size == 2
    assert picked.eta.sign[7][:, 0].tolist() == [10.0, 12.0]

    assert picked.histories is not None
    assert len(picked.histories) == 2
    assert picked.histories[0] is histories[0]
    assert picked.histories[1] is histories[2]

    assert picked.parent_margins is not None
    assert picked.parent_margins.tolist() == pytest.approx([0.1, 0.3])


def test_split_neuron_batched_relu_left_right_signs():
    bounds = Bounds(lb=torch.tensor([-1.0, -1.0]), ub=torch.tensor([1.0, 1.0]))
    root = SubproblemBatch.from_bounds_with_eta(bounds, layer_widths={1: 4})

    left, right = root.split_neuron_batched([(1, 2, "relu")])

    assert left.eta.sign[1][0, 2].item() == +1.0
    assert right.eta.sign[1][0, 2].item() == -1.0
    assert left.eta.point[1][0, 2].item() == 0.0
    assert right.eta.point[1][0, 2].item() == 0.0
    assert left.eta.val[1][0, 2].item() == 0.0
    assert right.eta.val[1][0, 2].item() == 0.0

    # Other slots untouched.
    for neuron in (0, 1, 3):
        assert left.eta.sign[1][0, neuron].item() == 0.0
        assert right.eta.sign[1][0, neuron].item() == 0.0

    assert left.depths.item() == 1
    assert right.depths.item() == 1
    assert torch.equal(left.lb, root.lb)
    assert torch.equal(left.ub, root.ub)

    assert left.histories is not None and len(left.histories) == 1
    assert left.histories[0][-1].sign == +1
    assert right.histories[0][-1].sign == -1


def test_split_neuron_batched_relu_warm_start_preserves_prior():
    bounds = Bounds(lb=torch.tensor([-1.0, -1.0]), ub=torch.tensor([1.0, 1.0]))
    root = SubproblemBatch.from_bounds_with_eta(bounds, layer_widths={1: 8})

    # Pretend neuron 5 was previously split INACTIVE with a warm eta value.
    root.eta.sign[1][0, 5] = +1.0
    root.eta.point[1][0, 5] = 0.0
    root.eta.val[1][0, 5] = 0.42

    left, right = root.split_neuron_batched([(1, 2, "relu")])

    # Prior split (neuron 5) is preserved in both children.
    for child in (left, right):
        assert child.eta.sign[1][0, 5].item() == +1.0
        assert child.eta.point[1][0, 5].item() == 0.0
        assert child.eta.val[1][0, 5].item() == pytest.approx(0.42)

    # New split (neuron 2) takes the canonical left/right signs.
    assert left.eta.sign[1][0, 2].item() == +1.0
    assert right.eta.sign[1][0, 2].item() == -1.0
    assert left.eta.val[1][0, 2].item() == 0.0
    assert right.eta.val[1][0, 2].item() == 0.0


def test_split_neuron_batched_smooth_uses_provided_points():
    lb = torch.tensor([[-1.0, -1.0], [-1.0, -1.0]])
    ub = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    depths = torch.tensor([0, 0], dtype=torch.long)
    eta = _make_eta(B=2, widths={3: 4})
    batch = SubproblemBatch(
        lb=lb,
        ub=ub,
        depths=depths,
        eta=eta,
        histories=[[], []],
    )

    decisions = [(3, 1, "smooth"), (3, 2, "smooth")]
    per_row = torch.tensor([0.0, 0.3])
    left, right = batch.split_neuron_batched(decisions, per_row_split_points=per_row)

    assert left.eta.point[3][0, 1].item() == pytest.approx(0.0)
    assert right.eta.point[3][0, 1].item() == pytest.approx(0.0)
    assert left.eta.point[3][1, 2].item() == pytest.approx(0.3)
    assert right.eta.point[3][1, 2].item() == pytest.approx(0.3)

    assert left.eta.sign[3][0, 1].item() == +1.0
    assert right.eta.sign[3][0, 1].item() == -1.0
    assert left.eta.sign[3][1, 2].item() == +1.0
    assert right.eta.sign[3][1, 2].item() == -1.0


def test_split_neuron_batched_relu_rejects_double_split():
    bounds = Bounds(lb=torch.tensor([-1.0, -1.0]), ub=torch.tensor([1.0, 1.0]))
    root = SubproblemBatch.from_bounds_with_eta(bounds, layer_widths={1: 4})

    left, _right = root.split_neuron_batched([(1, 2, "relu")])

    with pytest.raises(AssertionError, match="double-split"):
        left.split_neuron_batched([(1, 2, "relu")])


def test_split_neuron_batched_smooth_allows_double_split():
    bounds = Bounds(lb=torch.tensor([-1.0, -1.0]), ub=torch.tensor([1.0, 1.0]))
    root = SubproblemBatch.from_bounds_with_eta(bounds, layer_widths={3: 4})

    left, _right = root.split_neuron_batched(
        [(3, 2, "smooth")],
        per_row_split_points=torch.tensor([0.5]),
    )
    # Second split on the same neuron of the same child must NOT raise.
    left2, right2 = left.split_neuron_batched(
        [(3, 2, "smooth")],
        per_row_split_points=torch.tensor([0.25]),
    )
    # Overwrite semantics: the new point supersedes the old one.
    assert left2.eta.point[3][0, 2].item() == pytest.approx(0.25)
    assert right2.eta.point[3][0, 2].item() == pytest.approx(0.25)
    assert left2.eta.sign[3][0, 2].item() == +1.0
    assert right2.eta.sign[3][0, 2].item() == -1.0


def test_from_bounds_with_eta_zero_init():
    bounds = Bounds(lb=torch.tensor([-1.0, 0.0, 1.0]), ub=torch.tensor([1.0, 2.0, 3.0]))
    batch = SubproblemBatch.from_bounds_with_eta(
        bounds, layer_widths={1: 5, 3: 7}, depth=0
    )

    assert batch.batch_size == 1
    assert batch.input_dim == 3
    assert batch.depths.tolist() == [0]
    assert batch.eta is not None
    assert set(batch.eta.layer_ids) == {1, 3}

    for lid, width in {1: 5, 3: 7}.items():
        assert batch.eta.val[lid].shape == (1, width)
        assert batch.eta.sign[lid].shape == (1, width)
        assert batch.eta.point[lid].shape == (1, width)
        assert torch.all(batch.eta.val[lid] == 0)
        assert torch.all(batch.eta.sign[lid] == 0)
        assert torch.all(batch.eta.point[lid] == 0)

    assert batch.eta.is_empty()
    assert batch.eta.fast_path_skip()
    assert batch.histories == [[]]
    assert batch.parent_margins is None


def test_concat_subproblem_batches_preserves_all_fields():
    # Build two size-2 batches with eta/histories/parent_margins; concat to size-4.
    lb1 = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    ub1 = lb1 + 1.0
    d1 = torch.tensor([0, 0], dtype=torch.long)
    eta1 = _make_eta(B=2, widths={5: 3})
    eta1.sign[5][0, 0] = 1.0
    h1 = [[Split(5, 0, +1, 0.0, "relu")], []]
    pm1 = torch.tensor([0.5, 0.7])
    b1 = SubproblemBatch(
        lb=lb1, ub=ub1, depths=d1, eta=eta1, histories=h1, parent_margins=pm1
    )

    lb2 = torch.tensor([[2.0, 2.0], [3.0, 3.0]])
    ub2 = lb2 + 1.0
    d2 = torch.tensor([1, 1], dtype=torch.long)
    eta2 = _make_eta(B=2, widths={5: 3})
    eta2.sign[5][1, 1] = -1.0
    h2 = [[], [Split(5, 1, -1, 0.0, "relu")]]
    pm2 = torch.tensor([0.9, 1.1])
    b2 = SubproblemBatch(
        lb=lb2, ub=ub2, depths=d2, eta=eta2, histories=h2, parent_margins=pm2
    )

    merged = _concat_subproblem_batches(b1, b2)

    assert merged.batch_size == 4
    assert torch.equal(merged.lb, torch.cat([lb1, lb2], dim=0))
    assert merged.depths.tolist() == [0, 0, 1, 1]
    assert merged.eta is not None
    assert merged.eta.sign[5][0, 0].item() == 1.0
    assert merged.eta.sign[5][3, 1].item() == -1.0
    assert merged.histories is not None
    assert len(merged.histories) == 4
    assert merged.histories[0][0].sign == +1
    assert merged.histories[3][0].sign == -1
    assert merged.parent_margins is not None
    assert merged.parent_margins.tolist() == pytest.approx([0.5, 0.7, 0.9, 1.1])


def test_concat_subproblem_batches_preserves_alphas():
    lb1 = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    ub1 = lb1 + 1.0
    d1 = torch.tensor([0, 0], dtype=torch.long)
    a1 = AlphaState()
    a1.set(3, AlphaState.FINAL_SID, torch.tensor([[0.1, 0.2], [0.3, 0.4]]))
    a1.set(3, 4, torch.tensor([[0.5, 0.5], [0.6, 0.6]]))
    b1 = SubproblemBatch(lb=lb1, ub=ub1, depths=d1, alphas=a1)

    lb2 = torch.tensor([[2.0, 2.0]])
    ub2 = lb2 + 1.0
    d2 = torch.tensor([1], dtype=torch.long)
    a2 = AlphaState()
    a2.set(3, AlphaState.FINAL_SID, torch.tensor([[0.7, 0.8]]))
    a2.set(3, 4, torch.tensor([[0.9, 0.9]]))
    b2 = SubproblemBatch(lb=lb2, ub=ub2, depths=d2, alphas=a2)

    merged = _concat_subproblem_batches(b1, b2)

    assert merged.alphas is not None
    final_t = merged.alphas.get(3, AlphaState.FINAL_SID)
    inter_t = merged.alphas.get(3, 4)
    assert final_t is not None and inter_t is not None
    assert final_t.shape == (3, 2)
    assert torch.equal(final_t, torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.7, 0.8]]))
    assert torch.equal(inter_t, torch.tensor([[0.5, 0.5], [0.6, 0.6], [0.9, 0.9]]))


def test_concat_subproblem_batches_drops_alphas_when_keysets_disagree():
    lb1 = torch.tensor([[0.0]])
    ub1 = torch.tensor([[1.0]])
    d1 = torch.tensor([0], dtype=torch.long)
    a1 = AlphaState()
    a1.set(3, AlphaState.FINAL_SID, torch.tensor([[0.1]]))
    b1 = SubproblemBatch(lb=lb1, ub=ub1, depths=d1, alphas=a1)

    a2 = AlphaState()
    a2.set(3, AlphaState.FINAL_SID, torch.tensor([[0.2]]))
    a2.set(5, 6, torch.tensor([[0.3]]))
    b2 = SubproblemBatch(lb=lb1, ub=ub1, depths=d1, alphas=a2)

    merged = _concat_subproblem_batches(b1, b2)
    assert merged.alphas is None


def test_concat_subproblem_batches_drops_alphas_when_one_missing():
    lb1 = torch.tensor([[0.0]])
    ub1 = torch.tensor([[1.0]])
    d1 = torch.tensor([0], dtype=torch.long)
    a1 = AlphaState()
    a1.set(3, AlphaState.FINAL_SID, torch.tensor([[0.1]]))
    b1 = SubproblemBatch(lb=lb1, ub=ub1, depths=d1, alphas=a1)
    b2 = SubproblemBatch(lb=lb1, ub=ub1, depths=d1, alphas=None)

    merged = _concat_subproblem_batches(b1, b2)
    assert merged.alphas is None


def test_concat_subproblem_batches_preserves_eta_per_spec_flag_and_3d_val():
    """Tier 6 invariant (mirror of Tier 4 fix at node.py:531).

    EtaState now carries a per_spec flag (Tier 6 Phase 1). Sibling-batch
    concat MUST preserve it; otherwise downstream Adam routes per-spec val
    through legacy 2-D expansion and corrupts shape (same bug class as the
    Tier 4 _concat AlphaState flag drop).
    """
    from act.back_end.bab.eta import EtaState

    M, D = 3, 5
    lb1 = torch.tensor([[0.0, 0.0]])
    ub1 = lb1 + 1.0
    d1 = torch.tensor([0], dtype=torch.long)
    eta1 = EtaState(
        val={7: torch.full((1, M, D), 0.1)},
        sign={7: torch.zeros(1, D)},
        point={7: torch.zeros(1, D)},
        per_spec=True,
    )
    b1 = SubproblemBatch(lb=lb1, ub=ub1, depths=d1, eta=eta1)

    lb2 = torch.tensor([[2.0, 2.0]])
    ub2 = lb2 + 1.0
    d2 = torch.tensor([1], dtype=torch.long)
    eta2 = EtaState(
        val={7: torch.full((1, M, D), 0.7)},
        sign={7: torch.zeros(1, D)},
        point={7: torch.zeros(1, D)},
        per_spec=True,
    )
    b2 = SubproblemBatch(lb=lb2, ub=ub2, depths=d2, eta=eta2)

    merged = _concat_subproblem_batches(b1, b2)

    assert merged.eta is not None
    assert merged.eta.per_spec is True, (
        "regression: per_spec flag was dropped during eta concat; downstream "
        "evaluate_spec will mis-handle per-spec val through legacy expansion"
    )
    assert merged.eta.val[7].shape == (2, M, D), f"val shape regression: {merged.eta.val[7].shape}"
    assert merged.eta.sign[7].shape == (2, D), f"sign shape regression: {merged.eta.sign[7].shape}"


def test_concat_subproblem_batches_rejects_eta_per_spec_flag_mismatch():
    """Mixing per_spec=True and per_spec=False eta states must raise."""
    from act.back_end.bab.eta import EtaState

    lb = torch.tensor([[0.0, 0.0]])
    ub = lb + 1.0
    d = torch.tensor([0], dtype=torch.long)
    eta_off = EtaState(
        val={7: torch.zeros(1, 4)},
        sign={7: torch.zeros(1, 4)},
        point={7: torch.zeros(1, 4)},
        per_spec=False,
    )
    eta_on = EtaState(
        val={7: torch.zeros(1, 3, 4)},
        sign={7: torch.zeros(1, 4)},
        point={7: torch.zeros(1, 4)},
        per_spec=True,
    )
    b1 = SubproblemBatch(lb=lb, ub=ub, depths=d, eta=eta_off)
    b2 = SubproblemBatch(lb=lb, ub=ub, depths=d, eta=eta_on)

    with pytest.raises(ValueError, match=r"eta per_spec flag mismatch"):
        _concat_subproblem_batches(b1, b2)


def test_concat_subproblem_batches_preserves_per_spec_flag_and_3d_final_sid():
    """Tier 4 regression: per_spec=True must survive sibling-batch concat.

    Previously, ``_concat_subproblem_batches`` rebuilt the merged AlphaState
    via ``AlphaState()`` (per_spec=False default), silently downgrading the
    contract. The next ``evaluate_spec`` then routed the still-3-D FINAL_SID
    tensor through the legacy expand path, producing 4-D garbage and tripping
    the warm-alpha-width check. See HANDOFF_tier4_session4 fix-up notes.
    """
    M, D = 3, 5
    lb1 = torch.tensor([[0.0, 0.0]])
    ub1 = lb1 + 1.0
    d1 = torch.tensor([0], dtype=torch.long)
    a1 = AlphaState(per_spec=True)
    a1.set(3, AlphaState.FINAL_SID, torch.full((1, M, D), 0.1))
    a1.set(3, 4, torch.full((1, D), 0.5))
    b1 = SubproblemBatch(lb=lb1, ub=ub1, depths=d1, alphas=a1)

    lb2 = torch.tensor([[2.0, 2.0]])
    ub2 = lb2 + 1.0
    d2 = torch.tensor([1], dtype=torch.long)
    a2 = AlphaState(per_spec=True)
    a2.set(3, AlphaState.FINAL_SID, torch.full((1, M, D), 0.7))
    a2.set(3, 4, torch.full((1, D), 0.9))
    b2 = SubproblemBatch(lb=lb2, ub=ub2, depths=d2, alphas=a2)

    merged = _concat_subproblem_batches(b1, b2)

    assert merged.alphas is not None
    assert merged.alphas.per_spec is True, (
        "regression: per_spec flag was dropped during concat; downstream "
        "evaluate_spec will mis-route per-spec FINAL_SID through legacy expansion"
    )
    final_t = merged.alphas.get(3, AlphaState.FINAL_SID)
    inter_t = merged.alphas.get(3, 4)
    assert final_t is not None and inter_t is not None
    assert final_t.shape == (2, M, D), f"FINAL_SID shape regression: {final_t.shape}"
    assert inter_t.shape == (2, D), f"intermediate shape regression: {inter_t.shape}"
