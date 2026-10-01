"""Lossless frontier state concatenation across alpha cap transitions."""

import pytest
import torch

from act.back_end.bab.branching.bounding import TopKBounding, _concat_tree_nbytes
from act.back_end.bab.node import SubproblemBatch
from act.back_end.solver.solver_dual import _alpha_tree_concat_lanes


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("shared_rows", [False, True])
def test_mixed_alpha_concat_and_planned_bytes(reverse: bool, shared_rows: bool) -> None:
    left = torch.arange(8, dtype=torch.float64).reshape(2, 4) / 8
    if shared_rows:
        left = left[:, None, :]
    right = torch.arange(36, dtype=torch.float64).reshape(3, 3, 4) / 36
    if reverse:
        left, right = right, left
    actual = _alpha_tree_concat_lanes({3: left}, {3: right}, len(left), len(right))
    expected = torch.cat([
        value[:, None, :].expand(-1, 3, -1) if value.dim() == 2
        else value.expand(-1, 3, -1) for value in (left, right)
    ])
    assert isinstance(actual, dict)
    torch.testing.assert_close(actual[3], expected, rtol=0, atol=0)
    assert _concat_tree_nbytes({3: left}, {3: right}, len(left), len(right)) == expected.numel() * expected.element_size()

    def batch(value: torch.Tensor) -> SubproblemBatch:
        return SubproblemBatch(
            torch.zeros(len(value), 2), torch.ones(len(value), 2),
            torch.zeros(len(value), dtype=torch.long), incremental_alpha={3: value},
        )

    pool = TopKBounding()
    pool.push(batch(left))
    projected = pool.projected_push_nbytes(batch(right))
    pool.push(batch(right))
    assert pool.nbytes() == projected
    popped = pool.pop(5)
    assert popped.incremental_alpha is not None
    value = popped.incremental_alpha[3]
    assert isinstance(value, torch.Tensor)
    assert value.shape == expected.shape
    torch.testing.assert_close(value.sum(dim=0), expected.sum(dim=0), rtol=0, atol=1e-15)


def test_mixed_alpha_concat_still_rejects_different_widths() -> None:
    with pytest.raises(ValueError, match="alpha leaf trailing shape mismatch"):
        _alpha_tree_concat_lanes(torch.zeros(2, 4), torch.zeros(3, 3, 5), 2, 3)
