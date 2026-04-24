#===- act/back_end/test_analyze.py - Tests for analyze() -----------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#

# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnusedVariable=false, reportUnusedCallResult=false

"""Regression tests for multi-predecessor merge behavior in analyze()."""

from unittest.mock import patch

import pytest
import torch

from act.back_end.analyze import analyze
from act.back_end.core import Bounds, ConSet, Fact, Layer, Net
from act.back_end.layer_schema import LayerKind


def _make_bounds(shape: tuple[int, ...]) -> Bounds:
    """Create simple finite bounds with the requested shape."""
    return Bounds(
        lb=torch.zeros(shape, dtype=torch.float32),
        ub=torch.ones(shape, dtype=torch.float32),
    )


def _make_fact(shape: tuple[int, ...]) -> Fact:
    """Create a fact with empty constraints."""
    return Fact(bounds=_make_bounds(shape), cons=ConSet())


def _passthrough_fact(before: dict[int, Fact], layer_id: int) -> Fact:
    """Clone the current input fact for layers whose TF is irrelevant here."""
    bounds = before[layer_id].bounds
    return Fact(
        bounds=Bounds(lb=bounds.lb.clone(), ub=bounds.ub.clone()),
        cons=ConSet(),
    )


def _build_test_net(
    merge_kind: str,
    merge_params: dict[str, int],
    branch_a_vars: int,
    branch_b_vars: int,
    merge_vars: int,
) -> tuple[Net, int, int, int, int]:
    """Build a minimal wrapper-valid DAG with one two-input merge."""
    input_id = 0
    branch_a_id = 1
    branch_b_id = 2
    merge_id = 3
    input_spec_id = 4
    assert_id = 5

    input_vars = list(range(4))
    branch_a_out = list(range(10, 10 + branch_a_vars))
    branch_b_out = list(range(1000, 1000 + branch_b_vars))
    merge_out = list(range(2000, 2000 + merge_vars))

    layers = [
        Layer(
            id=input_id,
            kind=LayerKind.INPUT.value,
            params={
                "shape": [4],
                "dtype": "float32",
                "num_classes": 1,
                "value_range": [0.0, 1.0],
            },
            in_vars=input_vars,
            out_vars=input_vars,
        ),
        Layer(
            id=branch_a_id,
            kind=LayerKind.RELU.value,
            params={},
            in_vars=input_vars,
            out_vars=branch_a_out,
        ),
        Layer(
            id=branch_b_id,
            kind=LayerKind.RELU.value,
            params={},
            in_vars=input_vars,
            out_vars=branch_b_out,
        ),
        Layer(
            id=merge_id,
            kind=merge_kind,
            params=merge_params,
            in_vars=branch_a_out + branch_b_out,
            out_vars=merge_out,
        ),
        Layer(
            id=input_spec_id,
            kind=LayerKind.INPUT_SPEC.value,
            params={"kind": "BOX"},
            in_vars=merge_out,
            out_vars=merge_out,
        ),
        Layer(
            id=assert_id,
            kind=LayerKind.ASSERT.value,
            params={"kind": "RANGE"},
            in_vars=merge_out,
            out_vars=merge_out,
        ),
    ]
    net = Net(
        layers=layers,
        preds={
            input_id: [],
            branch_a_id: [input_id],
            branch_b_id: [input_id],
            merge_id: [branch_a_id, branch_b_id],
            input_spec_id: [merge_id],
            assert_id: [input_spec_id],
        },
        succs={
            input_id: [branch_a_id, branch_b_id],
            branch_a_id: [merge_id],
            branch_b_id: [merge_id],
            merge_id: [input_spec_id],
            input_spec_id: [assert_id],
            assert_id: [],
        },
    )
    return net, input_id, branch_a_id, branch_b_id, merge_id


def test_analyze_raises_on_mismatched_non_concat_predecessors() -> None:
    """ADD-family merges with mismatched predecessor shapes should raise."""
    net, input_id, branch_a_id, branch_b_id, _merge_id = _build_test_net(
        merge_kind=LayerKind.ADD.value,
        merge_params={},
        branch_a_vars=4,
        branch_b_vars=8,
        merge_vars=12,
    )

    def mock_dispatch(layer: Layer, before: dict[int, Fact], _after: dict[int, Fact], _net: Net) -> Fact:
        if layer.id == branch_a_id:
            return _make_fact((4,))
        if layer.id == branch_b_id:
            return _make_fact((8,))
        return _passthrough_fact(before, layer.id)

    with patch("act.back_end.analyze.dispatch_tf", side_effect=mock_dispatch):
        with pytest.raises(ValueError, match=r"mismatched shapes.*CONCAT-family"):
            analyze(net, entry_id=input_id, entry_fact=_make_fact((4,)))


def test_analyze_concat_preserves_batch_dim_with_concat_dim_1() -> None:
    """CONCAT should join along concat_dim instead of flattening all inputs."""
    net, input_id, branch_a_id, branch_b_id, concat_id = _build_test_net(
        merge_kind=LayerKind.CONCAT.value,
        merge_params={"concat_dim": 1},
        branch_a_vars=128,
        branch_b_vars=64,
        merge_vars=192,
    )

    def mock_dispatch(layer: Layer, before: dict[int, Fact], _after: dict[int, Fact], _net: Net) -> Fact:
        if layer.id == branch_a_id:
            return _make_fact((2, 64))
        if layer.id == branch_b_id:
            return _make_fact((2, 32))
        return _passthrough_fact(before, layer.id)

    with patch("act.back_end.analyze.dispatch_tf", side_effect=mock_dispatch):
        before, _after, _global_c = analyze(
            net,
            entry_id=input_id,
            entry_fact=_make_fact((4,)),
        )

    expected_shape = torch.Size([2, 96])
    assert before[concat_id].bounds.lb.shape == expected_shape, (
        "analyze() should preserve the batch dimension and concatenate "
        "predecessors along concat_dim=1 instead of flattening them into a "
        "single 1D bound"
    )
    assert before[concat_id].bounds.ub.shape == expected_shape, (
        "analyze() should preserve the batch dimension and concatenate "
        "upper bounds along concat_dim=1 instead of flattening them into a "
        "single 1D bound"
    )
