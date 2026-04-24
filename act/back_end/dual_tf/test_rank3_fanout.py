from __future__ import annotations

import pytest
import torch

from act.back_end.bab.branching.babsr import compute_lA_per_layer
from act.back_end.bab.node import SubproblemBatch
from act.back_end.bounds_dispatch import dispatch_add_forward, dispatch_concat_forward, expand_rank3
from act.back_end.core import Bounds, Layer, Net
from act.back_end.dual_tf import DualTF
from act.back_end.dual_tf.tf_forward import LinearBound, forward_add, forward_concat
from act.back_end.layer_schema import LayerKind
from act.util.device_manager import get_default_device, get_default_dtype, initialize_device


@pytest.fixture(scope="module", autouse=True)
def _cpu_float32() -> None:
    initialize_device("cpu", "float32")


def _t(data: object) -> torch.Tensor:
    return torch.tensor(data, dtype=get_default_dtype(), device=get_default_device())


def _make_linear_bound(batch: int = 2, out_dim: int = 4, input_dim: int = 3) -> LinearBound:
    A_lb = torch.arange(batch * out_dim * input_dim, dtype=get_default_dtype(), device=get_default_device()).reshape(batch, out_dim, input_dim) / 10.0
    A_ub = A_lb + 0.5
    b_lb = torch.arange(batch * out_dim, dtype=get_default_dtype(), device=get_default_device()).reshape(batch, out_dim) / 20.0
    b_ub = b_lb + 0.25
    return LinearBound(A_lb=A_lb, b_lb=b_lb, A_ub=A_ub, b_ub=b_ub)


def _expand_box(box: Bounds, M: int) -> Bounds:
    return expand_rank3({0: box}, M)[0]


def _expand_lin(lin: LinearBound, M: int) -> LinearBound:
    return expand_rank3({0: lin}, M)[0]


def _make_add_case() -> tuple[Layer, list[Bounds], list[LinearBound], list[tuple[torch.Tensor, torch.Tensor]], list[int], bool, torch.device, torch.dtype]:
    layer = Layer(id=30, kind=LayerKind.ADD.value, params={"bias": _t([0.1, -0.1, 0.05])}, in_vars=[], out_vars=[])
    parent_boxes = [
        Bounds(lb=_t([[-0.5, -0.3, 0.0, 0.2], [0.1, -0.4, -0.2, 0.0]]), ub=_t([[0.5, 0.3, 0.4, 0.6], [0.4, 0.1, 0.3, 0.5]])),
        Bounds(lb=_t([[-0.2, 0.0, -0.1, -0.3], [-0.1, -0.2, 0.0, -0.1]]), ub=_t([[0.3, 0.5, 0.2, 0.1], [0.2, 0.4, 0.5, 0.3]])),
    ]
    parent_lins = [_make_linear_bound(), _make_linear_bound()]
    frame = (_t([[-0.2, 0.0, -0.1], [0.1, -0.2, 0.0]]), _t([[0.4, 0.6, 0.5], [0.6, 0.2, 0.4]]))
    return layer, parent_boxes, parent_lins, [frame, frame], [0, 1], False, torch.device("cpu"), get_default_dtype()


def _make_concat_case() -> tuple[Layer, list[Bounds], list[LinearBound], list[tuple[torch.Tensor, torch.Tensor]], list[int], bool, torch.device, torch.dtype]:
    layer = Layer(id=31, kind=LayerKind.CONCAT.value, params={"concat_dim": 1}, in_vars=[], out_vars=[])
    parent_boxes = [
        Bounds(lb=_t([[-0.5, -0.3], [0.1, -0.4]]), ub=_t([[0.5, 0.3], [0.4, 0.1]])),
        Bounds(lb=_t([[-0.2, 0.0], [-0.1, -0.2]]), ub=_t([[0.3, 0.5], [0.2, 0.4]])),
    ]
    parent_lins = [_make_linear_bound(out_dim=2), _make_linear_bound(out_dim=2)]
    frame = (_t([[-0.2, 0.0, -0.1], [0.1, -0.2, 0.0]]), _t([[0.4, 0.6, 0.5], [0.6, 0.2, 0.4]]))
    return layer, parent_boxes, parent_lins, [frame, frame], [0, 1], False, torch.device("cpu"), get_default_dtype()


def _rank3_add_inputs(M: int = 3):
    layer, parent_boxes, parent_lins, parent_frames, preds, post_activation, device, dtype = _make_add_case()
    rank3_boxes = [_expand_box(box, M) for box in parent_boxes]
    rank3_lins = [_expand_lin(lin, M) for lin in parent_lins]
    return layer, rank3_boxes, rank3_lins, parent_frames, preds, post_activation, device, dtype


def _expand_frames(frames: list[tuple[torch.Tensor, torch.Tensor]], M: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    expanded = (
        frames[0][0].unsqueeze(1).expand(-1, M, -1),
        frames[0][1].unsqueeze(1).expand(-1, M, -1),
    )
    return [expanded for _ in frames]


def test_resnet_skip_no_silent_densify() -> None:
    layer, boxes, lins, frames, preds, post_activation, device, dtype = _rank3_add_inputs()
    args = (layer, boxes, lins, _expand_frames(frames, 3), preds, post_activation, device, dtype)
    out, _stored, lin, _frame = dispatch_add_forward(*args)
    assert out.lb.shape == (2, 3, 4)
    assert lin.A_lb.shape == (2, 3, 4, 3)


def test_forward_add_rank3_parity() -> None:
    layer, boxes, lins, frames, preds, post_activation, device, dtype = _rank3_add_inputs()
    rank3 = forward_add(layer, boxes, lins, _expand_frames(frames, 3), preds, post_activation, device, dtype)
    dense_boxes = [Bounds(lb=b.lb.contiguous().reshape(-1, b.lb.shape[-1]), ub=b.ub.contiguous().reshape(-1, b.ub.shape[-1])) for b in boxes]
    dense_lins = [LinearBound(A_lb=lin.A_lb.contiguous().reshape(-1, *lin.A_lb.shape[2:]), b_lb=lin.b_lb.contiguous().reshape(-1, lin.b_lb.shape[-1]), A_ub=lin.A_ub.contiguous().reshape(-1, *lin.A_ub.shape[2:]), b_ub=lin.b_ub.contiguous().reshape(-1, lin.b_ub.shape[-1])) for lin in lins]
    shared_dense_frame = (frames[0][0].repeat_interleave(3, dim=0), frames[0][1].repeat_interleave(3, dim=0))
    dense = forward_add(layer, dense_boxes, dense_lins, [shared_dense_frame, shared_dense_frame], preds, post_activation, device, dtype)
    torch.testing.assert_close(rank3[1].lb, dense[1].lb.reshape(rank3[1].lb.shape))


def test_forward_concat_rank3_shape() -> None:
    layer, parent_boxes, parent_lins, frames, preds, post_activation, device, dtype = _make_concat_case()
    boxes = [_expand_box(box, 3) for box in parent_boxes]
    lins = [_expand_lin(lin, 3) for lin in parent_lins]
    out, _stored, lin, _frame = dispatch_concat_forward(layer, boxes, lins, _expand_frames(frames, 3), preds, post_activation, device, dtype)
    assert out.lb.shape == (2, 3, 4)
    assert lin.A_lb.shape == (2, 3, 4, 3)


def test_forward_concat_rank3_parity() -> None:
    layer, parent_boxes, parent_lins, frames, preds, post_activation, device, dtype = _make_concat_case()
    boxes = [_expand_box(box, 2) for box in parent_boxes]
    lins = [_expand_lin(lin, 2) for lin in parent_lins]
    rank3 = forward_concat(layer, boxes, lins, _expand_frames(frames, 2), preds, post_activation, device, dtype)
    dense_boxes = [Bounds(lb=b.lb.contiguous().reshape(-1, b.lb.shape[-1]), ub=b.ub.contiguous().reshape(-1, b.ub.shape[-1])) for b in boxes]
    dense_lins = [LinearBound(A_lb=lin.A_lb.contiguous().reshape(-1, *lin.A_lb.shape[2:]), b_lb=lin.b_lb.contiguous().reshape(-1, lin.b_lb.shape[-1]), A_ub=lin.A_ub.contiguous().reshape(-1, *lin.A_ub.shape[2:]), b_ub=lin.b_ub.contiguous().reshape(-1, lin.b_ub.shape[-1])) for lin in lins]
    shared_dense_frame = (frames[0][0].repeat_interleave(2, dim=0), frames[0][1].repeat_interleave(2, dim=0))
    dense = forward_concat(layer, dense_boxes, dense_lins, [shared_dense_frame, shared_dense_frame], preds, post_activation, device, dtype)
    torch.testing.assert_close(rank3[1].lb.reshape(-1, rank3[1].lb.shape[-1]), dense[1].lb.reshape(-1, rank3[1].lb.shape[-1]))


def test_add_shape_mismatch_materializes_explicit(caplog: pytest.LogCaptureFixture) -> None:
    layer, parent_boxes, parent_lins, frames, preds, post_activation, device, dtype = _make_add_case()
    rank3_box = _expand_box(parent_boxes[0], 2)
    rank3_lin = _expand_lin(parent_lins[0], 2)
    with caplog.at_level("WARNING"):
        out, _stored, lin, _frame = forward_add(layer, [rank3_box, parent_boxes[1]], [rank3_lin, parent_lins[1]], frames, preds, post_activation, device, dtype)
    assert any("explicit materialize" in record.message for record in caplog.records)
    assert out.lb.dim() == 2
    assert lin.A_lb.dim() == 3


def test_add_M1_still_works() -> None:
    layer, parent_boxes, parent_lins, frames, preds, post_activation, device, dtype = _make_add_case()
    boxes = [_expand_box(box, 1) for box in parent_boxes]
    lins = [_expand_lin(lin, 1) for lin in parent_lins]
    out, _stored, lin, _frame = forward_add(layer, boxes, lins, frames, preds, post_activation, device, dtype)
    assert out.lb.shape == parent_boxes[0].lb.shape
    assert lin.A_lb.shape == parent_lins[0].A_lb.shape


def test_compute_lA_per_layer_rank3_matches_flat_baseline() -> None:
    net = Net(
        layers=[
            Layer(0, LayerKind.INPUT.value, {"shape": (2,), "dtype": "float32", "num_classes": 1, "value_range": (0.0, 1.0)}, [0, 1], [0, 1]),
            Layer(1, LayerKind.INPUT_SPEC.value, {"kind": "BOX"}, [0, 1], [0, 1]),
            Layer(2, LayerKind.DENSE.value, {"in_features": 2, "out_features": 2, "weight": _t([[1.0, 0.0], [0.0, 1.0]]), "bias": _t([0.0, 0.0])}, [0, 1], [10, 11]),
            Layer(3, LayerKind.RELU.value, {}, [10, 11], [10, 11]),
            Layer(4, LayerKind.DENSE.value, {"in_features": 2, "out_features": 1, "weight": _t([[1.0, -1.0]]), "bias": _t([0.0])}, [10, 11], [20]),
            Layer(5, LayerKind.ASSERT.value, {"kind": "RANGE"}, [20], [20]),
        ],
        preds={0: [], 1: [0], 2: [1], 3: [2], 4: [3], 5: [4]},
        succs={0: [1], 1: [2], 2: [3], 3: [4], 4: [5], 5: []},
    )
    bounds = {
        0: Bounds(lb=_t([[-1.0, -1.0], [0.0, 0.0]]), ub=_t([[1.0, 1.0], [1.0, 1.0]])),
        1: Bounds(lb=_t([[-1.0, -1.0], [0.0, 0.0]]), ub=_t([[1.0, 1.0], [1.0, 1.0]])),
        2: Bounds(lb=_t([[-1.0, -1.0], [0.0, 0.0]]), ub=_t([[1.0, 1.0], [1.0, 1.0]])),
        3: Bounds(lb=_t([[0.0, 0.0], [0.0, 0.0]]), ub=_t([[1.0, 1.0], [1.0, 1.0]])),
        4: Bounds(lb=_t([[-1.0], [-1.0]]), ub=_t([[1.0], [1.0]])),
    }
    rank3 = {lid: _expand_box(b, 2) for lid, b in bounds.items()}
    c = _t([[1.0], [1.0], [1.0], [1.0]])
    rank3_lA = compute_lA_per_layer(net, rank3, c, DualTF(), target_layer_ids=[2])
    flat_lA = compute_lA_per_layer(net, {lid: Bounds(lb=b.lb.repeat_interleave(2, dim=0), ub=b.ub.repeat_interleave(2, dim=0)) for lid, b in bounds.items()}, c, DualTF(), target_layer_ids=[2])
    torch.testing.assert_close(rank3_lA[2], flat_lA[2])


def test_subproblem_batch_select_rank3_alpha_propagation() -> None:
    batch = SubproblemBatch(lb=_t([[0.0, 0.0], [1.0, 1.0]]), ub=_t([[1.0, 1.0], [2.0, 2.0]]), depths=torch.tensor([0, 1], dtype=torch.long), alphas={3: _t([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])})
    picked = batch.select(torch.tensor([1], dtype=torch.long))
    assert picked.alphas is not None
    assert picked.alphas[3].shape == (1, 3)
    torch.testing.assert_close(picked.alphas[3], _t([[0.4, 0.5, 0.6]]))


def test_rank3_write_independence() -> None:
    base = torch.arange(12, dtype=get_default_dtype(), device=get_default_device()).reshape(2, 6)
    view = base.unsqueeze(1).expand(-1, 3, -1)
    row0 = view[:, 0, :].contiguous()
    row0[0, 0] = -999.0
    assert row0[0, 0].item() == -999.0
    assert view[0, 1, 0].item() == base[0, 0].item()


def test_subproblem_batch_select_alpha_shape_consistency() -> None:
    batch = SubproblemBatch(lb=_t([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]), ub=_t([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]), depths=torch.tensor([0, 1, 2], dtype=torch.long), alphas={3: _t([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])})
    picked = batch.select(torch.tensor([0, 2], dtype=torch.long))
    assert picked.alphas is not None and picked.alphas[3].shape == (2, 2)


def test_subproblem_batch_select_single_child() -> None:
    batch = SubproblemBatch(lb=_t([[0.0, 0.0]]), ub=_t([[1.0, 1.0]]), depths=torch.tensor([0], dtype=torch.long), alphas={3: _t([[0.1, 0.2]])})
    picked = batch.select(torch.tensor([0], dtype=torch.long))
    assert picked.batch_size == 1
    assert picked.alphas is not None and picked.alphas[3].shape == (1, 2)


def test_dispatch_add_forward_rank3_matches_direct() -> None:
    layer, boxes, lins, frames, preds, post_activation, device, dtype = _rank3_add_inputs(2)
    dispatched = dispatch_add_forward(layer, boxes, lins, _expand_frames(frames, 2), preds, post_activation, device, dtype)
    direct = forward_add(layer, boxes, lins, _expand_frames(frames, 2), preds, post_activation, device, dtype)
    torch.testing.assert_close(dispatched[0].lb, direct[0].lb)
    torch.testing.assert_close(dispatched[2].A_lb, direct[2].A_lb)


def test_dispatch_concat_forward_rank3_matches_direct() -> None:
    layer, parent_boxes, parent_lins, frames, preds, post_activation, device, dtype = _make_concat_case()
    boxes = [_expand_box(box, 2) for box in parent_boxes]
    lins = [_expand_lin(lin, 2) for lin in parent_lins]
    dispatched = dispatch_concat_forward(layer, boxes, lins, _expand_frames(frames, 2), preds, post_activation, device, dtype)
    direct = forward_concat(layer, boxes, lins, _expand_frames(frames, 2), preds, post_activation, device, dtype)
    torch.testing.assert_close(dispatched[0].lb, direct[0].lb)
    torch.testing.assert_close(dispatched[2].A_lb, direct[2].A_lb)


def test_subproblem_batch_select_preserves_parent_margins_and_ids() -> None:
    batch = SubproblemBatch(
        lb=_t([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]),
        ub=_t([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]),
        depths=torch.tensor([0, 1, 2], dtype=torch.long),
        alphas={3: _t([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])},
        parent_margins=_t([-0.1, 0.2, 0.3]),
        subproblem_ids=torch.tensor([10, 11, 12], dtype=torch.long),
    )
    picked = batch.select(torch.tensor([2, 0], dtype=torch.long))
    assert picked.parent_margins is not None
    assert picked.subproblem_ids is not None
    torch.testing.assert_close(picked.parent_margins, _t([0.3, -0.1]))
    assert picked.subproblem_ids.tolist() == [12, 10]


def test_rank3_box_expand_is_zero_stride() -> None:
    box = Bounds(lb=_t([[0.0, 1.0], [2.0, 3.0]]), ub=_t([[1.0, 2.0], [3.0, 4.0]]))
    expanded = _expand_box(box, 4)
    assert expanded.lb.shape == (2, 4, 2)
    assert expanded.lb.stride(1) == 0
    assert expanded.ub.stride(1) == 0
