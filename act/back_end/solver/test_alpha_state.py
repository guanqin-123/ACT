# pyright: reportMissingImports=false

from __future__ import annotations

import torch

from act.back_end.solver.alpha_state import AlphaState
from act.util.device_manager import (
    get_default_device,
    get_default_dtype,
    initialize_device,
)


def setup_module() -> None:
    initialize_device("cpu", "float64")


def _t(data: object, *, requires_grad: bool = False) -> torch.Tensor:
    return torch.tensor(
        data,
        dtype=get_default_dtype(),
        device=get_default_device(),
        requires_grad=requires_grad,
    )


def test_alpha_state_legacy_roundtrip() -> None:
    t1 = _t([[0.1, 0.2], [0.3, 0.4]])
    t2 = _t([[0.5], [0.6]])

    state = AlphaState.from_legacy({1: t1, 2: t2})
    legacy = state.to_legacy()

    torch.testing.assert_close(legacy[1], t1)
    torch.testing.assert_close(legacy[2], t2)


def test_alpha_state_for_start_node_isolation() -> None:
    final_t = _t([[0.1, 0.2], [0.3, 0.4]])
    intermediate_t = _t([[0.9, 0.8], [0.7, 0.6]])
    state = AlphaState()
    state.set(3, AlphaState.FINAL_SID, final_t)
    state.set(3, 11, intermediate_t)
    state.set(5, 11, _t([[0.4], [0.2]]))

    final_only = state.for_start_node(AlphaState.FINAL_SID)

    assert set(final_only) == {3}
    torch.testing.assert_close(final_only[3], final_t)


def test_alpha_state_clone_independence() -> None:
    original = AlphaState.from_legacy({1: _t([[0.1, 0.2], [0.3, 0.4]])})
    cloned = original.clone()

    original_tensor = original.get(1, AlphaState.FINAL_SID)
    cloned_tensor = cloned.get(1, AlphaState.FINAL_SID)
    assert original_tensor is not None
    assert cloned_tensor is not None

    cloned_tensor[0, 0] = 9.0

    assert original_tensor[0, 0].item() == 0.1
    assert cloned_tensor[0, 0].item() == 9.0


def test_alpha_state_select_batch_axis() -> None:
    state = AlphaState()
    state.set(
        3,
        AlphaState.FINAL_SID,
        _t(
            [
                [0.0, 1.0],
                [2.0, 3.0],
                [4.0, 5.0],
                [6.0, 7.0],
                [8.0, 9.0],
                [10.0, 11.0],
                [12.0, 13.0],
                [14.0, 15.0],
            ]
        ),
    )
    state.set(3, 7, _t([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]]))
    state.set(
        5,
        AlphaState.FINAL_SID,
        _t([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0]]),
    )

    idx = torch.tensor([0, 2, 4, 6], dtype=torch.long, device=get_default_device())
    selected = state.select(idx)

    for lid, by_sid in selected._store.items():
        for sid, tensor in by_sid.items():
            assert tensor.shape[0] == 4, (lid, sid, tensor.shape)


def test_alpha_state_empty_is_legacy_only() -> None:
    assert AlphaState().is_legacy_only() is True


def test_alpha_state_flat_params_preserves_grad() -> None:
    t1 = _t([[0.1, 0.2]], requires_grad=True)
    t2 = _t([[0.3]], requires_grad=True)

    state = AlphaState()
    state.set(1, AlphaState.FINAL_SID, t1)
    state.set(2, 5, t2)

    flat = state.flat_params()

    assert flat[0] is t1
    assert flat[1] is t2
