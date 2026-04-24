from __future__ import annotations

import pytest
import torch

from act.back_end.patches import Patches


def test_patches_construct_default() -> None:
    patches = Patches()
    assert patches.patches is None
    assert patches.stride == 1
    assert patches.padding == 0
    assert patches.identity == 0
    assert patches.inserted_zeros == 0


def test_patches_construct_with_tensor() -> None:
    tensor = torch.randn(2, 1, 3, 3, 1, 3, 3)
    patches = Patches(patches=tensor, stride=2, padding=1, shape=tuple(tensor.shape))
    assert patches.patches is tensor
    assert patches.shape == tuple(tensor.shape)
    assert not patches.is_identity


def test_patches_rejects_dilation() -> None:
    with pytest.raises(ValueError, match="inserted_zeros=0"):
        _ = Patches(inserted_zeros=1)


def test_patches_identity_mutex() -> None:
    tensor = torch.randn(1, 1, 1, 1, 1, 1, 1)
    with pytest.raises(ValueError, match="mutually exclusive"):
        _ = Patches(patches=tensor, identity=1)


def test_patches_is_identity_property() -> None:
    assert Patches(identity=1).is_identity is True
