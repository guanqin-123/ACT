from __future__ import annotations

from dataclasses import dataclass
from typing import override

import torch


def _tensor_shape_or_none(tensor: torch.Tensor | None) -> tuple[int, ...] | None:
    if tensor is None:
        return None
    return tuple(int(dim) for dim in tensor.shape)


@dataclass
class Patches:
    """Minimal sparse-conv scaffold for the Wave 1 dispatcher gate.

    TODO(W2b): Methods (.to_matrix, .clone, .detach, __eq__) and helpers
    (patches_to_matrix, insert_zeros, compute_patches_stride_padding,
    unify_shape, inplace_unfold) to be filled in Wave 2b.
    """

    patches: torch.Tensor | None = None
    stride: int | tuple[int, int] = 1
    padding: int | tuple[int, int] | tuple[int, int, int, int] = 0
    shape: tuple[int, ...] | None = None
    identity: int = 0
    unstable_idx: torch.Tensor | None = None
    output_shape: tuple[int, ...] | None = None
    input_shape: tuple[int, ...] | None = None
    inserted_zeros: int = 0
    output_padding: int | tuple[int, int] | tuple[int, int, int, int] = 0

    def __post_init__(self) -> None:
        if self.inserted_zeros > 0:
            raise ValueError(
                "Patches v1 only supports inserted_zeros=0 (no dilation)."
            )
        if self.identity != 0 and self.patches is not None:
            raise ValueError(
                "Patches.identity != 0 is mutually exclusive with a patches tensor."
            )

    @property
    def is_identity(self) -> bool:
        return self.identity != 0

    @override
    def __repr__(self) -> str:
        return (
            "Patches("
            f"patches_shape={_tensor_shape_or_none(self.patches)}, "
            f"stride={self.stride}, "
            f"padding={self.padding}, "
            f"shape={self.shape}, "
            f"identity={self.identity}, "
            f"unstable_idx_shape={_tensor_shape_or_none(self.unstable_idx)}, "
            f"output_shape={self.output_shape}, "
            f"input_shape={self.input_shape}, "
            f"inserted_zeros={self.inserted_zeros}, "
            f"output_padding={self.output_padding}"
            ")"
        )


__all__ = ["Patches"]
