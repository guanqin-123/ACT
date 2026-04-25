from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Dict, Iterator, List, Optional, Tuple

import torch


@dataclass
class AlphaState:
    """Typed α-store keyed by (layer_id, start_node_id).

    start_node_id = -1 (FINAL_SID) means "α used in the backward pass to the
    final spec margin" (today's only path). Intermediate start_node_ids are
    layer ids of pre-activation layers (added in Phase 2a).
    """

    FINAL_SID: ClassVar[int] = -1

    # _store[lid_relu][sid] -> Tensor[B, D_layer]
    _store: Dict[int, Dict[int, torch.Tensor]] = field(default_factory=dict)

    @property
    def start_nodes(self) -> Tuple[int, ...]:
        """Sorted unique sids across all ReLUs."""
        return tuple(sorted({sid for by_sid in self._store.values() for sid in by_sid}))

    def get(self, lid: int, sid: int) -> Optional[torch.Tensor]:
        return self._store.get(lid, {}).get(sid)

    def set(self, lid: int, sid: int, tensor: torch.Tensor) -> None:
        self._store.setdefault(lid, {})[sid] = tensor

    def for_start_node(self, sid: int) -> Dict[int, torch.Tensor]:
        """Project to legacy {lid: tensor} dict for one start_node."""
        return {
            lid: by_sid[sid]
            for lid, by_sid in self._store.items()
            if sid in by_sid
        }

    def flat_params(self) -> List[torch.Tensor]:
        """All tensors as a flat list, for Adam.parameters()."""
        params: List[torch.Tensor] = []
        for lid in sorted(self._store):
            for sid in sorted(self._store[lid]):
                params.append(self._store[lid][sid])
        return params

    def clone(self, *, detach: bool = True) -> "AlphaState":
        """Independent deep clone (for snapshot or BaB warm-start)."""
        cloned = AlphaState()
        for lid, by_sid in self._store.items():
            for sid, tensor in by_sid.items():
                copied = tensor.detach().clone() if detach else tensor.clone()
                cloned.set(lid, sid, copied)
        return cloned

    def to(self, device: torch.device, dtype: torch.dtype) -> "AlphaState":
        moved = AlphaState()
        for lid, by_sid in self._store.items():
            for sid, tensor in by_sid.items():
                moved.set(lid, sid, tensor.to(device=device, dtype=dtype))
        return moved

    def select(self, idx: torch.Tensor) -> "AlphaState":
        """Index along batch axis [B] -> [|idx|]; mirrors SubproblemBatch.select."""
        if idx.dim() != 1:
            raise ValueError(f"select: idx must be 1-D, got shape {tuple(idx.shape)}")
        selected = AlphaState()
        for lid, by_sid in self._store.items():
            for sid, tensor in by_sid.items():
                selected.set(lid, sid, tensor.index_select(0, idx))
        return selected

    @classmethod
    def from_legacy(cls, d: Optional[Dict[int, torch.Tensor]]) -> "AlphaState":
        """Wrap a legacy {lid: tensor} dict at FINAL_SID."""
        state = cls()
        if d is None:
            return state
        for lid, tensor in d.items():
            state.set(lid, cls.FINAL_SID, tensor)
        return state

    def to_legacy(self) -> Dict[int, torch.Tensor]:
        """Project FINAL_SID slice back to {lid: tensor}; raises if any non-FINAL_SID present."""
        if not self.is_legacy_only():
            raise ValueError(
                "AlphaState.to_legacy requires all entries to use FINAL_SID only"
            )
        return self.for_start_node(self.FINAL_SID)

    def is_legacy_only(self) -> bool:
        """True iff every entry is at FINAL_SID."""
        return all(sid == self.FINAL_SID for sid in self.start_nodes)

    def is_empty(self) -> bool:
        return not self._store

    def __bool__(self) -> bool:
        return not self.is_empty()

    def __contains__(self, lid: int) -> bool:
        return lid in self.to_legacy() if self.is_legacy_only() else lid in self._store

    def __getitem__(self, lid: int) -> torch.Tensor:
        return self.to_legacy()[lid]

    def __iter__(self) -> Iterator[int]:
        return iter(self.to_legacy())

    def __len__(self) -> int:
        return len(self._store) if not self.is_legacy_only() else len(self.to_legacy())

    def keys(self):
        return self.to_legacy().keys()

    def items(self):
        return self.to_legacy().items()

    def values(self):
        return self.to_legacy().values()
