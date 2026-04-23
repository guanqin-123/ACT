"""Bound-trajectory recording for BaB subproblems.

Opt-in instrumentation: caller passes a ``BoundTrace`` to ``verify_bab(trace=...)``.
When present, each BaB iteration records the current per-subproblem min_slack.
Additionally, each Adam step inside dual bound optimization can record the
current best objective for every subproblem row.
Default (``trace=None``) keeps the existing fast path unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class BoundTrace:
    """Per-subproblem bound trajectory across BaB iterations.

    Subproblem IDs are assigned at creation:
      - root gets id 0
      - each split spawns two new ids (left, right)
    Children record their parent id for lineage reconstruction.
    """

    min_slack_history: dict[int, list[float]] = field(default_factory=dict)
    iteration_history: dict[int, list[int]] = field(default_factory=dict)
    parent: dict[int, int] = field(default_factory=dict)
    depth: dict[int, int] = field(default_factory=dict)
    adam_trajectory: Dict[Tuple[int, int], List[float]] = field(default_factory=dict)
    next_id: int = 0

    def new_id(self, parent: int | None = None, depth: int = 0) -> int:
        sid = self.next_id
        self.next_id += 1
        self.min_slack_history[sid] = []
        self.iteration_history[sid] = []
        if parent is not None:
            self.parent[sid] = parent
        self.depth[sid] = depth
        return sid

    def record(self, sid: int, t: int, slack: float) -> None:
        if sid not in self.min_slack_history:
            raise KeyError(f"unregistered subproblem id {sid}; call new_id first")
        self.min_slack_history[sid].append(float(slack))
        self.iteration_history[sid].append(int(t))

    def record_adam_step(self, sid: int, bab_iter: int, obj_val: float) -> None:
        if sid not in self.min_slack_history:
            raise KeyError(f"unregistered subproblem id {sid}; call new_id first")
        key = (int(sid), int(bab_iter))
        self.adam_trajectory.setdefault(key, []).append(float(obj_val))

    def to_dict(self) -> dict[str, dict[str, object]]:
        return {
            "min_slack_history": {str(k): v for k, v in self.min_slack_history.items()},
            "iteration_history": {str(k): v for k, v in self.iteration_history.items()},
            "parent": {str(k): v for k, v in self.parent.items()},
            "depth": {str(k): v for k, v in self.depth.items()},
            "adam_trajectory": {
                f"{sid},{bab_iter}": v
                for (sid, bab_iter), v in self.adam_trajectory.items()
            },
        }
