# ===- act/back_end/bab/__init__.py - BaB Package -------------------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------====#

from act.back_end.config import BaBConfig
from act.back_end.bab.node import BabNode, Split, SubproblemBatch, split_subproblems
from act.back_end.bab.eta import EtaState, expand_eta_state, get_pre_activation_layer_id
from act.back_end.bab.branching.branching import BranchingStrategy
from act.back_end.bab.branching.bounding import BoundingStrategy, BFSBounding
from act.back_end.bab.trace import BoundTrace

__all__ = [
    "BaBConfig",
    "verify_bab",
    "BabNode",
    "Split",
    "SubproblemBatch",
    "split_subproblems",
    "EtaState",
    "expand_eta_state",
    "get_pre_activation_layer_id",
    "check_violation_at_point",
    "check_violation_at_point_batched",
    "BranchingStrategy",
    "BoundingStrategy",
    "BFSBounding",
    "BaBSRBranching",
    "BoundTrace",
]


def __getattr__(name: str):
    if name in {"verify_bab", "check_violation_at_point", "check_violation_at_point_batched"}:
        from act.back_end.bab.bab import (
            verify_bab,
            check_violation_at_point,
            check_violation_at_point_batched,
        )
        return {
            "verify_bab": verify_bab,
            "check_violation_at_point": check_violation_at_point,
            "check_violation_at_point_batched": check_violation_at_point_batched,
        }[name]
    if name == "BaBSRBranching":
        from act.back_end.bab.branching.babsr import BaBSRBranching
        return BaBSRBranching
    raise AttributeError(name)
