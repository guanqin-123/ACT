#===- act/back_end/analyze.py - Network Analysis Functions --------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Network analysis functions for ACT verification framework.
#   Provides analysis capabilities for neural network structures and properties.
#
#===---------------------------------------------------------------------===#

import torch
from collections import deque
from typing import Dict, Tuple
from act.back_end.core import Bounds, Fact, Net, ConSet
from act.back_end.utils import box_join, changed_or_maskdiff, update_cache
from act.back_end.transfer_functions import dispatch_tf, set_transfer_function_mode
from act.back_end.layer_schema import LayerKind

# Initialize default transfer function mode
def initialize_tf_mode(mode: str = "interval"):
    """Initialize transfer function mode. Call this before using analyze()."""
    set_transfer_function_mode(mode)

@torch.no_grad()
def analyze(net: Net, entry_id: int, entry_fact: Fact, eps: float=1e-9) -> Tuple[Dict[int, Fact], Dict[int, Fact], ConSet]:
    """
    Perform abstract interpretation on the network starting from entry_fact.
    Args:
        net: ACT network structure
        entry_id: ID of the entry (INPUT) layer
        entry_fact: Initial Fact containing bounds and constraints for the input
        eps: Convergence epsilon for fixpoint iteration
    
    Returns:
        Tuple of (before, after, globalC) containing propagated facts and global constraints
    """
    # Auto-initialize transfer function mode if not set
    try:
        from act.back_end.transfer_functions import get_transfer_function
        get_transfer_function()  # Check if already initialized
    except RuntimeError:
        initialize_tf_mode("interval")  # Default to interval mode
        
    before: Dict[int, Fact] = {}
    after:  Dict[int, Fact] = {}
    globalC = ConSet()

    # init with +/- inf boxes (vector length per layer's out_vars)
    for L in net.layers:
        n = len(L.out_vars)
        hi = torch.full((n,), float("inf"), device=entry_fact.bounds.lb.device, dtype=entry_fact.bounds.lb.dtype)
        lo = torch.full((n,), -float("inf"), device=entry_fact.bounds.lb.device, dtype=entry_fact.bounds.lb.dtype)
        before[L.id] = Fact(bounds=Bounds(lo.clone(), hi.clone()), cons=ConSet())
        after[L.id]  = Fact(bounds=Bounds(lo.clone(), hi.clone()), cons=ConSet())
        L.cache.clear()

    # Seed entry with provided Fact (includes all input constraints)
    before[entry_id] = entry_fact

    WL = deque([entry_id])
    while WL:
        lid = WL.popleft(); L = net.by_id[lid]

        # merge predecessors into before[lid]
        if net.preds.get(lid):
            preds_list = net.preds[lid]
            pred_bounds = [after[pid].bounds for pid in preds_list]
            kind = L.kind.upper() if isinstance(L.kind, str) else L.kind
            shapes_match = all(b.lb.shape == pred_bounds[0].lb.shape for b in pred_bounds)

            if kind == LayerKind.CONCAT.value:
                # CONCAT: use schema-required concat_dim; preserves batch dim.
                # Note: downstream TF handlers currently fetch bounds fresh via
                # get_all_predecessor_bounds(), so this Bjoin is not consumed —
                # but we still produce a correct shape for any future reader.
                concat_dim = L.params.get("concat_dim", 0)
                lb_cat = torch.cat([b.lb for b in pred_bounds], dim=concat_dim)
                ub_cat = torch.cat([b.ub for b in pred_bounds], dim=concat_dim)
                Bjoin = Bounds(lb=lb_cat, ub=ub_cat)
            elif not shapes_match:
                shapes = [tuple(b.lb.shape) for b in pred_bounds]
                raise ValueError(
                    f"Layer(id={lid}, kind={kind}) has {len(pred_bounds)} predecessors "
                    f"with mismatched shapes {shapes}. Only CONCAT-family layers may have "
                    f"shape-mismatched predecessors. Likely causes: graph wiring bug, "
                    f"missing TF handler for a new multi-input layer kind, or incorrect "
                    f"layer kind assignment."
                )
            else:
                first = pred_bounds[0]
                Bjoin = Bounds(lb=first.lb.clone(), ub=first.ub.clone())
                for b in pred_bounds[1:]:
                    Bjoin = box_join(Bjoin, b)

            Cjoin = ConSet()
            for pid in preds_list:
                for con in after[pid].cons: Cjoin.replace(con)
            before[lid] = Fact(Bjoin, Cjoin)

        out_fact = dispatch_tf(L, before, after, net)

        if changed_or_maskdiff(L, out_fact.bounds, None, eps):
            after[lid] = out_fact
            update_cache(L, out_fact.bounds, None)
            for con in out_fact.cons: globalC.replace(con)
            for sid in net.succs.get(lid, []): WL.append(sid)

    return before, after, globalC
