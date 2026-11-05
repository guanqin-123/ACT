# ===- act/back_end/bab/violation.py - Terminal candidate checking -------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------====#
"""BaB-specific node-box and activation-phase checks for terminal candidates."""
from __future__ import annotations

from typing import Any

import torch

from act.back_end.counterexample import (
    check_input_specs_batched, check_output_violations, concrete_forward, get_concrete_model,
)


@torch.no_grad()
def validate_terminal_candidates(export: Any, candidates: torch.Tensor) -> torch.Tensor:
    """Validate K-aligned candidates against the node and original property."""
    from act.pipeline.verification.act2torch import ActGraphModule

    box = export.input_bounds
    if candidates.shape != box.lb.shape:
        raise ValueError("terminal candidates must preserve the exported K lanes")
    specs = [layer for layer in export.net.layers if layer.kind == "INPUT_SPEC"]
    valid = check_input_specs_batched(candidates, specs, tolerance=0.0)
    valid &= ((candidates >= box.lb) & (candidates <= box.ub)).flatten(1).all(1)
    if not bool(valid.any()):
        return valid
    safe = torch.where(valid.reshape(-1, *([1] * (candidates.ndim - 1))),
                       candidates, box.lb)
    module = get_concrete_model(export.net, safe)
    observed, handles = {}, []
    try:
        for graph in module.modules():
            if not isinstance(graph, ActGraphModule):
                continue
            for lid in export.split_signs:
                if str(lid) not in graph.layers:
                    raise NotImplementedError(f"unsupported_phase_capture:{lid}")
                def capture(_module: torch.nn.Module, args: tuple[torch.Tensor, ...],
                            layer_id: int = lid) -> None:
                    """Capture the actual preactivation while concrete forward runs."""
                    observed[layer_id] = args[0].flatten(1).detach()
                handles.append(graph.layers[str(lid)].register_forward_pre_hook(capture))
        output = concrete_forward(module, safe)
    finally:
        for handle in handles:
            handle.remove()
    for lid, signs in export.split_signs.items():
        if lid not in observed:
            return torch.zeros_like(valid)
        values = observed[lid]
        valid &= (torch.isfinite(values) & ((signs == 0) | (values * signs >= 0))).all(1)
    assert_layer = next(layer for layer in export.net.layers if layer.kind == "ASSERT")
    return valid & check_output_violations(output, assert_layer)
