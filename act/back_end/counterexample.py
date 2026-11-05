# ===- act/back_end/counterexample.py - Concrete verification ------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------====#
"""Shared concrete input and property checks for ACT verification.

Solvers supply candidates; this module validates evidence without choosing
candidates or changing verification results. Batch lanes retain their original
input-specification and ASSERT alignment.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

from act.back_end.core import Layer, Net
from act.front_end.specs import InKind, OutputSpec
from act.util.model_inference import infer_single_model


_VIOLATION_CHECK_MODULE_CACHE: dict[int, torch.nn.Module] = {}


def clear_violation_check_module_cache() -> None:
    """Release reconstructed models at the end of a verification dispatch."""
    _VIOLATION_CHECK_MODULE_CACHE.clear()


def get_concrete_model(net: Net, reference: torch.Tensor) -> torch.nn.Module:
    """Reuse the concrete model in the candidate device and analysis dtype."""
    key = id(net)
    module = _VIOLATION_CHECK_MODULE_CACHE.get(key)
    if module is None:
        from act.pipeline.verification.act2torch import ACTToTorch

        module = ACTToTorch(net).run().to(device=reference.device, dtype=reference.dtype)
        _VIOLATION_CHECK_MODULE_CACHE[key] = module
    return module


def concrete_forward(net: Net | torch.nn.Module, candidates: torch.Tensor) -> torch.Tensor:
    """Evaluate concrete inputs; reconstruction and inference errors propagate."""
    module = net if isinstance(net, torch.nn.Module) else get_concrete_model(net, candidates)
    module.eval()
    success, output, error = infer_single_model("ce_validate_batched", module, candidates)
    if not success or output is None:
        raise RuntimeError(f"Counterexample model forward failed: {error}")
    if output.dim() < 2 or output.shape[0] != candidates.shape[0]:
        raise ValueError(f"Concrete output must preserve candidate lanes, got {tuple(output.shape)}")
    return output.flatten(1)


def check_input_specs_batched(
    candidates: torch.Tensor, spec_layers: Sequence[Layer], *, tolerance: float = 1e-7,
) -> torch.Tensor:
    """Check finite candidates against every input constraint, without dropping lanes."""
    valid = torch.isfinite(candidates).flatten(1).all(1)
    for layer in spec_layers:
        if layer.input_geometry is not None:
            valid &= layer.input_geometry.contains(candidates, tol=tolerance)
            continue
        params = layer.params
        match params["kind"]:
            case InKind.LIN_POLY:
                a, b = params["A"].to(candidates), params["b"].to(candidates)
                valid &= (candidates.flatten(1) @ a.T <= b).all(1)
            case InKind.BOX | InKind.LINF_BALL:
                lb, ub = params["lb"].to(candidates), params["ub"].to(candidates)
                valid &= ((candidates >= lb - tolerance) & (candidates <= ub + tolerance)).flatten(1).all(1)
            case _:
                raise NotImplementedError(f"Unsupported concrete input constraint: {params['kind']}")
    return valid


def check_output_violations(outputs: torch.Tensor, assert_layer: Layer) -> torch.Tensor:
    """Use the canonical ASSERT rows and OutputSpec's concrete boundary semantics."""
    params = assert_layer.params
    rows = int(params["M"])
    spec = OutputSpec(
        kind=params["kind"],
        c=params["C"].to(outputs).reshape(-1, rows, outputs.shape[1]),
        d=params["thresholds"].to(outputs),
        y_true=params.get("y_true"), margin=params.get("margin"),
        lb=params.get("lb"), ub=params.get("ub"),
    )
    violated, severity = spec.violation(outputs)
    return torch.isfinite(outputs).all(1) & torch.isfinite(severity) & violated


@torch.no_grad()
def validate_counterexamples(
    net: Net, candidates: torch.Tensor, *,
    model_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Return a per-lane mask of legal inputs that concretely violate the property."""
    if candidates.dim() < 2:
        raise ValueError("Counterexample candidates must have shape [B, *input_shape]")
    spec_layers = [layer for layer in net.layers if layer.kind == "INPUT_SPEC"]
    assert_layer = next(layer for layer in net.layers if layer.kind == "ASSERT")
    valid = check_input_specs_batched(candidates, spec_layers)
    if not bool(valid.any()):
        return valid
    # Keep lane-specific targets aligned, while preventing nonfinite candidates
    # from poisoning the concrete forward of otherwise valid lanes.
    safe = torch.where(valid.reshape(-1, *([1] * (candidates.ndim - 1))),
                       candidates, torch.zeros_like(candidates))
    outputs = concrete_forward(net, safe) if model_fn is None else model_fn(safe)
    if outputs.dim() < 2 or outputs.shape[0] != candidates.shape[0]:
        raise ValueError(f"Concrete output must preserve candidate lanes, got {tuple(outputs.shape)}")
    return valid & check_output_violations(outputs.flatten(1), assert_layer)
