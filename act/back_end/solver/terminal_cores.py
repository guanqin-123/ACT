"""Layered terminal LP dual costs and mandatory core-region admission."""

from collections.abc import Callable, Mapping
from typing import Any, Optional, cast

import numpy as np
from scipy import sparse
import torch

from act.back_end.core import Bounds, Net
from act.back_end.solver.terminal_lp import (
    _TerminalLP, _LPSolution, _collect_phase_signs, _phase_rows,
    _safe_dual_lower_bound, _certify_lp_solution, _solve_with_highs,
    solve_terminal_lp_lane,
)


def terminal_literal_cost(
    phase: float, lower: float, upper: float, sign_dual: float, equality_dual: float,
) -> float:
    """Charge sign rows and the invalid direction of ``z-slope*zhat=0``.

    Positive equality Pi uses universal ReLU lower facets and is free.
    Negative Pi uses the phase upper plane. Its maximum triangle gap is -l
    (active) or u (inactive), NOT just the chord intercept at zero: the
    opposite endpoint is in the enlarged region as well.
    """
    reach = max(0.0, -lower if phase > 0 else upper)
    return (max(0.0, -sign_dual) + max(0.0, -equality_dual)) * reach


class TerminalCoreLP:
    """Cache split-free bounds/triangles; never reuse split-dependent bounds."""

    def __init__(self, net: Net, input_bounds: Bounds) -> None:
        self.net = net
        self.dtype = input_bounds.lb.dtype
        templates: list[_TerminalLP] = []
        result = solve_terminal_lp_lane(net, input_bounds, _template=templates)
        if not templates:
            raise ValueError(f"terminal core template unavailable: {result.metadata}")
        self.base = templates[0]
        self.reference = max(abs(self.base.lower_bounds[-1]),
                             abs(self.base.upper_bounds[-1]), 1.0)

    def certify(
        self, signs: Mapping[int, torch.Tensor], timelimit: float,
    ) -> tuple[bool, _LPSolution, dict[str, Any]]:
        """Re-solve exactly these literals; dropped neurons keep triangles."""
        phases, _, _ = _collect_phase_signs(self.net, signs)
        eq, eq_rhs, le, le_rhs = _phase_rows(phases, self.base.objective.size)
        lp = _TerminalLP(
            self.base.objective, self.base.lower_bounds, self.base.upper_bounds,
            cast(sparse.csr_matrix, sparse.vstack([self.base.a_equal, eq], format="csr")),
            np.concatenate([self.base.b_equal, eq_rhs]),
            cast(sparse.csr_matrix, sparse.vstack([self.base.a_less_equal, le], format="csr")),
            np.concatenate([self.base.b_less_equal, le_rhs]),
        )
        solution = _solve_with_highs(lp, min(0.1, timelimit))
        metadata: dict[str, Any] = {}
        certified = _certify_lp_solution(lp, solution, self.reference, self.dtype, metadata)
        if certified and solution.status == "infeasible":
            assert solution.farkas_equal is not None
            assert solution.farkas_less_equal is not None
            plus = _safe_dual_lower_bound(
                np.zeros_like(lp.objective), 0., lp.lower_bounds, lp.upper_bounds,
                lp.a_equal, lp.b_equal, solution.farkas_equal,
                lp.a_less_equal, lp.b_less_equal, solution.farkas_less_equal,
            )
            orientation = 1. if plus > metadata["certification_tolerance"] else -1.
            solution.dual_equal = orientation * solution.farkas_equal
            solution.dual_less_equal = orientation * solution.farkas_less_equal
        return certified, solution, metadata

    def learn(
        self, signs: Mapping[int, torch.Tensor], *, theta: float,
        delta_abs: float, delta_rel: float, remaining: Callable[[], float],
        coarsening: bool = True,
    ) -> Optional[tuple[dict[int, torch.Tensor], np.ndarray, dict[str, Any]]]:
        """Propose deletions from dual costs, then safely re-solve the proposal."""
        if remaining() <= 0.0:
            return None
        valid, solution, metadata = self.certify(signs, remaining())
        if not valid or solution.dual_equal is None or solution.dual_less_equal is None:
            return None
        phases, _, _ = _collect_phase_signs(self.net, signs)
        literals = [(layer, neuron, phase) for layer, values in phases
                    for neuron, phase in enumerate(values) if phase != 0.]
        costs = np.asarray([
            terminal_literal_cost(
                phase, self.base.lower_bounds[layer.in_vars[neuron]],
                self.base.upper_bounds[layer.in_vars[neuron]],
                solution.dual_less_equal[self.base.b_less_equal.size + index],
                solution.dual_equal[self.base.b_equal.size + index],
            ) for index, (layer, neuron, phase) in enumerate(literals)
        ])
        candidate = {key: value.clone() for key, value in signs.items()}
        slack = metadata.get("safe_dual_bound", metadata.get("safe_farkas_bound", -np.inf))
        delta = delta_abs + delta_rel * abs(slack)
        budget = theta * slack - delta
        spent = 0.
        if coarsening and np.isfinite(costs).all() and slack > delta:
            for index in np.argsort(costs, kind="stable"):
                cost = costs[index]
                if cost != 0. and spent + cost > budget:
                    break
                layer, neuron, _ = literals[index]
                candidate[layer.id][..., neuron] = 0.
                spent += cost
        changed = any(not torch.equal(candidate[key], value) for key, value in signs.items())
        admitted = not changed
        if changed and remaining() > 0.:
            admitted, _, _ = self.certify(candidate, remaining())
        # The fallback full row passed its own admission LP above.
        metadata.update(admitted=True, fallback_full_row=not admitted)
        return candidate if admitted else dict(signs), costs, metadata
