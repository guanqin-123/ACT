"""Terminal LP solving for one Branch-and-Bound lane.

The LP contains the lane input box, all ACT linear constraints, the accumulated
ReLU split literals, and the joint ``UNSAFE_LINEAR`` output polytope.  Fixed
ReLUs are encoded exactly; any remaining unstable ReLU keeps the triangle
relaxation emitted by :mod:`act.back_end.cons_exportor`.

Gurobi solves the LP when it is usable and the model fits the restricted-license
guard.  Otherwise (no gurobipy or license, an oversized model, or a Gurobi
size-limit error) SciPy's HiGHS backend solves the same sparse LP.

Certification never trusts a solver's floating-point objective directly.  It
recomputes a Neumaier--Shcherbina residual-correction bound from the returned
dual multipliers with outward-rounded interval arithmetic.  A primal solution
is reported as a counterexample only after a concrete ACT forward pass.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import logging
import math
import time
from typing import Any, Optional, Union, cast

import numpy as np
from scipy import sparse
import torch

from act.back_end.bab.violation import (
    _check_input_specs_batched,
    check_violations_batched,
)
from act.back_end.core import Bounds, ConSet, Fact, Layer, Net
from act.back_end.solver.solver_gurobi import is_gurobi_available
from act.back_end.solver.solver_dual import certification_tolerance
from act.front_end.specs import InKind, OutKind
from act.util.device_manager import get_current_settings
from act.util.stats import VerifyResult, VerifyStatus

log = logging.getLogger(__name__)

_NEG_INF = float("-inf")
_POS_INF = float("inf")
# gurobipy errors (GRB_ERROR_NO_LICENSE, GRB_ERROR_SIZE_LIMIT_EXCEEDED) that
# route the LP to HiGHS instead of ending the lane UNKNOWN.
_GUROBI_FALLBACK_ERRORS = {
    10009: "gurobi_license_error",
    10010: "gurobi_size_limit_error",
}

_Matrix = Union[np.ndarray, sparse.spmatrix]


def _down(value: float) -> float:
    """Round one binary64 value toward negative infinity."""
    return float(np.nextafter(np.float64(value), np.float64(_NEG_INF)))


def _up(value: float) -> float:
    """Round one binary64 value toward positive infinity."""
    return float(np.nextafter(np.float64(value), np.float64(_POS_INF)))


def _outward_sums(
    products: np.ndarray, boundaries: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Enclose ``sum(products[start:stop])`` for each consecutive segment.

    Each correctly rounded product is widened by one ulp before an exact
    ``math.fsum``; the rounded sum is widened by one more ulp.
    """
    lower_terms = np.nextafter(products, _NEG_INF).tolist()
    upper_terms = np.nextafter(products, _POS_INF).tolist()
    segments = list(zip(boundaries[:-1], boundaries[1:]))
    lower = [math.fsum(lower_terms[start:stop]) for start, stop in segments]
    upper = [math.fsum(upper_terms[start:stop]) for start, stop in segments]
    return (
        np.nextafter(np.asarray(lower, dtype=np.float64), _NEG_INF),
        np.nextafter(np.asarray(upper, dtype=np.float64), _POS_INF),
    )


def _dot_interval(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    """Outward interval enclosing a binary64 dot product."""
    if left.shape != right.shape:
        raise ValueError(f"dot shape mismatch: {left.shape} != {right.shape}")
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    used = (left != 0.0) & (right != 0.0)
    lower, upper = _outward_sums(left[used] * right[used], [0, int(used.sum())])
    return float(lower[0]), float(upper[0])


def _column_dot_intervals(
    matrix: _Matrix, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """``_dot_interval(matrix[:, j], weights)`` for every column ``j``.

    Only stored nonzeros meet nonzero weights, exactly the products the dense
    scalar loop keeps, so dense and sparse inputs give bit-identical bounds.
    """
    columns = sparse.csc_matrix(matrix, dtype=np.float64, copy=True)
    columns.sum_duplicates()
    row_weights = np.asarray(weights, dtype=np.float64)[columns.indices]
    used = (columns.data != 0.0) & (row_weights != 0.0)
    column_of_entry = np.repeat(
        np.arange(columns.shape[1]), np.diff(columns.indptr)
    )
    counts = np.bincount(column_of_entry[used], minlength=columns.shape[1])
    boundaries = [0, *np.cumsum(counts).tolist()]
    return _outward_sums(columns.data[used] * row_weights[used], boundaries)


def _stored_values(array: _Matrix) -> np.ndarray:
    return array.data if sparse.issparse(array) else np.asarray(array)


def _add_intervals(*intervals: tuple[float, float]) -> tuple[float, float]:
    lower = math.fsum(interval[0] for interval in intervals)
    upper = math.fsum(interval[1] for interval in intervals)
    return _down(lower), _up(upper)


def _multiply_intervals(
    left: tuple[float, float], right: tuple[float, float]
) -> tuple[float, float]:
    products = [
        left[0] * right[0],
        left[0] * right[1],
        left[1] * right[0],
        left[1] * right[1],
    ]
    if any(math.isnan(product) for product in products):
        return _NEG_INF, _POS_INF
    return _down(min(products)), _up(max(products))


def _safe_dual_lower_bound(
    objective: np.ndarray,
    objective_constant: float,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    a_equal: _Matrix,
    b_equal: np.ndarray,
    dual_equal: np.ndarray,
    a_less_equal: _Matrix,
    b_less_equal: np.ndarray,
    dual_less_equal: np.ndarray,
) -> float:
    """Return a safe lower bound for a bounded minimization LP.

    Gurobi uses non-positive dual multipliers for ``A x <= b`` constraints in
    a minimization model.  Clamping tiny positive multiplier errors to zero
    restores dual feasibility.  For residual

    ``r = c - A_eq.T y_eq - A_le.T y_le``, the bound is

    ``c0 + b_eq.T y_eq + b_le.T y_le + inf_[l,u] r.T x``.

    Every dot product and residual operation is enclosed with binary64 outward
    rounding, so the returned number is no greater than the exact LP optimum.
    The constraint matrices may be dense arrays or SciPy sparse matrices.
    """
    n_variables = int(objective.size)
    expected_shapes = {
        "lower_bounds": lower_bounds.shape,
        "upper_bounds": upper_bounds.shape,
    }
    if any(shape != (n_variables,) for shape in expected_shapes.values()):
        raise ValueError(
            f"variable vector shape mismatch for n={n_variables}: {expected_shapes}"
        )
    if a_equal.shape != (b_equal.size, n_variables):
        raise ValueError("equality matrix shape mismatch")
    if a_less_equal.shape != (b_less_equal.size, n_variables):
        raise ValueError("inequality matrix shape mismatch")
    if dual_equal.shape != b_equal.shape:
        raise ValueError("equality dual shape mismatch")
    if dual_less_equal.shape != b_less_equal.shape:
        raise ValueError("inequality dual shape mismatch")
    if not all(
        np.isfinite(_stored_values(array)).all()
        for array in (
            objective,
            lower_bounds,
            upper_bounds,
            a_equal,
            b_equal,
            dual_equal,
            a_less_equal,
            b_less_equal,
            dual_less_equal,
        )
    ):
        return _NEG_INF

    feasible_dual_le = np.minimum(dual_less_equal, 0.0)
    dual_objective = _add_intervals(
        (objective_constant, objective_constant),
        _dot_interval(b_equal, dual_equal),
        _dot_interval(b_less_equal, feasible_dual_le),
    )

    eq_lower, eq_upper = _column_dot_intervals(a_equal, dual_equal)
    le_lower, le_upper = _column_dot_intervals(a_less_equal, feasible_dual_le)
    sum_lower = np.nextafter(
        np.asarray(
            [math.fsum(pair) for pair in zip(eq_lower.tolist(), le_lower.tolist())],
            dtype=np.float64,
        ),
        _NEG_INF,
    )
    sum_upper = np.nextafter(
        np.asarray(
            [math.fsum(pair) for pair in zip(eq_upper.tolist(), le_upper.tolist())],
            dtype=np.float64,
        ),
        _POS_INF,
    )
    costs = np.asarray(objective, dtype=np.float64)
    box_lower = np.asarray(lower_bounds, dtype=np.float64)
    box_upper = np.asarray(upper_bounds, dtype=np.float64)
    with np.errstate(all="ignore"):
        residual_lower = np.nextafter(costs - sum_upper, _NEG_INF)
        residual_upper = np.nextafter(costs - sum_lower, _POS_INF)
        corner_products = np.stack(
            [
                residual_lower * box_lower,
                residual_lower * box_upper,
                residual_upper * box_lower,
                residual_upper * box_upper,
            ]
        )
        correction_lowers = np.where(
            np.isnan(corner_products).any(axis=0),
            _NEG_INF,
            np.nextafter(corner_products.min(axis=0), _NEG_INF),
        )

    correction_lower = _down(math.fsum(correction_lowers.tolist()))
    return _down(dual_objective[0] + correction_lower)


def _unknown(reason: str, **metadata: Any) -> VerifyResult:
    return VerifyResult(
        VerifyStatus.UNKNOWN,
        metadata={"terminal_lp": True, "reason": reason, **metadata},
    )


def _one_lane_signs(
    signs: torch.Tensor, neuron_count: int
) -> tuple[np.ndarray, bool]:
    """Validate one lane's identical spec-row sign copies."""
    if signs.numel() == 0 or signs.dim() == 0 or int(signs.shape[-1]) != neuron_count:
        raise ValueError(
            f"split signs shape {tuple(signs.shape)} is incompatible with "
            f"{neuron_count} neurons"
        )
    if signs.dim() == 3:
        if signs.shape[0] != 1:
            raise ValueError(
                f"split signs must contain exactly one lane, got {tuple(signs.shape)}"
            )
        rows = signs[0].reshape(-1, neuron_count)
    elif signs.dim() in (1, 2):
        rows = signs.reshape(-1, neuron_count)
    else:
        raise ValueError(f"unsupported split signs shape {tuple(signs.shape)}")

    rows = rows.detach().to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(rows).all().item()):
        return np.zeros(neuron_count, dtype=np.float64), False
    if not bool(((rows == -1.0) | (rows == 0.0) | (rows == 1.0)).all().item()):
        return np.zeros(neuron_count, dtype=np.float64), False
    if not bool((rows == rows[0]).all().item()):
        return np.zeros(neuron_count, dtype=np.float64), False
    return rows[0].numpy().copy(), True


def _collect_phase_signs(
    net: Net,
    split_signs: Optional[Mapping[int, torch.Tensor]],
) -> tuple[list[tuple[Layer, np.ndarray]], bool, int]:
    """Validate split signs without allocating dense LP rows."""
    phase_layers: list[tuple[Layer, np.ndarray]] = []
    all_fixed = True
    fixed_count = 0
    split_signs = split_signs or {}

    for layer in net.layers:
        if layer.kind != "RELU":
            continue
        neuron_count = len(layer.out_vars)
        if len(layer.in_vars) != neuron_count:
            raise ValueError(
                f"RELU layer {layer.id} has {len(layer.in_vars)} inputs and "
                f"{neuron_count} outputs"
            )
        raw_signs = split_signs.get(layer.id)
        if raw_signs is None:
            phase_signs = np.zeros(neuron_count, dtype=np.float64)
        else:
            phase_signs, consistent = _one_lane_signs(raw_signs, neuron_count)
            if not consistent:
                raise ValueError(f"invalid split signs for RELU layer {layer.id}")
        phase_layers.append((layer, phase_signs))
        fixed_count += int(np.count_nonzero(phase_signs))
        all_fixed &= bool(np.all(phase_signs != 0.0))

    unknown_keys = set(split_signs) - {layer.id for layer, _ in phase_layers}
    if unknown_keys:
        raise ValueError(f"split signs reference non-RELU layers: {sorted(unknown_keys)}")
    return phase_layers, all_fixed, fixed_count


def _sparse_rows(
    rows: list[dict[int, float]], n_variables: int
) -> sparse.csr_matrix:
    row_ids = [row for row, entries in enumerate(rows) for _ in entries]
    column_ids = [column for entries in rows for column in entries]
    values = [value for entries in rows for value in entries.values()]
    return sparse.csr_matrix(
        (np.asarray(values, dtype=np.float64), (row_ids, column_ids)),
        shape=(len(rows), n_variables),
    )


def _phase_rows(
    phase_layers: list[tuple[Layer, np.ndarray]],
    n_variables: int,
) -> tuple[sparse.csr_matrix, np.ndarray, sparse.csr_matrix, np.ndarray]:
    """Build exact equalities and halfspaces for fixed ReLU phases."""
    equal_rows: list[dict[int, float]] = []
    less_rows: list[dict[int, float]] = []
    for layer, phase_signs in phase_layers:
        for neuron, phase in enumerate(phase_signs):
            if phase == 0.0:
                continue
            preactivation = layer.in_vars[neuron]
            activation = layer.out_vars[neuron]
            if max(preactivation, activation) >= n_variables:
                raise ValueError(f"RELU layer {layer.id} references unknown variable")
            equal: dict[int, float] = {activation: 1.0}
            if phase > 0.0:
                equal[preactivation] = -1.0
                less_rows.append({preactivation: -1.0})
            else:
                less_rows.append({preactivation: 1.0})
            equal_rows.append(equal)

    return (
        _sparse_rows(equal_rows, n_variables),
        np.zeros(len(equal_rows), dtype=np.float64),
        _sparse_rows(less_rows, n_variables),
        np.zeros(len(less_rows), dtype=np.float64),
    )


def _certificate_tolerance(reference: float, dtype: torch.dtype) -> float:
    tensor = torch.tensor(reference, device="cpu", dtype=dtype)
    return float(certification_tolerance(tensor).item())


def _exported_rows(matrix: torch.Tensor) -> sparse.csr_matrix:
    block = matrix.detach().to(device="cpu")
    if not block.is_sparse:
        block = block.to_sparse()
    block = block.coalesce()
    indices = block.indices().numpy()
    values = block.values().to(torch.float64).numpy()
    return sparse.csr_matrix(
        (values, (indices[0], indices[1])), shape=tuple(block.shape)
    )


@dataclass
class _TerminalLP:
    """``min c^T x`` over ``A_eq x = b_eq``, ``A_le x <= b_le``, ``l <= x <= u``."""

    objective: np.ndarray
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    a_equal: sparse.csr_matrix
    b_equal: np.ndarray
    a_less_equal: sparse.csr_matrix
    b_less_equal: np.ndarray


@dataclass
class _LPSolution:
    """Backend-neutral outcome; multipliers use Gurobi's sign convention.

    ``status`` is ``optimal``, ``infeasible``, ``time_limit`` or
    ``inconclusive``.  Row duals are non-positive for ``<=`` rows of the
    minimization, which is also SciPy's ``marginals`` convention.
    """

    status: str
    raw_status: int
    runtime_s: float
    primal: Optional[np.ndarray] = None
    objective_value: Optional[float] = None
    dual_equal: Optional[np.ndarray] = None
    dual_less_equal: Optional[np.ndarray] = None
    farkas_equal: Optional[np.ndarray] = None
    farkas_less_equal: Optional[np.ndarray] = None
    farkas_value: Optional[float] = None


def _finite_vector(values: Any, size: int) -> Optional[np.ndarray]:
    if values is None:
        return None
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.size != size or not np.isfinite(vector).all():
        return None
    return vector


def _solve_with_gurobi(lp: _TerminalLP, timelimit: float) -> _LPSolution:
    from act.back_end.solver import solver_gurobi as gurobi_backend

    gp = gurobi_backend.gp
    grb = gurobi_backend.GRB
    n_variables = int(lp.objective.size)
    with gp.Env(empty=True) as environment:
        environment.setParam("OutputFlag", 0)
        environment.start()
        with gp.Model("act_terminal_lp", env=environment) as model:
            model.Params.OutputFlag = 0
            model.Params.Threads = 1
            model.Params.TimeLimit = float(timelimit)
            model.Params.NumericFocus = 3
            model.Params.InfUnbdInfo = 1
            model.Params.DualReductions = 0
            variables = model.addMVar(
                n_variables,
                lb=lp.lower_bounds,
                ub=lp.upper_bounds,
                name="terminal_x",
            )
            equal_constraints = (
                model.addMConstr(lp.a_equal, variables, "=", lp.b_equal)
                if lp.b_equal.size
                else None
            )
            less_constraints = (
                model.addMConstr(lp.a_less_equal, variables, "<", lp.b_less_equal)
                if lp.b_less_equal.size
                else None
            )
            model.setObjective(lp.objective @ variables, grb.MINIMIZE)
            model.optimize()

            def row_values(constraints: Any, attribute: str, size: int) -> np.ndarray:
                if constraints is None:
                    return np.zeros(size, dtype=np.float64)
                return np.asarray(
                    constraints.getAttr(attribute), dtype=np.float64
                ).reshape(-1)

            primal = (
                _finite_vector(variables.X, n_variables)
                if int(model.SolCount) > 0
                else None
            )
            solution = _LPSolution(
                status="inconclusive",
                raw_status=int(model.Status),
                runtime_s=float(model.Runtime),
                primal=primal,
            )
            if model.Status == grb.OPTIMAL:
                solution.status = "optimal"
                solution.objective_value = float(model.ObjVal)
                solution.dual_equal = row_values(
                    equal_constraints, "Pi", lp.b_equal.size
                )
                solution.dual_less_equal = row_values(
                    less_constraints, "Pi", lp.b_less_equal.size
                )
            elif model.Status == grb.INFEASIBLE:
                solution.status = "infeasible"
                solution.farkas_equal = row_values(
                    equal_constraints, "FarkasDual", lp.b_equal.size
                )
                solution.farkas_less_equal = row_values(
                    less_constraints, "FarkasDual", lp.b_less_equal.size
                )
                solution.farkas_value = float(model.FarkasProof)
            elif model.Status == grb.TIME_LIMIT:
                solution.status = "time_limit"
            return solution


def _highs_phase_one_farkas(
    lp: _TerminalLP, timelimit: float
) -> Optional[tuple[np.ndarray, np.ndarray, float]]:
    """Farkas multipliers for an infeasible LP from its elastic phase one.

    ``min 1^T s`` over ``A_le x - s_le <= b_le``, ``A_eq x + s_p - s_n = b_eq``,
    ``l <= x <= u``, ``s >= 0`` is always feasible.  Its optimal row duals
    keep every slack reduced cost non-negative, so for the zero objective the
    Neumaier--Shcherbina bound on them equals the phase-one optimum; a
    positive safe bound therefore proves the original rows infeasible.
    """
    import scipy.optimize

    if not timelimit > 0.0:
        return None
    m_equal = int(lp.b_equal.size)
    m_less = int(lp.b_less_equal.size)
    n_slacks = m_less + 2 * m_equal
    less_block = sparse.hstack(
        [
            lp.a_less_equal,
            -sparse.identity(m_less, format="csr"),
            sparse.csr_matrix((m_less, 2 * m_equal)),
        ],
        format="csr",
    )
    equal_block = sparse.hstack(
        [
            lp.a_equal,
            sparse.csr_matrix((m_equal, m_less)),
            sparse.identity(m_equal, format="csr"),
            -sparse.identity(m_equal, format="csr"),
        ],
        format="csr",
    )
    bounds = np.vstack(
        [
            np.column_stack([lp.lower_bounds, lp.upper_bounds]),
            np.column_stack([np.zeros(n_slacks), np.full(n_slacks, np.inf)]),
        ]
    )
    result = scipy.optimize.linprog(
        np.concatenate([np.zeros(lp.objective.size), np.ones(n_slacks)]),
        A_ub=less_block,
        b_ub=lp.b_less_equal,
        A_eq=equal_block,
        b_eq=lp.b_equal,
        bounds=bounds,
        method="highs",
        options={"time_limit": float(timelimit)},
    )
    if result.status != 0:
        return None
    farkas_equal = _finite_vector(result.eqlin.marginals, m_equal)
    farkas_less_equal = _finite_vector(result.ineqlin.marginals, m_less)
    if farkas_equal is None or farkas_less_equal is None:
        return None
    return farkas_equal, farkas_less_equal, float(result.fun)


def _solve_with_highs(lp: _TerminalLP, timelimit: float) -> _LPSolution:
    import scipy.optimize

    started = time.perf_counter()
    n_variables = int(lp.objective.size)
    if not timelimit > 0.0:
        return _LPSolution(status="time_limit", raw_status=1, runtime_s=0.0)
    result = scipy.optimize.linprog(
        lp.objective,
        A_ub=lp.a_less_equal,
        b_ub=lp.b_less_equal,
        A_eq=lp.a_equal,
        b_eq=lp.b_equal,
        bounds=np.column_stack([lp.lower_bounds, lp.upper_bounds]),
        method="highs",
        options={"time_limit": float(timelimit)},
    )
    solution = _LPSolution(
        status="inconclusive",
        raw_status=int(result.status),
        runtime_s=time.perf_counter() - started,
        primal=_finite_vector(getattr(result, "x", None), n_variables),
    )
    if result.status == 0:
        solution.status = "optimal"
        solution.objective_value = float(result.fun)
        solution.dual_equal = _finite_vector(
            result.eqlin.marginals, int(lp.b_equal.size)
        )
        solution.dual_less_equal = _finite_vector(
            result.ineqlin.marginals, int(lp.b_less_equal.size)
        )
    elif result.status == 1:
        solution.status = "time_limit"
    elif result.status == 2:
        solution.status = "infeasible"
        farkas = _highs_phase_one_farkas(
            lp, timelimit - (time.perf_counter() - started)
        )
        if farkas is not None:
            (
                solution.farkas_equal,
                solution.farkas_less_equal,
                solution.farkas_value,
            ) = farkas
        solution.runtime_s = time.perf_counter() - started
    return solution


def _gurobi_fallback_reason(error: Exception) -> Optional[str]:
    from act.back_end.solver import solver_gurobi as gurobi_backend

    if not isinstance(error, gurobi_backend.gp.GurobiError):
        return None
    return _GUROBI_FALLBACK_ERRORS.get(getattr(error, "errno", None))


def _concrete_input_valid(
    candidate: torch.Tensor,
    lane_bounds: Bounds,
    input_spec_layers: list[Layer],
) -> bool:
    """Check the lane box and every input-spec constraint concretely."""
    tolerance = 1e-7
    lower = lane_bounds.lb.to(device=candidate.device, dtype=candidate.dtype)
    upper = lane_bounds.ub.to(device=candidate.device, dtype=candidate.dtype)
    if not bool(
        ((candidate >= lower - tolerance) & (candidate <= upper + tolerance))
        .flatten(start_dim=1)
        .all()
        .item()
    ):
        return False
    if not bool(_check_input_specs_batched(candidate, input_spec_layers)[0].item()):
        return False
    flat = candidate.reshape(1, -1)
    for layer in input_spec_layers:
        if layer.params.get("kind") != InKind.LIN_POLY:
            continue
        a_raw = layer.params.get("A")
        b_raw = layer.params.get("b")
        if not isinstance(a_raw, torch.Tensor) or not isinstance(b_raw, torch.Tensor):
            return False
        a = a_raw.to(device=flat.device, dtype=flat.dtype)
        b = b_raw.to(device=flat.device, dtype=flat.dtype).reshape(-1)
        lhs = a.reshape(-1, flat.shape[1]) @ flat[0]
        scale = 1.0 + b.abs() + a.abs() @ flat[0].abs()
        if bool((lhs > b + tolerance * scale).any().item()):
            return False
    return True


def _extract_joint_unsafe_rows(
    assert_layer: Layer, output_count: int
) -> tuple[np.ndarray, np.ndarray, int]:
    if assert_layer.params.get("kind") != OutKind.UNSAFE_LINEAR:
        raise NotImplementedError("terminal LP currently requires UNSAFE_LINEAR")
    row_count = int(assert_layer.params.get("M", 1))
    c_raw = assert_layer.params.get("C", assert_layer.params.get("c"))
    d_raw = assert_layer.params.get(
        "thresholds", assert_layer.params.get("d")
    )
    if not isinstance(c_raw, torch.Tensor) or not isinstance(d_raw, torch.Tensor):
        raise ValueError("UNSAFE_LINEAR requires tensor C and thresholds")
    coefficients = c_raw.detach().to(device="cpu", dtype=torch.float64)
    if coefficients.dim() == 3:
        coefficients = coefficients.reshape(-1, coefficients.shape[-1])
    else:
        coefficients = coefficients.reshape(-1, output_count)
    if coefficients.shape != (row_count, output_count):
        raise ValueError(
            f"UNSAFE_LINEAR C shape {tuple(coefficients.shape)} != "
            f"({row_count}, {output_count})"
        )
    thresholds = (
        d_raw.detach().to(device="cpu", dtype=torch.float64).reshape(-1).numpy()
    )
    if thresholds.shape != (row_count,):
        raise ValueError(
            f"UNSAFE_LINEAR thresholds shape {thresholds.shape} != ({row_count},)"
        )
    return coefficients.numpy(), thresholds, row_count


@torch.no_grad()
def solve_terminal_lp_lane(
    net: Net,
    input_bounds: Bounds,
    split_signs: Optional[Mapping[int, torch.Tensor]] = None,
    *,
    timelimit: float = 0.1,
    max_variables: int = 2000,
    max_constraints: int = 2000,
    _template: Optional[list[_TerminalLP]] = None,
) -> VerifyResult:
    """Solve one terminal BaB lane against its joint unsafe polytope.

    Args:
        net: ACT network ending in an ``UNSAFE_LINEAR`` ASSERT.
        input_bounds: One lane's batched input box, shaped ``[1, *input_shape]``.
        split_signs: ReLU phase literals keyed by layer id. Values may retain
            the spec-row axis (for example ``[1, M, n]``); copies must agree.
        timelimit: LP solver wall-clock limit in seconds (shared by the
            Gurobi attempt and any HiGHS fallback solves).
        max_variables: Gurobi restricted-license guard, including the epigraph
            var; larger models are solved by HiGHS.
        max_constraints: Gurobi restricted-license guard over all rows.

    Returns:
        ``CERTIFIED`` only when an outward-rounded dual lower bound clears the
        repository's dtype-scaled strict-comparison tolerance; ``FALSIFIED``
        only for a concrete checked witness; otherwise ``UNKNOWN``.
    """
    if input_bounds.lb.dim() < 2 or input_bounds.lb.shape[0] != 1:
        return _unknown(
            "expected_single_lane",
            input_shape=tuple(input_bounds.lb.shape),
        )
    if input_bounds.lb.shape != input_bounds.ub.shape:
        return _unknown("input_bounds_shape_mismatch")
    if not math.isfinite(timelimit) or timelimit <= 0.0:
        return _unknown("nonpositive_timelimit")
    if max_variables <= 0 or max_constraints <= 0:
        return _unknown("nonpositive_size_limit")

    managed_device, managed_dtype = get_current_settings()
    metadata: dict[str, Any] = {
        "terminal_lp": True,
        "device": str(managed_device),
        "dtype": str(managed_dtype),
    }

    try:
        from act.back_end.analyze import analyze
        from act.back_end.cons_exportor import export_to_batch_problem
        from act.back_end.verifier import (
            add_all_input_specs,
            gather_input_spec_layers,
            get_assert_layer,
            get_input_ids,
        )

        input_ids = get_input_ids(net)
        input_specs = gather_input_spec_layers(net)
        assert_layer = get_assert_layer(net)
        phase_layers, all_fixed, fixed_count = _collect_phase_signs(
            net, split_signs
        )
        entry_layer = next(layer for layer in net.layers if layer.kind == "INPUT")
        entry_fact = Fact(input_bounds, ConSet())
        add_all_input_specs(entry_fact.cons, input_ids, input_specs)
        _, _, constraints = analyze(net, entry_layer.id, entry_fact)
        problem = export_to_batch_problem(
            net, constraints, assert_layer, input_bounds
        )

        n_network_variables = problem.nvars
        unsafe_c, unsafe_d, unsafe_rows = _extract_joint_unsafe_rows(
            assert_layer, len(assert_layer.in_vars)
        )
        if problem.m_le < unsafe_rows:
            raise ValueError("exported LP has fewer rows than its ASSERT")
        n_variables = n_network_variables + 1
        n_constraints = int(problem.m_eq + problem.m_le + 2 * fixed_count)
        metadata.update(
            {
                "variables": n_variables,
                "constraints": n_constraints,
                "all_relus_phase_fixed": all_fixed,
            }
        )
        a_equal = _exported_rows(problem.A_eq_blockdiag)
        b_equal = (
            problem.b_eq[0].detach().cpu().numpy().astype(np.float64, copy=False)
        )
        network_less_rows = int(problem.m_le) - unsafe_rows
        a_less_equal = _exported_rows(problem.A_le_blockdiag)[:network_less_rows]
        b_less_equal = (
            problem.b_le[0].detach().cpu().numpy().astype(np.float64, copy=False)
        )[:network_less_rows]
        lower_bounds = (
            problem.lb[0].detach().cpu().numpy().astype(np.float64, copy=False)
        )
        upper_bounds = (
            problem.ub[0].detach().cpu().numpy().astype(np.float64, copy=False)
        )

        phase_eq, phase_eq_rhs, phase_le, phase_le_rhs = _phase_rows(
            phase_layers, n_network_variables
        )
        a_equal = sparse.vstack([a_equal, phase_eq], format="csr")
        b_equal = np.concatenate([b_equal, phase_eq_rhs])
        a_less_equal = sparse.vstack([a_less_equal, phase_le], format="csr")
        b_less_equal = np.concatenate([b_less_equal, phase_le_rhs])

        output_ids = np.asarray(assert_layer.in_vars, dtype=np.int64)
        if np.unique(output_ids).size != output_ids.size:
            raise ValueError("ASSERT input variable ids must be unique")
        output_lb = lower_bounds[output_ids]
        output_ub = upper_bounds[output_ids]
        if not np.isfinite(output_lb).all() or not np.isfinite(output_ub).all():
            return _unknown("nonfinite_output_bounds", **metadata)
        row_lower: list[float] = []
        row_upper: list[float] = []
        for row, threshold in zip(unsafe_c, unsafe_d, strict=True):
            products = [
                _multiply_intervals(
                    (float(coefficient), float(coefficient)),
                    (float(lower), float(upper)),
                )
                for coefficient, lower, upper in zip(
                    row, output_lb, output_ub, strict=True
                )
            ]
            row_interval = _add_intervals(*products)
            row_lower.append(_down(row_interval[0] - float(threshold)))
            row_upper.append(_up(row_interval[1] - float(threshold)))
        epigraph_lb = max(row_lower)
        epigraph_ub = max(row_upper)
        if not math.isfinite(epigraph_lb) or not math.isfinite(epigraph_ub):
            return _unknown("nonfinite_epigraph_bounds", **metadata)

        epigraph_column = sparse.csr_matrix((a_equal.shape[0], 1))
        a_equal = sparse.hstack([a_equal, epigraph_column], format="csr")
        a_less_equal = sparse.hstack(
            [a_less_equal, sparse.csr_matrix((a_less_equal.shape[0], 1))],
            format="csr",
        )
        epigraph_rows = sparse.csr_matrix(
            (
                np.concatenate([unsafe_c, -np.ones((unsafe_rows, 1))], axis=1)
                .reshape(-1),
                (
                    np.repeat(np.arange(unsafe_rows), output_ids.size + 1),
                    np.tile(np.append(output_ids, n_variables - 1), unsafe_rows),
                ),
            ),
            shape=(unsafe_rows, n_variables),
        )
        epigraph_rows.eliminate_zeros()
        a_less_equal = sparse.vstack([a_less_equal, epigraph_rows], format="csr")
        b_less_equal = np.concatenate([b_less_equal, unsafe_d])
        lower_bounds = np.concatenate(
            [lower_bounds, np.asarray([epigraph_lb], dtype=np.float64)]
        )
        upper_bounds = np.concatenate(
            [upper_bounds, np.asarray([epigraph_ub], dtype=np.float64)]
        )
        objective = np.zeros(n_variables, dtype=np.float64)
        objective[-1] = 1.0

        if int(a_equal.shape[0] + a_less_equal.shape[0]) != n_constraints:
            raise ValueError("terminal LP constraint count changed during assembly")
        if not np.isfinite(lower_bounds).all() or not np.isfinite(upper_bounds).all():
            return _unknown("nonfinite_variable_bounds", **metadata)

        certificate_reference = max(abs(epigraph_lb), abs(epigraph_ub), 1.0)
        certificate_tol = _certificate_tolerance(
            certificate_reference, input_bounds.lb.dtype
        )
        metadata["certification_tolerance"] = certificate_tol

        lp = _TerminalLP(
            objective=objective,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            a_equal=a_equal,
            b_equal=b_equal,
            a_less_equal=a_less_equal,
            b_less_equal=b_less_equal,
        )
        if _template is not None:
            # Build-only internal path, shared with terminal core admission.
            _template.append(lp)
            return _unknown("core_template", **metadata)
        solve_deadline = time.perf_counter() + float(timelimit)
        fallback_reason: Optional[str] = None
        if not is_gurobi_available():
            fallback_reason = "gurobi_unavailable"
        elif n_variables > max_variables or n_constraints > max_constraints:
            fallback_reason = "model_exceeds_license_size"
        else:
            try:
                solution = _solve_with_gurobi(lp, timelimit)
            except Exception as error:
                fallback_reason = _gurobi_fallback_reason(error)
                if fallback_reason is None:
                    raise
                log.debug("terminal LP: %s; retrying with HiGHS", error)
        if fallback_reason is None:
            metadata["lp_backend"] = "gurobi"
        else:
            metadata["lp_backend"] = "highs"
            metadata["highs_fallback_reason"] = fallback_reason
            solution = _solve_with_highs(
                lp, solve_deadline - time.perf_counter()
            )
        backend = metadata["lp_backend"]
        metadata["lp_status"] = solution.raw_status
        metadata["runtime_s"] = solution.runtime_s

        if (
            solution.status == "optimal"
            and solution.dual_equal is not None
            and solution.dual_less_equal is not None
        ):
            safe_bound = _safe_dual_lower_bound(
                objective,
                0.0,
                lower_bounds,
                upper_bounds,
                a_equal,
                b_equal,
                solution.dual_equal,
                a_less_equal,
                b_less_equal,
                solution.dual_less_equal,
            )
            metadata["safe_dual_bound"] = safe_bound
            metadata["primal_objective"] = solution.objective_value
            if safe_bound > certificate_tol:
                metadata["certificate_kind"] = "optimal_dual"
                return VerifyResult(VerifyStatus.CERTIFIED, metadata=metadata)

        if solution.status == "infeasible":
            if (
                solution.farkas_equal is not None
                and solution.farkas_less_equal is not None
            ):
                farkas_equal = solution.farkas_equal
                farkas_less_equal = solution.farkas_less_equal
                combination = (
                    a_equal.T @ farkas_equal + a_less_equal.T @ farkas_less_equal
                )
                bound_scale = float(
                    np.dot(
                        np.maximum(np.abs(lower_bounds), np.abs(upper_bounds)),
                        np.abs(combination),
                    )
                )
                rhs_scale = abs(float(np.dot(b_equal, farkas_equal))) + abs(
                    float(np.dot(b_less_equal, farkas_less_equal))
                )
                farkas_reference = max(
                    certificate_reference,
                    bound_scale,
                    rhs_scale,
                    abs(float(solution.farkas_value or 0.0)),
                )
                farkas_tol = _certificate_tolerance(
                    farkas_reference, input_bounds.lb.dtype
                )
                farkas_bounds = [
                    _safe_dual_lower_bound(
                        np.zeros_like(objective),
                        0.0,
                        lower_bounds,
                        upper_bounds,
                        a_equal,
                        b_equal,
                        orientation * farkas_equal,
                        a_less_equal,
                        b_less_equal,
                        orientation * farkas_less_equal,
                    )
                    for orientation in (-1.0, 1.0)
                ]
                safe_farkas_bound = max(farkas_bounds)
                metadata["safe_farkas_bound"] = safe_farkas_bound
                metadata["certification_tolerance"] = farkas_tol
                if safe_farkas_bound > farkas_tol:
                    metadata["certificate_kind"] = "farkas_dual"
                    return VerifyResult(VerifyStatus.CERTIFIED, metadata=metadata)
            return _unknown("infeasible_without_safe_farkas_bound", **metadata)

        if solution.primal is not None:
            input_values = torch.as_tensor(
                solution.primal[np.asarray(input_ids, dtype=np.int64)],
                device=input_bounds.lb.device,
                dtype=input_bounds.lb.dtype,
            ).reshape_as(input_bounds.lb)
            input_values = torch.maximum(
                torch.minimum(input_values, input_bounds.ub),
                input_bounds.lb,
            )
            input_valid = _concrete_input_valid(
                input_values, input_bounds, input_specs
            )
            concretely_unsafe = False
            if input_valid:
                concretely_unsafe = bool(
                    check_violations_batched(net, input_values, assert_layer)[0].item()
                )
            if input_valid and concretely_unsafe:
                return VerifyResult(
                    VerifyStatus.FALSIFIED,
                    counterexample=input_values[0].detach().cpu().clone(),
                    metadata=metadata,
                )
            metadata["concrete_input_valid"] = input_valid
            metadata["concretely_unsafe"] = concretely_unsafe
            return _unknown("primal_failed_concrete_validation", **metadata)

        if solution.status == "time_limit":
            return _unknown(f"{backend}_timeout", **metadata)
        return _unknown(f"{backend}_inconclusive", **metadata)
    except Exception as error:
        # Gurobi errors other than license/size limits, unsupported exporters,
        # and numerical solver failures are all inconclusive.  Never let an
        # optional terminal tier turn those conditions into an unsound verdict
        # or abort the enclosing BaB run.
        log.warning("terminal LP returned UNKNOWN: %s", error)
        return _unknown(
            "terminal_lp_error",
            error_type=type(error).__name__,
            **metadata,
        )


# ---------------------------------------------------------------------------
# Exact input-space tier for fully phase-fixed ReLU lanes (H3 speed, D22)
# ---------------------------------------------------------------------------

# Kinds that are affine once every ReLU phase is fixed.  A net containing any
# other kind keeps a non-ReLU nonlinearity (SIGMOID, TANH, MAXPOOL, ...) that
# BaB never splits, so its terminal LP is only a relaxation and is skipped.
_AFFINE_AFTER_PHASE_FIX = frozenset(
    {
        "INPUT", "INPUT_SPEC", "ASSERT", "DENSE", "BN", "CONV1D", "CONV2D",
        "CONV3D", "CONVTRANSPOSE2D", "AVGPOOL1D", "AVGPOOL2D", "AVGPOOL3D",
        "ADAPTIVEAVGPOOL2D", "RELU", "ADD", "SUB", "SCALE", "BIAS", "CONSTANT",
        "CONCAT", "STACK", "RESHAPE", "FLATTEN", "TRANSPOSE", "SQUEEZE",
        "UNSQUEEZE", "TILE", "EXPAND", "SLICE", "GATHER", "PAD", "MEAN",
        "REDUCE_SUM",
    }
)
_COMPOSABLE = frozenset({"INPUT", "INPUT_SPEC", "ASSERT", "DENSE", "RELU"})
# Measured on sat_relu unsat_v90_c111 lanes (91 vars, 293 rows): HiGHS solves
# in ~2-4 ms; the cap leaves a 10x margin without letting one lane stall BaB.
AFFINE_TERMINAL_TIMELIMIT_S = 0.05
GENERIC_TERMINAL_TIMELIMIT_S = 0.1
_UNIT_ROUNDOFF = 2.0 ** -53
# Large-K waves hand up to ~1024 lanes to the terminal tier.  The float64
# composition tensors of _affine_lane_lps are built at most this many bytes
# at a time, and independent lane LPs are solved together in block-diagonal
# HiGHS calls of at most these many lanes / constraint nonzeros.
_AFFINE_BUILD_MAX_BYTES = 256 * 1024 * 1024
_BATCHED_LP_MAX_LANES = 32
_BATCHED_LP_MAX_NONZEROS = 2_000_000
# Cost of the per-block elastic column relative to the lane's objective scale.
_ELASTIC_PENALTY = 1.0e4


def terminal_lp_mode(net: Net) -> str:
    """``affine`` (INPUT/DENSE/RELU chain), ``generic`` or ``nonlinear``."""
    kinds = [str(layer.kind).upper() for layer in net.layers]
    if any(kind not in _AFFINE_AFTER_PHASE_FIX for kind in kinds):
        return "nonlinear"
    if all(kind in _COMPOSABLE for kind in kinds) and all(
        len(net.preds.get(layer.id, [])) <= 1 for layer in net.layers
    ):
        return "affine"
    return "generic"


def _lane_phase_signs(
    split_signs: Optional[Mapping[int, torch.Tensor]],
    layer: Layer,
    lanes: int,
    preactivation_bounds: Optional[Mapping[int, Bounds]] = None,
) -> torch.Tensor:
    """``[K, n]`` phase per lane: split literals first, then neurons that the
    lane's sound pre-activation bounds make stable; 0 where still unfixed."""
    width = len(layer.out_vars)
    phase = torch.zeros(lanes, width, dtype=torch.float64, device="cpu")
    raw = (split_signs or {}).get(layer.id)
    if raw is not None:
        rows = raw.detach().to(device="cpu", dtype=torch.float64).reshape(lanes, -1, width)
        first = rows[:, :1, :]
        valid = ((rows == first) & ((rows == 1.0) | (rows == -1.0) | (rows == 0.0))).all(dim=1)
        phase = torch.where(valid, first[:, 0, :], phase)
    bounds = (preactivation_bounds or {}).get(layer.id)
    if bounds is not None and bounds.lb.numel() == lanes * width:
        lower = bounds.lb.detach().to(device="cpu", dtype=torch.float64).reshape(lanes, width)
        upper = bounds.ub.detach().to(device="cpu", dtype=torch.float64).reshape(lanes, width)
        stable = torch.where(lower >= 0.0, 1.0, torch.where(upper <= 0.0, -1.0, 0.0))
        phase = torch.where(phase == 0.0, stable.to(torch.float64), phase)
    return phase


def _certify_lp_solution(
    lp: _TerminalLP,
    solution: _LPSolution,
    reference: float,
    dtype: torch.dtype,
    metadata: dict[str, Any],
) -> bool:
    """Neumaier--Shcherbina check of an optimal or Farkas dual (no trust)."""
    tolerance = _certificate_tolerance(reference, dtype)
    metadata["certification_tolerance"] = tolerance
    if solution.status == "optimal" and solution.dual_less_equal is not None:
        dual_equal = (
            solution.dual_equal
            if solution.dual_equal is not None
            else np.zeros(lp.b_equal.size)
        )
        bound = _safe_dual_lower_bound(
            lp.objective, 0.0, lp.lower_bounds, lp.upper_bounds, lp.a_equal,
            lp.b_equal, dual_equal, lp.a_less_equal, lp.b_less_equal,
            solution.dual_less_equal,
        )
        metadata["safe_dual_bound"] = bound
        metadata["primal_objective"] = solution.objective_value
        if bound > tolerance:
            metadata["certificate_kind"] = "optimal_dual"
            return True
    if solution.status == "infeasible" and solution.farkas_less_equal is not None:
        farkas_equal = (
            solution.farkas_equal
            if solution.farkas_equal is not None
            else np.zeros(lp.b_equal.size)
        )
        farkas_less = solution.farkas_less_equal
        combination = lp.a_equal.T @ farkas_equal + lp.a_less_equal.T @ farkas_less
        farkas_reference = max(
            reference,
            float(np.dot(np.maximum(np.abs(lp.lower_bounds), np.abs(lp.upper_bounds)),
                         np.abs(combination))),
            abs(float(np.dot(lp.b_equal, farkas_equal)))
            + abs(float(np.dot(lp.b_less_equal, farkas_less))),
            abs(float(solution.farkas_value or 0.0)),
        )
        farkas_tol = _certificate_tolerance(farkas_reference, dtype)
        bound = max(
            _safe_dual_lower_bound(
                np.zeros_like(lp.objective), 0.0, lp.lower_bounds, lp.upper_bounds,
                lp.a_equal, lp.b_equal, orientation * farkas_equal, lp.a_less_equal,
                lp.b_less_equal, orientation * farkas_less,
            )
            for orientation in (-1.0, 1.0)
        )
        metadata["safe_farkas_bound"] = bound
        metadata["certification_tolerance"] = farkas_tol
        if bound > farkas_tol:
            metadata["certificate_kind"] = "farkas_dual"
            return True
    return False


def _solve_small_lp(lp: _TerminalLP, timelimit: float, metadata: dict[str, Any]) -> _LPSolution:
    """HiGHS for the small input-space LP (measured faster than a Gurobi env)."""
    metadata["lp_backend"] = "highs"
    solution = _solve_with_highs(lp, timelimit)
    metadata["lp_status"] = solution.raw_status
    metadata["runtime_s"] = solution.runtime_s
    return solution


def _solve_lps_block_diagonal(
    lps: list[_TerminalLP], references: list[float], timelimit: float
) -> Optional[list[tuple[_LPSolution, bool]]]:
    """Solve independent inequality-form lane LPs in one block-diagonal HiGHS call.

    Block ``k`` is ``min c_k^T x_k + M_k z_k`` over ``A_k x_k - z_k 1 <= b_k``,
    the lane box and ``z_k >= 0``.  The elastic column keeps every block
    feasible, so one infeasible lane cannot void the others.  A block returns
    only its own primal ``x_k`` and row duals: certification re-derives the
    Neumaier--Shcherbina bound of the lane's own LP from them, which is valid
    for any multipliers, so ``z_k`` never enters a certificate.  The flag is
    ``z_k == 0``: the block optimum is then an optimum of the lane LP.
    """
    import scipy.optimize

    if not timelimit > 0.0 or any(lp.b_equal.size for lp in lps):
        return None
    started = time.perf_counter()
    blocks: list[sparse.csr_matrix] = []
    objective: list[np.ndarray] = []
    lower: list[np.ndarray] = []
    upper: list[np.ndarray] = []
    offsets: list[tuple[int, int]] = []
    columns = rows = 0
    for lp, reference in zip(lps, references):
        m_rows, n_columns = lp.a_less_equal.shape
        blocks.append(sparse.hstack(
            [lp.a_less_equal, sparse.csr_matrix(-np.ones((m_rows, 1)))], format="csr"
        ))
        objective.append(np.append(lp.objective, _ELASTIC_PENALTY * max(1.0, abs(reference))))
        lower.append(np.append(lp.lower_bounds, 0.0))
        upper.append(np.append(lp.upper_bounds, np.inf))
        offsets.append((columns, rows))
        columns += n_columns + 1
        rows += m_rows
    result = scipy.optimize.linprog(
        np.concatenate(objective),
        A_ub=sparse.block_diag(blocks, format="csr"),
        b_ub=np.concatenate([lp.b_less_equal for lp in lps]),
        bounds=np.column_stack([np.concatenate(lower), np.concatenate(upper)]),
        method="highs",
        options={"time_limit": float(timelimit)},
    )
    if result.status != 0:
        return None
    primal = _finite_vector(getattr(result, "x", None), columns)
    duals = _finite_vector(result.ineqlin.marginals, rows)
    if primal is None or duals is None:
        return None
    runtime_s = (time.perf_counter() - started) / len(lps)
    solutions: list[tuple[_LPSolution, bool]] = []
    for lp, (column, row) in zip(lps, offsets):
        m_rows, n_columns = lp.a_less_equal.shape
        x = primal[column:column + n_columns].copy()
        solutions.append((
            _LPSolution(
                status="optimal",
                raw_status=int(result.status),
                runtime_s=runtime_s,
                primal=x,
                objective_value=float(lp.objective @ x),
                dual_equal=np.zeros(0),
                dual_less_equal=duals[row:row + m_rows].copy(),
            ),
            float(primal[column + n_columns]) <= 0.0,
        ))
    return solutions


def _batched_lane_solutions(
    lps: list[Optional[_TerminalLP]], references: list[float], remaining: Any
) -> dict[int, tuple[_LPSolution, bool]]:
    """Block-diagonal solves over the lanes that have an input-space LP."""
    eligible = [lane for lane, lp in enumerate(lps) if lp is not None]
    solved: dict[int, tuple[_LPSolution, bool]] = {}
    position = 0
    while len(eligible) >= 2 and position < len(eligible):
        chunk: list[int] = []
        nonzeros = 0
        while position < len(eligible) and len(chunk) < _BATCHED_LP_MAX_LANES:
            lp = lps[eligible[position]]
            assert lp is not None
            if chunk and nonzeros + lp.a_less_equal.nnz > _BATCHED_LP_MAX_NONZEROS:
                break
            chunk.append(eligible[position])
            nonzeros += lp.a_less_equal.nnz
            position += 1
        timelimit = min(AFFINE_TERMINAL_TIMELIMIT_S * len(chunk), remaining())
        try:
            solutions = _solve_lps_block_diagonal(
                [cast(_TerminalLP, lps[lane]) for lane in chunk],
                [references[lane] for lane in chunk],
                timelimit,
            )
        except Exception as error:  # per-lane solves remain the fallback
            log.warning("batched terminal LP unavailable: %s", error)
            solutions = None
        if solutions is not None:
            solved.update(zip(chunk, solutions))
    return solved


def _affine_lane_chunks(net: Net, lanes: int, n_inputs: int) -> list[tuple[int, int]]:
    """Lane ranges whose float64 composition tensors fit _AFFINE_BUILD_MAX_BYTES."""
    widths = [
        int(layer.params["weight"].shape[0])
        for layer in net.layers
        if str(layer.kind).upper() == "DENSE" and "weight" in layer.params
    ]
    per_lane = 8 * max(1, n_inputs) * (sum(widths) + 4 * max(widths, default=1))
    step = max(1, _AFFINE_BUILD_MAX_BYTES // per_lane)
    return [(start, min(lanes, start + step)) for start in range(0, lanes, step)]


@torch.no_grad()
def _affine_lane_lps(
    net: Net, lower: torch.Tensor, upper: torch.Tensor,
    split_signs: Optional[Mapping[int, torch.Tensor]],
    preactivation_bounds: Optional[Mapping[int, Bounds]] = None,
) -> tuple[list[Optional[_TerminalLP]], list[float]]:
    """Exact input-space LPs, built in batch by composing DENSE maps with masks.

    Variables are the ``n`` inputs plus the epigraph ``t``; rows are one
    halfspace per fixed ReLU literal, the joint ``UNSAFE_LINEAR`` rows and any
    LIN_POLY input rows; the lane box gives the variable bounds.  Every
    composed row's rhs is relaxed by a rigorous bound on the float64
    composition error, so each LP relaxes the exact lane problem.
    """
    lanes, n_inputs = lower.shape
    # The per-lane LPs are tiny: building them on CPU avoids CUDA launch and
    # sync overhead (measured 81 ms/lane on CUDA vs ~9 ms/lane on CPU).
    device = torch.device("cpu")
    lower, upper = lower.detach().to(device), upper.detach().to(device)
    f64 = torch.float64
    xmax = torch.maximum(lower.abs(), upper.abs()).to(f64).unsqueeze(-1)
    a_map: Optional[torch.Tensor] = None  # None encodes the identity map
    c_map = torch.zeros(lanes, n_inputs, dtype=f64, device=device)
    abs_a: Optional[torch.Tensor] = None
    abs_c = c_map.clone()
    gamma = 0.0
    fixed = torch.ones(lanes, dtype=torch.bool, device=device)
    rows: list[torch.Tensor] = []
    rhs: list[torch.Tensor] = []
    extra_rows: list[np.ndarray] = []
    extra_rhs: list[np.ndarray] = []

    def error_bound(abs_rows: torch.Tensor, abs_const: torch.Tensor) -> torch.Tensor:
        return 2.0 * gamma * (torch.matmul(abs_rows, xmax).squeeze(-1) + abs_const) + 1e-300

    def compose(weight: torch.Tensor, current: Optional[torch.Tensor]) -> torch.Tensor:
        if current is None:
            return weight.unsqueeze(0).expand(lanes, -1, -1)
        return torch.matmul(weight, current)

    for layer in net.layers:
        kind = str(layer.kind).upper()
        if kind == "INPUT_SPEC" and layer.params.get("kind") == InKind.LIN_POLY:
            a_poly = layer.params["A"].detach().to("cpu", f64).reshape(-1, n_inputs)
            extra_rows.append(a_poly.numpy())
            extra_rhs.append(layer.params["b"].detach().to("cpu", f64).reshape(-1).numpy())
        elif kind == "DENSE":
            weight = layer.params["weight"].detach().to(device, f64)
            bias = layer.params.get("bias")
            bias = (torch.zeros(weight.shape[0], dtype=f64, device=device) if bias is None
                    else bias.detach().to(device, f64).reshape(-1))
            gamma += (weight.shape[1] + 2) * _UNIT_ROUNDOFF
            a_map = compose(weight, a_map)
            abs_a = compose(weight.abs(), abs_a)
            c_map = c_map @ weight.T + bias
            abs_c = abs_c @ weight.abs().T + bias.abs()
        elif kind == "RELU":
            if a_map is None or abs_a is None:
                raise ValueError("RELU before any DENSE layer")
            phase = _lane_phase_signs(
                split_signs, layer, lanes, preactivation_bounds
            ).to(device)
            fixed &= (phase != 0.0).all(dim=1)
            rows.append(-phase.unsqueeze(-1) * a_map)
            rhs.append(phase * c_map + error_bound(abs_a, abs_c))
            active = (phase > 0.0).to(f64)
            a_map, c_map = a_map * active.unsqueeze(-1), c_map * active
            abs_a, abs_c = abs_a * active.unsqueeze(-1), abs_c * active
        elif kind == "ASSERT":
            if a_map is None or abs_a is None:
                raise ValueError("ASSERT before any DENSE layer")
            coefficients, thresholds, row_count = _extract_joint_unsafe_rows(
                layer, len(layer.in_vars)
            )
            c_rows = torch.from_numpy(coefficients).to(device)
            d_rows = torch.from_numpy(thresholds).to(device)
            gamma += (c_rows.shape[1] + 2) * _UNIT_ROUNDOFF
            unsafe_a = torch.matmul(c_rows, a_map).cpu()
            unsafe_c = (c_map @ c_rows.T - d_rows).cpu()
            unsafe_err = error_bound(
                torch.matmul(c_rows.abs(), abs_a), abs_c @ c_rows.abs().T + d_rows.abs()
            ).cpu()
    fixed = fixed.cpu()
    lps: list[Optional[_TerminalLP]] = []
    references: list[float] = []
    lower64 = lower.detach().to("cpu", torch.float64).numpy()
    upper64 = upper.detach().to("cpu", torch.float64).numpy()
    phase_a = torch.cat(rows, dim=1).cpu() if rows else torch.zeros(lanes, 0, n_inputs, dtype=f64, device=device)
    phase_b = torch.cat(rhs, dim=1).cpu() if rows else torch.zeros(lanes, 0, dtype=f64, device=device)
    for lane in range(lanes):
        if not bool(fixed[lane]):
            lps.append(None)
            references.append(1.0)
            continue
        a_u = unsafe_a[lane].numpy()
        const = unsafe_c[lane].numpy()
        err = unsafe_err[lane].numpy()
        low_part = np.minimum(a_u * lower64[lane], a_u * upper64[lane]).sum(axis=1)
        high_part = np.maximum(a_u * lower64[lane], a_u * upper64[lane]).sum(axis=1)
        scale = np.abs(a_u) @ np.abs(np.maximum(np.abs(lower64[lane]), np.abs(upper64[lane]))) + np.abs(const)
        margin = err + 1e-9 * (1.0 + scale)
        t_lb = float(np.max(low_part + const - margin))
        t_ub = float(np.max(high_part + const + margin))
        blocks = [np.hstack([phase_a[lane].numpy(), np.zeros((phase_a.shape[1], 1))]),
                  np.hstack([a_u, -np.ones((row_count, 1))])]
        rhs_blocks = [phase_b[lane].numpy(), -const + err]
        for a_poly, b_poly in zip(extra_rows, extra_rhs):
            blocks.append(np.hstack([a_poly, np.zeros((a_poly.shape[0], 1))]))
            rhs_blocks.append(b_poly)
        objective = np.zeros(n_inputs + 1)
        objective[-1] = 1.0
        lps.append(_TerminalLP(
            objective=objective,
            lower_bounds=np.append(lower64[lane], t_lb),
            upper_bounds=np.append(upper64[lane], t_ub),
            a_equal=sparse.csr_matrix((0, n_inputs + 1)),
            b_equal=np.zeros(0),
            a_less_equal=sparse.csr_matrix(np.vstack(blocks)),
            b_less_equal=np.concatenate(rhs_blocks),
        ))
        references.append(max(abs(t_lb), abs(t_ub), 1.0))
    return lps, references


@torch.no_grad()
def solve_terminal_lp_lanes(
    net: Net,
    lane_bounds: Bounds,
    split_signs: Optional[Mapping[int, torch.Tensor]],
    remaining: Any,
    preactivation_bounds: Optional[Mapping[int, Bounds]] = None,
) -> list[VerifyResult]:
    """Terminal tier for ``K`` candidate-less lanes (``lane_bounds`` ``[K, ...]``).

    ``remaining`` returns the BaB budget left in seconds; ``preactivation_bounds``
    (RELU layer id -> sound ``[K, n]`` input bounds) fixes stable neurons.  Nets with non-ReLU
    nonlinearities are skipped (UNKNOWN, no solve).  Fully phase-fixed lanes of
    INPUT/DENSE/RELU chains use the exact input-space LP; other lanes use the
    generic exported LP (``solve_terminal_lp_lane``).
    """
    lanes = int(lane_bounds.lb.shape[0])
    mode = terminal_lp_mode(net)
    if mode == "nonlinear":
        return [_unknown("nonlinear_layers_remain", skipped=True) for _ in range(lanes)]
    lps: list[Optional[_TerminalLP]] = [None] * lanes
    references = [1.0] * lanes
    flat_lb = lane_bounds.lb.reshape(lanes, -1)
    flat_ub = lane_bounds.ub.reshape(lanes, -1)
    batched: dict[int, tuple[_LPSolution, bool]] = {}
    if mode == "affine":
        try:
            lps, references = [], []
            for start, stop in _affine_lane_chunks(net, lanes, int(flat_lb.shape[1])):
                chunk_lps, chunk_references = _affine_lane_lps(
                    net, flat_lb[start:stop], flat_ub[start:stop],
                    None if split_signs is None else {
                        lid: signs[start:stop] for lid, signs in split_signs.items()
                    },
                    None if preactivation_bounds is None else {
                        lid: Bounds(bounds.lb[start:stop], bounds.ub[start:stop])
                        if bounds.lb.dim() > 0 and int(bounds.lb.shape[0]) == lanes else bounds
                        for lid, bounds in preactivation_bounds.items()
                    },
                )
                lps.extend(chunk_lps)
                references.extend(chunk_references)
        except Exception as error:  # unsupported layout: generic path per lane
            log.warning("affine terminal LP unavailable: %s", error)
            lps = [None] * lanes
            references = [1.0] * lanes
        batched = _batched_lane_solutions(lps, references, remaining)
    input_specs = None
    assert_layer = None
    results: list[VerifyResult] = []
    for lane in range(lanes):
        lane_box = Bounds(lane_bounds.lb[lane:lane + 1], lane_bounds.ub[lane:lane + 1])
        lp = lps[lane]
        if lp is None:
            lane_signs = None if split_signs is None else {
                lid: signs[lane:lane + 1] for lid, signs in split_signs.items()
            }
            results.append(solve_terminal_lp_lane(
                net, lane_box, lane_signs,
                timelimit=min(GENERIC_TERMINAL_TIMELIMIT_S, remaining()),
            ))
            continue
        metadata: dict[str, Any] = {"terminal_lp": True, "formulation": "input_space"}
        timelimit = min(AFFINE_TERMINAL_TIMELIMIT_S, remaining())
        if not timelimit > 0.0:
            results.append(_unknown("nonpositive_timelimit", **metadata))
            continue
        try:
            solution: Optional[_LPSolution] = None
            if lane in batched:
                block_solution, block_exact = batched[lane]
                metadata.update(lp_backend="highs_batched", lp_status=block_solution.raw_status,
                                runtime_s=block_solution.runtime_s)
                if _certify_lp_solution(lp, block_solution, references[lane], lane_box.lb.dtype, metadata):
                    results.append(VerifyResult(VerifyStatus.CERTIFIED, metadata=metadata))
                    continue
                if block_exact:
                    solution = block_solution
            if solution is None:
                solution = _solve_small_lp(lp, timelimit, metadata)
                if _certify_lp_solution(lp, solution, references[lane], lane_box.lb.dtype, metadata):
                    results.append(VerifyResult(VerifyStatus.CERTIFIED, metadata=metadata))
                    continue
            if solution.primal is None:
                results.append(_unknown(f"highs_{solution.status}", **metadata))
                continue
            if input_specs is None:
                from act.back_end.verifier import gather_input_spec_layers, get_assert_layer

                input_specs = gather_input_spec_layers(net)
                assert_layer = get_assert_layer(net)
            candidate = torch.as_tensor(
                solution.primal[:-1], device=lane_box.lb.device, dtype=lane_box.lb.dtype
            ).reshape_as(lane_box.lb)
            candidate = torch.maximum(torch.minimum(candidate, lane_box.ub), lane_box.lb)
            valid = _concrete_input_valid(candidate, lane_box, input_specs)
            unsafe = valid and bool(
                check_violations_batched(net, candidate, assert_layer)[0].item()
            )
            if unsafe:
                results.append(VerifyResult(
                    VerifyStatus.FALSIFIED,
                    counterexample=candidate[0].detach().cpu().clone(),
                    metadata=metadata,
                ))
                continue
            metadata.update(concrete_input_valid=valid, concretely_unsafe=unsafe)
            results.append(_unknown("primal_failed_concrete_validation", **metadata))
        except Exception as error:
            log.warning("input-space terminal LP returned UNKNOWN: %s", error)
            results.append(_unknown("terminal_lp_error", error_type=type(error).__name__, **metadata))
    return results
