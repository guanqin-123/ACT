from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

import numpy as np
import torch

from act.back_end.core import Bounds, ConSet, Fact, Layer, Net, get_topo_order
from act.back_end.layer_schema import LayerKind
from act.back_end.solver.solver_gurobi import is_gurobi_available, setup_gurobi_license
from act.front_end.specs import InKind, OutKind
from act.util.stats import VerifyResult, VerifyStatus


_SUPPORTED_BODY_KINDS = frozenset(
    {
        LayerKind.DENSE.value,
        LayerKind.CONV2D.value,
        LayerKind.RELU.value,
        LayerKind.ADD.value,
        LayerKind.FLATTEN.value,
    }
)


@dataclass(frozen=True)
class _ModelPlan:
    variable_ids: tuple[int, ...]
    variable_index: dict[int, int]
    lower: np.ndarray
    upper: np.ndarray
    ambiguous_relus: int
    constraint_count: int

    @property
    def variable_count(self) -> int:
        return len(self.variable_ids) + self.ambiguous_relus


class _UnsupportedNetwork(ValueError):
    pass


def _unknown(reason: str, **metadata: Any) -> VerifyResult:
    return VerifyResult(
        VerifyStatus.UNKNOWN,
        metadata={"solver": "gurobi_milp", "reason": reason, **metadata},
    )


def _one_lane(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if tensor.dim() < 2 or tensor.shape[0] != 1:
        raise _UnsupportedNetwork(
            f"{name} must have exactly one verification lane, got {tuple(tensor.shape)}"
        )
    return tensor.detach().reshape(1, -1)[0]


def _outward_bounds(bounds: Bounds, name: str) -> tuple[np.ndarray, np.ndarray]:
    lower = _one_lane(bounds.lb, f"{name}.lb").cpu().numpy().astype(np.float64)
    upper = _one_lane(bounds.ub, f"{name}.ub").cpu().numpy().astype(np.float64)
    if np.any(np.isnan(lower)) or np.any(np.isnan(upper)) or np.any(lower > upper):
        raise _UnsupportedNetwork(f"{name} has invalid root bounds")
    lower = np.where(np.isfinite(lower), np.nextafter(lower, -np.inf), lower)
    upper = np.where(np.isfinite(upper), np.nextafter(upper, np.inf), upper)
    return lower, upper


def _input_bounds(net: Net) -> Bounds:
    from act.back_end.verifier import gather_input_spec_layers, seed_from_input_specs

    spec_layers = gather_input_spec_layers(net)
    seed = seed_from_input_specs(spec_layers)
    _one_lane(seed.lb, "input lower bounds")
    _one_lane(seed.ub, "input upper bounds")
    lower = seed.lb.clone()
    upper = seed.ub.clone()
    for layer in spec_layers:
        kind = layer.params.get("kind")
        if kind == InKind.BOX or kind == InKind.LINF_BALL:
            lb = layer.params.get("lb")
            ub = layer.params.get("ub")
            if isinstance(lb, torch.Tensor) and isinstance(ub, torch.Tensor):
                lower = torch.maximum(lower, lb.to(lower))
                upper = torch.minimum(upper, ub.to(upper))
        elif kind != InKind.LIN_POLY:
            raise _UnsupportedNetwork(f"input kind {kind!r} is not MILP-supported")
    if bool((lower > upper).any().item()):
        raise _UnsupportedNetwork("input box intersection is empty")
    return Bounds(lower, upper)


def _root_facts(net: Net, input_box: Bounds):
    from act.back_end.analyze import analyze
    from act.back_end.verifier import (
        add_all_input_specs,
        find_entry_layer_id,
        gather_input_spec_layers,
        get_input_ids,
    )

    entry_id = find_entry_layer_id(net)
    entry_fact = Fact(input_box, ConSet())
    add_all_input_specs(
        entry_fact.cons, get_input_ids(net), gather_input_spec_layers(net)
    )
    return analyze(net, entry_id, entry_fact)[:2]


def _relu_partition(layer: Layer, before: dict[int, Fact]) -> tuple[np.ndarray, np.ndarray]:
    lower, upper = _outward_bounds(before[layer.id].bounds, f"RELU {layer.id} input")
    if lower.size != len(layer.in_vars) or upper.size != len(layer.out_vars):
        raise _UnsupportedNetwork(f"RELU {layer.id} variable/bound size mismatch")
    return lower, upper


def _linear_row_count(layer: Layer) -> int:
    if layer.kind == LayerKind.ADD.value:
        return len(layer.out_vars)
    return len(layer.out_vars)


def _property_rows(assert_layer: Layer) -> tuple[np.ndarray, np.ndarray]:
    if assert_layer.params.get("kind") != OutKind.UNSAFE_LINEAR:
        raise _UnsupportedNetwork("only joint UNSAFE_LINEAR output polytopes are supported")
    coefficients = assert_layer.params.get("C", assert_layer.params.get("c"))
    thresholds = assert_layer.params.get(
        "thresholds", assert_layer.params.get("d")
    )
    if not isinstance(coefficients, torch.Tensor) or not isinstance(
        thresholds, torch.Tensor
    ):
        raise _UnsupportedNetwork("UNSAFE_LINEAR requires tensor C and thresholds")
    coefficients = coefficients.detach()
    thresholds = thresholds.detach()
    if coefficients.dim() == 3 and coefficients.shape[0] == 1:
        coefficients = coefficients[0]
    if coefficients.dim() == 1:
        coefficients = coefficients.unsqueeze(0)
    if thresholds.dim() == 2 and thresholds.shape[0] == 1:
        thresholds = thresholds[0]
    thresholds = thresholds.reshape(-1)
    if coefficients.dim() != 2 or coefficients.shape[0] != thresholds.numel():
        raise _UnsupportedNetwork("UNSAFE_LINEAR C/threshold shape mismatch")
    return (
        coefficients.cpu().numpy().astype(np.float64),
        thresholds.cpu().numpy().astype(np.float64),
    )


def _build_plan(net: Net, before: dict[int, Fact], after: dict[int, Fact]) -> _ModelPlan:
    body_layers = [
        layer
        for layer in net.layers
        if layer.kind
        not in {LayerKind.INPUT.value, LayerKind.INPUT_SPEC.value, LayerKind.ASSERT.value}
    ]
    unsupported = sorted(
        {layer.kind for layer in body_layers if layer.kind not in _SUPPORTED_BODY_KINDS}
    )
    if unsupported:
        raise _UnsupportedNetwork(f"unsupported layer kinds: {', '.join(unsupported)}")

    variable_ids = tuple(
        sorted({var for layer in net.layers for var in layer.in_vars + layer.out_vars})
    )
    variable_index = {var_id: index for index, var_id in enumerate(variable_ids)}
    lower = np.full(len(variable_ids), -np.inf, dtype=np.float64)
    upper = np.full(len(variable_ids), np.inf, dtype=np.float64)

    for layer in net.layers:
        fact = after.get(layer.id)
        if fact is None or not layer.out_vars:
            continue
        layer_lower, layer_upper = _outward_bounds(fact.bounds, f"layer {layer.id}")
        if layer_lower.size != len(layer.out_vars):
            raise _UnsupportedNetwork(f"layer {layer.id} variable/bound size mismatch")
        for offset, var_id in enumerate(layer.out_vars):
            pos = variable_index[var_id]
            lower[pos] = max(lower[pos], layer_lower[offset])
            upper[pos] = min(upper[pos], layer_upper[offset])

    ambiguous_relus = 0
    constraint_count = 0
    for layer in body_layers:
        if layer.kind == LayerKind.RELU.value:
            relu_lower, relu_upper = _relu_partition(layer, before)
            active = relu_lower >= 0.0
            inactive = relu_upper <= 0.0
            ambiguous = ~(active | inactive)
            if np.any(ambiguous & (~np.isfinite(relu_lower) | ~np.isfinite(relu_upper))):
                raise _UnsupportedNetwork(f"RELU {layer.id} has non-finite big-M bounds")
            ambiguous_relus += int(ambiguous.sum())
            constraint_count += int(active.sum() + inactive.sum() + 3 * ambiguous.sum())
        else:
            constraint_count += _linear_row_count(layer)

    for layer in net.layers:
        if layer.kind != LayerKind.INPUT_SPEC.value:
            continue
        if layer.params.get("kind") == InKind.LIN_POLY:
            matrix = layer.params.get("A")
            if not isinstance(matrix, torch.Tensor):
                raise _UnsupportedNetwork("LIN_POLY requires tensor A")
            constraint_count += int(matrix.shape[-2])

    coefficients, _thresholds = _property_rows(net.layers[-1])
    constraint_count += int(coefficients.shape[0])
    return _ModelPlan(
        variable_ids,
        variable_index,
        lower,
        upper,
        ambiguous_relus,
        constraint_count,
    )


def _sum(gp: Any, variables: Any, positions: Iterable[int], coeffs: Iterable[float]):
    return gp.quicksum(float(coef) * variables[pos] for pos, coef in zip(positions, coeffs))


def _add_dense(gp: Any, model: Any, variables: Any, plan: _ModelPlan, layer: Layer) -> None:
    weight = layer.params["weight"]
    if not isinstance(weight, torch.Tensor):
        raise _UnsupportedNetwork(f"DENSE {layer.id} requires tensor weight")
    weight_np = weight.detach().cpu().numpy().astype(np.float64)
    bias = layer.params.get("bias")
    bias_np = (
        np.zeros(weight_np.shape[0], dtype=np.float64)
        if bias is None
        else bias.detach().cpu().numpy().astype(np.float64).reshape(-1)
    )
    token_wise = bool(layer.params.get("token_wise", False))
    if token_wise:
        in_width, out_width = weight_np.shape[1], weight_np.shape[0]
        if len(layer.in_vars) % in_width or len(layer.out_vars) % out_width:
            raise _UnsupportedNetwork(f"token-wise DENSE {layer.id} has invalid dimensions")
        blocks = len(layer.in_vars) // in_width
    else:
        in_width, out_width, blocks = len(layer.in_vars), len(layer.out_vars), 1
        if weight_np.shape != (out_width, in_width):
            raise _UnsupportedNetwork(f"DENSE {layer.id} weight shape mismatch")
    if bias_np.size != out_width // blocks:
        raise _UnsupportedNetwork(f"DENSE {layer.id} bias shape mismatch")

    local_out = weight_np.shape[0]
    for block in range(blocks):
        input_vars = layer.in_vars[block * in_width : (block + 1) * in_width]
        for row in range(local_out):
            output_var = layer.out_vars[block * local_out + row]
            lhs = variables[plan.variable_index[output_var]]
            rhs = _sum(
                gp,
                variables,
                (plan.variable_index[var] for var in input_vars),
                weight_np[row],
            ) + float(bias_np[row])
            model.addConstr(lhs == rhs, name=f"dense_{layer.id}_{block}_{row}")


def _pair(value: Any, default: int) -> tuple[int, int]:
    if value is None:
        return default, default
    if isinstance(value, int):
        return value, value
    pair = tuple(int(item) for item in value)
    if len(pair) != 2:
        raise _UnsupportedNetwork(f"expected a pair, got {value!r}")
    return pair


def _add_conv2d(gp: Any, model: Any, variables: Any, plan: _ModelPlan, layer: Layer) -> None:
    weight = layer.params.get("weight")
    input_shape = layer.params.get("input_shape")
    if not isinstance(weight, torch.Tensor) or input_shape is None:
        raise _UnsupportedNetwork(f"CONV2D {layer.id} lacks weight/input_shape")
    weight_np = weight.detach().cpu().numpy().astype(np.float64)
    bias = layer.params.get("bias")
    bias_np = (
        np.zeros(weight_np.shape[0], dtype=np.float64)
        if bias is None
        else bias.detach().cpu().numpy().astype(np.float64).reshape(-1)
    )
    shape = tuple(int(dim) for dim in input_shape)
    if len(shape) != 4:
        raise _UnsupportedNetwork(f"CONV2D {layer.id} input_shape must be NCHW")
    _, in_channels, in_height, in_width = shape
    out_channels, channels_per_group, kernel_height, kernel_width = weight_np.shape
    groups = int(layer.params.get("groups", 1))
    if in_channels != channels_per_group * groups or out_channels % groups:
        raise _UnsupportedNetwork(f"CONV2D {layer.id} has invalid groups")
    stride_h, stride_w = _pair(layer.params.get("stride"), 1)
    pad_h, pad_w = _pair(layer.params.get("padding"), 0)
    dilation_h, dilation_w = _pair(layer.params.get("dilation"), 1)
    out_height = (
        in_height + 2 * pad_h - dilation_h * (kernel_height - 1) - 1
    ) // stride_h + 1
    out_width = (
        in_width + 2 * pad_w - dilation_w * (kernel_width - 1) - 1
    ) // stride_w + 1
    if len(layer.out_vars) != out_channels * out_height * out_width:
        raise _UnsupportedNetwork(f"CONV2D {layer.id} output size mismatch")

    outputs_per_group = out_channels // groups
    for output_channel in range(out_channels):
        group = output_channel // outputs_per_group
        for output_row in range(out_height):
            for output_col in range(out_width):
                positions: list[int] = []
                coefficients: list[float] = []
                for local_channel in range(channels_per_group):
                    input_channel = group * channels_per_group + local_channel
                    for kernel_row in range(kernel_height):
                        input_row = output_row * stride_h - pad_h + kernel_row * dilation_h
                        if input_row < 0 or input_row >= in_height:
                            continue
                        for kernel_col in range(kernel_width):
                            input_col = output_col * stride_w - pad_w + kernel_col * dilation_w
                            if input_col < 0 or input_col >= in_width:
                                continue
                            flat_input = (
                                (input_channel * in_height + input_row) * in_width
                                + input_col
                            )
                            positions.append(plan.variable_index[layer.in_vars[flat_input]])
                            coefficients.append(
                                weight_np[
                                    output_channel,
                                    local_channel,
                                    kernel_row,
                                    kernel_col,
                                ]
                            )
                flat_output = (
                    (output_channel * out_height + output_row) * out_width + output_col
                )
                output = variables[plan.variable_index[layer.out_vars[flat_output]]]
                rhs = _sum(gp, variables, positions, coefficients) + float(
                    bias_np[output_channel]
                )
                model.addConstr(
                    output == rhs,
                    name=f"conv2d_{layer.id}_{output_channel}_{output_row}_{output_col}",
                )


def _add_relu(
    gp: Any,
    grb: Any,
    model: Any,
    variables: Any,
    plan: _ModelPlan,
    layer: Layer,
    before: dict[int, Fact],
) -> None:
    lower, upper = _relu_partition(layer, before)
    binaries = []
    for index, (input_id, output_id) in enumerate(zip(layer.in_vars, layer.out_vars)):
        input_var = variables[plan.variable_index[input_id]]
        output_var = variables[plan.variable_index[output_id]]
        if lower[index] >= 0.0:
            model.addConstr(output_var == input_var, name=f"relu_on_{layer.id}_{index}")
        elif upper[index] <= 0.0:
            model.addConstr(output_var == 0.0, name=f"relu_off_{layer.id}_{index}")
        else:
            phase = model.addVar(vtype=grb.BINARY, name=f"relu_phase_{layer.id}_{index}")
            binaries.append(phase)
            model.addConstr(output_var >= input_var, name=f"relu_lb_{layer.id}_{index}")
            model.addConstr(
                output_var <= input_var - float(lower[index]) * (1.0 - phase),
                name=f"relu_big_m_lower_{layer.id}_{index}",
            )
            model.addConstr(
                output_var <= float(upper[index]) * phase,
                name=f"relu_big_m_upper_{layer.id}_{index}",
            )


def _add_add(model: Any, variables: Any, plan: _ModelPlan, net: Net, layer: Layer) -> None:
    x_vars = layer.params.get("x_vars")
    y_vars = layer.params.get("y_vars")
    if not isinstance(x_vars, list) or not isinstance(y_vars, list):
        predecessors = net.preds.get(layer.id, [])
        if len(predecessors) != 2:
            raise _UnsupportedNetwork(f"ADD {layer.id} requires two operands")
        x_vars = net.by_id[predecessors[0]].out_vars
        y_vars = net.by_id[predecessors[1]].out_vars
    if not (len(x_vars) == len(y_vars) == len(layer.out_vars)):
        raise _UnsupportedNetwork(f"ADD {layer.id} broadcasting is not supported")
    for index, (output_id, x_id, y_id) in enumerate(
        zip(layer.out_vars, x_vars, y_vars)
    ):
        model.addConstr(
            variables[plan.variable_index[output_id]]
            == variables[plan.variable_index[x_id]] + variables[plan.variable_index[y_id]],
            name=f"add_{layer.id}_{index}",
        )


def _add_flatten(model: Any, variables: Any, plan: _ModelPlan, layer: Layer) -> None:
    if len(layer.in_vars) != len(layer.out_vars):
        raise _UnsupportedNetwork(f"FLATTEN {layer.id} changes element count")
    for index, (output_id, input_id) in enumerate(zip(layer.out_vars, layer.in_vars)):
        model.addConstr(
            variables[plan.variable_index[output_id]]
            == variables[plan.variable_index[input_id]],
            name=f"flatten_{layer.id}_{index}",
        )


def _add_input_polytopes(
    gp: Any, model: Any, variables: Any, plan: _ModelPlan, net: Net
) -> None:
    from act.back_end.verifier import get_input_ids

    input_ids = get_input_ids(net)
    for layer in net.layers:
        if layer.kind != LayerKind.INPUT_SPEC.value or layer.params.get("kind") != InKind.LIN_POLY:
            continue
        matrix = layer.params["A"].detach()
        rhs = layer.params["b"].detach()
        if matrix.dim() == 3 and matrix.shape[0] == 1:
            matrix = matrix[0]
        if rhs.dim() == 2 and rhs.shape[0] == 1:
            rhs = rhs[0]
        matrix_np = matrix.cpu().numpy().astype(np.float64)
        rhs_np = rhs.cpu().numpy().astype(np.float64).reshape(-1)
        if matrix_np.shape != (rhs_np.size, len(input_ids)):
            raise _UnsupportedNetwork("LIN_POLY A/b shape mismatch")
        positions = [plan.variable_index[var_id] for var_id in input_ids]
        for row, bound in enumerate(rhs_np):
            model.addConstr(
                _sum(gp, variables, positions, matrix_np[row]) <= float(bound),
                name=f"input_polytope_{layer.id}_{row}",
            )


def _add_unsafe_polytope(
    gp: Any, model: Any, variables: Any, plan: _ModelPlan, net: Net
) -> None:
    from act.back_end.verifier import get_assert_layer, get_output_ids

    assert_layer = get_assert_layer(net)
    coefficients, thresholds = _property_rows(assert_layer)
    output_ids = get_output_ids(net)
    if coefficients.shape[1] != len(output_ids):
        raise _UnsupportedNetwork("UNSAFE_LINEAR output width mismatch")
    positions = [plan.variable_index[var_id] for var_id in output_ids]
    for row, threshold in enumerate(thresholds):
        model.addConstr(
            _sum(gp, variables, positions, coefficients[row]) <= float(threshold),
            name=f"unsafe_polytope_{row}",
        )


def _validate_candidate(net: Net, candidate: torch.Tensor) -> bool:
    from act.back_end.bab.violation import (
        _check_input_specs_batched,
        check_violations_batched,
    )
    from act.back_end.verifier import gather_input_spec_layers, get_assert_layer

    input_valid = _check_input_specs_batched(candidate, gather_input_spec_layers(net))
    if not bool(input_valid.all().item()):
        return False
    violated = check_violations_batched(net, candidate, get_assert_layer(net))
    return bool(violated.all().item())


@torch.no_grad()
def escalate_milp(
    net: Net,
    *,
    timelimit: Optional[float] = 60.0,
    max_variables: int = 2000,
    max_constraints: int = 2000,
    threads: int = 1,
    output_flag: bool = False,
) -> VerifyResult:
    """Exactly solve one small ReLU ACT network against its joint unsafe set.

    The function is deliberately standalone so a later verifier/CLI task can
    call it behind a default-off escalation flag. Unsupported networks, batched
    properties, license-sized models, solver failures, and unvalidated
    incumbents all produce ``UNKNOWN``.
    """
    if max_variables <= 0 or max_constraints <= 0:
        return _unknown("model_size_limit", max_variables=max_variables, max_constraints=max_constraints)
    if not is_gurobi_available():
        return _unknown("gurobi_unavailable")

    try:
        input_box = _input_bounds(net)
        before, after = _root_facts(net, input_box)
        plan = _build_plan(net, before, after)
    except (KeyError, TypeError, ValueError, NotImplementedError) as error:
        return _unknown("unsupported_network", detail=str(error))

    size_metadata = {
        "variables": plan.variable_count,
        "continuous_variables": len(plan.variable_ids),
        "binary_variables": plan.ambiguous_relus,
        "constraints": plan.constraint_count,
        "max_variables": max_variables,
        "max_constraints": max_constraints,
    }
    if plan.variable_count > max_variables or plan.constraint_count > max_constraints:
        return _unknown("model_size_limit", **size_metadata)

    setup_gurobi_license()
    try:
        import gurobipy as gp
        from gurobipy import GRB

        environment = gp.Env(empty=True)
        environment.setParam("OutputFlag", int(output_flag))
        environment.start()
        model = gp.Model("act_whole_network_milp", env=environment)
        model.Params.OutputFlag = int(output_flag)
        model.Params.Threads = max(1, int(threads))
        model.Params.NumericFocus = 3
        model.Params.FeasibilityTol = 1e-9
        model.Params.IntFeasTol = 1e-9
        model.Params.OptimalityTol = 1e-9
        model.Params.MIPGap = 0.0
        model.Params.MIPGapAbs = 0.0
        if timelimit is not None:
            model.Params.TimeLimit = max(0.0, float(timelimit))

        variables = model.addMVar(
            len(plan.variable_ids),
            lb=plan.lower,
            ub=plan.upper,
            name="network",
        )
        for layer_id in get_topo_order(net):
            layer = net.by_id[layer_id]
            if layer.kind == LayerKind.DENSE.value:
                _add_dense(gp, model, variables, plan, layer)
            elif layer.kind == LayerKind.CONV2D.value:
                _add_conv2d(gp, model, variables, plan, layer)
            elif layer.kind == LayerKind.RELU.value:
                _add_relu(gp, GRB, model, variables, plan, layer, before)
            elif layer.kind == LayerKind.ADD.value:
                _add_add(model, variables, plan, net, layer)
            elif layer.kind == LayerKind.FLATTEN.value:
                _add_flatten(model, variables, plan, layer)
        _add_input_polytopes(gp, model, variables, plan, net)
        _add_unsafe_polytope(gp, model, variables, plan, net)
        model.setObjective(0.0, GRB.MINIMIZE)
        model.update()

        actual_variables = int(model.NumVars)
        actual_constraints = int(model.NumConstrs)
        size_metadata.update(
            actual_variables=actual_variables,
            actual_constraints=actual_constraints,
        )
        if actual_variables > max_variables or actual_constraints > max_constraints:
            model.dispose()
            environment.dispose()
            return _unknown("model_size_limit", **size_metadata)

        model.optimize()
        status = int(model.Status)
        size_metadata["gurobi_status"] = status
        size_metadata["solution_count"] = int(model.SolCount)
        if status == GRB.INFEASIBLE:
            result = VerifyResult(
                VerifyStatus.CERTIFIED,
                metadata={"solver": "gurobi_milp", **size_metadata},
            )
        elif model.SolCount > 0:
            from act.back_end.verifier import get_input_ids

            input_ids = get_input_ids(net)
            flat_input = np.asarray(
                [variables.X[plan.variable_index[var_id]] for var_id in input_ids],
                dtype=np.float64,
            )
            candidate = torch.from_numpy(flat_input).to(
                device=input_box.lb.device, dtype=input_box.lb.dtype
            ).reshape_as(input_box.lb)
            if _validate_candidate(net, candidate):
                result = VerifyResult(
                    VerifyStatus.FALSIFIED,
                    counterexample=candidate[0].detach().cpu().clone(),
                    metadata={"solver": "gurobi_milp", **size_metadata},
                )
            else:
                result = _unknown("candidate_validation_failed", **size_metadata)
        else:
            result = _unknown("solver_inconclusive", **size_metadata)
        model.dispose()
        environment.dispose()
        return result
    except Exception as error:
        return _unknown(
            "gurobi_error",
            detail=f"{type(error).__name__}: {error}",
            **size_metadata,
        )
