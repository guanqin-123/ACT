#===- act/back_end/hybridz_tf/tf_transformer.py - HybridZ Transformer TF -====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   HybridZ transfer functions for transformer operators.
#
#===---------------------------------------------------------------------===#

import torch

try:
    import numpy as np
    import scipy.sparse as sp
except ImportError:
    np = None
    sp = None

import act.back_end.interval_tf.tf_transformer as interval
from act.back_end.core import Bounds, Fact
from act.back_end.hybridz_tf.tf_mlp import _broadcast_flat
from act.back_end.solver.solver_hz import (
    hz_add_const,
    hz_bounds_are_liftable,
    hz_lift_bounds,
    sparse_hz_add_const,
    sparse_hz_lift_bounds,
)
from act.back_end.utils import scale_interval


def _dense_lift(
    L,
    fact,
    tf,
    output_equalities=None,
    equality_rhs=None,
    output_inequalities=None,
    inequality_rhs=None,
):
    hz = tf._hz_cache.get(L.id)
    if hz is None or not hz_bounds_are_liftable(fact.bounds):
        tf._hz_cache.pop(L.id, None)
        return fact
    out = hz_lift_bounds(
        hz,
        fact.bounds,
        output_equalities=output_equalities,
        equality_rhs=equality_rhs,
        output_inequalities=output_inequalities,
        inequality_rhs=inequality_rhs,
    )
    if max(out.Gc.shape[1] + out.Gb.shape[1], out.c.shape[0]) > tf._HZ_MAX_INPUT_DIM:
        tf._hz_cache.pop(L.id, None)
    else:
        tf._hz_cache[L.id] = out
    return fact


def _sparse_simplex(rowsize: int, n_out: int):
    if sp is None or rowsize <= 0 or n_out % rowsize:
        return None, None
    groups = n_out // rowsize
    rows = np.repeat(np.arange(groups, dtype=np.int64), rowsize)
    cols = np.arange(n_out, dtype=np.int64)
    matrix = sp.csr_matrix(
        (np.ones(n_out, dtype=np.float64), (rows, cols)),
        shape=(groups, n_out),
    )
    return matrix, np.ones(groups, dtype=np.float64)


def _softmax_ratio_inequalities(bounds: Bounds, rowsize: int):
    if sp is None or rowsize <= 1 or bounds.lb.numel() % rowsize:
        return None, None
    lb = bounds.lb.detach().cpu().double().numpy().reshape(-1, rowsize)
    ub = bounds.ub.detach().cpu().double().numpy().reshape(-1, rowsize)
    matrix_rows, matrix_cols, matrix_data = [], [], []
    row = 0
    for group in range(lb.shape[0]):
        if rowsize <= 8:
            pairs = ((i, j) for i in range(rowsize) for j in range(i + 1, rowsize))
        else:
            ref = int(np.argmax((lb[group] + ub[group]) * 0.5))
            pairs = ((i, ref) for i in range(rowsize) if i != ref)
        offset = group * rowsize
        for i, j in pairs:
            lower_delta = lb[group, i] - ub[group, j]
            upper_delta = ub[group, i] - lb[group, j]
            lower = (
                np.nextafter(np.exp(lower_delta), 0.0)
                if -745.0 < lower_delta <= np.log(1e12)
                else 0.0
            )
            upper = np.inf
            if upper_delta <= np.log(1e12):
                upper = np.nextafter(
                    np.exp(upper_delta) if upper_delta > -745.0 else 0.0,
                    np.inf,
                )
            if np.isfinite(upper):
                matrix_rows.extend((row, row))
                matrix_cols.extend((offset + i, offset + j))
                matrix_data.extend((1.0, -upper))
                row += 1
            if lower > 0.0:
                matrix_rows.extend((row, row))
                matrix_cols.extend((offset + i, offset + j))
                matrix_data.extend((-1.0, lower))
                row += 1
    if row == 0:
        return None, None
    matrix = sp.csr_matrix(
        (matrix_data, (matrix_rows, matrix_cols)),
        shape=(row, bounds.lb.numel()),
        dtype=np.float64,
    )
    return matrix, np.zeros(row, dtype=np.float64)


def _sparse_lift(
    L,
    hz,
    fact,
    tf,
    *,
    simplex_rowsize=None,
    output_equalities=None,
    equality_rhs=None,
    output_inequalities=None,
    inequality_rhs=None,
):
    if not hz_bounds_are_liftable(fact.bounds):
        return None, "nonfinite_sparse_transformer_bounds"
    reservation = tf._sparse_cont_slots_for(hz, L.id, fact.bounds.lb.numel())
    if reservation is None:
        return None, "sparse_transformer_size_limit"
    slots, n_cont = reservation
    equalities, rhs = output_equalities, equality_rhs
    if simplex_rowsize is not None:
        equalities, rhs = _sparse_simplex(
            int(simplex_rowsize), fact.bounds.lb.numel()
        )
    elif isinstance(equalities, torch.Tensor):
        equalities = sp.csr_matrix(equalities.detach().cpu().double().numpy())
        rhs = rhs.detach().cpu().double().numpy()
    inequalities, upper = output_inequalities, inequality_rhs
    if isinstance(inequalities, torch.Tensor):
        inequalities = sp.csr_matrix(
            inequalities.detach().cpu().double().numpy()
        )
        upper = upper.detach().cpu().double().numpy()
    return (
        sparse_hz_lift_bounds(
            hz,
            fact.bounds,
            slots,
            n_cont,
            output_equalities=equalities,
            equality_rhs=rhs,
            output_inequalities=inequalities,
            inequality_rhs=upper,
        ),
        None,
    )


def _softmax_box(bounds: Bounds, rowsize) -> Bounds:
    if rowsize is None or not hz_bounds_are_liftable(bounds):
        return Bounds(torch.zeros_like(bounds.lb), torch.ones_like(bounds.ub))
    lb = bounds.lb.reshape(-1, rowsize)
    ub = bounds.ub.reshape(-1, rowsize)
    eye = torch.eye(rowsize, dtype=torch.bool, device=lb.device).unsqueeze(0)
    other_ub = ub.unsqueeze(1).expand(-1, rowsize, -1).masked_fill(
        eye, float("-inf")
    )
    other_lb = lb.unsqueeze(1).expand(-1, rowsize, -1).masked_fill(
        eye, float("-inf")
    )
    lower_den = torch.logaddexp(lb, torch.logsumexp(other_ub, dim=-1))
    upper_den = torch.logaddexp(ub, torch.logsumexp(other_lb, dim=-1))
    return Bounds(
        torch.exp(lb - lower_den).reshape_as(bounds.lb),
        torch.exp(ub - upper_den).reshape_as(bounds.ub),
    )


def _layernorm_box(L, bounds: Bounds) -> Bounds:
    gamma = L.params["gamma"].flatten().to(bounds.lb)
    beta = L.params["beta"].flatten().to(bounds.lb)
    width = int(gamma.numel())
    if width == 0 or bounds.lb.shape[-1] % width:
        raise ValueError("LayerNorm parameters do not divide the flattened input")
    lb = bounds.lb.reshape(-1, width)
    ub = bounds.ub.reshape(-1, width)
    centered_lb = lb - ub.mean(dim=-1, keepdim=True)
    centered_ub = ub - lb.mean(dim=-1, keepdim=True)
    if L.params.get("variant", L.params.get("layer_norm", "standard")) == "no_var":
        norm_lb, norm_ub = centered_lb, centered_ub
    else:
        max_abs = torch.maximum(centered_lb.abs(), centered_ub.abs())
        variance_ub = (max_abs * max_abs).mean(dim=-1, keepdim=True)
        eps = bounds.lb.new_tensor(float(L.params.get("eps", 1e-5)))
        norm_lb, norm_ub = scale_interval(
            centered_lb,
            centered_ub,
            torch.rsqrt(variance_ub + eps),
            torch.rsqrt(eps).expand_as(variance_ub),
        )
    out_lb = torch.where(gamma >= 0, gamma * norm_lb + beta, gamma * norm_ub + beta)
    out_ub = torch.where(gamma >= 0, gamma * norm_ub + beta, gamma * norm_lb + beta)
    return Bounds(out_lb.reshape_as(bounds.lb), out_ub.reshape_as(bounds.ub))


def _layernorm_equalities(L, fact):
    gamma = L.params.get("gamma")
    beta = L.params.get("beta")
    if not isinstance(gamma, torch.Tensor) or not isinstance(beta, torch.Tensor):
        return None, None
    width = int(gamma.numel())
    total = int(fact.bounds.lb.numel())
    if width == 0 or total % width:
        return None, None
    gamma = gamma.flatten().to(fact.bounds.lb)
    beta = beta.flatten().to(fact.bounds.lb)
    if bool((gamma.abs() <= torch.finfo(gamma.dtype).eps).any()):
        return None, None
    weights = gamma.reciprocal()
    groups = total // width
    matrix = fact.bounds.lb.new_zeros((groups, total))
    cols = torch.arange(total, device=matrix.device).view(groups, width)
    matrix.scatter_(1, cols, weights.expand(groups, -1))
    return matrix, (weights * beta).sum().expand(groups)


def tf_posenc(L, bounds, tf):
    fact = interval.tf_posenc(L, bounds)
    hz = tf._hz_cache.get(L.id)
    if hz is not None:
        tf._hz_cache[L.id] = hz_add_const(
            hz, _broadcast_flat(L.params["pos_vec"], hz.c.shape[0])
        )
    return fact


def tf_layernorm(L, bounds, tf):
    fact = interval.tf_layernorm(L, bounds)
    fact = Fact(bounds=_layernorm_box(L, bounds), cons=fact.cons)
    fact.cons.add_box(L.id, L.out_vars, fact.bounds)
    equalities, rhs = _layernorm_equalities(L, fact)
    return _dense_lift(L, fact, tf, equalities, rhs)


def tf_gelu(L, bounds, tf):
    return _dense_lift(L, interval.tf_gelu(L, bounds), tf)


def tf_att_scores(L, bounds, tf):
    return interval.tf_att_scores(L,
        tf._before[L.params["q_src"]].bounds,
        tf._before[L.params["k_src"]].bounds)


def tf_softmax(L, bounds, tf):
    fact = interval.tf_softmax(L, bounds)
    rowsize = interval.softmax_rowsize(L, bounds)
    fact = Fact(bounds=_softmax_box(bounds, rowsize), cons=fact.cons)
    fact.cons.add_box(L.id, L.out_vars, fact.bounds)
    hz = tf._hz_cache.get(L.id)
    equalities = rhs = inequalities = upper = None
    if hz is not None and rowsize is not None:
        sparse_equalities, sparse_rhs = _sparse_simplex(
            rowsize, fact.bounds.lb.numel()
        )
        if sparse_equalities is not None:
            equalities = torch.from_numpy(sparse_equalities.toarray()).to(hz.c)
            rhs = torch.from_numpy(sparse_rhs).to(hz.c)
        sparse_inequalities, sparse_upper = _softmax_ratio_inequalities(
            bounds, rowsize
        )
        if sparse_inequalities is not None:
            inequalities = torch.from_numpy(sparse_inequalities.toarray()).to(hz.c)
            upper = torch.from_numpy(sparse_upper).to(hz.c)
    return _dense_lift(
        L, fact, tf, equalities, rhs, inequalities, upper
    )


def tf_att_mix(L, bounds, tf):
    return interval.tf_att_mix(L,
        tf._before[L.params["w_src"]].bounds,
        tf._before[L.params["v_src"]].bounds)


def tf_mha_split(L, bounds, tf):
    return interval.tf_mha_split(L, bounds)


def tf_mha_join(L, bounds, tf):
    return interval.tf_mha_join(L,
        tf._net.get_all_predecessor_bounds(L.id, tf._after, tf._before))


def tf_mask_add(L, bounds, tf):
    fact = interval.tf_mask_add(L, bounds)
    hz = tf._hz_cache.get(L.id)
    if hz is not None:
        tf._hz_cache[L.id] = hz_add_const(
            hz, _broadcast_flat(L.params["M"], hz.c.shape[0])
        )
    return fact


def sparse_hz_apply_layer(L, hz, input_bounds, result, tf):
    kind = L.kind.upper()
    if kind in {"POSENC", "MASK_ADD"}:
        key = "pos_vec" if kind == "POSENC" else "M"
        return True, sparse_hz_add_const(
            hz, _broadcast_flat(L.params[key], hz.n_out)
        ), None
    if kind == "SOFTMAX":
        rowsize = interval.softmax_rowsize(L, input_bounds)
        inequalities, upper = (
            _softmax_ratio_inequalities(input_bounds, rowsize)
            if rowsize is not None
            else (None, None)
        )
        out, reason = _sparse_lift(
            L,
            hz,
            result,
            tf,
            simplex_rowsize=rowsize,
            output_inequalities=inequalities,
            inequality_rhs=upper,
        )
        return True, out, reason
    if kind == "LAYERNORM":
        equalities, rhs = _layernorm_equalities(L, result)
        out, reason = _sparse_lift(
            L, hz, result, tf, output_equalities=equalities, equality_rhs=rhs
        )
        return True, out, reason
    if kind == "GELU":
        out, reason = _sparse_lift(L, hz, result, tf)
        return True, out, reason
    return False, None, None
