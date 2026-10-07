#===- act/back_end/dual_tf/tf_forward.py - Forward Bounds ----------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
"""Batched DAG-aware forward bound propagation for DualTF.

Per-layer linear state lives in ``lin_state`` and frame state in ``frame_dict``.
Traversal uses Kahn's algorithm over ``net.preds`` / ``net.succs``. ADD and
CONCAT read predecessor state explicitly for fan-out / fan-in DAGs such as
ResNet skips. Returned bounds stay batch-first flattened ``[B, n]`` tensors,
with activation bounds stored PRE-activation unless ``post_activation=True``.
"""
#===---------------------------------------------------------------------===#

# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false, reportMissingParameterType=false, reportUntypedFunctionDecorator=false, reportDeprecated=false

import contextvars
import logging
from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Tuple, cast

from act.back_end.core import Bounds, Layer, Net, topological_sort
from act.back_end.layer_schema import LayerKind
from act.back_end.utils import pair_2d
from act.config.config import DualConfig
from act.util.device_manager import get_default_device, get_default_dtype

log = logging.getLogger(__name__)


@dataclass
class LinearBound:
    A_lb: Optional[torch.Tensor]
    b_lb: torch.Tensor
    A_ub: Optional[torch.Tensor]
    b_ub: torch.Tensor

Frame = Tuple[torch.Tensor, torch.Tensor]  # (x_L, x_U) over which lin is defined


# Above this layer input dim, ``_fwd_X`` returns ``None`` to skip the
# dense linear-bound A matrix; caller ``forward_X`` falls back to
# interval-only via ``_box_X`` + ``_reset_forward_box``. Sound (interval
# is always sound), looser bounds at the bailed layer.
_DENSE_LIN_BOUND_MAX_DIM: int = 10_000


_LIN_BOUND_FREE_FRACTION: float = 0.8
_LIN_BOUND_MAX_MIB: float = 1024.0

# Peak transient working set of one lane of the symbolic composition, as a
# multiple of that lane's persistent coefficient pair
# ``2 * n_out * n_sym * itemsize``. Measured as ``max_memory_allocated()``
# deltas over the three helpers on toy CUDA layers (float32, B=4): CONV2D 2.38
# (3->64 channels, VGG conv1 shape) and 3.45 (64->64), DENSE 2.25 (widening)
# and 4.02 (square), MAXPOOL2D 2.51 — each rounded up to the next integer.
# Allocations that do NOT scale with the lane count sit outside this model and
# no lane split can shrink them: ``W.abs()`` plus the einsum's weight copies
# for DENSE (they follow ``n_out * n_in``, and dominate whenever
# ``n_in >> n_out``), and the im2col buffer for spatially tiny convolutions.
_LIN_BOUND_CHUNK_TRANSIENT_CONV2D: int = 4
_LIN_BOUND_CHUNK_TRANSIENT_DENSE: int = 4
_LIN_BOUND_CHUNK_TRANSIENT_MAXPOOL2D: int = 3

_LIN_BOUND_REPORTED: set[Tuple[int, str]] = set()


def _report_lin_bound_once(layer_id: int, mode: str, level: int, msg: str, *args: object) -> None:
    """Emit ``msg`` the first time a ``(layer id, mode)`` pair degrades.

    Forward bounds are recomputed for every BaB node, so an unconditional log
    call here produces one line per node per layer.
    """
    key = (layer_id, mode)
    if key in _LIN_BOUND_REPORTED:
        return
    _LIN_BOUND_REPORTED.add(key)
    log.log(level, msg, *args)


def _lin_bound_lane_chunk(B: int, n_out: int, n_sym: int, ref: torch.Tensor,
                          k_transient: int, layer_id: int, mode: str) -> Optional[int]:
    """Plan how many lanes a symbolic ``[B, n_out, n_sym]`` layer may do at once.

    Returns ``B`` when the whole batch fits ~80% of the free CUDA memory (the
    unguarded path, bit-for-bit unchanged), a chunk size ``1 <= c < B`` when the
    batch must be split lane-wise, or ``None`` when even a single lane's working
    set overruns the budget or the concatenated result cannot fit at all — the
    caller then returns ``None`` and ``forward_X`` degrades that layer to the
    interval box. CPU allocations are never split.

    Chunking only moves the *transient* peak: the result ``[B, n_out, n_sym]``
    pair is concatenated back and stays resident either way, so it is checked
    against the full free memory separately.
    """
    pair_bytes = 2 * n_out * n_sym * ref.element_size()
    if pair_bytes <= 0:
        return B
    persistent = B * pair_bytes
    cap_bytes = int(_LIN_BOUND_MAX_MIB * 2 ** 20)
    if persistent > cap_bytes:
        _report_lin_bound_once(
            layer_id, mode, logging.WARNING,
            "forward linear bound %.2f GiB persistent pair exceeds deterministic "
            "%.2f GiB cap for [%d, %d, %d]; falling back to interval for this layer",
            persistent / 2 ** 30, _LIN_BOUND_MAX_MIB / 1024.0,
            B, n_out, n_sym,
        )
        return None
    if not ref.is_cuda:
        return B
    free_bytes, _ = torch.cuda.mem_get_info(ref.device)
    budget = free_bytes * _LIN_BOUND_FREE_FRACTION
    if persistent <= budget:
        return B

    chunk = min(B, int(budget // (pair_bytes * k_transient)))
    if chunk >= 1 and persistent <= free_bytes:
        _report_lin_bound_once(
            layer_id, mode, logging.INFO,
            "forward linear bound splits %d lanes into chunks of %d for a "
            "[%d, %d, %d] coefficient pair (%.2f GiB resident, %.2f GiB CUDA free)",
            B, chunk, B, n_out, n_sym, persistent / 2 ** 30, free_bytes / 2 ** 30,
        )
        return chunk

    _report_lin_bound_once(
        layer_id, mode, logging.WARNING,
        "forward linear bound needs ~%.2f GiB for a [%d, %d, %d] coefficient pair "
        "(%.3f GiB for a single lane) with only %.2f GiB CUDA free; falling back "
        "to interval for this layer",
        persistent / 2 ** 30, B, n_out, n_sym,
        pair_bytes * k_transient / 2 ** 30, free_bytes / 2 ** 30,
    )
    return None


def _lin_lanes(lin: LinearBound, start: int, end: int) -> LinearBound:
    """View of lanes ``[start, end)`` — slices, so no copy is made."""
    return LinearBound(
        A_lb=None if lin.A_lb is None else lin.A_lb[start:end],
        b_lb=lin.b_lb[start:end],
        A_ub=None if lin.A_ub is None else lin.A_ub[start:end],
        b_ub=lin.b_ub[start:end],
    )


def _cat_lanes(parts: List[LinearBound]) -> LinearBound:
    """Reassemble lane chunks along the batch axis."""
    return LinearBound(
        A_lb=None if parts[0].A_lb is None else torch.cat(
            [cast(torch.Tensor, part.A_lb) for part in parts], dim=0),
        b_lb=torch.cat([part.b_lb for part in parts], dim=0),
        A_ub=None if parts[0].A_ub is None else torch.cat(
            [cast(torch.Tensor, part.A_ub) for part in parts], dim=0),
        b_ub=torch.cat([part.b_ub for part in parts], dim=0),
    )


def _resolve_perturbation_norm(value: Any) -> float:
    """Normalize a spec ``p_norm`` field to a float, defaulting to ``inf``.

    LP_EMBEDDING carries the input perturbation norm ``p``; box / L_inf specs
    omit it. A missing field (``None``) maps to ``inf`` so every box path keeps
    the exact box concretization with zero behavior change.
    """
    if value is None:
        return float("inf")
    if isinstance(value, str):
        token = value.strip().lower()
        if token in ("inf", "+inf", "infinity", "linf", "l_inf"):
            return float("inf")
        return float(token)
    return float(value)


def _dual_norm_exponent(p: float) -> float:
    """Return the Hölder dual exponent ``q`` with ``1/p + 1/q = 1``.

    Used to evaluate the exact ``min`` of a linear form over an Lp input ball
    (``min_{‖δ‖_p ≤ ε} ν·δ = −ε‖ν‖_q``): p=inf→q=1 (box), p=2→q=2, p=1→q=inf.
    """
    if p == float("inf"):
        return 1.0
    if p == 1.0:
        return float("inf")
    return p / (p - 1.0)


def input_block_slices(params: Dict[str, Any], orig_shape: torch.Size, n: int
                       ) -> List[Tuple[int, int]]:
    """Coordinate ranges ``[start, end)`` of each per-word embedding block.

    Each word occupies ``embed_dim`` contiguous embedding coordinates, so the
    dual norm is taken over those ``D`` coordinates independently and word
    balls do not couple. Splitting every token (not only perturbed ones) is
    sound and format-agnostic: a non-perturbed token has zero half-width, so
    its block penalty ``‖0 ⊙ nu_block‖_q`` is exactly 0 — the box already
    encodes which coordinates carry width, so ``perturbed_positions`` (index
    list or bool mask, possibly per-sample) need not be parsed here. The block
    size is read from ``embed_dim`` or, failing that, the trailing
    ``[..., L, D]`` axis of the spec ``center`` (or of ``orig_shape``) when
    ``perturbed_positions`` flags an embedding spec. The spec ``center`` is
    preferred because forward state and stored input boxes are flattened to
    ``[B, L*D]``, whose trailing axis is not the embedding width. Otherwise the
    whole input is one Lp ball (e.g. an image L2 spec).
    """
    embed_dim = params.get("embed_dim")
    if embed_dim is None and params.get("perturbed_positions") is not None:
        center = params.get("center")
        if isinstance(center, torch.Tensor) and center.dim() >= 2:
            embed_dim = int(center.shape[-1])
        elif len(orig_shape) >= 2:
            embed_dim = int(orig_shape[-1])
    if embed_dim is None:
        return [(0, n)]
    d = int(embed_dim)
    if d <= 0 or n % d != 0:
        return [(0, n)]
    return [(i * d, (i + 1) * d) for i in range(n // d)]


def lp_ball_support(a: torch.Tensor, center: torch.Tensor, half: torch.Tensor,
                    blocks: Iterable[Tuple[int, int]], q: float,
                    block_eps: Optional[torch.Tensor] = None,
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Support of the per-block Lp ball along every coefficient row of ``a``.

    Returns ``(dot, penalty)`` with ``dot = a·center`` and
    ``penalty = Σ_block ‖radius_block ⊙ a_block‖_q`` over the last axis, so
    ``min_{x ∈ ball} a·x = dot − penalty`` and ``max = dot + penalty``. The
    radius is the half-width of the frame (``half``) or, for BaB input-split
    lanes, the explicit per-lane table ``block_eps[b, j]`` of block ``j``.
    ``center`` / ``half`` broadcast against ``a`` (``[B|1, 1, n]`` against
    ``[B, M, n]``, or ``[1, n]`` against ``[BM, n]``). This is the single dual-norm
    concretization shared by the backward input contribution, the forward
    frame concretization and the spec-row frame bounds.
    """
    dot = (center * a).sum(dim=-1)
    penalty = torch.zeros_like(dot)
    if block_eps is not None:
        eps_b = block_eps.to(device=a.device, dtype=a.dtype)
        for block_idx, (s, e) in enumerate(blocks):
            penalty = penalty + eps_b[:, block_idx].unsqueeze(1) * torch.linalg.vector_norm(
                a[..., s:e], ord=q, dim=-1
            )
    else:
        for s, e in blocks:
            penalty = penalty + torch.linalg.vector_norm(
                half[..., s:e] * a[..., s:e], ord=q, dim=-1
            )
    return dot, penalty


@dataclass(frozen=True)
class InputBall:
    """Finite-p Lp-ball structure attached to an entry frame w.r.t. the INPUT.

    ``x_L`` / ``x_U`` are the frame tensors the ball belongs to; a frame is
    recognised by tensor identity, so any handler that resets or re-derives its
    frame silently falls back to the box concretization (sound). ``blocks`` are
    the per-word ``[start, end)`` column ranges in FRAME coordinates, ``q`` the
    Hölder dual exponent of the spec norm, ``block_eps`` the optional per-lane
    radius table ``[B, len(blocks)]`` installed by BaB input splits (otherwise
    the radius is the frame half-width).
    """

    x_L: torch.Tensor
    x_U: torch.Tensor
    q: float
    blocks: Tuple[Tuple[int, int], ...]
    block_eps: Optional[torch.Tensor] = None

    def owns(self, x_L: torch.Tensor, x_U: torch.Tensor) -> bool:
        return x_L is self.x_L and x_U is self.x_U

    def rebind(self, x_L: torch.Tensor, x_U: torch.Tensor) -> "InputBall":
        return replace(self, x_L=x_L, x_U=x_U)

    def _support(self, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        center = ((self.x_L + self.x_U) * 0.5).unsqueeze(1)
        half = ((self.x_U - self.x_L) * 0.5).unsqueeze(1)
        return lp_ball_support(a, center, half, self.blocks, self.q, self.block_eps)

    def lower(self, a: torch.Tensor) -> torch.Tensor:
        """``min_{x ∈ ball} a·x`` for coefficients ``a`` of shape ``[B, M, n]``."""
        dot, penalty = self._support(a)
        return dot - penalty

    def upper(self, a: torch.Tensor) -> torch.Tensor:
        """``max_{x ∈ ball} a·x`` for coefficients ``a`` of shape ``[B, M, n]``."""
        dot, penalty = self._support(a)
        return dot + penalty


_ACTIVE_INPUT_BALL: contextvars.ContextVar[Optional[InputBall]] = contextvars.ContextVar(
    "act_dual_forward_input_ball", default=None
)


def _active_input_ball() -> Optional[InputBall]:
    return _ACTIVE_INPUT_BALL.get()


def _input_spec_params(net: Net) -> Dict[str, Any]:
    spec_params: Optional[Dict[str, Any]] = None
    input_params: Dict[str, Any] = {}
    for layer in net.layers:
        kind = layer.kind.upper() if isinstance(layer.kind, str) else layer.kind
        if kind == LayerKind.INPUT_SPEC.value:
            spec_params = layer.params
        elif kind == LayerKind.INPUT.value:
            input_params = layer.params
    return spec_params if spec_params is not None else input_params


def input_ball_for_frame(net: Net, lb_in: torch.Tensor, ub_in: torch.Tensor,
                         frame: Frame, orig_shape: Optional[torch.Size] = None,
                         ) -> Optional[InputBall]:
    """Build the finite-p ball of the entry frame, ``None`` for box / L_inf specs.

    ``frame`` is either the shrunk frame ``(lb_in[:, pert], ub_in[:, pert])`` of
    :func:`_entry_lin_frame` or the full box ``(lb_in, ub_in)``; the input blocks
    of :func:`input_block_slices` are mapped onto the frame columns (a word
    block that has no perturbed column is dropped, its penalty being zero).
    """
    params = _input_spec_params(net)
    p_norm = _resolve_perturbation_norm(params.get("p_norm"))
    if p_norm == float("inf"):
        return None
    x_L, x_U = frame
    n = lb_in.shape[1]
    input_blocks = input_block_slices(params, lb_in.shape if orig_shape is None else orig_shape, n)
    if x_L.shape[1] == n:
        frame_cols = torch.arange(n, device=lb_in.device)
    else:
        frame_cols = (ub_in > lb_in).any(dim=0).nonzero(as_tuple=True)[0]
    blocks: List[Tuple[int, int]] = []
    present: List[int] = []
    for token, (s, e) in enumerate(input_blocks):
        lo = int(torch.searchsorted(frame_cols, torch.tensor(s, device=frame_cols.device)).item())
        hi = int(torch.searchsorted(frame_cols, torch.tensor(e, device=frame_cols.device)).item())
        if hi > lo:
            blocks.append((lo, hi))
            present.append(token)
    block_eps = params.get("bab_block_eps")
    if isinstance(block_eps, torch.Tensor):
        block_eps = block_eps[:, present]
    else:
        block_eps = None
    return InputBall(
        x_L=x_L, x_U=x_U, q=_dual_norm_exponent(p_norm), blocks=tuple(blocks), block_eps=block_eps,
    )


def _concretize(lin: LinearBound, x_L: torch.Tensor, x_U: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Concretize dual-track affine bounds over a batched input box.

    Lazy identity: when ``A_lb is None`` the affine bound is the identity,
    so ``lb = x_L + b_lb`` (and analogously for ub). This avoids materializing
    a dense ``[B, n, n]`` eye matrix at the INPUT layer, which for 224×224×3
    input would request 181 GB.

    When the frame is the entry frame of a finite-p spec (the ball registered
    by the running forward pass owns ``x_L`` / ``x_U``), the explicit tracks
    are additionally concretized over the per-word Lp ball and intersected with
    the box result, so the output is never looser than the box. Box / L_inf
    specs register no ball and take exactly the box path.
    """
    lb, ub = _concretize_box(lin, x_L, x_U)
    ball = _active_input_ball()
    if ball is None or lin.A_lb is None or lin.A_ub is None or not ball.owns(x_L, x_U):
        return lb, ub
    return _intersect_boxes(
        ball.lower(lin.A_lb) + lin.b_lb, ball.upper(lin.A_ub) + lin.b_ub, lb, ub,
    )


def _concretize_box(lin: LinearBound, x_L: torch.Tensor, x_U: torch.Tensor
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
    n_x = x_L.shape[1]
    if lin.A_lb is None:
        b_lb = lin.b_lb
        if b_lb.shape[1] != n_x:
            n_b = min(b_lb.shape[1], n_x)
            lb = x_L[..., :n_b] + b_lb[..., :n_b]
        else:
            lb = x_L + b_lb
    else:
        A_lb_p = lin.A_lb.clamp(min=0)
        A_lb_n = lin.A_lb.clamp(max=0)
        lb = (
            torch.einsum("boi,bi->bo", A_lb_p, x_L)
            + torch.einsum("boi,bi->bo", A_lb_n, x_U)
            + lin.b_lb
        )
    if lin.A_ub is None:
        b_ub = lin.b_ub
        if b_ub.shape[1] != n_x:
            n_b = min(b_ub.shape[1], n_x)
            ub = x_U[..., :n_b] + b_ub[..., :n_b]
        else:
            ub = x_U + b_ub
    else:
        A_ub_p = lin.A_ub.clamp(min=0)
        A_ub_n = lin.A_ub.clamp(max=0)
        ub = (
            torch.einsum("boi,bi->bo", A_ub_p, x_U)
            + torch.einsum("boi,bi->bo", A_ub_n, x_L)
            + lin.b_ub
        )
    return lb, ub

def _identity_lin(B: int, n: int, device, dtype) -> LinearBound:
    """Identity ``LinearBound`` over n features.

    Lazy identity: A_lb / A_ub are returned as ``None`` (sentinel for
    identity). Materializing ``torch.eye(n)`` for 224×224×3 input = 181 GB
    OOM-killed the verifier; the sentinel avoids it entirely. Downstream
    ``_fwd_X`` helpers and ``_concretize`` recognise None as identity.
    """
    zeros = torch.zeros(B, n, device=device, dtype=dtype)
    return LinearBound(A_lb=None, b_lb=zeros, A_ub=None, b_ub=zeros.clone())

def _reset_lin(lb: torch.Tensor, ub: torch.Tensor, device, dtype
               ) -> Tuple[LinearBound, torch.Tensor, torch.Tensor]:
    B, n = lb.shape[0], lb.shape[1]
    return _identity_lin(B, n, device, dtype), lb.clone(), ub.clone()

def _entry_lin_frame(lb_in: torch.Tensor, ub_in: torch.Tensor,
                     max_perturbed: int, device, dtype
                     ) -> Tuple[LinearBound, Frame]:
    """Entry LinearBound/Frame, symbolic only in the perturbed input dims.

    ``pert`` is the union over the batch of dims with ``ub > lb``. When
    ``0 < n_pert <= max_perturbed``, the entry bound is the explicit one-hot
    selection ``A[:, i, k] = 1`` iff input dim ``i`` is the ``k``-th perturbed
    dim; unperturbed dims fold into the constant term ``b`` and the frame
    shrinks to the perturbed columns. Downstream ``_fwd_X`` handlers then see
    ``A is not None`` and never take the ``_DENSE_LIN_BOUND_MAX_DIM`` bail,
    so e.g. VGG16 with <=100 perturbed pixels keeps linear bounds at every
    conv. Otherwise (``n_pert == 0`` or above the cap) fall back to the lazy
    identity over all dims, preserving the previous behaviour exactly.
    """
    B, input_dim = lb_in.shape
    pert = (ub_in > lb_in).any(dim=0)
    n_pert = int(pert.sum().item())
    if n_pert == 0 or n_pert > max_perturbed:
        return _identity_lin(B, input_dim, device, dtype), (lb_in, ub_in)
    selection = torch.zeros(input_dim, n_pert, device=device, dtype=dtype)
    selection[pert.nonzero(as_tuple=True)[0], torch.arange(n_pert, device=device)] = 1.0
    A = selection.unsqueeze(0).expand(B, input_dim, n_pert)
    b = torch.where(pert, torch.zeros_like(lb_in), lb_in)
    lin = LinearBound(A_lb=A, b_lb=b, A_ub=A, b_ub=b.clone())
    return lin, (lb_in[:, pert], ub_in[:, pert])

def _match_lin_input_dim(lin: LinearBound, n_in: int) -> LinearBound:
    """Pad or truncate the current output-feature axis to size n_in.

    Invariant: ``A_lb`` and ``A_ub`` are paired — either both ``None``
    (identity sentinel) or both ``Tensor``. We check both for explicit
    type narrowing.
    """
    if lin.A_lb is None or lin.A_ub is None:
        curr_out = lin.b_lb.shape[1]
    else:
        curr_out = lin.A_lb.shape[1]
    if curr_out == n_in:
        return lin

    B = lin.b_lb.shape[0]
    if lin.A_lb is None or lin.A_ub is None:
        if curr_out < n_in:
            pad = n_in - curr_out
            zeros_b = torch.zeros(B, pad, device=lin.b_lb.device, dtype=lin.b_lb.dtype)
            return LinearBound(
                A_lb=None,
                b_lb=torch.cat([lin.b_lb, zeros_b], dim=1),
                A_ub=None,
                b_ub=torch.cat([lin.b_ub, zeros_b.clone()], dim=1),
            )
        return LinearBound(
            A_lb=None,
            b_lb=lin.b_lb[:, :n_in],
            A_ub=None,
            b_ub=lin.b_ub[:, :n_in],
        )

    input_dim = lin.A_lb.shape[2]
    if curr_out < n_in:
        pad = n_in - curr_out
        zeros_A = torch.zeros(B, pad, input_dim, device=lin.A_lb.device, dtype=lin.A_lb.dtype)
        zeros_b = torch.zeros(B, pad, device=lin.b_lb.device, dtype=lin.b_lb.dtype)
        return LinearBound(
            A_lb=torch.cat([lin.A_lb, zeros_A], dim=1),
            b_lb=torch.cat([lin.b_lb, zeros_b], dim=1),
            A_ub=torch.cat([lin.A_ub, zeros_A.clone()], dim=1),
            b_ub=torch.cat([lin.b_ub, zeros_b.clone()], dim=1),
        )

    return LinearBound(
        A_lb=lin.A_lb[:, :n_in, :],
        b_lb=lin.b_lb[:, :n_in],
        A_ub=lin.A_ub[:, :n_in, :],
        b_ub=lin.b_ub[:, :n_in],
    )

def _intersect_boxes(lb_a: torch.Tensor, ub_a: torch.Tensor,
                     lb_b: torch.Tensor, ub_b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Intersect two sound boxes; EVERY crossing degrades to box ``b`` per entry.

    Both boxes enclose the same true value set, so in exact arithmetic they
    overlap and ``max(lbs) <= min(ubs)``. In float32 the linear-track
    concretization (box ``a``) and the interval track (box ``b``) are
    independent accumulations; when the true width is below the accumulation
    noise the pair can cross (RC2) — on layers whose partial sums dwarf the
    cancelled result (late VGG convs / FC, partials 1e5-1e7 vs results
    O(1-100)) by a small multiple of the PARTIAL-sum ulp. Any crossed entry,
    whatever its magnitude, falls back to box ``b`` verbatim (the
    centre-radius interval track, ordered by construction), which is sound
    and merely looser. A tolerance-gated "repair" that swaps the crossed pair
    to ``[min(ubs), max(lbs)]`` is UNSOUND: that interval lies between the two
    boxes and may contain points of neither — e.g. true range [1.0, 2.0],
    interval box [1.0001, 2.0001], linear box [0.0, 1.0] (1 ulp low at 1e5
    partial sums) swapped to [1.0, 1.0001], excluding the true value 2.0.
    Genuine producer bugs that emit inverted boxes directly remain visible to
    the downstream degenerate-interval check.
    """
    lb = torch.maximum(lb_a, lb_b)
    ub = torch.minimum(ub_a, ub_b)
    crossed = lb > ub
    out_lb = torch.where(crossed, lb_b, lb)
    out_ub = torch.where(crossed, ub_b, ub)
    return out_lb, out_ub

def _align_batch(a: torch.Tensor, n: int) -> torch.Tensor:
    if a.shape[1] == n:
        return a
    if a.shape[1] > n:
        return a[:, :n]
    repeats = (n + a.shape[1] - 1) // a.shape[1]
    return a.repeat(1, repeats)[:, :n]

def _shape_list(shape_param: object) -> Optional[list[int]]:
    return [int(v) for v in shape_param] if isinstance(shape_param, (tuple, list)) else None

def _int_param(value: object, default: int) -> int:
    if isinstance(value, bool):
        return int(value)
    return value if isinstance(value, int) else default


def _box_dense(layer: Layer, lb: torch.Tensor, ub: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Interval DENSE as ``m = c Wᵀ + b``, ``rho = r |W|ᵀ``, ``[m-rho, m+rho]``.

    ``rho >= 0`` structurally, so ``ub - lb = 2*rho >= 0`` per neuron whatever
    the rounding; the ``W⁺l + W⁻u`` / ``W⁺u + W⁻l`` form sums two independent
    accumulations that float32 cancellation can order the wrong way.
    """
    W = layer.params["weight"]
    b = layer.params.get("bias")
    lb = _align_batch(lb, W.shape[1])
    ub = _align_batch(ub, W.shape[1])
    centre = (lb + ub) * 0.5
    radius = (ub - lb) * 0.5
    mid = centre @ W.T
    spread = radius @ W.abs().T
    if b is not None:
        mid = mid + _align(b.flatten(), W.shape[0])
    return mid - spread, mid + spread


def _box_bias(layer: Layer, lb: torch.Tensor, ub: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    c = _align(layer.params["c"], lb.shape[1])
    return lb + c, ub + c


def _box_scale(layer: Layer, lb: torch.Tensor, ub: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    a = _align(layer.params["a"], lb.shape[1])
    out_lb = torch.where(a >= 0, a * lb, a * ub)
    out_ub = torch.where(a >= 0, a * ub, a * lb)
    return out_lb, out_ub


def _box_bn(layer: Layer, lb: torch.Tensor, ub: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    A_bn = _align(layer.params["A"], lb.shape[1])
    c = _align(layer.params["c"], lb.shape[1])
    out_lb = torch.where(A_bn >= 0, A_bn * lb + c, A_bn * ub + c)
    out_ub = torch.where(A_bn >= 0, A_bn * ub + c, A_bn * lb + c)
    return out_lb, out_ub


def _box_relu(lb: torch.Tensor, ub: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact interval propagation for ReLU."""
    return lb.clamp(min=0), ub.clamp(min=0)


def _box_lrelu(lb: torch.Tensor, ub: torch.Tensor, alpha: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact interval propagation for LeakyReLU."""
    alpha_tensor = torch.full_like(lb, alpha)
    out_lb = torch.where(lb >= 0, lb, alpha_tensor * lb)
    out_ub = torch.where(ub <= 0, alpha_tensor * ub, ub)
    return out_lb, out_ub


def _store_forward_state(bounds_dict: Dict[int, Bounds],
                          box_state: Dict[int, Bounds],
                          lin_state: Dict[int, LinearBound],
                          frame_dict: Dict[int, Frame],
                          layer_id: int,
                          stored: Bounds,
                          out_box: Bounds,
                          lin: LinearBound,
                          frame: Frame) -> None:
    """Store public bounds plus internal forward state for a layer."""
    bounds_dict[layer_id] = stored.copy()
    box_state[layer_id] = out_box.copy()
    lin_state[layer_id] = lin
    frame_dict[layer_id] = frame


def _reset_forward_box(lb: torch.Tensor, ub: torch.Tensor, device, dtype
                       ) -> Tuple[LinearBound, Tuple[torch.Tensor, torch.Tensor]]:
    """Reset dual-track state on a concrete box."""
    lin, x_L, x_U = _reset_lin(lb, ub, device, dtype)
    return lin, (x_L, x_U)


def _sum_interval_bounds(boxes: List[Bounds]) -> Bounds:
    """Sum predecessor boxes element-wise, trimming to the smallest width."""
    lbs = [box.lb.flatten(start_dim=1) for box in boxes]
    ubs = [box.ub.flatten(start_dim=1) for box in boxes]
    n = min(lb.shape[1] for lb in lbs)
    lbs = [lb[:, :n] for lb in lbs]
    ubs = [ub[:, :n] for ub in ubs]
    return Bounds(sum(lbs[1:], lbs[0]), sum(ubs[1:], ubs[0]))


def _sum_linear_bounds(lins: List[LinearBound]) -> LinearBound:
    """Sum dual-track affine bounds from multiple predecessors.

    Lazy identity: ``A = None`` denotes identity. Summing identities is
    a diagonal=k·I matrix; we still materialize only when at least one
    predecessor has a non-None A (i.e. is no longer identity). If all are
    None, result remains None (single-identity case is impossible because
    callers only sum from >=2 predecessors).
    """
    A_lbs = [lin.A_lb for lin in lins]
    A_ubs = [lin.A_ub for lin in lins]

    def _materialize(A: Optional[torch.Tensor], example: torch.Tensor) -> torch.Tensor:
        if A is not None:
            return A
        B, n = example.shape
        eye = torch.eye(n, device=example.device, dtype=example.dtype)
        return eye.unsqueeze(0).expand(B, n, n)

    if all(A is None for A in A_lbs):
        A_lb_sum = None
    else:
        any_real = next(A for A in A_lbs if A is not None)
        A_lb_sum = sum((_materialize(A, lins[i].b_lb) for i, A in enumerate(A_lbs[1:], 1)),
                       _materialize(A_lbs[0], lins[0].b_lb))

    if all(A is None for A in A_ubs):
        A_ub_sum = None
    else:
        any_real = next(A for A in A_ubs if A is not None)
        A_ub_sum = sum((_materialize(A, lins[i].b_ub) for i, A in enumerate(A_ubs[1:], 1)),
                       _materialize(A_ubs[0], lins[0].b_ub))

    return LinearBound(
        A_lb=A_lb_sum,
        b_lb=sum((lin.b_lb for lin in lins[1:]), lins[0].b_lb),
        A_ub=A_ub_sum,
        b_ub=sum((lin.b_ub for lin in lins[1:]), lins[0].b_ub),
    )


@dataclass(frozen=True)
class ForwardFrame:
    """Explicit forward linear frame of one layer's output.

    ``lin`` bounds the layer output ``y`` by ``A_lb x + b_lb <= y <= A_ub x + b_ub``
    for every symbolic input ``x`` in the box ``[x_L, x_U]`` (the perturbed
    input dims, see :func:`_entry_lin_frame`). The relation is valid for every
    input of the box it was computed on, so concretizing it over any sub-box is
    sound. Only explicit frames (``A_lb is not None``) are wrapped; a lazy
    identity frame carries no information beyond the stored box. ``ball`` is
    the finite-p :class:`InputBall` when the frame is the entry frame of an
    LP_EMBEDDING spec (``None`` for box / L_inf specs and for reset frames).
    """

    lid: int
    lin: LinearBound
    x_L: torch.Tensor
    x_U: torch.Tensor
    ball: Optional[InputBall] = None


def _canonical_input_box(
    input_lb: torch.Tensor, input_ub: torch.Tensor,
    device: torch.device, dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalize an input box exactly as the forward pass does."""
    if (
        input_lb.dtype != dtype or input_lb.device != device
        or input_ub.dtype != dtype or input_ub.device != device
    ):
        input_lb = input_lb.to(device=device, dtype=dtype)
        input_ub = input_ub.to(device=device, dtype=dtype)
    if input_lb.dim() < 2:
        input_lb = input_lb.unsqueeze(0)
        input_ub = input_ub.unsqueeze(0)
    batch_size = input_lb.shape[0]
    return input_lb.reshape(batch_size, -1), input_ub.reshape(batch_size, -1)


def _identical_input_lanes(
    input_lb: torch.Tensor, input_ub: torch.Tensor, net: Optional[Net] = None,
) -> int:
    """Return the batch size when every lane is bitwise identical, else one.

    BaB input splits of a finite-p spec keep every lane's box and only install
    a per-lane radius table ``bab_block_eps``; such lanes differ, so they are
    never shared (a one-lane frame would meet per-lane ball boxes downstream).
    """
    if input_lb.dim() < 2 or input_lb.shape[0] <= 1:
        return 1
    lanes = int(input_lb.shape[0])
    if input_ub.shape[0] != lanes:
        return 1
    if net is not None and isinstance(_input_spec_params(net).get("bab_block_eps"), torch.Tensor):
        return 1
    same_lb = torch.equal(input_lb, input_lb[:1].expand_as(input_lb))
    same_ub = torch.equal(input_ub, input_ub[:1].expand_as(input_ub))
    return lanes if same_lb and same_ub else 1


def _expand_bounds_lanes(bounds_dict: Dict[int, Bounds], lanes: int) -> Dict[int, Bounds]:
    """Broadcast single-lane public boxes without allocating lane copies."""
    if lanes == 1:
        return bounds_dict
    return {
        lid: Bounds(
            bounds.lb.expand(lanes, *bounds.lb.shape[1:]),
            bounds.ub.expand(lanes, *bounds.ub.shape[1:]),
        )
        for lid, bounds in bounds_dict.items()
    }


def _clone_cache_tensor(
    tensor: torch.Tensor, memo: Dict[int, torch.Tensor],
) -> torch.Tensor:
    """Clone a cache tensor once while preserving zero-stride broadcast views."""
    cached = memo.get(id(tensor))
    if cached is not None:
        return cached
    base = tensor._base
    if base is not None and 0 in tensor.stride():
        base_copy = _clone_cache_tensor(base, memo)
        relative_offset = tensor.storage_offset() - base.storage_offset()
        if relative_offset >= 0 and base_copy.stride() == base.stride():
            try:
                copied = base_copy.as_strided(tensor.shape, tensor.stride(), relative_offset)
            except RuntimeError:
                copied = tensor.clone()
        else:
            copied = tensor.clone()
    else:
        copied = tensor.clone()
    memo[id(tensor)] = copied
    return copied


def _clone_linear_bound(
    lin: LinearBound, memo: Optional[Dict[int, torch.Tensor]] = None,
) -> LinearBound:
    """Clone every tensor in a linear bound for private cache ownership."""
    tensor_memo = {} if memo is None else memo
    return LinearBound(
        A_lb=None if lin.A_lb is None else _clone_cache_tensor(lin.A_lb, tensor_memo),
        b_lb=_clone_cache_tensor(lin.b_lb, tensor_memo),
        A_ub=None if lin.A_ub is None else _clone_cache_tensor(lin.A_ub, tensor_memo),
        b_ub=_clone_cache_tensor(lin.b_ub, tensor_memo),
    )


_MAXPOOL_CACHE_KEYS = (
    "maxpool_argmax_flat", "maxpool_dominant", "maxpool_lb", "maxpool_ub",
)


class ForwardPrefixCache:
    """Immutable alpha-independent forward state for one optimization loop.

    The cache owns states of every layer that is *not* reachable from any
    ``alpha_relu_ids`` entry. :func:`compute_forward_bounds` can then seed the
    normal state maps from this object and execute only the ReLU descendants in
    the original topological order. No transfer function or arithmetic order is
    changed.

    The key is the ``net`` object, normalized input-box values, alpha-carrying
    ReLU id set, ``post_activation``, ``forward_lin_max_perturbed``, and current
    forward device/dtype. Every resume validates that key and raises
    ``ValueError`` on a mismatch. Cached tensors are privately cloned once and
    are only ever read by resumed passes. Returned public bounds are fresh
    copies, and cached MAXPOOL2D metadata is restored before each resume so an
    intervening forward on the same ``Net`` cannot alter the certificate.

    Example::

        cache = ForwardPrefixCache(
            net, input_lb, input_ub,
            alpha_relu_ids=forward_alphas.keys(),
            post_activation=False,
            forward_lin_max_perturbed=dual_config.forward_lin_max_perturbed,
        )
        for forward_alphas in alpha_steps:
            bounds = compute_forward_bounds(
                net, input_lb, input_ub, alphas=forward_alphas,
                forward_lin_max_perturbed=dual_config.forward_lin_max_perturbed,
                prefix_cache=cache,
            )
    """

    @torch.no_grad()
    def __init__(
        self,
        net: Net,
        input_lb: torch.Tensor,
        input_ub: torch.Tensor,
        *,
        alpha_relu_ids: Iterable[int],
        post_activation: bool = False,
        forward_lin_max_perturbed: Optional[int] = None,
    ) -> None:
        if forward_lin_max_perturbed is None:
            forward_lin_max_perturbed = DualConfig().forward_lin_max_perturbed
        device, dtype = get_default_device(), get_default_dtype()
        lb_key, ub_key = _canonical_input_box(input_lb, input_ub, device, dtype)
        alpha_ids = tuple(sorted({int(lid) for lid in alpha_relu_ids}))
        by_id = getattr(net, "by_id", {layer.id: layer for layer in net.layers})
        invalid = [
            lid for lid in alpha_ids
            if lid not in by_id or by_id[lid].kind.upper() != LayerKind.RELU.value
        ]
        if invalid:
            raise ValueError(
                "ForwardPrefixCache alpha_relu_ids must name ReLU layers; "
                f"invalid ids {invalid}"
            )

        descendants = set(alpha_ids)
        stack = list(alpha_ids)
        while stack:
            for successor in net.succs.get(stack.pop(), []):
                if successor not in descendants:
                    descendants.add(successor)
                    stack.append(successor)
        topo_order = topological_sort(net)
        independent = frozenset(lid for lid in topo_order if lid not in descendants)

        box_state: Dict[int, Bounds] = {}
        bounds_dict, lin_state, frame_dict, entry_ball = _forward_pass(
            net,
            lb_key,
            ub_key,
            post_activation,
            None,
            forward_lin_max_perturbed,
            _active_lids=independent,
            _box_state_out=box_state,
        )

        tensor_copies: Dict[int, torch.Tensor] = {}
        frame_copies: Dict[int, Frame] = {}
        self._frame_dict: Dict[int, Frame] = {}
        self._input_ball: Optional[InputBall] = None
        for lid, frame in frame_dict.items():
            copied = frame_copies.get(id(frame))
            if copied is None:
                copied = (
                    _clone_cache_tensor(frame[0], tensor_copies),
                    _clone_cache_tensor(frame[1], tensor_copies),
                )
                frame_copies[id(frame)] = copied
            self._frame_dict[lid] = copied
            if entry_ball is not None and entry_ball.owns(*frame):
                self._input_ball = entry_ball.rebind(*copied)
        self._bounds_dict: Dict[int, Bounds] = {
            lid: Bounds(
                _clone_cache_tensor(bounds.lb, tensor_copies),
                _clone_cache_tensor(bounds.ub, tensor_copies),
            )
            for lid, bounds in bounds_dict.items()
        }
        self._box_state: Dict[int, Bounds] = {
            lid: Bounds(
                _clone_cache_tensor(bounds.lb, tensor_copies),
                _clone_cache_tensor(bounds.ub, tensor_copies),
            )
            for lid, bounds in box_state.items()
        }
        self._lin_state: Dict[int, LinearBound] = {
            lid: _clone_linear_bound(lin, tensor_copies) for lid, lin in lin_state.items()
        }
        self._layer_cache_state: Dict[int, Dict[str, Optional[torch.Tensor]]] = {}
        for lid in independent:
            layer = by_id[lid]
            if layer.kind.upper() != LayerKind.MAXPOOL2D.value:
                continue
            self._layer_cache_state[lid] = {
                key: (
                    value.clone() if isinstance(value := layer.cache.get(key), torch.Tensor)
                    else None
                )
                for key in _MAXPOOL_CACHE_KEYS
            }

        self._net: Net = net
        self._input_lb: torch.Tensor = lb_key.clone()
        self._input_ub: torch.Tensor = ub_key.clone()
        self._alpha_relu_ids: Tuple[int, ...] = alpha_ids
        self._post_activation: bool = post_activation
        self._forward_lin_max_perturbed: int = forward_lin_max_perturbed
        self._device: torch.device = device
        self._dtype: torch.dtype = dtype
        self._recompute_lids: FrozenSet[int] = frozenset(descendants)
        self.cached_layer_ids: FrozenSet[int] = independent

    def _validate_key(
        self,
        net: Net,
        input_lb: torch.Tensor,
        input_ub: torch.Tensor,
        post_activation: bool,
        alphas: Optional[Dict[int, torch.Tensor]],
        forward_lin_max_perturbed: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Validate a resume request and return its normalized box/settings."""
        if forward_lin_max_perturbed is None:
            forward_lin_max_perturbed = DualConfig().forward_lin_max_perturbed
        device, dtype = get_default_device(), get_default_dtype()
        lb_key, ub_key = _canonical_input_box(input_lb, input_ub, device, dtype)
        alpha_ids = tuple(sorted(int(lid) for lid in (alphas or {})))
        matches = (
            net is self._net
            and device == self._device
            and dtype == self._dtype
            and post_activation == self._post_activation
            and forward_lin_max_perturbed == self._forward_lin_max_perturbed
            and alpha_ids == self._alpha_relu_ids
            and torch.equal(lb_key, self._input_lb)
            and torch.equal(ub_key, self._input_ub)
        )
        if not matches:
            raise ValueError("ForwardPrefixCache key mismatch")
        return lb_key, ub_key, forward_lin_max_perturbed

    @torch.no_grad()
    def _restore_layer_cache_state(self) -> None:
        """Restore forward side state needed by cached layers' backward handlers."""
        for lid, state in self._layer_cache_state.items():
            cache = self._net.by_id[lid].cache
            for key, value in state.items():
                if value is None:
                    cache.pop(key, None)
                else:
                    cache[key] = value.clone()

    @torch.no_grad()
    def _resume(
        self,
        net: Net,
        input_lb: torch.Tensor,
        input_ub: torch.Tensor,
        post_activation: bool,
        alphas: Optional[Dict[int, torch.Tensor]],
        forward_lin_max_perturbed: Optional[int],
        retain_lin_lids: FrozenSet[int],
    ) -> Tuple[Dict[int, Bounds], Dict[int, LinearBound], Dict[int, Frame], Optional[InputBall]]:
        """Resume the standard pass from the immutable cached state."""
        lb_key, ub_key, resolved_cap = self._validate_key(
            net, input_lb, input_ub, post_activation, alphas,
            forward_lin_max_perturbed,
        )
        self._restore_layer_cache_state()
        public_bounds = {lid: bounds.copy() for lid, bounds in self._bounds_dict.items()}
        return _forward_pass(
            net,
            lb_key,
            ub_key,
            post_activation,
            alphas,
            resolved_cap,
            _initial_state=(
                public_bounds,
                self._box_state,
                self._lin_state,
                self._frame_dict,
            ),
            _initial_ball=self._input_ball,
            _active_lids=self._recompute_lids,
            _retain_lin_lids=retain_lin_lids,
        )


@torch.no_grad()
def compute_forward_bounds(net: Net, input_lb: torch.Tensor, input_ub: torch.Tensor,
                           post_activation: bool = False,
                           alphas: Optional[Dict[int, torch.Tensor]] = None,
                           forward_lin_max_perturbed: Optional[int] = None,
                           prefix_cache: Optional[ForwardPrefixCache] = None,
                           ) -> Dict[int, Bounds]:
    """Forward bounds, natively batched with singleton auto-promotion.

    ``forward_lin_max_perturbed`` caps the number of perturbed input dims for
    which the linear track stays symbolic (see :func:`_entry_lin_frame`);
    ``None`` resolves to the DualConfig default at call time rather than being
    frozen at import time.
    """
    if prefix_cache is None:
        shared_lanes = _identical_input_lanes(input_lb, input_ub, net) if alphas is None else 1
        run_lb = input_lb[:1] if shared_lanes > 1 else input_lb
        run_ub = input_ub[:1] if shared_lanes > 1 else input_ub
        bounds_dict, _lin_state, _frame_dict, _ball = _forward_pass(
            net, run_lb, run_ub, post_activation, alphas, forward_lin_max_perturbed,
            _retain_lin_lids=frozenset(),
        )
        bounds_dict = _expand_bounds_lanes(bounds_dict, shared_lanes)
    else:
        bounds_dict, _lin_state, _frame_dict, _ball = prefix_cache._resume(
            net, input_lb, input_ub, post_activation, alphas, forward_lin_max_perturbed,
            frozenset(),
        )
    return bounds_dict


@torch.no_grad()
def compute_forward_bounds_with_frame(
    net: Net, input_lb: torch.Tensor, input_ub: torch.Tensor,
    *,
    frame_lid: int,
    post_activation: bool = False,
    alphas: Optional[Dict[int, torch.Tensor]] = None,
    forward_lin_max_perturbed: Optional[int] = None,
    prefix_cache: Optional[ForwardPrefixCache] = None,
) -> Tuple[Dict[int, Bounds], Optional[ForwardFrame]]:
    """Forward bounds plus the explicit linear frame of layer ``frame_lid``.

    Same arithmetic as :func:`compute_forward_bounds`; additionally returns the
    :class:`ForwardFrame` of ``frame_lid`` when its forward state carries an
    explicit frame, else ``None`` (the stored box is then the best forward
    information for that layer).
    """
    if prefix_cache is None:
        shared_lanes = _identical_input_lanes(input_lb, input_ub, net) if alphas is None else 1
        run_lb = input_lb[:1] if shared_lanes > 1 else input_lb
        run_ub = input_ub[:1] if shared_lanes > 1 else input_ub
        bounds_dict, lin_state, frame_dict, entry_ball = _forward_pass(
            net, run_lb, run_ub, post_activation, alphas, forward_lin_max_perturbed,
            _retain_lin_lids=frozenset({frame_lid}),
        )
        bounds_dict = _expand_bounds_lanes(bounds_dict, shared_lanes)
    else:
        bounds_dict, lin_state, frame_dict, entry_ball = prefix_cache._resume(
            net, input_lb, input_ub, post_activation, alphas, forward_lin_max_perturbed,
            frozenset({frame_lid}),
        )
    frame: Optional[ForwardFrame] = None
    lin = lin_state.get(frame_lid)
    if lin is not None and lin.A_lb is not None and lin.A_ub is not None:
        x_L, x_U = frame_dict[frame_lid]
        ball = entry_ball if entry_ball is not None and entry_ball.owns(x_L, x_U) else None
        if prefix_cache is not None:
            lin = _clone_linear_bound(lin)
            x_L, x_U = x_L.clone(), x_U.clone()
            ball = None if ball is None else ball.rebind(x_L, x_U)
        frame = ForwardFrame(lid=frame_lid, lin=lin, x_L=x_L, x_U=x_U, ball=ball)
    return bounds_dict, frame


def forward_frame_row_lower_bounds(frame: ForwardFrame, c: torch.Tensor, M: int
                                   ) -> torch.Tensor:
    """Lower bound of every spec row ``c_i . y`` from an explicit output frame.

    ``c`` is ``[B*M, n_out]`` sample-major (row ``b*M+j`` belongs to sample
    ``b``) while the frame is ``[B, ...]`` or ``[1, ...]`` (broadcast over the
    samples). Each row is composed with the signed tracks,
    ``a = c+ A_lb + c- A_ub`` and ``k = c+ . b_lb + c- . b_ub``, so that
    ``c . y >= a . x + k`` on the frame box; concretizing ``a`` over
    ``[x_L, x_U]`` keeps the correlation between the output coordinates that a
    box difference of the logits loses.
    """
    A_lb = cast(torch.Tensor, frame.lin.A_lb)
    A_ub = cast(torch.Tensor, frame.lin.A_ub)
    BM = c.shape[0]
    if BM % M != 0:
        raise ValueError(f"forward_frame_row_lower_bounds: c rows {BM} not divisible by M={M}")
    B = BM // M
    frame_B = A_lb.shape[0]
    if frame_B not in (1, B):
        raise ValueError(
            f"forward_frame_row_lower_bounds: frame batch {frame_B} incompatible with B={B}"
        )
    c_view = c.view(B, M, -1).to(device=A_lb.device, dtype=A_lb.dtype)
    if c_view.shape[-1] != A_lb.shape[1]:
        raise ValueError(
            f"forward_frame_row_lower_bounds: c has {c_view.shape[-1]} outputs, "
            f"frame has {A_lb.shape[1]}"
        )
    c_pos = c_view.clamp(min=0)
    c_neg = c_view.clamp(max=0)
    a = torch.einsum("bmo,bos->bms", c_pos, A_lb) + torch.einsum("bmo,bos->bms", c_neg, A_ub)
    const = (
        torch.einsum("bmo,bo->bm", c_pos, frame.lin.b_lb)
        + torch.einsum("bmo,bo->bm", c_neg, frame.lin.b_ub)
    )
    x_L = frame.x_L.unsqueeze(1)
    x_U = frame.x_U.unsqueeze(1)
    lower = (a.clamp(min=0) * x_L).sum(dim=-1) + (a.clamp(max=0) * x_U).sum(dim=-1) + const
    if frame.ball is not None:
        lower = torch.maximum(lower, frame.ball.lower(a) + const)
    return lower.reshape(BM)


def _forward_pass(net: Net, input_lb: torch.Tensor, input_ub: torch.Tensor,
                  post_activation: bool,
                  alphas: Optional[Dict[int, torch.Tensor]],
                  forward_lin_max_perturbed: Optional[int],
                  *,
                  _initial_state: Optional[Tuple[
                      Dict[int, Bounds], Dict[int, Bounds],
                      Dict[int, LinearBound], Dict[int, Frame],
                  ]] = None,
                  _initial_ball: Optional[InputBall] = None,
                  _active_lids: Optional[FrozenSet[int]] = None,
                  _retain_lin_lids: Optional[FrozenSet[int]] = None,
                  _box_state_out: Optional[Dict[int, Bounds]] = None,
                  ) -> Tuple[Dict[int, Bounds], Dict[int, LinearBound], Dict[int, Frame], Optional[InputBall]]:
    """Run the forward pass; return stored bounds, per-layer lins / frames and the entry ball.

    The entry ball (finite-p specs only) is registered for the duration of the
    pass so that every ``_concretize`` of a frame w.r.t. the INPUT also uses the
    exact per-word dual norm; a resumed pass receives the cache's rebound ball.
    """
    # Lazy import to break circular dep (dual_tf imports compute_forward_bounds)
    from .dual_tf import DualTF

    if forward_lin_max_perturbed is None:
        forward_lin_max_perturbed = DualConfig().forward_lin_max_perturbed

    device, dtype = get_default_device(), get_default_dtype()
    lb_in, ub_in = _canonical_input_box(input_lb, input_ub, device, dtype)

    if _initial_state is None:
        bounds_dict: Dict[int, Bounds] = {}
        box_state: Dict[int, Bounds] = {}
        lin_state: Dict[int, LinearBound] = {}
        frame_dict: Dict[int, Frame] = {}
    else:
        initial_bounds, initial_boxes, initial_lins, initial_frames = _initial_state
        bounds_dict = dict(initial_bounds)
        box_state = dict(initial_boxes)
        lin_state = dict(initial_lins)
        frame_dict = dict(initial_frames)
    topo_order = topological_sort(net)
    by_id = getattr(net, "by_id", {layer.id: layer for layer in net.layers})
    entry_box = Bounds(lb_in, ub_in)
    entry_lin: Optional[LinearBound] = None
    entry_frame: Optional[Frame] = None
    entry_ball: Optional[InputBall] = _initial_ball
    if _initial_state is None:
        entry_lin, entry_frame = _entry_lin_frame(
            lb_in, ub_in, forward_lin_max_perturbed, device, dtype,
        )
        entry_ball = input_ball_for_frame(net, lb_in, ub_in, entry_frame, input_lb.shape)

    ball_token = _ACTIVE_INPUT_BALL.set(entry_ball)
    try:
        _run_forward_layers(
            net, topo_order, by_id, post_activation, alphas, device, dtype,
            entry_box, entry_lin, entry_frame, _active_lids,
            bounds_dict, box_state, lin_state, frame_dict, DualTF,
            _retain_lin_lids,
        )
    finally:
        _ACTIVE_INPUT_BALL.reset(ball_token)

    if _box_state_out is not None:
        _box_state_out.update(box_state)
    return bounds_dict, lin_state, frame_dict, entry_ball


def _run_forward_layers(net: Net, topo_order: List[int], by_id: Dict[int, Layer],
                        post_activation: bool, alphas: Optional[Dict[int, torch.Tensor]],
                        device: torch.device, dtype: torch.dtype,
                        entry_box: Bounds, entry_lin: Optional[LinearBound],
                        entry_frame: Optional[Frame], _active_lids: Optional[FrozenSet[int]],
                        bounds_dict: Dict[int, Bounds], box_state: Dict[int, Bounds],
                        lin_state: Dict[int, LinearBound], frame_dict: Dict[int, Frame],
                        DualTF: Any,
                        retain_lin_lids: Optional[FrozenSet[int]]) -> None:
    remaining_uses: Dict[int, int] = {}
    if retain_lin_lids is not None:
        active = set(topo_order) if _active_lids is None else set(_active_lids)
        for successor in active:
            for predecessor in net.preds.get(successor, []) or []:
                remaining_uses[predecessor] = remaining_uses.get(predecessor, 0) + 1

    def retire_predecessors(preds: List[int]) -> None:
        if retain_lin_lids is None:
            return
        for predecessor in preds:
            uses = remaining_uses.get(predecessor, 0) - 1
            remaining_uses[predecessor] = uses
            if uses <= 0 and predecessor not in retain_lin_lids:
                lin_state.pop(predecessor, None)
                frame_dict.pop(predecessor, None)

    for lid in topo_order:
        if _active_lids is not None and lid not in _active_lids:
            continue
        layer = by_id[lid]
        lid = layer.id
        kind = layer.kind.upper()
        preds = list(net.preds.get(lid, []) or [])

        if not preds:
            if entry_lin is None or entry_frame is None:
                raise ValueError(
                    f"compute_forward_bounds: resumed layer {lid} has no cached source state"
                )
            if kind == LayerKind.CONSTANT.value:
                # CONSTANT is a data-independent source; materialize its value
                # broadcast to batch B (taken from entry_box), no input frame.
                handler = DualTF._FORWARD_REGISTRY[kind]
                stored, out, lin, frame = handler(
                    layer, [entry_box], [], [], [], post_activation, device, dtype,
                )
                _store_forward_state(
                    bounds_dict, box_state, lin_state, frame_dict,
                    lid, stored, out, lin, frame,
                )
                continue
            if kind not in (LayerKind.INPUT.value, LayerKind.INPUT_SPEC.value):
                raise ValueError(f"compute_forward_bounds: layer {lid} kind '{kind}' has no predecessors and is not INPUT / INPUT_SPEC")
            _store_forward_state(
                bounds_dict,
                box_state,
                lin_state,
                frame_dict,
                lid,
                entry_box,
                entry_box,
                entry_lin,
                entry_frame,
            )
            continue

        missing = [pid for pid in preds if pid not in box_state or pid not in lin_state or pid not in frame_dict]
        if missing:
            raise ValueError(f"compute_forward_bounds: layer {lid} missing predecessor state for {missing}")

        if len(preds) >= 2 or kind in (LayerKind.ADD.value, LayerKind.CONCAT.value):
            handler = DualTF._FORWARD_REGISTRY.get(kind)
            if handler is None:
                raise ValueError(
                    f"compute_forward_bounds: unknown multi-pred layer kind '{kind}' "
                    f"at layer {lid}. Registered kinds: "
                    f"{sorted(DualTF._FORWARD_REGISTRY.keys())}"
                )
            pred_boxes  = [box_state[pid]   for pid in preds]
            pred_lins   = [lin_state[pid]   for pid in preds]
            pred_frames = [frame_dict[pid]  for pid in preds]
            stored, out, lin, frame = handler(
                layer, pred_boxes, pred_lins, pred_frames, preds,
                post_activation, device, dtype,
            )
            _store_forward_state(bounds_dict, box_state, lin_state, frame_dict,
                                 lid, stored, out, lin, frame)
            retire_predecessors(preds)
            continue

        if kind == LayerKind.RELU.value and alphas is not None:
            parent_box = box_state[preds[0]]
            parent_lin = lin_state[preds[0]]
            parent_frame = frame_dict[preds[0]]
            x_L, x_U = parent_frame
            pre_lb, pre_ub = parent_box.lb, parent_box.ub
            alpha = alphas.get(lid)
            new_lin = _fwd_relu(parent_lin, pre_lb, pre_ub, alpha=alpha)
            int_lb, int_ub = _box_relu(pre_lb, pre_ub)
            if new_lin is None:
                lb, ub = int_lb, int_ub
                out = Bounds(lb, ub)
                stored = out if post_activation else Bounds(pre_lb, pre_ub)
                lin, frame = _reset_forward_box(lb, ub, device, dtype)
            else:
                lin = new_lin
                lin_lb, lin_ub = _concretize(lin, x_L, x_U)
                lb, ub = _intersect_boxes(lin_lb, lin_ub, int_lb, int_ub)
                out = Bounds(lb, ub)
                stored = out if post_activation else Bounds(pre_lb, pre_ub)
                frame = parent_frame
                if post_activation:
                    lin, frame = _reset_forward_box(lb, ub, device, dtype)
            _store_forward_state(bounds_dict, box_state, lin_state, frame_dict,
                                 lid, stored, out, lin, frame)
            retire_predecessors(preds)
            continue

        handler = DualTF._FORWARD_REGISTRY.get(kind)
        if handler is None:
            raise ValueError(
                f"compute_forward_bounds: unknown layer kind '{kind}' at layer {lid}. "
                f"Registered kinds: {sorted(DualTF._FORWARD_REGISTRY.keys())}"
            )
        pred_boxes  = [box_state[preds[0]]]
        pred_lins   = [lin_state[preds[0]]]
        pred_frames = [frame_dict[preds[0]]]
        stored, out, lin, frame = handler(
            layer, pred_boxes, pred_lins, pred_frames, preds,
            post_activation, device, dtype,
        )
        _store_forward_state(bounds_dict, box_state, lin_state, frame_dict,
                             lid, stored, out, lin, frame)
        retire_predecessors(preds)

    if retain_lin_lids is not None:
        for layer_id in tuple(lin_state):
            if layer_id not in retain_lin_lids:
                lin_state.pop(layer_id, None)
                frame_dict.pop(layer_id, None)


def _fwd_dense(layer: Layer, lin: LinearBound) -> Optional[LinearBound]:
    """Compose dual-track affine bounds through a dense layer.

    Lazy identity: if input ``A_lb`` is None (identity), output A would
    be ``W`` broadcast to ``[B, out_c, in_c]``. For wide DENSE layers
    (e.g. simple_cnn FC1 with in_c=200704), this allocates several GB
    transiently (W_broadcast_lb, W_broadcast_ub, W_pos, W_neg). When
    ``in_c > _DENSE_LIN_BOUND_MAX_DIM``, we return ``None`` instead, signalling
    the caller to fall back to interval-only forward via
    :func:`_reset_forward_box`. Same pattern as :func:`_fwd_conv2d` and
    :func:`_fwd_relu`.

    Under CUDA memory pressure the batch is composed in lane chunks (see
    :func:`_lin_bound_lane_chunk`) and reassembled; interval fallback is kept
    only for the case where a single lane does not fit.
    """
    W = layer.params["weight"]
    lin = _match_lin_input_dim(lin, W.shape[1])
    B = lin.b_lb.shape[0]
    n_sym = W.shape[1] if lin.A_lb is None else lin.A_lb.shape[2]
    chunk = _lin_bound_lane_chunk(B, W.shape[0], n_sym, lin.b_lb,
                                  _LIN_BOUND_CHUNK_TRANSIENT_DENSE, layer.id, "dense")
    if chunk is None:
        return None
    if chunk >= B:
        return _fwd_dense_lanes(layer, lin)

    parts: List[LinearBound] = []
    for start in range(0, B, chunk):
        part = _fwd_dense_lanes(layer, _lin_lanes(lin, start, min(start + chunk, B)))
        if part is None:
            return None
        parts.append(part)
    return _cat_lanes(parts)


def _fwd_dense_lanes(layer: Layer, lin: LinearBound) -> Optional[LinearBound]:
    """Dense composition for one lane chunk; ``lin`` must already be dim-matched."""
    W = layer.params["weight"]
    b = layer.params.get("bias")
    B = lin.b_lb.shape[0]
    bias_vec = torch.zeros(W.shape[0], device=lin.b_lb.device, dtype=lin.b_lb.dtype)
    if b is not None:
        bias_vec = _align(b.flatten(), W.shape[0])

    # Centre-radius composition; see _fwd_conv2d for the identity and the
    # float32 cancellation rationale.
    b_centre = (lin.b_lb + lin.b_ub) * 0.5
    b_radius = (lin.b_ub - lin.b_lb) * 0.5
    b_mid = torch.einsum("oc,bc->bo", W, b_centre) + bias_vec
    b_spread = torch.einsum("oc,bc->bo", W.abs(), b_radius)

    if lin.A_lb is None and lin.A_ub is None:
        if W.shape[1] > _DENSE_LIN_BOUND_MAX_DIM:
            return None
        W_broadcast_lb = W.unsqueeze(0).expand(B, -1, -1).contiguous()
        W_broadcast_ub = W.unsqueeze(0).expand(B, -1, -1).contiguous()
        return LinearBound(A_lb=W_broadcast_lb, b_lb=b_mid - b_spread,
                           A_ub=W_broadcast_ub, b_ub=b_mid + b_spread)

    A_centre = (lin.A_lb + lin.A_ub) * 0.5
    A_radius = (lin.A_ub - lin.A_lb) * 0.5
    A_mid = torch.einsum("oc,bci->boi", W, A_centre)
    A_spread = torch.einsum("oc,bci->boi", W.abs(), A_radius)
    return LinearBound(
        A_lb=A_mid - A_spread,
        b_lb=b_mid - b_spread,
        A_ub=A_mid + A_spread,
        b_ub=b_mid + b_spread,
    )


def _fwd_relu(lin: LinearBound, lb: torch.Tensor, ub: torch.Tensor,
              alpha: Optional[torch.Tensor] = None
              ) -> Optional[LinearBound]:
    """Apply forward ReLU linear relaxation with per-batch alpha choice.

    Lazy identity: if input A is None (identity), output A becomes a
    diagonal scaling matrix (alpha or up_slope on the diagonal). For high-dim
    inputs this would materialize a sparse-on-diagonal dense tensor — we
    return ``None`` instead, signalling the caller to fall back to
    interval-only forward via :func:`_reset_forward_box`. ReLU after INPUT
    is rare in practice (networks start with DENSE/CONV); the common case
    (RELU after DENSE/CONV) hits the second branch with a materialized A_lb.
    """
    on = lb >= 0
    off = ub <= 0
    amb = ~(on | off)
    denom = (ub - lb).clamp(min=1e-12)
    up_slope = torch.where(
        amb,
        ub / denom,
        torch.where(on, torch.ones_like(lb), torch.zeros_like(lb)),
    )
    up_inter = torch.where(amb, -up_slope * lb, torch.zeros_like(lb))
    if alpha is None:
        low_slope = (ub > -lb).to(lb.dtype)
    else:
        low_slope = alpha.to(device=lb.device, dtype=lb.dtype)
    low_slope = torch.where(on, torch.ones_like(low_slope), low_slope)
    low_slope = torch.where(off, torch.zeros_like(low_slope), low_slope)

    if lin.A_lb is None or lin.A_ub is None:
        B, n = lb.shape
        if n > _DENSE_LIN_BOUND_MAX_DIM:
            return None
        eye = torch.eye(n, device=lb.device, dtype=lb.dtype).unsqueeze(0).expand(B, n, n)
        A_lb_in = lin.A_lb if lin.A_lb is not None else eye
        A_ub_in = lin.A_ub if lin.A_ub is not None else eye
        return LinearBound(
            A_lb=low_slope.unsqueeze(-1) * A_lb_in,
            b_lb=low_slope * lin.b_lb,
            A_ub=up_slope.unsqueeze(-1) * A_ub_in,
            b_ub=up_slope * lin.b_ub + up_inter,
        )

    return LinearBound(
        A_lb=low_slope.unsqueeze(-1) * lin.A_lb,
        b_lb=low_slope * lin.b_lb,
        A_ub=up_slope.unsqueeze(-1) * lin.A_ub,
        b_ub=up_slope * lin.b_ub + up_inter,
    )


def _fwd_elementwise_planes(
    lin: LinearBound,
    k_lo: torch.Tensor, b_lo: torch.Tensor,
    k_hi: torch.Tensor, b_hi: torch.Tensor,
) -> Optional[LinearBound]:
    """Compose explicit dual-track bounds through element-wise planes.

    Given ``k_lo x + b_lo <= f(x) <= k_hi x + b_hi`` on the layer's input box
    (``[B, n]`` coefficients, either sign), each plane picks the input track of
    matching sign, exactly as :func:`_fwd_scale` does for a signed scale.
    Returns ``None`` on the lazy-identity frame or a feature-width mismatch,
    which callers treat as "keep the interval-reset behaviour".
    """
    if lin.A_lb is None or lin.A_ub is None:
        return None
    if lin.A_lb.shape[1] != k_lo.shape[1] or lin.b_lb.shape[1] != k_lo.shape[1]:
        return None
    k_lo_p, k_lo_n = k_lo.clamp(min=0), k_lo.clamp(max=0)
    k_hi_p, k_hi_n = k_hi.clamp(min=0), k_hi.clamp(max=0)
    return LinearBound(
        A_lb=k_lo_p.unsqueeze(-1) * lin.A_lb + k_lo_n.unsqueeze(-1) * lin.A_ub,
        b_lb=k_lo_p * lin.b_lb + k_lo_n * lin.b_ub + b_lo,
        A_ub=k_hi_p.unsqueeze(-1) * lin.A_ub + k_hi_n.unsqueeze(-1) * lin.A_lb,
        b_ub=k_hi_p * lin.b_ub + k_hi_n * lin.b_lb + b_hi,
    )


def _fwd_rowwise_planes(
    lin: LinearBound, row: int,
    a_lo: torch.Tensor, c_lo: torch.Tensor,
    a_hi: torch.Tensor, c_hi: torch.Tensor,
) -> Optional[LinearBound]:
    """Compose explicit dual-track bounds through per-row matrix planes.

    The op acts independently on contiguous ``row``-sized feature blocks with
    planes ``a_lo x + c_lo <= f(x) <= a_hi x + c_hi`` per block (``a`` shaped
    ``[B*n_rows, row, row]``, ``c`` shaped ``[B*n_rows, row]``, as returned by
    the local vector relaxation). Signed entries pick the matching input track
    per coefficient. Returns ``None`` on the lazy-identity frame or a width
    mismatch.
    """
    if lin.A_lb is None or lin.A_ub is None:
        return None
    B, n, n_sym = lin.A_lb.shape
    if n % row != 0 or lin.b_lb.shape[1] != n or a_lo.shape[0] != B * (n // row):
        return None
    n_rows = n // row
    A_lb = lin.A_lb.reshape(B, n_rows, row, n_sym)
    A_ub = lin.A_ub.reshape(B, n_rows, row, n_sym)
    b_lb = lin.b_lb.reshape(B, n_rows, row)
    b_ub = lin.b_ub.reshape(B, n_rows, row)
    a_lo = a_lo.reshape(B, n_rows, row, row)
    a_hi = a_hi.reshape(B, n_rows, row, row)
    a_lo_p, a_lo_n = a_lo.clamp(min=0), a_lo.clamp(max=0)
    a_hi_p, a_hi_n = a_hi.clamp(min=0), a_hi.clamp(max=0)
    A_lb_new = (torch.einsum("brij,brjs->bris", a_lo_p, A_lb)
                + torch.einsum("brij,brjs->bris", a_lo_n, A_ub))
    b_lb_new = (torch.einsum("brij,brj->bri", a_lo_p, b_lb)
                + torch.einsum("brij,brj->bri", a_lo_n, b_ub)
                + c_lo.reshape(B, n_rows, row))
    A_ub_new = (torch.einsum("brij,brjs->bris", a_hi_p, A_ub)
                + torch.einsum("brij,brjs->bris", a_hi_n, A_lb))
    b_ub_new = (torch.einsum("brij,brj->bri", a_hi_p, b_ub)
                + torch.einsum("brij,brj->bri", a_hi_n, b_lb)
                + c_hi.reshape(B, n_rows, row))
    return LinearBound(
        A_lb=A_lb_new.reshape(B, n, n_sym), b_lb=b_lb_new.reshape(B, n),
        A_ub=A_ub_new.reshape(B, n, n_sym), b_ub=b_ub_new.reshape(B, n),
    )


def _fwd_bias(layer: Layer, lin: LinearBound) -> LinearBound:
    """Compose dual-track affine bounds through a bias layer.

    Bias is purely additive on b, so A_lb / A_ub pass through unchanged —
    including the None (identity) sentinel.
    """
    c = _align(layer.params["c"], lin.b_lb.shape[1])
    return LinearBound(
        A_lb=lin.A_lb,
        b_lb=lin.b_lb + c,
        A_ub=lin.A_ub,
        b_ub=lin.b_ub + c,
    )


def _fwd_scale(layer: Layer, lin: LinearBound) -> Optional[LinearBound]:
    """Compose dual-track affine bounds through an element-wise scale.

    Lazy identity: SCALE following INPUT is rare; if input A is None,
    materialize a diagonal eye·a representation. Common case (SCALE after
    DENSE/CONV) hits the second branch with materialized A_lb.
    """
    a = _align(layer.params["a"], lin.b_lb.shape[1])
    a_pos = a.clamp(min=0)
    a_neg = a.clamp(max=0)
    a_pos_A = a_pos.view(1, -1, 1)
    a_neg_A = a_neg.view(1, -1, 1)

    if lin.A_lb is None or lin.A_ub is None:
        B, n = lin.b_lb.shape
        if n > _DENSE_LIN_BOUND_MAX_DIM:
            return None
        eye = torch.eye(n, device=lin.b_lb.device, dtype=lin.b_lb.dtype).unsqueeze(0).expand(B, n, n)
        A_lb_in = lin.A_lb if lin.A_lb is not None else eye
        A_ub_in = lin.A_ub if lin.A_ub is not None else eye
        return LinearBound(
            A_lb=a_pos_A * A_lb_in + a_neg_A * A_ub_in,
            b_lb=a_pos * lin.b_lb + a_neg * lin.b_ub,
            A_ub=a_pos_A * A_ub_in + a_neg_A * A_lb_in,
            b_ub=a_pos * lin.b_ub + a_neg * lin.b_lb,
        )
    return LinearBound(
        A_lb=a_pos_A * lin.A_lb + a_neg_A * lin.A_ub,
        b_lb=a_pos * lin.b_lb + a_neg * lin.b_ub,
        A_ub=a_pos_A * lin.A_ub + a_neg_A * lin.A_lb,
        b_ub=a_pos * lin.b_ub + a_neg * lin.b_lb,
    )


def _fwd_bn(layer: Layer, lin: LinearBound) -> Optional[LinearBound]:
    """Compose dual-track affine bounds through batch normalization.

    Lazy identity: BN following INPUT (rare) materializes diagonal.
    """
    A_bn = _align(layer.params["A"], lin.b_lb.shape[1])
    c = _align(layer.params["c"], lin.b_lb.shape[1])
    A_pos = A_bn.clamp(min=0)
    A_neg = A_bn.clamp(max=0)
    A_pos_A = A_pos.view(1, -1, 1)
    A_neg_A = A_neg.view(1, -1, 1)

    if lin.A_lb is None or lin.A_ub is None:
        B, n = lin.b_lb.shape
        if n > _DENSE_LIN_BOUND_MAX_DIM:
            return None
        eye = torch.eye(n, device=lin.b_lb.device, dtype=lin.b_lb.dtype).unsqueeze(0).expand(B, n, n)
        A_lb_in = lin.A_lb if lin.A_lb is not None else eye
        A_ub_in = lin.A_ub if lin.A_ub is not None else eye
        return LinearBound(
            A_lb=A_pos_A * A_lb_in + A_neg_A * A_ub_in,
            b_lb=A_pos * lin.b_lb + A_neg * lin.b_ub + c,
            A_ub=A_pos_A * A_ub_in + A_neg_A * A_lb_in,
            b_ub=A_pos * lin.b_ub + A_neg * lin.b_lb + c,
        )
    return LinearBound(
        A_lb=A_pos_A * lin.A_lb + A_neg_A * lin.A_ub,
        b_lb=A_pos * lin.b_lb + A_neg * lin.b_ub + c,
        A_ub=A_pos_A * lin.A_ub + A_neg_A * lin.A_lb,
        b_ub=A_pos * lin.b_ub + A_neg * lin.b_lb + c,
    )


def _fwd_lrelu(lin: LinearBound, lb: torch.Tensor, ub: torch.Tensor, alpha: float) -> Optional[LinearBound]:
    """Apply forward triangle linear relaxation for LeakyReLU.

    Lazy identity: LReLU after INPUT (rare) materializes diagonal.
    """
    on = lb >= 0
    off = ub <= 0
    amb = ~(on | off)
    denom = (ub - lb).clamp(min=1e-12)
    alpha_tensor = torch.full_like(lb, alpha)
    up_slope = torch.where(
        amb,
        (ub - alpha * lb) / denom,
        torch.where(on, torch.ones_like(lb), alpha_tensor),
    )
    up_inter = torch.where(amb, alpha * lb - up_slope * lb, torch.zeros_like(lb))
    low_slope = torch.where(on, torch.ones_like(lb), alpha_tensor)

    if lin.A_lb is None or lin.A_ub is None:
        B, n = lb.shape
        if n > _DENSE_LIN_BOUND_MAX_DIM:
            return None
        eye = torch.eye(n, device=lb.device, dtype=lb.dtype).unsqueeze(0).expand(B, n, n)
        A_lb_in = lin.A_lb if lin.A_lb is not None else eye
        A_ub_in = lin.A_ub if lin.A_ub is not None else eye
        return LinearBound(
            A_lb=low_slope.unsqueeze(-1) * A_lb_in,
            b_lb=low_slope * lin.b_lb,
            A_ub=up_slope.unsqueeze(-1) * A_ub_in,
            b_ub=up_slope * lin.b_ub + up_inter,
        )
    return LinearBound(
        A_lb=low_slope.unsqueeze(-1) * lin.A_lb,
        b_lb=low_slope * lin.b_lb,
        A_ub=up_slope.unsqueeze(-1) * lin.A_ub,
        b_ub=up_slope * lin.b_ub + up_inter,
    )


def _fwd_conv2d(layer: Layer, lin: LinearBound) -> Optional[LinearBound]:
    """Propagate dual-track affine bounds through Conv2D via batched F.conv2d.

    Under CUDA memory pressure the batch is convolved in lane chunks (see
    :func:`_lin_bound_lane_chunk`) and reassembled; interval fallback is kept
    only for the case where a single lane does not fit.
    """
    weight = cast(torch.Tensor, layer.params["weight"])
    bias = layer.params.get("bias")
    stride = layer.params.get("stride", 1)
    padding = layer.params.get("padding", 0)
    dilation = layer.params.get("dilation", 1)
    groups = layer.params.get("groups", 1)
    if isinstance(stride, (list, tuple)):
        stride = stride[0]
    if isinstance(padding, (list, tuple)):
        padding = padding[0]
    if isinstance(dilation, (list, tuple)):
        dilation = dilation[0]

    out_c, in_c_per_g, _, _ = weight.shape
    in_c = in_c_per_g * groups

    if lin.A_lb is None or lin.A_ub is None:
        B, curr_dim = lin.b_lb.shape
        if curr_dim > _DENSE_LIN_BOUND_MAX_DIM:
            return None
        device, dtype = lin.b_lb.device, lin.b_lb.dtype
        eye = torch.eye(curr_dim, device=device, dtype=dtype).unsqueeze(0).expand(B, curr_dim, curr_dim)
        A_lb_in = lin.A_lb if lin.A_lb is not None else eye
        A_ub_in = lin.A_ub if lin.A_ub is not None else eye
        lin = LinearBound(A_lb=A_lb_in, b_lb=lin.b_lb,
                          A_ub=A_ub_in, b_ub=lin.b_ub)

    B, curr_dim, input_dim = lin.A_lb.shape

    in_h = in_w = 0
    input_shape = _shape_list(layer.params.get("input_shape"))
    if input_shape is not None and len(input_shape) >= 3:
        shape = input_shape
        if len(shape) == 4:
            _, _, h, w = shape
        else:
            _, h, w = shape[-3], shape[-2], shape[-1]
        if h * w * in_c == curr_dim:
            in_h, in_w = h, w

    if in_h == 0 or in_w == 0:
        spatial = curr_dim // in_c if in_c > 0 else 0
        side = int(spatial ** 0.5) if spatial > 0 else 0
        if spatial > 0 and side * side * in_c == curr_dim:
            in_h = in_w = side

    if in_h == 0 or in_w == 0:
        return None

    k_h, k_w = int(weight.shape[2]), int(weight.shape[3])
    stride_i = cast(int, stride)
    padding_i = cast(int, padding)
    dilation_i = cast(int, dilation)
    out_h = (in_h + 2 * padding_i - dilation_i * (k_h - 1) - 1) // stride_i + 1
    out_w = (in_w + 2 * padding_i - dilation_i * (k_w - 1) - 1) // stride_i + 1
    chunk = _lin_bound_lane_chunk(B, out_c * out_h * out_w, input_dim, lin.b_lb,
                                  _LIN_BOUND_CHUNK_TRANSIENT_CONV2D, layer.id, "conv2d")
    if chunk is None:
        return None

    def conv_A(A_mat: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        lanes = A_mat.shape[0]
        A_t = A_mat.transpose(1, 2).contiguous().view(lanes * input_dim, in_c, in_h, in_w)
        out = F.conv2d(A_t, kernel, None, stride, padding, dilation, groups)
        return out.flatten(start_dim=1).reshape(lanes, input_dim, -1).transpose(1, 2).contiguous()

    def conv_b(vec: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        b_4d = vec.view(vec.shape[0], in_c, in_h, in_w)
        return F.conv2d(b_4d, kernel, None, stride, padding, dilation, groups).flatten(start_dim=1)

    # Centre-radius composition, algebraically identical to the split-sign
    # form: W+ @ A_lb + W- @ A_ub == W @ Ac - |W| @ Ar with Ac=(A_lb+A_ub)/2,
    # Ar=(A_ub-A_lb)/2 (and the mirrored identity for the upper track). The
    # split-sign form runs four INDEPENDENT accumulations whose float32
    # rounding crosses by ~n*eps*sum|W||b| once the tracks nearly coincide
    # (RC2's linear-track sibling; vgg spec2 inverted by 1e2 at ~1e5
    # magnitudes). Sharing the centre term keeps bit-equal input tracks
    # bit-equal on output, so concretized boxes cannot invert along affine
    # chains. cuDNN is disabled for the radius convs: Winograd's subtractive
    # transforms emit small negatives for a zero/nonnegative radius (see
    # _fwd_conv2d_interval).
    def compose(sub: LinearBound) -> LinearBound:
        A_centre = (sub.A_lb + sub.A_ub) * 0.5
        A_radius = (sub.A_ub - sub.A_lb) * 0.5
        b_centre = (sub.b_lb + sub.b_ub) * 0.5
        b_radius = (sub.b_ub - sub.b_lb) * 0.5
        A_mid = conv_A(A_centre, weight)
        b_mid = conv_b(b_centre, weight)
        with torch.backends.cudnn.flags(enabled=False):
            A_spread = conv_A(A_radius, weight.abs())
            b_spread = conv_b(b_radius, weight.abs())
        A_lb_new = A_mid - A_spread
        A_ub_new = A_mid + A_spread
        b_lb_new = b_mid - b_spread
        b_ub_new = b_mid + b_spread

        if bias is not None:
            out_spatial = b_lb_new.shape[1] // out_c
            bias_bc = bias.view(out_c, 1).expand(out_c, out_spatial).reshape(-1)
            b_lb_new = b_lb_new + bias_bc
            b_ub_new = b_ub_new + bias_bc

        return LinearBound(A_lb=A_lb_new, b_lb=b_lb_new, A_ub=A_ub_new, b_ub=b_ub_new)

    if chunk >= B:
        return compose(lin)
    return _cat_lanes([
        compose(_lin_lanes(lin, start, min(start + chunk, B)))
        for start in range(0, B, chunk)
    ])


def _fwd_conv2d_interval(layer: Layer, lb: torch.Tensor, ub: torch.Tensor
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fallback interval Conv2D for when the linear-relaxation path cannot infer shape.

    Centre-radius form: ``m = c ⊛ W + b``, ``rho = r ⊛ |W|``, ``[m-rho, m+rho]``.
    ``rho >= 0`` structurally, so ``ub - lb = 2*rho >= 0`` per neuron whatever
    the rounding; the ``W⁺l + W⁻u`` / ``W⁺u + W⁻l`` form sums two independent
    4608-term accumulations that float32 cancellation can order the wrong way
    (VGG16 L30: 11 515 inverted neurons, worst excess 3.9e-1). Two convs, not four.

    The radius conv runs with cuDNN disabled: cuDNN picks Winograd for 3x3, whose
    transforms contain subtractions, so it returns small *negative* values for a
    non-negative input/kernel pair (measured -1.3 where neighbouring tiles carry a
    large radius). Plain accumulation keeps ``rho >= 0`` exact, hence ``ub >= lb``.
    """
    weight = cast(torch.Tensor, layer.params["weight"])
    bias = layer.params.get("bias")
    stride = layer.params.get("stride", 1)
    padding = layer.params.get("padding", 0)
    dilation = layer.params.get("dilation", 1)
    groups = layer.params.get("groups", 1)
    if isinstance(stride, (list, tuple)):
        stride = stride[0]
    if isinstance(padding, (list, tuple)):
        padding = padding[0]
    if isinstance(dilation, (list, tuple)):
        dilation = dilation[0]

    B = lb.shape[0]
    _, in_c_per_g, _, _ = weight.shape
    in_c = in_c_per_g * groups
    input_shape = _shape_list(layer.params.get("input_shape"))
    if input_shape is not None and len(input_shape) >= 3:
        shape = input_shape
        if len(shape) == 4:
            _, _, in_h, in_w = shape
        else:
            _, in_h, in_w = shape[-3], shape[-2], shape[-1]
    else:
        spatial = lb.shape[1] // in_c if in_c > 0 else 0
        side = int(spatial ** 0.5) if spatial > 0 else 0
        if side * side * in_c != lb.shape[1]:
            raise ValueError(
                f"_fwd_conv2d_interval: cannot infer spatial shape for "
                f"{lb.shape[1]} features with in_c={in_c}; layer {layer.id} "
                f"needs an explicit 'input_shape' param"
            )
        in_h = in_w = side

    try:
        centre = ((lb + ub) * 0.5).view(B, in_c, in_h, in_w)
        radius = ((ub - lb) * 0.5).view(B, in_c, in_h, in_w)
    except RuntimeError as e:
        raise ValueError(
            f"_fwd_conv2d_interval: reshape to [B={B}, {in_c}, {in_h}, {in_w}] "
            f"failed for lb.shape={tuple(lb.shape)}"
        ) from e

    conv_kw = dict(stride=stride, padding=padding, dilation=dilation, groups=groups)
    mid = F.conv2d(centre, weight, None, **conv_kw)
    with torch.backends.cudnn.flags(enabled=False):
        spread = F.conv2d(radius, weight.abs(), None, **conv_kw)
    if bias is not None:
        mid = mid + bias.view(1, -1, 1, 1)
    return (mid - spread).flatten(start_dim=1), (mid + spread).flatten(start_dim=1)


def _fwd_maxpool2d(layer: Layer, lb: torch.Tensor, ub: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """MaxPool2D interval propagation with runtime batch size."""
    kernel_size = layer.params.get("kernel_size", 2)
    stride = layer.params.get("stride", kernel_size)
    padding = layer.params.get("padding", 0)
    dilation = layer.params.get("dilation", 1)
    input_shape = _shape_list(layer.params.get("input_shape"))
    if input_shape is None:
        raise ValueError(
            f"_fwd_maxpool2d: layer {layer.id} missing required 'input_shape' param"
        )

    shape = input_shape
    if len(shape) == 4:
        _, c, h, w = shape
    else:
        c, h, w = shape[-3], shape[-2], shape[-1]
    B = lb.shape[0]
    lb_out = F.max_pool2d(lb.view(B, c, h, w), kernel_size, stride, padding, dilation)
    ub_out = F.max_pool2d(ub.view(B, c, h, w), kernel_size, stride, padding, dilation)
    return lb_out.flatten(start_dim=1), ub_out.flatten(start_dim=1)


def _fwd_maxpool2d_lin(
    layer: Layer, lin: LinearBound, lb: torch.Tensor, ub: torch.Tensor,
    frame: Frame,
) -> Optional[Tuple[LinearBound, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """DeepPoly MaxPool2D on the dual-track affine bounds (explicit ``A`` only).

    Per window let ``i*`` be the argmax of the interval lower bounds. Since
    ``max(x_window) >= x_{i*}`` always, input ``i*``'s affine lower bound is a
    sound lower bound for the pool output. When ``i*`` dominates —
    ``lb_{i*} >= max_{j != i*} ub_j`` — the pool output *is* ``x_{i*}``, so its
    affine upper bound is exact; otherwise the constant ``max_j ub_j`` is used,
    lifted to also cover the concretized max of the gathered lower row over
    ``frame``. ``ub_max`` comes from the interval track while the lower row
    comes from the linear track; once the box width collapses (deep input
    splits) the two independent float32 accumulations can cross by a few ulps,
    and a crossed (lower_fn, upper_const) pair is amplified by every following
    affine layer's |W| mass until the backward's degenerate-interval check
    raises. Raising the upper constant to ``max(ub_max, max_frame lower_fn)``
    is sound (uppers may only grow) and restores the exact-arithmetic
    invariant ``upper >= lower`` pointwise on the frame.
    ``ub_second`` (largest ub among the window's other inputs) is computed via
    ``F.unfold`` with the argmax slot and zero-filled padding slots masked to
    ``-inf``.

    Under CUDA memory pressure the gathers run in lane chunks (see
    :func:`_lin_bound_lane_chunk`); the window statistics above them do not
    scale with ``n_sym`` and stay batched.

    Returns ``None`` on the lazy-identity path (``A is None``), on a feature
    width mismatch, or when a single lane's gathered ``[1, n_out, n_sym]``
    coefficients would overrun CUDA memory — the caller then keeps the
    interval-reset behaviour. Otherwise returns ``(lin_out, idx_flat, dominant, lb_max,
    ub_max)`` where ``idx_flat`` is the flat argmax index per output (long
    ``[B, n_out]``), ``dominant`` is a bool ``[B, n_out]`` mask, and
    ``lb_max``/``ub_max`` are the interval pool box ``[B, n_out]``.
    """
    if lin.A_lb is None or lin.A_ub is None:
        return None
    kernel_size = pair_2d(layer.params.get("kernel_size", 2))
    stride = pair_2d(layer.params.get("stride", layer.params.get("kernel_size", 2)))
    padding = pair_2d(layer.params.get("padding", 0))
    dilation = pair_2d(layer.params.get("dilation", 1))
    input_shape = _shape_list(layer.params.get("input_shape"))
    if input_shape is None:
        raise ValueError(
            f"_fwd_maxpool2d_lin: layer {layer.id} missing required 'input_shape' param"
        )
    shape = input_shape
    if len(shape) == 4:
        _, c, h, w = shape
    else:
        c, h, w = shape[-3], shape[-2], shape[-1]
    B, n_in = lb.shape
    if n_in != c * h * w or lin.A_lb.shape[1] != n_in or lin.b_lb.shape[1] != n_in:
        return None

    lb_4d = lb.view(B, c, h, w)
    ub_4d = ub.view(B, c, h, w)
    lb_max_4d, idx_plane = F.max_pool2d(
        lb_4d, kernel_size, stride, padding, dilation, return_indices=True
    )
    ub_max_4d = F.max_pool2d(ub_4d, kernel_size, stride, padding, dilation)
    n_out = c * lb_max_4d.shape[2] * lb_max_4d.shape[3]
    n_sym = lin.A_lb.shape[2]
    chunk = _lin_bound_lane_chunk(B, n_out, n_sym, lin.b_lb,
                                  _LIN_BOUND_CHUNK_TRANSIENT_MAXPOOL2D, layer.id, "maxpool2d")
    if chunk is None:
        return None

    # Window membership via im2col: map each kernel slot to its input plane
    # index. Unfold zero-fills padding, so shift indices by one to tell a
    # padded slot (0) from plane position 0; float64 keeps indices exact.
    plane = torch.arange(
        1, h * w + 1, device=lb.device, dtype=torch.float64
    ).view(1, 1, h, w)
    plane_unf = F.unfold(
        plane, kernel_size, dilation=dilation, padding=padding, stride=stride
    )
    pad_slot = plane_unf == 0
    slot_plane = plane_unf.long() - 1
    k2, spatial_out = plane_unf.shape[1], plane_unf.shape[2]

    # Largest ub among the window's OTHER inputs: mask the argmax-lb slot and
    # the padded slots (unfold zero-fill) to -inf, then reduce over the kernel.
    ub_unf = F.unfold(
        ub_4d, kernel_size, dilation=dilation, padding=padding, stride=stride
    ).view(B, c, k2, spatial_out)
    is_argmax = slot_plane.view(1, 1, k2, spatial_out) == idx_plane.view(B, c, 1, spatial_out)
    ub_second = ub_unf.masked_fill(
        is_argmax | pad_slot.view(1, 1, k2, spatial_out), float("-inf")
    ).amax(dim=2)

    lb_max = lb_max_4d.flatten(start_dim=1)
    ub_max = ub_max_4d.flatten(start_dim=1)
    dominant = (lb_max_4d.view(B, c, spatial_out) >= ub_second).view(B, n_out)

    channel_base = (torch.arange(c, device=lb.device) * (h * w)).view(1, c, 1)
    idx_flat = (idx_plane.view(B, c, spatial_out) + channel_base).view(B, n_out)

    x_L, x_U = frame

    def gather_lanes(start: int, end: int) -> LinearBound:
        idx = idx_flat[start:end]
        gather_idx = idx.unsqueeze(-1).expand(end - start, n_out, n_sym)
        dom = dominant[start:end]
        A_lb_out = cast(torch.Tensor, lin.A_lb)[start:end].gather(1, gather_idx)
        b_lb_out = lin.b_lb[start:end].gather(1, idx)
        A_ub_out = cast(torch.Tensor, lin.A_ub)[start:end].gather(1, gather_idx).masked_fill(
            ~dom.unsqueeze(-1), 0.0)
        lower_row_max = (
            torch.einsum("boi,bi->bo", A_lb_out.clamp(min=0), x_U[start:end])
            + torch.einsum("boi,bi->bo", A_lb_out.clamp(max=0), x_L[start:end])
            + b_lb_out
        )
        b_ub_out = torch.where(
            dom, lin.b_ub[start:end].gather(1, idx),
            torch.maximum(ub_max[start:end], lower_row_max),
        )
        return LinearBound(A_lb=A_lb_out, b_lb=b_lb_out, A_ub=A_ub_out, b_ub=b_ub_out)

    if chunk >= B:
        lin_out = gather_lanes(0, B)
    else:
        lin_out = _cat_lanes([
            gather_lanes(start, min(start + chunk, B)) for start in range(0, B, chunk)
        ])
    return lin_out, idx_flat, dominant, lb_max, ub_max


def _fwd_avgpool2d(layer: Layer, lb: torch.Tensor, ub: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """AvgPool2D interval propagation with runtime batch size."""
    kernel_size = layer.params.get("kernel_size", 2)
    stride = layer.params.get("stride", kernel_size)
    padding = layer.params.get("padding", 0)
    ceil_mode = bool(layer.params.get("ceil_mode", False))
    count_include_pad = bool(layer.params.get("count_include_pad", True))
    divisor_override = layer.params.get("divisor_override")
    input_shape = _shape_list(layer.params.get("input_shape"))
    if input_shape is None:
        raise ValueError(
            f"_fwd_avgpool2d: layer {layer.id} missing required 'input_shape' param"
        )

    shape = input_shape
    if len(shape) == 4:
        _, c, h, w = shape
    else:
        c, h, w = shape[-3], shape[-2], shape[-1]
    B = lb.shape[0]
    pool_kwargs = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override,
    }
    lb_out = F.avg_pool2d(lb.view(B, c, h, w), **pool_kwargs)
    ub_out = F.avg_pool2d(ub.view(B, c, h, w), **pool_kwargs)
    return lb_out.flatten(start_dim=1), ub_out.flatten(start_dim=1)


def _align(a: torch.Tensor, n: int) -> torch.Tensor:
    """Align a 1-D parameter tensor to size n."""
    a = a.flatten()
    if a.numel() == n:
        return a
    if a.numel() > n:
        return a[:n]
    return a.repeat((n + a.numel() - 1) // a.numel())[:n]
