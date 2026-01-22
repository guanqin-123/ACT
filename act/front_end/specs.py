#===- act/front_end/specs.py - Specification Data Types ----------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Defines InputSpec, OutputSpec and their batched variants for verification
#   specifications including safety, robustness, and constraint types.
#
#===---------------------------------------------------------------------===#

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union, List, Tuple
import torch


class InKind:
    BOX = "BOX"
    LINF_BALL = "LINF_BALL"
    LIN_POLY = "LIN_POLY"


class OutKind:
    LINEAR_LE = "LINEAR_LE"
    TOP1_ROBUST = "TOP1_ROBUST"
    MARGIN_ROBUST = "MARGIN_ROBUST"
    RANGE = "RANGE"


# =============================================================================
# Single-Sample Specifications
# =============================================================================

@dataclass
class InputSpec:
    kind: str
    lb: Optional[torch.Tensor] = None
    ub: Optional[torch.Tensor] = None
    center: Optional[torch.Tensor] = None
    eps: Optional[float] = None
    A: Optional[torch.Tensor] = None
    b: Optional[torch.Tensor] = None


@dataclass
class OutputSpec:
    kind: str
    c: Optional[torch.Tensor] = None
    d: Optional[float] = None
    y_true: Optional[int] = None
    margin: float = 0.0
    lb: Optional[torch.Tensor] = None
    ub: Optional[torch.Tensor] = None
    meta: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Batched Specifications
# =============================================================================

@dataclass
class BatchedInputSpec:
    """
    Batched input specification for B samples.
    
    Shapes: lb/ub/center: [B, ...], eps: scalar or [B], A: [B, M, D], b: [B, M]
    """
    kind: str
    lb: Optional[torch.Tensor] = None
    ub: Optional[torch.Tensor] = None
    center: Optional[torch.Tensor] = None
    eps: Optional[Union[float, torch.Tensor]] = None
    A: Optional[torch.Tensor] = None
    b: Optional[torch.Tensor] = None
    
    def __post_init__(self):
        if self.kind == InKind.BOX:
            assert self.lb is not None and self.ub is not None, "BOX requires lb, ub"
        elif self.kind == InKind.LINF_BALL:
            assert self.center is not None and self.eps is not None, "LINF_BALL requires center, eps"
        elif self.kind == InKind.LIN_POLY:
            assert self.A is not None and self.b is not None, "LIN_POLY requires A, b"
    
    def _get_tensor(self) -> torch.Tensor:
        """Get the primary tensor for this spec kind."""
        if self.kind == InKind.BOX:
            return self.lb
        elif self.kind == InKind.LINF_BALL:
            return self.center
        elif self.kind == InKind.LIN_POLY:
            return self.A
        raise ValueError(f"Unknown kind: {self.kind}")
    
    @property
    def batch_size(self) -> int:
        return self._get_tensor().shape[0]
    
    @property
    def device(self) -> torch.device:
        return self._get_tensor().device
    
    @property
    def dtype(self) -> torch.dtype:
        return self._get_tensor().dtype
    
    def get_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (lb, ub) tensors. For LINF_BALL: lb=center-eps, ub=center+eps."""
        if self.kind == InKind.BOX:
            return self.lb, self.ub
        elif self.kind == InKind.LINF_BALL:
            eps = self.eps.view(-1, *([1]*(self.center.dim()-1))) if isinstance(self.eps, torch.Tensor) else self.eps
            return self.center - eps, self.center + eps
        raise NotImplementedError(f"get_bounds not supported for {self.kind}")
    
    def to(self, device: Union[str, torch.device]) -> 'BatchedInputSpec':
        def mv(t): return t.to(device) if isinstance(t, torch.Tensor) else t
        return BatchedInputSpec(self.kind, mv(self.lb), mv(self.ub), mv(self.center), mv(self.eps), mv(self.A), mv(self.b))
    
    def __getitem__(self, i: int) -> InputSpec:
        if self.kind == InKind.BOX:
            return InputSpec(self.kind, lb=self.lb[i:i+1], ub=self.ub[i:i+1])
        elif self.kind == InKind.LINF_BALL:
            eps = self.eps[i].item() if isinstance(self.eps, torch.Tensor) else self.eps
            return InputSpec(self.kind, center=self.center[i:i+1], eps=eps)
        elif self.kind == InKind.LIN_POLY:
            return InputSpec(self.kind, A=self.A[i:i+1], b=self.b[i:i+1])
        raise NotImplementedError(f"__getitem__ not supported for {self.kind}")
    
    def __len__(self) -> int:
        return self.batch_size
    
    @classmethod
    def from_single_specs(cls, specs: List[InputSpec]) -> 'BatchedInputSpec':
        """Stack multiple InputSpecs into a batched spec."""
        if not specs:
            raise ValueError("Empty spec list")
        kind = specs[0].kind
        assert all(s.kind == kind for s in specs), "All specs must have same kind"
        
        if kind == InKind.BOX:
            return cls(kind, lb=torch.cat([s.lb for s in specs]), ub=torch.cat([s.ub for s in specs]))
        elif kind == InKind.LINF_BALL:
            center = torch.cat([s.center for s in specs])
            eps_list = [s.eps for s in specs]
            eps = eps_list[0] if len(set(eps_list)) == 1 else torch.tensor(eps_list, dtype=center.dtype, device=center.device)
            return cls(kind, center=center, eps=eps)
        elif kind == InKind.LIN_POLY:
            return cls(kind, A=torch.cat([s.A for s in specs]), b=torch.cat([s.b for s in specs]))
        raise NotImplementedError(f"from_single_specs not supported for {kind}")
    
    def __repr__(self) -> str:
        shape = list(self._get_tensor().shape)
        eps_s = f", eps={self.eps}" if self.eps is not None and not isinstance(self.eps, torch.Tensor) else ""
        return f"BatchedInputSpec({self.kind}, B={self.batch_size}, shape={shape}{eps_s})"


@dataclass
class BatchedOutputSpec:
    """
    Batched output specification for B samples.
    
    Shapes: y_true: [B], c: [B, C], d/margin: scalar or [B], lb/ub: [B, C]
    """
    kind: str
    c: Optional[torch.Tensor] = None
    d: Optional[Union[float, torch.Tensor]] = None
    y_true: Optional[torch.Tensor] = None
    margin: Union[float, torch.Tensor] = 0.0
    lb: Optional[torch.Tensor] = None
    ub: Optional[torch.Tensor] = None
    
    def __post_init__(self):
        if self.kind in (OutKind.TOP1_ROBUST, OutKind.MARGIN_ROBUST):
            assert self.y_true is not None, f"{self.kind} requires y_true"
        elif self.kind == OutKind.LINEAR_LE:
            assert self.c is not None and self.d is not None, "LINEAR_LE requires c, d"
        elif self.kind == OutKind.RANGE:
            assert self.lb is not None or self.ub is not None, "RANGE requires lb or ub"
    
    def _get_tensor(self) -> torch.Tensor:
        """Get the primary tensor for this spec kind."""
        if self.kind in (OutKind.TOP1_ROBUST, OutKind.MARGIN_ROBUST):
            return self.y_true
        elif self.kind == OutKind.LINEAR_LE:
            return self.c
        elif self.kind == OutKind.RANGE:
            return self.lb if self.lb is not None else self.ub
        raise ValueError(f"Unknown kind: {self.kind}")
    
    @property
    def batch_size(self) -> int:
        return self._get_tensor().shape[0]
    
    @property
    def device(self) -> torch.device:
        return self._get_tensor().device
    
    def to(self, device: Union[str, torch.device]) -> 'BatchedOutputSpec':
        def mv(t): return t.to(device) if isinstance(t, torch.Tensor) else t
        return BatchedOutputSpec(self.kind, mv(self.c), mv(self.d), mv(self.y_true), mv(self.margin), mv(self.lb), mv(self.ub))
    
    def __getitem__(self, i: int) -> OutputSpec:
        if self.kind in (OutKind.TOP1_ROBUST, OutKind.MARGIN_ROBUST):
            margin = self.margin[i].item() if isinstance(self.margin, torch.Tensor) else self.margin
            return OutputSpec(self.kind, y_true=self.y_true[i].item(), margin=margin)
        elif self.kind == OutKind.LINEAR_LE:
            d = self.d[i].item() if isinstance(self.d, torch.Tensor) else self.d
            return OutputSpec(self.kind, c=self.c[i:i+1], d=d)
        elif self.kind == OutKind.RANGE:
            return OutputSpec(self.kind, lb=self.lb[i:i+1] if self.lb is not None else None,
                            ub=self.ub[i:i+1] if self.ub is not None else None)
        raise NotImplementedError(f"__getitem__ not supported for {self.kind}")
    
    def __len__(self) -> int:
        return self.batch_size
    
    @classmethod
    def from_single_specs(cls, specs: List[OutputSpec]) -> 'BatchedOutputSpec':
        """Stack multiple OutputSpecs into a batched spec."""
        if not specs:
            raise ValueError("Empty spec list")
        kind = specs[0].kind
        assert all(s.kind == kind for s in specs), "All specs must have same kind"
        
        if kind in (OutKind.TOP1_ROBUST, OutKind.MARGIN_ROBUST):
            y_true = torch.tensor([s.y_true for s in specs], dtype=torch.long)
            margins = [s.margin for s in specs]
            margin = margins[0] if len(set(margins)) == 1 else torch.tensor(margins)
            return cls(kind, y_true=y_true, margin=margin)
        elif kind == OutKind.LINEAR_LE:
            c = torch.cat([s.c for s in specs])
            d_list = [s.d for s in specs]
            d = d_list[0] if len(set(d_list)) == 1 else torch.tensor(d_list)
            return cls(kind, c=c, d=d)
        elif kind == OutKind.RANGE:
            lb = torch.cat([s.lb for s in specs]) if specs[0].lb is not None else None
            ub = torch.cat([s.ub for s in specs]) if specs[0].ub is not None else None
            return cls(kind, lb=lb, ub=ub)
        raise NotImplementedError(f"from_single_specs not supported for {kind}")
    
    def __repr__(self) -> str:
        extra = f", labels={len(self.y_true.unique())}" if self.y_true is not None else ""
        return f"BatchedOutputSpec({self.kind}, B={self.batch_size}{extra})"
