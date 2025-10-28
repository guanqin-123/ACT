#===- act/front_end/wrapper_layers.py - PyTorch Wrapper Layers ---------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   PyTorch wrapper layers for spec-free verification. Provides nn.Module
#   components that embed specifications directly into models, enabling
#   constraint checking during inference and seamless ACT conversion.
#
# Key Features:
#   - Spec-free: Constraints embedded in model architecture, not external
#   - PyTorch-native: Full nn.Module compatibility for training/inference
#   - Bidirectional: Converts to/from ACT format via torch2act/act2torch
#   - Automatic verification: Returns constraint satisfaction status
#   - Rich metadata: Tracks input shapes, dtypes, devices for verification
#
# Core Wrapper Layers:
#
#   InputLayer:
#     Declares symbolic input with metadata (shape, dtype, device).
#     No-op at inference, converted to INPUT layer in ACT.
#
#   InputAdapterLayer:
#     Preprocessing operations (normalize, permute, flatten, slice).
#     Converted to PERMUTE/SCALE_SHIFT/FLATTEN/etc. in ACT.
#
#   InputSpecLayer:
#     Input constraint checking (BOX, L_INF, LIN_POLY).
#     Returns (x, satisfied, explanation) tuple during inference.
#     Converted to INPUT_SPEC layer in ACT.
#
#   OutputSpecLayer:
#     Output constraint checking (SAFETY, TOP1_ROBUST, MARGIN, etc.).
#     Returns (x, satisfied, explanation) tuple during inference.
#     Converted to ASSERT layer in ACT.
#
# Verification Workflow:
#   1. Build model with wrapper layers:
#      model = nn.Sequential(
#          InputLayer(shape=(1, 28, 28)),
#          InputSpecLayer(InputSpec(kind=InKind.L_INF, eps=0.03)),
#          nn.Flatten(),
#          nn.Linear(784, 128),
#          nn.ReLU(),
#          nn.Linear(128, 10),
#          OutputSpecLayer(OutputSpec(kind=OutKind.TOP1_ROBUST, y_true=5))
#      )
#
#   2. Wrap with VerifiableModel (from act2torch.py):
#      verifiable = VerifiableModel(*model)
#
#   3. Run with automatic constraint checking:
#      results = verifiable(input_tensor)
#      # Returns: {'output', 'input_satisfied', 'output_satisfied', ...}
#
#   4. Convert to ACT for formal verification:
#      from act.pipeline.torch2act import TorchToACT
#      act_net = TorchToACT(verifiable).run()
#
# Specification Support:
#   Input constraints (InKind):
#     - BOX: Interval bounds [lb, ub]
#     - L_INF: ε-ball around center
#     - LIN_POLY: Linear polyhedron Ax ≤ b
#
#   Output constraints (OutKind):
#     - SAFETY: Linear constraints cx ≤ d
#     - TOP1_ROBUST: Classification robustness (true label stays top-1)
#     - MARGIN: Classification margin > threshold
#     - LOCAL_ROBUST: Local robustness verification
#
#===---------------------------------------------------------------------===#

from __future__ import annotations
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple, Union

# Import ACT components
from act.front_end.specs import InputSpec, OutputSpec, InKind, OutKind
from act.back_end.layer_schema import LayerKind, REGISTRY
from act.back_end.layer_util import create_layer
from act.util.device_manager import get_default_device, get_default_dtype


def prod(seq: Tuple[int, ...]) -> int:
    """Helper function to compute product of shape dimensions."""
    p = 1
    for s in seq:
        p *= s
    return p


class InputLayer(nn.Module):
    """
    Declares the symbolic input block with rich metadata. No-op at inference.
    
    Supports comprehensive metadata tracking for verification including data type,
    layout, dataset information, and optional ground truth labels.
    
    NOTE: dtype is REQUIRED for verification soundness (different dtypes have
    different precision/range affecting bound propagation).
    """
    def __init__(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype,  # REQUIRED: Critical for verification soundness
        center: Optional[torch.Tensor] = None,
        desc: str = "input",
        # Tier 1: Essential metadata (strongly recommended)
        layout: Optional[str] = None,
        dataset_name: Optional[str] = None,
        # Tier 2: Important metadata
        num_classes: Optional[int] = None,
        value_range: Optional[Tuple[float, float]] = None,
        scale_hint: Optional[str] = None,
        distribution: Optional[str] = None,  # NEW: "uniform", "normal", "normalized", "unknown", or custom
        # Tier 3: Optional metadata
        label: Optional[Union[torch.Tensor, int]] = None,
        sample_id: Optional[Union[int, str]] = None,
        domain: Optional[str] = None,
        channels: Optional[int] = None,
    ):
        super().__init__()
        if shape[0] != 1:
            raise ValueError(f"Verification wrapper assumes batch=1, got batch size {shape[0]}")
        
        # Core attributes (dtype now required)
        self.shape = tuple(shape)
        self.dtype = dtype  # REQUIRED
        self.desc = desc
        
        # Tier 1: Essential metadata
        self.layout = layout
        self.dataset_name = dataset_name
        
        # Tier 2: Important metadata
        self.num_classes = num_classes
        self.value_range = tuple(value_range) if value_range else None
        self.scale_hint = scale_hint
        self.distribution = distribution  # NEW
        
        # Tier 3: Optional metadata
        self.sample_id = sample_id
        self.domain = domain
        self.channels = channels
        
        # GPU-first device management for tensor buffers
        if center is not None:
            center = center.to(device=get_default_device(), dtype=get_default_dtype())
            self.register_buffer("center", center.reshape(-1))
        else:
            self.center = None
        
        # Label as tensor buffer (device-aware, auto-converts from int)
        if label is not None:
            if isinstance(label, int):
                label = torch.tensor(label, dtype=torch.long)
            label = label.to(device=get_default_device(), dtype=torch.long)
            self.register_buffer("label", label.reshape(-1))  # Always 1-D
        else:
            self.label = None
        
        self._validate_schema()

    def _validate_schema(self):
        """Validate parameters against INPUT layer schema"""
        schema = REGISTRY[LayerKind.INPUT.value]
        
        # Collect params (only tensors)
        params = {}
        if self.center is not None:
            params["center"] = self.center
        
        # Collect meta (everything else - dtype now REQUIRED)
        meta = {
            "shape": self.shape,
            "dtype": str(self.dtype)  # REQUIRED - must always be present
        }
        
        # Add non-default desc
        if self.desc != "input":
            meta["desc"] = self.desc
        
        # Add Tier 1-3 metadata (only if not None)
        if self.layout is not None:
            meta["layout"] = self.layout
        if self.dataset_name is not None:
            meta["dataset_name"] = self.dataset_name
        if self.num_classes is not None:
            meta["num_classes"] = self.num_classes
        if self.value_range is not None:
            meta["value_range"] = self.value_range
        if self.scale_hint is not None:
            meta["scale_hint"] = self.scale_hint
        if self.distribution is not None:  # NEW
            meta["distribution"] = self.distribution
        if self.label is not None:
            meta["label"] = self.label  # Tensor stored directly
        if self.sample_id is not None:
            meta["sample_id"] = self.sample_id
        if self.domain is not None:
            meta["domain"] = self.domain
        if self.channels is not None:
            meta["channels"] = self.channels
        
        # Check required/optional params and meta
        for key in schema["params_required"]:
            if key not in params:
                raise ValueError(f"InputLayer missing required param: {key}")
        for key in params:
            if key not in schema["params_required"] + schema["params_optional"]:
                raise ValueError(f"InputLayer has unknown param: {key}")
        for key in schema["meta_required"]:
            if key not in meta:
                raise ValueError(f"InputLayer missing required meta: {key}")
        for key in meta:
            if key not in schema["meta_required"] + schema["meta_optional"]:
                raise ValueError(f"InputLayer has unknown meta: {key}")

    def to_act_layers(self, layer_id_start: int, in_vars: List[int]) -> Tuple[List, List[int]]:
        """Convert to ACT Layer(s) and return (layers, out_vars)"""
        N = prod(self.shape[1:])
        out_vars = list(range(len(in_vars), len(in_vars) + N))
        
        # Collect params (only center)
        params = {}
        if self.center is not None:
            params["center"] = self.center
        
        # Collect meta (dtype is REQUIRED, always present)
        meta = {
            "shape": self.shape,
            "dtype": str(self.dtype)  # REQUIRED
        }
        
        if self.desc != "input":
            meta["desc"] = self.desc
        
        # Add all optional metadata fields (only if not None)
        if self.layout is not None:
            meta["layout"] = self.layout
        if self.dataset_name is not None:
            meta["dataset_name"] = self.dataset_name
        if self.num_classes is not None:
            meta["num_classes"] = self.num_classes
        if self.value_range is not None:
            meta["value_range"] = self.value_range
        if self.scale_hint is not None:
            meta["scale_hint"] = self.scale_hint
        if self.distribution is not None:  # NEW
            meta["distribution"] = self.distribution
        if self.label is not None:
            meta["label"] = self.label
        if self.sample_id is not None:
            meta["sample_id"] = self.sample_id
        if self.domain is not None:
            meta["domain"] = self.domain
        if self.channels is not None:
            meta["channels"] = self.channels
        
        layer = create_layer(
            id=layer_id_start,
            kind=LayerKind.INPUT.value,
            params=params,
            meta=meta,
            in_vars=in_vars,
            out_vars=out_vars
        )
        return [layer], out_vars

    def get_metadata_summary(self) -> Dict[str, Any]:
        """Return a summary of all metadata for debugging/logging"""
        return {
            "shape": self.shape,
            "desc": self.desc,
            "dtype": str(self.dtype),  # Always present (required)
            "layout": self.layout,
            "dataset_name": self.dataset_name,
            "num_classes": self.num_classes,
            "value_range": self.value_range,
            "scale_hint": self.scale_hint,
            "distribution": self.distribution,  # NEW
            "label": self.label.item() if self.label is not None else None,
            "sample_id": self.sample_id,
            "domain": self.domain,
            "channels": self.channels,
            "has_center": self.center is not None,
        }

    def __repr__(self) -> str:
        """Enhanced string representation with key metadata"""
        meta_str = f"shape={self.shape}"
        if self.dataset_name:
            meta_str += f", dataset={self.dataset_name}"
        if self.label is not None:
            meta_str += f", label={self.label.item()}"
        if self.layout:
            meta_str += f", layout={self.layout}"
        return f"InputLayer({meta_str})"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class InputAdapterLayer(nn.Module):
    """
    General, config-driven input adapter. Applies any subset of:
      - permute over non-batch dims (e.g., HWC->CHW)
      - reorder indices (gather)
      - slice indices (subset)
      - pad constants (append features)
      - per-element affine: z = a ⊙ x + c (a,c scalar or per-element)
      - linear projection: z = A x + b
    """
    def __init__(
        self,
        permute_axes: Optional[Tuple[int, ...]] = None,
        reorder_idx: Optional[torch.Tensor] = None,
        slice_idx: Optional[torch.Tensor] = None,
        pad_values: Optional[torch.Tensor] = None,
        affine_a: Optional[torch.Tensor | float] = None,
        affine_c: Optional[torch.Tensor | float] = None,
        linproj_A: Optional[torch.Tensor] = None,
        linproj_b: Optional[torch.Tensor] = None,
        reshape_to: Optional[Tuple[int, ...]] = None,  # New: target shape for model input (excluding batch)
        adapt_channels: Optional[str] = None,  # New: channel adaptation strategy
    ):
        super().__init__()
        self.permute_axes = permute_axes
        self.reorder_idx = reorder_idx
        self.slice_idx = slice_idx
        self.pad_values = pad_values
        self.affine_a = affine_a
        self.affine_c = affine_c
        self.linproj_A = linproj_A
        self.linproj_b = linproj_b
        self.reshape_to = reshape_to
        self.adapt_channels = adapt_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x
        B = t.shape[0]

        # 1) Permute over non-batch dims
        if self.permute_axes is not None and t.dim() >= 2:
            axes = (0,) + tuple(a + 1 for a in self.permute_axes)
            t = t.permute(*axes)

        # 2) Flatten to features for index ops
        t = t.reshape(B, -1)

        # 3) Reorder / slice / pad
        if self.reorder_idx is not None:
            t = t.index_select(1, self.reorder_idx.to(t.device))
        if self.slice_idx is not None:
            t = t.index_select(1, self.slice_idx.to(t.device))
        if self.pad_values is not None:
            pad = self.pad_values.to(t.device, t.dtype).reshape(1, -1).expand(B, -1)
            t = torch.cat([t, pad], dim=1)

        # 4) Elementwise affine
        if self.affine_a is not None:
            a = torch.as_tensor(self.affine_a, device=t.device, dtype=t.dtype)
            if a.numel() == 1:
                a = a.expand_as(t)
            t = t * a
        if self.affine_c is not None:
            c = torch.as_tensor(self.affine_c, device=t.device, dtype=t.dtype)
            if c.numel() == 1:
                c = c.expand_as(t)
            t = t + c

        # 5) Linear projection
        if self.linproj_A is not None:
            A = self.linproj_A.to(t.device, t.dtype)  # [M, N]
            t = t @ A.t()
            if self.linproj_b is not None:
                t = t + self.linproj_b.to(t.device, t.dtype)

        # 6) Final reshaping for model compatibility (e.g., flatten -> image shape for CNNs)
        if self.reshape_to is not None:
            target_shape = (B,) + self.reshape_to
            t = t.reshape(target_shape)

        # 7) Channel adaptation for model compatibility
        if self.adapt_channels is not None and t.dim() == 4:  # Only for image tensors (B, C, H, W)
            B, C, H, W = t.shape
            if self.adapt_channels == "1to3" and C == 1:
                # Convert 1-channel (grayscale) to 3-channel (RGB) by replicating
                t = t.repeat(1, 3, 1, 1)
            elif self.adapt_channels == "3to1" and C == 3:
                # Convert 3-channel (RGB) to 1-channel (grayscale) by averaging
                t = t.mean(dim=1, keepdim=True)
            elif self.adapt_channels == "1to3_pad" and C == 1:
                # Convert 1-channel to 3-channel by padding with zeros
                zeros = torch.zeros(B, 2, H, W, device=t.device, dtype=t.dtype)
                t = torch.cat([t, zeros], dim=1)
            elif self.adapt_channels == "resize" and (C != self.reshape_to[0] if self.reshape_to else False):
                # Generic resize by interpolation (more advanced)
                target_channels = self.reshape_to[0] if self.reshape_to else C
                if C < target_channels:
                    # Repeat channels to match target
                    repeat_factor = target_channels // C
                    remainder = target_channels % C
                    t_repeated = t.repeat(1, repeat_factor, 1, 1)
                    if remainder > 0:
                        t_extra = t[:, :remainder, :, :]
                        t = torch.cat([t_repeated, t_extra], dim=1)
                elif C > target_channels:
                    # Take subset of channels
                    t = t[:, :target_channels, :, :]

        return t

    def to_act_layers(self, layer_id_start: int, in_vars: List[int]) -> Tuple[List, List[int]]:
        """Convert InputAdapterLayer to multiple ACT layers"""
        from act.back_end.core import Layer
        from act.back_end.layer_schema import LayerKind
        from act.back_end.layer_util import create_layer
        
        layers = []
        current_vars = in_vars
        current_id = layer_id_start
        
        # Convert each adapter operation to corresponding ACT layer
        if self.permute_axes is not None:
            params = {}
            meta = {"perm": self.permute_axes}
            layer = create_layer(
                id=current_id,
                kind=LayerKind.PERMUTE.value,
                params=params,
                meta=meta,
                in_vars=current_vars,
                out_vars=current_vars.copy()  # Same size
            )
            layers.append(layer)
            current_id += 1
        
        if self.reorder_idx is not None:
            params = {"idx": self.reorder_idx}
            meta = {}
            layer = create_layer(
                id=current_id,
                kind=LayerKind.REORDER.value,
                params=params,
                meta=meta,
                in_vars=current_vars,
                out_vars=current_vars.copy()  # Same size for reorder
            )
            layers.append(layer)
            current_id += 1
        
        if self.slice_idx is not None:
            params = {"idx": self.slice_idx}
            meta = {}
            new_vars = list(range(len(current_vars), len(current_vars) + len(self.slice_idx)))
            layer = create_layer(
                id=current_id,
                kind=LayerKind.SLICE.value,
                params=params,
                meta=meta,
                in_vars=current_vars,
                out_vars=new_vars
            )
            layers.append(layer)
            current_vars = new_vars
            current_id += 1
        
        if self.pad_values is not None:
            params = {"values": self.pad_values}
            meta = {}
            new_size = len(current_vars) + len(self.pad_values)
            new_vars = list(range(len(current_vars), len(current_vars) + new_size))
            layer = create_layer(
                id=current_id,
                kind=LayerKind.PAD.value,
                params=params,
                meta=meta,
                in_vars=current_vars,
                out_vars=new_vars
            )
            layers.append(layer)
            current_vars = new_vars
            current_id += 1
        
        if self.affine_a is not None or self.affine_c is not None:
            params = {}
            if self.affine_a is not None:
                params["scale"] = torch.as_tensor(self.affine_a)
            if self.affine_c is not None:
                params["shift"] = torch.as_tensor(self.affine_c)
            meta = {}
            layer = create_layer(
                id=current_id,
                kind=LayerKind.SCALE_SHIFT.value,
                params=params,
                meta=meta,
                in_vars=current_vars,
                out_vars=current_vars.copy()  # Same size
            )
            layers.append(layer)
            current_id += 1
        
        if self.linproj_A is not None:
            params = {"A": self.linproj_A}
            if self.linproj_b is not None:
                params["b"] = self.linproj_b
            meta = {}
            new_size = self.linproj_A.shape[0]
            new_vars = list(range(len(current_vars), len(current_vars) + new_size))
            layer = create_layer(
                id=current_id,
                kind=LayerKind.LINEAR_PROJ.value,
                params=params,
                meta=meta,
                in_vars=current_vars,
                out_vars=new_vars
            )
            layers.append(layer)
            current_vars = new_vars
            current_id += 1
        
        return layers, current_vars


class InputSpecLayer(nn.Module):
    """
    Wraps ACT's InputSpec AND is an nn.Module. No-op in forward; used by converters.
    The spec it carries should already be EXPRESSED IN POST-ADAPTER SPACE.
    """
    def __init__(self, spec: Optional[InputSpec] = None, **kwargs):
        super().__init__()
        self.spec = spec or InputSpec(**kwargs)
        self.kind = self.spec.kind
        self.eps = float(self.spec.eps) if self.spec.eps is not None else None

        # Register tensor fields as buffers so .to(device) works
        for name in ("lb", "ub", "center", "A", "b"):
            val = getattr(self.spec, name, None)
            if isinstance(val, torch.Tensor):
                self.register_buffer(name, val)
            else:
                setattr(self, name, None)
        self._validate_schema()

    def _validate_schema(self):
        """Validate parameters against INPUT_SPEC layer schema"""
        schema = REGISTRY[LayerKind.INPUT_SPEC.value]
        params = {}
        for name in ("lb", "ub", "center", "A", "b"):
            val = getattr(self, name, None)
            if val is not None:
                params[name] = val
        meta = {"kind": self.kind}
        if self.eps is not None:
            meta["eps"] = self.eps
        
        # Check schema compliance
        for key in schema["meta_required"]:
            if key not in meta:
                raise ValueError(f"InputSpecLayer missing required meta: {key}")
        for key in meta:
            if key not in schema["meta_required"] + schema["meta_optional"]:
                raise ValueError(f"InputSpecLayer has unknown meta: {key}")

    def to_act_layers(self, layer_id_start: int, in_vars: List[int]) -> Tuple[List, List[int]]:
        """Convert to ACT Layer(s) - INPUT_SPEC doesn't create new vars"""
        params = {}
        for name in ("lb", "ub", "center", "A", "b"):
            val = getattr(self, name, None)
            if val is not None:
                params[name] = val
        meta = {"kind": self.kind}
        if self.eps is not None:
            meta["eps"] = self.eps
        
        layer = create_layer(
            id=layer_id_start,
            kind=LayerKind.INPUT_SPEC.value,
            params=params,
            meta=meta,
            in_vars=in_vars,
            out_vars=in_vars  # INPUT_SPEC doesn't change variables
        )
        return [layer], in_vars

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, bool, str]:
        """
        Forward pass with constraint checking.
        
        Returns:
            Tuple of (tensor, satisfied, explanation):
            - tensor: The input tensor (unchanged)
            - satisfied: True if input satisfies all constraints
            - explanation: Human-readable constraint check result
        """
        # If no spec, pass through without checking
        if self.spec is None:
            return x, True, "✅ INPUT: No constraints"
        
        # Check constraints based on kind
        if self.kind == InKind.BOX:
            # Box constraint: lb <= x <= ub
            if self.lb is None or self.ub is None:
                return x, True, "⚠️ INPUT BOX: Missing lb/ub"
            
            lb_satisfied = (x >= self.lb).all()
            ub_satisfied = (x <= self.ub).all()
            satisfied = bool(lb_satisfied and ub_satisfied)
            
            if satisfied:
                margin_lb = (x - self.lb).min().item()
                margin_ub = (self.ub - x).min().item()
                margin = min(margin_lb, margin_ub)
                explanation = f"✅ INPUT BOX: lb≤x≤ub (margin={margin:.4f})"
            else:
                lb_viol = (x < self.lb).sum().item()
                ub_viol = (x > self.ub).sum().item()
                explanation = f"❌ INPUT BOX: {lb_viol} lb violations, {ub_viol} ub violations"
            
            return x, satisfied, explanation
        
        elif self.kind == InKind.LINF_BALL:
            # L∞-ball constraint: ||x - center||∞ <= eps
            if self.center is None or self.eps is None:
                return x, True, "⚠️ INPUT L∞: Missing center/eps"
            
            center = self.center.reshape(x.shape)
            linf_dist = (x - center).abs().max().item()
            satisfied = linf_dist <= self.eps
            
            if satisfied:
                explanation = f"✅ INPUT L∞: ||x-c||∞={linf_dist:.4f}≤ε={self.eps:.4f}"
            else:
                explanation = f"❌ INPUT L∞: ||x-c||∞={linf_dist:.4f}>ε={self.eps:.4f}"
            
            return x, satisfied, explanation
        
        elif self.kind == InKind.LIN_POLY:
            # Linear polytope: Ax <= b
            if self.A is None or self.b is None:
                return x, True, "⚠️ INPUT LIN_POLY: Missing A/b"
            
            x_flat = x.reshape(-1)
            residuals = self.A @ x_flat - self.b  # Should be <= 0
            max_violation = residuals.max().item()
            satisfied = max_violation <= 0
            
            if satisfied:
                margin = -max_violation  # How much slack we have
                explanation = f"✅ INPUT LIN_POLY: Ax≤b (margin={margin:.4f})"
            else:
                num_violations = (residuals > 0).sum().item()
                explanation = f"❌ INPUT LIN_POLY: {num_violations} constraints violated (max={max_violation:.4f})"
            
            return x, satisfied, explanation
        
        else:
            return x, True, f"⚠️ INPUT: Unknown kind {self.kind}"


class OutputSpecLayer(nn.Module):
    """
    Wraps ACT's OutputSpec AND is an nn.Module. No-op in forward; used by converters.
    """
    def __init__(self, spec: Optional[OutputSpec] = None, **kwargs):
        super().__init__()
        self.spec = spec or OutputSpec(**kwargs)
        self.kind = self.spec.kind
        self.y_true = self.spec.y_true
        self.margin = float(self.spec.margin)
        self.d = None if self.spec.d is None else float(self.spec.d)
        self.meta = dict(self.spec.meta)

        for name in ("c", "lb", "ub"):
            val = getattr(self.spec, name, None)
            if isinstance(val, torch.Tensor):
                self.register_buffer(name, val)
            else:
                setattr(self, name, None)
        self._validate_schema()

    def _validate_schema(self):
        """Validate parameters against ASSERT layer schema"""
        schema = REGISTRY[LayerKind.ASSERT.value]
        params = {}
        for name in ("c", "lb", "ub"):
            val = getattr(self, name, None)
            if val is not None:
                params[name] = val
        meta = {"kind": self.kind}
        if self.y_true is not None:
            meta["y_true"] = self.y_true
        if self.margin is not None:
            meta["margin"] = self.margin
        if self.d is not None:
            meta["d"] = self.d
        
        # Check schema compliance
        for key in schema["meta_required"]:
            if key not in meta:
                raise ValueError(f"OutputSpecLayer missing required meta: {key}")

    def to_act_layers(self, layer_id_start: int, in_vars: List[int]) -> Tuple[List, List[int]]:
        """Convert to ACT Layer(s) - ASSERT doesn't create new vars"""
        params = {}
        for name in ("c", "lb", "ub"):
            val = getattr(self, name, None)
            if val is not None:
                params[name] = val
        meta = {"kind": self.kind}
        if self.y_true is not None:
            meta["y_true"] = self.y_true
        if self.margin is not None:
            meta["margin"] = self.margin
        if self.d is not None:
            meta["d"] = self.d
        
        layer = create_layer(
            id=layer_id_start,
            kind=LayerKind.ASSERT.value,
            params=params,
            meta=meta,
            in_vars=in_vars,
            out_vars=in_vars  # ASSERT doesn't change variables
        )
        return [layer], in_vars

    def forward(self, y: torch.Tensor) -> Tuple[torch.Tensor, bool, str]:
        """
        Forward pass with constraint checking.
        
        Returns:
            Tuple of (tensor, satisfied, explanation):
            - tensor: The output tensor (unchanged)
            - satisfied: True if output satisfies all constraints
            - explanation: Human-readable constraint check result
        """
        # If no spec, pass through without checking
        if self.spec is None:
            return y, True, "✅ OUTPUT: No constraints"
        
        # Check constraints based on kind
        if self.kind == OutKind.TOP1_ROBUST:
            # Top-1 robustness: y_true class has highest score
            if self.y_true is None:
                return y, True, "⚠️ OUTPUT TOP1: Missing y_true"
            
            y_flat = y.reshape(-1)
            pred_class = y_flat.argmax().item()
            y_true_score = y_flat[self.y_true].item()
            max_other_score = y_flat[[i for i in range(len(y_flat)) if i != self.y_true]].max().item()
            margin = y_true_score - max_other_score
            
            satisfied = pred_class == self.y_true
            
            if satisfied:
                explanation = f"✅ OUTPUT TOP1: Class {self.y_true} wins (margin={margin:.4f})"
            else:
                explanation = f"❌ OUTPUT TOP1: Class {pred_class} wins, expected {self.y_true} (margin={margin:.4f})"
            
            return y, satisfied, explanation
        
        elif self.kind == OutKind.MARGIN_ROBUST:
            # Margin robustness: y_true class score exceeds others by margin
            if self.y_true is None or self.margin is None:
                return y, True, "⚠️ OUTPUT MARGIN: Missing y_true/margin"
            
            y_flat = y.reshape(-1)
            y_true_score = y_flat[self.y_true].item()
            max_other_score = y_flat[[i for i in range(len(y_flat)) if i != self.y_true]].max().item()
            actual_margin = y_true_score - max_other_score
            
            satisfied = actual_margin >= self.margin
            
            if satisfied:
                explanation = f"✅ OUTPUT MARGIN: margin={actual_margin:.4f}≥{self.margin:.4f}"
            else:
                explanation = f"❌ OUTPUT MARGIN: margin={actual_margin:.4f}<{self.margin:.4f}"
            
            return y, satisfied, explanation
        
        elif self.kind == OutKind.LINEAR_LE:
            # Linear inequality: c^T y <= d
            if self.c is None or self.d is None:
                return y, True, "⚠️ OUTPUT LINEAR_LE: Missing c/d"
            
            y_flat = y.reshape(-1)
            # Ensure dtype consistency for dot product
            c_typed = self.c.to(dtype=y_flat.dtype, device=y_flat.device)
            lhs = (c_typed @ y_flat).item()
            satisfied = lhs <= self.d
            
            if satisfied:
                margin = self.d - lhs
                explanation = f"✅ OUTPUT LINEAR_LE: c^T·y={lhs:.4f}≤d={self.d:.4f} (margin={margin:.4f})"
            else:
                violation = lhs - self.d
                explanation = f"❌ OUTPUT LINEAR_LE: c^T·y={lhs:.4f}>d={self.d:.4f} (violation={violation:.4f})"
            
            return y, satisfied, explanation
        
        elif self.kind == OutKind.RANGE:
            # Range constraint: lb <= y <= ub
            if self.lb is None or self.ub is None:
                return y, True, "⚠️ OUTPUT RANGE: Missing lb/ub"
            
            lb_satisfied = (y >= self.lb).all()
            ub_satisfied = (y <= self.ub).all()
            satisfied = bool(lb_satisfied and ub_satisfied)
            
            if satisfied:
                margin_lb = (y - self.lb).min().item()
                margin_ub = (self.ub - y).min().item()
                margin = min(margin_lb, margin_ub)
                explanation = f"✅ OUTPUT RANGE: lb≤y≤ub (margin={margin:.4f})"
            else:
                lb_viol = (y < self.lb).sum().item()
                ub_viol = (y > self.ub).sum().item()
                explanation = f"❌ OUTPUT RANGE: {lb_viol} lb violations, {ub_viol} ub violations"
            
            return y, satisfied, explanation
        
        else:
            return y, True, f"⚠️ OUTPUT: Unknown kind {self.kind}"