#===- act/pipeline/finetune/provable.py - Provable Robust Loss ----------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Provable robust loss using dual bounds (Wong & Kolter style).
#   Computes certified lower bounds on class margins using Lagrangian dual,
#   then applies loss on worst-case logits.
#
#   Key: NO @torch.no_grad() - gradients flow for training.
#
#   Architecture:
#     - Registry-based dispatch (like DualTF) for extensibility
#     - Supports nn.Module, VerifiableModel, and ACT Net formats
#     - Batched operations for efficient training
#
#   Note: Batched bound functions are defined here (not in dual_tf) because
#   dual_tf only handles single-sample verification. Training needs batched
#   operations with gradient flow.
#
#===---------------------------------------------------------------------===#

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Dict, Optional, Tuple, List, Union

from .actloss import RobustLoss


# ============================================================================
# Batched Forward Bound Functions (for training)
# ============================================================================

def fwd_linear_batched(
    W: torch.Tensor, b: Optional[torch.Tensor],
    A: torch.Tensor, bias: torch.Tensor,
    x0: torch.Tensor, eps: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Batched linear forward: output = W @ (A @ x + bias) + b
    
    Args:
        W: Weight matrix [out_dim, in_dim]
        b: Bias vector [out_dim] or None
        A: Coefficient matrix [curr_dim, input_dim]
        bias: Bias vector [curr_dim]
        x0: Input centers [B, input_dim]
        eps: Input half-widths [B, input_dim]
        
    Returns:
        A_new: [out_dim, input_dim]
        bias_new: [out_dim]
        lb, ub: [B, out_dim]
    """
    n_in = W.shape[1]
    
    # Align A dimensions if needed
    if A.shape[0] != n_in:
        if A.shape[0] < n_in:
            pad = n_in - A.shape[0]
            A = torch.cat([A, torch.zeros(pad, A.shape[1], dtype=A.dtype, device=A.device)], dim=0)
            bias = torch.cat([bias, torch.zeros(pad, dtype=bias.dtype, device=bias.device)])
        else:
            A, bias = A[:n_in, :], bias[:n_in]
    
    # Update coefficients: new = W @ (A @ x + bias) + b = (W @ A) @ x + (W @ bias + b)
    A_new = W @ A
    bias_new = W @ bias
    if b is not None:
        bias_new = bias_new + b
    
    # Compute bounds: center ± radius [B, out_dim]
    center = x0 @ A_new.T + bias_new  # [B, out_dim]
    radius = eps @ A_new.abs().T      # [B, out_dim]
    
    return A_new, bias_new, center - radius, center + radius


def fwd_relu_batched(
    A: torch.Tensor, bias: torch.Tensor,
    x0: torch.Tensor, eps: torch.Tensor,
    prev_lb: torch.Tensor, prev_ub: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Batched ReLU forward with CROWN-style relaxation.
    
    Args:
        A: Coefficient matrix [curr_dim, input_dim]
        bias: Bias vector [curr_dim]
        x0: Input centers [B, input_dim]
        eps: Input half-widths [B, input_dim]
        prev_lb, prev_ub: Pre-activation bounds [B, curr_dim]
        
    Returns:
        A_new: [curr_dim, input_dim]
        bias_new: [curr_dim]
        lb, ub: [B, curr_dim]
    """
    device, dtype = prev_lb.device, prev_lb.dtype
    curr_dim = prev_lb.size(1)
    
    # Use worst-case bounds across batch for coefficient update
    batch_lb = prev_lb.min(dim=0)[0]  # [curr_dim]
    batch_ub = prev_ub.max(dim=0)[0]  # [curr_dim]
    
    # Classify neurons: active (lb >= 0), inactive (ub <= 0), crossing
    on = batch_lb >= 0
    off = batch_ub <= 0
    amb = ~(on | off)
    
    # Compute slopes: active=1, inactive=0, crossing=u/(u-l)
    d = torch.zeros(curr_dim, device=device, dtype=dtype)
    d = torch.where(on, torch.ones_like(d), d)
    offset = torch.zeros(curr_dim, device=device, dtype=dtype)
    
    if amb.any():
        denom = (batch_ub - batch_lb).clamp(min=1e-12)
        slope = batch_ub / denom
        d = torch.where(amb, slope, d)
        offset = torch.where(amb, -slope * batch_lb, offset)
    
    # Update coefficients: output = d * (A @ x + bias) + offset
    A_new = d.unsqueeze(1) * A
    bias_new = d * bias + offset
    
    # Post-ReLU bounds [B, curr_dim]
    center = x0 @ A_new.T + bias_new
    radius = eps @ A_new.abs().T
    lb_out = (center - radius).clamp(min=0)
    ub_out = center + radius
    
    return A_new, bias_new, lb_out, ub_out


# ============================================================================
# Batched Dual Backward Functions (for training)
# ============================================================================

def _get_relu_masks(l: torch.Tensor, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get boolean masks: (on, off, amb) for ReLU neurons."""
    on, off = l >= 0, u <= 0
    return on, off, ~(on | off)


def dual_relu_backward_batched(
    nu: torch.Tensor,  # [B, dim]
    l: torch.Tensor,   # [B, dim] 
    u: torch.Tensor,   # [B, dim]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Batched ReLU backward for provable training.
    
    Args:
        nu: Dual variables [B, dim]
        l, u: Pre-activation bounds [B, dim]
        
    Returns:
        nu_out: [B, dim]
        contrib: [B] contribution per sample
    """
    # Get neuron masks
    on, off, amb = _get_relu_masks(l, u)
    
    # Compute slope d: active=1, inactive=0, crossing=u/(u-l)
    d = torch.zeros_like(l)
    d = torch.where(on, torch.ones_like(d), d)
    if amb.any():
        denom = (u - l).clamp(min=1e-12)
        d = torch.where(amb, u / denom, d)
    
    # Apply slope
    nu_out = d * nu
    
    # Contribution: [nu]_+ * l for crossing neurons, summed per sample
    contrib = torch.zeros(nu.size(0), device=nu.device, dtype=nu.dtype)
    if amb.any():
        crossing_contrib = torch.where(amb, nu_out.clamp(min=0) * l, torch.zeros_like(l))
        contrib = crossing_contrib.sum(dim=1)
    
    return nu_out, contrib


def dual_dense_backward_batched(
    nu: torch.Tensor,  # [B, out_features]
    W: torch.Tensor,   # [out_features, in_features]
    b: Optional[torch.Tensor] = None,  # [out_features]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Batched Dense backward for provable training.
    
    Args:
        nu: Dual variables [B, out_features]
        W: Weight matrix [out_features, in_features]
        b: Bias vector [out_features]
        
    Returns:
        nu_out: [B, in_features]
        contrib: [B] contribution per sample
    """
    # Transpose: nu_out = nu @ W
    nu_out = nu @ W  # [B, in_features]
    
    # Bias contribution: -nu @ b per sample
    if b is not None:
        contrib = -(nu @ b)  # [B]
    else:
        contrib = torch.zeros(nu.size(0), device=nu.device, dtype=nu.dtype)
    
    return nu_out, contrib


def dual_input_contrib_batched(
    nu: torch.Tensor,  # [B, dim]
    lb: torch.Tensor,  # [B, dim]
    ub: torch.Tensor,  # [B, dim]
) -> torch.Tensor:
    """
    Batched input contribution for provable training.
    
    Wong-Kolter's input contribution: -[nu]_- @ lb - [nu]_+ @ ub
    
    Args:
        nu: Dual variables [B, dim]
        lb, ub: Input bounds [B, dim]
        
    Returns:
        contrib: [B] contribution per sample
    """
    nu_pos = nu.clamp(min=0)
    nu_neg = nu.clamp(max=0)
    return (-nu_neg * lb - nu_pos * ub).sum(dim=1)


# ============================================================================
# Layer Type Helpers
# ============================================================================

def _get_layer_type(layer: nn.Module) -> str:
    """Get canonical layer type string for registry lookup."""
    return type(layer).__name__


# ============================================================================
# ProvableLoss Class
# ============================================================================

class ProvableLoss(RobustLoss):
    """
    Provable robust loss using dual bounds.
    
    Computes certified lower bounds on class margins using Lagrangian dual,
    then applies CrossEntropyLoss on worst-case logits.
    
    Architecture:
        - Registry-based dispatch for forward/backward handlers (like DualTF)
        - Supports nn.Module (Sequential), VerifiableModel, and ACT Net
        - Batched operations for efficient training
    
    Usage:
        loss_fn = ProvableLoss(input_clamp=(0, 1))
        loss, metrics = loss_fn(model, X, y, epsilon=0.1)
        loss.backward()  # Gradients flow!
        
        # Register custom layer support
        loss_fn.register_forward('MyLayer', my_fwd_handler)
        loss_fn.register_backward('MyLayer', my_bwd_handler)
    """
    
    # -------- Forward Bounds Registry --------
    # Maps layer type -> method name for forward bound propagation
    _FORWARD_REGISTRY: Dict[str, str] = {
        # Affine layers
        'Linear': '_fwd_linear',
        'Conv2d': '_fwd_conv2d',
        'BatchNorm1d': '_fwd_bn',
        'BatchNorm2d': '_fwd_bn',
        # Activations
        'ReLU': '_fwd_relu',
        'Sigmoid': '_fwd_sigmoid',
        'Tanh': '_fwd_tanh',
        # Identity-like (pass-through)
        'Flatten': '_fwd_identity',
        'Identity': '_fwd_identity',
        'Dropout': '_fwd_identity',  # No-op at eval
        # Spec layers (VerifiableModel)
        'InputLayer': '_fwd_identity',
        'InputSpecLayer': '_fwd_input_spec',
        'OutputSpecLayer': '_fwd_identity',
    }
    
    # -------- Backward Bounds Registry --------
    # Maps layer type -> method name for dual backward pass
    _BACKWARD_REGISTRY: Dict[str, str] = {
        # Affine layers
        'Linear': '_bwd_linear',
        'Conv2d': '_bwd_conv2d',
        'BatchNorm1d': '_bwd_bn',
        'BatchNorm2d': '_bwd_bn',
        # Activations
        'ReLU': '_bwd_relu',
        'Sigmoid': '_bwd_sigmoid',
        'Tanh': '_bwd_tanh',
        # Identity-like (pass-through)
        'Flatten': '_bwd_identity',
        'Identity': '_bwd_identity',
        'Dropout': '_bwd_identity',
        # Spec layers (VerifiableModel)
        'InputLayer': '_bwd_identity',
        'InputSpecLayer': '_bwd_identity',
        'OutputSpecLayer': '_bwd_identity',
    }
    
    def __init__(
        self,
        input_clamp: Tuple[float, float] = (0.0, 1.0),
        loss_fn: str = 'ce',
    ):
        """
        Args:
            input_clamp: (min, max) bounds for input space (e.g., (0,1) for images)
            loss_fn: 'ce' (CrossEntropy) or 'hinge' (margin loss)
        """
        self.input_min, self.input_max = input_clamp
        self.loss_fn = loss_fn
        # Instance-level registry overrides (for custom layers)
        self._fwd_registry_overrides: Dict[str, Callable] = {}
        self._bwd_registry_overrides: Dict[str, Callable] = {}
    
    # -------- Registry Extension API --------
    def register_forward(self, layer_type: str, handler: Callable) -> None:
        """
        Register a custom forward bounds handler for a layer type.
        
        Args:
            layer_type: Layer class name (e.g., 'MyCustomLayer')
            handler: Function(self, layer, A, bias, x0, eps, prev_bounds) -> (A, bias, lb, ub)
        """
        self._fwd_registry_overrides[layer_type] = handler
    
    def register_backward(self, layer_type: str, handler: Callable) -> None:
        """
        Register a custom backward bounds handler for a layer type.
        
        Args:
            layer_type: Layer class name (e.g., 'MyCustomLayer')
            handler: Function(self, layer, nu, bounds) -> (nu, contrib)
        """
        self._bwd_registry_overrides[layer_type] = handler
    
    def supports_layer(self, layer_type: str) -> bool:
        """Check if a layer type is supported."""
        return (layer_type in self._FORWARD_REGISTRY or 
                layer_type in self._fwd_registry_overrides)
    
    @property
    def name(self) -> str: return "provable"
    
    @property
    def is_certified(self) -> bool: return True
    
    # -------- Main Interface --------
    def __call__(
        self,
        model: nn.Module,
        X: torch.Tensor,        # [B, ...]
        y: torch.Tensor,        # [B]
        epsilon: float,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute provable robust loss.
        
        Args:
            model: nn.Sequential network (Linear + ReLU)
            X: Input batch [B, ...]
            y: Labels [B]
            epsilon: L-inf perturbation radius
            
        Returns:
            loss: Differentiable loss tensor
            metrics: {'certified_acc', 'clean_acc', 'loss'}
        """
        B = X.size(0)
        device, dtype = X.device, X.dtype
        num_classes = self._get_num_classes(model)
        
        # Input bounds: [X - eps, X + eps] clamped to valid range
        X_flat = X.view(B, -1)
        lb = (X_flat - epsilon).clamp(min=self.input_min, max=self.input_max)
        ub = (X_flat + epsilon).clamp(min=self.input_min, max=self.input_max)
        
        # Forward pass: get pre-activation bounds at each layer (CROWN)
        layer_bounds = self._forward_bounds(model, lb, ub)
        
        # Dual bounds for all class margins
        worst_logits = self._compute_dual_bounds(model, layer_bounds, lb, ub, y, num_classes)
        
        # Compute loss
        loss = self._compute_loss(worst_logits, y, B, device)
        
        # Compute metrics (no grad needed)
        metrics = self._compute_metrics(model, X, y, worst_logits, loss)
        
        return loss, metrics
    
    # -------- Loss Computation --------
    def _compute_loss(
        self,
        worst_logits: torch.Tensor,  # [B, num_classes]
        y: torch.Tensor,             # [B]
        B: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Compute loss from worst-case logits."""
        if self.loss_fn == 'ce':
            return F.cross_entropy(-worst_logits, y)
        else:  # hinge
            margins = worst_logits[torch.arange(B, device=device), y]
            return F.relu(1.0 - margins).mean()
    
    def _compute_metrics(
        self,
        model: nn.Module,
        X: torch.Tensor,
        y: torch.Tensor,
        worst_logits: torch.Tensor,
        loss: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute accuracy metrics (no grad)."""
        with torch.no_grad():
            clean_out = model(X)
            clean_pred = clean_out.argmax(dim=1)
            clean_acc = (clean_pred == y).float().mean().item()
            
            certified = self._compute_certified(worst_logits, y)
            certified_acc = certified.float().mean().item()
        
        return {
            'certified_acc': certified_acc,
            'clean_acc': clean_acc,
            'loss': loss.item(),
        }
    
    def _forward_bounds(
        self,
        model: nn.Module,
        lb: torch.Tensor,  # [B, input_dim]
        ub: torch.Tensor,  # [B, input_dim]
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        CROWN-style forward bounds for batched inputs.
        Uses registry-based dispatch for extensibility.
        """
        layers = self._get_layers(model)
        input_dim = lb.size(1)
        device, dtype = lb.device, lb.dtype
        
        # Center and half-width [B, input_dim]
        x0 = (lb + ub) / 2
        eps = (ub - lb) / 2
        
        # Coefficient tracking: output = A @ input + bias
        A = torch.eye(input_dim, device=device, dtype=dtype)
        bias = torch.zeros(input_dim, device=device, dtype=dtype)
        layer_bounds = [(lb, ub)]
        
        for layer in layers:
            layer_type = _get_layer_type(layer)
            prev_bounds = layer_bounds[-1]
            
            # Check instance overrides first, then class registry
            if layer_type in self._fwd_registry_overrides:
                handler = self._fwd_registry_overrides[layer_type]
                A, bias, l, u = handler(self, layer, A, bias, x0, eps, prev_bounds)
            elif layer_type in self._FORWARD_REGISTRY:
                handler_name = self._FORWARD_REGISTRY[layer_type]
                handler = getattr(self, handler_name)
                A, bias, l, u = handler(layer, A, bias, x0, eps, prev_bounds)
            else:
                # Unknown layer: warn and use identity
                import warnings
                warnings.warn(f"ProvableLoss: unknown layer '{layer_type}', using identity")
                A, bias, l, u = self._fwd_identity(layer, A, bias, x0, eps, prev_bounds)
            
            layer_bounds.append((l, u))
        
        return layer_bounds
    
    # -------- Forward Handlers --------
    def _fwd_linear(
        self, layer: nn.Module, A: torch.Tensor, bias: torch.Tensor,
        x0: torch.Tensor, eps: torch.Tensor, prev_bounds: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward bounds through Linear layer."""
        return fwd_linear_batched(layer.weight, layer.bias, A, bias, x0, eps)
    
    def _fwd_relu(
        self, layer: nn.Module, A: torch.Tensor, bias: torch.Tensor,
        x0: torch.Tensor, eps: torch.Tensor, prev_bounds: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward bounds through ReLU layer."""
        prev_lb, prev_ub = prev_bounds
        return fwd_relu_batched(A, bias, x0, eps, prev_lb, prev_ub)
    
    def _fwd_identity(
        self, layer: nn.Module, A: torch.Tensor, bias: torch.Tensor,
        x0: torch.Tensor, eps: torch.Tensor, prev_bounds: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward bounds through identity-like layer (Flatten, Identity, Dropout)."""
        prev_lb, prev_ub = prev_bounds
        return A, bias, prev_lb, prev_ub
    
    def _fwd_conv2d(
        self, layer: nn.Module, A: torch.Tensor, bias: torch.Tensor,
        x0: torch.Tensor, eps: torch.Tensor, prev_bounds: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward bounds through Conv2d layer. TODO: implement batched conv bounds."""
        raise NotImplementedError("Conv2d forward bounds not yet implemented for training")
    
    def _fwd_bn(
        self, layer: nn.Module, A: torch.Tensor, bias: torch.Tensor,
        x0: torch.Tensor, eps: torch.Tensor, prev_bounds: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward bounds through BatchNorm layer. TODO: implement batched BN bounds."""
        raise NotImplementedError("BatchNorm forward bounds not yet implemented for training")
    
    def _fwd_sigmoid(
        self, layer: nn.Module, A: torch.Tensor, bias: torch.Tensor,
        x0: torch.Tensor, eps: torch.Tensor, prev_bounds: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward bounds through Sigmoid layer. TODO: implement batched sigmoid bounds."""
        raise NotImplementedError("Sigmoid forward bounds not yet implemented for training")
    
    def _fwd_tanh(
        self, layer: nn.Module, A: torch.Tensor, bias: torch.Tensor,
        x0: torch.Tensor, eps: torch.Tensor, prev_bounds: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward bounds through Tanh layer. TODO: implement batched tanh bounds."""
        raise NotImplementedError("Tanh forward bounds not yet implemented for training")
    
    def _fwd_input_spec(
        self, layer: nn.Module, A: torch.Tensor, bias: torch.Tensor,
        x0: torch.Tensor, eps: torch.Tensor, prev_bounds: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward bounds through InputSpecLayer (VerifiableModel). Uses spec bounds."""
        # TODO: Extract bounds from InputSpecLayer.input_spec
        prev_lb, prev_ub = prev_bounds
        return A, bias, prev_lb, prev_ub
    
    # -------- Dual Backward Pass --------
    def _compute_dual_bounds(
        self,
        model: nn.Module,
        layer_bounds: List[Tuple[torch.Tensor, torch.Tensor]],
        input_lb: torch.Tensor,  # [B, d]
        input_ub: torch.Tensor,  # [B, d]
        y: torch.Tensor,         # [B]
        num_classes: int,
    ) -> torch.Tensor:
        """
        Compute dual lower bounds on margins: output[y] - output[j] for all j.
        Returns: [B, num_classes] where [b,j] = lower bound on out[y[b]] - out[j]
        """
        B = input_lb.size(0)
        device, dtype = input_lb.device, input_lb.dtype
        
        # Coefficient matrix: c[b,j,k] = 1 if k==y[b], -1 if k==j, else 0
        eye = torch.eye(num_classes, device=device, dtype=dtype)
        c = eye[y].unsqueeze(1) - eye.unsqueeze(0)  # [B, num_classes, num_classes]
        
        # Dual backward for each target class
        all_bounds = []
        for j in range(num_classes):
            c_j = c[:, j, :]  # [B, num_classes]
            bound_j = self._dual_backward(model, layer_bounds, input_lb, input_ub, c_j)
            all_bounds.append(bound_j)
        
        return torch.stack(all_bounds, dim=1)  # [B, num_classes]
    
    def _dual_backward(
        self,
        model: nn.Module,
        layer_bounds: List[Tuple[torch.Tensor, torch.Tensor]],
        input_lb: torch.Tensor,  # [B, d]
        input_ub: torch.Tensor,  # [B, d]
        c: torch.Tensor,         # [B, num_classes]
    ) -> torch.Tensor:
        """
        Dual backward pass to compute lower bound on c @ output.
        Uses registry-based dispatch for extensibility.
        """
        layers = self._get_layers(model)
        num_layers = len(layers)
        
        nu = -c  # Wong-Kolter: nu = -c
        obj = torch.zeros(c.size(0), device=c.device, dtype=c.dtype)
        bound_idx = num_layers
        
        for layer in reversed(layers):
            bound_idx -= 1
            layer_type = _get_layer_type(layer)
            bounds = layer_bounds[bound_idx] if bound_idx >= 0 else None
            
            # Check instance overrides first, then class registry
            if layer_type in self._bwd_registry_overrides:
                handler = self._bwd_registry_overrides[layer_type]
                nu, contrib = handler(self, layer, nu, bounds)
            elif layer_type in self._BACKWARD_REGISTRY:
                handler_name = self._BACKWARD_REGISTRY[layer_type]
                handler = getattr(self, handler_name)
                nu, contrib = handler(layer, nu, bounds)
            else:
                # Unknown layer: warn and use identity
                import warnings
                warnings.warn(f"ProvableLoss: unknown layer '{layer_type}' in backward, using identity")
                nu, contrib = self._bwd_identity(layer, nu, bounds)
            
            obj = obj + contrib
        
        # Input contribution
        obj = obj + dual_input_contrib_batched(nu, input_lb, input_ub)
        return obj
    
    # -------- Backward Handlers --------
    def _bwd_linear(
        self, layer: nn.Module, nu: torch.Tensor, bounds: Optional[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Backward through Linear layer."""
        return dual_dense_backward_batched(nu, layer.weight, layer.bias)
    
    def _bwd_relu(
        self, layer: nn.Module, nu: torch.Tensor, bounds: Optional[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Backward through ReLU layer."""
        if bounds is None:
            return nu, torch.tensor(0.0, dtype=nu.dtype, device=nu.device)
        
        l, u = bounds
        l, u = l.view(l.size(0), -1), u.view(u.size(0), -1)
        
        # Align sizes if needed
        n = min(nu.size(1), l.size(1))
        if nu.size(1) != l.size(1):
            l, u, nu = l[:, :n], u[:, :n], nu[:, :n]
        
        return dual_relu_backward_batched(nu, l, u)
    
    def _bwd_identity(
        self, layer: nn.Module, nu: torch.Tensor, bounds: Optional[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Backward through identity-like layer (Flatten, Identity, Dropout)."""
        return nu, torch.tensor(0.0, dtype=nu.dtype, device=nu.device)
    
    def _bwd_conv2d(
        self, layer: nn.Module, nu: torch.Tensor, bounds: Optional[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Backward through Conv2d layer. TODO: implement batched conv backward."""
        raise NotImplementedError("Conv2d backward not yet implemented for training")
    
    def _bwd_bn(
        self, layer: nn.Module, nu: torch.Tensor, bounds: Optional[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Backward through BatchNorm layer. TODO: implement batched BN backward."""
        raise NotImplementedError("BatchNorm backward not yet implemented for training")
    
    def _bwd_sigmoid(
        self, layer: nn.Module, nu: torch.Tensor, bounds: Optional[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Backward through Sigmoid layer. TODO: implement batched sigmoid backward."""
        raise NotImplementedError("Sigmoid backward not yet implemented for training")
    
    def _bwd_tanh(
        self, layer: nn.Module, nu: torch.Tensor, bounds: Optional[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Backward through Tanh layer. TODO: implement batched tanh backward."""
        raise NotImplementedError("Tanh backward not yet implemented for training")
    
    # -------- Certification --------
    def _compute_certified(
        self,
        worst_logits: torch.Tensor,  # [B, C]
        y: torch.Tensor,             # [B]
    ) -> torch.Tensor:
        """
        Compute certified mask.
        Certified if worst_logits[b,j] > 0 for all j != y[b]
        """
        B, C = worst_logits.shape
        device = worst_logits.device
        
        # Mask true class by setting to inf
        worst_logits_masked = worst_logits.clone()
        worst_logits_masked[torch.arange(B, device=device), y] = float('inf')
        
        # Certified if min margin > 0
        return worst_logits_masked.min(dim=1)[0] > 0
    
    # -------- Helpers --------
    def _get_num_classes(self, model: nn.Module) -> int:
        """Get number of output classes from last Linear layer."""
        for layer in reversed(list(model.modules())):
            if isinstance(layer, nn.Linear):
                return layer.out_features
        raise ValueError("Could not determine number of classes from model")
    
    def _get_layers(self, model: nn.Module) -> List[nn.Module]:
        """Extract layer list from nn.Module."""
        if isinstance(model, nn.Sequential):
            return list(model)
        if hasattr(model, 'layers'):
            layers = model.layers
            if isinstance(layers, (nn.Sequential, nn.ModuleList)):
                return list(layers)
            return list(layers) if isinstance(layers, (list, tuple)) else [layers]
        if hasattr(model, 'model') and isinstance(model.model, nn.Sequential):
            return list(model.model)
        children = list(model.children())
        if children:
            if len(children) == 1 and isinstance(children[0], nn.Sequential):
                return list(children[0])
            return children
        raise TypeError(f"Cannot extract layers from {type(model).__name__}")
