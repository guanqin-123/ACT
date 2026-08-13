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
#   
#   Key Idea: Reuse DualTF verification code directly for training.
#   - Convert nn.Module to ACT Net format
#   - Use DualTF(mode='t') for gradient-enabled bound computation
#   - Aggregate bounds across batch to compute loss
#
#   This ensures correctness by using the same code path as verification.
#
#===---------------------------------------------------------------------===#

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List

from .actloss import RobustLoss
from act.back_end.core import Net, Layer, Bounds
from act.back_end.dual_tf import DualTF, compute_forward_bounds
from act.back_end.layer_schema import LayerKind


# ============================================================================
# ProvableLoss - Single Sample Training with DualTF
# ============================================================================

class ProvableLoss(RobustLoss):
    """
    Provable robust loss using DualTF verification bounds.
    
    Computes certified lower bounds on class margins using Lagrangian dual,
    then applies CrossEntropyLoss on worst-case logits.
    
    Architecture:
        - Converts nn.Module to ACT Net (once, cached)
        - Uses DualTF(mode='t') for gradient-enabled bounds
        - Loops over batch samples (single-sample verification per sample)
    
    Usage:
        loss_fn = ProvableLoss(input_clamp=(0, 1))
        loss, metrics = loss_fn(model, X, y, epsilon=0.1)
        loss.backward()  # Gradients flow!
    """
    
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
        
        # Cache for ACT Net conversion (avoid re-converting each forward pass)
        self._cached_net: Optional[Net] = None
        self._cached_model_id: Optional[int] = None
        
        # DualTF instance with training mode (gradients enabled)
        self._dual_tf = DualTF(mode='t')
    
    @property
    def name(self) -> str:
        return "provable"
    
    @property
    def is_certified(self) -> bool:
        return True
    
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
            model: nn.Sequential network
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
        
        # Convert model to ACT Net (cached)
        net = self._get_or_build_net(model, X.shape[1:], dtype)
        
        # Compute worst-case logits for each sample
        worst_logits_list = []
        for i in range(B):
            x_i = X[i]  # [...]
            y_i = y[i].item()
            
            # Input bounds for this sample
            lb_i = (x_i - epsilon).clamp(min=self.input_min, max=self.input_max)
            ub_i = (x_i + epsilon).clamp(min=self.input_min, max=self.input_max)
            
            # Compute bounds using DualTF
            margins_i = self._compute_margins(net, lb_i, ub_i, y_i, num_classes)
            worst_logits_list.append(margins_i)
        
        # Stack margins: [B, num_classes]
        worst_logits = torch.stack(worst_logits_list, dim=0)
        
        # Compute loss
        loss = self._compute_loss(worst_logits, y, B, device)
        
        # Compute metrics (no grad needed)
        metrics = self._compute_metrics(model, X, y, worst_logits, loss)
        
        return loss, metrics
    
    # -------- ACT Net Conversion --------
    def _get_or_build_net(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        dtype: torch.dtype,
    ) -> Net:
        """
        Build ACT Net from nn.Module WITH gradient preservation.
        
        Unlike build_act() which detaches weights for verification,
        this method preserves requires_grad for training.
        """
        model_id = id(model)
        
        if self._cached_net is not None and self._cached_model_id == model_id:
            # Check if model weights changed (would need re-caching)
            # For now, just return cached net
            return self._cached_net
        
        layers = []
        next_var = 0
        
        # Helper to allocate variable IDs
        def alloc_vars(n):
            nonlocal next_var
            ids = list(range(next_var, next_var + n))
            next_var += n
            return ids
        
        # Input dimension
        input_dim = 1
        for s in input_shape:
            input_dim *= s
        
        # INPUT layer
        input_vars = alloc_vars(input_dim)
        input_layer = Layer.__new__(Layer)
        input_layer.id = 0
        input_layer.kind = LayerKind.INPUT.value
        input_layer.params = {}
        input_layer.meta = {"shape": (1,) + tuple(input_shape)}
        input_layer.in_vars = []
        input_layer.out_vars = input_vars
        layers.append(input_layer)
        
        prev_vars = input_vars
        
        # Convert each nn.Module layer (NO detach - preserve gradients!)
        for mod in self._get_layers(model):
            if isinstance(mod, nn.Linear):
                out_vars = alloc_vars(mod.out_features)
                layer = Layer.__new__(Layer)
                layer.id = len(layers)
                layer.kind = LayerKind.DENSE.value
                # Keep requires_grad for training!
                layer.params = {"W": mod.weight, "b": mod.bias if mod.bias is not None else torch.zeros(mod.out_features, dtype=dtype)}
                layer.meta = {}
                layer.in_vars = prev_vars
                layer.out_vars = out_vars
                layers.append(layer)
                prev_vars = out_vars
                
            elif isinstance(mod, nn.ReLU):
                out_vars = alloc_vars(len(prev_vars))
                layer = Layer.__new__(Layer)
                layer.id = len(layers)
                layer.kind = LayerKind.RELU.value
                layer.params = {}
                layer.meta = {}
                layer.in_vars = prev_vars
                layer.out_vars = out_vars
                layers.append(layer)
                prev_vars = out_vars
                
            elif isinstance(mod, (nn.Flatten, nn.Identity)):
                # Identity-like layers - just pass through
                pass
            else:
                raise NotImplementedError(f"Unsupported layer type: {type(mod).__name__}")
        
        # Build preds/succs (simple sequential chain)
        n_layers = len(layers)
        preds = {i: [i-1] if i > 0 else [] for i in range(n_layers)}
        succs = {i: [i+1] if i < n_layers - 1 else [] for i in range(n_layers)}
        
        # Create Net (skip validation)
        net = Net.__new__(Net)
        net.layers = layers
        net.preds = preds
        net.succs = succs
        
        # Cache
        self._cached_net = net
        self._cached_model_id = model_id
        
        return net
    
    def _get_layers(self, model: nn.Module) -> List[nn.Module]:
        """Extract layer list from nn.Module."""
        if isinstance(model, nn.Sequential):
            return list(model)
        children = list(model.children())
        if children:
            if len(children) == 1 and isinstance(children[0], nn.Sequential):
                return list(children[0])
            return children
        return []
    
    # -------- Dual Bound Computation --------
    def _compute_margins(
        self,
        net: Net,
        lb: torch.Tensor,  # Input lower bound [...]
        ub: torch.Tensor,  # Input upper bound [...]
        y_true: int,
        num_classes: int,
    ) -> torch.Tensor:
        """
        Compute margin lower bounds: output[y_true] - output[j] for all j.
        
        Returns: [num_classes] tensor of margin bounds
        """
        device, dtype = lb.device, lb.dtype
        
        # Compute forward bounds for all layers
        bounds_dict = compute_forward_bounds(net, lb.flatten(), ub.flatten())
        
        # Store bounds in DualTF for backward pass
        self._dual_tf._bounds_dict = bounds_dict
        self._dual_tf.clear_cache()
        
        # Compute margin bound for each class j != y_true
        margins = []
        for j in range(num_classes):
            if j == y_true:
                # Margin to self is always 0 (or inf)
                margins.append(torch.tensor(float('inf'), dtype=dtype, device=device))
            else:
                # Coefficient: c[y_true] = 1, c[j] = -1
                c = torch.zeros(num_classes, dtype=dtype, device=device)
                c[y_true] = 1.0
                c[j] = -1.0
                
                # Compute lower bound on margin
                bound = self._dual_tf.compute_bound(net, bounds_dict, c)
                margins.append(bound)
        
        return torch.stack(margins)  # [num_classes]
    
    # -------- Loss Computation --------
    def _compute_loss(
        self,
        worst_logits: torch.Tensor,  # [B, num_classes] - margins
        y: torch.Tensor,             # [B]
        B: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Compute loss from worst-case margins.
        
        worst_logits[b, j] = lower bound on output[y[b]] - output[j]
        worst_logits[b, y[b]] = inf (margin to self)
        
        For CE loss: we want the true class to have highest margin
        For hinge loss: we want all margins > 1
        """
        if self.loss_fn == 'ce':
            # Replace inf with 0 for the true class (neutral in softmax)
            logits = worst_logits.clone()
            logits[torch.arange(B, device=device), y] = 0.0
            # CrossEntropy expects logits where higher = more confident
            # Our margins are: positive = correctly classified
            # So we use -margins for non-true classes, 0 for true class
            return F.cross_entropy(-logits, y)
        else:  # hinge
            # Hinge loss: want min margin > 1
            # Mask out self-margin (inf)
            logits = worst_logits.clone()
            logits[torch.arange(B, device=device), y] = float('inf')
            min_margins = logits.min(dim=1)[0]  # [B]
            return F.relu(1.0 - min_margins).mean()
    
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
            # Clean accuracy
            clean_out = model(X)
            clean_pred = clean_out.argmax(dim=1)
            clean_acc = (clean_pred == y).float().mean().item()
            
            # Certified accuracy
            certified = self._compute_certified(worst_logits, y)
            certified_acc = certified.float().mean().item()
        
        return {
            'certified_acc': certified_acc,
            'clean_acc': clean_acc,
            'loss': loss.item(),
        }
    
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
        
        # Mask true class by setting to inf (already inf from _compute_margins)
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
