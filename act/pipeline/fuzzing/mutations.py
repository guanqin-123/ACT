"""
Mutation strategies for ACTFuzzer.

Implements gradient-guided, activation-guided, boundary, and random mutations.
All mutations automatically respect InputSpec constraints via projection.

Copyright (C) 2025 SVF-tools/ACT
License: AGPLv3+
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import torch
import torch.nn as nn
import numpy as np

from act.front_end.specs import InputSpec, InKind


class MutationStrategy(ABC):
    """Base class for mutation strategies."""
    
    @abstractmethod
    def mutate(self, 
               input_tensor: torch.Tensor,
               model: nn.Module,
               activations: Optional[Dict[str, torch.Tensor]] = None
              ) -> torch.Tensor:
        """
        Apply mutation to input tensor.
        
        Args:
            input_tensor: Seed input
            model: Model for gradient computation
            activations: Activations from previous inference (optional)
        
        Returns:
            Mutated input tensor
        """
        pass


class GradientMutation(MutationStrategy):
    """
    FGSM-style gradient-guided mutation.
    
    Computes gradients to maximize output variance, then applies
    signed gradient perturbation.
    """
    
    def __init__(self, epsilon: float = 0.01):
        """
        Initialize gradient mutation.
        
        Args:
            epsilon: Perturbation magnitude
        """
        self.epsilon = epsilon
    
    def mutate(self, input_tensor, model, activations=None):
        """Apply gradient-based perturbation."""
        # Enable gradients
        x = input_tensor.clone().detach().requires_grad_(True)
        
        # Forward pass
        output = model(x)
        
        # Extract output tensor if dict (from VerifiableModel)
        if isinstance(output, dict):
            output = output['output']
        
        # Compute loss: maximize output variance
        loss = output.var()
        
        # Backward pass
        loss.backward()
        
        # Get gradient
        grad = x.grad.detach()
        
        # FGSM: sign of gradient
        perturbation = self.epsilon * torch.sign(grad)
        
        # Apply perturbation
        return input_tensor + perturbation


class ActivationMutation(MutationStrategy):
    """
    Mutation to maximize neuron activation changes.
    
    Uses random direction weighted by recent activation patterns.
    """
    
    def __init__(self, epsilon: float = 0.01):
        """
        Initialize activation mutation.
        
        Args:
            epsilon: Perturbation magnitude
        """
        self.epsilon = epsilon
    
    def mutate(self, input_tensor, model, activations=None):
        """Apply activation-guided perturbation."""
        # Random direction (future: weight by inactive neurons)
        direction = torch.randn_like(input_tensor)
        
        # Normalize and scale
        direction = direction / (direction.norm() + 1e-8)
        perturbation = self.epsilon * direction
        
        return input_tensor + perturbation


class BoundaryMutation(MutationStrategy):
    """
    Mutation toward InputSpec boundaries.
    
    Explores edge cases where properties are more likely to fail.
    """
    
    def __init__(self, epsilon: float = 0.005):
        """
        Initialize boundary mutation.
        
        Args:
            epsilon: Perturbation magnitude toward boundary
        """
        self.epsilon = epsilon
    
    def mutate(self, input_tensor, model, activations=None):
        """Push toward boundaries (will be projected by engine)."""
        # Random direction
        direction = torch.sign(torch.randn_like(input_tensor))
        
        # Scale
        perturbation = self.epsilon * direction
        
        return input_tensor + perturbation


class RandomMutation(MutationStrategy):
    """Random Gaussian perturbation (baseline)."""
    
    def __init__(self, epsilon: float = 0.005):
        """
        Initialize random mutation.
        
        Args:
            epsilon: Standard deviation of Gaussian noise
        """
        self.epsilon = epsilon
    
    def mutate(self, input_tensor, model, activations=None):
        """Apply random Gaussian noise."""
        noise = torch.randn_like(input_tensor) * self.epsilon
        return input_tensor + noise


class MutationEngine:
    """
    Mutation engine with strategy selection and constraint projection.
    
    Features:
    - Weighted random strategy selection
    - Automatic InputSpec projection
    - Activation capture via forward hooks
    - Statistics tracking
    
    Example:
        >>> engine = MutationEngine(model, input_spec, weights, device)
        >>> mutated = engine.mutate(seed_tensor)
        >>> activations = engine.get_last_activations()
    """
    
    def __init__(self,
                 model: nn.Module,
                 input_spec: Optional[InputSpec],
                 weights: Dict[str, float],
                 device: torch.device):
        """
        Initialize mutation engine.
        
        Args:
            model: Model for gradient computation
            input_spec: InputSpec for constraint projection
            weights: Strategy weights (e.g., {"gradient": 0.4, "random": 0.1})
            device: Torch device
        """
        self.model = model
        self.input_spec = input_spec
        self.device = device
        
        # Initialize strategies
        self.strategies = {
            "gradient": GradientMutation(),
            "activation": ActivationMutation(),
            "boundary": BoundaryMutation(),
            "random": RandomMutation()
        }
        
        # Normalize weights
        total = sum(weights.values())
        self.weights = {k: v/total for k, v in weights.items()}
        
        # Statistics
        self.total_mutations = 0
        self.last_activations: Dict[str, torch.Tensor] = {}
        self.last_strategy: Optional[str] = None  # NEW: track last mutation strategy
        self.last_gradients: Optional[Dict[str, torch.Tensor]] = None  # NEW: for Level 3 tracing
        self.last_loss: Optional[float] = None  # NEW: for Level 3 tracing
        
        # Setup hooks for activation capture
        self._setup_hooks()
    
    def _setup_hooks(self):
        """Setup forward hooks to capture activations."""
        def make_hook(name):
            def hook(module, input, output):
                # Store activation (handle both tensor and dict outputs)
                if isinstance(output, torch.Tensor):
                    self.last_activations[name] = output.detach()
                elif isinstance(output, dict) and 'output' in output:
                    self.last_activations[name] = output['output'].detach()
            return hook
        
        # Register hooks on computational layers
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.ReLU, nn.Linear, nn.Conv2d)):
                module.register_forward_hook(make_hook(name))
    
    def mutate(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply random mutation strategy and project to InputSpec.
        
        Args:
            input_tensor: Seed input
        
        Returns:
            Mutated input satisfying InputSpec constraints
        """
        # Select strategy
        strategy_names = list(self.weights.keys())
        strategy_probs = list(self.weights.values())
        strategy_name = np.random.choice(strategy_names, p=strategy_probs)
        strategy = self.strategies[strategy_name]
        
        # NEW: Store strategy for tracing
        self.last_strategy = strategy_name
        
        # Apply mutation
        input_device = input_tensor.to(self.device)
        mutated = strategy.mutate(
            input_device,
            self.model,
            self.last_activations
        )
        
        # Project to InputSpec constraints
        mutated = self._project(mutated)
        
        self.total_mutations += 1
        return mutated
    
    def _project(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Project tensor to satisfy InputSpec constraints.
        
        Supports:
        - BOX: Clip to [lb, ub]
        - LINF_BALL: Clamp to L∞ ball around center
        - LIN_POLY: (TODO) Project to linear polytope
        
        Note: InputSpec bounds should always match tensor shape (enforced by spec creators).
        """
        if self.input_spec is None:
            return tensor
        
        if self.input_spec.kind == InKind.BOX:
            # Box constraints: clip to bounds
            lb = self.input_spec.lb.to(tensor.device)
            ub = self.input_spec.ub.to(tensor.device)
            
            # Verify shape consistency (should be guaranteed by spec creators)
            assert lb.shape == tensor.shape, (
                f"Shape mismatch in BOX projection: "
                f"input_spec.lb.shape={lb.shape} != tensor.shape={tensor.shape}. "
                f"This indicates a bug in the spec creator - bounds should be reshaped during spec creation."
            )
            assert ub.shape == tensor.shape, (
                f"Shape mismatch in BOX projection: "
                f"input_spec.ub.shape={ub.shape} != tensor.shape={tensor.shape}. "
                f"This indicates a bug in the spec creator - bounds should be reshaped during spec creation."
            )
            
            return torch.clamp(tensor, lb, ub)
        
        elif self.input_spec.kind == InKind.LINF_BALL:
            # L∞ ball: clamp perturbation to epsilon
            center = self.input_spec.center.to(tensor.device)
            eps = self.input_spec.eps
            
            # Verify shape consistency (center has batch dimension matching tensor)
            assert center.shape == tensor.shape, (
                f"Shape mismatch in LINF_BALL projection: "
                f"input_spec.center.shape={center.shape} != tensor.shape={tensor.shape}. "
                f"This indicates a bug in the spec creator - center should have batch dimension."
            )
            
            delta = tensor - center
            delta = torch.clamp(delta, -eps, eps)
            return center + delta
        
        elif self.input_spec.kind == InKind.LIN_POLY:
            # Linear polytope: Ax <= b
            # TODO: Implement quadratic programming projection
            # For now, just return the tensor
            return tensor
        
        return tensor
    
    def get_last_activations(self) -> Dict[str, torch.Tensor]:
        """Get activations from last inference."""
        return self.last_activations
    
    def get_last_gradients(self) -> Optional[Dict[str, torch.Tensor]]:
        """Get gradients from last mutation (Level 3 tracing only)."""
        return self.last_gradients
    
    def get_last_loss(self) -> Optional[float]:
        """Get loss value from last mutation (Level 3 tracing only)."""
        return self.last_loss
    
    def get_stats(self) -> Dict:
        """Get mutation statistics."""
        return {
            "total_mutations": self.total_mutations,
            "strategy_weights": self.weights,
        }
