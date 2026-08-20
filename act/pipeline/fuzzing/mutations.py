"""
Mutation strategies for ACTFuzzer.

Implements gradient-guided, activation-guided, boundary, and random mutations.
All mutations automatically respect InputSpec constraints via projection.

Gradient-guided now accommodates two mutated input generation methods: FGSM (Fast Gradient Sign Method) and PGD (Projected Gradient Descent).
    1) FGSM: single-step gradient-based perturbation.
    2) PGD: iterative gradient-based perturbation.

## Batch Tensor-Based Mutation

All mutation strategies operate on batched inputs [B, C, H, W] for GPU parallelism.
The MutationEngine selects a single strategy per batch and applies it to all seeds
simultaneously, enabling efficient gradient computation (FGSM/PGD) across the batch.
The batch size is aligned with model synthesis (N VNNLib instances), so InputSpec bounds
are already [N, ...] and match the batch dimension directly. After mutation, projection
ensures each sample respects its InputSpec bounds (BOX per-sample bounds via original_index,
or LINF_BALL eps-ball around each seed's original_tensor).

## Adaptive Perturbation Sizing

NOTE: We use "perturb_size" (not "epsilon") to avoid confusion with InputSpec.eps (L∞ radius).
- InputSpec.eps: Defines constraint boundaries (e.g., center ± eps for LINF_BALL)
- Mutation perturb_size: Controls mutation perturbation magnitude (exploration granularity)

This module supports adaptive perturbation sizing that scales with InputSpec bounds to ensure
consistent exploration across different problem scales.

### What is perturb_scale?

`perturb_scale` is the **fraction of the feasible range** that each mutation perturbation covers.

**Interpretation Formula:**
    steps_to_traverse = 1 / perturb_scale

**Calculation:**
    range / perturb_size = range / (range * perturb_scale) = 1 / perturb_scale

**Examples:**
    - perturb_scale=0.1  → Each perturbation covers 10% of range → Takes ~10 steps to traverse from lb to ub
    - perturb_scale=0.2  → Each perturbation covers 20% of range → Takes ~5 steps to traverse from lb to ub
    - perturb_scale=0.05 → Each perturbation covers 5% of range  → Takes ~20 steps to traverse from lb to ub

### Perturbation Modes

1. **adaptive_scalar** (default):
   - Computes single perturb_size from mean range: perturb_size = mean(ub - lb) * perturb_scale
   - Best for: Uniform ranges (e.g., VNNLib BOX constraints with consistent bounds)
   - Example: VNNLib with lb=0.0, ub=1.0 → range=1.0, perturb_size=0.1 (10 steps)

2. **adaptive_perdim** (advanced):
   - Computes per-dimension perturb_size tensor: perturb_size[i] = (ub[i] - lb[i]) * perturb_scale
   - Best for: Non-uniform ranges (e.g., different features with vastly different scales)
   - Example: lb=[0, -100], ub=[1, 100] → perturb_size=[0.1, 20.0] (10 steps per dimension)

3. **fixed** (conventional):
   - Computes perturb_size from mean range: perturb_size = mean(ub - lb)
   - Note: Uses full feasible range as perturbation size 

### Configuration

Set in `act/config/pipeline.yaml`:
```yaml
perturb_mode: "adaptive_scalar"  # Options: "adaptive_scalar", "adaptive_perdim", "fixed"
perturb_scale: 0.1               # Fraction of range per step (default: 0.1 = 10 steps)
```

Copyright (C) 2025 SVF-tools/ACT
License: AGPLv3+
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from act.front_end.specs import InputSpec, InKind, OutputSpec
from act.front_end.verifiable_model import VerifiableModel
from act.pipeline.fuzzing.checker import INPUT_FEASIBILITY_ATTR
from act.util.device_manager import get_default_device

if TYPE_CHECKING:
    from act.pipeline.fuzzing.corpus import FuzzingSeed


class MutationStrategy(ABC):
    """Base class for mutation strategies."""
    
    @abstractmethod
    def mutate(self, 
               input_tensor: torch.Tensor,
               model: nn.Module,
               activations: Optional[Dict[str, torch.Tensor]] = None,
               rows: Optional[torch.Tensor] = None
              ) -> torch.Tensor:
        """
        Apply mutation to input tensor.
        
        Args:
            input_tensor: Seed input
            model: Inner network for gradient computation
            activations: Activations from previous inference (optional)
            rows: [B] int64 — spec row backing each batch lane (optional)
        
        Returns:
            Mutated input tensor
        """
        pass


class GradientMutation(MutationStrategy, ABC):
    """
    Base class for the gradient-guided strategies (FGSM, PGD).

    Both ascend the same objective, and this is the only place it is defined:
    the violation severity of the model's ``OutputSpec``. Severity is signed so
    that ``severity > 0`` means the property is broken, which makes it a direct
    gradient-ascent target for every ``OutKind``. For ``TOP1_ROBUST`` it is
    exactly the Carlini-Wagner margin ``max_{j != t} z_j - z_t`` that PGD used
    to compute inline; routing through ``OutputSpec.violation`` generalises that
    loss to the other four kinds instead of falling back to a label-free proxy.
    """

    def __init__(
        self,
        output_spec: OutputSpec,
        perturb_size: Union[float, torch.Tensor],
    ):
        """
        Initialize a gradient-guided mutation.

        Args:
            output_spec: Property to attack. Supplies the ascent objective.
            perturb_size: Mutation perturbation magnitude (scalar or per-dimension tensor)
        """
        self.output_spec = output_spec
        self.perturb_size = perturb_size

    def _severity_sum(
        self,
        x: torch.Tensor,
        model: nn.Module,
        rows: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Forward ``x`` and return the batch's total violation severity.

        The returned scalar stays inside the autograd graph, so
        ``torch.autograd.grad(loss, x)`` yields the direction that drives the
        batch toward violating the property. Lanes are summed because they are
        independent: ``d severity_i / d x_j`` is zero for ``i != j``, so the sum
        gives each lane its own gradient.

        Args:
            x: Input tensor [B, ...] that requires grad.
            model: Inner network; returns logits [B, n_out].
            rows: [B] int64 mapping each lane onto its spec row, or None for
                identity. The corpus samples with replacement, so lane ``i`` is
                generally NOT spec row ``i``.

        Returns:
            Scalar tensor to ascend.
        """
        _, severity = self.output_spec.violation(model(x), rows=rows)
        return severity.sum()


class FGSMMutation(GradientMutation):
    """
    FGSM-style gradient-guided mutation (single-step).

    Takes one sign-gradient step along the violation-severity gradient.
    """

    def __init__(
        self,
        output_spec: OutputSpec,
        perturb_size: Union[float, torch.Tensor] = 8/255,
    ):
        """
        Initialize FGSM mutation.

        Args:
            output_spec: Property to attack (supplies the ascent objective)
            perturb_size: Mutation perturbation magnitude (scalar or per-dimension tensor)
        """
        super().__init__(output_spec, perturb_size)

    def mutate(self, input_tensor, model, activations=None, rows=None):
        """Apply FGSM gradient-based perturbation (single-step).
        
        Args:
            input_tensor: Seed input tensor
            model: Inner network for gradient computation
            activations: Activations from previous inference (unused)
            rows: [B] int64 — spec row backing each batch lane
        """
        # Enable gradients
        x = input_tensor.clone().detach().requires_grad_(True)

        loss = self._severity_sum(x, model, rows)

        # Get gradient w.r.t. input only (avoid accumulating grads on model params)
        grad = torch.autograd.grad(loss, x, retain_graph=False, create_graph=False)[0].detach()

        # FGSM: sign of gradient
        perturb_size = self.perturb_size.to(input_tensor.device) if isinstance(self.perturb_size, torch.Tensor) else self.perturb_size
        perturbation = perturb_size * torch.sign(grad)

        # Apply perturbation
        return input_tensor + perturbation


class PGDMutation(GradientMutation):
    """
    PGD-style gradient-guided mutation (iterative).

    Implementation follows the notebook approach:
    - Define a feasible box around x0: [x0 - perturb_size, x0 + perturb_size]
    - Optional random start within the feasible box
    - Iterative sign-gradient ascent with projection back to the feasible box

    Note: Global InputSpec constraints are enforced by MutationEngine projection after mutation.
    """

    def __init__(
        self,
        output_spec: OutputSpec,
        perturb_size: Union[float, torch.Tensor] = 8/255,
        num_steps: int = 50,
        step_size: Optional[float] = None,
        random_start: bool = True,
    ):
        """
        Initialize PGD mutation.

        Args:
            output_spec: Property to attack (supplies the ascent objective)
            perturb_size: L_infinity radius of local feasible box around the seed (scalar or per-dimension tensor)
            num_steps: Number of PGD iterations
            step_size: Per-iteration step size (if None, computed from feasible box range / steps as in notebook)
            random_start: Whether to start uniformly within the feasible box (recommended)
        """
        super().__init__(output_spec, perturb_size)
        self.num_steps = int(num_steps)
        self.step_size = step_size
        self.random_start = random_start

    def mutate(self, input_tensor, model, activations=None, rows=None):
        """Apply PGD mutation.
        
        Args:
            input_tensor: Seed input tensor [B, C, H, W] or [1, C, H, W]
            model: Inner network for gradient computation
            activations: Activations from previous inference (unused by PGD)
            rows: [B] int64 — spec row backing each batch lane
        
        Returns:
            Adversarially perturbed input tensor [B, C, H, W]
        """
        x0 = input_tensor.detach()
        B = x0.shape[0]

        perturb_size = self.perturb_size.to(input_tensor.device) if isinstance(self.perturb_size, torch.Tensor) else self.perturb_size
        x_low = x0 - perturb_size
        x_high = x0 + perturb_size

        # Default step size: spread movement across the available range (notebook heuristic)
        if self.step_size is None:
            # (x_high - x_low) == 2*perturb_size; take max range element as scalar step size
            step_size = float((x_high - x_low).abs().max().item()) / max(self.num_steps, 1)
            step_size = max(step_size, 1e-6)
        else:
            step_size = float(self.step_size)

        # Random start inside feasible box
        if self.random_start:
            x_adv = x_low + torch.rand_like(x0) * (x_high - x_low)
        else:
            x_adv = x0.clone()

        # Ensure start in-bounds
        x_adv = torch.max(torch.min(x_adv, x_high), x_low).detach()

        for _ in range(self.num_steps):
            x_adv.requires_grad_(True)

            loss = self._severity_sum(x_adv, model, rows)

            grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0].detach()

            # Gradient ascent on loss
            x_adv = (x_adv + step_size * torch.sign(grad)).detach()

            # Project back to feasible box
            x_adv = torch.max(torch.min(x_adv, x_high), x_low).detach()

        return x_adv.detach()




class ActivationMutation(MutationStrategy):
    """
    Mutation to maximize neuron activation changes.
    
    Uses random direction weighted by recent activation patterns.
    """
    
    def __init__(self, perturb_size: Union[float, torch.Tensor] = 0.01):
        """
        Initialize activation mutation.
        
        Args:
            perturb_size: Mutation perturbation magnitude (scalar or per-dimension tensor)
        """
        self.perturb_size = perturb_size
    
    def mutate(self, input_tensor, model, activations=None, rows=None):
        """Apply activation-guided perturbation.
        
        Args:
            input_tensor: Seed input tensor
            model: Model (unused)
            activations: Activations from previous inference (unused currently)
            rows: Spec rows (unused, kept for interface consistency)
        """
        # Random direction (future: weight by inactive neurons)
        direction = torch.randn_like(input_tensor)
        
        # Normalize and scale
        direction = direction / (direction.norm() + 1e-8)
        # Handle both scalar and tensor perturb_size
        perturb_size = self.perturb_size.to(input_tensor.device) if isinstance(self.perturb_size, torch.Tensor) else self.perturb_size
        perturbation = perturb_size * direction
        
        return input_tensor + perturbation


class BoundaryMutation(MutationStrategy):
    """
    Mutation toward InputSpec boundaries.
    
    Explores edge cases where properties are more likely to fail.
    """
    
    def __init__(self, perturb_size: Union[float, torch.Tensor] = 0.005):
        """
        Initialize boundary mutation.
        
        Args:
            perturb_size: Mutation perturbation magnitude toward boundary (scalar or per-dimension tensor)
        """
        self.perturb_size = perturb_size
    
    def mutate(self, input_tensor, model, activations=None, rows=None):
        """Push toward boundaries (will be projected by engine).
        
        Args:
            input_tensor: Seed input tensor
            model: Model (unused)
            activations: Activations (unused)
            rows: Spec rows (unused, kept for interface consistency)
        """
        # Random direction
        direction = torch.sign(torch.randn_like(input_tensor))
        
        # Scale
        # Handle both scalar and tensor perturb_size
        perturb_size = self.perturb_size.to(input_tensor.device) if isinstance(self.perturb_size, torch.Tensor) else self.perturb_size
        perturbation = perturb_size * direction
        
        return input_tensor + perturbation


class RandomMutation(MutationStrategy):
    """Random Gaussian perturbation (baseline)."""
    
    def __init__(self, perturb_size: Union[float, torch.Tensor] = 0.005):
        """
        Initialize random mutation.
        
        Args:
            perturb_size: Standard deviation of Gaussian noise (scalar or per-dimension tensor)
        """
        self.perturb_size = perturb_size
    
    def mutate(self, input_tensor, model, activations=None, rows=None):
        """Apply random Gaussian noise.
        
        Args:
            input_tensor: Seed input tensor
            model: Model (unused)
            activations: Activations (unused)
            rows: Spec rows (unused, kept for interface consistency)
        """
        # Handle both scalar and tensor perturb_size
        perturb_size = self.perturb_size.to(input_tensor.device) if isinstance(self.perturb_size, torch.Tensor) else self.perturb_size
        noise = torch.randn_like(input_tensor) * perturb_size
        return input_tensor + noise


class MutationEngine:
    """
    Mutation engine with strategy selection and constraint projection.
    
    Features:
    - Weighted random strategy selection
    - Automatic InputSpec projection
    - Activation capture via forward hooks
    - Single strategy per mutation call for GPU parallelism
    
    Example:
        >>> engine = MutationEngine(model, input_spec, weights)
        >>> mutated = engine.mutate([seed1, seed2])
        >>> activations = engine.get_activation_map()
    """
    
    def __init__(self,
                 model: nn.Module,
                 input_spec: Optional[InputSpec],
                 weights: Dict[str, float],
                 perturb_mode: str = "fixed",
                 perturb_scale: float = 0.1):
        """
        Initialize mutation engine.
        
        Args:
            model: VerifiableModel for gradient computation. Hooks are registered
                across its whole module tree, but the gradient strategies attack
                its inner network (see :meth:`_resolve_attack_target`).
            input_spec: InputSpec for constraint projection (batched bounds from model synthesis)
            weights: Strategy weights (e.g., {"gradient": 0.4, "random": 0.1})
            perturb_mode: Perturbation size computation mode ("adaptive_scalar", "adaptive_perdim", "fixed")
            perturb_scale: Fraction of range per mutation perturbation (e.g., 0.1 = 10% = ~10 steps to traverse)
        
        Raises:
            TypeError: If ``model`` is not a VerifiableModel.
            ValueError: If ``model`` carries no OutputSpec, or weights are invalid.
        """
        self.model = model
        self.attack_model, self.output_spec = self._resolve_attack_target(model)
        self.input_spec = input_spec
        self.device = get_default_device()
        self.perturb_mode = perturb_mode
        self.perturb_scale = perturb_scale
        
        valid_perturb_modes = {"adaptive_scalar", "adaptive_perdim", "fixed"}
        if self.perturb_mode not in valid_perturb_modes:
            raise ValueError(
                f"Unknown perturb_mode: {self.perturb_mode}. "
                "Valid options: 'adaptive_scalar', 'adaptive_perdim', 'fixed'"
            )

        # Adaptive modes replace this placeholder with the dynamic per-seed
        # schedule before the selected strategy is used. Fixed mode retains its
        # range-aware baseline; without an InputSpec, 0.01 remains the fallback.
        perturb_size = (
            self._compute_fixed_perturb_size()
            if self.perturb_mode == "fixed" or self.input_spec is None
            else 0.0
        )
        
        # Initialize strategies with computed perturb_size
        self.strategies = {
            "gradient": FGSMMutation(self.output_spec, perturb_size=perturb_size),
            "pgd": PGDMutation(self.output_spec, perturb_size=perturb_size),
            "activation": ActivationMutation(perturb_size=perturb_size),
            "boundary": BoundaryMutation(perturb_size=perturb_size),
            "random": RandomMutation(perturb_size=perturb_size),
        }

        # Validate and normalize weights
        unknown = set(weights.keys()) - set(self.strategies.keys())
        if unknown:
            raise ValueError(
                f"Unknown mutation strategy keys in weights: {sorted(unknown)}. "
                f"Valid options: {sorted(self.strategies.keys())}"
            )
        total = sum(float(v) for v in weights.values())
        if total <= 0.0:
            raise ValueError(f"Mutation weights must sum to > 0. Got total={total}.")
        self.weights = {k: float(v) / total for k, v in weights.items()}
        
        # Statistics
        self.total_mutations = 0
        self.activation_map: Dict[str, torch.Tensor] = {}
        self.last_strategy: Optional[str] = None  # Track last mutation strategy for tracing
        self.last_gradients: Optional[Dict[str, torch.Tensor]] = None  # For Level 3 tracing: gradient capture
        self.last_loss: Optional[float] = None  # For Level 3 tracing: loss value capture
        self._pending_rows: Optional[torch.Tensor] = None
        self._feasibility_forward_active = False
        
        # Setup hooks for activation capture
        self._setup_hooks()
        self._setup_feasibility_hooks()
    
    @staticmethod
    def _resolve_attack_target(model: nn.Module) -> tuple[nn.Module, OutputSpec]:
        """Split a wrapped model into the network to attack and the property to attack it with.
        
        The gradient strategies attack the INNER network rather than the
        VerifiableModel wrapper. ``InputLayer`` and ``InputSpecLayer`` are tensor
        pass-throughs, so the logits are identical, but every wrapper forward runs
        ``.sum().item()`` in each spec-layer branch — roughly 4 CUDA syncs that PGD
        would otherwise pay ``num_steps`` times per mutation. Coverage hooks are
        unaffected: they are registered across the whole module tree, and the
        inner network is part of it.
        
        Args:
            model: VerifiableModel from model synthesis.
        
        Returns:
            Tuple of (inner network returning logits, output property to ascend).
        
        Raises:
            TypeError: If ``model`` is not a VerifiableModel, so no inner network
                and no property can be identified.
            ValueError: If the wrapper carries no OutputSpec; the gradient
                strategies have no objective without one.
        """
        if not isinstance(model, VerifiableModel):
            raise TypeError(
                f"MutationEngine requires a VerifiableModel to source the attack "
                f"objective, got {type(model).__name__}."
            )
        output_spec = model.output_spec.spec
        if output_spec is None:
            raise ValueError(
                "MutationEngine requires an OutputSpec: the gradient strategies "
                "ascend its violation severity."
            )
        return model.model, output_spec
    
    def _compute_fixed_perturb_size(self) -> float:
        """Return the range-aware perturbation baseline used by fixed mode."""
        if self.input_spec is None:
            print(
                "[MutationEngine] No InputSpec provided, falling back to "
                "fixed perturb_size=0.01"
            )
            return 0.01
        
        # Extract bounds based on InputSpec kind
        # BOX: explicit lb/ub bounds define the feasible region directly
        # LINF_BALL: feasible region is [center - eps, center + eps] (L∞ ball around center)
        if self.input_spec.kind == InKind.BOX:
            # BOX constraints: lb and ub are directly specified
            assert self.input_spec.lb is not None and self.input_spec.ub is not None
            lb = self.input_spec.lb
            ub = self.input_spec.ub
        elif self.input_spec.kind == InKind.LINF_BALL:
            # L∞ ball constraints: range is 2*eps around center point
            # The feasible region is all points x such that ||x - center||_∞ <= eps
            assert self.input_spec.center is not None and self.input_spec.eps is not None
            center = self.input_spec.center
            eps = torch.as_tensor(self.input_spec.eps, device=center.device, dtype=center.dtype)
            lb = center - eps
            ub = center + eps
        elif self.input_spec.kind == InKind.LP_EMBEDDING:
            lb, ub = self.input_spec.materialize_box_seed()
        else:
            print(
                f"[MutationEngine] Unsupported InputSpec kind "
                f"'{self.input_spec.kind}', falling back to fixed "
                "perturb_size=0.01"
            )
            return 0.01

        return (ub - lb).mean().item()
    
    def _setup_hooks(self):
        """Setup forward hooks to capture activations."""
        def make_hook(name):
            def hook(module, input, output):
                # Store activation (handle both tensor and dict outputs)
                if isinstance(output, torch.Tensor):
                    self.activation_map[name] = output.detach()
                elif isinstance(output, dict) and 'output' in output:
                    self.activation_map[name] = output['output'].detach()
            
            return hook
        
        # Register hooks on computational layers (ReLU, Linear, Conv2d)
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.ReLU, nn.Linear, nn.Conv2d)):
                module.register_forward_hook(make_hook(name))

    def _setup_feasibility_hooks(self) -> None:
        """Carry lane rows and input feasibility through the fuzzer inference."""
        def inject_rows(module, args, kwargs):
            rows = self._pending_rows
            if rows is None:
                return None
            self._pending_rows = None
            self._feasibility_forward_active = True
            if "rows" in kwargs:
                raise RuntimeError("Fuzzer inference rows were provided twice")
            kwargs["rows"] = rows
            return args, kwargs

        def attach_feasibility(module, args, kwargs, output):
            if not self._feasibility_forward_active:
                return None
            self._feasibility_forward_active = False
            if not isinstance(output, dict):
                raise RuntimeError(
                    "VerifiableModel.forward must return input feasibility metadata"
                )
            model_output = output.get("output")
            input_satisfied = output.get("input_satisfied_per_sample")
            if not isinstance(model_output, torch.Tensor) or not isinstance(
                input_satisfied, torch.Tensor
            ):
                raise RuntimeError(
                    "VerifiableModel.forward did not return a per-sample input "
                    "feasibility mask"
                )
            setattr(model_output, INPUT_FEASIBILITY_ATTR, input_satisfied)
            return None

        self.model.register_forward_pre_hook(inject_rows, with_kwargs=True)
        self.model.register_forward_hook(attach_feasibility, with_kwargs=True)
                
    def mutate(self, seeds: 'FuzzingSeed') -> torch.Tensor:
        """
        Apply mutation to seeds.
        
        A single strategy is selected for all seeds, enabling GPU parallelism
        for gradient-based strategies (FGSM/PGD).
        
        Args:
            seeds: FuzzingSeed batch with .tensor [B,C,H,W] and .original_index [B] int64
        
        Returns:
            Mutated tensor [B, C, H, W] satisfying InputSpec constraints
        """
        if not seeds:
            raise ValueError("Empty seed batch")
        
        B = len(seeds)
        
        # Use batch tensor directly: [B, C, H, W]
        batch_input = seeds.tensor.to(self.device)
        # Spec row backing each lane. The corpus samples WITH REPLACEMENT, so
        # lane i is not spec row i; identity rows would attack another
        # instance's property (and gather another instance's bounds below).
        rows = seeds.original_index.to(self.device)
        
        # Select strategy (same for all samples)
        strategy_names = list(self.weights.keys())
        strategy_probs = list(self.weights.values())
        strategy_name = np.random.choice(strategy_names, p=strategy_probs)
        strategy = self.strategies[strategy_name]
        
        # Store strategy for tracing
        self.last_strategy = strategy_name
        
        # Dynamic per-seed scale: s_b = 1 - (1 - s₀)^{n_b+1}
        if self.perturb_mode != "fixed" and self.input_spec is not None:
            if self.input_spec.kind == InKind.LINF_BALL:
                assert self.input_spec.eps is not None
                _orig = seeds.original_tensor.to(self.device)
                _eps = torch.as_tensor(self.input_spec.eps, device=_orig.device, dtype=_orig.dtype)
                _lb = _orig - _eps
                _ub = _orig + _eps
            else:
                if self.input_spec.kind == InKind.LP_EMBEDDING:
                    _lb, _ub = self.input_spec.materialize_box_seed()
                    _lb = _lb.to(self.device)
                    _ub = _ub.to(self.device)
                else:
                    assert self.input_spec.lb is not None and self.input_spec.ub is not None
                    _lb = self.input_spec.lb.to(self.device)
                    _ub = self.input_spec.ub.to(self.device)
                if _lb.shape[0] > 1:
                    assert bool((rows < _lb.shape[0]).all()), (
                        f"original_index out of range for InputSpec bounds: "
                        f"max={int(rows.max().item())} >= {_lb.shape[0]} rows"
                    )
                    _lb, _ub = _lb[rows], _ub[rows]
                elif _lb.shape[0] == 1 and B > 1:
                    _lb = _lb.expand(B, *_lb.shape[1:])
                    _ub = _ub.expand(B, *_ub.shape[1:])
                else:
                    _lb, _ub = _lb[:B], _ub[:B]
            n = seeds.select_count.float().to(self.device)
            s_b = 1.0 - (1.0 - self.perturb_scale) ** (n + 1.0)
            reach = torch.max(batch_input - _lb, _ub - batch_input)
            shape = [B] + [1] * (reach.dim() - 1)
            perturb_size = s_b.view(*shape) * reach
            if self.perturb_mode == "adaptive_scalar":
                perturb_size = perturb_size.mean(dim=list(range(1, perturb_size.dim())), keepdim=True)
            strategy.perturb_size = perturb_size
        
        # Apply mutation
        mutated = strategy.mutate(
            batch_input,
            self.attack_model,
            self.activation_map,
            rows=rows
        )
        
        # Project to InputSpec constraints
        mutated = self._project(mutated, seeds)

        # The next wrapped-model inference must check each lane against the
        # InputSpec row from which that seed originated.
        self._pending_rows = rows
        
        self.total_mutations += B
        return mutated
    
    def _project(self, tensor: torch.Tensor, seeds: 'Optional[FuzzingSeed]' = None) -> torch.Tensor:
        """
        Project mutated tensor back into the feasible region defined by InputSpec constraints.
        
        Since fuzzing batch size is aligned with model synthesis (N VNNLib instances),
        the InputSpec bounds (lb/ub) are already [N, ...] and match the batch dimension directly.
        
        **BOX (InKind.BOX)**:
            Clamps to per-sample lb/ub bounds from InputSpec (already batch-aligned).
            Uses seeds.original_index tensor to gather correct per-sample bounds.
        
        **LINF_BALL (InKind.LINF_BALL)**:
            Clamps perturbation delta to [-eps, +eps] around each seed's ORIGINAL input,
            preserving the L∞ distance invariant across mutation chains.
        
        **LIN_POLY (InKind.LIN_POLY)**:
            Not yet implemented — returns tensor unchanged.
        
        Args:
            tensor: Mutated input tensor [B, ...] to project
            seeds: FuzzingSeed batch with original_index [B] and original_tensor [B, ...]
        
        Returns:
            Projected tensor [B, ...] satisfying InputSpec constraints
        """
        if self.input_spec is None:
            return tensor
        
        B = tensor.shape[0]
        
        if self.input_spec.kind in (InKind.BOX, InKind.LP_EMBEDDING):
            if self.input_spec.kind == InKind.LP_EMBEDDING:
                lb, ub = self.input_spec.materialize_box_seed()
                lb = lb.to(tensor.device)
                ub = ub.to(tensor.device)
            else:
                assert self.input_spec.lb is not None and self.input_spec.ub is not None
                lb = self.input_spec.lb.to(tensor.device)
                ub = self.input_spec.ub.to(tensor.device)
            
            # bounds: use seeds.original_index to gather correct bounds
            if lb.shape[0] > 1 and seeds is not None:
                indices = seeds.original_index.to(lb.device)
                assert bool((indices < lb.shape[0]).all()), (
                    f"original_index out of range for InputSpec bounds: "
                    f"max={int(indices.max().item())} >= {lb.shape[0]} rows"
                )
                lb = lb[indices]  # (B, ...) vectorized gather
                ub = ub[indices]
            elif lb.shape[0] == 1 and B > 1:
                lb = lb.expand(B, *lb.shape[1:])
                ub = ub.expand(B, *ub.shape[1:])
            else:
                lb = lb[:B]
                ub = ub[:B]
            
            return torch.clamp(tensor, lb, ub)
        
        elif self.input_spec.kind == InKind.LINF_BALL:
            eps = self.input_spec.eps
            assert eps is not None
            
            assert seeds is not None and len(seeds) == B, \
                f"LINF_BALL projection requires seeds (got {len(seeds) if seeds else 0}, expected {B})"
            
            # Use original_tensor as center to maintain L∞ distance from original
            center = seeds.original_tensor.to(tensor.device)
            eps = torch.as_tensor(eps, device=tensor.device, dtype=tensor.dtype)
            
            delta = tensor - center
            delta = torch.clamp(delta, -eps, eps)
            projected = center + delta

            # Reconstructing a point exactly at ±eps can round one ULP outside
            # the ball. Pull only those coordinates one representable value
            # toward the center so the exact InputSpecLayer check agrees with
            # this projection.
            overshot = (projected - center).abs() > eps
            return torch.where(
                overshot, torch.nextafter(projected, center), projected
            )
        
        elif self.input_spec.kind == InKind.LIN_POLY:
            # TODO: Implement quadratic programming projection
            return tensor
        
        return tensor
    
    def get_activation_map(self) -> Dict[str, torch.Tensor]:
        """Get activations from last inference."""
        return self.activation_map
    
    
    def get_last_gradients(self) -> Optional[Dict[str, torch.Tensor]]:
        """Get gradients from last mutation (Level 3 tracing only)."""
        return self.last_gradients
    
    def get_last_loss(self) -> Optional[float]:
        """Get loss value from last mutation (Level 3 tracing only)."""
        return self.last_loss
    
    def get_stats(self) -> Dict[str, Any]:
        """Get mutation statistics."""
        perturb_size_info = {}
        for strategy_name, strategy in self.strategies.items():
            perturb_size = strategy.perturb_size
            if isinstance(perturb_size, torch.Tensor):
                perturb_size_info[strategy_name] = {
                    "type": "tensor",
                    "shape": list(perturb_size.shape),
                    "min": perturb_size.min().item(),
                    "max": perturb_size.max().item(),
                    "mean": perturb_size.mean().item()
                }
            else:
                perturb_size_info[strategy_name] = {
                    "type": "scalar",
                    "value": perturb_size
                }
        
        return {
            "total_mutations": self.total_mutations,
            "strategy_weights": self.weights,
            "perturb_mode": self.perturb_mode,
            "perturb_scale": self.perturb_scale,
            "perturb_size_values": perturb_size_info,
        }
