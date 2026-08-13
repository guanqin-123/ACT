"""
Property violation detection for ACTFuzzer.

Checks if model outputs violate OutputSpec properties and records counterexamples.

Copyright (C) 2025 SVF-tools/ACT
License: AGPLv3+
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Callable
import time
import torch

from act.front_end.specs import OutputSpec, OutKind


@dataclass
class Counterexample:
    """
    Counterexample with full details.
    
    Represents an input that violates the OutputSpec property.
    This is the primary output of ACTFuzzer.
    
    Attributes:
        input: Input tensor that caused violation (perturbed)
        output: Model's output on this input
        expected: Expected value (e.g., true label)
        actual: Actual value (e.g., predicted label)
        kind: Type of violation (TOP1_ROBUST, MARGIN_ROBUST, etc.)
        confidence: Confidence score of the prediction
        timestamp: When the counterexample was found
        seed_index: Index of the original seed (optional, for tracking)
        seed_input: Original unperturbed input (optional, for visualization)
    """
    input: torch.Tensor
    output: torch.Tensor
    expected: int
    actual: int
    kind: str
    confidence: float
    timestamp: float
    seed_index: Optional[int] = None
    seed_input: Optional[torch.Tensor] = None
    
    def summary(self) -> str:
        """One-line summary of the counterexample."""
        return f"{self.kind}: expected {self.expected}, got {self.actual} (conf={self.confidence:.3f})"
    
    def save(self, path):
        """Save counterexample to disk."""
        torch.save({
            "input": self.input,
            "output": self.output,
            "expected": self.expected,
            "actual": self.actual,
            "kind": self.kind,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "seed_index": self.seed_index,
            "seed_input": self.seed_input,
        }, path)
    
    @staticmethod
    def load(path):
        """Load counterexample from disk."""
        data = torch.load(path)
        return Counterexample(
            input=data["input"],
            output=data["output"],
            expected=data["expected"],
            actual=data["actual"],
            kind=data["kind"],
            confidence=data["confidence"],
            timestamp=data["timestamp"],
            seed_index=data.get("seed_index"),
            seed_input=data.get("seed_input")
        )


# =============================================================================
# Property Checking (Batched - handles both single and batched inputs)
# =============================================================================

@dataclass
class _BatchContext:
    """Pre-computed tensors shared across check methods."""
    inputs: torch.Tensor       # [B, ...]
    outputs: torch.Tensor      # [B, num_classes]
    labels: List[Optional[int]]
    y_true: torch.Tensor       # [B] - labels as tensor (-1 for None)
    valid_mask: torch.Tensor   # [B] - True where label is not None
    seed_tensors: Optional[List[torch.Tensor]]
    seed_indices: Optional[List[int]]
    device: torch.device
    B: int
    num_classes: int


class PropertyChecker:
    """
    Property checker for violation detection (handles both single and batched inputs).
    
    Supports all OutKind types:
    - TOP1_ROBUST: Top prediction must equal true label
    - MARGIN_ROBUST: Margin to runner-up must exceed threshold
    - RANGE: Output must be within [lb, ub]
    - LINEAR_LE: Linear constraint c^T y <= d must hold
    
    For single samples, use check(). For batched samples, use check_batch().
    
    Example (single):
        >>> checker = PropertyChecker(output_spec)
        >>> violation = checker.check(input_tensor, output, label=5)
        >>> if violation:
        ...     print(f"Found counterexample: {violation.summary()}")
    
    Example (batched):
        >>> checker = PropertyChecker(output_spec)
        >>> violations = checker.check_batch(inputs, outputs, labels)
        >>> # violations is List[Counterexample | None] of length B
    """
    
    def __init__(self, output_spec: Optional[OutputSpec]):
        """Initialize batched property checker."""
        self.spec = output_spec
        
        # Dispatch table for spec kinds
        self._dispatch: dict[str, Callable[[_BatchContext], List[Optional[Counterexample]]]] = {
            OutKind.TOP1_ROBUST: self._check_top1,
            OutKind.MARGIN_ROBUST: self._check_margin,
            OutKind.RANGE: self._check_range,
            OutKind.LINEAR_LE: self._check_linear,
        }
    
    def check_batch(
        self,
        inputs: torch.Tensor,
        outputs: torch.Tensor,
        labels: List[Optional[int]],
        seed_tensors: Optional[List[torch.Tensor]] = None,
        seed_indices: Optional[List[int]] = None
    ) -> List[Optional[Counterexample]]:
        """
        Check B samples for violations in parallel.
        
        Args:
            inputs: Input tensors [B, C, H, W] or [B, D]
            outputs: Model outputs [B, num_classes]
            labels: List of B ground truth labels (None entries skip checking)
            seed_tensors: Optional list of original seed tensors
            seed_indices: Optional list of seed indices
        
        Returns:
            List of B elements, each Counterexample or None
        """
        B = inputs.shape[0]
        
        if self.spec is None:
            return [None] * B
        
        handler = self._dispatch.get(self.spec.kind)
        if handler is None:
            return [None] * B
        
        # Build shared context (computed once, used by all check methods)
        device = outputs.device
        y_true = torch.tensor(
            [l if l is not None else -1 for l in labels],
            dtype=torch.long, device=device
        )
        
        ctx = _BatchContext(
            inputs=inputs,
            outputs=outputs,
            labels=labels,
            y_true=y_true,
            valid_mask=(y_true >= 0),
            seed_tensors=seed_tensors,
            seed_indices=seed_indices,
            device=device,
            B=B,
            num_classes=outputs.shape[1],
        )
        
        return handler(ctx)
    
    def _build_results(
        self,
        ctx: _BatchContext,
        violations_mask: torch.Tensor,
        kind: str,
        actual_values: torch.Tensor,
        confidence_values: torch.Tensor,
    ) -> List[Optional[Counterexample]]:
        """Build Counterexample list from violation mask."""
        timestamp = time.time()
        violation_indices = violations_mask.nonzero(as_tuple=True)[0]
        
        results: List[Optional[Counterexample]] = [None] * ctx.B
        
        for idx in violation_indices:
            i = idx.item()
            results[i] = Counterexample(
                input=ctx.inputs[i].detach().cpu(),
                output=ctx.outputs[i].detach().cpu(),
                expected=ctx.labels[i],
                actual=int(actual_values[i].item()),
                kind=kind,
                confidence=float(confidence_values[i].item()),
                timestamp=timestamp,
                seed_index=ctx.seed_indices[i] if ctx.seed_indices else None,
                seed_input=ctx.seed_tensors[i].detach().cpu() if ctx.seed_tensors else None,
            )
        
        return results
    
    def _check_top1(self, ctx: _BatchContext) -> List[Optional[Counterexample]]:
        """Check if top prediction != y_true for B samples."""
        pred_classes = ctx.outputs.argmax(dim=1)
        violations_mask = ctx.valid_mask & (pred_classes != ctx.y_true)
        
        # Confidence = probability of predicted class
        probs = torch.softmax(ctx.outputs, dim=1)
        confidences = probs.gather(1, pred_classes.unsqueeze(1)).squeeze(1)
        
        return self._build_results(ctx, violations_mask, "TOP1_ROBUST", pred_classes, confidences)
    
    def _check_margin(self, ctx: _BatchContext) -> List[Optional[Counterexample]]:
        """Check if margin(y_true) < threshold for B samples."""
        # Use 0 for invalid labels to avoid index errors (valid_mask handles filtering)
        y_safe = ctx.y_true.clamp(min=0)
        
        # True class logits
        true_logits = ctx.outputs.gather(1, y_safe.unsqueeze(1)).squeeze(1)
        
        # Runner-up logits (max over non-true classes)
        mask = torch.ones(ctx.B, ctx.num_classes, dtype=torch.bool, device=ctx.device)
        mask.scatter_(1, y_safe.unsqueeze(1), False)
        runner_up_logits = ctx.outputs.masked_fill(~mask, float('-inf')).max(dim=1).values
        
        margins = true_logits - runner_up_logits
        threshold = getattr(self.spec, 'margin', 0.0) or 0.0
        violations_mask = ctx.valid_mask & (margins < threshold)
        
        actual = torch.full((ctx.B,), -1, dtype=torch.long, device=ctx.device)
        return self._build_results(ctx, violations_mask, "MARGIN_ROBUST", actual, margins)
    
    def _check_range(self, ctx: _BatchContext) -> List[Optional[Counterexample]]:
        """Check if outputs are outside [lb, ub] bounds for B samples."""
        if self.spec.lb is None or self.spec.ub is None:
            return [None] * ctx.B
        
        lb = self._to_tensor(self.spec.lb, ctx.device)
        ub = self._to_tensor(self.spec.ub, ctx.device)
        
        # Violation if any output is outside [lb, ub]
        violations_mask = ((ctx.outputs < lb) | (ctx.outputs > ub)).any(dim=1)
        
        # Confidence = max violation magnitude
        lb_viol = (lb - ctx.outputs).clamp(min=0).max(dim=1).values
        ub_viol = (ctx.outputs - ub).clamp(min=0).max(dim=1).values
        confidences = torch.maximum(lb_viol, ub_viol)
        
        actual = torch.full((ctx.B,), -1, dtype=torch.long, device=ctx.device)
        return self._build_results(ctx, violations_mask, "RANGE", actual, confidences)
    
    def _check_linear(self, ctx: _BatchContext) -> List[Optional[Counterexample]]:
        """Check if linear constraint c^T y <= d is violated for B samples."""
        if self.spec.c is None or self.spec.d is None:
            return [None] * ctx.B
        
        c = self.spec.c.to(ctx.device)
        d = float(self.spec.d)
        
        # c^T y for all samples
        values = (ctx.outputs * c).sum(dim=1)
        violations_mask = (values > d)
        confidences = values - d  # violation magnitude
        
        actual = torch.full((ctx.B,), -1, dtype=torch.long, device=ctx.device)
        return self._build_results(ctx, violations_mask, "LINEAR_LE", actual, confidences)
    
    def check(
        self,
        input_tensor: torch.Tensor,
        output: torch.Tensor,
        label: Optional[int],
        seed_tensor: Optional[torch.Tensor] = None
    ) -> Optional[Counterexample]:
        """
        Check single sample for violation (convenience wrapper).
        
        Args:
            input_tensor: Input tensor [1, ...] or [...]
            output: Model output [1, num_classes] or [num_classes]
            label: Ground truth label
            seed_tensor: Original unperturbed input
        
        Returns:
            Counterexample if violation found, None otherwise
        """
        # Ensure batch dimension
        if input_tensor.dim() == output.dim():
            inp = input_tensor.unsqueeze(0) if input_tensor.dim() < 2 or input_tensor.shape[0] != 1 else input_tensor
            out = output.unsqueeze(0) if output.dim() < 2 or output.shape[0] != 1 else output
        else:
            inp = input_tensor if input_tensor.shape[0] == 1 else input_tensor.unsqueeze(0)
            out = output if output.shape[0] == 1 else output.unsqueeze(0)
        
        results = self.check_batch(
            inputs=inp,
            outputs=out,
            labels=[label],
            seed_tensors=[seed_tensor] if seed_tensor is not None else None
        )
        return results[0]
    
    @staticmethod
    def _to_tensor(val, device: torch.device) -> torch.Tensor:
        """Convert value to tensor on device."""
        if isinstance(val, torch.Tensor):
            return val.to(device)
        return torch.tensor(val, device=device)
