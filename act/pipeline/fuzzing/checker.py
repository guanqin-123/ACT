"""
Property violation detection for ACTFuzzer.

Checks if model outputs violate OutputSpec properties and records counterexamples.

Copyright (C) 2025 SVF-tools/ACT
License: AGPLv3+
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
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
        input: Input tensor that caused violation
        output: Model's output on this input
        expected: Expected value (e.g., true label)
        actual: Actual value (e.g., predicted label)
        kind: Type of violation (TOP1_ROBUST, MARGIN_ROBUST, etc.)
        confidence: Confidence score of the prediction
        timestamp: When the counterexample was found
    """
    input: torch.Tensor
    output: torch.Tensor
    expected: int
    actual: int
    kind: str
    confidence: float
    timestamp: float
    
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
            timestamp=data["timestamp"]
        )


class PropertyChecker:
    """
    Check OutputSpec violations and record counterexamples.
    
    Supports all OutKind types:
    - TOP1_ROBUST: Top prediction must equal true label
    - MARGIN_ROBUST: Margin to runner-up must exceed threshold
    - RANGE: Output must be within [lb, ub]
    - LINEAR_LE: Linear constraint c^T y <= d must hold
    
    Example:
        >>> checker = PropertyChecker(output_spec)
        >>> violation = checker.check(input_tensor, output, label=5)
        >>> if violation:
        ...     print(f"Found counterexample: {violation.summary()}")
        ...     violation.save("counterexample.pt")
    """
    
    def __init__(self, output_spec: Optional[OutputSpec]):
        """
        Initialize property checker.
        
        Args:
            output_spec: OutputSpec to check against (None = no checking)
        """
        self.spec = output_spec
    
    def check(self,
              input_tensor: torch.Tensor,
              output: torch.Tensor,
              label: Optional[int],
              seed_tensor: Optional[torch.Tensor] = None
             ) -> Optional[Counterexample]:
        """
        Check if output violates OutputSpec.
        
        Args:
            input_tensor: Input that was tested
            output: Model's output on input_tensor
            label: Ground truth label (if available)
            seed_tensor: Original unperturbed input (for distance computation)
        
        Returns:
            Counterexample if violation found, None otherwise
        """
        if self.spec is None or label is None:
            return None
        
        # Check based on spec kind
        if self.spec.kind == OutKind.TOP1_ROBUST:
            return self._check_top1(input_tensor, output, label)
        elif self.spec.kind == OutKind.MARGIN_ROBUST:
            return self._check_margin(input_tensor, output, label)
        elif self.spec.kind == OutKind.RANGE:
            return self._check_range(input_tensor, output, label)
        elif self.spec.kind == OutKind.LINEAR_LE:
            return self._check_linear(input_tensor, output, label)
        
        return None
    
    def _check_top1(self, 
                   input_tensor: torch.Tensor,
                   output: torch.Tensor, 
                   y_true: int
                  ) -> Optional[Counterexample]:
        """
        Check if top prediction != y_true (misclassification).
        
        This is a counterexample for robustness: the model changed its
        prediction from the true label to a different class.
        """
        # Handle both batched and unbatched outputs
        if output.dim() > 1:
            pred_class = output.argmax(dim=1).item()
            logits = output[0]
        else:
            pred_class = output.argmax().item()
            logits = output
        
        if pred_class != y_true:
            # Compute confidence
            probs = torch.softmax(logits, dim=0)
            confidence = probs[pred_class].item()
            
            return Counterexample(
                input=input_tensor.detach().cpu(),
                output=output.detach().cpu(),
                expected=y_true,
                actual=pred_class,
                kind="TOP1_ROBUST",
                confidence=confidence,
                timestamp=time.time()
            )
        
        return None
    
    def _check_margin(self,
                     input_tensor: torch.Tensor,
                     output: torch.Tensor,
                     y_true: int
                    ) -> Optional[Counterexample]:
        """
        Check if margin(y_true) < threshold.
        
        Margin = logit[y_true] - max(logit[i] for i != y_true)
        Counterexample if margin is too small (model not confident enough).
        """
        # Handle batched output
        if output.dim() > 1:
            logits = output[0]
        else:
            logits = output
        
        true_logit = logits[y_true]
        
        # Find runner-up (max logit excluding true class)
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[y_true] = False
        runner_up_logit = logits[mask].max()
        
        margin = (true_logit - runner_up_logit).item()
        
        if margin < self.spec.margin:
            return Counterexample(
                input=input_tensor.detach().cpu(),
                output=output.detach().cpu(),
                expected=y_true,
                actual=-1,  # Not misclassified, just low margin
                kind="MARGIN_ROBUST",
                confidence=margin,
                timestamp=time.time()
            )
        
        return None
    
    def _check_range(self,
                    input_tensor: torch.Tensor,
                    output: torch.Tensor,
                    y_true: int
                   ) -> Optional[Counterexample]:
        """Check if output is outside [lb, ub] bounds."""
        if self.spec.lb is None or self.spec.ub is None:
            return None
        
        # Check if any output element violates bounds
        below_lb = (output < self.spec.lb).any()
        above_ub = (output > self.spec.ub).any()
        
        if below_lb or above_ub:
            return Counterexample(
                input=input_tensor.detach().cpu(),
                output=output.detach().cpu(),
                expected=y_true,
                actual=-1,
                kind="RANGE",
                confidence=0.0,
                timestamp=time.time()
            )
        
        return None
    
    def _check_linear(self,
                     input_tensor: torch.Tensor,
                     output: torch.Tensor,
                     y_true: int
                    ) -> Optional[Counterexample]:
        """Check if linear constraint c^T y <= d is violated."""
        if self.spec.c is None or self.spec.d is None:
            return None
        
        # Compute c^T y
        if output.dim() > 1:
            y = output[0]
        else:
            y = output
        
        value = torch.dot(self.spec.c.to(y.device), y).item()
        
        if value > self.spec.d:
            return Counterexample(
                input=input_tensor.detach().cpu(),
                output=output.detach().cpu(),
                expected=y_true,
                actual=-1,
                kind="LINEAR_LE",
                confidence=value - self.spec.d,
                timestamp=time.time()
            )
        
        return None
