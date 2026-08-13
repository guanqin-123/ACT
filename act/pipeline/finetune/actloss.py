#===- act/pipeline/finetune/actloss.py - Robust Loss Base Classes -------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Abstract base class for robust training losses.
#   Supports both provable (certified) and adversarial (empirical) training.
#
#===---------------------------------------------------------------------===#

from abc import ABC, abstractmethod
from typing import Dict, Tuple
import torch
import torch.nn as nn


class RobustLoss(ABC):
    """Abstract base class for robust training losses."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the loss strategy."""
        pass
    
    @property
    def is_certified(self) -> bool:
        """Whether this loss provides certified guarantees."""
        return False
    
    @abstractmethod
    def __call__(
        self,
        model: nn.Module,
        X: torch.Tensor,
        y: torch.Tensor,
        epsilon: float,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute robust loss.
        
        Args:
            model: Neural network (nn.Sequential or similar)
            X: Input batch [B, ...]
            y: Labels [B]
            epsilon: Perturbation radius (L-inf)
            
        Returns:
            loss: Scalar loss tensor (differentiable for training)
            metrics: Dict with 'clean_acc', 'robust_acc', 'certified_acc', etc.
        """
        pass
    
    def eval_metrics(
        self,
        model: nn.Module,
        X: torch.Tensor,
        y: torch.Tensor,
        epsilon: float,
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics without loss (for validation).
        Default implementation calls __call__ and discards loss.
        """
        with torch.no_grad():
            _, metrics = self(model, X, y, epsilon)
        return metrics
