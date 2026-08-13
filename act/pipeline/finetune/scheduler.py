#===- act/pipeline/finetune/scheduler.py - Epsilon Scheduler ------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Epsilon scheduler for curriculum learning in robust training.
#   Gradually increase epsilon from small to target value.
#
#===---------------------------------------------------------------------===#

from typing import Literal
import math


class EpsilonScheduler:
    """
    Scheduler for epsilon in robust training (curriculum learning).
    
    Strategies:
    - linear: eps(t) = eps_start + (eps_end - eps_start) * t / T
    - exponential: eps(t) = eps_start * (eps_end/eps_start)^(t/T)
    - step: eps increases at fixed intervals
    - warmup: keep eps_start for warmup epochs, then linear ramp
    """
    
    def __init__(
        self,
        eps_start: float,
        eps_end: float,
        total_epochs: int,
        schedule: Literal['linear', 'exponential', 'step', 'warmup'] = 'linear',
        warmup_epochs: int = 0,
        step_epochs: int = 10,
    ):
        """
        Args:
            eps_start: Starting epsilon (usually small, e.g., 0.0 or 0.01)
            eps_end: Target epsilon (e.g., 0.1 for MNIST, 8/255 for CIFAR)
            total_epochs: Total training epochs
            schedule: Scheduling strategy
            warmup_epochs: Epochs to keep eps_start before ramping (for 'warmup')
            step_epochs: Epochs between steps (for 'step')
        """
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.total_epochs = total_epochs
        self.schedule = schedule
        self.warmup_epochs = warmup_epochs
        self.step_epochs = step_epochs
    
    def get(self, epoch: int) -> float:
        """Get epsilon for given epoch (0-indexed)."""
        if epoch >= self.total_epochs:
            return self.eps_end
        
        if self.schedule == 'linear':
            return self._linear(epoch)
        elif self.schedule == 'exponential':
            return self._exponential(epoch)
        elif self.schedule == 'step':
            return self._step(epoch)
        elif self.schedule == 'warmup':
            return self._warmup(epoch)
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")
    
    def _linear(self, epoch: int) -> float:
        """Linear interpolation."""
        t = epoch / max(self.total_epochs - 1, 1)
        return self.eps_start + (self.eps_end - self.eps_start) * t
    
    def _exponential(self, epoch: int) -> float:
        """Exponential interpolation."""
        if self.eps_start <= 0:
            # Fallback to linear if start is 0
            return self._linear(epoch)
        t = epoch / max(self.total_epochs - 1, 1)
        ratio = self.eps_end / self.eps_start
        return self.eps_start * (ratio ** t)
    
    def _step(self, epoch: int) -> float:
        """Step-wise increase."""
        num_steps = self.total_epochs // self.step_epochs
        current_step = epoch // self.step_epochs
        if num_steps <= 0:
            return self.eps_end
        t = current_step / num_steps
        return self.eps_start + (self.eps_end - self.eps_start) * t
    
    def _warmup(self, epoch: int) -> float:
        """Warmup then linear ramp."""
        if epoch < self.warmup_epochs:
            return self.eps_start
        remaining = self.total_epochs - self.warmup_epochs
        t = (epoch - self.warmup_epochs) / max(remaining - 1, 1)
        return self.eps_start + (self.eps_end - self.eps_start) * t
    
    def __repr__(self) -> str:
        return (f"EpsilonScheduler({self.schedule}, "
                f"eps={self.eps_start:.4f}->{self.eps_end:.4f}, "
                f"epochs={self.total_epochs})")
