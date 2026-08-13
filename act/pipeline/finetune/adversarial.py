#===- act/pipeline/finetune/adversarial.py - Adversarial Robust Loss ----====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Adversarial robust loss using front-end fuzzer mutations for attack generation.
#   Reuses PGDMutation/FGSMMutation from act.pipeline.fuzzing.mutations.
#
#   This is empirical training (not certified) - provides robustness against
#   tested attacks but no formal guarantees.
#
#===---------------------------------------------------------------------===#

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Literal

from .actloss import RobustLoss


# ============================================================================
# Evaluation Functions
# ============================================================================

def evaluate_pgd(
    model: nn.Module,
    loader,
    epsilon: float,
    num_steps: int = 40,
    step_size: Optional[float] = None,
    device: Optional[torch.device] = None,
) -> float:
    """
    Evaluate PGD adversarial error using ACT front-end.
    
    Args:
        model: Neural network to evaluate
        loader: DataLoader with (X, y) batches
        epsilon: L-inf perturbation bound
        num_steps: Number of PGD iterations (default: 40)
        step_size: Per-step size (default: 2.5 * epsilon / num_steps)
        device: Device to use (default: infer from model)
        
    Returns:
        pgd_error: Percentage of samples where PGD attack succeeds (0-100)
    """
    attacker = AdversarialLoss(
        attack='pgd',
        attack_steps=num_steps,
        step_size=step_size,
        random_start=True,
    )
    
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    total = 0
    misclassified = 0
    
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        
        # Generate adversarial examples using front-end
        X_adv = attacker.generate_adversarial(model, X, y, epsilon)
        
        # Check if attack succeeds
        with torch.no_grad():
            outputs = model(X_adv)
            preds = outputs.argmax(dim=1)
            misclassified += (preds != y).sum().item()
            total += y.size(0)
    
    return (misclassified / total) * 100 if total > 0 else 0.0


def evaluate_fgsm(
    model: nn.Module,
    loader,
    epsilon: float,
    device: Optional[torch.device] = None,
) -> float:
    """
    Evaluate FGSM adversarial error using ACT front-end.
    
    Args:
        model: Neural network to evaluate
        loader: DataLoader with (X, y) batches
        epsilon: L-inf perturbation bound
        device: Device to use (default: infer from model)
        
    Returns:
        fgsm_error: Percentage of samples where FGSM attack succeeds (0-100)
    """
    attacker = AdversarialLoss(attack='fgsm')
    
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    total = 0
    misclassified = 0
    
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        X_adv = attacker.generate_adversarial(model, X, y, epsilon)
        
        with torch.no_grad():
            outputs = model(X_adv)
            preds = outputs.argmax(dim=1)
            misclassified += (preds != y).sum().item()
            total += y.size(0)
    
    return (misclassified / total) * 100 if total > 0 else 0.0


# ============================================================================
# AdversarialLoss Class
# ============================================================================

class AdversarialLoss(RobustLoss):
    """
    Adversarial robust loss using front-end fuzzer mutations.
    
    Reuses PGDMutation/FGSMMutation from act.pipeline.fuzzing.mutations
    to generate adversarial examples, then computes training loss.
    
    This is empirical (not certified) - provides robustness against tested samples.
    
    Example:
        >>> loss_fn = AdversarialLoss(attack='pgd', attack_steps=10)
        >>> loss, metrics = loss_fn(model, X, y, epsilon=0.1)
        >>> loss.backward()
    """
    
    def __init__(
        self,
        attack: Literal['pgd', 'fgsm'] = 'pgd',
        attack_steps: int = 10,
        step_size: Optional[float] = None,
        random_start: bool = True,
        input_clamp: Tuple[float, float] = (0.0, 1.0),
        loss_type: Literal['ce', 'trades'] = 'ce',
        trades_beta: float = 6.0,
    ):
        """
        Initialize AdversarialLoss with front-end mutation strategy.
        
        Args:
            attack: Attack method - 'pgd' (iterative) or 'fgsm' (single-step)
            attack_steps: Number of PGD iterations (ignored for FGSM)
            step_size: Per-step size for PGD (default: 2.5 * epsilon / attack_steps)
            random_start: Random initialization within epsilon ball
            input_clamp: (min, max) valid input range (e.g., (0, 1) for images)
            loss_type: Training loss - 'ce' (standard AT) or 'trades'
            trades_beta: Beta parameter for TRADES loss (weight of KL term)
        """
        self.attack = attack
        self.attack_steps = attack_steps
        self.step_size = step_size
        self.random_start = random_start
        self.input_min, self.input_max = input_clamp
        self.loss_type = loss_type
        self.trades_beta = trades_beta
        
        # Mutation strategy will be created lazily with correct epsilon
        self._mutation = None
        self._current_epsilon = None
    
    @property
    def name(self) -> str:
        return f"adversarial-{self.attack}{self.attack_steps if self.attack == 'pgd' else ''}"
    
    @property
    def is_certified(self) -> bool:
        return False
    
    def _get_mutation(self, epsilon: float):
        """
        Get or create mutation strategy for current epsilon.
        
        imports from front-end to avoid circular imports.
        Recreates mutation if epsilon changes.
        """
        if self._mutation is None or self._current_epsilon != epsilon:
            from act.pipeline.fuzzing.mutations import PGDMutation, FGSMMutation
            
            # Compute step size: default heuristic from Madry et al.
            step_size = self.step_size if self.step_size else 2.5 * epsilon / self.attack_steps
            
            if self.attack == 'pgd':
                self._mutation = PGDMutation(
                    perturb_size=epsilon,
                    num_steps=self.attack_steps,
                    step_size=step_size,
                    random_start=self.random_start,
                )
            elif self.attack == 'fgsm':
                self._mutation = FGSMMutation(perturb_size=epsilon)
            else:
                raise ValueError(f"Unknown attack: {self.attack}. Use 'pgd' or 'fgsm'.")
            
            self._current_epsilon = epsilon
        
        return self._mutation
    
    def __call__(
        self,
        model: nn.Module,
        X: torch.Tensor,
        y: torch.Tensor,
        epsilon: float,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute adversarial robust loss.
        
        Args:
            model: Neural network (nn.Module)
            X: Input batch [B, ...]
            y: Labels [B]
            epsilon: L-inf perturbation radius
            
        Returns:
            loss: Differentiable loss tensor (for training)
            metrics: {'robust_acc', 'clean_acc', 'loss'}
        """
        # Generate adversarial examples using front-end mutation
        X_adv = self._generate_adversarial_batch(model, X, y, epsilon)
        
        # Compute loss
        if self.loss_type == 'ce':
            # Standard adversarial training: CE on adversarial examples
            adv_out = model(X_adv)
            loss = F.cross_entropy(adv_out, y)
        else:  # TRADES
            # TRADES: clean CE + KL divergence between clean and adv predictions
            clean_out = model(X)
            adv_out = model(X_adv)
            
            loss_clean = F.cross_entropy(clean_out, y)
            loss_kl = F.kl_div(
                F.log_softmax(adv_out, dim=1),
                F.softmax(clean_out, dim=1).detach(),  # Detach clean for KL
                reduction='batchmean'
            )
            loss = loss_clean + self.trades_beta * loss_kl
        
        # Compute metrics (no grad needed)
        with torch.no_grad():
            clean_out = model(X)
            clean_pred = clean_out.argmax(dim=1)
            clean_acc = (clean_pred == y).float().mean().item()
            
            adv_out = model(X_adv)
            adv_pred = adv_out.argmax(dim=1)
            robust_acc = (adv_pred == y).float().mean().item()
        
        metrics = {
            'robust_acc': robust_acc,
            'clean_acc': clean_acc,
            'loss': loss.item(),
        }
        
        return loss, metrics
    
    def _generate_adversarial_batch(
        self,
        model: nn.Module,
        X: torch.Tensor,
        y: torch.Tensor,
        epsilon: float,
    ) -> torch.Tensor:
        """
        Generate adversarial examples for a batch using front-end mutation.
        
        Args:
            model: Neural network
            X: Input batch [B, ...]
            y: Labels [B]
            epsilon: L-inf perturbation radius
            
        Returns:
            X_adv: Adversarial batch [B, ...] (detached, no grad)
        """
        mutation = self._get_mutation(epsilon)
        
        # Save and restore default dtype (front-end may change it via device_manager)
        original_dtype = torch.get_default_dtype()
        
        # Set model to eval for attack generation
        was_training = model.training
        model.eval()
        
        B = X.size(0)
        X_adv_list = []
        
        for i in range(B):
            x_i = X[i:i+1]  # Keep batch dimension [1, ...]
            label_i = y[i].item()
            
            # Use front-end mutation (handles gradient computation internally)
            x_adv_i = mutation.mutate(x_i, model, label=label_i)
            
            # Ensure output matches input dtype (mutation may change it)
            if x_adv_i.dtype != X.dtype:
                x_adv_i = x_adv_i.to(X.dtype)
            
            # Clamp to valid input range
            x_adv_i = x_adv_i.clamp(self.input_min, self.input_max)
            
            X_adv_list.append(x_adv_i)
        
        # Restore model training mode
        if was_training:
            model.train()
        
        # Restore original default dtype (in case mutation changed it)
        if torch.get_default_dtype() != original_dtype:
            torch.set_default_dtype(original_dtype)
        
        return torch.cat(X_adv_list, dim=0).detach()
    
    def generate_adversarial(
        self,
        model: nn.Module,
        X: torch.Tensor,
        y: torch.Tensor,
        epsilon: float,
    ) -> torch.Tensor:
        """
        Public API to generate adversarial examples without computing loss.
        
        Useful for evaluation or visualization.
        
        Args:
            model: Neural network
            X: Input batch [B, ...]
            y: Labels [B]
            epsilon: L-inf perturbation radius
            
        Returns:
            X_adv: Adversarial batch [B, ...]
        """
        return self._generate_adversarial_batch(model, X, y, epsilon)
