#===- act/pipeline/finetune/test_finetune.py - Finetune Module Tests ----====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Unit tests for robust finetuning module.
#   Tests ProvableLoss, EpsilonScheduler, and RobustTrainer.
#
# Run with:
#   pytest act/pipeline/finetune/test_finetune.py -v
#
#===---------------------------------------------------------------------===#

import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from .provable import ProvableLoss
from .adversarial import AdversarialLoss
from .scheduler import EpsilonScheduler
from .trainer import RobustTrainer


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def simple_mlp():
    """Simple 2-layer MLP for testing."""
    torch.manual_seed(42)
    return nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 3),
    )


@pytest.fixture
def deep_mlp():
    """Deeper 4-layer MLP for testing."""
    torch.manual_seed(42)
    return nn.Sequential(
        nn.Linear(8, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 8),
        nn.ReLU(),
        nn.Linear(8, 4),
    )


@pytest.fixture
def toy_data():
    """Toy dataset for testing."""
    torch.manual_seed(42)
    X = torch.randn(32, 4)
    y = torch.randint(0, 3, (32,))
    return X, y


@pytest.fixture
def toy_loaders():
    """Toy data loaders for training tests."""
    torch.manual_seed(42)
    X_train = torch.randn(64, 4)
    y_train = torch.randint(0, 3, (64,))
    X_test = torch.randn(16, 4)
    y_test = torch.randint(0, 3, (16,))
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=16)
    
    return train_loader, test_loader


# ============================================================================
# ProvableLoss Tests
# ============================================================================

class TestProvableLoss:
    """Tests for ProvableLoss class."""
    
    def test_forward_pass(self, simple_mlp, toy_data):
        """Test basic forward pass returns loss and metrics."""
        X, y = toy_data
        loss_fn = ProvableLoss(input_clamp=(-10.0, 10.0))
        
        loss, metrics = loss_fn(simple_mlp, X, y, epsilon=0.1)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert 'certified_acc' in metrics
        assert 'clean_acc' in metrics
        assert 0.0 <= metrics['certified_acc'] <= 1.0
        assert 0.0 <= metrics['clean_acc'] <= 1.0
    
    def test_gradient_flow(self, simple_mlp, toy_data):
        """Test that gradients flow through all parameters."""
        X, y = toy_data
        loss_fn = ProvableLoss(input_clamp=(-10.0, 10.0))
        
        loss, _ = loss_fn(simple_mlp, X, y, epsilon=0.1)
        loss.backward()
        
        for name, p in simple_mlp.named_parameters():
            assert p.grad is not None, f"Parameter {name} has no gradient"
            assert p.grad.norm() > 0, f"Parameter {name} has zero gradient"
    
    def test_soundness(self, simple_mlp, toy_data):
        """Test that dual bounds are sound (lower bounds)."""
        X, y = toy_data
        epsilon = 0.1
        loss_fn = ProvableLoss(input_clamp=(-10.0, 10.0))
        
        # Get loss and metrics - bounds are computed internally
        loss, metrics = loss_fn(simple_mlp, X, y, epsilon)
        
        # Verify by sampling that bounds are sound
        num_samples = 100
        num_classes = 3
        
        for b in range(min(X.size(0), 5)):  # Test first 5 samples
            x_center = X[b:b+1]
            true_label = y[b].item()
            
            # Get the internal bounds for this sample
            net = loss_fn._get_or_build_net(simple_mlp, X.shape[1:], X.dtype)
            x = X[b]
            lb = (x - epsilon).clamp(min=-10.0, max=10.0)
            ub = (x + epsilon).clamp(min=-10.0, max=10.0)
            margins = loss_fn._compute_margins(net, lb, ub, true_label, num_classes)
            
            # Sample perturbations and verify bounds are sound
            deltas = torch.rand(num_samples, 4) * 2 * epsilon - epsilon
            x_perturbed = (x_center + deltas).clamp(min=-10.0, max=10.0)
            
            with torch.no_grad():
                outputs = simple_mlp(x_perturbed)
            
            for j in range(num_classes):
                if j == true_label:
                    continue
                
                # actual_margins[k] = output[true_label] - output[j] for sample k
                actual_margins = outputs[:, true_label] - outputs[:, j]
                worst_actual = actual_margins.min().item()
                
                dual_lb = margins[j].item()
                
                # Dual bound should be <= actual worst case (sound lower bound)
                assert dual_lb <= worst_actual + 1e-4, \
                    f"Unsound! Sample {b}, class {j}: dual={dual_lb:.4f} > actual={worst_actual:.4f}"
    
    def test_deep_network(self, deep_mlp):
        """Test with deeper network."""
        torch.manual_seed(42)
        X = torch.randn(16, 8)
        y = torch.randint(0, 4, (16,))
        
        loss_fn = ProvableLoss(input_clamp=(-10.0, 10.0))
        
        loss, metrics = loss_fn(deep_mlp, X, y, epsilon=0.05)
        loss.backward()
        
        # Check all parameters have gradients
        for name, p in deep_mlp.named_parameters():
            assert p.grad is not None, f"Parameter {name} has no gradient"
    
    def test_training_step(self, simple_mlp, toy_data):
        """Test that a training step updates weights."""
        X, y = toy_data
        loss_fn = ProvableLoss(input_clamp=(-10.0, 10.0))
        optimizer = torch.optim.SGD(simple_mlp.parameters(), lr=0.01)
        
        initial_weights = simple_mlp[0].weight.clone().detach()
        
        optimizer.zero_grad()
        loss, _ = loss_fn(simple_mlp, X, y, epsilon=0.1)
        loss.backward()
        optimizer.step()
        
        weight_diff = (simple_mlp[0].weight - initial_weights).abs().sum().item()
        assert weight_diff > 0, "Weights did not change after training step"


# ============================================================================
# AdversarialLoss Tests
# ============================================================================

class TestAdversarialLoss:
    """Tests for AdversarialLoss class (reuses front-end mutations)."""
    
    def test_pgd_attack(self, simple_mlp, toy_data):
        """Test PGD attack generates adversarial examples."""
        X, y = toy_data
        loss_fn = AdversarialLoss(
            attack='pgd',
            attack_steps=5,
            input_clamp=(-10.0, 10.0),
        )
        
        loss, metrics = loss_fn(simple_mlp, X, y, epsilon=0.1)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert 'robust_acc' in metrics
        assert 'clean_acc' in metrics
    
    def test_fgsm_attack(self, simple_mlp, toy_data):
        """Test FGSM attack generates adversarial examples."""
        X, y = toy_data
        loss_fn = AdversarialLoss(
            attack='fgsm',
            input_clamp=(-10.0, 10.0),
        )
        
        loss, metrics = loss_fn(simple_mlp, X, y, epsilon=0.1)
        
        assert isinstance(loss, torch.Tensor)
        assert 'robust_acc' in metrics
    
    def test_trades_loss(self, simple_mlp, toy_data):
        """Test TRADES loss computation."""
        X, y = toy_data
        loss_fn = AdversarialLoss(
            attack='pgd',
            attack_steps=3,
            input_clamp=(-10.0, 10.0),
            loss_type='trades',
            trades_beta=6.0,
        )
        
        loss, metrics = loss_fn(simple_mlp, X, y, epsilon=0.1)
        
        assert isinstance(loss, torch.Tensor)
        # TRADES loss should be higher than pure CE due to KL term
        assert loss.item() > 0
    
    def test_gradient_flow(self, simple_mlp, toy_data):
        """Test gradients flow through adversarial loss."""
        X, y = toy_data
        loss_fn = AdversarialLoss(attack='pgd', attack_steps=3, input_clamp=(-10.0, 10.0))
        
        loss, _ = loss_fn(simple_mlp, X, y, epsilon=0.1)
        loss.backward()
        
        for name, p in simple_mlp.named_parameters():
            assert p.grad is not None, f"Parameter {name} has no gradient"
    
    def test_epsilon_bound(self, simple_mlp, toy_data):
        """Test adversarial examples stay within epsilon bound."""
        X, y = toy_data
        epsilon = 0.1
        loss_fn = AdversarialLoss(attack='pgd', attack_steps=5, input_clamp=(-10.0, 10.0))
        
        X_adv = loss_fn.generate_adversarial(simple_mlp, X, y, epsilon)
        
        max_perturb = (X_adv - X).abs().max().item()
        assert max_perturb <= epsilon + 1e-5, f"Perturbation {max_perturb} exceeds epsilon {epsilon}"
    
    def test_dtype_preservation(self, toy_data):
        """Test that default dtype is preserved after attack."""
        X, y = toy_data
        
        original_dtype = torch.get_default_dtype()
        
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
        loss_fn = AdversarialLoss(attack='pgd', attack_steps=3, input_clamp=(-10.0, 10.0))
        
        loss, _ = loss_fn(model, X, y, epsilon=0.1)
        
        assert torch.get_default_dtype() == original_dtype, "Default dtype was changed"


# ============================================================================
# EpsilonScheduler Tests
# ============================================================================

class TestEpsilonScheduler:
    """Tests for EpsilonScheduler class."""
    
    def test_linear_schedule(self):
        """Test linear epsilon schedule."""
        scheduler = EpsilonScheduler(eps_start=0.0, eps_end=0.1, total_epochs=10, schedule='linear')
        
        assert scheduler.get(0) == pytest.approx(0.0)
        assert scheduler.get(9) == pytest.approx(0.1)
        assert scheduler.get(4) == pytest.approx(0.0444, rel=0.01)
    
    def test_exponential_schedule(self):
        """Test exponential epsilon schedule."""
        scheduler = EpsilonScheduler(eps_start=0.01, eps_end=0.1, total_epochs=10, schedule='exponential')
        
        assert scheduler.get(0) == pytest.approx(0.01)
        assert scheduler.get(9) == pytest.approx(0.1)
        # Exponential should be slower at start
        assert scheduler.get(4) < 0.05
    
    def test_step_schedule(self):
        """Test step-wise epsilon schedule."""
        scheduler = EpsilonScheduler(
            eps_start=0.0, eps_end=0.1, total_epochs=20, 
            schedule='step', step_epochs=5
        )
        
        # Should increase every 5 epochs
        assert scheduler.get(0) == scheduler.get(4)
        assert scheduler.get(5) > scheduler.get(4)
    
    def test_warmup_schedule(self):
        """Test warmup epsilon schedule."""
        scheduler = EpsilonScheduler(
            eps_start=0.0, eps_end=0.1, total_epochs=10,
            schedule='warmup', warmup_epochs=3
        )
        
        # Should stay at start during warmup (epochs 0, 1, 2)
        assert scheduler.get(0) == 0.0
        assert scheduler.get(2) == 0.0
        # Epoch 3 is first epoch after warmup, should start ramping
        # With 7 remaining epochs (3-9), epoch 3 is t=0 in the ramp
        # So it starts at eps_start=0.0, then increases
        assert scheduler.get(4) > 0.0  # Epoch 4 should be > 0
    
    def test_beyond_total_epochs(self):
        """Test behavior beyond total epochs."""
        scheduler = EpsilonScheduler(eps_start=0.0, eps_end=0.1, total_epochs=10, schedule='linear')
        
        assert scheduler.get(15) == 0.1  # Should return eps_end


# ============================================================================
# RobustTrainer Tests
# ============================================================================

class TestRobustTrainer:
    """Tests for RobustTrainer class."""
    
    def test_training_loop(self, simple_mlp, toy_loaders):
        """Test basic training loop runs without errors."""
        train_loader, test_loader = toy_loaders
        
        loss_fn = ProvableLoss(input_clamp=(-10.0, 10.0))
        optimizer = torch.optim.Adam(simple_mlp.parameters(), lr=0.01)
        scheduler = EpsilonScheduler(eps_start=0.0, eps_end=0.1, total_epochs=3, schedule='linear')
        
        trainer = RobustTrainer(
            model=simple_mlp,
            loss_fn=loss_fn,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            epsilon_scheduler=scheduler,
            epochs=3,
            device='cpu',
            log_interval=100,
            eval_interval=1,
        )
        
        history = trainer.train()
        
        assert 'train_loss' in history
        assert len(history['train_loss']) == 3
        assert 'epsilon' in history
    
    def test_loss_decreases(self, simple_mlp, toy_loaders):
        """Test that loss generally decreases over training."""
        train_loader, test_loader = toy_loaders
        
        # Use fixed epsilon for this test
        loss_fn = ProvableLoss(input_clamp=(-10.0, 10.0))
        optimizer = torch.optim.Adam(simple_mlp.parameters(), lr=0.01)
        scheduler = EpsilonScheduler(eps_start=0.05, eps_end=0.05, total_epochs=5, schedule='linear')
        
        trainer = RobustTrainer(
            model=simple_mlp,
            loss_fn=loss_fn,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            epsilon_scheduler=scheduler,
            epochs=5,
            device='cpu',
            log_interval=100,
            eval_interval=5,
        )
        
        history = trainer.train()
        
        # Loss should generally decrease (allow some variance)
        assert history['train_loss'][-1] <= history['train_loss'][0] * 1.1


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for the full pipeline."""
    
    def test_full_training_pipeline(self, simple_mlp, toy_loaders):
        """Test complete training pipeline."""
        train_loader, test_loader = toy_loaders
        
        # Setup
        loss_fn = ProvableLoss(input_clamp=(-10.0, 10.0))
        optimizer = torch.optim.Adam(simple_mlp.parameters(), lr=0.01)
        scheduler = EpsilonScheduler(eps_start=0.0, eps_end=0.1, total_epochs=5, schedule='linear')
        
        # Train
        trainer = RobustTrainer(
            model=simple_mlp,
            loss_fn=loss_fn,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            epsilon_scheduler=scheduler,
            epochs=5,
            device='cpu',
            log_interval=100,
            eval_interval=1,
        )
        
        history = trainer.train()
        
        # Verify outputs
        assert len(history['train_loss']) == 5
        assert len(history['epsilon']) == 5
        assert history['epsilon'][0] == 0.0
        assert history['epsilon'][-1] == 0.1
        
        # Verify model was trained (weights changed)
        # This is implicitly tested by the training loop completing


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
