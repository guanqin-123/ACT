#===- act/pipeline/finetune/trainer.py - Robust Trainer -----------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Unified training loop for robust finetuning.
#   Supports provable, adversarial, and mixed training strategies.
#
#===---------------------------------------------------------------------===#

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Callable
from collections import defaultdict
import time
import logging

from .actloss import RobustLoss
from .scheduler import EpsilonScheduler

logger = logging.getLogger(__name__)


class RobustTrainer:
    """
    Unified training loop for robust finetuning.
    
    Features:
    - Multiple loss strategies (provable, adversarial, mixed)
    - Epsilon scheduling (curriculum learning)
    - Logging and checkpointing
    - Evaluation on clean and robust metrics
    """
    
    def __init__(
        self,
        model: nn.Module,
        loss_fn: RobustLoss,
        train_loader: DataLoader,
        test_loader: DataLoader,
        optimizer: optim.Optimizer,
        epsilon_scheduler: EpsilonScheduler,
        epochs: int,
        device: str = 'cuda',
        checkpoint_dir: Optional[str] = None,
        log_interval: int = 10,
        eval_interval: int = 1,
        callbacks: Optional[List[Callable]] = None,
    ):
        """
        Args:
            model: Neural network to train
            loss_fn: RobustLoss instance (provable, adversarial, mixed)
            train_loader: Training data loader
            test_loader: Test/validation data loader
            optimizer: PyTorch optimizer
            epsilon_scheduler: Scheduler for epsilon during training
            epochs: Total training epochs
            device: Device to train on ('cuda' or 'cpu')
            checkpoint_dir: Directory to save checkpoints (None = no saving)
            log_interval: Log every N batches
            eval_interval: Evaluate every N epochs
            callbacks: List of callback functions called after each epoch
        """
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.epsilon_scheduler = epsilon_scheduler
        self.epochs = epochs
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.callbacks = callbacks or []
        
        # Best model tracking
        self.best_metric = 0.0
        self.best_epoch = 0
    
    def train(self) -> Dict[str, List[float]]:
        """
        Run full training loop.
        
        Returns:
            history: Dict mapping metric names to list of values per epoch
        """
        history = defaultdict(list)
        
        logger.info(f"Starting training: {self.epochs} epochs, loss={self.loss_fn.name}")
        logger.info(f"Epsilon schedule: {self.epsilon_scheduler}")
        
        for epoch in range(self.epochs):
            epsilon = self.epsilon_scheduler.get(epoch)
            
            # Training epoch
            train_metrics = self._train_epoch(epoch, epsilon)
            
            # Evaluation
            if (epoch + 1) % self.eval_interval == 0 or epoch == self.epochs - 1:
                eval_metrics = self._evaluate(epsilon)
            else:
                eval_metrics = {}
            
            # Merge metrics
            all_metrics = {**train_metrics, **eval_metrics, 'epsilon': epsilon}
            
            # Update history
            for k, v in all_metrics.items():
                history[k].append(v)
            
            # Logging
            self._log_epoch(epoch, all_metrics)
            
            # Checkpointing
            self._checkpoint(epoch, all_metrics)
            
            # Callbacks
            for callback in self.callbacks:
                callback(epoch, self.model, all_metrics)
        
        logger.info(f"Training complete. Best epoch: {self.best_epoch}, "
                   f"best metric: {self.best_metric:.4f}")
        
        return dict(history)
    
    def _train_epoch(self, epoch: int, epsilon: float) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_samples = 0
        metrics_sum = defaultdict(float)
        
        start_time = time.time()
        
        for batch_idx, (X, y) in enumerate(self.train_loader):
            X, y = X.to(self.device), y.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Compute robust loss
            loss, batch_metrics = self.loss_fn(self.model, X, y, epsilon)
            
            # Backward and optimize
            loss.backward()
            self.optimizer.step()
            
            # Accumulate metrics
            batch_size = X.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            for k, v in batch_metrics.items():
                metrics_sum[k] += v * batch_size
            
            # Logging
            if (batch_idx + 1) % self.log_interval == 0:
                avg_loss = total_loss / total_samples
                logger.debug(f"Epoch {epoch}, Batch {batch_idx+1}/{len(self.train_loader)}, "
                           f"Loss: {avg_loss:.4f}")
        
        elapsed = time.time() - start_time
        
        # Average metrics
        metrics = {f'train_{k}': v / total_samples for k, v in metrics_sum.items()}
        metrics['train_loss'] = total_loss / total_samples
        metrics['train_time'] = elapsed
        
        return metrics
    
    def _evaluate(self, epsilon: float) -> Dict[str, float]:
        """Evaluate on test set."""
        self.model.eval()
        
        total_samples = 0
        metrics_sum = defaultdict(float)
        
        with torch.no_grad():
            for X, y in self.test_loader:
                X, y = X.to(self.device), y.to(self.device)
                
                # Use loss function's eval_metrics or __call__
                batch_metrics = self.loss_fn.eval_metrics(self.model, X, y, epsilon)
                
                batch_size = X.size(0)
                total_samples += batch_size
                
                for k, v in batch_metrics.items():
                    metrics_sum[k] += v * batch_size
        
        # Average metrics
        metrics = {f'test_{k}': v / total_samples for k, v in metrics_sum.items()}
        
        return metrics
    
    def _log_epoch(self, epoch: int, metrics: Dict[str, float]):
        """Log epoch metrics."""
        eps = metrics.get('epsilon', 0)
        train_loss = metrics.get('train_loss', 0)
        
        msg = f"Epoch {epoch+1}/{self.epochs} | eps={eps:.4f} | loss={train_loss:.4f}"
        
        if 'test_clean_acc' in metrics:
            msg += f" | clean={metrics['test_clean_acc']:.4f}"
        if 'test_certified_acc' in metrics:
            msg += f" | certified={metrics['test_certified_acc']:.4f}"
        if 'test_robust_acc' in metrics:
            msg += f" | robust={metrics['test_robust_acc']:.4f}"
        
        logger.info(msg)
    
    def _checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save checkpoint if improved."""
        if self.checkpoint_dir is None:
            return
        
        # Determine metric to track (certified > robust > clean)
        if 'test_certified_acc' in metrics:
            current_metric = metrics['test_certified_acc']
        elif 'test_robust_acc' in metrics:
            current_metric = metrics['test_robust_acc']
        else:
            current_metric = metrics.get('test_clean_acc', 0)
        
        if current_metric > self.best_metric:
            self.best_metric = current_metric
            self.best_epoch = epoch
            
            import os
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            
            path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'metrics': metrics,
            }, path)
            
            logger.info(f"Saved best model (epoch {epoch}, metric={current_metric:.4f})")
    
    def load_checkpoint(self, path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Loaded checkpoint from {path}")
        return checkpoint.get('metrics', {})
