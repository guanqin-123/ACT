#===- act/pipeline/finetune/__init__.py - Robust Finetuning Module ------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Robust finetuning module for neural network training with certified
#   and empirical robustness guarantees.
#
#   Supports:
#   - Provable training (dual bounds, certified robustness)
#   - Adversarial training (PGD, empirical robustness)
#   - Mixed training (combine both strategies)
#
#===---------------------------------------------------------------------===#

# Base classes
from .actloss import RobustLoss

# Loss implementations
from .provable import ProvableLoss
from .adversarial import AdversarialLoss, evaluate_pgd, evaluate_fgsm

# Training utilities
from .scheduler import EpsilonScheduler
from .trainer import RobustTrainer

__all__ = [
    # Base
    'RobustLoss',
    # Loss implementations
    'ProvableLoss',
    'AdversarialLoss', 
    # Evaluation
    'evaluate_pgd',
    'evaluate_fgsm',
    # Training
    'EpsilonScheduler',
    'RobustTrainer',
]
