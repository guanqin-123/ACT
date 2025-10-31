#!/usr/bin/env python3
"""
Custom Model Architectures for TorchVision Datasets.

Provides custom neural network architectures (SimpleCNN, LeNet5) optimized
for smaller datasets like MNIST, along with code generation utilities.

Copyright (C) 2025 SVF-tools/ACT
License: AGPLv3+
"""

import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    Simple CNN that adapts to standard preprocessing (3-channel RGB, 224×224).
    
    Compatible with the same preprocessing pipeline as standard TorchVision models.
    """
    
    def __init__(self, num_classes: int = 10):
        super(SimpleCNN, self).__init__()
        # Accept 3 channels (RGB) after standard preprocessing
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # After 2 pooling layers on 224x224: 224 -> 112 -> 56
        self.fc1 = nn.Linear(64 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class LeNet5(nn.Module):
    """
    LeNet-5 architecture adapted for standard preprocessing (3-channel RGB, 224×224).
    
    Compatible with the same preprocessing pipeline as standard TorchVision models.
    """
    
    def __init__(self, num_classes: int = 10):
        super(LeNet5, self).__init__()
        # Accept 3 channels (RGB) after standard preprocessing
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        # After conv1 (224->220) + pool (220->110) + conv2 (110->106) + pool (106->53)
        self.fc1 = nn.Linear(16 * 53 * 53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.pool = nn.AvgPool2d(2, 2)
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        x = self.pool(self.tanh(self.conv1(x)))
        x = self.pool(self.tanh(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


def get_model(model_name: str, num_classes: int = 10):
    """
    Get a custom model instance by name.
    
    Args:
        model_name: Name of the custom model (simple_cnn or lenet5, case-insensitive)
        num_classes: Number of output classes
        
    Returns:
        Model instance
        
    Raises:
        ValueError: If model_name is not recognized
    """
    model_name_lower = model_name.lower()
    
    if model_name_lower == "simple_cnn" or model_name_lower == "simplecnn":
        return SimpleCNN(num_classes=num_classes)
    elif model_name_lower == "lenet5":
        return LeNet5(num_classes=num_classes)
    else:
        raise ValueError(
            f"Unknown custom model: {model_name}. "
            f"Available models: simple_cnn, lenet5"
        )


def _get_custom_model_definition(model_name: str, num_classes: int) -> str:
    """
    Get the architecture definition for custom models as Python code string.
    
    Args:
        model_name: Name of the custom model (simple_cnn or lenet5)
        num_classes: Number of output classes
        
    Returns:
        Python code string defining the model architecture
    """
    if model_name == "simple_cnn":
        return f"""
import torch.nn as nn

class SimpleCNN(nn.Module):
    \"\"\"Simple CNN that adapts to standard preprocessing (3-channel RGB, 224×224).\"\"\"
    def __init__(self, num_classes={num_classes}):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 56 * 56, 128)  # After 2 pooling layers on 224x224
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Create model
model = SimpleCNN(num_classes={num_classes})
"""
    elif model_name == "lenet5":
        return f"""
import torch.nn as nn

class LeNet5(nn.Module):
    \"\"\"LeNet-5 adapted for standard preprocessing (3-channel RGB, 224×224).\"\"\"
    def __init__(self, num_classes={num_classes}):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 53 * 53, 120)  # After convs and pooling on 224x224
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.pool = nn.AvgPool2d(2, 2)
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        x = self.pool(self.tanh(self.conv1(x)))
        x = self.pool(self.tanh(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

# Create model
model = LeNet5(num_classes={num_classes})
"""
    else:
        raise ValueError(f"Unknown custom model: {model_name}")
