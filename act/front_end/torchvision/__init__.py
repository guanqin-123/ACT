#!/usr/bin/env python3
"""
TorchVision Dataset-Model Integration for ACT.

Copyright (C) 2025 SVF-tools/ACT
License: AGPLv3+
"""

# Core data and API
from act.front_end.torchvision.data_model_mapping import (
    DATASET_MODEL_MAPPING,
    get_dataset_info,
    list_datasets_by_category,
    get_all_categories,
    search_datasets,
    get_preprocessing_transforms,
    create_preprocessing_pipeline,
    validate_dataset_model_compatibility,
    find_dataset_name,
    find_model_name,
)

# Dataset-Model loader
from act.front_end.torchvision.data_model_loader import (
    download_dataset_model_pair,
    load_dataset_model_pair,
    list_downloaded_pairs,
    model_inference_with_dataset,
)

# Model definitions
from act.front_end.torchvision.model_definitions import (
    SimpleCNN,
    LeNet5,
)

# CLI is NOT exported - users must explicitly import
# from act.front_end.torchvision.cli import main

__all__ = [
    'DATASET_MODEL_MAPPING',
    'get_dataset_info',
    'list_datasets_by_category',
    'get_all_categories',
    'search_datasets',
    'get_preprocessing_transforms',
    'create_preprocessing_pipeline',
    'validate_dataset_model_compatibility',
    'find_dataset_name',
    'find_model_name',
    'download_dataset_model_pair',
    'load_dataset_model_pair',
    'list_downloaded_pairs',
    'model_inference_with_dataset',
    'SimpleCNN',
    'LeNet5',
]
