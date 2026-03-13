"""
Data pipeline for soil fertility classification.
"""

from .augmentation import (
    get_conservative_augmentation,
    get_aggressive_augmentation,
    get_validation_transforms,
    get_test_transforms,
    get_all_transforms,
    IMAGENET_MEAN,
    IMAGENET_STD,
)

from .dataset import (
    SoilImageDataset,
    create_train_val_test_datasets,
)

from .dataloader import (
    create_train_loader,
    create_val_loader,
    create_test_loader,
    get_optimal_num_workers,
)

__all__ = [
    # Augmentation
    'get_conservative_augmentation',
    'get_aggressive_augmentation',
    'get_validation_transforms',
    'get_test_transforms',
    'get_all_transforms',
    'IMAGENET_MEAN',
    'IMAGENET_STD',
    # Dataset
    'SoilImageDataset',
    'create_train_val_test_datasets',
    # DataLoader
    'create_train_loader',
    'create_val_loader',
    'create_test_loader',
    'get_optimal_num_workers',
]
