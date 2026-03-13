"""
DataLoader utilities for soil fertility classification.
"""

import logging
import platform
from typing import Optional

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .dataset import SoilImageDataset

logger = logging.getLogger(__name__)


def get_optimal_num_workers() -> int:
    """
    Get optimal number of workers for DataLoader based on platform.

    Returns 0 for Windows (avoids multiprocessing issues),
    4 for Linux/Mac.
    """
    if platform.system() == 'Windows':
        return 0
    return 4


def create_train_loader(
    dataset: SoilImageDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: Optional[int] = None,
    use_weighted_sampler: bool = False,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Create DataLoader for training set.

    Args:
        dataset: SoilImageDataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle data (ignored if use_weighted_sampler=True)
        num_workers: Number of worker processes (auto-detected if None)
        use_weighted_sampler: Use weighted sampling for class imbalance
        pin_memory: Pin memory for faster GPU transfer

    Returns:
        DataLoader instance
    """
    if num_workers is None:
        num_workers = get_optimal_num_workers()

    sampler = None
    if use_weighted_sampler:
        class_weights = dataset.get_class_weights()
        if class_weights is not None:
            sample_weights = [class_weights[label] for _, label, _ in dataset]
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            shuffle = False  # Sampler and shuffle are mutually exclusive
            logger.info("Using weighted random sampler for training")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=True,  # Drop incomplete batch
    )


def create_val_loader(
    dataset: SoilImageDataset,
    batch_size: int = 32,
    num_workers: Optional[int] = None,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Create DataLoader for validation set.

    Args:
        dataset: SoilImageDataset instance
        batch_size: Batch size
        num_workers: Number of worker processes (auto-detected if None)
        pin_memory: Pin memory for faster GPU transfer

    Returns:
        DataLoader instance
    """
    if num_workers is None:
        num_workers = get_optimal_num_workers()

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=False,
    )


def create_test_loader(
    dataset: SoilImageDataset,
    batch_size: int = 32,
    num_workers: Optional[int] = None,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Create DataLoader for test set (alias for validation loader).

    Args:
        dataset: SoilImageDataset instance
        batch_size: Batch size
        num_workers: Number of worker processes (auto-detected if None)
        pin_memory: Pin memory for faster GPU transfer

    Returns:
        DataLoader instance
    """
    return create_val_loader(dataset, batch_size, num_workers, pin_memory)
