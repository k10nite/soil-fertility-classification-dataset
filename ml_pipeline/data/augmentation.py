"""
Data augmentation pipelines for soil image classification.

This module provides three augmentation strategies:
1. Conservative: Recommended for soil images with realistic transformations
2. Aggressive: Maximum variety for scenarios with limited training data
3. Validation/Test: Minimal augmentation (resize + normalize only)
"""

from typing import Tuple
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ImageNet normalization values (commonly used for transfer learning)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_conservative_augmentation(
    image_size: Tuple[int, int] = (224, 224),
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD,
) -> A.Compose:
    """
    Create a conservative augmentation pipeline for soil images.

    Recommended for production use with soil classification. Applies realistic
    transformations that preserve soil texture and color characteristics while
    providing sufficient variety for model generalization.

    Transformations:
    - Horizontal and vertical flips
    - Rotation (±180°) - soil has no fixed orientation
    - Random resized crop (80-100% scale)
    - Color jitter (brightness, contrast, saturation, hue)
    - Gaussian blur (simulates focus variations)
    - Gaussian noise (simulates camera sensor noise)

    Args:
        image_size: Target image dimensions (height, width)
        mean: Normalization mean for RGB channels
        std: Normalization std for RGB channels

    Returns:
        Albumentations composition pipeline
    """
    return A.Compose([
        # Geometric transformations
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=180, p=0.7, border_mode=0),  # BORDER_CONSTANT

        # Crop and resize
        A.RandomResizedCrop(
            height=image_size[0],
            width=image_size[1],
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1),  # Keep aspect ratio similar
            p=1.0
        ),

        # Color augmentations - conservative for soil color preservation
        A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.15,
            hue=0.05,
            p=0.6
        ),

        # Blur and noise - simulates real-world image quality variations
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.GaussNoise(var_limit=(10.0, 30.0), p=0.3),

        # Normalization and tensor conversion
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def get_aggressive_augmentation(
    image_size: Tuple[int, int] = (224, 224),
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD,
) -> A.Compose:
    """
    Create an aggressive augmentation pipeline for maximum variety.

    Use when training data is severely limited or when model needs to be
    robust to extreme variations. Includes all conservative transforms plus
    additional photometric and motion effects.

    Additional transformations over conservative:
    - Random shadows
    - Motion blur (simulates camera shake)
    - Stronger crops (75-100% scale)
    - More aggressive color jitter
    - Coarse dropout (random rectangular cutouts)

    Args:
        image_size: Target image dimensions (height, width)
        mean: Normalization mean for RGB channels
        std: Normalization std for RGB channels

    Returns:
        Albumentations composition pipeline
    """
    return A.Compose([
        # Geometric transformations
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=180, p=0.8, border_mode=0),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.2,
            rotate_limit=180,
            border_mode=0,
            p=0.5
        ),

        # Crop and resize - more aggressive
        A.RandomResizedCrop(
            height=image_size[0],
            width=image_size[1],
            scale=(0.75, 1.0),  # Stronger crops
            ratio=(0.85, 1.15),
            p=1.0
        ),

        # Color augmentations - stronger
        A.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.25,
            hue=0.08,
            p=0.7
        ),

        # Additional photometric effects
        A.RandomShadow(
            shadow_roi=(0, 0, 1, 1),
            num_shadows_limit=(1, 2),
            shadow_dimension=5,
            p=0.3
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.25,
            contrast_limit=0.25,
            p=0.5
        ),

        # Blur and motion effects
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=(3, 7), p=1.0),
        ], p=0.4),

        # Noise
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),

        # Random cutout (simulates occlusions)
        A.CoarseDropout(
            max_holes=3,
            max_height=32,
            max_width=32,
            min_holes=1,
            min_height=8,
            min_width=8,
            fill_value=0,
            p=0.3
        ),

        # Normalization and tensor conversion
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def get_validation_transforms(
    image_size: Tuple[int, int] = (224, 224),
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD,
) -> A.Compose:
    """
    Create minimal augmentation pipeline for validation and test sets.

    Only applies essential preprocessing (resize and normalize) without
    any data augmentation. Ensures consistent, reproducible evaluation.

    Transformations:
    - Resize to target dimensions
    - Normalize using provided mean and std
    - Convert to PyTorch tensor

    Args:
        image_size: Target image dimensions (height, width)
        mean: Normalization mean for RGB channels
        std: Normalization std for RGB channels

    Returns:
        Albumentations composition pipeline
    """
    return A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def get_test_transforms(
    image_size: Tuple[int, int] = (224, 224),
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD,
) -> A.Compose:
    """
    Create transforms for test set (alias for validation transforms).

    Identical to validation transforms - no augmentation, only resize
    and normalize. Provided as separate function for code clarity.

    Args:
        image_size: Target image dimensions (height, width)
        mean: Normalization mean for RGB channels
        std: Normalization std for RGB channels

    Returns:
        Albumentations composition pipeline
    """
    return get_validation_transforms(image_size=image_size, mean=mean, std=std)


def get_all_transforms(
    image_size: Tuple[int, int] = (224, 224),
    augmentation_strategy: str = "conservative",
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD,
) -> dict:
    """
    Get all three transform pipelines (train, validation, test) at once.

    Args:
        image_size: Target image dimensions (height, width)
        augmentation_strategy: "conservative" or "aggressive" for training
        mean: Normalization mean for RGB channels
        std: Normalization std for RGB channels

    Returns:
        Dictionary with keys: 'train', 'val', 'test'

    Raises:
        ValueError: If augmentation_strategy is not 'conservative' or 'aggressive'

    Example:
        >>> transforms = get_all_transforms(image_size=(256, 256))
        >>> train_transform = transforms['train']
        >>> val_transform = transforms['val']
    """
    if augmentation_strategy not in ["conservative", "aggressive"]:
        raise ValueError(
            f"augmentation_strategy must be 'conservative' or 'aggressive', "
            f"got '{augmentation_strategy}'"
        )

    train_transform = (
        get_conservative_augmentation(image_size, mean, std)
        if augmentation_strategy == "conservative"
        else get_aggressive_augmentation(image_size, mean, std)
    )

    return {
        "train": train_transform,
        "val": get_validation_transforms(image_size, mean, std),
        "test": get_test_transforms(image_size, mean, std),
    }
