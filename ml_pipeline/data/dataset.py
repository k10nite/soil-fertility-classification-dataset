"""
PyTorch Dataset for soil fertility classification.

Loads soil images and metadata from CSV file.
"""

import os
import logging
from typing import Optional, Tuple, Dict, List
from pathlib import Path

import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SoilImageDataset(Dataset):
    """
    PyTorch Dataset for soil fertility classification.

    Loads images and metadata from CSV file with columns:
    - image_filename: Path to image file
    - municipality: Location (Atok, La Trinidad, etc.)
    - crops: Crop types grown
    - latitude: GPS latitude
    - longitude: GPS longitude
    - soil_class: Fertility class label (0, 1, 2 for Low, Medium, High)

    Args:
        csv_path: Path to CSV file with image metadata
        img_dir: Root directory containing images
        transform: Albumentations transform to apply to images
        validate_images: If True, check that all images exist on initialization

    Example:
        >>> from ml_pipeline.data.augmentation import get_conservative_augmentation
        >>> transform = get_conservative_augmentation()
        >>> dataset = SoilImageDataset(
        ...     csv_path='data/soil_data.csv',
        ...     img_dir='data/images',
        ...     transform=transform
        ... )
        >>> image, label, metadata = dataset[0]
    """

    def __init__(
        self,
        csv_path: str,
        img_dir: str,
        transform: Optional[object] = None,
        validate_images: bool = True,
    ):
        self.csv_path = csv_path
        self.img_dir = Path(img_dir)
        self.transform = transform

        # Load CSV
        logger.info(f"Loading dataset from {csv_path}")
        self.data = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(self.data)} samples")

        # Validate required columns
        required_cols = ['image_filename']
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Validate images exist
        if validate_images:
            self._validate_images()

        # Get class distribution
        if 'soil_class' in self.data.columns:
            self.num_classes = self.data['soil_class'].nunique()
            self.class_counts = self.data['soil_class'].value_counts().to_dict()
            logger.info(f"Class distribution: {self.class_counts}")
        else:
            self.num_classes = None
            logger.warning("No 'soil_class' column found - labels not available")

    def _validate_images(self):
        """Check that all image files exist."""
        logger.info("Validating image files...")
        missing = []
        for idx, row in self.data.iterrows():
            img_path = self.img_dir / row['image_filename']
            if not img_path.exists():
                missing.append(str(img_path))

        if missing:
            logger.warning(f"Missing {len(missing)} image files")
            logger.warning(f"First 5 missing: {missing[:5]}")
            # Remove missing images from dataset
            valid_mask = self.data['image_filename'].apply(
                lambda x: (self.img_dir / x).exists()
            )
            self.data = self.data[valid_mask].reset_index(drop=True)
            logger.info(f"Dataset size after removing missing images: {len(self.data)}")
        else:
            logger.info("All image files validated successfully")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict]:
        """
        Get a single sample.

        Args:
            idx: Index of sample to retrieve

        Returns:
            image: Transformed image tensor (C, H, W)
            label: Soil fertility class label (or -1 if not available)
            metadata: Dictionary with additional information
        """
        row = self.data.iloc[idx]

        # Load image
        img_path = self.img_dir / row['image_filename']
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)

        # Apply transform
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        else:
            # Convert to tensor if no transform
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        # Get label
        label = int(row['soil_class']) if 'soil_class' in row and pd.notna(row['soil_class']) else -1

        # Get metadata
        metadata = {
            'image_filename': row['image_filename'],
            'municipality': row.get('municipality', 'unknown'),
            'crops': row.get('crops', 'unknown'),
            'latitude': float(row['latitude']) if 'latitude' in row and pd.notna(row['latitude']) else None,
            'longitude': float(row['longitude']) if 'longitude' in row and pd.notna(row['longitude']) else None,
        }

        return image, label, metadata

    def get_class_weights(self) -> Optional[torch.Tensor]:
        """
        Calculate class weights for handling imbalanced datasets.

        Returns:
            Tensor of class weights (inverse frequency), or None if labels not available
        """
        if 'soil_class' not in self.data.columns:
            return None

        class_counts = self.data['soil_class'].value_counts().sort_index()
        total = len(self.data)
        weights = total / (len(class_counts) * class_counts.values)
        return torch.FloatTensor(weights)


def create_train_val_test_datasets(
    csv_path: str,
    img_dir: str,
    train_transform: object,
    val_transform: object,
    test_transform: Optional[object] = None,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[SoilImageDataset, SoilImageDataset, SoilImageDataset]:
    """
    Create train, validation, and test datasets from a single CSV file.

    Performs stratified splitting to maintain class distribution across splits.

    Args:
        csv_path: Path to CSV file
        img_dir: Root directory containing images
        train_transform: Transform for training set
        val_transform: Transform for validation set
        test_transform: Transform for test set (uses val_transform if None)
        train_size: Proportion for training (default 0.7)
        val_size: Proportion for validation (default 0.15)
        test_size: Proportion for test (default 0.15)
        random_state: Random seed for reproducibility
        stratify: Whether to stratify split by class labels

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)

    Example:
        >>> from ml_pipeline.data.augmentation import get_all_transforms
        >>> transforms = get_all_transforms()
        >>> train_ds, val_ds, test_ds = create_train_val_test_datasets(
        ...     csv_path='data/soil_data.csv',
        ...     img_dir='data/images',
        ...     train_transform=transforms['train'],
        ...     val_transform=transforms['val'],
        ... )
    """
    # Validate split sizes
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError(f"Split sizes must sum to 1.0, got {train_size + val_size + test_size}")

    # Load data
    data = pd.read_csv(csv_path)
    logger.info(f"Total samples: {len(data)}")

    # Determine stratification
    stratify_col = None
    if stratify and 'soil_class' in data.columns:
        stratify_col = data['soil_class']

    # First split: train vs (val + test)
    train_data, temp_data = train_test_split(
        data,
        train_size=train_size,
        random_state=random_state,
        stratify=stratify_col
    )

    # Second split: val vs test
    val_ratio = val_size / (val_size + test_size)
    stratify_temp = temp_data['soil_class'] if stratify and 'soil_class' in temp_data.columns else None

    val_data, test_data = train_test_split(
        temp_data,
        train_size=val_ratio,
        random_state=random_state,
        stratify=stratify_temp
    )

    logger.info(f"Train samples: {len(train_data)}")
    logger.info(f"Val samples: {len(val_data)}")
    logger.info(f"Test samples: {len(test_data)}")

    # Save splits to temporary CSV files
    temp_dir = Path(csv_path).parent / 'splits'
    temp_dir.mkdir(exist_ok=True)

    train_csv = temp_dir / 'train.csv'
    val_csv = temp_dir / 'val.csv'
    test_csv = temp_dir / 'test.csv'

    train_data.to_csv(train_csv, index=False)
    val_data.to_csv(val_csv, index=False)
    test_data.to_csv(test_csv, index=False)

    # Create datasets
    if test_transform is None:
        test_transform = val_transform

    train_dataset = SoilImageDataset(train_csv, img_dir, train_transform)
    val_dataset = SoilImageDataset(val_csv, img_dir, val_transform)
    test_dataset = SoilImageDataset(test_csv, img_dir, test_transform)

    return train_dataset, val_dataset, test_dataset
