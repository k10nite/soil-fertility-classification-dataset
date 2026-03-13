"""
Example usage of the soil fertility classification data pipeline.

This script demonstrates how to:
1. Load and augment soil images
2. Create train/val/test datasets
3. Create DataLoaders
4. Iterate through batches
"""

import logging
from pathlib import Path

import torch

# Import from our data pipeline
from data import (
    get_all_transforms,
    create_train_val_test_datasets,
    create_train_loader,
    create_val_loader,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main example function."""

    # Configuration
    CSV_PATH = '../organized_images/combined_field_data.csv'
    IMG_DIR = '../organized_images'
    IMAGE_SIZE = (224, 224)  # ResNet18 input size
    BATCH_SIZE = 16

    logger.info("=== Soil Fertility Classification - Data Pipeline Example ===\n")

    # Step 1: Get transforms
    logger.info("Step 1: Creating data transforms...")
    transforms = get_all_transforms(
        image_size=IMAGE_SIZE,
        augmentation_strategy='conservative'  # or 'aggressive'
    )
    logger.info(f"✓ Created transforms: {list(transforms.keys())}\n")

    # Step 2: Create datasets
    logger.info("Step 2: Creating train/val/test datasets...")
    try:
        train_dataset, val_dataset, test_dataset = create_train_val_test_datasets(
            csv_path=CSV_PATH,
            img_dir=IMG_DIR,
            train_transform=transforms['train'],
            val_transform=transforms['val'],
            train_size=0.7,
            val_size=0.15,
            test_size=0.15,
            random_state=42,
            stratify=True,
        )
        logger.info(f"✓ Train: {len(train_dataset)} samples")
        logger.info(f"✓ Val:   {len(val_dataset)} samples")
        logger.info(f"✓ Test:  {len(test_dataset)} samples\n")
    except FileNotFoundError:
        logger.error(f"CSV file not found at {CSV_PATH}")
        logger.info("Please update CSV_PATH and IMG_DIR in this script")
        return

    # Step 3: Create DataLoaders
    logger.info("Step 3: Creating DataLoaders...")
    train_loader = create_train_loader(
        train_dataset,
        batch_size=BATCH_SIZE,
        use_weighted_sampler=True,  # Handle class imbalance
    )
    val_loader = create_val_loader(
        val_dataset,
        batch_size=BATCH_SIZE,
    )
    logger.info(f"✓ Train batches: {len(train_loader)}")
    logger.info(f"✓ Val batches:   {len(val_loader)}\n")

    # Step 4: Iterate through one batch
    logger.info("Step 4: Inspecting one training batch...")
    images, labels, metadata = next(iter(train_loader))
    logger.info(f"✓ Batch image shape: {images.shape}")  # [B, C, H, W]
    logger.info(f"✓ Batch labels shape: {labels.shape}")  # [B]
    logger.info(f"✓ First sample metadata: {metadata[0]['municipality']}\n")

    # Step 5: Get class weights for loss function
    logger.info("Step 5: Computing class weights...")
    class_weights = train_dataset.get_class_weights()
    if class_weights is not None:
        logger.info(f"✓ Class weights: {class_weights}")
        logger.info("  Use with: nn.CrossEntropyLoss(weight=class_weights)\n")

    # Step 6: Example training loop structure
    logger.info("Step 6: Example training loop structure...")
    logger.info("""
    # Pseudo-code for training:
    model = YourModel()
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(num_epochs):
        # Training
        model.train()
        for images, labels, metadata in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            for images, labels, metadata in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                # Compute metrics...
    """)

    logger.info("=== Example Complete ===")
    logger.info("\nNext steps:")
    logger.info("1. Obtain labeled soil fertility data (NPK values, pH)")
    logger.info("2. Update CSV with 'soil_class' column (0=Low, 1=Medium, 2=High)")
    logger.info("3. Implement model in ../models/")
    logger.info("4. Create training script")


if __name__ == '__main__':
    main()
