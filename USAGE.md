# Usage Guide

Complete guide for using the soil fertility classification dataset.

## Table of Contents
1. [Installation](#installation)
2. [Data Augmentation](#data-augmentation)
3. [Creating Datasets](#creating-datasets)
4. [Creating DataLoaders](#creating-dataloaders)
5. [Training Example](#training-example)
6. [Common Issues](#common-issues)

## Installation

```bash
# Clone and setup
git clone git@github.com:k10nite/soil-fertility-classification-dataset.git
cd soil-fertility-classification-dataset
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Data Augmentation

### Conservative Strategy (Recommended)

```python
from ml_pipeline.data import get_conservative_augmentation

transform = get_conservative_augmentation(
    image_size=(224, 224),  # Target size
)
```

### Aggressive Strategy

```python
from ml_pipeline.data import get_aggressive_augmentation

transform = get_aggressive_augmentation(
    image_size=(224, 224),
)
```

### Get All Transforms At Once

```python
from ml_pipeline.data import get_all_transforms

transforms = get_all_transforms(
    image_size=(224, 224),
    augmentation_strategy='conservative',  # or 'aggressive'
)

train_transform = transforms['train']
val_transform = transforms['val']
test_transform = transforms['test']
```

## Creating Datasets

### Method 1: Single Dataset

```python
from ml_pipeline.data import SoilImageDataset, get_validation_transforms

dataset = SoilImageDataset(
    csv_path='organized_images/combined_field_data.csv',
    img_dir='organized_images',
    transform=get_validation_transforms(),
    validate_images=True,  # Check all images exist
)

print(f"Dataset size: {len(dataset)}")
print(f"Number of classes: {dataset.num_classes}")
print(f"Class distribution: {dataset.class_counts}")
```

### Method 2: Train/Val/Test Split

```python
from ml_pipeline.data import create_train_val_test_datasets, get_all_transforms

transforms = get_all_transforms()

train_ds, val_ds, test_ds = create_train_val_test_datasets(
    csv_path='organized_images/combined_field_data.csv',
    img_dir='organized_images',
    train_transform=transforms['train'],
    val_transform=transforms['val'],
    train_size=0.7,
    val_size=0.15,
    test_size=0.15,
    random_state=42,
    stratify=True,  # Maintain class distribution
)
```

## Creating DataLoaders

### Training DataLoader

```python
from ml_pipeline.data import create_train_loader

train_loader = create_train_loader(
    dataset=train_ds,
    batch_size=32,
    shuffle=True,
    use_weighted_sampler=True,  # Handle class imbalance
    pin_memory=True,  # Faster GPU transfer
)
```

### Validation/Test DataLoader

```python
from ml_pipeline.data import create_val_loader

val_loader = create_val_loader(
    dataset=val_ds,
    batch_size=32,
)

test_loader = create_val_loader(
    dataset=test_ds,
    batch_size=32,
)
```

## Training Example

### Complete Training Script

```python
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from ml_pipeline.data import (
    get_all_transforms,
    create_train_val_test_datasets,
    create_train_loader,
    create_val_loader,
)

# Configuration
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Create datasets
transforms = get_all_transforms(image_size=(224, 224))
train_ds, val_ds, test_ds = create_train_val_test_datasets(
    csv_path='organized_images/combined_field_data.csv',
    img_dir='organized_images',
    train_transform=transforms['train'],
    val_transform=transforms['val'],
)

# 2. Create data loaders
train_loader = create_train_loader(train_ds, BATCH_SIZE, use_weighted_sampler=True)
val_loader = create_val_loader(val_ds, BATCH_SIZE)

# 3. Initialize model (example with ResNet18)
from torchvision.models import resnet18
model = resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 3)  # 3 classes
model = model.to(DEVICE)

# 4. Loss function with class weights
class_weights = train_ds.get_class_weights().to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

# 5. Training loop
for epoch in range(EPOCHS):
    # Training
    model.train()
    train_loss = 0.0
    train_correct = 0

    for images, labels, metadata in tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}'):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Metrics
        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        train_correct += (predicted == labels).sum().item()

    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0

    with torch.no_grad():
        for images, labels, metadata in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()

    # Print metrics
    print(f'Train Loss: {train_loss/len(train_loader):.4f}, '
          f'Train Acc: {100*train_correct/len(train_ds):.2f}%')
    print(f'Val Loss: {val_loss/len(val_loader):.4f}, '
          f'Val Acc: {100*val_correct/len(val_ds):.2f}%')

# 6. Save model
torch.save(model.state_dict(), 'soil_model.pth')
```

## Common Issues

### Issue: "Missing image files"

**Solution**: Ensure `img_dir` points to the correct directory containing images.

```python
# Check paths
import os
print(os.path.exists('organized_images'))
print(os.path.exists('organized_images/combined_field_data.csv'))
```

### Issue: "No 'soil_class' column found"

**Solution**: Dataset doesn't have labels yet. This is expected if you haven't added ground truth labels.

```python
# Dataset will work but labels will be -1
dataset = SoilImageDataset(csv_path, img_dir, transform)
image, label, metadata = dataset[0]
print(label)  # Will be -1 if no labels
```

### Issue: "CUDA out of memory"

**Solution**: Reduce batch size or image size.

```python
# Reduce batch size
train_loader = create_train_loader(train_ds, batch_size=16)  # Instead of 32

# Or reduce image size
transforms = get_all_transforms(image_size=(112, 112))  # Instead of 224
```

### Issue: "DataLoader multiprocessing error on Windows"

**Solution**: Set `num_workers=0` or use the auto-detection (already handled).

```python
train_loader = create_train_loader(train_ds, num_workers=0)
```

## Next Steps

1. Add ground truth labels to your CSV (column: `soil_class` with values 0, 1, 2)
2. Implement your model in `models/`
3. Create a training script
4. Run experiments and track metrics
5. Evaluate on test set

For more examples, see `ml_pipeline/example_usage.py`.
