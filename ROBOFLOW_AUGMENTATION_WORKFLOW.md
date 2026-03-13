# Roboflow Data Augmentation Workflow

## Using Roboflow for Data Augmentation Only

This guide provides a streamlined workflow for using Roboflow specifically for data augmentation of the soil fertility dataset.

---

## Workflow Overview

```
┌────────────────────────────────────────────────────────────────┐
│                     AUGMENTATION WORKFLOW                       │
└────────────────────────────────────────────────────────────────┘

Step 1: Clone Repository
    ↓
Step 2: Upload Images to Roboflow
    ↓
Step 3: Configure Augmentation Settings
    ↓
Step 4: Generate Augmented Dataset
    ↓
Step 5: Download Augmented Images
    ↓
Step 6: Train with Augmented Data

Total Time: ~2-3 hours (mostly upload/download)
```

---

## Prerequisites

### Required
- GitHub repository access: `git@github.com:k10nite/soil-fertility-classification-dataset.git`
- Roboflow account (free): https://roboflow.com
- Internet connection (for upload/download ~400MB)
- Python 3.7+ (for API method)

### Dataset Info
```
Original Dataset:
├─ Images: 909 soil images (1080×1080px)
├─ Locations: Atok (239), La Trinidad (697)
├─ Size: ~370 MB
└─ Format: JPG

After Augmentation:
├─ Images: 909 × augmentation multiplier
│  └─ Example: 5× augmentation = 4,545 images
├─ Size: ~1.8 GB (for 5× multiplier)
└─ Format: JPG (224×224px resized)
```

---

## Step 1: Clone Repository & Prepare Data

### 1.1 Clone Repository

```bash
# Clone the repository
git clone git@github.com:k10nite/soil-fertility-classification-dataset.git
cd soil-fertility-classification-dataset

# Verify images exist
ls organized_images/Atok/ | wc -l      # Should show 239
ls organized_images/Latrinidad/ | wc -l # Should show 697

# Check total size
du -sh organized_images/  # Should be ~370 MB
```

### 1.2 Prepare Images for Upload

**Option A: Upload All Images (Recommended for Full Augmentation)**
```bash
# No preparation needed - upload directly from organized_images/
# Structure:
organized_images/
├── Atok/               # 239 images
├── Latrinidad/         # 697 images
└── combined_field_data.csv
```

**Option B: Create Combined Folder (Simpler Upload)**
```bash
# Create single folder with all images
mkdir -p upload_to_roboflow
cp organized_images/Atok/*.jpg upload_to_roboflow/
cp organized_images/Latrinidad/*.jpg upload_to_roboflow/

# Verify
ls upload_to_roboflow/ | wc -l  # Should show 909
```

**Option C: Upload Subset for Testing**
```bash
# Test with smaller subset first (e.g., 100 images)
mkdir -p test_upload
cp organized_images/Atok/*.jpg test_upload/ | head -50
cp organized_images/Latrinidad/*.jpg test_upload/ | head -50

ls test_upload/ | wc -l  # Should show 100
```

---

## Step 2: Upload Images to Roboflow

### 2.1 Create Roboflow Account & Project

```
1. Go to https://roboflow.com
2. Sign up (FREE plan)
3. Create New Project:
   ├─ Name: "Soil Fertility Augmentation"
   ├─ Type: "Classification" ← IMPORTANT
   └─ Classes: "Low", "Medium", "High" (create placeholders)
```

**Note**: We're using classification type even though we're only doing augmentation. This is required by Roboflow's project structure.

### 2.2 Upload Images

#### Method 1: Web Interface (Good for <500 images)

```
1. Go to project dashboard
2. Click "Upload"
3. Select "Upload Folder" or drag-and-drop
4. Choose folder:
   ├─ Option A: organized_images/Atok/ then organized_images/Latrinidad/
   └─ Option B: upload_to_roboflow/
5. Upload settings:
   ├─ Split: "100% Train" (we'll handle splits later)
   ├─ Assign to Class: "Unlabeled" (we don't need labels for augmentation)
   └─ Start Upload
6. Wait for upload: ~10-30 minutes for 909 images
```

**Progress Tracking**:
```
Upload Progress:
[████████████████████████░░░░░░░░] 85% (773/909 images)
Estimated time remaining: 5 minutes
```

#### Method 2: Python API (Recommended for 909 images)

```python
# Install Roboflow SDK
pip install roboflow

# Upload script
from roboflow import Roboflow
import os
from pathlib import Path

# Initialize (get API key from Roboflow settings)
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("YOUR_WORKSPACE").project("soil-fertility-augmentation")

# Upload all images
image_dirs = [
    "organized_images/Atok",
    "organized_images/Latrinidad"
]

uploaded_count = 0
total_images = 909

for image_dir in image_dirs:
    for image_file in os.listdir(image_dir):
        if image_file.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(image_dir, image_file)

            try:
                project.upload(
                    image_path=image_path,
                    split="train",  # Put all in train for now
                    num_retry_uploads=3
                )
                uploaded_count += 1
                print(f"✓ Uploaded {uploaded_count}/{total_images}: {image_file}")
            except Exception as e:
                print(f"✗ Failed to upload {image_file}: {e}")

print(f"\n✓ Upload complete: {uploaded_count}/{total_images} images")
```

**Get API Key**:
```
1. Click profile icon (top-right)
2. Go to "Account Settings"
3. Scroll to "API Keys" section
4. Copy "Private API Key"
5. Keep secure (don't commit to Git)
```

### 2.3 Verify Upload

```
1. Go to project "Images" tab
2. Check count: Should show 909 images
3. Check "Health Check":
   ├─ Duplicate images: 0 ✓
   ├─ Image quality warnings: Review if any
   └─ Format issues: 0 ✓
4. Preview some images to ensure quality
```

---

## Step 3: Configure Augmentation Settings

### 3.1 Create Dataset Version

```
1. Go to "Generate" tab
2. Click "Create New Version"
3. You'll see two sections:
   ├─ Preprocessing (applied to ALL images)
   └─ Augmentation (creates additional images)
```

### 3.2 Preprocessing Settings

```
┌─────────────────────────────────────────────────────────────┐
│ PREPROCESSING (Applied to original images)                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│ ✓ Auto-Orient                                               │
│   └─ Applies EXIF rotation automatically                    │
│                                                              │
│ ✓ Resize                                                     │
│   ├─ Method: "Stretch to 224×224"                           │
│   └─ Why: Match ResNet18/EfficientNet input size            │
│                                                              │
│ ✗ Static Crop (disable)                                     │
│   └─ We'll use RandomResizedCrop in augmentation            │
│                                                              │
│ ✗ Modify Classes (disable)                                  │
│   └─ Not needed for augmentation only                       │
│                                                              │
│ ✗ Filter Null (disable)                                     │
│   └─ Keep all images                                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 Augmentation Settings

#### Conservative Augmentation (Recommended)

```
┌─────────────────────────────────────────────────────────────┐
│ AUGMENTATION (Creates additional images)                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│ Output Per Training Example: 3                              │
│ └─ Original + 2 augmented versions = 3× dataset size        │
│                                                              │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ FLIP                                                    │ │
│ ├─────────────────────────────────────────────────────────┤ │
│ │ ✓ Horizontal: 50% probability                          │ │
│ │ ✓ Vertical: 50% probability                            │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                              │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ ROTATION                                                │ │
│ ├─────────────────────────────────────────────────────────┤ │
│ │ ✓ Enable rotation                                      │ │
│ │ ├─ Degrees: Between -180° and +180°                    │ │
│ │ └─ Fill: Black (0, 0, 0)                               │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                              │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ CROP                                                    │ │
│ ├─────────────────────────────────────────────────────────┤ │
│ │ ✓ Random Crop                                          │ │
│ │ ├─ Min Zoom: 80% (crop to 80% of original)            │ │
│ │ └─ Max Zoom: 100% (keep full image)                   │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                              │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ BRIGHTNESS & EXPOSURE                                   │ │
│ ├─────────────────────────────────────────────────────────┤ │
│ │ ✓ Brightness adjustment                                │ │
│ │ └─ Range: -20% to +20%                                 │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                              │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ BLUR                                                    │ │
│ ├─────────────────────────────────────────────────────────┤ │
│ │ ✓ Blur                                                 │ │
│ │ └─ Up to 1.5px                                         │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                              │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ NOISE                                                   │ │
│ ├─────────────────────────────────────────────────────────┤ │
│ │ ✓ Noise                                                │ │
│ │ └─ Up to 2% of pixels                                  │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                              │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ SATURATION & HUE (CONSERVATIVE)                         │ │
│ ├─────────────────────────────────────────────────────────┤ │
│ │ ✓ Saturation                                           │ │
│ │ └─ Range: -15% to +15% ← CONSERVATIVE for soil        │ │
│ │                                                         │ │
│ │ ✓ Hue                                                  │ │
│ │ └─ Range: -5° to +5° ← VERY CONSERVATIVE for soil     │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                              │
│ ✗ Cutout (disable for conservative)                         │
│ ✗ Mosaic (disable - not needed)                             │
│ ✗ Bounding Box (disable - classification only)              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Result**: 909 original + 1,818 augmented = **2,727 total images**

#### Aggressive Augmentation (For Limited Data)

```
Additional settings beyond conservative:

Output Per Training Example: 5
└─ Original + 4 augmented = 5× dataset size

Additional Augmentations:
├─ ✓ Cutout: 3 boxes, 10% of image
├─ ✓ Brightness: -30% to +30% (more aggressive)
├─ ✓ Saturation: -25% to +25% (more aggressive)
├─ ✓ Hue: -8° to +8°
└─ ✓ Blur: Up to 2.5px
```

**Result**: 909 original + 3,636 augmented = **4,545 total images**

### 3.4 Train/Valid/Test Split

```
┌─────────────────────────────────────────────────────────────┐
│ DATASET SPLIT                                                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│ Train: 70% (636 images → 1,908 after 3× aug)               │
│ Valid: 15% (136 images → 408 after 3× aug)                 │
│ Test:  15% (137 images → 411 after 3× aug)                 │
│                                                              │
│ ✓ Stratify (if labels available)                            │
│ ✗ Random seed: 42 (for reproducibility)                     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 3.5 Review & Generate

```
1. Click "Preview" to see example augmented images
2. Verify augmentations look realistic:
   ✓ Soil texture still visible
   ✓ Colors look natural
   ✓ No extreme distortions
3. If satisfied, click "Generate"
4. Wait for processing: ~5-10 minutes
```

**Preview Example**:
```
Original Image          Aug 1 (HFlip+Rotate)    Aug 2 (Crop+Bright)
┌─────────────┐        ┌─────────────┐         ┌─────────────┐
│   [Soil]    │   →    │  [lioS]     │    →    │  [Soil]     │
│  Brown      │        │  Brown      │         │  Lighter    │
│  1080×1080  │        │  Rotated 45°│         │  Cropped 90%│
└─────────────┘        └─────────────┘         └─────────────┘
```

---

## Step 4: Download Augmented Dataset

### 4.1 Export Dataset

```
1. Go to generated dataset version page
2. Click "Download"
3. Choose format:
   ┌─────────────────────────────────────────────────────┐
   │ RECOMMENDED FOR PYTORCH:                            │
   ├─────────────────────────────────────────────────────┤
   │                                                      │
   │ Format: "Folder Structure"                          │
   │ └─ Organizes images into train/valid/test folders   │
   │                                                      │
   │ OR                                                   │
   │                                                      │
   │ Format: "COCO JSON"                                 │
   │ └─ Images + annotations.json (more metadata)        │
   │                                                      │
   └─────────────────────────────────────────────────────┘
4. Click "Download ZIP"
5. Wait for download: ~500MB-2GB depending on augmentation
```

### 4.2 Extract Downloaded Dataset

```bash
# Download will be named something like:
# soil-fertility-augmentation-v1.zip

# Extract
unzip soil-fertility-augmentation-v1.zip -d roboflow_augmented

# Check structure (Folder format)
tree roboflow_augmented/
```

**Folder Structure Output**:
```
roboflow_augmented/
├── train/
│   ├── image_001.jpg          # Original
│   ├── image_001_aug_1.jpg    # Augmented version 1
│   ├── image_001_aug_2.jpg    # Augmented version 2
│   ├── image_002.jpg
│   ├── image_002_aug_1.jpg
│   └── ... (1,908 total)
├── valid/
│   ├── image_450.jpg
│   ├── image_450_aug_1.jpg
│   └── ... (408 total)
├── test/
│   ├── image_800.jpg
│   ├── image_800_aug_1.jpg
│   └── ... (411 total)
└── README.roboflow.txt
```

**COCO JSON Structure**:
```
roboflow_augmented/
├── train/
│   ├── images/
│   │   ├── image_001.jpg
│   │   ├── image_001_aug_1.jpg
│   │   └── ... (1,908 total)
│   └── _annotations.coco.json
├── valid/
│   ├── images/
│   │   └── ... (408 total)
│   └── _annotations.coco.json
├── test/
│   ├── images/
│   │   └── ... (411 total)
│   └── _annotations.coco.json
└── README.dataset.txt
```

---

## Step 5: Integrate with Training Pipeline

### 5.1 Organize Augmented Data

```bash
# Create organized structure for our pipeline
mkdir -p augmented_dataset/images
mkdir -p augmented_dataset/splits

# Copy all images to single folder
cp roboflow_augmented/train/*.jpg augmented_dataset/images/
cp roboflow_augmented/valid/*.jpg augmented_dataset/images/
cp roboflow_augmented/test/*.jpg augmented_dataset/images/

# Count images
ls augmented_dataset/images/ | wc -l  # Should show 2,727 (3× augmentation)
```

### 5.2 Create CSV Mapping

```python
import pandas as pd
import os

# Create CSV mapping for augmented images
data = []

for split in ['train', 'valid', 'test']:
    split_dir = f'roboflow_augmented/{split}'

    for image_file in os.listdir(split_dir):
        if image_file.endswith('.jpg'):
            # Extract original filename (before _aug_)
            if '_aug_' in image_file:
                original_file = image_file.split('_aug_')[0] + '.jpg'
                is_augmented = True
                aug_number = int(image_file.split('_aug_')[1].split('.')[0])
            else:
                original_file = image_file
                is_augmented = False
                aug_number = 0

            data.append({
                'image_filename': image_file,
                'original_filename': original_file,
                'split': split,
                'is_augmented': is_augmented,
                'augmentation_number': aug_number
            })

# Create DataFrame
df = pd.DataFrame(data)
df.to_csv('augmented_dataset/augmented_mapping.csv', index=False)

print(f"✓ Created mapping for {len(df)} images")
print(f"\nSplit distribution:")
print(df['split'].value_counts())
print(f"\nAugmentation distribution:")
print(df['is_augmented'].value_counts())
```

### 5.3 Train with Augmented Dataset

#### Option A: Use Augmented Images Directly (No Runtime Augmentation)

```python
from ml_pipeline.data import SoilImageDataset, create_val_loader
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd

# Load augmented data
df = pd.read_csv('augmented_dataset/augmented_mapping.csv')

# Separate by split
train_df = df[df['split'] == 'train']
valid_df = df[df['split'] == 'valid']
test_df = df[df['split'] == 'test']

# Minimal transform (images already augmented)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# Create datasets (using augmented images, no additional augmentation)
train_dataset = SoilImageDataset(
    csv_path='augmented_dataset/train_data.csv',  # Create from train_df
    img_dir='augmented_dataset/images',
    transform=transform,  # No augmentation, images already augmented
)

val_dataset = SoilImageDataset(
    csv_path='augmented_dataset/valid_data.csv',
    img_dir='augmented_dataset/images',
    transform=transform,
)

# Create loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print(f"✓ Train: {len(train_dataset)} images (includes augmented)")
print(f"✓ Valid: {len(val_dataset)} images (includes augmented)")
```

#### Option B: Combine Roboflow + Runtime Augmentation

```python
from ml_pipeline.data import get_all_transforms, SoilImageDataset

# Use Roboflow-augmented images as base
# Apply ADDITIONAL runtime augmentation during training
transforms = get_all_transforms(
    image_size=(224, 224),
    augmentation_strategy='conservative'  # Light additional augmentation
)

train_dataset = SoilImageDataset(
    csv_path='augmented_dataset/train_data.csv',
    img_dir='augmented_dataset/images',
    transform=transforms['train'],  # Additional augmentation on top of Roboflow
)

# This gives you: 1,908 base images × additional runtime augmentation
# Effective dataset: ~5,000-10,000 variations
```

**Recommendation**: Use **Option A** (augmented images directly) if:
- You want faster training (no runtime augmentation overhead)
- Roboflow augmentation is sufficient
- Training on CPU or limited GPU

Use **Option B** (combine augmentations) if:
- You want maximum variety
- Have powerful GPU
- Need best possible generalization

### 5.4 Complete Training Example

```python
import torch
import torch.nn as nn
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd

# Setup
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 50
BATCH_SIZE = 32

# Transform (minimal, images already augmented)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# Load augmented datasets
# (Assuming you've created train/valid/test CSV files)
train_dataset = SoilImageDataset('augmented_dataset/train_data.csv',
                                'augmented_dataset/images', transform)
val_dataset = SoilImageDataset('augmented_dataset/valid_data.csv',
                              'augmented_dataset/images', transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model
model = resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 3)  # 3 classes
model = model.to(DEVICE)

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(EPOCHS):
    # Train
    model.train()
    train_loss = 0.0

    for images, labels, _ in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Validate
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels, _ in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"  Train Loss: {train_loss/len(train_loader):.4f}")
    print(f"  Val Loss: {val_loss/len(val_loader):.4f}")
    print(f"  Val Accuracy: {100*correct/total:.2f}%")

# Save model
torch.save(model.state_dict(), 'soil_model_roboflow_augmented.pth')
print("✓ Training complete!")
```

---

## Workflow Summary

### Complete Flow Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│ STEP 1: PREPARE DATA                                             │
├──────────────────────────────────────────────────────────────────┤
│ • Clone GitHub repository                                        │
│ • Verify 909 images in organized_images/                        │
│ Time: 5-10 minutes                                               │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│ STEP 2: UPLOAD TO ROBOFLOW                                       │
├──────────────────────────────────────────────────────────────────┤
│ • Create Roboflow account & project                             │
│ • Upload 909 images (API or web interface)                      │
│ Time: 30-60 minutes (upload)                                     │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│ STEP 3: CONFIGURE AUGMENTATION                                   │
├──────────────────────────────────────────────────────────────────┤
│ • Set preprocessing: Resize to 224×224                          │
│ • Set augmentation: Conservative (3×) or Aggressive (5×)        │
│   - Flip, Rotate, Crop, Brightness, Blur, Noise                │
│ • Configure split: 70/15/15                                     │
│ • Generate dataset version                                       │
│ Time: 10-15 minutes (including processing)                       │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│ STEP 4: DOWNLOAD AUGMENTED DATA                                  │
├──────────────────────────────────────────────────────────────────┤
│ • Export as Folder Structure or COCO JSON                       │
│ • Download ZIP (~500MB-2GB)                                     │
│ • Extract to local directory                                    │
│ Time: 20-40 minutes (download)                                   │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│ STEP 5: ORGANIZE & PREPARE                                       │
├──────────────────────────────────────────────────────────────────┤
│ • Copy images to training directory                             │
│ • Create CSV mapping file                                        │
│ • Verify split distribution                                     │
│ Time: 5-10 minutes                                               │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│ STEP 6: TRAIN MODEL                                              │
├──────────────────────────────────────────────────────────────────┤
│ • Use augmented images with minimal runtime transforms          │
│ • Train ResNet18 or EfficientNetV2-S                            │
│ • Monitor training metrics                                       │
│ Time: 45-120 minutes (depending on GPU)                          │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│ RESULT: TRAINED MODEL                                            │
├──────────────────────────────────────────────────────────────────┤
│ • Model trained on 2,727 images (3×) or 4,545 images (5×)      │
│ • Expected accuracy: 80-85% (depending on labels)               │
│ • Ready for deployment                                           │
└──────────────────────────────────────────────────────────────────┘

TOTAL TIME: ~2-3 hours (mostly upload/download)
```

### Quick Reference

| Step | Action | Time | Output |
|------|--------|------|--------|
| 1 | Clone repo | 5 min | 909 images locally |
| 2 | Upload to Roboflow | 30-60 min | 909 images on cloud |
| 3 | Configure & generate | 10 min | Augmented dataset |
| 4 | Download | 20-40 min | 2,727-4,545 images |
| 5 | Organize | 5 min | CSV + organized folders |
| 6 | Train | 45-120 min | Trained model |

### Expected Results

```
Conservative Augmentation (3×):
├─ Original: 909 images
├─ After augmentation: 2,727 images
├─ Train: 1,908 images
├─ Valid: 408 images
├─ Test: 411 images
└─ Storage: ~800 MB-1 GB

Aggressive Augmentation (5×):
├─ Original: 909 images
├─ After augmentation: 4,545 images
├─ Train: 3,180 images
├─ Valid: 680 images
├─ Test: 685 images
└─ Storage: ~1.5-2 GB
```

---

## Tips & Best Practices

### Before Upload
- ✓ Test with small batch (50-100 images) first
- ✓ Verify image quality (no corrupted files)
- ✓ Check internet speed (upload 400MB)

### During Configuration
- ✓ Preview augmentations before generating
- ✓ Start conservative, increase if needed
- ✓ Keep saturation/hue changes minimal (soil color important)
- ✓ Document settings for reproducibility

### After Download
- ✓ Verify image count matches expected
- ✓ Check augmented images look realistic
- ✓ Compare file sizes (should be reasonable)
- ✓ Back up augmented dataset (in case need to regenerate)

### Training
- ✓ Use augmented images directly (Option A) for simplicity
- ✓ No additional runtime augmentation needed (already augmented)
- ✓ Monitor if model benefits from augmentation
- ✓ Compare vs training on original 909 images

---

## Troubleshooting

### Upload Issues
```
Problem: Upload fails or timeout
Solution:
├─ Split into smaller batches (100 images at a time)
├─ Use API method instead of web interface
├─ Check internet connection stability
└─ Retry failed uploads
```

### Augmentation Issues
```
Problem: Augmented images look unrealistic
Solution:
├─ Reduce brightness range (±20% → ±10%)
├─ Reduce saturation/hue changes
├─ Disable cutout for conservative approach
├─ Preview before generating
└─ Regenerate version with adjusted settings
```

### Download Issues
```
Problem: Download slow or fails
Solution:
├─ Use download manager (resume capability)
├─ Download during off-peak hours
├─ Split into train/valid/test separate downloads
└─ Contact Roboflow support if persistent
```

### Training Issues
```
Problem: Model not improving with augmented data
Solution:
├─ Verify labels are correct
├─ Check if augmentation too aggressive
├─ Try smaller augmentation multiplier (5× → 3×)
├─ Compare with training on original images
└─ Ensure images properly normalized
```

---

## Cost

**FREE** - Using Roboflow for augmentation only requires:
- Free plan (10,000 images limit)
- No inference costs (only augmentation)
- No monthly fees

**Storage**: You'll need ~2-3 GB local storage for augmented dataset

---

**Next Steps**: Follow Step 1 to start the workflow!
