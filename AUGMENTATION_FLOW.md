# Soil Image Data Augmentation Flow

## Complete Pipeline Documentation

This document provides a comprehensive, step-by-step explanation of how data augmentation is conducted for the soil fertility classification dataset.

---

## Table of Contents

1. [Overview](#overview)
2. [Pipeline Architecture](#pipeline-architecture)
3. [Augmentation Flow Diagram](#augmentation-flow-diagram)
4. [Transformation Stages (Conservative)](#transformation-stages-conservative)
5. [Transformation Stages (Aggressive)](#transformation-stages-aggressive)
6. [Decision Tree](#decision-tree)
7. [Integration with Training Loop](#integration-with-training-loop)
8. [Performance & Memory Considerations](#performance--memory-considerations)
9. [Before/After Examples](#beforeafter-examples)
10. [Best Practices](#best-practices)

---

## Overview

### Purpose
Data augmentation artificially expands our training dataset from **936 original soil images** to an effective dataset size of **5,000-15,000 training samples** through on-the-fly transformations during training.

### Why Augmentation?
- **Prevents overfitting**: Model sees slightly different variations each epoch
- **Improves generalization**: Model learns robust features invariant to lighting, orientation, camera position
- **Simulates real-world conditions**: Reproduces natural variations in field photography
- **Cost-effective**: Eliminates need to physically collect thousands more images

### Key Principle
All transformations preserve soil-specific characteristics (texture, color, moisture patterns) while introducing realistic variations that might occur during actual field data collection.

---

## Pipeline Architecture

### Three Distinct Pipelines

```
┌─────────────────────────────────────────────────────────────┐
│                   AUGMENTATION SYSTEM                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────┐ │
│  │ TRAINING         │  │ VALIDATION       │  │ TEST      │ │
│  │ Pipeline         │  │ Pipeline         │  │ Pipeline  │ │
│  ├──────────────────┤  ├──────────────────┤  ├───────────┤ │
│  │ • Conservative   │  │ • Resize only    │  │ • Resize  │ │
│  │   OR             │  │ • Normalize      │  │ • Normal. │ │
│  │ • Aggressive     │  │ • To Tensor      │  │ • Tensor  │ │
│  └──────────────────┘  └──────────────────┘  └───────────┘ │
│         ▲                      ▲                    ▲       │
│         │                      │                    │       │
└─────────┼──────────────────────┼────────────────────┼───────┘
          │                      │                    │
    ┌─────┴─────┐          ┌────┴────┐         ┌────┴────┐
    │ Train Set │          │ Val Set │         │Test Set │
    │ (70%)     │          │ (15%)   │         │ (15%)   │
    └───────────┘          └─────────┘         └─────────┘
```

### Pipeline Selection

| Pipeline | When Applied | Augmentation | Dataset Split |
|----------|-------------|--------------|---------------|
| **Training (Conservative)** | Default for training | 7 transforms + normalize | 70% of data |
| **Training (Aggressive)** | Limited data scenarios | 13 transforms + normalize | 70% of data |
| **Validation** | Model evaluation during training | Resize + normalize only | 15% of data |
| **Test** | Final model evaluation | Resize + normalize only | 15% of data |

---

## Augmentation Flow Diagram

### Complete Flow (Conservative Strategy)

```
┌──────────────────────────────────────────────────────────────────────┐
│                    ORIGINAL SOIL IMAGE                                │
│                    (1080×1080px, RGB)                                 │
│                    Size: ~400KB per image                             │
└───────────────────────────┬──────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────────┐
│ STAGE 1: GEOMETRIC TRANSFORMATIONS                                    │
│ Purpose: Orientation invariance (soil has no canonical orientation)   │
├──────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  ┌────────────────────┐                                              │
│  │ HorizontalFlip     │  Probability: 50%                            │
│  │ p=0.5              │  Why: Camera can photograph from either side  │
│  └─────────┬──────────┘  Effect: Mirrors image left↔right            │
│            ▼                                                          │
│  ┌────────────────────┐                                              │
│  │ VerticalFlip       │  Probability: 50%                            │
│  │ p=0.5              │  Why: Soil texture same from all angles      │
│  └─────────┬──────────┘  Effect: Mirrors image top↔bottom            │
│            ▼                                                          │
│  ┌────────────────────┐                                              │
│  │ Rotate             │  Probability: 70%                            │
│  │ limit=±180°        │  Why: Camera can be at any rotation          │
│  │ p=0.7              │  Effect: Random rotation (any angle)         │
│  └─────────┬──────────┘  Border: Filled with black (BORDER_CONSTANT) │
│            │                                                          │
└────────────┼──────────────────────────────────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────────────────────────────────┐
│ STAGE 2: SPATIAL CROPPING & RESIZING                                  │
│ Purpose: Scale invariance + focus on different soil regions           │
├──────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  ┌────────────────────────────────────────────────────────┐          │
│  │ RandomResizedCrop                                       │          │
│  │ • Scale: 80-100% of original                           │          │
│  │ • Aspect ratio: 0.9-1.1 (nearly square)                │          │
│  │ • Output: 224×224px (ResNet18 input size)              │          │
│  │ • Probability: 100% (always applied)                   │          │
│  └─────────────────────────┬──────────────────────────────┘          │
│                            │                                          │
│  Why this works for soil:                                            │
│  ✓ Simulates different camera distances                              │
│  ✓ Focuses on different soil patches within frame                    │
│  ✓ Maintains aspect ratio (soil texture not distorted)               │
│  ✓ Resizes to network input size (224×224px)                         │
│                            │                                          │
└────────────────────────────┼──────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│ STAGE 3: PHOTOMETRIC TRANSFORMATIONS                                  │
│ Purpose: Lighting & color invariance (field conditions vary)          │
├──────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  ┌────────────────────────────────────────────────────────┐          │
│  │ ColorJitter                      Probability: 60%       │          │
│  ├────────────────────────────────────────────────────────┤          │
│  │ • Brightness: ±20% (0.8 to 1.2×)                       │          │
│  │   → Simulates: Cloudy vs sunny days, time of day       │          │
│  │                                                         │          │
│  │ • Contrast: ±20% (0.8 to 1.2×)                         │          │
│  │   → Simulates: Hazy conditions, direct sunlight        │          │
│  │                                                         │          │
│  │ • Saturation: ±15% (0.85 to 1.15×)                     │          │
│  │   → Simulates: Soil moisture variations                │          │
│  │   → CONSERVATIVE: Preserves brown/reddish soil tones    │          │
│  │                                                         │          │
│  │ • Hue: ±5% (±0.05 range)                               │          │
│  │   → Simulates: White balance differences                │          │
│  │   → VERY CONSERVATIVE: Soil color critical for NPK      │          │
│  └─────────────────────────┬──────────────────────────────┘          │
│                            │                                          │
│  Why conservative color changes?                                     │
│  • Soil color correlates with fertility (organic matter = darker)    │
│  • Reddish hues may indicate iron content                            │
│  • Over-augmentation could destroy these diagnostic features         │
│                            │                                          │
└────────────────────────────┼──────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│ STAGE 4: IMAGE QUALITY DEGRADATION                                    │
│ Purpose: Camera quality & focus invariance                            │
├──────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  ┌────────────────────┐                                              │
│  │ GaussianBlur       │  Probability: 30%                            │
│  │ kernel: 3-5px      │  Why: Camera autofocus variations            │
│  │ p=0.3              │  Effect: Slight blur (simulates defocus)     │
│  └─────────┬──────────┘  Safe: Texture still visible                 │
│            ▼                                                          │
│  ┌────────────────────┐                                              │
│  │ GaussNoise         │  Probability: 30%                            │
│  │ variance: 10-30    │  Why: Camera sensor noise (especially phones)│
│  │ p=0.3              │  Effect: Grainy appearance                   │
│  └─────────┬──────────┘  Safe: Doesn't obscure soil features         │
│            │                                                          │
│  Why degradation helps:                                              │
│  • Field photos taken with various phone cameras (quality varies)    │
│  • Model must work on older devices with noisy sensors               │
│  • Simulates realistic data collection conditions                    │
│            │                                                          │
└────────────┼──────────────────────────────────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────────────────────────────────┐
│ STAGE 5: NORMALIZATION & TENSOR CONVERSION                            │
│ Purpose: Prepare for neural network input                             │
├──────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  ┌────────────────────────────────────────────────────────┐          │
│  │ Normalize (ImageNet statistics)                        │          │
│  ├────────────────────────────────────────────────────────┤          │
│  │ Mean: [0.485, 0.456, 0.406] (R, G, B)                  │          │
│  │ Std:  [0.229, 0.224, 0.225] (R, G, B)                  │          │
│  │                                                         │          │
│  │ Formula: output = (input - mean) / std                 │          │
│  │                                                         │          │
│  │ Why ImageNet stats?                                    │          │
│  │ • Using transfer learning (ResNet18/EfficientNetV2)    │          │
│  │ • Pre-trained on ImageNet with these normalization     │          │
│  │ • MUST use same stats for compatibility                │          │
│  └─────────────────────────┬──────────────────────────────┘          │
│                            ▼                                          │
│  ┌────────────────────────────────────────────────────────┐          │
│  │ ToTensorV2 (Albumentations → PyTorch)                  │          │
│  ├────────────────────────────────────────────────────────┤          │
│  │ • Convert: numpy array → torch.Tensor                  │          │
│  │ • Transpose: (H, W, C) → (C, H, W)                     │          │
│  │ • Data type: uint8 [0-255] → float32 [0.0-1.0]        │          │
│  │ • Output shape: (3, 224, 224)                          │          │
│  └─────────────────────────┬──────────────────────────────┘          │
│                            │                                          │
└────────────────────────────┼──────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    AUGMENTED IMAGE OUTPUT                             │
│                    torch.Tensor (3, 224, 224)                         │
│                    dtype: float32                                     │
│                    Ready for model input                              │
└──────────────────────────────────────────────────────────────────────┘
```

### Effective Dataset Size Calculation

```
Original dataset: 936 images
├─ Train split (70%): 655 images
├─ Val split (15%): 141 images
└─ Test split (15%): 140 images

CONSERVATIVE AUGMENTATION:
───────────────────────────
Each training image can generate ~8-12 unique variations per epoch
655 images × 10 variations (average) = 6,550 effective training samples/epoch
Over 50 epochs: Model sees ~327,500 total training samples
Unique combinations: ~6,000-7,000 distinct augmented images

AGGRESSIVE AUGMENTATION:
───────────────────────────
Each training image can generate ~15-25 unique variations per epoch
655 images × 20 variations (average) = 13,100 effective training samples/epoch
Over 50 epochs: Model sees ~655,000 total training samples
Unique combinations: ~10,000-15,000 distinct augmented images
```

---

## Transformation Stages (Conservative)

### Stage 1: Geometric Transformations

#### 1.1 Horizontal Flip
```
Original:                    Flipped:
[Soil texture →]            [← Soil texture]
probability: 50%
```

**Purpose**: Soil has no inherent left/right orientation
**When applied**: Random 50% chance
**Effect**: Mirror image horizontally
**Safe for soil**: YES - texture patterns are symmetric
**Example**: Photo taken from north vs south side of plot

#### 1.2 Vertical Flip
```
Original:                    Flipped:
[Top of frame]              [Bottom becomes top]
[Bottom of frame]           [Top becomes bottom]
probability: 50%
```

**Purpose**: Soil has no inherent top/bottom orientation
**When applied**: Random 50% chance
**Effect**: Mirror image vertically
**Safe for soil**: YES - gravity doesn't affect 2D texture appearance
**Example**: Camera held normally vs upside down

#### 1.3 Rotation (±180°)
```
Original (0°)    Rotated (45°)    Rotated (90°)    Rotated (180°)
[Soil]           [/Soil/]         [⟲Soil]          [lioS]
probability: 70%
```

**Purpose**: Camera can be at any angle relative to soil
**When applied**: Random 70% chance, random angle ±180°
**Effect**: Rotate image by random degrees
**Border handling**: Black padding (BORDER_CONSTANT)
**Safe for soil**: YES - soil particles have no preferred direction
**Example**: Photographer holding phone at different angles

---

### Stage 2: Spatial Transformations

#### 2.1 Random Resized Crop
```
Original 1080×1080px          Crop + Resize to 224×224px
┌─────────────────────┐
│                     │       ┌──────────┐
│    ┌──────────┐    │       │          │
│    │ CROPPED  │    │  →    │ RESIZED  │
│    │  REGION  │    │       │ TO 224px │
│    │ (80-100%)│    │       │          │
│    └──────────┘    │       └──────────┘
│                     │
└─────────────────────┘
Scale: 80-100% of original
Aspect ratio: 0.9-1.1 (nearly square)
```

**Purpose**:
1. Resize to network input size (224×224 for ResNet18)
2. Simulate different camera distances
3. Focus on different soil regions

**Parameters**:
- **Scale range**: 80-100% (conservative, preserves most soil)
- **Aspect ratio**: 0.9-1.1 (prevents distortion)
- **Interpolation**: Bilinear (smooth, no artifacts)

**Why this range**:
- 80% minimum: Still captures representative soil texture
- 100% maximum: Full image preserved
- Avoids extreme crops that might miss diagnostic features

**Example scenarios**:
- 100% crop: Camera 1 meter above soil
- 90% crop: Camera 0.9 meters above soil
- 80% crop: Camera 0.8 meters (closer view)

---

### Stage 3: Photometric Transformations

#### 3.1 Color Jitter (60% probability)

##### Brightness: ±20%
```
Dark (-20%)         Normal (0%)         Bright (+20%)
[██████]           [██████]            [██████]
Simulates: Heavy    Original            Direct
cloud cover        conditions           sunlight
```

**Purpose**: Handle varying lighting conditions
**Range**: 0.8× to 1.2× brightness multiplier
**Safe for soil**: YES - natural field lighting varies significantly

##### Contrast: ±20%
```
Low Contrast        Normal              High Contrast
[▓▓▓▓▓▓]           [███▒▒▒]            [███░░░]
Simulates: Hazy     Original            Harsh
atmosphere         conditions           shadows
```

**Purpose**: Handle atmospheric conditions
**Range**: 0.8× to 1.2× contrast multiplier
**Safe for soil**: YES - weather affects contrast naturally

##### Saturation: ±15% (CONSERVATIVE)
```
Desaturated         Normal              Saturated
[Brown→Gray]       [Brown]             [Brown→Red]
Simulates: Dry      Original            Moist
soil               conditions           soil
```

**Purpose**: Handle soil moisture variations
**Range**: 0.85× to 1.15× saturation multiplier
**Conservative**: Limited to ±15% to preserve soil color
**Critical**: Soil color indicates organic matter, moisture, minerals

##### Hue: ±5% (VERY CONSERVATIVE)
```
Shift: ±0.05 in HSV color space
Effect: Minimal color shift (brown ↔ reddish-brown)
```

**Purpose**: Handle white balance variations
**Range**: ±0.05 in HSV hue channel
**Very conservative**: Soil color is diagnostic feature
**Safe**: Minimal shift preserves brown/red/gray soil tones

---

### Stage 4: Image Quality Degradation

#### 4.1 Gaussian Blur (30% probability)
```
Original (Sharp)         Blurred (kernel=5px)
[Clear texture]         [Soft texture]
┌──┬──┬──┐             ┌─────────┐
│▓▓│░░│▓▓│             │  ░▓░    │
│░░│▓▓│░░│      →      │  ▓░▓    │
│▓▓│░░│▓▓│             │  ░▓░    │
└──┴──┴──┘             └─────────┘
```

**Purpose**: Simulate camera autofocus variations
**Kernel size**: 3-5 pixels
**Effect**: Slight blur (texture still distinguishable)
**When**: Random 30% chance
**Safe for soil**: YES - simulates realistic phone camera defocus

#### 4.2 Gaussian Noise (30% probability)
```
Original (Clean)         Noisy (variance=30)
[Smooth soil]           [Grainy soil]
████████                ▓█▓█▓█▓█
████████        →       █▓█▓█▓█▓
████████                ▓█▓█▓█▓█
```

**Purpose**: Simulate camera sensor noise
**Variance range**: 10-30 (mild to moderate)
**Effect**: Grainy appearance
**When**: Random 30% chance
**Safe for soil**: YES - phone cameras have sensor noise, especially in low light

---

### Stage 5: Normalization & Preparation

#### 5.1 ImageNet Normalization
```python
# Per-channel normalization
R_channel = (R - 0.485) / 0.229
G_channel = (G - 0.456) / 0.224
B_channel = (B - 0.406) / 0.225

# Example:
Input pixel:  [120, 95, 80] (RGB, range 0-255)
After /255:   [0.47, 0.37, 0.31] (normalized to 0-1)
After norm:   [−0.065, −0.384, −0.431] (standardized)
```

**Purpose**: Match pre-trained model expectations
**Statistics**: ImageNet dataset mean and std
**Why required**: Transfer learning from ImageNet-trained models
**Effect**: Converts [0, 1] range to approximately [-2, 2] range

#### 5.2 Tensor Conversion
```
Input:  numpy.ndarray, shape (224, 224, 3), dtype uint8
        Dimensions: (Height, Width, Channels)
        Range: 0-255

Output: torch.Tensor, shape (3, 224, 224), dtype float32
        Dimensions: (Channels, Height, Width)
        Range: ~[-2.0, 2.0] (after normalization)
```

**Purpose**: Convert to PyTorch format
**Changes**:
1. Data type: uint8 → float32
2. Dimension order: HWC → CHW
3. Value range: [0, 255] → [0, 1] → normalized

---

## Transformation Stages (Aggressive)

### Additional Transforms Beyond Conservative

#### A1. Shift-Scale-Rotate (50% probability)
```
Original              Shifted + Scaled + Rotated
┌─────────────┐      ┌─────────────┐
│   [Soil]    │      │  ╱[Soil]╲   │
│             │  →   │ (moved+     │
│             │      │  rotated)   │
└─────────────┘      └─────────────┘
```

**Purpose**: Combined geometric transformation
**Parameters**:
- Shift: ±10% of image size
- Scale: ±20% zoom
- Rotate: ±180°

**When to use**: Extreme variation needed

#### A2. Random Shadow (30% probability)
```
Original              With Shadow
┌─────────────┐      ┌─────────────┐
│   ░░░░░░    │      │   ░░░░░░    │
│   ░░░░░░    │  →   │   ░░▓▓▓▓    │ ← Shadow
│   ░░░░░░    │      │   ░░▓▓▓▓    │
└─────────────┘      └─────────────┘
```

**Purpose**: Simulate shadows from plants, clouds, photographer
**Parameters**: 1-2 shadows, dimension=5
**Effect**: Darkened regions
**Realistic**: YES - field photos often have shadows

#### A3. Motion Blur (40% probability, either Gaussian OR Motion)
```
Original              Motion Blurred
┌──┬──┬──┐           ┌───────────┐
│▓▓│░░│▓▓│           │ ▓▓░░▓▓──→ │
│░░│▓▓│░░│      →    │ ░░▓▓░░──→ │
│▓▓│░░│▓▓│           │ ▓▓░░▓▓──→ │
└──┴──┴──┘           └───────────┘
```

**Purpose**: Simulate camera shake
**Kernel**: 3-7 pixels
**Direction**: Random
**When**: Alternative to Gaussian blur (40% either/or)

#### A4. Coarse Dropout (30% probability)
```
Original              With Cutouts
┌─────────────┐      ┌─────────────┐
│   ░░░░░░    │      │   ░░██░░    │ ← Black rectangle
│   ░░░░░░    │  →   │   ░░░░██    │ ← Black rectangle
│   ░░░░░░    │      │   ░░░░░░    │
└─────────────┘      └─────────────┘
```

**Purpose**: Simulate partial occlusions (twigs, leaves, debris)
**Parameters**:
- Holes: 1-3 per image
- Size: 8×8 to 32×32 pixels
- Fill: Black (0)

**Why safe**: Forces model to focus on visible regions, not rely on all pixels

---

## Decision Tree

### Choosing Augmentation Strategy

```
START: Do I need data augmentation for TRAINING?
│
├─ YES (Training Set)
│  │
│  └─ How much labeled data do I have?
│     │
│     ├─ >500 labeled images
│     │  └─→ CONSERVATIVE augmentation ✓
│     │      • Realistic transformations
│     │      • Preserves soil characteristics
│     │      • Effective dataset: 5,000-7,000 images
│     │      • Recommended for production
│     │
│     ├─ 200-500 labeled images
│     │  └─→ CONSERVATIVE (default) or AGGRESSIVE (if underfitting)
│     │      • Start with conservative
│     │      • Switch to aggressive if:
│     │        - Validation accuracy plateaus early
│     │        - Large train/val accuracy gap
│     │        - Model memorizes training data
│     │
│     └─ <200 labeled images
│        └─→ AGGRESSIVE augmentation ⚠
│            • Maximum variety needed
│            • Effective dataset: 10,000-15,000 images
│            • Monitor: May introduce unrealistic samples
│            • Better: Collect more data if possible
│
└─ NO (Validation/Test Sets)
   └─→ MINIMAL augmentation (resize + normalize only) ✓
       • No randomness
       • Reproducible evaluation
       • Fair comparison across runs
```

### Implementation Selection

```python
# Method 1: Get single pipeline
from ml_pipeline.data import get_conservative_augmentation

train_transform = get_conservative_augmentation(
    image_size=(224, 224)  # ResNet18 input
)

# Method 2: Get all pipelines at once
from ml_pipeline.data import get_all_transforms

transforms = get_all_transforms(
    image_size=(224, 224),
    augmentation_strategy='conservative'  # or 'aggressive'
)

train_transform = transforms['train']
val_transform = transforms['val']
test_transform = transforms['test']
```

---

## Integration with Training Loop

### Complete Pipeline Flow

```
┌───────────────────────────────────────────────────────────────────┐
│ STEP 1: DATASET PREPARATION (ONE-TIME SETUP)                       │
└───────────────────────────────────────────────────────────────────┘
│
├─ Load CSV with image metadata
│  • 936 images total
│  • Columns: image_filename, municipality, crops, soil_class, etc.
│
├─ Split into train/val/test (70/15/15)
│  • Train: 655 images (will be augmented)
│  • Val:   141 images (no augmentation)
│  • Test:  140 images (no augmentation)
│  • Stratified: Maintains class distribution
│
└─ Create Dataset objects
   • train_dataset = SoilImageDataset(train_csv, train_transform)
   • val_dataset   = SoilImageDataset(val_csv, val_transform)
   • test_dataset  = SoilImageDataset(test_csv, test_transform)

┌───────────────────────────────────────────────────────────────────┐
│ STEP 2: DATALOADER CREATION                                        │
└───────────────────────────────────────────────────────────────────┘
│
├─ Training DataLoader
│  • Batch size: 32 images
│  • Shuffle: YES (randomize order each epoch)
│  • Weighted sampler: YES (handle class imbalance)
│  • Num workers: 0 (Windows) or 4 (Linux/Mac)
│  • Pin memory: YES (faster GPU transfer)
│
└─ Validation/Test DataLoaders
   • Batch size: 32 images
   • Shuffle: NO (consistent evaluation)
   • Weighted sampler: NO
   • Num workers: 0 (Windows) or 4 (Linux/Mac)

┌───────────────────────────────────────────────────────────────────┐
│ STEP 3: TRAINING LOOP (HAPPENS EVERY EPOCH)                        │
└───────────────────────────────────────────────────────────────────┘

for epoch in range(50):  # Train for 50 epochs

    ┌─────────────────────────────────────────────────────────────┐
    │ TRAINING PHASE                                              │
    └─────────────────────────────────────────────────────────────┘

    for batch_idx, (images, labels, metadata) in enumerate(train_loader):

        ┌──────────────────────────────────────────────────────┐
        │ AUGMENTATION HAPPENS HERE (ON-THE-FLY)              │
        │                                                      │
        │ 1. DataLoader requests batch of 32 images           │
        │ 2. For each image:                                  │
        │    a. Load image from disk (1080×1080px)           │
        │    b. Apply augmentation pipeline:                  │
        │       • HorizontalFlip (maybe)                      │
        │       • VerticalFlip (maybe)                        │
        │       • Rotate (maybe)                              │
        │       • RandomResizedCrop (always)                  │
        │       • ColorJitter (maybe)                         │
        │       • GaussianBlur (maybe)                        │
        │       • GaussNoise (maybe)                          │
        │       • Normalize (always)                          │
        │       • ToTensor (always)                           │
        │    c. Result: Tensor (3, 224, 224)                  │
        │ 3. Stack 32 tensors → batch (32, 3, 224, 224)      │
        │ 4. Return batch to training loop                    │
        │                                                      │
        │ CRITICAL: Different augmentation EVERY TIME!        │
        │ Same image in epoch 1 vs epoch 2 = different result │
        └──────────────────────────────────────────────────────┘

        # Move to GPU
        images = images.to(device)  # (32, 3, 224, 224)
        labels = labels.to(device)  # (32,)

        # Forward pass
        outputs = model(images)  # (32, 3) - 3 classes
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    ┌─────────────────────────────────────────────────────────────┐
    │ VALIDATION PHASE                                            │
    └─────────────────────────────────────────────────────────────┘

    with torch.no_grad():
        for images, labels, metadata in val_loader:

            ┌──────────────────────────────────────────────────┐
            │ NO AUGMENTATION (Resize + Normalize only)       │
            │                                                  │
            │ 1. Load image from disk (1080×1080px)          │
            │ 2. Resize to 224×224px (deterministic)         │
            │ 3. Normalize with ImageNet stats               │
            │ 4. ToTensor                                     │
            │                                                  │
            │ CRITICAL: Same image = same result every time   │
            │ Ensures reproducible evaluation                 │
            └──────────────────────────────────────────────────┘

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            # Compute metrics (accuracy, precision, recall)

┌───────────────────────────────────────────────────────────────────┐
│ KEY INSIGHT: On-the-Fly Augmentation                              │
├───────────────────────────────────────────────────────────────────┤
│                                                                    │
│ • Images NOT pre-augmented and saved to disk                      │
│ • Augmentation happens DURING training (real-time)                │
│ • Each epoch sees DIFFERENT augmented versions                    │
│                                                                    │
│ Example:                                                          │
│   Original image: "Atok_001.jpg"                                  │
│                                                                    │
│   Epoch 1, Batch 5: HFlip=YES, Rotate=45°, Brightness=+10%       │
│   Epoch 2, Batch 12: HFlip=NO, Rotate=−30°, Brightness=−5%       │
│   Epoch 3, Batch 3: HFlip=YES, Rotate=120°, Brightness=+15%      │
│                                                                    │
│ → Same original image, 50 different augmented versions           │
│                                                                    │
└───────────────────────────────────────────────────────────────────┘
```

### Memory Efficiency

```
TRADITIONAL APPROACH (Not Used):
─────────────────────────────────
Original dataset:     936 images × 400KB    = 374 MB
Augmented (×10):    9,360 images × 400KB    = 3,740 MB (3.7 GB)
Total storage:                              = 4.1 GB
Problem: Massive storage, inflexible, fixed variations

OUR APPROACH (On-the-Fly):
─────────────────────────────────
Original dataset:     936 images × 400KB    = 374 MB
Augmented storage:                          = 0 MB (not saved)
RAM during training:  1 batch × 32 images   = ~25 MB (GPU)
Total storage:                              = 374 MB
Benefits: Minimal storage, infinite variations, flexible
```

---

## Performance & Memory Considerations

### Computational Cost

| Stage | CPU Time | GPU Time | Bottleneck? |
|-------|----------|----------|-------------|
| Load image from disk | 5-10ms | - | Disk I/O |
| Geometric transforms | 2-5ms | - | NO |
| Color transforms | 1-3ms | - | NO |
| Blur/Noise | 1-2ms | - | NO |
| Normalize + ToTensor | 1ms | - | NO |
| **Total per image** | **10-21ms** | - | NO |
| **Total per batch (32)** | **320-672ms** | - | Manageable |
| Model forward pass | - | 50-100ms | MAIN COST |

**Conclusion**: Augmentation overhead is ~3-7× smaller than model forward pass. Not a bottleneck.

### GPU Memory Usage

```
Single image:  (3, 224, 224) × 4 bytes/float32 = 602 KB
Batch of 32:   32 × 602 KB                      = 19.3 MB
Model weights: ResNet18                         = ~45 MB
Gradients:     Same as weights                  = ~45 MB
Optimizer:     Adam (2× parameters)             = ~90 MB
────────────────────────────────────────────────────────
Total GPU memory:                               = ~200 MB

Available on RTX 3060: 12 GB
Headroom: 60× larger than needed ✓
```

### Optimization Strategies

#### 1. Parallel Data Loading (Implemented)
```python
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    num_workers=4,  # 4 CPU threads load/augment in parallel
    pin_memory=True,  # Faster CPU→GPU transfer
)
```

**Effect**:
- Single worker: 320-672ms per batch
- 4 workers: ~100-200ms per batch (3-4× faster)
- GPU stays busy while next batch loads

#### 2. Mixed Precision Training (Optional)
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():  # Use float16 for forward pass
    outputs = model(images)
    loss = criterion(outputs, labels)
```

**Benefits**:
- 2× faster training
- 2× less GPU memory
- No accuracy loss (tested on ImageNet)

#### 3. Gradient Accumulation (For limited GPU memory)
```python
accumulation_steps = 4
effective_batch_size = 32 × 4 = 128

for i, (images, labels, _) in enumerate(train_loader):
    outputs = model(images)
    loss = criterion(outputs, labels) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Use case**: Simulate larger batch size with limited GPU memory

---

## Before/After Examples

### Example 1: Conservative Augmentation

```
ORIGINAL IMAGE: Atok_001.jpg (1080×1080px)
Location: Atok, Benguet
Soil type: Highland volcanic soil
Visual: Reddish-brown, medium texture, moist

AUGMENTED VERSION 1 (Epoch 1, Batch 3):
├─ HorizontalFlip: Applied (50% coin flip → YES)
├─ VerticalFlip: Not applied (50% coin flip → NO)
├─ Rotate: Applied 67° clockwise (70% chance → YES)
├─ RandomResizedCrop: 89% scale, aspect 0.95
├─ ColorJitter: Applied (60% chance → YES)
│  ├─ Brightness: +12%
│  ├─ Contrast: −8%
│  ├─ Saturation: +5%
│  └─ Hue: +0.02
├─ GaussianBlur: Not applied (30% chance → NO)
├─ GaussNoise: Applied, variance=22 (30% chance → YES)
├─ Normalize: ImageNet stats
└─ Output: (3, 224, 224) tensor

VISUAL RESULT:
• Flipped horizontally + rotated 67°
• Slightly brighter and less contrasty
• Grainy appearance from noise
• Focuses on center-left soil region

AUGMENTED VERSION 2 (Epoch 2, Batch 17):
├─ HorizontalFlip: Not applied
├─ VerticalFlip: Applied
├─ Rotate: Applied −123° (70% chance → YES)
├─ RandomResizedCrop: 95% scale
├─ ColorJitter: Not applied (60% chance → NO)
├─ GaussianBlur: Applied, kernel=5 (30% chance → YES)
├─ GaussNoise: Not applied (30% chance → NO)
├─ Normalize: ImageNet stats
└─ Output: (3, 224, 224) tensor

VISUAL RESULT:
• Flipped vertically + rotated −123°
• Original colors preserved
• Slightly blurred (simulates defocus)
• Focuses on top-right soil region

KEY TAKEAWAY: Same original image produces vastly different augmented results
```

### Example 2: Aggressive vs Conservative

```
ORIGINAL IMAGE: Latrinidad_042.jpg
Location: La Trinidad, Benguet
Soil type: Alluvial loam
Visual: Dark brown, fine texture, organic-rich

CONSERVATIVE AUGMENTATION:
├─ Transformations: 7 stages
├─ Color jitter: ±20% brightness, ±5% hue
├─ No shadows, no motion blur, no cutouts
└─ Result: Realistic variation, soil features preserved

AGGRESSIVE AUGMENTATION:
├─ Transformations: 13 stages (6 extra)
├─ Color jitter: ±30% brightness, ±8% hue
├─ RandomShadow: Applied (30% chance)
│  └─ Effect: Dark shadow across bottom-right quadrant
├─ MotionBlur: Applied instead of Gaussian (40% chance)
│  └─ Effect: Directional blur (camera shake simulation)
├─ CoarseDropout: Applied (30% chance)
│  └─ Effect: 2 black rectangles (16×16px, 28×24px)
└─ Result: Extreme variation, some soil features obscured

WHEN TO USE EACH:
───────────────────
Conservative:
• Default for ≥500 labeled images
• Production deployment
• Preserves diagnostic soil features

Aggressive:
• <200 labeled images
• Model underfitting
• Need maximum variety
• Risk: May hurt performance if too extreme
```

---

## Best Practices

### Do's ✓

1. **Use conservative augmentation by default**
   - Proven effective for soil images
   - Preserves color and texture
   - Sufficient variety for most cases

2. **Always use ImageNet normalization**
   - Required for transfer learning
   - Pre-trained models expect these stats
   - Mean: (0.485, 0.456, 0.406), Std: (0.229, 0.224, 0.225)

3. **No augmentation for validation/test**
   - Only resize + normalize
   - Ensures reproducible metrics
   - Fair comparison across experiments

4. **Monitor augmentation effects**
   - Visualize augmented samples: `matplotlib.pyplot.imshow()`
   - Check if soil features still visible
   - Verify colors realistic

5. **Use on-the-fly augmentation**
   - Don't pre-generate and save
   - Saves storage
   - Infinite variations

6. **Enable parallel data loading**
   - `num_workers=4` on Linux/Mac
   - `num_workers=0` on Windows (multiprocessing issues)
   - `pin_memory=True` for faster GPU transfer

### Don'ts ✗

1. **Don't augment validation/test sets**
   - Breaks reproducibility
   - Inflates metrics artificially
   - Makes debugging impossible

2. **Don't use extreme color augmentation**
   - Soil color correlates with NPK/fertility
   - Large hue shifts destroy diagnostic info
   - Keep saturation/hue changes minimal

3. **Don't over-augment**
   - Start conservative, increase if needed
   - Too much augmentation hurts performance
   - Monitor train/val accuracy gap

4. **Don't skip normalization**
   - Transfer learning requires ImageNet stats
   - Model won't converge without proper normalization
   - Always final step before ToTensor

5. **Don't save augmented images to disk**
   - Wastes storage (GB → TB)
   - Fixed variations (defeats purpose)
   - Inflexible (can't change strategy)

6. **Don't use different augmentation each run**
   - Fix `random_state=42` for dataset splits
   - Augmentation randomness is OK (that's the point)
   - Splits must be consistent for fair comparison

### Validation Checklist

Before training:
- [ ] Training set uses conservative/aggressive augmentation
- [ ] Validation set uses minimal augmentation (resize + normalize)
- [ ] Test set uses minimal augmentation (resize + normalize)
- [ ] All sets use same normalization stats (ImageNet)
- [ ] Image size matches model input (224×224 for ResNet18)
- [ ] DataLoader has `num_workers > 0` (if Linux/Mac)
- [ ] DataLoader has `pin_memory=True` (if using GPU)
- [ ] Weighted sampler enabled for imbalanced classes
- [ ] Visualized augmented samples (look realistic)

### Debugging Tips

#### Problem: Model not learning (loss stuck)
```
Possible causes:
1. Wrong normalization stats
   → Check: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

2. Augmentation too aggressive
   → Try: Switch to conservative strategy

3. Learning rate too high
   → Try: Reduce from 1e-3 to 1e-4
```

#### Problem: Overfitting (train acc >> val acc)
```
Possible causes:
1. Not enough augmentation
   → Try: Switch from conservative to aggressive

2. Too few training samples
   → Try: Collect more labeled data

3. Model too large
   → Try: Use ResNet18 instead of EfficientNetV2-S
```

#### Problem: Augmented images look weird
```
Check:
1. Are colors realistic?
   → If not: Reduce ColorJitter parameters

2. Is texture visible?
   → If not: Reduce blur/noise probability

3. Are crops too extreme?
   → If yes: Increase scale range (e.g., 0.8→0.9)
```

---

## Summary

### Augmentation Pipeline Overview

| Aspect | Details |
|--------|---------|
| **Input** | 936 soil images (1080×1080px) |
| **Output** | 5,000-15,000 effective training samples |
| **Method** | On-the-fly augmentation during training |
| **Strategies** | Conservative (default), Aggressive (limited data) |
| **Transforms** | 7-13 stages depending on strategy |
| **Integration** | PyTorch DataLoader with albumentations |
| **Performance** | 10-21ms per image, not a bottleneck |
| **Memory** | ~200 MB GPU for batch size 32 |

### Key Principles

1. **Soil-specific design**: All transforms preserve texture and color
2. **Conservative by default**: Start with realistic augmentation
3. **On-the-fly processing**: Never pre-generate augmented images
4. **No val/test augmentation**: Only resize + normalize
5. **Transfer learning ready**: ImageNet normalization required

### Expected Results

```
With conservative augmentation:
├─ Effective dataset: 5,000-7,000 images
├─ Training epochs: 50-100
├─ Expected accuracy: 75-85% (with 500+ labeled samples)
└─ Generalization: Good (model robust to field variations)

With aggressive augmentation:
├─ Effective dataset: 10,000-15,000 images
├─ Training epochs: 100-150
├─ Expected accuracy: 70-80% (with <200 labeled samples)
└─ Generalization: Variable (may introduce artifacts)
```

---

**Document Version**: 1.0
**Last Updated**: March 2026
**Corresponds to**: `ml_pipeline/data/augmentation.py` v1.0
