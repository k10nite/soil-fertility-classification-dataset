# Soil Fertility Classification Dataset

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A deep learning dataset and training pipeline for automated soil fertility classification using transfer learning. Designed to support precision agriculture in the Philippine Benguet region (Cordillera Administrative Region).

## Overview

This repository provides:
- **1,144 soil images** collected from Benguet municipalities (Atok, La Trinidad)
- **Production-ready data augmentation pipeline** using albumentations
- **PyTorch Dataset and DataLoader** implementations
- **Training/validation/test split utilities** with stratification
- **Documentation** and usage examples

### Problem Statement

Farmers in the Benguet region lack affordable, timely soil fertility data, leading to:
- Inefficient fertilizer use
- Reduced crop yields
- Increased agricultural costs
- Environmental degradation from over-fertilization

### Solution

A machine learning dataset and pipeline that enables:
- Cost-effective soil fertility assessment at scale
- Data-driven fertilizer recommendations
- Reduced environmental impact
- Improved agricultural productivity

## Dataset

### Current Status
- **Total Images**: 1,144 soil images (1080x1080px and 1920x1080px)
- **Total Records**: 1,462 data points with metadata
- **Locations**: Atok and La Trinidad municipalities in Benguet
- **Format**: Organized by location with comprehensive metadata
- **Metadata**: GPS coordinates, altitude, temperature, humidity, crops, barangay

### Dataset Structure
```
organized_images/
├── Atok/                          # Soil images from Atok municipality
├── Latrinidad/                    # Soil images from La Trinidad municipality
└── combined_field_data.csv        # 1,462 rows with metadata
```

### Required Labels (In Progress)
- **Laboratory Analysis**: NPK (Nitrogen, Phosphorus, Potassium) values and pH measurements
- **Ground Truth Labels**: Soil fertility classifications (Low, Medium, High) based on lab results

### CSV Columns
```
uuid, spot_number, shot_number, image_filename, latitude, longitude,
altitude_m, gps_accuracy_m, municipality, barangay, farm_name, crops,
temperature_c, humidity_percent, camera_pitch, camera_roll, camera_heading
```

## Features

- **Transfer Learning Ready**: Works with EfficientNetV2-S and ResNet18 models
- **Advanced Data Augmentation**: Conservative and aggressive pipelines using albumentations
- **Class Imbalance Handling**: Weighted sampling and class weight computation
- **Geospatial Metadata**: GPS coordinates, municipality, barangay, farm location
- **Production-Ready**: Clean, documented, type-hinted Python code
- **Extensible**: Easy integration with fertilizer recommendation systems

## Requirements

### System Requirements
- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)
- 8GB+ RAM
- 20GB+ storage space

### Dependencies

```
torch>=2.0.0
torchvision>=0.15.0
albumentations>=1.3.0
pandas>=1.5.0
numpy>=1.24.0
Pillow>=9.5.0
scikit-learn>=1.2.0
matplotlib>=3.7.0
tqdm>=4.65.0
opencv-python>=4.7.0
```

## Installation

```bash
# 1. Clone repository
git clone git@github.com:k10nite/soil-fertility-classification-dataset.git
cd soil-fertility-classification-dataset

# 2. Create virtual environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on Linux/Mac
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import albumentations; print(f'Albumentations: {albumentations.__version__}')"
```

## Quick Start

```python
from ml_pipeline.data import (
    get_all_transforms,
    create_train_val_test_datasets,
    create_train_loader,
    create_val_loader,
)

# 1. Get transforms
transforms = get_all_transforms(
    image_size=(224, 224),
    augmentation_strategy='conservative'
)

# 2. Create datasets (70/15/15 split)
train_ds, val_ds, test_ds = create_train_val_test_datasets(
    csv_path='organized_images/combined_field_data.csv',
    img_dir='organized_images',
    train_transform=transforms['train'],
    val_transform=transforms['val'],
)

# 3. Create data loaders
train_loader = create_train_loader(train_ds, batch_size=32)
val_loader = create_val_loader(val_ds, batch_size=32)

# 4. Iterate
for images, labels, metadata in train_loader:
    # images: [B, 3, 224, 224]
    # labels: [B]
    # metadata: list of dicts with GPS, crops, etc.
    pass
```

For complete example, see `ml_pipeline/example_usage.py`.

## Data Augmentation

### Conservative Pipeline (Recommended)
Preserves soil texture and color while providing sufficient variety:
- Horizontal/vertical flips
- Rotation ±180°
- Random crop (80-100%)
- Color jitter (brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05)
- Gaussian blur + Gaussian noise

**Effective dataset size**: ~6,000-8,500 images (from 1,144 base images)

### Aggressive Pipeline
Maximum variety for limited training data:
- All conservative transforms
- Random shadows
- Motion blur
- Stronger crops (75-100%)
- Coarse dropout

**Effective dataset size**: ~12,000-17,000 images (from 1,144 base images)

## Usage

### Run Example
```bash
cd ml_pipeline
python example_usage.py
```

### Custom Training
```python
import torch
import torch.nn as nn
from ml_pipeline.data import create_train_loader, create_val_loader

# Get class weights for imbalanced data
class_weights = train_dataset.get_class_weights()
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Training loop
model = YourModel()
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(num_epochs):
    for images, labels, metadata in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

## Project Structure

```
soil-fertility-classification-dataset/
├── ml_pipeline/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── augmentation.py      # Augmentation pipelines
│   │   ├── dataset.py            # PyTorch Dataset
│   │   └── dataloader.py         # DataLoader utilities
│   └── example_usage.py          # Complete working example
├── organized_images/
│   ├── Atok/                     # Soil images from Atok
│   ├── Latrinidad/               # Soil images from La Trinidad
│   └── combined_field_data.csv   # 1,462 rows of metadata
├── .gitignore
├── README.md
└── requirements.txt
```

## Roadmap

### Phase 1: Data Collection (Current)
- [x] Collect 1,144 soil images from Benguet (Atok, La Trinidad)
- [x] Organize images by location
- [x] Create comprehensive metadata CSV (1,462 data points)
- [x] Upload dataset to GitHub with Git LFS
- [ ] Obtain laboratory NPK and pH analysis
- [ ] Create labeled dataset with fertility classifications

### Phase 2: Model Development (Next)
- [ ] Implement EfficientNetV2-S model
- [ ] Implement ResNet18 baseline
- [ ] Setup training pipeline with logging
- [ ] Implement evaluation metrics
- [ ] Hyperparameter tuning

### Phase 3: Deployment
- [ ] Model optimization (quantization, pruning)
- [ ] Create inference API
- [ ] Integration with fertilizer recommendation system
- [ ] Field testing with farmers

## Contributing

Contributions welcome! Areas for contribution:
- Dataset labeling and validation
- Model architecture improvements
- Data augmentation strategies
- Documentation and tutorials
- Field testing and validation

## Research Context

This dataset is part of a graduate thesis in agricultural technology, conducted in collaboration with:
- **DOST (Department of Science and Technology)** - Research funding and support
- **DA (Department of Agriculture)** - Agricultural expertise and validation
- **Benguet Farmers** - Ground truth data and field testing

## Citation

If you use this dataset in your research:

```bibtex
@dataset{soil_fertility_benguet_2026,
  title={Soil Fertility Classification Dataset for Philippine Agriculture},
  author={[Your Name]},
  year={2026},
  publisher={GitHub},
  url={https://github.com/k10nite/soil-fertility-classification-dataset}
}
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- **DOST (Department of Science and Technology)** - Research funding
- **DA (Department of Agriculture)** - Agricultural expertise
- **Benguet Farmers** - Data collection collaboration
- **PyTorch Team** - Deep learning framework
- **albumentations Contributors** - Augmentation library

## Contact

- **Issues**: [GitHub Issues](https://github.com/k10nite/soil-fertility-classification-dataset/issues)
- **Repository**: [https://github.com/k10nite/soil-fertility-classification-dataset](https://github.com/k10nite/soil-fertility-classification-dataset)

---

**Status**: Dataset collection complete. Pending laboratory soil analysis for ground truth labels.

**Last Updated**: March 2026
