"""
Soil Fertility Dataset Augmentation - Operation-Based Pipeline
Standalone version - all dependencies included
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Callable, Tuple, Any
import albumentations as A
from albumentations.core.composition import Compose
import logging
from dataclasses import dataclass, field
from tqdm import tqdm


# ============================================================
# Configuration
# ============================================================

@dataclass
class PipelineSettings:
    """Settings for the augmentation pipeline"""
    input_dir: str = ""
    output_dir: str = ""
    preserve_structure: bool = True
    file_suffix: str = "_aug"
    output_format: str = "jpg"
    resize: Tuple[int, int] = None
    quality: int = 95
    max_workers: int = 4

    # Operation settings
    operations: List[str] = field(default_factory=list)
    operation_params: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# Operation Registry
# ============================================================

class OperationRegistry:
    """Registry of available augmentation operations"""

    @staticmethod
    def get_all_operations() -> Dict[str, Dict]:
        """Get all available operations with their metadata"""
        return {
            # Geometric Operations
            "rotate_90": {
                "name": "Rotate 90°",
                "category": "Geometric",
                "description": "Rotate image 90 degrees clockwise",
                "params": {}
            },
            "rotate_180": {
                "name": "Rotate 180°",
                "category": "Geometric",
                "description": "Rotate image 180 degrees",
                "params": {}
            },
            "rotate_270": {
                "name": "Rotate 270°",
                "category": "Geometric",
                "description": "Rotate image 270 degrees clockwise",
                "params": {}
            },
            "flip_horizontal": {
                "name": "Flip Horizontal",
                "category": "Geometric",
                "description": "Flip image horizontally (left-right)",
                "params": {}
            },
            "flip_vertical": {
                "name": "Flip Vertical",
                "category": "Geometric",
                "description": "Flip image vertically (up-down)",
                "params": {}
            },
            "random_rotate": {
                "name": "Random Rotate",
                "category": "Geometric",
                "description": "Rotate image by random angle",
                "params": {"limit": 45}
            },

            # Color Operations
            "brightness": {
                "name": "Brightness Adjustment",
                "category": "Color",
                "description": "Adjust image brightness",
                "params": {"limit": 0.2}
            },
            "contrast": {
                "name": "Contrast Adjustment",
                "category": "Color",
                "description": "Adjust image contrast",
                "params": {"limit": 0.2}
            },
            "hue_saturation": {
                "name": "Hue/Saturation",
                "category": "Color",
                "description": "Adjust hue and saturation",
                "params": {"hue_shift_limit": 20, "sat_shift_limit": 30}
            },
            "rgb_shift": {
                "name": "RGB Shift",
                "category": "Color",
                "description": "Randomly shift RGB channels",
                "params": {"r_shift_limit": 20, "g_shift_limit": 20, "b_shift_limit": 20}
            },
            "grayscale": {
                "name": "Grayscale",
                "category": "Color",
                "description": "Convert to grayscale",
                "params": {}
            },

            # Noise Operations
            "gaussian_noise": {
                "name": "Gaussian Noise",
                "category": "Noise",
                "description": "Add Gaussian noise",
                "params": {"var_limit": (10.0, 50.0)}
            },
            "blur": {
                "name": "Blur",
                "category": "Noise",
                "description": "Apply blur effect",
                "params": {"blur_limit": 7}
            },
            "motion_blur": {
                "name": "Motion Blur",
                "category": "Noise",
                "description": "Apply motion blur",
                "params": {"blur_limit": 7}
            },
            "gaussian_blur": {
                "name": "Gaussian Blur",
                "category": "Noise",
                "description": "Apply Gaussian blur",
                "params": {"blur_limit": (3, 7)}
            },

            # Enhancement Operations
            "sharpen": {
                "name": "Sharpen",
                "category": "Enhancement",
                "description": "Sharpen the image",
                "params": {"alpha": (0.2, 0.5)}
            },
            "clahe": {
                "name": "CLAHE",
                "category": "Enhancement",
                "description": "Contrast Limited Adaptive Histogram Equalization",
                "params": {"clip_limit": 4.0}
            },
            "equalize": {
                "name": "Histogram Equalize",
                "category": "Enhancement",
                "description": "Equalize histogram",
                "params": {}
            },
        }

    @staticmethod
    def get_operation_transform(operation_name: str, params: Dict = None) -> Callable:
        """Get the albumentations transform for an operation"""
        params = params or {}

        # Geometric transforms
        if operation_name == "rotate_90":
            return lambda img: cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif operation_name == "rotate_180":
            return lambda img: cv2.rotate(img, cv2.ROTATE_180)
        elif operation_name == "rotate_270":
            return lambda img: cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif operation_name == "flip_horizontal":
            return lambda img: cv2.flip(img, 1)
        elif operation_name == "flip_vertical":
            return lambda img: cv2.flip(img, 0)
        elif operation_name == "random_rotate":
            limit = params.get("limit", 45)
            return A.Rotate(limit=limit, p=1.0)

        # Color transforms
        elif operation_name == "brightness":
            limit = params.get("limit", 0.2)
            return A.RandomBrightness(limit=limit, p=1.0)
        elif operation_name == "contrast":
            limit = params.get("limit", 0.2)
            return A.RandomContrast(limit=limit, p=1.0)
        elif operation_name == "hue_saturation":
            hue = params.get("hue_shift_limit", 20)
            sat = params.get("sat_shift_limit", 30)
            return A.HueSaturationValue(hue_shift_limit=hue, sat_shift_limit=sat, p=1.0)
        elif operation_name == "rgb_shift":
            r = params.get("r_shift_limit", 20)
            g = params.get("g_shift_limit", 20)
            b = params.get("b_shift_limit", 20)
            return A.RGBShift(r_shift_limit=r, g_shift_limit=g, b_shift_limit=b, p=1.0)
        elif operation_name == "grayscale":
            return A.ToGray(p=1.0)

        # Noise transforms
        elif operation_name == "gaussian_noise":
            var_limit = params.get("var_limit", (10.0, 50.0))
            return A.GaussNoise(var_limit=var_limit, p=1.0)
        elif operation_name == "blur":
            blur_limit = params.get("blur_limit", 7)
            return A.Blur(blur_limit=blur_limit, p=1.0)
        elif operation_name == "motion_blur":
            blur_limit = params.get("blur_limit", 7)
            return A.MotionBlur(blur_limit=blur_limit, p=1.0)
        elif operation_name == "gaussian_blur":
            blur_limit = params.get("blur_limit", (3, 7))
            return A.GaussianBlur(blur_limit=blur_limit, p=1.0)

        # Enhancement transforms
        elif operation_name == "sharpen":
            alpha = params.get("alpha", (0.2, 0.5))
            return A.Sharpen(alpha=alpha, p=1.0)
        elif operation_name == "clahe":
            clip_limit = params.get("clip_limit", 4.0)
            return A.CLAHE(clip_limit=clip_limit, p=1.0)
        elif operation_name == "equalize":
            return A.Equalize(p=1.0)

        else:
            raise ValueError(f"Unknown operation: {operation_name}")


# ============================================================
# Image Processor
# ============================================================

class ImageProcessor:
    """Processes individual images with selected operations"""

    def __init__(self, operations: List[str], operation_params: Dict[str, Any] = None):
        self.operations = operations
        self.operation_params = operation_params or {}
        self.logger = logging.getLogger(__name__)

    def process_image(self, image_path: str) -> np.ndarray:
        """Load and process a single image"""
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # Convert BGR to RGB for albumentations
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Apply operations
        for op_name in self.operations:
            params = self.operation_params.get(op_name, {})
            transform = OperationRegistry.get_operation_transform(op_name, params)

            # Handle both callable and Albumentations transforms
            if callable(transform) and not isinstance(transform, A.BasicTransform):
                # Direct callable (like cv2.rotate)
                img = transform(img)
            else:
                # Albumentations transform
                augmented = transform(image=img)
                img = augmented['image']

        # Convert back to BGR for saving
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def save_image(self, image: np.ndarray, output_path: str, quality: int = 95):
        """Save processed image"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save with appropriate format
        if output_path.suffix.lower() in ['.jpg', '.jpeg']:
            cv2.imwrite(str(output_path), image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        elif output_path.suffix.lower() == '.png':
            cv2.imwrite(str(output_path), image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        else:
            cv2.imwrite(str(output_path), image)


# ============================================================
# Pipeline
# ============================================================

class OperationPipeline:
    """Main augmentation pipeline"""

    def __init__(self, settings: PipelineSettings):
        self.settings = settings
        self.processor = ImageProcessor(settings.operations, settings.operation_params)
        self.logger = logging.getLogger(__name__)

    def get_image_files(self) -> List[Path]:
        """Get all image files from input directory"""
        input_path = Path(self.settings.input_dir)
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

        image_files = []
        if self.settings.preserve_structure:
            # Recursively find all images
            for ext in extensions:
                image_files.extend(input_path.rglob(f'*{ext}'))
                image_files.extend(input_path.rglob(f'*{ext.upper()}'))
        else:
            # Only top-level images
            for ext in extensions:
                image_files.extend(input_path.glob(f'*{ext}'))
                image_files.extend(input_path.glob(f'*{ext.upper()}'))

        return sorted(image_files)

    def process_file(self, image_path: Path) -> Tuple[bool, str]:
        """Process a single file"""
        try:
            # Generate output path
            if self.settings.preserve_structure:
                rel_path = image_path.relative_to(self.settings.input_dir)
                output_path = Path(self.settings.output_dir) / rel_path.parent / f"{rel_path.stem}{self.settings.file_suffix}{rel_path.suffix}"
            else:
                output_path = Path(self.settings.output_dir) / f"{image_path.stem}{self.settings.file_suffix}{image_path.suffix}"

            # Change extension if needed
            if self.settings.output_format:
                output_path = output_path.with_suffix(f'.{self.settings.output_format}')

            # Process image
            processed_img = self.processor.process_image(str(image_path))

            # Resize if needed
            if self.settings.resize:
                processed_img = cv2.resize(processed_img, self.settings.resize)

            # Save
            self.processor.save_image(processed_img, str(output_path), self.settings.quality)

            return True, str(output_path)

        except Exception as e:
            self.logger.error(f"Error processing {image_path}: {e}")
            return False, str(e)

    def run(self, progress_callback: Callable = None) -> Dict[str, Any]:
        """Run the augmentation pipeline"""
        self.logger.info("Starting augmentation pipeline")
        self.logger.info(f"Input: {self.settings.input_dir}")
        self.logger.info(f"Output: {self.settings.output_dir}")
        self.logger.info(f"Operations: {', '.join(self.settings.operations)}")

        # Get all image files
        image_files = self.get_image_files()
        total_files = len(image_files)

        if total_files == 0:
            self.logger.warning("No image files found!")
            return {
                'success': False,
                'total': 0,
                'processed': 0,
                'failed': 0,
                'message': 'No image files found'
            }

        self.logger.info(f"Found {total_files} images to process")

        # Process files
        processed = 0
        failed = 0

        for idx, image_path in enumerate(image_files):
            success, msg = self.process_file(image_path)

            if success:
                processed += 1
            else:
                failed += 1

            # Progress callback
            if progress_callback:
                progress_callback(idx + 1, total_files, image_path.name)

        # Summary
        result = {
            'success': True,
            'total': total_files,
            'processed': processed,
            'failed': failed,
            'message': f'Successfully processed {processed}/{total_files} images'
        }

        self.logger.info(f"Pipeline complete: {processed} processed, {failed} failed")
        return result


# ============================================================
# Utility Functions
# ============================================================

def setup_logging(log_file: str = None, level: int = logging.INFO):
    """Set up logging configuration"""
    handlers = []

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    handlers.append(console_handler)

    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)

    # Configure root logger
    logging.basicConfig(
        level=level,
        handlers=handlers
    )
