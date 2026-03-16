"""
OPERATION-BASED AUGMENTATION PIPELINE
Independent operation-per-folder architecture for soil fertility dataset

Architecture:
    Input: 2nd_Attempt_20260316_000111/ → 438 images
    ↓
    Each operation creates its own folder:
    - brightness/      (±10%, ±20%, ±30%)
    - contrast/        (±15%, ±30%)
    - rotation/        (90°, 180°, 270°)
    - flip_horizontal/
    - flip_vertical/
    - hue_shift/       (±5°, ±10°, ±15°)
    - saturation/      (±15%, ±30%)
    - clahe/           (clip 2.0, 4.0)
    - gaussian_noise/  (var 10, 20, 30)
    - perspective/     (scale 0.05, 0.1)
    - blur/            (sigma 3, 5)
    - sharpen/         (alpha 0.2, 0.4)
    ↓
    Caching system tracks completion
    ↓
    All outputs: 1920x1080, BLACK background, LANCZOS4 interpolation

Author: Claude
Date: 2026-03-16
"""

import sys
import os
import json
import time
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import defaultdict
import traceback
import hashlib

import cv2
import numpy as np
from tqdm import tqdm
import albumentations as A
from PIL import Image


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('operation_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class OperationConfig:
    """Configuration for a single operation"""
    name: str                    # Operation name (e.g., "bright_+20")
    folder: str                  # Output folder name (e.g., "brightness")
    pipeline: Any                # Albumentations pipeline
    description: str             # Human-readable description


@dataclass
class PipelineSettings:
    """Global pipeline settings"""
    INPUT_DIR: str = "2nd_Attempt_20260316_000111"
    OUTPUT_BASE: str = "augmented_operations"
    TARGET_SIZE: Tuple[int, int] = (1920, 1080)
    BACKGROUND_COLOR: Tuple[int, int, int] = (0, 0, 0)  # BLACK
    INTERPOLATION: int = cv2.INTER_LANCZOS4
    PNG_COMPRESSION: int = 3
    JPEG_QUALITY: int = 95
    MIN_BLUR_SCORE: float = 50.0
    ENABLE_CACHING: bool = True
    PARALLEL_OPERATIONS: bool = False  # Set True for multiprocessing

    def __post_init__(self):
        """Convert paths to Path objects"""
        self.INPUT_DIR = Path(self.INPUT_DIR)
        self.OUTPUT_BASE = Path(self.OUTPUT_BASE)


@dataclass
class OperationResult:
    """Result from processing a single operation"""
    operation_name: str
    folder: str
    success: bool
    processed_count: int
    failed_count: int
    skipped_count: int
    failed_images: List[str]
    blur_scores: List[float]
    execution_time_seconds: float
    error_message: Optional[str] = None


class CacheManager:
    """
    Manages operation caching to enable resume capability

    Cache format:
    {
        "last_run": "2026-03-16T12:00:00",
        "input_dir": "2nd_Attempt_20260316_000111",
        "settings": {...},
        "completed_operations": {
            "brightness": {
                "bright_+10": ["img1.png", "img2.png", ...],
                "bright_-10": ["img1.png", ...]
            }
        },
        "failed_images": {
            "brightness/bright_+10": ["failed1.png"]
        }
    }
    """

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = cache_dir / "operation_cache.json"
        self.cache = self._load_cache()

    def _load_cache(self) -> Dict:
        """Load cache from disk"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}, starting fresh")
                return self._empty_cache()
        else:
            return self._empty_cache()

    def _empty_cache(self) -> Dict:
        """Create empty cache structure"""
        return {
            "last_run": None,
            "input_dir": None,
            "settings": {},
            "completed_operations": defaultdict(dict),
            "failed_images": defaultdict(list)
        }

    def save_cache(self):
        """Persist cache to disk"""
        try:
            # Convert defaultdict to dict for JSON serialization
            cache_copy = dict(self.cache)
            cache_copy['completed_operations'] = dict(cache_copy['completed_operations'])
            cache_copy['failed_images'] = dict(cache_copy['failed_images'])

            with open(self.cache_file, 'w') as f:
                json.dump(cache_copy, f, indent=2)

            logger.debug(f"Cache saved to {self.cache_file}")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    def is_completed(self, folder: str, operation_name: str, image_name: str) -> bool:
        """Check if operation was already completed for an image"""
        if folder not in self.cache['completed_operations']:
            return False

        if operation_name not in self.cache['completed_operations'][folder]:
            return False

        return image_name in self.cache['completed_operations'][folder][operation_name]

    def mark_completed(self, folder: str, operation_name: str, image_name: str):
        """Mark an operation as completed for an image"""
        if folder not in self.cache['completed_operations']:
            self.cache['completed_operations'][folder] = {}

        if operation_name not in self.cache['completed_operations'][folder]:
            self.cache['completed_operations'][folder][operation_name] = []

        if image_name not in self.cache['completed_operations'][folder][operation_name]:
            self.cache['completed_operations'][folder][operation_name].append(image_name)

    def mark_failed(self, folder: str, operation_name: str, image_name: str):
        """Mark an image as failed for an operation"""
        key = f"{folder}/{operation_name}"
        if key not in self.cache['failed_images']:
            self.cache['failed_images'][key] = []

        if image_name not in self.cache['failed_images'][key]:
            self.cache['failed_images'][key].append(image_name)

    def update_metadata(self, input_dir: str, settings: Dict):
        """Update cache metadata"""
        self.cache['last_run'] = datetime.now().isoformat()
        self.cache['input_dir'] = str(input_dir)
        self.cache['settings'] = settings


class ImageProcessor:
    """
    High-quality image processing with transparency handling
    """

    def __init__(self, settings: PipelineSettings):
        self.settings = settings

    def load_and_prepare_image(self, image_path: Path) -> Optional[np.ndarray]:
        """
        Load image and prepare for processing

        Steps:
        1. Load with PIL (best for PNG alpha handling)
        2. Handle transparency → BLACK background
        3. Resize to target resolution (1920x1080)
        4. Convert to RGB numpy array

        Args:
            image_path: Path to input image

        Returns:
            RGB numpy array at target resolution, or None on error
        """
        try:
            with Image.open(image_path) as img:
                # Handle transparency
                if img.mode in ('RGBA', 'LA', 'PA'):
                    logger.debug(f"  [ALPHA] {image_path.name} has transparency, compositing to BLACK")

                    # Create BLACK background
                    background = Image.new('RGB', img.size, self.settings.BACKGROUND_COLOR)

                    # Alpha composite
                    img_rgba = img.convert('RGBA')
                    background.paste(img_rgba, mask=img_rgba.split()[3])  # Use alpha channel as mask
                    img = background
                else:
                    # Convert to RGB
                    img = img.convert('RGB')

                # Resize to target resolution with LANCZOS4 (highest quality)
                if img.size != self.settings.TARGET_SIZE:
                    img = img.resize(self.settings.TARGET_SIZE, Image.Resampling.LANCZOS)
                    logger.debug(f"  [RESIZE] {img.size} → {self.settings.TARGET_SIZE}")

                # Convert to numpy array (RGB format)
                image_array = np.array(img)
                return image_array

        except Exception as e:
            logger.error(f"Error loading {image_path}: {e}")
            return None

    def save_image(self, image: np.ndarray, output_path: Path) -> bool:
        """
        Save image with high quality settings

        Args:
            image: RGB numpy array
            output_path: Output file path

        Returns:
            True if successful
        """
        try:
            # Convert RGB to BGR for OpenCV
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Determine save parameters
            ext = output_path.suffix.lower()

            if ext in ['.jpg', '.jpeg']:
                params = [cv2.IMWRITE_JPEG_QUALITY, self.settings.JPEG_QUALITY]
            elif ext == '.png':
                params = [cv2.IMWRITE_PNG_COMPRESSION, self.settings.PNG_COMPRESSION]
            else:
                params = []

            # Save
            success = cv2.imwrite(str(output_path), image_bgr, params)
            return success

        except Exception as e:
            logger.error(f"Error saving {output_path}: {e}")
            return False

    @staticmethod
    def calculate_blur_score(image: np.ndarray) -> float:
        """Calculate blur score using Laplacian variance"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        return float(variance)


class OperationRegistry:
    """
    Registry of all available augmentation operations

    Each operation is independent and applied to original images
    """

    def __init__(self, settings: PipelineSettings):
        self.settings = settings
        self.operations: List[OperationConfig] = []
        self._register_all_operations()

    def _register_all_operations(self):
        """Register all available operations"""

        # === BRIGHTNESS ===
        self._register_brightness_operations()

        # === CONTRAST ===
        self._register_contrast_operations()

        # === ROTATION ===
        self._register_rotation_operations()

        # === FLIPS ===
        self._register_flip_operations()

        # === HUE SHIFT ===
        self._register_hue_operations()

        # === SATURATION ===
        self._register_saturation_operations()

        # === CLAHE ===
        self._register_clahe_operations()

        # === GAUSSIAN NOISE ===
        self._register_noise_operations()

        # === PERSPECTIVE ===
        self._register_perspective_operations()

        # === BLUR ===
        self._register_blur_operations()

        # === SHARPEN ===
        self._register_sharpen_operations()

        logger.info(f"Registered {len(self.operations)} operations across {len(self.get_folders())} folders")

    def _register_brightness_operations(self):
        """Register brightness adjustment operations"""
        brightness_levels = [
            ('bright_+10', 0.1, 0.1, 'Increase brightness by 10%'),
            ('bright_+20', 0.2, 0.2, 'Increase brightness by 20%'),
            ('bright_+30', 0.3, 0.3, 'Increase brightness by 30%'),
            ('bright_-10', -0.1, -0.1, 'Decrease brightness by 10%'),
            ('bright_-20', -0.2, -0.2, 'Decrease brightness by 20%'),
        ]

        for name, limit_min, limit_max, desc in brightness_levels:
            pipeline = A.Compose([
                A.RandomBrightnessContrast(
                    brightness_limit=(limit_min, limit_max),
                    contrast_limit=0,
                    p=1.0
                )
            ])

            self.operations.append(OperationConfig(
                name=name,
                folder='brightness',
                pipeline=pipeline,
                description=desc
            ))

    def _register_contrast_operations(self):
        """Register contrast adjustment operations"""
        contrast_levels = [
            ('contrast_+15', 0.15, 0.15, 'Increase contrast by 15%'),
            ('contrast_+30', 0.3, 0.3, 'Increase contrast by 30%'),
            ('contrast_-15', -0.15, -0.15, 'Decrease contrast by 15%'),
        ]

        for name, limit_min, limit_max, desc in contrast_levels:
            pipeline = A.Compose([
                A.RandomBrightnessContrast(
                    brightness_limit=0,
                    contrast_limit=(limit_min, limit_max),
                    p=1.0
                )
            ])

            self.operations.append(OperationConfig(
                name=name,
                folder='contrast',
                pipeline=pipeline,
                description=desc
            ))

    def _register_rotation_operations(self):
        """Register rotation operations"""
        rotations = [
            ('rot_90', 90, '90° clockwise rotation'),
            ('rot_180', 180, '180° rotation'),
            ('rot_270', 270, '270° clockwise rotation'),
        ]

        for name, angle, desc in rotations:
            pipeline = A.Compose([
                A.Rotate(
                    limit=(angle, angle),
                    p=1.0,
                    interpolation=self.settings.INTERPOLATION,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=self.settings.BACKGROUND_COLOR
                )
            ])

            self.operations.append(OperationConfig(
                name=name,
                folder='rotation',
                pipeline=pipeline,
                description=desc
            ))

    def _register_flip_operations(self):
        """Register flip operations"""
        # Horizontal flip
        self.operations.append(OperationConfig(
            name='flip_h',
            folder='flip_horizontal',
            pipeline=A.Compose([A.HorizontalFlip(p=1.0)]),
            description='Horizontal flip (mirror)'
        ))

        # Vertical flip
        self.operations.append(OperationConfig(
            name='flip_v',
            folder='flip_vertical',
            pipeline=A.Compose([A.VerticalFlip(p=1.0)]),
            description='Vertical flip'
        ))

    def _register_hue_operations(self):
        """Register hue shift operations"""
        hue_shifts = [
            ('hue_+5', 5, 5, 'Shift hue by +5°'),
            ('hue_+10', 10, 10, 'Shift hue by +10°'),
            ('hue_+15', 15, 15, 'Shift hue by +15°'),
            ('hue_-5', -5, -5, 'Shift hue by -5°'),
            ('hue_-10', -10, -10, 'Shift hue by -10°'),
        ]

        for name, shift_min, shift_max, desc in hue_shifts:
            pipeline = A.Compose([
                A.HueSaturationValue(
                    hue_shift_limit=(shift_min, shift_max),
                    sat_shift_limit=0,
                    val_shift_limit=0,
                    p=1.0
                )
            ])

            self.operations.append(OperationConfig(
                name=name,
                folder='hue_shift',
                pipeline=pipeline,
                description=desc
            ))

    def _register_saturation_operations(self):
        """Register saturation adjustment operations"""
        saturation_levels = [
            ('sat_+15', 15, 15, 'Increase saturation by 15%'),
            ('sat_+30', 30, 30, 'Increase saturation by 30%'),
            ('sat_-15', -15, -15, 'Decrease saturation by 15%'),
        ]

        for name, shift_min, shift_max, desc in saturation_levels:
            pipeline = A.Compose([
                A.HueSaturationValue(
                    hue_shift_limit=0,
                    sat_shift_limit=(shift_min, shift_max),
                    val_shift_limit=0,
                    p=1.0
                )
            ])

            self.operations.append(OperationConfig(
                name=name,
                folder='saturation',
                pipeline=pipeline,
                description=desc
            ))

    def _register_clahe_operations(self):
        """Register CLAHE operations"""
        clahe_configs = [
            ('clahe_2.0', 2.0, (8, 8), 'CLAHE with clip limit 2.0'),
            ('clahe_4.0', 4.0, (8, 8), 'CLAHE with clip limit 4.0'),
        ]

        for name, clip_limit, tile_grid, desc in clahe_configs:
            pipeline = A.Compose([
                A.CLAHE(
                    clip_limit=clip_limit,
                    tile_grid_size=tile_grid,
                    p=1.0
                )
            ])

            self.operations.append(OperationConfig(
                name=name,
                folder='clahe',
                pipeline=pipeline,
                description=desc
            ))

    def _register_noise_operations(self):
        """Register Gaussian noise operations"""
        noise_levels = [
            ('noise_10', (10, 10), 'Gaussian noise (var=10)'),
            ('noise_20', (20, 20), 'Gaussian noise (var=20)'),
            ('noise_30', (30, 30), 'Gaussian noise (var=30)'),
        ]

        for name, var_limit, desc in noise_levels:
            pipeline = A.Compose([
                A.GaussNoise(
                    var_limit=var_limit,
                    p=1.0
                )
            ])

            self.operations.append(OperationConfig(
                name=name,
                folder='gaussian_noise',
                pipeline=pipeline,
                description=desc
            ))

    def _register_perspective_operations(self):
        """Register perspective transform operations"""
        perspective_configs = [
            ('perspective_0.05', (0.05, 0.05), 'Perspective transform (scale=0.05)'),
            ('perspective_0.1', (0.1, 0.1), 'Perspective transform (scale=0.1)'),
        ]

        for name, scale, desc in perspective_configs:
            pipeline = A.Compose([
                A.Perspective(
                    scale=scale,
                    p=1.0,
                    interpolation=self.settings.INTERPOLATION,
                    fit_output=True
                )
            ])

            self.operations.append(OperationConfig(
                name=name,
                folder='perspective',
                pipeline=pipeline,
                description=desc
            ))

    def _register_blur_operations(self):
        """Register blur operations"""
        blur_configs = [
            ('blur_3', (3, 3), 'Gaussian blur (sigma=3)'),
            ('blur_5', (5, 5), 'Gaussian blur (sigma=5)'),
        ]

        for name, blur_limit, desc in blur_configs:
            pipeline = A.Compose([
                A.GaussianBlur(
                    blur_limit=blur_limit,
                    p=1.0
                )
            ])

            self.operations.append(OperationConfig(
                name=name,
                folder='blur',
                pipeline=pipeline,
                description=desc
            ))

    def _register_sharpen_operations(self):
        """Register sharpen operations"""
        sharpen_configs = [
            ('sharpen_0.2', (0.2, 0.2), (0.5, 1.0), 'Sharpen (alpha=0.2)'),
            ('sharpen_0.4', (0.4, 0.4), (0.5, 1.0), 'Sharpen (alpha=0.4)'),
        ]

        for name, alpha, lightness, desc in sharpen_configs:
            pipeline = A.Compose([
                A.Sharpen(
                    alpha=alpha,
                    lightness=lightness,
                    p=1.0
                )
            ])

            self.operations.append(OperationConfig(
                name=name,
                folder='sharpen',
                pipeline=pipeline,
                description=desc
            ))

    def get_folders(self) -> List[str]:
        """Get unique list of output folders"""
        return sorted(list(set(op.folder for op in self.operations)))

    def get_operations_by_folder(self, folder: str) -> List[OperationConfig]:
        """Get all operations for a specific folder"""
        return [op for op in self.operations if op.folder == folder]


class OperationPipeline:
    """
    Main pipeline orchestrator for operation-based augmentation
    """

    def __init__(self, settings: PipelineSettings):
        self.settings = settings
        self.processor = ImageProcessor(settings)
        self.registry = OperationRegistry(settings)
        self.cache = CacheManager(settings.OUTPUT_BASE / ".cache")

        # Create output directories
        settings.OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
        for folder in self.registry.get_folders():
            (settings.OUTPUT_BASE / folder).mkdir(exist_ok=True)

        logger.info("=" * 80)
        logger.info("OPERATION-BASED AUGMENTATION PIPELINE INITIALIZED")
        logger.info("=" * 80)
        logger.info(f"Input directory:  {settings.INPUT_DIR}")
        logger.info(f"Output directory: {settings.OUTPUT_BASE}")
        logger.info(f"Target size:      {settings.TARGET_SIZE}")
        logger.info(f"Background color: RGB{settings.BACKGROUND_COLOR}")
        logger.info(f"Total operations: {len(self.registry.operations)}")
        logger.info(f"Output folders:   {len(self.registry.get_folders())}")
        logger.info("=" * 80)

    def process_operation(
        self,
        operation: OperationConfig,
        image_files: List[Path]
    ) -> OperationResult:
        """
        Process a single operation across all images

        Args:
            operation: Operation configuration
            image_files: List of input image paths

        Returns:
            OperationResult with processing statistics
        """
        logger.info("=" * 80)
        logger.info(f"OPERATION: {operation.folder}/{operation.name}")
        logger.info(f"Description: {operation.description}")
        logger.info("=" * 80)

        start_time = time.time()
        output_dir = self.settings.OUTPUT_BASE / operation.folder

        processed_count = 0
        failed_count = 0
        skipped_count = 0
        failed_images = []
        blur_scores = []

        with tqdm(total=len(image_files), desc=f"{operation.folder}/{operation.name}") as pbar:
            for img_path in image_files:
                # Check cache
                if self.settings.ENABLE_CACHING and self.cache.is_completed(
                    operation.folder, operation.name, img_path.name
                ):
                    skipped_count += 1
                    pbar.update(1)
                    continue

                try:
                    # Load and prepare image
                    image = self.processor.load_and_prepare_image(img_path)
                    if image is None:
                        failed_count += 1
                        failed_images.append(img_path.name)
                        self.cache.mark_failed(operation.folder, operation.name, img_path.name)
                        pbar.update(1)
                        continue

                    # Apply augmentation
                    augmented = operation.pipeline(image=image)
                    augmented_image = augmented['image']

                    # Generate output filename
                    output_name = f"{img_path.stem}_{operation.name}{img_path.suffix}"
                    output_path = output_dir / output_name

                    # Save image
                    success = self.processor.save_image(augmented_image, output_path)

                    if success:
                        # Calculate quality metrics
                        blur_score = self.processor.calculate_blur_score(augmented_image)
                        blur_scores.append(blur_score)

                        # Update cache
                        self.cache.mark_completed(operation.folder, operation.name, img_path.name)
                        processed_count += 1
                    else:
                        failed_count += 1
                        failed_images.append(img_path.name)
                        self.cache.mark_failed(operation.folder, operation.name, img_path.name)

                except Exception as e:
                    logger.error(f"Error processing {img_path.name}: {e}")
                    failed_count += 1
                    failed_images.append(img_path.name)
                    self.cache.mark_failed(operation.folder, operation.name, img_path.name)

                pbar.update(1)

        execution_time = time.time() - start_time

        # Save cache after each operation
        if self.settings.ENABLE_CACHING:
            self.cache.save_cache()

        # Calculate statistics
        avg_blur = np.mean(blur_scores) if blur_scores else 0

        logger.info(f"✓ Operation complete: {processed_count} processed, {failed_count} failed, {skipped_count} skipped")
        logger.info(f"  Execution time: {execution_time:.1f}s")
        logger.info(f"  Avg blur score: {avg_blur:.1f}")

        return OperationResult(
            operation_name=operation.name,
            folder=operation.folder,
            success=True,
            processed_count=processed_count,
            failed_count=failed_count,
            skipped_count=skipped_count,
            failed_images=failed_images,
            blur_scores=blur_scores,
            execution_time_seconds=execution_time
        )

    def run(self) -> Dict[str, List[OperationResult]]:
        """
        Execute all operations

        Returns:
            Dictionary mapping folder names to operation results
        """
        start_time = time.time()

        # Validate input directory
        if not self.settings.INPUT_DIR.exists():
            logger.error(f"Input directory not found: {self.settings.INPUT_DIR}")
            return {}

        # Get input images
        image_files = list(self.settings.INPUT_DIR.glob("*.png")) + \
                     list(self.settings.INPUT_DIR.glob("*.jpg"))

        logger.info(f"Found {len(image_files)} input images")

        if len(image_files) == 0:
            logger.error("No images found in input directory!")
            return {}

        # Update cache metadata
        self.cache.update_metadata(
            str(self.settings.INPUT_DIR),
            {
                'target_size': self.settings.TARGET_SIZE,
                'background_color': self.settings.BACKGROUND_COLOR,
                'operations_count': len(self.registry.operations)
            }
        )

        # Process all operations
        results_by_folder = defaultdict(list)

        for operation in self.registry.operations:
            result = self.process_operation(operation, image_files)
            results_by_folder[operation.folder].append(result)

        total_time = time.time() - start_time

        # Print summary
        self._print_summary(results_by_folder, len(image_files), total_time)

        return dict(results_by_folder)

    def _print_summary(
        self,
        results_by_folder: Dict[str, List[OperationResult]],
        input_count: int,
        total_time: float
    ):
        """Print execution summary"""
        logger.info("\n" + "=" * 80)
        logger.info("=== OPERATION-BASED PIPELINE COMPLETE ===")
        logger.info("=" * 80)
        logger.info(f"Input images:     {input_count}")
        logger.info(f"Operations run:   {sum(len(ops) for ops in results_by_folder.values())}")
        logger.info(f"Output folders:   {len(results_by_folder)}")
        logger.info("")

        total_processed = 0
        total_failed = 0
        total_skipped = 0

        for folder, operations in sorted(results_by_folder.items()):
            folder_processed = sum(op.processed_count for op in operations)
            folder_failed = sum(op.failed_count for op in operations)
            folder_skipped = sum(op.skipped_count for op in operations)

            total_processed += folder_processed
            total_failed += folder_failed
            total_skipped += folder_skipped

            logger.info(
                f"{folder:20s}: {len(operations)} operations, "
                f"{folder_processed} processed, {folder_failed} failed, {folder_skipped} skipped"
            )

        logger.info("")
        logger.info(f"Total processed:  {total_processed}")
        logger.info(f"Total failed:     {total_failed}")
        logger.info(f"Total skipped:    {total_skipped}")
        logger.info(f"Total time:       {total_time / 60:.1f} minutes")
        logger.info("=" * 80)


def main():
    """Main entry point"""
    # Create configuration
    settings = PipelineSettings(
        INPUT_DIR="2nd_Attempt_20260316_000111",
        OUTPUT_BASE="augmented_operations",
        TARGET_SIZE=(1920, 1080),
        BACKGROUND_COLOR=(0, 0, 0),  # BLACK
        INTERPOLATION=cv2.INTER_LANCZOS4,
        ENABLE_CACHING=True
    )

    # Run pipeline
    pipeline = OperationPipeline(settings)
    results = pipeline.run()

    # Exit with appropriate code
    success = all(
        all(op.success for op in ops)
        for ops in results.values()
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
