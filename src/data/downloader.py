"""
SKU-110K Dataset Downloader.

Downloads and extracts the SKU-110K dataset for high-density object detection.
"""

import os
import zipfile
import tarfile
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve
from tqdm import tqdm

from ..utils.logger import get_logger

logger = get_logger("hdos.data.downloader")


class DownloadProgressBar:
    """Progress bar for downloads."""

    def __init__(self):
        self.pbar = None

    def __call__(self, block_num: int, block_size: int, total_size: int):
        if self.pbar is None:
            self.pbar = tqdm(total=total_size, unit='B', unit_scale=True)
        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(block_size)
        else:
            self.pbar.close()


class SKU110KDownloader:
    """
    Downloader for the SKU-110K dataset.

    The SKU-110K dataset contains ~11K images of retail store shelves
    with extremely dense object annotations (average 150 objects/image).

    Reference:
        Goldman et al. "Precise Detection in Densely Packed Scenes" (CVPR 2019)
        https://github.com/eg4000/SKU110K_CVPR19
    """

    # Dataset URLs (Google Drive links - may need manual download)
    URLS = {
        "images": "https://drive.google.com/uc?id=1iq93lCdhaPUN0fWbLieMtzfB1850pKwd",
        "annotations": "https://drive.google.com/uc?id=1V4sDbAICZkKvR8RI6mKdqy8Y2yFHlLGg",
    }

    DATASET_INFO = {
        "name": "SKU-110K",
        "num_images": 11762,
        "num_objects": 1739777,
        "avg_objects_per_image": 147.9,
        "splits": {
            "train": 8219,
            "val": 588,
            "test": 2936
        }
    }

    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize the downloader.

        Args:
            data_dir: Directory to store the dataset
        """
        if data_dir is None:
            data_dir = Path(__file__).parent.parent.parent / "data" / "raw"
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download(self, use_gdown: bool = True) -> None:
        """
        Download the SKU-110K dataset.

        Args:
            use_gdown: Use gdown for Google Drive downloads (recommended)
        """
        logger.info("Starting SKU-110K dataset download...")
        logger.info(f"Dataset will be saved to: {self.data_dir}")

        if use_gdown:
            self._download_with_gdown()
        else:
            self._download_manual_instructions()

    def _download_with_gdown(self) -> None:
        """Download using gdown library."""
        try:
            import gdown
        except ImportError:
            logger.error("gdown not installed. Run: pip install gdown")
            self._download_manual_instructions()
            return

        # Download images
        images_zip = self.data_dir / "SKU110K_images.zip"
        if not images_zip.exists():
            logger.info("Downloading images...")
            gdown.download(self.URLS["images"], str(images_zip), quiet=False)

        # Download annotations
        annotations_zip = self.data_dir / "SKU110K_annotations.zip"
        if not annotations_zip.exists():
            logger.info("Downloading annotations...")
            gdown.download(self.URLS["annotations"], str(annotations_zip), quiet=False)

        # Extract
        self._extract_dataset()

    def _download_manual_instructions(self) -> None:
        """Print manual download instructions."""
        logger.info("\n" + "="*60)
        logger.info("MANUAL DOWNLOAD INSTRUCTIONS")
        logger.info("="*60)
        logger.info("\nThe SKU-110K dataset requires manual download from Google Drive.")
        logger.info("\n1. Visit: https://github.com/eg4000/SKU110K_CVPR19")
        logger.info("2. Download the images and annotations from the provided links")
        logger.info(f"3. Place the zip files in: {self.data_dir}")
        logger.info("4. Run this script again to extract the files")
        logger.info("\nAlternatively, use Kaggle:")
        logger.info("   kaggle datasets download -d thedatasith/sku110k-annotations")
        logger.info("="*60 + "\n")

    def _extract_dataset(self) -> None:
        """Extract downloaded zip files."""
        images_zip = self.data_dir / "SKU110K_images.zip"
        annotations_zip = self.data_dir / "SKU110K_annotations.zip"

        # Extract images
        if images_zip.exists():
            logger.info("Extracting images...")
            with zipfile.ZipFile(images_zip, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
            logger.info("Images extracted successfully")

        # Extract annotations
        if annotations_zip.exists():
            logger.info("Extracting annotations...")
            with zipfile.ZipFile(annotations_zip, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
            logger.info("Annotations extracted successfully")

    def verify_dataset(self) -> bool:
        """
        Verify that the dataset is properly downloaded and extracted.

        Returns:
            True if dataset is valid, False otherwise
        """
        images_dir = self.data_dir / "images"
        annotations_dir = self.data_dir / "annotations"

        # Check directories exist
        if not images_dir.exists() or not annotations_dir.exists():
            logger.warning("Dataset directories not found")
            return False

        # Count images
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        num_images = len(image_files)

        logger.info(f"Found {num_images} images")

        if num_images < 1000:
            logger.warning(f"Expected ~11K images, found only {num_images}")
            return False

        logger.info("Dataset verified successfully")
        return True

    def get_split_info(self) -> dict:
        """Get information about train/val/test splits."""
        return self.DATASET_INFO["splits"]

    def create_coco_format(self) -> None:
        """Convert SKU-110K annotations to COCO format for easier use."""
        logger.info("Converting annotations to COCO format...")

        import csv
        import json
        from datetime import datetime

        annotations_dir = self.data_dir / "annotations"
        output_dir = self.data_dir / "coco_format"
        output_dir.mkdir(exist_ok=True)

        for split in ["train", "val", "test"]:
            csv_file = annotations_dir / f"annotations_{split}.csv"
            if not csv_file.exists():
                logger.warning(f"Annotation file not found: {csv_file}")
                continue

            coco_dict = {
                "info": {
                    "description": f"SKU-110K {split} set",
                    "date_created": datetime.now().isoformat()
                },
                "licenses": [],
                "images": [],
                "annotations": [],
                "categories": [{"id": 1, "name": "product", "supercategory": "object"}]
            }

            image_id_map = {}
            annotation_id = 1

            with open(csv_file, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header

                for row in reader:
                    image_name = row[0]
                    x1, y1, x2, y2 = map(float, row[1:5])
                    width, height = int(row[6]), int(row[7])

                    # Add image if not seen
                    if image_name not in image_id_map:
                        image_id = len(image_id_map) + 1
                        image_id_map[image_name] = image_id
                        coco_dict["images"].append({
                            "id": image_id,
                            "file_name": image_name,
                            "width": width,
                            "height": height
                        })

                    # Add annotation
                    bbox_width = x2 - x1
                    bbox_height = y2 - y1
                    coco_dict["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id_map[image_name],
                        "category_id": 1,
                        "bbox": [x1, y1, bbox_width, bbox_height],
                        "area": bbox_width * bbox_height,
                        "iscrowd": 0
                    })
                    annotation_id += 1

            # Save COCO format
            output_file = output_dir / f"{split}.json"
            with open(output_file, 'w') as f:
                json.dump(coco_dict, f)

            logger.info(f"Created {output_file} with {len(coco_dict['images'])} images")


def download_dataset(
    dataset: str = "sku110k",
    output_dir: Optional[Path] = None
) -> None:
    """
    Download a dataset by name.

    Args:
        dataset: Dataset name ('sku110k')
        output_dir: Output directory for the dataset
    """
    if dataset.lower() == "sku110k":
        downloader = SKU110KDownloader(output_dir)
        downloader.download()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
