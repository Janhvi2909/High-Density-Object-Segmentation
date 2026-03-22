"""
Data preprocessing utilities for High-Density Object Segmentation.

Provides image preprocessing, normalization, and format conversion.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from ..utils.logger import get_logger
from ..utils.config import load_config

logger = get_logger("hdos.data.preprocessing")


def resize_with_aspect(
    image: np.ndarray,
    target_size: Tuple[int, int],
    pad_value: int = 114
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Resize image while maintaining aspect ratio with padding.

    Args:
        image: Input image (H, W, C)
        target_size: Target size (height, width)
        pad_value: Padding value (default: 114 for YOLO)

    Returns:
        Tuple of (resized_image, scale_factor, padding)
    """
    h, w = image.shape[:2]
    target_h, target_w = target_size

    # Calculate scale factor
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    # Resize image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create padded image
    padded = np.full((target_h, target_w, 3), pad_value, dtype=np.uint8)

    # Calculate padding
    pad_w = (target_w - new_w) // 2
    pad_h = (target_h - new_h) // 2

    # Place resized image
    padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

    return padded, scale, (pad_w, pad_h)


def preprocess_image(
    image: Union[np.ndarray, str, Path],
    size: Tuple[int, int] = (640, 640),
    normalize: bool = True,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225)
) -> torch.Tensor:
    """
    Preprocess an image for model inference.

    Args:
        image: Input image (numpy array or path)
        size: Target size (height, width)
        normalize: Whether to normalize with ImageNet stats
        mean: Normalization mean
        std: Normalization std

    Returns:
        Preprocessed image tensor (C, H, W)
    """
    # Load image if path
    if isinstance(image, (str, Path)):
        image = cv2.imread(str(image))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize with aspect ratio
    image, _, _ = resize_with_aspect(image, size)

    # Convert to float and normalize to [0, 1]
    image = image.astype(np.float32) / 255.0

    # Apply ImageNet normalization
    if normalize:
        image = (image - np.array(mean)) / np.array(std)

    # Convert to tensor (C, H, W)
    image = torch.from_numpy(image).permute(2, 0, 1)

    return image


class DataPreprocessor:
    """
    Data preprocessor for batch processing.

    Handles loading, preprocessing, and batching of images and annotations.
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (640, 640),
        normalize: bool = True,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225)
    ):
        """
        Initialize the preprocessor.

        Args:
            image_size: Target image size (height, width)
            normalize: Whether to apply normalization
            mean: Normalization mean values
            std: Normalization std values
        """
        self.image_size = image_size
        self.normalize = normalize
        self.mean = mean
        self.std = std

    def process_image(self, image: Union[np.ndarray, str, Path]) -> Dict:
        """
        Process a single image.

        Args:
            image: Input image or path

        Returns:
            Dictionary with processed image and metadata
        """
        # Load image if path
        if isinstance(image, (str, Path)):
            image_path = Path(image)
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_path = None

        original_size = image.shape[:2]

        # Resize with aspect ratio
        processed, scale, padding = resize_with_aspect(image, self.image_size)

        # Normalize
        processed = processed.astype(np.float32) / 255.0
        if self.normalize:
            processed = (processed - np.array(self.mean)) / np.array(self.std)

        # Convert to tensor
        tensor = torch.from_numpy(processed).permute(2, 0, 1)

        return {
            "image": tensor,
            "original_size": original_size,
            "scale": scale,
            "padding": padding,
            "image_path": str(image_path) if image_path else None
        }

    def process_batch(
        self,
        images: List[Union[np.ndarray, str, Path]]
    ) -> Dict:
        """
        Process a batch of images.

        Args:
            images: List of images or paths

        Returns:
            Dictionary with batched tensors and metadata
        """
        results = [self.process_image(img) for img in images]

        return {
            "images": torch.stack([r["image"] for r in results]),
            "original_sizes": [r["original_size"] for r in results],
            "scales": [r["scale"] for r in results],
            "paddings": [r["padding"] for r in results],
            "image_paths": [r["image_path"] for r in results]
        }

    def denormalize(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Denormalize a tensor back to image.

        Args:
            tensor: Normalized tensor (C, H, W) or (B, C, H, W)

        Returns:
            Numpy image array
        """
        if tensor.dim() == 4:
            tensor = tensor[0]  # Take first image from batch

        # Convert to numpy and transpose
        image = tensor.permute(1, 2, 0).numpy()

        # Denormalize
        if self.normalize:
            image = image * np.array(self.std) + np.array(self.mean)

        # Clip and convert to uint8
        image = np.clip(image * 255, 0, 255).astype(np.uint8)

        return image


class SKU110KDataset(Dataset):
    """
    PyTorch Dataset for SKU-110K.

    Loads images and bounding box annotations for training/evaluation.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str = "train",
        transform=None,
        target_size: Tuple[int, int] = (640, 640)
    ):
        """
        Initialize the dataset.

        Args:
            data_dir: Path to the dataset root
            split: Dataset split ('train', 'val', 'test')
            transform: Optional albumentations transform
            target_size: Target image size
        """
        import json

        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.target_size = target_size

        # Load COCO format annotations
        self.coco_file = self.data_dir / "coco_format" / f"{split}.json"
        if self.coco_file.exists():
            with open(self.coco_file) as f:
                self.coco_data = json.load(f)
            self.images = self.coco_data["images"]
            self._build_annotation_index()
        else:
            logger.warning(f"COCO annotations not found at {self.coco_file}")
            self.images = []
            self.annotations_by_image = {}

        self.images_dir = self.data_dir / "images"

        logger.info(f"Loaded {len(self.images)} images for {split} split")

    def _build_annotation_index(self) -> None:
        """Build index of annotations by image ID."""
        self.annotations_by_image = {}
        for ann in self.coco_data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.annotations_by_image:
                self.annotations_by_image[img_id] = []
            self.annotations_by_image[img_id].append(ann)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary with image and annotations
        """
        # Load image info
        img_info = self.images[idx]
        img_path = self.images_dir / img_info["file_name"]

        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get annotations
        annotations = self.annotations_by_image.get(img_info["id"], [])

        # Extract bounding boxes and labels
        bboxes = []
        labels = []
        for ann in annotations:
            x, y, w, h = ann["bbox"]
            bboxes.append([x, y, x + w, y + h])  # Convert to xyxy format
            labels.append(ann["category_id"])

        bboxes = np.array(bboxes, dtype=np.float32) if bboxes else np.zeros((0, 4))
        labels = np.array(labels, dtype=np.int64) if labels else np.zeros((0,))

        # Apply transforms
        if self.transform:
            transformed = self.transform(
                image=image,
                bboxes=bboxes,
                labels=labels
            )
            image = transformed["image"]
            bboxes = np.array(transformed["bboxes"])
            labels = np.array(transformed["labels"])

        # Resize image
        original_size = image.shape[:2]
        image, scale, padding = resize_with_aspect(image, self.target_size)

        # Adjust bboxes for resize and padding
        if len(bboxes) > 0:
            bboxes = bboxes * scale
            bboxes[:, [0, 2]] += padding[0]
            bboxes[:, [1, 3]] += padding[1]

        # Normalize and convert to tensor
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)

        return {
            "image": image,
            "bboxes": torch.from_numpy(bboxes),
            "labels": torch.from_numpy(labels),
            "image_id": img_info["id"],
            "original_size": original_size,
            "scale": scale,
            "padding": padding,
            "num_objects": len(annotations)
        }


def create_dataloader(
    data_dir: Union[str, Path],
    split: str = "train",
    batch_size: int = 16,
    num_workers: int = 4,
    shuffle: bool = True,
    transform=None
) -> DataLoader:
    """
    Create a DataLoader for the SKU-110K dataset.

    Args:
        data_dir: Path to dataset root
        split: Dataset split
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        transform: Optional transforms

    Returns:
        DataLoader instance
    """
    dataset = SKU110KDataset(
        data_dir=data_dir,
        split=split,
        transform=transform
    )

    def collate_fn(batch):
        """Custom collate function to handle variable number of boxes."""
        images = torch.stack([item["image"] for item in batch])
        return {
            "images": images,
            "bboxes": [item["bboxes"] for item in batch],
            "labels": [item["labels"] for item in batch],
            "image_ids": [item["image_id"] for item in batch],
            "num_objects": [item["num_objects"] for item in batch]
        }

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
