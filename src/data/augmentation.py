"""
Data augmentation pipelines for High-Density Object Segmentation.

Provides comprehensive augmentation using albumentations library.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
import albumentations as A
from albumentations.pytorch import ToTensorV2

from ..utils.logger import get_logger

logger = get_logger("hdos.data.augmentation")


def get_train_transforms(
    image_size: Tuple[int, int] = (640, 640),
    augment_level: str = "medium"
) -> A.Compose:
    """
    Get training augmentation transforms.

    Args:
        image_size: Target image size (height, width)
        augment_level: Augmentation level ('light', 'medium', 'heavy')

    Returns:
        Albumentations Compose object
    """
    height, width = image_size

    # Base transforms
    base_transforms = [
        A.LongestMaxSize(max_size=max(height, width)),
        A.PadIfNeeded(
            min_height=height,
            min_width=width,
            border_mode=cv2.BORDER_CONSTANT,
            value=(114, 114, 114)
        ),
    ]

    if augment_level == "light":
        aug_transforms = [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.3
            ),
        ]
    elif augment_level == "medium":
        aug_transforms = [
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=10,
                border_mode=cv2.BORDER_CONSTANT,
                value=(114, 114, 114),
                p=0.5
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=20,
                p=0.3
            ),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        ]
    else:  # heavy
        aug_transforms = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.ShiftScaleRotate(
                shift_limit=0.15,
                scale_limit=0.3,
                rotate_limit=15,
                border_mode=cv2.BORDER_CONSTANT,
                value=(114, 114, 114),
                p=0.7
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.6
            ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=30,
                p=0.5
            ),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 80.0)),
                A.ISONoise(),
                A.MultiplicativeNoise(),
            ], p=0.3),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7)),
                A.MotionBlur(blur_limit=7),
                A.MedianBlur(blur_limit=5),
            ], p=0.3),
            A.OneOf([
                A.CLAHE(clip_limit=4.0),
                A.Sharpen(),
                A.Emboss(),
            ], p=0.2),
            A.RandomShadow(p=0.2),
            A.CoarseDropout(
                max_holes=8,
                max_height=32,
                max_width=32,
                fill_value=114,
                p=0.2
            ),
        ]

    # Combine all transforms
    transforms = base_transforms + aug_transforms

    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format='pascal_voc',  # [x_min, y_min, x_max, y_max]
            label_fields=['labels'],
            min_visibility=0.3
        )
    )


def get_val_transforms(
    image_size: Tuple[int, int] = (640, 640)
) -> A.Compose:
    """
    Get validation/test transforms (no augmentation).

    Args:
        image_size: Target image size (height, width)

    Returns:
        Albumentations Compose object
    """
    height, width = image_size

    return A.Compose(
        [
            A.LongestMaxSize(max_size=max(height, width)),
            A.PadIfNeeded(
                min_height=height,
                min_width=width,
                border_mode=cv2.BORDER_CONSTANT,
                value=(114, 114, 114)
            ),
        ],
        bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['labels']
        )
    )


class AugmentationPipeline:
    """
    Advanced augmentation pipeline with mosaic and mixup support.

    Designed for high-density object detection where preserving
    all objects and their relationships is critical.
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (640, 640),
        mosaic_prob: float = 0.5,
        mixup_prob: float = 0.1,
        copy_paste_prob: float = 0.0
    ):
        """
        Initialize the augmentation pipeline.

        Args:
            image_size: Target image size
            mosaic_prob: Probability of applying mosaic augmentation
            mixup_prob: Probability of applying mixup augmentation
            copy_paste_prob: Probability of copy-paste augmentation
        """
        self.image_size = image_size
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob
        self.copy_paste_prob = copy_paste_prob

        self.basic_transform = get_train_transforms(image_size, "medium")
        self.val_transform = get_val_transforms(image_size)

    def mosaic_augmentation(
        self,
        images: List[np.ndarray],
        bboxes_list: List[np.ndarray],
        labels_list: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply 4-image mosaic augmentation.

        Creates a 2x2 grid of images, commonly used in YOLO training.

        Args:
            images: List of 4 images
            bboxes_list: List of 4 bbox arrays
            labels_list: List of 4 label arrays

        Returns:
            Tuple of (mosaic_image, combined_bboxes, combined_labels)
        """
        assert len(images) == 4, "Mosaic requires exactly 4 images"

        h, w = self.image_size
        mosaic_image = np.full((h * 2, w * 2, 3), 114, dtype=np.uint8)

        # Random center point
        cx, cy = np.random.randint(w // 2, w * 3 // 2), np.random.randint(h // 2, h * 3 // 2)

        all_bboxes = []
        all_labels = []

        # Place 4 images
        positions = [
            (0, 0, cx, cy),           # top-left
            (cx, 0, w * 2, cy),       # top-right
            (0, cy, cx, h * 2),       # bottom-left
            (cx, cy, w * 2, h * 2)    # bottom-right
        ]

        for i, (img, bboxes, labels) in enumerate(zip(images, bboxes_list, labels_list)):
            x1, y1, x2, y2 = positions[i]

            # Resize image to fit the allocated space
            region_w, region_h = x2 - x1, y2 - y1
            img_h, img_w = img.shape[:2]

            scale = min(region_w / img_w, region_h / img_h)
            new_w, new_h = int(img_w * scale), int(img_h * scale)

            resized = cv2.resize(img, (new_w, new_h))

            # Place in mosaic
            offset_x = x1 + (region_w - new_w) // 2
            offset_y = y1 + (region_h - new_h) // 2

            mosaic_image[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized

            # Adjust bboxes
            if len(bboxes) > 0:
                adjusted_bboxes = bboxes.copy()
                adjusted_bboxes[:, [0, 2]] = adjusted_bboxes[:, [0, 2]] * scale + offset_x
                adjusted_bboxes[:, [1, 3]] = adjusted_bboxes[:, [1, 3]] * scale + offset_y

                all_bboxes.append(adjusted_bboxes)
                all_labels.append(labels)

        # Combine all bboxes and labels
        combined_bboxes = np.concatenate(all_bboxes) if all_bboxes else np.zeros((0, 4))
        combined_labels = np.concatenate(all_labels) if all_labels else np.zeros((0,))

        # Crop to final size
        crop_x = (w * 2 - w) // 2
        crop_y = (h * 2 - h) // 2
        final_image = mosaic_image[crop_y:crop_y + h, crop_x:crop_x + w]

        # Adjust bboxes for crop
        if len(combined_bboxes) > 0:
            combined_bboxes[:, [0, 2]] -= crop_x
            combined_bboxes[:, [1, 3]] -= crop_y

            # Clip to image bounds
            combined_bboxes[:, [0, 2]] = np.clip(combined_bboxes[:, [0, 2]], 0, w)
            combined_bboxes[:, [1, 3]] = np.clip(combined_bboxes[:, [1, 3]], 0, h)

            # Filter out invalid boxes
            valid = (combined_bboxes[:, 2] > combined_bboxes[:, 0]) & \
                    (combined_bboxes[:, 3] > combined_bboxes[:, 1])
            combined_bboxes = combined_bboxes[valid]
            combined_labels = combined_labels[valid]

        return final_image, combined_bboxes, combined_labels.astype(np.int64)

    def mixup_augmentation(
        self,
        image1: np.ndarray,
        bboxes1: np.ndarray,
        labels1: np.ndarray,
        image2: np.ndarray,
        bboxes2: np.ndarray,
        labels2: np.ndarray,
        alpha: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply mixup augmentation.

        Blends two images and concatenates their annotations.

        Args:
            image1: First image
            bboxes1: First image bboxes
            labels1: First image labels
            image2: Second image
            bboxes2: Second image bboxes
            labels2: Second image labels
            alpha: Mixup ratio

        Returns:
            Tuple of (mixed_image, combined_bboxes, combined_labels)
        """
        # Ensure same size
        if image1.shape != image2.shape:
            image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

        # Mix images
        mixed = cv2.addWeighted(image1, alpha, image2, 1 - alpha, 0)

        # Combine annotations
        combined_bboxes = np.concatenate([bboxes1, bboxes2]) if len(bboxes1) > 0 or len(bboxes2) > 0 else np.zeros((0, 4))
        combined_labels = np.concatenate([labels1, labels2]) if len(labels1) > 0 or len(labels2) > 0 else np.zeros((0,))

        return mixed, combined_bboxes, combined_labels.astype(np.int64)

    def __call__(
        self,
        image: np.ndarray,
        bboxes: np.ndarray,
        labels: np.ndarray,
        is_training: bool = True
    ) -> Dict:
        """
        Apply augmentation pipeline.

        Args:
            image: Input image
            bboxes: Bounding boxes (N, 4) in xyxy format
            labels: Object labels (N,)
            is_training: Whether in training mode

        Returns:
            Dictionary with augmented image, bboxes, and labels
        """
        if not is_training:
            result = self.val_transform(
                image=image,
                bboxes=bboxes.tolist() if len(bboxes) > 0 else [],
                labels=labels.tolist() if len(labels) > 0 else []
            )
        else:
            result = self.basic_transform(
                image=image,
                bboxes=bboxes.tolist() if len(bboxes) > 0 else [],
                labels=labels.tolist() if len(labels) > 0 else []
            )

        return {
            "image": result["image"],
            "bboxes": np.array(result["bboxes"]) if result["bboxes"] else np.zeros((0, 4)),
            "labels": np.array(result["labels"]) if result["labels"] else np.zeros((0,))
        }


class DensityAwareAugmentation:
    """
    Density-aware augmentation that applies different strategies
    based on object density in the image.

    For high-density images, uses more conservative augmentations
    to preserve object relationships.
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (640, 640),
        density_threshold: int = 50
    ):
        """
        Initialize density-aware augmentation.

        Args:
            image_size: Target image size
            density_threshold: Object count threshold for high density
        """
        self.image_size = image_size
        self.density_threshold = density_threshold

        # Light augmentation for high-density images
        self.high_density_transform = get_train_transforms(image_size, "light")

        # Heavy augmentation for low-density images
        self.low_density_transform = get_train_transforms(image_size, "heavy")

    def __call__(
        self,
        image: np.ndarray,
        bboxes: np.ndarray,
        labels: np.ndarray
    ) -> Dict:
        """
        Apply density-aware augmentation.

        Args:
            image: Input image
            bboxes: Bounding boxes
            labels: Object labels

        Returns:
            Augmented data dictionary
        """
        num_objects = len(bboxes)

        if num_objects > self.density_threshold:
            # High density - conservative augmentation
            transform = self.high_density_transform
        else:
            # Low density - aggressive augmentation
            transform = self.low_density_transform

        result = transform(
            image=image,
            bboxes=bboxes.tolist() if len(bboxes) > 0 else [],
            labels=labels.tolist() if len(labels) > 0 else []
        )

        return {
            "image": result["image"],
            "bboxes": np.array(result["bboxes"]) if result["bboxes"] else np.zeros((0, 4)),
            "labels": np.array(result["labels"]) if result["labels"] else np.zeros((0,))
        }
