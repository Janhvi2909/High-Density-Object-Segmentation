"""
Segment Anything Model (SAM) adapter.

Zero-shot segmentation using Meta's Segment Anything.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

try:
    from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False


class SAMAdapter:
    """
    Adapter for Segment Anything Model.

    Provides zero-shot segmentation and prompt-based refinement.
    """

    MODEL_URLS = {
        'vit_h': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
        'vit_l': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth',
        'vit_b': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth'
    }

    def __init__(
        self,
        model_type: str = 'vit_b',
        checkpoint_path: Optional[str] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize SAM adapter.

        Args:
            model_type: Model type ('vit_h', 'vit_l', 'vit_b')
            checkpoint_path: Path to checkpoint
            device: Device to use
        """
        if not SAM_AVAILABLE:
            raise ImportError("segment-anything is required. Install with: pip install segment-anything")

        self.device = device
        self.model_type = model_type

        # Load model
        if checkpoint_path is None:
            checkpoint_path = self._download_checkpoint(model_type)

        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam.to(device)

        # Create predictor and mask generator
        self.predictor = SamPredictor(self.sam)
        self.mask_generator = None  # Lazy initialization

    def _download_checkpoint(self, model_type: str) -> str:
        """Download checkpoint if not available."""
        import urllib.request
        import os

        checkpoint_dir = Path.home() / '.cache' / 'sam'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / f'sam_{model_type}.pth'

        if not checkpoint_path.exists():
            print(f"Downloading SAM {model_type} checkpoint...")
            url = self.MODEL_URLS[model_type]
            urllib.request.urlretrieve(url, checkpoint_path)
            print("Download complete.")

        return str(checkpoint_path)

    def set_image(self, image: np.ndarray) -> None:
        """
        Set image for prediction.

        Args:
            image: RGB image (H, W, 3)
        """
        self.predictor.set_image(image)

    def predict_with_points(
        self,
        points: np.ndarray,
        labels: np.ndarray,
        multimask_output: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Predict masks using point prompts.

        Args:
            points: Point coordinates (N, 2)
            labels: Point labels (N,) - 1 for foreground, 0 for background
            multimask_output: Return multiple masks

        Returns:
            Dictionary with masks, scores, and logits
        """
        masks, scores, logits = self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=multimask_output
        )

        return {
            'masks': masks,
            'scores': scores,
            'logits': logits
        }

    def predict_with_box(
        self,
        box: np.ndarray,
        multimask_output: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Predict mask using box prompt.

        Args:
            box: Box coordinates [x1, y1, x2, y2]
            multimask_output: Return multiple masks

        Returns:
            Dictionary with masks, scores, and logits
        """
        masks, scores, logits = self.predictor.predict(
            box=box,
            multimask_output=multimask_output
        )

        return {
            'masks': masks,
            'scores': scores,
            'logits': logits
        }

    def refine_mask(
        self,
        initial_mask: np.ndarray,
        iterations: int = 1
    ) -> np.ndarray:
        """
        Refine an initial mask using SAM.

        Args:
            initial_mask: Initial binary mask
            iterations: Number of refinement iterations

        Returns:
            Refined mask
        """
        mask = initial_mask.copy()

        for _ in range(iterations):
            # Find contour points
            import cv2
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if not contours:
                break

            # Sample points from contour
            contour = max(contours, key=cv2.contourArea)
            points = contour.reshape(-1, 2)

            # Sample foreground and background points
            fg_points = points[::max(1, len(points) // 10)]

            # Background points from outside mask
            bg_mask = ~mask.astype(bool)
            bg_coords = np.array(np.where(bg_mask)).T[:, ::-1]
            if len(bg_coords) > 0:
                bg_indices = np.random.choice(len(bg_coords), min(5, len(bg_coords)), replace=False)
                bg_points = bg_coords[bg_indices]
            else:
                bg_points = np.zeros((0, 2))

            # Combine points
            all_points = np.vstack([fg_points, bg_points]) if len(bg_points) > 0 else fg_points
            labels = np.array([1] * len(fg_points) + [0] * len(bg_points))

            # Predict refined mask
            result = self.predict_with_points(all_points, labels, multimask_output=False)
            mask = result['masks'][0]

        return mask.astype(np.uint8) * 255


class SAMSegmenter:
    """
    Full segmentation using SAM's automatic mask generator.

    Automatically segments all objects in an image.
    Expected performance: Variable (depends on prompts)
    """

    def __init__(
        self,
        model_type: str = 'vit_b',
        checkpoint_path: Optional[str] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        min_mask_region_area: int = 100
    ):
        """
        Initialize SAM segmenter.

        Args:
            model_type: Model type
            checkpoint_path: Path to checkpoint
            device: Device to use
            points_per_side: Points per side for grid sampling
            pred_iou_thresh: Predicted IoU threshold
            stability_score_thresh: Stability score threshold
            min_mask_region_area: Minimum mask area
        """
        self.adapter = SAMAdapter(model_type, checkpoint_path, device)
        self.points_per_side = points_per_side
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.min_mask_region_area = min_mask_region_area

        # Initialize mask generator
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.adapter.sam,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            min_mask_region_area=min_mask_region_area
        )

    def segment(
        self,
        image: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Automatically segment all objects in image.

        Args:
            image: RGB image

        Returns:
            Dictionary with masks, bboxes, and scores
        """
        # Generate masks
        sam_results = self.mask_generator.generate(image)

        if not sam_results:
            h, w = image.shape[:2]
            return {
                'masks': np.zeros((0, h, w)),
                'bboxes': np.zeros((0, 4)),
                'scores': np.zeros((0,))
            }

        # Extract results
        masks = []
        bboxes = []
        scores = []

        for result in sam_results:
            mask = result['segmentation'].astype(np.uint8) * 255
            bbox = result['bbox']  # [x, y, w, h]
            score = result['predicted_iou']

            masks.append(mask)
            bboxes.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
            scores.append(score)

        return {
            'masks': np.array(masks),
            'bboxes': np.array(bboxes),
            'scores': np.array(scores)
        }

    def segment_with_boxes(
        self,
        image: np.ndarray,
        boxes: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Segment objects given bounding box prompts.

        Args:
            image: RGB image
            boxes: Bounding boxes (N, 4) in xyxy format

        Returns:
            Segmentation results
        """
        self.adapter.set_image(image)

        masks = []
        scores = []
        refined_boxes = []

        for box in boxes:
            result = self.adapter.predict_with_box(box, multimask_output=False)
            mask = result['masks'][0]
            score = result['scores'][0]

            masks.append(mask.astype(np.uint8) * 255)
            scores.append(score)
            refined_boxes.append(box)

        return {
            'masks': np.array(masks) if masks else np.zeros((0, *image.shape[:2])),
            'bboxes': np.array(refined_boxes) if refined_boxes else np.zeros((0, 4)),
            'scores': np.array(scores) if scores else np.zeros((0,))
        }
