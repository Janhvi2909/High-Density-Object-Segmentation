"""
Adaptive thresholding-based segmentation.

Baseline method using traditional image processing techniques.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional


class AdaptiveThresholdSegmenter:
    """
    Segmentation using adaptive thresholding and morphological operations.

    This is a baseline method that uses intensity-based segmentation.
    Expected performance: Low (15-25% mAP) due to inability to handle
    overlapping objects and varying illumination.
    """

    def __init__(
        self,
        block_size: int = 11,
        constant: int = 2,
        min_area: int = 100,
        max_area: int = 50000
    ):
        """
        Initialize adaptive threshold segmenter.

        Args:
            block_size: Size of neighborhood for threshold calculation
            constant: Constant subtracted from mean
            min_area: Minimum contour area to consider
            max_area: Maximum contour area to consider
        """
        self.block_size = block_size
        self.constant = constant
        self.min_area = min_area
        self.max_area = max_area

    def segment(
        self,
        image: np.ndarray,
        preprocess: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Segment objects using adaptive thresholding.

        Args:
            image: Input image (RGB or grayscale)
            preprocess: Whether to apply preprocessing

        Returns:
            Dictionary with masks, bboxes, and scores
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        # Preprocessing
        if preprocess:
            # Gaussian blur to reduce noise
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            # CLAHE for contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)

        # Adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self.block_size,
            self.constant
        )

        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter and extract bounding boxes
        masks = []
        bboxes = []
        scores = []

        for contour in contours:
            area = cv2.contourArea(contour)

            if self.min_area <= area <= self.max_area:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)

                # Aspect ratio filter (products are typically vertical)
                aspect_ratio = w / h if h > 0 else 0
                if 0.1 <= aspect_ratio <= 3.0:
                    bboxes.append([x, y, x + w, y + h])

                    # Create mask for this contour
                    mask = np.zeros(binary.shape, dtype=np.uint8)
                    cv2.drawContours(mask, [contour], -1, 255, -1)
                    masks.append(mask)

                    # Score based on contour solidity
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    solidity = area / hull_area if hull_area > 0 else 0
                    scores.append(solidity)

        return {
            'masks': np.array(masks) if masks else np.zeros((0, *binary.shape)),
            'bboxes': np.array(bboxes) if bboxes else np.zeros((0, 4)),
            'scores': np.array(scores) if scores else np.zeros((0,)),
            'binary': binary
        }

    def segment_with_multiple_thresholds(
        self,
        image: np.ndarray,
        block_sizes: List[int] = [7, 11, 15, 21]
    ) -> Dict[str, np.ndarray]:
        """
        Segment using multiple threshold parameters and combine results.

        Args:
            image: Input image
            block_sizes: List of block sizes to try

        Returns:
            Combined segmentation results
        """
        all_masks = []
        all_bboxes = []
        all_scores = []

        for bs in block_sizes:
            self.block_size = bs
            result = self.segment(image)

            if len(result['bboxes']) > 0:
                all_masks.extend(result['masks'])
                all_bboxes.extend(result['bboxes'])
                all_scores.extend(result['scores'])

        if not all_bboxes:
            return {
                'masks': np.zeros((0, image.shape[0], image.shape[1])),
                'bboxes': np.zeros((0, 4)),
                'scores': np.zeros((0,))
            }

        # Non-maximum suppression to remove duplicates
        bboxes = np.array(all_bboxes)
        scores = np.array(all_scores)
        masks = all_masks

        keep = self._nms(bboxes, scores, threshold=0.5)

        return {
            'masks': np.array([masks[i] for i in keep]),
            'bboxes': bboxes[keep],
            'scores': scores[keep]
        }

    @staticmethod
    def _nms(
        bboxes: np.ndarray,
        scores: np.ndarray,
        threshold: float = 0.5
    ) -> List[int]:
        """
        Apply Non-Maximum Suppression.

        Args:
            bboxes: Bounding boxes (N, 4)
            scores: Confidence scores (N,)
            threshold: IoU threshold for suppression

        Returns:
            List of indices to keep
        """
        if len(bboxes) == 0:
            return []

        # Sort by score
        order = np.argsort(scores)[::-1]

        x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        keep = []

        while len(order) > 0:
            i = order[0]
            keep.append(i)

            if len(order) == 1:
                break

            # Calculate IoU with rest
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter)

            # Keep boxes with IoU below threshold
            inds = np.where(iou <= threshold)[0]
            order = order[inds + 1]

        return keep
