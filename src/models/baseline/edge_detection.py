"""
Edge detection-based segmentation.

Baseline method using Canny edge detection and contour analysis.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional


class CannyContourSegmenter:
    """
    Segmentation using Canny edge detection and contour finding.

    Uses edge information to identify object boundaries.
    Expected performance: Low-Medium (18-28% mAP)
    """

    def __init__(
        self,
        canny_low: int = 50,
        canny_high: int = 150,
        min_area: int = 100,
        max_area: int = 50000,
        use_auto_canny: bool = True
    ):
        """
        Initialize Canny contour segmenter.

        Args:
            canny_low: Lower threshold for Canny
            canny_high: Upper threshold for Canny
            min_area: Minimum contour area
            max_area: Maximum contour area
            use_auto_canny: Automatically determine thresholds
        """
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.min_area = min_area
        self.max_area = max_area
        self.use_auto_canny = use_auto_canny

    def _auto_canny(self, image: np.ndarray, sigma: float = 0.33) -> np.ndarray:
        """
        Apply automatic Canny edge detection.

        Args:
            image: Grayscale image
            sigma: Threshold adjustment factor

        Returns:
            Edge image
        """
        median = np.median(image)
        lower = int(max(0, (1.0 - sigma) * median))
        upper = int(min(255, (1.0 + sigma) * median))
        return cv2.Canny(image, lower, upper)

    def segment(
        self,
        image: np.ndarray,
        use_morphology: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Segment objects using edge detection.

        Args:
            image: Input image (RGB or grayscale)
            use_morphology: Apply morphological operations

        Returns:
            Dictionary with masks, bboxes, and scores
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        # Preprocessing
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Edge detection
        if self.use_auto_canny:
            edges = self._auto_canny(gray)
        else:
            edges = cv2.Canny(gray, self.canny_low, self.canny_high)

        # Morphological closing to connect edges
        if use_morphology:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            edges = cv2.dilate(edges, kernel, iterations=2)
            edges = cv2.erode(edges, kernel, iterations=1)

        # Find contours
        contours, hierarchy = cv2.findContours(
            edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        # Process contours
        masks = []
        bboxes = []
        scores = []

        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)

            if self.min_area <= area <= self.max_area:
                # Approximate contour
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                # Get bounding box
                x, y, w, h = cv2.boundingRect(approx)

                # Filter by aspect ratio
                aspect_ratio = w / h if h > 0 else 0
                if 0.1 <= aspect_ratio <= 3.0:
                    bboxes.append([x, y, x + w, y + h])

                    # Create filled mask
                    mask = np.zeros(gray.shape, dtype=np.uint8)
                    cv2.drawContours(mask, [contour], -1, 255, -1)
                    masks.append(mask)

                    # Score based on contour characteristics
                    perimeter = cv2.arcLength(contour, True)
                    circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    solidity = area / hull_area if hull_area > 0 else 0

                    # Combined score
                    score = 0.5 * solidity + 0.5 * min(circularity * 2, 1.0)
                    scores.append(score)

        return {
            'masks': np.array(masks) if masks else np.zeros((0, *gray.shape)),
            'bboxes': np.array(bboxes) if bboxes else np.zeros((0, 4)),
            'scores': np.array(scores) if scores else np.zeros((0,)),
            'edges': edges
        }

    def segment_with_multiple_scales(
        self,
        image: np.ndarray,
        scales: List[float] = [0.5, 1.0, 1.5]
    ) -> Dict[str, np.ndarray]:
        """
        Multi-scale edge detection for better coverage.

        Args:
            image: Input image
            scales: Gaussian blur scales to use

        Returns:
            Combined segmentation results
        """
        all_masks = []
        all_bboxes = []
        all_scores = []

        for scale in scales:
            # Apply blur at different scales
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()

            ksize = int(5 * scale)
            if ksize % 2 == 0:
                ksize += 1

            blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0)

            # Detect edges
            if self.use_auto_canny:
                edges = self._auto_canny(blurred)
            else:
                edges = cv2.Canny(
                    blurred,
                    int(self.canny_low * scale),
                    int(self.canny_high * scale)
                )

            # Find contours
            contours, _ = cv2.findContours(
                edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                area = cv2.contourArea(contour)
                if self.min_area <= area <= self.max_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    if 0.1 <= w / max(h, 1) <= 3.0:
                        all_bboxes.append([x, y, x + w, y + h])
                        mask = np.zeros(gray.shape, dtype=np.uint8)
                        cv2.drawContours(mask, [contour], -1, 255, -1)
                        all_masks.append(mask)
                        all_scores.append(0.5)  # Default score

        if not all_bboxes:
            return {
                'masks': np.zeros((0, image.shape[0], image.shape[1])),
                'bboxes': np.zeros((0, 4)),
                'scores': np.zeros((0,))
            }

        # NMS
        bboxes = np.array(all_bboxes)
        scores = np.array(all_scores)
        keep = self._nms(bboxes, scores, threshold=0.5)

        return {
            'masks': np.array([all_masks[i] for i in keep]),
            'bboxes': bboxes[keep],
            'scores': scores[keep]
        }

    @staticmethod
    def _nms(bboxes: np.ndarray, scores: np.ndarray, threshold: float) -> List[int]:
        """Apply Non-Maximum Suppression."""
        if len(bboxes) == 0:
            return []

        order = np.argsort(scores)[::-1]
        x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(i)

            if len(order) == 1:
                break

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

            inds = np.where(iou <= threshold)[0]
            order = order[inds + 1]

        return keep
