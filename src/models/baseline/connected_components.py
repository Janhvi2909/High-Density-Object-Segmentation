"""
Connected components-based segmentation.

Baseline method using connected component labeling.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional


class ConnectedComponentsSegmenter:
    """
    Segmentation using connected component analysis.

    Simple baseline using binary segmentation and component labeling.
    Expected performance: Low (12-20% mAP)
    """

    def __init__(
        self,
        threshold_method: str = 'otsu',
        min_area: int = 50,
        max_area: int = 50000,
        connectivity: int = 8
    ):
        """
        Initialize connected components segmenter.

        Args:
            threshold_method: 'otsu', 'adaptive', or 'fixed'
            min_area: Minimum component area
            max_area: Maximum component area
            connectivity: 4 or 8 connectivity
        """
        self.threshold_method = threshold_method
        self.min_area = min_area
        self.max_area = max_area
        self.connectivity = connectivity

    def _binarize(self, image: np.ndarray) -> np.ndarray:
        """
        Convert image to binary using specified method.

        Args:
            image: Grayscale image

        Returns:
            Binary image
        """
        if self.threshold_method == 'otsu':
            _, binary = cv2.threshold(
                image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
        elif self.threshold_method == 'adaptive':
            binary = cv2.adaptiveThreshold(
                image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )
        else:  # fixed
            _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)

        return binary

    def segment(
        self,
        image: np.ndarray,
        return_labels: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Segment using connected component analysis.

        Args:
            image: Input image
            return_labels: Whether to return label image

        Returns:
            Dictionary with masks, bboxes, scores, and optionally labels
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        # Preprocessing
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Binarization
        binary = self._binarize(gray)

        # Morphological cleaning
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # Connected component analysis
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=self.connectivity
        )

        masks = []
        bboxes = []
        scores = []

        # Skip label 0 (background)
        for label_id in range(1, num_labels):
            area = stats[label_id, cv2.CC_STAT_AREA]

            if self.min_area <= area <= self.max_area:
                x = stats[label_id, cv2.CC_STAT_LEFT]
                y = stats[label_id, cv2.CC_STAT_TOP]
                w = stats[label_id, cv2.CC_STAT_WIDTH]
                h = stats[label_id, cv2.CC_STAT_HEIGHT]

                # Aspect ratio filter
                aspect_ratio = w / h if h > 0 else 0
                if 0.1 <= aspect_ratio <= 3.0:
                    bboxes.append([x, y, x + w, y + h])

                    # Create mask
                    mask = (labels == label_id).astype(np.uint8) * 255
                    masks.append(mask)

                    # Score based on compactness
                    perimeter = self._estimate_perimeter(mask)
                    compactness = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
                    scores.append(min(compactness, 1.0))

        result = {
            'masks': np.array(masks) if masks else np.zeros((0, *gray.shape)),
            'bboxes': np.array(bboxes) if bboxes else np.zeros((0, 4)),
            'scores': np.array(scores) if scores else np.zeros((0,)),
            'binary': binary
        }

        if return_labels:
            result['labels'] = labels

        return result

    @staticmethod
    def _estimate_perimeter(mask: np.ndarray) -> float:
        """Estimate perimeter from binary mask."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            return cv2.arcLength(contours[0], True)
        return 0

    def segment_color(
        self,
        image: np.ndarray,
        color_space: str = 'LAB'
    ) -> Dict[str, np.ndarray]:
        """
        Segment using color-based connected components.

        Args:
            image: RGB image
            color_space: Color space to use ('LAB', 'HSV', 'RGB')

        Returns:
            Segmentation results
        """
        if len(image.shape) != 3:
            return self.segment(image)

        # Convert color space
        if color_space == 'LAB':
            converted = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        elif color_space == 'HSV':
            converted = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        else:
            converted = image

        # Process each channel
        all_masks = []
        all_bboxes = []
        all_scores = []

        for channel in range(converted.shape[2]):
            result = self.segment(converted[:, :, channel])
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

        # NMS
        bboxes = np.array(all_bboxes)
        scores = np.array(all_scores)
        keep = self._nms(bboxes, scores, 0.5)

        return {
            'masks': np.array([all_masks[i] for i in keep]),
            'bboxes': bboxes[keep],
            'scores': scores[keep]
        }

    @staticmethod
    def _nms(bboxes: np.ndarray, scores: np.ndarray, threshold: float) -> List[int]:
        """Non-Maximum Suppression."""
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
