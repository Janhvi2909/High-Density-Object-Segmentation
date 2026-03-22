"""
Marker-controlled watershed segmentation.

Advanced CV method for separating touching objects.
"""

import cv2
import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from typing import Dict, List, Tuple, Optional


class MarkerControlledWatershed:
    """
    Marker-controlled watershed segmentation.

    Uses distance transform and local maxima to create markers,
    then applies watershed to separate touching objects.
    Expected performance: Medium (30-40% mAP)
    """

    def __init__(
        self,
        distance_threshold: float = 0.5,
        min_distance: int = 10,
        min_area: int = 50,
        max_area: int = 50000
    ):
        """
        Initialize watershed segmenter.

        Args:
            distance_threshold: Threshold for distance transform
            min_distance: Minimum distance between markers
            min_area: Minimum segment area
            max_area: Maximum segment area
        """
        self.distance_threshold = distance_threshold
        self.min_distance = min_distance
        self.min_area = min_area
        self.max_area = max_area

    def segment(
        self,
        image: np.ndarray,
        foreground_mask: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Segment using marker-controlled watershed.

        Args:
            image: Input image
            foreground_mask: Optional binary mask of foreground

        Returns:
            Dictionary with masks, bboxes, and scores
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        # Create foreground mask if not provided
        if foreground_mask is None:
            foreground_mask = self._create_foreground_mask(gray)

        # Distance transform
        dist_transform = cv2.distanceTransform(
            foreground_mask, cv2.DIST_L2, 5
        )

        # Normalize distance transform
        dist_normalized = dist_transform / dist_transform.max() if dist_transform.max() > 0 else dist_transform

        # Find local maxima as markers
        coordinates = peak_local_max(
            dist_normalized,
            min_distance=self.min_distance,
            threshold_abs=self.distance_threshold,
            exclude_border=False
        )

        # Create marker image
        markers = np.zeros_like(foreground_mask, dtype=np.int32)
        for i, (y, x) in enumerate(coordinates, start=1):
            markers[y, x] = i

        # Expand markers
        markers = ndimage.label(markers)[0]

        # Apply watershed
        if len(image.shape) == 3:
            watershed_image = image
        else:
            watershed_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        labels = watershed(-dist_transform, markers, mask=foreground_mask)

        # Extract segments
        masks = []
        bboxes = []
        scores = []

        unique_labels = np.unique(labels)
        for label_id in unique_labels:
            if label_id <= 0:  # Skip background and watershed lines
                continue

            mask = (labels == label_id).astype(np.uint8) * 255
            area = np.sum(mask > 0)

            if self.min_area <= area <= self.max_area:
                # Get bounding box
                coords = np.where(mask > 0)
                if len(coords[0]) == 0:
                    continue

                y1, y2 = coords[0].min(), coords[0].max()
                x1, x2 = coords[1].min(), coords[1].max()

                # Aspect ratio filter
                w, h = x2 - x1, y2 - y1
                if h > 0 and 0.1 <= w / h <= 3.0:
                    masks.append(mask)
                    bboxes.append([x1, y1, x2, y2])

                    # Score based on marker strength
                    marker_value = dist_transform[coords[0], coords[1]].max()
                    score = marker_value / (dist_transform.max() + 1e-6)
                    scores.append(score)

        return {
            'masks': np.array(masks) if masks else np.zeros((0, *gray.shape)),
            'bboxes': np.array(bboxes) if bboxes else np.zeros((0, 4)),
            'scores': np.array(scores) if scores else np.zeros((0,)),
            'labels': labels,
            'dist_transform': dist_transform
        }

    def _create_foreground_mask(self, gray: np.ndarray) -> np.ndarray:
        """Create foreground mask using Otsu's thresholding."""
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Otsu's thresholding
        _, binary = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        return binary

    def segment_with_gradient(
        self,
        image: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Watershed using gradient magnitude as the flooding surface.

        Better for separating objects with distinct edges.

        Args:
            image: Input image

        Returns:
            Segmentation results
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        # Compute gradient
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(grad_x**2 + grad_y**2)
        gradient = (gradient / gradient.max() * 255).astype(np.uint8)

        # Create markers using adaptive thresholding
        _, markers_thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # Sure background
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        sure_bg = cv2.dilate(markers_thresh, kernel, iterations=3)

        # Sure foreground using distance transform
        dist_transform = cv2.distanceTransform(markers_thresh, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(
            dist_transform, self.distance_threshold * dist_transform.max(), 255, 0
        )
        sure_fg = np.uint8(sure_fg)

        # Unknown region
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Label markers
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        # Apply watershed using gradient as surface
        if len(image.shape) == 3:
            watershed_image = image.copy()
        else:
            watershed_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        cv2.watershed(watershed_image, markers)

        # Extract segments
        masks = []
        bboxes = []
        scores = []

        for label_id in range(2, markers.max() + 1):
            mask = (markers == label_id).astype(np.uint8) * 255
            area = np.sum(mask > 0)

            if self.min_area <= area <= self.max_area:
                coords = np.where(mask > 0)
                if len(coords[0]) > 0:
                    y1, y2 = coords[0].min(), coords[0].max()
                    x1, x2 = coords[1].min(), coords[1].max()

                    w, h = x2 - x1, y2 - y1
                    if h > 0 and 0.1 <= w / h <= 3.0:
                        masks.append(mask)
                        bboxes.append([x1, y1, x2, y2])
                        scores.append(0.5)

        return {
            'masks': np.array(masks) if masks else np.zeros((0, *gray.shape)),
            'bboxes': np.array(bboxes) if bboxes else np.zeros((0, 4)),
            'scores': np.array(scores) if scores else np.zeros((0,)),
            'markers': markers
        }
