"""
Superpixel-based segmentation using SLIC.

Oversegmentation followed by merging for object detection.
"""

import cv2
import numpy as np
from skimage.segmentation import slic, mark_boundaries
from skimage import graph 
from typing import Dict, List, Tuple, Optional


class SLICSegmenter:
    """
    Segmentation using SLIC superpixels.

    Creates superpixels and merges them based on similarity.
    Expected performance: Medium (28-38% mAP)
    """

    def __init__(
        self,
        n_segments: int = 500,
        compactness: float = 10.0,
        sigma: float = 1.0,
        min_area: int = 50,
        max_area: int = 50000
    ):
        """
        Initialize SLIC segmenter.

        Args:
            n_segments: Approximate number of superpixels
            compactness: Balance between color and space proximity
            sigma: Gaussian smoothing
            min_area: Minimum segment area
            max_area: Maximum segment area
        """
        self.n_segments = n_segments
        self.compactness = compactness
        self.sigma = sigma
        self.min_area = min_area
        self.max_area = max_area

    def segment(
        self,
        image: np.ndarray,
        merge_threshold: float = 0.3
    ) -> Dict[str, np.ndarray]:
        """
        Segment using SLIC superpixels with merging.

        Args:
            image: Input image
            merge_threshold: Threshold for merging similar superpixels

        Returns:
            Segmentation results
        """
        if len(image.shape) != 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        h, w = image.shape[:2]

        # Generate superpixels
        segments = slic(
            image,
            n_segments=self.n_segments,
            compactness=self.compactness,
            sigma=self.sigma,
            start_label=0
        )

        # Build region adjacency graph
        rag = graph.rag_mean_color(image, segments)

        # Merge similar regions
        merged = graph.merge_hierarchical(
            segments, rag,
            thresh=merge_threshold,
            rag_copy=False,
            in_place_merge=True,
            merge_func=self._merge_boundary,
            weight_func=self._weight_boundary
        )

        # Extract objects from merged segments
        masks = []
        bboxes = []
        scores = []

        unique_labels = np.unique(merged)
        for label_id in unique_labels:
            mask = (merged == label_id).astype(np.uint8) * 255
            area = np.sum(mask > 0)

            if self.min_area <= area <= self.max_area:
                # Get bounding box
                coords = np.where(mask > 0)
                if len(coords[0]) > 0:
                    y1, y2 = coords[0].min(), coords[0].max()
                    x1, x2 = coords[1].min(), coords[1].max()

                    box_w, box_h = x2 - x1, y2 - y1
                    if box_h > 0 and 0.1 <= box_w / box_h <= 3.0:
                        masks.append(mask)
                        bboxes.append([x1, y1, x2, y2])

                        # Score based on region compactness
                        perimeter = self._estimate_perimeter(mask)
                        compactness = (4 * np.pi * area) / (perimeter ** 2 + 1e-6)
                        scores.append(min(compactness, 1.0))

        return {
            'masks': np.array(masks) if masks else np.zeros((0, h, w)),
            'bboxes': np.array(bboxes) if bboxes else np.zeros((0, 4)),
            'scores': np.array(scores) if scores else np.zeros((0,)),
            'superpixels': segments,
            'merged': merged
        }

    def segment_with_color_histogram(
        self,
        image: np.ndarray,
        n_bins: int = 32
    ) -> Dict[str, np.ndarray]:
        """
        Segment by comparing superpixel color histograms.

        Args:
            image: Input image
            n_bins: Number of histogram bins

        Returns:
            Segmentation results
        """
        if len(image.shape) != 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        h, w = image.shape[:2]

        # Generate superpixels
        segments = slic(
            image,
            n_segments=self.n_segments,
            compactness=self.compactness,
            sigma=self.sigma,
            start_label=0
        )

        # Compute histogram for each superpixel
        unique_labels = np.unique(segments)
        histograms = {}

        for label_id in unique_labels:
            mask = segments == label_id
            pixels = image[mask]

            # Compute color histogram
            hist = []
            for c in range(3):
                h_c, _ = np.histogram(pixels[:, c], bins=n_bins, range=(0, 255))
                h_c = h_c.astype(np.float32)
                h_c /= (h_c.sum() + 1e-6)
                hist.append(h_c)

            histograms[label_id] = np.concatenate(hist)

        # Merge similar superpixels using histogram comparison
        merged = segments.copy()
        label_map = {l: l for l in unique_labels}

        for i, l1 in enumerate(unique_labels):
            for l2 in unique_labels[i+1:]:
                if label_map[l1] != label_map[l2]:
                    # Check adjacency
                    if self._are_adjacent(segments, l1, l2):
                        # Compare histograms
                        dist = np.linalg.norm(histograms[l1] - histograms[l2])
                        if dist < 0.5:  # Similar enough to merge
                            # Merge l2 into l1
                            old_label = label_map[l2]
                            new_label = label_map[l1]
                            for k, v in label_map.items():
                                if v == old_label:
                                    label_map[k] = new_label

        # Apply label mapping
        for old, new in label_map.items():
            merged[segments == old] = new

        # Extract objects
        masks = []
        bboxes = []
        scores = []

        for label_id in np.unique(merged):
            mask = (merged == label_id).astype(np.uint8) * 255
            area = np.sum(mask > 0)

            if self.min_area <= area <= self.max_area:
                coords = np.where(mask > 0)
                if len(coords[0]) > 0:
                    y1, y2 = coords[0].min(), coords[0].max()
                    x1, x2 = coords[1].min(), coords[1].max()

                    box_w, box_h = x2 - x1, y2 - y1
                    if box_h > 0 and 0.1 <= box_w / box_h <= 3.0:
                        masks.append(mask)
                        bboxes.append([x1, y1, x2, y2])
                        scores.append(0.5)

        return {
            'masks': np.array(masks) if masks else np.zeros((0, h, w)),
            'bboxes': np.array(bboxes) if bboxes else np.zeros((0, 4)),
            'scores': np.array(scores) if scores else np.zeros((0,)),
            'superpixels': segments
        }

    @staticmethod
    def _merge_boundary(graph, src, dst, n):
        """Callback for region merging."""
        pass

    @staticmethod
    def _weight_boundary(graph, src, dst, n):
        """Weight function for region adjacency graph."""
        diff = graph[dst][src].get('weight', 0)
        return diff

    @staticmethod
    def _estimate_perimeter(mask: np.ndarray) -> float:
        """Estimate perimeter from mask."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            return cv2.arcLength(contours[0], True)
        return 1.0

    @staticmethod
    def _are_adjacent(segments: np.ndarray, l1: int, l2: int) -> bool:
        """Check if two superpixels are adjacent."""
        mask1 = segments == l1
        mask2 = segments == l2

        # Dilate mask1 and check overlap with mask2
        kernel = np.ones((3, 3), dtype=np.uint8)
        dilated = cv2.dilate(mask1.astype(np.uint8), kernel)

        return np.any(dilated & mask2)

    def visualize_superpixels(
        self,
        image: np.ndarray,
        segments: np.ndarray
    ) -> np.ndarray:
        """
        Visualize superpixel boundaries.

        Args:
            image: Original image
            segments: Superpixel labels

        Returns:
            Image with superpixel boundaries
        """
        return (mark_boundaries(image, segments) * 255).astype(np.uint8)
