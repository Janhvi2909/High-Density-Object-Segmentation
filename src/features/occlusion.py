"""
Occlusion-aware feature extraction.

Estimates object visibility and overlap to help models handle occluded objects.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple


class OcclusionEstimator:
    """
    Estimates occlusion levels between objects.

    Provides visibility scores for objects based on overlap analysis.
    """

    def __init__(
        self,
        iou_threshold: float = 0.1,
        visibility_threshold: float = 0.7
    ):
        """
        Initialize occlusion estimator.

        Args:
            iou_threshold: IoU threshold to consider objects as overlapping
            visibility_threshold: Visibility ratio below which object is occluded
        """
        self.iou_threshold = iou_threshold
        self.visibility_threshold = visibility_threshold

    @staticmethod
    def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """
        Calculate Intersection over Union between two boxes.

        Args:
            box1: First box [x1, y1, x2, y2]
            box2: Second box [x1, y1, x2, y2]

        Returns:
            IoU value
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0

    @staticmethod
    def calculate_overlap_ratio(box1: np.ndarray, box2: np.ndarray) -> float:
        """
        Calculate what fraction of box1 is covered by box2.

        Args:
            box1: Target box [x1, y1, x2, y2]
            box2: Potentially occluding box [x1, y1, x2, y2]

        Returns:
            Overlap ratio (0-1)
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])

        return inter_area / box1_area if box1_area > 0 else 0

    def estimate_visibility(
        self,
        bboxes: np.ndarray,
        depth_order: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Estimate visibility score for each object.

        Higher score = more visible (less occluded).

        Args:
            bboxes: Bounding boxes (N, 4) in xyxy format
            depth_order: Optional depth ordering (0 = front)

        Returns:
            Visibility scores (N,) in range [0, 1]
        """
        n = len(bboxes)
        if n == 0:
            return np.array([])

        visibility = np.ones(n)

        # If no depth order provided, estimate from y2 (bottom of box)
        # Objects lower in image are typically in front
        if depth_order is None:
            depth_order = np.argsort(-bboxes[:, 3])  # Sort by y2 descending

        # Calculate occlusion for each object
        for i in range(n):
            occluder_indices = depth_order[:np.where(depth_order == i)[0][0]]

            total_occlusion = 0
            for j in occluder_indices:
                overlap = self.calculate_overlap_ratio(bboxes[i], bboxes[j])
                total_occlusion = max(total_occlusion, overlap)

            visibility[i] = 1 - total_occlusion

        return visibility

    def get_occlusion_graph(
        self,
        bboxes: np.ndarray
    ) -> np.ndarray:
        """
        Build occlusion graph between objects.

        Returns adjacency matrix where entry[i,j] = overlap of box i by box j.

        Args:
            bboxes: Bounding boxes (N, 4)

        Returns:
            Occlusion matrix (N, N)
        """
        n = len(bboxes)
        graph = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i != j:
                    graph[i, j] = self.calculate_overlap_ratio(bboxes[i], bboxes[j])

        return graph

    def identify_heavily_occluded(
        self,
        bboxes: np.ndarray,
        threshold: float = 0.5
    ) -> np.ndarray:
        """
        Identify heavily occluded objects.

        Args:
            bboxes: Bounding boxes (N, 4)
            threshold: Visibility threshold (below = heavily occluded)

        Returns:
            Boolean mask of heavily occluded objects
        """
        visibility = self.estimate_visibility(bboxes)
        return visibility < threshold


class VisibilityFeatures:
    """
    Extract features encoding object visibility and occlusion patterns.

    These features help models learn to handle partially visible objects.
    """

    def __init__(self):
        """Initialize visibility feature extractor."""
        self.occlusion_estimator = OcclusionEstimator()

    def extract_per_object(
        self,
        bboxes: np.ndarray,
        image: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Extract visibility features for each object.

        Args:
            bboxes: Bounding boxes (N, 4)
            image: Optional image for additional features

        Returns:
            Dictionary of per-object features
        """
        n = len(bboxes)

        if n == 0:
            return {
                'visibility_score': np.array([]),
                'num_overlapping': np.array([]),
                'max_overlap': np.array([]),
                'is_edge_object': np.array([])
            }

        # Visibility scores
        visibility = self.occlusion_estimator.estimate_visibility(bboxes)

        # Occlusion graph analysis
        occlusion_graph = self.occlusion_estimator.get_occlusion_graph(bboxes)

        # Number of objects overlapping with each object
        num_overlapping = np.sum(occlusion_graph > 0.1, axis=1)

        # Maximum overlap for each object
        max_overlap = np.max(occlusion_graph, axis=1)

        # Edge object detection (touching image boundary)
        if image is not None:
            h, w = image.shape[:2]
        else:
            h, w = bboxes[:, 3].max(), bboxes[:, 2].max()

        is_edge = (
            (bboxes[:, 0] < 5) |  # left edge
            (bboxes[:, 1] < 5) |  # top edge
            (bboxes[:, 2] > w - 5) |  # right edge
            (bboxes[:, 3] > h - 5)    # bottom edge
        ).astype(float)

        return {
            'visibility_score': visibility,
            'num_overlapping': num_overlapping.astype(float),
            'max_overlap': max_overlap,
            'is_edge_object': is_edge
        }

    def extract_global(
        self,
        bboxes: np.ndarray,
        image_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Extract global occlusion statistics.

        Args:
            bboxes: Bounding boxes (N, 4)
            image_size: Image size (height, width)

        Returns:
            Global feature vector
        """
        if len(bboxes) == 0:
            return np.zeros(10)

        visibility = self.occlusion_estimator.estimate_visibility(bboxes)
        heavily_occluded = visibility < 0.5

        occlusion_graph = self.occlusion_estimator.get_occlusion_graph(bboxes)

        features = [
            visibility.mean(),                          # Mean visibility
            visibility.std(),                           # Visibility variance
            (visibility < 0.3).mean(),                  # Fraction severely occluded
            (visibility < 0.5).mean(),                  # Fraction moderately occluded
            (visibility > 0.8).mean(),                  # Fraction mostly visible
            np.sum(occlusion_graph > 0) / (len(bboxes) ** 2),  # Overlap density
            np.max(occlusion_graph) if len(bboxes) > 0 else 0, # Max overlap
            np.mean(np.sum(occlusion_graph > 0.1, axis=1)),    # Avg overlapping neighbors
            len(bboxes) / (image_size[0] * image_size[1]) * 1e6,  # Object density
            len(bboxes),                                # Total objects
        ]

        return np.array(features)

    def create_visibility_map(
        self,
        bboxes: np.ndarray,
        image_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Create spatial map of visibility.

        Each pixel contains expected visibility of objects in that region.

        Args:
            bboxes: Bounding boxes (N, 4)
            image_size: Image size (height, width)

        Returns:
            Visibility map (H, W)
        """
        h, w = image_size
        visibility_map = np.ones((h, w), dtype=np.float32)

        if len(bboxes) == 0:
            return visibility_map

        visibility_scores = self.occlusion_estimator.estimate_visibility(bboxes)

        # Weight each pixel by visibility of objects covering it
        count_map = np.zeros((h, w), dtype=np.float32)

        for bbox, vis in zip(bboxes, visibility_scores):
            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            visibility_map[y1:y2, x1:x2] += vis
            count_map[y1:y2, x1:x2] += 1

        # Average where multiple objects
        mask = count_map > 0
        visibility_map[mask] /= count_map[mask]

        return visibility_map
