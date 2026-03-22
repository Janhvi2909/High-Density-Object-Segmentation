"""
Gaussian Mixture Model-based segmentation.

Advanced unsupervised segmentation using color clustering.
"""

import cv2
import numpy as np
from sklearn.mixture import GaussianMixture
from typing import Dict, List, Tuple, Optional


class GMMSegmenter:
    """
    Segmentation using Gaussian Mixture Models.

    Clusters pixels by color and extracts connected regions.
    Expected performance: Medium (25-35% mAP)
    """

    def __init__(
        self,
        n_components: int = 5,
        covariance_type: str = 'full',
        min_area: int = 50,
        max_area: int = 50000,
        random_state: int = 42
    ):
        """
        Initialize GMM segmenter.

        Args:
            n_components: Number of Gaussian components
            covariance_type: Type of covariance ('full', 'tied', 'diag', 'spherical')
            min_area: Minimum segment area
            max_area: Maximum segment area
            random_state: Random seed
        """
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.min_area = min_area
        self.max_area = max_area
        self.random_state = random_state

    def segment(
        self,
        image: np.ndarray,
        color_space: str = 'LAB',
        use_spatial: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Segment image using GMM clustering.

        Args:
            image: Input RGB image
            color_space: Color space for clustering ('RGB', 'LAB', 'HSV')
            use_spatial: Include spatial coordinates as features

        Returns:
            Dictionary with masks, bboxes, and scores
        """
        if len(image.shape) != 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        h, w = image.shape[:2]

        # Convert color space
        if color_space == 'LAB':
            converted = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        elif color_space == 'HSV':
            converted = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        else:
            converted = image.copy()

        # Prepare features
        pixels = converted.reshape(-1, 3).astype(np.float32)

        if use_spatial:
            # Add normalized spatial coordinates
            xx, yy = np.meshgrid(np.arange(w), np.arange(h))
            xx = (xx.flatten() / w - 0.5) * 50  # Scale to similar range as colors
            yy = (yy.flatten() / h - 0.5) * 50
            spatial = np.column_stack([xx, yy])
            features = np.hstack([pixels, spatial])
        else:
            features = pixels

        # Fit GMM
        gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
            n_init=3
        )
        labels = gmm.fit_predict(features)
        labels = labels.reshape(h, w)

        # Probabilities for scoring
        probs = gmm.predict_proba(features).reshape(h, w, -1)

        # Extract segments from each cluster
        masks = []
        bboxes = []
        scores = []

        for cluster_id in range(self.n_components):
            cluster_mask = (labels == cluster_id).astype(np.uint8) * 255

            # Find connected components within cluster
            num_labels, comp_labels, stats, _ = cv2.connectedComponentsWithStats(
                cluster_mask, connectivity=8
            )

            for comp_id in range(1, num_labels):
                area = stats[comp_id, cv2.CC_STAT_AREA]

                if self.min_area <= area <= self.max_area:
                    x = stats[comp_id, cv2.CC_STAT_LEFT]
                    y = stats[comp_id, cv2.CC_STAT_TOP]
                    comp_w = stats[comp_id, cv2.CC_STAT_WIDTH]
                    comp_h = stats[comp_id, cv2.CC_STAT_HEIGHT]

                    # Aspect ratio filter
                    if comp_h > 0 and 0.1 <= comp_w / comp_h <= 3.0:
                        mask = (comp_labels == comp_id).astype(np.uint8) * 255
                        masks.append(mask)
                        bboxes.append([x, y, x + comp_w, y + comp_h])

                        # Score based on cluster probability
                        mask_coords = np.where(mask > 0)
                        avg_prob = probs[mask_coords[0], mask_coords[1], cluster_id].mean()
                        scores.append(avg_prob)

        return {
            'masks': np.array(masks) if masks else np.zeros((0, h, w)),
            'bboxes': np.array(bboxes) if bboxes else np.zeros((0, 4)),
            'scores': np.array(scores) if scores else np.zeros((0,)),
            'cluster_labels': labels,
            'gmm': gmm
        }

    def segment_hierarchical(
        self,
        image: np.ndarray,
        component_range: Tuple[int, int] = (3, 8)
    ) -> Dict[str, np.ndarray]:
        """
        Hierarchical GMM segmentation with automatic component selection.

        Args:
            image: Input image
            component_range: Range of components to try (min, max)

        Returns:
            Best segmentation results based on BIC
        """
        best_bic = float('inf')
        best_result = None

        h, w = image.shape[:2]
        pixels = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)

        for n_comp in range(component_range[0], component_range[1] + 1):
            gmm = GaussianMixture(
                n_components=n_comp,
                covariance_type=self.covariance_type,
                random_state=self.random_state
            )
            gmm.fit(pixels)
            bic = gmm.bic(pixels)

            if bic < best_bic:
                best_bic = bic
                self.n_components = n_comp
                best_result = self.segment(image)

        return best_result

    def refine_with_morphology(
        self,
        result: Dict[str, np.ndarray],
        kernel_size: int = 5
    ) -> Dict[str, np.ndarray]:
        """
        Refine GMM segmentation with morphological operations.

        Args:
            result: Segmentation result from segment()
            kernel_size: Kernel size for morphology

        Returns:
            Refined segmentation
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        refined_masks = []
        refined_bboxes = []
        refined_scores = []

        for mask, bbox, score in zip(result['masks'], result['bboxes'], result['scores']):
            # Close small holes
            refined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            # Remove small protrusions
            refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel)

            # Update bounding box
            coords = np.where(refined > 0)
            if len(coords[0]) > 0:
                area = len(coords[0])
                if self.min_area <= area <= self.max_area:
                    y1, y2 = coords[0].min(), coords[0].max()
                    x1, x2 = coords[1].min(), coords[1].max()

                    refined_masks.append(refined)
                    refined_bboxes.append([x1, y1, x2, y2])
                    refined_scores.append(score)

        return {
            'masks': np.array(refined_masks) if refined_masks else np.zeros((0, *result['masks'].shape[1:])),
            'bboxes': np.array(refined_bboxes) if refined_bboxes else np.zeros((0, 4)),
            'scores': np.array(refined_scores) if refined_scores else np.zeros((0,))
        }
