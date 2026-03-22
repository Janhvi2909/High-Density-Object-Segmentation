"""
Density-aware feature extraction for high-density object segmentation.

Novel feature engineering: estimates local object density to guide model behavior.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.ndimage import gaussian_filter


class DensityEstimator:
    """
    Estimates local object density in images.

    Creates density maps that can guide attention mechanisms
    and model selection in hybrid approaches.
    """

    def __init__(
        self,
        window_size: int = 64,
        stride: int = 32,
        sigma: float = 16.0
    ):
        """
        Initialize density estimator.

        Args:
            window_size: Size of sliding window for density estimation
            stride: Stride for sliding window
            sigma: Gaussian smoothing sigma for density map
        """
        self.window_size = window_size
        self.stride = stride
        self.sigma = sigma

    def estimate_from_bboxes(
        self,
        bboxes: np.ndarray,
        image_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Create density map from bounding box annotations.

        Args:
            bboxes: Bounding boxes (N, 4) in xyxy format
            image_size: Image size (height, width)

        Returns:
            Density map (H, W) with object counts per pixel neighborhood
        """
        h, w = image_size
        density_map = np.zeros((h, w), dtype=np.float32)

        if len(bboxes) == 0:
            return density_map

        # Place Gaussian at center of each bbox
        for bbox in bboxes:
            x1, y1, x2, y2 = map(int, bbox)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Ensure within bounds
            cx = max(0, min(w - 1, cx))
            cy = max(0, min(h - 1, cy))

            density_map[cy, cx] += 1.0

        # Smooth with Gaussian
        density_map = gaussian_filter(density_map, sigma=self.sigma)

        return density_map

    def estimate_from_image(
        self,
        image: np.ndarray,
        edge_based: bool = True
    ) -> np.ndarray:
        """
        Estimate density directly from image (no annotations needed).

        Uses edge density as proxy for object density.

        Args:
            image: Input image
            edge_based: Use edge-based estimation

        Returns:
            Estimated density map
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        h, w = gray.shape

        if edge_based:
            # Edge-based density estimation
            edges = cv2.Canny(gray, 50, 150)

            # Compute local edge density using sliding window
            density_map = np.zeros((h, w), dtype=np.float32)

            for y in range(0, h - self.window_size + 1, self.stride):
                for x in range(0, w - self.window_size + 1, self.stride):
                    window = edges[y:y + self.window_size, x:x + self.window_size]
                    density = np.sum(window > 0) / window.size

                    # Fill the stride region
                    y_end = min(y + self.stride, h)
                    x_end = min(x + self.stride, w)
                    density_map[y:y_end, x:x_end] = density

            # Smooth
            density_map = gaussian_filter(density_map, sigma=self.sigma)

        else:
            # Texture-based density estimation using local variance
            from scipy.ndimage import uniform_filter

            local_mean = uniform_filter(gray.astype(np.float32), size=self.window_size)
            local_sq_mean = uniform_filter(gray.astype(np.float32)**2, size=self.window_size)
            local_var = local_sq_mean - local_mean**2
            density_map = local_var / (local_var.max() + 1e-6)

        return density_map

    def get_density_regions(
        self,
        density_map: np.ndarray,
        low_threshold: float = 0.3,
        high_threshold: float = 0.7
    ) -> Dict[str, np.ndarray]:
        """
        Segment density map into low/medium/high density regions.

        Args:
            density_map: Density map from estimate_*
            low_threshold: Threshold for low density
            high_threshold: Threshold for high density

        Returns:
            Dictionary of binary masks for each density level
        """
        normalized = density_map / (density_map.max() + 1e-6)

        return {
            'low': normalized < low_threshold,
            'medium': (normalized >= low_threshold) & (normalized < high_threshold),
            'high': normalized >= high_threshold
        }


class DensityAwareFeatures:
    """
    Novel density-aware feature representation.

    Creates features that encode local object density context,
    helping models adapt their behavior based on scene complexity.
    """

    def __init__(
        self,
        feature_scales: List[int] = [32, 64, 128],
        n_density_bins: int = 5
    ):
        """
        Initialize density-aware feature extractor.

        Args:
            feature_scales: Scales for multi-scale density features
            n_density_bins: Number of density level bins
        """
        self.feature_scales = feature_scales
        self.n_density_bins = n_density_bins
        self.density_estimator = DensityEstimator()

    def extract(
        self,
        image: np.ndarray,
        bboxes: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Extract density-aware features.

        Args:
            image: Input image
            bboxes: Optional bounding boxes for supervised density

        Returns:
            Dictionary of density features
        """
        h, w = image.shape[:2]

        features = {}

        # Multi-scale density maps
        for scale in self.feature_scales:
            estimator = DensityEstimator(window_size=scale, stride=scale // 2)

            if bboxes is not None:
                density = estimator.estimate_from_bboxes(bboxes, (h, w))
            else:
                density = estimator.estimate_from_image(image)

            # Downsample to fixed size
            density_resized = cv2.resize(density, (32, 32))
            features[f'density_scale_{scale}'] = density_resized

        # Global density statistics
        primary_density = features[f'density_scale_{self.feature_scales[0]}']
        features['global_stats'] = np.array([
            primary_density.mean(),
            primary_density.std(),
            primary_density.max(),
            np.percentile(primary_density, 90),
            (primary_density > primary_density.mean()).sum() / primary_density.size
        ])

        # Density histogram
        hist, _ = np.histogram(
            primary_density.flatten(),
            bins=self.n_density_bins,
            range=(0, primary_density.max() + 1e-6),
            density=True
        )
        features['density_histogram'] = hist

        return features

    def compute_density_attention(
        self,
        density_map: np.ndarray,
        feature_map_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Compute attention weights from density map.

        Higher density regions get higher attention.

        Args:
            density_map: Density map
            feature_map_size: Size to resize attention to

        Returns:
            Attention map (feature_map_size)
        """
        # Normalize density to [0, 1]
        density_norm = density_map / (density_map.max() + 1e-6)

        # Apply softmax-like transformation
        # Higher density = higher attention
        attention = np.exp(density_norm * 2)
        attention = attention / attention.sum()

        # Resize to feature map size
        attention = cv2.resize(attention, (feature_map_size[1], feature_map_size[0]))

        return attention


class MultiScaleDensityPyramid:
    """
    Multi-scale density pyramid for hierarchical density analysis.

    Inspired by image pyramids, creates density representations
    at multiple scales to capture both local and global density patterns.
    """

    def __init__(
        self,
        n_levels: int = 4,
        base_size: int = 256
    ):
        """
        Initialize multi-scale density pyramid.

        Args:
            n_levels: Number of pyramid levels
            base_size: Size of base (finest) level
        """
        self.n_levels = n_levels
        self.base_size = base_size

    def build_pyramid(
        self,
        image: np.ndarray,
        bboxes: Optional[np.ndarray] = None
    ) -> List[np.ndarray]:
        """
        Build density pyramid from image.

        Args:
            image: Input image
            bboxes: Optional bounding boxes

        Returns:
            List of density maps (coarse to fine)
        """
        h, w = image.shape[:2]
        pyramid = []

        for level in range(self.n_levels):
            scale = 2 ** level
            window_size = self.base_size // scale
            window_size = max(16, window_size)

            estimator = DensityEstimator(
                window_size=window_size,
                stride=window_size // 2
            )

            if bboxes is not None:
                density = estimator.estimate_from_bboxes(bboxes, (h, w))
            else:
                density = estimator.estimate_from_image(image)

            # Resize to pyramid level size
            level_size = self.base_size // (2 ** level)
            density_resized = cv2.resize(density, (level_size, level_size))

            pyramid.append(density_resized)

        return pyramid

    def get_hierarchical_features(
        self,
        pyramid: List[np.ndarray]
    ) -> np.ndarray:
        """
        Extract hierarchical features from density pyramid.

        Args:
            pyramid: Density pyramid from build_pyramid

        Returns:
            Feature vector encoding multi-scale density
        """
        features = []

        for level, density_map in enumerate(pyramid):
            # Statistics at each level
            features.extend([
                density_map.mean(),
                density_map.std(),
                density_map.max(),
                np.percentile(density_map, 75),
                np.percentile(density_map, 25)
            ])

            # Quadrant statistics (spatial structure)
            h, w = density_map.shape
            mid_h, mid_w = h // 2, w // 2

            quadrants = [
                density_map[:mid_h, :mid_w],    # top-left
                density_map[:mid_h, mid_w:],    # top-right
                density_map[mid_h:, :mid_w],    # bottom-left
                density_map[mid_h:, mid_w:]     # bottom-right
            ]

            for q in quadrants:
                features.append(q.mean())

        return np.array(features)
