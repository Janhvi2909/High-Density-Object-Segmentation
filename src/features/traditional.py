"""
Traditional feature extraction methods.

Implements classical computer vision features: HOG, SIFT, edge descriptors.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from skimage.feature import hog
from skimage import exposure


class HOGExtractor:
    """
    Histogram of Oriented Gradients (HOG) feature extractor.

    HOG captures edge and gradient structure, useful for shape detection.
    """

    def __init__(
        self,
        orientations: int = 9,
        pixels_per_cell: Tuple[int, int] = (8, 8),
        cells_per_block: Tuple[int, int] = (2, 2),
        transform_sqrt: bool = True,
        block_norm: str = 'L2-Hys'
    ):
        """
        Initialize HOG extractor.

        Args:
            orientations: Number of orientation bins
            pixels_per_cell: Size of a cell in pixels
            cells_per_block: Number of cells in each block
            transform_sqrt: Apply power law compression
            block_norm: Block normalization method
        """
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.transform_sqrt = transform_sqrt
        self.block_norm = block_norm

    def extract(
        self,
        image: np.ndarray,
        visualize: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Extract HOG features from an image.

        Args:
            image: Input image (grayscale or RGB)
            visualize: Whether to return visualization

        Returns:
            HOG feature vector, optionally with visualization
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        return hog(
            gray,
            orientations=self.orientations,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block,
            transform_sqrt=self.transform_sqrt,
            block_norm=self.block_norm,
            visualize=visualize,
            feature_vector=True
        )

    def extract_from_region(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        target_size: Tuple[int, int] = (64, 128)
    ) -> np.ndarray:
        """
        Extract HOG features from a specific region.

        Args:
            image: Full image
            bbox: Bounding box (x1, y1, x2, y2)
            target_size: Resize region to this size

        Returns:
            HOG feature vector for the region
        """
        x1, y1, x2, y2 = map(int, bbox)
        region = image[y1:y2, x1:x2]

        if region.size == 0:
            return np.zeros(self._get_feature_size(target_size))

        # Resize to fixed size for consistent features
        region = cv2.resize(region, target_size)

        return self.extract(region, visualize=False)

    def _get_feature_size(self, image_size: Tuple[int, int]) -> int:
        """Calculate expected feature vector size."""
        h, w = image_size
        n_cells_x = w // self.pixels_per_cell[0]
        n_cells_y = h // self.pixels_per_cell[1]
        n_blocks_x = n_cells_x - self.cells_per_block[0] + 1
        n_blocks_y = n_cells_y - self.cells_per_block[1] + 1
        return (n_blocks_x * n_blocks_y *
                self.cells_per_block[0] * self.cells_per_block[1] *
                self.orientations)


class SIFTExtractor:
    """
    Scale-Invariant Feature Transform (SIFT) extractor.

    SIFT provides robust keypoint detection and description.
    """

    def __init__(
        self,
        n_features: int = 500,
        n_octave_layers: int = 3,
        contrast_threshold: float = 0.04,
        edge_threshold: float = 10,
        sigma: float = 1.6
    ):
        """
        Initialize SIFT extractor.

        Args:
            n_features: Maximum number of features to retain
            n_octave_layers: Number of layers in each octave
            contrast_threshold: Contrast threshold for filtering
            edge_threshold: Edge threshold for filtering
            sigma: Sigma for Gaussian blur
        """
        self.sift = cv2.SIFT_create(
            nfeatures=n_features,
            nOctaveLayers=n_octave_layers,
            contrastThreshold=contrast_threshold,
            edgeThreshold=edge_threshold,
            sigma=sigma
        )
        self.n_features = n_features

    def extract(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[List, np.ndarray]:
        """
        Extract SIFT keypoints and descriptors.

        Args:
            image: Input image
            mask: Optional mask for region of interest

        Returns:
            Tuple of (keypoints, descriptors)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        keypoints, descriptors = self.sift.detectAndCompute(gray, mask)

        if descriptors is None:
            descriptors = np.zeros((0, 128))

        return keypoints, descriptors

    def compute_bag_of_words(
        self,
        descriptors: np.ndarray,
        vocabulary: np.ndarray
    ) -> np.ndarray:
        """
        Compute Bag of Visual Words histogram.

        Args:
            descriptors: SIFT descriptors (N, 128)
            vocabulary: Visual vocabulary (K, 128)

        Returns:
            Histogram of visual word frequencies (K,)
        """
        if len(descriptors) == 0:
            return np.zeros(len(vocabulary))

        # Assign each descriptor to nearest visual word
        from scipy.spatial.distance import cdist
        distances = cdist(descriptors, vocabulary)
        assignments = np.argmin(distances, axis=1)

        # Create histogram
        histogram = np.bincount(assignments, minlength=len(vocabulary))
        histogram = histogram.astype(np.float32)

        # L2 normalize
        norm = np.linalg.norm(histogram)
        if norm > 0:
            histogram /= norm

        return histogram


class EdgeFeatureExtractor:
    """
    Edge-based feature extractor.

    Extracts edge density, orientation histograms, and structure features.
    """

    def __init__(
        self,
        canny_low: int = 50,
        canny_high: int = 150,
        n_orientation_bins: int = 8
    ):
        """
        Initialize edge feature extractor.

        Args:
            canny_low: Canny edge detector low threshold
            canny_high: Canny edge detector high threshold
            n_orientation_bins: Number of edge orientation bins
        """
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.n_orientation_bins = n_orientation_bins

    def extract(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract comprehensive edge features.

        Args:
            image: Input image

        Returns:
            Dictionary of edge features
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        # Compute gradients
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # Gradient magnitude and orientation
        magnitude = np.sqrt(gx**2 + gy**2)
        orientation = np.arctan2(gy, gx)

        # Canny edges
        edges = cv2.Canny(gray, self.canny_low, self.canny_high)

        # Edge density (fraction of edge pixels)
        edge_density = np.sum(edges > 0) / edges.size

        # Orientation histogram (weighted by magnitude)
        hist, _ = np.histogram(
            orientation.flatten(),
            bins=self.n_orientation_bins,
            range=(-np.pi, np.pi),
            weights=magnitude.flatten(),
            density=True
        )

        # Edge length statistics
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        edge_lengths = [cv2.arcLength(c, False) for c in contours]

        return {
            'edge_density': np.array([edge_density]),
            'orientation_histogram': hist,
            'mean_edge_length': np.array([np.mean(edge_lengths) if edge_lengths else 0]),
            'num_contours': np.array([len(contours)]),
            'magnitude_mean': np.array([magnitude.mean()]),
            'magnitude_std': np.array([magnitude.std()])
        }

    def extract_local_edge_map(
        self,
        image: np.ndarray,
        grid_size: Tuple[int, int] = (8, 8)
    ) -> np.ndarray:
        """
        Compute local edge density map.

        Divides image into grid and computes edge density per cell.

        Args:
            image: Input image
            grid_size: Number of grid cells (rows, cols)

        Returns:
            Edge density map (grid_size)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        edges = cv2.Canny(gray, self.canny_low, self.canny_high)

        h, w = edges.shape
        cell_h, cell_w = h // grid_size[0], w // grid_size[1]

        density_map = np.zeros(grid_size)

        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                y1, y2 = i * cell_h, (i + 1) * cell_h
                x1, x2 = j * cell_w, (j + 1) * cell_w
                cell = edges[y1:y2, x1:x2]
                density_map[i, j] = np.sum(cell > 0) / cell.size

        return density_map
