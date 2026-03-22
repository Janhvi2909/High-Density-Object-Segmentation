"""
GrabCut-based segmentation.

Interactive/automatic foreground extraction using graph cuts.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional


class GrabCutSegmenter:
    """
    Segmentation using GrabCut algorithm.

    Uses graph cuts for foreground/background separation.
    Expected performance: Low-Medium (20-30% mAP)
    """

    def __init__(
        self,
        iter_count: int = 5,
        min_area: int = 50,
        max_area: int = 50000
    ):
        """
        Initialize GrabCut segmenter.

        Args:
            iter_count: Number of GrabCut iterations
            min_area: Minimum segment area
            max_area: Maximum segment area
        """
        self.iter_count = iter_count
        self.min_area = min_area
        self.max_area = max_area

    def segment_with_rect(
        self,
        image: np.ndarray,
        rect: Tuple[int, int, int, int]
    ) -> Dict[str, np.ndarray]:
        """
        Segment using rectangle initialization.

        Args:
            image: Input RGB image
            rect: Rectangle (x, y, width, height) containing object

        Returns:
            Segmentation result for single object
        """
        if len(image.shape) != 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        h, w = image.shape[:2]

        # Initialize mask
        mask = np.zeros((h, w), dtype=np.uint8)

        # Background/foreground models
        bgd_model = np.zeros((1, 65), dtype=np.float64)
        fgd_model = np.zeros((1, 65), dtype=np.float64)

        # Run GrabCut
        cv2.grabCut(
            image, mask, rect,
            bgd_model, fgd_model,
            self.iter_count,
            cv2.GC_INIT_WITH_RECT
        )

        # Extract foreground mask
        fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0)
        fg_mask = fg_mask.astype(np.uint8)

        # Get bounding box
        coords = np.where(fg_mask > 0)
        if len(coords[0]) > 0:
            y1, y2 = coords[0].min(), coords[0].max()
            x1, x2 = coords[1].min(), coords[1].max()
            bbox = np.array([[x1, y1, x2, y2]])
            score = np.array([1.0])
        else:
            bbox = np.zeros((0, 4))
            score = np.zeros((0,))

        return {
            'masks': fg_mask[np.newaxis, ...] if fg_mask.any() else np.zeros((0, h, w)),
            'bboxes': bbox,
            'scores': score
        }

    def segment_automatic(
        self,
        image: np.ndarray,
        edge_margin: float = 0.1
    ) -> Dict[str, np.ndarray]:
        """
        Automatic segmentation using edge-based initialization.

        Args:
            image: Input image
            edge_margin: Margin from edges to initialize rectangle

        Returns:
            Segmentation results
        """
        if len(image.shape) != 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        h, w = image.shape[:2]

        # Use edge detection to find potential object regions
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Dilate edges to connect nearby edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        edges_dilated = cv2.dilate(edges, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(
            edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        all_masks = []
        all_bboxes = []
        all_scores = []

        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            area = cw * ch

            if self.min_area <= area <= self.max_area:
                # Add margin to rectangle
                margin_x = int(cw * edge_margin)
                margin_y = int(ch * edge_margin)
                rect = (
                    max(0, x - margin_x),
                    max(0, y - margin_y),
                    min(w - x + margin_x, cw + 2 * margin_x),
                    min(h - y + margin_y, ch + 2 * margin_y)
                )

                try:
                    result = self.segment_with_rect(image, rect)
                    if len(result['bboxes']) > 0:
                        all_masks.append(result['masks'][0])
                        all_bboxes.append(result['bboxes'][0])
                        all_scores.append(result['scores'][0])
                except cv2.error:
                    continue

        if not all_bboxes:
            return {
                'masks': np.zeros((0, h, w)),
                'bboxes': np.zeros((0, 4)),
                'scores': np.zeros((0,))
            }

        return {
            'masks': np.array(all_masks),
            'bboxes': np.array(all_bboxes),
            'scores': np.array(all_scores)
        }

    def refine_mask(
        self,
        image: np.ndarray,
        initial_mask: np.ndarray,
        iter_count: int = 3
    ) -> np.ndarray:
        """
        Refine an initial mask using GrabCut.

        Args:
            image: Input image
            initial_mask: Initial binary mask
            iter_count: Number of iterations

        Returns:
            Refined mask
        """
        if len(image.shape) != 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        h, w = image.shape[:2]

        # Initialize GrabCut mask
        gc_mask = np.zeros((h, w), dtype=np.uint8)
        gc_mask[initial_mask > 0] = cv2.GC_PR_FGD
        gc_mask[initial_mask == 0] = cv2.GC_PR_BGD

        # Background/foreground models
        bgd_model = np.zeros((1, 65), dtype=np.float64)
        fgd_model = np.zeros((1, 65), dtype=np.float64)

        # Run GrabCut
        cv2.grabCut(
            image, gc_mask, None,
            bgd_model, fgd_model,
            iter_count,
            cv2.GC_INIT_WITH_MASK
        )

        # Extract refined mask
        refined = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0)
        return refined.astype(np.uint8)
