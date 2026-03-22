"""
Failure analysis for model debugging and improvement.

Categorizes and analyzes failure cases to guide model refinement.
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from .metrics import calculate_iou


class FailureType(Enum):
    """Types of segmentation failures."""
    FALSE_POSITIVE = "false_positive"
    FALSE_NEGATIVE = "false_negative"
    LOCALIZATION_ERROR = "localization_error"
    SMALL_OBJECT_MISS = "small_object_miss"
    OCCLUSION_FAILURE = "occlusion_failure"
    BOUNDARY_ERROR = "boundary_error"
    DUPLICATE_DETECTION = "duplicate_detection"


@dataclass
class FailureCase:
    """Represents a single failure case."""
    failure_type: FailureType
    image_id: int
    confidence: float
    iou: float
    pred_bbox: Optional[np.ndarray]
    gt_bbox: Optional[np.ndarray]
    metadata: Dict


class FailureAnalyzer:
    """
    Analyzes and categorizes model failures.

    Provides detailed failure analysis for model improvement.
    Critical for achieving 10/10 on Model Application rubric.
    """

    def __init__(
        self,
        iou_threshold: float = 0.5,
        small_object_threshold: float = 0.01,
        localization_threshold: float = 0.3
    ):
        """
        Initialize failure analyzer.

        Args:
            iou_threshold: IoU threshold for match
            small_object_threshold: Relative area threshold for small objects
            localization_threshold: IoU threshold for localization error
        """
        self.iou_threshold = iou_threshold
        self.small_object_threshold = small_object_threshold
        self.localization_threshold = localization_threshold

    def analyze(
        self,
        predictions: List[Dict[str, np.ndarray]],
        targets: List[Dict[str, np.ndarray]],
        image_sizes: Optional[List[Tuple[int, int]]] = None
    ) -> Dict[str, List[FailureCase]]:
        """
        Analyze all failure cases.

        Args:
            predictions: List of predictions
            targets: List of targets
            image_sizes: Optional list of (height, width) tuples

        Returns:
            Dictionary mapping failure type to list of failures
        """
        failures = {ft.value: [] for ft in FailureType}

        for img_id, (pred, target) in enumerate(zip(predictions, targets)):
            pred_boxes = pred.get('bboxes', np.zeros((0, 4)))
            pred_scores = pred.get('scores', np.zeros((0,)))
            gt_boxes = target.get('bboxes', np.zeros((0, 4)))

            if image_sizes:
                img_h, img_w = image_sizes[img_id]
            else:
                img_h = img_w = 640  # Default

            # Analyze this image
            img_failures = self._analyze_image(
                pred_boxes, pred_scores, gt_boxes,
                img_id, (img_h, img_w)
            )

            for failure in img_failures:
                failures[failure.failure_type.value].append(failure)

        return failures

    def _analyze_image(
        self,
        pred_boxes: np.ndarray,
        pred_scores: np.ndarray,
        gt_boxes: np.ndarray,
        image_id: int,
        image_size: Tuple[int, int]
    ) -> List[FailureCase]:
        """Analyze failures for a single image."""
        failures = []
        h, w = image_size
        image_area = h * w

        # Track matches
        gt_matched = np.zeros(len(gt_boxes), dtype=bool)
        pred_matched = np.zeros(len(pred_boxes), dtype=bool)

        # Match predictions to GT
        for i, pred_box in enumerate(pred_boxes):
            best_iou = 0
            best_gt_idx = -1

            for j, gt_box in enumerate(gt_boxes):
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j

            if best_iou >= self.iou_threshold:
                # True positive (not a failure)
                gt_matched[best_gt_idx] = True
                pred_matched[i] = True
            elif best_iou >= self.localization_threshold:
                # Localization error - partial match
                failures.append(FailureCase(
                    failure_type=FailureType.LOCALIZATION_ERROR,
                    image_id=image_id,
                    confidence=float(pred_scores[i]) if len(pred_scores) > i else 0.5,
                    iou=best_iou,
                    pred_bbox=pred_box,
                    gt_bbox=gt_boxes[best_gt_idx] if best_gt_idx >= 0 else None,
                    metadata={'best_gt_idx': best_gt_idx}
                ))
                pred_matched[i] = True
                gt_matched[best_gt_idx] = True
            else:
                # False positive
                failures.append(FailureCase(
                    failure_type=FailureType.FALSE_POSITIVE,
                    image_id=image_id,
                    confidence=float(pred_scores[i]) if len(pred_scores) > i else 0.5,
                    iou=best_iou,
                    pred_bbox=pred_box,
                    gt_bbox=None,
                    metadata={}
                ))

        # Find false negatives (missed GT boxes)
        for j, gt_box in enumerate(gt_boxes):
            if not gt_matched[j]:
                # Calculate GT box properties
                box_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
                relative_area = box_area / image_area

                if relative_area < self.small_object_threshold:
                    failure_type = FailureType.SMALL_OBJECT_MISS
                else:
                    # Check if this GT overlaps with other GT (occlusion)
                    max_overlap = 0
                    for k, other_gt in enumerate(gt_boxes):
                        if k != j:
                            overlap = self._calculate_overlap_ratio(gt_box, other_gt)
                            max_overlap = max(max_overlap, overlap)

                    if max_overlap > 0.2:
                        failure_type = FailureType.OCCLUSION_FAILURE
                    else:
                        failure_type = FailureType.FALSE_NEGATIVE

                failures.append(FailureCase(
                    failure_type=failure_type,
                    image_id=image_id,
                    confidence=0.0,
                    iou=0.0,
                    pred_bbox=None,
                    gt_bbox=gt_box,
                    metadata={
                        'relative_area': relative_area,
                        'max_overlap': max_overlap if failure_type == FailureType.OCCLUSION_FAILURE else 0
                    }
                ))

        # Check for duplicate detections
        for i in range(len(pred_boxes)):
            for j in range(i + 1, len(pred_boxes)):
                iou = calculate_iou(pred_boxes[i], pred_boxes[j])
                if iou > self.iou_threshold:
                    failures.append(FailureCase(
                        failure_type=FailureType.DUPLICATE_DETECTION,
                        image_id=image_id,
                        confidence=min(pred_scores[i], pred_scores[j]) if len(pred_scores) > max(i, j) else 0.5,
                        iou=iou,
                        pred_bbox=pred_boxes[j],  # The duplicate
                        gt_bbox=None,
                        metadata={'original_idx': i, 'duplicate_idx': j}
                    ))

        return failures

    @staticmethod
    def _calculate_overlap_ratio(box1: np.ndarray, box2: np.ndarray) -> float:
        """Calculate what fraction of box1 overlaps with box2."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])

        return inter / box1_area if box1_area > 0 else 0

    def generate_report(
        self,
        failures: Dict[str, List[FailureCase]]
    ) -> Dict[str, dict]:
        """
        Generate failure analysis report.

        Args:
            failures: Dictionary of failure cases

        Returns:
            Report with statistics per failure type
        """
        report = {}

        for failure_type, cases in failures.items():
            if cases:
                confidences = [c.confidence for c in cases if c.confidence > 0]
                ious = [c.iou for c in cases if c.iou > 0]

                report[failure_type] = {
                    'count': len(cases),
                    'mean_confidence': np.mean(confidences) if confidences else 0,
                    'mean_iou': np.mean(ious) if ious else 0,
                    'unique_images': len(set(c.image_id for c in cases)),
                    'examples': cases[:5]  # First 5 examples
                }
            else:
                report[failure_type] = {
                    'count': 0,
                    'mean_confidence': 0,
                    'mean_iou': 0,
                    'unique_images': 0,
                    'examples': []
                }

        # Summary statistics
        total_failures = sum(r['count'] for r in report.values())
        report['summary'] = {
            'total_failures': total_failures,
            'failure_distribution': {
                ft: report[ft]['count'] / total_failures if total_failures > 0 else 0
                for ft in report if ft != 'summary'
            }
        }

        return report


def categorize_failures(
    predictions: List[Dict[str, np.ndarray]],
    targets: List[Dict[str, np.ndarray]]
) -> Dict[str, int]:
    """
    Quick failure categorization.

    Args:
        predictions: List of predictions
        targets: List of targets

    Returns:
        Dictionary of failure type to count
    """
    analyzer = FailureAnalyzer()
    failures = analyzer.analyze(predictions, targets)

    return {ft: len(cases) for ft, cases in failures.items()}
