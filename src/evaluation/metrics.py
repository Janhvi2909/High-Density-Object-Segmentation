"""
Evaluation metrics for object segmentation.

Implements mAP, IoU, precision, recall, and F1 score.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union


def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Calculate Intersection over Union between two boxes.

    Args:
        box1: First box [x1, y1, x2, y2]
        box2: Second box [x1, y1, x2, y2]

    Returns:
        IoU value in [0, 1]
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


def calculate_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Calculate IoU between two binary masks.

    Args:
        mask1: First binary mask
        mask2: Second binary mask

    Returns:
        Mask IoU value
    """
    intersection = np.logical_and(mask1 > 0, mask2 > 0).sum()
    union = np.logical_or(mask1 > 0, mask2 > 0).sum()

    return intersection / union if union > 0 else 0


def calculate_precision_recall(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray,
    iou_threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate precision and recall at different score thresholds.

    Args:
        pred_boxes: Predicted boxes (N, 4)
        pred_scores: Prediction scores (N,)
        gt_boxes: Ground truth boxes (M, 4)
        iou_threshold: IoU threshold for matching

    Returns:
        Tuple of (precision, recall) arrays
    """
    if len(pred_boxes) == 0:
        return np.array([0]), np.array([0])

    if len(gt_boxes) == 0:
        return np.array([0] * len(pred_boxes)), np.array([0] * len(pred_boxes))

    # Sort by score
    sorted_indices = np.argsort(pred_scores)[::-1]
    pred_boxes = pred_boxes[sorted_indices]
    pred_scores = pred_scores[sorted_indices]

    # Track which GT boxes have been matched
    gt_matched = np.zeros(len(gt_boxes), dtype=bool)

    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))

    for i, pred_box in enumerate(pred_boxes):
        best_iou = 0
        best_gt_idx = -1

        for j, gt_box in enumerate(gt_boxes):
            if gt_matched[j]:
                continue

            iou = calculate_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j

        if best_iou >= iou_threshold:
            tp[i] = 1
            gt_matched[best_gt_idx] = True
        else:
            fp[i] = 1

    # Cumulative sums
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    # Precision and recall
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    recall = tp_cumsum / len(gt_boxes)

    return precision, recall


def calculate_ap(precision: np.ndarray, recall: np.ndarray) -> float:
    """
    Calculate Average Precision using 11-point interpolation.

    Args:
        precision: Precision values
        recall: Recall values

    Returns:
        Average Precision value
    """
    # Add sentinel values
    precision = np.concatenate([[0], precision, [0]])
    recall = np.concatenate([[0], recall, [1]])

    # Make precision monotonically decreasing
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    # Find points where recall changes
    recall_change = recall[1:] != recall[:-1]

    # Sum areas under curve
    ap = np.sum((recall[1:] - recall[:-1]) * precision[1:])

    return ap


def calculate_map(
    predictions: List[Dict[str, np.ndarray]],
    targets: List[Dict[str, np.ndarray]],
    iou_thresholds: List[float] = [0.5]
) -> float:
    """
    Calculate mean Average Precision across images.

    Args:
        predictions: List of prediction dictionaries
        targets: List of target dictionaries
        iou_thresholds: IoU thresholds to evaluate

    Returns:
        mAP value
    """
    aps = []

    for iou_thresh in iou_thresholds:
        all_precision = []
        all_recall = []

        for pred, target in zip(predictions, targets):
            pred_boxes = pred.get('bboxes', np.zeros((0, 4)))
            pred_scores = pred.get('scores', np.zeros((0,)))
            gt_boxes = target.get('bboxes', np.zeros((0, 4)))

            if len(pred_boxes) > 0 and len(gt_boxes) > 0:
                precision, recall = calculate_precision_recall(
                    pred_boxes, pred_scores, gt_boxes, iou_thresh
                )
                ap = calculate_ap(precision, recall)
                aps.append(ap)
            elif len(gt_boxes) == 0 and len(pred_boxes) == 0:
                aps.append(1.0)  # Perfect: no GT, no predictions
            elif len(gt_boxes) == 0:
                aps.append(0.0)  # False positives
            else:
                aps.append(0.0)  # Missed detections

    return np.mean(aps) if aps else 0.0


def calculate_f1_score(
    predictions: Dict[str, np.ndarray],
    targets: Dict[str, np.ndarray],
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate F1 score, precision, and recall.

    Args:
        predictions: Prediction dictionary
        targets: Target dictionary
        iou_threshold: IoU threshold for matching

    Returns:
        Dictionary with precision, recall, and F1 score
    """
    pred_boxes = predictions.get('bboxes', np.zeros((0, 4)))
    pred_scores = predictions.get('scores', np.zeros((0,)))
    gt_boxes = targets.get('bboxes', np.zeros((0, 4)))

    if len(pred_boxes) == 0 and len(gt_boxes) == 0:
        return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}

    if len(pred_boxes) == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    if len(gt_boxes) == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    precision, recall = calculate_precision_recall(
        pred_boxes, pred_scores, gt_boxes, iou_threshold
    )

    final_precision = precision[-1] if len(precision) > 0 else 0
    final_recall = recall[-1] if len(recall) > 0 else 0

    f1 = 2 * final_precision * final_recall / (final_precision + final_recall + 1e-6)

    return {
        'precision': final_precision,
        'recall': final_recall,
        'f1': f1
    }


class SegmentationMetrics:
    """
    Comprehensive metrics calculator for segmentation evaluation.

    Computes all standard metrics and provides detailed analysis.
    """

    def __init__(
        self,
        iou_thresholds: List[float] = [0.5, 0.75],
        use_mask_iou: bool = True
    ):
        """
        Initialize metrics calculator.

        Args:
            iou_thresholds: IoU thresholds for evaluation
            use_mask_iou: Use mask IoU instead of box IoU
        """
        self.iou_thresholds = iou_thresholds
        self.use_mask_iou = use_mask_iou

    def evaluate(
        self,
        predictions: List[Dict[str, np.ndarray]],
        targets: List[Dict[str, np.ndarray]]
    ) -> Dict[str, float]:
        """
        Evaluate predictions against targets.

        Args:
            predictions: List of prediction dictionaries
            targets: List of target dictionaries

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # mAP at different thresholds
        for thresh in self.iou_thresholds:
            mAP = calculate_map(predictions, targets, [thresh])
            metrics[f'mAP@{thresh}'] = mAP

        # mAP@0.5:0.95 (COCO-style)
        thresholds = np.arange(0.5, 0.96, 0.05)
        mAP_coco = calculate_map(predictions, targets, thresholds.tolist())
        metrics['mAP@0.5:0.95'] = mAP_coco

        # Per-image F1 scores
        f1_scores = []
        precisions = []
        recalls = []

        for pred, target in zip(predictions, targets):
            result = calculate_f1_score(pred, target, iou_threshold=0.5)
            f1_scores.append(result['f1'])
            precisions.append(result['precision'])
            recalls.append(result['recall'])

        metrics['mean_f1'] = np.mean(f1_scores)
        metrics['mean_precision'] = np.mean(precisions)
        metrics['mean_recall'] = np.mean(recalls)

        # Detection rate (images with any correct detection)
        detection_rate = sum(1 for f1 in f1_scores if f1 > 0) / len(f1_scores)
        metrics['detection_rate'] = detection_rate

        return metrics

    def evaluate_single(
        self,
        prediction: Dict[str, np.ndarray],
        target: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        Evaluate a single image.

        Args:
            prediction: Prediction dictionary
            target: Target dictionary

        Returns:
            Per-image metrics
        """
        return calculate_f1_score(prediction, target, iou_threshold=0.5)

    def generate_confusion_matrix_data(
        self,
        predictions: List[Dict[str, np.ndarray]],
        targets: List[Dict[str, np.ndarray]],
        iou_threshold: float = 0.5
    ) -> Dict[str, int]:
        """
        Generate confusion matrix data.

        Args:
            predictions: List of predictions
            targets: List of targets
            iou_threshold: IoU threshold

        Returns:
            Dictionary with TP, FP, FN counts
        """
        total_tp = 0
        total_fp = 0
        total_fn = 0

        for pred, target in zip(predictions, targets):
            pred_boxes = pred.get('bboxes', np.zeros((0, 4)))
            gt_boxes = target.get('bboxes', np.zeros((0, 4)))

            gt_matched = np.zeros(len(gt_boxes), dtype=bool)

            for pred_box in pred_boxes:
                matched = False
                for j, gt_box in enumerate(gt_boxes):
                    if not gt_matched[j]:
                        iou = calculate_iou(pred_box, gt_box)
                        if iou >= iou_threshold:
                            total_tp += 1
                            gt_matched[j] = True
                            matched = True
                            break

                if not matched:
                    total_fp += 1

            total_fn += np.sum(~gt_matched)

        return {
            'tp': total_tp,
            'fp': total_fp,
            'fn': total_fn
        }
