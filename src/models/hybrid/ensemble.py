"""
Ensemble segmentation combining multiple models.

Uses weighted voting and confidence calibration for optimal results.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union


class WeightedVotingEnsemble:
    """
    Ensemble that combines predictions using weighted voting.

    Aggregates masks from multiple models with learned weights.
    """

    def __init__(
        self,
        model_weights: Optional[Dict[str, float]] = None,
        iou_threshold: float = 0.5,
        min_votes: int = 2
    ):
        """
        Initialize weighted voting ensemble.

        Args:
            model_weights: Weights for each model (default: equal weights)
            iou_threshold: IoU threshold for matching predictions
            min_votes: Minimum votes needed to keep a prediction
        """
        self.model_weights = model_weights or {}
        self.iou_threshold = iou_threshold
        self.min_votes = min_votes

    def combine(
        self,
        predictions: Dict[str, Dict[str, np.ndarray]],
        method: str = 'weighted_boxes'
    ) -> Dict[str, np.ndarray]:
        """
        Combine predictions from multiple models.

        Args:
            predictions: Dictionary mapping model names to their predictions
            method: Combination method ('weighted_boxes', 'soft_nms', 'mask_voting')

        Returns:
            Combined predictions
        """
        if method == 'weighted_boxes':
            return self._weighted_boxes_fusion(predictions)
        elif method == 'soft_nms':
            return self._soft_nms_fusion(predictions)
        elif method == 'mask_voting':
            return self._mask_voting(predictions)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _weighted_boxes_fusion(
        self,
        predictions: Dict[str, Dict[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """
        Weighted Boxes Fusion for combining detections.

        Clusters similar boxes and computes weighted average.
        """
        all_boxes = []
        all_scores = []
        all_masks = []
        all_model_weights = []

        for model_name, pred in predictions.items():
            weight = self.model_weights.get(model_name, 1.0)

            for i in range(len(pred['bboxes'])):
                all_boxes.append(pred['bboxes'][i])
                all_scores.append(pred['scores'][i] * weight)
                all_masks.append(pred['masks'][i] if len(pred['masks']) > i else None)
                all_model_weights.append(weight)

        if not all_boxes:
            return {
                'masks': np.zeros((0,)),
                'bboxes': np.zeros((0, 4)),
                'scores': np.zeros((0,))
            }

        boxes = np.array(all_boxes)
        scores = np.array(all_scores)

        # Cluster similar boxes
        clusters = self._cluster_boxes(boxes, self.iou_threshold)

        # Aggregate clusters
        final_boxes = []
        final_scores = []
        final_masks = []

        for cluster_indices in clusters:
            if len(cluster_indices) >= self.min_votes:
                # Weighted average of boxes
                cluster_boxes = boxes[cluster_indices]
                cluster_scores = scores[cluster_indices]
                cluster_weights = np.array([all_model_weights[i] for i in cluster_indices])

                # Normalize weights
                weights = cluster_scores * cluster_weights
                weights /= weights.sum()

                # Weighted box
                fused_box = np.average(cluster_boxes, axis=0, weights=weights)
                fused_score = np.average(cluster_scores, weights=cluster_weights)

                final_boxes.append(fused_box)
                final_scores.append(fused_score)

                # Use mask from highest confidence prediction
                best_idx = cluster_indices[np.argmax(cluster_scores)]
                if all_masks[best_idx] is not None:
                    final_masks.append(all_masks[best_idx])

        return {
            'masks': np.array(final_masks) if final_masks else np.zeros((0,)),
            'bboxes': np.array(final_boxes) if final_boxes else np.zeros((0, 4)),
            'scores': np.array(final_scores) if final_scores else np.zeros((0,))
        }

    def _cluster_boxes(
        self,
        boxes: np.ndarray,
        iou_threshold: float
    ) -> List[List[int]]:
        """Cluster boxes based on IoU."""
        n = len(boxes)
        if n == 0:
            return []

        # Compute pairwise IoU
        clusters = []
        used = set()

        for i in range(n):
            if i in used:
                continue

            cluster = [i]
            used.add(i)

            for j in range(i + 1, n):
                if j in used:
                    continue

                iou = self._compute_iou(boxes[i], boxes[j])
                if iou >= iou_threshold:
                    cluster.append(j)
                    used.add(j)

            clusters.append(cluster)

        return clusters

    @staticmethod
    def _compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute IoU between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter

        return inter / union if union > 0 else 0

    def _soft_nms_fusion(
        self,
        predictions: Dict[str, Dict[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """Combine using Soft-NMS across all predictions."""
        all_boxes = []
        all_scores = []
        all_masks = []

        for model_name, pred in predictions.items():
            weight = self.model_weights.get(model_name, 1.0)

            for i in range(len(pred['bboxes'])):
                all_boxes.append(pred['bboxes'][i])
                all_scores.append(pred['scores'][i] * weight)
                if len(pred['masks']) > i:
                    all_masks.append(pred['masks'][i])

        if not all_boxes:
            return {
                'masks': np.zeros((0,)),
                'bboxes': np.zeros((0, 4)),
                'scores': np.zeros((0,))
            }

        # Apply Soft-NMS
        boxes = np.array(all_boxes)
        scores = np.array(all_scores)
        keep = self._soft_nms(boxes, scores)

        return {
            'masks': np.array([all_masks[i] for i in keep]) if all_masks else np.zeros((0,)),
            'bboxes': boxes[keep],
            'scores': scores[keep]
        }

    def _soft_nms(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        sigma: float = 0.5
    ) -> List[int]:
        """Apply Soft-NMS."""
        N = len(boxes)
        indices = list(range(N))
        keep = []

        while indices:
            max_idx = max(indices, key=lambda i: scores[i])
            keep.append(max_idx)
            indices.remove(max_idx)

            if not indices:
                break

            max_box = boxes[max_idx]
            for idx in indices[:]:
                iou = self._compute_iou(max_box, boxes[idx])
                scores[idx] *= np.exp(-(iou ** 2) / sigma)
                if scores[idx] < 0.01:
                    indices.remove(idx)

        return keep

    def _mask_voting(
        self,
        predictions: Dict[str, Dict[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """Pixel-wise mask voting."""
        # Get image shape from first prediction
        first_pred = next(iter(predictions.values()))
        if len(first_pred['masks']) == 0:
            return {
                'masks': np.zeros((0,)),
                'bboxes': np.zeros((0, 4)),
                'scores': np.zeros((0,))
            }

        h, w = first_pred['masks'][0].shape

        # Aggregate all masks
        all_masks_by_region = []

        # Use WBF to cluster, then vote on masks
        combined = self._weighted_boxes_fusion(predictions)

        return combined


class EnsembleSegmenter:
    """
    Full ensemble segmenter combining multiple models.

    Manages multiple models and provides unified inference.
    """

    def __init__(
        self,
        models: Dict[str, object],
        weights: Optional[Dict[str, float]] = None,
        combination_method: str = 'weighted_boxes'
    ):
        """
        Initialize ensemble segmenter.

        Args:
            models: Dictionary of model name to model instance
            weights: Model weights for combination
            combination_method: Method for combining predictions
        """
        self.models = models
        self.weights = weights or {name: 1.0 for name in models}
        self.combination_method = combination_method
        self.ensemble = WeightedVotingEnsemble(
            model_weights=self.weights,
            min_votes=1
        )

    def predict(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Run ensemble prediction.

        Args:
            image: Input image

        Returns:
            Combined predictions
        """
        predictions = {}

        for name, model in self.models.items():
            try:
                pred = model.predict(image)
                predictions[name] = pred
            except Exception as e:
                print(f"Error in model {name}: {e}")
                continue

        if not predictions:
            h, w = image.shape[:2]
            return {
                'masks': np.zeros((0, h, w)),
                'bboxes': np.zeros((0, 4)),
                'scores': np.zeros((0,))
            }

        return self.ensemble.combine(predictions, self.combination_method)

    def calibrate_weights(
        self,
        val_images: List[np.ndarray],
        val_targets: List[Dict[str, np.ndarray]]
    ) -> None:
        """
        Calibrate model weights based on validation performance.

        Args:
            val_images: Validation images
            val_targets: Validation targets
        """
        from ..evaluation.metrics import calculate_map

        model_maps = {}

        for name, model in self.models.items():
            all_preds = []
            for img in val_images:
                pred = model.predict(img)
                all_preds.append(pred)

            # Calculate mAP for this model
            mAP = calculate_map(all_preds, val_targets)
            model_maps[name] = mAP

        # Normalize to weights
        total = sum(model_maps.values())
        self.weights = {name: mAP / total for name, mAP in model_maps.items()}
        self.ensemble.model_weights = self.weights

        print("Calibrated weights:", self.weights)
