"""
Mask R-CNN for instance segmentation.

High-performance deep learning model for object detection and segmentation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

try:
    import torchvision
    from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False


def build_maskrcnn(
    num_classes: int = 2,
    pretrained: bool = True,
    trainable_backbone_layers: int = 3
) -> nn.Module:
    """
    Build Mask R-CNN model with custom number of classes.

    Args:
        num_classes: Number of classes including background
        pretrained: Use pretrained backbone
        trainable_backbone_layers: Number of trainable backbone layers

    Returns:
        Mask R-CNN model
    """
    if not TORCHVISION_AVAILABLE:
        raise ImportError("torchvision is required for Mask R-CNN")

    # Load pretrained model
    if pretrained:
        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        model = maskrcnn_resnet50_fpn(
            weights=weights,
            trainable_backbone_layers=trainable_backbone_layers
        )
    else:
        model = maskrcnn_resnet50_fpn(
            weights=None,
            trainable_backbone_layers=trainable_backbone_layers
        )

    # Replace classification head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Replace mask head
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    return model


class MaskRCNNSegmenter:
    """
    Wrapper for Mask R-CNN inference and training.

    Expected performance: High (55-70% mAP)
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        checkpoint_path: Optional[str] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.3
    ):
        """
        Initialize Mask R-CNN segmenter.

        Args:
            num_classes: Number of classes
            pretrained: Use pretrained weights
            checkpoint_path: Path to fine-tuned checkpoint
            device: Device to use
            conf_threshold: Confidence threshold
            nms_threshold: NMS IoU threshold
        """
        self.device = device
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.num_classes = num_classes

        # Build model
        self.model = build_maskrcnn(num_classes, pretrained)

        # Load checkpoint if provided
        if checkpoint_path and Path(checkpoint_path).exists():
            state_dict = torch.load(checkpoint_path, map_location=device)
            self.model.load_state_dict(state_dict)

        self.model.to(device)
        self.model.eval()

    def predict(
        self,
        image: Union[np.ndarray, torch.Tensor],
        return_features: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Run inference on a single image.

        Args:
            image: Input image (HWC numpy or CHW tensor)
            return_features: Return intermediate features

        Returns:
            Dictionary with masks, bboxes, scores, and labels
        """
        # Preprocess
        if isinstance(image, np.ndarray):
            # Normalize to [0, 1]
            if image.max() > 1:
                image = image.astype(np.float32) / 255.0
            # HWC to CHW
            image = torch.from_numpy(image).permute(2, 0, 1)

        image = image.to(self.device)

        # Run inference
        with torch.no_grad():
            predictions = self.model([image])[0]

        # Extract results
        boxes = predictions['boxes'].cpu().numpy()
        labels = predictions['labels'].cpu().numpy()
        scores = predictions['scores'].cpu().numpy()
        masks = predictions['masks'].cpu().numpy()

        # Filter by confidence
        keep = scores >= self.conf_threshold
        boxes = boxes[keep]
        labels = labels[keep]
        scores = scores[keep]
        masks = masks[keep]

        # Apply NMS
        if len(boxes) > 0:
            keep_indices = self._soft_nms(boxes, scores, self.nms_threshold)
            boxes = boxes[keep_indices]
            labels = labels[keep_indices]
            scores = scores[keep_indices]
            masks = masks[keep_indices]

        # Threshold masks to binary
        masks = (masks > 0.5).astype(np.uint8).squeeze(1)

        return {
            'masks': masks,
            'bboxes': boxes,
            'scores': scores,
            'labels': labels
        }

    def predict_batch(
        self,
        images: List[Union[np.ndarray, torch.Tensor]]
    ) -> List[Dict[str, np.ndarray]]:
        """
        Run inference on a batch of images.

        Args:
            images: List of images

        Returns:
            List of prediction dictionaries
        """
        return [self.predict(img) for img in images]

    @staticmethod
    def _soft_nms(
        boxes: np.ndarray,
        scores: np.ndarray,
        iou_threshold: float,
        sigma: float = 0.5
    ) -> List[int]:
        """
        Apply Soft-NMS for better handling of overlapping objects.

        Args:
            boxes: Bounding boxes (N, 4)
            scores: Confidence scores (N,)
            iou_threshold: IoU threshold
            sigma: Gaussian decay parameter

        Returns:
            Indices to keep
        """
        N = len(boxes)
        indices = list(range(N))
        keep = []

        while indices:
            # Get highest scoring box
            max_idx = max(indices, key=lambda i: scores[i])
            keep.append(max_idx)
            indices.remove(max_idx)

            if not indices:
                break

            # Calculate IoU with remaining boxes
            max_box = boxes[max_idx]
            remaining_boxes = boxes[indices]

            x1 = np.maximum(max_box[0], remaining_boxes[:, 0])
            y1 = np.maximum(max_box[1], remaining_boxes[:, 1])
            x2 = np.minimum(max_box[2], remaining_boxes[:, 2])
            y2 = np.minimum(max_box[3], remaining_boxes[:, 3])

            inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
            area_max = (max_box[2] - max_box[0]) * (max_box[3] - max_box[1])
            area_others = (remaining_boxes[:, 2] - remaining_boxes[:, 0]) * \
                         (remaining_boxes[:, 3] - remaining_boxes[:, 1])
            union = area_max + area_others - inter
            iou = inter / (union + 1e-6)

            # Soft-NMS: decay scores using Gaussian
            decay = np.exp(-(iou ** 2) / sigma)
            for i, idx in enumerate(indices):
                scores[idx] *= decay[i]

            # Remove low-confidence boxes
            indices = [i for i in indices if scores[i] > 0.01]

        return keep

    def train_step(
        self,
        images: List[torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """
        Perform a single training step.

        Args:
            images: Batch of images
            targets: List of target dictionaries
            optimizer: Optimizer

        Returns:
            Dictionary of losses
        """
        self.model.train()

        images = [img.to(self.device) for img in images]
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

        # Forward pass
        loss_dict = self.model(images, targets)

        # Total loss
        losses = sum(loss for loss in loss_dict.values())

        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        return {k: v.item() for k, v in loss_dict.items()}

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save(self.model.state_dict(), path)

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)
