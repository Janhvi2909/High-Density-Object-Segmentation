"""
Visualization utilities for segmentation results.

Creates publication-quality overlays and visualizations.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt


def get_color_palette(n_colors: int = 20) -> List[Tuple[int, int, int]]:
    """Generate distinct colors for visualization."""
    colors = []
    for i in range(n_colors):
        hue = i / n_colors
        rgb = plt.cm.hsv(hue)[:3]
        colors.append(tuple(int(c * 255) for c in rgb))
    return colors


def draw_segmentation(
    image: np.ndarray,
    masks: np.ndarray,
    bboxes: np.ndarray,
    scores: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    class_names: Optional[Dict[int, str]] = None,
    alpha: float = 0.5,
    show_scores: bool = True,
    show_labels: bool = True
) -> np.ndarray:
    """
    Draw segmentation masks and bounding boxes on image.

    Args:
        image: Input RGB image
        masks: Binary masks (N, H, W)
        bboxes: Bounding boxes (N, 4)
        scores: Confidence scores (N,)
        labels: Class labels (N,)
        class_names: Mapping from label ID to name
        alpha: Mask transparency
        show_scores: Show confidence scores
        show_labels: Show class labels

    Returns:
        Image with overlaid segmentation
    """
    result = image.copy()
    colors = get_color_palette(len(masks))

    for i, (mask, bbox) in enumerate(zip(masks, bboxes)):
        color = colors[i % len(colors)]

        # Draw mask
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

        mask_bool = mask > 0
        overlay = result.copy()
        overlay[mask_bool] = color
        result = cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0)

        # Draw bounding box
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

        # Draw label
        label_text = ""
        if show_labels and labels is not None:
            label_id = labels[i]
            if class_names and label_id in class_names:
                label_text = class_names[label_id]
            else:
                label_text = f"class_{label_id}"

        if show_scores and scores is not None:
            score_text = f"{scores[i]:.2f}"
            label_text = f"{label_text}: {score_text}" if label_text else score_text

        if label_text:
            # Background for text
            (text_w, text_h), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(result, (x1, y1 - text_h - 4), (x1 + text_w, y1), color, -1)
            cv2.putText(
                result, label_text, (x1, y1 - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )

    return result


def overlay_masks(
    image: np.ndarray,
    masks: np.ndarray,
    alpha: float = 0.5,
    colors: Optional[List[Tuple[int, int, int]]] = None
) -> np.ndarray:
    """
    Overlay multiple masks on image.

    Args:
        image: Input image
        masks: Binary masks (N, H, W)
        alpha: Transparency
        colors: Optional color list

    Returns:
        Image with mask overlay
    """
    if colors is None:
        colors = get_color_palette(len(masks))

    result = image.copy()

    for i, mask in enumerate(masks):
        color = colors[i % len(colors)]

        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

        mask_bool = mask > 0
        overlay = result.copy()
        overlay[mask_bool] = color
        result = cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0)

    return result


def visualize_predictions(
    image: np.ndarray,
    predictions: Dict[str, np.ndarray],
    ground_truth: Optional[Dict[str, np.ndarray]] = None,
    title: str = "Predictions"
) -> np.ndarray:
    """
    Visualize predictions with optional ground truth comparison.

    Args:
        image: Input image
        predictions: Prediction dictionary
        ground_truth: Optional GT dictionary
        title: Plot title

    Returns:
        Visualization figure as numpy array
    """
    n_cols = 2 if ground_truth else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(8 * n_cols, 8))

    if n_cols == 1:
        axes = [axes]

    # Draw predictions
    pred_vis = draw_segmentation(
        image,
        predictions.get('masks', np.zeros((0, *image.shape[:2]))),
        predictions.get('bboxes', np.zeros((0, 4))),
        predictions.get('scores', None)
    )
    axes[0].imshow(pred_vis)
    axes[0].set_title(f"{title}\n({len(predictions.get('bboxes', []))} detections)")
    axes[0].axis('off')

    # Draw ground truth
    if ground_truth:
        gt_vis = draw_segmentation(
            image,
            ground_truth.get('masks', np.zeros((0, *image.shape[:2]))),
            ground_truth.get('bboxes', np.zeros((0, 4))),
            show_scores=False
        )
        axes[1].imshow(gt_vis)
        axes[1].set_title(f"Ground Truth\n({len(ground_truth.get('bboxes', []))} objects)")
        axes[1].axis('off')

    plt.tight_layout()

    # Convert to numpy
    fig.canvas.draw()
    result = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    result = result.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return result


def create_comparison_grid(
    image: np.ndarray,
    results: Dict[str, Dict[str, np.ndarray]],
    ground_truth: Optional[Dict[str, np.ndarray]] = None,
    figsize: Tuple[int, int] = (16, 12)
) -> np.ndarray:
    """
    Create grid comparing multiple model results.

    Args:
        image: Original image
        results: Dict mapping model name to predictions
        ground_truth: Optional ground truth
        figsize: Figure size

    Returns:
        Comparison grid as numpy array
    """
    n_models = len(results)
    n_cols = min(3, n_models + (1 if ground_truth else 0))
    n_rows = (n_models + (1 if ground_truth else 0) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.array(axes).flatten() if n_rows * n_cols > 1 else [axes]

    idx = 0

    # Ground truth first
    if ground_truth:
        gt_vis = draw_segmentation(
            image,
            ground_truth.get('masks', np.zeros((0, *image.shape[:2]))),
            ground_truth.get('bboxes', np.zeros((0, 4))),
            show_scores=False
        )
        axes[idx].imshow(gt_vis)
        axes[idx].set_title(f"Ground Truth ({len(ground_truth.get('bboxes', []))} objects)")
        axes[idx].axis('off')
        idx += 1

    # Model predictions
    for name, pred in results.items():
        pred_vis = draw_segmentation(
            image,
            pred.get('masks', np.zeros((0, *image.shape[:2]))),
            pred.get('bboxes', np.zeros((0, 4))),
            pred.get('scores', None)
        )
        axes[idx].imshow(pred_vis)
        axes[idx].set_title(f"{name} ({len(pred.get('bboxes', []))} detections)")
        axes[idx].axis('off')
        idx += 1

    # Hide unused axes
    for i in range(idx, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    fig.canvas.draw()
    result = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    result = result.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return result
