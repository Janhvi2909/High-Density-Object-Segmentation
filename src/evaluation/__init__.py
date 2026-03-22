"""Evaluation metrics and analysis tools."""

from .metrics import (
    calculate_map,
    calculate_iou,
    calculate_precision_recall,
    calculate_f1_score,
    SegmentationMetrics
)
from .density_analysis import DensityAnalyzer, stratified_evaluation
from .failure_analysis import FailureAnalyzer, categorize_failures
