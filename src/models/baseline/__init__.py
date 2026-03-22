"""Baseline ML models using traditional image processing."""

from .thresholding import AdaptiveThresholdSegmenter
from .edge_detection import CannyContourSegmenter
from .connected_components import ConnectedComponentsSegmenter
