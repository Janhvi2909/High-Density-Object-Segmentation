"""
Density-based performance analysis.

Analyzes model performance across different object density levels.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from .metrics import calculate_map, calculate_f1_score


class DensityAnalyzer:
    """
    Analyzes model performance stratified by object density.

    Critical for understanding model behavior on high-density scenarios.
    """

    def __init__(
        self,
        density_buckets: List[Tuple[int, int]] = None
    ):
        """
        Initialize density analyzer.

        Args:
            density_buckets: List of (min, max) object count ranges
        """
        self.density_buckets = density_buckets or [
            (1, 20),    # Low
            (21, 50),   # Medium
            (51, 100),  # High
            (101, 200), # Very high
            (201, 500)  # Extreme
        ]

    def stratify_by_density(
        self,
        predictions: List[Dict[str, np.ndarray]],
        targets: List[Dict[str, np.ndarray]]
    ) -> Dict[str, Dict]:
        """
        Stratify evaluation results by object density.

        Args:
            predictions: List of predictions
            targets: List of targets

        Returns:
            Dictionary mapping density bucket to evaluation metrics
        """
        results = {}

        for min_obj, max_obj in self.density_buckets:
            bucket_name = f"{min_obj}-{max_obj}"
            bucket_preds = []
            bucket_targets = []

            for pred, target in zip(predictions, targets):
                num_gt = len(target.get('bboxes', []))
                if min_obj <= num_gt <= max_obj:
                    bucket_preds.append(pred)
                    bucket_targets.append(target)

            if bucket_preds:
                # Calculate metrics for this bucket
                mAP = calculate_map(bucket_preds, bucket_targets, [0.5])

                f1_scores = []
                precisions = []
                recalls = []

                for pred, target in zip(bucket_preds, bucket_targets):
                    metrics = calculate_f1_score(pred, target)
                    f1_scores.append(metrics['f1'])
                    precisions.append(metrics['precision'])
                    recalls.append(metrics['recall'])

                results[bucket_name] = {
                    'mAP@0.5': mAP,
                    'mean_f1': np.mean(f1_scores),
                    'mean_precision': np.mean(precisions),
                    'mean_recall': np.mean(recalls),
                    'num_images': len(bucket_preds),
                    'avg_objects': np.mean([len(t.get('bboxes', [])) for t in bucket_targets])
                }
            else:
                results[bucket_name] = {
                    'mAP@0.5': 0.0,
                    'mean_f1': 0.0,
                    'mean_precision': 0.0,
                    'mean_recall': 0.0,
                    'num_images': 0,
                    'avg_objects': 0
                }

        return results

    def analyze_density_correlation(
        self,
        predictions: List[Dict[str, np.ndarray]],
        targets: List[Dict[str, np.ndarray]]
    ) -> Dict[str, float]:
        """
        Analyze correlation between density and performance.

        Args:
            predictions: List of predictions
            targets: List of targets

        Returns:
            Correlation statistics
        """
        densities = []
        f1_scores = []
        recalls = []

        for pred, target in zip(predictions, targets):
            num_gt = len(target.get('bboxes', []))
            metrics = calculate_f1_score(pred, target)

            densities.append(num_gt)
            f1_scores.append(metrics['f1'])
            recalls.append(metrics['recall'])

        densities = np.array(densities)
        f1_scores = np.array(f1_scores)
        recalls = np.array(recalls)

        return {
            'density_f1_correlation': np.corrcoef(densities, f1_scores)[0, 1],
            'density_recall_correlation': np.corrcoef(densities, recalls)[0, 1],
            'mean_density': densities.mean(),
            'std_density': densities.std()
        }

    def generate_report(
        self,
        predictions: List[Dict[str, np.ndarray]],
        targets: List[Dict[str, np.ndarray]]
    ) -> pd.DataFrame:
        """
        Generate comprehensive density analysis report.

        Args:
            predictions: List of predictions
            targets: List of targets

        Returns:
            DataFrame with density analysis
        """
        stratified = self.stratify_by_density(predictions, targets)

        # Convert to DataFrame
        rows = []
        for bucket, metrics in stratified.items():
            row = {'Density Bucket': bucket}
            row.update(metrics)
            rows.append(row)

        df = pd.DataFrame(rows)
        df = df.set_index('Density Bucket')

        return df


def stratified_evaluation(
    predictions: List[Dict[str, np.ndarray]],
    targets: List[Dict[str, np.ndarray]],
    stratification_fn: callable
) -> Dict[str, Dict[str, float]]:
    """
    Perform stratified evaluation using custom stratification function.

    Args:
        predictions: List of predictions
        targets: List of targets
        stratification_fn: Function that takes target and returns category

    Returns:
        Dictionary of category to metrics
    """
    categories = {}

    for pred, target in zip(predictions, targets):
        category = stratification_fn(target)

        if category not in categories:
            categories[category] = {'preds': [], 'targets': []}

        categories[category]['preds'].append(pred)
        categories[category]['targets'].append(target)

    results = {}
    for category, data in categories.items():
        mAP = calculate_map(data['preds'], data['targets'], [0.5])

        f1_scores = []
        for pred, target in zip(data['preds'], data['targets']):
            metrics = calculate_f1_score(pred, target)
            f1_scores.append(metrics['f1'])

        results[category] = {
            'mAP@0.5': mAP,
            'mean_f1': np.mean(f1_scores),
            'num_samples': len(data['preds'])
        }

    return results
