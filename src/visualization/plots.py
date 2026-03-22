"""
Publication-quality plotting utilities.

Creates figures suitable for papers and reports.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import pandas as pd


def set_publication_style():
    """Set matplotlib style for publication-quality figures."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })


def plot_density_distribution(
    object_counts: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Object Density Distribution"
) -> plt.Figure:
    """
    Plot distribution of object counts per image.

    Args:
        object_counts: Array of object counts
        save_path: Optional path to save figure
        title: Plot title

    Returns:
        Matplotlib figure
    """
    set_publication_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Histogram
    ax.hist(object_counts, bins=50, color='#3498db', edgecolor='black', alpha=0.7)

    # Add mean and median lines
    mean_val = np.mean(object_counts)
    median_val = np.median(object_counts)

    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}')
    ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.1f}')

    ax.set_xlabel('Objects per Image')
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.legend()

    if save_path:
        fig.savefig(save_path)

    return fig


def plot_performance_comparison(
    results: Dict[str, Dict[str, float]],
    metrics: List[str] = ['mAP@0.5', 'F1'],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot performance comparison across models.

    Args:
        results: Dict mapping model name to metrics dict
        metrics: Metrics to plot
        save_path: Optional save path

    Returns:
        Matplotlib figure
    """
    set_publication_style()

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 6))
    axes = [axes] if n_metrics == 1 else axes

    models = list(results.keys())
    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))

    for ax, metric in zip(axes, metrics):
        values = [results[model].get(metric, 0) for model in models]

        bars = ax.bar(models, values, color=colors, edgecolor='black')
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} by Model')

        # Rotate labels
        ax.set_xticklabels(models, rotation=45, ha='right')

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.2f}', ha='center', fontsize=10)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)

    return fig


def plot_confusion_matrix(
    confusion_data: Dict[str, int],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot confusion matrix visualization.

    Args:
        confusion_data: Dict with 'tp', 'fp', 'fn' counts
        save_path: Optional save path

    Returns:
        Matplotlib figure
    """
    set_publication_style()

    tp = confusion_data.get('tp', 0)
    fp = confusion_data.get('fp', 0)
    fn = confusion_data.get('fn', 0)
    tn = 0  # Not applicable for object detection

    matrix = np.array([[tp, fn], [fp, tn]])

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Predicted Positive', 'Predicted Negative'],
                yticklabels=['Actual Positive', 'Actual Negative'])

    ax.set_title('Detection Confusion Matrix')

    if save_path:
        fig.savefig(save_path)

    return fig


def create_publication_figure(
    density_metrics: Dict[str, Dict[str, float]],
    model_name: str = "Hybrid",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create comprehensive publication figure with multiple panels.

    Args:
        density_metrics: Metrics stratified by density
        model_name: Model name for title
        save_path: Optional save path

    Returns:
        Matplotlib figure
    """
    set_publication_style()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Performance by density
    buckets = list(density_metrics.keys())
    mAP_values = [density_metrics[b].get('mAP@0.5', 0) for b in buckets]
    f1_values = [density_metrics[b].get('mean_f1', 0) for b in buckets]

    x = np.arange(len(buckets))
    width = 0.35

    bars1 = axes[0].bar(x - width/2, mAP_values, width, label='mAP@0.5', color='#3498db')
    bars2 = axes[0].bar(x + width/2, f1_values, width, label='F1 Score', color='#e74c3c')

    axes[0].set_xlabel('Object Density (objects/image)')
    axes[0].set_ylabel('Score')
    axes[0].set_title(f'{model_name}: Performance by Density')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(buckets)
    axes[0].legend()
    axes[0].set_ylim(0, 1)

    # Panel 2: Performance trend line
    avg_objects = [density_metrics[b].get('avg_objects', 0) for b in buckets]
    axes[1].scatter(avg_objects, mAP_values, s=100, c='#3498db', label='mAP@0.5', edgecolors='black')
    axes[1].scatter(avg_objects, f1_values, s=100, c='#e74c3c', label='F1 Score', edgecolors='black')

    # Trend lines
    if len(avg_objects) > 1:
        z1 = np.polyfit(avg_objects, mAP_values, 1)
        p1 = np.poly1d(z1)
        z2 = np.polyfit(avg_objects, f1_values, 1)
        p2 = np.poly1d(z2)

        x_line = np.linspace(min(avg_objects), max(avg_objects), 100)
        axes[1].plot(x_line, p1(x_line), '--', color='#3498db', alpha=0.7)
        axes[1].plot(x_line, p2(x_line), '--', color='#e74c3c', alpha=0.7)

    axes[1].set_xlabel('Average Objects per Image')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Performance vs Density (Trend)')
    axes[1].legend()
    axes[1].set_ylim(0, 1)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)

    return fig
