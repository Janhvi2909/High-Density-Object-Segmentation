"""
Logging utilities for High-Density Object Segmentation.

Provides consistent logging across all modules with rich formatting.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler


def setup_logger(
    name: str = "hdos",
    level: str = "INFO",
    log_dir: Optional[Path] = None,
    log_to_file: bool = True
) -> logging.Logger:
    """
    Set up a logger with rich console output and optional file logging.

    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files. If None, uses default logs/ directory
        log_to_file: Whether to also log to a file

    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # Rich console handler
    console_handler = RichHandler(
        console=Console(),
        show_time=True,
        show_path=False,
        rich_tracebacks=True
    )
    console_handler.setLevel(getattr(logging, level.upper()))
    console_format = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler
    if log_to_file:
        if log_dir is None:
            log_dir = Path(__file__).parent.parent.parent / "logs"
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{name}_{timestamp}.log"

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Log everything to file
        file_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "hdos") -> logging.Logger:
    """
    Get an existing logger or create a new one.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger


class ProgressLogger:
    """
    A progress logger for tracking training/evaluation progress.

    Example:
        progress = ProgressLogger("Training", total_steps=1000)
        for step in range(1000):
            progress.update(step, loss=0.5, accuracy=0.9)
    """

    def __init__(self, name: str, total_steps: int):
        """
        Initialize progress logger.

        Args:
            name: Name of the progress (e.g., "Training", "Evaluation")
            total_steps: Total number of steps
        """
        self.name = name
        self.total_steps = total_steps
        self.logger = get_logger("hdos.progress")
        self.start_time = datetime.now()

    def update(self, step: int, **metrics) -> None:
        """
        Update progress with current step and metrics.

        Args:
            step: Current step number
            **metrics: Key-value pairs of metrics to log
        """
        progress = (step + 1) / self.total_steps * 100
        elapsed = (datetime.now() - self.start_time).total_seconds()
        eta = elapsed / (step + 1) * (self.total_steps - step - 1) if step > 0 else 0

        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(
            f"{self.name} [{step + 1}/{self.total_steps}] ({progress:.1f}%) "
            f"- ETA: {eta:.0f}s - {metrics_str}"
        )

    def finish(self, **final_metrics) -> None:
        """
        Log final completion message.

        Args:
            **final_metrics: Final metrics to report
        """
        total_time = (datetime.now() - self.start_time).total_seconds()
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in final_metrics.items()])
        self.logger.info(
            f"{self.name} completed in {total_time:.1f}s - {metrics_str}"
        )
