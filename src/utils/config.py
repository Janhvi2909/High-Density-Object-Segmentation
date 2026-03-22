"""
Configuration management for High-Density Object Segmentation.

Provides utilities for loading and accessing YAML configuration files.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml


class Config:
    """
    Configuration wrapper that allows attribute-style access to config values.

    Example:
        config = Config.from_yaml('config/config.yaml')
        print(config.data.image_size)
        print(config.training.epochs)
    """

    def __init__(self, config_dict: Dict[str, Any]):
        """
        Initialize Config from a dictionary.

        Args:
            config_dict: Dictionary containing configuration values
        """
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> 'Config':
        """
        Load configuration from a YAML file.

        Args:
            yaml_path: Path to the YAML configuration file

        Returns:
            Config object with loaded values
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        return cls(config_dict)

    def __repr__(self) -> str:
        return f"Config({self.__dict__})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert Config to dictionary recursively."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value with optional default."""
        return getattr(self, key, default)


def load_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file. If None, loads default config.

    Returns:
        Config object
    """
    if config_path is None:
        # Default config path
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "config" / "config.yaml"

    return Config.from_yaml(config_path)


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


def get_data_dir() -> Path:
    """Get the data directory path."""
    return get_project_root() / "data"


def get_checkpoint_dir() -> Path:
    """Get the checkpoints directory path."""
    checkpoint_dir = get_project_root() / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    return checkpoint_dir
