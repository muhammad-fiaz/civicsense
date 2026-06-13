"""Configuration loader for CivicSense.

Loads detection settings from TOML config file with Python defaults.
Provides typed access to all detection thresholds, waste categories,
and storage paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import tomllib

_CONFIG_PATH = Path(__file__).parent / "detection.toml"
_config: dict[str, Any] | None = None


def _load_toml() -> dict[str, Any]:
    """Load and cache the TOML configuration file.

    Returns:
        Parsed TOML configuration dictionary.
    """
    global _config
    if _config is None:
        if _CONFIG_PATH.exists():
            with open(_CONFIG_PATH, "rb") as f:
                _config = tomllib.load(f)
        else:
            _config = {}
    return _config


def get_config_value(section: str, key: str, default: Any = None) -> Any:
    """Get a configuration value from the TOML file.

    Args:
        section: Top-level TOML section (e.g., 'detection').
        key: Key within the section.
        default: Fallback value if key is not found.

    Returns:
        The configuration value.
    """
    config = _load_toml()
    return config.get(section, {}).get(key, default)


def get_nested_config(*keys: str, default: Any = None) -> Any:
    """Get a nested configuration value.

    Args:
        *keys: Chain of keys to traverse (e.g., 'detection', 'thresholds', 'hand_proximity').
        default: Fallback value.

    Returns:
        The nested value or default.
    """
    config = _load_toml()
    current = config
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key, default)
        else:
            return default
    return current


def get_waste_classes() -> list[str]:
    """Get all waste class names from the TOML config.

    Returns:
        Flat list of all COCO class names that count as potential waste.
    """
    config = _load_toml()
    categories = config.get("waste", {}).get("categories", {})
    waste_classes: list[str] = []
    for category_classes in categories.values():
        if isinstance(category_classes, list):
            waste_classes.extend(category_classes)
    return waste_classes


def get_dustbin_candidates() -> list[str]:
    """Get dustbin candidate class names.

    Returns:
        List of COCO class names that could be dustbins.
    """
    config = _load_toml()
    return config.get("waste", {}).get("categories", {}).get("dustbin_candidates", [])


def get_waste_categories() -> dict[str, list[str]]:
    """Get waste categories with their class names.

    Returns:
        Dictionary mapping category names to lists of class names.
    """
    config = _load_toml()
    return config.get("waste", {}).get("categories", {})


def reload_config() -> None:
    """Force reload the configuration from disk."""
    global _config
    _config = None
    _load_toml()
