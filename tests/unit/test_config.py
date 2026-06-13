"""Tests for core configuration module."""

from __future__ import annotations

import pytest
from civicsense.core.config import (
    AIConfig,
    AppConfig,
    CameraConfig,
    DatabaseConfig,
    LoggingConfig,
    StorageConfig,
    ThemeMode,
    get_config,
    load_config,
)


class TestDatabaseConfig:
    """Tests for DatabaseConfig."""

    def test_default_values(self) -> None:
        """Verify default database configuration values."""
        config = DatabaseConfig()
        assert "sqlite" in config.url
        assert config.echo is False
        assert config.pool_size == 5

    def test_custom_values(self) -> None:
        """Verify custom database configuration values."""
        config = DatabaseConfig(url="postgresql://localhost/test", echo=True)
        assert config.url == "postgresql://localhost/test"
        assert config.echo is True


class TestAIConfig:
    """Tests for AIConfig."""

    def test_default_model(self) -> None:
        """Verify default detection model is yolo26n."""
        config = AIConfig()
        assert config.detection_model == "yolo26n.pt"
        assert config.pose_model == "yolo26n-pose.pt"

    def test_confidence_bounds(self) -> None:
        """Verify confidence threshold is bounded 0-1."""
        config = AIConfig(confidence_threshold=0.75)
        assert config.confidence_threshold == 0.75

    def test_invalid_confidence_raises(self) -> None:
        """Verify invalid confidence threshold raises validation error."""
        with pytest.raises(Exception):
            AIConfig(confidence_threshold=1.5)


class TestAppConfig:
    """Tests for AppConfig."""

    def test_default_theme(self) -> None:
        """Verify default theme is DARK."""
        config = AppConfig()
        assert config.theme == ThemeMode.DARK

    def test_singleton_get_config(self) -> None:
        """Verify get_config returns consistent instance."""
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2

    def test_load_config_override(self) -> None:
        """Verify load_config accepts overrides."""
        config = load_config(debug=True)
        assert config.debug is True
