"""Application configuration using Pydantic Settings.

Provides typed, validated configuration for all CivicSense subsystems.
Defaults are loaded from civicsense/config/detection.toml.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from civicsense.config import get_nested_config


class ThemeMode(str, Enum):
    """Supported GUI theme modes."""

    DARK = "dark"
    LIGHT = "light"
    SYSTEM = "system"


class DatabaseConfig(BaseSettings):
    """Database connection configuration."""

    model_config = SettingsConfigDict(env_prefix="CIVICSENSE_DB_")

    url: str = Field(default="sqlite:///civicsense.db", description="Database URL")
    echo: bool = Field(default=False, description="Enable SQL query logging")
    pool_size: int = Field(default=5, description="Connection pool size")
    max_overflow: int = Field(default=10, description="Max overflow connections")


class AIConfig(BaseSettings):
    """AI model configuration — defaults from detection.toml."""

    model_config = SettingsConfigDict(env_prefix="CIVICSENSE_AI_")

    detection_model: str = Field(
        default=get_nested_config(
            "detection", "models", "detection", default="yolo26n.pt"
        ),
        description="Detection model name",
    )
    pose_model: str = Field(
        default=get_nested_config(
            "detection", "models", "pose", default="yolo26n-pose.pt"
        ),
        description="Pose estimation model name",
    )
    confidence_threshold: float = Field(
        default=float(get_nested_config("detection", "confidence", default=0.15)),
        ge=0.0,
        le=1.0,
    )
    iou_threshold: float = Field(
        default=float(get_nested_config("detection", "iou", default=0.45)),
        ge=0.0,
        le=1.0,
    )
    device: str = Field(
        default=get_nested_config("detection", "device", default="auto"),
        description="Device: auto, cpu, cuda, cuda:0",
    )
    image_size: int = Field(
        default=int(get_nested_config("detection", "image_size", default=640)),
        ge=320,
        le=1920,
    )
    half_precision: bool = Field(default=True, description="Use FP16 inference")
    max_detections: int = Field(default=300, ge=1, le=1000)
    tracking_threshold: float = Field(
        default=float(get_nested_config("detection", "tracking", "min_hits", default=1))
        / 10.0,
        ge=0.0,
        le=1.0,
    )
    tracker_type: str = Field(default="bytetrack", description="Tracker algorithm")


class CameraConfig(BaseSettings):
    """Camera source configuration."""

    model_config = SettingsConfigDict(env_prefix="CIVICSENSE_CAM_")

    source: str | int = Field(
        default=0, description="Camera source (index or RTSP URL)"
    )
    fps: int = Field(default=30, ge=1, le=120)
    width: int = Field(default=1920, ge=320, le=3840)
    height: int = Field(default=1080, ge=240, le=2160)
    buffer_size: int = Field(default=5, ge=1, le=30)

    @field_validator("source", mode="before")
    @classmethod
    def coerce_camera_source(cls, v: str | int) -> str | int:
        """Convert string digits to int for camera index."""
        if isinstance(v, str) and v.isdigit():
            return int(v)
        return v


class StorageConfig(BaseSettings):
    """Storage path configuration — defaults from detection.toml."""

    model_config = SettingsConfigDict(env_prefix="CIVICSENSE_STORAGE_")

    base_dir: Path = Field(
        default=Path(get_nested_config("storage", "detected_dir", default="detected")),
        description="Base output directory",
    )
    evidence_dir: Path = Field(
        default=Path(
            get_nested_config("storage", "evidence_dir", default="detected/evidence")
        ),
        description="Evidence storage",
    )
    snapshots_dir: Path = Field(
        default=Path(
            get_nested_config(
                "storage", "screenshots_dir", default="detected/screenshots"
            )
        ),
        description="Screenshot storage",
    )
    clips_dir: Path = Field(
        default=Path(
            get_nested_config("storage", "clips_dir", default="detected/clips")
        ),
        description="Video clip storage",
    )
    annotated_dir: Path = Field(
        default=Path(
            get_nested_config("storage", "annotated_dir", default="detected/annotated")
        ),
        description="Annotated frame storage",
    )
    exports_dir: Path = Field(
        default=Path(get_nested_config("storage", "exports_dir", default="exports")),
        description="Export storage",
    )
    logs_dir: Path = Field(
        default=Path(get_nested_config("storage", "logs_dir", default="logs")),
        description="Log file directory",
    )
    max_storage_gb: float = Field(default=50.0, ge=1.0)


class LoggingConfig(BaseSettings):
    """Logging configuration — defaults from detection.toml."""

    model_config = SettingsConfigDict(env_prefix="CIVICSENSE_LOG_")

    level: str = Field(
        default=get_nested_config("logging", "level", default="INFO"),
        description="Global log level",
    )
    json_output: bool = Field(
        default=get_nested_config("logging", "json_output", default=False),
        description="Structured JSON logging",
    )
    rotation: str = Field(
        default=get_nested_config("logging", "rotation", default="10 MB"),
        description="Log rotation size",
    )
    retention: str = Field(
        default=get_nested_config("logging", "retention", default="7 days"),
        description="Log retention period",
    )
    compression: str = Field(
        default=get_nested_config("logging", "compression", default="gz"),
        description="Log compression format",
    )
    rich_tracebacks: bool = Field(
        default=get_nested_config("logging", "rich_tracebacks", default=True),
        description="Use Rich tracebacks",
    )


class AppConfig(BaseSettings):
    """Root application configuration aggregating all subsystem configs."""

    model_config = SettingsConfigDict(
        env_prefix="CIVICSENSE_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    app_name: str = Field(default="CivicSense")
    version: str = Field(default="0.1.0")
    debug: bool = Field(default=False)
    theme: ThemeMode = Field(default=ThemeMode.DARK)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    ai: AIConfig = Field(default_factory=AIConfig)
    camera: CameraConfig = Field(default_factory=CameraConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @field_validator("storage", mode="after")
    @classmethod
    def ensure_storage_dirs(cls, v: StorageConfig) -> StorageConfig:
        """Create output directories if they do not exist."""
        for d in [
            v.base_dir,
            v.snapshots_dir,
            v.clips_dir,
            v.annotated_dir,
            v.logs_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)
        return v


_config: AppConfig | None = None


def get_config() -> AppConfig:
    """Return the global application configuration singleton.

    Returns:
        The application configuration instance.
    """
    global _config
    if _config is None:
        _config = AppConfig()
    return _config


def load_config(**overrides: Any) -> AppConfig:
    """Load configuration with optional overrides.

    Args:
        **overrides: Key-value pairs to override default settings.

    Returns:
        The loaded application configuration.
    """
    global _config
    _config = AppConfig(**overrides)
    return _config
