"""AI model manager for loading and coordinating detection models.

Centralizes model lifecycle management including loading, unloading,
device selection, and model swapping through configuration changes.
"""

from __future__ import annotations

from typing import Any

from civicsense.ai.detector import YOLODetector
from civicsense.ai.pose_detector import YOLOPoseDetector
from civicsense.ai.tracker import ByteTracker
from civicsense.core.config import get_config
from civicsense.core.constants import DETECTION_MODELS, POSE_MODELS
from civicsense.core.exceptions import ModelLoadError
from civicsense.core.logging import get_logger

logger = get_logger("ai")


class ModelManager:
    """Manages the lifecycle of all AI models and the tracker.

    Provides centralized control over detection, pose estimation,
    and tracking model initialization and resource management.
    """

    def __init__(self) -> None:
        """Initialize the ModelManager with empty model instances."""
        self._detector = YOLODetector()
        self._pose_detector = YOLOPoseDetector()
        self._tracker = ByteTracker()

    @property
    def detector(self) -> YOLODetector:
        """Return the detection model instance."""
        return self._detector

    @property
    def pose_detector(self) -> YOLOPoseDetector:
        """Return the pose estimation model instance."""
        return self._pose_detector

    @property
    def tracker(self) -> ByteTracker:
        """Return the object tracker instance."""
        return self._tracker

    def load_models(
        self,
        detection_model: str | None = None,
        pose_model: str | None = None,
        device: str | None = None,
    ) -> None:
        """Load detection and pose models from configuration.

        Args:
            detection_model: Override detection model path.
            pose_model: Override pose model path.
            device: Override computation device.

        Raises:
            ModelLoadError: If either model fails to load.
        """
        config = get_config()
        det_path = detection_model or config.ai.detection_model
        pose_path = pose_model or config.ai.pose_model
        dev = device or config.ai.device

        try:
            self._detector.load(det_path, dev)
        except ModelLoadError:
            logger.error(f"Failed to load detection model: {det_path}", module="ai")
            raise

        try:
            self._pose_detector.load(pose_path, dev)
        except ModelLoadError:
            logger.warning(
                f"Pose model load failed, pose estimation disabled: {pose_path}",
                module="ai",
            )

        self._tracker.initialize()
        logger.info("All AI models loaded successfully", module="ai")

    def unload_models(self) -> None:
        """Unload all models and release resources."""
        self._detector.unload()
        self._pose_detector.unload()
        self._tracker.reset()
        logger.info("All AI models unloaded", module="ai")

    def swap_detection_model(self, model_name: str) -> None:
        """Swap the active detection model at runtime.

        Args:
            model_name: Name of the new model (e.g., 'yolo26s.pt').

        Raises:
            ModelLoadError: If the new model fails to load.
        """
        config = get_config()
        self._detector.unload()
        self._detector.load(model_name, config.ai.device)
        logger.info(f"Detection model swapped to {model_name}", module="ai")

    def swap_pose_model(self, model_name: str) -> None:
        """Swap the active pose model at runtime.

        Args:
            model_name: Name of the new model (e.g., 'yolo26s-pose.pt').

        Raises:
            ModelLoadError: If the new model fails to load.
        """
        config = get_config()
        self._pose_detector.unload()
        self._pose_detector.load(model_name, config.ai.device)
        logger.info(f"Pose model swapped to {model_name}", module="ai")

    @property
    def status(self) -> dict[str, Any]:
        """Return the current status of all managed models.

        Returns:
            Dictionary with detector, pose, and tracker status.
        """
        return {
            "detector": self._detector.model_info,
            "pose_detector": self._pose_detector.model_info,
            "tracker": {
                "active_tracks": len(self._tracker._tracks),
                "frame_count": self._tracker._frame_count,
            },
        }

    @staticmethod
    def available_models() -> dict[str, dict[str, str]]:
        """Return available model options.

        Returns:
            Dictionary mapping model types to their size-to-filename maps.
        """
        return {
            "detection": DETECTION_MODELS,
            "pose": POSE_MODELS,
        }
