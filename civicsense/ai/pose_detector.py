"""YOLO26-based pose estimation module.

Wraps the Ultralytics YOLO26-Pose model for estimating human body keypoints
used in determining hand-waste proximity and littering events.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from ultralytics import YOLO

from civicsense.core.config import get_config
from civicsense.core.constants import DEFAULT_CONFIDENCE_THRESHOLD, resolve_model_path
from civicsense.core.exceptions import ModelLoadError
from civicsense.core.logging import get_logger
from civicsense.dto.detection import BoundingBox, Keypoint, PoseResult

logger = get_logger("ai")


class YOLOPoseDetector:
    """YOLO26-Pose estimation engine.

    Estimates human body keypoints for tracking hand positions
    relative to waste objects to determine littering events.
    """

    def __init__(self) -> None:
        """Initialize the pose detector with no loaded model."""
        self._model: YOLO | None = None
        self._model_path: str = ""
        self._device: str = "cpu"

    def load(self, model_path: str, device: str = "auto") -> None:
        """Load the YOLO26-Pose model.

        Args:
            model_path: Path or name of the pose model weights.
            device: Target device for inference.

        Raises:
            ModelLoadError: If the model fails to load.
        """
        try:
            resolved = resolve_model_path(model_path)
            self._model = YOLO(resolved)
            self._model_path = resolved
            self._device = self._resolve_device(device)
            logger.info(
                f"Loaded pose model {resolved} on {self._device}",
                module="ai",
            )
        except Exception as e:
            raise ModelLoadError(f"Failed to load pose model {model_path}: {e}") from e

    def estimate(
        self,
        frame: NDArray[np.uint8],
        confidence: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> PoseResult:
        """Estimate pose keypoints for all detected persons.

        Args:
            frame: Input image as a BGR numpy array.
            confidence: Minimum confidence threshold for keypoints.

        Returns:
            PoseResult containing keypoints and bounding boxes for each person.

        Raises:
            RuntimeError: If no model is loaded.
        """
        if self._model is None:
            raise RuntimeError("No pose model loaded. Call load() first.")

        config = get_config()
        results = self._model.predict(
            source=frame,
            conf=confidence,
            imgsz=config.ai.image_size,
            device=self._device,
            verbose=False,
        )

        all_keypoints: list[list[Keypoint]] = []
        person_bboxes: list[BoundingBox] = []
        inference_time = 0.0

        for result in results:
            if result.speed:
                inference_time = result.speed.get("inference", 0.0) or 0.0

            if result.keypoints is None or result.boxes is None:
                continue

            keypoints_data = result.keypoints
            boxes = result.boxes

            if keypoints_data.xy is None or keypoints_data.conf is None:
                continue

            for i in range(len(keypoints_data)):
                kps_raw = keypoints_data.xy[i]
                kps_conf_raw = keypoints_data.conf[i]
                if kps_raw is None or kps_conf_raw is None:
                    continue
                kps = kps_raw.cpu().numpy()
                kps_conf = kps_conf_raw.cpu().numpy()

                person_keypoints: list[Keypoint] = []
                for j in range(len(kps)):
                    person_keypoints.append(
                        Keypoint(
                            x=float(kps[j][0]),
                            y=float(kps[j][1]),
                            confidence=float(kps_conf[j]) if j < len(kps_conf) else 0.0,
                        )
                    )
                all_keypoints.append(person_keypoints)

                box = boxes.xyxy[i].cpu().numpy()
                person_bboxes.append(
                    BoundingBox(
                        x1=float(box[0]),
                        y1=float(box[1]),
                        x2=float(box[2]),
                        y2=float(box[3]),
                    )
                )

        return PoseResult(
            keypoints=all_keypoints,
            person_bboxes=person_bboxes,
            frame_shape=frame.shape,
            inference_time_ms=inference_time,
        )

    def unload(self) -> None:
        """Release model resources and free memory."""
        if self._model is not None:
            del self._model
            self._model = None
            self._model_path = ""
            logger.info("Pose model unloaded", module="ai")

    @property
    def is_loaded(self) -> bool:
        """Return True if a model is loaded and ready for inference."""
        return self._model is not None

    @property
    def model_info(self) -> dict[str, Any]:
        """Return metadata about the loaded pose model.

        Returns:
            Dictionary with model path, device, and load status.
        """
        return {
            "model_path": self._model_path,
            "device": self._device,
            "is_loaded": self.is_loaded,
        }

    @staticmethod
    def _resolve_device(device: str) -> str:
        """Resolve the computation device.

        Defaults to CUDA GPU when available, falls back to CPU.

        Args:
            device: Requested device string.

        Returns:
            The resolved device string.
        """
        if device != "auto":
            return device
        try:
            import torch

            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                logger.info(f"CUDA available: {device_name}, using GPU", module="ai")
                return "cuda:0"
            logger.info("CUDA not available, falling back to CPU", module="ai")
        except ImportError:
            logger.warning("PyTorch not installed, using CPU", module="ai")
        return "cpu"
