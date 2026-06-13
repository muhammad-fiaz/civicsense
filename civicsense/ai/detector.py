"""YOLO26-based object detection module.

Wraps the Ultralytics YOLO26 model for detecting persons, waste objects,
and dustbins with configurable confidence and IoU thresholds.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from ultralytics import YOLO

from civicsense.core.config import get_config
from civicsense.core.constants import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
    resolve_model_path,
)
from civicsense.core.exceptions import ModelLoadError
from civicsense.core.logging import get_logger
from civicsense.dto.detection import BoundingBox, Detection, DetectionResult

logger = get_logger("ai")


class YOLODetector:
    """YOLO26 object detection engine.

    Provides detection of persons, waste objects, and dustbins using
    Ultralytics YOLO26 models with automatic GPU/CPU device selection.
    """

    def __init__(self) -> None:
        """Initialize the detector with no loaded model."""
        self._model: YOLO | None = None
        self._model_path: str = ""
        self._device: str = "cpu"

    def load(self, model_path: str, device: str = "auto") -> None:
        """Load the YOLO26 detection model.

        Args:
            model_path: Path or name of the model weights file.
            device: Target device. 'auto' selects GPU if available.

        Raises:
            ModelLoadError: If the model fails to load.
        """
        try:
            resolved = resolve_model_path(model_path)
            self._model = YOLO(resolved)
            self._model_path = resolved
            self._device = self._resolve_device(device)
            logger.info(
                f"Loaded detection model {resolved} on {self._device}",
                module="ai",
            )
        except Exception as e:
            raise ModelLoadError(f"Failed to load model {model_path}: {e}") from e

    def detect(
        self,
        frame: NDArray[np.uint8],
        confidence: float = DEFAULT_CONFIDENCE_THRESHOLD,
        iou: float = DEFAULT_IOU_THRESHOLD,
    ) -> DetectionResult:
        """Run object detection on a single frame.

        Args:
            frame: Input image as a BGR numpy array.
            confidence: Minimum confidence threshold for detections.
            iou: IoU threshold for non-maximum suppression.

        Returns:
            DetectionResult containing all detected objects.

        Raises:
            RuntimeError: If no model is loaded.
        """
        if self._model is None:
            raise RuntimeError("No detection model loaded. Call load() first.")

        config = get_config()
        results = self._model.predict(
            source=frame,
            conf=confidence,
            iou=iou,
            imgsz=config.ai.image_size,
            device=self._device,
            verbose=False,
        )

        detections: list[Detection] = []
        inference_time = 0.0

        for result in results:
            if result.speed:
                inference_time = result.speed.get("inference", 0.0) or 0.0

            if result.boxes is None:
                continue

            boxes = result.boxes
            for i in range(len(boxes)):
                box = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                cls_name = result.names.get(cls_id, f"class_{cls_id}")

                bbox = BoundingBox(
                    x1=float(box[0]),
                    y1=float(box[1]),
                    x2=float(box[2]),
                    y2=float(box[3]),
                )
                detections.append(
                    Detection(
                        class_id=cls_id,
                        class_name=cls_name,
                        confidence=conf,
                        bbox=bbox,
                    )
                )

        return DetectionResult(
            detections=detections,
            frame_shape=frame.shape,
            inference_time_ms=inference_time,
        )

    def unload(self) -> None:
        """Release model resources and free memory."""
        if self._model is not None:
            del self._model
            self._model = None
            self._model_path = ""
            logger.info("Detection model unloaded", module="ai")

    @property
    def is_loaded(self) -> bool:
        """Return True if a model is loaded and ready for inference."""
        return self._model is not None

    @property
    def model_info(self) -> dict[str, Any]:
        """Return metadata about the loaded model.

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
            device: Requested device string ('auto', 'cpu', 'cuda', etc.).

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
