"""Background inference worker for the AI pipeline.

Runs the full detection -> pose -> tracking -> classification pipeline
on video frames in a worker thread to keep the UI responsive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
from numpy.typing import NDArray

from civicsense.ai.event_classifier import EventClassifier
from civicsense.ai.littering_engine import LitteringClassifier, LitteringState
from civicsense.ai.model_manager import ModelManager
from civicsense.ai.tracker import ByteTracker
from civicsense.core.config import get_config
from civicsense.core.constants import WASTE_CLASSES
from civicsense.core.logging import get_logger
from civicsense.dto.detection import DetectionResult, PoseResult, TrackedObject

logger = get_logger("ai")


@dataclass
class InferenceResult:
    """Complete result of processing a single frame through the pipeline."""

    frame_idx: int
    frame: NDArray[np.uint8]
    detection_result: DetectionResult
    pose_result: PoseResult | None
    tracked_persons: list[TrackedObject]
    tracked_waste: list[TrackedObject]
    tracked_dustbins: list[TrackedObject]
    littering_events: list[LitteringState]
    fps: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class InferenceWorker:
    """Runs the full AI inference pipeline on video frames.

    Coordinates detection, pose estimation, tracking, and littering
    classification in a single pipeline for each input frame.
    """

    def __init__(self, model_manager: ModelManager) -> None:
        """Initialize the inference worker with a model manager.

        Args:
            model_manager: The shared ModelManager instance.
        """
        self._model_manager = model_manager
        self._event_classifier = EventClassifier()
        self._littering_classifier = LitteringClassifier()
        self._frame_idx: int = 0
        self._last_fps: float = 0.0

    def process_frame(
        self,
        frame: NDArray[np.uint8],
    ) -> InferenceResult:
        """Process a single frame through the full inference pipeline.

        Args:
            frame: Input video frame as a BGR numpy array.

        Returns:
            InferenceResult with all detection, tracking, and classification data.
        """
        self._frame_idx += 1
        config = get_config()

        detector = self._model_manager.detector
        pose_detector = self._model_manager.pose_detector
        tracker = self._model_manager.tracker

        detection_result = detector.detect(
            frame,
            confidence=config.ai.confidence_threshold,
            iou=config.ai.iou_threshold,
        )

        pose_result = None
        if pose_detector.is_loaded:
            try:
                pose_result = pose_detector.estimate(
                    frame,
                    confidence=config.ai.confidence_threshold,
                )
            except Exception as e:
                logger.warning(f"Pose estimation failed: {e}", module="ai")

        persons = self._track_objects(detection_result, "person", tracker, frame)
        waste = self._track_objects(
            detection_result,
            None,
            tracker,
            frame,
            class_filter=WASTE_CLASSES,
        )
        dustbins = self._track_objects(detection_result, "dustbin", tracker, frame)

        if pose_result and persons:
            self._attach_keypoints(persons, pose_result)

        littering_events = self._littering_classifier.classify(
            persons, waste, dustbins, frame.shape[0]
        )

        return InferenceResult(
            frame_idx=self._frame_idx,
            frame=frame,
            detection_result=detection_result,
            pose_result=pose_result,
            tracked_persons=persons,
            tracked_waste=waste,
            tracked_dustbins=dustbins,
            littering_events=littering_events,
            fps=self._last_fps,
        )

    def reset(self) -> None:
        """Reset the inference pipeline state."""
        self._frame_idx = 0
        self._littering_classifier.reset()
        self._event_classifier.reset_stats()
        self._model_manager.tracker.reset()

    def _track_objects(
        self,
        detection_result: DetectionResult,
        class_name: str | None,
        tracker: ByteTracker,
        frame: NDArray[np.uint8],
        class_filter: list[str] | None = None,
    ) -> list[TrackedObject]:
        """Filter detections and run them through the tracker.

        Args:
            detection_result: Raw detection results.
            class_name: Specific class to filter by, or None for multi-class.
            tracker: The ByteTracker instance.
            frame: The current video frame.
            class_filter: Alternative to class_name for multi-class filtering.

        Returns:
            List of tracked objects with stable IDs.
        """
        if class_filter:
            dets = [
                d for d in detection_result.detections if d.class_name in class_filter
            ]
        elif class_name:
            dets = detection_result.filter_by_class(class_name)
        else:
            dets = detection_result.detections

        tracked = [
            TrackedObject(
                track_id=0,
                class_name=d.class_name,
                bbox=d.bbox,
                confidence=d.confidence,
                frame_idx=self._frame_idx,
            )
            for d in dets
        ]

        return tracker.update(tracked, frame)

    def _attach_keypoints(
        self,
        persons: list[TrackedObject],
        pose_result: PoseResult,
    ) -> None:
        """Attach pose keypoints to tracked person objects.

        Args:
            persons: List of tracked person objects.
            pose_result: Pose estimation results.
        """
        for person in persons:
            best_iou = 0.0
            best_kps: list = []
            for i, person_bbox in enumerate(pose_result.person_bboxes):
                iou = self._compute_iou(person.bbox, person_bbox)
                if iou > best_iou:
                    best_iou = iou
                    if i < len(pose_result.keypoints):
                        best_kps = pose_result.keypoints[i]
            if best_kps:
                person.keypoints = best_kps

    @staticmethod
    def _compute_iou(box_a: Any, box_b: Any) -> float:
        """Compute IoU between two bounding boxes.

        Args:
            box_a: First bounding box with x1, y1, x2, y2 attributes.
            box_b: Second bounding box with x1, y1, x2, y2 attributes.

        Returns:
            IoU value between 0.0 and 1.0.
        """
        x1 = max(box_a.x1, box_b.x1)
        y1 = max(box_a.y1, box_b.y1)
        x2 = min(box_a.x2, box_b.x2)
        y2 = min(box_a.y2, box_b.y2)

        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_a = box_a.area
        area_b = box_b.area
        union = area_a + area_b - intersection

        return intersection / union if union > 0 else 0.0
