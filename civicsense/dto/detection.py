"""Data Transfer Objects for detection pipeline results.

Provides typed dataclasses for passing detection, pose, and tracking
results between the AI pipeline, services, and GUI layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


@dataclass
class BoundingBox:
    """A 2D bounding box with coordinates and dimensions."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        """Return the width of the bounding box."""
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """Return the height of the bounding box."""
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        """Return the center point of the bounding box."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def area(self) -> float:
        """Return the area of the bounding box."""
        return self.width * self.height


@dataclass
class Keypoint:
    """A single pose keypoint with coordinates and visibility."""

    x: float
    y: float
    confidence: float


@dataclass
class Detection:
    """A single object detection result."""

    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox


@dataclass
class DetectionResult:
    """Aggregated detection results for a single frame."""

    detections: list[Detection] = field(default_factory=list)
    frame_shape: tuple[int, int, int] = (0, 0, 0)
    inference_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def count(self) -> int:
        """Return the number of detections."""
        return len(self.detections)

    def filter_by_class(self, class_name: str) -> list[Detection]:
        """Return detections matching the given class name.

        Args:
            class_name: The class label to filter by.

        Returns:
            List of matching Detection objects.
        """
        return [d for d in self.detections if d.class_name == class_name]

    def filter_by_confidence(self, min_confidence: float) -> list[Detection]:
        """Return detections above the confidence threshold.

        Args:
            min_confidence: Minimum confidence value.

        Returns:
            List of detections meeting the threshold.
        """
        return [d for d in self.detections if d.confidence >= min_confidence]


@dataclass
class PoseResult:
    """Pose estimation results for detected persons."""

    keypoints: list[list[Keypoint]] = field(default_factory=list)
    person_bboxes: list[BoundingBox] = field(default_factory=list)
    frame_shape: tuple[int, int, int] = (0, 0, 0)
    inference_time_ms: float = 0.0

    @property
    def person_count(self) -> int:
        """Return the number of persons with detected poses."""
        return len(self.keypoints)


@dataclass
class TrackedObject:
    """A tracked object with stable ID across frames."""

    track_id: int
    class_name: str
    bbox: BoundingBox
    confidence: float
    frame_idx: int = 0
    keypoints: list[Keypoint] = field(default_factory=list)
    velocity: tuple[float, float] = (0.0, 0.0)
    age: int = 0
    hits: int = 0

    @property
    def wrist_positions(self) -> list[tuple[float, float]]:
        """Return wrist keypoint positions if available.

        Returns:
            List of (x, y) tuples for detected wrists.
        """
        if len(self.keypoints) < 11:
            return []
        positions = []
        for idx in (9, 10):
            kp = self.keypoints[idx]
            if kp.confidence > 0:
                positions.append((kp.x, kp.y))
        return positions

    @property
    def hand_center(self) -> tuple[float, float] | None:
        """Return the average position of detected wrists.

        Returns:
            Average (x, y) of wrists, or None if no wrists detected.
        """
        wrists = self.wrist_positions
        if not wrists:
            return None
        avg_x = sum(w[0] for w in wrists) / len(wrists)
        avg_y = sum(w[1] for w in wrists) / len(wrists)
        return (avg_x, avg_y)


class IncidentStatus(str, Enum):
    """Incident review status."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    UNDER_REVIEW = "under_review"


@dataclass
class IncidentDTO:
    """Data transfer object for an incident record."""

    id: int | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    camera_id: str = ""
    camera_name: str = ""
    confidence: float = 0.0
    person_track_id: int = 0
    waste_type: str = ""
    waste_bbox: BoundingBox | None = None
    snapshot_path: str = ""
    annotated_path: str = ""
    clip_path: str = ""
    detection_metadata: dict[str, object] = field(default_factory=dict)
    status: IncidentStatus = IncidentStatus.PENDING
    review_notes: str = ""
    frame_width: int = 0
    frame_height: int = 0

    def to_dict(self) -> dict[str, object]:
        """Serialize the DTO to a dictionary.

        Returns:
            Dictionary representation of the incident.
        """
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "camera_id": self.camera_id,
            "camera_name": self.camera_name,
            "confidence": self.confidence,
            "person_track_id": self.person_track_id,
            "waste_type": self.waste_type,
            "snapshot_path": self.snapshot_path,
            "annotated_path": self.annotated_path,
            "clip_path": self.clip_path,
            "status": self.status.value,
            "review_notes": self.review_notes,
        }
