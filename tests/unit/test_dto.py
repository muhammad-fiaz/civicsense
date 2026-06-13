"""Tests for detection DTOs."""

from __future__ import annotations

import pytest
from civicsense.dto.detection import (
    BoundingBox,
    Detection,
    DetectionResult,
    IncidentDTO,
    IncidentStatus,
    Keypoint,
    PoseResult,
    TrackedObject,
)


class TestBoundingBox:
    """Tests for BoundingBox dataclass."""

    def test_basic_properties(self) -> None:
        """Verify bounding box computed properties."""
        bbox = BoundingBox(x1=10, y1=20, x2=110, y2=220)
        assert bbox.width == 100
        assert bbox.height == 200
        assert bbox.area == 20000

    def test_center(self) -> None:
        """Verify center point calculation."""
        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=100)
        assert bbox.center == (50.0, 50.0)

    def test_center_offset(self) -> None:
        """Verify center with offset coordinates."""
        bbox = BoundingBox(x1=20, y1=30, x2=80, y2=70)
        assert bbox.center == (50.0, 50.0)


class TestDetection:
    """Tests for Detection dataclass."""

    def test_creation(self) -> None:
        """Verify detection object creation."""
        det = Detection(
            class_id=0,
            class_name="person",
            confidence=0.95,
            bbox=BoundingBox(x1=0, y1=0, x2=100, y2=100),
        )
        assert det.class_name == "person"
        assert det.confidence == 0.95


class TestDetectionResult:
    """Tests for DetectionResult dataclass."""

    def test_count(self) -> None:
        """Verify detection count."""
        result = DetectionResult(
            detections=[
                Detection(0, "person", 0.9, BoundingBox(0, 0, 10, 10)),
                Detection(1, "bottle", 0.8, BoundingBox(20, 20, 30, 30)),
            ]
        )
        assert result.count == 2

    def test_filter_by_class(self) -> None:
        """Verify class filtering."""
        result = DetectionResult(
            detections=[
                Detection(0, "person", 0.9, BoundingBox(0, 0, 10, 10)),
                Detection(1, "bottle", 0.8, BoundingBox(20, 20, 30, 30)),
            ]
        )
        persons = result.filter_by_class("person")
        assert len(persons) == 1
        assert persons[0].class_name == "person"

    def test_filter_by_confidence(self) -> None:
        """Verify confidence filtering."""
        result = DetectionResult(
            detections=[
                Detection(0, "person", 0.9, BoundingBox(0, 0, 10, 10)),
                Detection(1, "bottle", 0.3, BoundingBox(20, 20, 30, 30)),
            ]
        )
        high_conf = result.filter_by_confidence(0.5)
        assert len(high_conf) == 1


class TestTrackedObject:
    """Tests for TrackedObject dataclass."""

    def test_wrist_positions_empty(self) -> None:
        """Verify empty wrist positions when insufficient keypoints."""
        obj = TrackedObject(
            track_id=1,
            class_name="person",
            bbox=BoundingBox(0, 0, 10, 10),
            confidence=0.9,
        )
        assert obj.wrist_positions == []

    def test_hand_center_none(self) -> None:
        """Verify hand center is None when no wrists detected."""
        obj = TrackedObject(
            track_id=1,
            class_name="person",
            bbox=BoundingBox(0, 0, 10, 10),
            confidence=0.9,
        )
        assert obj.hand_center is None


class TestIncidentDTO:
    """Tests for IncidentDTO."""

    def test_to_dict(self) -> None:
        """Verify dictionary serialization."""
        dto = IncidentDTO(
            id=1,
            camera_id="cam1",
            camera_name="Camera 1",
            confidence=0.95,
            person_track_id=5,
            waste_type="bottle",
            status=IncidentStatus.PENDING,
        )
        d = dto.to_dict()
        assert d["id"] == 1
        assert d["camera_id"] == "cam1"
        assert d["status"] == "pending"

    def test_default_status(self) -> None:
        """Verify default status is PENDING."""
        dto = IncidentDTO()
        assert dto.status == IncidentStatus.PENDING
