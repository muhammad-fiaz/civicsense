"""Tests for ByteTracker."""

from __future__ import annotations

import numpy as np
import pytest
from civicsense.ai.tracker import ByteTracker
from civicsense.dto.detection import BoundingBox, TrackedObject


def _make_tracked(
    track_id: int = 0,
    x1: float = 0,
    y1: float = 0,
    x2: float = 50,
    y2: float = 50,
    class_name: str = "person",
) -> TrackedObject:
    """Create a TrackedObject for testing."""
    return TrackedObject(
        track_id=track_id,
        class_name=class_name,
        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
        confidence=0.9,
    )


class TestByteTracker:
    """Tests for ByteTracker."""

    def test_initialization(self) -> None:
        """Verify tracker initializes with no tracks."""
        tracker = ByteTracker()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = tracker.update([], frame)
        assert result == []

    def test_first_detection_gets_id(self) -> None:
        """Verify first detection gets a stable ID."""
        tracker = ByteTracker(min_hits=1)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        det = _make_tracked(x1=100, y1=100, x2=200, y2=200)
        result = tracker.update([det], frame)
        assert len(result) == 1
        assert result[0].track_id == 1

    def test_stable_id_across_frames(self) -> None:
        """Verify tracking ID remains stable across frames."""
        tracker = ByteTracker(min_hits=1, max_age=5)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        det1 = _make_tracked(x1=100, y1=100, x2=200, y2=200)
        result1 = tracker.update([det1], frame)
        first_id = result1[0].track_id

        det2 = _make_tracked(x1=105, y1=105, x2=205, y2=205)
        result2 = tracker.update([det2], frame)
        assert result2[0].track_id == first_id

    def test_multiple_objects(self) -> None:
        """Verify multiple objects get unique IDs."""
        tracker = ByteTracker(min_hits=1)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        dets = [
            _make_tracked(x1=10, y1=10, x2=60, y2=60),
            _make_tracked(x1=200, y1=200, x2=250, y2=250),
        ]
        result = tracker.update(dets, frame)
        ids = {r.track_id for r in result}
        assert len(ids) == 2

    def test_reset(self) -> None:
        """Verify reset clears all tracks."""
        tracker = ByteTracker(min_hits=1)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        det = _make_tracked()
        tracker.update([det], frame)
        tracker.reset()

        result = tracker.update([], frame)
        assert result == []

    def test_iou_computation(self) -> None:
        """Verify IoU calculation between boxes."""
        box_a = BoundingBox(x1=0, y1=0, x2=100, y2=100)
        box_b = BoundingBox(x1=50, y1=50, x2=150, y2=150)
        iou = ByteTracker._compute_iou(box_a, box_b)
        assert 0.1 < iou < 0.2

    def test_iou_no_overlap(self) -> None:
        """Verify IoU is 0 for non-overlapping boxes."""
        box_a = BoundingBox(x1=0, y1=0, x2=10, y2=10)
        box_b = BoundingBox(x1=100, y1=100, x2=200, y2=200)
        iou = ByteTracker._compute_iou(box_a, box_b)
        assert iou == 0.0

    def test_track_expiry(self) -> None:
        """Verify tracks expire after max_age frames."""
        tracker = ByteTracker(min_hits=1, max_age=2)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        det = _make_tracked()
        tracker.update([det], frame)
        tracker.update([], frame)
        tracker.update([], frame)

        assert len(tracker._tracks) == 0
