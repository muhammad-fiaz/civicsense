"""Tests for utility functions."""

from __future__ import annotations

import numpy as np
from civicsense.utils.helpers import (
    annotate_frame,
    compute_distance,
    ensure_directory,
)
from civicsense.dto.detection import BoundingBox, TrackedObject


class TestComputeDistance:
    """Tests for compute_distance."""

    def test_same_point(self) -> None:
        """Verify distance between same point is zero."""
        assert compute_distance((0, 0), (0, 0)) == 0.0

    def test_unit_distance(self) -> None:
        """Verify unit distance calculation."""
        assert compute_distance((0, 0), (3, 4)) == 5.0

    def test_negative_coordinates(self) -> None:
        """Verify distance with negative coordinates."""
        assert compute_distance((-1, -1), (2, 3)) == 5.0


class TestAnnotateFrame:
    """Tests for annotate_frame."""

    def test_annotate_empty(self) -> None:
        """Verify annotating with no objects returns frame."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = annotate_frame(frame, [], [], [])
        assert result.shape == frame.shape

    def test_annotate_with_objects(self) -> None:
        """Verify annotating with objects modifies frame."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        person = TrackedObject(
            track_id=1,
            class_name="person",
            bbox=BoundingBox(100, 100, 200, 300),
            confidence=0.9,
        )
        result = annotate_frame(frame, [person], [], [])
        assert result.shape == frame.shape
        assert not np.array_equal(result, frame)


class TestEnsureDirectory:
    """Tests for ensure_directory."""

    def test_creates_directory(self, tmp_path) -> None:
        """Verify directory is created."""
        target = tmp_path / "new_dir"
        result = ensure_directory(target)
        assert result.exists()
        assert result.is_dir()
