"""Tests for littering engine."""

from __future__ import annotations

import pytest
from civicsense.ai.littering_engine import LitteringClassifier, LitteringState
from civicsense.dto.detection import BoundingBox, TrackedObject


def _make_person(
    track_id: int = 1,
    x1: float = 100,
    y1: float = 100,
    x2: float = 200,
    y2: float = 300,
    wrist_x: float = 180,
    wrist_y: float = 250,
) -> TrackedObject:
    """Create a tracked person with keypoints for testing."""
    from civicsense.dto.detection import Keypoint

    keypoints = [Keypoint(x=0, y=0, confidence=0.0)] * 11
    keypoints[9] = Keypoint(x=wrist_x, y=wrist_y, confidence=0.9)
    keypoints[10] = Keypoint(x=wrist_x + 10, y=wrist_y, confidence=0.9)

    return TrackedObject(
        track_id=track_id,
        class_name="person",
        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
        confidence=0.9,
        keypoints=keypoints,
    )


def _make_waste(
    track_id: int = 2,
    x1: float = 170,
    y1: float = 240,
    x2: float = 190,
    y2: float = 260,
) -> TrackedObject:
    """Create a tracked waste object for testing."""
    return TrackedObject(
        track_id=track_id,
        class_name="bottle",
        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
        confidence=0.85,
    )


def _make_dustbin(
    track_id: int = 3,
    x1: float = 500,
    y1: float = 300,
    x2: float = 600,
    y2: float = 450,
) -> TrackedObject:
    """Create a tracked dustbin for testing."""
    return TrackedObject(
        track_id=track_id,
        class_name="dustbin",
        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
        confidence=0.9,
    )


class TestLitteringClassifier:
    """Tests for LitteringClassifier."""

    def test_initialization(self) -> None:
        """Verify classifier initializes empty state."""
        classifier = LitteringClassifier()
        assert len(classifier._states) == 0

    def test_no_persons_no_events(self) -> None:
        """Verify no events when no persons detected."""
        classifier = LitteringClassifier()
        waste = [_make_waste()]
        dustbins = [_make_dustbin()]
        events = classifier.classify([], waste, dustbins, 480)
        assert len(events) == 0

    def test_no_waste_no_events(self) -> None:
        """Verify no events when no waste detected."""
        classifier = LitteringClassifier()
        persons = [_make_person()]
        dustbins = [_make_dustbin()]
        events = classifier.classify(persons, [], dustbins, 480)
        assert len(events) == 0

    def test_near_hand_detection(self) -> None:
        """Verify waste near hand is detected."""
        classifier = LitteringClassifier()
        person = _make_person(wrist_x=175, wrist_y=245)
        waste = _make_waste(x1=170, y1=240, x2=190, y2=260)
        dustbins = [_make_dustbin()]

        classifier.classify([person], [waste], dustbins, 480)
        state = classifier._states.get(2)
        assert state is not None
        assert state.near_hand is True

    def test_reset(self) -> None:
        """Verify reset clears all state."""
        classifier = LitteringClassifier()
        person = _make_person()
        waste = _make_waste()
        classifier.classify([person], [waste], [_make_dustbin()], 480)
        assert len(classifier._states) > 0

        classifier.reset()
        assert len(classifier._states) == 0

    def test_state_is_littering_conditions(self) -> None:
        """Verify LitteringState.is_littering checks all conditions."""
        state = LitteringState(
            waste_track_id=1,
            person_track_id=1,
            waste_type="bottle",
            near_hand=True,
            left_hand=True,
            falling=True,
            reached_ground=True,
            entered_dustbin=False,
        )
        assert state.is_littering is True

    def test_state_not_littering_if_in_dustbin(self) -> None:
        """Verify littering is false when waste enters dustbin."""
        state = LitteringState(
            waste_track_id=1,
            person_track_id=1,
            waste_type="bottle",
            near_hand=True,
            left_hand=True,
            falling=True,
            reached_ground=True,
            entered_dustbin=True,
        )
        assert state.is_littering is False

    def test_state_not_littering_if_not_fallen(self) -> None:
        """Verify littering is false when waste hasn't reached ground."""
        state = LitteringState(
            waste_track_id=1,
            person_track_id=1,
            waste_type="bottle",
            near_hand=True,
            left_hand=True,
            falling=True,
            reached_ground=False,
            entered_dustbin=False,
        )
        assert state.is_littering is False
