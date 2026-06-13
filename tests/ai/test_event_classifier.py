"""Tests for event classifier."""

from __future__ import annotations

import numpy as np
from civicsense.ai.event_classifier import EventClassifier, EventCategory
from civicsense.dto.detection import BoundingBox, Detection, DetectionResult


class TestEventClassifier:
    """Tests for EventClassifier."""

    def test_initialization(self) -> None:
        """Verify classifier initializes with zero counts."""
        classifier = EventClassifier()
        stats = classifier.event_stats
        assert all(v == 0 for v in stats.values())

    def test_classify_person(self) -> None:
        """Verify person detections are categorized."""
        classifier = EventClassifier()
        result = DetectionResult(
            detections=[
                Detection(0, "person", 0.9, BoundingBox(100, 100, 200, 300)),
            ],
            frame_shape=(480, 640, 3),
        )
        events = classifier.classify_frame(result)
        assert len(events[EventCategory.PERSON_DETECTED]) == 1

    def test_classify_waste(self) -> None:
        """Verify waste detections are categorized."""
        classifier = EventClassifier()
        result = DetectionResult(
            detections=[
                Detection(1, "bottle", 0.85, BoundingBox(150, 200, 180, 250)),
            ],
            frame_shape=(480, 640, 3),
        )
        events = classifier.classify_frame(result)
        assert len(events[EventCategory.WASTE_DETECTED]) == 1

    def test_classify_dustbin(self) -> None:
        """Verify dustbin detections are categorized."""
        classifier = EventClassifier()
        result = DetectionResult(
            detections=[
                Detection(8, "dustbin", 0.9, BoundingBox(500, 300, 600, 450)),
            ],
            frame_shape=(480, 640, 3),
        )
        events = classifier.classify_frame(result)
        assert len(events[EventCategory.DUSTBIN_DETECTED]) == 1

    def test_person_carrying_waste(self) -> None:
        """Verify person-carrying-waste is detected when near."""
        classifier = EventClassifier()
        result = DetectionResult(
            detections=[
                Detection(0, "person", 0.9, BoundingBox(100, 100, 200, 300)),
                Detection(1, "bottle", 0.85, BoundingBox(150, 200, 180, 250)),
            ],
            frame_shape=(480, 640, 3),
        )
        events = classifier.classify_frame(result)
        assert len(events[EventCategory.PERSON_CARRYING_WASTE]) >= 1

    def test_reset_stats(self) -> None:
        """Verify reset clears all counters."""
        classifier = EventClassifier()
        result = DetectionResult(
            detections=[
                Detection(0, "person", 0.9, BoundingBox(0, 0, 10, 10)),
            ],
            frame_shape=(480, 640, 3),
        )
        classifier.classify_frame(result)
        classifier.reset_stats()
        assert all(v == 0 for v in classifier.event_stats.values())
