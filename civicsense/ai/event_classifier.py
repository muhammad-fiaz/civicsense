"""Event classifier for categorizing detection events.

Maps raw detection results to semantic event types used by the
littering engine and incident creation pipeline.
"""

from __future__ import annotations

from enum import StrEnum

from civicsense.core.constants import (
    DUSTBIN_PROXIMITY_THRESHOLD,
    PERSON_WASTE_DISTANCE_THRESHOLD,
    WASTE_CLASSES,
)
from civicsense.core.logging import get_logger
from civicsense.dto.detection import Detection, DetectionResult

logger = get_logger("ai")


class EventCategory(StrEnum):
    """Classification categories for detection events."""

    PERSON_DETECTED = "person_detected"
    WASTE_DETECTED = "waste_detected"
    DUSTBIN_DETECTED = "dustbin_detected"
    PERSON_CARRYING_WASTE = "person_carrying_waste"
    WASTE_DROPPED = "waste_dropped"
    WASTE_IN_DUSTBIN = "waste_in_dustbin"
    LITTERING = "littering"


class EventClassifier:
    """Classifies detection results into semantic event categories.

    Analyzes spatial relationships between detections to determine
    the nature of each event in the monitoring pipeline.
    """

    def __init__(self) -> None:
        """Initialize the event classifier."""
        self._event_counts: dict[EventCategory, int] = {cat: 0 for cat in EventCategory}

    def classify_frame(
        self,
        detection_result: DetectionResult,
    ) -> dict[EventCategory, list[Detection]]:
        """Classify all detections in a frame into event categories.

        Args:
            detection_result: The raw detection results for a frame.

        Returns:
            Dictionary mapping event categories to relevant detections.
        """
        events: dict[EventCategory, list[Detection]] = {
            cat: [] for cat in EventCategory
        }

        persons = detection_result.filter_by_class("person")
        waste = [
            d for d in detection_result.detections if d.class_name in WASTE_CLASSES
        ]
        dustbins = detection_result.filter_by_class("dustbin")

        events[EventCategory.PERSON_DETECTED] = persons
        events[EventCategory.WASTE_DETECTED] = waste
        events[EventCategory.DUSTBIN_DETECTED] = dustbins

        for p in persons:
            for w in waste:
                if self._is_near(
                    p.bbox.center,
                    w.bbox.center,
                    threshold=PERSON_WASTE_DISTANCE_THRESHOLD,
                ):
                    events[EventCategory.PERSON_CARRYING_WASTE].append(w)
                    break

        for w in waste:
            near_dustbin = any(
                self._is_near(
                    w.bbox.center, d.bbox.center, threshold=DUSTBIN_PROXIMITY_THRESHOLD
                )
                for d in dustbins
            )
            if near_dustbin:
                events[EventCategory.WASTE_IN_DUSTBIN].append(w)

        for cat, detections in events.items():
            self._event_counts[cat] += len(detections)

        return events

    @property
    def event_stats(self) -> dict[str, int]:
        """Return cumulative event counts.

        Returns:
            Dictionary of event category names to their cumulative counts.
        """
        return {cat.value: count for cat, count in self._event_counts.items()}

    def reset_stats(self) -> None:
        """Reset all cumulative event counters."""
        self._event_counts = {cat: 0 for cat in EventCategory}

    @staticmethod
    def _is_near(
        point_a: tuple[float, float],
        point_b: tuple[float, float],
        threshold: float = 100.0,
    ) -> bool:
        """Check if two points are within a distance threshold.

        Args:
            point_a: First point (x, y).
            point_b: Second point (x, y).
            threshold: Maximum distance in pixels.

        Returns:
            True if the points are within the threshold.
        """
        dist = ((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2) ** 0.5
        return dist <= threshold
