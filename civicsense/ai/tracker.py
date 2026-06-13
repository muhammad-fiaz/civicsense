"""ByteTrack multi-object tracker implementation.

Provides stable tracking IDs for detected persons and waste objects
across consecutive video frames using the ByteTrack algorithm.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from civicsense.core.constants import MAX_TRACKING_AGE, MAX_TRACKING_DISTANCE
from civicsense.core.logging import get_logger
from civicsense.dto.detection import BoundingBox, TrackedObject

logger = get_logger("ai")


class ByteTracker:
    """ByteTrack-inspired multi-object tracker.

    Associates detections across frames using IoU and Hungarian matching
    with configurable age and distance limits for track lifecycle management.
    """

    def __init__(
        self,
        max_age: int = MAX_TRACKING_AGE,
        max_distance: float = MAX_TRACKING_DISTANCE,
        min_hits: int = 3,
    ) -> None:
        """Initialize the ByteTracker.

        Args:
            max_age: Maximum frames a track can survive without detection.
            max_distance: Maximum association distance in pixels.
            min_hits: Minimum consecutive detections to confirm a track.
        """
        self.max_age = max_age
        self.max_distance = max_distance
        self.min_hits = min_hits
        self._tracks: dict[int, TrackedObject] = {}
        self._next_id: int = 1
        self._frame_count: int = 0

    def initialize(self, **kwargs: Any) -> None:
        """Initialize or reinitialize the tracker.

        Args:
            **kwargs: Configuration overrides (max_age, max_distance, min_hits).
        """
        if "max_age" in kwargs:
            self.max_age = kwargs["max_age"]
        if "max_distance" in kwargs:
            self.max_distance = kwargs["max_distance"]
        if "min_hits" in kwargs:
            self.min_hits = kwargs["min_hits"]
        self.reset()
        logger.debug("ByteTracker initialized", module="ai")

    def update(
        self,
        detections: list[TrackedObject],
        frame: NDArray[np.uint8],
    ) -> list[TrackedObject]:
        """Update tracking with new detections.

        Args:
            detections: Current frame detections.
            frame: The current video frame.

        Returns:
            List of tracked objects with stable IDs.
        """
        self._frame_count += 1

        if not detections:
            self._age_tracks()
            return self._get_active_tracks()

        matched, unmatched_dets, _unmatched_tracks = self._associate(detections)

        for det_idx, track_idx in matched:
            det = detections[det_idx]
            track = list(self._tracks.values())[track_idx]
            track.bbox = det.bbox
            track.confidence = det.confidence
            track.class_name = det.class_name
            track.frame_idx = self._frame_count
            track.hits += 1
            track.age = 0
            track.keypoints = det.keypoints

        for det_idx in unmatched_dets:
            det = detections[det_idx]
            track_id = self._next_id
            self._next_id += 1
            self._tracks[track_id] = TrackedObject(
                track_id=track_id,
                class_name=det.class_name,
                bbox=det.bbox,
                confidence=det.confidence,
                frame_idx=self._frame_count,
                keypoints=det.keypoints,
                age=0,
                hits=1,
            )

        self._age_tracks()
        return self._get_active_tracks()

    def reset(self) -> None:
        """Reset tracker state and clear all tracks."""
        self._tracks.clear()
        self._next_id = 1
        self._frame_count = 0
        logger.debug("ByteTracker reset", module="ai")

    def _associate(
        self,
        detections: list[TrackedObject],
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Associate detections with existing tracks using IoU.

        Args:
            detections: Current frame detections.

        Returns:
            Tuple of (matched_pairs, unmatched_detection_indices,
            unmatched_track_indices).
        """
        if not self._tracks:
            return [], list(range(len(detections))), []

        track_list = list(self._tracks.values())
        iou_matrix = np.zeros((len(detections), len(track_list)))

        for d, det in enumerate(detections):
            for t, track in enumerate(track_list):
                iou_matrix[d, t] = self._compute_iou(det.bbox, track.bbox)

        matched: list[tuple[int, int]] = []
        unmatched_dets = list(range(len(detections)))
        unmatched_tracks = list(range(len(track_list)))

        for _ in range(min(len(detections), len(track_list))):
            if iou_matrix.size == 0:
                break
            max_idx = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
            max_val = iou_matrix[max_idx]

            if max_val <= 0:
                break

            d_idx, t_idx = int(max_idx[0]), int(max_idx[1])
            matched.append((d_idx, t_idx))
            unmatched_dets.remove(d_idx)
            unmatched_tracks.remove(t_idx)
            iou_matrix[d_idx, :] = 0
            iou_matrix[:, t_idx] = 0

        return matched, unmatched_dets, unmatched_tracks

    def _age_tracks(self) -> None:
        """Increment age for all tracks and remove expired ones."""
        to_remove: list[int] = []
        for track_id, track in self._tracks.items():
            track.age += 1
            if track.age > self.max_age:
                to_remove.append(track_id)
        for track_id in to_remove:
            del self._tracks[track_id]

    def _get_active_tracks(self) -> list[TrackedObject]:
        """Return tracks that meet the minimum hit threshold.

        Returns:
            List of confirmed tracked objects.
        """
        return [t for t in self._tracks.values() if t.hits >= self.min_hits]

    @staticmethod
    def _compute_iou(box_a: BoundingBox, box_b: BoundingBox) -> float:
        """Compute Intersection over Union between two bounding boxes.

        Args:
            box_a: First bounding box.
            box_b: Second bounding box.

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

        if union <= 0:
            return 0.0
        return intersection / union
