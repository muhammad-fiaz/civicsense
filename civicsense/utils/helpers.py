"""Utility functions for CivicSense.

Provides helper functions for image annotation, file operations,
geometry calculations, and other common operations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from civicsense.dto.detection import TrackedObject


def annotate_frame(
    frame: NDArray[np.uint8],
    persons: list[TrackedObject],
    waste_objects: list[TrackedObject],
    dustbins: list[TrackedObject],
    show_pose: bool = True,
    show_tracking_ids: bool = True,
    show_confidence: bool = True,
) -> NDArray[np.uint8]:
    """Draw bounding boxes, labels, and skeletons on a video frame.

    Args:
        frame: Input video frame.
        persons: Tracked person objects.
        waste_objects: Tracked waste objects.
        dustbins: Tracked dustbin objects.
        show_pose: Whether to draw pose skeletons.
        show_tracking_ids: Whether to show tracking IDs.
        show_confidence: Whether to show confidence scores.

    Returns:
        Annotated copy of the input frame.
    """
    annotated = frame.copy()

    for person in persons:
        _draw_bbox(
            annotated,
            person,
            (0, 255, 0),
            "Person",
            show_tracking_ids,
            show_confidence,
        )
        if show_pose and person.keypoints:
            _draw_skeleton(annotated, person.keypoints)

    for waste in waste_objects:
        _draw_bbox(
            annotated,
            waste,
            (0, 0, 255),
            "Waste",
            show_tracking_ids,
            show_confidence,
        )

    for dustbin in dustbins:
        _draw_bbox(
            annotated,
            dustbin,
            (255, 165, 0),
            "Dustbin",
            show_tracking_ids,
            show_confidence,
        )

    return annotated


def _draw_bbox(
    frame: NDArray[np.uint8],
    obj: TrackedObject,
    color: tuple[int, int, int],
    label: str,
    show_id: bool,
    show_conf: bool,
) -> None:
    """Draw a bounding box with optional labels.

    Args:
        frame: Frame to draw on.
        obj: The tracked object.
        color: BGR color tuple.
        label: Class label prefix.
        show_id: Whether to display tracking ID.
        show_conf: Whether to display confidence.
    """
    bbox = obj.bbox
    pt1 = (int(bbox.x1), int(bbox.y1))
    pt2 = (int(bbox.x2), int(bbox.y2))
    cv2.rectangle(frame, pt1, pt2, color, 2)

    parts = [label]
    if show_id:
        parts.append(f"ID:{obj.track_id}")
    if show_conf:
        parts.append(f"{obj.confidence:.2f}")

    text = " ".join(parts)
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    text_pt = (int(bbox.x1), int(bbox.y1) - 5)
    bg_pt1 = (text_pt[0], text_pt[1] - text_size[1] - 4)
    bg_pt2 = (text_pt[0] + text_size[0], text_pt[1] + 4)

    cv2.rectangle(frame, bg_pt1, bg_pt2, color, -1)
    cv2.putText(
        frame,
        text,
        text_pt,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )


def _draw_skeleton(
    frame: NDArray[np.uint8],
    keypoints: list[Any],
    color: tuple[int, int, int] = (255, 255, 0),
) -> None:
    """Draw pose keypoints and connecting lines.

    Args:
        frame: Frame to draw on.
        keypoints: List of Keypoint objects.
        color: BGR color tuple for skeleton lines.
    """
    skeleton_pairs = [
        (5, 6),
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),
        (11, 12),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
        (5, 11),
        (6, 12),
    ]

    for kp in keypoints:
        if kp.confidence > 0.3:
            pt = (int(kp.x), int(kp.y))
            cv2.circle(frame, pt, 4, color, -1)

    for i, j in skeleton_pairs:
        if i < len(keypoints) and j < len(keypoints):
            kpi = keypoints[i]
            kpj = keypoints[j]
            if kpi.confidence > 0.3 and kpj.confidence > 0.3:
                pt1 = (int(kpi.x), int(kpi.y))
                pt2 = (int(kpj.x), int(kpj.y))
                cv2.line(frame, pt1, pt2, color, 2)


def compute_distance(
    point_a: tuple[float, float],
    point_b: tuple[float, float],
) -> float:
    """Compute Euclidean distance between two points.

    Args:
        point_a: First point (x, y).
        point_b: Second point (x, y).

    Returns:
        The Euclidean distance.
    """
    return ((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2) ** 0.5


def ensure_directory(path: Path) -> Path:
    """Create a directory and parents if they do not exist.

    Args:
        path: The directory path to create.

    Returns:
        The same path, guaranteed to exist.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path
