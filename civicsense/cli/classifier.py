"""Classification module for CLI detection.

Analyzes frame detections and classifies littering behavior.
Identifies all ground objects, waste types, and reports detailed findings.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from civicsense.ai.detector import YOLODetector
from civicsense.ai.pose_detector import YOLOPoseDetector
from civicsense.ai.tracker import ByteTracker
from civicsense.config import get_dustbin_candidates, get_waste_classes
from civicsense.core.config import get_config
from civicsense.core.constants import (
    DUSTBIN_PROXIMITY_THRESHOLD,
    GROUND_OBJECT_DISTANCE_THRESHOLD,
    HAND_PROXIMITY_THRESHOLD,
    PERSON_WASTE_DISTANCE_THRESHOLD,
    WASTE_GROUND_Y_THRESHOLD,
)
from civicsense.core.logging import get_logger
from civicsense.dto.detection import BoundingBox, Detection, DetectionResult, PoseResult

logger = get_logger("ai")

# Waste classes loaded from TOML config
WASTE_CANDIDATES: frozenset[str] = frozenset(get_waste_classes())
DUSTBIN_CANDIDATES_SET: frozenset[str] = frozenset(get_dustbin_candidates())
GROUND_OBJECTS: frozenset[str] = WASTE_CANDIDATES


@dataclass
class PersonInfo:
    """Detected person with pose data."""

    bbox: BoundingBox
    confidence: float
    left_wrist: tuple[float, float] | None = None
    right_wrist: tuple[float, float] | None = None
    left_elbow: tuple[float, float] | None = None
    right_elbow: tuple[float, float] | None = None

    @property
    def hand_center(self) -> tuple[float, float] | None:
        """Return the midpoint of both wrists if available."""
        if self.left_wrist and self.right_wrist:
            return (
                (self.left_wrist[0] + self.right_wrist[0]) / 2,
                (self.left_wrist[1] + self.right_wrist[1]) / 2,
            )
        return self.left_wrist or self.right_wrist


@dataclass
class WasteInfo:
    """Detected waste object."""

    class_name: str
    bbox: BoundingBox
    confidence: float
    on_ground: bool = False
    near_person: bool = False
    near_hand: bool = False


@dataclass
class DustbinInfo:
    """Detected dustbin."""

    class_name: str
    bbox: BoundingBox
    confidence: float


@dataclass
class GroundObject:
    """Object detected on the ground level."""

    class_name: str
    bbox: BoundingBox
    confidence: float
    distance_to_nearest_person: float = 0.0


@dataclass
class FrameResult:
    """Analysis result for a frame."""

    frame_idx: int
    persons: list[PersonInfo] = field(default_factory=list)
    waste_objects: list[WasteInfo] = field(default_factory=list)
    dustbins: list[DustbinInfo] = field(default_factory=list)
    ground_objects: list[GroundObject] = field(default_factory=list)
    classification: str = "NO WASTE DETECTED"
    details: str = ""
    waste_count: int = 0
    ground_count: int = 0
    summary: str = ""


def classify_frame(
    frame: np.ndarray,
    detector: YOLODetector,
    pose_detector: YOLOPoseDetector,
    frame_idx: int,
) -> FrameResult:
    """Run detection and classify littering behavior.

    Identifies all persons, waste objects, dustbins, and ground-level items.
    Reports detailed findings including waste types and disposal status.

    Args:
        frame: BGR video frame.
        detector: YOLODetector instance.
        pose_detector: YOLOPoseDetector instance.
        frame_idx: Current frame index.

    Returns:
        FrameResult with classification and detailed report.
    """
    result = FrameResult(frame_idx=frame_idx)

    config = get_config()

    # Detect objects
    det_result = detector.detect(frame, confidence=config.ai.confidence_threshold)

    # Detect poses (non-fatal — classification works without pose)
    pose_result = None
    try:
        pose_result = pose_detector.estimate(
            frame, confidence=config.ai.confidence_threshold
        )
    except Exception as e:
        logger.debug(f"Pose estimation skipped: {e}", module="ai")

    frame_height = frame.shape[0]
    ground_y = frame_height * WASTE_GROUND_Y_THRESHOLD

    # Categorize all detections
    for d in det_result.detections:
        if d.class_name == "person":
            pi = _extract_person(d, pose_result)
            result.persons.append(pi)

        elif d.class_name in WASTE_CANDIDATES:
            on_ground = d.bbox.center[1] > ground_y
            near_person = any(
                _dist(d.bbox.center, p.bbox.center) < PERSON_WASTE_DISTANCE_THRESHOLD
                for p in result.persons
            )
            near_hand = any(
                p.hand_center
                and _dist(d.bbox.center, p.hand_center) < HAND_PROXIMITY_THRESHOLD
                for p in result.persons
            )
            wi = WasteInfo(
                class_name=d.class_name,
                bbox=d.bbox,
                confidence=d.confidence,
                on_ground=on_ground,
                near_person=near_person,
                near_hand=near_hand,
            )
            result.waste_objects.append(wi)

        elif d.class_name in DUSTBIN_CANDIDATES_SET:
            bbox = d.bbox
            aspect = bbox.width / max(bbox.height, 1)
            if 0.3 < aspect < 2.0 and bbox.area > 2000:
                result.dustbins.append(DustbinInfo(d.class_name, d.bbox, d.confidence * 0.7))

    # Identify all ground-level objects (any object near the bottom of frame)
    _identify_ground_objects(result, det_result, frame_height, ground_y)

    # Update waste counts
    result.waste_count = len(result.waste_objects)
    result.ground_count = len(result.ground_objects)

    # Classify
    result.classification, result.details = _classify(result)
    result.summary = _build_summary(result)

    logger.debug(
        f"Frame {frame_idx}: {result.classification} | "
        f"persons={len(result.persons)} waste={result.waste_count} "
        f"ground={result.ground_count} dustbins={len(result.dustbins)}",
        module="ai",
    )

    return result


def _extract_person(detection: Detection, pose_result: PoseResult | None) -> PersonInfo:
    """Extract person info with pose keypoints.

    Args:
        detection: Detection object with bbox and confidence.
        pose_result: Pose estimation result.

    Returns:
        PersonInfo with pose data attached.
    """
    pi = PersonInfo(bbox=detection.bbox, confidence=detection.confidence)
    if pose_result and pose_result.person_bboxes:
        best_iou = 0
        for j, pb in enumerate(pose_result.person_bboxes):
            iou = ByteTracker._compute_iou(detection.bbox, pb)
            if iou > best_iou:
                best_iou = iou
                if j < len(pose_result.keypoints):
                    kps = pose_result.keypoints[j]
                    if len(kps) > 10:
                        if kps[9].confidence > 0.3:
                            pi.left_wrist = (kps[9].x, kps[9].y)
                        if kps[10].confidence > 0.3:
                            pi.right_wrist = (kps[10].x, kps[10].y)
                        if kps[7].confidence > 0.3:
                            pi.left_elbow = (kps[7].x, kps[7].y)
                        if kps[8].confidence > 0.3:
                            pi.right_elbow = (kps[8].x, kps[8].y)
    return pi


def _identify_ground_objects(
    result: FrameResult,
    det_result: DetectionResult,
    frame_height: int,
    ground_y: float,
) -> None:
    """Identify all objects on the ground level.

    Args:
        result: FrameResult to populate ground_objects.
        det_result: Raw detection results.
        frame_height: Height of the frame.
        ground_y: Y threshold for ground level.
    """
    for d in det_result.detections:
        if d.class_name == "person":
            continue

        bbox = d.bbox
        # Object is on ground if its bottom edge is below ground_y
        is_on_ground = bbox.y2 > ground_y

        if is_on_ground:
            # Find distance to nearest person
            min_dist = float("inf")
            for person in result.persons:
                dist = _dist(bbox.center, person.bbox.center)
                min_dist = min(min_dist, dist)

            result.ground_objects.append(
                GroundObject(
                    class_name=d.class_name,
                    bbox=bbox,
                    confidence=d.confidence,
                    distance_to_nearest_person=min_dist
                    if min_dist != float("inf")
                    else 0.0,
                )
            )


def _classify(result: FrameResult) -> tuple[str, str]:
    """Classify waste disposal behavior.

    Analyzes spatial relationships between persons, waste, and dustbins
    to determine if littering is occurring.

    Returns:
        Tuple of (classification, details).
    """
    if not result.persons:
        if result.ground_objects:
            object_names = _format_object_list(result.ground_objects)
            return (
                "GROUND LITTER - No person",
                f"Objects on ground ({object_names}) - pre-existing litter",
            )
        return "NO PERSON DETECTED", "No persons in frame"

    has_dustbin = len(result.dustbins) > 0
    has_waste = len(result.waste_objects) > 0

    if not has_waste and not result.ground_objects:
        return (
            "NO WASTE DETECTED",
            f"Person(s) detected ({len(result.persons)}), no waste",
        )

    # Check each person for littering behavior
    for person in result.persons:
        throwing = _is_throwing(person)
        arm_raised = _is_arm_raised(person)

        # Case 1: Waste + Dustbin present
        if has_waste and has_dustbin:
            for waste in result.waste_objects:
                near_dustbin = any(
                    _dist(waste.bbox.center, db.bbox.center)
                    < DUSTBIN_PROXIMITY_THRESHOLD
                    for db in result.dustbins
                )
                near_hand = waste.near_hand
                if near_dustbin:
                    return (
                        "VALID - Disposed in dustbin",
                        f"Waste ({waste.class_name}) correctly placed near dustbin",
                    )
                if near_hand and not throwing:
                    return (
                        "VALID - Holding waste near dustbin",
                        f"Person holding {waste.class_name} near dustbin",
                    )

        # Case 2: Waste present, no dustbin
        if has_waste and not has_dustbin:
            for waste in result.waste_objects:
                near_hand = waste.near_hand
                on_ground = waste.on_ground

                if near_hand and throwing:
                    return (
                        "FLAGGED - Littering (throwing)",
                        f"Waste ({waste.class_name}) being thrown, no dustbin nearby",
                    )
                if near_hand and arm_raised:
                    return (
                        "FLAGGED - Littering (arm raised)",
                        f"Waste ({waste.class_name}) in raised hand, no dustbin",
                    )
                if near_hand:
                    return (
                        "FLAGGED - Likely littering",
                        f"Waste ({waste.class_name}) held, no dustbin - probable disposal",
                    )
                if (
                    on_ground
                    and _dist(waste.bbox.center, person.bbox.center)
                    < PERSON_WASTE_DISTANCE_THRESHOLD
                ):
                    return (
                        "FLAGGED - Waste on ground near person",
                        f"Waste ({waste.class_name}) on ground near person",
                    )
                if on_ground:
                    return (
                        "FLAGGED - Waste on ground",
                        f"Waste ({waste.class_name}) on ground, no dustbin",
                    )

        # Case 3: No waste detected but throwing motion
        if not has_waste and has_dustbin and throwing:
            return (
                "FLAGGED - Throwing without visible waste",
                "Throwing motion detected, dustbin present but not targeted",
            )

        # Case 4: Ground objects near person without dustbin
        if not has_waste and result.ground_objects and not has_dustbin:
            nearby_ground = [
                g
                for g in result.ground_objects
                if g.distance_to_nearest_person < GROUND_OBJECT_DISTANCE_THRESHOLD
            ]
            if nearby_ground:
                object_names = _format_object_list(nearby_ground)
                if throwing:
                    return (
                        "FLAGGED - Littering (ground objects)",
                        f"Throwing near ground objects ({object_names}), no dustbin",
                    )
                return (
                    "INCONCLUSIVE - Ground objects near person",
                    f"Objects on ground ({object_names}) near person, needs context",
                )

    # Case 5: Ground objects with no person
    if result.ground_objects and not result.waste_objects:
        object_names = _format_object_list(result.ground_objects)
        return (
            "GROUND LITTER DETECTED",
            f"Objects on ground ({object_names}) - possible litter",
        )

    return (
        "INCONCLUSIVE - Needs more frames",
        f"Waste ({result.waste_count} items), ground ({result.ground_count} items)",
    )


def _is_throwing(person: PersonInfo) -> bool:
    """Detect throwing posture using arm extension analysis.

    Args:
        person: PersonInfo with pose data.

    Returns:
        True if throwing posture detected.
    """
    body_center_x = person.bbox.center[0]
    body_width = max(person.bbox.width, 1)
    body_height = person.bbox.height
    shoulder_y = person.bbox.y1 + body_height * 0.25
    waist_y = person.bbox.y1 + body_height * 0.55

    for wrist, elbow in [
        (person.right_wrist, person.right_elbow),
        (person.left_wrist, person.left_elbow),
    ]:
        if wrist and elbow:
            wx, wy = wrist
            _ex, ey = elbow
            if (
                abs(wx - body_center_x) > body_width * 0.7
                and wy > shoulder_y
                and wy > ey
                and wy > waist_y
            ):
                return True
    return False


def _is_arm_raised(person: PersonInfo) -> bool:
    """Detect if arm is raised above shoulder level.

    Args:
        person: PersonInfo with pose data.

    Returns:
        True if arm is raised.
    """
    body_height = person.bbox.height
    shoulder_y = person.bbox.y1 + body_height * 0.25

    return any(
        wrist and wrist[1] < shoulder_y
        for wrist in [person.left_wrist, person.right_wrist]
    )


def _format_object_list(objects: Sequence[GroundObject | WasteInfo | DustbinInfo]) -> str:
    """Format a list of objects into a readable string.

    Args:
        objects: List of objects with class_name attribute.

    Returns:
        Comma-separated string of object class names.
    """
    names = {}
    for obj in objects:
        name = obj.class_name
        names[name] = names.get(name, 0) + 1
    parts = []
    for name, count in sorted(names.items()):
        parts.append(f"{name}x{count}" if count > 1 else name)
    return ", ".join(parts) if parts else "unknown"


def _build_summary(result: FrameResult) -> str:
    """Build a human-readable summary of the frame analysis.

    Args:
        result: FrameResult with all detection data.

    Returns:
        Summary string.
    """
    parts = []
    parts.append(f"Persons: {len(result.persons)}")
    if result.waste_count:
        waste_names = {}
        for w in result.waste_objects:
            waste_names[w.class_name] = waste_names.get(w.class_name, 0) + 1
        waste_str = ", ".join(
            f"{name}x{count}" if count > 1 else name
            for name, count in sorted(waste_names.items())
        )
        parts.append(f"Waste: {waste_str}")
    if result.ground_count:
        ground_names = {}
        for g in result.ground_objects:
            ground_names[g.class_name] = ground_names.get(g.class_name, 0) + 1
        ground_str = ", ".join(
            f"{name}x{count}" if count > 1 else name
            for name, count in sorted(ground_names.items())
        )
        parts.append(f"Ground: {ground_str}")
    parts.append(f"Dustbins: {len(result.dustbins)}")
    return " | ".join(parts)


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Euclidean distance between two points.

    Args:
        a: First point (x, y).
        b: Second point (x, y).

    Returns:
        Distance in pixels.
    """
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
