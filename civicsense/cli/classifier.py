"""Classification module for CLI detection.

Analyzes frame detections and classifies littering behavior.
Identifies all ground objects, waste types, and reports detailed findings.
Handles partial persons (hands from cars), hand-only throwing, and
waste detection without person presence.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from civicsense.ai.detector import YOLODetector
from civicsense.ai.pose_detector import YOLOPoseDetector
from civicsense.ai.tracker import ByteTracker
from civicsense.config import get_dustbin_candidates, get_non_litter_classes, get_waste_classes
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
NON_LITTER: frozenset[str] = frozenset(get_non_litter_classes())
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
    is_partial: bool = False

    @property
    def hand_center(self) -> tuple[float, float] | None:
        """Return the midpoint of both wrists if available, or estimate from bbox."""
        if self.left_wrist and self.right_wrist:
            return (
                (self.left_wrist[0] + self.right_wrist[0]) / 2,
                (self.left_wrist[1] + self.right_wrist[1]) / 2,
            )
        if self.left_wrist or self.right_wrist:
            return self.left_wrist or self.right_wrist
        # Estimate hand position from bbox (bottom 40% of body, at sides)
        return self._estimate_hand_position()

    def _estimate_hand_position(self) -> tuple[float, float]:
        """Estimate hand position from bounding box when no pose data.

        Hands are typically at 60-80% of body height, at the sides.
        """
        bbox = self.bbox
        hand_y = bbox.y1 + bbox.height * 0.65
        # Return right side hand estimate (most common throwing side)
        hand_x = bbox.x2
        return (hand_x, hand_y)

    @property
    def estimated_hands(self) -> list[tuple[float, float]]:
        """Return all available hand positions (pose + estimated)."""
        hands = []
        if self.left_wrist:
            hands.append(self.left_wrist)
        if self.right_wrist:
            hands.append(self.right_wrist)
        if not hands:
            # Add estimated positions for both sides
            bbox = self.bbox
            hand_y = bbox.y1 + bbox.height * 0.65
            hands.append((bbox.x2, hand_y))  # right side
            hands.append((bbox.x1, hand_y))  # left side
        return hands


@dataclass
class WasteInfo:
    """Detected waste object."""

    class_name: str
    bbox: BoundingBox
    confidence: float
    on_ground: bool = False
    near_person: bool = False
    near_hand: bool = False
    in_air: bool = False


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

    Identifies all persons (including partial), waste objects, dustbins,
    and ground-level items. Always checks hand-waste proximity using
    estimated hand positions when pose data is unavailable.

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

    # Detect objects with lower threshold to catch partial persons
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
    frame_width = frame.shape[1]
    ground_y = frame_height * WASTE_GROUND_Y_THRESHOLD

    # Categorize all detections
    for d in det_result.detections:
        if d.class_name == "person":
            pi = _extract_person(d, pose_result, frame_width)
            result.persons.append(pi)

        elif d.class_name in WASTE_CANDIDATES and d.class_name not in NON_LITTER:
            on_ground = d.bbox.center[1] > ground_y
            # Waste is "in air" if it's in the middle of frame and not on ground
            in_air = (
                not on_ground
                and d.bbox.center[1] < ground_y
                and d.bbox.center[1] > frame_height * 0.2
            )
            near_person = any(
                _dist(d.bbox.center, p.bbox.center) < PERSON_WASTE_DISTANCE_THRESHOLD
                for p in result.persons
            )
            near_hand = _check_waste_near_hand(d, result.persons, frame_width)
            wi = WasteInfo(
                class_name=d.class_name,
                bbox=d.bbox,
                confidence=d.confidence,
                on_ground=on_ground,
                near_person=near_person,
                near_hand=near_hand,
                in_air=in_air,
            )
            result.waste_objects.append(wi)

        elif d.class_name in DUSTBIN_CANDIDATES_SET:
            bbox = d.bbox
            aspect = bbox.width / max(bbox.height, 1)
            if 0.3 < aspect < 2.0 and bbox.area > 2000:
                result.dustbins.append(DustbinInfo(d.class_name, d.bbox, d.confidence * 0.7))

    # Identify all ground-level objects (any object near the bottom of frame)
    _identify_ground_objects(result, det_result, frame_height, ground_y)

    # Ground litter analysis: detect small litter that YOLO misses at normal confidence
    # by running a second pass with very low threshold on the ground region only
    _analyze_ground_litter(result, detector, frame, frame_height, frame_width, ground_y)

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


def _check_waste_near_hand(
    waste_detection: Detection,
    persons: list[PersonInfo],
    frame_width: int,
) -> bool:
    """Check if waste is near any person's hand (pose or estimated).

    Uses actual wrist positions when available, falls back to
    estimated hand positions from bounding box.

    Args:
        waste_detection: The waste detection to check.
        persons: List of detected persons.
        frame_width: Frame width for proximity scaling.

    Returns:
        True if waste is near a hand.
    """
    waste_center = waste_detection.bbox.center

    for person in persons:
        # Check all available hand positions (pose + estimated)
        for hand_pos in person.estimated_hands:
            dist = _dist(waste_center, hand_pos)
            # Scale threshold based on person size (larger person = larger reach)
            reach = max(person.bbox.width * 0.8, HAND_PROXIMITY_THRESHOLD)
            if dist < reach:
                return True

        # Also check if waste overlaps with person's bbox (being held)
        pb = person.bbox
        if (
            waste_center[0] >= pb.x1
            and waste_center[0] <= pb.x2
            and waste_center[1] >= pb.y1
            and waste_center[1] <= pb.y2
        ):
            return True

    return False


def _extract_person(
    detection: Detection, pose_result: PoseResult | None, frame_width: int
) -> PersonInfo:
    """Extract person info with pose keypoints.

    Detects partial persons (hands/arms from cars) by checking if the
    person bbox is at the frame edge and relatively small.

    Args:
        detection: Detection object with bbox and confidence.
        pose_result: Pose estimation result.
        frame_width: Frame width for partial detection.

    Returns:
        PersonInfo with pose data attached.
    """
    bbox = detection.bbox
    # A person is "partial" if bbox is at frame edge or very narrow
    is_partial = (
        bbox.x1 < frame_width * 0.05
        or bbox.x2 > frame_width * 0.95
        or bbox.width < frame_width * 0.08
    )

    pi = PersonInfo(bbox=bbox, confidence=detection.confidence, is_partial=is_partial)
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
        if d.class_name in NON_LITTER:
            continue

        bbox = d.bbox
        # Object is on ground if its bottom edge is below ground_y
        is_on_ground = bbox.y2 > ground_y

        # Filter out objects that are too small (noise) or too large (vehicles, buildings)
        min_area = 500
        max_area = frame_height * frame_height * 0.3
        if bbox.area < min_area or bbox.area > max_area:
            continue

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


# Vehicle/infrastructure classes to ignore in ground litter analysis
_VEHICLE_CLASSES = frozenset({
    "car", "truck", "bus", "motorcycle", "bicycle", "train", "boat", "airplane",
})
_INFRA_CLASSES = frozenset({
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "street sign", "pole", "wall", "fence",
})


def _analyze_ground_litter(
    result: FrameResult,
    detector: YOLODetector,
    frame: np.ndarray,
    frame_height: int,
    frame_width: int,
    ground_y: float,
) -> None:
    """Analyze ground region for scattered litter using visual analysis.

    When YOLO cannot detect individual litter items (too small, no matching
    COCO class), this function analyzes the ground region visually:
    1. Checks for color/texture variation indicating scattered debris
    2. Detects dark spots and irregular patterns on the ground
    3. Uses scene context (person present + ground clutter = litter)

    Args:
        result: FrameResult to update with ground litter findings.
        detector: YOLODetector instance (unused, kept for API consistency).
        frame: Full BGR frame.
        frame_height: Frame height.
        frame_width: Frame width.
        ground_y: Y threshold for ground level.
    """
    # Only analyze if person is present
    if not result.persons:
        return

    # Skip if ground objects already detected
    if result.ground_objects:
        return

    # Extract ground region (bottom 35% of frame)
    ground_start = int(frame_height * 0.65)
    ground_region = frame[ground_start:, :]

    if ground_region.size == 0 or ground_region.shape[0] < 10:
        return

    # Convert to grayscale for texture analysis
    import cv2

    gray = cv2.cvtColor(ground_region, cv2.COLOR_BGR2GRAY)

    # Detect edges (litter creates irregular edges on ground)
    edges = cv2.Canny(gray, 30, 100)

    # Count edge pixels (more edges = more objects on ground)
    edge_ratio = np.count_nonzero(edges) / max(edges.size, 1)

    # Detect dark spots (potential litter) using thresholding
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    dark_ratio = np.count_nonzero(thresh) / max(thresh.size, 1)

    # Detect contours (individual litter pieces)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Filter small contours (noise) vs medium contours (litter)
    litter_contours = [
        c for c in contours
        if 50 < cv2.contourArea(c) < 5000
    ]

    # Calculate ground clutter score
    # High edge ratio + high dark ratio + many contours = likely litter
    clutter_score = (edge_ratio * 10) + (dark_ratio * 5) + len(litter_contours) * 0.5

    logger.debug(
        f"Ground analysis: edge={edge_ratio:.3f} dark={dark_ratio:.3f} "
        f"contours={len(litter_contours)} clutter={clutter_score:.1f}",
        module="ai",
    )

    # Threshold for ground litter detection
    # If clutter score > 5, ground likely has scattered objects
    if clutter_score > 5.0 and len(litter_contours) >= 3:
        # Find person's feet position for proximity check
        for person in result.persons:
            person_feet_y = person.bbox.y2
            # Check if ground clutter is near the person
            person_ground_dist = abs(person_feet_y - ground_start)
            if person_ground_dist < frame_height * 0.4:
                # Create a synthetic ground litter marker
                # Place it at the center of the ground region near the person
                marker_x = person.bbox.center[0]
                marker_y = ground_start + (ground_region.shape[0] // 2)
                bbox = BoundingBox(
                    x1=marker_x - 30,
                    y1=marker_y - 15,
                    x2=marker_x + 30,
                    y2=marker_y + 15,
                )
                result.ground_objects.append(
                    GroundObject(
                        class_name="ground_litter",
                        bbox=bbox,
                        confidence=min(clutter_score / 20.0, 0.9),
                        distance_to_nearest_person=person_ground_dist,
                    )
                )
                result.ground_count = len(result.ground_objects)
                logger.debug(
                    f"Ground litter detected: clutter_score={clutter_score:.1f}, "
                    f"contours={len(litter_contours)} near person",
                    module="ai",
                )
                return


def _classify(result: FrameResult) -> tuple[str, str]:
    """Classify waste disposal behavior.

    Analyzes spatial relationships between persons, waste, and dustbins
    to determine if littering is occurring. Handles:
    - Full person with hand detection
    - Partial person (hand/arm from car window)
    - Waste without person (mid-air or on ground)
    - Multiple waste items

    Returns:
        Tuple of (classification, details).
    """
    has_dustbin = len(result.dustbins) > 0
    has_waste = len(result.waste_objects) > 0
    has_persons = len(result.persons) > 0

    # Case A: No persons detected
    if not has_persons:
        if has_waste:
            # Check if any waste is in mid-air (being thrown)
            air_waste = [w for w in result.waste_objects if w.in_air]
            ground_waste = [w for w in result.waste_objects if w.on_ground]
            if air_waste:
                names = _format_waste_list(air_waste)
                return (
                    "FLAGGED - Waste in air (no person)",
                    f"Waste ({names}) detected in mid-air, likely thrown",
                )
            if ground_waste:
                names = _format_waste_list(ground_waste)
                return (
                    "GROUND LITTER - No person",
                    f"Waste ({names}) on ground - pre-existing litter",
                )
            names = _format_waste_list(result.waste_objects)
            return (
                "FLAGGED - Waste detected (no person)",
                f"Waste ({names}) detected, no person visible",
            )
        if result.ground_objects:
            object_names = _format_object_list(result.ground_objects)
            return (
                "GROUND LITTER - No person",
                f"Objects on ground ({object_names}) - pre-existing litter",
            )
        return "NO PERSON DETECTED", "No persons in frame"

    # Case B: Persons detected, no waste
    if not has_waste and not result.ground_objects:
        return (
            "NO WASTE DETECTED",
            f"Person(s) detected ({len(result.persons)}), no waste",
        )

    # Check each person for littering behavior
    for person in result.persons:
        throwing = _is_throwing(person)
        arm_raised = _is_arm_raised(person)
        hand_lowered = _is_hand_lowered(person)

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
                near_person = waste.near_person
                in_air = waste.in_air

                # Active throwing/dropping from hand
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
                if near_hand and hand_lowered:
                    return (
                        "FLAGGED - Littering (dropping)",
                        f"Waste ({waste.class_name}) being dropped from hand, no dustbin",
                    )
                # Waste in hand without action evidence — still suspicious
                if near_hand:
                    return (
                        "FLAGGED - Likely littering",
                        f"Waste ({waste.class_name}) held, no dustbin - probable disposal",
                    )
                # Waste in mid-air near person (being thrown)
                if in_air and near_person:
                    return (
                        "FLAGGED - Waste thrown",
                        f"Waste ({waste.class_name}) in air near person, no dustbin",
                    )
                # Waste on ground near person
                if on_ground and near_person:
                    return (
                        "FLAGGED - Waste on ground near person",
                        f"Waste ({waste.class_name}) on ground near person",
                    )
                if on_ground:
                    return (
                        "FLAGGED - Waste on ground",
                        f"Waste ({waste.class_name}) on ground, no dustbin",
                    )
                # Waste near person (any position)
                if near_person:
                    return (
                        "FLAGGED - Waste near person",
                        f"Waste ({waste.class_name}) near person, no dustbin nearby",
                    )

        # Case 3: No waste but throwing motion + dustbin
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
                # Multiple ground items or visual clutter detected near person = littering
                if len(nearby_ground) >= 2:
                    return (
                        "FLAGGED - Ground litter near person",
                        f"Litter on ground ({object_names}) near person, no dustbin",
                    )
                # Single ground item with visual clutter = likely litter
                if any(g.class_name == "ground_litter" for g in nearby_ground):
                    return (
                        "FLAGGED - Ground litter detected",
                        f"Visual ground clutter ({object_names}) near person, likely litter",
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

    # Case 6: Waste exists but no matching case above
    if has_waste:
        names = _format_waste_list(result.waste_objects)
        return (
            "FLAGGED - Waste detected",
            f"Waste ({names}) detected, needs review",
        )

    return (
        "INCONCLUSIVE - Needs more frames",
        f"Waste ({result.waste_count} items), ground ({result.ground_count} items)",
    )


def _is_throwing(person: PersonInfo) -> bool:
    """Detect throwing posture using arm extension analysis.

    Checks for arm extended to side or forward with hand lowered,
    indicating a throwing or dropping motion. Works with both
    pose-detected and estimated hand positions.

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
            # Arm extended to side + hand below waist = throwing/dropping
            if (
                abs(wx - body_center_x) > body_width * 0.5
                and wy > shoulder_y
                and wy > ey
                and wy > waist_y
            ):
                return True
            # Hand below waist + wrist below elbow = dropping motion
            if wy > waist_y and wy > ey:
                return True
    return False


def _is_hand_lowered(person: PersonInfo) -> bool:
    """Detect if hand is lowered below waist (dropping/throwing gesture).

    Args:
        person: PersonInfo with pose data.

    Returns:
        True if hand is in lowered position.
    """
    body_height = person.bbox.height
    waist_y = person.bbox.y1 + body_height * 0.55

    return any(
        wrist and wrist[1] > waist_y
        for wrist in [person.left_wrist, person.right_wrist]
    )


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


def _format_waste_list(waste: list[WasteInfo]) -> str:
    """Format waste objects into a readable string."""
    names: dict[str, int] = {}
    for w in waste:
        names[w.class_name] = names.get(w.class_name, 0) + 1
    parts = [f"{name}x{count}" if count > 1 else name for name, count in sorted(names.items())]
    return ", ".join(parts) if parts else "unknown"


def _format_object_list(objects: Sequence[GroundObject | WasteInfo | DustbinInfo]) -> str:
    """Format a list of objects into a readable string.

    Args:
        objects: List of objects with class_name attribute.

    Returns:
        Comma-separated string of object class names.
    """
    names: dict[str, int] = {}
    for obj in objects:
        name = obj.class_name
        names[name] = names.get(name, 0) + 1
    parts = [f"{name}x{count}" if count > 1 else name for name, count in sorted(names.items())]
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
        waste_names: dict[str, int] = {}
        for w in result.waste_objects:
            waste_names[w.class_name] = waste_names.get(w.class_name, 0) + 1
        waste_str = ", ".join(
            f"{name}x{count}" if count > 1 else name
            for name, count in sorted(waste_names.items())
        )
        parts.append(f"Waste: {waste_str}")
    if result.ground_count:
        ground_names: dict[str, int] = {}
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
