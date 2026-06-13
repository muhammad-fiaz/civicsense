"""Application-wide constants for CivicSense.

All detection thresholds and settings are loaded from config/detection.toml.
Python enums and class mappings kept here for type safety.
"""

from __future__ import annotations

from enum import IntEnum, StrEnum
from pathlib import Path

from civicsense.config import (
    get_dustbin_candidates,
    get_nested_config,
    get_waste_classes,
)


class DetectionClass(StrEnum):
    """Object detection class labels."""

    PERSON = "person"
    BOTTLE = "bottle"
    CUP = "cup"
    PAPER = "paper"
    PLASTIC_BAG = "plastic bag"
    FOOD_WRAPPER = "food wrapper"
    TRASH = "trash"
    GARBAGE = "garbage"
    DUSTBIN = "dustbin"


class PoseKeypointIndex(IntEnum):
    """COCO-style pose keypoint indices."""

    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16


PERSON_CLASSES: list[str] = [DetectionClass.PERSON]

DETECTION_CLASSES: list[str] = [cls.value for cls in DetectionClass]

WEIGHTS_DIR: Path = Path(__file__).parent.parent.parent / "weights"

MODEL_SIZES: list[str] = ["n", "s", "m", "l", "x"]


def resolve_model_path(filename: str) -> str:
    """Resolve a model filename to the weights/ directory.

    If the file exists in weights/, return that path.
    Otherwise return the filename as-is (ultralytics will download).

    Args:
        filename: Model filename like 'yolo26n.pt'.

    Returns:
        Resolved path string.
    """
    weights_path = WEIGHTS_DIR / filename
    if weights_path.exists():
        return str(weights_path)
    return filename


DETECTION_MODELS: dict[str, str] = {size: f"yolo26{size}.pt" for size in MODEL_SIZES}

POSE_MODELS: dict[str, str] = {size: f"yolo26{size}-pose.pt" for size in MODEL_SIZES}

# Thresholds loaded from TOML config (single source of truth)
DEFAULT_CONFIDENCE_THRESHOLD: float = float(
    get_nested_config("detection", "confidence", default=0.15)
)
DEFAULT_IOU_THRESHOLD: float = float(
    get_nested_config("detection", "iou", default=0.45)
)
DEFAULT_IMAGE_SIZE: int = int(get_nested_config("detection", "image_size", default=640))

HAND_PROXIMITY_THRESHOLD: float = float(
    get_nested_config("detection", "thresholds", "hand_proximity", default=80.0)
)
WASTE_FALL_SPEED_THRESHOLD: float = float(
    get_nested_config("detection", "thresholds", "waste_fall_speed", default=2.0)
)
WASTE_GROUND_Y_THRESHOLD: float = float(
    get_nested_config("detection", "thresholds", "waste_ground_y", default=0.75)
)
DUSTBIN_PROXIMITY_THRESHOLD: float = float(
    get_nested_config("detection", "thresholds", "dustbin_proximity", default=100.0)
)
PERSON_WASTE_DISTANCE_THRESHOLD: float = float(
    get_nested_config("detection", "thresholds", "person_waste_distance", default=200.0)
)
GROUND_OBJECT_DISTANCE_THRESHOLD: float = float(
    get_nested_config(
        "detection", "thresholds", "ground_object_distance", default=250.0
    )
)
ARM_RAISED_RATIO_THRESHOLD: float = float(
    get_nested_config("detection", "thresholds", "arm_raised_ratio", default=0.25)
)

INCIDENT_COOLDOWN_SECONDS: float = 5.0

MAX_TRACKING_AGE: int = int(
    get_nested_config("detection", "tracking", "max_age", default=30)
)
MAX_TRACKING_DISTANCE: float = float(
    get_nested_config("detection", "tracking", "max_distance", default=150.0)
)

# Waste classes loaded from TOML config
WASTE_CLASSES: list[str] = get_waste_classes()

# Dustbin candidates loaded from TOML config
DUSTBIN_CANDIDATES: list[str] = get_dustbin_candidates()

# Clip settings from TOML
CLIP_DURATION_SECONDS: int = int(
    get_nested_config("detection", "clip", "duration_seconds", default=10)
)
CLIP_PRE_EVENT_SECONDS: int = int(
    get_nested_config("detection", "clip", "pre_event_seconds", default=3)
)
CLIP_POST_EVENT_SECONDS: int = int(
    get_nested_config("detection", "clip", "post_event_seconds", default=7)
)
