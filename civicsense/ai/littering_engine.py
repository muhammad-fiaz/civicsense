"""Littering event classification engine.

Implements the core logic for determining when a littering event occurs
by analyzing spatial relationships between persons, waste objects,
and dustbins across video frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from civicsense.core.constants import (
    DUSTBIN_PROXIMITY_THRESHOLD,
    HAND_PROXIMITY_THRESHOLD,
    WASTE_FALL_SPEED_THRESHOLD,
    WASTE_GROUND_Y_THRESHOLD,
)
from civicsense.core.logging import get_logger
from civicsense.dto.detection import TrackedObject

logger = get_logger("ai")


@dataclass
class LitteringState:
    """Tracks the state of a potential littering event over time."""

    waste_track_id: int
    person_track_id: int
    waste_type: str
    near_hand: bool = False
    left_hand: bool = False
    falling: bool = False
    reached_ground: bool = False
    entered_dustbin: bool = False
    frame_count: int = 0
    first_seen: datetime = field(default_factory=datetime.now)
    positions: list[tuple[float, float]] = field(default_factory=list)

    @property
    def is_littering(self) -> bool:
        """Return True if all littering conditions are met."""
        return (
            self.near_hand
            and self.left_hand
            and self.falling
            and self.reached_ground
            and not self.entered_dustbin
        )


class LitteringClassifier:
    """Classifies littering events by analyzing object trajectories and spatial relationships.

    Maintains state across frames to detect the full sequence:
    waste near hand -> waste leaves hand -> waste falls -> waste reaches ground.
    """

    def __init__(self) -> None:
        """Initialize the classifier with empty state tracking."""
        self._states: dict[int, LitteringState] = {}
        self._frame_idx: int = 0

    def classify(
        self,
        persons: list[TrackedObject],
        waste_objects: list[TrackedObject],
        dustbins: list[TrackedObject],
        frame_height: int,
    ) -> list[LitteringState]:
        """Analyze a frame and update littering event states.

        Args:
            persons: Detected and tracked persons in the frame.
            waste_objects: Detected and tracked waste objects.
            dustbins: Detected dustbin locations.
            frame_height: Height of the frame in pixels.

        Returns:
            List of LitteringState objects that have reached the littering condition.
        """
        self._frame_idx += 1
        triggered: list[LitteringState] = []

        ground_y = frame_height * WASTE_GROUND_Y_THRESHOLD

        for waste in waste_objects:
            wid = waste.track_id
            waste_center = waste.bbox.center

            if wid not in self._states:
                best_person = self._find_nearest_person(waste, persons)
                if best_person is not None:
                    self._states[wid] = LitteringState(
                        waste_track_id=wid,
                        person_track_id=best_person.track_id,
                        waste_type=waste.class_name,
                    )

            state = self._states.get(wid)
            if state is None:
                continue

            state.frame_count += 1
            state.positions.append(waste_center)

            state.near_hand = self._check_near_hand(waste, persons)

            if state.near_hand and len(state.positions) >= 2:
                state.left_hand = self._check_left_hand(state, waste, persons)

            if state.left_hand and len(state.positions) >= 3:
                state.falling = self._check_falling(state, waste_center)

            if state.falling:
                state.reached_ground = waste_center[1] >= ground_y

            for dustbin in dustbins:
                if self._near_dustbin(waste, dustbin):
                    state.entered_dustbin = True
                    break

            if state.is_littering:
                triggered.append(state)
                logger.info(
                    f"Littering detected: waste_track={wid}, "
                    f"person_track={state.person_track_id}, "
                    f"type={state.waste_type}",
                    module="ai",
                )

        self._cleanup_states(persons, waste_objects)
        return triggered

    def reset(self) -> None:
        """Clear all tracking state."""
        self._states.clear()
        self._frame_idx = 0

    def _find_nearest_person(
        self,
        waste: TrackedObject,
        persons: list[TrackedObject],
    ) -> TrackedObject | None:
        """Find the person nearest to a waste object.

        Args:
            waste: The waste object.
            persons: List of detected persons.

        Returns:
            The nearest person, or None if no persons detected.
        """
        if not persons:
            return None

        waste_center = waste.bbox.center
        min_dist = float("inf")
        nearest: TrackedObject | None = None

        for person in persons:
            person_center = person.bbox.center
            dist = (
                (waste_center[0] - person_center[0]) ** 2
                + (waste_center[1] - person_center[1]) ** 2
            ) ** 0.5
            if dist < min_dist:
                min_dist = dist
                nearest = person

        return nearest

    def _check_near_hand(
        self,
        waste: TrackedObject,
        persons: list[TrackedObject],
    ) -> bool:
        """Check if waste is within proximity of any person's hand.

        Args:
            waste: The waste object.
            persons: List of detected persons.

        Returns:
            True if waste is near a person's hand.
        """
        waste_center = waste.bbox.center

        for person in persons:
            hand_pos = person.hand_center
            if hand_pos is None:
                continue
            dist = (
                (waste_center[0] - hand_pos[0]) ** 2
                + (waste_center[1] - hand_pos[1]) ** 2
            ) ** 0.5
            if dist <= HAND_PROXIMITY_THRESHOLD:
                return True

        return False

    def _check_left_hand(
        self,
        state: LitteringState,
        waste: TrackedObject,
        persons: list[TrackedObject],
    ) -> bool:
        """Check if waste has moved away from the person's hand.

        Args:
            state: The current littering state.
            waste: The waste object.
            persons: List of detected persons.

        Returns:
            True if waste has left the hand.
        """
        if len(state.positions) < 2:
            return False

        prev_pos = state.positions[-2]
        curr_pos = state.positions[-1]
        movement = (
            (curr_pos[0] - prev_pos[0]) ** 2 + (curr_pos[1] - prev_pos[1]) ** 2
        ) ** 0.5

        for person in persons:
            if person.track_id != state.person_track_id:
                continue
            hand_pos = person.hand_center
            if hand_pos is None:
                continue
            curr_dist = (
                (curr_pos[0] - hand_pos[0]) ** 2 + (curr_pos[1] - hand_pos[1]) ** 2
            ) ** 0.5
            prev_dist = (
                (prev_pos[0] - hand_pos[0]) ** 2 + (prev_pos[1] - hand_pos[1]) ** 2
            ) ** 0.5
            if curr_dist > prev_dist and movement > 5.0:
                return True

        return False

    def _check_falling(
        self,
        state: LitteringState,
        current_pos: tuple[float, float],
    ) -> bool:
        """Check if waste is falling based on vertical velocity.

        Args:
            state: The current littering state.
            current_pos: Current position of the waste.

        Returns:
            True if the waste appears to be falling.
        """
        if len(state.positions) < 3:
            return False

        recent = state.positions[-3:]
        vertical_velocities = [
            recent[i + 1][1] - recent[i][1] for i in range(len(recent) - 1)
        ]
        avg_vertical_velocity = sum(vertical_velocities) / len(vertical_velocities)
        return avg_vertical_velocity > WASTE_FALL_SPEED_THRESHOLD

    def _near_dustbin(self, waste: TrackedObject, dustbin: TrackedObject) -> bool:
        """Check if waste is near a dustbin.

        Args:
            waste: The waste object.
            dustbin: The dustbin object.

        Returns:
            True if waste is within dustbin proximity threshold.
        """
        waste_center = waste.bbox.center
        dustbin_center = dustbin.bbox.center
        dist = (
            (waste_center[0] - dustbin_center[0]) ** 2
            + (waste_center[1] - dustbin_center[1]) ** 2
        ) ** 0.5
        return dist <= DUSTBIN_PROXIMITY_THRESHOLD

    def _cleanup_states(
        self,
        persons: list[TrackedObject],
        waste_objects: list[TrackedObject],
    ) -> None:
        """Remove states for waste objects no longer detected.

        Args:
            persons: Currently detected persons.
            waste_objects: Currently detected waste objects.
        """
        active_waste_ids = {w.track_id for w in waste_objects}
        to_remove = [wid for wid in self._states if wid not in active_waste_ids]
        for wid in to_remove:
            del self._states[wid]
