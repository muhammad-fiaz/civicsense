"""Event-driven architecture components for CivicSense.

Provides a lightweight event bus for decoupled communication between
the AI pipeline, services, and GUI layers.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


class EventType(StrEnum):
    """Registered event types."""

    FRAME_PROCESSED = "frame.processed"
    DETECTION_COMPLETE = "detection.complete"
    INCIDENT_CREATED = "incident.created"
    INCIDENT_UPDATED = "incident.updated"
    CAMERA_CONNECTED = "camera.connected"
    CAMERA_DISCONNECTED = "camera.disconnected"
    MODEL_LOADED = "model.loaded"
    MODEL_UNLOADED = "model.unloaded"
    TRACKING_UPDATED = "tracking.updated"
    POSE_ESTIMATED = "pose.estimated"
    LITTERING_DETECTED = "littering.detected"
    ERROR_OCCURRED = "error.occurred"
    STATUS_CHANGED = "status.changed"
    SETTINGS_UPDATED = "settings.updated"
    EXPORT_COMPLETE = "export.complete"
    ANALYTICS_READY = "analytics.ready"


@dataclass
class Event:
    """An event payload."""

    event_type: EventType
    data: dict[str, Any] = field(default_factory=dict)
    source: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


EventHandler = Callable[[Event], None]


class EventBus:
    """Lightweight publish-subscribe event bus.

    Enables decoupled communication between application subsystems.
    """

    def __init__(self) -> None:
        """Initialize the event bus with empty subscriber lists."""
        self._subscribers: dict[EventType, list[EventHandler]] = {}

    def subscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """Register a handler for a specific event type.

        Args:
            event_type: The event type to subscribe to.
            handler: The callback function to invoke.
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)

    def unsubscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """Remove a handler from an event type.

        Args:
            event_type: The event type to unsubscribe from.
            handler: The handler to remove.

        Raises:
            ValueError: If the handler is not subscribed.
        """
        handlers = self._subscribers.get(event_type, [])
        if handler not in handlers:
            raise ValueError(f"Handler not subscribed to {event_type}")
        handlers.remove(handler)

    def publish(self, event: Event) -> None:
        """Publish an event to all subscribed handlers.

        Args:
            event: The event to publish.
        """
        handlers = self._subscribers.get(event.event_type, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                import sys

                print(
                    f"Event handler error ({event.event_type.value}): {e}",
                    file=sys.stderr,
                )

    def clear(self) -> None:
        """Remove all subscriptions."""
        self._subscribers.clear()


_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    """Return the global event bus singleton.

    Returns:
        The application event bus instance.
    """
    global _bus
    if _bus is None:
        _bus = EventBus()
    return _bus
