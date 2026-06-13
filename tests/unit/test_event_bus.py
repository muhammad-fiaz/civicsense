"""Tests for event bus."""

from __future__ import annotations

from civicsense.events.event_bus import Event, EventBus, EventType, get_event_bus


class TestEventBus:
    """Tests for the EventBus publish-subscribe system."""

    def test_subscribe_and_publish(self) -> None:
        """Verify handler receives published events."""
        bus = EventBus()
        received: list[Event] = []

        def handler(event: Event) -> None:
            received.append(event)

        bus.subscribe(EventType.FRAME_PROCESSED, handler)
        event = Event(event_type=EventType.FRAME_PROCESSED, data={"frame": 1})
        bus.publish(event)

        assert len(received) == 1
        assert received[0].data == {"frame": 1}

    def test_unsubscribe(self) -> None:
        """Verify unsubscribed handler does not receive events."""
        bus = EventBus()
        received: list[Event] = []

        def handler(event: Event) -> None:
            received.append(event)

        bus.subscribe(EventType.FRAME_PROCESSED, handler)
        bus.unsubscribe(EventType.FRAME_PROCESSED, handler)
        bus.publish(Event(event_type=EventType.FRAME_PROCESSED))

        assert len(received) == 0

    def test_unsubscribe_nonexistent_raises(self) -> None:
        """Verify unsubscribing non-existent handler raises ValueError."""
        bus = EventBus()

        def handler(event: Event) -> None:
            pass

        import pytest

        with pytest.raises(ValueError):
            bus.unsubscribe(EventType.FRAME_PROCESSED, handler)

    def test_multiple_subscribers(self) -> None:
        """Verify multiple handlers receive the same event."""
        bus = EventBus()
        count = {"a": 0, "b": 0}

        def handler_a(event: Event) -> None:
            count["a"] += 1

        def handler_b(event: Event) -> None:
            count["b"] += 1

        bus.subscribe(EventType.INCIDENT_CREATED, handler_a)
        bus.subscribe(EventType.INCIDENT_CREATED, handler_b)
        bus.publish(Event(event_type=EventType.INCIDENT_CREATED))

        assert count["a"] == 1
        assert count["b"] == 1

    def test_clear(self) -> None:
        """Verify clear removes all subscriptions."""
        bus = EventBus()
        received: list[Event] = []

        def handler(event: Event) -> None:
            received.append(event)

        bus.subscribe(EventType.STATUS_CHANGED, handler)
        bus.clear()
        bus.publish(Event(event_type=EventType.STATUS_CHANGED))

        assert len(received) == 0

    def test_singleton(self) -> None:
        """Verify get_event_bus returns the same instance."""
        bus1 = get_event_bus()
        bus2 = get_event_bus()
        assert bus1 is bus2
