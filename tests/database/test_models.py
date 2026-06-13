"""Tests for database ORM models."""

from __future__ import annotations

import pytest
from civicsense.database.models.orm import (
    AnalyticsSnapshot,
    ApplicationSettings,
    AuditLog,
    Base,
    Camera,
    Evidence,
    Incident,
    ModelConfiguration,
)


class TestCameraModel:
    """Tests for Camera ORM model."""

    def test_repr(self) -> None:
        """Verify Camera string representation."""
        cam = Camera(name="TestCam", source="0")
        assert "TestCam" in repr(cam)

    def test_default_values(self) -> None:
        """Verify Camera default values."""
        cam = Camera(name="Cam", source="0", is_active=False, fps=30)
        assert cam.is_active is False
        assert cam.fps == 30


class TestIncidentModel:
    """Tests for Incident ORM model."""

    def test_repr(self) -> None:
        """Verify Incident string representation."""
        inc = Incident(
            timestamp=__import__("datetime").datetime.now(),
            camera_id="cam1",
            camera_name="Camera 1",
            confidence=0.95,
            person_track_id=1,
            waste_type="bottle",
            status="pending",
        )
        assert "bottle" in repr(inc)

    def test_default_status(self) -> None:
        """Verify default incident status is pending."""
        inc = Incident(
            timestamp=__import__("datetime").datetime.now(),
            camera_id="cam1",
            camera_name="Camera 1",
            confidence=0.9,
            person_track_id=1,
            waste_type="bottle",
            status="pending",
        )
        assert inc.status == "pending"


class TestBaseMetadata:
    """Tests for SQLAlchemy Base metadata."""

    def test_all_tables_exist(self) -> None:
        """Verify all expected tables are registered."""
        table_names = Base.metadata.tables.keys()
        expected = [
            "cameras",
            "incidents",
            "evidence",
            "application_settings",
            "model_configurations",
            "audit_logs",
            "analytics_snapshots",
        ]
        for name in expected:
            assert name in table_names
