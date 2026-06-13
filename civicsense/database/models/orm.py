"""SQLAlchemy ORM models for CivicSense.

Implements declarative models for cameras, incidents, evidence,
application settings, model configuration, audit logs, and analytics snapshots.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""

    pass


class Camera(Base):
    """Camera configuration and status model."""

    __tablename__ = "cameras"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    source: Mapped[str] = mapped_column(String(512), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    fps: Mapped[int] = mapped_column(Integer, default=30)
    width: Mapped[int] = mapped_column(Integer, default=1920)
    height: Mapped[int] = mapped_column(Integer, default=1080)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    def __repr__(self) -> str:
        """Return string representation of the Camera."""
        return f"<Camera(id={self.id}, name={self.name!r}, active={self.is_active})>"


class Incident(Base):
    """Littering incident record model."""

    __tablename__ = "incidents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    camera_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    camera_name: Mapped[str] = mapped_column(String(255), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    person_track_id: Mapped[int] = mapped_column(Integer, nullable=False)
    waste_type: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    snapshot_path: Mapped[str] = mapped_column(String(1024), default="")
    annotated_path: Mapped[str] = mapped_column(String(1024), default="")
    clip_path: Mapped[str] = mapped_column(String(1024), default="")
    detection_metadata: Mapped[str] = mapped_column(Text, default="{}")
    status: Mapped[str] = mapped_column(String(50), default="pending", index=True)
    review_notes: Mapped[str] = mapped_column(Text, default="")
    frame_width: Mapped[int] = mapped_column(Integer, default=0)
    frame_height: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    def __repr__(self) -> str:
        """Return string representation of the Incident."""
        return f"<Incident(id={self.id}, waste={self.waste_type!r}, status={self.status!r})>"


class Evidence(Base):
    """Evidence file metadata model."""

    __tablename__ = "evidence"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    incident_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    file_type: Mapped[str] = mapped_column(String(50), nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, default=0)
    mime_type: Mapped[str] = mapped_column(String(100), default="")
    description: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    def __repr__(self) -> str:
        """Return string representation of the Evidence."""
        return f"<Evidence(id={self.id}, incident={self.incident_id}, type={self.file_type!r})>"


class ApplicationSettings(Base):
    """Application settings key-value store model."""

    __tablename__ = "application_settings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    key: Mapped[str] = mapped_column(
        String(255), nullable=False, unique=True, index=True
    )
    value: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    category: Mapped[str] = mapped_column(String(100), default="general")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    def __repr__(self) -> str:
        """Return string representation of the ApplicationSettings."""
        return f"<ApplicationSettings(key={self.key!r})>"


class ModelConfiguration(Base):
    """AI model configuration model."""

    __tablename__ = "model_configurations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    model_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    model_type: Mapped[str] = mapped_column(String(50), nullable=False)
    device: Mapped[str] = mapped_column(String(50), default="auto")
    confidence_threshold: Mapped[float] = mapped_column(Float, default=0.5)
    iou_threshold: Mapped[float] = mapped_column(Float, default=0.45)
    image_size: Mapped[int] = mapped_column(Integer, default=640)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    def __repr__(self) -> str:
        """Return string representation of the ModelConfiguration."""
        return f"<ModelConfiguration(name={self.name!r}, type={self.model_type!r})>"


class AuditLog(Base):
    """Audit trail log model."""

    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    action: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    entity_type: Mapped[str] = mapped_column(String(100), nullable=False)
    entity_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    details: Mapped[str] = mapped_column(Text, default="{}")
    user: Mapped[str] = mapped_column(String(255), default="system")
    timestamp: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    def __repr__(self) -> str:
        """Return string representation of the AuditLog."""
        return f"<AuditLog(action={self.action!r}, entity={self.entity_type!r})>"


class AnalyticsSnapshot(Base):
    """Periodic analytics snapshot model."""

    __tablename__ = "analytics_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    snapshot_date: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, index=True
    )
    period: Mapped[str] = mapped_column(String(50), nullable=False)
    total_incidents: Mapped[int] = mapped_column(Integer, default=0)
    approved_incidents: Mapped[int] = mapped_column(Integer, default=0)
    rejected_incidents: Mapped[int] = mapped_column(Integer, default=0)
    pending_incidents: Mapped[int] = mapped_column(Integer, default=0)
    unique_cameras: Mapped[int] = mapped_column(Integer, default=0)
    waste_type_breakdown: Mapped[str] = mapped_column(Text, default="{}")
    hourly_distribution: Mapped[str] = mapped_column(Text, default="{}")
    average_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    def __repr__(self) -> str:
        """Return string representation of the AnalyticsSnapshot."""
        return f"<AnalyticsSnapshot(date={self.snapshot_date}, incidents={self.total_incidents})>"
