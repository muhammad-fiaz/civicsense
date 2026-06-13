"""Repository implementations for CivicSense database models.

Provides typed CRUD operations for each database entity using the
repository pattern with SQLAlchemy sessions.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Generic, TypeVar

from sqlalchemy import func

from civicsense.core.logging import get_logger
from civicsense.database.engine import get_session
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

logger = get_logger("database")

ModelType = TypeVar("ModelType", bound=Base)


class BaseRepository(Generic[ModelType]):
    """Generic base repository with common CRUD operations.

    Provides type-safe database access for any SQLAlchemy model.
    """

    def __init__(self, model: type[ModelType]) -> None:
        """Initialize the repository with a model class.

        Args:
            model: The SQLAlchemy model class this repository manages.
        """
        self.model = model

    def get_by_id(self, id: int) -> ModelType | None:
        """Retrieve a record by its primary key.

        Args:
            id: The primary key value.

        Returns:
            The matching record or None.
        """
        with get_session() as session:
            result = session.get(self.model, id)
            if result is not None:
                session.expunge(result)
            return result

    def get_all(self, offset: int = 0, limit: int = 100) -> list[ModelType]:
        """Retrieve all records with pagination.

        Args:
            offset: Number of records to skip.
            limit: Maximum number of records to return.

        Returns:
            List of matching records.
        """
        with get_session() as session:
            results = session.query(self.model).offset(offset).limit(limit).all()
            for r in results:
                session.expunge(r)
            return list(results)

    def create(self, data: dict[str, Any]) -> ModelType:
        """Create a new record.

        Args:
            data: Dictionary of column values.

        Returns:
            The created record with its assigned ID.
        """
        with get_session() as session:
            instance = self.model(**data)
            session.add(instance)
            session.flush()
            session.refresh(instance)
            session.expunge(instance)
            logger.info(
                f"Created {self.model.__name__} id={instance.id}",
                module="database",
            )
            return instance

    def update(self, id: int, data: dict[str, Any]) -> ModelType | None:
        """Update an existing record.

        Args:
            id: The primary key value.
            data: Dictionary of column values to update.

        Returns:
            The updated record or None if not found.
        """
        with get_session() as session:
            instance = session.get(self.model, id)
            if instance is None:
                return None
            for key, value in data.items():
                setattr(instance, key, value)
            session.flush()
            session.refresh(instance)
            session.expunge(instance)
            logger.info(
                f"Updated {self.model.__name__} id={id}",
                module="database",
            )
            return instance

    def delete(self, id: int) -> bool:
        """Delete a record by its primary key.

        Args:
            id: The primary key value.

        Returns:
            True if the record was deleted.
        """
        with get_session() as session:
            instance = session.get(self.model, id)
            if instance is None:
                return False
            session.delete(instance)
            logger.info(
                f"Deleted {self.model.__name__} id={id}",
                module="database",
            )
            return True

    def count(self) -> int:
        """Count total records.

        Returns:
            The total number of records.
        """
        with get_session() as session:
            return session.query(func.count()).select_from(self.model).scalar() or 0


class CameraRepository(BaseRepository[Camera]):
    """Repository for Camera model operations."""

    def __init__(self) -> None:
        """Initialize the Camera repository."""
        super().__init__(Camera)

    def get_active_cameras(self) -> list[Camera]:
        """Retrieve all active cameras.

        Returns:
            List of active Camera records.
        """
        with get_session() as session:
            results = session.query(Camera).filter(Camera.is_active.is_(True)).all()
            for r in results:
                session.expunge(r)
            return list(results)

    def set_active(self, id: int, active: bool) -> Camera | None:
        """Set the active status of a camera.

        Args:
            id: The camera primary key.
            active: Whether the camera should be active.

        Returns:
            The updated Camera or None.
        """
        return self.update(id, {"is_active": active})


class IncidentRepository(BaseRepository[Incident]):
    """Repository for Incident model operations."""

    def __init__(self) -> None:
        """Initialize the Incident repository."""
        super().__init__(Incident)

    def get_by_status(
        self, status: str, offset: int = 0, limit: int = 100
    ) -> list[Incident]:
        """Retrieve incidents filtered by review status.

        Args:
            status: The incident status value.
            offset: Number of records to skip.
            limit: Maximum records to return.

        Returns:
            List of matching Incident records.
        """
        with get_session() as session:
            results = (
                session.query(Incident)
                .filter(Incident.status == status)
                .order_by(Incident.timestamp.desc())
                .offset(offset)
                .limit(limit)
                .all()
            )
            for r in results:
                session.expunge(r)
            return list(results)

    def get_by_camera(
        self, camera_id: str, offset: int = 0, limit: int = 100
    ) -> list[Incident]:
        """Retrieve incidents for a specific camera.

        Args:
            camera_id: The camera identifier.
            offset: Number of records to skip.
            limit: Maximum records to return.

        Returns:
            List of matching Incident records.
        """
        with get_session() as session:
            results = (
                session.query(Incident)
                .filter(Incident.camera_id == camera_id)
                .order_by(Incident.timestamp.desc())
                .offset(offset)
                .limit(limit)
                .all()
            )
            for r in results:
                session.expunge(r)
            return list(results)

    def get_by_date_range(self, start: datetime, end: datetime) -> list[Incident]:
        """Retrieve incidents within a date range.

        Args:
            start: Start datetime (inclusive).
            end: End datetime (inclusive).

        Returns:
            List of Incident records in the range.
        """
        with get_session() as session:
            results = (
                session.query(Incident)
                .filter(Incident.timestamp >= start, Incident.timestamp <= end)
                .order_by(Incident.timestamp.desc())
                .all()
            )
            for r in results:
                session.expunge(r)
            return list(results)

    def get_today_count(self) -> int:
        """Count incidents created today.

        Returns:
            Number of today's incidents.
        """
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        with get_session() as session:
            return (
                session.query(func.count())
                .select_from(Incident)
                .filter(Incident.timestamp >= today)
                .scalar()
                or 0
            )

    def search(self, query: str, offset: int = 0, limit: int = 100) -> list[Incident]:
        """Search incidents by waste type or camera name.

        Args:
            query: Search string.
            offset: Number of records to skip.
            limit: Maximum records to return.

        Returns:
            List of matching Incident records.
        """
        with get_session() as session:
            results = (
                session.query(Incident)
                .filter(
                    Incident.waste_type.ilike(f"%{query}%")
                    | Incident.camera_name.ilike(f"%{query}%")
                )
                .order_by(Incident.timestamp.desc())
                .offset(offset)
                .limit(limit)
                .all()
            )
            for r in results:
                session.expunge(r)
            return list(results)


class EvidenceRepository(BaseRepository[Evidence]):
    """Repository for Evidence model operations."""

    def __init__(self) -> None:
        """Initialize the Evidence repository."""
        super().__init__(Evidence)

    def get_by_incident(self, incident_id: int) -> list[Evidence]:
        """Retrieve all evidence for an incident.

        Args:
            incident_id: The parent incident ID.

        Returns:
            List of Evidence records.
        """
        with get_session() as session:
            results = (
                session.query(Evidence)
                .filter(Evidence.incident_id == incident_id)
                .all()
            )
            for r in results:
                session.expunge(r)
            return list(results)


class SettingsRepository(BaseRepository[ApplicationSettings]):
    """Repository for ApplicationSettings model operations."""

    def __init__(self) -> None:
        """Initialize the Settings repository."""
        super().__init__(ApplicationSettings)

    def get_value(self, key: str) -> str | None:
        """Retrieve a setting value by key.

        Args:
            key: The setting key.

        Returns:
            The setting value or None.
        """
        with get_session() as session:
            result = (
                session.query(ApplicationSettings)
                .filter(ApplicationSettings.key == key)
                .first()
            )
            if result is not None:
                return result.value
            return None

    def set_value(self, key: str, value: str, description: str = "") -> None:
        """Create or update a setting value.

        Args:
            key: The setting key.
            value: The setting value.
            description: Optional description of the setting.
        """
        with get_session() as session:
            existing = (
                session.query(ApplicationSettings)
                .filter(ApplicationSettings.key == key)
                .first()
            )
            if existing:
                existing.value = value
                if description:
                    existing.description = description
            else:
                session.add(
                    ApplicationSettings(key=key, value=value, description=description)
                )


class ModelConfigRepository(BaseRepository[ModelConfiguration]):
    """Repository for ModelConfiguration model operations."""

    def __init__(self) -> None:
        """Initialize the ModelConfiguration repository."""
        super().__init__(ModelConfiguration)

    def get_active(self) -> ModelConfiguration | None:
        """Retrieve the currently active model configuration.

        Returns:
            The active ModelConfiguration or None.
        """
        with get_session() as session:
            result = (
                session.query(ModelConfiguration)
                .filter(ModelConfiguration.is_active.is_(True))
                .first()
            )
            if result is not None:
                session.expunge(result)
            return result

    def set_active(self, id: int) -> None:
        """Set a model configuration as active, deactivating all others.

        Args:
            id: The configuration to activate.
        """
        with get_session() as session:
            session.query(ModelConfiguration).update({"is_active": False})
            instance = session.get(ModelConfiguration, id)
            if instance:
                instance.is_active = True


class AuditLogRepository(BaseRepository[AuditLog]):
    """Repository for AuditLog model operations."""

    def __init__(self) -> None:
        """Initialize the AuditLog repository."""
        super().__init__(AuditLog)

    def log_action(
        self,
        action: str,
        entity_type: str,
        entity_id: int | None = None,
        details: str = "{}",
        user: str = "system",
    ) -> AuditLog:
        """Log an audit event.

        Args:
            action: The action performed.
            entity_type: The type of entity affected.
            entity_id: The entity's primary key.
            details: JSON-serialized action details.
            user: The user performing the action.

        Returns:
            The created AuditLog record.
        """
        return self.create(
            {
                "action": action,
                "entity_type": entity_type,
                "entity_id": entity_id,
                "details": details,
                "user": user,
            }
        )


class AnalyticsRepository(BaseRepository[AnalyticsSnapshot]):
    """Repository for AnalyticsSnapshot model operations."""

    def __init__(self) -> None:
        """Initialize the AnalyticsSnapshot repository."""
        super().__init__(AnalyticsSnapshot)

    def get_latest(self, period: str) -> AnalyticsSnapshot | None:
        """Retrieve the most recent snapshot for a period.

        Args:
            period: The analytics period (daily, weekly, monthly).

        Returns:
            The latest AnalyticsSnapshot or None.
        """
        with get_session() as session:
            result = (
                session.query(AnalyticsSnapshot)
                .filter(AnalyticsSnapshot.period == period)
                .order_by(AnalyticsSnapshot.snapshot_date.desc())
                .first()
            )
            if result is not None:
                session.expunge(result)
            return result
