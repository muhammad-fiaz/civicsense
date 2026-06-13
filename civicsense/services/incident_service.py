"""Incident management service.

Handles creation, review, and management of littering incidents
by coordinating between the database repositories and evidence storage.
"""

from __future__ import annotations

import json
from typing import Any

from civicsense.core.logging import get_logger
from civicsense.database.repositories.crud import IncidentRepository
from civicsense.dto.detection import IncidentDTO, IncidentStatus
from civicsense.events.event_bus import Event, EventType, get_event_bus

logger = get_logger("app")


class IncidentService:
    """Manages the full lifecycle of littering incidents.

    Provides operations for creating, querying, reviewing,
    and updating incidents with audit logging.
    """

    def __init__(self) -> None:
        """Initialize the IncidentService with repository and event bus."""
        self._repository = IncidentRepository()
        self._event_bus = get_event_bus()

    def create_incident(self, dto: IncidentDTO) -> IncidentDTO:
        """Create a new littering incident.

        Args:
            dto: The incident data transfer object.

        Returns:
            The created incident with assigned ID.
        """
        data = {
            "timestamp": dto.timestamp,
            "camera_id": dto.camera_id,
            "camera_name": dto.camera_name,
            "confidence": dto.confidence,
            "person_track_id": dto.person_track_id,
            "waste_type": dto.waste_type,
            "snapshot_path": dto.snapshot_path,
            "annotated_path": dto.annotated_path,
            "clip_path": dto.clip_path,
            "detection_metadata": json.dumps(dto.detection_metadata),
            "status": dto.status.value,
            "review_notes": dto.review_notes,
            "frame_width": dto.frame_width,
            "frame_height": dto.frame_height,
        }

        record = self._repository.create(data)
        dto.id = record.id

        self._event_bus.publish(
            Event(
                event_type=EventType.INCIDENT_CREATED,
                data=dto.to_dict(),
                source="IncidentService",
            )
        )
        logger.info(f"Incident created: id={record.id}", module="app")
        return dto

    def get_incident(self, incident_id: int) -> IncidentDTO | None:
        """Retrieve an incident by ID.

        Args:
            incident_id: The incident primary key.

        Returns:
            The matching IncidentDTO or None.
        """
        record = self._repository.get_by_id(incident_id)
        if record is None:
            return None
        return self._record_to_dto(record)

    def get_incidents(
        self,
        offset: int = 0,
        limit: int = 100,
    ) -> list[IncidentDTO]:
        """Retrieve incidents with pagination.

        Args:
            offset: Number of records to skip.
            limit: Maximum records to return.

        Returns:
            List of IncidentDTO objects.
        """
        records = self._repository.get_all(offset=offset, limit=limit)
        return [self._record_to_dto(r) for r in records]

    def search_incidents(
        self,
        query: str,
        offset: int = 0,
        limit: int = 100,
    ) -> list[IncidentDTO]:
        """Search incidents by waste type or camera name.

        Args:
            query: Search string.
            offset: Number of records to skip.
            limit: Maximum records to return.

        Returns:
            List of matching IncidentDTO objects.
        """
        records = self._repository.search(query, offset=offset, limit=limit)
        return [self._record_to_dto(r) for r in records]

    def get_by_status(
        self,
        status: IncidentStatus,
        offset: int = 0,
        limit: int = 100,
    ) -> list[IncidentDTO]:
        """Retrieve incidents filtered by review status.

        Args:
            status: The review status filter.
            offset: Number of records to skip.
            limit: Maximum records to return.

        Returns:
            List of matching IncidentDTO objects.
        """
        records = self._repository.get_by_status(
            status.value, offset=offset, limit=limit
        )
        return [self._record_to_dto(r) for r in records]

    def review_incident(
        self,
        incident_id: int,
        status: IncidentStatus,
        notes: str = "",
    ) -> IncidentDTO | None:
        """Update the review status of an incident.

        Args:
            incident_id: The incident to review.
            status: The new review status.
            notes: Optional review notes.

        Returns:
            The updated IncidentDTO or None.
        """
        data = {"status": status.value, "review_notes": notes}
        record = self._repository.update(incident_id, data)
        if record is None:
            return None

        dto = self._record_to_dto(record)
        self._event_bus.publish(
            Event(
                event_type=EventType.INCIDENT_UPDATED,
                data=dto.to_dict(),
                source="IncidentService",
            )
        )
        logger.info(
            f"Incident {incident_id} reviewed: status={status.value}",
            module="app",
        )
        return dto

    def approve_incident(self, incident_id: int, notes: str = "") -> IncidentDTO | None:
        """Approve an incident.

        Args:
            incident_id: The incident to approve.
            notes: Optional approval notes.

        Returns:
            The updated IncidentDTO or None.
        """
        return self.review_incident(incident_id, IncidentStatus.APPROVED, notes)

    def reject_incident(self, incident_id: int, notes: str = "") -> IncidentDTO | None:
        """Reject an incident.

        Args:
            incident_id: The incident to reject.
            notes: Optional rejection notes.

        Returns:
            The updated IncidentDTO or None.
        """
        return self.review_incident(incident_id, IncidentStatus.REJECTED, notes)

    def get_total_count(self) -> int:
        """Return the total number of incidents.

        Returns:
            Total incident count.
        """
        return self._repository.count()

    def get_today_count(self) -> int:
        """Return the number of incidents created today.

        Returns:
            Today's incident count.
        """
        return self._repository.get_today_count()

    @staticmethod
    def _record_to_dto(record: Any) -> IncidentDTO:
        """Convert a database record to an IncidentDTO.

        Args:
            record: The SQLAlchemy Incident record.

        Returns:
            The corresponding IncidentDTO.
        """
        return IncidentDTO(
            id=record.id,
            timestamp=record.timestamp,
            camera_id=record.camera_id,
            camera_name=record.camera_name,
            confidence=record.confidence,
            person_track_id=record.person_track_id,
            waste_type=record.waste_type,
            snapshot_path=record.snapshot_path,
            annotated_path=record.annotated_path,
            clip_path=record.clip_path,
            detection_metadata=json.loads(record.detection_metadata or "{}"),
            status=IncidentStatus(record.status),
            review_notes=record.review_notes,
            frame_width=record.frame_width,
            frame_height=record.frame_height,
        )
