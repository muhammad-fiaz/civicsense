"""Data export service for incidents and analytics.

Supports exporting incident data to CSV and JSON formats
for external reporting and analysis.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from civicsense.core.config import get_config
from civicsense.core.exceptions import ExportError
from civicsense.core.logging import get_logger
from civicsense.database.repositories.crud import IncidentRepository
from civicsense.events.event_bus import Event, EventType, get_event_bus

logger = get_logger("app")


class ExportService:
    """Handles exporting incident data to various file formats.

    Provides CSV and JSON export capabilities with configurable
    output paths and filtering options.
    """

    def __init__(self) -> None:
        """Initialize the ExportService with repository and configuration."""
        self._repository = IncidentRepository()
        self._config = get_config()
        self._event_bus = get_event_bus()

    def export_csv(
        self,
        output_path: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> str:
        """Export incidents to CSV format.

        Args:
            output_path: Destination file path. Auto-generated if None.
            filters: Optional filters (status, camera_id, date range).

        Returns:
            The absolute path of the exported file.

        Raises:
            ExportError: If the export operation fails.
        """
        try:
            incidents = self._get_filtered_incidents(filters)

            if output_path is None:
                export_dir = Path(self._config.storage.exports_dir)
                export_dir.mkdir(parents=True, exist_ok=True)
                output_path = str(export_dir / f"incidents_export_{id(incidents)}.csv")

            fieldnames = [
                "id",
                "timestamp",
                "camera_id",
                "camera_name",
                "confidence",
                "person_track_id",
                "waste_type",
                "status",
                "review_notes",
                "created_at",
            ]

            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for inc in incidents:
                    writer.writerow(
                        {
                            "id": inc.id,
                            "timestamp": inc.timestamp.isoformat(),
                            "camera_id": inc.camera_id,
                            "camera_name": inc.camera_name,
                            "confidence": inc.confidence,
                            "person_track_id": inc.person_track_id,
                            "waste_type": inc.waste_type,
                            "status": inc.status,
                            "review_notes": inc.review_notes,
                            "created_at": inc.created_at.isoformat()
                            if inc.created_at
                            else "",
                        }
                    )

            self._event_bus.publish(
                Event(
                    event_type=EventType.EXPORT_COMPLETE,
                    data={
                        "format": "csv",
                        "path": output_path,
                        "count": len(incidents),
                    },
                    source="ExportService",
                )
            )
            logger.info(
                f"CSV exported: {output_path} ({len(incidents)} records)", module="app"
            )
            return str(Path(output_path).resolve())

        except Exception as e:
            raise ExportError(f"CSV export failed: {e}") from e

    def export_json(
        self,
        output_path: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> str:
        """Export incidents to JSON format.

        Args:
            output_path: Destination file path. Auto-generated if None.
            filters: Optional filters (status, camera_id, date range).

        Returns:
            The absolute path of the exported file.

        Raises:
            ExportError: If the export operation fails.
        """
        try:
            incidents = self._get_filtered_incidents(filters)

            if output_path is None:
                export_dir = Path(self._config.storage.exports_dir)
                export_dir.mkdir(parents=True, exist_ok=True)
                output_path = str(export_dir / f"incidents_export_{id(incidents)}.json")

            data = [
                {
                    "id": inc.id,
                    "timestamp": inc.timestamp.isoformat(),
                    "camera_id": inc.camera_id,
                    "camera_name": inc.camera_name,
                    "confidence": inc.confidence,
                    "person_track_id": inc.person_track_id,
                    "waste_type": inc.waste_type,
                    "snapshot_path": inc.snapshot_path,
                    "annotated_path": inc.annotated_path,
                    "clip_path": inc.clip_path,
                    "status": inc.status,
                    "review_notes": inc.review_notes,
                    "created_at": inc.created_at.isoformat() if inc.created_at else "",
                }
                for inc in incidents
            ]

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            self._event_bus.publish(
                Event(
                    event_type=EventType.EXPORT_COMPLETE,
                    data={
                        "format": "json",
                        "path": output_path,
                        "count": len(incidents),
                    },
                    source="ExportService",
                )
            )
            logger.info(
                f"JSON exported: {output_path} ({len(incidents)} records)",
                module="app",
            )
            return str(Path(output_path).resolve())

        except Exception as e:
            raise ExportError(f"JSON export failed: {e}") from e

    def _get_filtered_incidents(self, filters: dict[str, Any] | None) -> list[Any]:
        """Retrieve incidents with optional filters.

        Args:
            filters: Filter criteria dictionary.

        Returns:
            List of incident records matching the filters.
        """
        if filters is None:
            return self._repository.get_all(offset=0, limit=100000)

        if "status" in filters:
            return self._repository.get_by_status(
                filters["status"], offset=0, limit=100000
            )

        if "camera_id" in filters:
            return self._repository.get_by_camera(
                filters["camera_id"], offset=0, limit=100000
            )

        if "search" in filters:
            return self._repository.search(filters["search"], offset=0, limit=100000)

        return self._repository.get_all(offset=0, limit=100000)
