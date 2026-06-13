"""Evidence storage and management service.

Handles saving, retrieving, and managing evidence files (snapshots,
annotated images, video clips) associated with incidents.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from civicsense.core.config import get_config
from civicsense.core.exceptions import StorageError
from civicsense.core.logging import get_logger
from civicsense.database.repositories.crud import EvidenceRepository

logger = get_logger("app")


class EvidenceService:
    """Manages evidence file storage and retrieval.

    Handles snapshot capture, annotated frame saving, video clip storage,
    and evidence metadata tracking in the database.
    """

    def __init__(self) -> None:
        """Initialize the EvidenceService with configuration and repository."""
        self._config = get_config()
        self._repository = EvidenceRepository()
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Create evidence storage directories if they do not exist."""
        dirs = [
            self._config.storage.snapshots_dir,
            self._config.storage.evidence_dir,
            self._config.storage.clips_dir,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    def save_snapshot(
        self,
        frame: NDArray[np.uint8],
        incident_id: int,
        filename: str | None = None,
    ) -> str:
        """Save a frame snapshot as evidence.

        Args:
            frame: The video frame to save.
            incident_id: The associated incident ID.
            filename: Optional custom filename.

        Returns:
            The path where the snapshot was saved.
        """
        if filename is None:
            filename = f"snapshot_{incident_id}_{id(frame)}.jpg"

        path = self._config.storage.snapshots_dir / filename
        try:
            cv2.imwrite(str(path), frame)
        except Exception as e:
            raise StorageError(f"Failed to save snapshot: {e}") from e

        self._repository.create(
            {
                "incident_id": incident_id,
                "file_path": str(path),
                "file_type": "snapshot",
                "file_size": path.stat().st_size,
                "mime_type": "image/jpeg",
            }
        )
        logger.debug(f"Snapshot saved: {path}", module="app")
        return str(path)

    def save_annotated(
        self,
        frame: NDArray[np.uint8],
        incident_id: int,
        filename: str | None = None,
    ) -> str:
        """Save an annotated frame with bounding boxes as evidence.

        Args:
            frame: The annotated video frame.
            incident_id: The associated incident ID.
            filename: Optional custom filename.

        Returns:
            The path where the annotated frame was saved.
        """
        if filename is None:
            filename = f"annotated_{incident_id}_{id(frame)}.jpg"

        path = self._config.storage.evidence_dir / filename
        try:
            cv2.imwrite(str(path), frame)
        except Exception as e:
            raise StorageError(f"Failed to save annotated frame: {e}") from e

        self._repository.create(
            {
                "incident_id": incident_id,
                "file_path": str(path),
                "file_type": "annotated",
                "file_size": path.stat().st_size,
                "mime_type": "image/jpeg",
            }
        )
        logger.debug(f"Annotated frame saved: {path}", module="app")
        return str(path)

    def save_clip(
        self,
        frames: list[NDArray[np.uint8]],
        incident_id: int,
        fps: float = 30.0,
        filename: str | None = None,
    ) -> str:
        """Save a video clip from a sequence of frames.

        Args:
            frames: List of video frames forming the clip.
            incident_id: The associated incident ID.
            fps: Frames per second for the output video.
            filename: Optional custom filename.

        Returns:
            The path where the video clip was saved.
        """
        if not frames:
            raise StorageError("No frames provided for clip")

        if filename is None:
            filename = f"clip_{incident_id}_{id(frames)}.mp4"

        path = self._config.storage.clips_dir / filename
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))

        try:
            for frame in frames:
                writer.write(frame)
        finally:
            writer.release()

        self._repository.create(
            {
                "incident_id": incident_id,
                "file_path": str(path),
                "file_type": "clip",
                "file_size": path.stat().st_size,
                "mime_type": "video/mp4",
            }
        )
        logger.debug(f"Video clip saved: {path}", module="app")
        return str(path)

    def get_evidence(self, incident_id: int) -> list[dict[str, Any]]:
        """Retrieve all evidence for an incident.

        Args:
            incident_id: The incident to retrieve evidence for.

        Returns:
            List of evidence metadata dictionaries.
        """
        records = self._repository.get_by_incident(incident_id)
        return [
            {
                "id": r.id,
                "file_path": r.file_path,
                "file_type": r.file_type,
                "file_size": r.file_size,
                "mime_type": r.mime_type,
            }
            for r in records
        ]

    def delete_evidence(self, evidence_id: int) -> bool:
        """Delete an evidence file and its database record.

        Args:
            evidence_id: The evidence record ID.

        Returns:
            True if the evidence was deleted.
        """
        record = self._repository.get_by_id(evidence_id)
        if record is None:
            return False

        path = Path(record.file_path)
        if path.exists():
            path.unlink()

        self._repository.delete(evidence_id)
        logger.info(f"Evidence deleted: {evidence_id}", module="app")
        return True

    def get_storage_usage(self) -> dict[str, Any]:
        """Calculate current storage usage.

        Returns:
            Dictionary with storage paths and their sizes in bytes.
        """
        usage: dict[str, int] = {}
        dirs = {
            "snapshots": self._config.storage.snapshots_dir,
            "evidence": self._config.storage.evidence_dir,
            "clips": self._config.storage.clips_dir,
        }
        for name, d in dirs.items():
            total = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            usage[name] = total
        usage["total"] = sum(usage.values())
        return usage
