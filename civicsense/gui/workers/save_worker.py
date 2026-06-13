"""File save worker thread.

Handles saving evidence files (snapshots, clips, annotated frames)
in a background thread to avoid blocking the GUI.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PySide6.QtCore import QThread, Signal

from civicsense.core.logging import get_logger
from civicsense.services.evidence_service import EvidenceService

logger = get_logger("app")


class SaveWorker(QThread):
    """Background worker for saving evidence files.

    Accepts frames and metadata, then saves them to disk
    without blocking the main thread.
    """

    save_complete = Signal(str)
    error_occurred = Signal(str)

    def __init__(
        self,
        evidence_service: EvidenceService,
        parent: Any = None,
    ) -> None:
        """Initialize the save worker.

        Args:
            evidence_service: The evidence storage service.
            parent: Parent QObject.
        """
        super().__init__(parent)
        self._evidence_service = evidence_service
        self._tasks: list[dict[str, Any]] = []
        self._running = False

    def queue_snapshot(
        self,
        frame: np.ndarray,
        incident_id: int,
    ) -> None:
        """Queue a snapshot for saving.

        Args:
            frame: The frame to save.
            incident_id: Associated incident ID.
        """
        self._tasks.append(
            {
                "type": "snapshot",
                "frame": frame.copy(),
                "incident_id": incident_id,
            }
        )

    def queue_annotated(
        self,
        frame: np.ndarray,
        incident_id: int,
    ) -> None:
        """Queue an annotated frame for saving.

        Args:
            frame: The annotated frame to save.
            incident_id: Associated incident ID.
        """
        self._tasks.append(
            {
                "type": "annotated",
                "frame": frame.copy(),
                "incident_id": incident_id,
            }
        )

    def run(self) -> None:
        """Process the save queue."""
        self._running = True

        while self._running or self._tasks:
            if not self._tasks:
                self.msleep(10)
                continue

            task = self._tasks.pop(0)
            try:
                if task["type"] == "snapshot":
                    path = self._evidence_service.save_snapshot(
                        task["frame"], task["incident_id"]
                    )
                elif task["type"] == "annotated":
                    path = self._evidence_service.save_annotated(
                        task["frame"], task["incident_id"]
                    )
                else:
                    continue
                self.save_complete.emit(path)
            except Exception as e:
                self.error_occurred.emit(str(e))
                logger.error(f"Save error: {e}", module="app")

    def stop(self) -> None:
        """Stop the save worker after current tasks complete."""
        self._running = False
        self.wait()
