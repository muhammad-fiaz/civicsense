"""Camera capture worker thread.

Runs camera frame reading in a background thread to keep
the GUI responsive while continuously capturing video frames.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PySide6.QtCore import QThread, Signal

from civicsense.core.logging import get_logger
from civicsense.services.camera_service import CameraService

logger = get_logger("app")


class CameraWorker(QThread):
    """Background worker for capturing frames from a camera.

    Emits frames at the camera's frame rate and handles
    connection errors gracefully.
    """

    frame_captured = Signal(np.ndarray)
    error_occurred = Signal(str)
    fps_updated = Signal(float)

    def __init__(
        self,
        camera_service: CameraService,
        camera_id: str,
        parent: Any = None,
    ) -> None:
        """Initialize the camera worker.

        Args:
            camera_service: The camera service instance.
            camera_id: The camera to capture from.
            parent: Parent QObject.
        """
        super().__init__(parent)
        self._camera_service = camera_service
        self._camera_id = camera_id
        self._running = False

    def run(self) -> None:
        """Main capture loop running in background thread."""
        self._running = True
        import time

        frame_count = 0
        start_time = time.time()

        while self._running:
            try:
                success, frame = self._camera_service.read(self._camera_id)
                if success and frame is not None:
                    self.frame_captured.emit(frame)
                    frame_count += 1
                    elapsed = time.time() - start_time
                    if elapsed > 1.0:
                        self.fps_updated.emit(frame_count / elapsed)
                        frame_count = 0
                        start_time = time.time()
                else:
                    self.msleep(10)
            except Exception as e:
                self.error_occurred.emit(str(e))
                self._running = False

    def stop(self) -> None:
        """Stop the capture loop."""
        self._running = False
        self.wait()
