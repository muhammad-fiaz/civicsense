"""AI inference worker thread.

Runs the full detection -> pose -> tracking -> classification pipeline
in a background thread to keep the GUI responsive.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from PySide6.QtCore import QThread, Signal

from civicsense.ai.inference_worker import InferenceWorker
from civicsense.ai.model_manager import ModelManager
from civicsense.core.logging import get_logger

logger = get_logger("app")


class AIWorker(QThread):
    """Background worker for AI inference processing.

    Receives video frames, runs them through the inference pipeline,
    and emits results back to the GUI thread.
    """

    result_ready = Signal(object)
    error_occurred = Signal(str)
    status_changed = Signal(str)

    def __init__(
        self,
        model_manager: ModelManager,
        parent: Any = None,
    ) -> None:
        """Initialize the AI worker.

        Args:
            model_manager: The shared ModelManager instance.
            parent: Parent QObject.
        """
        super().__init__(parent)
        self._model_manager = model_manager
        self._inference_worker = InferenceWorker(model_manager)
        self._frame: np.ndarray | None = None
        self._running = False
        self._processing = False
        self._error_count = 0
        self._last_error_time = 0.0

    def set_frame(self, frame: np.ndarray) -> None:
        """Queue a frame for processing.

        Args:
            frame: The video frame to process.
        """
        self._frame = frame

    def run(self) -> None:
        """Main processing loop running in background thread."""
        self._running = True
        self.status_changed.emit("AI Worker started")

        while self._running:
            if self._frame is not None and not self._processing:
                self._processing = True
                try:
                    result = self._inference_worker.process_frame(self._frame)
                    self.result_ready.emit(result)
                    self._error_count = 0
                except Exception as e:
                    self._error_count += 1
                    now = time.time()
                    # Rate limit: emit at most once per 2 seconds
                    if now - self._last_error_time > 2.0:
                        self.error_occurred.emit(str(e))
                        self._last_error_time = now
                    logger.error(
                        f"AI inference error ({self._error_count}): {e}", module="app"
                    )
                finally:
                    self._processing = False
                    self._frame = None
            else:
                self.msleep(1)

    def stop(self) -> None:
        """Stop the processing loop."""
        self._running = False
        self._inference_worker.reset()
        self.wait()
