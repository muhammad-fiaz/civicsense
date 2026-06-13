"""Camera management service.

Handles camera lifecycle operations including opening, closing,
frame capture, and stream management for multiple camera sources.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from civicsense.core.exceptions import CameraConnectionError
from civicsense.core.logging import get_logger
from civicsense.events.event_bus import Event, EventType, get_event_bus

logger = get_logger("app")


class CameraService:
    """Manages camera connections and frame capture.

    Provides a unified interface for opening camera streams,
    reading frames, and managing camera lifecycle.
    """

    _cached_cameras: list[dict[str, str | int]] | None = None

    def __init__(self) -> None:
        """Initialize the CameraService with no open cameras."""
        self._cameras: dict[str, cv2.VideoCapture] = {}
        self._event_bus = get_event_bus()

    def open(self, camera_id: str, source: str | int, **kwargs: Any) -> bool:
        """Open a camera stream.

        Args:
            camera_id: Unique identifier for this camera.
            source: Camera index or RTSP/video stream URL.
            **kwargs: Additional OpenCV VideoCapture properties.

        Returns:
            True if the camera was opened successfully.

        Raises:
            CameraConnectionError: If the camera cannot be connected.
        """
        try:
            cap = cv2.VideoCapture(source)
            for key, value in kwargs.items():
                prop = getattr(cv2, f"CAP_PROP_{key.upper()}", None)
                if prop is not None:
                    cap.set(prop, value)

            if not cap.isOpened():
                raise CameraConnectionError(f"Cannot open camera: {source}")

            self._cameras[camera_id] = cap
            self._event_bus.publish(
                Event(
                    event_type=EventType.CAMERA_CONNECTED,
                    data={"camera_id": camera_id, "source": str(source)},
                    source="CameraService",
                )
            )
            logger.info(f"Camera {camera_id} opened from {source}", module="app")
            return True
        except CameraConnectionError:
            raise
        except Exception as e:
            raise CameraConnectionError(
                f"Failed to open camera {camera_id}: {e}"
            ) from e

    def read(self, camera_id: str) -> tuple[bool, NDArray[np.uint8] | None]:
        """Read a frame from a camera.

        Args:
            camera_id: The camera identifier.

        Returns:
            Tuple of (success, frame). Frame is None on failure.

        Raises:
            KeyError: If the camera_id is not found.
        """
        if camera_id not in self._cameras:
            raise KeyError(f"Camera not found: {camera_id}")

        cap = self._cameras[camera_id]
        ret, frame = cap.read()

        if not ret or frame is None:
            return False, None

        return True, np.asarray(frame, dtype=np.uint8)

    def release(self, camera_id: str) -> None:
        """Release a camera stream.

        Args:
            camera_id: The camera identifier to release.
        """
        cap = self._cameras.pop(camera_id, None)
        if cap is not None:
            cap.release()
            self._event_bus.publish(
                Event(
                    event_type=EventType.CAMERA_DISCONNECTED,
                    data={"camera_id": camera_id},
                    source="CameraService",
                )
            )
            logger.info(f"Camera {camera_id} released", module="app")

    def release_all(self) -> None:
        """Release all open camera streams."""
        for camera_id in list(self._cameras.keys()):
            self.release(camera_id)

    def get_fps(self, camera_id: str) -> float:
        """Get the frame rate of a camera.

        Args:
            camera_id: The camera identifier.

        Returns:
            The camera frame rate.
        """
        if camera_id not in self._cameras:
            return 0.0
        cap = self._cameras[camera_id]
        return cap.get(cv2.CAP_PROP_FPS)

    def get_resolution(self, camera_id: str) -> tuple[int, int]:
        """Get the resolution of a camera.

        Args:
            camera_id: The camera identifier.

        Returns:
            Tuple of (width, height).
        """
        if camera_id not in self._cameras:
            return (0, 0)
        cap = self._cameras[camera_id]
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width, height)

    @property
    def active_cameras(self) -> list[str]:
        """Return list of active camera IDs."""
        return list(self._cameras.keys())

    @staticmethod
    def enumerate_cameras(max_test: int = 6) -> list[dict[str, str | int]]:
        """Detect available cameras on the system.

        Args:
            max_test: Maximum number of camera indices to probe.

        Returns:
            List of dicts with 'id', 'name', and 'index' keys.
        """
        if CameraService._cached_cameras is not None:
            return CameraService._cached_cameras

        available = []
        found_any = False
        for i in range(max_test):
            cap = cv2.VideoCapture(i)
            try:
                if not cap.isOpened():
                    if found_any:
                        break
                    continue
                ret, _ = cap.read()
                if ret:
                    found_any = True
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    available.append(
                        {
                            "id": f"camera_{i}",
                            "name": f"Camera {i} ({width}x{height})",
                            "index": i,
                        }
                    )
                else:
                    if found_any:
                        break
            except Exception:
                if found_any:
                    break
            finally:
                cap.release()
        CameraService._cached_cameras = available
        return available
