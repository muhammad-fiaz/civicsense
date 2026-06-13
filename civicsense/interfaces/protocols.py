"""Reusable interface and protocol definitions for CivicSense.

All subsystems depend on these interfaces rather than concrete implementations,
enabling dependency injection, testability, and swappable backends.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from civicsense.dto.detection import DetectionResult, PoseResult, TrackedObject


@runtime_checkable
class DetectorInterface(Protocol):
    """Protocol for object detection backends."""

    def load(self, model_path: str, device: str = "auto") -> None:
        """Load the detection model.

        Args:
            model_path: Path or name of the model weights.
            device: Target computation device.
        """
        ...

    def detect(
        self,
        frame: NDArray[np.uint8],
        confidence: float = 0.5,
        iou: float = 0.45,
    ) -> DetectionResult:
        """Run detection on a single frame.

        Args:
            frame: Input image as a BGR numpy array.
            confidence: Minimum confidence threshold.
            iou: IoU threshold for NMS.

        Returns:
            Detection result containing bounding boxes and class info.
        """
        ...

    def unload(self) -> None:
        """Release model resources."""
        ...

    @property
    def is_loaded(self) -> bool:
        """Return True if the model is loaded and ready."""
        ...


@runtime_checkable
class PoseEstimatorInterface(Protocol):
    """Protocol for pose estimation backends."""

    def load(self, model_path: str, device: str = "auto") -> None:
        """Load the pose estimation model.

        Args:
            model_path: Path or name of the pose model weights.
            device: Target computation device.
        """
        ...

    def estimate(
        self,
        frame: NDArray[np.uint8],
        confidence: float = 0.5,
    ) -> PoseResult:
        """Estimate pose keypoints for detected persons.

        Args:
            frame: Input image as a BGR numpy array.
            confidence: Minimum confidence threshold.

        Returns:
            Pose result containing keypoints for each detected person.
        """
        ...

    def unload(self) -> None:
        """Release model resources."""
        ...

    @property
    def is_loaded(self) -> bool:
        """Return True if the model is loaded and ready."""
        ...


@runtime_checkable
class TrackerInterface(Protocol):
    """Protocol for multi-object tracking backends."""

    def initialize(self, **kwargs: Any) -> None:
        """Initialize the tracker with configuration.

        Args:
            **kwargs: Tracker-specific configuration parameters.
        """
        ...

    def update(
        self,
        detections: list[TrackedObject],
        frame: NDArray[np.uint8],
    ) -> list[TrackedObject]:
        """Update tracking with new detections.

        Args:
            detections: Current frame detections.
            frame: The current video frame.

        Returns:
            Updated list of tracked objects with stable IDs.
        """
        ...

    def reset(self) -> None:
        """Reset tracker state."""
        ...


@runtime_checkable
class CameraInterface(Protocol):
    """Protocol for camera capture backends."""

    def open(self, source: str | int, **kwargs: Any) -> bool:
        """Open the camera stream.

        Args:
            source: Camera index or stream URL.
            **kwargs: Additional backend-specific options.

        Returns:
            True if the camera was opened successfully.
        """
        ...

    def read(self) -> tuple[bool, NDArray[np.uint8] | None]:
        """Read a frame from the camera.

        Returns:
            Tuple of (success_flag, frame). Frame is None on failure.
        """
        ...

    def release(self) -> None:
        """Release camera resources."""
        ...

    @property
    def is_opened(self) -> bool:
        """Return True if the camera is currently open."""
        ...

    @property
    def fps(self) -> float:
        """Return the camera frame rate."""
        ...

    @property
    def resolution(self) -> tuple[int, int]:
        """Return the camera resolution as (width, height)."""
        ...


@runtime_checkable
class RepositoryInterface(Protocol):
    """Protocol for data repository backends."""

    def get_by_id(self, id: int) -> Any:
        """Retrieve a record by its ID.

        Args:
            id: The primary key.

        Returns:
            The matching record or None.
        """
        ...

    def get_all(self, offset: int = 0, limit: int = 100) -> list[Any]:
        """Retrieve all records with pagination.

        Args:
            offset: Number of records to skip.
            limit: Maximum number of records to return.

        Returns:
            List of matching records.
        """
        ...

    def create(self, data: Any) -> Any:
        """Create a new record.

        Args:
            data: The data for the new record.

        Returns:
            The created record.
        """
        ...

    def update(self, id: int, data: Any) -> Any:
        """Update an existing record.

        Args:
            id: The primary key.
            data: The updated data.

        Returns:
            The updated record.
        """
        ...

    def delete(self, id: int) -> bool:
        """Delete a record.

        Args:
            id: The primary key.

        Returns:
            True if the record was deleted.
        """
        ...

    def count(self) -> int:
        """Count total records.

        Returns:
            The total number of records.
        """
        ...


@runtime_checkable
class StorageInterface(Protocol):
    """Protocol for file storage backends."""

    def save(self, data: bytes, path: str) -> str:
        """Save binary data to storage.

        Args:
            data: The bytes to save.
            path: Relative storage path.

        Returns:
            The absolute path where data was saved.
        """
        ...

    def load(self, path: str) -> bytes:
        """Load binary data from storage.

        Args:
            path: Relative storage path.

        Returns:
            The loaded bytes.
        """
        ...

    def delete(self, path: str) -> bool:
        """Delete a file from storage.

        Args:
            path: Relative storage path.

        Returns:
            True if the file was deleted.
        """
        ...

    def exists(self, path: str) -> bool:
        """Check if a file exists in storage.

        Args:
            path: Relative storage path.

        Returns:
            True if the file exists.
        """
        ...

    def get_size(self, path: str) -> int:
        """Get the size of a file in bytes.

        Args:
            path: Relative storage path.

        Returns:
            File size in bytes.
        """
        ...


@runtime_checkable
class AnalyticsInterface(Protocol):
    """Protocol for analytics computation backends."""

    def compute_daily(self, date: str) -> dict[str, Any]:
        """Compute analytics for a given day.

        Args:
            date: Date string in YYYY-MM-DD format.

        Returns:
            Dictionary of daily analytics metrics.
        """
        ...

    def compute_weekly(self, start_date: str) -> dict[str, Any]:
        """Compute analytics for a week starting from the given date.

        Args:
            start_date: Start date in YYYY-MM-DD format.

        Returns:
            Dictionary of weekly analytics metrics.
        """
        ...

    def compute_monthly(self, year: int, month: int) -> dict[str, Any]:
        """Compute analytics for a given month.

        Args:
            year: The calendar year.
            month: The calendar month (1-12).

        Returns:
            Dictionary of monthly analytics metrics.
        """
        ...


@runtime_checkable
class ExporterInterface(Protocol):
    """Protocol for data export backends."""

    def export_csv(self, data: list[Any], output_path: str) -> str:
        """Export data to CSV format.

        Args:
            data: List of records to export.
            output_path: Destination file path.

        Returns:
            The absolute path of the exported file.
        """
        ...

    def export_json(self, data: list[Any], output_path: str) -> str:
        """Export data to JSON format.

        Args:
            data: List of records to export.
            output_path: Destination file path.

        Returns:
            The absolute path of the exported file.
        """
        ...
