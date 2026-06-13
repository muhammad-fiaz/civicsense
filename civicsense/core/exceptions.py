"""Application-wide exception hierarchy for CivicSense.

Defines typed exceptions for each subsystem to enable precise error handling
and structured error reporting throughout the application.
"""

from __future__ import annotations


class CivicSenseError(Exception):
    """Base exception for all CivicSense errors."""

    def __init__(self, message: str = "An unexpected error occurred") -> None:
        """Initialize the exception.

        Args:
            message: Human-readable error description.
        """
        super().__init__(message)
        self.message = message


class DatabaseError(CivicSenseError):
    """Raised when a database operation fails."""


class ConnectionError(DatabaseError):
    """Raised when a database connection cannot be established."""


class MigrationError(DatabaseError):
    """Raised when an Alembic migration fails."""


class RepositoryError(DatabaseError):
    """Raised when a repository operation fails."""


class AIError(CivicSenseError):
    """Raised when an AI inference operation fails."""


class ModelLoadError(AIError):
    """Raised when a YOLO model fails to load."""


class InferenceError(AIError):
    """Raised when an inference pass fails."""


class TrackerError(AIError):
    """Raised when the object tracker encounters an error."""


class CameraError(CivicSenseError):
    """Raised when a camera operation fails."""


class CameraConnectionError(CameraError):
    """Raised when a camera cannot be connected."""


class CameraFrameError(CameraError):
    """Raised when a frame cannot be captured."""


class StorageError(CivicSenseError):
    """Raised when a file storage operation fails."""


class IncidentError(CivicSenseError):
    """Raised when an incident processing operation fails."""


class ExportError(CivicSenseError):
    """Raised when an export operation fails."""


class ConfigurationError(CivicSenseError):
    """Raised when configuration validation fails."""


class ValidationError(CivicSenseError):
    """Raised when data validation fails."""
