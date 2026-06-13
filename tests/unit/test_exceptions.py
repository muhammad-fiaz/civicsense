"""Tests for exception hierarchy."""

from __future__ import annotations

import pytest
from civicsense.core.exceptions import (
    AIError,
    CameraConnectionError,
    CameraError,
    CivicSenseError,
    ConfigurationError,
    DatabaseError,
    ExportError,
    InferenceError,
    IncidentError,
    MigrationError,
    ModelLoadError,
    RepositoryError,
    StorageError,
    ValidationError,
)


class TestExceptionHierarchy:
    """Tests for the exception class hierarchy."""

    def test_base_exception(self) -> None:
        """Verify base exception has correct message."""
        exc = CivicSenseError("test error")
        assert str(exc) == "test error"
        assert exc.message == "test error"

    def test_database_error_is_civic_sense_error(self) -> None:
        """Verify DatabaseError inherits from CivicSenseError."""
        assert issubclass(DatabaseError, CivicSenseError)

    def test_ai_error_is_civic_sense_error(self) -> None:
        """Verify AIError inherits from CivicSenseError."""
        assert issubclass(AIError, CivicSenseError)

    def test_model_load_error_is_ai_error(self) -> None:
        """Verify ModelLoadError inherits from AIError."""
        assert issubclass(ModelLoadError, AIError)

    def test_inference_error_is_ai_error(self) -> None:
        """Verify InferenceError inherits from AIError."""
        assert issubclass(InferenceError, AIError)

    def test_camera_error_is_civic_sense_error(self) -> None:
        """Verify CameraError inherits from CivicSenseError."""
        assert issubclass(CameraError, CivicSenseError)

    def test_camera_connection_error(self) -> None:
        """Verify CameraConnectionError can be raised and caught."""
        with pytest.raises(CameraConnectionError):
            raise CameraConnectionError("camera not found")

    def test_storage_error(self) -> None:
        """Verify StorageError can be raised and caught."""
        with pytest.raises(StorageError):
            raise StorageError("disk full")

    def test_export_error(self) -> None:
        """Verify ExportError can be raised and caught."""
        with pytest.raises(ExportError):
            raise ExportError("export failed")

    def test_migration_error(self) -> None:
        """Verify MigrationError can be raised and caught."""
        with pytest.raises(MigrationError):
            raise MigrationError("migration failed")

    def test_configuration_error(self) -> None:
        """Verify ConfigurationError can be raised and caught."""
        with pytest.raises(ConfigurationError):
            raise ConfigurationError("invalid config")
