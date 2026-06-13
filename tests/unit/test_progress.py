"""Tests for progress manager."""

from __future__ import annotations

import pytest
from civicsense.core.progress import ProgressManager


class TestProgressManager:
    """Tests for ProgressManager."""

    def test_initialization(self) -> None:
        """Verify manager initializes with no active bars."""
        manager = ProgressManager()
        assert len(manager._active_bars) == 0

    def test_cancel_nonexistent_raises(self) -> None:
        """Verify cancelling nonexistent task raises KeyError."""
        manager = ProgressManager()
        with pytest.raises(KeyError):
            manager.cancel("nonexistent")

    def test_cancel_all(self) -> None:
        """Verify cancel_all clears all active bars."""
        manager = ProgressManager()
        manager.cancel_all()
        assert len(manager._active_bars) == 0
