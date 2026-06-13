"""Centralized progress tracking using tqdm and Rich.

Provides a ProgressManager that coordinates progress bars across
model downloads, video processing, exports, and other long-running tasks.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from tqdm import tqdm


class ProgressManager:
    """Manages progress bars for long-running operations.

    Supports both Rich progress bars for the GUI console and
    tqdm progress bars for CLI and batch operations.
    """

    def __init__(self) -> None:
        """Initialize the ProgressManager."""
        self._active_bars: dict[str, Any] = {}

    @contextmanager
    def rich_progress(
        self,
        description: str,
        total: int | None = None,
        task_id: str | None = None,
    ) -> Generator[Progress, None, None]:
        """Context manager for Rich-based progress bars.

        Args:
            description: Description of the task being tracked.
            total: Total number of steps. None for indeterminate progress.
            task_id: Optional identifier for the progress task.

        Yields:
            The Rich Progress instance.
        """
        columns: list[ProgressColumn] = [
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ]
        with Progress(*columns) as progress:
            progress.add_task(description, total=total)
            if task_id:
                self._active_bars[task_id] = progress
            try:
                yield progress
            finally:
                if task_id:
                    self._active_bars.pop(task_id, None)

    @contextmanager
    def tqdm_progress(
        self,
        description: str,
        total: int | None = None,
        task_id: str | None = None,
    ) -> Generator[tqdm, None, None]:
        """Context manager for tqdm-based progress bars.

        Args:
            description: Description of the task being tracked.
            total: Total number of steps. None for indeterminate progress.
            task_id: Optional identifier for the progress task.

        Yields:
            The tqdm progress bar instance.
        """
        bar = tqdm(total=total, desc=description, unit="it")
        if task_id:
            self._active_bars[task_id] = bar
        try:
            yield bar
        finally:
            bar.close()
            if task_id:
                self._active_bars.pop(task_id, None)

    def cancel(self, task_id: str) -> None:
        """Cancel an active progress bar.

        Args:
            task_id: The identifier of the progress task to cancel.

        Raises:
            KeyError: If the task_id is not found.
        """
        bar = self._active_bars.pop(task_id, None)
        if bar is None:
            raise KeyError(f"No active progress bar with id: {task_id}")
        if hasattr(bar, "cancel"):
            bar.cancel()

    def cancel_all(self) -> None:
        """Cancel all active progress bars."""
        for task_id in list(self._active_bars.keys()):
            self.cancel(task_id)
