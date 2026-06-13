"""Dashboard page for CivicSense.

Displays overview statistics including incident counts, camera status,
AI model status, GPU information, and storage usage.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)


class StatCard(QFrame):
    """A styled card displaying a single statistic."""

    def __init__(
        self,
        title: str,
        value: str = "0",
        parent: QWidget | None = None,
    ) -> None:
        """Initialize the stat card.

        Args:
            title: The statistic label.
            value: The initial value.
            parent: Parent widget.
        """
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet("""
            StatCard {
                background-color: #2b2b2b;
                border-radius: 8px;
                padding: 16px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        self._value_label = QLabel(value)
        self._value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._value_label.setStyleSheet(
            "font-size: 28px; font-weight: bold; color: #00d4aa;"
        )
        layout.addWidget(self._value_label)

        self._title_label = QLabel(title)
        self._title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._title_label.setStyleSheet(
            "font-size: 12px; color: #888888; text-transform: uppercase;"
        )
        layout.addWidget(self._title_label)

    def set_value(self, value: str) -> None:
        """Update the displayed value.

        Args:
            value: New value to display.
        """
        self._value_label.setText(value)


class StatusIndicator(QWidget):
    """A colored status indicator with label."""

    def __init__(
        self,
        label: str,
        is_active: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        """Initialize the status indicator.

        Args:
            label: Status label text.
            is_active: Whether the status is active/good.
            parent: Parent widget.
        """
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._dot = QLabel()
        self._dot.setFixedSize(12, 12)
        self._update_dot_color(is_active)
        layout.addWidget(self._dot)

        self._label = QLabel(label)
        self._label.setStyleSheet("color: #cccccc; font-size: 13px;")
        layout.addWidget(self._label)
        layout.addStretch()

    def set_active(self, active: bool) -> None:
        """Update the active state.

        Args:
            active: Whether the status is active/good.
        """
        self._update_dot_color(active)

    def _update_dot_color(self, active: bool) -> None:
        """Update the dot color based on state.

        Args:
            active: Whether the status is active.
        """
        color = "#00d4aa" if active else "#ff4444"
        self._dot.setStyleSheet(f"background-color: {color}; border-radius: 6px;")


class DashboardPage(QWidget):
    """Dashboard overview page.

    Displays key metrics, system status indicators, and quick actions.
    """

    refresh_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the dashboard page.

        Args:
            parent: Parent widget.
        """
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Build the dashboard layout."""
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(24, 24, 24, 24)

        header = QLabel("Dashboard")
        header.setStyleSheet("font-size: 24px; font-weight: bold; color: #ffffff;")
        main_layout.addWidget(header)

        stats_grid = QGridLayout()
        stats_grid.setSpacing(16)

        self._total_incidents = StatCard("Total Incidents", "0")
        self._today_incidents = StatCard("Today", "0")
        self._weekly_incidents = StatCard("This Week", "0")
        self._monthly_incidents = StatCard("This Month", "0")

        stats_grid.addWidget(self._total_incidents, 0, 0)
        stats_grid.addWidget(self._today_incidents, 0, 1)
        stats_grid.addWidget(self._weekly_incidents, 0, 2)
        stats_grid.addWidget(self._monthly_incidents, 0, 3)

        main_layout.addLayout(stats_grid)

        status_frame = QFrame()
        status_frame.setStyleSheet("""
            QFrame {
                background-color: #2b2b2b;
                border-radius: 8px;
                padding: 16px;
            }
        """)
        status_layout = QVBoxLayout(status_frame)

        status_title = QLabel("System Status")
        status_title.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: #ffffff; border: none;"
        )
        status_layout.addWidget(status_title)

        self._camera_status = StatusIndicator("Active Cameras")
        self._ai_status = StatusIndicator("AI Models")
        self._gpu_status = StatusIndicator("GPU Acceleration")
        self._db_status = StatusIndicator("Database")

        status_layout.addWidget(self._camera_status)
        status_layout.addWidget(self._ai_status)
        status_layout.addWidget(self._gpu_status)
        status_layout.addWidget(self._db_status)

        main_layout.addWidget(status_frame)

        self._storage_label = QLabel("Storage Usage: --")
        self._storage_label.setStyleSheet("color: #888888; font-size: 13px;")
        main_layout.addWidget(self._storage_label)

        main_layout.addStretch()

    def update_stats(
        self,
        total: int = 0,
        today: int = 0,
        weekly: int = 0,
        monthly: int = 0,
    ) -> None:
        """Update the dashboard statistics.

        Args:
            total: Total incident count.
            today: Today's incident count.
            weekly: This week's incident count.
            monthly: This month's incident count.
        """
        self._total_incidents.set_value(str(total))
        self._today_incidents.set_value(str(today))
        self._weekly_incidents.set_value(str(weekly))
        self._monthly_incidents.set_value(str(monthly))

    def update_status(
        self,
        cameras_active: bool = False,
        ai_loaded: bool = False,
        gpu_available: bool = False,
        db_connected: bool = False,
    ) -> None:
        """Update the system status indicators.

        Args:
            cameras_active: Whether cameras are connected.
            ai_loaded: Whether AI models are loaded.
            gpu_available: Whether GPU is available.
            db_connected: Whether the database is connected.
        """
        self._camera_status.set_active(cameras_active)
        self._ai_status.set_active(ai_loaded)
        self._gpu_status.set_active(gpu_available)
        self._db_status.set_active(db_connected)

    def update_storage(self, usage_bytes: int) -> None:
        """Update storage usage display.

        Args:
            usage_bytes: Total storage usage in bytes.
        """
        if usage_bytes < 1024 * 1024:
            text = f"{usage_bytes / 1024:.1f} KB"
        elif usage_bytes < 1024 * 1024 * 1024:
            text = f"{usage_bytes / (1024 * 1024):.1f} MB"
        else:
            text = f"{usage_bytes / (1024 * 1024 * 1024):.2f} GB"
        self._storage_label.setText(f"Storage Usage: {text}")
