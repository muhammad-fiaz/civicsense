"""Analytics page for CivicSense.

Displays trend charts, camera statistics, waste type breakdowns,
and provides report generation capabilities.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class ChartPlaceholder(QFrame):
    """Placeholder frame for chart display areas."""

    def __init__(
        self,
        title: str,
        parent: QWidget | None = None,
    ) -> None:
        """Initialize the chart placeholder.

        Args:
            title: Chart title.
            parent: Parent widget.
        """
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setMinimumHeight(200)
        self.setStyleSheet("""
            ChartPlaceholder {
                background-color: #2b2b2b;
                border-radius: 8px;
                padding: 16px;
            }
        """)

        layout = QVBoxLayout(self)
        label = QLabel(title)
        label.setStyleSheet("font-size: 14px; font-weight: bold; color: #00d4aa;")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)

        self._content = QLabel("No data available")
        self._content.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._content.setStyleSheet("color: #666666; font-size: 12px;")
        layout.addWidget(self._content)

    def update_content(self, text: str) -> None:
        """Update the placeholder content.

        Args:
            text: Text to display.
        """
        self._content.setText(text)


class AnalyticsPage(QWidget):
    """Analytics page with trend charts and statistics."""

    generate_report_requested = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the analytics page.

        Args:
            parent: Parent widget.
        """
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Build the analytics layout."""
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(16)
        main_layout.setContentsMargins(24, 24, 24, 24)

        header_layout = QHBoxLayout()
        header = QLabel("Analytics")
        header.setStyleSheet("font-size: 24px; font-weight: bold; color: #ffffff;")
        header_layout.addWidget(header)
        header_layout.addStretch()

        for period in ["Daily", "Weekly", "Monthly"]:
            btn = QPushButton(f"{period} Report")
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #3498db;
                    color: #ffffff;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 4px;
                }
                QPushButton:hover { background-color: #2980b9; }
            """)
            btn.clicked.connect(
                lambda checked, p=period.lower(): self.generate_report_requested.emit(p)  # noqa: B008
            )
            header_layout.addWidget(btn)

        main_layout.addLayout(header_layout)

        charts_grid = QGridLayout()
        charts_grid.setSpacing(16)

        self._trend_chart = ChartPlaceholder("Incident Trend")
        self._waste_chart = ChartPlaceholder("Waste Type Distribution")
        self._camera_chart = ChartPlaceholder("Camera Statistics")
        self._hourly_chart = ChartPlaceholder("Hourly Distribution")

        charts_grid.addWidget(self._trend_chart, 0, 0)
        charts_grid.addWidget(self._waste_chart, 0, 1)
        charts_grid.addWidget(self._camera_chart, 1, 0)
        charts_grid.addWidget(self._hourly_chart, 1, 1)

        main_layout.addLayout(charts_grid, stretch=1)

        self._report_text = QTextEdit()
        self._report_text.setReadOnly(True)
        self._report_text.setMaximumHeight(200)
        self._report_text.setStyleSheet("""
            QTextEdit {
                background-color: #2b2b2b;
                color: #cccccc;
                border-radius: 8px;
                padding: 12px;
                font-family: monospace;
                font-size: 12px;
            }
        """)
        self._report_text.setPlaceholderText("Report output will appear here...")
        main_layout.addWidget(self._report_text)

    def update_trend(self, data: list[dict[str, Any]]) -> None:
        """Update the trend chart with data.

        Args:
            data: List of daily count dictionaries.
        """
        if not data:
            self._trend_chart.update_content("No trend data")
            return
        lines = [f"{d['date']}: {d['count']} incidents" for d in data[-7:]]
        self._trend_chart.update_content("\n".join(lines))

    def update_waste_distribution(self, data: dict[str, int]) -> None:
        """Update the waste type chart.

        Args:
            data: Waste type to count mapping.
        """
        if not data:
            self._waste_chart.update_content("No waste data")
            return
        lines = [f"{k}: {v}" for k, v in sorted(data.items(), key=lambda x: -x[1])]
        self._waste_chart.update_content("\n".join(lines))

    def update_camera_stats(self, data: dict[str, dict[str, int]]) -> None:
        """Update the camera statistics chart.

        Args:
            data: Camera ID to stats mapping.
        """
        if not data:
            self._camera_chart.update_content("No camera data")
            return
        lines = [f"{k}: {v.get('total', 0)} total" for k, v in data.items()]
        self._camera_chart.update_content("\n".join(lines))

    def update_hourly(self, data: dict[str, int]) -> None:
        """Update the hourly distribution chart.

        Args:
            data: Hour to count mapping.
        """
        if not data:
            self._hourly_chart.update_content("No hourly data")
            return
        max_val = max(data.values()) if data.values() else 1
        lines = []
        for hour in range(24):
            count = data.get(str(hour), 0)
            bar_len = int((count / max_val) * 20) if max_val > 0 else 0
            bar = "#" * bar_len
            lines.append(f"{hour:02d}:00 | {bar}")
        self._hourly_chart.update_content("\n".join(lines))

    def display_report(self, report: dict[str, Any]) -> None:
        """Display a formatted analytics report.

        Args:
            report: The analytics report dictionary.
        """
        lines = [
            f"Period: {report.get('period', 'unknown')}",
            f"Date: {report.get('date', 'N/A')}",
            f"Total Incidents: {report.get('total_incidents', 0)}",
            f"Approved: {report.get('approved', 0)}",
            f"Rejected: {report.get('rejected', 0)}",
            f"Pending: {report.get('pending', 0)}",
            f"Unique Cameras: {report.get('unique_cameras', 0)}",
            f"Avg Confidence: {report.get('average_confidence', 0):.3f}",
            "",
            "Waste Type Breakdown:",
        ]
        for wtype, count in report.get("waste_type_breakdown", {}).items():
            lines.append(f"  {wtype}: {count}")

        self._report_text.setText("\n".join(lines))
