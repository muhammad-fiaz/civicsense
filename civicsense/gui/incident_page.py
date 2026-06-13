"""Incident manager page for CivicSense.

Provides search, filter, review, and evidence viewing capabilities
for managing littering incidents.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class IncidentManagerPage(QWidget):
    """Incident management page with search, filtering, and review."""

    search_requested = Signal(str)
    approve_requested = Signal(int)
    reject_requested = Signal(int)
    export_csv_requested = Signal()
    export_json_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the incident manager page.

        Args:
            parent: Parent widget.
        """
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Build the incident manager layout."""
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(16)
        main_layout.setContentsMargins(24, 24, 24, 24)

        header = QLabel("Incident Manager")
        header.setStyleSheet("font-size: 24px; font-weight: bold; color: #ffffff;")
        main_layout.addWidget(header)

        toolbar_layout = QHBoxLayout()

        self._search_input = QLineEdit()
        self._search_input.setPlaceholderText("Search incidents...")
        self._search_input.setStyleSheet("""
            QLineEdit {
                background-color: #2b2b2b;
                color: #ffffff;
                border: 1px solid #444444;
                padding: 8px 12px;
                border-radius: 4px;
                font-size: 14px;
            }
        """)
        self._search_input.returnPressed.connect(
            lambda: self.search_requested.emit(self._search_input.text())
        )
        toolbar_layout.addWidget(self._search_input, stretch=1)

        search_btn = QPushButton("Search")
        search_btn.setStyleSheet("""
            QPushButton {
                background-color: #00d4aa;
                color: #1a1a1a;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #00b894; }
        """)
        search_btn.clicked.connect(
            lambda: self.search_requested.emit(self._search_input.text())
        )
        toolbar_layout.addWidget(search_btn)

        csv_btn = QPushButton("Export CSV")
        csv_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: #ffffff;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #2980b9; }
        """)
        csv_btn.clicked.connect(self.export_csv_requested.emit)
        toolbar_layout.addWidget(csv_btn)

        json_btn = QPushButton("Export JSON")
        json_btn.setStyleSheet("""
            QPushButton {
                background-color: #9b59b6;
                color: #ffffff;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #8e44ad; }
        """)
        json_btn.clicked.connect(self.export_json_requested.emit)
        toolbar_layout.addWidget(json_btn)

        main_layout.addLayout(toolbar_layout)

        self._table = QTableWidget()
        self._table.setColumnCount(8)
        self._table.setHorizontalHeaderLabels(
            [
                "ID",
                "Date/Time",
                "Camera",
                "Waste Type",
                "Confidence",
                "Status",
                "Actions",
                "Notes",
            ]
        )
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self._table.setStyleSheet("""
            QTableWidget {
                background-color: #2b2b2b;
                color: #ffffff;
                border: 1px solid #444444;
                gridline-color: #3a3a3a;
            }
            QTableWidget::item { padding: 8px; }
            QHeaderView::section {
                background-color: #1a1a1a;
                color: #00d4aa;
                padding: 8px;
                border: 1px solid #444444;
                font-weight: bold;
            }
        """)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setAlternatingRowColors(True)
        main_layout.addWidget(self._table, stretch=1)

    def populate_incidents(self, incidents: list[dict[str, Any]]) -> None:
        """Populate the table with incident data.

        Args:
            incidents: List of incident dictionaries.
        """
        self._table.setRowCount(len(incidents))

        for row, inc in enumerate(incidents):
            self._table.setItem(row, 0, QTableWidgetItem(str(inc.get("id", ""))))
            self._table.setItem(row, 1, QTableWidgetItem(inc.get("timestamp", "")))
            self._table.setItem(row, 2, QTableWidgetItem(inc.get("camera_name", "")))
            self._table.setItem(row, 3, QTableWidgetItem(inc.get("waste_type", "")))
            self._table.setItem(
                row, 4, QTableWidgetItem(f"{inc.get('confidence', 0):.2f}")
            )
            self._table.setItem(row, 5, QTableWidgetItem(inc.get("status", "")))
            self._table.setItem(row, 7, QTableWidgetItem(inc.get("review_notes", "")))

            actions_widget = QWidget()
            actions_layout = QHBoxLayout(actions_widget)
            actions_layout.setContentsMargins(4, 2, 4, 2)

            approve_btn = QPushButton("Approve")
            approve_btn.setStyleSheet("""
                QPushButton {
                    background-color: #00d4aa;
                    color: #1a1a1a;
                    border: none;
                    padding: 4px 8px;
                    border-radius: 3px;
                    font-size: 11px;
                }
            """)
            approve_btn.clicked.connect(
                lambda checked, inc_id=inc.get("id"): self.approve_requested.emit(
                    inc_id
                )  # noqa: B008
            )
            actions_layout.addWidget(approve_btn)

            reject_btn = QPushButton("Reject")
            reject_btn.setStyleSheet("""
                QPushButton {
                    background-color: #ff4444;
                    color: #ffffff;
                    border: none;
                    padding: 4px 8px;
                    border-radius: 3px;
                    font-size: 11px;
                }
            """)
            reject_btn.clicked.connect(
                lambda checked, inc_id=inc.get("id"): self.reject_requested.emit(inc_id)  # noqa: B008
            )
            actions_layout.addWidget(reject_btn)

            self._table.setCellWidget(row, 5, actions_widget)
