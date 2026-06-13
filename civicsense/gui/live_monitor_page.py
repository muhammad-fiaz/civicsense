"""Live monitoring page with video feed and detection overlays."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
import numpy as np


class VideoWidget(QLabel):
    """Widget for displaying video frames with aspect ratio preservation."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(640, 480)
        self.setStyleSheet(
            "background-color: #1a1a1a; border: 1px solid #333333; border-radius: 8px; color: #666666; font-size: 16px;"
        )
        self.setText("No Video Feed")

    def update_frame(self, frame: np.ndarray) -> None:
        h, w, ch = frame.shape
        rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])
        q_image = QImage(rgb_frame.tobytes(), w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        scaled = pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)


class InfoPanel(QWidget):
    """Panel displaying detection statistics and system info."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setStyleSheet("""
            InfoPanel {
                background-color: #2b2b2b;
                border: 1px solid #333333;
                border-radius: 8px;
                padding: 12px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        self._fps_label = QLabel("FPS: --")
        self._fps_label.setStyleSheet(
            "color: #00d4aa; font-size: 14px; font-weight: bold;"
        )
        layout.addWidget(self._fps_label)

        self._persons_label = QLabel("Persons: 0")
        self._persons_label.setStyleSheet("color: #ffffff; font-size: 14px;")
        layout.addWidget(self._persons_label)

        self._waste_label = QLabel("Waste Objects: 0")
        self._waste_label.setStyleSheet("color: #ffffff; font-size: 14px;")
        layout.addWidget(self._waste_label)

        self._dustbin_label = QLabel("Dustbins: 0")
        self._dustbin_label.setStyleSheet("color: #ffffff; font-size: 14px;")
        layout.addWidget(self._dustbin_label)

        self._gpu_label = QLabel("Device: CPU")
        self._gpu_label.setStyleSheet("color: #888888; font-size: 12px;")
        layout.addWidget(self._gpu_label)

        layout.addStretch()

    def update_stats(
        self,
        fps: float = 0.0,
        persons: int = 0,
        waste: int = 0,
        dustbins: int = 0,
        device: str = "CPU",
    ) -> None:
        self._fps_label.setText(f"FPS: {fps:.1f}")
        self._persons_label.setText(f"Persons: {persons}")
        self._waste_label.setText(f"Waste Objects: {waste}")
        self._dustbin_label.setText(f"Dustbins: {dustbins}")
        self._gpu_label.setText(f"Device: {device}")


class LiveMonitorPage(QWidget):
    """Live monitoring page with video feed and detection overlays."""

    camera_toggle_requested = Signal()
    camera_switch_requested = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._camera_active = False
        self._loading = False
        self._setup_ui()

    def _setup_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(16)
        main_layout.setContentsMargins(24, 24, 24, 24)

        header_layout = QHBoxLayout()
        header = QLabel("Live Monitor")
        header.setStyleSheet("font-size: 24px; font-weight: bold; color: #ffffff;")
        header_layout.addWidget(header)
        header_layout.addStretch()

        self._camera_combo = QComboBox()
        self._camera_combo.setMinimumWidth(200)
        self._camera_combo.setStyleSheet("""
            QComboBox {
                background-color: #2b2b2b;
                color: #ffffff;
                border: 1px solid #444444;
                padding: 6px 12px;
                border-radius: 6px;
                font-size: 13px;
            }
            QComboBox:hover { border-color: #00d4aa; }
            QComboBox:focus { border-color: #00d4aa; }
            QComboBox::drop-down { border: none; width: 24px; }
            QComboBox QAbstractItemView {
                background-color: #2b2b2b;
                color: #ffffff;
                selection-background-color: #00d4aa;
                selection-color: #1a1a1a;
                border: 1px solid #444444;
                border-radius: 4px;
                padding: 4px;
            }
        """)
        self._camera_combo.currentIndexChanged.connect(self._on_camera_changed)
        header_layout.addWidget(self._camera_combo)

        self._camera_url = QLineEdit()
        self._camera_url.setPlaceholderText("RTSP/HTTP URL")
        self._camera_url.setMaximumWidth(250)
        self._camera_url.setStyleSheet("""
            QLineEdit {
                background-color: #2b2b2b;
                color: #ffffff;
                border: 1px solid #444444;
                padding: 6px 12px;
                border-radius: 6px;
                font-size: 13px;
            }
            QLineEdit:hover { border-color: #00d4aa; }
            QLineEdit:focus { border-color: #00d4aa; }
            QLineEdit::placeholder { color: #666666; }
        """)
        header_layout.addWidget(self._camera_url)

        self._toggle_btn = QPushButton("Start Camera")
        self._toggle_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._toggle_btn.setFixedWidth(140)
        self._apply_start_style()
        self._toggle_btn.clicked.connect(self._on_toggle_clicked)
        header_layout.addWidget(self._toggle_btn)

        main_layout.addLayout(header_layout)

        self._status_label = QLabel("")
        self._status_label.setStyleSheet("color: #888888; font-size: 12px;")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self._status_label)

        content_layout = QHBoxLayout()

        self._video_widget = VideoWidget()
        content_layout.addWidget(self._video_widget, stretch=3)

        right_panel = QVBoxLayout()

        self._info_panel = InfoPanel()
        right_panel.addWidget(self._info_panel)

        self._alerts_text = QTextEdit()
        self._alerts_text.setReadOnly(True)
        self._alerts_text.setStyleSheet("""
            QTextEdit {
                background-color: #2b2b2b;
                color: #cccccc;
                border: 1px solid #333333;
                border-radius: 8px;
                padding: 12px;
                font-size: 12px;
                selection-background-color: #00d4aa;
                selection-color: #1a1a1a;
            }
        """)
        self._alerts_text.setPlaceholderText("Alerts will appear here...")
        right_panel.addWidget(self._alerts_text, stretch=1)

        content_layout.addLayout(right_panel, stretch=1)

        main_layout.addLayout(content_layout)

    def _apply_start_style(self) -> None:
        self._toggle_btn.setStyleSheet("""
            QPushButton {
                background-color: #00d4aa;
                color: #1a1a1a;
                border: none;
                padding: 8px 20px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover { background-color: #00b894; }
            QPushButton:pressed { background-color: #009e80; }
            QPushButton:disabled { background-color: #333333; color: #666666; }
        """)

    def _apply_stop_style(self) -> None:
        self._toggle_btn.setStyleSheet("""
            QPushButton {
                background-color: #ff4444;
                color: #ffffff;
                border: none;
                padding: 8px 20px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover { background-color: #e63333; }
            QPushButton:pressed { background-color: #cc2222; }
            QPushButton:disabled { background-color: #333333; color: #666666; }
        """)

    def _apply_loading_style(self) -> None:
        self._toggle_btn.setStyleSheet("""
            QPushButton {
                background-color: #f0ad4e;
                color: #1a1a1a;
                border: none;
                padding: 8px 20px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:disabled { background-color: #333333; color: #666666; }
        """)

    def _on_toggle_clicked(self) -> None:
        if self._loading:
            return
        self.camera_toggle_requested.emit()

    def populate_cameras(self, cameras: list[dict[str, str | int]]) -> None:
        self._camera_combo.blockSignals(True)
        self._camera_combo.clear()
        self._camera_combo.addItem("Select a camera...")
        self._camera_combo.setItemData(0, -1)
        for i, cam in enumerate(cameras, start=1):
            self._camera_combo.addItem(str(cam["name"]))
            self._camera_combo.setItemData(i, cam["index"])
        self._camera_combo.blockSignals(False)

    def get_selected_camera_index(self) -> int:
        return self._camera_combo.currentData()

    def get_stream_url(self) -> str:
        return self._camera_url.text().strip()

    def _on_camera_changed(self, index: int) -> None:
        if index > 0:
            cam_index = self._camera_combo.currentData()
            if cam_index is not None and cam_index >= 0:
                self.camera_switch_requested.emit(cam_index)

    def update_frame(self, frame: np.ndarray) -> None:
        self._video_widget.update_frame(frame)

    def update_stats(
        self,
        fps: float = 0.0,
        persons: int = 0,
        waste: int = 0,
        dustbins: int = 0,
        device: str = "CPU",
    ) -> None:
        self._info_panel.update_stats(fps, persons, waste, dustbins, device)

    def add_alert(self, message: str) -> None:
        from datetime import datetime

        timestamp = datetime.now().strftime("%H:%M:%S")
        self._alerts_text.append(f"[{timestamp}] {message}")

    def set_camera_active(self, active: bool) -> None:
        self._camera_active = active
        self._loading = False
        self._camera_combo.setEnabled(not active)
        self._camera_url.setEnabled(not active)
        if active:
            self._toggle_btn.setText("Stop Camera")
            self._apply_stop_style()
            self._toggle_btn.setEnabled(True)
            self._status_label.setText("")
        else:
            self._toggle_btn.setText("Start Camera")
            self._apply_start_style()
            self._toggle_btn.setEnabled(True)
            self._status_label.setText("")

    def set_loading(self, loading: bool, message: str = "") -> None:
        self._loading = loading
        if loading:
            self._toggle_btn.setText("Starting...")
            self._apply_loading_style()
            self._toggle_btn.setEnabled(True)
            self._camera_combo.setEnabled(False)
            self._camera_url.setEnabled(False)
            self._status_label.setText(message)
            self._status_label.setStyleSheet("color: #f0ad4e; font-size: 12px;")
        else:
            self._status_label.setText("")
            self._status_label.setStyleSheet("color: #888888; font-size: 12px;")
