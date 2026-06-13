"""Settings page for CivicSense.

Provides configuration UI for cameras, AI models, confidence thresholds,
storage paths, database settings, and theme selection.
All defaults are loaded from the live config singleton.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from civicsense.core.config import get_config
from civicsense.core.logging import get_logger

logger = get_logger("app")


class SettingsPage(QWidget):
    """Application settings configuration page."""

    settings_saved = Signal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the settings page.

        Args:
            parent: Parent widget.
        """
        super().__init__(parent)
        self._settings: dict[str, Any] = {}
        self._camera_list: list[dict[str, str | int]] = []
        self._setup_ui()
        self._load_defaults_from_config()

    def _setup_ui(self) -> None:
        """Build the settings layout."""
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(16)
        main_layout.setContentsMargins(24, 24, 24, 24)

        header = QLabel("Settings")
        header.setStyleSheet("font-size: 24px; font-weight: bold; color: #ffffff;")
        main_layout.addWidget(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(16)

        ai_group = self._create_ai_settings()
        layout.addWidget(ai_group)

        camera_group = self._create_camera_settings()
        layout.addWidget(camera_group)

        storage_group = self._create_storage_settings()
        layout.addWidget(storage_group)

        db_group = self._create_database_settings()
        layout.addWidget(db_group)

        gui_group = self._create_gui_settings()
        layout.addWidget(gui_group)

        layout.addStretch()

        save_btn = QPushButton("Save Settings")
        save_btn.setStyleSheet("""
            QPushButton {
                background-color: #00d4aa;
                color: #1a1a1a;
                border: none;
                padding: 10px 24px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover { background-color: #00b894; }
        """)
        save_btn.clicked.connect(self._on_save)
        layout.addWidget(save_btn)

        scroll.setWidget(container)
        main_layout.addWidget(scroll)

    def _create_group(self, title: str) -> tuple[QGroupBox, QFormLayout]:
        """Create a styled settings group box.

        Args:
            title: Group box title.

        Returns:
            Tuple of (group_box, form_layout).
        """
        group = QGroupBox(title)
        group.setStyleSheet("""
            QGroupBox {
                font-size: 16px;
                font-weight: bold;
                color: #00d4aa;
                border: 1px solid #444444;
                border-radius: 8px;
                padding: 16px;
                margin-top: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 16px;
                padding: 0 8px;
            }
        """)
        layout = QFormLayout(group)
        layout.setSpacing(12)
        return group, layout

    def _create_ai_settings(self) -> QGroupBox:
        """Create the AI model settings group.

        Returns:
            Configured QGroupBox.
        """
        group, layout = self._create_group("AI Model Settings")

        self._det_model = QComboBox()
        self._det_model.addItems(
            [
                "yolo26n.pt",
                "yolo26s.pt",
                "yolo26m.pt",
                "yolo26l.pt",
                "yolo26x.pt",
            ]
        )
        layout.addRow("Detection Model:", self._det_model)

        self._pose_model = QComboBox()
        self._pose_model.addItems(
            [
                "yolo26n-pose.pt",
                "yolo26s-pose.pt",
                "yolo26m-pose.pt",
                "yolo26l-pose.pt",
                "yolo26x-pose.pt",
            ]
        )
        layout.addRow("Pose Model:", self._pose_model)

        self._confidence_slider = QSlider(Qt.Orientation.Horizontal)
        self._confidence_slider.setRange(0, 100)
        self._conf_label = QLabel("0.15")
        self._confidence_slider.valueChanged.connect(
            lambda v: self._conf_label.setText(f"{v / 100:.2f}")
        )
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(self._confidence_slider)
        conf_layout.addWidget(self._conf_label)
        layout.addRow("Confidence:", conf_layout)

        self._device_combo = QComboBox()
        self._device_combo.addItems(["auto", "cpu", "cuda", "cuda:0"])
        layout.addRow("Device:", self._device_combo)

        self._image_size = QSpinBox()
        self._image_size.setRange(320, 1920)
        self._image_size.setSingleStep(32)
        layout.addRow("Image Size:", self._image_size)

        return group

    def _create_camera_settings(self) -> QGroupBox:
        """Create the camera settings group with camera enumeration.

        Returns:
            Configured QGroupBox.
        """
        group, layout = self._create_group("Camera Settings")

        self._camera_combo = QComboBox()
        self._camera_combo.addItem("None (select a camera)")
        self._refresh_cameras_btn = QPushButton("Refresh Cameras")
        self._refresh_cameras_btn.setStyleSheet("""
            QPushButton {
                background-color: #333333;
                color: #ffffff;
                border: 1px solid #555555;
                padding: 4px 12px;
                border-radius: 4px;
                font-size: 12px;
            }
            QPushButton:hover { background-color: #444444; }
        """)
        self._refresh_cameras_btn.clicked.connect(self._refresh_cameras)

        cam_layout = QHBoxLayout()
        cam_layout.addWidget(self._camera_combo, stretch=1)
        cam_layout.addWidget(self._refresh_cameras_btn)
        layout.addRow("Camera:", cam_layout)

        self._camera_url = QLineEdit()
        self._camera_url.setPlaceholderText("Or enter RTSP/HTTP stream URL")
        layout.addRow("Stream URL:", self._camera_url)

        self._cam_fps = QSpinBox()
        self._cam_fps.setRange(1, 120)
        layout.addRow("FPS:", self._cam_fps)

        return group

    def _create_storage_settings(self) -> QGroupBox:
        """Create the storage settings group.

        Returns:
            Configured QGroupBox.
        """
        group, layout = self._create_group("Storage Settings")

        self._evidence_path = QLineEdit()
        layout.addRow("Evidence Path:", self._evidence_path)

        self._snapshots_path = QLineEdit()
        layout.addRow("Snapshots Path:", self._snapshots_path)

        self._clips_path = QLineEdit()
        layout.addRow("Clips Path:", self._clips_path)

        self._exports_path = QLineEdit()
        layout.addRow("Exports Path:", self._exports_path)

        return group

    def _create_database_settings(self) -> QGroupBox:
        """Create the database settings group.

        Returns:
            Configured QGroupBox.
        """
        group, layout = self._create_group("Database Settings")

        self._db_url = QLineEdit()
        layout.addRow("Database URL:", self._db_url)

        return group

    def _create_gui_settings(self) -> QGroupBox:
        """Create the GUI settings group.

        Returns:
            Configured QGroupBox.
        """
        group, layout = self._create_group("Interface Settings")

        self._theme_combo = QComboBox()
        self._theme_combo.addItems(["dark", "light", "system"])
        layout.addRow("Theme:", self._theme_combo)

        return group

    def _load_defaults_from_config(self) -> None:
        """Load all widget defaults from the current config singleton."""
        config = get_config()

        # AI settings
        idx = self._det_model.findText(config.ai.detection_model)
        if idx >= 0:
            self._det_model.setCurrentIndex(idx)

        idx = self._pose_model.findText(config.ai.pose_model)
        if idx >= 0:
            self._pose_model.setCurrentIndex(idx)

        self._confidence_slider.setValue(int(config.ai.confidence_threshold * 100))

        idx = self._device_combo.findText(config.ai.device)
        if idx >= 0:
            self._device_combo.setCurrentIndex(idx)

        self._image_size.setValue(config.ai.image_size)

        # Camera settings
        self._cam_fps.setValue(config.camera.fps)
        if isinstance(config.camera.source, int):
            cam_idx = self._camera_combo.findText(
                f"Camera {config.camera.source}", Qt.MatchFlag.MatchContains
            )
            if cam_idx >= 0:
                self._camera_combo.setCurrentIndex(cam_idx)
        elif isinstance(config.camera.source, str) and config.camera.source.startswith(
            ("rtsp://", "http://")
        ):
            self._camera_url.setText(config.camera.source)

        # Storage settings
        self._evidence_path.setText(str(config.storage.evidence_dir))
        self._snapshots_path.setText(str(config.storage.snapshots_dir))
        self._clips_path.setText(str(config.storage.clips_dir))
        self._exports_path.setText(str(config.storage.exports_dir))

        # Database settings
        self._db_url.setText(config.database.url)

        # Theme
        idx = self._theme_combo.findText(config.theme.value)
        if idx >= 0:
            self._theme_combo.setCurrentIndex(idx)

        # Enumerate cameras
        self._refresh_cameras()

    def _refresh_cameras(self) -> None:
        """Scan for available cameras and populate the dropdown."""
        from civicsense.services.camera_service import CameraService

        self._camera_combo.clear()
        self._camera_combo.addItem("None (select a camera)")

        self._camera_list = CameraService.enumerate_cameras()
        for i, cam in enumerate(self._camera_list, start=1):
            self._camera_combo.addItem(str(cam["name"]))
            self._camera_combo.setItemData(i, cam["index"])

        if not self._camera_list:
            self._camera_combo.addItem("No cameras found")

        logger.info(f"Found {len(self._camera_list)} camera(s)", module="app")

    def _on_save(self) -> None:
        """Handle save button click — emit all current widget values."""
        # Determine camera source
        cam_url = self._camera_url.text().strip()
        if cam_url:
            camera_source: str | int = cam_url
        elif self._camera_combo.currentIndex() > 0:
            camera_source = self._camera_combo.currentData()
        else:
            camera_source = 0

        self._settings = {
            "ai.detection_model": self._det_model.currentText(),
            "ai.pose_model": self._pose_model.currentText(),
            "ai.confidence_threshold": self._confidence_slider.value() / 100,
            "ai.device": self._device_combo.currentText(),
            "ai.image_size": self._image_size.value(),
            "camera.source": camera_source,
            "camera.fps": self._cam_fps.value(),
            "storage.evidence_dir": self._evidence_path.text(),
            "storage.snapshots_dir": self._snapshots_path.text(),
            "storage.clips_dir": self._clips_path.text(),
            "storage.exports_dir": self._exports_path.text(),
            "database.url": self._db_url.text(),
            "theme": self._theme_combo.currentText(),
        }
        self.settings_saved.emit(self._settings)

    def load_settings(self, settings: dict[str, Any]) -> None:
        """Load settings into the form widgets.

        Args:
            settings: Dictionary of settings values.
        """
        if "ai.detection_model" in settings:
            idx = self._det_model.findText(settings["ai.detection_model"])
            if idx >= 0:
                self._det_model.setCurrentIndex(idx)

        if "ai.confidence_threshold" in settings:
            self._confidence_slider.setValue(
                int(settings["ai.confidence_threshold"] * 100)
            )

        if "ai.device" in settings:
            idx = self._device_combo.findText(settings["ai.device"])
            if idx >= 0:
                self._device_combo.setCurrentIndex(idx)

        if "theme" in settings:
            idx = self._theme_combo.findText(settings["theme"])
            if idx >= 0:
                self._theme_combo.setCurrentIndex(idx)
