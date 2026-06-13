"""Main application window for CivicSense.

Coordinates all pages, navigation, and application lifecycle.
Provides the primary user interface with sidebar navigation.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QCloseEvent
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from civicsense.core.config import get_config
from civicsense.core.logging import get_logger
from civicsense.gui.analytics_page import AnalyticsPage
from civicsense.gui.dashboard_page import DashboardPage
from civicsense.gui.incident_page import IncidentManagerPage
from civicsense.gui.live_monitor_page import LiveMonitorPage
from civicsense.gui.settings_page import SettingsPage

logger = get_logger("app")


class NavButton(QPushButton):
    """Styled navigation button for the sidebar."""

    def __init__(self, text: str, parent: QWidget | None = None) -> None:
        """Initialize the navigation button.

        Args:
            text: Button label text.
            parent: Parent widget.
        """
        super().__init__(text, parent)
        self.setCheckable(True)
        self.setFixedHeight(44)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #888888;
                border: none;
                border-radius: 6px;
                padding: 0 16px;
                text-align: left;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QPushButton:checked {
                background-color: #00d4aa;
                color: #1a1a1a;
                font-weight: bold;
            }
        """)


class MainWindow(QMainWindow):
    """Main application window.

    Provides sidebar navigation between Dashboard, Live Monitor,
    Incident Manager, Analytics, and Settings pages.
    """

    def __init__(self) -> None:
        """Initialize the main window with all pages and navigation."""
        super().__init__()
        self.setWindowTitle("CivicSense - AI Littering Detection")
        self.setMinimumSize(1280, 800)
        self.resize(1440, 900)

        # Workers (created on demand)
        self._camera_worker = None
        self._ai_worker = None
        self._save_worker = None
        self._camera_service = None
        self._model_manager = None
        self._incident_service = None
        self._analytics_service = None
        self._incident_count = 0

        self._setup_menubar()
        self._setup_ui()
        self._connect_signals()
        self._apply_dark_theme()
        self._init_services()

    def _setup_menubar(self) -> None:
        """Configure the application menu bar."""
        menubar = self.menuBar()
        if menubar is None:
            return
        menubar.setStyleSheet("""
            QMenuBar {
                background-color: #1a1a1a;
                color: #ffffff;
                border-bottom: 1px solid #333333;
            }
            QMenuBar::item:selected { background-color: #2b2b2b; }
        """)

        file_menu = menubar.addMenu("File")
        if file_menu:
            exit_action = QAction("Exit", self)
            exit_action.setShortcut("Ctrl+Q")
            exit_action.triggered.connect(self.close)
            file_menu.addAction(exit_action)

        view_menu = menubar.addMenu("View")
        if view_menu:
            dashboard_action = QAction("Dashboard", self)
            dashboard_action.triggered.connect(lambda: self._navigate(0))
            view_menu.addAction(dashboard_action)

            monitor_action = QAction("Live Monitor", self)
            monitor_action.triggered.connect(lambda: self._navigate(1))
            view_menu.addAction(monitor_action)

        help_menu = menubar.addMenu("Help")
        if help_menu:
            about_action = QAction("About", self)
            about_action.triggered.connect(self._show_about)
            help_menu.addAction(about_action)

    def _setup_ui(self) -> None:
        """Build the main window layout with sidebar and content area."""
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self._nav_buttons: list[NavButton] = []
        self._pages = QStackedWidget()
        self._pages.setStyleSheet("background-color: #1e1e1e;")

        self._dashboard_page = DashboardPage()
        self._live_monitor_page = LiveMonitorPage()
        self._incident_page = IncidentManagerPage()
        self._analytics_page = AnalyticsPage()
        self._settings_page = SettingsPage()

        self._pages.addWidget(self._dashboard_page)
        self._pages.addWidget(self._live_monitor_page)
        self._pages.addWidget(self._incident_page)
        self._pages.addWidget(self._analytics_page)
        self._pages.addWidget(self._settings_page)

        sidebar = self._create_sidebar()
        main_layout.addWidget(sidebar)

        main_layout.addWidget(self._pages, stretch=1)

    def _connect_signals(self) -> None:
        """Connect page signals to handlers."""
        self._settings_page.settings_saved.connect(self._on_settings_saved)
        self._live_monitor_page.camera_toggle_requested.connect(self._on_camera_toggle)

    def _init_services(self) -> None:
        """Initialize shared services and populate camera list."""
        from civicsense.database.engine import init_database
        from civicsense.services.analytics_service import AnalyticsService
        from civicsense.services.camera_service import CameraService
        from civicsense.services.incident_service import IncidentService

        self._camera_service = CameraService()
        cameras = CameraService.enumerate_cameras()
        self._live_monitor_page.populate_cameras(cameras)

        try:
            init_database()
            self._incident_service = IncidentService()
            self._analytics_service = AnalyticsService()
            self._dashboard_page.update_status(
                cameras_active=False,
                ai_loaded=False,
                gpu_available=False,
                db_connected=True,
            )
            self._refresh_dashboard()
            self._refresh_analytics()
        except Exception as e:
            logger.error(f"Service init failed: {e}", module="app")

    def _create_sidebar(self) -> QWidget:
        """Create the navigation sidebar.

        Returns:
            The configured sidebar widget.
        """
        sidebar = QWidget()
        sidebar.setFixedWidth(220)
        sidebar.setStyleSheet("""
            QWidget {
                background-color: #1a1a1a;
                border-right: 1px solid #333333;
            }
        """)

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(12, 16, 12, 16)
        layout.setSpacing(4)

        logo = QLabel("CivicSense")
        logo.setStyleSheet(
            "font-size: 20px; font-weight: bold; color: #00d4aa; padding: 8px 0 16px 0;"
        )
        layout.addWidget(logo)

        pages = [
            ("Dashboard", 0),
            ("Live Monitor", 1),
            ("Incidents", 2),
            ("Analytics", 3),
            ("Settings", 4),
        ]

        for name, idx in pages:
            btn = NavButton(name)
            btn.clicked.connect(lambda checked, i=idx: self._navigate(i))
            layout.addWidget(btn)
            self._nav_buttons.append(btn)

        layout.addStretch()

        version = QLabel("v0.1.0")
        version.setStyleSheet("color: #555555; font-size: 11px;")
        layout.addWidget(version)

        self._navigate(0)
        return sidebar

    def _navigate(self, page_index: int) -> None:
        """Navigate to a page by index.

        Args:
            page_index: The page index to display.
        """
        self._pages.setCurrentIndex(page_index)
        for i, btn in enumerate(self._nav_buttons):
            btn.setChecked(i == page_index)

    def _on_settings_saved(self, settings: dict) -> None:
        """Apply saved settings to the config singleton.

        Args:
            settings: Dictionary of setting key-value pairs.
        """
        config = get_config()

        # Apply AI settings
        ai_keys = (
            "detection_model",
            "pose_model",
            "confidence_threshold",
            "device",
            "image_size",
        )
        for key in ai_keys:
            full_key = f"ai.{key}"
            if full_key in settings:
                setattr(config.ai, key, settings[full_key])

        # Apply camera settings
        for key in ("source", "fps"):
            full_key = f"camera.{key}"
            if full_key in settings:
                setattr(config.camera, key, settings[full_key])

        # Apply storage settings
        from pathlib import Path

        for key in ("evidence_dir", "snapshots_dir", "clips_dir", "exports_dir"):
            full_key = f"storage.{key}"
            if full_key in settings:
                setattr(config.storage, key, Path(settings[full_key]))

        # Apply database settings
        if "database.url" in settings:
            config.database.url = settings["database.url"]

        logger.info("Settings applied to config", module="app")
        self.statusBar().showMessage("Settings saved", 3000)

    def _on_camera_toggle(self) -> None:
        """Toggle camera on/off."""
        if self._camera_worker is not None:
            self._stop_camera()
        else:
            self._start_camera()

    def _start_camera(self) -> None:
        """Start camera capture and AI processing."""
        config = get_config()
        source = self._live_monitor_page.get_selected_camera_index()

        if source < 0:
            url_text = self._live_monitor_page.get_stream_url()
            if not url_text:
                self._show_error(
                    "No Camera Selected",
                    "Please select a camera or enter a stream URL first.",
                )
                return
            source = url_text

        if self._camera_service is None:
            self._show_error("Camera Error", "Camera service not initialized.")
            return

        self._live_monitor_page.set_loading(True, "Opening camera...")

        try:
            from civicsense.ai.model_manager import ModelManager
            from civicsense.gui.workers.ai_worker import AIWorker
            from civicsense.gui.workers.camera_worker import CameraWorker

            camera_id = "live_monitor"
            self._camera_service.open(camera_id, source)
            self._live_monitor_page.add_alert(f"Camera opened: {source}")

            self._live_monitor_page.set_loading(True, "Loading AI models...")

            self._camera_worker = CameraWorker(self._camera_service, camera_id)
            self._camera_worker.frame_captured.connect(self._on_frame_captured)
            self._camera_worker.error_occurred.connect(self._on_worker_error)
            self._camera_worker.fps_updated.connect(
                lambda fps: self._live_monitor_page.update_stats(fps=fps)
            )

            self._model_manager = ModelManager()
            try:
                self._model_manager.load_models()
                self._ai_worker = AIWorker(self._model_manager)
                self._ai_worker.result_ready.connect(self._on_ai_result)
                self._ai_worker.error_occurred.connect(self._on_worker_error)
                self._ai_worker.start()
                device = config.ai.device
                self._live_monitor_page.update_stats(device=device.upper())
                self._live_monitor_page.add_alert("AI models loaded")
            except Exception as e:
                self._show_error("AI Model Error", f"Failed to load AI models:\n{e}")
                logger.error(f"Model load failed: {e}", module="app")

            self._camera_worker.start()
            self._live_monitor_page.set_camera_active(True)
            self._live_monitor_page.add_alert("Camera active")
            self.statusBar().showMessage("Camera active", 5000)

            self._dashboard_page.update_status(
                cameras_active=True,
                ai_loaded=self._model_manager is not None,
                gpu_available=config.ai.device != "cpu",
                db_connected=True,
            )

        except Exception as e:
            self._live_monitor_page.set_loading(False)
            self._show_error("Camera Error", f"Failed to start camera:\n{e}")
            logger.error(f"Camera start failed: {e}", module="app")

    def _stop_camera(self) -> None:
        """Stop camera capture and AI processing."""
        if self._camera_worker is not None:
            self._camera_worker.stop()
            self._camera_worker = None

        if self._ai_worker is not None:
            self._ai_worker.stop()
            self._ai_worker = None

        if self._camera_service is not None:
            self._camera_service.release_all()

        self._live_monitor_page.set_camera_active(False)
        self._live_monitor_page.add_alert("Camera stopped")
        self.statusBar().showMessage("Camera stopped", 5000)

        self._dashboard_page.update_status(
            cameras_active=False,
            ai_loaded=False,
            gpu_available=False,
            db_connected=True,
        )
        self._refresh_dashboard()
        self._refresh_analytics()

    def _on_frame_captured(self, frame) -> None:
        """Handle a new camera frame.

        Args:
            frame: BGR numpy array from camera.
        """
        self._live_monitor_page.update_frame(frame)

        # Queue frame for AI processing
        if self._ai_worker is not None:
            self._ai_worker.set_frame(frame)

    def _on_ai_result(self, result) -> None:
        """Handle AI inference result.

        Args:
            result: InferenceResult from InferenceWorker.
        """
        from civicsense.ai.inference_worker import InferenceResult

        if not isinstance(result, InferenceResult):
            return

        persons = len(result.tracked_persons)
        waste = len(result.tracked_waste)
        dustbins = len(result.tracked_dustbins)

        self._live_monitor_page.update_stats(
            persons=persons,
            waste=waste,
            dustbins=dustbins,
        )

        if result.littering_events:
            self._handle_littering_events(result)

    def _handle_littering_events(self, result) -> None:
        """Create incidents from detected littering events.

        Args:
            result: InferenceResult with littering_events.
        """
        from civicsense.dto.detection import IncidentDTO, IncidentStatus

        if self._incident_service is None:
            return

        for event in result.littering_events:
            if not event.is_littering:
                continue

            self._incident_count += 1
            dto = IncidentDTO(
                camera_id="live_monitor",
                camera_name="Live Camera",
                confidence=1.0,
                person_track_id=event.person_track_id,
                waste_type=event.waste_type,
                frame_width=result.frame.shape[1],
                frame_height=result.frame.shape[0],
                status=IncidentStatus.PENDING,
            )
            try:
                created = self._incident_service.create_incident(dto)
                self._show_alert(
                    f"Littering detected: {event.waste_type} "
                    f"(person #{event.person_track_id})"
                )
                self._refresh_dashboard()
                self._refresh_analytics()
                logger.info(
                    f"Incident created: id={created.id}, type={event.waste_type}",
                    module="app",
                )
            except Exception as e:
                logger.error(f"Failed to create incident: {e}", module="app")

    def _refresh_dashboard(self) -> None:
        """Refresh dashboard stats from incident service."""
        if self._incident_service is None:
            return
        try:
            total = self._incident_service.get_total_count()
            today = self._incident_service.get_today_count()
            self._dashboard_page.update_stats(
                total=total,
                today=today,
            )
        except Exception as e:
            logger.error(f"Dashboard refresh failed: {e}", module="app")

    def _refresh_analytics(self) -> None:
        """Refresh analytics charts from analytics service."""
        if self._analytics_service is None:
            return
        try:
            trend = self._analytics_service.get_trend_data(days=30)
            self._analytics_page.update_trend(trend)

            waste_dist = self._analytics_service.get_waste_type_statistics()
            self._analytics_page.update_waste_distribution(waste_dist)

            camera_stats = self._analytics_service.get_camera_statistics()
            self._analytics_page.update_camera_stats(camera_stats)
        except Exception as e:
            logger.error(f"Analytics refresh failed: {e}", module="app")

    def _on_worker_error(self, message: str) -> None:
        """Handle error from any worker thread.

        Args:
            message: Error description string.
        """
        self._show_alert(f"ERROR: {message}")
        logger.error(f"Worker error: {message}", module="app")

    def _show_error(self, title: str, message: str) -> None:
        """Display an error dialog to the user.

        Args:
            title: Dialog title.
            message: Error description.
        """
        QMessageBox.critical(self, title, message)

    def _show_alert(self, message: str) -> None:
        """Add an alert to the live monitor page.

        Args:
            message: Alert text.
        """
        self._live_monitor_page.add_alert(message)

    def _show_about(self) -> None:
        """Show the About dialog."""
        QMessageBox.about(
            self,
            "About CivicSense",
            "CivicSense - AI-Powered Littering Detection\n"
            "Version 0.1.0\n\n"
            "Detects littering events in real-time using\n"
            "YOLO26 object detection and pose estimation.",
        )

    def _apply_dark_theme(self) -> None:
        """Apply the dark theme stylesheet to the application."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QToolTip {
                background-color: #2b2b2b;
                color: #ffffff;
                border: 1px solid #444444;
                padding: 4px;
                border-radius: 4px;
            }
            QScrollBar:vertical {
                background-color: #1a1a1a;
                width: 10px;
                border: none;
                margin: 2px;
            }
            QScrollBar::handle:vertical {
                background-color: #444444;
                border-radius: 5px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #555555;
            }
            QScrollBar::handle:vertical:pressed {
                background-color: #00d4aa;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none;
            }
            QStatusBar {
                background-color: #1a1a1a;
                color: #888888;
                border-top: 1px solid #333333;
            }
            QComboBox QAbstractItemView {
                background-color: #2b2b2b;
                color: #ffffff;
                selection-background-color: #00d4aa;
                selection-color: #1a1a1a;
                border: 1px solid #444444;
            }
        """)
        self.statusBar().showMessage("Ready")

    @property
    def dashboard(self) -> DashboardPage:
        """Return the dashboard page instance."""
        return self._dashboard_page

    @property
    def live_monitor(self) -> LiveMonitorPage:
        """Return the live monitor page instance."""
        return self._live_monitor_page

    @property
    def incidents(self) -> IncidentManagerPage:
        """Return the incident manager page instance."""
        return self._incident_page

    @property
    def analytics(self) -> AnalyticsPage:
        """Return the analytics page instance."""
        return self._analytics_page

    @property
    def settings(self) -> SettingsPage:
        """Return the settings page instance."""
        return self._settings_page

    def closeEvent(self, event: QCloseEvent) -> None:
        """Clean up threads on window close."""
        if self._camera_worker is not None:
            self._camera_worker.stop()
            self._camera_worker = None
        if self._ai_worker is not None:
            self._ai_worker.stop()
            self._ai_worker = None
        if self._camera_service is not None:
            self._camera_service.release_all()
        event.accept()
