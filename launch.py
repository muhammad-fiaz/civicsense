"""CivicSense Launch Entry Point.

Unified entry point for running CivicSense via CLI, GUI, or API server.
Supports image detection, video detection, live camera, and web API.
"""

from __future__ import annotations

import sys
import argparse


def main() -> None:
    """Parse arguments and launch the appropriate CivicSense mode."""
    parser = argparse.ArgumentParser(
        prog="civicsense",
        description="CivicSense - AI-Powered Littering Detection Platform",
    )
    subparsers = parser.add_subparsers(dest="command", help="Mode to run")

    # CLI mode - detect images/videos
    cli_parser = subparsers.add_parser(
        "detect", help="Detect littering in image/video/camera"
    )
    cli_parser.add_argument(
        "source", help="Image path, video path, 'camera', or RTSP/HTTP URL"
    )
    cli_parser.add_argument(
        "--save", help="Save annotated output to path", default=None
    )
    cli_parser.add_argument(
        "--conf", type=float, default=0.15, help="Confidence threshold"
    )
    cli_parser.add_argument(
        "--no-gui", action="store_true", help="Disable OpenCV window display"
    )

    # GUI mode
    gui_parser = subparsers.add_parser("gui", help="Launch the PySide6 GUI application")
    gui_parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    # API mode
    api_parser = subparsers.add_parser("api", help="Launch the FastAPI server")
    api_parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    api_parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    api_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()

    if args.command == "detect":
        _run_detect(args)
    elif args.command == "gui":
        _run_gui(args)
    elif args.command == "api":
        _run_api(args)
    else:
        parser.print_help()


def _run_detect(args: argparse.Namespace) -> None:
    """Run detection in CLI mode."""
    from civicsense.cli.detector import run_detection

    run_detection(
        source=args.source,
        save_path=args.save,
        confidence=args.conf,
        show_window=not args.no_gui,
    )


def _run_gui(args: argparse.Namespace) -> None:
    """Launch the PySide6 GUI."""
    from civicsense.core.logging import setup_logging

    setup_logging()

    from PySide6.QtCore import Qt
    from PySide6.QtGui import QFont
    from PySide6.QtWidgets import QApplication

    QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("CivicSense")
    app.setApplicationVersion("0.1.0")

    fallback_font = QFont()
    fallback_font.setPointSize(10)
    app.setFont(fallback_font)

    from civicsense.gui.main_window import MainWindow

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


def _run_api(args: argparse.Namespace) -> None:
    """Launch the FastAPI server."""
    from civicsense.core.logging import setup_logging

    setup_logging()

    import uvicorn
    from civicsense.api.app import create_app

    app = create_app()
    print(f"Starting CivicSense API server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
