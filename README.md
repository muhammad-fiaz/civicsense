<div align="center">

# CivicSense

**AI-Powered Littering Detection Platform**

*Real-time monitoring of video feeds to detect and flag littering incidents*

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![YOLO26](https://img.shields.io/badge/YOLO26-Real--Time-Red.svg)](https://github.com/ultralytics/ultralytics)
[![UV](https://img.shields.io/badge/UV-Package-Manager-Purple.svg)](https://docs.astral.sh/uv/)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-Green.svg)](https://fastapi.tiangolo.com/)
[![PySide6](https://img.shields.io/badge/PySide6-GUI-Blue.svg)](https://doc.qt.io/qtforpython-6/)
[![Loguru](https://img.shields.io/badge/Loguru-Logging-Orange.svg)](https://github.com/Delgan/loguru)
[![Ruff](https://img.shields.io/badge/Ruff-Linting-Cyan.svg)](https://docs.astral.sh/ruff/)
[![Tests](https://img.shields.io/badge/Tests-72%20Passed-brightgreen.svg)]()

</div>

---

## Overview

CivicSense monitors video feeds in real-time and detects when people throw waste onto the ground instead of disposing of it in a dustbin. Built with YOLO26 object detection, pose estimation, multi-object tracking, and comprehensive reporting.

## Features

- **Real-time littering detection** using YOLO26 object detection
- **Human pose estimation** for hand-waste proximity analysis
- **ByteTrack multi-object tracking** with stable IDs across frames
- **Comprehensive ground object identification** - detects all waste types on the ground
- **Detailed waste classification** - identifies bottles, cups, food wrappers, bags, and 30+ COCO classes
- **Incident management** with review workflow (pending/approved/rejected)
- **Analytics dashboard** with daily, weekly, and monthly reports
- **Evidence storage** - snapshots, annotated frames, video clips
- **CSV and JSON data export** for external reporting
- **SQLite database** with SQLAlchemy ORM and Alembic migrations
- **Rich CLI output** with progress bars, styled panels, and formatted tables
- **Structured logging** with Loguru (app, AI, database, error, audit logs)
- **Three interface modes**: CLI, GUI (PySide6), and REST API (FastAPI)

## Quick Start

### Installation

```bash
git clone https://github.com/muhammad-fiaz/civicsense.git
cd civicsense
uv sync
```

### Usage

**CLI - Detect in image/video/camera:**

```bash
# Image detection
uv run launch.py detect photo.jpg
uv run launch.py detect photo.avif --save output.jpg

# Video detection
uv run launch.py detect video.mp4
uv run launch.py detect video.mp4 --save output.mp4 --conf 0.3

# Live camera
uv run launch.py detect camera
uv run launch.py detect camera --no-gui

# RTSP/HTTP stream
uv run launch.py detect rtsp://192.168.1.100:554/stream
uv run launch.py detect http://192.168.1.100:8080/video
```

**GUI - Launch desktop application:**

```bash
uv run launch.py gui
```

**API - Launch FastAPI server:**

```bash
uv run launch.py api --port 8000
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/detect/image` | Detect littering in uploaded image |
| POST | `/detect/video` | Detect littering in uploaded video |
| GET | `/health` | Health check |
| GET | `/docs` | Interactive API documentation |

## Detection Capabilities

### Waste Types Detected

CivicSense identifies all COCO dataset classes that commonly appear as litter:

| Category | Objects |
|----------|---------|
| **Beverages** | bottle, wine glass, cup |
| **Food** | banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake |
| **Containers** | bowl, fork, knife, spoon |
| **Personal Items** | backpack, umbrella, handbag, suitcase |
| **Electronics** | cell phone |
| **Miscellaneous** | book, scissors, teddy bear, sports ball, kite, frisbee |
| **Potential Dustbins** | potted plant, vase, sink, toilet |

### Classification Logic

| Condition | Classification | Details |
|-----------|---------------|---------|
| Waste + near dustbin | VALID | Correctly disposed |
| Waste in hand + throwing + no dustbin | FLAGGED | Active littering |
| Waste held + no dustbin | FLAGGED | Likely littering |
| Waste on ground + no dustbin | FLAGGED | Litter on ground |
| Throwing motion + dustbin not targeted | FLAGGED | Potential littering |
| Ground objects near person | INCONCLUSIVE | Needs more context |
| Ground litter, no person | GROUND LITTER | Pre-existing litter |

### Reporting

Each detection provides:
- **Classification** (FLAGGED/VALID/GROUND LITTER/INCONCLUSIVE)
- **Detailed breakdown** of all detected objects
- **Person count** with pose keypoints (wrists, elbows)
- **Waste count** with per-type breakdown
- **Ground object count** with distance to nearest person
- **Dustbin count** with proximity analysis
- **Timestamped screenshots** saved automatically

## Project Structure

```
civicsense/
  launch.py              # Unified entry point (CLI, GUI, API)
  pyproject.toml         # Project configuration
  LICENSE                # GPL v3 License
  CONTRIBUTING.md        # Contribution guidelines
  civicsense/            # Library package
    __init__.py
    config/              # TOML configuration
    core/                # Configuration, logging, exceptions, constants
    database/            # SQLAlchemy models, repositories, migrations
    ai/                  # YOLO26 detection, pose estimation, tracking
    services/            # Business logic (incidents, evidence, analytics, export)
    gui/                 # PySide6 desktop application (5 pages)
    cli/                 # Command-line interface (detector, classifier)
    api/                 # FastAPI REST API
    dto/                 # Data Transfer Objects
    interfaces/          # Runtime-checkable protocols
    events/              # Event bus (pub-sub)
    utils/               # Helper functions
  weights/               # Model weights (yolo26n.pt, yolo26n-pose.pt)
  detected/              # Detection outputs (screenshots, clips, annotated)
  tests/                 # 72 tests (unit, ai, database)
  logs/                  # Application logs
```

## Technology Stack

| Component | Technology |
|-----------|------------|
| **Detection** | YOLO26 (Ultralytics) |
| **Pose Estimation** | YOLO26-Pose |
| **Tracking** | ByteTrack |
| **Database** | SQLite + SQLAlchemy 2.x + Alembic |
| **Configuration** | TOML + Pydantic Settings |
| **Logging** | Loguru + Rich |
| **CLI** | Rich (panels, tables, progress) |
| **GUI** | PySide6 (Qt) |
| **API** | FastAPI + Uvicorn |
| **Testing** | pytest |
| **Linting** | Ruff |
| **Package Manager** | UV |

## Configuration

All detection thresholds, waste categories, and storage paths are configured in `civicsense/config/detection.toml`.

Environment variables (`.env`):

```env
CIVICSENSE_LOG_LEVEL=INFO
CIVICSENSE_AI_DEVICE=auto
CIVICSENSE_AI_CONFIDENCE_THRESHOLD=0.15
```

## Requirements

- Python 3.10+
- UV package manager
- Camera (for live detection)
- GPU recommended (CUDA support via PyTorch)

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2026 Muhammad Fiaz
