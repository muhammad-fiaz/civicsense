"""API routes for CivicSense detection endpoints."""

from __future__ import annotations

import io
import os
import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from civicsense.cli.classifier import FrameResult, classify_frame

router = APIRouter()

_models_loaded = False
_detector = None
_pose_detector = None


def _ensure_models() -> None:
    """Lazy-load AI models on first request."""
    global _models_loaded, _detector, _pose_detector
    if _models_loaded:
        return

    from civicsense.ai.detector import YOLODetector
    from civicsense.ai.pose_detector import YOLOPoseDetector
    from civicsense.core.config import get_config

    config = get_config()
    _detector = YOLODetector()
    _detector.load(config.ai.detection_model, config.ai.device)
    _pose_detector = YOLOPoseDetector()
    _pose_detector.load(config.ai.pose_model, config.ai.device)
    _models_loaded = True


def _read_image(file_bytes: bytes) -> np.ndarray | None:
    """Convert uploaded bytes to OpenCV image.

    Args:
        file_bytes: Raw image bytes.

    Returns:
        BGR numpy array or None.
    """
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is not None:
        return img

    # Try PIL for AVIF/WEBP
    try:
        from PIL import Image

        pil_img = Image.open(io.BytesIO(file_bytes))
        rgb = np.array(pil_img.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        return None


@router.post("/detect/image")
async def detect_image(file: UploadFile = File(...)) -> JSONResponse:  # noqa: B008
    """Detect littering in an uploaded image.

    Accepts: jpg, jpeg, png, bmp, tiff, webp, avif
    """
    _ensure_models()
    assert _detector is not None
    assert _pose_detector is not None

    contents = await file.read()
    frame = _read_image(contents)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    result = classify_frame(frame, _detector, _pose_detector, frame_idx=0)
    annotated = _annotate_frame(frame, result)

    # Save to detected/screenshots/
    screenshots_dir = Path("detected/screenshots")
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = screenshots_dir / f"detection_{ts}.jpg"
    cv2.imwrite(str(out_path), annotated)

    return JSONResponse(
        content={
            "classification": result.classification,
            "details": result.details,
            "waste_count": result.waste_count,
            "ground_count": result.ground_count,
            "persons": len(result.persons),
            "dustbins": len(result.dustbins),
            "output_path": str(out_path),
        }
    )


@router.post("/detect/video")
async def detect_video(
    file: UploadFile = File(...),  # noqa: B008
    max_frames: int = 100,
) -> JSONResponse:
    """Detect littering in an uploaded video.

    Processes up to max_frames and returns aggregate results.
    """
    _ensure_models()
    assert _detector is not None
    assert _pose_detector is not None

    suffix = Path(file.filename or "video.mp4").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video")

        flagged_frames = 0
        total_frames = 0
        results = []

        while total_frames < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            result = classify_frame(
                frame, _detector, _pose_detector, frame_idx=total_frames
            )
            if "FLAGGED" in result.classification:
                flagged_frames += 1
                results.append(
                    {
                        "frame": total_frames,
                        "classification": result.classification,
                        "waste_count": result.waste_count,
                    }
                )
            total_frames += 1

        cap.release()

        return JSONResponse(
            content={
                "total_frames": total_frames,
                "flagged_frames": flagged_frames,
                "results": results,
            }
        )
    finally:
        os.unlink(tmp_path)


@router.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "civicsense"}


def _annotate_frame(frame: np.ndarray, result: FrameResult) -> np.ndarray:
    """Annotate frame with detections — delegates to shared CLI annotate."""
    from civicsense.cli.detector import _annotate

    return _annotate(frame, result)
