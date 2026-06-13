"""CLI detection module for CivicSense.

Handles image, video, and live camera detection from the command line
with screenshot capture, timestamped results, and rich progress display.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)
from rich.table import Table

from civicsense.ai.detector import YOLODetector
from civicsense.ai.pose_detector import YOLOPoseDetector
from civicsense.cli.classifier import FrameResult
from civicsense.core.config import get_config
from civicsense.core.logging import get_logger

logger = get_logger("app")
console = Console()


def run_detection(
    source: str,
    save_path: str | None = None,
    confidence: float | None = None,
    show_window: bool = True,
) -> None:
    """Run littering detection from CLI.

    Args:
        source: Image path, video path, 'camera', or stream URL.
        save_path: Optional path to save annotated output.
        confidence: Detection confidence threshold (None = use config default).
        show_window: Whether to display OpenCV window.
    """
    from civicsense.ai.detector import YOLODetector
    from civicsense.ai.pose_detector import YOLOPoseDetector

    config = get_config()
    if confidence is None:
        confidence = config.ai.confidence_threshold

    console.print(
        Panel.fit(
            "[bold green]CivicSense[/bold green] - AI-Powered Littering Detection",
            border_style="green",
        )
    )

    config = get_config()

    # Load models with progress
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading AI models...", total=None)

        detector = YOLODetector()
        detector.load(config.ai.detection_model, config.ai.device)
        progress.update(task, description="[green]Detection model loaded")

        pose_detector = YOLOPoseDetector()
        pose_detector.load(config.ai.pose_model, config.ai.device)
        progress.update(task, description="[green]Pose model loaded")

        progress.update(task, description="[bold green]All models ready", completed=1)

    logger.info("Models loaded successfully", module="app")

    # Determine source type
    camera_aliases = {"camera", "cam", "webcam"}
    rtsp_prefixes = ("rtsp://", "rtmp://", "http://", "https://")
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".avif"}
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}

    # Output directories
    detected_dir = Path("detected")
    screenshots_dir = detected_dir / "screenshots"
    clips_dir = detected_dir / "clips"
    annotated_dir = detected_dir / "annotated"
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)

    if source.lower() in camera_aliases or source.startswith(rtsp_prefixes):
        _run_live_detection(
            source=source if source.lower() not in camera_aliases else 0,
            detector=detector,
            pose_detector=pose_detector,
            save_path=save_path,
            show_window=show_window,
            screenshots_dir=screenshots_dir,
            clips_dir=clips_dir,
        )
    else:
        if not os.path.exists(source):
            console.print(f"[red]ERROR:[/red] File not found: {source}")
            sys.exit(1)

        ext = Path(source).suffix.lower()
        if ext in image_exts:
            _run_image_detection(
                source, detector, pose_detector, save_path, screenshots_dir
            )
        elif ext in video_exts:
            _run_video_detection(
                source,
                detector,
                pose_detector,
                save_path,
                show_window,
                screenshots_dir,
                clips_dir,
            )
        else:
            console.print(f"[red]ERROR:[/red] Unsupported file type: {ext}")
            sys.exit(1)


def _run_image_detection(
    source: str,
    detector: YOLODetector,
    pose_detector: YOLOPoseDetector,
    save_path: str | None,
    screenshots_dir: Path,
) -> None:
    """Detect littering in a single image.

    Args:
        source: Image file path.
        detector: YOLODetector instance.
        pose_detector: YOLOPoseDetector instance.
        save_path: Optional output path.
        screenshots_dir: Directory for screenshots.
    """
    from civicsense.cli.classifier import classify_frame

    console.print(f"\n[bold]Processing:[/bold] {source}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Analyzing image...", total=None)

        frame = _load_image(source)
        if frame is None:
            progress.update(task, description="[red]Could not load image")
            return

        result = classify_frame(frame, detector, pose_detector, 0)
        annotated = _annotate(frame, result)
        progress.update(task, description="[green]Analysis complete", completed=1)

    # Save with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = f"{Path(source).stem}_{timestamp}.jpg"
    out_path = save_path or str(screenshots_dir / out_name)
    cv2.imwrite(out_path, annotated)

    _print_result(result, out_path)
    logger.info(f"Image detection complete: {result.classification}", module="app")


def _run_video_detection(
    source: str,
    detector: YOLODetector,
    pose_detector: YOLOPoseDetector,
    save_path: str | None,
    show_window: bool,
    screenshots_dir: Path,
    clips_dir: Path,
) -> None:
    """Detect littering in a video file with 10-second clip extraction.

    Processes the video, flags littering incidents, and saves 10-second
    clips around each flagged event as proof.

    Args:
        source: Video file path.
        detector: YOLODetector instance.
        pose_detector: YOLOPoseDetector instance.
        save_path: Optional output path.
        show_window: Whether to display window.
        screenshots_dir: Directory for screenshots.
        clips_dir: Directory for extracted clips.
    """
    from civicsense.cli.classifier import classify_frame
    from civicsense.core.constants import (
        CLIP_POST_EVENT_SECONDS,
        CLIP_PRE_EVENT_SECONDS,
    )

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        console.print(f"[red]ERROR:[/red] Cannot open video: {source}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    console.print(f"\n[bold]Video:[/bold] {source} ({w}x{h} @ {fps:.1f} FPS)")
    console.print(f"[dim]Total frames: {total_frames}[/dim]")

    writer = None
    if save_path:
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))

    frame_idx = 0
    flagged_count = 0
    valid_count = 0
    prev_class = ""

    # Clip extraction state
    clip_buffer: list[np.ndarray] = []
    clip_buffer_max = int(fps * (CLIP_PRE_EVENT_SECONDS + CLIP_POST_EVENT_SECONDS))
    flagged_frames_in_clip: list[int] = []
    recording_clip = False
    clip_start_frame = 0

    console.print("[dim]Press 'q' to quit, 's' to save screenshot[/dim]\n")

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        total = total_frames if total_frames > 0 else None
        task = progress.add_task("Processing frames...", total=total)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = classify_frame(frame, detector, pose_detector, frame_idx)
            annotated = _annotate(frame, result)

            if writer:
                writer.write(annotated)

            # Clip buffer management
            clip_buffer.append(annotated.copy())
            if len(clip_buffer) > clip_buffer_max:
                clip_buffer.pop(0)

            if "FLAGGED" in result.classification:
                if not recording_clip:
                    recording_clip = True
                    clip_start_frame = max(
                        0, frame_idx - int(fps * CLIP_PRE_EVENT_SECONDS)
                    )
                    flagged_frames_in_clip = []
                flagged_frames_in_clip.append(frame_idx)
                flagged_count += 1

                if flagged_count <= 5:
                    ts = datetime.now().strftime("%H%M%S")
                    ss_path = screenshots_dir / f"FLAGGED_{frame_idx}_{ts}.jpg"
                    cv2.imwrite(str(ss_path), annotated)
                    console.print(
                        f"  [bold red]FLAGGED screenshot saved:[/bold red] {ss_path}"
                    )
            elif recording_clip:
                # End of flagged event - save the clip
                if frame_idx - flagged_frames_in_clip[-1] > int(
                    fps * CLIP_POST_EVENT_SECONDS
                ):
                    _save_clip(clip_buffer, clips_dir, clip_start_frame, frame_idx, fps)
                    recording_clip = False
                    clip_buffer.clear()
                    flagged_frames_in_clip.clear()

            if show_window:
                cv2.imshow("CivicSense Detection", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("s"):
                    ts = datetime.now().strftime("%H%M%S")
                    ss_path = screenshots_dir / f"screenshot_{frame_idx}_{ts}.jpg"
                    cv2.imwrite(str(ss_path), annotated)
                    console.print(f"  [cyan]Screenshot saved:[/cyan] {ss_path}")

            if result.classification != prev_class:
                if "FLAGGED" in result.classification:
                    console.print(
                        f"  [red]Frame {frame_idx}: {result.classification}[/red]"
                    )
                elif "VALID" in result.classification:
                    console.print(
                        f"  [green]Frame {frame_idx}: {result.classification}[/green]"
                    )
                else:
                    console.print(
                        f"  [dim]Frame {frame_idx}: {result.classification}[/dim]"
                    )
                prev_class = result.classification

            if "VALID" in result.classification:
                valid_count += 1

            frame_idx += 1
            progress.advance(task)

    # Save any remaining clip
    if recording_clip and clip_buffer:
        _save_clip(clip_buffer, clips_dir, clip_start_frame, frame_idx, fps)

    cap.release()
    if writer:
        writer.release()
    if show_window:
        cv2.destroyAllWindows()

    _print_video_summary(frame_idx, flagged_count, valid_count)


def _run_live_detection(
    source: str | int,
    detector: YOLODetector,
    pose_detector: YOLOPoseDetector,
    save_path: str | None,
    show_window: bool,
    screenshots_dir: Path,
    clips_dir: Path,
) -> None:
    """Run live camera detection with clip extraction.

    Args:
        source: Camera index or stream URL.
        detector: YOLODetector instance.
        pose_detector: YOLOPoseDetector instance.
        save_path: Optional output path.
        show_window: Whether to display window.
        screenshots_dir: Directory for screenshots.
        clips_dir: Directory for extracted clips.
    """
    from civicsense.cli.classifier import classify_frame

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        console.print(f"[red]ERROR:[/red] Cannot open source: {source}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    console.print(f"\n[bold]Camera:[/bold] {w}x{h} @ {fps:.1f} FPS")
    console.print("[dim]Press 'q' to quit, 's' to save screenshot[/dim]\n")

    writer = None
    if save_path:
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))

    frame_idx = 0
    flagged_count = 0
    prev_class = ""

    # Clip buffer for live camera
    clip_buffer: list[np.ndarray] = []
    from civicsense.core.constants import (
        CLIP_POST_EVENT_SECONDS,
        CLIP_PRE_EVENT_SECONDS,
    )

    clip_buffer_max = int(fps * (CLIP_PRE_EVENT_SECONDS + CLIP_POST_EVENT_SECONDS))
    recording_clip = False
    clip_start_frame = 0
    last_flagged_frame = -1

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Monitoring...", total=None)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = classify_frame(frame, detector, pose_detector, frame_idx)
            annotated = _annotate(frame, result)

            # Add timestamp overlay
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(
                annotated,
                ts,
                (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            # Clip buffer management
            clip_buffer.append(annotated.copy())
            if len(clip_buffer) > clip_buffer_max:
                clip_buffer.pop(0)

            if writer:
                writer.write(annotated)

            if show_window:
                cv2.imshow("CivicSense Live", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("s"):
                    ts_file = datetime.now().strftime("%Y%m%d_%H%M%S")
                    ss_path = screenshots_dir / f"live_{frame_idx}_{ts_file}.jpg"
                    cv2.imwrite(str(ss_path), annotated)
                    console.print(f"  [cyan]Screenshot:[/cyan] {ss_path}")

            if "FLAGGED" in result.classification:
                flagged_count += 1
                last_flagged_frame = frame_idx
                if not recording_clip:
                    recording_clip = True
                    clip_start_frame = max(
                        0, frame_idx - int(fps * CLIP_PRE_EVENT_SECONDS)
                    )
                if flagged_count <= 10:
                    ts_file = datetime.now().strftime("%Y%m%d_%H%M%S")
                    ss_path = screenshots_dir / f"FLAGGED_{frame_idx}_{ts_file}.jpg"
                    cv2.imwrite(str(ss_path), annotated)
                    console.print(
                        f"  [bold red]FLAGGED:[/bold red] {result.details[:80]}"
                    )
                    console.print(f"     Saved: {ss_path}")
            elif recording_clip:
                if frame_idx - last_flagged_frame > int(fps * CLIP_POST_EVENT_SECONDS):
                    _save_clip(clip_buffer, clips_dir, clip_start_frame, frame_idx, fps)
                    recording_clip = False
                    clip_buffer.clear()

            if result.classification != prev_class:
                if "FLAGGED" in result.classification:
                    console.print(
                        f"  [red]Frame {frame_idx}: {result.classification}[/red]"
                    )
                elif "VALID" in result.classification:
                    console.print(
                        f"  [green]Frame {frame_idx}: {result.classification}[/green]"
                    )
                else:
                    console.print(
                        f"  [dim]Frame {frame_idx}: {result.classification}[/dim]"
                    )
                prev_class = result.classification

            progress.update(
                task, description=f"Frame {frame_idx} | Flagged: {flagged_count}"
            )
            frame_idx += 1

    # Save any remaining clip
    if recording_clip and clip_buffer:
        _save_clip(clip_buffer, clips_dir, clip_start_frame, frame_idx, fps)

    cap.release()
    if writer:
        writer.release()
    if show_window:
        cv2.destroyAllWindows()

    console.print(
        f"\n[bold]Session:[/bold] {frame_idx} frames, {flagged_count} FLAGGED"
    )
    logger.info(
        f"Live session complete: {frame_idx} frames, {flagged_count} flagged",
        module="app",
    )


def _save_clip(
    frames: list[np.ndarray],
    clips_dir: Path,
    start_frame: int,
    end_frame: int,
    fps: float,
) -> None:
    """Save a 10-second clip from buffered frames.

    Args:
        frames: List of annotated frames to include.
        clips_dir: Directory to save the clip.
        start_frame: Starting frame index.
        end_frame: Ending frame index.
        fps: Frames per second.
    """
    if not frames:
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    clip_path = clips_dir / f"clip_{start_frame}_{end_frame}_{ts}.mp4"

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(clip_path), fourcc, fps, (w, h))

    for frame in frames:
        writer.write(frame)
    writer.release()

    duration = len(frames) / fps
    console.print(
        f"  [yellow]Clip saved:[/yellow] {clip_path} ({duration:.1f}s, {len(frames)} frames)"
    )
    logger.info(
        f"Clip extracted: {clip_path} ({duration:.1f}s)",
        module="app",
    )


def _load_image(path: str) -> np.ndarray | None:
    """Load image supporting AVIF and other formats.

    Args:
        path: Image file path.

    Returns:
        BGR numpy array or None on failure.
    """
    ext = Path(path).suffix.lower()
    if ext in (".avif", ".webp", ".tiff", ".bmp"):
        try:
            from PIL import Image

            pil_img = Image.open(path)
            rgb = np.array(pil_img.convert("RGB"))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception:
            return None
    return cv2.imread(path)


def _annotate(frame: np.ndarray, result: FrameResult) -> np.ndarray:
    """Annotate frame with detections and classification.

    Args:
        frame: Input video frame.
        result: FrameResult with detection data.

    Returns:
        Annotated frame with bounding boxes and labels.
    """
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # Colors
    color_person = (0, 255, 0)
    color_waste = (0, 0, 255)
    color_dustbin = (255, 165, 0)
    color_valid = (0, 200, 0)
    color_flagged = (0, 0, 255)
    color_other = (0, 165, 255)
    color_ground = (128, 128, 0)

    # Draw persons
    for p in result.persons:
        pt1 = (int(p.bbox.x1), int(p.bbox.y1))
        pt2 = (int(p.bbox.x2), int(p.bbox.y2))
        cv2.rectangle(annotated, pt1, pt2, color_person, 2)
        cv2.putText(
            annotated,
            f"Person {p.confidence:.2f}",
            (pt1[0], pt1[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color_person,
            2,
        )
        for wrist in [p.left_wrist, p.right_wrist]:
            if wrist:
                cv2.circle(
                    annotated, (int(wrist[0]), int(wrist[1])), 6, (0, 255, 255), -1
                )

    # Draw waste
    for waste in result.waste_objects:
        pt1 = (int(waste.bbox.x1), int(waste.bbox.y1))
        pt2 = (int(waste.bbox.x2), int(waste.bbox.y2))
        cv2.rectangle(annotated, pt1, pt2, color_waste, 2)
        label = waste.class_name
        if waste.on_ground:
            label += " [GROUND]"
        if waste.near_hand:
            label += " [HAND]"
        cv2.putText(
            annotated,
            label,
            (pt1[0], pt1[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color_waste,
            2,
        )

    # Draw dustbins
    for d in result.dustbins:
        pt1 = (int(d.bbox.x1), int(d.bbox.y1))
        pt2 = (int(d.bbox.x2), int(d.bbox.y2))
        cv2.rectangle(annotated, pt1, pt2, color_dustbin, 2)
        cv2.putText(
            annotated,
            f"Dustbin {d.confidence:.2f}",
            (pt1[0], pt1[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color_dustbin,
            2,
        )

    # Draw ground objects (not already drawn as waste)
    waste_bboxes = {(w.bbox.x1, w.bbox.y1) for w in result.waste_objects}
    for g in result.ground_objects:
        if (g.bbox.x1, g.bbox.y1) not in waste_bboxes:
            pt1 = (int(g.bbox.x1), int(g.bbox.y1))
            pt2 = (int(g.bbox.x2), int(g.bbox.y2))
            cv2.rectangle(annotated, pt1, pt2, color_ground, 1)
            cv2.putText(
                annotated,
                f"ground:{g.class_name}",
                (pt1[0], pt1[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color_ground,
                1,
            )

    # Banner
    cls = result.classification
    if "FLAGGED" in cls:
        color = color_flagged
        text = "LITTERING DETECTED"
    elif "VALID" in cls:
        color = color_valid
        text = "VALID DISPOSAL"
    elif "GROUND" in cls:
        color = color_ground
        text = "GROUND LITTER"
    else:
        color = color_other
        text = cls.split(" - ")[0] if " - " in cls else cls

    cv2.rectangle(annotated, (0, 0), (w, 35), color, -1)
    cv2.putText(
        annotated, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
    )

    # Details
    if result.details:
        cv2.rectangle(annotated, (0, h - 40), (w, h), (0, 0, 0), -1)
        cv2.putText(
            annotated,
            result.details[:100],
            (10, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    # Timestamp overlay on all output images
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(
        annotated,
        ts,
        (w - 200, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (200, 200, 200),
        1,
    )

    return annotated


def _print_result(result: FrameResult, out_path: str) -> None:
    """Print detection result summary with rich formatting.

    Args:
        result: FrameResult with detection data.
        out_path: Path where annotated image was saved.
    """
    table = Table(
        title="Detection Result", show_header=True, header_style="bold magenta"
    )
    table.add_column("Field", style="cyan")
    table.add_column("Value")

    classification = result.classification
    if "FLAGGED" in classification:
        style = "bold red"
    elif "VALID" in classification:
        style = "bold green"
    else:
        style = "yellow"

    table.add_row("Classification", f"[{style}]{classification}[/{style}]")
    table.add_row("Details", result.details)
    table.add_row("Persons", str(len(result.persons)))
    table.add_row("Waste Objects", str(result.waste_count))
    table.add_row("Ground Objects", str(result.ground_count))
    table.add_row("Dustbins", str(len(result.dustbins)))
    table.add_row("Saved", out_path)

    console.print(table)
    console.print(f"\n{result.summary}")


def _print_video_summary(total: int, flagged: int, valid: int) -> None:
    """Print video processing summary.

    Args:
        total: Total frames processed.
        flagged: Number of flagged frames.
        valid: Number of valid frames.
    """
    table = Table(
        title="Video Analysis Summary", show_header=True, header_style="bold magenta"
    )
    table.add_column("Metric", style="cyan")
    table.add_column("Value")

    table.add_row("Total Frames", str(total))
    table.add_row("[red]Flagged (Littering)", str(flagged))
    table.add_row("[green]Valid (Disposal)", str(valid))
    table.add_row("Clean/Inconclusive", str(total - flagged - valid))

    if total > 0:
        pct = (flagged / total) * 100
        table.add_row("Flagged Rate", f"{pct:.1f}%")

    console.print(table)
