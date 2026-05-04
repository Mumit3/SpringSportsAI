"""Process a ZED SVO file through the basketball analytics pipeline.

CLI usage:
    python main.py svo_files/my_session.svo --label tensorrt_v3

Programmatic usage (called by Flask):
    from main import process_svo
    result = process_svo("svo_files/my_session.svo", label="v1", progress_cb=callback)

Output layout:
    outputs/videos/<svo_stem>__<label>.mp4              ← all annotated MP4s, flat
    outputs/data/<svo_stem>__<label>/{analytics.json, shots.csv, traces.json}
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from pipeline import (
    SVOReader, BallHoopDetector, BallTracker,
    ShotDetector, Analytics, Annotator,
)
from pipeline import config


def _default_label() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M")


def _transcode_for_browser(video_path: Path, progress_cb=None) -> None:
    """Re-encode an MP4 to browser-compatible H.264 + faststart, in-place.

    OpenCV on Jetson typically doesn't ship with H.264 — the writer falls back
    to mp4v, which HTML5 <video> can't decode. ffmpeg fixes it. If ffmpeg
    isn't on PATH the function returns silently (file stays as written).
    """
    if shutil.which("ffmpeg") is None:
        print("[Pipeline] ffmpeg not installed — skipping browser transcode. "
              "Install with: sudo apt install ffmpeg")
        return

    if progress_cb:
        progress_cb(0.99, "Transcoding for browser…")

    tmp = video_path.with_suffix(".h264.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-hide_banner", "-loglevel", "error",
        "-i", str(video_path),
        "-c:v", "libx264",
        "-profile:v", "baseline",   # widest browser support
        "-level", "3.0",
        "-pix_fmt", "yuv420p",      # required for browser HTML5 video
        "-preset", "fast",
        "-crf", "23",
        "-movflags", "+faststart",  # moov atom at start so streaming works
        "-an",                      # drop audio (none anyway)
        str(tmp),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"[Pipeline] ffmpeg transcode failed: {result.stderr.strip()[:300]}")
            tmp.unlink(missing_ok=True)
            return
        tmp.replace(video_path)
        print(f"[Pipeline] Transcoded MP4 to browser-compatible H.264")
    except subprocess.TimeoutExpired:
        print("[Pipeline] ffmpeg transcode timed out — leaving original file.")
        tmp.unlink(missing_ok=True)
    except Exception as e:
        print(f"[Pipeline] ffmpeg transcode error: {e}")
        tmp.unlink(missing_ok=True)


def process_svo(
    svo_path: str,
    label:    Optional[str] = None,
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> dict:
    """Run the full pipeline on an SVO file.

    Args:
        svo_path:    Path to the .svo file.
        label:       Tag for this run; appears in output filenames.
                     Defaults to a timestamp if not provided.
        progress_cb: Optional callback(fraction_done, status_message).

    Returns:
        Dict with keys: session_id, video_path, analytics_json, traces_json, summary.
    """
    svo_path = Path(svo_path)
    stem     = svo_path.stem
    label    = (label or _default_label()).strip()
    session  = f"{stem}__{label}"

    videos_dir = config.OUTPUT_DIR / "videos"
    data_dir   = config.OUTPUT_DIR / "data" / session
    videos_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    video_path = videos_dir / f"{session}.mp4"

    def _cb(frac: float, msg: str = ""):
        if progress_cb:
            progress_cb(frac, msg)

    _cb(0.0, "Opening SVO file…")

    # ── initialise components ─────────────────────────────────────────────────
    reader   = SVOReader(str(svo_path))
    detector = BallHoopDetector()
    tracker  = BallTracker(fps=reader.info.fps, reader=reader)
    shot_det = ShotDetector(
        fps          = reader.info.fps,
        frame_width  = reader.info.width,
        frame_height = reader.info.height,
    )
    analytics = Analytics(svo_filename=svo_path.name, fps=reader.info.fps)
    annotator = Annotator(frame_width=reader.info.width, frame_height=reader.info.height)

    # ── video writer ──────────────────────────────────────────────────────────
    # Try H.264 first (browser-friendly), fall back to mp4v if OpenCV's build
    # doesn't have H.264 support. Browsers can't play raw mp4v in HTML5 video,
    # so without H.264 the user has to download the MP4 to view it.
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"avc1"),
        reader.info.fps,
        (reader.info.width, reader.info.height),
    )
    if not writer.isOpened():
        print("[Pipeline] avc1 (H.264) unavailable in this OpenCV build, "
              "falling back to mp4v — output may not play in browsers.")
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            reader.info.fps,
            (reader.info.width, reader.info.height),
        )

    total     = reader.info.total_frames
    hoop_3d:  Optional[np.ndarray] = None
    hoop_3d_locked = False
    hoop_ema_alpha = 0.15           # smooth hoop position over time

    _cb(0.02, f"Processing {total} frames at {reader.info.fps:.0f} fps…")

    t0 = time.time()

    try:
        with reader:
            for frame in reader:
                idx = frame.index

                # ── dynamic ball confidence during flight ─────────────────────────
                ball_conf = (
                    config.BALL_CONF_FLIGHT
                    if tracker._in_flight
                    else config.BALL_CONF_NORMAL
                )

                # ── detect ────────────────────────────────────────────────────────
                ball_det, hoop_det = detector.detect(frame.image, ball_conf)

                # ── update 3-D hoop position (EMA until locked) ───────────────
                if hoop_det is not None and not hoop_3d_locked:
                    new_h3d = reader.pixel_to_3d(frame.point_cloud, hoop_det.cx, hoop_det.cy)
                    if new_h3d is not None and config.HOOP_DEPTH_MIN < new_h3d[2] < config.HOOP_DEPTH_MAX:
                        if hoop_3d is None:
                            hoop_3d = new_h3d.copy()
                        else:
                            hoop_3d = (1 - hoop_ema_alpha) * hoop_3d + hoop_ema_alpha * new_h3d

                    # When the 2-D bbox is locked, also lock the 3-D position
                    if hoop_det.source == "locked" and hoop_3d is not None:
                        hoop_3d_locked = True
                        print(f"[Hoop] 3D position locked at "
                              f"X={hoop_3d[0]:.2f} Y={hoop_3d[1]:.2f} Z={hoop_3d[2]:.2f}")

                # ── track ball ────────────────────────────────────────────────────
                tracker_result = tracker.update(
                    frame.image, frame.point_cloud, ball_det, hoop_det, hoop_3d,
                )

                # ── detect shots ──────────────────────────────────────────────────
                completed_shot = shot_det.update(idx, tracker_result, hoop_det, hoop_3d)
                if completed_shot is not None:
                    analytics.add(completed_shot)

                # ── annotate + write frame ─────────────────────────────────────────
                annotated = annotator.draw(
                    frame.image, ball_det, hoop_det,
                    tracker_result, completed_shot, hoop_3d, idx,
                )
                writer.write(annotated)

                # ── progress ──────────────────────────────────────────────────────
                if idx % 30 == 0 or idx == total - 1:
                    frac = min((idx + 1) / max(total, 1), 0.98)
                    elapsed = time.time() - t0
                    fps_est = (idx + 1) / elapsed if elapsed > 0 else 0
                    _cb(frac, f"Frame {idx+1}/{total}  ({fps_est:.1f} fps)")
    finally:
        writer.release()

    # ── transcode to browser-compatible H.264 (no-op if ffmpeg missing) ───────
    _transcode_for_browser(video_path, progress_cb=_cb)

    # ── save analytics ────────────────────────────────────────────────────────
    paths   = analytics.save(data_dir)
    summary = analytics.summary()

    # Save Plotly trace data for the web UI
    traces_path = data_dir / "traces.json"
    with open(traces_path, "w") as f:
        json.dump(analytics.plotly_traces(reader.info.width, reader.info.height), f)

    _cb(1.0, "Done.")

    elapsed = time.time() - t0
    print(f"\n[Pipeline] Finished in {elapsed:.1f}s")
    print(f"  Shots detected : {summary['total_shots']}")
    print(f"  Makes / Misses : {summary['makes']} / {summary['misses']}")
    print(f"  FG%            : {summary['fg_pct']}%")
    print(f"  Video          : {video_path}")
    print(f"  Data           : {data_dir}")

    return {
        "session_id":    session,
        "video_path":    str(video_path),
        "analytics_json": str(paths["json"]),
        "traces_json":   str(traces_path),
        "summary":       summary,
    }


# ── CLI entry point ───────────────────────────────────────────────────────────

def _cli():
    parser = argparse.ArgumentParser(description="Basketball SVO analytics pipeline")
    parser.add_argument("svo", help="Path to the .svo file")
    parser.add_argument(
        "--label",
        help="Tag for this run; shows up in output filenames. "
             "Defaults to a timestamp if omitted.",
    )
    args = parser.parse_args()

    def _print_progress(frac: float, msg: str):
        bar_len = 30
        filled  = int(bar_len * frac)
        bar     = "█" * filled + "░" * (bar_len - filled)
        print(f"\r[{bar}] {frac*100:5.1f}%  {msg:<40}", end="", flush=True)

    process_svo(args.svo, label=args.label, progress_cb=_print_progress)
    print()   # newline after progress bar


if __name__ == "__main__":
    _cli()
