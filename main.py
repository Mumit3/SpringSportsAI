"""Process a ZED SVO file through the basketball analytics pipeline.

CLI usage:
    python main.py svo_files/my_session.svo

Programmatic usage (called by Flask):
    from main import process_svo
    result = process_svo("svo_files/my_session.svo", progress_cb=callback)
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from pipeline import (
    SVOReader, BallHoopDetector, BallTracker,
    ShotDetector, Analytics, Annotator,
)
from pipeline import config


def _sanitize_label(label: str) -> str:
    """Make a user-supplied label safe for filesystem and URL use."""
    label = label.strip().lower()
    label = re.sub(r"[^a-z0-9_\-]+", "_", label)
    label = re.sub(r"_+", "_", label)   # collapse multi-underscores (delimiter is __)
    label = label.strip("_")
    return label or "untitled"


def process_svo(
    svo_path: str,
    progress_cb: Optional[Callable[[float, str], None]] = None,
    label:       Optional[str] = None,
) -> dict:
    """Run the full pipeline on an SVO file.

    Args:
        svo_path:    Path to the .svo file.
        progress_cb: Optional callback(fraction_done, status_message).
        label:       Optional human label for this run. Used in the output
                     folder name so multiple runs of the same SVO don't collide.

    Returns:
        Dict with keys: output_dir, analytics_json, video_path, summary, job_id.
    """
    svo_path  = Path(svo_path)
    stem      = svo_path.stem

    # Output layout:
    #   EXPERIMENT_MODE: flat — outputs/experiment/<run_name>.mp4 plus
    #                    <run_name>_analytics.json / _traces.json / _shots.csv
    #                    All files sit directly under outputs/experiment/.
    #   else (legacy):   outputs/<stem>/annotated.mp4 + analytics.json + ...
    safe_label = _sanitize_label(label) if label else None
    if config.EXPERIMENT_MODE:
        out_dir   = config.OUTPUT_DIR / "experiment"
        run_name  = safe_label or stem
        job_id    = f"experiment__{run_name}"
        prefix    = run_name           # → <run_name>.mp4, <run_name>_analytics.json
    else:
        out_dir   = config.OUTPUT_DIR / stem
        run_name  = stem
        job_id    = stem
        prefix    = ""                 # legacy: annotated.mp4 / analytics.json
    out_dir.mkdir(parents=True, exist_ok=True)

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
    overlay_label = safe_label or "v3"
    annotator = Annotator(
        frame_width  = reader.info.width,
        frame_height = reader.info.height,
        version      = overlay_label,
    )

    # ── video writer ──────────────────────────────────────────────────────────
    video_path = out_dir / (f"{prefix}.mp4" if prefix else "annotated.mp4")
    fourcc     = cv2.VideoWriter_fourcc(*"mp4v")
    writer     = cv2.VideoWriter(
        str(video_path), fourcc,
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
                tracker_result = tracker.update(frame.image, frame.point_cloud, ball_det, hoop_det)

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

    # ── save analytics ────────────────────────────────────────────────────────
    paths   = analytics.save(out_dir, name_prefix=prefix)
    summary = analytics.summary()

    # Save Plotly trace data for the web UI
    traces_filename = f"{prefix}_traces.json" if prefix else "traces.json"
    traces_path     = out_dir / traces_filename
    with open(traces_path, "w") as f:
        json.dump(analytics.plotly_traces(reader.info.width, reader.info.height), f)

    _cb(1.0, "Done.")

    elapsed = time.time() - t0
    print(f"\n[Pipeline] Finished in {elapsed:.1f}s")
    print(f"  Shots detected : {summary['total_shots']}")
    print(f"  Makes / Misses : {summary['makes']} / {summary['misses']}")
    print(f"  FG%            : {summary['fg_pct']}%")
    print(f"  Output         : {out_dir}")

    return {
        "job_id":         job_id,
        "label":          safe_label,
        "output_dir":     str(out_dir),
        "video_path":     str(video_path),
        "analytics_json": str(paths["json"]),
        "traces_json":    str(traces_path),
        "summary":        summary,
    }


# ── CLI entry point ───────────────────────────────────────────────────────────

def _cli():
    parser = argparse.ArgumentParser(description="Basketball SVO analytics pipeline")
    parser.add_argument("svo", help="Path to the .svo file")
    parser.add_argument("--label", default=None,
                        help="Optional label for this run (used in the output folder name)")
    args = parser.parse_args()

    def _print_progress(frac: float, msg: str):
        bar_len = 30
        filled  = int(bar_len * frac)
        bar     = "█" * filled + "░" * (bar_len - filled)
        print(f"\r[{bar}] {frac*100:5.1f}%  {msg:<40}", end="", flush=True)

    process_svo(args.svo, progress_cb=_print_progress, label=args.label)
    print()   # newline after progress bar


if __name__ == "__main__":
    _cli()
