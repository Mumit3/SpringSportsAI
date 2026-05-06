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
from pipeline import background_mask
from pipeline.body_tracker import (
    BodyReleaseDetector,
    extract_snapshot_from_zed_body,
    extract_keypoints_2d,
)
from pipeline.profiles import profile_scope, available as available_profiles


class CancelledError(Exception):
    """Raised inside the pipeline when the caller signals a cancel.
    Caught by the Flask job runner so it can clean up partial output."""
    pass


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
    profile:  str = "regulation",
    ball_only: bool = False,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> dict:
    """Run the full pipeline on an SVO file.

    Args:
        svo_path:    Path to the .svo file.
        label:       Tag for this run; appears in output filenames.
                     Defaults to a timestamp if not provided.
        progress_cb: Optional callback(fraction_done, status_message).
        profile:     Named processing profile applied for the duration of
                     this job. "regulation" (default) is no-op; "mini" applies
                     mini-hoop overrides.
        ball_only:   When True, skip rim detection and shot classification.
                     Pipeline still tracks the ball and produces 2-D / 3-D
                     trail visualisations of the entire video.

    Returns:
        Dict with keys: session_id, video_path, analytics_json, traces_json, summary.
    """
    with profile_scope(profile):
        return _process_svo_inner(
            svo_path, label, progress_cb, profile, ball_only, cancel_check,
        )


def _process_svo_inner(
    svo_path:     str,
    label:        Optional[str],
    progress_cb:  Optional[Callable[[float, str], None]],
    profile:      str,
    ball_only:    bool,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> dict:
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

    mode_msg = f" [profile={profile}{', ball-only' if ball_only else ''}]"
    _cb(0.0, f"Opening SVO file…{mode_msg}")

    # ── initialise components ─────────────────────────────────────────────────
    body_tracking_wanted = bool(config.BODY_TRACKING_ENABLED) and not ball_only
    reader   = SVOReader(str(svo_path), enable_body_tracking=body_tracking_wanted)
    body_release_det = (
        BodyReleaseDetector(fps=reader.info.fps)
        if reader.body_tracking_enabled else None
    )
    detector = BallHoopDetector()
    tracker  = BallTracker(fps=reader.info.fps, reader=reader)
    shot_det = ShotDetector(
        fps          = reader.info.fps,
        frame_width  = reader.info.width,
        frame_height = reader.info.height,
    )
    analytics = Analytics(
        svo_filename = svo_path.name,
        fps          = reader.info.fps,
        mode         = profile,
        ball_only    = ball_only,
    )
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

                # ── cancel check (caller can abort mid-pipeline) ──────────────────
                if cancel_check is not None and cancel_check():
                    raise CancelledError("cancelled by caller")

                # ── dynamic ball confidence ───────────────────────────────────────
                # Three tiers: NORMAL when YOLO has the ball, FLIGHT when in
                # flight, RECOVERY when we've lost the ball for several frames.
                # Working videos never reach the recovery tier (they don't lose
                # the ball for long), so this can't regress them.
                if tracker._missed >= config.BALL_RECOVERY_MISS_THRESHOLD:
                    ball_conf = config.BALL_CONF_RECOVERY
                elif tracker._in_flight:
                    ball_conf = config.BALL_CONF_FLIGHT
                else:
                    ball_conf = config.BALL_CONF_NORMAL

                # ── detect ────────────────────────────────────────────────────────
                # Optional: dim non-foreground pixels before YOLO sees the
                # frame. The original frame.image is unchanged — annotator and
                # writer still see the natural-looking image.
                det_image = background_mask.apply(
                    frame.image, frame.point_cloud, hoop_3d,
                )
                ball_det, hoop_det = detector.detect(det_image, ball_conf)

                # In ball-only mode, ignore the rim entirely. Tracker, shot
                # detector, and annotator behave as if no hoop was ever found.
                if ball_only:
                    hoop_det = None

                # ── body tracking — detect release from wrist motion ──────────────
                # When enabled, this overrides the late Method A/B trigger by
                # firing a shot at the body-detected release frame, with the
                # wrist position as the release point.
                body_keypoints_2d = []   # for skeleton overlay
                if body_release_det is not None:
                    raw_bodies = reader.retrieve_bodies()
                    snapshots  = []
                    for b in raw_bodies:
                        s = extract_snapshot_from_zed_body(b, idx)
                        if s is not None:
                            snapshots.append(s)
                        kp2d = extract_keypoints_2d(b)
                        if kp2d is not None:
                            body_keypoints_2d.append(kp2d)
                    release_event = body_release_det.update(idx, snapshots, hoop_3d)
                    if release_event is not None:
                        print(
                            f"[BodyRelease] Shot release detected at frame "
                            f"{release_event.release_frame} via {release_event.wrist} "
                            f"wrist; pos = ({release_event.release_pos_3d[0]:.2f}, "
                            f"{release_event.release_pos_3d[1]:.2f}, "
                            f"{release_event.release_pos_3d[2]:.2f})"
                        )
                        shot_det.inject_release(
                            frame_idx   = release_event.release_frame,
                            release_pos = release_event.release_pos_3d,
                            hoop_3d     = hoop_3d,
                            wrist_trail = release_event.trail_3d,
                        )

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
                if ball_only:
                    completed_shot = None
                    analytics.add_ball_position(
                        tracker_result.position_2d, tracker_result.position_3d,
                    )
                else:
                    completed_shot = shot_det.update(idx, tracker_result, hoop_det, hoop_3d)
                    if completed_shot is not None:
                        analytics.add(completed_shot)

                # ── annotate + write frame ─────────────────────────────────────────
                annotated = annotator.draw(
                    frame.image, ball_det, hoop_det,
                    tracker_result, completed_shot, hoop_3d, idx,
                    body_keypoints_2d=body_keypoints_2d if body_keypoints_2d else None,
                    shot_in_flight=shot_det.is_in_flight,
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

    traces_3d_path = data_dir / "traces_3d.json"
    with open(traces_3d_path, "w") as f:
        json.dump(
            analytics.plotly_traces_3d(hoop_3d, config.MAKE_CYLINDER_RADIUS),
            f,
        )

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
    parser.add_argument(
        "--profile",
        choices=available_profiles(),
        default="regulation",
        help="Processing profile. 'regulation' uses the default constants; "
             "'mini' applies mini-hoop overrides (smaller ball, smaller "
             "make cylinder, lower floor minimum).",
    )
    parser.add_argument(
        "--ball-only",
        action="store_true",
        help="Skip rim detection and shot classification. Tracks the ball "
             "across the whole video for trajectory visualisation only.",
    )
    args = parser.parse_args()

    def _print_progress(frac: float, msg: str):
        bar_len = 30
        filled  = int(bar_len * frac)
        bar     = "█" * filled + "░" * (bar_len - filled)
        print(f"\r[{bar}] {frac*100:5.1f}%  {msg:<40}", end="", flush=True)

    process_svo(
        args.svo,
        label       = args.label,
        progress_cb = _print_progress,
        profile     = args.profile,
        ball_only   = args.ball_only,
    )
    print()   # newline after progress bar


if __name__ == "__main__":
    _cli()
