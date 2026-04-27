"""Frame annotator — draws all overlays onto a BGR frame in-place."""
from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np
import cv2

from .detector import Detection
from .tracker import TrackerResult
from .shot_detector import ShotEvent, Outcome
from . import config


def _lerp_color(c1, c2, t):
    return tuple(int(a + (b-a)*t) for a, b in zip(c1, c2))


class Annotator:

    def __init__(self, frame_width: int, frame_height: int, version: str = "v3"):
        self._fw = frame_width
        self._fh = frame_height
        self._version = version

        # Font
        self._font       = cv2.FONT_HERSHEY_SIMPLEX
        self._font_small = 0.55
        self._font_med   = 0.75

        # Active shot overlay timer
        self._result_display_frames = 0
        self._last_outcome: Optional[Outcome] = None
        self._last_metrics: Optional[dict]    = None

    # ── main entry ────────────────────────────────────────────────────────────

    def draw(
        self,
        frame:     np.ndarray,
        ball_det:  Optional[Detection],
        hoop_det:  Optional[Detection],
        tracker:   TrackerResult,
        shot:      Optional[ShotEvent],
        hoop_3d:   Optional[np.ndarray],
        frame_idx: int,
    ) -> np.ndarray:
        out = frame.copy()

        self._draw_hoop(out, hoop_det)
        self._draw_trail(out, tracker.trail_2d)
        self._draw_predicted_arc(out, tracker.predicted_arc_2d)
        self._draw_ball(out, ball_det, tracker)
        self._draw_hud(out, tracker, frame_idx)
        self._draw_version(out)

        if shot is not None:
            self._result_display_frames = 90   # show result for 3 s @ 30 fps
            self._last_outcome = shot.outcome
            self._last_metrics = shot.to_dict()

        if self._result_display_frames > 0:
            self._draw_shot_result(out)
            self._result_display_frames -= 1

        return out

    # ── hoop ──────────────────────────────────────────────────────────────────

    def _draw_hoop(self, frame: np.ndarray, hoop_det: Optional[Detection]) -> None:
        if hoop_det is None:
            return
        x1, y1, x2, y2 = hoop_det.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), config.COLOR_HOOP, 2)
        cv2.putText(frame, f"RIM {hoop_det.confidence:.0%}",
                    (x1, y1-6), self._font, self._font_small,
                    config.COLOR_HOOP, 1, cv2.LINE_AA)

    # ── ball bounding box ─────────────────────────────────────────────────────

    def _draw_ball(
        self,
        frame:    np.ndarray,
        det:      Optional[Detection],
        tracker:  TrackerResult,
    ) -> None:
        pos2d = tracker.position_2d
        if pos2d is None:
            return

        source_colors = {
            "yolo":         config.COLOR_BALL_YOLO,
            "optical_flow": config.COLOR_BALL_OF,
            "hough":        config.COLOR_BALL_HOUGH,
            "kalman":       config.COLOR_BALL_KALMAN,
            "none":         (100, 100, 100),
        }
        color = source_colors.get(tracker.source, (100, 100, 100))

        if det is not None and tracker.source == "yolo":
            # Draw YOLO bounding box
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            radius = max(det.w, det.h) // 2
        else:
            # Draw circle for estimated position
            radius = 18
            cv2.circle(frame, pos2d, radius, color, 2)

        # Centre dot
        cv2.circle(frame, pos2d, 4, color, -1)

        # Source label
        label = tracker.source.replace("_", " ").upper()
        cv2.putText(frame, label,
                    (pos2d[0]+radius+4, pos2d[1]+5),
                    self._font, 0.45, color, 1, cv2.LINE_AA)

    # ── trajectory trail ──────────────────────────────────────────────────────

    def _draw_trail(
        self,
        frame: np.ndarray,
        trail: List[Tuple[int,int]],
    ) -> None:
        n = len(trail)
        if n < 2:
            return

        for i in range(1, n):
            t = i / n   # 0 (tail) → 1 (head)
            color  = _lerp_color(config.COLOR_TRAIL_TAIL, config.COLOR_TRAIL_HEAD, t)
            radius = max(2, int(3 + 4 * t))
            cv2.circle(frame, trail[i], radius, color, -1)

        # Connect with lines
        for i in range(1, n):
            t = i / n
            color = _lerp_color(config.COLOR_TRAIL_TAIL, config.COLOR_TRAIL_HEAD, t)
            thickness = max(1, int(1 + 2 * t))
            cv2.line(frame, trail[i-1], trail[i], color, thickness, cv2.LINE_AA)

    # ── predicted arc ─────────────────────────────────────────────────────────

    def _draw_predicted_arc(
        self,
        frame: np.ndarray,
        arc:   List[Tuple[int,int]],
    ) -> None:
        if len(arc) < 2:
            return

        for i in range(1, len(arc)):
            if i % 2 == 0:   # dashed effect
                cv2.line(frame, arc[i-1], arc[i],
                         config.COLOR_ARC_PRED, 1, cv2.LINE_AA)

    # ── HUD overlay (top-left corner) ─────────────────────────────────────────

    def _draw_hud(
        self,
        frame:    np.ndarray,
        tracker:  TrackerResult,
        frame_idx: int,
    ) -> None:
        lines = [f"Frame: {frame_idx}"]

        if tracker.velocity_3d is not None:
            vy  = tracker.velocity_3d[1]
            spd = float(np.linalg.norm(tracker.velocity_3d))
            lines.append(f"Spd: {spd:.1f} m/s  Vy: {vy:+.1f}")

        if tracker.position_3d is not None:
            z = tracker.position_3d[2]
            lines.append(f"Dist: {z:.2f} m")

        lines.append(f"Src: {tracker.source}")

        y_start = 24
        for line in lines:
            # Shadow
            cv2.putText(frame, line, (11, y_start+1),
                        self._font, self._font_small, (0,0,0), 2, cv2.LINE_AA)
            cv2.putText(frame, line, (10, y_start),
                        self._font, self._font_small, (230,230,230), 1, cv2.LINE_AA)
            y_start += 22

    # ── version label (top-center) ────────────────────────────────────────────

    def _draw_version(self, frame: np.ndarray) -> None:
        text = self._version
        scale = 1.1
        thickness = 2
        (tw, th), _ = cv2.getTextSize(text, self._font, scale, thickness)
        x = (self._fw - tw) // 2
        y = th + 14
        # Shadow
        cv2.putText(frame, text, (x+2, y+2), self._font, scale, (0,0,0), thickness+2, cv2.LINE_AA)
        # Foreground
        cv2.putText(frame, text, (x, y), self._font, scale, (255,255,255), thickness, cv2.LINE_AA)

    # ── shot result banner ────────────────────────────────────────────────────

    def _draw_shot_result(self, frame: np.ndarray) -> None:
        if self._last_outcome is None or self._last_metrics is None:
            return

        if self._last_outcome == Outcome.MAKE:
            color  = config.COLOR_MAKE
            label  = "MAKE"
        elif self._last_outcome == Outcome.MISS:
            color  = config.COLOR_MISS
            label  = "MISS"
        else:
            color  = config.COLOR_PENDING
            label  = "..."

        m = self._last_metrics

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, self._fh-120), (self._fw, self._fh), (20,20,20), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        # Big result label
        cv2.putText(frame, label,
                    (self._fw//2 - 80, self._fh - 70),
                    self._font, 2.4, color, 4, cv2.LINE_AA)

        # Metrics row
        stats = (
            f"Angle: {m.get('release_angle_deg',0):.1f}°  |  "
            f"Arc: {m.get('arc_height_m',0):.2f} m  |  "
            f"Dist: {m.get('shot_distance_m',0):.2f} m  |  "
            f"Speed: {m.get('release_speed_mps',0):.1f} m/s"
        )
        tw, _ = cv2.getTextSize(stats, self._font, self._font_small, 1)
        x_off = max(0, (self._fw - tw[0]) // 2)
        cv2.putText(frame, stats, (x_off, self._fh - 20),
                    self._font, self._font_small, (220, 220, 220), 1, cv2.LINE_AA)
