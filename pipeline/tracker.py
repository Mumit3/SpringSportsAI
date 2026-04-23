"""Ball tracker combining three layers:

  1. YOLOv8 detection   — primary, frame-by-frame
  2. Lucas-Kanade OF    — fallback when YOLO misses (ball apex / motion blur)
  3. Kalman filter       — always running; provides smooth 3-D position & arc

The `source` field tells callers which layer produced the estimate so the
annotator can colour-code the bounding circle accordingly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
import cv2

from .kalman_tracker import KalmanFilter3D
from .detector import Detection
from .svo_reader import SVOReader
from . import config


@dataclass
class TrackerResult:
    position_3d: Optional[np.ndarray]          # [X, Y, Z] metres, or None
    velocity_3d: Optional[np.ndarray]          # [vx, vy, vz] m/s, or None
    position_2d: Optional[Tuple[int, int]]     # projected pixel (u, v), or None
    source: str                                 # 'yolo' | 'optical_flow' | 'kalman' | 'none'
    confidence: float                           # 0–1
    in_flight: bool
    trail_2d: List[Tuple[int, int]] = field(default_factory=list)
    predicted_arc_2d: List[Tuple[int, int]] = field(default_factory=list)


class OpticalFlowTracker:
    """Lucas-Kanade sparse optical flow for short-gap ball tracking."""

    _LK_PARAMS = dict(
        winSize    = (21, 21),
        maxLevel   = 3,
        criteria   = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 25, 0.01),
    )

    def __init__(self):
        self._prev_gray:   Optional[np.ndarray] = None
        self._prev_points: Optional[np.ndarray] = None  # shape (N,1,2) float32

    def init(self, gray: np.ndarray, bbox: Tuple[int,int,int,int]) -> None:
        x1, y1, x2, y2 = bbox
        cx, cy = (x1+x2)/2.0, (y1+y2)/2.0
        r = min(x2-x1, y2-y1) / 3.5

        # Ring of points around ball centre + centre itself
        angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        pts = [[cx + r*np.cos(a), cy + r*np.sin(a)] for a in angles]
        pts.append([cx, cy])

        self._prev_gray   = gray.copy()
        self._prev_points = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)

    def track(self, curr_gray: np.ndarray) -> Optional[Tuple[int,int]]:
        """Return estimated ball centre in curr_gray, or None if tracking lost."""
        if self._prev_points is None or self._prev_gray is None:
            return None

        p1, st, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, curr_gray,
            self._prev_points, None,
            **self._LK_PARAMS,
        )
        if p1 is None or st is None:
            return None

        good = p1[st.ravel() == 1].reshape(-1, 2)   # ensure shape (M, 2)
        if len(good) < 3:
            self.reset()
            return None

        cx = float(np.mean(good[:, 0]))
        cy = float(np.mean(good[:, 1]))

        self._prev_gray   = curr_gray.copy()
        self._prev_points = good.reshape(-1, 1, 2)
        return (int(cx), int(cy))

    def update_gray(self, gray: np.ndarray) -> None:
        """Advance prev_gray without tracking (keeps OF ready for next frame)."""
        self._prev_gray = gray.copy()

    def reset(self) -> None:
        self._prev_gray   = None
        self._prev_points = None


class BallTracker:
    """Fuses YOLO detections, optical flow, and a Kalman filter.

    Call `update()` once per frame in frame order.
    """

    def __init__(self, fps: float, reader: SVOReader):
        self._fps    = fps
        self._reader = reader
        self._kf     = KalmanFilter3D(dt=1.0/fps)
        self._of     = OpticalFlowTracker()

        self._missed   = 0          # consecutive frames without YOLO
        self._in_flight = False
        self._trail: List[Tuple[int,int]] = []

    # ── public ────────────────────────────────────────────────────────────────

    def update(
        self,
        frame_bgr: np.ndarray,
        frame_pc:  np.ndarray,
        ball_det:  Optional[Detection],
    ) -> TrackerResult:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        return self._compute(gray, frame_pc, ball_det)

    def reset(self) -> None:
        self._kf.reset()
        self._of.reset()
        self._missed    = 0
        self._in_flight = False
        self._trail.clear()

    # ── internal ──────────────────────────────────────────────────────────────

    def _compute(
        self,
        gray:     np.ndarray,
        pc:       np.ndarray,
        ball_det: Optional[Detection],
    ) -> TrackerResult:
        source = "none"
        pos3d: Optional[np.ndarray] = None

        # ── Layer 1: YOLO ─────────────────────────────────────────────────────
        if ball_det is not None:
            p3d = self._reader.pixel_to_3d(pc, ball_det.cx, ball_det.cy)
            if p3d is not None:
                pos3d  = p3d
                source = "yolo"
                self._missed = 0
                # Re-seed optical flow from fresh detection
                self._of.init(gray, ball_det.bbox)
                # Kalman update
                if not self._kf.initialized:
                    self._kf.initialize(pos3d)
                else:
                    self._kf.predict()
                    self._kf.update(pos3d)

        # ── Layer 2: Optical Flow ─────────────────────────────────────────────
        if pos3d is None and self._missed < config.OF_MAX_MISSED_FRAMES:
            of_2d = self._of.track(gray)
            if of_2d is not None:
                u, v   = of_2d
                p3d    = self._reader.pixel_to_3d(pc, u, v)
                if p3d is not None:
                    pos3d  = p3d
                    source = "optical_flow"
                    # Kalman update with higher measurement noise
                    if self._kf.initialized:
                        self._kf.predict()
                        self._kf.update(pos3d, noise_scale=3.0)
                else:
                    # OF gave 2D but no valid depth — still use KF predict
                    if self._kf.initialized:
                        pos3d  = self._kf.predict()
                        source = "kalman"
                self._missed += 1
            else:
                self._missed += 1

        # ── Layer 3: Kalman predict-only ──────────────────────────────────────
        if pos3d is None:
            if self._kf.initialized and self._missed < config.KF_MAX_MISSED_FRAMES:
                pos3d  = self._kf.predict()
                source = "kalman"
                self._of.update_gray(gray)   # keep OF gray fresh without tracking
            else:
                self._kf.reset()
                self._of.reset()
                self._missed = 0
            self._missed += 1

        # ── Derived values ────────────────────────────────────────────────────
        vel3d  = self._kf.get_velocity() if self._kf.initialized else None
        speed  = float(np.linalg.norm(vel3d)) if vel3d is not None else 0.0

        # In-flight: upward velocity > 0.5 m/s or fresh detection within last 10 frames
        if vel3d is not None:
            vy = float(vel3d[1])
            self._in_flight = vy > 0.3 or (source in ("yolo", "optical_flow") and self._missed < 10)
        else:
            self._in_flight = False

        # 2-D projection of 3-D position
        pos2d: Optional[Tuple[int,int]] = None
        if pos3d is not None:
            pos2d = self._reader.project_3d_to_2d(pos3d)

        # Update trail
        if pos2d is not None:
            self._trail.append(pos2d)
        if len(self._trail) > config.TRAIL_LENGTH:
            self._trail = self._trail[-config.TRAIL_LENGTH:]

        # Predicted arc in 2D
        arc2d: List[Tuple[int,int]] = []
        if self._kf.initialized and self._in_flight:
            arc3d = self._kf.predict_arc(config.PREDICT_AHEAD_FRAMES)
            for p in arc3d:
                p2 = self._reader.project_3d_to_2d(p)
                if p2 is not None:
                    arc2d.append(p2)
                else:
                    break   # stop at first out-of-frame point

        conf_map = {"yolo": 1.0, "optical_flow": 0.72, "kalman": 0.45, "none": 0.0}

        return TrackerResult(
            position_3d      = pos3d,
            velocity_3d      = vel3d,
            position_2d      = pos2d,
            source           = source,
            confidence       = conf_map[source],
            in_flight        = self._in_flight,
            trail_2d         = list(self._trail),
            predicted_arc_2d = arc2d,
        )
