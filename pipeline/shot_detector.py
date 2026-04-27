"""Shot event detection and make/miss classification.

Two independent detection methods run in parallel; either can trigger a shot.
A 60-frame cooldown prevents double-counting.

Make/miss uses a 3-D cylinder check as primary and a 2-D bounding-box overlap
as fallback.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional
import numpy as np

from .tracker import TrackerResult
from .detector import Detection
from . import config


class Outcome(str, Enum):
    MAKE    = "make"
    MISS    = "miss"
    PENDING = "pending"


@dataclass
class ShotEvent:
    shot_id:        int
    trigger_frame:  int
    release_frame:  int
    release_pos:    np.ndarray          # 3-D
    release_vel:    np.ndarray          # 3-D m/s
    apex_pos:       Optional[np.ndarray]

    release_angle_deg: float            # degrees above horizontal
    arc_height_m:      float            # apex Y − release Y
    shot_distance_m:   float            # 3-D distance to hoop at release
    release_speed_mps: float

    outcome: Outcome = Outcome.PENDING
    outcome_frame: int = -1

    # 2-D pixel trail of ball positions during this shot arc
    trail_2d: List = field(default_factory=list)

    def to_dict(self) -> dict:
        vel  = self.release_vel
        horiz_speed = float(np.linalg.norm([vel[0], vel[2]]))
        return {
            "shot_id":           self.shot_id,
            "trigger_frame":     self.trigger_frame,
            "release_angle_deg": round(self.release_angle_deg, 1),
            "arc_height_m":      round(self.arc_height_m, 3),
            "shot_distance_m":   round(self.shot_distance_m, 3),
            "release_speed_mps": round(self.release_speed_mps, 2),
            "horiz_speed_mps":   round(horiz_speed, 2),
            "outcome":           self.outcome.value,
            "outcome_frame":     self.outcome_frame,
        }


class _ArcState(Enum):
    IDLE   = auto()
    FLIGHT = auto()


class ShotDetector:
    """Stateful per-frame shot detector.

    Call `update()` every frame; it returns a completed ShotEvent when
    a shot is newly classified, otherwise None.
    """

    def __init__(self, fps: float, frame_width: int, frame_height: int):
        self._fps   = fps
        self._fw    = frame_width
        self._fh    = frame_height

        self._state       = _ArcState.IDLE
        self._cooldown    = 0
        self._shot_id     = 0
        self._arc_frames  = 0
        self._current: Optional[ShotEvent] = None

        # Pixel-rise rolling window (2-D method)
        self._ball_y_window: List[int] = []

        # For computing arc height
        self._apex_y3d: float = -1e9

        self.completed_shots: List[ShotEvent] = []

    # ── public ────────────────────────────────────────────────────────────────

    def update(
        self,
        frame_idx:      int,
        tracker:        TrackerResult,
        hoop_det:       Optional[Detection],
        hoop_3d:        Optional[np.ndarray],
        ball_in_basket: Optional[Detection] = None,
    ) -> Optional[ShotEvent]:
        """Returns a newly completed ShotEvent, or None."""
        if self._cooldown > 0:
            self._cooldown -= 1

        # Track 2-D ball y position for pixel-rise method
        if tracker.position_2d is not None:
            self._ball_y_window.append(tracker.position_2d[1])
        if len(self._ball_y_window) > config.PIXEL_RISE_WINDOW:
            self._ball_y_window.pop(0)

        # ball-in-basket is a definitive MAKE signal
        if ball_in_basket is not None and self._state == _ArcState.FLIGHT:
            return self._finalise(Outcome.MAKE, frame_idx)

        if self._state == _ArcState.IDLE:
            return self._check_trigger(frame_idx, tracker, hoop_3d)
        else:
            return self._track_arc(frame_idx, tracker, hoop_det, hoop_3d)

    # ── trigger logic ─────────────────────────────────────────────────────────

    def _check_trigger(
        self,
        frame_idx: int,
        tracker:   TrackerResult,
        hoop_3d:   Optional[np.ndarray],
    ) -> None:
        if self._cooldown > 0 or not tracker.in_flight:
            return None

        triggered = False

        # Method A: 3-D arc — upward Kalman velocity exceeds threshold
        if tracker.velocity_3d is not None:
            vy = float(tracker.velocity_3d[1])
            if vy > config.ARC_VELOCITY_THRESHOLD:
                triggered = True

        # Method B: 2-D pixel rise — ball rises > threshold in rolling window
        if not triggered and len(self._ball_y_window) >= config.PIXEL_RISE_WINDOW:
            y_vals = self._ball_y_window
            rise_px = y_vals[-1] - y_vals[0]   # negative = rising (image y flipped)
            if -rise_px / self._fh > config.PIXEL_RISE_THRESHOLD:
                triggered = True

        if not triggered:
            return None

        # Initialise a new shot event
        self._shot_id     += 1
        self._arc_frames   = 0
        self._apex_y3d     = tracker.position_3d[1] if tracker.position_3d is not None else 0.0
        self._state        = _ArcState.FLIGHT

        dist = 0.0
        if tracker.position_3d is not None and hoop_3d is not None:
            dist = float(np.linalg.norm(tracker.position_3d - hoop_3d))

        vel  = tracker.velocity_3d if tracker.velocity_3d is not None else np.zeros(3)
        spd  = float(np.linalg.norm(vel))
        vy   = float(vel[1])
        horiz = float(np.linalg.norm([vel[0], vel[2]]))
        angle = float(np.degrees(np.arctan2(vy, horiz))) if horiz > 0 else 90.0

        self._current = ShotEvent(
            shot_id         = self._shot_id,
            trigger_frame   = frame_idx,
            release_frame   = frame_idx,
            release_pos     = tracker.position_3d.copy() if tracker.position_3d is not None else np.zeros(3),
            release_vel     = vel.copy(),
            apex_pos        = None,
            release_angle_deg = angle,
            arc_height_m    = 0.0,
            shot_distance_m = dist,
            release_speed_mps = spd,
        )
        return None

    # ── arc tracking ──────────────────────────────────────────────────────────

    def _track_arc(
        self,
        frame_idx:  int,
        tracker:    TrackerResult,
        hoop_det:   Optional[Detection],
        hoop_3d:    Optional[np.ndarray],
    ) -> Optional[ShotEvent]:
        self._arc_frames += 1

        if tracker.position_2d is not None:
            self._current.trail_2d.append(tracker.position_2d)

        # Update apex
        if tracker.position_3d is not None:
            if tracker.position_3d[1] > self._apex_y3d:
                self._apex_y3d      = float(tracker.position_3d[1])
                self._current.apex_pos = tracker.position_3d.copy()

        # Arc height
        self._current.arc_height_m = max(
            0.0, self._apex_y3d - float(self._current.release_pos[1])
        )

        # Check make/miss
        outcome = self._classify(tracker, hoop_det, hoop_3d)
        if outcome != Outcome.PENDING:
            return self._finalise(outcome, frame_idx)

        # Terminate arc when ball starts descending or max frames reached
        ball_descending = (
            tracker.velocity_3d is not None
            and float(tracker.velocity_3d[1]) < -0.5
            and self._arc_frames > config.ARC_MIN_FRAMES
        )
        arc_timeout = self._arc_frames >= config.ARC_MAX_FRAMES

        if (ball_descending or arc_timeout) and self._arc_frames >= config.ARC_MIN_FRAMES:
            return self._finalise(Outcome.MISS, frame_idx)

        return None

    # ── classification ────────────────────────────────────────────────────────

    def _classify(
        self,
        tracker:  TrackerResult,
        hoop_det: Optional[Detection],
        hoop_3d:  Optional[np.ndarray],
    ) -> Outcome:
        # Primary: 3-D cylinder through hoop plane
        if tracker.position_3d is not None and hoop_3d is not None:
            outcome = self._classify_3d(tracker.position_3d, tracker.velocity_3d, hoop_3d)
            if outcome != Outcome.PENDING:
                return outcome

        # Fallback: 2-D bounding-box overlap with downward velocity
        if tracker.position_2d is not None and hoop_det is not None:
            outcome = self._classify_2d(tracker, hoop_det)
            if outcome != Outcome.PENDING:
                return outcome

        return Outcome.PENDING

    def _classify_3d(
        self,
        ball_pos: np.ndarray,
        ball_vel: Optional[np.ndarray],
        hoop_3d:  np.ndarray,
    ) -> Outcome:
        """Ball must cross hoop plane (Y == hoop Y) within cylinder of radius R."""
        # Only classify when ball is descending through hoop height
        if ball_vel is not None and float(ball_vel[1]) > 0.1:
            return Outcome.PENDING   # still rising

        hoop_y  = float(hoop_3d[1])
        ball_y  = float(ball_pos[1])

        if ball_y > hoop_y + 0.1:
            return Outcome.PENDING   # ball hasn't reached hoop height yet

        # Check horizontal distance from hoop centre
        dx = float(ball_pos[0]) - float(hoop_3d[0])
        dz = float(ball_pos[2]) - float(hoop_3d[2])
        horiz_dist = np.sqrt(dx**2 + dz**2)

        if horiz_dist <= config.MAKE_CYLINDER_RADIUS:
            return Outcome.MAKE
        elif ball_y < hoop_y - 0.5:
            # Ball passed well below the hoop — miss
            return Outcome.MISS

        return Outcome.PENDING

    def _classify_2d(
        self,
        tracker:  TrackerResult,
        hoop_det: Detection,
    ) -> Outcome:
        """Bounding-box overlap + ball moving downward."""
        if tracker.position_2d is None:
            return Outcome.PENDING

        # Ball must be moving downward in image (y increasing)
        if len(self._ball_y_window) >= 4:
            if self._ball_y_window[-1] <= self._ball_y_window[-3]:
                return Outcome.PENDING   # not descending in image

        bx, by = tracker.position_2d
        hx1, hy1, hx2, hy2 = hoop_det.bbox

        # Expand hoop bbox slightly
        pad = 20
        if (hx1-pad <= bx <= hx2+pad) and (hy1-pad <= by <= hy2+pad):
            return Outcome.MAKE

        return Outcome.PENDING

    # ── finalise ──────────────────────────────────────────────────────────────

    def _finalise(self, outcome: Outcome, frame_idx: int) -> ShotEvent:
        self._current.outcome       = outcome
        self._current.outcome_frame = frame_idx
        self._state                 = _ArcState.IDLE
        self._cooldown              = config.SHOT_COOLDOWN_FRAMES
        shot = self._current
        self._current = None
        self.completed_shots.append(shot)
        return shot
