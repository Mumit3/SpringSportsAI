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


def _segment_intersects_rect(p1, p2, rect) -> bool:
    """Liang-Barsky test: does segment p1→p2 intersect axis-aligned rect?"""
    x1, y1 = p1
    x2, y2 = p2
    xmin, ymin, xmax, ymax = rect
    dx, dy = x2 - x1, y2 - y1
    t_enter, t_exit = 0.0, 1.0
    for p, q in ((-dx, x1 - xmin), (dx, xmax - x1),
                 (-dy, y1 - ymin), (dy, ymax - y1)):
        if p == 0:
            if q < 0:
                return False
            continue
        t = q / p
        if p < 0:
            if t > t_exit:
                return False
            t_enter = max(t_enter, t)
        else:
            if t < t_enter:
                return False
            t_exit = min(t_exit, t)
    return t_enter <= t_exit


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

        # Velocity-sign make/miss state — reset per shot
        self._cylinder_entry_frame: int = -1
        self._cylinder_entry_vy:    float = 0.0

        self.completed_shots: List[ShotEvent] = []

    # ── public ────────────────────────────────────────────────────────────────

    def update(
        self,
        frame_idx:  int,
        tracker:    TrackerResult,
        hoop_det:   Optional[Detection],
        hoop_3d:    Optional[np.ndarray],
    ) -> Optional[ShotEvent]:
        """Returns a newly completed ShotEvent, or None."""
        if self._cooldown > 0:
            self._cooldown -= 1

        # Track 2-D ball y position for pixel-rise method
        if tracker.position_2d is not None:
            self._ball_y_window.append(tracker.position_2d[1])
        if len(self._ball_y_window) > config.PIXEL_RISE_WINDOW:
            self._ball_y_window.pop(0)

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
        self._cylinder_entry_frame = -1
        self._cylinder_entry_vy    = 0.0

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

        # Terminate arc only if the ball has clearly passed the hoop or timed out
        ball_below_hoop = (
            tracker.position_3d is not None
            and hoop_3d is not None
            and float(tracker.position_3d[1]) < float(hoop_3d[1]) - 0.5
        )
        # Fallback: if no hoop info, fall back to descent + min frames
        ball_falling_no_hoop = (
            hoop_3d is None
            and tracker.velocity_3d is not None
            and float(tracker.velocity_3d[1]) < -0.5
            and self._arc_frames > config.ARC_MIN_FRAMES * 2
        )
        arc_timeout = self._arc_frames >= config.ARC_MAX_FRAMES

        if ((ball_below_hoop or ball_falling_no_hoop or arc_timeout)
                and self._arc_frames >= config.ARC_MIN_FRAMES):
            return self._finalise(Outcome.MISS, frame_idx)

        return None

    # ── classification ────────────────────────────────────────────────────────

    def _classify(
        self,
        tracker:  TrackerResult,
        hoop_det: Optional[Detection],
        hoop_3d:  Optional[np.ndarray],
    ) -> Outcome:
        # 2-D first: only ever returns MAKE or PENDING. At long range (~8 m to
        # the hoop) ZED depth on a small fast ball is noisier than the 3-D
        # cylinder radius, so trusting 3-D as primary causes valid makes to be
        # ruled out before 2-D ever sees the trajectory crossing.
        if tracker.position_2d is not None and hoop_det is not None:
            outcome = self._classify_2d(tracker, hoop_det)
            if outcome == Outcome.MAKE:
                return outcome

        # 3-D second: confirms makes via cylinder pass-through and detects
        # rim bounces / sideways slips.
        if tracker.position_3d is not None and hoop_3d is not None:
            outcome = self._classify_3d(tracker.position_3d, tracker.velocity_3d, hoop_3d)
            if outcome != Outcome.PENDING:
                return outcome

        return Outcome.PENDING

    def _classify_3d(
        self,
        ball_pos: np.ndarray,
        ball_vel: Optional[np.ndarray],
        hoop_3d:  np.ndarray,
    ) -> Outcome:
        """Velocity-sign make/miss classification.

        A MAKE requires the ball to (1) enter the cylinder while descending,
        and (2) continue descending out the bottom. Any upward Vy reversal
        after entry is a rim-out → MISS.
        """
        hoop_y = float(hoop_3d[1])
        ball_y = float(ball_pos[1])

        dx = float(ball_pos[0]) - float(hoop_3d[0])
        dz = float(ball_pos[2]) - float(hoop_3d[2])
        horiz_dist = float(np.sqrt(dx*dx + dz*dz))
        in_cylinder = horiz_dist <= config.MAKE_CYLINDER_RADIUS
        descending  = ball_vel is not None and float(ball_vel[1]) < 0.1

        # Hasn't reached the hoop area yet
        if ball_y > hoop_y + 0.15:
            return Outcome.PENDING

        # First time entering cylinder while descending → mark entry
        if (in_cylinder
            and self._cylinder_entry_frame < 0
            and descending
            and ball_y < hoop_y + 0.15):
            self._cylinder_entry_frame = self._arc_frames
            self._cylinder_entry_vy    = float(ball_vel[1]) if ball_vel is not None else 0.0

        # After entry, watch what happens
        if self._cylinder_entry_frame >= 0:
            frames_since_entry = self._arc_frames - self._cylinder_entry_frame

            # Rim bounce: Vy reverses upward after a descending entry
            if (ball_vel is not None
                and float(ball_vel[1]) > 0.5
                and frames_since_entry >= 1):
                return Outcome.MISS

            # Slipped sideways out of cylinder while still at hoop height
            if (not in_cylinder
                and frames_since_entry >= 2
                and ball_y > hoop_y - 0.3):
                return Outcome.MISS

            # Cleared through the bottom of the hoop
            if ball_y < hoop_y - 0.4:
                return Outcome.MAKE

            # Held inside cylinder for several descending frames → MAKE
            if frames_since_entry >= 6 and in_cylinder and descending:
                return Outcome.MAKE

        # NOTE: do NOT fire MISS here just because ball dropped below hoop
        # without "entering the cylinder" — at long range the cylinder check
        # is unreliable due to depth noise. The arc termination in
        # _track_arc will mark MISS if no MAKE is detected anywhere.
        return Outcome.PENDING

    def _classify_2d(
        self,
        tracker:  TrackerResult,
        hoop_det: Detection,
    ) -> Outcome:
        """Trajectory crossing through hoop bbox + ball moving downward.

        Checks the line segment between the previous and current ball position
        for intersection with the (padded) hoop bbox — this catches makes where
        the ball moves fast enough to skip past the bbox between samples.
        """
        if tracker.position_2d is None:
            return Outcome.PENDING

        # Ball must be moving downward in image (y increasing)
        if len(self._ball_y_window) >= 4:
            if self._ball_y_window[-1] <= self._ball_y_window[-3]:
                return Outcome.PENDING

        bx, by = tracker.position_2d
        hx1, hy1, hx2, hy2 = hoop_det.bbox
        pad = 40
        rx1, ry1, rx2, ry2 = hx1-pad, hy1-pad, hx2+pad, hy2+pad

        # Current point inside padded bbox
        if rx1 <= bx <= rx2 and ry1 <= by <= ry2:
            return Outcome.MAKE

        # Trajectory crossing — segment from previous to current ball position
        if (self._current is not None
                and len(self._current.trail_2d) >= 2):
            px, py = self._current.trail_2d[-2]
            if _segment_intersects_rect((px, py), (bx, by), (rx1, ry1, rx2, ry2)):
                return Outcome.MAKE

        return Outcome.PENDING

    # ── finalise ──────────────────────────────────────────────────────────────

    def _finalise(self, outcome: Outcome, frame_idx: int) -> ShotEvent:
        self._current.outcome       = outcome
        self._current.outcome_frame = frame_idx
        self._state                 = _ArcState.IDLE
        self._cooldown              = config.SHOT_COOLDOWN_FRAMES
        self._cylinder_entry_frame  = -1
        self._cylinder_entry_vy     = 0.0
        shot = self._current
        self._current = None
        self.completed_shots.append(shot)
        return shot
