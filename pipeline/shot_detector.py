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
    # 3-D world trail (X, Y, Z) in metres — for interactive 3-D plotting
    trail_3d: List = field(default_factory=list)

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

        # Per-shot debug stats — reset per shot
        self._dbg_trigger_method: str = ""
        self._dbg_min_horiz_dist: float = float("inf")
        self._dbg_min_horiz_frame: int = -1
        self._dbg_min_horiz_descending: bool = False
        self._dbg_min_2d_dist:    float = float("inf")
        self._dbg_was_in_2d_bbox: bool = False
        self._dbg_hoop_3d:        Optional[np.ndarray] = None
        self._dbg_hoop_bbox:      Optional[tuple] = None
        self._dbg_classify_reason: str = ""

        # Rolling buffer of per-frame tracker state — used to dump pre-trigger
        # context when a shot fires, so we can see why Method A did/didn't fire.
        self._dbg_frame_buf: List[tuple] = []

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

        # Rolling capture of per-frame tracker state for pre-trigger diagnostics.
        if config.DEBUG_SHOT_DETECTION:
            p3d_copy = tracker.position_3d.copy() if tracker.position_3d is not None else None
            v3d_copy = tracker.velocity_3d.copy() if tracker.velocity_3d is not None else None
            self._dbg_frame_buf.append((
                frame_idx, tracker.source, tracker.position_2d, p3d_copy, v3d_copy,
            ))
            if len(self._dbg_frame_buf) > config.DEBUG_PRETRIGGER_FRAMES:
                self._dbg_frame_buf.pop(0)

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
        method = ""

        # Method A: 3-D arc — upward Kalman velocity exceeds threshold
        if tracker.velocity_3d is not None:
            vy = float(tracker.velocity_3d[1])
            if vy > config.ARC_VELOCITY_THRESHOLD:
                triggered = True
                method = "A (3D vy)"

        # Method B: 2-D pixel rise — ball rises > threshold in rolling window
        if not triggered and len(self._ball_y_window) >= config.PIXEL_RISE_WINDOW:
            y_vals = self._ball_y_window
            rise_px = y_vals[-1] - y_vals[0]   # negative = rising (image y flipped)
            if -rise_px / self._fh > config.PIXEL_RISE_THRESHOLD:
                triggered = True
                method = "B (2D rise)"

        if not triggered:
            return None

        # Dump pre-trigger frame buffer so we can see what 3D state we had
        # leading up to the trigger (in particular, why Method A didn't fire).
        if config.DEBUG_SHOT_DETECTION:
            self._print_pre_trigger_buffer(method, frame_idx)

        # Reset per-shot debug stats
        self._dbg_trigger_method  = method
        self._dbg_min_horiz_dist  = float("inf")
        self._dbg_min_horiz_frame = -1
        self._dbg_min_horiz_descending = False
        self._dbg_min_2d_dist     = float("inf")
        self._dbg_was_in_2d_bbox  = False
        self._dbg_hoop_3d         = hoop_3d.copy() if hoop_3d is not None else None
        self._dbg_hoop_bbox       = None
        self._dbg_classify_reason = ""

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
        if tracker.position_3d is not None:
            p = tracker.position_3d
            self._current.trail_3d.append((float(p[0]), float(p[1]), float(p[2])))

        # Update apex
        if tracker.position_3d is not None:
            if tracker.position_3d[1] > self._apex_y3d:
                self._apex_y3d      = float(tracker.position_3d[1])
                self._current.apex_pos = tracker.position_3d.copy()

        # Debug: track closest 3-D horizontal approach to hoop
        if (config.DEBUG_SHOT_DETECTION
                and tracker.position_3d is not None
                and hoop_3d is not None):
            dx = float(tracker.position_3d[0]) - float(hoop_3d[0])
            dz = float(tracker.position_3d[2]) - float(hoop_3d[2])
            hdist = float(np.sqrt(dx*dx + dz*dz))
            if hdist < self._dbg_min_horiz_dist:
                self._dbg_min_horiz_dist = hdist
                self._dbg_min_horiz_frame = frame_idx
                self._dbg_min_horiz_descending = (
                    tracker.velocity_3d is not None
                    and float(tracker.velocity_3d[1]) < 0.1
                )

        # Debug: track 2-D bbox proximity / containment
        if (config.DEBUG_SHOT_DETECTION
                and tracker.position_2d is not None
                and hoop_det is not None):
            self._dbg_hoop_bbox = hoop_det.bbox
            bx, by = tracker.position_2d
            hx1, hy1, hx2, hy2 = hoop_det.bbox
            if hx1 <= bx <= hx2 and hy1 <= by <= hy2:
                self._dbg_was_in_2d_bbox = True
                self._dbg_min_2d_dist = 0.0
            else:
                cx, cy = (hx1+hx2)//2, (hy1+hy2)//2
                d2 = float(np.sqrt((bx-cx)**2 + (by-cy)**2))
                if d2 < self._dbg_min_2d_dist:
                    self._dbg_min_2d_dist = d2

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
            if ball_below_hoop:
                self._dbg_classify_reason = "arc terminated: ball >0.5 m below hoop in 3D"
            elif arc_timeout:
                self._dbg_classify_reason = f"arc terminated: timeout at {self._arc_frames} frames"
            else:
                self._dbg_classify_reason = "arc terminated: falling (no hoop info)"
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
                self._dbg_classify_reason = f"3D rim bounce (vy={float(ball_vel[1]):.2f}>0.5 after entry)"
                return Outcome.MISS

            # Slipped sideways out of cylinder while still at hoop height
            if (not in_cylinder
                and frames_since_entry >= 2
                and ball_y > hoop_y - 0.3):
                self._dbg_classify_reason = f"3D slipped sideways (horiz_dist={horiz_dist:.2f}>{config.MAKE_CYLINDER_RADIUS:.2f})"
                return Outcome.MISS

            # Cleared through the bottom of the hoop
            if ball_y < hoop_y - 0.4:
                self._dbg_classify_reason = "3D cleared bottom of hoop"
                return Outcome.MAKE

            # Held inside cylinder for several descending frames → MAKE
            if frames_since_entry >= 6 and in_cylinder and descending:
                self._dbg_classify_reason = "3D held in cylinder ≥6 descending frames"
                return Outcome.MAKE

        # Never entered cylinder, but ball passed well below hoop → MISS
        if ball_y < hoop_y - 0.4 and self._cylinder_entry_frame < 0:
            self._dbg_classify_reason = (
                f"3D below hoop without cylinder entry "
                f"(min_horiz={self._dbg_min_horiz_dist:.2f}>{config.MAKE_CYLINDER_RADIUS:.2f})"
            )
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
            self._dbg_classify_reason = "2D ball inside padded hoop bbox while descending"
            return Outcome.MAKE

        return Outcome.PENDING

    # ── finalise ──────────────────────────────────────────────────────────────

    def _finalise(self, outcome: Outcome, frame_idx: int) -> ShotEvent:
        self._current.outcome       = outcome
        self._current.outcome_frame = frame_idx

        if config.DEBUG_SHOT_DETECTION:
            self._print_debug(outcome, frame_idx)

        self._state                 = _ArcState.IDLE
        self._cooldown              = config.SHOT_COOLDOWN_FRAMES
        self._cylinder_entry_frame  = -1
        self._cylinder_entry_vy     = 0.0
        shot = self._current
        self._current = None
        self.completed_shots.append(shot)
        return shot

    def _print_pre_trigger_buffer(self, method: str, trigger_frame: int) -> None:
        """Print the last N frames of tracker state leading up to the trigger.

        Reveals YOLO detection continuity (source field) and 3D Kalman state
        (position/velocity), so we can diagnose why Method A (3D vy > threshold)
        did or didn't fire and whether ball detection was stable during ascent.
        """
        if not self._dbg_frame_buf:
            return
        n = len(self._dbg_frame_buf)
        print(
            f"\n[Pre-trigger trace] Shot #{self._shot_id + 1}, "
            f"trigger frame {trigger_frame} via {method} "
            f"({n} frames captured, A threshold = {config.ARC_VELOCITY_THRESHOLD} m/s)"
        )
        # Count detection sources to summarise YOLO continuity
        src_counts = {}
        max_vy = -1e9
        for _, src, _, _, v3d in self._dbg_frame_buf:
            src_counts[src] = src_counts.get(src, 0) + 1
            if v3d is not None:
                max_vy = max(max_vy, float(v3d[1]))
        src_summary = ", ".join(f"{k}={v}" for k, v in src_counts.items())
        max_vy_str = f"{max_vy:+.2f}" if max_vy > -1e8 else "n/a"
        print(f"  source breakdown: {src_summary} | max vy seen = {max_vy_str} m/s")
        print(f"  {'frame':>5}  {'source':<13} {'2D':>11}  {'3D (X,Y,Z)':>22}  {'vel (vx,vy,vz)':>22}")
        for f, src, p2d, p3d, v3d in self._dbg_frame_buf:
            p2d_s = f"({p2d[0]},{p2d[1]})" if p2d is not None else "—"
            p3d_s = (f"({p3d[0]:+.2f},{p3d[1]:+.2f},{p3d[2]:+.2f})"
                     if p3d is not None else "—")
            v3d_s = (f"({v3d[0]:+.1f},{v3d[1]:+.1f},{v3d[2]:+.1f})"
                     if v3d is not None else "—")
            print(f"  {f:>5}  {src:<13} {p2d_s:>11}  {p3d_s:>22}  {v3d_s:>22}")
        print()

    def _print_debug(self, outcome: Outcome, frame_idx: int) -> None:
        s = self._current
        rel = s.release_pos
        vel = s.release_vel
        cyl = self._cylinder_entry_frame
        descend = "descending" if self._dbg_min_horiz_descending else "ascending/level"

        if self._dbg_was_in_2d_bbox:
            bbox_line = "ball entered hoop bbox at some frame"
        else:
            bbox_line = f"ball never inside hoop bbox (closest 2D = {self._dbg_min_2d_dist:.0f} px)"

        if self._dbg_hoop_3d is not None:
            hoop_line = f"hoop @ ({self._dbg_hoop_3d[0]:.2f}, {self._dbg_hoop_3d[1]:.2f}, {self._dbg_hoop_3d[2]:.2f})"
        else:
            hoop_line = "hoop 3D not available"

        if self._dbg_min_horiz_dist == float("inf"):
            min_horiz_line = "no 3D ball/hoop samples during arc"
        else:
            min_horiz_line = (
                f"min 3D horiz = {self._dbg_min_horiz_dist:.2f} m "
                f"@ frame {self._dbg_min_horiz_frame} ({descend}); "
                f"cylinder = {config.MAKE_CYLINDER_RADIUS:.2f} m"
            )

        print(
            f"\n[Shot {s.shot_id}] {outcome.value.upper()} at frame {frame_idx}\n"
            f"  trigger: frame {s.trigger_frame}, method {self._dbg_trigger_method}, "
            f"angle {s.release_angle_deg:+.1f}°, speed {s.release_speed_mps:.1f} m/s\n"
            f"  release pos: ({rel[0]:.2f}, {rel[1]:.2f}, {rel[2]:.2f})  "
            f"vel: ({vel[0]:+.1f}, {vel[1]:+.1f}, {vel[2]:+.1f})\n"
            f"  {hoop_line}\n"
            f"  {min_horiz_line}\n"
            f"  cylinder entry: "
            f"{'frame ' + str(s.trigger_frame + cyl) if cyl >= 0 else 'never'}\n"
            f"  2D: {bbox_line}\n"
            f"  reason: {self._dbg_classify_reason}\n"
        )
