"""Detect basketball release events from ZED body keypoints.

The pipeline normally infers "release" from ball motion (Method A 3D vy, or
Method B 2D pixel rise). Both have known weaknesses — Method A rarely fires
because the Kalman filter trails reality, and Method B fires late, after the
ball has already passed apex. The ANNOTATED video then shows mostly Kalman
predictions with only late YOLO detections.

Body tracking gives a much more reliable signal: a shooter's wrist follows a
distinctive motion pattern during a shot — accelerating up + forward, peaking
near the moment the ball leaves the hand, then decelerating into the follow-
through. We detect that peak in the wrist trajectory and call it the release.

This module is decoupled from the ZED SDK: you pass in already-extracted
body keypoint snapshots, not raw ZED objects, so it's testable and won't
crash when pyzed isn't available.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np


# ZED BODY_18 indices we care about (verified against ZED SDK 5.x docs).
# Documented as: 0=NOSE, 1=NECK, 2=RIGHT_SHOULDER, 3=RIGHT_ELBOW, 4=RIGHT_WRIST,
# 5=LEFT_SHOULDER, 6=LEFT_ELBOW, 7=LEFT_WRIST, 8=RIGHT_HIP, 9=RIGHT_KNEE,
# 10=RIGHT_ANKLE, 11=LEFT_HIP, 12=LEFT_KNEE, 13=LEFT_ANKLE.
KP_NECK         = 1
KP_RIGHT_WRIST  = 4
KP_LEFT_WRIST   = 7
KP_RIGHT_HIP    = 8
KP_LEFT_HIP     = 11


@dataclass
class BodySnapshot:
    """A single body's keypoints at one frame, in 3D world coordinates.

    Any keypoint that wasn't detected reliably is None.
    """
    frame_idx:   int
    body_id:     int
    torso_3d:    Optional[np.ndarray]   # rough centre-of-mass (neck or hip-mean)
    left_wrist:  Optional[np.ndarray]
    right_wrist: Optional[np.ndarray]


@dataclass
class ReleaseEvent:
    release_frame:   int
    release_pos_3d:  np.ndarray         # wrist position at apex
    wrist:           str                # 'left' or 'right'
    trail_3d:        List               # wrist history from rise → apex (rough ball trail)


def extract_snapshot_from_zed_body(zed_body, frame_idx: int) -> Optional[BodySnapshot]:
    """Convert a ZED `sl.BodyData` into a BodySnapshot.

    Returns None if the body is too low-confidence to use. Defensive against
    keypoints reported as NaN/inf, which ZED uses for "not detected".
    """
    try:
        kp = np.asarray(zed_body.keypoint)         # (N, 3) float
    except AttributeError:
        return None

    def _get(idx):
        if idx >= len(kp):
            return None
        p = kp[idx]
        if not np.all(np.isfinite(p)):
            return None
        return p.astype(float)

    neck = _get(KP_NECK)
    rh   = _get(KP_RIGHT_HIP)
    lh   = _get(KP_LEFT_HIP)
    if neck is not None:
        torso = neck
    elif rh is not None and lh is not None:
        torso = (rh + lh) * 0.5
    elif rh is not None:
        torso = rh
    elif lh is not None:
        torso = lh
    else:
        return None

    return BodySnapshot(
        frame_idx   = frame_idx,
        body_id     = int(getattr(zed_body, "id", -1)),
        torso_3d    = torso,
        left_wrist  = _get(KP_LEFT_WRIST),
        right_wrist = _get(KP_RIGHT_WRIST),
    )


class BodyReleaseDetector:
    """Stateful detector. Feed it one frame's worth of bodies at a time.

    Per-frame steps:
      1. Of all bodies in the frame, pick the one furthest from the rim
         (rebounders stand near the hoop; shooters stand back).
      2. Append the chosen body's wrist positions to a rolling history.
      3. Look at the recent wrist Y trajectory. If it shows a clean rise +
         peak + decline, classify the peak as a release event and return it.

    Parameters tuned for typical 30 fps recordings; adjust via config.
    """

    def __init__(
        self,
        fps:               float,
        history_frames:    int   = 30,
        min_rise_m:        float = 0.20,
        min_decline_m:     float = 0.05,
        cooldown_seconds:  float = 1.5,
    ):
        self._fps               = fps
        self._history: List[BodySnapshot] = []
        self._max_history       = history_frames
        self._min_rise          = min_rise_m
        self._min_decline       = min_decline_m
        self._cooldown_frames   = int(fps * cooldown_seconds)
        self._last_release_frame = -10**9

    def reset(self) -> None:
        self._history.clear()
        self._last_release_frame = -10**9

    def update(
        self,
        frame_idx: int,
        bodies:    list,                    # list of BodySnapshot
        hoop_3d:   Optional[np.ndarray],
    ) -> Optional[ReleaseEvent]:
        """Update with this frame's bodies. Return a ReleaseEvent if one fires."""
        chosen = self._pick_shooter(bodies, hoop_3d)
        if chosen is not None:
            self._history.append(chosen)
            while len(self._history) > self._max_history:
                self._history.pop(0)

        # Cooldown — don't fire two releases right next to each other
        if frame_idx - self._last_release_frame < self._cooldown_frames:
            return None

        return self._detect_release()

    # ── internals ─────────────────────────────────────────────────────────────

    def _pick_shooter(
        self,
        bodies:  list,
        hoop_3d: Optional[np.ndarray],
    ) -> Optional[BodySnapshot]:
        """Of the bodies in this frame, return the one furthest from the rim.

        Rationale: in a two-person scene (shooter + rebounder) the shooter
        stands further back. With one person, that one is always picked.
        """
        if not bodies:
            return None
        if hoop_3d is None:
            # No rim known yet — fall back to the body whose torso is closest
            # to the centre of frame; in practice that's usually the shooter.
            return bodies[0]
        best       = None
        best_dist  = -1.0
        for b in bodies:
            if b.torso_3d is None:
                continue
            d = float(np.linalg.norm(b.torso_3d - hoop_3d))
            if d > best_dist:
                best_dist = d
                best      = b
        return best

    def _detect_release(self) -> Optional[ReleaseEvent]:
        """Look at the wrist Y trajectory in the history and detect a peak.

        Returns a ReleaseEvent if a clean rise → peak → decline pattern is
        seen for either wrist, else None.
        """
        if len(self._history) < 8:
            return None

        for wrist_attr in ("right_wrist", "left_wrist"):
            event = self._detect_for_wrist(wrist_attr)
            if event is not None:
                self._last_release_frame = event.release_frame
                return event
        return None

    def _detect_for_wrist(self, wrist_attr: str) -> Optional[ReleaseEvent]:
        # Build (frame_idx, y, full_3d) list, skipping None entries.
        samples: List[Tuple[int, float, np.ndarray]] = []
        for s in self._history:
            w = getattr(s, wrist_attr)
            if w is None:
                continue
            samples.append((s.frame_idx, float(w[1]), w))
        if len(samples) < 6:
            return None

        # Locate the peak-Y sample, ignoring boundary points so we know there's
        # both a rise BEFORE and a decline AFTER.
        n = len(samples)
        peak_idx = max(range(n), key=lambda i: samples[i][1])
        if peak_idx < 2 or peak_idx > n - 3:
            return None

        peak_y = samples[peak_idx][1]
        rise   = peak_y - samples[0][1]
        decline = peak_y - samples[-1][1]
        if rise < self._min_rise or decline < self._min_decline:
            return None

        release_frame = samples[peak_idx][0]
        release_pos   = samples[peak_idx][2]

        # Trail from the start of the rise up to the peak, in 3D — used to
        # backfill the visual ball trail so the shot's upward arc isn't missing.
        trail = [s[2] for s in samples[:peak_idx + 1]]

        return ReleaseEvent(
            release_frame   = release_frame,
            release_pos_3d  = release_pos,
            wrist           = ("right" if wrist_attr == "right_wrist" else "left"),
            trail_3d        = trail,
        )
