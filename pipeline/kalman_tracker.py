"""3-D Kalman filter with constant-gravity model for basketball tracking.

State vector:  [x, y, z, vx, vy, vz]
Measurement:   [x, y, z]
Coordinate:    LEFT_HANDED_Y_UP  →  gravity acts on -y  (vy decreases by g·dt)
"""
from __future__ import annotations
import numpy as np
from typing import Optional
from . import config


class KalmanFilter3D:

    def __init__(self, dt: float = 1.0 / 30.0):
        self.dt   = dt
        self.g    = config.GRAVITY
        self.initialized = False

        # ── state & covariance ────────────────────────────────────────────────
        self.x = np.zeros(6)          # [x, y, z, vx, vy, vz]
        self.P = np.eye(6) * 50.0     # initial uncertainty

        # ── transition matrix F ───────────────────────────────────────────────
        self.F = np.eye(6)
        self._update_F(dt)

        # ── measurement matrix H  (we observe x, y, z only) ──────────────────
        self.H = np.zeros((3, 6))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0

        q = config.KALMAN_PROCESS_NOISE
        r = config.KALMAN_MEASUREMENT_NOISE

        # ── process noise Q ───────────────────────────────────────────────────
        # Higher noise on velocity components — ball can accelerate on contact
        self.Q = np.diag([q*0.5, q*0.5, q*0.5, q*2.0, q*2.0, q*2.0])

        # ── measurement noise R ───────────────────────────────────────────────
        self.R = np.eye(3) * (r ** 2)

    # ── internal ──────────────────────────────────────────────────────────────

    def _update_F(self, dt: float) -> None:
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt

    # ── public API ────────────────────────────────────────────────────────────

    def initialize(self, pos: np.ndarray, velocity: Optional[np.ndarray] = None) -> None:
        self.x[:3] = pos
        self.x[3:] = velocity if velocity is not None else np.zeros(3)
        self.P = np.eye(6) * 5.0
        self.initialized = True

    def predict(self, dt: Optional[float] = None) -> np.ndarray:
        """Predict next state. Returns predicted position [x, y, z]."""
        if not self.initialized:
            return self.x[:3].copy()

        if dt is not None and dt != self.dt:
            self.dt = dt
            self._update_F(dt)

        self.x = self.F @ self.x
        self.x[4] -= self.g * self.dt   # gravity on vy

        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:3].copy()

    def peek_predict(self, dt: Optional[float] = None) -> np.ndarray:
        """Return what predict() WOULD return, without modifying state."""
        if not self.initialized:
            return self.x[:3].copy()

        use_dt = dt if dt is not None else self.dt
        if use_dt == self.dt:
            F = self.F
        else:
            F = np.eye(6)
            F[0, 3] = use_dt
            F[1, 4] = use_dt
            F[2, 5] = use_dt

        x_next = F @ self.x
        x_next[4] -= self.g * use_dt
        return x_next[:3].copy()

    def update(self, measurement: np.ndarray, noise_scale: float = 1.0) -> np.ndarray:
        """Correct with a 3-D measurement. Returns corrected position."""
        if not self.initialized:
            self.initialize(measurement)
            return self.x[:3].copy()

        z   = np.asarray(measurement, dtype=float)
        R   = self.R * (noise_scale ** 2)

        y   = z - self.H @ self.x                     # innovation
        S   = self.H @ self.P @ self.H.T + R          # innovation cov
        K   = self.P @ self.H.T @ np.linalg.inv(S)    # Kalman gain

        self.x = self.x + K @ y
        I      = np.eye(6)
        self.P = (I - K @ self.H) @ self.P

        return self.x[:3].copy()

    def get_position(self) -> np.ndarray:
        return self.x[:3].copy()

    def get_velocity(self) -> np.ndarray:
        return self.x[3:].copy()

    def get_speed(self) -> float:
        return float(np.linalg.norm(self.x[3:]))

    def predict_arc(self, n_frames: int, dt: Optional[float] = None) -> np.ndarray:
        """Return (n_frames, 3) array of predicted future positions."""
        dt  = dt or self.dt
        x, y, z, vx, vy, vz = self.x.copy()
        g   = self.g
        out = np.empty((n_frames, 3))
        for i in range(n_frames):
            t      = (i + 1) * dt
            out[i] = [
                x  + vx * t,
                y  + vy * t - 0.5 * g * t**2,
                z  + vz * t,
            ]
        return out

    def reset(self) -> None:
        self.x[:] = 0.0
        self.P    = np.eye(6) * 50.0
        self.initialized = False
