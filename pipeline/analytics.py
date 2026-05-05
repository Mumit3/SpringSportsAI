"""Session analytics — aggregates shot events and serialises to JSON/CSV."""
from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

from .shot_detector import ShotEvent, Outcome


class Analytics:

    def __init__(
        self,
        svo_filename: str,
        fps: float,
        mode: str = "regulation",
        ball_only: bool = False,
    ):
        self.svo_filename = svo_filename
        self.fps          = fps
        self.mode         = mode
        self.ball_only    = ball_only
        self.shots: List[ShotEvent] = []

        # Whole-video ball trails — used when ball_only is True so the 3-D plot
        # can render every tracked frame, not just frames that fall inside a
        # detected shot arc. Empty in normal (shot-detection) mode.
        self.ball_trail_2d: List = []
        self.ball_trail_3d: List = []

    def add(self, shot: ShotEvent) -> None:
        self.shots.append(shot)

    def add_ball_position(self, pos_2d, pos_3d) -> None:
        """Record a per-frame ball position for ball-only mode trails."""
        if pos_2d is not None:
            self.ball_trail_2d.append((int(pos_2d[0]), int(pos_2d[1])))
        if pos_3d is not None:
            self.ball_trail_3d.append((float(pos_3d[0]), float(pos_3d[1]), float(pos_3d[2])))

    # ── summary ───────────────────────────────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        total = len(self.shots)
        makes = sum(1 for s in self.shots if s.outcome == Outcome.MAKE)

        def _avg(key):
            vals = [getattr(s, key) for s in self.shots]
            return round(float(np.mean(vals)), 2) if vals else 0.0

        return {
            "svo_file":         self.svo_filename,
            "mode":             self.mode,
            "ball_only":        self.ball_only,
            "total_shots":      total,
            "makes":            makes,
            "misses":           total - makes,
            "fg_pct":           round(makes / total * 100, 1) if total else 0.0,
            "avg_release_angle_deg": _avg("release_angle_deg"),
            "avg_arc_height_m":      _avg("arc_height_m"),
            "avg_shot_distance_m":   _avg("shot_distance_m"),
            "avg_release_speed_mps": _avg("release_speed_mps"),
        }

    def shot_list(self) -> List[Dict[str, Any]]:
        return [s.to_dict() for s in self.shots]

    def trail_data(self) -> List[List]:
        """Per-shot list of 2-D pixel trails for trajectory visualisation."""
        return [s.trail_2d for s in self.shots]

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, output_dir: Path) -> Dict[str, Path]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "summary": self.summary(),
            "shots":   self.shot_list(),
        }

        json_path = output_dir / "analytics.json"
        with open(json_path, "w") as f:
            json.dump(payload, f, indent=2)

        csv_path = output_dir / "shots.csv"
        rows     = self.shot_list()
        if rows:
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

        return {"json": json_path, "csv": csv_path}

    # ── trajectory export for Plotly ─────────────────────────────────────────

    def plotly_traces(self, frame_width: int, frame_height: int) -> List[Dict]:
        """Return list of Plotly trace dicts — one per shot arc.

        Y axis is inverted so that 'up' in image space corresponds to 'up'
        visually in the chart.
        """
        traces = []
        for idx, shot in enumerate(self.shots):
            if not shot.trail_2d:
                continue
            xs = [p[0] for p in shot.trail_2d]
            ys = [frame_height - p[1] for p in shot.trail_2d]   # invert Y
            color = "#22c55e" if shot.outcome == Outcome.MAKE else "#ef4444"
            label = f"Shot {shot.shot_id} – {shot.outcome.value.upper()}"
            traces.append({
                "x":    xs,
                "y":    ys,
                "mode": "lines+markers",
                "name": label,
                "line": {"color": color, "width": 2},
                "marker": {"size": 4},
            })

        # Whole-video ball trail for ball-only mode
        if self.ball_trail_2d:
            xs = [p[0] for p in self.ball_trail_2d]
            ys = [frame_height - p[1] for p in self.ball_trail_2d]
            traces.append({
                "x":    xs,
                "y":    ys,
                "mode": "lines+markers",
                "name": "Ball trail",
                "line": {"color": "#fb923c", "width": 2},
                "marker": {"size": 3},
            })
        return traces

    # ── 3-D trajectory export for Plotly ─────────────────────────────────────

    def plotly_traces_3d(
        self,
        hoop_3d: "np.ndarray | None" = None,
        cylinder_radius: float = 0.30,
    ) -> List[Dict]:
        """Return list of Plotly 3-D scatter traces — one per shot arc, plus a
        marker for the locked hoop position and a rim circle.

        ZED coordinates (LEFT_HANDED_Y_UP): X = horizontal across, Y = vertical,
        Z = distance from camera. Plotly's 3-D scene treats Z as up by default,
        so we map ZED's Y to Plotly's Z for the most natural orientation.
        """
        traces: List[Dict] = []
        for shot in self.shots:
            if not shot.trail_3d:
                continue
            xs = [p[0] for p in shot.trail_3d]
            ys = [p[2] for p in shot.trail_3d]   # ZED Z → Plotly Y (depth)
            zs = [p[1] for p in shot.trail_3d]   # ZED Y → Plotly Z (up)
            color = "#22c55e" if shot.outcome == Outcome.MAKE else "#ef4444"
            label = f"Shot {shot.shot_id} – {shot.outcome.value.upper()}"
            traces.append({
                "type":   "scatter3d",
                "mode":   "lines+markers",
                "x":      xs,
                "y":      ys,
                "z":      zs,
                "name":   label,
                "line":   {"color": color, "width": 4},
                "marker": {"size": 3, "color": color},
            })

        # Whole-video ball trail for ball-only mode
        if self.ball_trail_3d:
            xs = [p[0] for p in self.ball_trail_3d]
            ys = [p[2] for p in self.ball_trail_3d]
            zs = [p[1] for p in self.ball_trail_3d]
            traces.append({
                "type":   "scatter3d",
                "mode":   "lines+markers",
                "x":      xs,
                "y":      ys,
                "z":      zs,
                "name":   "Ball trail",
                "line":   {"color": "#fb923c", "width": 3},
                "marker": {"size": 2, "color": "#fb923c"},
            })

        if hoop_3d is not None:
            hx, hy, hz = float(hoop_3d[0]), float(hoop_3d[1]), float(hoop_3d[2])
            # Hoop centre marker
            traces.append({
                "type":   "scatter3d",
                "mode":   "markers",
                "x":      [hx],
                "y":      [hz],
                "z":      [hy],
                "name":   "Hoop centre",
                "marker": {"size": 6, "color": "#f97316", "symbol": "diamond"},
            })
            # Rim circle (drawn as line trace in the hoop's horizontal plane)
            ring_x, ring_y, ring_z = [], [], []
            for i in range(33):
                angle = 2 * np.pi * i / 32
                ring_x.append(hx + cylinder_radius * float(np.cos(angle)))
                ring_y.append(hz + cylinder_radius * float(np.sin(angle)))
                ring_z.append(hy)
            traces.append({
                "type":   "scatter3d",
                "mode":   "lines",
                "x":      ring_x,
                "y":      ring_y,
                "z":      ring_z,
                "name":   "Rim",
                "line":   {"color": "#f97316", "width": 4},
                "showlegend": True,
            })

        return traces
