"""Session analytics — aggregates shot events and serialises to JSON/CSV."""
from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

from .shot_detector import ShotEvent, Outcome


class Analytics:

    def __init__(self, svo_filename: str, fps: float):
        self.svo_filename = svo_filename
        self.fps          = fps
        self.shots: List[ShotEvent] = []

    def add(self, shot: ShotEvent) -> None:
        self.shots.append(shot)

    # ── summary ───────────────────────────────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        total = len(self.shots)
        makes = sum(1 for s in self.shots if s.outcome == Outcome.MAKE)

        def _avg(key):
            vals = [getattr(s, key) for s in self.shots]
            return round(float(np.mean(vals)), 2) if vals else 0.0

        return {
            "svo_file":         self.svo_filename,
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
        return traces
