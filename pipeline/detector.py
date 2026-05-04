"""Ball and hoop detection.

Primary:  YOLOv8 (custom basketball.pt if present, else COCO yolov8n.pt)
Fallback: Colour-segmentation + Hough ellipse for hoop when YOLO confidence is low.

Detection sources are transparent to callers — the result just carries a
`source` field ('yolo_custom', 'yolo_coco', 'color') for diagnostics.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import cv2

from . import config


@dataclass
class Detection:
    x1: int; y1: int; x2: int; y2: int
    confidence: float
    class_name: str
    source: str = "yolo"

    @property
    def cx(self) -> int:  return (self.x1 + self.x2) // 2
    @property
    def cy(self) -> int:  return (self.y1 + self.y2) // 2
    @property
    def w(self)  -> int:  return self.x2 - self.x1
    @property
    def h(self)  -> int:  return self.y2 - self.y1
    @property
    def bbox(self) -> Tuple[int,int,int,int]: return (self.x1, self.y1, self.x2, self.y2)


class BallHoopDetector:
    """Detects basketball and hoop in a BGR frame.

    Hoop detection is polled every `HOOP_POLL_FRAMES` frames and cached
    between polls — the hoop doesn't move during a session.
    """

    def __init__(self):
        self._model       = None
        self._model_type  = None   # 'custom' | 'coco'
        self._frame_count = 0

        # Hoop state
        self._hoop_cache: Optional[Detection] = None
        self._hoop_stable_count = 0
        self._hoop_locked: Optional[Detection] = None
        self._hoop_candidates: list = []

        self._load_model()

    # ── model loading ─────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        try:
            from ultralytics import YOLO  # noqa: import inside fn – optional dep
        except ImportError:
            print("[Detector] ultralytics not installed — run setup.sh first.")
            return

        custom = Path(config.YOLO_MODEL_PATH)
        if custom.exists():
            print(f"[Detector] Loading custom model: {custom}")
            self._model      = YOLO(str(custom))
            self._model_type = "custom"
            return

        # Prefer TensorRT engine on Jetson (3-5× faster than .pt with same accuracy).
        engine_path = Path(config.YOLO_FALLBACK).with_suffix(".engine")
        if config.USE_TENSORRT and engine_path.exists():
            print(f"[Detector] Loading TensorRT engine: {engine_path}")
            self._model      = YOLO(str(engine_path))
            self._model_type = "coco"
            return

        print(f"[Detector] Using COCO {config.YOLO_FALLBACK}")
        self._model      = YOLO(config.YOLO_FALLBACK)
        self._model_type = "coco"

    # ── public API ────────────────────────────────────────────────────────────

    def detect(
        self,
        frame: np.ndarray,
        ball_conf_override: Optional[float] = None,
    ) -> Tuple[Optional[Detection], Optional[Detection]]:
        """Return (ball_detection, hoop_detection). Either may be None."""
        self._frame_count += 1
        ball = self._detect_ball(frame, ball_conf_override)
        hoop = self._get_hoop(frame)
        return ball, hoop

    # ── ball detection ────────────────────────────────────────────────────────

    def _detect_ball(
        self,
        frame: np.ndarray,
        conf_override: Optional[float],
    ) -> Optional[Detection]:
        if self._model is None:
            return None

        conf = conf_override or config.BALL_CONF_NORMAL
        results = self._model(frame, verbose=False, conf=conf, imgsz=config.YOLO_IMGSZ)[0]

        best: Optional[Detection] = None

        if self._model_type == "custom":
            best = self._best_from_results(results, config.CUSTOM_BALL_NAMES, conf)
        else:
            best = self._best_by_class_id(results, config.COCO_BALL_CLASS, conf)

        return best

    # ── hoop detection ────────────────────────────────────────────────────────

    def _get_hoop(self, frame: np.ndarray) -> Optional[Detection]:
        """Return locked hoop, or run calibration / fallback detection."""
        # Once locked, the hoop is fixed for the rest of the video
        if self._hoop_locked is not None:
            return self._hoop_locked

        # ── Calibration phase: detect every frame for first N frames ──────────
        if self._frame_count <= config.HOOP_CALIBRATION_FRAMES:
            fresh = self._detect_hoop(frame)
            if fresh is not None:
                self._hoop_candidates.append(fresh)
                self._hoop_cache = fresh

            if self._frame_count == config.HOOP_CALIBRATION_FRAMES:
                self._hoop_locked = self._cluster_and_lock_hoop()
                if self._hoop_locked is not None:
                    print(f"[Hoop] Locked at ({self._hoop_locked.cx},{self._hoop_locked.cy}) "
                          f"from {len(self._hoop_candidates)} candidates")
                else:
                    print("[Hoop] Calibration failed — using fallback per-frame detection")

            return self._hoop_cache

        # ── Post-calibration fallback (lock failed) ───────────────────────────
        if self._frame_count % config.HOOP_POLL_FRAMES != 0:
            return self._hoop_cache

        fresh = self._detect_hoop(frame)
        if fresh is not None:
            if self._hoop_cache is not None and self._hoop_stable_count > 5:
                dx = abs(fresh.cx - self._hoop_cache.cx)
                dy = abs(fresh.cy - self._hoop_cache.cy)
                if dx > 120 or dy > 120:
                    fresh = None

        if fresh is not None:
            self._hoop_cache = fresh
            self._hoop_stable_count += 1
        elif self._hoop_stable_count > 0:
            self._hoop_stable_count = max(0, self._hoop_stable_count - 1)

        return self._hoop_cache

    def _cluster_and_lock_hoop(self) -> Optional[Detection]:
        """Cluster calibration detections by position and return the dominant one."""
        if len(self._hoop_candidates) < 5:
            return None

        tol = config.HOOP_CLUSTER_TOLERANCE_PX
        clusters: list = []
        for det in self._hoop_candidates:
            placed = False
            for cluster in clusters:
                cx_avg = sum(d.cx for d in cluster) / len(cluster)
                cy_avg = sum(d.cy for d in cluster) / len(cluster)
                if abs(det.cx - cx_avg) < tol and abs(det.cy - cy_avg) < tol:
                    cluster.append(det)
                    placed = True
                    break
            if not placed:
                clusters.append([det])

        clusters.sort(key=len, reverse=True)
        best = clusters[0]
        if len(best) < 3:
            return None   # dominant cluster too small to trust

        x1 = int(sum(d.x1 for d in best) / len(best))
        y1 = int(sum(d.y1 for d in best) / len(best))
        x2 = int(sum(d.x2 for d in best) / len(best))
        y2 = int(sum(d.y2 for d in best) / len(best))

        # Expand bbox downward to include net area (HSV usually only catches rim arc)
        rim_height = max(1, y2 - y1)
        y2_expanded = y2 + int(rim_height * 1.2)
        return Detection(x1, y1, x2, y2_expanded, 0.95, "hoop", "locked")

    def _detect_hoop(self, frame: np.ndarray) -> Optional[Detection]:
        """Try YOLO first, fall back to colour segmentation."""
        det = self._detect_hoop_yolo(frame)
        if det is not None:
            return det
        return self._detect_hoop_color(frame)

    def _detect_hoop_yolo(self, frame: np.ndarray) -> Optional[Detection]:
        if self._model is None:
            return None
        results = self._model(frame, verbose=False, conf=config.HOOP_CONF, imgsz=config.YOLO_IMGSZ)[0]
        if self._model_type == "custom":
            return self._best_from_results(results, config.CUSTOM_HOOP_NAMES, config.HOOP_CONF)
        return None   # COCO has no hoop class

    def _detect_hoop_color(self, frame: np.ndarray) -> Optional[Detection]:
        """Colour + shape fallback for the orange steel rim."""
        h, w = frame.shape[:2]
        # Hoop is in the upper 65 % of the frame
        roi_frame = frame[:int(h * 0.65), :]

        hsv  = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
        lo1, hi1 = np.array(config.HOOP_HSV_LOWER1), np.array(config.HOOP_HSV_UPPER1)
        lo2, hi2 = np.array(config.HOOP_HSV_LOWER2), np.array(config.HOOP_HSV_UPPER2)
        mask = cv2.bitwise_or(
            cv2.inRange(hsv, lo1, hi1),
            cv2.inRange(hsv, lo2, hi2),
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_area = 0
        best_det: Optional[Detection] = None

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 80 or area > 8000:
                continue
            if len(cnt) < 5:
                continue

            rx, ry, rw, rh = cv2.boundingRect(cnt)
            aspect = rw / max(rh, 1)
            # Hoop appears as a wide, thin ellipse or arc from the side
            if not (0.8 < aspect < 8.0):
                continue

            if area > best_area:
                best_area = area
                # Adjust y back to full-frame coordinates
                best_det = Detection(
                    x1=rx, y1=ry, x2=rx+rw, y2=ry+rh,
                    confidence=0.40,
                    class_name="hoop",
                    source="color",
                )

        return best_det

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _best_from_results(
        results,
        target_names: set,
        min_conf: float,
    ) -> Optional[Detection]:
        """Pick highest-confidence detection whose class name is in target_names."""
        names  = results.names            # {id: name}
        boxes  = results.boxes

        best_conf = min_conf - 1e-6
        best_det: Optional[Detection] = None

        for box in boxes:
            cls_id = int(box.cls[0])
            name   = names.get(cls_id, "").lower()
            if name not in target_names:
                continue
            conf = float(box.conf[0])
            if conf > best_conf:
                best_conf = conf
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                best_det = Detection(x1, y1, x2, y2, conf, name, "yolo_custom")

        return best_det

    @staticmethod
    def _best_by_class_id(
        results,
        class_id: int,
        min_conf: float,
    ) -> Optional[Detection]:
        """Pick highest-confidence detection with a given COCO class id."""
        names     = results.names
        boxes     = results.boxes
        best_conf = min_conf - 1e-6
        best_det: Optional[Detection] = None

        for box in boxes:
            if int(box.cls[0]) != class_id:
                continue
            conf = float(box.conf[0])
            if conf > best_conf:
                best_conf = conf
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                name = names.get(class_id, str(class_id))
                best_det = Detection(x1, y1, x2, y2, conf, name, "yolo_coco")

        return best_det
