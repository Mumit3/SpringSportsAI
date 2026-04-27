"""Ball and hoop detection.

Primary:  YOLOv8 (custom basketball.pt if present, else COCO yolov8n.pt)
Fallback: Colour-segmentation + Hough ellipse for hoop when YOLO confidence is low.

Detection sources are transparent to callers — the result just carries a
`source` field ('yolo_custom', 'yolo_coco', 'color') for diagnostics.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
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

        # Cached hoop state
        self._hoop_cache: Optional[Detection] = None
        self._hoop_stable_count = 0

        # Ball detection history for cleaning
        self._ball_history: List[Detection] = []

        self._load_model()

    # ── model loading ─────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        # Try Roboflow inference SDK first (downloads + caches model on first run)
        if config.USE_ROBOFLOW_API:
            try:
                from inference import get_model
                self._model      = get_model(
                    model_id = config.ROBOFLOW_MODEL_ID,
                    api_key  = config.ROBOFLOW_API_KEY,
                )
                self._model_type = "roboflow_api"
                print(f"[Detector] Roboflow API model loaded: {config.ROBOFLOW_MODEL_ID}")
                return
            except ImportError:
                print("[Detector] inference SDK not installed — run: pip3 install inference")
            except Exception as e:
                print(f"[Detector] Roboflow API unavailable ({e}). Falling back to local model.")

        # Fall back to local YOLO
        try:
            from ultralytics import YOLO
        except ImportError:
            print("[Detector] ultralytics not installed — run setup.sh first.")
            return

        custom = Path(config.YOLO_MODEL_PATH)
        if custom.exists():
            print(f"[Detector] Loading custom model: {custom}")
            self._model      = YOLO(str(custom))
            self._model_type = "custom"
        else:
            print(f"[Detector] Custom model not found. Using COCO {config.YOLO_FALLBACK}")
            self._model      = YOLO(config.YOLO_FALLBACK)
            self._model_type = "coco"

    # ── public API ────────────────────────────────────────────────────────────

    def detect(
        self,
        frame: np.ndarray,
        ball_conf_override: Optional[float] = None,
    ) -> Tuple[Optional[Detection], Optional[Detection], Optional[Detection]]:
        """Return (ball, hoop, ball_in_basket). Any may be None."""
        self._frame_count += 1

        if self._model_type == "roboflow_api":
            all_dets       = self._infer_roboflow(frame, ball_conf_override)
            ball           = self._clean_ball(
                self._filter_dets(all_dets, config.CUSTOM_BALL_NAMES,
                                  ball_conf_override or config.BALL_CONF_NORMAL)
            )
            ball_in_basket = self._filter_dets(
                all_dets, config.CUSTOM_BALL_IN_BASKET_NAMES, config.BALL_IN_BASKET_CONF
            )
            hoop = self._get_hoop(frame, None)   # model has no hoop class — use HSV
        else:
            # Run local YOLO once per frame at the lowest required confidence
            results = None
            if self._model is not None:
                min_conf = min(
                    ball_conf_override or config.BALL_CONF_NORMAL,
                    config.HOOP_CONF,
                    config.BALL_IN_BASKET_CONF,
                )
                results = self._model(frame, verbose=False, conf=min_conf)[0]

            ball           = self._clean_ball(self._detect_ball(results, ball_conf_override))
            hoop           = self._get_hoop(frame, results)
            ball_in_basket = self._detect_ball_in_basket(results)

        return ball, hoop, ball_in_basket

    # ── Roboflow API inference ────────────────────────────────────────────────

    def _infer_roboflow(
        self,
        frame: np.ndarray,
        conf_override: Optional[float] = None,
    ) -> List[Detection]:
        """Call hosted Roboflow model, return all detections as Detection objects."""
        min_conf = min(
            conf_override or config.BALL_CONF_NORMAL,
            config.BALL_IN_BASKET_CONF,
        )
        try:
            result = self._model.infer(frame, confidence=min_conf)[0]
        except Exception:
            return []

        dets = []
        for pred in result.predictions:
            x1 = int(pred.x - pred.width  / 2)
            y1 = int(pred.y - pred.height / 2)
            x2 = int(pred.x + pred.width  / 2)
            y2 = int(pred.y + pred.height / 2)
            dets.append(Detection(x1, y1, x2, y2,
                                  float(pred.confidence),
                                  pred.class_name.lower(),
                                  "roboflow_api"))
        return dets

    @staticmethod
    def _filter_dets(
        dets: List[Detection],
        target_names: set,
        min_conf: float,
    ) -> Optional[Detection]:
        """Pick highest-confidence detection whose class name is in target_names."""
        best_conf = min_conf - 1e-6
        best: Optional[Detection] = None
        for det in dets:
            if det.class_name in target_names and det.confidence > best_conf:
                best_conf = det.confidence
                best = det
        return best

    # ── ball detection ────────────────────────────────────────────────────────

    def _detect_ball(
        self,
        results,
        conf_override: Optional[float],
    ) -> Optional[Detection]:
        if results is None:
            return None
        conf = conf_override or config.BALL_CONF_NORMAL
        if self._model_type == "custom":
            return self._best_from_results(results, config.CUSTOM_BALL_NAMES, conf)
        return self._best_by_class_id(results, config.COCO_BALL_CLASS, conf)

    def _detect_ball_in_basket(self, results) -> Optional[Detection]:
        if results is None or self._model_type != "custom":
            return None
        return self._best_from_results(
            results, config.CUSTOM_BALL_IN_BASKET_NAMES, config.BALL_IN_BASKET_CONF
        )

    def _clean_ball(self, det: Optional[Detection]) -> Optional[Detection]:
        """Reject detections that are implausibly shaped or moving too fast."""
        if det is None:
            self._ball_history.append(None)
            if len(self._ball_history) > 30:
                self._ball_history.pop(0)
            return None

        # Reject non-round detections (ball should be roughly square bbox)
        aspect = det.w / max(det.h, 1)
        if aspect > 1.4 or aspect < 0.6:
            return None

        # Reject if ball moved more than 4x its diameter in the last 5 frames
        recent = [h for h in self._ball_history[-5:] if h is not None]
        if recent:
            diameter = (det.w + det.h) / 2
            last = recent[-1]
            dist = ((det.cx - last.cx)**2 + (det.cy - last.cy)**2) ** 0.5
            if dist > 4 * diameter:
                return None

        self._ball_history.append(det)
        if len(self._ball_history) > 30:
            self._ball_history.pop(0)
        return det

    # ── hoop detection ────────────────────────────────────────────────────────

    def _get_hoop(self, frame: np.ndarray, results=None) -> Optional[Detection]:
        """Return cached hoop or re-detect if stale."""
        if self._frame_count % config.HOOP_POLL_FRAMES != 0:
            return self._hoop_cache

        fresh = self._detect_hoop(frame, results)
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

    def _detect_hoop(self, frame: np.ndarray, results=None) -> Optional[Detection]:
        """Try YOLO first, fall back to colour segmentation."""
        det = self._detect_hoop_yolo(results)
        if det is not None:
            return det
        return self._detect_hoop_color(frame)

    def _detect_hoop_yolo(self, results) -> Optional[Detection]:
        if results is None or self._model_type != "custom":
            return None
        return self._best_from_results(results, config.CUSTOM_HOOP_NAMES, config.HOOP_CONF)

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
