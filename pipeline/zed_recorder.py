"""Live ZED 2i camera capture with optional SVO recording.

All ZED API calls happen inside a single capture thread to avoid the
non-thread-safety of pyzed.Camera. API methods (start_recording, stop_recording,
shutdown, get_state, get_jpeg_frame) only set commands or read state through
locks — they never touch the camera object directly.

State machine:
    IDLE → STARTING → PREVIEW → RECORDING → PREVIEW → IDLE (on shutdown)

The pipeline must call shutdown() before opening the same SVO file, so the
ZED SDK fully releases the camera handle.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    import pyzed.sl as sl
    ZED_AVAILABLE = True
except ImportError:
    ZED_AVAILABLE = False

from . import config


class ZedRecorder:
    """Manages live ZED capture + SVO recording for the Flask UI.

    Resolution and fps are passed when starting preview, so the UI can
    let the user pick from HD2K/HD1080/HD720 per session.
    """

    def __init__(self):
        if not ZED_AVAILABLE:
            raise RuntimeError("pyzed not available — install ZED SDK Python bindings")

        # Command channel (API thread → capture thread)
        self._cmd_lock = threading.Lock()
        self._cmd_record_start: Optional[Tuple[str, int]] = None
        self._cmd_record_stop  = False
        self._cmd_shutdown     = False

        # State (capture thread → API thread)
        self._state_lock = threading.Lock()
        self._state              = "IDLE"
        self._resolution         = config.RECORD_RESOLUTION
        self._fps                = config.RECORD_FPS
        self._recording_path: Optional[str] = None
        self._recording_started_at = 0.0
        self._recording_max_seconds = config.RECORDING_MAX_SECONDS
        self._completed_svo_path: Optional[str] = None
        self._error: Optional[str] = None

        # Shared preview frame
        self._frame_lock = threading.Lock()
        self._latest_jpeg: Optional[bytes] = None

        self._capture_thread: Optional[threading.Thread] = None

    # ── public API ────────────────────────────────────────────────────────────

    def start_preview(self, resolution: str = "HD2K", fps: int = 15) -> None:
        if self._capture_thread is not None and self._capture_thread.is_alive():
            return
        with self._state_lock:
            self._state      = "STARTING"
            self._error      = None
            self._completed_svo_path = None
            self._resolution = resolution
            self._fps        = fps
        with self._cmd_lock:
            self._cmd_shutdown     = False
            self._cmd_record_start = None
            self._cmd_record_stop  = False

        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()

    def start_recording(self, svo_path: str, max_seconds: Optional[int] = None) -> None:
        max_sec = max_seconds if max_seconds is not None else config.RECORDING_MAX_SECONDS
        with self._cmd_lock:
            self._cmd_record_start = (svo_path, int(max_sec))
        with self._state_lock:
            self._completed_svo_path = None

    def stop_recording(self) -> None:
        with self._cmd_lock:
            self._cmd_record_stop = True

    def shutdown(self, timeout: float = 4.0) -> Optional[str]:
        """Stop recording, close the camera, fully release the handle.

        Must be called before the analytics pipeline opens the SVO — the
        SDK refuses to open a file while the camera handle is held.
        Returns the path of the most recently completed SVO, or None.
        """
        with self._cmd_lock:
            self._cmd_shutdown = True
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=timeout)
            self._capture_thread = None
        with self._state_lock:
            return self._completed_svo_path

    def get_state(self) -> dict:
        with self._state_lock:
            elapsed = 0
            if self._state == "RECORDING":
                elapsed = int(time.time() - self._recording_started_at)
            return {
                "state":           self._state,
                "elapsed_seconds": elapsed,
                "max_seconds":     self._recording_max_seconds,
                "completed_svo":   self._completed_svo_path,
                "resolution":      self._resolution,
                "fps":             self._fps,
                "error":           self._error,
            }

    def get_jpeg_frame(self) -> Optional[bytes]:
        with self._frame_lock:
            return self._latest_jpeg

    # ── capture thread ────────────────────────────────────────────────────────

    def _capture_loop(self) -> None:
        zed: Optional["sl.Camera"] = None
        try:
            zed = self._open_camera()
        except Exception as e:
            with self._state_lock:
                self._state = "ERROR"
                self._error = str(e)
            print(f"[ZedRecorder] Failed to open camera: {e}")
            return

        with self._state_lock:
            self._state = "PREVIEW"

        mat_img = sl.Mat()
        runtime = sl.RuntimeParameters()

        recording      = False
        recording_path = None
        recording_max  = config.RECORDING_MAX_SECONDS
        recording_t0   = 0.0

        try:
            while True:
                with self._cmd_lock:
                    start_cmd = self._cmd_record_start
                    stop_cmd  = self._cmd_record_stop
                    shutdown  = self._cmd_shutdown
                    self._cmd_record_start = None
                    self._cmd_record_stop  = False

                if shutdown:
                    break

                if start_cmd is not None and not recording:
                    path, max_sec = start_cmd
                    p = Path(path)
                    p.parent.mkdir(parents=True, exist_ok=True)
                    if p.exists():
                        try:
                            p.unlink()
                        except Exception as e:
                            print(f"[ZedRecorder] Could not delete old SVO: {e}")

                    rec_params = sl.RecordingParameters()
                    rec_params.video_filename = str(p)
                    rec_params.compression_mode = sl.SVO_COMPRESSION_MODE.H264

                    err = zed.enable_recording(rec_params)
                    if err == sl.ERROR_CODE.SUCCESS:
                        recording      = True
                        recording_path = str(p)
                        recording_max  = max_sec
                        recording_t0   = time.time()
                        with self._state_lock:
                            self._state                = "RECORDING"
                            self._recording_path       = recording_path
                            self._recording_started_at = recording_t0
                            self._recording_max_seconds = recording_max
                        print(f"[ZedRecorder] Recording → {recording_path} (max {max_sec}s)")
                    else:
                        with self._state_lock:
                            self._error = f"enable_recording failed: {err}"
                        print(f"[ZedRecorder] enable_recording failed: {err}")

                if recording and (time.time() - recording_t0) >= recording_max:
                    stop_cmd = True

                if stop_cmd and recording:
                    zed.disable_recording()
                    with self._state_lock:
                        self._state = "PREVIEW"
                        self._completed_svo_path = recording_path
                    print(f"[ZedRecorder] Recording stopped — {recording_path}")
                    recording      = False
                    recording_path = None

                err = zed.grab(runtime)
                if err == sl.ERROR_CODE.SUCCESS:
                    zed.retrieve_image(mat_img, sl.VIEW.LEFT)
                    bgr = mat_img.get_data()[:, :, :3].copy()
                    self._encode_preview(bgr)

                time.sleep(0.001)
        except Exception as e:
            print(f"[ZedRecorder] Capture loop error: {e}")
            with self._state_lock:
                self._state = "ERROR"
                self._error = str(e)
        finally:
            try:
                if recording:
                    zed.disable_recording()
                    with self._state_lock:
                        self._completed_svo_path = recording_path
            except Exception:
                pass
            try:
                if zed is not None:
                    zed.close()
            except Exception:
                pass
            with self._state_lock:
                self._state = "IDLE"
            print("[ZedRecorder] Camera released")

    # ── helpers ───────────────────────────────────────────────────────────────

    def _open_camera(self) -> "sl.Camera":
        zed = sl.Camera()
        init = sl.InitParameters()

        res_map = {
            "HD2K":   sl.RESOLUTION.HD2K,
            "HD1080": sl.RESOLUTION.HD1080,
            "HD720":  sl.RESOLUTION.HD720,
        }
        with self._state_lock:
            res = self._resolution
            fps = self._fps
        init.camera_resolution = res_map.get(res, sl.RESOLUTION.HD2K)
        init.camera_fps        = int(fps)

        depth_map = {
            "NEURAL":      sl.DEPTH_MODE.NEURAL,
            "ULTRA":       sl.DEPTH_MODE.ULTRA,
            "QUALITY":     sl.DEPTH_MODE.QUALITY,
            "PERFORMANCE": sl.DEPTH_MODE.PERFORMANCE,
        }
        init.depth_mode             = depth_map.get(config.DEPTH_MODE, sl.DEPTH_MODE.NEURAL)
        init.coordinate_system      = sl.COORDINATE_SYSTEM.LEFT_HANDED_Y_UP
        init.coordinate_units       = sl.UNIT.METER
        init.depth_minimum_distance = config.DEPTH_MIN
        init.depth_maximum_distance = config.DEPTH_MAX

        err = zed.open(init)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED open failed: {err}")
        print(f"[ZedRecorder] Camera opened ({res} @ {fps} fps)")
        return zed

    def _encode_preview(self, bgr: np.ndarray) -> None:
        h, w = bgr.shape[:2]
        max_w = config.PREVIEW_MAX_WIDTH
        if w > max_w:
            scale = max_w / w
            bgr = cv2.resize(bgr, (max_w, int(h * scale)))
        ok, jpeg = cv2.imencode(".jpg", bgr,
                                [int(cv2.IMWRITE_JPEG_QUALITY), config.PREVIEW_JPEG_QUALITY])
        if ok:
            with self._frame_lock:
                self._latest_jpeg = jpeg.tobytes()
