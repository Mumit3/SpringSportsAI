"""ZED 2i SVO file reader — wraps ZED SDK 5.x for frame-by-frame iteration."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np

try:
    import pyzed.sl as sl
    ZED_AVAILABLE = True
except ImportError:
    ZED_AVAILABLE = False
    print("[SVOReader] WARNING: pyzed not found. ZED features disabled.")

from . import config


@dataclass
class Frame:
    index: int
    image: np.ndarray        # (H, W, 3) BGR uint8
    depth: np.ndarray        # (H, W) float32, metres; NaN where invalid
    point_cloud: np.ndarray  # (H, W, 3) float32, XYZ metres in camera space
    timestamp_s: float       # seconds since epoch


@dataclass
class CameraInfo:
    width: int
    height: int
    fps: float
    fx: float
    fy: float
    cx: float
    cy: float
    total_frames: int

    @property
    def dt(self) -> float:
        return 1.0 / self.fps if self.fps > 0 else 1.0 / 30.0


class SVOReader:
    """Iterates over frames in a ZED SVO file.

    Usage::

        with SVOReader("path/to/file.svo") as reader:
            print(reader.info)
            for frame in reader:
                process(frame)
    """

    def __init__(self, svo_path: str, enable_body_tracking: bool = False):
        if not ZED_AVAILABLE:
            raise RuntimeError("pyzed is not installed. Install ZED SDK 5.x first.")

        self._zed = sl.Camera()
        self._body_tracking_ready = False

        init = sl.InitParameters()
        init.set_from_svo_file(str(svo_path))
        init.svo_real_time_mode     = False
        init.coordinate_units       = sl.UNIT.METER
        init.coordinate_system      = sl.COORDINATE_SYSTEM.LEFT_HANDED_Y_UP
        init.depth_minimum_distance = config.DEPTH_MIN
        init.depth_maximum_distance = config.DEPTH_MAX

        depth_map = {
            "NEURAL":      sl.DEPTH_MODE.NEURAL,
            "ULTRA":       sl.DEPTH_MODE.ULTRA,
            "QUALITY":     sl.DEPTH_MODE.QUALITY,
            "PERFORMANCE": sl.DEPTH_MODE.PERFORMANCE,
        }
        init.depth_mode = depth_map.get(config.DEPTH_MODE, sl.DEPTH_MODE.NEURAL)

        err = self._zed.open(init)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Cannot open SVO '{svo_path}': {err}")

        cam_cfg  = self._zed.get_camera_information().camera_configuration
        calib    = self._zed.get_camera_information().camera_configuration.calibration_parameters.left_cam

        self.info = CameraInfo(
            width        = cam_cfg.resolution.width,
            height       = cam_cfg.resolution.height,
            fps          = cam_cfg.fps,
            fx           = calib.fx,
            fy           = calib.fy,
            cx           = calib.cx,
            cy           = calib.cy,
            total_frames = self._zed.get_svo_number_of_frames(),
        )

        self._runtime = sl.RuntimeParameters()
        self._runtime.enable_depth = True

        self._mat_img = sl.Mat()
        self._mat_dep = sl.Mat()
        self._mat_pc  = sl.Mat()

        # Optional ZED body tracking. We enable it lazily so SVOReader works
        # with or without the body tracking module on this Jetson — if the
        # SDK rejects enable_body_tracking we just disable the feature and
        # the pipeline falls back to ball-only detection.
        if enable_body_tracking:
            try:
                # Positional tracking is required for body tracking in 3D world coords.
                tracking_params = sl.PositionalTrackingParameters()
                err = self._zed.enable_positional_tracking(tracking_params)
                if err != sl.ERROR_CODE.SUCCESS:
                    print(f"[SVOReader] positional tracking failed: {err} — body tracking disabled")
                else:
                    body_params = sl.BodyTrackingParameters()
                    body_params.enable_tracking      = True
                    body_params.enable_body_fitting  = False
                    body_params.detection_model      = sl.BODY_TRACKING_MODEL.HUMAN_BODY_MEDIUM
                    body_params.body_format          = sl.BODY_FORMAT.BODY_18
                    err = self._zed.enable_body_tracking(body_params)
                    if err != sl.ERROR_CODE.SUCCESS:
                        print(f"[SVOReader] body tracking enable failed: {err} — disabled")
                    else:
                        self._body_tracking_ready = True
                        self._bodies_runtime = sl.BodyTrackingRuntimeParameters()
                        self._bodies_runtime.detection_confidence_threshold = 40
                        self._bodies          = sl.Bodies()
                        print("[SVOReader] body tracking enabled (BODY_18, MEDIUM model)")
            except Exception as e:
                print(f"[SVOReader] body tracking init exception: {e} — disabled")
                self._body_tracking_ready = False

    # ── iteration ─────────────────────────────────────────────────────────────

    def __iter__(self):
        return self

    def __next__(self) -> Frame:
        err = self._zed.grab(self._runtime)
        if err != sl.ERROR_CODE.SUCCESS:
            raise StopIteration

        self._zed.retrieve_image(self._mat_img, sl.VIEW.LEFT)
        self._zed.retrieve_measure(self._mat_dep, sl.MEASURE.DEPTH)
        self._zed.retrieve_measure(self._mat_pc,  sl.MEASURE.XYZRGBA)

        bgr = self._mat_img.get_data()[:, :, :3].copy()          # drop alpha
        dep = self._mat_dep.get_data().copy().astype(np.float32)
        pc  = self._mat_pc.get_data()[:, :, :3].copy()           # XYZ only

        # Replace inf/negative with NaN so downstream code can use np.isnan
        dep[~np.isfinite(dep)] = np.nan
        dep[dep < config.DEPTH_MIN] = np.nan
        dep[dep > config.DEPTH_MAX] = np.nan

        idx = self._zed.get_svo_position()
        ts  = self._zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_seconds()

        return Frame(idx, bgr, dep, pc, ts)

    # ── helpers ───────────────────────────────────────────────────────────────

    def seek(self, frame_index: int) -> None:
        self._zed.set_svo_position(frame_index)

    def depth_at_pixel(
        self,
        depth: np.ndarray,
        u: int,
        v: int,
        patch: int = 11,
    ) -> Optional[float]:
        """Robust depth at (u, v) via patch median. Returns None if unreliable."""
        h, w = depth.shape
        r    = patch // 2
        roi  = depth[max(0, v-r):v+r+1, max(0, u-r):u+r+1]
        valid = roi[np.isfinite(roi)]
        return float(np.median(valid)) if len(valid) >= 4 else None

    def pixel_to_3d(
        self,
        pc: np.ndarray,
        u: int,
        v: int,
        patch: int = 9,
    ) -> Optional[np.ndarray]:
        """Return (X, Y, Z) world point for pixel (u, v), or None."""
        h, w = pc.shape[:2]
        if not (0 <= u < w and 0 <= v < h):
            return None

        r   = patch // 2
        roi = pc[max(0,v-r):v+r+1, max(0,u-r):u+r+1].reshape(-1, 3)
        ok  = roi[np.all(np.isfinite(roi), axis=1)]
        ok  = ok[(ok[:, 2] > config.DEPTH_MIN) & (ok[:, 2] < config.DEPTH_MAX)]

        if len(ok) < 4:
            return None
        return np.median(ok, axis=0)

    def retrieve_bodies(self) -> list:
        """Return the list of detected bodies for the most recently grabbed
        frame. Empty list if body tracking wasn't enabled or no bodies detected.

        Each item is a `pyzed.sl.BodyData`; the body_tracker module knows how
        to convert these into our snapshot format.
        """
        if not self._body_tracking_ready:
            return []
        try:
            err = self._zed.retrieve_bodies(self._bodies, self._bodies_runtime)
            if err != sl.ERROR_CODE.SUCCESS:
                return []
            return list(self._bodies.body_list)
        except Exception as e:
            print(f"[SVOReader] retrieve_bodies failed: {e}")
            return []

    @property
    def body_tracking_enabled(self) -> bool:
        return self._body_tracking_ready

    def project_3d_to_2d(self, xyz: np.ndarray) -> Optional[Tuple[int, int]]:
        """Project world point back to image pixel using camera intrinsics.

        Coordinate system: LEFT_HANDED_Y_UP
          X → right,  Y → up,  Z → forward (depth positive)
        Image coords: u → right, v → down (v = 0 at top)
        So: u = fx*X/Z + cx,  v = -fy*Y/Z + cy
        """
        x, y, z = xyz
        if z <= 0.05:
            return None
        u = int(round(self.info.fx * x / z + self.info.cx))
        v = int(round(-self.info.fy * y / z + self.info.cy))
        if 0 <= u < self.info.width and 0 <= v < self.info.height:
            return (u, v)
        return None

    # ── context manager ───────────────────────────────────────────────────────

    def close(self) -> None:
        self._zed.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
