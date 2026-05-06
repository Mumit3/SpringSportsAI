from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).parent.parent
SVO_DIR    = ROOT_DIR / "svo_files"
OUTPUT_DIR = ROOT_DIR / "outputs"
MODELS_DIR = ROOT_DIR / "models"

# Curated SVO folders shown as a dropdown in the web UI. First entry is the
# default. The folder names are real top-level directory names under ROOT_DIR;
# the labels are what the dropdown displays.
SVO_FOLDERS = [
    ("expo",          "Expo"),
    ("svo_files",     "Default"),
    ("new_svo_files", "New Recordings"),
    ("mini",          "Mini Hoop"),
]
SVO_FOLDER_DEFAULT = SVO_FOLDERS[0][0]
SVO_FOLDER_NAMES = [name for name, _ in SVO_FOLDERS]

# Default model: YOLOv8s pretrained on COCO (auto-downloaded by ultralytics).
# To use a custom basketball model, set this to the .pt file path.
YOLO_MODEL_PATH = str(MODELS_DIR / "basketball.pt")   # custom if exists
YOLO_FALLBACK   = "yolov8s.pt"                        # COCO fallback (small ≈ 3-4× better small-object recall than nano)

# TensorRT acceleration on Jetson. If a yolov8s.engine file exists alongside
# the .pt, the detector loads the engine instead — typically 3-5× faster on
# Xavier NX with no accuracy loss. Build it with: python tools/export_tensorrt.py
USE_TENSORRT = True

# YOLO inference resolution. Default is 640 — too small for a distant ball in
# HD2K (ball becomes ~9 px). 1280 keeps it ~18 px and significantly improves
# recall against busy backgrounds. Must be multiple of 32.
YOLO_IMGSZ = 1280

# ── ZED SDK ──────────────────────────────────────────────────────────────────
DEPTH_MODE  = "NEURAL"   # NEURAL | ULTRA | QUALITY | PERFORMANCE
COORD_SYSTEM = "LEFT_HANDED_Y_UP"
DEPTH_MIN   = 0.3        # metres
DEPTH_MAX   = 20.0       # metres

# ── Live recording (Flask UI) ────────────────────────────────────────────────
RECORD_RESOLUTION       = "HD2K"   # default; UI lets user override
RECORD_FPS              = 15       # default; clamped to camera limits per resolution
RECORDING_MAX_SECONDS   = 60
PREVIEW_MAX_WIDTH       = 800      # downscale MJPEG preview for browser bandwidth
PREVIEW_JPEG_QUALITY    = 70

# ── Detection thresholds ─────────────────────────────────────────────────────
BALL_CONF_NORMAL   = 0.35
BALL_CONF_FLIGHT   = 0.10   # lowered while ball is confirmed in-flight
# Recovery mode: when the tracker has missed the ball for many consecutive
# frames, drop YOLO confidence aggressively to recover faint detections.
# Working videos never enter this mode (they don't lose the ball for long),
# so this can't regress them — it only kicks in when detection is failing.
BALL_CONF_RECOVERY = 0.05
BALL_RECOVERY_MISS_THRESHOLD = 5   # consecutive missed frames before recovery
HOOP_CONF        = 0.25
HOOP_POLL_FRAMES = 5      # re-run hoop detector every N frames (post-calibration)

# Hoop calibration: detect every frame for first N frames, cluster, then LOCK
HOOP_CALIBRATION_FRAMES = 60
HOOP_CLUSTER_TOLERANCE_PX = 30   # detections within this distance count as same hoop

# COCO class indices (used when falling back to yolov8n.pt)
COCO_BALL_CLASS = 32      # "sports ball"
COCO_PERSON_CLASS = 0     # "person"

# Custom model class names (set to match your .pt label order)
CUSTOM_BALL_NAMES = {"basketball", "ball"}
CUSTOM_HOOP_NAMES = {"basketball-hoop", "hoop", "rim", "net"}

# ── Kalman filter ─────────────────────────────────────────────────────────────
KALMAN_PROCESS_NOISE     = 2.0   # higher = trusts measurements more
KALMAN_MEASUREMENT_NOISE = 0.04  # metres std-dev (ZED depth accuracy)
GRAVITY = 9.81                   # m/s²

# ── Ball tracker ──────────────────────────────────────────────────────────────
TRAIL_LENGTH          = 45   # frames of position history to show
PREDICT_AHEAD_FRAMES  = 30   # frames to project forward for arc preview
OF_MAX_MISSED_FRAMES  = 20   # use optical flow up to this many missed frames
KF_MAX_MISSED_FRAMES  = 60   # keep Kalman-only estimate up to this limit

# Depth-based ball size validation
BALL_DIAMETER_M       = 0.24   # regulation basketball
BALL_SIZE_TOLERANCE   = 0.7    # accept ±70 % of expected pixel size at given depth

# 3-D plausibility filter — reject ball detections at impossible positions
# (a real basketball during a shot can't be at floor level, right against the
# camera lens, or wildly out of plane with the hoop). yolov8s @ 1280 picks up
# more low-confidence false positives than yolov8n @ 640, so this filter
# becomes important to suppress them before the tracker locks onto noise.
BALL_MIN_Y             = 0.3   # metres above floor — below this is dribble/floor
BALL_MIN_Z             = 1.5   # metres from camera — closer than this is a hand
BALL_HOOP_Z_TOLERANCE  = 6.0   # metres — ball's Z must be within this of hoop's Z

# ── Background masking (preprocessing for YOLO) ──────────────────────────────
# When enabled, frames are pre-processed before detection: pixels far from the
# hoop's depth and not orange-ish are darkened. The original frame in the
# annotated output video is unchanged — this only affects what YOLO sees.
# Off by default; enable to test on videos where busy backgrounds are hurting
# detection. Risk: YOLO is trained on natural images, so heavily masked input
# can occasionally hurt instead of help — flip back to False if it regresses.
BACKGROUND_MASK_ENABLED        = False
BACKGROUND_DIM_FACTOR          = 0.30   # 0.0 = pitch black bg, 1.0 = unchanged
BACKGROUND_DEPTH_NEAR          = 0.5    # always keep pixels from this distance
BACKGROUND_DEPTH_BEHIND_HOOP   = 1.5    # keep up to N metres past hoop (catches rim, person holding it)
BACKGROUND_DEPTH_FALLBACK_MAX  = 12.0   # cap when hoop hasn't locked yet
BACKGROUND_MASK_DILATE_PX      = 9      # grow mask by N pixels so ball edges aren't shaved

# HSV range for basketball orange — used by the background mask to keep ball
# pixels visible regardless of depth. Tuned for typical indoor lighting; if
# your venue has very warm/cool lights and the ball gets dimmed, widen the
# H range slightly.
BALL_HSV_LOWER = (5, 100, 80)
BALL_HSV_UPPER = (25, 255, 255)

# ── ZED body tracking ────────────────────────────────────────────────────────
# When enabled, the pipeline runs ZED's body tracking module each frame,
# identifies the shooter (the body furthest from the rim), and detects the
# release moment from wrist motion. Detected release events override the
# late Method A / Method B triggers — the shot's release_pos becomes the
# wrist position at apex, and the visual trail is backfilled with wrist
# history so the upward arc is visible.
#
# Cost: roughly +50–100 ms per frame on Jetson Xavier NX. Falls back
# gracefully if the body tracking module isn't available on this machine.
# Off by default — flip to True to enable.
BODY_TRACKING_ENABLED = True

# Hough Circle recovery — search for ball as a circle in Kalman-predicted ROI
HOUGH_ROI_FACTOR      = 5.0    # ROI half-size = factor × expected radius
HOUGH_RADIUS_MIN_FRAC = 0.7    # search radii from 0.7× to 1.4× expected
HOUGH_RADIUS_MAX_FRAC = 1.4
HOUGH_ACCUMULATOR_THR = 20     # cv2.HoughCircles param2

# Depth-validated optical flow — reject OF if measured depth differs from prediction
OF_DEPTH_TOLERANCE_M  = 1.2    # metres

# ── Shot detector ─────────────────────────────────────────────────────────────
ARC_VELOCITY_THRESHOLD = 1.0   # m/s upward Kalman velocity → shot triggered
ARC_MIN_FRAMES         = 8     # arc must last ≥ this many frames to be valid
ARC_MAX_FRAMES         = 150   # arc auto-terminates after this many frames
PIXEL_RISE_THRESHOLD   = 0.10  # fraction of frame height risen in window
PIXEL_RISE_WINDOW      = 8     # rolling frame window for 2D method
SHOT_COOLDOWN_FRAMES   = 40    # minimum frames between detected shots (1.33 s @ 30 fps)

# Make/miss: ball must pass through hoop plane within this radius (metres).
# Real rim radius is 0.23 m, but yolov8s @ imgsz=1280 places ball/hoop bboxes a
# few pixels different from yolov8n @ 640, which shifts the 3-D hoop center by
# a few cm. 0.30 m absorbs that shift so makes still register.
MAKE_CYLINDER_RADIUS = 0.30
# How many frames after shot release to keep checking for make/miss
MAKE_CHECK_FRAMES = 60

# Post-rim verification — once the ball goes below the rim, the zone
# classifier waits this many frames AND watches the horizontal drift before
# declaring a MAKE. This catches deflections where the ball briefly dips
# below then flies sideways: the ball's horizontal distance from the rim
# centre should stay within MAKE_POST_RIM_DRIFT_M for those frames, otherwise
# it's classified as a MISS.
MAKE_POST_RIM_FRAMES   = 3       # ~100 ms at 30 fps
MAKE_POST_RIM_DRIFT_M  = 0.50    # 50 cm: real makes drop straight down through the net

# Stricter values used only when a job opts in via the per-card "Strict
# make/miss" checkbox. Tightening these makes MAKE harder to fire and so
# reduces false MAKEs on rim-outs and balls that just pass near the rim.
STRICT_MAKE_POST_RIM_DRIFT_M = 0.35
STRICT_MAKE_BBOX_3D_GATE_M   = 0.30

# 2-D bbox MAKE classifier — when ball's 3D position is also known, require
# the ball to be horizontally near the hoop's 3D centre. Stops false MAKEs
# from balls that pass IN FRONT of the rim and happen to overlap its pixel
# bbox without actually going through.
MAKE_BBOX_3D_GATE_M = 0.50

# Method C — trajectory-direction shot trigger. Fires when the ball is
# clearly headed toward the rim from a distance, even if body tracking
# missed the release. All conditions must hold simultaneously.
SHOT_TRIGGER_MIN_RIM_DIST_M  = 1.0    # ball must be at least this far from rim (no layup-range false fires)
SHOT_TRIGGER_MIN_VY_MPS      = 0.5    # ball must be ascending in 3D
SHOT_TRIGGER_MIN_TOWARD_MPS  = 1.0    # horizontal velocity component toward rim
SHOT_TRIGGER_OBS_FRAMES      = 5      # window of recent observations to compute velocity from

# Print per-shot diagnostic block at finalise time. Useful for figuring out
# why a shot was classified the way it was — shows closest 3-D approach,
# cylinder entry, 2-D bbox proximity, and which rule fired.
DEBUG_SHOT_DETECTION = True
# How many frames of pre-trigger tracker state to keep in the rolling buffer
# (printed at trigger time). At 15 fps, 30 frames = 2 seconds of context.
DEBUG_PRETRIGGER_FRAMES = 30

# ── Hoop colour segmentation fallback ────────────────────────────────────────
# HSV range for the orange steel rim
HOOP_HSV_LOWER1 = (0,  90, 80)
HOOP_HSV_UPPER1 = (22, 255, 255)
HOOP_HSV_LOWER2 = (158, 90, 80)
HOOP_HSV_UPPER2 = (180, 255, 255)
HOOP_DEPTH_MIN  = 2.5    # metres — hoop is always this far away
HOOP_DEPTH_MAX  = 14.0   # metres

# ── Annotation colours (BGR) ──────────────────────────────────────────────────
COLOR_BALL_YOLO    = (0,   165, 255)   # orange
COLOR_BALL_OF      = (0,   215, 255)   # gold
COLOR_BALL_HOUGH   = (255, 200,   0)   # bright cyan
COLOR_BALL_KALMAN  = (255, 255,   0)   # cyan
COLOR_HOOP         = (0,   165, 255)   # orange
COLOR_TRAIL_HEAD   = (255, 255, 255)   # white
COLOR_TRAIL_TAIL   = (50,  100, 200)   # faded blue
COLOR_ARC_PRED     = (180, 180, 180)   # light grey dashed
COLOR_MAKE         = (50,  205,  50)   # green
COLOR_MISS         = (50,   50, 220)   # red
COLOR_PENDING      = (200, 200,   0)   # yellow
