from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).parent.parent
SVO_DIR    = ROOT_DIR / "svo_files"
OUTPUT_DIR = ROOT_DIR / "outputs"
MODELS_DIR = ROOT_DIR / "models"

# Default model: YOLOv8n pretrained on COCO (auto-downloaded by ultralytics).
# To use a custom basketball model, set this to the .pt file path.
YOLO_MODEL_PATH = str(MODELS_DIR / "basketball.pt")   # custom if exists
YOLO_FALLBACK   = "yolov8n.pt"                        # COCO fallback

# ── ZED SDK ──────────────────────────────────────────────────────────────────
DEPTH_MODE  = "NEURAL"   # NEURAL | ULTRA | QUALITY | PERFORMANCE
COORD_SYSTEM = "LEFT_HANDED_Y_UP"
DEPTH_MIN   = 0.3        # metres
DEPTH_MAX   = 20.0       # metres

# ── Detection thresholds ─────────────────────────────────────────────────────
BALL_CONF_NORMAL = 0.35
BALL_CONF_FLIGHT = 0.18   # lowered while ball is confirmed in-flight
HOOP_CONF        = 0.25
HOOP_POLL_FRAMES = 5      # re-run hoop detector every N frames

# COCO class indices (used when falling back to yolov8n.pt)
COCO_BALL_CLASS = 32      # "sports ball"
COCO_PERSON_CLASS = 0     # "person"

# Custom model class names (set to match your .pt label order)
CUSTOM_BALL_NAMES          = {"basketball", "ball"}
CUSTOM_HOOP_NAMES          = {"basketball-hoop", "hoop", "rim", "net"}
CUSTOM_BALL_IN_BASKET_NAMES = {"ball-in-basket"}
BALL_IN_BASKET_CONF        = 0.35

# ── Kalman filter ─────────────────────────────────────────────────────────────
KALMAN_PROCESS_NOISE     = 2.0   # higher = trusts measurements more
KALMAN_MEASUREMENT_NOISE = 0.04  # metres std-dev (ZED depth accuracy)
GRAVITY = 9.81                   # m/s²

# ── Ball tracker ──────────────────────────────────────────────────────────────
TRAIL_LENGTH          = 45   # frames of position history to show
PREDICT_AHEAD_FRAMES  = 30   # frames to project forward for arc preview
OF_MAX_MISSED_FRAMES  = 20   # use optical flow up to this many missed frames
KF_MAX_MISSED_FRAMES  = 60   # keep Kalman-only estimate up to this limit

# ── Shot detector ─────────────────────────────────────────────────────────────
ARC_VELOCITY_THRESHOLD = 1.0   # m/s upward Kalman velocity → shot triggered
ARC_MIN_FRAMES         = 8     # arc must last ≥ this many frames to be valid
ARC_MAX_FRAMES         = 150   # arc auto-terminates after this many frames
PIXEL_RISE_THRESHOLD   = 0.10  # fraction of frame height risen in window
PIXEL_RISE_WINDOW      = 8     # rolling frame window for 2D method
SHOT_COOLDOWN_FRAMES   = 70    # minimum frames between detected shots

# Make/miss: ball must pass through hoop plane within this radius (metres)
MAKE_CYLINDER_RADIUS = 0.23
# How many frames after shot release to keep checking for make/miss
MAKE_CHECK_FRAMES = 60

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
COLOR_BALL_KALMAN  = (255, 255,   0)   # cyan
COLOR_HOOP         = (0,   165, 255)   # orange
COLOR_TRAIL_HEAD   = (255, 255, 255)   # white
COLOR_TRAIL_TAIL   = (50,  100, 200)   # faded blue
COLOR_ARC_PRED     = (180, 180, 180)   # light grey dashed
COLOR_MAKE         = (50,  205,  50)   # green
COLOR_MISS         = (50,   50, 220)   # red
COLOR_PENDING      = (200, 200,   0)   # yellow
