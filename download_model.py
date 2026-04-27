"""One-time model download script.

Downloads the basketball + hoop YOLOv8 model from Roboflow Universe and saves
the weights to  models/basketball.pt.  After this runs once, the pipeline
operates fully offline.

Usage:
    python3 download_model.py --api-key YOUR_ROBOFLOW_API_KEY

If you already exported a .pt file from Roboflow, just copy it to
models/basketball.pt and skip this script.
"""
import argparse
import shutil
import sys
from pathlib import Path


MODELS_DIR  = Path(__file__).parent / "models"
OUTPUT_PATH = MODELS_DIR / "basketball.pt"

# Roboflow project details
WORKSPACE   = "roboflow-jvuqo"
PROJECT     = "basketball-player-detection-3-ycjdo"
VERSION     = 4
FORMAT      = "yolov8"


def download(api_key: str) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        from roboflow import Roboflow
    except ImportError:
        print("[ERROR] roboflow package not installed.")
        print("        Run:  pip3 install roboflow")
        sys.exit(1)

    print(f"[*] Connecting to Roboflow (project: {PROJECT} v{VERSION})…")
    rf      = Roboflow(api_key=api_key)

    # Try the workspace from the report first, fall back to auto-detect
    try:
        project = rf.workspace(WORKSPACE).project(PROJECT)
    except Exception:
        project = rf.project(PROJECT)

    version = project.version(VERSION)

    print(f"[*] Downloading {FORMAT} dataset to models/…")
    dataset = version.download(FORMAT, location=str(MODELS_DIR / "dataset"))

    # Extract zip if Roboflow didn't unpack it automatically
    dataset_dir = Path(dataset.location)
    for zf in dataset_dir.rglob("*.zip"):
        print(f"[*] Extracting {zf} …")
        import zipfile
        with zipfile.ZipFile(zf, "r") as z:
            z.extractall(dataset_dir)
        zf.unlink()

    pt_candidates = list(dataset_dir.rglob("*.pt"))

    if pt_candidates:
        src = pt_candidates[0]
        shutil.copy(src, OUTPUT_PATH)
        print(f"[OK] Model weights saved to {OUTPUT_PATH}")
        print(f"     (source: {src})")
    else:
        # The dataset download gives us label data only — need to train briefly
        print("[!] No pre-trained weights found in download.")
        print("    Training YOLOv8n for 30 epochs on the downloaded dataset…")
        _train(dataset_dir)


def _train(dataset_dir: Path) -> None:
    """Quick fine-tune of YOLOv8n on the downloaded dataset."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics not installed. Run: pip3 install ultralytics")
        sys.exit(1)

    yaml_files = list(dataset_dir.rglob("data.yaml"))
    if not yaml_files:
        print("[ERROR] data.yaml not found in downloaded dataset.")
        sys.exit(1)

    yaml_path = yaml_files[0]
    print(f"[*] Training on {yaml_path} …")

    model = YOLO("yolov8n.pt")
    model.train(
        data    = str(yaml_path),
        epochs  = 30,
        imgsz   = 640,
        batch   = 8,
        device  = 0,           # CUDA GPU 0
        project = str(MODELS_DIR),
        name    = "train",
        verbose = False,
    )

    # Copy best weights
    trained_pt = MODELS_DIR / "train" / "weights" / "best.pt"
    if trained_pt.exists():
        shutil.copy(trained_pt, OUTPUT_PATH)
        print(f"[OK] Trained weights saved to {OUTPUT_PATH}")
    else:
        print("[WARN] Training completed but best.pt not found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", required=True, help="Roboflow API key")
    args = parser.parse_args()
    download(args.api_key)
