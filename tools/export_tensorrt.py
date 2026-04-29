"""Export a YOLO PyTorch model to a Jetson-optimised TensorRT engine.

Run ONCE on the Jetson (not on Windows — engines are hardware-specific):

    python tools/export_tensorrt.py

This builds yolov8s.engine alongside yolov8s.pt. The detector picks it up
automatically on subsequent pipeline runs.

The first build takes 5-15 minutes on Xavier NX. After that, inference is
roughly 3-5x faster than the .pt model with no accuracy loss.
"""
from __future__ import annotations
import sys
from pathlib import Path

# Make the pipeline package importable when running from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import config


def _find_pt(name: str) -> Path:
    """Look for the .pt file in the repo root, then ultralytics cache."""
    here = Path.cwd() / name
    if here.exists():
        return here

    # ultralytics caches downloads in ~/.config/Ultralytics/<name>
    cache = Path.home() / ".config" / "Ultralytics" / name
    if cache.exists():
        return cache

    raise FileNotFoundError(
        f"{name} not found in {here} or {cache}. "
        f"Run the pipeline once first so ultralytics downloads it."
    )


def main() -> None:
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[Export] ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    pt_path = _find_pt(config.YOLO_FALLBACK)
    print(f"[Export] Source: {pt_path}")

    model = YOLO(str(pt_path))

    print(f"[Export] Building TensorRT engine at imgsz={config.YOLO_IMGSZ} (FP16)…")
    print("[Export] This takes 5-15 minutes on first build. Sit tight.")

    engine = model.export(
        format   = "engine",
        imgsz    = config.YOLO_IMGSZ,
        half     = True,    # FP16 on Jetson
        workspace= 4,       # GB of TRT workspace
        verbose  = False,
    )

    print(f"\n[Export] Done. Engine: {engine}")
    print("[Export] Next pipeline run will load this engine automatically.")


if __name__ == "__main__":
    main()
