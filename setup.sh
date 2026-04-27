#!/usr/bin/env bash
# setup.sh — Install Python dependencies on Jetson Xavier NX (JetPack 5.x)
# Run once:  bash setup.sh
set -e

echo "=== Basketball Analytics — Jetson Setup ==="

# ── PyTorch for JetPack 5.x (aarch64) ──────────────────────────────────────
# If torch is already installed, skip.
if python3 -c "import torch; print('torch', torch.__version__)" 2>/dev/null; then
  echo "[OK] PyTorch already installed."
else
  echo "[*] Installing PyTorch for JetPack 5.x…"
  # Official NVIDIA wheel index for JetPack 5.1.x
  pip3 install --no-cache-dir \
    torch torchvision \
    --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch
fi

# ── Ultralytics (YOLOv8 / YOLO11) ───────────────────────────────────────────
echo "[*] Installing ultralytics…"
pip3 install --no-cache-dir "ultralytics>=8.2.0"

# ── Flask + CORS ─────────────────────────────────────────────────────────────
echo "[*] Installing Flask…"
pip3 install --no-cache-dir "flask>=3.0.0" "flask-cors>=4.0.0"

# ── Plotly + Pandas ───────────────────────────────────────────────────────────
echo "[*] Installing Plotly and Pandas…"
pip3 install --no-cache-dir "plotly>=5.18.0" "pandas>=1.5.0"

# ── Roboflow inference SDK ────────────────────────────────────────────────────
echo "[*] Installing Roboflow inference SDK…"
pip3 install --no-cache-dir "inference"

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Next steps:"
echo "  1. Copy your .svo files into  svo_files/"
echo "  2. (Optional) Run  python3 download_model.py  to get a custom basketball model"
echo "  3. Start the web server:  python3 web/app.py"
echo "  4. Or run the pipeline directly:  python3 main.py svo_files/<file>.svo"
echo ""
echo "Access the dashboard at: http://<jetson-ip>:5000"
