# Ai Assisted Sports Analytics

**AI-Assisted Basketball Shot Analytics**

An offline computer vision pipeline that processes stereo video recordings of basketball shooting sessions and produces per-shot analytics: release angle, arc height, shot distance, release speed, and make/miss classification.

Senior Design Project SDP-113, Spring 2026 — Binghamton University, Thomas J. Watson College of Engineering and Applied Science.

---

## Hardware requirements

- **NVIDIA Jetson Xavier NX** (8 GB or 16 GB) on a carrier board with USB 3.0
- **Stereolabs ZED 2i** stereo camera, connected via USB 3.0
- Tripod mount, kept stationary throughout each recording
- Wall power for the Jetson (5V/4A barrel jack)
- A device on the same network (phone, tablet, or laptop) to access the dashboard

The pipeline requires a CUDA-capable GPU. There is no CPU-only fallback.

---

## Software requirements

Install these in order before cloning the repo. Steps 1, 2, and 7 install software outside the repo. Steps 3–6 happen inside the cloned repo.

### 1. NVIDIA JetPack 5.1.2

The Jetson must be flashed with **JetPack 5.1.2** specifically. This bundles Ubuntu 20.04, Python 3.8, CUDA 11.4, cuDNN 8.6, and TensorRT 8.5 — the exact versions the rest of the stack depends on.

Download from NVIDIA's SDK Manager:
https://developer.nvidia.com/embedded/jetpack-sdk-512

Follow NVIDIA's flashing instructions for the Jetson Xavier NX. Plan for 1–2 hours.

### 2. Stereolabs ZED SDK

The ZED SDK provides camera drivers, depth processing, and body tracking. It is a system-level install — not a Python package — and must be installed before cloning this repo.

Download the **ZED SDK 5.x build for JetPack 5.1.2** from:
https://www.stereolabs.com/developers/release/

Then on the Jetson:

```bash
chmod +x ZED_SDK_Tegra_L4T35.4_v5.0.run
./ZED_SDK_Tegra_L4T35.4_v5.0.run
```

Verify by plugging in the ZED 2i and launching:

```bash
/usr/local/zed/tools/ZED_Explorer
```

You should see live left+right video and a depth map. If the depth map is black, the camera is on a USB 2.0 port — switch to one of the blue USB 3.0 ports.

### 3. Clone this repository

If git is not already installed:
```bash
sudo apt update
sudo apt install -y git
```

Then clone:
```bash
git clone https://github.com/<your-username>/SpringSportsAI.git
cd SpringSportsAI
```

(Replace `<your-username>` with the actual GitHub URL.)

### 4. Set up the Python environment

The project uses Python 3.8 (the version JetPack 5.1.2 ships with).

```bash
python3.8 -m venv env
source env/bin/activate
pip install --upgrade pip wheel
pip install -r requirements.txt
```

### 5. Install the ZED Python bindings

The `pyzed` package does not come from PyPI. It ships with the ZED SDK and must be built against your active Python interpreter.

With the virtual environment still activated:

```bash
cd /usr/local/zed
python3 get_python_api.py
cd -
```

Verify:
```bash
python -c "import pyzed.sl as sl; print(sl.Camera.get_sdk_version())"
```

### 6. Set up the YOLO model

The detector uses a custom YOLOv8s model trained on basketball footage. Place the weights file (`yolov8s.pt`) in the `models/` directory.

For deployment on the Jetson, export the weights to a TensorRT engine for a 3–5× speedup:

```bash
yolo export model=models/yolov8s.pt format=engine device=0 half=True
```

This produces `models/yolov8s.engine`, which the pipeline will use automatically when present.

> **Note:** TensorRT engines are not portable across CUDA versions. After any JetPack upgrade, regenerate the engine.

### 7. Install FFmpeg

```bash
sudo apt update
sudo apt install -y ffmpeg
```

---

## Quick install (alternative)

If steps 1 and 2 are already done, the rest can be run as a single script:

```bash
git clone https://github.com/<your-username>/SpringSportsAI.git
cd SpringSportsAI
chmod +x install.sh
./install.sh
```

The script handles steps 4–7. You'll still need to drop `yolov8s.pt` into `models/` before it can export the TensorRT engine.

---

## Running the code

Always activate the virtual environment first:

```bash
cd ~/SpringSportsAI
source env/bin/activate
```

### Option A — Web dashboard (recommended)

```bash
python -m web.app
```

The dashboard is available at:

```
http://<jetson-ip>:5000
```

To find the Jetson's IP:
```bash
hostname -I
```

From any phone, tablet, or laptop on the same network, open that URL in a browser. From the dashboard you can:

- Record a new session live (`/record` page)
- Process an existing SVO file from the file picker
- View past sessions with annotated video, summary stats, and 2D + 3D trajectory plots

### Option B — Command line

For batch processing or scripted use:

```bash
python main.py svo_files/your_recording.svo2 \
    --label session_name \
    --profile regulation
```

Flags:
- `--label NAME` — name for this session, used in output filenames
- `--profile {regulation,mini}` — `regulation` for full-size court, `mini` for office testing with a small hoop
- `--ball-only` — skip body tracking (faster, less accurate release detection)
- `--strict-make` — tighter make/miss thresholds

### Recording new sessions

You can record SVO files in two ways:

**From the web dashboard:** open `/record`, enter a label, pick a resolution, hit Start. The file is saved into the chosen folder under the project directory.

**From the ZED Explorer tool:** use the record button in `/usr/local/zed/tools/ZED_Explorer` and save the resulting `.svo2` into `svo_files/`.

Either way, recordings appear automatically in the dashboard's file picker.

---

## Where output files go

After processing, look in:

```
outputs/
├── videos/
│   └── <session>.mp4              # Annotated video
└── data/
    └── <session>/
        ├── analytics.json         # Session summary + per-shot data
        ├── shots.csv              # Per-shot metrics (open in Excel)
        ├── traces.json            # 2D trajectory data
        └── traces_3d.json         # 3D trajectory data
```

---

## Configuration

Tunable parameters live in `pipeline/config.py` — detection thresholds, make/miss tolerances, etc. Edit and re-run; no rebuild needed.

---

## Troubleshooting

**ZED camera not detected.** The ZED 2i requires USB 3.0. Confirm the cable is in a blue USB 3.0 port. The ZED Explorer will see the camera on USB 2.0 but fail to open the depth stream.

**`ImportError: No module named 'pyzed.sl'`** Re-run step 5 from inside the activated venv. The bindings must be regenerated after every ZED SDK upgrade.

**TensorRT engine load error.** Regenerate the engine on the deployment device:
```bash
yolo export model=models/yolov8s.pt format=engine device=0 half=True
```

**Flask port 5000 already in use.** Find and kill the old process:
```bash
lsof -i :5000
kill <pid>
```
Or run on a different port: `PORT=5050 python -m web.app`

**Pipeline runs but shot count is zero.** Usually a hoop-detection issue — check the console for `[Hoop] Locked at (...)` early in the run. If the rim never locks in, re-record with the hoop clearly visible and centered in the frame.

**Browser can't reach the dashboard.** Confirm the device is on the same network as the Jetson, and check `hostname -I` for the correct IP. Some networks block peer-to-peer connections — try a phone hotspot if the gym Wi-Fi has client isolation enabled.

---

## License

MIT — see [LICENSE](LICENSE).
