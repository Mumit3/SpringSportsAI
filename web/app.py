"""Flask web application for Basketball Analytics.

Accessible over the Jetson hotspot at  http://<jetson-ip>:5000

Routes:
  GET  /                          — landing page (record or process existing)
  GET  /record                    — live recording page
  GET  /api/files                 — list SVO files
  POST /api/process               — start processing a chosen SVO (with optional label)
  GET  /api/status/<job>          — SSE stream of progress
  GET  /results/<job>             — results page
  GET  /api/results/<job>         — analytics JSON
  GET  /video/<job>               — serve annotated MP4

  GET  /api/preview               — MJPEG live preview of ZED camera
  POST /api/recorder/start        — open camera and begin preview
  POST /api/recorder/record       — start SVO recording
  POST /api/recorder/stop         — stop recording, kick off pipeline
  POST /api/recorder/shutdown     — close camera (no pipeline)
  GET  /api/recorder/state        — JSON: state, elapsed time, etc.
"""
from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path

from flask import (
    Flask, Response, jsonify, render_template,
    request, send_file, abort,
)
from flask_cors import CORS

# Make sure the project root is importable even when running from web/
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import config
from main import process_svo, _sanitize_label

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static",
)
CORS(app)

# ── job registry ──────────────────────────────────────────────────────────────
# job_id → {"status": "pending|running|done|error", "progress": float,
#           "message": str, "result": dict|None, "queue": Queue, ...}
_jobs: dict = {}
_jobs_lock  = threading.Lock()

# ── recorder singleton (lazy) ─────────────────────────────────────────────────
_recorder = None
_recorder_lock = threading.Lock()


def _get_recorder():
    """Lazy-initialise the ZedRecorder. Returns None if pyzed unavailable."""
    global _recorder
    with _recorder_lock:
        if _recorder is None:
            try:
                from pipeline.zed_recorder import ZedRecorder
                _recorder = ZedRecorder()
            except Exception as e:
                print(f"[App] ZedRecorder unavailable: {e}")
                return None
        return _recorder


# ── helpers ───────────────────────────────────────────────────────────────────

def _list_svo_files():
    svo_dir = config.SVO_DIR
    svo_dir.mkdir(parents=True, exist_ok=True)
    # Skip temporary live-capture files (filenames starting with underscore)
    return sorted([
        f.name for f in svo_dir.glob("*.svo*")
        if not f.name.startswith("_")
    ])


def _job_id_to_files(job_id: str) -> dict:
    """Map a job_id to its output files. Handles both new flat layout and
    legacy nested layout. Returns dict: dir, video, analytics, traces."""
    out = config.OUTPUT_DIR

    if job_id.startswith("experiment__"):
        rest = job_id[len("experiment__"):]

        # Legacy nested: experiment/<stem>/<label>/annotated.mp4
        if "__" in rest:
            stem, label = rest.split("__", 1)
            legacy = out / "experiment" / stem / label
            if (legacy / "annotated.mp4").exists():
                return {
                    "dir":       legacy,
                    "video":     legacy / "annotated.mp4",
                    "analytics": legacy / "analytics.json",
                    "traces":    legacy / "traces.json",
                }
            rest = label   # fall through to flat lookup using label

        # Legacy single-level: experiment/<rest>/annotated.mp4
        legacy_single = out / "experiment" / rest
        if (legacy_single / "annotated.mp4").exists():
            return {
                "dir":       legacy_single,
                "video":     legacy_single / "annotated.mp4",
                "analytics": legacy_single / "analytics.json",
                "traces":    legacy_single / "traces.json",
            }

        # New flat layout
        flat = out / "experiment"
        return {
            "dir":       flat,
            "video":     flat / f"{rest}.mp4",
            "analytics": flat / f"{rest}_analytics.json",
            "traces":    flat / f"{rest}_traces.json",
        }

    # Non-experiment legacy
    legacy_root = out / job_id
    return {
        "dir":       legacy_root,
        "video":     legacy_root / "annotated.mp4",
        "analytics": legacy_root / "analytics.json",
        "traces":    legacy_root / "traces.json",
    }


def _list_results():
    """Return previously processed sessions across both layouts."""
    out = config.OUTPUT_DIR
    if not out.exists():
        return []

    sessions = []

    # Legacy/main-branch layout: outputs/<stem>/analytics.json
    for d in sorted(out.iterdir()):
        if d.name == "experiment" or not d.is_dir():
            continue
        j = d / "analytics.json"
        if j.exists():
            try:
                with open(j) as f:
                    data = json.load(f)
                sessions.append({
                    "job_id":  d.name,
                    "label":   None,
                    "summary": data.get("summary", {}),
                })
            except Exception:
                pass

    # Experiment layouts (any of the three):
    #   1. Flat (current):   outputs/experiment/<name>_analytics.json
    #   2. Single-nested:    outputs/experiment/<stem>/analytics.json
    #   3. Double-nested:    outputs/experiment/<stem>/<label>/analytics.json
    exp_root = out / "experiment"
    if exp_root.exists():
        # ── 1. Flat layout — analytics files directly under experiment/ ──
        for f in sorted(exp_root.glob("*_analytics.json")):
            name = f.name[:-len("_analytics.json")]
            try:
                with open(f) as fh:
                    data = json.load(fh)
                sessions.append({
                    "job_id":  f"experiment__{name}",
                    "label":   name,
                    "summary": data.get("summary", {}),
                })
            except Exception:
                pass

        # ── 2/3. Legacy nested layouts ──
        for stem_dir in sorted(exp_root.iterdir()):
            if not stem_dir.is_dir():
                continue
            label_dirs = [c for c in stem_dir.iterdir() if c.is_dir()]
            if label_dirs:
                for label_dir in sorted(label_dirs):
                    j = label_dir / "analytics.json"
                    if j.exists():
                        try:
                            with open(j) as f:
                                data = json.load(f)
                            sessions.append({
                                "job_id":  f"experiment__{stem_dir.name}__{label_dir.name}",
                                "label":   label_dir.name,
                                "summary": data.get("summary", {}),
                            })
                        except Exception:
                            pass
            else:
                j = stem_dir / "analytics.json"
                if j.exists():
                    try:
                        with open(j) as f:
                            data = json.load(f)
                        sessions.append({
                            "job_id":  f"experiment__{stem_dir.name}",
                            "label":   None,
                            "summary": data.get("summary", {}),
                        })
                    except Exception:
                        pass

    return sessions


def _start_job(svo_path: str, label: str = None) -> str:
    """Submit a processing job and return its job_id."""
    svo_p = Path(svo_path)
    safe = _sanitize_label(label) if label else None

    if config.EXPERIMENT_MODE:
        run_name = safe or svo_p.stem
        job_id   = f"experiment__{run_name}"
    else:
        job_id = svo_p.stem

    with _jobs_lock:
        existing = _jobs.get(job_id)
        if existing and existing["status"] == "running":
            return job_id  # already in progress, reuse

        _jobs[job_id] = {
            "status":   "pending",
            "progress": 0.0,
            "message":  "Queued",
            "result":   None,
            "queue":    queue.Queue(),
            "svo_path": svo_path,
            "label":    safe,
        }

    t = threading.Thread(
        target=_run_job,
        args=(job_id, svo_path, safe),
        daemon=True,
    )
    t.start()
    return job_id


def _run_job(job_id: str, svo_path: str, label: str = None) -> None:
    q = _jobs[job_id]["queue"]

    def _progress(frac: float, msg: str = ""):
        with _jobs_lock:
            _jobs[job_id]["progress"] = frac
            _jobs[job_id]["message"]  = msg
        q.put({"progress": frac, "message": msg})

    try:
        with _jobs_lock:
            _jobs[job_id]["status"] = "running"

        result = process_svo(svo_path, progress_cb=_progress, label=label)

        with _jobs_lock:
            _jobs[job_id]["status"]   = "done"
            _jobs[job_id]["progress"] = 1.0
            _jobs[job_id]["result"]   = result

        q.put({"progress": 1.0, "message": "done", "status": "done", "job_id": job_id})

    except Exception as exc:
        with _jobs_lock:
            _jobs[job_id]["status"]  = "error"
            _jobs[job_id]["message"] = str(exc)
        q.put({"progress": 0.0, "message": str(exc), "status": "error"})


# ── pages ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template(
        "index.html",
        svo_files = _list_svo_files(),
        sessions  = _list_results(),
    )


@app.route("/record")
def record_page():
    return render_template("record.html")


@app.route("/results/<job_id>")
def results_page(job_id: str):
    files = _job_id_to_files(job_id)
    analytics_path = files["analytics"]
    traces_path    = files["traces"]

    if not analytics_path.exists():
        abort(404)

    with open(analytics_path) as f:
        analytics_data = json.load(f)

    traces = []
    if traces_path.exists():
        with open(traces_path) as f:
            traces = json.load(f)

    return render_template(
        "results.html",
        job_id   = job_id,
        summary  = analytics_data.get("summary", {}),
        shots    = analytics_data.get("shots", []),
        traces   = json.dumps(traces),
    )


# ── processing API ────────────────────────────────────────────────────────────

@app.route("/api/files")
def api_files():
    return jsonify({"files": _list_svo_files()})


@app.route("/api/process", methods=["POST"])
def api_process():
    data     = request.get_json(force=True)
    filename = (data.get("filename") or "").strip()
    label    = (data.get("label") or "").strip() or None

    if not filename:
        return jsonify({"error": "No filename provided"}), 400

    svo_path = config.SVO_DIR / filename
    if not svo_path.exists():
        return jsonify({"error": f"File not found: {filename}"}), 404

    job_id = _start_job(str(svo_path), label=label)
    return jsonify({"job_id": job_id, "status": "started"}), 202


@app.route("/api/status/<job_id>")
def api_status_sse(job_id: str):
    if job_id not in _jobs:
        abort(404)

    def _generate():
        q = _jobs[job_id]["queue"]
        while True:
            try:
                msg = q.get(timeout=30)
                yield f"data: {json.dumps(msg)}\n\n"
                if msg.get("status") in ("done", "error"):
                    break
            except queue.Empty:
                yield "data: {\"heartbeat\": true}\n\n"

    return Response(
        _generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/api/results/<job_id>")
def api_results(job_id: str):
    p = _job_id_to_files(job_id)["analytics"]
    if not p.exists():
        abort(404)
    with open(p) as f:
        return jsonify(json.load(f))


@app.route("/video/<job_id>")
def serve_video(job_id: str):
    video = _job_id_to_files(job_id)["video"]
    if not video.exists():
        abort(404)
    return send_file(str(video), mimetype="video/mp4", conditional=True)


# ── live recording API ────────────────────────────────────────────────────────

@app.route("/api/recorder/start", methods=["POST"])
def api_recorder_start():
    rec = _get_recorder()
    if rec is None:
        return jsonify({"error": "ZED SDK not available"}), 500
    try:
        rec.start_preview()
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/recorder/record", methods=["POST"])
def api_recorder_record():
    rec = _get_recorder()
    if rec is None:
        return jsonify({"error": "ZED SDK not available"}), 500

    state = rec.get_state()
    if state["state"] not in ("PREVIEW", "STARTING"):
        return jsonify({"error": f"Cannot record from state {state['state']}"}), 400

    svo_path = config.SVO_DIR / config.LIVE_SVO_FILENAME
    rec.start_recording(str(svo_path), config.RECORDING_MAX_SECONDS)
    return jsonify({"status": "recording"})


@app.route("/api/recorder/stop", methods=["POST"])
def api_recorder_stop():
    """Stop recording, fully release camera, kick off the pipeline."""
    rec = _get_recorder()
    if rec is None:
        return jsonify({"error": "ZED SDK not available"}), 500

    data  = request.get_json(silent=True) or {}
    label = (data.get("label") or "").strip() or None

    rec.stop_recording()
    # Give the capture thread a moment to flush the SVO close
    time.sleep(0.4)

    # Fully release camera so the pipeline can open the SVO file
    completed = rec.shutdown(timeout=5.0)
    if completed is None:
        # Maybe it already finalised; check state for the recorded path
        state = rec.get_state()
        completed = state.get("completed_svo")

    if completed is None or not Path(completed).exists():
        return jsonify({"error": "No recording was finalised"}), 500

    job_id = _start_job(completed, label=label)
    return jsonify({"job_id": job_id, "status": "processing"}), 202


@app.route("/api/recorder/shutdown", methods=["POST"])
def api_recorder_shutdown():
    rec = _get_recorder()
    if rec is None:
        return jsonify({"status": "ok"})
    rec.shutdown(timeout=3.0)
    return jsonify({"status": "ok"})


@app.route("/api/recorder/state")
def api_recorder_state():
    rec = _get_recorder()
    if rec is None:
        return jsonify({"state": "UNAVAILABLE"})
    return jsonify(rec.get_state())


@app.route("/api/preview")
def api_preview():
    rec = _get_recorder()
    if rec is None:
        abort(503)

    boundary = b"--frame"

    def _generate():
        while True:
            frame = rec.get_jpeg_frame()
            if frame is not None:
                yield (boundary + b"\r\n"
                       + b"Content-Type: image/jpeg\r\n\r\n"
                       + frame + b"\r\n")
            time.sleep(0.05)   # ~20 fps preview

    return Response(
        _generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


# ── run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
