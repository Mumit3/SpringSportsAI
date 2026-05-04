"""Flask web application for Basketball Analytics.

Accessible over the Jetson hotspot at  http://<jetson-ip>:5000

Routes:
  GET  /                       — home (file selector + sessions)
  GET  /record                 — recording page
  GET  /api/files              — list SVO files (recursive scan)
  GET  /api/folders            — list folders under SpringSportsAI/
  POST /api/process            — start processing a chosen SVO
  GET  /api/status/<job>       — SSE stream of progress
  POST /api/record/start       — begin live preview / recording
  GET  /api/record/preview     — MJPEG stream of camera feed
  POST /api/record/stop        — stop recording (preview keeps running)
  POST /api/record/shutdown    — release camera
  GET  /api/record/state       — recorder state polling
  GET  /results/<job>          — results page
  GET  /api/results/<job>      — analytics JSON
  POST /api/sessions/<job>/rename  — rename a session
  POST /api/sessions/<job>/notes   — save free-text notes
  POST /api/sessions/<job>/delete  — delete a session
  GET  /video/<job>            — serve annotated MP4
"""
from __future__ import annotations

import json
import os
import queue
import shutil
import threading
import time
import uuid
from datetime import datetime
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
from main import process_svo

try:
    from pipeline.zed_recorder import ZedRecorder
    _RECORDER_AVAILABLE = True
except Exception as _e:
    print(f"[Flask] ZED recorder unavailable: {_e}")
    _RECORDER_AVAILABLE = False

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static",
)
CORS(app)

# ── job registry ──────────────────────────────────────────────────────────────
# job_id → {"status": "pending|running|done|error",
#            "progress": 0.0-1.0,
#            "message": str,
#            "result": dict|None,
#            "queue": Queue}
_jobs: dict = {}
_jobs_lock  = threading.Lock()

# ── recorder singleton ────────────────────────────────────────────────────────
_recorder:      "ZedRecorder | None" = None
_recorder_lock = threading.Lock()


def _get_recorder():
    global _recorder
    if not _RECORDER_AVAILABLE:
        return None
    with _recorder_lock:
        if _recorder is None:
            _recorder = ZedRecorder()
        return _recorder


# ── helpers ───────────────────────────────────────────────────────────────────

def _list_svo_files():
    """Recursively find all .svo / .svo2 files under the project root.

    Returns dicts with relative paths so the UI can show where each lives.
    """
    root = config.ROOT_DIR
    files = []
    for ext in ("*.svo", "*.svo2"):
        for p in root.rglob(ext):
            # Skip anything under outputs/ (not useful as input)
            try:
                rel = p.relative_to(root)
            except ValueError:
                continue
            if rel.parts and rel.parts[0] == "outputs":
                continue
            files.append({
                "name": p.name,
                "path": str(rel).replace("\\", "/"),
                "folder": str(rel.parent).replace("\\", "/"),
                "size_mb": round(p.stat().st_size / (1024*1024), 1),
            })
    files.sort(key=lambda f: (f["folder"], f["name"]))
    return files


def _list_record_folders():
    """List folder names directly under SpringSportsAI/ for the recording UI.

    Hides typical noise (.git, outputs, web, etc.) so the dropdown is short.
    """
    root = config.ROOT_DIR
    hide = {"outputs", "web", "pipeline", "tools", "tests", "models", ".git", ".claude", "__pycache__"}
    folders = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        if p.name.startswith(".") or p.name.startswith("__"):
            continue
        if p.name in hide:
            continue
        folders.append(p.name)
    if "svo_files" not in folders:
        folders.insert(0, "svo_files")
    return folders


def _flask_default_label() -> str:
    """Default label for runs kicked off from the web UI without one."""
    return datetime.now().strftime("%Y-%m-%d_%H-%M")


def _list_results():
    """Return previously processed sessions that have analytics.json.

    Sessions live under outputs/data/<svo_stem>__<label>/.
    """
    data_root = config.OUTPUT_DIR / "data"
    if not data_root.exists():
        return []
    sessions = []
    for d in sorted(data_root.iterdir()):
        if not d.is_dir():
            continue
        j = d / "analytics.json"
        if j.exists():
            try:
                with open(j) as f:
                    data = json.load(f)
                sessions.append({
                    "job_id":  d.name,
                    "summary": data.get("summary", {}),
                })
            except Exception:
                pass
    return sessions


def _run_job(job_id: str, svo_path: str, label: str) -> None:
    """Worker function executed in a background thread."""
    q = _jobs[job_id]["queue"]

    def _progress(frac: float, msg: str = ""):
        with _jobs_lock:
            _jobs[job_id]["progress"] = frac
            _jobs[job_id]["message"]  = msg
        q.put({"progress": frac, "message": msg})

    try:
        with _jobs_lock:
            _jobs[job_id]["status"] = "running"

        result = process_svo(svo_path, label=label, progress_cb=_progress)

        with _jobs_lock:
            _jobs[job_id]["status"]   = "done"
            _jobs[job_id]["progress"] = 1.0
            _jobs[job_id]["result"]   = result

        q.put({"progress": 1.0, "message": "done", "status": "done"})

    except Exception as exc:
        with _jobs_lock:
            _jobs[job_id]["status"]  = "error"
            _jobs[job_id]["message"] = str(exc)
        q.put({"progress": 0.0, "message": str(exc), "status": "error"})


# ── routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template(
        "index.html",
        svo_files = _list_svo_files(),
        sessions  = _list_results(),
    )


@app.route("/api/files")
def api_files():
    return jsonify({"files": _list_svo_files()})


@app.route("/api/folders")
def api_folders():
    return jsonify({"folders": _list_record_folders()})


@app.route("/api/process", methods=["POST"])
def api_process():
    data     = request.get_json(force=True)
    rel_path = (data.get("path") or data.get("filename") or "").strip()
    label    = (data.get("label") or "").strip() or _flask_default_label()

    if not rel_path:
        return jsonify({"error": "No file path provided"}), 400

    # Accept either a relative path under SpringSportsAI/ or just a filename
    candidate = (config.ROOT_DIR / rel_path).resolve()
    if not candidate.exists():
        # Fallback: search by basename
        matches = [p for p in config.ROOT_DIR.rglob(Path(rel_path).name)
                   if p.is_file()]
        if matches:
            candidate = matches[0]
        else:
            return jsonify({"error": f"File not found: {rel_path}"}), 404

    # Reject paths that escape the project root
    try:
        candidate.relative_to(config.ROOT_DIR.resolve())
    except ValueError:
        return jsonify({"error": "File outside project"}), 400

    svo_path = candidate

    # job_id matches the on-disk session folder name
    job_id = f"{svo_path.stem}__{label}"

    with _jobs_lock:
        existing = _jobs.get(job_id)
        if existing and existing["status"] == "running":
            return jsonify({"job_id": job_id, "status": "already_running"}), 200

        _jobs[job_id] = {
            "status":   "pending",
            "progress": 0.0,
            "message":  "Queued",
            "result":   None,
            "queue":    queue.Queue(),
            "svo_path": str(svo_path),
            "label":    label,
        }

    t = threading.Thread(target=_run_job, args=(job_id, str(svo_path), label), daemon=True)
    t.start()

    return jsonify({"job_id": job_id, "status": "started"}), 202


@app.route("/api/status/<job_id>")
def api_status_sse(job_id: str):
    """Server-Sent Events stream for live progress."""
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
                # heartbeat
                yield "data: {\"heartbeat\": true}\n\n"

    return Response(
        _generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":  "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/results/<job_id>")
def results_page(job_id: str):
    out_dir = config.OUTPUT_DIR / "data" / job_id
    if not out_dir.exists():
        abort(404)

    analytics_path = out_dir / "analytics.json"
    traces_path    = out_dir / "traces.json"
    notes_path     = out_dir / "notes.txt"

    if not analytics_path.exists():
        abort(404)

    with open(analytics_path) as f:
        analytics_data = json.load(f)

    traces = []
    if traces_path.exists():
        with open(traces_path) as f:
            traces = json.load(f)

    notes = ""
    if notes_path.exists():
        notes = notes_path.read_text(encoding="utf-8")

    return render_template(
        "results.html",
        job_id   = job_id,
        summary  = analytics_data.get("summary", {}),
        shots    = analytics_data.get("shots", []),
        traces   = json.dumps(traces),
        notes    = notes,
    )


@app.route("/api/results/<job_id>")
def api_results(job_id: str):
    p = config.OUTPUT_DIR / "data" / job_id / "analytics.json"
    if not p.exists():
        abort(404)
    with open(p) as f:
        return jsonify(json.load(f))


@app.route("/video/<job_id>")
def serve_video(job_id: str):
    video = config.OUTPUT_DIR / "videos" / f"{job_id}.mp4"
    if not video.exists():
        abort(404)
    return send_file(str(video), mimetype="video/mp4", conditional=True)


# ── recording ─────────────────────────────────────────────────────────────────

@app.route("/record")
def record_page():
    return render_template(
        "record.html",
        folders     = _list_record_folders(),
        recorder_ok = _RECORDER_AVAILABLE,
    )


@app.route("/api/record/start", methods=["POST"])
def api_record_start():
    rec = _get_recorder()
    if rec is None:
        return jsonify({"error": "Recorder unavailable (pyzed not installed?)"}), 503

    data       = request.get_json(force=True)
    label      = (data.get("label") or "").strip()
    folder     = (data.get("folder") or "svo_files").strip()
    resolution = (data.get("resolution") or "HD2K").strip()
    fps        = int(data.get("fps") or 15)

    if not label:
        return jsonify({"error": "Filename label is required"}), 400

    # Sanitize label — disallow path separators
    if any(c in label for c in ("/", "\\", "..")):
        return jsonify({"error": "Label can't contain slashes or '..'"}), 400

    # Block re-entry while a job is running
    with _jobs_lock:
        for j in _jobs.values():
            if j["status"] == "running":
                return jsonify({"error": "A processing job is running — wait for it to finish"}), 409

    save_dir = (config.ROOT_DIR / folder).resolve()
    try:
        save_dir.relative_to(config.ROOT_DIR.resolve())
    except ValueError:
        return jsonify({"error": "Folder outside project"}), 400
    save_dir.mkdir(parents=True, exist_ok=True)

    svo_path = save_dir / f"{label}.svo2"

    rec.start_preview(resolution=resolution, fps=fps)
    # Wait briefly for camera to come up before starting the recording
    for _ in range(40):   # up to 4 s
        st = rec.get_state()["state"]
        if st == "PREVIEW":
            break
        if st == "ERROR":
            return jsonify({"error": rec.get_state().get("error") or "camera error"}), 500
        time.sleep(0.1)

    rec.start_recording(str(svo_path))
    return jsonify({
        "ok": True,
        "svo_path": str(svo_path.relative_to(config.ROOT_DIR)).replace("\\", "/"),
    })


@app.route("/api/record/preview")
def api_record_preview():
    rec = _get_recorder()
    if rec is None:
        abort(503)

    def _stream():
        while True:
            frame = rec.get_jpeg_frame()
            if frame is None:
                time.sleep(0.05)
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                   + str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n")
    return Response(
        _stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/record/state")
def api_record_state():
    rec = _get_recorder()
    if rec is None:
        return jsonify({"state": "UNAVAILABLE"}), 503
    return jsonify(rec.get_state())


@app.route("/api/record/stop", methods=["POST"])
def api_record_stop():
    rec = _get_recorder()
    if rec is None:
        return jsonify({"error": "Recorder unavailable"}), 503
    rec.stop_recording()
    # Wait for the capture thread to flush the SVO file
    for _ in range(50):   # up to 5 s
        st = rec.get_state()
        if st["state"] != "RECORDING":
            break
        time.sleep(0.1)
    return jsonify(rec.get_state())


@app.route("/api/record/shutdown", methods=["POST"])
def api_record_shutdown():
    rec = _get_recorder()
    if rec is None:
        return jsonify({"ok": True})
    completed = rec.shutdown()
    # Drop the singleton so the next start gets a fresh handle
    global _recorder
    with _recorder_lock:
        _recorder = None
    return jsonify({"ok": True, "completed_svo": completed})


# ── session management ────────────────────────────────────────────────────────

def _session_paths(job_id: str):
    data_dir = config.OUTPUT_DIR / "data"  / job_id
    video    = config.OUTPUT_DIR / "videos" / f"{job_id}.mp4"
    return data_dir, video


@app.route("/api/sessions/<job_id>/rename", methods=["POST"])
def api_session_rename(job_id: str):
    data      = request.get_json(force=True)
    new_label = (data.get("label") or "").strip()
    if not new_label:
        return jsonify({"error": "New label required"}), 400
    if any(c in new_label for c in ("/", "\\", "..")):
        return jsonify({"error": "Label can't contain slashes or '..'"}), 400

    data_dir, video = _session_paths(job_id)
    if not data_dir.exists():
        abort(404)

    # Existing job_id is "<stem>__<old_label>" — preserve the stem
    if "__" not in job_id:
        return jsonify({"error": "Malformed job id"}), 400
    stem = job_id.rsplit("__", 1)[0]
    new_id = f"{stem}__{new_label}"

    new_data_dir = config.OUTPUT_DIR / "data"  / new_id
    new_video    = config.OUTPUT_DIR / "videos" / f"{new_id}.mp4"
    if new_data_dir.exists() or new_video.exists():
        return jsonify({"error": "A session with this label already exists"}), 409

    data_dir.rename(new_data_dir)
    if video.exists():
        new_video.parent.mkdir(parents=True, exist_ok=True)
        video.rename(new_video)
    return jsonify({"ok": True, "job_id": new_id})


@app.route("/api/sessions/<job_id>/notes", methods=["POST"])
def api_session_notes(job_id: str):
    data  = request.get_json(force=True)
    notes = data.get("notes", "")
    data_dir, _ = _session_paths(job_id)
    if not data_dir.exists():
        abort(404)
    notes_file = data_dir / "notes.txt"
    notes_file.write_text(notes, encoding="utf-8")
    return jsonify({"ok": True})


@app.route("/api/sessions/<job_id>/delete", methods=["POST"])
def api_session_delete(job_id: str):
    data_dir, video = _session_paths(job_id)
    if not data_dir.exists() and not video.exists():
        abort(404)
    if data_dir.exists():
        shutil.rmtree(data_dir, ignore_errors=True)
    if video.exists():
        try:
            video.unlink()
        except Exception:
            pass
    return jsonify({"ok": True})


# ── run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Bind on all interfaces so it's reachable over the Jetson hotspot
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
