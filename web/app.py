"""Flask web application for Basketball Analytics.

Accessible over the Jetson hotspot at  http://<jetson-ip>:5000

Routes:
  GET  /                   — home page (file selector)
  GET  /api/files          — list SVO files
  POST /api/process        — start processing a chosen SVO
  GET  /api/status/<job>   — SSE stream of progress
  GET  /results/<job>      — results page
  GET  /api/results/<job>  — analytics JSON
  GET  /video/<job>        — serve annotated MP4
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
import uuid
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


# ── helpers ───────────────────────────────────────────────────────────────────

def _list_svo_files():
    svo_dir = config.SVO_DIR
    svo_dir.mkdir(parents=True, exist_ok=True)
    return sorted([f.name for f in svo_dir.glob("*.svo")])


def _list_results():
    """Return previously processed sessions that have analytics.json."""
    out = config.OUTPUT_DIR
    if not out.exists():
        return []
    sessions = []
    for d in sorted(out.iterdir()):
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


def _run_job(job_id: str, svo_path: str) -> None:
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

        result = process_svo(svo_path, progress_cb=_progress)

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


@app.route("/api/process", methods=["POST"])
def api_process():
    data     = request.get_json(force=True)
    filename = data.get("filename", "").strip()

    if not filename:
        return jsonify({"error": "No filename provided"}), 400

    svo_path = config.SVO_DIR / filename
    if not svo_path.exists():
        return jsonify({"error": f"File not found: {filename}"}), 404

    # Use the stem as job_id so results are stable across calls
    job_id = svo_path.stem

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
        }

    t = threading.Thread(target=_run_job, args=(job_id, str(svo_path)), daemon=True)
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
    out_dir = config.OUTPUT_DIR / job_id
    if not out_dir.exists():
        abort(404)

    analytics_path = out_dir / "analytics.json"
    traces_path    = out_dir / "traces.json"

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


@app.route("/api/results/<job_id>")
def api_results(job_id: str):
    out_dir = config.OUTPUT_DIR / job_id
    p = out_dir / "analytics.json"
    if not p.exists():
        abort(404)
    with open(p) as f:
        return jsonify(json.load(f))


@app.route("/video/<job_id>")
def serve_video(job_id: str):
    video = config.OUTPUT_DIR / job_id / "annotated.mp4"
    if not video.exists():
        abort(404)
    return send_file(str(video), mimetype="video/mp4", conditional=True)


# ── run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Bind on all interfaces so it's reachable over the Jetson hotspot
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
