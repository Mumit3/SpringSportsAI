/* record.js — live ZED recording page */

document.addEventListener('DOMContentLoaded', () => {
  const previewImg     = document.getElementById('preview-img');
  const previewOverlay = document.getElementById('preview-overlay');
  const labelInput     = document.getElementById('label-input');
  const btnRecord      = document.getElementById('btn-record');
  const btnStop        = document.getElementById('btn-stop');
  const recTimer       = document.getElementById('rec-timer');
  const recTimeEl      = document.getElementById('rec-time');
  const recMaxEl       = document.getElementById('rec-max');
  const recBarFill     = document.getElementById('rec-bar-fill');
  const statusMsg      = document.getElementById('status-msg');
  const progressPanel  = document.getElementById('progress-panel');
  const progressBar    = document.getElementById('progress-bar');
  const progressMsg    = document.getElementById('progress-msg');
  const progressPct    = document.getElementById('progress-pct');

  let stateTimer  = null;
  let activeSSE   = null;
  let cameraReady = false;

  function fmt(secs) {
    const m = Math.floor(secs / 60);
    const s = Math.floor(secs % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
  }

  function showStatus(msg, kind = 'info') {
    statusMsg.textContent = msg;
    statusMsg.dataset.kind = kind;
  }

  // ── start preview on page load ──────────────────────────────────────────────
  fetch('/api/recorder/start', { method: 'POST' })
    .then(r => r.json())
    .then(data => {
      if (data.error) {
        previewOverlay.textContent = `Camera error: ${data.error}`;
        showStatus(`Camera error: ${data.error}`, 'error');
        return;
      }
      // Load MJPEG stream
      previewImg.src = '/api/preview';
      previewOverlay.classList.add('hidden');
      // Poll state until camera is in PREVIEW state
      stateTimer = setInterval(pollState, 500);
    })
    .catch(err => {
      previewOverlay.textContent = `Network error: ${err}`;
      showStatus(`Network error: ${err}`, 'error');
    });

  function pollState() {
    fetch('/api/recorder/state')
      .then(r => r.json())
      .then(s => updateState(s))
      .catch(() => {});
  }

  let wasRecording = false;
  function updateState(s) {
    if (s.state === 'PREVIEW' && !cameraReady) {
      cameraReady = true;
      btnRecord.disabled = false;
      showStatus('Camera ready — adjust framing then click Start Recording.');
    } else if (s.state === 'RECORDING') {
      wasRecording = true;
      const elapsed = s.elapsed_seconds || 0;
      const max     = s.max_seconds || 60;
      recTimeEl.textContent = fmt(elapsed);
      recMaxEl.textContent  = fmt(max);
      recBarFill.style.width = `${(elapsed / max) * 100}%`;
    } else if (s.state === 'PREVIEW' && wasRecording && !btnStop.dataset.processing) {
      // Backend auto-stopped (timer hit max). Trigger pipeline.
      wasRecording = false;
      autoStop();
    } else if (s.state === 'ERROR') {
      showStatus(`Camera error: ${s.error || 'unknown'}`, 'error');
      clearInterval(stateTimer);
    }
  }

  // ── start recording ─────────────────────────────────────────────────────────
  btnRecord.addEventListener('click', () => {
    btnRecord.disabled = true;
    fetch('/api/recorder/record', { method: 'POST' })
      .then(r => r.json())
      .then(data => {
        if (data.error) {
          showStatus(`Error: ${data.error}`, 'error');
          btnRecord.disabled = false;
          return;
        }
        btnRecord.classList.add('hidden');
        btnStop.classList.remove('hidden');
        recTimer.classList.remove('hidden');
        showStatus('● Recording…', 'recording');
      })
      .catch(err => {
        showStatus(`Network error: ${err}`, 'error');
        btnRecord.disabled = false;
      });
  });

  // ── stop & process ──────────────────────────────────────────────────────────
  btnStop.addEventListener('click', () => stopAndProcess(false));

  function autoStop() {
    showStatus('⏱ Time limit reached — auto-stopping.', 'recording');
    stopAndProcess(true);
  }

  function stopAndProcess(autoTriggered) {
    if (btnStop.dataset.processing) return;
    btnStop.dataset.processing = 'true';
    btnStop.disabled = true;
    btnStop.textContent = autoTriggered ? '⏱ Processing…' : '■ Processing…';

    if (stateTimer) { clearInterval(stateTimer); stateTimer = null; }

    const label = (labelInput.value || '').trim();

    fetch('/api/recorder/stop', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ label }),
    })
    .then(r => r.json())
    .then(data => {
      if (data.error) {
        showStatus(`Error: ${data.error}`, 'error');
        btnStop.disabled = false;
        return;
      }
      // Stop preview img since camera is now released
      previewImg.removeAttribute('src');
      previewOverlay.textContent = 'Camera released. Processing…';
      previewOverlay.classList.remove('hidden');
      progressPanel.classList.remove('hidden');
      recTimer.classList.add('hidden');
      listenForProgress(data.job_id);
    })
    .catch(err => {
      showStatus(`Network error: ${err}`, 'error');
      btnStop.disabled = false;
    });
  }

  // ── pipeline progress via SSE ───────────────────────────────────────────────
  function listenForProgress(jobId) {
    activeSSE = new EventSource(`/api/status/${jobId}`);
    activeSSE.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.heartbeat) return;
        const pct = Math.round((msg.progress || 0) * 100);
        setProgress(pct, msg.message || '');
        if (msg.status === 'done') {
          activeSSE.close();
          setProgress(100, 'Done — redirecting to results…');
          setTimeout(() => { window.location.href = `/results/${jobId}`; }, 600);
        } else if (msg.status === 'error') {
          activeSSE.close();
          setProgress(0, `Error: ${msg.message}`);
        }
      } catch (e) { /* ignore */ }
    };
    activeSSE.onerror = () => {
      fetch(`/api/results/${jobId}`).then(r => {
        if (r.ok) window.location.href = `/results/${jobId}`;
      }).catch(() => {});
    };
  }

  function setProgress(pct, msg) {
    if (progressBar) progressBar.style.width = `${pct}%`;
    if (progressMsg) progressMsg.textContent  = msg;
    if (progressPct) progressPct.textContent  = `${pct}%`;
  }

  // ── shutdown camera if user navigates away before recording ────────────────
  window.addEventListener('beforeunload', () => {
    if (!btnStop.dataset.processing) {
      // best-effort — fire-and-forget
      navigator.sendBeacon &&
        navigator.sendBeacon('/api/recorder/shutdown', new Blob([], {type:'application/json'}));
    }
  });
});
