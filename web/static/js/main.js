/* main.js — index page interactions */

document.addEventListener('DOMContentLoaded', () => {
  // ── mode picker ──────────────────────────────────────────────
  const modeExisting   = document.getElementById('mode-existing');
  const existingPanel  = document.getElementById('existing-panel');
  if (modeExisting && existingPanel) {
    modeExisting.addEventListener('click', () => {
      existingPanel.classList.remove('hidden');
      existingPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
  }

  // ── existing-SVO processing ──────────────────────────────────
  const fileBtns      = document.querySelectorAll('.file-btn');
  const labelInput    = document.getElementById('label-input');
  const progressPanel = document.getElementById('progress-panel');
  const progressBar   = document.getElementById('progress-bar');
  const progressMsg   = document.getElementById('progress-msg');
  const progressPct   = document.getElementById('progress-pct');

  if (!fileBtns.length) return;

  let activeEventSource = null;

  fileBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      fileBtns.forEach(b => b.classList.remove('selected'));
      btn.classList.add('selected');
      const filename = btn.dataset.filename;
      const label    = (labelInput && labelInput.value.trim()) || null;
      startProcessing(filename, label);
    });
  });

  function startProcessing(filename, label) {
    if (progressPanel) progressPanel.classList.remove('hidden');
    setProgress(0, `Starting ${filename}…`);

    if (activeEventSource) activeEventSource.close();

    fetch('/api/process', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ filename, label }),
    })
    .then(r => r.json())
    .then(data => {
      if (data.error) {
        setProgress(0, `Error: ${data.error}`);
        return;
      }
      listenForProgress(data.job_id);
    })
    .catch(err => setProgress(0, `Network error: ${err}`));
  }

  function listenForProgress(jobId) {
    activeEventSource = new EventSource(`/api/status/${jobId}`);
    activeEventSource.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.heartbeat) return;
        const pct = Math.round((msg.progress || 0) * 100);
        setProgress(pct, msg.message || '');
        if (msg.status === 'done') {
          activeEventSource.close();
          setProgress(100, 'Done — redirecting…');
          setTimeout(() => { window.location.href = `/results/${jobId}`; }, 600);
        } else if (msg.status === 'error') {
          activeEventSource.close();
          setProgress(0, `Error: ${msg.message}`);
        }
      } catch (e) { /* ignore parse errors */ }
    };
    activeEventSource.onerror = () => {
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
});
