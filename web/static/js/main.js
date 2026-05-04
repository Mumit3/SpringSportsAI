/* main.js — index page interactions
 *
 * Flow:
 *   click .file-btn  → select it, reveal #label-prompt
 *   type label, click #confirm-btn → POST /api/process
 *   listen on /api/status/<job_id>, redirect to /results/<job_id> when done
 */

document.addEventListener('DOMContentLoaded', () => {
  const fileBtns       = document.querySelectorAll('.file-btn');
  const labelPrompt    = document.getElementById('label-prompt');
  const labelInput     = document.getElementById('process-label');
  const confirmBtn     = document.getElementById('confirm-btn');
  const cancelBtn      = document.getElementById('cancel-btn');
  const progressPanel  = document.getElementById('progress-panel');
  const progressBar    = document.getElementById('progress-bar');
  const progressMsg    = document.getElementById('progress-msg');
  const progressPct    = document.getElementById('progress-pct');

  let selectedPath = null;
  let activeEventSource = null;

  fileBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      fileBtns.forEach(b => b.classList.remove('selected'));
      btn.classList.add('selected');

      selectedPath = btn.dataset.path;

      // Reveal the label prompt; hide any prior progress
      labelPrompt.classList.remove('hidden');
      progressPanel.classList.add('hidden');
      labelInput.focus();
    });
  });

  if (cancelBtn) {
    cancelBtn.addEventListener('click', () => {
      labelPrompt.classList.add('hidden');
      fileBtns.forEach(b => b.classList.remove('selected'));
      selectedPath = null;
    });
  }

  if (confirmBtn) {
    confirmBtn.addEventListener('click', () => {
      if (!selectedPath) return;
      const label = (labelInput.value || '').trim();
      startProcessing(selectedPath, label);
    });
  }

  // Allow Enter to submit
  if (labelInput) {
    labelInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') confirmBtn.click();
    });
  }

  function startProcessing(path, label) {
    labelPrompt.classList.add('hidden');
    progressPanel.classList.remove('hidden');
    setProgress(0, `Starting ${path}…`);

    if (activeEventSource) activeEventSource.close();

    fetch('/api/process', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path, label }),
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
          setProgress(100, 'Processing complete — redirecting…');
          setTimeout(() => { window.location.href = `/results/${jobId}`; }, 800);
        } else if (msg.status === 'error') {
          activeEventSource.close();
          setProgress(0, `Error: ${msg.message}`);
        }
      } catch (e) { /* ignore */ }
    };

    activeEventSource.onerror = () => {
      fetch(`/api/results/${jobId}`)
        .then(r => { if (r.ok) window.location.href = `/results/${jobId}`; })
        .catch(() => {});
    };
  }

  function setProgress(pct, msg) {
    if (progressBar) progressBar.style.width = `${pct}%`;
    if (progressMsg) progressMsg.textContent  = msg;
    if (progressPct) progressPct.textContent  = `${pct}%`;
  }

  // If we landed here from "Process Now" on the record page, auto-resume tracking
  const params = new URLSearchParams(window.location.search);
  const incomingJob = params.get('job');
  if (incomingJob) {
    progressPanel.classList.remove('hidden');
    setProgress(0, `Resuming ${incomingJob}…`);
    listenForProgress(incomingJob);
  }
});
