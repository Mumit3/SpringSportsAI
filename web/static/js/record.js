/* record.js — live recording page logic */

document.addEventListener('DOMContentLoaded', () => {
  const setupCard = document.getElementById('setup-card');
  const liveCard  = document.getElementById('live-card');
  const doneCard  = document.getElementById('done-card');
  if (!setupCard) return;   // recorder unavailable

  const labelInput     = document.getElementById('record-label');
  const resolutionSel  = document.getElementById('record-resolution');
  const folderSel      = document.getElementById('record-folder');
  const folderCustom   = document.getElementById('record-folder-custom');

  const startBtn       = document.getElementById('start-btn');
  const stopBtn        = document.getElementById('stop-btn');

  const previewImg     = document.getElementById('preview-img');
  const elapsedBadge   = document.getElementById('live-elapsed');
  const liveStatus     = document.getElementById('live-status');

  const donePath       = document.getElementById('done-svo-path');
  const processBtn     = document.getElementById('process-btn');
  const recordAgainBtn = document.getElementById('record-again-btn');
  const processLabelRow = document.getElementById('process-label-row');
  const processLabelInp = document.getElementById('process-label');
  const processConfirm = document.getElementById('process-confirm-btn');

  let pollTimer = null;
  let lastSvoPath = null;

  // Toggle the custom folder text input
  folderSel.addEventListener('change', () => {
    if (folderSel.value === '__custom__') {
      folderCustom.classList.remove('form-input-hidden');
      folderCustom.focus();
    } else {
      folderCustom.classList.add('form-input-hidden');
    }
  });

  startBtn.addEventListener('click', async () => {
    const label = (labelInput.value || '').trim();
    if (!label) {
      labelInput.focus();
      return;
    }

    let folder = folderSel.value;
    if (folder === '__custom__') {
      folder = (folderCustom.value || '').trim();
      if (!folder) {
        folderCustom.focus();
        return;
      }
    }

    const [resolution, fps] = resolutionSel.value.split('|');

    startBtn.disabled = true;
    startBtn.textContent = 'Starting…';

    try {
      const r = await fetch('/api/record/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ label, folder, resolution, fps: parseInt(fps, 10) }),
      });
      const data = await r.json();
      if (!r.ok) {
        alert(data.error || 'Failed to start recording');
        startBtn.disabled = false;
        startBtn.textContent = '● Start Recording';
        return;
      }
      lastSvoPath = data.svo_path;
      // Force the preview image to reload (fresh stream)
      previewImg.src = '/api/record/preview?t=' + Date.now();
      setupCard.classList.add('hidden');
      liveCard.classList.remove('hidden');
      pollState();
    } catch (e) {
      alert('Network error: ' + e);
      startBtn.disabled = false;
      startBtn.textContent = '● Start Recording';
    }
  });

  stopBtn.addEventListener('click', async () => {
    stopBtn.disabled = true;
    stopBtn.textContent = 'Stopping…';
    try {
      await fetch('/api/record/stop', { method: 'POST' });
      // The poll loop will detect the state transition and show the done card
    } catch (e) {
      alert('Stop error: ' + e);
      stopBtn.disabled = false;
      stopBtn.textContent = '■ Stop';
    }
  });

  function pollState() {
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(async () => {
      try {
        const r = await fetch('/api/record/state');
        const s = await r.json();
        if (s.state === 'RECORDING') {
          elapsedBadge.textContent = `${s.elapsed_seconds}s / ${s.max_seconds}s`;
          liveStatus.textContent   = 'Recording';
        } else if (s.state === 'PREVIEW' && s.completed_svo) {
          // Recording finished
          clearInterval(pollTimer);
          pollTimer = null;
          await fetch('/api/record/shutdown', { method: 'POST' });
          showDone(s.completed_svo);
        } else if (s.state === 'ERROR') {
          clearInterval(pollTimer);
          alert('Camera error: ' + (s.error || 'unknown'));
          resetToSetup();
        }
      } catch (e) {
        // ignore transient errors
      }
    }, 500);
  }

  function showDone(svoPath) {
    // The state response gives an absolute path; convert to a project-relative
    // string for display + later processing.
    const m = svoPath.replace(/\\/g, '/').match(/SpringSportsAI\/(.+)$/);
    const rel = m ? m[1] : svoPath;
    lastSvoPath = rel;
    donePath.textContent = rel;
    liveCard.classList.add('hidden');
    doneCard.classList.remove('hidden');
    stopBtn.disabled = false;
    stopBtn.textContent = '■ Stop';
  }

  recordAgainBtn.addEventListener('click', () => {
    resetToSetup();
  });

  processBtn.addEventListener('click', () => {
    processLabelRow.classList.remove('hidden');
    processLabelInp.focus();
  });

  processConfirm.addEventListener('click', async () => {
    const procLabel = (processLabelInp.value || '').trim();
    processConfirm.disabled = true;
    processConfirm.textContent = 'Starting…';
    try {
      const r = await fetch('/api/process', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: lastSvoPath, label: procLabel }),
      });
      const data = await r.json();
      if (data.error) {
        alert(data.error);
        processConfirm.disabled = false;
        processConfirm.textContent = 'Process';
        return;
      }
      // Hop to the home page; the user can watch progress there.
      // (Could also build a dedicated progress page; reusing index keeps things simple.)
      window.location.href = '/?job=' + encodeURIComponent(data.job_id);
    } catch (e) {
      alert('Network error: ' + e);
      processConfirm.disabled = false;
      processConfirm.textContent = 'Process';
    }
  });

  function resetToSetup() {
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = null;
    doneCard.classList.add('hidden');
    liveCard.classList.add('hidden');
    setupCard.classList.remove('hidden');
    startBtn.disabled = false;
    startBtn.textContent = '● Start Recording';
    processLabelRow.classList.add('hidden');
    processLabelInp.value = '';
    processConfirm.disabled = false;
    processConfirm.textContent = 'Process';
  }
});
