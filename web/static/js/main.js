/* main.js — index page interactions
 *
 * Each .file-card has an inline action area that's hidden until the card is
 * clicked. Clicking another card collapses any previously-expanded one.
 */

document.addEventListener('DOMContentLoaded', () => {
  const cards          = document.querySelectorAll('.file-card');
  const progressPanel  = document.getElementById('progress-panel');
  const progressBar    = document.getElementById('progress-bar');
  const progressMsg    = document.getElementById('progress-msg');
  const progressPct    = document.getElementById('progress-pct');
  const cancelBtn      = document.getElementById('cancel-btn');

  let activeEventSource = null;
  let activeJobId       = null;

  cards.forEach(card => {
    const labelInput  = card.querySelector('.file-card-label');
    const startBtn    = card.querySelector('.file-card-start');
    const cancelBtn   = card.querySelector('.file-card-cancel');
    const modeSelect  = card.querySelector('.file-card-mode');
    const ballOnlyWrap = card.querySelector('.file-card-ball-only-wrap');

    // Show "Ball only" checkbox only when Mini Hoop is selected
    if (modeSelect && ballOnlyWrap) {
      const sync = () => {
        if (modeSelect.value === 'mini') ballOnlyWrap.classList.remove('hidden');
        else ballOnlyWrap.classList.add('hidden');
      };
      modeSelect.addEventListener('change', sync);
      modeSelect.addEventListener('click',  (e) => e.stopPropagation());
      sync();
    }

    // Click on card body (but not action area) → expand
    card.addEventListener('click', (e) => {
      if (e.target.closest('.file-card-action')) return;
      cards.forEach(c => { if (c !== card) c.classList.remove('selected'); });
      card.classList.toggle('selected');
      if (card.classList.contains('selected') && labelInput) {
        labelInput.focus();
      }
    });

    // Keyboard support — Enter/Space when card is focused
    card.addEventListener('keydown', (e) => {
      if (e.target !== card) return;   // ignore key events from inputs
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();   // prevent page scroll on Space
        card.click();
      }
    });

    if (labelInput) {
      labelInput.addEventListener('keydown', (e) => {
        e.stopPropagation();   // don't bubble Enter into card click
        if (e.key === 'Enter') startBtn.click();
      });
      labelInput.addEventListener('click', (e) => e.stopPropagation());
    }

    if (cancelBtn) {
      cancelBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        card.classList.remove('selected');
      });
    }

    if (startBtn) {
      startBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        const path  = card.dataset.path;
        const mode  = (modeSelect && modeSelect.value) || 'regulation';
        const label = (labelInput.value || '').trim();
        const ballOnlyEl = card.querySelector('.file-card-ball-only');
        const ballOnly = mode === 'mini'
                         && !!(ballOnlyEl && ballOnlyEl.checked);
        startProcessing(path, label, mode, ballOnly);
      });
    }
  });

  function startProcessing(path, label, mode, ballOnly) {
    cards.forEach(c => c.classList.remove('selected'));
    progressPanel.classList.remove('hidden');
    const tag = mode === 'mini' ? (ballOnly ? 'mini · ball-only' : 'mini') : 'regulation';
    setProgress(0, `Starting ${path} [${tag}]…`);

    if (activeEventSource) activeEventSource.close();

    fetch('/api/process', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path, label, mode, ball_only: !!ballOnly }),
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
    activeJobId       = jobId;
    activeEventSource = new EventSource(`/api/status/${jobId}`);

    activeEventSource.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.heartbeat) return;
        const pct = Math.round((msg.progress || 0) * 100);
        setProgress(pct, msg.message || '');
        if (msg.status === 'done') {
          activeEventSource.close();
          activeJobId = null;
          setProgress(100, 'Processing complete — redirecting…');
          setTimeout(() => { window.location.href = `/results/${jobId}`; }, 800);
        } else if (msg.status === 'error') {
          activeEventSource.close();
          activeJobId = null;
          setProgress(0, `Error: ${msg.message}`);
        } else if (msg.status === 'cancelled') {
          activeEventSource.close();
          activeJobId = null;
          setProgress(0, 'Cancelled — partial files deleted.');
          // Return to the file list (current folder), drop ?job=...
          setTimeout(() => {
            const url = new URL(window.location.href);
            url.searchParams.delete('job');
            window.location.href = url.toString();
          }, 1000);
        }
      } catch (e) { /* ignore */ }
    };

    activeEventSource.onerror = () => {
      fetch(`/api/results/${jobId}`)
        .then(r => { if (r.ok) window.location.href = `/results/${jobId}`; })
        .catch(() => {});
    };
  }

  if (cancelBtn) {
    cancelBtn.addEventListener('click', async () => {
      if (!activeJobId) return;
      const ok = window.confirm(
        'Cancel processing?\n\n' +
        'The pipeline will stop and any partial output files will be deleted.\n' +
        'This cannot be undone.'
      );
      if (!ok) return;
      cancelBtn.disabled = true;
      cancelBtn.textContent = 'Cancelling…';
      try {
        const r = await fetch(`/api/process/cancel/${encodeURIComponent(activeJobId)}`, {
          method: 'POST',
        });
        if (!r.ok) {
          const data = await r.json().catch(() => ({}));
          window.alert(data.error || 'Failed to cancel');
          cancelBtn.disabled = false;
          cancelBtn.textContent = '✕ Cancel';
        }
        // On success, the SSE will deliver `status: cancelled` and JS will
        // redirect to the file list.
      } catch (e) {
        window.alert('Network error: ' + e);
        cancelBtn.disabled = false;
        cancelBtn.textContent = '✕ Cancel';
      }
    });
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

  // ── Folder picker: reload page with ?folder=... when changed ─────────────
  const folderPicker = document.getElementById('folder-picker');
  if (folderPicker) {
    folderPicker.addEventListener('change', () => {
      const url = new URL(window.location.href);
      url.searchParams.set('folder', folderPicker.value);
      // Drop any incoming-job param so reloading doesn't auto-resume
      url.searchParams.delete('job');
      window.location.href = url.toString();
    });
  }

  // ── Previous Sessions password gate (cosmetic — see threat-model in design) ─
  const SESSIONS_PASSWORD = 'mumitmichael';
  const SESSIONS_UNLOCK_KEY = 'sessions_unlocked';
  const sessionsToggle = document.getElementById('sessions-toggle');
  const sessionsBody   = document.getElementById('sessions-body');
  const sessionsSection = document.getElementById('sessions-section');

  function unlockSessions() {
    if (sessionsBody)   sessionsBody.hidden = false;
    if (sessionsSection) sessionsSection.classList.remove('locked');
    if (sessionsToggle) sessionsToggle.setAttribute('aria-expanded', 'true');
  }

  if (sessionsToggle && sessionsBody) {
    if (sessionStorage.getItem(SESSIONS_UNLOCK_KEY) === '1') {
      unlockSessions();
    }
    sessionsToggle.addEventListener('click', () => {
      const expanded = sessionsToggle.getAttribute('aria-expanded') === 'true';
      if (expanded) {
        sessionsBody.hidden = true;
        sessionsSection.classList.add('locked');
        sessionsToggle.setAttribute('aria-expanded', 'false');
        sessionStorage.removeItem(SESSIONS_UNLOCK_KEY);
        return;
      }
      const pw = window.prompt('Enter password to view previous sessions:');
      if (pw === null) return;          // user cancelled
      if (pw === SESSIONS_PASSWORD) {
        sessionStorage.setItem(SESSIONS_UNLOCK_KEY, '1');
        unlockSessions();
      } else {
        window.alert('Incorrect password.');
      }
    });
  }
});
