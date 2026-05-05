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

  let activeEventSource = null;

  cards.forEach(card => {
    const labelInput = card.querySelector('.file-card-label');
    const startBtn   = card.querySelector('.file-card-start');
    const cancelBtn  = card.querySelector('.file-card-cancel');

    // Click on card body (but not action area) → expand
    card.addEventListener('click', (e) => {
      // If the click was on a button/input inside the action area, let the
      // dedicated handler deal with it.
      if (e.target.closest('.file-card-action')) return;
      cards.forEach(c => { if (c !== card) c.classList.remove('selected'); });
      card.classList.toggle('selected');
      if (card.classList.contains('selected') && labelInput) {
        labelInput.focus();
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
        const label = (labelInput.value || '').trim();
        startProcessing(path, label);
      });
    }
  });

  function startProcessing(path, label) {
    cards.forEach(c => c.classList.remove('selected'));
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
