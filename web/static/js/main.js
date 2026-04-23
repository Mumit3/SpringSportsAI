/* main.js — index page interactions */

document.addEventListener('DOMContentLoaded', () => {
  const fileBtns     = document.querySelectorAll('.file-btn');
  const progressPanel = document.getElementById('progress-panel');
  const progressBar   = document.getElementById('progress-bar');
  const progressMsg   = document.getElementById('progress-msg');
  const progressPct   = document.getElementById('progress-pct');

  if (!fileBtns.length) return;

  let activeEventSource = null;

  fileBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      // Highlight selected button
      fileBtns.forEach(b => b.classList.remove('selected'));
      btn.classList.add('selected');

      const filename = btn.dataset.filename;
      startProcessing(filename);
    });
  });

  function startProcessing(filename) {
    // Show progress panel
    progressPanel.classList.remove('hidden');
    setProgress(0, `Starting ${filename}…`);

    // Close any existing SSE connection
    if (activeEventSource) {
      activeEventSource.close();
    }

    fetch('/api/process', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ filename }),
    })
    .then(r => r.json())
    .then(data => {
      if (data.status === 'error') {
        setProgress(0, `Error: ${data.error}`);
        return;
      }
      const jobId = data.job_id;
      listenForProgress(jobId);
    })
    .catch(err => {
      setProgress(0, `Network error: ${err}`);
    });
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
          setTimeout(() => {
            window.location.href = `/results/${jobId}`;
          }, 800);
        } else if (msg.status === 'error') {
          activeEventSource.close();
          setProgress(0, `Error: ${msg.message}`);
        }
      } catch (e) {
        // Ignore parse errors on heartbeat
      }
    };

    activeEventSource.onerror = () => {
      // SSE connection dropped — check if job finished
      fetch(`/api/results/${jobId}`)
        .then(r => {
          if (r.ok) {
            window.location.href = `/results/${jobId}`;
          }
        })
        .catch(() => {});
    };
  }

  function setProgress(pct, msg) {
    if (progressBar) progressBar.style.width = `${pct}%`;
    if (progressMsg) progressMsg.textContent  = msg;
    if (progressPct) progressPct.textContent  = `${pct}%`;
  }
});
