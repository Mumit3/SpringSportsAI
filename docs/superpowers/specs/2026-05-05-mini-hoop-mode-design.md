# Mini-Hoop Mode — Design

**Date:** 2026-05-05
**Branch:** 3DPlotting
**Author:** brainstorm with Mumit

## Goal

Add an opt-in "Mini Hoop" processing mode to the existing pipeline so the
system can analyze recordings of mini-hoop / Pop-A-Shot style setups. Must
not affect any current regulation behavior — existing SVOs, existing sessions,
existing outputs, and the existing UI flow remain identical for regulation.

## Constraints

- Regulation pipeline behavior is unchanged. No regression risk on existing videos.
- Demo deadline: 2026-05-06. Implementation must be small enough to land safely tonight.
- Single-user-on-Jetson assumption: jobs are processed serially by the Flask backend.
- YOLO model is unchanged (no retraining). Mini-hoop accuracy depends on whether the
  current model generalizes to smaller ball / smaller rim. This is a known unknown.

## User-facing surface

### Index page (`/`)

Two stacked sections under the existing stats overview:

1. **Process SVO (Regulation)** — existing section, just retitled.
   - Lists SVO files found anywhere under the project tree, **excluding** any path
     that contains `mini/` as a directory component. Backwards-compatible: every
     SVO file in the project today appears here.
2. **Process SVO (Mini Hoop)** — new.
   - Lists SVO files found only in `svo_files/mini/`.
   - Each card has the same label-input + Start button as regulation.
   - Each card also has a checkbox: **"Ball tracking only (skip make/miss)"**.
     When checked, the pipeline runs ball detection and the 3-D trail but skips
     rim detection and shot classification entirely.

The Previous Sessions table adds a small `REG` / `MINI` badge per row, derived
from the `mode` field in each session's `analytics.json`. Sessions without that
field (existing sessions) display `REG` by default.

### CLI (`main.py`)

Two new optional flags:

- `--profile {regulation,mini}` — defaults to `regulation`.
- `--ball-only` — skips rim detection / classification. Independent of profile;
  most useful when paired with `--profile mini`.

Behavior with no flags is identical to the current CLI.

## Implementation

### 1. New module: `pipeline/profiles.py`

Two named profiles as a dict, plus an `apply_profile(name)` function that
mutates `pipeline.config` module attributes and returns a context-manager
object that restores the originals on exit.

```python
PROFILES = {
    "regulation": {
        # Empty — represents "no overrides", values remain as defined in config.py
    },
    "mini": {
        "BALL_DIAMETER_M":        0.13,
        "BALL_SIZE_TOLERANCE":    0.50,
        "BALL_MIN_Y":             0.10,
        "BALL_MIN_Z":             1.00,
        "BALL_HOOP_Z_TOLERANCE":  3.00,
        "MAKE_CYLINDER_RADIUS":   0.10,
    },
}
```

Usage:

```python
with profile_scope(name):
    # pipeline runs with overridden values
    ...
# config restored automatically, even on exception
```

### 2. Pipeline entry: `main.py`

`process_svo(...)` gains two optional kwargs:

- `profile: str = "regulation"`
- `ball_only: bool = False`

Job execution wraps the existing pipeline body in `with profile_scope(profile):`.

When `ball_only=True`, the pipeline skips:
- Hoop detection / locking (`hoop_det = None` always)
- Shot detection (`ShotDetector.update()` is not called)
- Make/miss classification

Ball tracking, 3-D trail capture per frame, and the `traces_3d.json` output
still happen. The output `analytics.json` will have:
- `mode`: `"regulation"` or `"mini"`
- `ball_only`: `true` or `false`
- `total_shots`: 0
- `summary` fields: zeroed out or marked `"n/a"`

When ball_only is False but no shots are detected (rim never locked, etc.),
behavior is identical to current.

### 3. CLI flags: `main.py:_cli()`

Two new `argparse` arguments. The constructed call to `process_svo` passes
them through.

### 4. Flask backend: `web/app.py`

- `_list_svo_files()` is split or parameterized to return either:
  - `regulation` files: recursive scan, exclude any path containing `/mini/`.
  - `mini` files: recursive scan inside `svo_files/mini/` only.
- `/api/process` accepts new optional JSON fields:
  - `mode`: `"regulation"` (default) or `"mini"`
  - `ball_only`: bool, default false
- Job runner passes both into `process_svo()`.
- Sessions index reads `mode` from each session's `analytics.json` (defaulting
  to `"regulation"` when absent) and exposes it on the row data.

### 5. Templates: `web/templates/index.html`

Add a second `<section class="section">` for mini hoop after the existing
regulation section. Same card markup; cards include the new ball-only checkbox.

Sessions row template adds a `<span class="badge badge-{{mode}}">` displaying
`REG` or `MINI`.

### 6. JS: `web/static/js/main.js`

Existing card click / start logic is reused. Card section determines the
`mode` value sent to `/api/process`. Ball-only checkbox value is included
in the request payload when present.

### 7. CSS: `web/static/css/style.css`

Add badge styling: small pill, neutral grey for REG, accent color for MINI.

## What this design does NOT solve

1. **YOLO accuracy on mini gear is unverified.** If the current model can't
   detect the mini ball or rim, this design provides no fallback beyond
   `--ball-only` (which at least visualizes whatever IS detected).
2. **Make/miss tuning is unvalidated.** The mini profile values
   (`MAKE_CYLINDER_RADIUS = 0.10` etc.) are educated guesses. Real values
   depend on the specific mini hoop geometry and may need iteration.
3. **Concurrent jobs aren't safe.** Profile mutates `pipeline.config` at
   module level. If two jobs ran in parallel they could clash. Mitigated
   by the existing single-worker job runner.

## Risk summary

| Risk | Likelihood | Impact | Mitigation |
|--|--|--|--|
| Profile values wrong for mini hoop | High | Bad classifications | `--ball-only` escape hatch |
| YOLO doesn't detect mini gear | Medium | No detections at all | Test tonight before demo |
| Concurrent job config clash | Low | Wrong values used | Existing serial job runner |
| Regression on regulation videos | Very low | Demo failure | No regulation code paths change |

## Out of scope

- Retraining YOLO for mini-scale gear.
- Concurrent-safe config refactor (separate task if multi-user becomes a need).
- Per-mode output folders. Mode lives in `analytics.json`, files share folders.
- Migration of existing session JSONs to add `mode` field (defaults to regulation
  on read).
