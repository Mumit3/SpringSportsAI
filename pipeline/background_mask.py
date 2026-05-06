"""Background dimming for YOLO inference.

When `config.BACKGROUND_MASK_ENABLED` is True, the pipeline pre-processes each
frame before passing it to the detector. The aim is to keep foreground pixels
(the ball, the rim, the shooter) at full brightness while dimming everything
else — busy whiteboards, posters, distant clutter, oddly-coloured walls — so
YOLO is less confused by them.

Two cues are unioned to identify "foreground":

  1. Depth — pixels whose Z is close to the locked hoop's depth (or, when the
     hoop hasn't locked yet, a generous near-field range).
  2. Colour — HSV pixels that match a basketball-orange range.

Either cue alone is enough to mark a pixel as foreground (a player isn't
orange but is at the right depth; a ball might briefly fall outside the depth
band due to depth dropouts but is still orange). The mask is then dilated a
little so the ball's edges aren't shaved off, and finally used to blend the
original frame with a darkened version of itself.

The original frame is never modified — only a copy is returned. The annotated
output video continues to look exactly as before; this masking is purely a
preprocessing step for the detector.
"""
from __future__ import annotations

from typing import Optional
import numpy as np
import cv2

from . import config


def _build_mask(
    image:       np.ndarray,
    point_cloud: Optional[np.ndarray],
    hoop_3d:     Optional[np.ndarray],
) -> np.ndarray:
    """Return a uint8 foreground mask (255 = keep, 0 = background)."""
    h, w = image.shape[:2]

    # ── depth band ────────────────────────────────────────────────────────────
    # Foreground = pixels from the camera up to slightly past the hoop. The
    # range is asymmetric: we want to keep the ball (always closer than or at
    # the hoop's depth) and reject the wall behind the hoop. A small margin
    # past hoop_z catches the rim itself plus a person holding a portable hoop.
    depth_mask = np.zeros((h, w), dtype=np.uint8)
    if point_cloud is not None and point_cloud.shape[:2] == (h, w):
        z = point_cloud[..., 2]
        depth_lo = float(config.BACKGROUND_DEPTH_NEAR)
        if hoop_3d is not None:
            depth_hi = float(hoop_3d[2]) + float(config.BACKGROUND_DEPTH_BEHIND_HOOP)
        else:
            depth_hi = float(config.BACKGROUND_DEPTH_FALLBACK_MAX)
        depth_ok = (z > depth_lo) & (z < depth_hi) & np.isfinite(z)
        depth_mask[depth_ok] = 255
    else:
        # No depth available — fall back to keeping everything; the colour
        # mask alone will still focus things, and the detector behaves as today.
        depth_mask[:] = 255

    # ── orange colour band ────────────────────────────────────────────────────
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lo  = np.array(config.BALL_HSV_LOWER, dtype=np.uint8)
    hi  = np.array(config.BALL_HSV_UPPER, dtype=np.uint8)
    color_mask = cv2.inRange(hsv, lo, hi)

    # ── union + dilate so we don't shave object edges ─────────────────────────
    mask = cv2.bitwise_or(depth_mask, color_mask)
    if config.BACKGROUND_MASK_DILATE_PX > 0:
        k = config.BACKGROUND_MASK_DILATE_PX
        kernel = np.ones((k, k), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def apply(
    image:       np.ndarray,
    point_cloud: Optional[np.ndarray],
    hoop_3d:     Optional[np.ndarray],
) -> np.ndarray:
    """Return a new frame with non-foreground pixels dimmed for YOLO input.

    No-op (returns the original array) if the feature is disabled.
    """
    if not config.BACKGROUND_MASK_ENABLED:
        return image

    mask = _build_mask(image, point_cloud, hoop_3d)

    dim = float(config.BACKGROUND_DIM_FACTOR)
    if dim < 0.0: dim = 0.0
    if dim > 1.0: dim = 1.0

    # Smooth the mask boundary so we don't get sharp halos that confuse YOLO.
    mask_blur = cv2.GaussianBlur(mask, (15, 15), 0)
    alpha = (mask_blur.astype(np.float32) / 255.0)[..., None]

    fg = image.astype(np.float32)
    bg = fg * dim
    out = fg * alpha + bg * (1.0 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)
