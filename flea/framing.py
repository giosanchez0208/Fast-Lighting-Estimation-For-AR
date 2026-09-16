"""Canvas geometry.

Every FLEA sample, synthetic or real, satisfies the same framing contract:

    - square canvas of CANVAS px
    - subject bounding box scaled so its longest side is FIT_PX
    - subject centered

The residual border is background. It is not padding. It carries illumination
context, so its width is a design parameter.

The contract is deterministic, which is what makes it applicable to a detector
bounding box at inference: normalize_detection() below applies the identical
rule to a real crop.
"""

from __future__ import annotations

import numpy as np

CANVAS = 224
FIT_PX = 180
MARGIN = (CANVAS - FIT_PX) / 2  # 22.0 px per side

# Scale jitter applied at training time to absorb detector bbox slop. A tight
# fit of exactly FIT_PX is the canonical rule; real detections will not honour
# it to the pixel.
JITTER = (0.85, 1.00)


def fit_box(canvas: int = CANVAS, fit_px: int = FIT_PX) -> tuple[float, float, float, float]:
    """Subject bounding box in canvas pixels as (x0, y0, x1, y1)."""
    m = (canvas - fit_px) / 2
    return (m, m, canvas - m, canvas - m)


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Tight bounding box of a boolean mask as (x0, y0, x1, y1), exclusive."""
    if not mask.any():
        return None
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def mask_extent(mask: np.ndarray) -> float:
    """Longest side of the mask bounding box, in pixels."""
    box = mask_bbox(mask)
    if box is None:
        return 0.0
    x0, y0, x1, y1 = box
    return float(max(x1 - x0, y1 - y0))


def fit_error(mask: np.ndarray, target_px: float = FIT_PX) -> tuple[float, float]:
    """Return (achieved_extent, signed_error_px) against the target extent."""
    e = mask_extent(mask)
    return e, e - target_px


def normalize_detection(
    image: np.ndarray,
    bbox: tuple[float, float, float, float],
    canvas: int = CANVAS,
    fit_px: int = FIT_PX,
    fill: float = 0.0,
) -> np.ndarray:
    """Apply the training framing contract to a real detection.

    Crops `image` around `bbox`, scales the box's longest side to fit_px, and
    centers it on a canvas of the training resolution. Regions falling outside
    the source image are filled with `fill`, which is the one place a real crop
    can differ from a synthetic sample: a subject near the frame edge cannot
    supply its full border context.

    This is the inference-time counterpart of the synthetic generator, and the
    reason the generator centers its subject at a fixed extent.
    """
    x0, y0, x1, y1 = bbox
    scale = fit_px / max(x1 - x0, y1 - y0)

    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = canvas / (2 * scale)

    # Source-space window that maps onto the full canvas.
    sx0, sy0 = cx - half, cy - half
    src_h, src_w = image.shape[:2]

    yy, xx = np.mgrid[0:canvas, 0:canvas]
    su = np.floor(sx0 + (xx + 0.5) / scale).astype(np.int64)
    sv = np.floor(sy0 + (yy + 0.5) / scale).astype(np.int64)

    inside = (su >= 0) & (su < src_w) & (sv >= 0) & (sv < src_h)
    out = np.full((canvas, canvas) + image.shape[2:], fill, dtype=image.dtype)
    out[inside] = image[sv[inside], su[inside]]
    return out
