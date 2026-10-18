"""Automatic background / gradient removal (PixInsight ABE, a.k.a. "auto flat")."""
import numpy as np


def auto_flat(rgb, bins=16, degree=3, pct=25.0, mode="subtract", strength=1.0):
    """Remove the smooth low-frequency background, PixInsight
    AutomaticBackgroundExtractor (ABE, a.k.a. "auto flat") style.

    Splits the image into a grid of tiles, estimates each tile's sky level with a
    low percentile (robust to the sparse stars), fits a 2D polynomial surface per
    channel through those tile levels, then subtracts (or divides by) the model.
    Run on LINEAR data, where gradients (light pollution, vignetting, airglow)
    are additive and therefore physically meaningful to fit.  The polynomial is
    deliberately low order so the narrow Milky Way is *not* absorbed.

    In `subtract` mode only the gradient *shape* is removed: a pedestal at the
    model's 10th percentile (the dark-sky level) is kept so the background is not
    floored to zero, which would clip the sky noise and starve later steps.

    `strength` (0..1) blends the correction.  When the field is filled edge to
    edge with the Milky Way / a broad glow, a low-order model cannot tell that
    glow from the sky background, so a full-strength correction carves the centre
    and leaves a bright rim (ring) at the border; dialing strength down trades
    some gradient removal for a clean, ring-free background.

    Returns (corrected_rgb, per-channel model info)."""
    h, w = rgb.shape[:2]
    nx = max(2, int(bins))
    ny = max(2, int(round(bins * h / w)))
    xs = ((np.arange(nx) + 0.5) / nx) * 2.0 - 1.0
    ys = ((np.arange(ny) + 0.5) / ny) * 2.0 - 1.0
    terms = [(i, j) for i in range(degree + 1) for j in range(degree + 1)
             if i + j <= degree]
    A = np.empty((nx * ny, len(terms)), np.float64)
    for yi in range(ny):
        for xi in range(nx):
            for t, (i, j) in enumerate(terms):
                A[yi * nx + xi, t] = (xs[xi] ** i) * (ys[yi] ** j)
    xe = np.linspace(0, w, nx + 1).astype(int)
    ye = np.linspace(0, h, ny + 1).astype(int)
    gx = (np.linspace(0, w - 1, w) / max(w - 1, 1)) * 2.0 - 1.0
    gy = (np.linspace(0, h - 1, h) / max(h - 1, 1)) * 2.0 - 1.0

    out = rgb.copy()
    info = []
    for c in range(3):
        vals = np.empty(nx * ny, np.float64)
        for yi in range(ny):
            for xi in range(nx):
                tile = rgb[ye[yi]:ye[yi + 1], xe[xi]:xe[xi + 1], c]
                vals[yi * nx + xi] = np.percentile(tile, pct) if tile.size else 0.0
        coef = np.linalg.lstsq(A, vals, rcond=None)[0]
        model = np.zeros((h, w), np.float32)
        for (i, j), cc in zip(terms, coef):
            model += cc * ((gy ** j)[:, None] * (gx ** i)[None, :])
        if mode == "divide":
            corrected = np.clip(rgb[..., c] / np.maximum(model, 1e-6), 0, 1)
        else:
            # Subtract only the gradient *shape*: keep a pedestal at the model's
            # 10th percentile (the dark-sky level) so flat regions are not
            # floored to zero -- otherwise the whole background clips and later
            # steps break.
            ped = max(0.0, float(np.percentile(model, 10.0)))
            corrected = np.clip(rgb[..., c] - np.clip(model - ped, 0, None), 0, 1)
        out[..., c] = np.clip(rgb[..., c] + strength * (corrected - rgb[..., c]), 0, 1)
        info.append((float(model.min()), float(model.max()), float(model.mean())))
    return out, info
