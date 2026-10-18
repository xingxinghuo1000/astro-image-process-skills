"""Adaptive brightness stretch (PixInsight auto MTF style)."""
import numpy as np


def _mtf(m, x):
    """Midtones transfer function (PixInsight style); m=0.5 is identity."""
    return ((m - 1.0) * x) / ((2.0 * m - 1.0) * x - m)


def auto_stretch(rgb_lin, target_bg=0.20, black_pct=0.3, white_pct=99.95):
    """Adaptive stretch: robust per-channel black point, normalize by a high
    percentile, then pick the midtone-transfer parameter so the sky background
    lands at `target_bg` -- lifts the faint Milky Way without blowing stars.
    Returns (stretched_rgb, mtf_m, background_level)."""
    x = np.clip(rgb_lin, 0, None)
    blk = np.percentile(x, black_pct, axis=(0, 1))
    x = np.clip(x - blk, 0, None)
    wht = float(np.percentile(x, white_pct))
    xn = x / wht if wht > 0 else x
    bg = float(np.median(xn.mean(axis=2)))
    lo, hi = 1e-4, 0.999
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _mtf(mid, bg) > target_bg:
            lo = mid
        else:
            hi = mid
    m = 0.5 * (lo + hi)
    y = _mtf(m, np.clip(xn, 0, 1))
    return np.clip(y, 0, 1), m, bg


def stretch_preview(rgb_lin, gain=50.0):
    """Simple arcsinh preview of a linear image (for the linear master thumb)."""
    x = np.clip(rgb_lin, 0, None)
    x = np.arcsinh(x * gain) / np.arcsinh(gain)
    return np.clip(x * 255, 0, 255).astype(np.uint8)
