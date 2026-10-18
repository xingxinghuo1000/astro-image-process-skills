"""Per-channel sky-background neutralization (PixInsight style)."""
import numpy as np


def background_reference(rgb, bg_pct=0.5):
    """Per-channel robust sky-background level.

    Only the darker `bg_pct` fraction of pixels (by luminance) is used, so the
    estimate is not biased by stars or the bright Milky Way core.  Median (not
    mean) keeps it robust to the few remaining bright pixels."""
    lum = rgb.mean(axis=2)
    thr = np.percentile(lum, bg_pct * 100.0)
    m = lum <= thr
    if m.sum() < 100:
        m = np.ones_like(lum, dtype=bool)
    return np.array([np.median(rgb[..., c][m]) for c in range(3)], np.float64)


def background_neutralize(rgb, bg_pct=0.5):
    """Make the sky background a neutral grey, PixInsight BackgroundNeutralization
    style: measure the background reference per channel and scale each channel so
    they all match the average reference (a pure grey).  Applied on linear data.
    Returns (neutralized_rgb, reference, gains)."""
    ref = background_reference(rgb, bg_pct)
    target = float(ref.mean())
    if target < 1e-6:
        return rgb, ref, np.ones(3)
    gains = target / np.maximum(ref, 1e-6)
    out = np.clip(rgb * gains[None, None, :], 0, 1)
    return out, ref, gains
