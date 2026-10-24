"""Per-channel sky-background neutralization (PixInsight style).

On composites the measurement AND the correction are restricted to the sky
(sky_mask): the darkest pixels of a frame with a large dark foreground are
ground, not sky, and equalising global gains onto the whole frame turns the
aurora band neon and the zenith magenta.
"""
import numpy as np


def background_reference(rgb, bg_pct=0.5, sky_mask=None):
    """Per-channel robust sky-background level.

    Only the darker `bg_pct` fraction of pixels (by luminance) is used, so the
    estimate is not biased by stars or the bright Milky Way core.  Median (not
    mean) keeps it robust to the few remaining bright pixels.  With `sky_mask`
    the percentile is taken over SKY pixels only (1 = sky, feathered): a dark
    foreground no longer monopolises the estimate."""
    lum = rgb.mean(axis=2)
    if sky_mask is not None:
        sky = np.asarray(sky_mask) > 0.5
        if sky.sum() < 1000:
            sky = np.ones_like(lum, dtype=bool)
        thr = np.percentile(lum[sky], bg_pct * 100.0)
        m = sky & (lum <= thr)
    else:
        thr = np.percentile(lum, bg_pct * 100.0)
        m = lum <= thr
    if m.sum() < 100:
        m = np.ones_like(lum, dtype=bool)
    return np.array([np.median(rgb[..., c][m]) for c in range(3)], np.float64)


def background_neutralize(rgb, bg_pct=0.5, sky_mask=None, strength=1.0):
    """Make the sky background a neutral grey, PixInsight BackgroundNeutralization
    style: measure the background reference per channel and scale each channel so
    they all match the average reference (a pure grey).  Applied on stretched
    data.  `strength` blends the log-gain (0 = off, 1 = full).  With `sky_mask`
    (1 = sky, feathered float) the gains are applied to the sky only, leaving
    the ground untouched.  Returns (neutralized_rgb, reference, measured gains)."""
    ref = background_reference(rgb, bg_pct, sky_mask)
    target = float(ref.mean())
    if target < 1e-6:
        return rgb, ref, np.ones(3)
    gains = target / np.maximum(ref, 1e-6)
    g = np.exp(float(strength) * np.log(np.maximum(gains, 1e-8)))
    if sky_mask is not None:
        w = np.clip(sky_mask, 0, 1)[..., None]
        out = rgb * (1.0 + (g - 1.0) * w)
    else:
        out = rgb * g[None, None, :]
    return np.clip(out, 0, 1), ref, gains
