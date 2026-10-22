"""Automatic white balance after sky-background neutralization.

Neutralization equalizes the *background*; it does not guarantee that the stars
and the Milky Way are colour-neutral.  This step measures the sky's colour
(white-patch on the brightest sky pixels, optionally mixed with a gray-world
term) and applies a luminance-preserving per-channel gain.
"""
import numpy as np


def _gains_from_refs(refs):
    refs = np.asarray(refs, np.float64)
    target = float(refs.mean())
    return target / np.maximum(refs, 1e-8)


def auto_white_balance(rgb, mode="whitepatch", strength=0.6, gains=None,
                       sky_mask=None, wp_pct=99.8):
    """Return (white-balanced rgb, effective gains).

    mode: 'whitepatch' (brightest sky pixels), 'grayworld' (sky mean) or
    'mixed' (geometric mean of both).  `gains` overrides the measurement.
    `strength` blends in log space (0 = off, 1 = full).  Gains are normalised
    to preserve overall luminance.
    """
    if gains is not None:
        g = np.asarray(gains, np.float64)
    else:
        lum = rgb.mean(axis=2)
        if sky_mask is not None:
            sky = sky_mask > 0.5
        else:
            sky = lum <= np.percentile(lum, 80.0)
        if sky.sum() < 1000:
            sky = np.ones_like(lum, dtype=bool)
        g_gray = _gains_from_refs([np.median(rgb[..., c][sky]) for c in range(3)])
        hi = np.percentile(lum[sky], wp_pct)
        bright = sky & (lum >= hi)
        if bright.sum() < 100:
            bright = sky
        g_white = _gains_from_refs(
            [np.percentile(rgb[..., c][bright], 50.0) for c in range(3)])
        if mode == "grayworld":
            g = g_gray
        elif mode == "mixed":
            g = np.sqrt(g_gray * g_white)
        else:
            g = g_white
    g = np.exp(float(strength) * np.log(np.maximum(g, 1e-8)))
    g /= np.exp(np.log(g).mean())
    out = np.clip(rgb * g[None, None, :], 0, 1)
    return out, g
