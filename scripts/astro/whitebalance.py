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
                       sky_mask=None, wp_pct=99.98):
    """Return (white-balanced rgb, effective gains).

    mode: 'whitepatch' (brightest sky pixels), 'grayworld' (sky mean) or
    'mixed' (geometric mean of both).  `gains` overrides the measurement.
    `strength` blends in log space (0 = off, 1 = full).  Gains are normalised
    to preserve overall luminance.

    `sky_mask` (1 = sky, feathered float) restricts BOTH the measurement and
    the correction to the sky: a dark ground must not bias the stats, and the
    ground must not be tinted by star-colour gains.  wp_pct=99.98 keeps the
    white patch on star cores; the diffuse aurora/glow fills the brightest
    0.2% and would be read as "white", desaturating the very subject.
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
        # star cores, NOT saturated ones: a stretched sequence clips its
        # brightest stars at 1.0, and "white patch = saturated pixels" is
        # neutral by construction -- the gain collapses to a no-op. Keep the
        # bright star pixels just below saturation, whose colour is real.
        hi = np.percentile(lum[sky], wp_pct)
        sat = lum >= 0.995
        bright = sky & (lum >= hi) & (~sat)
        if bright.sum() < 100:
            bright = sky & (lum >= np.percentile(lum[sky], 99.9)) & (~sat)
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
    if sky_mask is not None:
        w = np.clip(sky_mask, 0, 1)[..., None]
        out = rgb * (1.0 + (g - 1.0) * w)
    else:
        out = rgb * g[None, None, :]
    return np.clip(out, 0, 1), g
