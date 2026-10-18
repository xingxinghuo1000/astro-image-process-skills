"""Sky-ground boundary ("天地分界线") detection.

On a fixed tripod the sky moves while the foreground is static, so the two
regions are separated by *temporal* variance computed over the frames: the
variance of the raw (un-aligned) frames is high over the sky (the stars move)
and ~0 over the ground.  Aggregated over a few tens of pixels -- so the sparse
stars bleed into the smooth sky between them -- it is a reliable sky cue even
where the sky barely moves (e.g. near the celestial pole).

The boundary is the *lower envelope* of that sky cue: the lowest row per column
that still looks like sky.  Taking the lower envelope (rather than a connected
component) naturally bridges low-motion holes inside the sky and follows a
ragged tree/terrain line.  A robust straight-line fit through the same envelope
is offered as an alternative (`mode='line'`).
"""
import cv2
import numpy as np


def variance(sum_, sumsq, n):
    """Per-pixel temporal variance from running sum / sum-of-squares."""
    mean = sum_ / max(n, 1)
    return np.maximum(sumsq / max(n, 1) - mean * mean, 0.0)


def sky_evidence(var_raw, sigma, norm_pct=99.5):
    """Blur the raw variance into a smooth sky-likelihood and normalize it to a
    99.5th-percentile of 1, so the threshold is scene-independent."""
    e = cv2.GaussianBlur(var_raw, (0, 0), sigma)
    hi = float(np.percentile(e, norm_pct))
    return (e / hi).astype(np.float32) if hi > 0 else e.astype(np.float32)


def _lower_envelope(evidence, t=0.05, win=41, smooth=30):
    """Lowest row per column whose `win`-tall average evidence exceeds `t`,
    then filled and smoothed across columns.  Returns None if there is no
    confident sky anywhere."""
    sm = cv2.blur(evidence, (1, win))
    rows = np.arange(sm.shape[0], dtype=np.float32)[:, None]
    env = np.where(sm > t, rows, -1.0).max(0)
    env[env < 0] = np.nan
    x = np.arange(env.size)
    ok = np.isfinite(env)
    if ok.sum() < 10:
        return None
    env = np.interp(x, x[ok], env[ok]).astype(np.float32)
    if smooth > 0:
        env = cv2.GaussianBlur(env.reshape(1, -1), (0, 0), smooth).ravel()
    return env


def _robust_line(xs, ys, iters=6, min_pts=10):
    """Robust y = a*x + b fit (iteratively re-weighted least squares);
    returns (a, b, inlier_count) or (None, None, 0)."""
    ys = np.asarray(ys, np.float64)
    xs = np.asarray(xs, np.float64)
    if xs.size < min_pts:
        return None, None, 0
    a, b = np.polyfit(xs, ys, 1)
    for _ in range(iters):
        r = ys - (a * xs + b)
        thr = max(3.0 * 1.4826 * np.median(np.abs(r)), 6.0)
        keep = np.abs(r) < thr
        if keep.sum() < min_pts:
            break
        a, b = np.polyfit(xs[keep], ys[keep], 1)
    r = np.abs(ys - (a * xs + b))
    n = int((r < max(3.0 * 1.4826 * np.median(r), 6.0)).sum())
    return float(a), float(b), n


def _envelope_mask(shape, env, feather):
    """Feathered half-plane mask: 1 above the per-column envelope (sky)."""
    yy = np.arange(shape[0], dtype=np.float32)[:, None]
    return np.clip((env[None, :] - yy) / feather + 0.5, 0, 1)


def _line_mask(shape, a, b, feather):
    """Feathered half-plane mask: 1 above y = a*x + b (sky)."""
    xx = np.arange(shape[1], dtype=np.float32)[None, :]
    yy = np.arange(shape[0], dtype=np.float32)[:, None]
    return np.clip(((a * xx + b) - yy) / feather + 0.5, 0, 1)


def detect(evidence, full_shape, mode="mask", t=0.05, win=41, smooth=30,
           feather=25.0):
    """Return (full-res float sky mask, info dict).

    `mode='mask'` follows the per-column envelope (ragged tree/terrain line);
    `mode='line'` substitutes a robust straight-line fit through the envelope."""
    hh, ww = evidence.shape
    scale = full_shape[1] / float(ww)          # full-res px per evidence px
    feather_h = max(1.0, feather / scale)      # blend band in evidence px
    env = _lower_envelope(evidence, t=t, win=win, smooth=smooth)
    if env is None:
        return None, dict(failed=True, reason="no sky region found")
    a, b, npts = _robust_line(np.arange(ww), env)
    if mode == "line" and a is not None:
        half = _line_mask((hh, ww), a, b, feather_h)
    else:
        half = _envelope_mask((hh, ww), env, feather_h)
    full = np.clip(cv2.resize(half, (full_shape[1], full_shape[0]),
                              interpolation=cv2.INTER_LINEAR), 0, 1)
    info = dict(failed=False, mode=mode, slope=a, intercept=b, inliers=npts,
                coverage=float(full.mean()))
    return full, info
