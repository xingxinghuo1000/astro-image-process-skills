"""Single-frame, per-channel hot-pixel detection and repair.

A hot pixel is one photosensor with a high dark current: after demosaic it shows
up as an isolated, colored (pure R/G/B) single-pixel dot that survives stacking.
It is a *fixed* sensor defect, so with a tracking mount it moves relative to the
sky and -- under a mean stack -- is diluted to a faint colored streak; median
stacking already rejects it.  For the residuals near the rotation centre (where
the defect barely moves) we repair it directly.

The detector works per channel on a single frame: the local background is a
median filter (which ignores an isolated spike), so ``channel - median`` peaks
at a defect.  A pixel is a defect only if that residual clears a robust (MAD)
and absolute threshold, it clearly dominates *every* one of its 8 neighbours (a
hot pixel is a lone peak; a star edge has a bright neighbour), and it is
dominated by this one channel (a hot pixel is a colored dot; a white star is
not).  Repair replaces the flagged channel-pixel with that channel's local
median, leaving every other pixel untouched.
"""
import os
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np

from .stretch import stretch_preview
from .render import _label

_NEIGH_KERNEL = np.ones((3, 3), np.uint8)
_NEIGH_KERNEL[1, 1] = 0


def _neighbor_max(a):
    """Max of the 8 neighbours of every pixel (excludes the centre)."""
    return cv2.dilate(a.astype(np.float32), _NEIGH_KERNEL)


def _residuals(rgb, ksize, workers=3):
    """Per-channel ``channel - medianBlur(channel)``, three channels in parallel."""
    h, w = rgb.shape[:2]
    resid = np.empty((h, w, 3), np.float32)

    def one(c):
        ch = rgb[..., c].astype(np.float32)
        return c, ch - cv2.medianBlur(ch, ksize)

    workers = max(1, min(int(workers), 3))
    if workers == 1:
        for c in range(3):
            resid[..., c] = one(c)[1]
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for c, r in ex.map(one, range(3)):
                resid[..., c] = r
    return resid


def hot_pixel_flags(rgb, sigma=6.0, ksize=5, thr_abs=0.0, color_ratio=2.0,
                    neigh_ratio=1.0, workers=3):
    """Per-channel bool map of the *peak* pixel of each colored hot pixel.

    `sigma` is the robust (MAD) threshold multiplier, `thr_abs` an absolute
    residual floor, `color_ratio` how much a defect must exceed the other two
    channels, `neigh_ratio` how much it must exceed the brightest of its 8
    neighbours (1.0 = strict local maximum).  A single hot photosite spreads
    over several px after demosaic, so this flags only the peak; `repair`
    expands each peak over the rest of the (colored, elevated) blob."""
    h, w = rgb.shape[:2]
    resid = _residuals(rgb, ksize, workers)
    flags = np.zeros((h, w, 3), bool)

    def one(c):
        r = resid[..., c]
        med = float(np.median(r))
        mad = float(np.median(np.abs(r - med))) * 1.4826 + 1e-9
        thr = max(med + sigma * mad, med + thr_abs)
        cand = r > thr
        if cand.any():
            cand &= r > neigh_ratio * np.maximum(_neighbor_max(r), 0.0)
            others = np.maximum(resid[..., (c + 1) % 3],
                                resid[..., (c + 2) % 3])
            cand &= r > color_ratio * np.maximum(others, 0.0)
        return c, cand

    workers = max(1, min(int(workers), 3))
    if workers == 1:
        for c in range(3):
            flags[..., c] = one(c)[1]
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for c, cand in ex.map(one, range(3)):
                flags[..., c] = cand
    return flags


def apply_hot_pixels(rgb, flags, grow=3, thr_low=0.03, color_ratio=2.0,
                     workers=3):
    """Cosmetic repair of the flagged hot pixels.

    A single flagged peak is only one pixel of a multi-pixel demosaic blob, so
    the replacement is grown over the whole blob: every pixel within `grow` of a
    peak that is still colored (dominates the other two channels) and elevated
    (above `thr_low`) is replaced.  The replacement value is a wide (double 5x5)
    median which -- unlike a single 5x5 -- ignores the blob, yet -- unlike a
    morphological opening -- does not erase nearby stars.

    Returns (repaired_rgb, replaced_channel_px)."""
    out = rgb.copy()
    resid = _residuals(rgb, 5, workers)
    rect = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * grow + 1,) * 2)

    def one(c):
        ch = rgb[..., c].astype(np.float32)
        bg1 = cv2.medianBlur(ch, 5)
        bg = cv2.medianBlur(bg1, 5)
        others = np.maximum(resid[..., (c + 1) % 3],
                            resid[..., (c + 2) % 3])
        grown = cv2.dilate(flags[..., c].astype(np.uint8), rect) > 0
        mask = grown & (resid[..., c] > thr_low) & \
            (resid[..., c] > color_ratio * np.maximum(others, 0.0))
        return c, bg, mask, int(mask.sum())

    n = 0
    workers = max(1, min(int(workers), 3))
    if workers == 1:
        results = (one(c) for c in range(3))
    else:
        ex = ThreadPoolExecutor(max_workers=workers)
        results = ex.map(one, range(3))
    try:
        for c, bg, mask, count in results:
            out[..., c][mask] = bg[mask]
            n += count
    finally:
        if workers > 1:
            ex.shutdown()
    return np.clip(out, 0, 1), n


def detect_and_fix(rgb, sigma=6.0, ksize=5, thr_abs=0.0, color_ratio=2.0,
                   neigh_ratio=1.0, grow=3, workers=3):
    """Convenience wrapper: return (repaired_rgb, replaced_channel_px)."""
    flags = hot_pixel_flags(rgb, sigma, ksize, thr_abs, color_ratio,
                            neigh_ratio, workers)
    return apply_hot_pixels(rgb, flags, grow=grow, color_ratio=color_ratio,
                            workers=workers)


def metrics(rgb, sigma=6.0, ksize=5, thr_abs=0.0, color_ratio=2.0,
            neigh_ratio=1.0, workers=3):
    """Audit of a linear image: flagged peaks and distinct hot pixels."""
    flags = hot_pixel_flags(rgb, sigma, ksize, thr_abs, color_ratio,
                            neigh_ratio, workers)
    return dict(hot_channel_px=int(flags.sum()),
                hot_pixels=int(flags.reshape(-1, 3).any(1).sum()))


def write_compare(out_dir, before, after, tag, sigma=6.0, ksize=5,
                  thr_abs=0.0, color_ratio=2.0, neigh_ratio=1.0,
                  n_crops=3, crop=90, scale=4.0):
    """Before/after figure: full-frame previews plus zoomed crops centred on
    the strongest colored hot pixels, so the effect can be judged by eye."""
    flags = hot_pixel_flags(before, sigma, ksize, thr_abs, color_ratio,
                            neigh_ratio)
    resid = _residuals(before, ksize)
    score = np.where(flags, resid, 0).max(2)
    ys, xs = np.where(flags.any(2))
    picked = []
    if len(ys):
        val = score[ys, xs]
        for i in np.argsort(-val):
            y, x = int(ys[i]), int(xs[i])
            if all((y - py) ** 2 + (x - px) ** 2 > crop ** 2 for py, px in picked):
                picked.append((y, x))
            if len(picked) >= n_crops:
                break

    bs = stretch_preview(before)
    as_ = stretch_preview(after)
    H, W = bs.shape[:2]

    cw = int(2 * crop * scale)
    full_b = cv2.resize(bs, (cw, max(1, int(bs.shape[0] * cw / bs.shape[1]))),
                        interpolation=cv2.INTER_AREA)
    full_a = cv2.resize(as_, (cw, max(1, int(as_.shape[0] * cw / as_.shape[1]))),
                        interpolation=cv2.INTER_AREA)

    def panel(img):
        if img.shape[1] > cw:
            img = cv2.resize(img, (cw, max(1, int(img.shape[0] * cw / img.shape[1]))))
        if img.shape[1] < cw:
            img = np.hstack([img, np.zeros((img.shape[0], cw - img.shape[1], 3),
                                           np.uint8)])
        return img

    left = [panel(_label(full_b, "before"))]
    right = [panel(_label(full_a, "after"))]
    for y, x in picked:
        y0, y1 = max(0, y - crop), min(H, y + crop)
        x0, x1 = max(0, x - crop), min(W, x + crop)
        left.append(panel(cv2.resize(bs[y0:y1, x0:x1], None, fx=scale, fy=scale,
                                     interpolation=cv2.INTER_NEAREST)))
        right.append(panel(cv2.resize(as_[y0:y1, x0:x1], None, fx=scale, fy=scale,
                                      interpolation=cv2.INTER_NEAREST)))
    gap = np.zeros((4, cw, 3), np.uint8)

    def column(panels):
        return np.vstack([p for pair in zip(panels, [gap] * len(panels))
                          for p in pair][:2 * len(panels) - 1])

    lcol, rcol = column(left), column(right)
    h = min(lcol.shape[0], rcol.shape[0])
    sep = np.zeros((h, 4, 3), np.uint8)
    fig = np.hstack([lcol[:h], sep, rcol[:h]])
    path = os.path.join(out_dir, "hot_compare_%s.jpg" % tag)
    cv2.imwrite(path, cv2.cvtColor(fig, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 93])
    return path, int(flags.reshape(-1, 3).any(1).sum())
