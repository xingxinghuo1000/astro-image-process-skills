"""Star reduction: shrink stars under a high-pass star mask (PixInsight style)."""
import cv2
import numpy as np


def make_star_mask(lum, sigma=12.0, nsigma=5.0, grow=2.0):
    """Soft mask covering stars (not the large-scale Milky Way / foreground).

    A high-pass isolates small bright blobs (stars + their halos) from the smooth
    background; a noise-robust `mean + nsigma*std` threshold keeps the stars, the
    binary mask is then dilated by `grow` px -- so it also covers the pixels the
    erosion will pull in -- and slightly blurred for soft edges.  A percentile
    ramp instead of a hard threshold leaves typical stars almost unmasked (only
    the 99.9th percentile reaches ~1), which makes the reduction invisible."""
    hp = np.clip(lum - cv2.GaussianBlur(lum, (0, 0), sigma), 0, None)
    thr = float(hp.mean()) + nsigma * float(hp.std())
    mask = (hp > thr).astype(np.float32)
    if grow > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (2 * int(grow) + 1, 2 * int(grow) + 1))
        mask = cv2.GaussianBlur(cv2.dilate(mask, k), (0, 0), 1.5)
    return np.clip(mask, 0, 1)


def reduce_stars(rgb, amount=2.0, sigma=12.0):
    """Shrink stars, PixInsight StarReduction style.

    `amount` is the erosion radius in pixels: a grey erosion of the luminance
    pulls every bright edge inward, and the darkening ratio is applied to the
    RGB (so hue is preserved) only inside the star mask -- background and the
    smooth Milky Way are left untouched.  Returns (rgb, star_mask)."""
    r = int(round(amount))
    lum = rgb.mean(axis=2).astype(np.float32)
    if r < 1:
        return rgb, np.zeros(lum.shape, np.float32)
    mask = make_star_mask(lum, sigma=sigma)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    lum_e = cv2.erode(lum, k)
    scale = np.where(lum > 1e-6, lum_e / np.maximum(lum, 1e-6), 1.0).astype(np.float32)
    scale = 1.0 + mask * (scale - 1.0)
    out = np.clip(rgb * scale[..., None], 0, 1)
    return out, mask
