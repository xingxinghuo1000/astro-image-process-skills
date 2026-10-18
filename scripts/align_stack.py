#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Star-field alignment + stacking for a sequence of Canon CR2 frames shot on a
fixed tripod (sky rotates while the foreground stays static).

Method
------
1. Decode grayscale previews of every frame and build a per-pixel static-pattern
   map (high percentile): this isolates hot pixels / fixed-pattern noise, which
   is subtracted so feature matching can only latch onto real, moving stars.
2. High-pass the cleaned gray, run ORB feature matching against a reference
   frame and fit a similarity transform (rotation + translation, scale ~1) with
   RANSAC, then refine on star centroids to sub-pixel accuracy.
3. Warp the full-resolution linear RGB of every frame into the reference frame
   and accumulate a weighted mean or a median (median removes hot-pixel trails).
4. Write an adaptive brightness-stretched copy (auto MTF) so faint structure
   such as the Milky Way is visible.

4b. Optional post-processing, PixInsight style, applied in stages (each enabled
    step adds a stage; the stretch is always applied):
    stacked      linear stacked image
    flat         auto flat -- fit and subtract the smooth per-channel background
                 gradient (light pollution / vignetting / airglow), PixInsight
                 AutomaticBackgroundExtractor style; runs on the LINEAR image,
                 where gradients are additive, i.e. before the stretch, via
                 --flat
    stretched    adaptive stretch (auto MTF)
    neutralized  equalize the per-channel sky background to a neutral grey
                 (PixInsight BackgroundNeutralization), via --neutralize
    star_reduced erode the stars under a high-pass star mask so they shrink
                 without touching the background / Milky Way, via
                 --star-reduce PX (default 1; 0 disables)

Outputs (in --out) -- every stage is written, plus a comparison montage:
    transforms.json                       per-frame similarity transforms
    stage<i>_<name>.tif                   every stage, 16-bit linear TIFF
    stage<i>_<name>_preview.jpg           half-size preview of every stage
    stages_montage.jpg                    all stages side by side, labelled
    star_mask.png                         star mask (with --star-reduce)
    flat_compare.jpg                      no-flat vs auto-flat, + strength sweep
                                          (with --flat; for the user to judge the
                                          flat and pick a strength)
    stack_<method>.tif                    linear 16-bit master (stage1)
    stack_<method>_stretched.tif/.jpg     finished image, full-res TIFF + JPEG
    stack_preview.jpg / stretched_preview.jpg
    aligned/IMG_xxxx.tif                  sky-aligned frames (with --save-frames)

Usage (Windows, using the repo's virtualenv):
    .venv/Scripts/python.exe scripts/align_stack.py \
        --dir "D:\\pic\\DCIM-2026\\08-shangduhu-2" \
        --start 5994 --end 6012 --out out --ref 6003 --method median
"""
import argparse, json, os, sys, time
import numpy as np
import cv2
import rawpy


def log(*a):
    print(*a, flush=True)


def find_files(d, start, end):
    names = []
    for i in range(start, end + 1):
        n = f"IMG_{i}.CR2"
        p = os.path.join(d, n)
        if os.path.exists(p):
            names.append((i, p))
        else:
            log(f"  (!) {n} not found, skipped")
    return names


def load_linear(path):
    """Decode CR2 -> float32 linear RGB in [0,1] (camera WB, no tone curve)."""
    raw = rawpy.imread(path)
    rgb16 = raw.postprocess(use_camera_wb=True, no_auto_bright=True,
                            output_bps=16, gamma=(1, 1),
                            output_color=rawpy.ColorSpace.sRGB,
                            demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
                            half_size=False)
    return rgb16.astype(np.float32) / 65535.0


def enhance_for_registration(gray_half, pre_blur=1.0, hp_sigma=20.0):
    """Suppress single-pixel hot pixels, then high-pass so real stars become
    clean local features for ORB (a pre-blur is essential: without it ORB
    matches static hot pixels and the fit collapses to identity)."""
    g = cv2.GaussianBlur(gray_half, (0, 0), pre_blur)
    bg = cv2.GaussianBlur(g, (0, 0), hp_sigma)
    e = np.clip(g - bg, 0, None)
    e = cv2.normalize(e, None, 0, 255, cv2.NORM_MINMAX)
    return e.astype(np.uint8)


def highpass_float(gray_half, pre_blur=1.0, hp_sigma=20.0):
    g = cv2.GaussianBlur(gray_half, (0, 0), pre_blur)
    return np.clip(g - cv2.GaussianBlur(g, (0, 0), hp_sigma), 0, None)


def detect_stars(hpf, nsigma=5.0, minarea=3, maxarea=400):
    """Connected-component star centroids.  minarea>=3 rejects isolated
    1-pixel hot pixels, which are the main source of false matches."""
    m, s = hpf.mean(), hpf.std()
    mask = (hpf > m + nsigma * s).astype(np.uint8)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    pts = [cent[i] for i in range(1, n)
           if minarea <= stats[i, cv2.CC_STAT_AREA] <= maxarea]
    return np.array(pts, np.float32)


def refine_transform(M, ref_stars, mov_stars, tols=(8.0, 4.0, 2.0, 1.2, 0.8)):
    """Iterative nearest-neighbour star matching + RANSAC similarity fit.
    Turns the coarse ORB estimate into a sub-pixel transform."""
    M = M.copy()
    if len(ref_stars) < 8 or len(mov_stars) < 8:
        return M
    for tol in tols:
        m2 = cv2.transform(mov_stars.reshape(-1, 1, 2), M).reshape(-1, 2)
        d2 = ((m2[:, None, :] - ref_stars[None, :, :]) ** 2).sum(-1)
        j = d2.argmin(1)
        dist = np.sqrt(d2[np.arange(len(m2)), j])
        keep = dist < tol
        if keep.sum() < 10:
            break
        src = mov_stars[keep].reshape(-1, 1, 2)
        dst = ref_stars[j[keep]].reshape(-1, 1, 2)
        Mn, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                            ransacReprojThreshold=tol,
                                            maxIters=20000, confidence=0.999)
        if Mn is not None:
            M = Mn
    return M


def star_score(ref_enh, mov_enh, M, frac=0.55):
    """Normalized correlation of the high-pass fields over the sky region.
    >0.6 means the transform really locks the stars; ~0.1..0.3 means failure."""
    H, W = ref_enh.shape
    w = cv2.warpAffine(mov_enh.astype(np.float32), M, (W, H),
                       flags=cv2.INTER_LINEAR)
    reg = slice(0, int(H * frac))
    a = ref_enh[reg].astype(np.float32).ravel()
    b = w[reg].ravel()
    a -= a.mean(); b -= b.mean()
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def star_residual(ref_stars, mov_stars, M, tol=1.5):
    """Median distance from warped moving stars to the nearest reference star,
    and the fraction of them within tol.  Small median = a true star lock."""
    if len(ref_stars) < 4 or len(mov_stars) < 4:
        return 1e9, 0.0
    m2 = cv2.transform(mov_stars.reshape(-1, 1, 2), M).reshape(-1, 2)
    d2 = ((m2[:, None, :] - ref_stars[None, :, :]) ** 2).sum(-1)
    d = np.sqrt(d2.min(1))
    return float(np.median(d)), float((d < tol).mean())


def transform_is_plausible(M, W, H, inliers, sc, resid_med, resid_frac):
    rot = np.degrees(np.arctan2(M[1, 0], M[0, 0]))
    tx, ty = M[0, 2], M[1, 2]
    return (abs(rot) < 15 and abs(tx) < 0.15 * W and abs(ty) < 0.15 * H
            and inliers >= 25 and sc >= 0.55 and resid_frac >= 0.20
            and resid_med < 3.0)


def make_detector():
    return cv2.ORB_create(nfeatures=12000, scaleFactor=1.2, nlevels=8,
                          fastThreshold=5, edgeThreshold=19)


def register(ref_desc, ref_kp, mob_desc, mob_kp):
    """Return 2x3 similarity M mapping moving coords -> reference coords."""
    if ref_desc is None or mob_desc is None:
        return None, 0, 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = bf.knnMatch(mob_desc, ref_desc, k=2)
    good = [m for m, n in (p for p in knn if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 8:
        return None, len(good), 0
    src = np.float32([mob_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([ref_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                         ransacReprojThreshold=3.0, maxIters=8000,
                                         confidence=0.999)
    ninl = int(inl.sum()) if inl is not None else 0
    return M, len(good), ninl


def scale_matrix(M, s):
    """M was fit at 1/s resolution; return equivalent matrix at full res."""
    t = M[:, 2] * s
    return np.column_stack([M[:, 0:2], t]).astype(np.float32)


def stretch_preview(rgb_lin, gain=50.0):
    x = np.clip(rgb_lin, 0, None)
    x = np.arcsinh(x * gain) / np.arcsinh(gain)
    return np.clip(x * 255, 0, 255).astype(np.uint8)


def write_tiff16(path, rgb_uint16):
    """Write a TIFF from an RGB (not BGR) array.  cv2.imwrite expects BGR, so
    convert first -- otherwise the red and blue channels end up swapped."""
    cv2.imwrite(path, cv2.cvtColor(rgb_uint16, cv2.COLOR_RGB2BGR))


def _mtf(m, x):
    """Midtones transfer function (PixInsight style); m=0.5 is identity."""
    return ((m - 1.0) * x) / ((2.0 * m - 1.0) * x - m)


def auto_stretch(rgb_lin, target_bg=0.20, black_pct=0.3, white_pct=99.95):
    """Adaptive stretch: robust per-channel black point, normalize by a high
    percentile, then pick the midtone-transfer parameter so the sky background
    lands at `target_bg` -- lifts the faint Milky Way without blowing stars."""
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
    they all match the average reference (a pure grey).  Applied on linear data."""
    ref = background_reference(rgb, bg_pct)
    target = float(ref.mean())
    if target < 1e-6:
        return rgb, ref, np.ones(3)
    gains = target / np.maximum(ref, 1e-6)
    out = np.clip(rgb * gains[None, None, :], 0, 1)
    return out, ref, gains


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
    some gradient removal for a clean, ring-free background."""
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


def _label(img, text):
    cv2.putText(img, text, (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    return img


def _product_preview(lin, target_bg, neutralize, scale=0.5):
    """Stretched (and optionally neutralized) preview of a linear image."""
    ys, _, _ = auto_stretch(lin, target_bg=target_bg)
    if neutralize:
        ys, _, _ = background_neutralize(ys)
    p = (np.clip(ys, 0, 1) * 255).astype(np.uint8)
    return cv2.resize(p, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def write_flat_compare(stack_lin, flat_lin, target_bg, neutralize, out_dir,
                       sweep=(), applied=1.0, scale=0.5):
    """Side-by-side stretched previews -- no-flat vs auto-flat, plus an optional
    strength sweep -- so the user can judge whether the flat helps and pick a
    `--flat-strength`.  The flat is a judgement call (it removes the light-
    pollution gradient but can carve the glow of a frame-filling Milky Way), so
    the pipeline always writes this figure whenever --flat is used.  With no
    sweep the second panel shows the strength actually applied."""
    def blend(s):
        return np.clip(stack_lin + s * (flat_lin - stack_lin), 0, 1)
    panels = [("no flat", _product_preview(stack_lin, target_bg, neutralize, scale))]
    for s in sweep:
        panels.append(("flat s=%.2f" % s,
                       _product_preview(blend(s), target_bg, neutralize, scale)))
    if not sweep:
        panels.append(("flat s=%.1f" % applied,
                       _product_preview(blend(applied), target_bg, neutralize, scale)))
    row = np.hstack([_label(im, t) for t, im in panels])
    path = os.path.join(out_dir, "flat_compare.jpg")
    cv2.imwrite(path, cv2.cvtColor(row, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 90])
    return path


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
    smooth Milky Way are left untouched."""
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


def write_stage(out_dir, name, rgb, preview_scale=0.5, quality=92):
    """Write one pipeline stage as a 16-bit TIFF plus a half-size JPEG preview."""
    tif = os.path.join(out_dir, name + ".tif")
    write_tiff16(tif, np.clip(rgb * 65535.0, 0, 65535).astype(np.uint16))
    prev = cv2.resize((np.clip(rgb, 0, 1) * 255).astype(np.uint8), None,
                      fx=preview_scale, fy=preview_scale, interpolation=cv2.INTER_AREA)
    jpg = os.path.join(out_dir, name + "_preview.jpg")
    cv2.imwrite(jpg, cv2.cvtColor(prev, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, quality])
    log(f"    {name}.tif  ({prev.shape[1]}x{prev.shape[0]} preview)")


def write_montage(out_dir, stages, target_bg=0.20, cols=3, scale=0.22,
                  quality=90):
    """Tile every stage into one labelled figure so the whole progression can be
    compared at a glance.  Linear stages (stacked / flat) are shown stretched so
    they are visible next to the already-stretched stages."""
    panels = []
    for name, img in stages:
        disp = (auto_stretch(img, target_bg=target_bg)[0]
                if ("stacked" in name or "flat" in name) else img)
        p = cv2.resize((np.clip(disp, 0, 1) * 255).astype(np.uint8), None,
                       fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        panels.append(_label(p, name))
    while len(panels) % cols:
        panels.append(np.zeros_like(panels[0]))
    rows = [np.hstack(panels[i:i + cols]) for i in range(0, len(panels), cols)]
    grid = np.vstack(rows)
    path = os.path.join(out_dir, "stages_montage.jpg")
    cv2.imwrite(path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, quality])
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=r"D:\pic\DCIM-2026\08-shangduhu-2")
    ap.add_argument("--start", type=int, default=5994)
    ap.add_argument("--end", type=int, default=6012)
    ap.add_argument("--out", default="out")
    ap.add_argument("--ref", type=int, default=6003,
                    help="reference frame number (default: middle of sequence)")
    ap.add_argument("--save-frames", action="store_true",
                    help="also write each sky-aligned full-res frame (.tif)")
    ap.add_argument("--method", choices=["mean", "median"], default="mean")
    ap.add_argument("--exclude", default="", help="comma separated frame numbers to drop")
    ap.add_argument("--stretch-bg", type=float, default=0.20,
                    help="target sky background level of the stretched output "
                         "(higher = brighter; default 0.20)")
    ap.add_argument("--neutralize", action="store_true",
                    help="neutralize the sky background to a grey before "
                         "stretching (PixInsight BackgroundNeutralization style)")
    ap.add_argument("--star-reduce", type=float, default=1.0, metavar="PX",
                    help="shrink stars by PX pixels with a morphology + star-mask "
                         "step applied to the stretched image (PixInsight "
                         "StarReduction style); default 1 (gentle), 0 disables")
    ap.add_argument("--flat", action="store_true",
                    help="remove the smooth background gradient on the LINEAR "
                         "image before stretching (PixInsight "
                         "AutomaticBackgroundExtractor / 'auto flat')")
    ap.add_argument("--flat-mode", choices=["subtract", "divide"],
                    default="subtract",
                    help="subtract the fitted background (light-pollution / "
                         "airglow gradients, default) or divide by it "
                         "(vignetting-style flat)")
    ap.add_argument("--flat-bins", type=int, default=16,
                    help="grid columns for the background model (default 16)")
    ap.add_argument("--flat-degree", type=int, default=3,
                    help="2D polynomial degree of the background surface "
                         "(default 3)")
    ap.add_argument("--flat-pct", type=float, default=25.0,
                    help="per-tile background percentile (default 25; low keeps "
                         "the estimate robust to stars)")
    ap.add_argument("--flat-strength", type=float, default=1.0, metavar="S",
                    help="blend the flat correction, 0..1 (default 1.0). Lower it "
                         "if a full correction carves the middle of a "
                         "Milky-Way-filled field and leaves a bright border ring; "
                         "~0.3-0.5 removes the ring while keeping the gradient")
    ap.add_argument("--flat-sweep", default="", metavar="S1,S2,...",
                    help="also render these flat strengths (0..1) into "
                         "flat_compare.jpg so you can pick --flat-strength from a "
                         "single run, e.g. 0.4,0.5,0.6,0.7")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    frames_dir = os.path.join(args.out, "aligned")
    if args.save_frames:
        os.makedirs(frames_dir, exist_ok=True)

    excl = {int(x) for x in args.exclude.split(",") if x.strip()}
    files = [(i, p) for i, p in find_files(args.dir, args.start, args.end) if i not in excl]
    if not files:
        log("no input files"); return
    log(f"{len(files)} frames: {files[0][0]}..{files[-1][0]}")

    # pick reference
    ids = [i for i, _ in files]
    ref_id = args.ref if args.ref in ids else ids[len(ids) // 2]
    log(f"reference frame: IMG_{ref_id}")

    orb = make_detector()
    t0 = time.time()

    # ---- pass 1: build a static-pattern map (hot pixels / fixed pattern) ----
    # A pixel that is bright in *most* frames is a hot pixel (static), while a
    # real star is bright in only a frame or two because the sky moves.  A high
    # percentile across frames therefore isolates the static pattern, which we
    # subtract so ORB can only latch onto real moving stars.
    log("pass 1: decoding grayscale previews to remove static pattern ...")
    grays = []
    for idx, path in files:
        raw = rawpy.imread(path)
        rgb = raw.postprocess(half_size=True, use_camera_wb=True,
                              no_auto_bright=True, output_bps=8)
        grays.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32))
    gh, gw = grays[0].shape
    if len(grays) >= 5:
        static = np.percentile(np.stack(grays), 75, axis=0)
    else:
        static = np.zeros((gh, gw), np.float32)
    cleaned = [np.clip(g - static, 0, None) for g in grays]
    del grays
    log(f"  static-pattern map built (half-res {gh}x{gw}, {len(files)} frames)")

    reg_scale = 0.5
    H, W = gh * 2, gw * 2
    k_ref = [i for i, (idx, _) in enumerate(files) if idx == ref_id][0]

    ref_clean = cleaned[k_ref]
    ref_enh = enhance_for_registration(ref_clean)
    ref_hpf = highpass_float(ref_clean)
    ref_kp, ref_desc = orb.detectAndCompute(ref_enh, None)
    ref_stars = detect_stars(ref_hpf)
    log(f"  reference ORB features: {len(ref_kp)}  stars: {len(ref_stars)}")

    acc_sum = None
    acc_w = np.zeros((H, W), np.float32)
    ones = np.ones((H, W), np.float32)
    records = []

    # median needs all aligned frames kept (memory heavy) -> use memmap on disk
    median_store = None
    if args.method == "median":
        mm_path = os.path.join(args.out, "_aligned_uint16.dat")
        median_store = np.memmap(mm_path, dtype=np.uint16, mode="w+",
                                 shape=(len(files), H, W, 3))

    log("pass 2: registering + stacking ...")
    for k, (idx, path) in enumerate(files):
        ts = time.time()
        clean = cleaned[k]
        if idx == ref_id:
            M_half = np.array([[1, 0, 0], [0, 1, 0]], np.float32)
            M_full = M_half.copy()
            enh = ref_enh
            minl = gd = 0
        else:
            hpf = highpass_float(clean)
            enh = enhance_for_registration(clean)
            kp, desc = orb.detectAndCompute(enh, None)
            M_half, gd, minl = register(ref_desc, ref_kp, desc, kp)
            if M_half is None:
                log(f"  IMG_{idx}: FAILED to register (good={gd}, inliers={minl}); skipped")
                continue
            mov_stars = detect_stars(hpf)
            M_half = refine_transform(M_half, ref_stars, mov_stars)
            M_full = scale_matrix(M_half, 1.0 / reg_scale)

        sc = star_score(ref_enh, enh, M_half)
        resid_med, resid_frac = (0.0, 1.0) if idx == ref_id else \
            star_residual(ref_stars, detect_stars(highpass_float(clean)),
                          M_half)
        if idx != ref_id and not transform_is_plausible(M_half, gw, gh, minl,
                                                        sc, resid_med, resid_frac):
            log(f"  !! IMG_{idx}: registration rejected "
                f"(score={sc:.2f} inl={minl} resid={resid_med:.2f}px "
                f"frac={resid_frac:.2f}); frame excluded from stack")
            continue

        lin = load_linear(path)
        if acc_sum is None:
            acc_sum = np.zeros_like(lin)

        if args.method == "mean":
            warped = cv2.warpAffine(lin, M_full, (W, H), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            w = cv2.warpAffine(ones, M_full, (W, H), flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            acc_sum += warped * w[..., None]
            acc_w += w
        else:
            if idx == ref_id:
                warped = lin
            else:
                warped = cv2.warpAffine(lin, M_full, (W, H), flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            median_store[k] = np.clip(warped * 65535.0, 0, 65535).astype(np.uint16)

        if args.save_frames:
            p = os.path.join(frames_dir, f"IMG_{idx}.tif")
            write_tiff16(p, np.clip(warped * 65535.0, 0, 65535).astype(np.uint16))

        # report rotation/translation in degrees / px
        rot = np.degrees(np.arctan2(M_full[1, 0], M_full[0, 0]))
        tx, ty = M_full[0, 2], M_full[1, 2]
        log(f"  IMG_{idx}: good={gd} inliers={minl} score={sc:.3f} rot={rot:+.3f}deg "
            f"shift=({tx:+.2f},{ty:+.2f})px  [{time.time()-ts:.1f}s]")
        records.append(dict(frame=idx, good=int(gd), inliers=int(minl),
                            score=float(sc), rot_deg=float(rot),
                            tx=float(tx), ty=float(ty),
                            M=[[float(v) for v in row] for row in M_full]))

        del lin

    # ---- stack ----
    log("stacking ...")
    if args.method == "mean":
        stack = acc_sum / np.maximum(acc_w[..., None], 1)
    else:
        log(f"  computing median over {len(files)} frames (block-wise) ...")
        stack = np.empty((H, W, 3), np.float32)
        step = 128
        for r0 in range(0, H, step):
            r1 = min(H, r0 + step)
            block = median_store[:, r0:r1, :, :].astype(np.float32) / 65535.0
            stack[r0:r1] = np.median(block, axis=0)

    # ---- staged post-processing -------------------------------------------------
    # Each enabled step appends a stage; the number reflects the real order so
    # the flattened output of the linear master is always stage1's successor.
    stages = []

    def add_stage(name, img):
        stages.append((f"stage{len(stages) + 1}_{name}", img))

    # stage: the linear stacked image
    linear = stack
    add_stage("stacked", linear)

    # stage: auto flat (ABE) -- fit and subtract the smooth per-channel
    # background gradient on the LINEAR image, where gradients are additive,
    # i.e. before the nonlinear stretch.  `flat_full` is the full-strength
    # correction; the applied result is a strength blend of it, and a
    # no-flat/flat comparison is always written so the user can judge it.
    if args.flat:
        flat_full, finfo = auto_flat(linear, bins=args.flat_bins,
                                     degree=args.flat_degree, pct=args.flat_pct,
                                     mode=args.flat_mode)
        linear = np.clip(linear + args.flat_strength * (flat_full - linear), 0, 1)
        log("  auto-flat (ABE, %s, strength=%.2f): model level " %
            (args.flat_mode, args.flat_strength)
            + ", ".join("ch%d=%.4f..%.4f" % (c, mn, mx)
                        for c, (mn, mx, _) in enumerate(finfo)))
        add_stage("flat", linear)
        sweep = [float(x) for x in args.flat_sweep.split(",") if x.strip()]
        cmp_path = write_flat_compare(stack, flat_full, args.stretch_bg,
                                     args.neutralize, args.out, sweep,
                                     applied=args.flat_strength)
        log(f"  wrote {cmp_path}  (no flat vs auto flat"
            + ("" if not sweep else " vs strength sweep " + args.flat_sweep) + ")")

    # stage: adaptive brightness stretch (auto MTF), so faint structure shows
    ys, mtf_m, bg = auto_stretch(linear, target_bg=args.stretch_bg)
    add_stage("stretched", ys)

    # stage: background neutralization -- scale each channel so the sky
    # background becomes a neutral grey (PixInsight BackgroundNeutralization).
    if args.neutralize:
        ys, sref, sgain = background_neutralize(ys)
        log(f"  neutralized background: ref={np.round(sref, 4).tolist()} "
            f"gains={np.round(sgain, 4).tolist()}")
        add_stage("neutralized", ys)

    # stage: star reduction -- shrink stars under a high-pass star mask.
    if args.star_reduce > 0:
        ys, star_mask = reduce_stars(ys, amount=args.star_reduce)
        m8 = (np.clip(star_mask, 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(args.out, "star_mask.png"),
                    cv2.cvtColor(m8, cv2.COLOR_GRAY2BGR))
        log(f"  stars reduced by {args.star_reduce:g}px "
            f"(mask {100.0 * (star_mask > 0.05).mean():.2f}% of frame)")
        add_stage("star_reduced", ys)

    # ---- every stage as its own TIFF + preview, plus a comparison montage ----
    log("  writing per-stage images ...")
    for name, img in stages:
        write_stage(args.out, name, img)
    montage = write_montage(args.out, stages, target_bg=args.stretch_bg)
    log(f"  wrote {montage}  (all stages side by side)")

    # ---- final deliverables (linear master + finished image, TIFF and JPEG) ----
    stack_path = os.path.join(args.out, f"stack_{args.method}.tif")
    write_tiff16(stack_path, np.clip(linear * 65535.0, 0, 65535).astype(np.uint16))
    log(f"  wrote {stack_path}")

    prev = stretch_preview(linear)
    prev = cv2.resize(prev, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(args.out, "stack_preview.jpg"),
                cv2.cvtColor(prev, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])

    st_path = os.path.join(args.out, f"stack_{args.method}_stretched.tif")
    write_tiff16(st_path, (ys * 65535.0).astype(np.uint16))
    st_jpg = os.path.join(args.out, f"stack_{args.method}_stretched.jpg")
    cv2.imwrite(st_jpg,
                cv2.cvtColor((np.clip(ys, 0, 1) * 255 + 0.5).astype(np.uint8),
                             cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 95])
    ys_prev = cv2.resize((ys * 255).astype(np.uint8), None, fx=0.5, fy=0.5,
                         interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(args.out, "stretched_preview.jpg"),
                cv2.cvtColor(ys_prev, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    log(f"  wrote {st_path} + stack_{args.method}_stretched.jpg (full-res)"
        f"  (auto-stretch: mtf_m={mtf_m:.4f}, bg={bg:.4f}, target={args.stretch_bg})")

    with open(os.path.join(args.out, "transforms.json"), "w") as f:
        json.dump(dict(reference=ref_id, reg_scale=reg_scale, frames=records), f, indent=2)

    if median_store is not None:
        del median_store
        try:
            os.remove(mm_path)
        except OSError:
            pass

    log(f"done in {time.time()-t0:.1f}s -> {os.path.abspath(args.out)}")
    log(f"  {len(records)} frames stacked ({args.method})")


if __name__ == "__main__":
    main()
