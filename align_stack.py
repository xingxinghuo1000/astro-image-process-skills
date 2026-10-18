#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Star-field alignment + stacking for a sequence of Canon CR2 frames shot on a
fixed tripod (sky rotates while the foreground stays static).

Method
------
1. Decode each CR2 once to linear 16-bit RGB (rawpy, camera WB, no auto-bright).
2. Build a high-pass grayscale image, run ORB feature matching against a
   reference frame and fit a similarity transform (rotation + translation,
   scale fixed ~1) with RANSAC.  RANSAC locks onto the *stars* (which move
   coherently) and rejects the sensor-noise / static foreground (which do not).
3. Warp the full-resolution linear RGB of every frame into the reference
   frame's coordinate system and accumulate a weighted mean (sky-aligned stack).

Outputs (in --out):
    transforms.json / transforms.txt   per-frame similarity transforms
    aligned/IMG_xxxx.tif               sky-aligned linear 16-bit frames (opt)
    stack_mean.tif                     linear 16-bit stacked image
    stack_preview.jpg                  arcsinh-stretched preview of the stack

Usage:
    python align_stack.py --dir "D:\\pic\\DCIM-2026\\08-shangduhu-2" \
                          --start 5994 --end 6012 \
                          --out out --ref 6003
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

    stack16 = np.clip(stack * 65535.0, 0, 65535).astype(np.uint16)
    stack_path = os.path.join(args.out, f"stack_{args.method}.tif")
    write_tiff16(stack_path, stack16)
    log(f"  wrote {stack_path}")

    prev = stretch_preview(stack)
    prev = cv2.resize(prev, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(args.out, "stack_preview.jpg"),
                cv2.cvtColor(prev, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])

    # adaptive brightness stretch -> extra file, so the Milky Way is visible
    ys, mtf_m, bg = auto_stretch(stack, target_bg=args.stretch_bg)
    st_path = os.path.join(args.out, f"stack_{args.method}_stretched.tif")
    write_tiff16(st_path, (ys * 65535.0).astype(np.uint16))
    ys_prev = cv2.resize((ys * 255).astype(np.uint8), None, fx=0.5, fy=0.5,
                         interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(args.out, "stretched_preview.jpg"),
                cv2.cvtColor(ys_prev, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    log(f"  wrote {st_path}  (auto-stretch: mtf_m={mtf_m:.4f}, bg={bg:.4f}, "
        f"target={args.stretch_bg})")

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
