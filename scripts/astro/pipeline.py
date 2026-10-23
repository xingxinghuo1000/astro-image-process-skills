"""Pipeline orchestration: wire the individual steps together.

    decode previews -> (static-pattern map) -> register -> stack
        -> (optional) sky/ground composite -> (optional) flat
        -> stretch -> (optional) neutralize -> (optional) white balance
        -> (optional) star reduction
        -> per-stage TIFFs + montage -> final TIFF + JPEG

The step implementations live in the sibling modules; this file only sequences
them and writes the deliverables.
"""
import os
import json
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2

from .util import log, auto_workers, cpu_count, available_memory_gb
from . import imgio
from . import registration as reg
from . import stacking
from . import flat as flatmod
from . import stretch
from . import neutralize
from . import whitebalance
from . import star_reduce
from . import composite as composite_step
from . import horizon
from . import render
from . import hotpixels


def _register_task(task):
    """Per-frame registration work, safe to run in a thread pool."""
    (k, idx, clean, ref_id, ref_enh, ref_kp, ref_desc, ref_stars, reg_mask,
     gw, gh, reg_scale) = task
    eye = np.eye(3, dtype=np.float32)
    if idx == ref_id:
        rot, tx, ty, scale, _ = reg.homography_metrics(eye, gw * 2, gh * 2)
        return dict(k=k, idx=idx, ok=True, reason="", M_half=eye, M_full=eye,
                    gd=0, inl=0, sc=1.0, resid_med=0.0, resid_frac=1.0,
                    rot=rot, tx=tx, ty=ty, scale=scale)
    hpf = reg.highpass_float(clean)
    enh = reg.enhance_for_registration(clean)
    orb = reg.make_detector()
    kp, desc = orb.detectAndCompute(enh, reg_mask)
    M_half, gd, minl = reg.register(ref_desc, ref_kp, desc, kp)
    if M_half is None:
        return dict(k=k, idx=idx, ok=False,
                    reason=f"FAILED to register (good={gd}, inliers={minl})",
                    gd=gd, inl=minl)
    mov_stars = reg.detect_stars(hpf, valid=reg_mask)
    M_half = reg.refine_transform(M_half, ref_stars, mov_stars)
    M_full = reg.scale_matrix(M_half, 1.0 / reg_scale)
    sc = reg.star_score(ref_enh, enh, M_half)
    resid_med, resid_frac = reg.star_residual(ref_stars, mov_stars, M_half)
    ok = reg.transform_is_plausible(M_half, gw, gh, minl, sc,
                                    resid_med, resid_frac)
    rot, tx, ty, scale, _ = reg.homography_metrics(M_full, gw * 2, gh * 2)
    return dict(k=k, idx=idx, ok=ok,
                reason=(f"registration rejected (score={sc:.2f} inl={minl} "
                        f"resid={resid_med:.2f}px frac={resid_frac:.2f})"
                        if not ok else ""),
                M_half=M_half, M_full=M_full, gd=gd, inl=minl, sc=sc,
                resid_med=resid_med, resid_frac=resid_frac,
                rot=rot, tx=tx, ty=ty, scale=scale)


def _stack_ground_from_dir(args):
    gstart = args.ground_start if args.ground_start is not None else args.start
    gend = args.ground_end if args.ground_end is not None else args.end
    gfiles = imgio.find_files(args.ground_dir, gstart, gend)
    if not gfiles:
        return None
    log(f"  ground layer: {len(gfiles)} frames from {args.ground_dir} "
        f"({gstart}..{gend})")
    acc = None
    for idx, path in gfiles:
        lin = imgio.load_linear(path)
        acc = np.zeros_like(lin) if acc is None else acc
        acc += lin
    return acc / len(gfiles)


def run(args):
    os.makedirs(args.out, exist_ok=True)
    frames_dir = os.path.join(args.out, "aligned")
    if args.save_frames:
        os.makedirs(frames_dir, exist_ok=True)

    if args.method == "mean":
        log("WARNING: --method mean is discouraged for astro sequences. A hot "
            "pixel is fixed on the sensor while the frame rotates, so mean "
            "keeps it as a colored streak (hot/N) that no threshold can catch. "
            "Use --method median (default).")

    excl = {int(x) for x in args.exclude.split(",") if x.strip()}
    files = [(i, p) for i, p in imgio.find_files(args.dir, args.start, args.end)
             if i not in excl]
    if not files:
        log("no input files")
        return
    log(f"{len(files)} frames: {files[0][0]}..{files[-1][0]}")

    ids = [i for i, _ in files]
    ref_id = args.ref if args.ref in ids else ids[len(ids) // 2]
    log(f"reference frame: IMG_{ref_id}")

    t0 = time.perf_counter()
    timings = {}

    def mark(name, started):
        timings[name] = round(time.perf_counter() - started, 3)

    cpus = cpu_count()
    avail_mem = available_memory_gb()
    workers = args.workers if args.workers > 0 else auto_workers(
        "registration", len(files))
    io_workers = args.io_workers if args.io_workers > 0 else auto_workers(
        "io", len(files))
    stack_workers = args.stack_workers if args.stack_workers > 0 else auto_workers(
        "stack", len(files))
    stack_band_mb = args.stack_band_mb if args.stack_band_mb > 0 else 16
    render_workers = args.render_workers if args.render_workers > 0 else auto_workers(
        "render", 16)
    workers = max(1, min(workers, len(files)))
    io_workers = max(1, min(io_workers, len(files)))
    mem_txt = f"{avail_mem:.1f} GB" if avail_mem is not None else "unknown"
    log(f"workers: registration={workers}, io={io_workers}, "
        f"stack={stack_workers}@{stack_band_mb}MB, render={render_workers} "
        f"(cpu={cpus}, available RAM={mem_txt})")

    ground_mask_img = None
    if args.ground_mask:
        ground_mask_img = cv2.imread(args.ground_mask, cv2.IMREAD_GRAYSCALE)
        if ground_mask_img is None:
            log(f"  !! cannot read --ground-mask {args.ground_mask}")

    # ---- pass 1: build a static-pattern map (hot pixels / fixed pattern) ----
    # On a fixed tripod a pixel that is bright in most frames is a hot pixel or
    # fixed pattern; a real star is bright in only a frame or two because the
    # sky moves.  A high percentile across frames isolates that static pattern.
    # On a *sky-tracked* sequence the stars themselves are static, so the same
    # percentile map contains the whole star field and subtracting it erases
    # the very features ORB needs.  Detect the tracked case first.
    log(f"pass 1: decoding grayscale previews ({io_workers} workers) ...")
    t = time.perf_counter()
    if io_workers > 1:
        with ThreadPoolExecutor(max_workers=io_workers) as ex:
            grays = list(ex.map(imgio.load_gray8, [p for _, p in files]))
    else:
        grays = [imgio.load_gray8(path) for _, path in files]
    mark("decode_previews", t)
    gh, gw = grays[0].shape
    t = time.perf_counter()

    reg_mask = None
    if ground_mask_img is not None:
        sky_half = cv2.resize((ground_mask_img < 127).astype(np.uint8),
                              (gw, gh), interpolation=cv2.INTER_NEAREST)
        reg_mask = cv2.erode(sky_half, np.ones((5, 5), np.uint8))
        log(f"  registration restricted to sky ({100.0 * reg_mask.mean():.1f}% "
            f"of the half-res frame)")

    k_ref = [i for i, (idx, _) in enumerate(files) if idx == ref_id][0]
    tracked = False
    if len(grays) >= 5:
        a = reg.highpass_float(grays[k_ref])
        sl = slice(0, int(gh * 0.55))
        probes = sorted({len(grays) // 4, len(grays) // 2,
                         3 * len(grays) // 4, len(grays) - 1})
        corrs = []
        for k in probes:
            if k == k_ref:
                continue
            b = reg.highpass_float(grays[k])
            if reg_mask is not None:
                m = reg_mask[sl] > 0
                aa, bb = a[sl][m], b[sl][m]
            else:
                aa, bb = a[sl].ravel(), b[sl].ravel()
            aa = aa - aa.mean()
            bb = bb - bb.mean()
            corrs.append(float((aa * bb).sum() /
                               (np.linalg.norm(aa) * np.linalg.norm(bb) + 1e-9)))
        corr = float(np.median(corrs)) if corrs else 0.0
        if corr > 0.65:
            tracked = True
            log(f"  sky is nearly static around the reference frame "
                f"(high-pass corr={corr:.2f}): treating as tracked; "
                f"static-pattern subtraction disabled")
    if not args.no_static and not tracked:
        static = reg.build_static_map(grays)
        cleaned = [np.clip(g - static, 0, None) for g in grays]
        log(f"  static-pattern map built (half-res {gh}x{gw}, {len(files)} frames)")
    else:
        cleaned = grays
        log("  static-pattern subtraction disabled")
    del grays
    if tracked and not args.hot_post:
        args.hot_post = True
        log("  tracked sequence: enabling --hot-post (fixed sensor hot pixels "
            "are not rejected by a median of static stars)")
    mark("tracked_and_static", t)

    reg_scale = 0.5
    H, W = gh * 2, gw * 2

    t = time.perf_counter()
    orb = reg.make_detector()
    ref_clean = cleaned[k_ref]
    ref_enh = reg.enhance_for_registration(ref_clean)
    ref_kp, ref_desc = orb.detectAndCompute(ref_enh, reg_mask)
    ref_stars = reg.detect_stars(reg.highpass_float(ref_clean), valid=reg_mask)
    log(f"  reference ORB features: {len(ref_kp)}  stars: {len(ref_stars)}")
    mark("reference", t)
    if not ref_kp:
        log("  !! reference frame has no ORB features; aborting")
        return

    ground_bbox = None
    if args.composite and not args.ground_dir:
        if ground_mask_img is not None:
            ys, xs = np.where(ground_mask_img > 127)
            if len(ys):
                gy0, gy1 = int(ys.min()), int(ys.max()) + 1
                gx0, gx1 = int(xs.min()), int(xs.max()) + 1
            else:
                gy0, gy1, gx0, gx1 = int(H * 0.65), H, 0, W
        else:
            gy0, gy1, gx0, gx1 = int(H * 0.65), H, 0, W
        gy0, gy1 = max(0, gy0), min(H, gy1)
        gx0, gx1 = max(0, gx0), min(W, gx1)
        if gy1 - gy0 >= 32 and gx1 - gx0 >= 32:
            ground_bbox = (gy0, gy1, gx0, gx1)

    ground_shift = None
    ground_resp = None
    ground_static = True
    if args.composite and tracked and not args.ground_dir:
        if ground_bbox is not None:
            gy0, gy1, gx0, gx1 = ground_bbox
            hy0, hy1 = max(0, gy0 // 2), min(gh, (gy1 + 1) // 2)
            hx0, hx1 = max(0, gx0 // 2), min(gw, (gx1 + 1) // 2)
            if hy1 - hy0 >= 16 and hx1 - hx0 >= 16:
                ground_shift = np.zeros((len(files), 2), np.float32)
                ground_resp = np.zeros(len(files), np.float32)
                ref_ground = cleaned[k_ref][hy0:hy1, hx0:hx1].astype(np.float32)
                win = cv2.createHanningWindow((hx1 - hx0, hy1 - hy0), cv2.CV_32F)
                for k in range(len(files)):
                    if k == k_ref:
                        ground_resp[k] = 1.0
                        continue
                    mov = cleaned[k][hy0:hy1, hx0:hx1].astype(np.float32)
                    try:
                        (dx, dy), resp = cv2.phaseCorrelate(mov, ref_ground, win)
                    except cv2.error:
                        continue
                    if not np.isfinite([dx, dy, resp]).all() or resp < 0.005:
                        dx = dy = 0.0
                    ground_shift[k] = (dx, dy)
                    ground_resp[k] = resp
                norms = np.linalg.norm(ground_shift, axis=1)
                max_shift = float(np.max(norms))
                p95_shift = float(np.percentile(norms, 95))
                ground_static = p95_shift < 1.5
                log(f"  ground motion: max {max_shift:.2f} px, p95 {p95_shift:.2f} px "
                    f"at half-res ({'static' if ground_static else 'translation-aligned'})")
            else:
                log("  ground motion: ground region too small; averaging unaligned frames")
        else:
            log("  ground motion: no usable ground region; averaging unaligned frames")

    ground_store = None
    ground_path = None
    ground_kept = 0
    if (args.composite and args.ground_method == "median" and not args.ground_dir
            and ground_bbox is not None):
        gy0, gy1, gx0, gx1 = ground_bbox
        ground_path = os.path.join(args.out, "_ground_uint16.dat")
        ground_store = stacking.make_median_store(
            ground_path, len(files), gy1 - gy0, gx1 - gx0)
        log(f"  ground median store: {gx1-gx0}x{gy1-gy0} px, {len(files)} frames")

    acc_sum = None
    acc_w = np.zeros((H, W), np.float32)
    ones = np.ones((H, W), np.float32)
    ground_acc = (np.zeros((H, W, 3), np.float32)
                  if args.composite and ground_store is None else None)
    ground_n = 0
    # Temporal-motion accumulator (half-res) for the sky/ground boundary: the
    # raw frames vary over the sky (the stars move) and are static over the
    # ground, so their variance is a clean sky cue even near the celestial pole.
    var_sum = var_sq = None
    var_n = 0
    records = []
    hot_pre_count = 0
    hot_pre_before = None
    hot_pre_after = None

    # median needs all aligned frames kept (memory heavy) -> use memmap on disk
    median_store = None
    mm_path = None
    kept = 0
    if args.method == "median":
        mm_path = os.path.join(args.out, "_aligned_uint16.dat")
        median_store = stacking.make_median_store(mm_path, len(files), H, W)

    log(f"pass 2: registering {len(files)} frames with {workers} worker threads ...")
    t = time.perf_counter()
    tasks = [(k, idx, cleaned[k], ref_id, ref_enh, ref_kp, ref_desc,
              ref_stars, reg_mask, gw, gh, reg_scale)
             for k, (idx, _) in enumerate(files)]
    prev_threads = cv2.getNumThreads()
    if workers > 1:
        cv2.setNumThreads(1)
    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            reg_results = list(ex.map(_register_task, tasks))
    finally:
        cv2.setNumThreads(prev_threads)
    mark("registration", t)

    io_pool = (ThreadPoolExecutor(max_workers=io_workers)
               if io_workers > 1 else None)
    pending = {}
    next_submit = 0
    io_limit = max(1, io_workers * 2)

    def submit_frames():
        nonlocal next_submit
        if io_pool is None:
            return
        while next_submit < len(files) and len(pending) < io_limit:
            pending[next_submit] = io_pool.submit(
                imgio.load_linear, files[next_submit][1])
            next_submit += 1

    def take_frame(k):
        if io_pool is None:
            return imgio.load_linear(files[k][1])
        return pending.pop(k).result()

    submit_frames()
    t = time.perf_counter()

    for k, (idx, path) in enumerate(files):
        ts = time.time()
        clean = cleaned[k]
        r = reg_results[k]
        if not r["ok"]:
            if io_pool is not None:
                fut = pending.pop(k, None)
                if fut is not None:
                    fut.cancel()
                submit_frames()
            log(f"  !! IMG_{idx}: {r['reason']}; frame excluded from stack")
            continue
        M_half = r["M_half"]
        M_full = r["M_full"]
        gd, minl, sc = r["gd"], r["inl"], r["sc"]

        if args.composite:
            if var_sum is None:
                var_sum = np.zeros((gh, gw), np.float32)
                var_sq = np.zeros((gh, gw), np.float32)
            var_sum += clean
            var_sq += clean * clean
            var_n += 1

        lin = take_frame(k)
        submit_frames()
        if acc_sum is None:
            acc_sum = np.zeros_like(lin)

        if args.hot_pre:
            cap = hot_pre_before is None
            if cap:
                hot_pre_before = lin.copy()
            lin, nh = hotpixels.detect_and_fix(
                lin, args.hot_sigma, args.hot_ksize, args.hot_abs,
                args.hot_color, args.hot_neigh, args.hot_grow)
            hot_pre_count += nh
            if cap:
                hot_pre_after = lin.copy()

        if args.method == "mean":
            warped = stacking.warp_into(lin, M_full, W, H)
            w = stacking.valid_mask(M_full, W, H, ones)
            acc_sum += warped * w[..., None]
            acc_w += w
        else:
            warped = lin if idx == ref_id else stacking.warp_into(lin, M_full, W, H)
            median_store[kept] = np.clip(warped * 65535.0, 0, 65535).astype(np.uint16)
            kept += 1

        if ground_acc is not None or ground_store is not None:
            glin = lin
            if tracked and not ground_static:
                dx, dy = ground_shift[k]
                if ground_resp[k] >= 0.005:
                    gm = np.float32([[1, 0, dx * 2], [0, 1, dy * 2]])
                    glin = cv2.warpAffine(
                        lin, gm, (W, H), flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            if ground_store is not None:
                gy0, gy1, gx0, gx1 = ground_bbox
                ground_store[ground_kept] = np.clip(
                    glin[gy0:gy1, gx0:gx1] * 65535.0, 0, 65535).astype(np.uint16)
                ground_kept += 1
            else:
                ground_acc += glin
                ground_n += 1

        if args.save_frames:
            imgio.write_tiff16(os.path.join(frames_dir, f"IMG_{idx}.tif"),
                               np.clip(warped * 65535.0, 0, 65535).astype(np.uint16))

        rot, tx, ty, scale = r["rot"], r["tx"], r["ty"], r["scale"]
        log(f"  IMG_{idx}: good={gd} inliers={minl} score={sc:.3f} rot={rot:+.3f}deg "
            f"shift=({tx:+.2f},{ty:+.2f})px scale={scale:.5f}  "
            f"[{time.time()-ts:.1f}s]")
        records.append(dict(frame=idx, good=int(gd), inliers=int(minl),
                            score=float(sc), rot_deg=float(rot),
                            tx=float(tx), ty=float(ty),
                            M=[[float(v) for v in row] for row in M_full]))
        del lin

    if io_pool is not None:
        io_pool.shutdown(wait=True)
    mark("frame_loop", t)

    # ---- stack ----
    log("stacking ...")
    t = time.perf_counter()
    if args.method == "mean":
        stack = stacking.mean_stack(acc_sum, acc_w)
    else:
        if kept == 0:
            log("  no frames registered; nothing to stack")
            return
        log(f"  computing median over {kept} accepted frames "
            f"({stack_workers} row-band workers) ...")
        stack = stacking.median_stack(median_store[:kept], H, W,
                                      workers=stack_workers,
                                      band_mb=stack_band_mb)
    mark("stacking", t)

    # ---- staged post-processing ----
    stages = []

    def add_stage(name, img, linear=False):
        stages.append((f"stage{len(stages) + 1}_{name}", img, linear))

    add_stage("stacked", stack, linear=True)
    master = stack

    # ---- optional hot-pixel removal on the stacked image (single frame) ---------
    if args.hot_post:
        t = time.perf_counter()
        pre_hot = master.copy()
        master, nh = hotpixels.detect_and_fix(
            master, args.hot_sigma, args.hot_ksize, args.hot_abs,
            args.hot_color, args.hot_neigh, args.hot_grow)
        log(f"  post-stacking hot-pixel removal: {nh} channel-pixels replaced")
        add_stage("hotfix", master, linear=True)
        p, _ = hotpixels.write_compare(
            args.out, pre_hot, master, "post", args.hot_sigma, args.hot_ksize,
            args.hot_abs, args.hot_color, args.hot_neigh)
        log(f"  wrote {p}  (before vs after post-stacking hot-pixel removal)")
        mark("hotfix", t)

    # ---- optional sky/ground composite -----------------------------------------
    # Stitch the star-aligned sky onto a temporal ground layer, splitting them at
    # the detected sky-ground boundary.  For tracked sequences the ground is
    # translation-aligned first; for static tripod sequences every frame is
    # already aligned.  This replaces the linear master that later steps use.
    sky_mask = None
    t = time.perf_counter()
    if args.composite:
        if args.ground_dir:
            ground = _stack_ground_from_dir(args)
            if ground is None:
                log("  !! composite: no ground frames found; using the input frames")
                ground = stacking.mean_ground(ground_acc, ground_n)
        elif ground_store is not None:
            gy0, gy1, gx0, gx1 = ground_bbox
            log(f"  ground layer: median of {ground_kept} frames "
                f"({gx1-gx0}x{gy1-gy0} px, "
                f"{'static foreground' if ground_static else 'translation-aligned'})")
            ground = master.copy()
            gcrop = stacking.median_stack(
                ground_store[:ground_kept], gy1 - gy0, gx1 - gx0,
                workers=stack_workers, band_mb=stack_band_mb)
            if ground_mask_img is not None:
                gm = ground_mask_img[gy0:gy1, gx0:gx1] > 127
                dst = ground[gy0:gy1, gx0:gx1]
                dst[gm] = gcrop[gm]
            else:
                ground[gy0:gy1, gx0:gx1] = gcrop
            ground_n = ground_kept
        else:
            log(f"  ground layer: mean of {ground_n} frames "
                f"({'static foreground' if ground_static else 'translation-aligned'})")
            ground = stacking.mean_ground(ground_acc, ground_n)
        sigma = max(8.0, 0.01 * gw)
        evidence = horizon.sky_evidence(
            horizon.variance(var_sum, var_sq, var_n), sigma)
        if os.environ.get("ASTRO_DEBUG_VAR"):
            np.savez(os.path.join(args.out, "_dbg_var.npz"),
                     var_sum=var_sum, var_sq=var_sq, var_n=var_n)
        # A boundary recognised by a large image model can be supplied as a
        # ground mask (white = ground); it overrides the variance detection.
        sky_mask = None
        if args.ground_mask:
            gm = ground_mask_img
            if gm is None:
                log(f"  !! composite: cannot read --ground-mask "
                    f"{args.ground_mask}; falling back to auto detection")
            else:
                gm = cv2.resize(gm, (W, H), interpolation=cv2.INTER_LINEAR)
                sky = 1.0 - gm.astype(np.float32) / 255.0
                sky = cv2.GaussianBlur(sky, (0, 0),
                                       max(1.0, args.composite_feather))
                sky_mask = np.clip(sky, 0, 1).astype(np.float32)
                log(f"  ground mask (model): {args.ground_mask}  "
                    f"ground={100.0 * (gm > 127).mean():.1f}%")
        comp, info = composite_step.run_composite(
            master, ground, evidence, args.out, target_bg=args.stretch_bg,
            mode=args.composite_horizon, feather=args.composite_feather,
            thresh=args.composite_thresh, sky_mask=sky_mask)
        if comp is not None:
            add_stage("composite", comp, linear=True)
            master = comp
        linear = master
    else:
        linear = master
    mark("composite", t)

    # stage: auto flat (ABE) -- fit and subtract the smooth per-channel
    # background gradient on the LINEAR image, where gradients are additive,
    # i.e. before the nonlinear stretch.
    if args.flat:
        t_flat = time.perf_counter()
        flat_full, finfo = flatmod.auto_flat(linear, bins=args.flat_bins,
                                             degree=args.flat_degree,
                                             pct=args.flat_pct, mode=args.flat_mode)
        linear = np.clip(linear + args.flat_strength * (flat_full - linear), 0, 1)
        log("  auto-flat (ABE, %s, strength=%.2f): model level " %
            (args.flat_mode, args.flat_strength)
            + ", ".join("ch%d=%.4f..%.4f" % (c, mn, mx)
                        for c, (mn, mx, _) in enumerate(finfo)))
        add_stage("flat", linear, linear=True)
        mark("flat_model", t_flat)
        t_cmp = time.perf_counter()
        sweep = [float(x) for x in args.flat_sweep.split(",") if x.strip()]
        cmp_path = render.write_flat_compare(master, flat_full, args.stretch_bg,
                                             args.neutralize, args.out, sweep,
                                             applied=args.flat_strength)
        mark("flat_compare", t_cmp)
        log(f"  wrote {cmp_path}  (no flat vs auto flat"
            + ("" if not sweep else " vs strength sweep " + args.flat_sweep) + ")")

    # stage: adaptive brightness stretch (auto MTF), so faint structure shows
    t = time.perf_counter()
    ys, mtf_m, bg = stretch.auto_stretch(linear, target_bg=args.stretch_bg)
    add_stage("stretched", ys)
    mark("stretch", t)

    # stage: background neutralization -- scale each channel so the sky
    # background becomes a neutral grey (PixInsight BackgroundNeutralization).
    if args.neutralize:
        t = time.perf_counter()
        ys, sref, sgain = neutralize.background_neutralize(ys)
        log(f"  neutralized background: ref={np.round(sref, 4).tolist()} "
            f"gains={np.round(sgain, 4).tolist()}")
        add_stage("neutralized", ys)
        mark("neutralize", t)

    # stage: automatic white balance -- the neutralization equalizes the sky
    # background, this corrects the star/Milky-Way colour balance.  The
    # before/after figure is for visual review (and a manual --wb-gains retry).
    if args.wb:
        t = time.perf_counter()
        manual = None
        if args.wb_gains.strip():
            try:
                manual = np.array([float(v) for v in args.wb_gains.split(",")],
                                  np.float64)
                if manual.size != 3:
                    raise ValueError
            except ValueError:
                log(f"  !! invalid --wb-gains '{args.wb_gains}'; using auto")
                manual = None
        before = ys.copy()
        ys, wgains = whitebalance.auto_white_balance(
            ys, mode=args.wb_mode, strength=args.wb_strength, gains=manual,
            sky_mask=sky_mask)
        log(f"  white balance ({args.wb_mode}, strength={args.wb_strength:g}): "
            f"gains={np.round(wgains, 4).tolist()}")
        add_stage("whitebalanced", ys)
        p = render.write_wb_compare(args.out, before, ys, wgains)
        log(f"  wrote {p}  (before vs after white balance)")
        del before
        mark("white_balance", t)

    # stage: star reduction -- shrink stars under a high-pass star mask.
    if args.star_reduce > 0:
        t = time.perf_counter()
        ys, star_mask = star_reduce.reduce_stars(ys, amount=args.star_reduce)
        cv2.imwrite(os.path.join(args.out, "star_mask.png"),
                    cv2.cvtColor((np.clip(star_mask, 0, 1) * 255).astype(np.uint8),
                                 cv2.COLOR_GRAY2BGR))
        log(f"  stars reduced by {args.star_reduce:g}px "
            f"(mask {100.0 * (star_mask > 0.05).mean():.2f}% of frame)")
        add_stage("star_reduced", ys)
        mark("star_reduce", t)

    # ---- every stage as its own TIFF + preview, plus a comparison montage ----
    log("  writing per-stage images ...")
    t = time.perf_counter()
    render_workers = max(1, min(render_workers, len(stages)))
    if render_workers > 1:
        with ThreadPoolExecutor(max_workers=render_workers) as ex:
            list(ex.map(lambda item: render.write_stage(args.out, item[0],
                                                        item[1]), stages))
    else:
        for name, img, _ in stages:
            render.write_stage(args.out, name, img)
    mark("stage_writes", t)
    t = time.perf_counter()
    montage = render.write_montage(args.out, stages, target_bg=args.stretch_bg)
    mark("montage", t)
    log(f"  wrote {montage}  (all stages side by side)")

    # ---- hot-pixel comparison products + audit ----
    if hot_pre_before is not None:
        p, _ = hotpixels.write_compare(
            args.out, hot_pre_before, hot_pre_after, "pre", args.hot_sigma,
            args.hot_ksize, args.hot_abs, args.hot_color, args.hot_neigh)
        log(f"  pre-stacking hot-pixel removal: {hot_pre_count} channel-pixels "
            f"replaced over {len(records)} frames")
        log(f"  wrote {p}  (before vs after pre-stacking hot-pixel removal)")
        del hot_pre_before, hot_pre_after

    t = time.perf_counter()
    met = hotpixels.metrics(linear, args.hot_sigma, args.hot_ksize,
                            args.hot_abs, args.hot_color, args.hot_neigh)
    mark("hot_metrics", t)
    log("  hot-pixel audit on the linear master: %d hot pixels (%d channel-px)"
        % (met["hot_pixels"], met["hot_channel_px"]))
    with open(os.path.join(args.out, "hot_metrics.json"), "w") as f:
        json.dump(dict(method=args.method, hot_pre=bool(args.hot_pre),
                       hot_post=bool(args.hot_post), pre_px=hot_pre_count,
                       **met), f, indent=2)

    # ---- final deliverables: the finished image, named for the merged range --
    # Final products are merge_final_<start>_<end>.tif/.jpg (the frame numbers
    # of the merged sequence); every intermediate is a stageN_* file.
    first_id, last_id = files[0][0], files[-1][0]
    final_base = f"merge_final_{first_id}_{last_id}"
    t = time.perf_counter()

    stack_path = os.path.join(args.out, f"stack_{args.method}.tif")
    imgio.write_tiff16(stack_path, np.clip(linear * 65535.0, 0, 65535).astype(np.uint16))
    log(f"  wrote {stack_path} (linear master)")

    prev = stretch.stretch_preview(linear)
    prev = cv2.resize(prev, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(args.out, "stack_preview.jpg"),
                cv2.cvtColor(prev, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])

    final_tif = os.path.join(args.out, final_base + ".tif")
    final_jpg = os.path.join(args.out, final_base + ".jpg")
    imgio.write_tiff16(final_tif, (ys * 65535.0).astype(np.uint16))
    cv2.imwrite(final_jpg,
                cv2.cvtColor((np.clip(ys, 0, 1) * 255 + 0.5).astype(np.uint8),
                             cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    log(f"  wrote {final_tif} + {final_base}.jpg (full-res)"
        f"  (auto-stretch: mtf_m={mtf_m:.4f}, bg={bg:.4f}, target={args.stretch_bg})")
    mark("final_writes", t)
    timings["total"] = round(time.perf_counter() - t0, 3)

    with open(os.path.join(args.out, "transforms.json"), "w") as f:
        json.dump(dict(reference=ref_id, reg_scale=reg_scale, frames=records),
                  f, indent=2)
    with open(os.path.join(args.out, "timing.json"), "w") as f:
        json.dump(dict(frames=len(files), stacked=len(records),
                       cpu=cpus, available_ram_gb=(round(avail_mem, 1)
                                                   if avail_mem is not None else None),
                       workers=workers, io_workers=io_workers,
                       stack_workers=stack_workers, stack_band_mb=stack_band_mb,
                       render_workers=render_workers,
                       timings=timings), f, indent=2)
    log("  stage timings (s): "
        + ", ".join(f"{k}={v:g}" for k, v in timings.items()))

    if median_store is not None:
        del median_store
        try:
            os.remove(mm_path)
        except OSError:
            pass
    if ground_store is not None:
        del ground_store
        try:
            os.remove(ground_path)
        except OSError:
            pass

    log(f"done in {time.perf_counter()-t0:.1f}s -> {os.path.abspath(args.out)}")
    log(f"  {len(records)} frames stacked ({args.method})")
