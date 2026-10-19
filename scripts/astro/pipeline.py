"""Pipeline orchestration: wire the individual steps together.

    decode previews -> static-pattern map -> register -> stack
        -> (optional) sky/ground composite -> (optional) flat
        -> stretch -> (optional) neutralize -> (optional) star reduction
        -> per-stage TIFFs + montage -> final TIFF + JPEG

The step implementations live in the sibling modules; this file only sequences
them and writes the deliverables.
"""
import os
import json
import time
import numpy as np
import cv2

from .util import log
from . import imgio
from . import registration as reg
from . import stacking
from . import flat as flatmod
from . import stretch
from . import neutralize
from . import star_reduce
from . import composite as composite_step
from . import horizon
from . import render
from . import hotpixels


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

    orb = reg.make_detector()
    t0 = time.time()

    # ---- pass 1: build a static-pattern map (hot pixels / fixed pattern) ----
    # A pixel that is bright in *most* frames is a hot pixel (static), while a
    # real star is bright in only a frame or two because the sky moves.  A high
    # percentile across frames therefore isolates the static pattern, which we
    # subtract so ORB can only latch onto real moving stars.
    log("pass 1: decoding grayscale previews to remove static pattern ...")
    grays = [imgio.load_gray8(path) for _, path in files]
    gh, gw = grays[0].shape
    static = reg.build_static_map(grays)
    cleaned = [np.clip(g - static, 0, None) for g in grays]
    del grays
    log(f"  static-pattern map built (half-res {gh}x{gw}, {len(files)} frames)")

    reg_scale = 0.5
    H, W = gh * 2, gw * 2
    k_ref = [i for i, (idx, _) in enumerate(files) if idx == ref_id][0]

    ref_clean = cleaned[k_ref]
    ref_enh = reg.enhance_for_registration(ref_clean)
    ref_kp, ref_desc = orb.detectAndCompute(ref_enh, None)
    ref_stars = reg.detect_stars(reg.highpass_float(ref_clean))
    log(f"  reference ORB features: {len(ref_kp)}  stars: {len(ref_stars)}")

    acc_sum = None
    acc_w = np.zeros((H, W), np.float32)
    ones = np.ones((H, W), np.float32)
    ground_acc = np.zeros((H, W, 3), np.float32) if args.composite else None
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
    if args.method == "median":
        mm_path = os.path.join(args.out, "_aligned_uint16.dat")
        median_store = stacking.make_median_store(mm_path, len(files), H, W)

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
            hpf = reg.highpass_float(clean)
            enh = reg.enhance_for_registration(clean)
            kp, desc = orb.detectAndCompute(enh, None)
            M_half, gd, minl = reg.register(ref_desc, ref_kp, desc, kp)
            if M_half is None:
                log(f"  IMG_{idx}: FAILED to register (good={gd}, inliers={minl}); skipped")
                continue
            mov_stars = reg.detect_stars(hpf)
            M_half = reg.refine_transform(M_half, ref_stars, mov_stars)
            M_full = reg.scale_matrix(M_half, 1.0 / reg_scale)

        sc = reg.star_score(ref_enh, enh, M_half)
        resid_med, resid_frac = (0.0, 1.0) if idx == ref_id else \
            reg.star_residual(ref_stars, reg.detect_stars(reg.highpass_float(clean)),
                              M_half)
        if idx != ref_id and not reg.transform_is_plausible(M_half, gw, gh, minl,
                                                            sc, resid_med, resid_frac):
            log(f"  !! IMG_{idx}: registration rejected "
                f"(score={sc:.2f} inl={minl} resid={resid_med:.2f}px "
                f"frac={resid_frac:.2f}); frame excluded from stack")
            continue

        if ground_acc is not None:
            if var_sum is None:
                var_sum = np.zeros((gh, gw), np.float32)
                var_sq = np.zeros((gh, gw), np.float32)
            var_sum += clean
            var_sq += clean * clean
            var_n += 1

        lin = imgio.load_linear(path)
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
            median_store[k] = np.clip(warped * 65535.0, 0, 65535).astype(np.uint16)

        if ground_acc is not None:
            ground_acc += lin
            ground_n += 1

        if args.save_frames:
            imgio.write_tiff16(os.path.join(frames_dir, f"IMG_{idx}.tif"),
                               np.clip(warped * 65535.0, 0, 65535).astype(np.uint16))

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
        stack = stacking.mean_stack(acc_sum, acc_w)
    else:
        log(f"  computing median over {len(files)} frames (block-wise) ...")
        stack = stacking.median_stack(median_store, H, W)

    # ---- staged post-processing ----
    stages = []

    def add_stage(name, img, linear=False):
        stages.append((f"stage{len(stages) + 1}_{name}", img, linear))

    add_stage("stacked", stack, linear=True)
    master = stack

    # ---- optional hot-pixel removal on the stacked image (single frame) ---------
    if args.hot_post:
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

    # ---- optional sky/ground composite -----------------------------------------
    # Stitch the star-aligned sky onto a static ground layer (the plain average of
    # the frames), splitting them at the detected sky-ground boundary.  This
    # replaces the linear master that every later step operates on.
    if args.composite:
        if args.ground_dir:
            ground = _stack_ground_from_dir(args)
            if ground is None:
                log("  !! composite: no ground frames found; using the input frames")
                ground = stacking.mean_ground(ground_acc, ground_n)
        else:
            log(f"  ground layer: averaging {ground_n} unaligned frames "
                "(static foreground)")
            ground = stacking.mean_ground(ground_acc, ground_n)
        sigma = max(8.0, 0.01 * gw)
        evidence = horizon.sky_evidence(
            horizon.variance(var_sum, var_sq, var_n), sigma)
        if os.environ.get("ASTRO_DEBUG_VAR"):
            np.savez(os.path.join(args.out, "_dbg_var.npz"),
                     var_sum=var_sum, var_sq=var_sq, var_n=var_n)
        comp, info = composite_step.run_composite(
            master, ground, evidence, args.out, target_bg=args.stretch_bg,
            mode=args.composite_horizon, feather=args.composite_feather,
            thresh=args.composite_thresh)
        if comp is not None:
            add_stage("composite", comp, linear=True)
            master = comp
        linear = master
    else:
        linear = master

    # stage: auto flat (ABE) -- fit and subtract the smooth per-channel
    # background gradient on the LINEAR image, where gradients are additive,
    # i.e. before the nonlinear stretch.
    if args.flat:
        flat_full, finfo = flatmod.auto_flat(linear, bins=args.flat_bins,
                                             degree=args.flat_degree,
                                             pct=args.flat_pct, mode=args.flat_mode)
        linear = np.clip(linear + args.flat_strength * (flat_full - linear), 0, 1)
        log("  auto-flat (ABE, %s, strength=%.2f): model level " %
            (args.flat_mode, args.flat_strength)
            + ", ".join("ch%d=%.4f..%.4f" % (c, mn, mx)
                        for c, (mn, mx, _) in enumerate(finfo)))
        add_stage("flat", linear, linear=True)
        sweep = [float(x) for x in args.flat_sweep.split(",") if x.strip()]
        cmp_path = render.write_flat_compare(master, flat_full, args.stretch_bg,
                                             args.neutralize, args.out, sweep,
                                             applied=args.flat_strength)
        log(f"  wrote {cmp_path}  (no flat vs auto flat"
            + ("" if not sweep else " vs strength sweep " + args.flat_sweep) + ")")

    # stage: adaptive brightness stretch (auto MTF), so faint structure shows
    ys, mtf_m, bg = stretch.auto_stretch(linear, target_bg=args.stretch_bg)
    add_stage("stretched", ys)

    # stage: background neutralization -- scale each channel so the sky
    # background becomes a neutral grey (PixInsight BackgroundNeutralization).
    if args.neutralize:
        ys, sref, sgain = neutralize.background_neutralize(ys)
        log(f"  neutralized background: ref={np.round(sref, 4).tolist()} "
            f"gains={np.round(sgain, 4).tolist()}")
        add_stage("neutralized", ys)

    # stage: star reduction -- shrink stars under a high-pass star mask.
    if args.star_reduce > 0:
        ys, star_mask = star_reduce.reduce_stars(ys, amount=args.star_reduce)
        cv2.imwrite(os.path.join(args.out, "star_mask.png"),
                    cv2.cvtColor((np.clip(star_mask, 0, 1) * 255).astype(np.uint8),
                                 cv2.COLOR_GRAY2BGR))
        log(f"  stars reduced by {args.star_reduce:g}px "
            f"(mask {100.0 * (star_mask > 0.05).mean():.2f}% of frame)")
        add_stage("star_reduced", ys)

    # ---- every stage as its own TIFF + preview, plus a comparison montage ----
    log("  writing per-stage images ...")
    for name, img, _ in stages:
        render.write_stage(args.out, name, img)
    montage = render.write_montage(args.out, stages, target_bg=args.stretch_bg)
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

    met = hotpixels.metrics(linear, args.hot_sigma, args.hot_ksize,
                            args.hot_abs, args.hot_color, args.hot_neigh)
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

    with open(os.path.join(args.out, "transforms.json"), "w") as f:
        json.dump(dict(reference=ref_id, reg_scale=reg_scale, frames=records),
                  f, indent=2)

    if median_store is not None:
        del median_store
        try:
            os.remove(mm_path)
        except OSError:
            pass

    log(f"done in {time.time()-t0:.1f}s -> {os.path.abspath(args.out)}")
    log(f"  {len(records)} frames stacked ({args.method})")
