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

# Median full-res star displacement (px) under which the sequence is judged
# sky-tracked: registration barely had to move the stars, so the camera rode a
# tracking mount and the *ground* is what sweeps across the frames.
STAR_TRACK_MAX_PX = 12.0


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


def _plan_ground_motion(args, files, cleaned, ground_mask_img, ground_bbox,
                        gw, gh, stars_tracked, ground_keep):
    """Per-frame ground warps (full-res 3x3, crop-space model scaled x2) for
    the composite ground layer.  Returns (ground_shift, ground_M,
    ground_static, ground_keep).

    Fixed tripod: the ground is static, no warp.  Sky-tracked: the camera
    rotates *against the ground*, so the apparent ground motion is a
    homography (rotation around the polar-axis point + wide-angle
    perspective terms).  A translation-only warp locks the crop centroid and
    leaves a lever-arm residual growing toward the edges -- "aligned in the
    middle, ghosted at the corners".  We fit a per-frame homography from 8
    block phase-correlations on the ground crop (the dark flashlight-lit
    grass has no features that match; correlation does not need them), and
    restrict the ground layer to a temporally CONTIGUOUS lit cluster:
    correlation across flashlight clusters is meaningless (the lit scene
    patch changes with the beam) and a short-window model must never be
    extrapolated over the whole sequence.  The layer is anchored at the
    cluster centre so the model mask (drawn from the ground preview) lines
    up.  Fallback: the legacy translation arc; last resort: no warp.
    """
    n = len(files)
    ground_shift = None
    ground_M = [None] * n
    ground_static = True
    if not stars_tracked:
        log("  ground: fixed tripod (stars sweep) -> stacking unaligned "
            "frames (foreground is static)")
        return ground_shift, ground_M, ground_static, ground_keep
    if ground_bbox is None:
        log("  ground motion: no usable ground region; averaging unaligned frames")
        return ground_shift, ground_M, ground_static, ground_keep

    gy0, gy1, gx0, gx1 = ground_bbox
    hy0, hy1 = max(0, gy0 // 2), min(gh, (gy1 + 1) // 2)
    hx0, hx1 = max(0, gx0 // 2), min(gw, (gx1 + 1) // 2)
    # On a tracked rig the stars in a rectangular ground bbox are static and
    # out-bright the dark moving ground, so a plain phase correlation locks
    # onto the stars and reports "static".  Weight the crop by the ground
    # mask (sky pixels zeroed) so only the moving foreground carries the
    # correlation.
    gw_half = None
    if ground_mask_img is not None:
        gm_half = cv2.resize((ground_mask_img > 127).astype(np.uint8),
                             (gw, gh), interpolation=cv2.INTER_NEAREST) > 0
        # Correlation blocks weigh ground texture INCLUDING the fade band
        # above the composite mask line: fog over the foreground is scene
        # content that moves with it, and zero-weighting it starves the top
        # block row (01: head window fits dropped to 1, every homography
        # rejected).  The composite mask keeps its fade-free line.
        upm = np.roll(gm_half, -175, axis=0); upm[-175:] = False
        gm_half |= upm
        gw_half = np.ascontiguousarray(
            gm_half[hy0:hy1, hx0:hx1].astype(np.float32))
        gw_half = cv2.erode(gw_half, np.ones((5, 5), np.uint8))
        if gw_half.sum() < 0.02 * gw_half.size:
            gw_half = None
    elif ground_bbox[0] == int(gh * 2 * 0.65):
        log("  ground motion: no --ground-mask; stars may dominate the "
            "crop on a tracked rig -> supply a model mask for reliable alignment")
    if hy1 - hy0 < 16 or hx1 - hx0 < 16:
        log("  ground motion: ground region too small; averaging unaligned frames")
        return ground_shift, ground_M, ground_static, ground_keep

    crop_w, crop_h = hx1 - hx0, hy1 - hy0
    win = cv2.createHanningWindow((crop_w, crop_h), cv2.CV_32F)
    wm = gw_half if gw_half is not None else np.ones((crop_h, crop_w), np.float32)
    win = win * wm
    crops = np.stack([cleaned[k][hy0:hy1, hx0:hx1].astype(np.float32)
                      for k in range(n)])
    # A dark tracked ground is out-signalled by the sensor-fixed pattern
    # (vignetting, amp glow, hot pixels), which phase correlation locks onto
    # at shift 0 ("static").  The temporal median of the crop is exactly
    # that fixed pattern, so subtract it and correlate only the residual.
    fixed = np.median(crops, axis=0)
    # Ground texture (flashlight-lit grass) exists in only a few frames, and
    # the sky reference frame is often a pitch-black silhouette.  Each
    # candidate flashlight window gets its own reference frame and its own
    # correlation pass: a beam that sweeps exposes a LIT PATCH that changed
    # between windows, so cross-window correlation is meaningless (measured
    # response <0.05), and a window whose blocks are too dark to fit must
    # hand selection to the next window (01: the stones-lit tail window fits
    # no homography; the arc fallback smeared).
    # Candidate windows: top-brightness frames UNION lit-increment frames
    # (positive energy over the pixel-wise temporal median -- the flashlight
    # signature proper: 01's horizon-lit head window ranks below the
    # stones-lit tail in ground-mean brightness because its light lives in
    # the fog band the composite mask excludes, but its lit increment is
    # huge; sky glow cancels in the median subtraction).
    # Cluster score = sqrt(n) x brightness clipped at the 99.5th ground
    # percentile: a narrow direct beam blows highlights (area x P), a broad
    # grass-lit window wins the clipped mean (SKILL: direct needs >1.8x to
    # flip -- the unclipped mean lets a blown window win on highlights).
    pos = np.clip((crops - fixed) * wm[None], 0.0, None)
    allv = (crops * wm)[:, wm > 0][:, ::20].ravel()
    P = float(np.percentile(allv, 99.5)) if allv.size else 0.0
    bright = np.array([float((np.clip(crops[k], 0, max(P, 1e-6)) * wm).sum()
                              / max(wm.sum(), 1.0)) for k in range(n)])
    e = pos.reshape(n, -1).sum(axis=1)
    emed = float(np.median(e))
    top_frames = ([int(k) for k in np.flatnonzero(ground_keep)]
                  if ground_keep is not None else [])
    cand = sorted(set(top_frames)
                  | set(int(k) for k in np.flatnonzero(e > 1.6 * emed + 1e-9)))
    groups = []
    for k in cand:
        if groups and k - groups[-1][-1] <= 4:
            groups[-1].append(k)
        else:
            groups.append([k])
    gref = int(np.argmax(bright))
    refs = sorted({max(g, key=lambda gk: bright[gk]) for g in groups} | {gref},
                  key=lambda gk: -bright[gk])
    arc = None
    for gref_r in refs:
        ref_ground = (crops[gref_r] - fixed) * win
        idxs, shs = [], []
        for k in range(n):
            if k == gref_r:
                idxs.append(k)
                shs.append((0.0, 0.0))
                continue
            mov = (crops[k] - fixed) * win
            try:
                (dx, dy), resp = cv2.phaseCorrelate(mov, ref_ground, None)
            except cv2.error:
                continue
            if np.isfinite([dx, dy, resp]).all() and resp >= 0.05:
                idxs.append(k)
                shs.append((dx, dy))
        for k in top_frames:
            if k not in idxs:
                idxs.append(k)
                shs.append((float("nan"), float("nan")))
        idxs.sort()
        if len(idxs) < 3:
            continue
        idxs_f = np.asarray(idxs, np.float64)
        shs_f = np.asarray(shs, np.float64)
        meas = np.isfinite(shs_f).all(axis=1)
        kk = np.arange(n, dtype=np.float64)
        if meas.sum() >= 3:
            pred = np.stack([np.polyval(np.polyfit(idxs_f[meas],
                                                   shs_f[meas][:, a], 1), kk)
                             for a in (0, 1)], axis=1)
            shift = (pred - pred[int(np.median(idxs_f[meas]))]).astype(np.float32)
        else:
            shift = np.zeros((n, 2), np.float32)
        # Temporally contiguous lit clusters, tried best-scored first; each
        # anchors its own short-range model.  A candidate reference pass can
        # contain several windows (the beam-lit one + a correlation-joined
        # one); every window gets its fit attempt, first to measure >=3
        # homographies wins.
        clusters = [[idxs[0]]]
        for k in idxs[1:]:
            if k - clusters[-1][-1] > 4:
                clusters.append([k])
            else:
                clusters[-1].append(k)
        clusters.sort(key=lambda c: -(np.sqrt(len(c))
                                      * float(np.mean(bright[c]))))
        for cl in clusters:
            if len(cl) < 3:
                continue
            # Reference = the cluster's BRIGHTEST frame (cross-cluster
            # reference sits on a different lit patch, measures nothing).
            gref_c = max(cl, key=lambda k: bright[k])
            blocks = reg.ground_blocks(wm)
            ref_crop = crops[gref_c] - fixed
            items = []
            for k in cl:
                if k == gref_c:
                    items.append((k, np.eye(3)))
                    continue
                rows = reg.ground_block_shifts(crops[k] - fixed, ref_crop,
                                               blocks)
                Mf, st = reg.homography_from_shifts(rows, crop_w, crop_h)
                if Mf is not None:
                    items.append((k, Mf))
            if arc is None:
                arc = (idxs, idxs_f, shift, gref_r)
            if len(items) < 3:
                continue
            best = cl
            # Stack only frames whose homography was actually measured: gate-
            # passed frames with no usable fit are flashlight-dark silhouettes;
            # stacking them medians the lit detail to black, and a model
            # extrapolated onto them puts the anchor in a phase nothing was
            # measured at (01: 25-frame cluster, 10 fits in its lit head --
            # stones from the 3 lit frames smeared to black).
            orig_n = len(best)
            orig_cluster = list(best)
            runs, cur = [], [items[0][0]]
            for kf in [k for k, _ in items[1:]]:
                if kf - cur[-1] > 4:
                    runs.append(cur); cur = [kf]
                else:
                    cur.append(kf)
            runs.append(cur)
            best = max(runs, key=lambda c: np.sqrt(len(c))
                       * float(np.mean(bright[c])))
            coef, tinfo = reg.temporal_homography(
                [(k, M) for k, M in items if k in best], crop_w, crop_h)
            k_mid = int(np.median(best))
            A = np.linalg.inv(reg.eval_temporal_homography(coef, k_mid))
            S2 = np.diag([2.0, 2.0, 1.0])
            S2i = np.linalg.inv(S2)
            # In-plane rotation rate: slope of the per-frame rotation angles
            # (evaluated at the crop centre by homography_metrics, which
            # accounts for the perspective terms -- h21/h12 slopes alone
            # do not equal rotation).
            rot_rate = float(np.polyfit([float(k) for k, _ in items],
                                        [reg.homography_metrics(Mf, crop_w, crop_h)[0]
                                         for _, Mf in items], 1)[0])
            corner_disp = []
            # Hh is crop-local: applied to the FULL frame it pivots the
            # rotation/scale at the scaled crop centre, offset from the ground
            # strip by 2*(hx0, hy0) -- a 1.6deg roll becomes a 70px lever-arm
            # smear that medians the grass rows into a ghost.  Conjugate the
            # crop's full-res origin in before use.
            T = np.array([[1.0, 0.0, 2.0 * hx0],
                          [0.0, 1.0, 2.0 * hy0],
                          [0.0, 0.0, 1.0]])
            Ti = np.linalg.inv(T)
            for k in best:
                Hh = A @ reg.eval_temporal_homography(coef, k)
                ground_M[k] = (T @ S2 @ Hh @ S2i @ Ti).astype(np.float32)
                corner_disp.append(reg.corner_displacement(Hh, crop_w, crop_h))
            p95 = float(np.percentile(corner_disp, 95))
            ground_static = p95 < 1.5
            # Empirical ground-motion model choice.  The 8-block correlation
            # measures the *illumination patch* movement; with a static camera
            # (tripod creep, flashlight hand-off) that movement is parallax +
            # light-spot drift, NOT a camera transform, and warping by it
            # smears a sharp static foreground (01: raw stack grad^2 3.1e5,
            # every warped variant 1.0e4).  Score each model by the gradient
            # energy of the median stack of the measured half-res crops --
            # alignment only survives if it actually sharpens.
            def _model_score(Mrel):
                st = []
                for k in best:
                    d = cv2.warpPerspective(
                        crops[k] - fixed, Mrel[k].astype(np.float32),
                        (crop_w, crop_h), flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    st.append(d)
                m = np.median(np.stack(st), axis=0)
                gy = np.diff(m, axis=0)[:, :-1]
                gx = np.diff(m, axis=1)[:-1, :]
                # Score the INTERIOR of the real ground mask only: fade-band
                # glow has no ground to align, and warped variants pull black
                # border in up to p95 px deep -- scoring that edge lets a
                # smeared raw stack outscore an aligned one on border
                # artefacts (02 regression: raw beat homography 2.4:0.8).
                sel = (gm_base[:-1, :-1] > 0) & (wm[:-1, :-1] > 0.5)
                k_sel = 2 * int(np.ceil(p95)) + 13
                sel = cv2.erode(sel.astype(np.uint8),
                                np.ones((k_sel, k_sel), np.uint8)) > 0
                if sel.sum() < 2000:
                    sel = (gm_base[:-1, :-1] > 0) & (wm[:-1, :-1] > 0.5)
                # MEDIAN |grad|, not mean grad^2: an unwarped median keeps
                # every sensor hot pixel (spikes inflate mean grad^2), while
                # warped variants displace hot pixels so the median rejects
                # them -- 02: raw "outscored" homography 2.4:0.8 on that bias
                # alone while the stacked stones are visibly sharper aligned.
                g = np.abs(np.concatenate([gy[sel].ravel(), gx[sel].ravel()]))
                return float(np.median(g))
            gm_base = cv2.resize((ground_mask_img > 127).astype(np.uint8),
                                 (gw, gh), interpolation=cv2.INTER_NEAREST)[
                hy0:hy1, hx0:hx1]
            I3 = np.eye(3)
            Mrel = {m: {} for m in ("raw", "translation", "homography")}
            tM = {}
            for k in best:
                Mrel["homography"][k] = A @ reg.eval_temporal_homography(coef, k)
                Mrel["raw"][k] = I3
                dx, dy = shift[k] if np.isfinite(shift[k]).all() else (0.0, 0.0)
                tM[k] = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]])
            tinv = np.linalg.inv(tM[k_mid])
            for k in best:
                Mrel["translation"][k] = tM[k] @ tinv
            scores = {m: _model_score(Mrel[m]) for m in Mrel}
            model = max(scores, key=scores.get)
            # Every structural sharpness metric on the median stack is biased
            # toward "raw": an unaligned median preserves the sensor's fixed
            # pattern and photon-noise texture, while alignment warps move
            # them and the temporal median rejects them -- so inflated raw
            # scores win on 02 (real tracked motion, accepted look is the
            # homography stack) too.  auto is only trusted for near-static
            # flashlight scenes; tracked sequences pass --ground-model.
            forced = getattr(args, "ground_model", "auto")
            if forced != "auto":
                model = forced
            ground_model = model
            if model == "raw":
                ground_static = True
                for k in best:
                    ground_M[k] = None
            elif model == "translation":
                ground_static = False
                for k in best:
                    ground_M[k] = None
            diag_scores = {m: round(s, 4) for m, s in scores.items()}
            tmax = max(best, key=lambda k: abs(k - k_mid))
            diag = dict(mode=ground_model,
                        model_sharpness=diag_scores,
                        gref=int(files[gref_c][0]),
                        anchor=int(files[k_mid][0]),
                        cluster=[int(files[k][0]) for k in best],
                        cluster_all=([int(files[k][0]) for k in orig_cluster]
                                     if orig_n != len(best) else None),
                        fits=len(items), kept=tinfo["kept"],
                        rot_rate_deg_per_frame=round(rot_rate, 4),
                        corner_disp_p95_px=round(p95, 2),
                        frames=[])
            for k, Mf in items:
                rot, tx, ty, sc, _ = reg.homography_metrics(Mf, crop_w, crop_h)
                diag["frames"].append(dict(
                    frame=int(files[k][0]), rot_deg=round(rot, 4),
                    tx=round(tx, 2), ty=round(ty, 2), scale=round(sc, 5)))
            _write_ground_diag(args, diag)
            _write_ground_compare(args, files, crops, fixed, coef, A, shift,
                                  k_mid, best, tmax)
            log(f"  ground motion: lit cluster IMG_{files[min(best)][0]}.."
                f"IMG_{files[max(best)][0]} ({len(best)} frames"
                + (f", pruned from {orig_n}-frame cluster to measured fits"
                   if orig_n != len(best) else "")
                + f", ref IMG_"
                f"{files[gref_c][0]}): 8-block correlation, anchor IMG_{files[k_mid][0]}: "
                f"corner lever-arm displacement p95 {p95:.1f}px at half-res; "
                f"ground model={ground_model}"
                + (f" (forced by --ground-model)" if forced != "auto" else
                   f" (stack sharpness " + " ".join(f"{m}={s:.1e}"
                    for m, s in sorted(diag_scores.items())) + ")")
                + f" -> {'static' if ground_static else ground_model + '-aligned'}")
            ground_keep = np.zeros(n, bool)
            ground_keep[list(best)] = True
            log(f"  ground layer frames: lit cluster only ({len(best)} frames; "
                "replaces the top-brightness selection so every stacked frame "
                "is measured against the same lit scene patch)")
            return (shift if ground_model == "translation" else None), \
                ground_M, ground_static, ground_keep

    if arc is None:
        log("  ground motion: no textured (lit) ground frames to correlate; "
            "foreground is a silhouette -> stacking unaligned")
        return ground_shift, ground_M, ground_static, ground_keep
    idxs, idxs_f, shift, gref_r = arc
    norms = np.linalg.norm(shift, axis=1)
    p95_shift = float(np.percentile(norms, 95))
    ground_static = p95_shift < 1.5
    ground_shift = shift
    k_mid = int(np.median(idxs_f))
    log(f"  ground motion: {len(idxs)} lit frames correlated vs IMG_"
        f"{files[gref_r][0]}, arc fit -> shift vs arc centre IMG_{files[k_mid][0]}: "
        f"max {float(np.max(norms)):.2f} px, p95 {p95_shift:.2f} px at half-res "
        f"({'static' if ground_static else 'translation-aligned'})")
    return ground_shift, ground_M, ground_static, ground_keep


def _write_ground_diag(args, diag):
    path = os.path.join(args.out, "ground_align.json")
    with open(path, "w") as f:
        json.dump(diag, f, indent=2)
    log(f"  wrote {path}  (per-frame ground homographies, rotation rate)")


def _write_ground_compare(args, files, crops, fixed, coef, A, shift, k_mid,
                          best, tmax):
    """Inspection figure: the most extreme cluster frame warped into the
    anchor position by the legacy translation arc vs the block homography,
    next to the anchor crop.  Ghosting in the translation panel (grass tips
    doubled, worse toward the edges) is exactly what the homography removes."""
    u8 = lambda a: (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)
    anchor = crops[k_mid] - fixed
    p999 = float(np.percentile(anchor, 99.9))
    def show(a):
        g = np.clip(a / max(p999, 1e-6), 0, 1)
        return cv2.cvtColor(u8(g), cv2.COLOR_GRAY2RGB)
    Hh = A @ reg.eval_temporal_homography(coef, tmax)
    ch, cw = crops.shape[1], crops.shape[2]
    mov_t = cv2.warpAffine(crops[tmax] - fixed,
                           np.float32([[1, 0, shift[tmax][0]], [0, 1, shift[tmax][1]]]),
                           (cw, ch), borderMode=cv2.BORDER_CONSTANT)
    mov_h = cv2.warpPerspective(crops[tmax] - fixed, Hh.astype(np.float32),
                                (cw, ch), borderMode=cv2.BORDER_CONSTANT)
    panels = [show(anchor), show(mov_t), show(mov_h)]
    labels = (f"anchor IMG_{files[k_mid][0]}",
              f"IMG_{files[tmax][0]} translation warp",
              f"IMG_{files[tmax][0]} homography warp")
    for p, txt in zip(panels, labels):
        cv2.putText(p, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2)
    path = os.path.join(args.out, "ground_align_compare.jpg")
    cv2.imwrite(path, cv2.cvtColor(np.hstack(panels), cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 88])
    log(f"  wrote {path}  (translation vs homography warp of IMG_"
        f"{files[tmax][0]}, ghosting check)")


def _build_ground_layer(args, ground_mask_img, master, ground_store, ground_bbox,
                        ground_kept, ground_acc, ground_n, ground_static,
                        stack_workers, stack_band_mb, ground_mode="translation"):
    """Linear full-res ground layer: median of the kept (lit-cluster, aligned)
    frames pasted into a master-based canvas inside the ground bbox, with
    warp-edge pixels filled from the master; mean fallback."""
    if args.ground_dir:
        ground = _stack_ground_from_dir(args)
        if ground is None:
            log("  !! composite: no ground frames found; using the input frames")
            ground = stacking.mean_ground(ground_acc, ground_n)
        return ground
    if ground_store is not None:
        gy0, gy1, gx0, gx1 = ground_bbox
        ground = master.copy()
        if not ground_kept:
            log("  !! composite: no ground frames stacked; using the sky stack")
            return ground
        log(f"  ground layer: median of {ground_kept} frames "
            f"({gx1-gx0}x{gy1-gy0} px, "
            f"{'static foreground' if ground_static else ground_mode + '-aligned'})")
        gcrop = stacking.median_stack(
            ground_store[:ground_kept], gy1 - gy0, gx1 - gx0,
            workers=stack_workers, band_mb=stack_band_mb)
        # Pixels the shifted frames never cover (warp edges) stay 0 in every
        # stacked frame; fill them from the sky master, whose foreground sits
        # at the same aligned position as the ground layer (dark, continuous
        # -- no seam, no black strip).
        inv = (gcrop.sum(axis=2) <= 0)
        if inv.any():
            gcrop[inv] = master[gy0:gy1, gx0:gx1][inv]
            log(f"  ground layer: filled {100.0 * inv.mean():.1f}% "
                f"warp-edge pixels from the sky master")
        dst = ground[gy0:gy1, gx0:gx1]
        if ground_mask_img is not None:
            gm = ground_mask_img[gy0:gy1, gx0:gx1] > 127
            dst[gm] = gcrop[gm]
        else:
            dst[...] = gcrop
        return ground
    if ground_n:
        log(f"  ground layer: mean of {ground_n} frames "
            f"({'static foreground' if ground_static else ground_mode + '-aligned'})")
        return stacking.mean_ground(ground_acc, ground_n)
    log("  !! composite: no ground frames stacked; using the sky stack")
    return master.copy()


def run(args):
    os.makedirs(args.out, exist_ok=True)
    frames_dir = os.path.join(args.out, "aligned")
    if args.save_frames:
        os.makedirs(frames_dir, exist_ok=True)

    # A model mask boundary is an exact silhouette line, so a wide feather band
    # only smears sky glow into the dark foreground (visible foggy strip);
    # the ragged variance fallback is the case that needs a soft blend.
    if args.composite_feather is None:
        args.composite_feather = 3.0 if (args.composite and args.ground_mask) \
            else 8.0

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
    # Pixel correlation collapses under a 1-2 px shift, so a tracking mount with
    # a little periodic-error drift reads as "untracked" and the static-pattern
    # subtraction erases the stars it needs.  Phase correlation tolerates the
    # shift, so use it as a second probe: on a tracking mount the sky shifts by
    # tens of px between distant probes; on a fixed tripod it sweeps hundreds.
    tracked = False
    if len(grays) >= 5:
        a = reg.highpass_float(grays[k_ref])
        sl = slice(0, int(gh * 0.55))
        probes = sorted({len(grays) // 4, len(grays) // 2,
                         3 * len(grays) // 4, len(grays) - 1})
        crop = (0, int(gh * 0.55), 0, gw)
        if reg_mask is not None:
            ys = np.flatnonzero(reg_mask[:crop[1]].any(axis=1))
            xs = np.flatnonzero(reg_mask[:crop[1]].any(axis=0))
            if len(ys) >= 32 and len(xs) >= 32:
                crop = (int(ys.min()), int(ys.max()) + 1,
                        int(xs.min()), int(xs.max()) + 1)
        ca = np.ascontiguousarray(
            a[crop[0]:crop[1], crop[2]:crop[3]].astype(np.float32))
        win = cv2.createHanningWindow((crop[3] - crop[2], crop[1] - crop[0]),
                                      cv2.CV_32F)
        corrs, shifts, resps = [], [], []
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
            cb = np.ascontiguousarray(
                b[crop[0]:crop[1], crop[2]:crop[3]].astype(np.float32))
            try:
                (dx, dy), resp = cv2.phaseCorrelate(cb, ca, win)
            except cv2.error:
                continue
            if np.isfinite([dx, dy, resp]).all():
                shifts.append(float(np.hypot(dx, dy)))
                resps.append(float(resp))
        corr = float(np.median(corrs)) if corrs else 0.0
        probe_px = float(np.median(shifts)) if shifts else None
        resp_med = float(np.median(resps)) if resps else 0.0
        if corr > 0.65:
            tracked = True
            log(f"  sky is nearly static around the reference frame "
                f"(high-pass corr={corr:.2f}): treating as tracked; "
                f"static-pattern subtraction disabled")
        elif probe_px is not None and probe_px < 40.0 and resp_med > 0.02:
            tracked = True
            log(f"  sky shifts by only {probe_px:.1f} px between distant probe "
                f"frames (phase-corr response {resp_med:.3f}): treating as "
                f"tracked; static-pattern subtraction disabled")
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

    # Feathered sky mask (1 = sky) from the model ground mask: used to keep the
    # sky/ground stitch, the background neutralization and the white balance
    # all restricted to the sky.
    sky_mask = None
    if args.composite and ground_mask_img is not None:
        gmfull = cv2.resize(ground_mask_img, (W, H), interpolation=cv2.INTER_LINEAR)
        sky = 1.0 - gmfull.astype(np.float32) / 255.0
        sky = cv2.GaussianBlur(sky, (0, 0), max(1.0, args.composite_feather))
        sky_mask = np.clip(sky, 0, 1).astype(np.float32)
        log(f"  ground mask (model): {args.ground_mask}  "
            f"ground={100.0 * (gmfull > 127).mean():.1f}%")

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

    # A foreground light (flashlight / light painting) is often present in only
    # a handful of frames.  Median over the whole sequence rejects it and leaves
    # a dark, detail-free ground; select the brightest ground frames instead.
    ground_keep = None
    if args.composite and not args.ground_dir:
        if ground_mask_img is not None:
            gm_half = cv2.resize(
                (ground_mask_img > 127).astype(np.uint8), (gw, gh),
                interpolation=cv2.INTER_NEAREST) > 0
        else:
            gm_half = np.zeros((gh, gw), bool)
            gm_half[int(gh * 0.65):] = True
        if ground_bbox is not None:
            gy0, gy1, gx0, gx1 = ground_bbox
            region = np.zeros((gh, gw), bool)
            region[max(0, gy0 // 2):min(gh, (gy1 + 1) // 2),
                   max(0, gx0 // 2):min(gw, (gx1 + 1) // 2)] = True
            gm_half &= region
        if gm_half.any():
            n_all = len(files)
            if args.ground_top < 0:
                n_top = n_all
            elif args.ground_top > 0:
                n_top = min(args.ground_top, n_all)
            else:
                n_top = min(n_all, max(3, int(round(0.05 * n_all))))
            scores = np.array(
                [float(cleaned[k][gm_half].mean()) for k in range(n_all)],
                np.float32)
            order = np.argsort(scores)[::-1][:n_top]
            ground_keep = np.zeros(n_all, bool)
            ground_keep[order] = True
            chosen = ", ".join(f"IMG_{files[k][0]}" for k in sorted(order))
            log(f"  ground selection: {n_top}/{n_all} brightest frames ({chosen})")

    acc_sum = None
    acc_w = np.zeros((H, W), np.float32)
    ones = np.ones((H, W), np.float32)
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

    # Mount decision from the registration itself -- the robust test.  If the
    # stars barely needed aligning (small median displacement), the camera rode
    # a sky-tracking mount, so the *ground* sweeps across the frames and the
    # ground layer must be translation-aligned.  If the stars move a lot, it is
    # a fixed tripod: the ground is static and frames stack unaligned.
    dnorm = [float(np.hypot(r["tx"], r["ty"])) for r in reg_results
             if r["ok"] and r["idx"] != ref_id]
    star_disp = float(np.median(dnorm)) if dnorm else None
    stars_tracked = tracked
    if star_disp is not None:
        stars_tracked = star_disp < STAR_TRACK_MAX_PX
        log(f"  star motion vs reference: median {star_disp:.1f} px over "
            f"{len(dnorm)} frames -> "
            + ("sky-tracked mount: ground must be aligned" if stars_tracked
               else "fixed tripod: ground is static"))
    if stars_tracked and not args.hot_post:
        args.hot_post = True
        log("  tracked sequence: enabling --hot-post (fixed sensor hot pixels "
            "are not rejected by a median of static stars)")

    # Ground alignment plan: per-frame full-res warps for the ground layer
    # (homography on tracked rigs, see _plan_ground_motion).  May narrow
    # ground_keep to a lit cluster, so the median store is sized after this.
    ground_shift = None
    ground_M = [None] * len(files)
    ground_static = True
    if args.composite and not args.ground_dir:
        ground_shift, ground_M, ground_static, ground_keep = _plan_ground_motion(
            args, files, cleaned, ground_mask_img, ground_bbox, gw, gh,
            stars_tracked, ground_keep)

    ground_store = None
    ground_path = None
    ground_kept = 0
    n_ground = int(ground_keep.sum()) if ground_keep is not None else len(files)
    if (args.composite and args.ground_method == "median" and not args.ground_dir
            and ground_bbox is not None):
        gy0, gy1, gx0, gx1 = ground_bbox
        ground_path = os.path.join(args.out, "_ground_uint16.dat")
        ground_store = stacking.make_median_store(
            ground_path, n_ground, gy1 - gy0, gx1 - gx0)
        log(f"  ground median store: {gx1-gx0}x{gy1-gy0} px, {n_ground} frames")
    ground_acc = (np.zeros((H, W, 3), np.float32)
                  if args.composite and ground_store is None else None)
    ground_n = 0

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

        use_ground = ground_keep is None or ground_keep[k]
        if (ground_acc is not None or ground_store is not None) and use_ground:
            glin = lin
            if not ground_static:
                Mf = ground_M[k]
                if Mf is not None:
                    glin = cv2.warpPerspective(
                        lin, Mf, (W, H), flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                elif ground_shift is not None:
                    dx, dy = ground_shift[k]
                    if np.isfinite([dx, dy]).all():
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
    composite_late = (args.composite and
                      getattr(args, "composite_stage", "early") == "late")
    ground_mode = ("homography" if any(m is not None for m in ground_M)
                   else "translation")
    t = time.perf_counter()
    if args.composite and not composite_late:
        ground = _build_ground_layer(args, ground_mask_img, master, ground_store,
                                     ground_bbox, ground_kept, ground_acc,
                                     ground_n, ground_static, stack_workers,
                                     stack_band_mb, ground_mode)
        sigma = max(8.0, 0.01 * gw)
        evidence = horizon.sky_evidence(
            horizon.variance(var_sum, var_sq, var_n), sigma)
        if os.environ.get("ASTRO_DEBUG_VAR"):
            np.savez(os.path.join(args.out, "_dbg_var.npz"),
                     var_sum=var_sum, var_sq=var_sq, var_n=var_n)
        comp, info = composite_step.run_composite(
            master, ground, evidence, args.out, target_bg=args.stretch_bg,
            mode=args.composite_horizon, feather=args.composite_feather,
            thresh=args.composite_thresh, sky_mask=sky_mask)
        if comp is not None:
            add_stage("composite", comp, linear=True)
            master = comp
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
    ys, mtf_m, bg, sparams = stretch.auto_stretch(
        linear, target_bg=args.stretch_bg, return_params=True)
    add_stage("stretched", ys)
    mark("stretch", t)

    # stage: background neutralization -- scale each channel so the sky
    # background becomes a neutral grey (PixInsight BackgroundNeutralization).
    if args.neutralize:
        t = time.perf_counter()
        ys, sref, sgain = neutralize.background_neutralize(
            ys, sky_mask=sky_mask, strength=args.neutralize_strength)
        log(f"  neutralized background{' (sky only)' if sky_mask is not None else ''}: "
            f"ref={np.round(sref, 4).tolist()} "
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

    # ---- late composite: the sky ran the whole chain first (flat / stretch /
    # neutralize / white balance / star reduction all measured on PURE sky, no
    # foreground contamination); stretch the ground with the sky's curve and
    # stitch at the very end.  Needs --ground-mask for the boundary.
    if composite_late:
        t = time.perf_counter()
        ground = _build_ground_layer(args, ground_mask_img, master, ground_store,
                                     ground_bbox, ground_kept, ground_acc,
                                     ground_n, ground_static, stack_workers,
                                     stack_band_mb, ground_mode)
        if getattr(args, "ground_curve", "sky") == "own":
            gnd = None if sky_mask is None else 1.0 - sky_mask
            rows = np.flatnonzero(gnd.max(axis=1) > 0.9) if gnd is not None \
                else np.array([], np.int64)
            if len(rows) < 64:
                rows = np.arange(int(H * 0.75), H)
            _, _, _, gcurve = stretch.auto_stretch(
                ground[rows[0]:rows[-1] + 1],
                target_bg=args.ground_target_bg, return_params=True)
            log(f"  ground curve: own MTF (stats over rows {rows[0]}..{rows[-1]}, "
                f"target_bg {args.ground_target_bg:g})")
        else:
            gcurve = sparams
            log("  ground curve: sky MTF")
        ygr = np.clip(stretch.apply_stretch(ground, gcurve)
                      * getattr(args, "ground_gain", 1.0), 0.0, 1.0)
        if sky_mask is None:
            sky_mask = np.ones((H, W), np.float32)
            sky_mask[int(H * 0.65):] = 0.0
            sky_mask = cv2.GaussianBlur(sky_mask, (0, 0),
                                        max(1.0, args.composite_feather))
            log("  late composite: no --ground-mask, straight bottom-line "
                "boundary (the variance horizon cannot be used after stretching)")
        imgio.write_tiff16(os.path.join(args.out, "ground_stack.tif"),
                           np.clip(ground * 65535.0, 0, 65535).astype(np.uint16))
        gp = cv2.resize((np.clip(ygr, 0, 1) * 255 + 0.5).astype(np.uint8),
                        None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(args.out, "ground_stack_preview.jpg"),
                    cv2.cvtColor(gp, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        ys = composite_step.compose(ys, ygr, sky_mask)
        cv2.imwrite(os.path.join(args.out, "sky_mask.png"),
                    (np.clip(sky_mask, 0, 1) * 255).astype(np.uint8))
        disp = cv2.resize((np.clip(ys, 0, 1) * 255 + 0.5).astype(np.uint8),
                          None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        half = cv2.resize(sky_mask, (disp.shape[1], disp.shape[0]),
                          interpolation=cv2.INTER_AREA)
        cnts, _ = cv2.findContours((half > 0.5).astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(disp, cnts, -1, (0, 200, 255), 2)
        cv2.putText(disp, "sky / ground boundary (late composite)", (15, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(args.out, "sky_ground_overlay.jpg"),
                    cv2.cvtColor(disp, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
        log(f"  late composite: sky {100.0 * sky_mask.mean():.1f}% / "
            f"ground {100.0 * (1.0 - sky_mask.mean()):.1f}%  "
            f"(feather {args.composite_feather:g}px)")
        add_stage("composite", ys)
        mark("composite_late", t)

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
