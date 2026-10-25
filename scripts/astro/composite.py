"""Pipeline 2 -- the sky/ground ("天地") composite.

This pipeline stitches pipeline 1's two exported layers at a sky/ground
boundary:

* ``sky_final_<s>_<e>.tif``   -- the graded sky (the whole chain measured on
  pure sky), display domain;
* ``ground_stack.tif``        -- the linear temporal ground layer (per-frame
  homography-aligned, lit-cluster median);
* a ground mask PNG           -- white = ground, drawn by a VISION MODEL after
  viewing ``ground_stack_preview.jpg``.

One-shot sky+ground compositing in a single command is forbidden (SKILL.md):
the mask must come from a model/human that LOOKED at the preview, which is
only possible once pipeline 1 has exported it.  This module therefore never
decodes CR2 frames and never detects the boundary from image statistics --
no mask, no composite.
"""
import json
import os
import re

import cv2
import numpy as np

from .util import log
from .imgio import write_tiff16
from .stretch import auto_stretch, apply_stretch


def compose(sky, ground, mask):
    m = mask[..., None]
    return np.clip(sky * m + ground * (1.0 - m), 0, 1)


def load_tiff16_rgb(path):
    """Read a 16-bit TIFF written by write_tiff16 back to float RGB [0, 1]."""
    a = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if a is None:
        raise SystemExit(f"cannot read {path}")
    if a.ndim == 2:
        a = cv2.cvtColor(a, cv2.COLOR_GRAY2RGB)
    else:
        a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    return a.astype(np.float32) / (65535.0 if a.dtype == np.uint16 else 255.0)


def run_ground_composite(args):
    """Grade the ground layer, stitch it under the sky at the model mask,
    and write merge_final_*.tif/.jpg + sky_mask.png + sky_ground_overlay.jpg."""
    os.makedirs(args.out, exist_ok=True)
    for p in (args.sky, args.ground):
        if not os.path.isfile(p):
            raise SystemExit(f"missing layer file: {p} (run pipeline 1 first: "
                             "scripts/align_stack.py --composite)")
    sky = load_tiff16_rgb(args.sky)
    ground = load_tiff16_rgb(args.ground)
    if sky.shape != ground.shape:
        raise SystemExit(f"layer sizes differ: sky {sky.shape} vs "
                         f"ground {ground.shape} (both must come from the "
                         "same pipeline-1 run)")
    H, W = sky.shape[:2]

    # The model mask is white = ground; the stitch mask is 1 = sky (the two
    # conventions are inverted -- feeding the model mask straight in shows the
    # sky layer over the foreground, the classic "all ground looks blurred"
    # symptom, SKILL.md).
    m = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise SystemExit(f"cannot read --mask {args.mask}")
    m = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)
    w = (m > 127).astype(np.float32)
    # A model mask boundary is an exact silhouette line, so the feather band
    # stays narrow: a wide band smears sky glow into the dark foreground
    # (visible foggy strip).
    sigma = max(1.0, args.feather)
    sky_mask = np.clip(cv2.GaussianBlur(1.0 - w, (0, 0), sigma), 0, 1)
    cov = float(sky_mask.mean())
    if cov < 0.02 or cov > 0.98:
        raise SystemExit(f"mask covers almost the whole frame or none of it "
                         f"(sky {100.0 * cov:.1f}%) -- check the mask polarity "
                         "(white = ground) and its size")
    log(f"  mask {args.mask}: ground {100.0 * (1.0 - cov):.1f}% "
        f"(feather sigma {sigma:g}px)")

    if args.ground_curve == "own":
        rows = np.flatnonzero(w.max(axis=1) > 0.9)
        if len(rows) < 64:
            rows = np.arange(int(H * 0.75), H)
        _, _, _, gcurve = auto_stretch(
            ground[rows[0]:rows[-1] + 1],
            target_bg=args.ground_target_bg, return_params=True)
        log(f"  ground curve: own MTF (stats over rows {rows[0]}..{rows[-1]}, "
            f"target_bg {args.ground_target_bg:g})")
    else:
        curve = args.curve or os.path.join(
            os.path.dirname(os.path.abspath(args.sky)), "sky_curve.json")
        if not os.path.isfile(curve):
            raise SystemExit(f"missing sky curve: {curve} (pipeline 1 writes "
                             "sky_curve.json next to sky_final_*.tif)")
        with open(curve) as f:
            cj = json.load(f)
        gcurve = dict(blk=np.asarray(cj["blk"], np.float64),
                      wht=float(cj["wht"]), m=float(cj["m"]))
        log(f"  ground curve: sky MTF ({curve})")
    ygr = np.clip(apply_stretch(ground, gcurve) * args.ground_gain, 0.0, 1.0)

    ys = compose(sky, ygr, sky_mask)

    rng = re.search(r"(\d+)_(\d+)", os.path.basename(args.sky))
    base = f"merge_final_{rng.group(1)}_{rng.group(2)}" if rng else "merge_final"
    final_tif = os.path.join(args.out, base + ".tif")
    final_jpg = os.path.join(args.out, base + ".jpg")
    write_tiff16(final_tif, (np.clip(ys, 0, 1) * 65535.0 + 0.5).astype(np.uint16))
    cv2.imwrite(final_jpg,
                cv2.cvtColor((np.clip(ys, 0, 1) * 255 + 0.5).astype(np.uint8),
                             cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])

    cv2.imwrite(os.path.join(args.out, "sky_mask.png"),
                (np.clip(sky_mask, 0, 1) * 255).astype(np.uint8))
    disp = cv2.resize((np.clip(ys, 0, 1) * 255 + 0.5).astype(np.uint8),
                      None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    half = cv2.resize(sky_mask, (disp.shape[1], disp.shape[0]),
                      interpolation=cv2.INTER_AREA)
    cnts, _ = cv2.findContours((half > 0.5).astype(np.uint8),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(disp, cnts, -1, (0, 200, 255), 2)
    cv2.putText(disp, "sky / ground boundary", (15, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    overlay = os.path.join(args.out, "sky_ground_overlay.jpg")
    cv2.imwrite(overlay, cv2.cvtColor(disp, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 88])
    log(f"  composite: sky {100.0 * cov:.1f}% / ground {100.0 * (1.0 - cov):.1f}%")
    log(f"  wrote {final_tif} + {base}.jpg (full-res)")
    log(f"  wrote {overlay}  (cyan = mask boundary; verify it follows the ridge)")
    log(f"done -> {os.path.abspath(args.out)}")
