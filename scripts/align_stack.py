#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Star-field alignment + stacking for a sequence of Canon CR2 frames shot on a
fixed tripod (sky rotates while the foreground stays static).

The pipeline is split into single-purpose modules under ``scripts/astro/``;
this file is only the command-line front end.  See ``astro/pipeline.py`` for the
orchestration and the module docstrings for the individual steps:

    registration  ORB + star-centroid homography registration (align the sky)
    stacking      weighted mean / block-wise median accumulation
    hotpixels     single-frame, per-channel colored hot-pixel detection/repair
    flat          auto flat (PixInsight ABE style) background-gradient removal
    stretch       adaptive (auto MTF) brightness stretch
    neutralize    per-channel sky-background neutralization
    whitebalance  automatic star-colour / gray-world white balance
    star_reduce   star-mask morphology star shrink
    composite     sky/ground ("天地") stitch at the detected horizon
    render        per-stage TIFF/JPEG, montage and comparison figures

Stages (each enabled step adds one; the stretch is always applied):
    stacked -> (hotfix) -> (composite) -> (flat) -> stretched -> (neutralized)
        -> (whitebalanced) -> (star_reduced)

Every stage is written as ``stage<i>_<name>.tif`` + ``_preview.jpg`` and tiled
into ``stages_montage.jpg``; the finished image is written as a full-resolution
TIFF and JPEG.  Outputs default to ``--out`` (``out/``).

Usage (Windows, using the repo's virtualenv):
    .venv/Scripts/python.exe scripts/align_stack.py \
        --dir "D:\\pic\\DCIM-2026\\08-shangduhu-2" \
        --start 5994 --end 6012 --out out --ref 6003 --method median \
        --composite --flat --flat-strength 0.5 --neutralize --wb
"""
import argparse

from astro.pipeline import run


def build_parser():
    ap = argparse.ArgumentParser(
        description="Align + stack a fixed-tripod star-field CR2 sequence.")
    ap.add_argument("--dir", default=r"D:\pic\DCIM-2026\08-shangduhu-2")
    ap.add_argument("--start", type=int, default=5994)
    ap.add_argument("--end", type=int, default=6012)
    ap.add_argument("--out", default="out")
    ap.add_argument("--ref", type=int, default=6003,
                    help="reference frame number (default: middle of sequence)")
    ap.add_argument("--save-frames", action="store_true",
                    help="also write each sky-aligned full-res frame (.tif)")
    ap.add_argument("--method", choices=["mean", "median"], default="median",
                    help="stacking combiner. MEDIAN is required for astro "
                         "sequences: a hot pixel is fixed on the sensor while "
                         "the frame rotates, so mean leaves a colored streak "
                         "(value/N) that survives every threshold; median "
                         "rejects it as an outlier. 'mean' is kept only for "
                         "special cases and is discouraged")
    ap.add_argument("--exclude", default="", help="comma separated frame numbers to drop")
    ap.add_argument("--workers", type=int, default=0, metavar="N",
                    help="parallel worker threads for per-frame registration "
                         "(default 0 = one per frame up to the CPU count)")
    ap.add_argument("--io-workers", type=int, default=0, metavar="N",
                    help="parallel CR2 decode workers for previews and the "
                         "full-res stacking pass (default 0 = auto from CPU "
                         "count and RAM, capped at 8; ~2*N decoded frames "
                         "are held in RAM)")
    ap.add_argument("--stack-workers", type=int, default=0, metavar="N",
                    help="parallel row-band workers for the median stack "
                         "(default 0 = auto from CPU count and RAM; on a "
                         "72-core machine this selects 72)")
    ap.add_argument("--stack-band-mb", type=int, default=0, metavar="MB",
                    help="target working set per median row band (default 0 = "
                         "16 MB). Smaller bands use less RAM and expose more "
                         "parallel work; the minimum is one image row")
    ap.add_argument("--render-workers", type=int, default=0, metavar="N",
                    help="parallel workers for per-stage TIFF/JPEG writing "
                         "(default 0 = auto from CPU count, capped at 8)")
    ap.add_argument("--no-static", action="store_true",
                    help="disable static-pattern subtraction (automatic for "
                         "sky-tracked sequences, where a percentile map would "
                         "remove the stars themselves)")
    ap.add_argument("--stretch-bg", type=float, default=0.20,
                    help="target sky background level of the stretched output "
                         "(higher = brighter; default 0.20)")
    ap.add_argument("--neutralize", action="store_true",
                    help="neutralize the sky background to a grey after "
                         "stretching (PixInsight BackgroundNeutralization "
                         "style); with --ground-mask the measurement and the "
                         "correction are restricted to the sky")
    ap.add_argument("--neutralize-strength", type=float, default=1.0, metavar="S",
                    help="blend the neutralization gains in log space, 0..1 "
                         "(default 1.0 = full; 0.4-0.6 is gentler on strong "
                         "aurora / airglow scenes)")
    ap.add_argument("--star-reduce", type=float, default=1.0, metavar="PX",
                    help="shrink stars by PX pixels with a morphology + star-mask "
                         "step applied to the stretched image (PixInsight "
                         "StarReduction style); default 1 (gentle), 0 disables")

    w = ap.add_argument_group("white balance (after neutralization)")
    w.add_argument("--wb", action="store_true",
                   help="apply an automatic white balance after background "
                        "neutralization; writes wb_compare.jpg for visual review")
    w.add_argument("--wb-mode", choices=["whitepatch", "grayworld", "mixed"],
                   default="whitepatch",
                   help="whitepatch = brightest sky pixels (default), grayworld "
                        "= whole-sky channel means, mixed = geometric mean")
    w.add_argument("--wb-strength", type=float, default=0.6, metavar="S",
                   help="blend the correction in log space: 0 off, 1 full "
                        "(default 0.6)")
    w.add_argument("--wb-gains", default="", metavar="R,G,B",
                   help="manual gains (overrides the automatic measurement), "
                        "e.g. 1.02,0.99,0.99")

    g = ap.add_argument_group("flat (auto background / ABE)")
    g.add_argument("--flat", action="store_true",
                   help="remove the smooth background gradient on the LINEAR "
                        "image before stretching (PixInsight "
                        "AutomaticBackgroundExtractor / 'auto flat')")
    g.add_argument("--flat-mode", choices=["subtract", "divide"], default="subtract",
                   help="subtract the fitted background (light-pollution / "
                        "airglow gradients, default) or divide by it "
                        "(vignetting-style flat)")
    g.add_argument("--flat-bins", type=int, default=16,
                   help="grid columns for the background model (default 16)")
    g.add_argument("--flat-degree", type=int, default=3,
                   help="2D polynomial degree of the background surface (default 3)")
    g.add_argument("--flat-pct", type=float, default=25.0,
                   help="per-tile background percentile (default 25; low keeps "
                        "the estimate robust to stars)")
    g.add_argument("--flat-strength", type=float, default=1.0, metavar="S",
                   help="blend the flat correction, 0..1 (default 1.0). Lower it "
                        "if a full correction carves the middle of a "
                        "Milky-Way-filled field and leaves a bright border ring; "
                        "~0.3-0.5 removes the ring while keeping the gradient")
    g.add_argument("--flat-sweep", default="", metavar="S1,S2,...",
                   help="also render these flat strengths (0..1) into "
                        "flat_compare.jpg so you can pick --flat-strength from a "
                        "single run, e.g. 0.4,0.5,0.6,0.7")

    h = ap.add_argument_group("hot-pixel removal (single frame, per channel)")
    h.add_argument("--hot-pre", action="store_true",
                   help="detect and repair colored hot pixels in EACH raw frame "
                        "before stacking (removes the sensor defects at the "
                        "source; independent of tracking)")
    h.add_argument("--hot-post", action="store_true",
                   help="detect and repair colored hot pixels on the stacked "
                        "linear master (faster; prefer --method median first)")
    h.add_argument("--hot-sigma", type=float, default=6.0, metavar="N",
                   help="robust (MAD) threshold multiplier (default 6)")
    h.add_argument("--hot-abs", type=float, default=0.02, metavar="T",
                   help="absolute residual floor in [0,1] (default 0.02; raise "
                        "it to keep only stronger defects)")
    h.add_argument("--hot-ksize", type=int, default=5, choices=[3, 5],
                   help="median-filter size for the local background (default 5)")
    h.add_argument("--hot-color", type=float, default=2.0, metavar="R",
                   help="a defect must exceed R times the other two channels "
                        "(default 2; a white star is rejected)")
    h.add_argument("--hot-neigh", type=float, default=1.0, metavar="R",
                   help="a defect must exceed R times the brightest of its 8 "
                        "neighbours (default 1.0 = strict local maximum; a star "
                        "edge is rejected)")
    h.add_argument("--hot-grow", type=int, default=3, metavar="PX",
                   help="repair radius: also replace colored, elevated pixels "
                        "within PX of a detected peak, covering the whole "
                        "demosaic blob (a hot photosite spreads over ~3-5 px; "
                        "default 3)")

    c = ap.add_argument_group("sky/ground composite (天地合成)")
    c.add_argument("--composite", action="store_true",
                   help="stitch the star-aligned sky onto a temporally stacked "
                        "ground layer, split at the detected sky-ground boundary "
                        "('天地分界线')")
    c.add_argument("--composite-stage", choices=["early", "late"],
                   default="late",
                   help="'late' (default): the sky runs the whole chain first "
                        "(flat / stretch / neutralize / white balance / star "
                        "reduction measured on pure sky), the ground is "
                        "stretched with the sky's curve and stitched last -- "
                        "sky stats uncontaminated by the foreground, and the "
                        "ground layer (ground_stack.tif) stays a separate, "
                        "re-adjustable product you can re-grade and re-stitch "
                        "without rerunning the chain; needs --ground-mask. "
                        "'early': stitch in the linear domain (legacy), all "
                        "post-processing acts on the composite")
    c.add_argument("--composite-horizon", choices=["mask", "line"], default="mask",
                   help="'mask' follows a ragged tree/terrain line (default); "
                        "'line' forces a robust straight horizon fit")
    c.add_argument("--composite-feather", type=float, default=None, metavar="PX",
                   help="sigma (full-res px) of the blend band across the boundary "
                        "(default: 3 with --ground-mask -- a wide band over a dark "
                        "silhouette renders as a foggy gradient; 8 with the "
                        "variance fallback)")
    c.add_argument("--composite-thresh", type=float, default=0.05, metavar="T",
                   help="sky-evidence threshold (fraction of the 99.5th percentile); "
                        "raise it to pull the boundary up, lower it to include "
                        "fainter sky (default 0.05)")
    c.add_argument("--ground-mask", default="",
                   help="path to a ground mask (grayscale PNG, white = ground) "
                        "recognised by a large image model; overrides the "
                        "variance-based boundary detection")
    c.add_argument("--ground-method", choices=["mean", "median"], default="median",
                   help="combine the ground layer: median (default) rejects "
                        "transient flashlight/people and hot pixels; mean keeps "
                        "light painting")
    c.add_argument("--ground-top", type=int, default=0, metavar="N",
                   help="use only the N brightest ground frames (default: 5%% "
                        "of the frames, at least 3; -1 = all frames)")
    c.add_argument("--ground-dir", default="",
                   help="use a separate foreground sequence for the ground layer "
                        "(default: the input frames, un-registered)")
    c.add_argument("--ground-start", type=int, default=None,
                   help="first frame of the ground sequence (default: --start)")
    c.add_argument("--ground-end", type=int, default=None,
                   help="last frame of the ground sequence (default: --end)")
    c.add_argument("--ground-model",
                   choices=["auto", "raw", "translation", "homography"],
                   default="auto",
                   help="ground-motion model for stacking the lit ground layer: "
                        "auto (default) picks raw/translation/homography by "
                        "measured stack sharpness -- reliable for near-static "
                        "flashlight scenes (the lit-patch/parallax 'motion' is "
                        "not a camera transform, warping smears the static "
                        "foreground); for sequences with REAL tracked ground "
                        "motion pass homography (auto's sharpness metric is "
                        "biased toward raw by the median stack's fixed-pattern "
                        "preservation). 'raw' stacks the ground unaligned")
    c.add_argument("--ground-curve", choices=["sky", "own"], default="sky",
                   help="grading curve for the ground layer (late composite): "
                        "'sky' (default) reuses the sky's MTF curve (matched tone "
                        "scale); 'own' measures the MTF on the ground pixels alone "
                        "(presses black down -> dark foreground with less noise)")
    c.add_argument("--ground-target-bg", type=float, default=0.10, metavar="BG",
                   help="with --ground-curve own: MTF target background of the "
                        "ground strip (default 0.10; higher = brighter ground)")
    c.add_argument("--ground-gain", type=float, default=1.0, metavar="F",
                   help="display-domain multiplier on the graded ground "
                        "(default 1.0; flashlight-dark ground: 1.5-2.0)")
    return ap


def main():
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
