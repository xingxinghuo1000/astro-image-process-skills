#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Star-field alignment + stacking for a sequence of Canon CR2 frames shot on a
fixed tripod (sky rotates while the foreground stays static).

The pipeline is split into single-purpose modules under ``scripts/astro/``;
this file is only the command-line front end.  See ``astro/pipeline.py`` for the
orchestration and the module docstrings for the individual steps:

    registration  ORB + star-centroid similarity registration (align the sky)
    stacking      weighted mean / block-wise median accumulation
    hotpixels     single-frame, per-channel colored hot-pixel detection/repair
    flat          auto flat (PixInsight ABE style) background-gradient removal
    stretch       adaptive (auto MTF) brightness stretch
    neutralize    per-channel sky-background neutralization
    star_reduce   star-mask morphology star shrink
    composite     sky/ground ("天地") stitch at the detected horizon
    render        per-stage TIFF/JPEG, montage and comparison figures

Stages (each enabled step adds one; the stretch is always applied):
    stacked -> (hotfix) -> (composite) -> (flat) -> stretched -> (neutralized)
        -> (star_reduced)

Every stage is written as ``stage<i>_<name>.tif`` + ``_preview.jpg`` and tiled
into ``stages_montage.jpg``; the finished image is written as a full-resolution
TIFF and JPEG.  Outputs default to ``--out`` (``out/``).

Usage (Windows, using the repo's virtualenv):
    .venv/Scripts/python.exe scripts/align_stack.py \
        --dir "D:\\pic\\DCIM-2026\\08-shangduhu-2" \
        --start 5994 --end 6012 --out out --ref 6003 --method median \
        --composite --flat --flat-strength 0.5 --neutralize
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
                   help="stitch the star-aligned sky onto a static ground layer "
                        "(the un-registered average of the frames), split at the "
                        "detected sky-ground boundary ('天地分界线')")
    c.add_argument("--composite-horizon", choices=["mask", "line"], default="mask",
                   help="'mask' follows a ragged tree/terrain line (default); "
                        "'line' forces a robust straight horizon fit")
    c.add_argument("--composite-feather", type=float, default=25.0, metavar="PX",
                   help="width (full-res px) of the blend band across the boundary "
                        "(default 25)")
    c.add_argument("--composite-thresh", type=float, default=0.05, metavar="T",
                   help="sky-evidence threshold (fraction of the 99.5th percentile); "
                        "raise it to pull the boundary up, lower it to include "
                        "fainter sky (default 0.05)")
    c.add_argument("--ground-dir", default="",
                   help="use a separate foreground sequence for the ground layer "
                        "(default: the input frames, un-registered)")
    c.add_argument("--ground-start", type=int, default=None,
                   help="first frame of the ground sequence (default: --start)")
    c.add_argument("--ground-end", type=int, default=None,
                   help="last frame of the ground sequence (default: --end)")
    return ap


def main():
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
