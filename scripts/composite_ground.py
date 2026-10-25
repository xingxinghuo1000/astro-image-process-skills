#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ground composite -- PIPELINE 2 of the sky/ground ("天地") workflow.

Inputs are the layers pipeline 1 (scripts/align_stack.py --composite)
exported, plus the ground mask the agent drew AFTER viewing the exported
preview (SKILL.md forbids one-shot sky+ground compositing: the boundary must
come from a model that looked at ground_stack_preview.jpg):

    --sky    out/sky_final_<s>_<e>.tif   graded sky (display domain)
    --ground out/ground_stack.tif        linear temporal ground layer
    --mask   ground mask PNG, white = ground

It grades the ground (sky MTF from sky_curve.json, or its own MTF, plus an
optional gain), stitches at the feathered mask, and writes
merge_final_<s>_<e>.tif/.jpg + sky_mask.png + sky_ground_overlay.jpg.
No CR2 decoding, no registration: pure layer work.

Usage (Windows, using the repo's virtualenv):
    .venv/Scripts/python.exe scripts/composite_ground.py \
        --sky out/sky_final_4744_4869.tif --ground out/ground_stack.tif \
        --mask out/ground_mask.png --out out \
        --ground-curve own --ground-target-bg 0.10 --feather 3
"""
import argparse

from astro.composite import run_ground_composite


def build_parser():
    ap = argparse.ArgumentParser(
        description="sky/ground composite (pipeline 2): stitch the graded sky "
                    "and the exported ground layer at a model-drawn mask")
    ap.add_argument("--sky", required=True, metavar="PATH",
                    help="pipeline-1 graded sky TIFF (sky_final_<s>_<e>.tif)")
    ap.add_argument("--ground", required=True, metavar="PATH",
                    help="pipeline-1 linear ground layer (ground_stack.tif)")
    ap.add_argument("--mask", required=True, metavar="PATH",
                    help="ground mask PNG recognised by a vision model from "
                         "ground_stack_preview.jpg (white = ground, black = "
                         "sky); the composite refuses to run without it")
    ap.add_argument("--curve", default="", metavar="PATH",
                    help="sky MTF curve JSON (default: sky_curve.json next to "
                         "--sky)")
    ap.add_argument("--ground-curve", choices=["sky", "own"], default="sky",
                    help="grading curve for the ground layer: 'sky' (default) "
                         "reuses the sky's MTF curve (matched tone scale); "
                         "'own' measures the MTF on the ground pixels alone "
                         "(presses black down -> dark foreground with less "
                         "noise)")
    ap.add_argument("--ground-target-bg", type=float, default=0.10, metavar="BG",
                    help="with --ground-curve own: MTF target background of "
                         "the ground strip (default 0.10; higher = brighter)")
    ap.add_argument("--ground-gain", type=float, default=1.0, metavar="F",
                    help="display-domain multiplier on the graded ground "
                         "(default 1.0; flashlight-dark ground: 1.5-2.0)")
    ap.add_argument("--feather", type=float, default=3.0, metavar="PX",
                    help="sigma (full-res px) of the blend band across the "
                         "boundary (default 3: a model mask is an exact "
                         "silhouette line, a wide band smears sky glow into "
                         "the dark foreground as a visible foggy strip)")
    ap.add_argument("--out", default="out", metavar="DIR",
                    help="output directory (default out/)")
    return ap


def main():
    args = build_parser().parse_args()
    run_ground_composite(args)


if __name__ == "__main__":
    main()
