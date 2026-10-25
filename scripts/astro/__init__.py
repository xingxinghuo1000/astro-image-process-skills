"""Modular astro alignment / stacking / post-processing toolkit.

The package splits the pipeline into single-purpose modules so
``scripts/align_stack.py`` stays a thin command-line entry point:

    util          logging + small array helpers
    imgio         CR2 decoding and TIFF writing
    registration  ORB + star-centroid homography registration
    stacking      mean / median frame accumulation
    flat          auto flat (PixInsight ABE style) background-gradient removal
    stretch       adaptive (auto MTF) brightness stretch
    neutralize    per-channel sky-background neutralization
    whitebalance  automatic star/sky white balance
    star_reduce   star-mask morphology star shrink
    composite     sky/ground ("天地") composite -- pipeline 2, run by
                  scripts/composite_ground.py (mask-only boundary; pipeline 1
                  never stitches)
    render        per-stage TIFF/JPEG, montage and comparison figures
    pipeline      the orchestration that wires the steps together
"""
