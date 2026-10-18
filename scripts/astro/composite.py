"""Sky/ground composite -- stitch a star-aligned sky stack onto a static ground
stack, splitting them at the detected sky-ground boundary ("天地分界线").

Both layers come from the same frames:

* the **sky** stack is registered on the stars, so the sky is sharp and the
  static foreground is smeared by the frame-to-frame rotation;
* the **ground** stack is the plain (un-registered) average, so the foreground
  is sharp and the sky is smeared into star trails.

The boundary comes from ``horizon.detect`` (temporal motion variance); this
module builds the feathered mask, blends the two layers with it, and writes the
inspection figures (the ground layer, the mask and an overlay).
"""
import os
import cv2
import numpy as np

from .util import log, to_u8
from .imgio import write_tiff16
from .stretch import auto_stretch
from . import horizon


def compose(sky, ground, mask):
    m = mask[..., None]
    return np.clip(sky * m + ground * (1.0 - m), 0, 1)


def _overlay(comp, half_mask, line, target_bg, path, quality=88):
    """Inspection figure: the composited result with the detected boundary."""
    disp = to_u8(auto_stretch(comp, target_bg=target_bg)[0])
    disp = cv2.resize(disp, (half_mask.shape[1], half_mask.shape[0]),
                      interpolation=cv2.INTER_AREA)
    cnts, _ = cv2.findContours((half_mask > 0.5).astype(np.uint8),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(disp, cnts, -1, (0, 200, 255), 2)
    a, b = line
    if a is not None:
        h, w = disp.shape[:2]
        cv2.line(disp, (0, int(round(b))), (w - 1, int(round(a * (w - 1) + b))),
                 (0, 255, 0), 2)
    cv2.putText(disp, "sky / ground boundary", (15, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.imwrite(path, cv2.cvtColor(disp, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, quality])


def run_composite(sky, ground, evidence, out_dir, target_bg=0.20, mode="mask",
                  feather=25.0, thresh=0.05, min_sky=0.02, preview_scale=0.5):
    """Write the ground layer + inspection figures, detect the boundary from the
    sky `evidence` map, composite, and return (composite or None, info dict)."""
    write_tiff16(os.path.join(out_dir, "ground_stack.tif"),
                 np.clip(ground * 65535.0, 0, 65535).astype(np.uint16))
    prev = cv2.resize(to_u8(auto_stretch(ground, target_bg=target_bg)[0]), None,
                      fx=preview_scale, fy=preview_scale,
                      interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(out_dir, "ground_stack_preview.jpg"),
                cv2.cvtColor(prev, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 92])

    mask, info = horizon.detect(evidence, sky.shape[:2], mode=mode, t=thresh,
                                feather=feather)
    if mask is None:
        log(f"  !! composite: {info.get('reason', 'horizon detection failed')}"
            " -- skipping sky/ground stitch")
        return None, info
    cov = info["coverage"]
    log(f"  sky/ground boundary ({mode}): coverage={100.0 * cov:.1f}%"
        + ("" if info["slope"] is None else
           f", line slope={info['slope']:+.3f} intercept={info['intercept']:.1f}px"
           f" ({info['inliers']} pts)"))
    if cov < min_sky or cov > 1.0 - min_sky:
        log("  !! composite: sky or ground layer is essentially absent "
            f"(coverage {100.0 * cov:.1f}%) -- keeping the sky stack")
        return None, info

    comp = compose(sky, ground, mask)
    cv2.imwrite(os.path.join(out_dir, "sky_mask.png"), to_u8(mask))
    half_mask = cv2.resize(mask, (evidence.shape[1], evidence.shape[0]),
                           interpolation=cv2.INTER_AREA)
    _overlay(comp, half_mask, (info["slope"], info["intercept"]), target_bg,
             os.path.join(out_dir, "sky_ground_overlay.jpg"))
    log(f"  sky/ground composite: sky {100.0 * cov:.1f}% / "
        f"ground {100.0 * (1.0 - cov):.1f}%  (feather {feather:g}px)")
    return comp, info
