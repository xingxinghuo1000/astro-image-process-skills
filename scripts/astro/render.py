"""Figures and per-stage outputs: TIFF+preview, montage, flat comparison."""
import os
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np

from .util import log, to_u8
from .imgio import write_tiff16
from .stretch import auto_stretch
from .neutralize import background_neutralize


def _label(img, text):
    cv2.putText(img, text, (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    return img


def _product_preview(lin, target_bg, neutralize, scale=0.5):
    """Stretched (and optionally neutralized) preview of a linear image."""
    ys, _, _ = auto_stretch(lin, target_bg=target_bg)
    if neutralize:
        ys, _, _ = background_neutralize(ys)
    p = to_u8(ys)
    return cv2.resize(p, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def write_stage(out_dir, name, rgb, preview_scale=0.5, quality=92):
    """Write one pipeline stage as a 16-bit TIFF plus a half-size JPEG preview."""
    tif = os.path.join(out_dir, name + ".tif")
    write_tiff16(tif, np.clip(rgb * 65535.0, 0, 65535).astype(np.uint16))
    prev = cv2.resize(to_u8(rgb), None, fx=preview_scale, fy=preview_scale,
                      interpolation=cv2.INTER_AREA)
    jpg = os.path.join(out_dir, name + "_preview.jpg")
    cv2.imwrite(jpg, cv2.cvtColor(prev, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, quality])
    log(f"    {name}.tif  ({prev.shape[1]}x{prev.shape[0]} preview)")


def write_montage(out_dir, stages, target_bg=0.20, cols=3, scale=0.22, quality=90):
    """Tile every stage into one labelled figure so the whole progression can be
    compared at a glance.  `stages` is a sequence of (name, image, linear) and
    linear stages are shown stretched so they are visible next to the already
    stretched ones."""
    def panel(stage):
        name, img, linear = stage
        disp = auto_stretch(img, target_bg=target_bg)[0] if linear else img
        p = cv2.resize(to_u8(disp), None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_AREA)
        return _label(p, name)

    workers = max(1, min(4, len(stages)))
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            panels = list(ex.map(panel, stages))
    else:
        panels = [panel(s) for s in stages]
    while len(panels) % cols:
        panels.append(np.zeros_like(panels[0]))
    rows = [np.hstack(panels[i:i + cols]) for i in range(0, len(panels), cols)]
    grid = np.vstack(rows)
    path = os.path.join(out_dir, "stages_montage.jpg")
    cv2.imwrite(path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, quality])
    return path


def write_wb_compare(out_dir, before, after, gains, scale=0.30, quality=92):
    """Before/after white-balance figure for visual review."""
    def panel(img, text):
        p = cv2.resize(to_u8(img), None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_AREA)
        return _label(p, text)
    g = np.asarray(gains, np.float64)
    row = np.hstack([
        panel(before, "before WB"),
        panel(after, "after WB  gains %.3f/%.3f/%.3f" % tuple(g)),
    ])
    path = os.path.join(out_dir, "wb_compare.jpg")
    cv2.imwrite(path, cv2.cvtColor(row, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, quality])
    return path


def write_flat_compare(stack_lin, flat_lin, target_bg, neutralize, out_dir,
                       sweep=(), applied=1.0, scale=0.5):
    """Side-by-side stretched previews -- no-flat vs auto-flat, plus an optional
    strength sweep -- so the user can judge whether the flat helps and pick a
    `--flat-strength`.  The flat is a judgement call (it removes the light-
    pollution gradient but can carve the glow of a frame-filling Milky Way), so
    the pipeline always writes this figure whenever --flat is used.  With no
    sweep the second panel shows the strength actually applied."""
    def blend(s):
        return np.clip(stack_lin + s * (flat_lin - stack_lin), 0, 1)

    specs = [("no flat", stack_lin)]
    specs += [("flat s=%.2f" % s, blend(s)) for s in sweep]
    if not sweep:
        specs.append(("flat s=%.1f" % applied, blend(applied)))

    def panel(spec):
        title, img = spec
        return _label(_product_preview(img, target_bg, neutralize, scale), title)

    workers = max(1, min(4, len(specs)))
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            panels = list(ex.map(panel, specs))
    else:
        panels = [panel(s) for s in specs]
    row = np.hstack(panels)
    path = os.path.join(out_dir, "flat_compare.jpg")
    cv2.imwrite(path, cv2.cvtColor(row, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 90])
    return path
