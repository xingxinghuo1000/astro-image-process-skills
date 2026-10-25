# Star-field Align & Stack

**Languages:** English | [简体中文](docs/README-cn.md)

An **Agent Skill** (with a standalone Python CLI) that **aligns and stacks** a
sequence of Canon **CR2** frames shot from a **fixed tripod** (the sky rotates
between frames while the foreground stays still) or a **sky-tracked rig**, then
applies a PixInsight-style post-processing chain — auto flat (ABE), background
neutralization, automatic white balance, star reduction — and an optional
**sky/ground ("天地") composite** that keeps both the star field and the
foreground sharp.

![Processing pipeline](docs/pipeline-en.png)

**Deep dives:** [Single-image parallelization report](docs/single-image-parallelization-report.md) ·
[Wide-field registration algorithm](docs/wide-field-registration.md)

---

## What it does

| Capability | Description |
| --- | --- |
| **RAW decode** | Reads `IMG_<n>.CR2` with `rawpy` (camera WB, linear, AHD demosaic). |
| **Static-pattern removal** | A two-pass scheme builds a static-pattern map (hot pixels / fixed pattern) from the whole sequence and subtracts it so registration only locks onto real stars. Automatically skipped for sky-tracked sequences, where the stars themselves are static (the percentile map would erase them); `--hot-post` is then enabled automatically. |
| **Sky registration** | ORB feature matching + star-centroid refinement, fitting a projective **homography** per frame (the correct model for a rotating sky: `K R K^-1`), with plausibility checks (match score, inliers, star residual) that reject bad frames. Frames register in parallel with `--workers` (default: one per CPU core). |
| **Stacking** | Block-wise **median** (recommended) or weighted **mean**; median uses an on-disk memmap so long sequences fit in RAM, and independent row bands are processed in parallel (`--stack-workers`). |
| **Hot-pixel removal** | Per-channel, single-frame detection of colored hot pixels (isolated local maxima dominated by one channel), with cosmetic repair and before/after figures. `--hot-pre` (per raw frame) or `--hot-post` (stacked). |
| **Sky/ground composite** | Stitches the star-aligned sky onto a sharp ground layer at the sky/ground boundary. Fixed tripod: the ground is static, the layer is a median of the **brightest 5% of frames** (light painting is usually transient). Sky-tracked (equatorial mount): the ground sweeps across the frames, and the apparent motion is a camera rotation — a **per-frame homography** (fit from 8 block phase-correlations, flashlight-lit frame cluster only) aligns it; a plain translation aligns only the centre and ghosts the wide-angle edges. The boundary mask must be drawn by a **large vision model after viewing the exported ground preview** — compositing is a separate second command (`composite_ground.py`) that refuses to run without the mask; one-shot sky+ground compositing is forbidden and variance auto-detection was removed. |
| **Auto flat (ABE)** | PixInsight `AutomaticBackgroundExtractor`-style 2D polynomial background fit on the **linear** image, subtract/divide, with a blend strength and a strength sweep. |
| **Auto stretch** | Adaptive mid-tone-transfer (MTF) stretch that places the sky background at a target level. |
| **Background neutralization** | Per-channel scaling so the sky background becomes neutral grey (removes light-pollution color casts). |
| **Auto white balance** | Luminance-preserving star/sky white balance after neutralization (whitepatch, grayworld or mixed), with a `wb_compare.jpg` for visual review and manual `--wb-gains` override. |
| **Star reduction** | High-pass star mask + morphological shrink to make nebulae/the Milky Way stand out. |
| **Deliverables** | Every intermediate stage as 16-bit TIFF + preview, a stage montage, the full-resolution sky `sky_final_<start>_<end>.tif/.jpg` (pipeline 1), and the composite `merge_final_<start>_<end>.tif/.jpg` (pipeline 2). |

---

## Requirements

- Python 3.9+
- [numpy](https://numpy.org/), [opencv-python](https://pypi.org/project/opencv-python/), [rawpy](https://pypi.org/project/rawpy/)

This repository is developed on Windows and expects a virtualenv at `.venv/`.

```bash
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt \
    -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
```

On macOS/Linux use `./.venv/bin/python` instead of `./.venv/Scripts/python.exe`.

Input frames are **not** stored in the repo; point `--dir` at the folder that
contains `IMG_<n>.CR2`.

---

## Install as a skill

This repository **is an Agent Skill**: `SKILL.md` carries the frontmatter
(`name: astro-image-process-skills` + trigger keywords) and the agent-facing
instructions; `scripts/` is the implementation. Put the repository in a skills
directory, keeping the folder name equal to the skill name:

| Scope | Path |
| --- | --- |
| opencode (global) | `~/.config/opencode/skills/astro-image-process-skills/` |
| opencode (project) | `.opencode/skills/astro-image-process-skills/` |
| Claude Code / compatible | `~/.claude/skills/astro-image-process-skills/` |

```bash
git clone git@github.com:xingxinghuo1000/astro-image-process-skills.git \
    ~/.config/opencode/skills/astro-image-process-skills
```

Alternatively register the folder via `skills.paths` in `opencode.json`
(opencode scans those directories recursively for `**/SKILL.md`):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "skills": { "paths": ["~/src/astro-image-process-skills"] }
}
```

Restart opencode afterwards — skills are loaded at startup.

---

## Use the skill

You do **not** call the script by hand: ask the agent in natural language. The
skill's description advertises the trigger keywords (星野, 星空, 银河, 对齐,
叠加, *star field alignment*, *astro stacking*, *CR2 alignment*), so the agent
loads it automatically when a request matches. It then reads `SKILL.md`, prepares
the `.venv`, runs `scripts/align_stack.py` with the right flags and inspects the
results — and because it can see images, it will also produce the `--ground-mask`
for the sky/ground composite. Things you can ask for:

**1. Plain align + stack (the everyday case)**

> "Stack this nightscape CR2 sequence: `D:\pic\DCIM-2026\08-shangduhu-2`, frames 5994–6012."

**2. Stack + full post-processing (flat, neutralize, white balance, star reduction)**

> "把这些星野照片对齐叠加，去掉光害梯度、做背景中性化，星点稍微收一点：`D:\pic\DCIM-2026\08-shangduhu-2`，5994–6012。"

**3. Full workflow + sky/ground composite — sharp tree *and* sharp sky**

This is a multi-turn exchange:

> **You:** "`D:\pic\DCIM-2026\08-shangduhu-2` 的 5994–6012 走完整流程：平场 + 中性化 + 缩星，最后做天地合成，前景要清晰。"

Sky and ground are **two separate pipelines — one-shot compositing is forbidden**.
The agent runs the sky pipeline with `--composite` (which exports the ground
layer but never stitches), then **looks at the exported
`ground_stack_preview.jpg` with its vision**, marks the foreground (tree, dome,
terrain) as a white-on-black `ground_mask.png`, and runs the ground-composite
pipeline on the two layers — under the hood:

```bash
# pipeline 1: sky chain + ground-layer export (never composites)
./.venv/Scripts/python.exe scripts/align_stack.py \
    --dir "D:\pic\DCIM-2026\08-shangduhu-2" --method median \
    --flat --flat-strength 0.5 --neutralize --wb --star-reduce 1 \
    --composite --out out

# the agent views out/ground_stack_preview.jpg and draws out/ground_mask.png

# pipeline 2: ground composite (refuses to run without --mask)
./.venv/Scripts/python.exe scripts/composite_ground.py \
    --sky out/sky_final_5994_6012.tif --ground out/ground_stack.tif \
    --mask out/ground_mask.png --out out
```

It reports the final `out/merge_final_5994_6012.tif/.jpg` and shows you
`out/sky_ground_overlay.jpg` so you can confirm the split.

**4. Check a flat before committing to it**

> "这个画面用平场会不会压出亮环？给我看几个强度的对比。" — the agent runs
> `--flat-sweep 0.4,0.5,0.6,0.7` and shows you `flat_compare.jpg`.

**5. Colored hot-pixel streaks**

> "银河里有很多彩色小点拖线，帮我去掉。" — the agent adds `--hot-post` (or
> `--hot-pre`) and shows `hot_compare_post.jpg`.

**6. Keep refining conversationally**

> "树太糊了 / 天空又出现星轨了 / 背景还偏紫 / 星星再收一点" — the agent tweaks
> `--ground-top`, `--feather`, `--ground-gain`, `--flat-strength`, `--neutralize`, `--wb-strength`
> or `--star-reduce` and re-runs, re-checking the overlay and montage each time.

Prefer driving it by hand? See [Manual CLI usage](#manual-cli-usage).

---

## Manual CLI usage

Run the script directly (useful for scripting or debugging); the arguments are
the same ones the skill uses under the hood.

```bash
./.venv/Scripts/python.exe scripts/align_stack.py \
    --dir "D:\pic\DCIM-2026\08-shangduhu-2" \
    --start 5994 --end 6012 \
    --method median --out out

./.venv/Scripts/python.exe scripts/align_stack.py --help
```

```text
align_stack.py
  --dir DIR                folder containing IMG_<n>.CR2
  --start N --end N        frame-number range (missing numbers are skipped)
  --out DIR                output directory (default: out)
  --ref N                  reference frame (default: middle of the range)
  --method {mean,median}   combiner (default: median; mean is discouraged)
  --exclude 6007,6012      drop specific frames
  --workers N              parallel registration threads (default 0 = auto)
  --io-workers N           parallel CR2 decoders (default 0 = auto, cap 8)
  --stack-workers N        parallel median row bands (default 0 = auto)
  --stack-band-mb MB       target median band working set (default 0 = 16 MB)
  --render-workers N       parallel stage TIFF/JPEG writers (default 0 = auto, cap 8)
  --no-static              disable static-pattern subtraction
  --stretch-bg 0.20        target sky-background level after stretch
  --save-frames            also save each aligned full-res frame
  --neutralize             background neutralization (after stretch)
  --star-reduce PX         star shrink in pixels (default 1; 0 disables)

White balance (after neutralization)
  --wb                     enable automatic white balance
  --wb-mode {whitepatch,grayworld,mixed}   (default: whitepatch)
  --wb-strength S          0..1 log-space blend (default 0.6)
  --wb-gains R,G,B         manual gains (overrides the measurement)

Flat (auto background / ABE)
  --flat                   enable
  --flat-mode {subtract,divide}   (default: subtract)
  --flat-bins 16 --flat-degree 3 --flat-pct 25
  --flat-strength S        0..1 blend (default 1.0; use ~0.3-0.6 if a ring appears)
  --flat-sweep 0.4,0.5,0.6,0.7    render several strengths into flat_compare.jpg

Hot pixels
  --hot-pre / --hot-post   detect+repair per raw frame / on the stacked master
  --hot-sigma 6 --hot-abs 0.02 --hot-ksize 5 --hot-color 2 --hot-neigh 1 --hot-grow 3

Ground layer for the composite (pipeline 1 exports, never stitches)
  --composite              build+export ground_stack.tif, preview, sky_curve.json
  --ground-band 0.35       ground crop band (fraction of frame height)
  --ground-method {mean,median}   ground combiner (default: median)
  --ground-top N           use the N brightest ground frames (default: 5%, min 3;
                           -1 = all frames)
  --ground-model {auto,raw,translation,homography}
  --ground-dir DIR --ground-start N --ground-end N   ground from another sequence

Ground composite (pipeline 2: scripts/composite_ground.py)
  --sky PATH               pipeline-1 sky_final_<s>_<e>.tif   (required)
  --ground PATH            pipeline-1 ground_stack.tif        (required)
  --mask PATH              model-drawn ground mask, white = ground (required:
                           no mask, no composite)
  --curve PATH             sky MTF curve (default: sky_curve.json next to --sky)
  --ground-curve {sky,own} --ground-target-bg 0.10 --ground-gain 1.0
  --feather 3              blend sigma px
  --out DIR
```

### Defaults and conventions

- `--method median` is the default and is **strongly recommended** for astro
  sequences: a hot pixel is fixed on the sensor while the frame rotates, so a
  mean leaves a colored streak (`value / N`) that no threshold can remove;
  median rejects it as an outlier.
- Frames must be named `IMG_<n>.CR2`; the range is `--start .. --end` and any
  missing number is logged and skipped.
- All products (processed frames, every intermediate stage, final images) go to
  `--out` (default `out/`).

---

## Typical CLI workflows

**1. Straight stack + gentle star reduction (the everyday case)**

```bash
./.venv/Scripts/python.exe scripts/align_stack.py \
    --dir "D:\pic\DCIM-2026\08-shangduhu-2" --method median --out out
```

**2. Stack + full post-processing (flat, neutralize, white balance, star reduction)**

```bash
./.venv/Scripts/python.exe scripts/align_stack.py \
    --dir "D:\pic\DCIM-2026\08-shangduhu-2" --method median \
    --flat --flat-strength 0.5 --neutralize --wb --star-reduce 1 --out out
```

**3. Full workflow + sky/ground composite (all stages, sharp foreground)**

Step 1 — run once to build the ground layer and produce the preview:

```bash
./.venv/Scripts/python.exe scripts/align_stack.py \
    --dir "D:\pic\DCIM-2026\08-shangduhu-2" --method median \
    --flat --flat-strength 0.5 --neutralize --wb --star-reduce 1 \
    --composite --out out
```

Step 2 — have a large vision model look at `out/ground_stack_preview.jpg` and
mark the **ground / foreground** (tree, building, terrain, horizon) as a
grayscale mask with **white = ground, black = sky**, e.g. `out/ground_mask.png`.
(When you use the skill, the agent does this step for you.)

Step 3 — pipeline 2 stitches the two layers (it refuses to run without the
mask; one-shot compositing does not exist):

```bash
./.venv/Scripts/python.exe scripts/composite_ground.py \
    --sky out/sky_final_5994_6012.tif --ground out/ground_stack.tif \
    --mask out/ground_mask.png --out out
```

Check `out/sky_ground_overlay.jpg`: the boundary is drawn in cyan and should hug
the foreground; the sky must be free of star trails. The final image is
`out/merge_final_5994_6012.tif/.jpg`.

> The composite needs a **static foreground**. By default the ground layer is
> stacked from the **brightest 5% of accepted frames** (median; `--ground-method
> mean` keeps more light painting). This avoids the dark, detail-free foreground
> produced when a flashlight or light painting appears in only a few frames.
> Use `--ground-top N` to choose a different count, or `--ground-top -1` to use
> every frame. On a sky-tracked sequence the ground is translation-aligned only
> when the measured motion is significant. With dedicated foreground frames,
> use `--ground-dir` (scenario 6).

**4. Light-pollution gradient only — compare flat strengths first**

```bash
./.venv/Scripts/python.exe scripts/align_stack.py \
    --dir "D:\pic\DCIM-2026\08-shangduhu-2" --method median \
    --flat --flat-sweep 0.4,0.5,0.6,0.7 --out out
```

Look at `out/flat_compare.jpg` (no flat vs each strength), then re-run with the
chosen `--flat-strength` (a full correction of a Milky-Way-filled field can leave
a bright border ring; `~0.5` usually removes it).

**5. Colored hot-pixel streaks**

```bash
./.venv/Scripts/python.exe scripts/align_stack.py \
    --dir "D:\pic\..." --method median --hot-post --out out
```

Prefer `--method median` first; add `--hot-post` for residual colored dots near
the rotation centre, or `--hot-pre` to clean every raw frame at the source
(slower). Compare `out/hot_compare_post.jpg`.

**6. Foreground from a dedicated sequence**

```bash
./.venv/Scripts/python.exe scripts/align_stack.py \
    --dir "D:\pic\...\sky" --method median --composite \
    --ground-dir "D:\pic\...\ground" --ground-start 7001 --ground-end 7020 \
    --out out
```

Add `--ground-mask out/ground_mask.png` here too if you want the boundary to
follow a recognised foreground instead of the automatic detection.

---

## Processing pipeline

The order below is the actual sequence run by `scripts/astro/pipeline.py`.
Optional steps are added to the stage list only when their flag is set, and the
`stageN` numbering follows that order.

1. **Decode + static-pattern map** — half-resolution grayscale previews are
   decoded for every frame; a high percentile across frames isolates the static
   pattern (hot pixels / fixed pattern), which is subtracted before registration.
   If the reference frame correlates with the rest of the sequence (sky-tracked
   rig), the static map is skipped automatically and `--hot-post` is enabled,
   because the map would otherwise delete the stars (or `--no-static` forces it).
2. **Register** — ORB features on the enhanced previews, then a star-centroid
   refinement; the result is a projective homography per frame. Implausible fits
   (low score / inliers or high star residual) are rejected; the frame is skipped.
   Frames are processed in parallel (`--workers`, default one per CPU core).
   With `--hot-pre`, each full-resolution frame is repaired here, before stacking.
3. **Stack** — sky-aligned frames are combined with a **block-wise median**
   (memmap-backed) or a weighted mean → `stage1_stacked`.
4. **Hot-pixel repair** (`--hot-post`) — per-channel detection on the stacked
   master and cosmetic repair → `stage2_hotfix`.
5. **Ground-layer export** (`--composite`) — build a ground layer (tracked
   rigs: per-frame homography over the flashlight-lit frame cluster; fixed
   tripod: median of the **brightest 5% of frames**, `--ground-top` overrides,
   `--ground-method mean` keeps more light painting) and export it as
   `ground_stack.tif` + preview + `sky_curve.json`. Pipeline 1 **never
   composites**: a large vision model must view the preview and draw the
   boundary mask, then pipeline 2 (`composite_ground.py`) grades the ground
   with the sky curve and blends the two layers there.
6. **Auto flat / ABE** (`--flat`) — fit a smooth per-channel 2D background on the
   **linear** image and subtract/divide it, blended by `--flat-strength` →
   `stageN_flat`. Writes `flat_compare.jpg` for a visual decision.
7. **Auto stretch** (always) — adaptive MTF stretch to the target background
   level → `stageN_stretched`.
8. **Background neutralization** (`--neutralize`) — per-channel scaling to a
   neutral sky background → `stageN_neutralized`.
9. **White balance** (`--wb`) — automatic star/sky channel gains (whitepatch /
   grayworld / mixed), blended by `--wb-strength` → `stageN_whitebalanced`;
   writes `wb_compare.jpg` for a visual decision.
10. **Star reduction** (`--star-reduce`, default `1`) — morphology shrink under a
    high-pass star mask → `stageN_star_reduced`.
11. **Deliverables** — every stage is written as `stageN_<name>.tif` +
    `_preview.jpg` and tiled into `stages_montage.jpg`; the finished image is
    written full-resolution as `merge_final_<start>_<end>.tif` and `.jpg`.

---

## Output files

| File | Meaning |
| --- | --- |
| `stageN_<name>.tif` + `_preview.jpg` | Each processing stage (16-bit TIFF + half-size preview); `N` is the stage number. |
| `stages_montage.jpg` | All stages side by side, labelled. |
| `sky_final_<start>_<end>.tif` + `.jpg` | **Pipeline-1 finished sky** at full resolution; the `--sky` input of pipeline 2 (the deliverable when not compositing). |
| `merge_final_<start>_<end>.tif` + `.jpg` | **Sky/ground composite final image** — produced by **pipeline 2** (`composite_ground.py`), never by pipeline 1. |
| `sky_curve.json` | Sky MTF stretch curve (black points, white point, m), written by pipeline 1 so pipeline 2 grades the ground on the same tone scale. |
| `stack_<method>.tif` | Linear master after stacking/composite (for external post). |
| `stack_preview.jpg` | Preview of the linear master. |
| `transforms.json` | Per-frame homography, inliers, score. |
| `timing.json` | Per-stage timings and worker counts for performance profiling. |
| `hot_metrics.json`, `hot_compare_pre.jpg`, `hot_compare_post.jpg` | Hot-pixel audit and before/after figures. |
| `flat_compare.jpg` | No-flat vs flat (and a strength sweep when `--flat-sweep` is used). |
| `wb_compare.jpg` | Before vs after white balance, labelled with the applied gains. |
| `ground_stack.tif` + `_preview.jpg` | Linear ground layer + sky-curve-graded preview (with `--composite`); the preview is what the model must view to draw the mask. |
| `sky_mask.png`, `sky_ground_overlay.jpg` | Composite boundary mask (white = sky, inverted from your mask) and inspection overlay — **pipeline 2** outputs. |
| `star_mask.png` | Star mask (when star reduction runs). |
| `aligned/IMG_<n>.tif` | Each aligned frame (with `--save-frames`). |

---

## Notes & caveats

- **Median first.** Use `--method median` for astro sequences; `mean` is kept
  only for special cases.
- **Flat ring.** A full-strength ABE correction of a Milky-Way-filled field can
  carve the centre and leave a bright border ring — lower `--flat-strength` to
  ~0.3–0.6.
- **Composite requires a static foreground.** On a fixed tripod the ground layer
  comes from the plain (un-registered) selected frames; on a sky-tracked rig it
  is translation-aligned when the measured ground motion is significant.
- **Ground-mask polarity.** `--ground-mask` expects a grayscale image where
  **white = ground/foreground** and black = sky. Edges are feathered by
  `--composite-feather`.
- The default `--dir/--start/--end` point at the example sequence; override them
  for your own data.

---

## Project layout

```text
scripts/align_stack.py     command-line front end (argparse only)
scripts/astro/             implementation, one module per step
  util.py imgio.py         logging/array helpers, CR2 decode / TIFF write
  registration.py          ORB + star-centroid homography registration
  stacking.py              mean / block-wise median accumulation
  hotpixels.py             per-channel hot-pixel detection/repair
  flat.py stretch.py       auto flat (ABE), auto MTF stretch
  neutralize.py            background neutralization
  whitebalance.py          automatic star/sky white balance
  star_reduce.py           star shrink
  composite.py horizon.py  sky/ground stitch + boundary detection
  render.py                per-stage TIFF/JPEG, montage, comparisons
  pipeline.py              orchestration
docs/                      design notes and diagrams
out/                       default output directory (git-ignored)
```
