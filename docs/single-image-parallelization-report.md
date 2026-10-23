# Single-image parallelization report

**Language:** English | [简体中文](single-image-parallelization-report-cn.md)

> Executive summary: this work had two performance passes. The first moved the complete
> **124-frame, 5496×3670** workflow from “serial per frame / per channel” to **frame-level
> pipelining plus intra-image parallelism**, reaching **362.7 s** end-to-end (about
> **2.9 s/frame**; the old 160-frame baseline was about **17.4 s/frame**, roughly **6×**
> normalized). The second pass targeted median stacking with **smaller bands + higher
> concurrency**: the band target went from ~1 GB to 16 MB and workers from 16 to the CPU
> count, cutting median stacking from **115.8 s to 55.3 s** and stack peak working set
> from **~63 GB to ~18 GB**. The full pipeline measured **343.9 s** (total time is
> disk-sensitive; treat it as indicative). Final TIFF md5 values are identical across
> both passes.

![Parallelization overview](images/parallelization-map.png)

*Figure 1: Parallel work distribution and memory bounds.*

---

## 1. Scope and test data

| Item | Value |
| --- | --- |
| Input | `D:\pic\DCIM-2023\0812-liuxingyu\02\serial`, `IMG_4744..4869` |
| Frames | 124 (`4746/4747` missing) |
| Resolution | 5496×3670 full-res linear RGB; registration at 2748×1835 grayscale |
| Reference | `IMG_4780` |
| Scene | Sky-tracked sequence; high-pass correlation 0.89, static-pattern subtraction skipped automatically |
| Composite | `--ground-mask`; ground occupies 24.93% of the half-res frame |
| Stacking | `--method median`; all 124/124 frames passed registration validation |
| Hardware | 72 logical cores, 137 GB RAM, Windows |
| Output | `out_4744_4869/` |
| Final TIFF | `merge_final_4744_4869.tif` |

> “Parallelization” here has two distinct layers:
>
> - **Frame-level parallelism / pipelining**: different CR2 decodes and different
>   frame registrations can run at the same time.
> - **Intra-image parallelism**: within one image, work is split by channel, row band
>   or output panel.
>
> The first controls throughput; the second addresses “one large image is itself slow.”
> They should not be conflated.

---

## 2. Design principles

1. **Only parallelize independent work units.** Median row bands, RGB channels and
   separate output files have no write conflicts. That is the precondition for threads.
2. **Prefer threads over processes.** NumPy’s sort/partition kernels and OpenCV filters
   release the GIL; threads share `numpy` arrays and the memmap. A process pool would
   copy or re-map the 15 GB intermediate.
3. **Bound in-flight data.** The prefetch queue holds about `2N` frames; each median row
   band targets 16 MB (one row minimum). This prevents 72 full-image tasks from
   exhausting RAM.
4. **Keep output deterministic.** Row bands write disjoint row ranges; channels write
   disjoint channel slices; stage files are independent. Thread completion order does
   not affect the final pixels.
5. **Preserve numerical correctness before parallelizing.** Fully covered pixels still
   use `np.median`; only invalid warp borders use `np.nanmedian`.

---

## 3. Where the parallelism lives

See the overview figure at the top. The table below breaks it down:

| Location | Parallel unit | Default concurrency | Sharing / synchronization |
| --- | --- | --- | --- |
| Grayscale preview decode | One task per frame | 8 | Thread pool; results collected in frame order |
| Registration | One task per frame | Up to 72 (capped by frame count) | `ThreadPoolExecutor`; independent ORB per task |
| Full-resolution decode | One prefetch task per frame | 8 | Bounded future queue, about `2N` frames |
| Median stacking | Output row band | 8 | Each band reads the memmap and writes its own rows |
| Hot-pixel repair | R/G/B channels | 3 | Each channel writes its own slice |
| Auto flat | R/G/B channels | 3 | Each channel fits and writes independently |
| Stage TIFF/JPEG | One task per stage | 4 | Each task writes its own file |
| Stage comparison panels | One task per panel | 4 | Independent panels; assembled afterwards |

**Key point:** intra-image parallelism does not copy the whole image into multiple
processes. Each thread handles one channel or one row band, which uses multiple cores
while keeping peak memory bounded.

---

## 4. Intra-image changes (core)

### 4.1 Median stack: serial 128-row blocks to adaptive small bands

Before, `stacking.py` used a fixed `step=128` serial loop:

```python
# Before: scripts/astro/stacking.py
def median_stack(store, H, W, step=128):
    stack = np.empty((H, W, 3), np.float32)
    for r0 in range(0, H, step):
        r1 = min(H, r0 + step)
        block = store[:, r0:r1, :, :].astype(np.float32) / 65535.0
        stack[r0:r1] = np.median(block, axis=0)
    return stack
```

After, the band size targets 16 MB (one row minimum), bands run in a thread pool, and a
fast-median path avoids a full-frame NaN-aware pass:

```python
# After: scripts/astro/stacking.py
def median_stack(store, H, W, step=None, workers=1, band_mb=16):
    n = store.shape[0]
    if step is None:
        target = max(1.0, float(band_mb)) * 1e6
        step = int(target / max(1, n * W * 3 * 4))
        step = max(1, min(256, step))
    bands = [(r0, min(H, r0 + step)) for r0 in range(0, H, step)]
    stack = np.empty((H, W, 3), np.float32)

    def one(band):
        r0, r1 = band
        block = store[:, r0:r1, :, :].astype(np.float32) / 65535.0
        med = np.median(block, axis=0)
        bad = block <= 0
        invalid = bad.any(axis=0)
        if invalid.any():
            flat = block.reshape(n, -1)
            badflat = bad.reshape(n, -1)
            idx = np.flatnonzero(invalid.reshape(-1))
            sub = flat[:, idx]
            sub[badflat[:, idx]] = np.nan
            med.flat[idx] = np.nanmedian(sub, axis=0)
        stack[r0:r1] = med

    workers = max(1, min(int(workers), len(bands)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, bands))
    return np.nan_to_num(stack, nan=0.0)
```

#### Algorithmic differences

| Aspect | Before | After |
| --- | --- | --- |
| Parallel unit | None; fixed 128-row serial | Adaptive row bands, CPU-count threads |
| Band size | Fixed 128 rows | About 16 MB per band from `band_mb` (one row minimum) |
| Invalid border | `0` treated as a real pixel | Fast median first; only pixels containing `0` use `nanmedian` |
| Fully covered pixels | `np.median` | Still `np.median`, pixel-for-pixel identical |
| Border result | Can be pulled black by zeros | Uses only valid frames; no black border |
| Memory | One 128-row float32 block | `workers` × ~8–16 MB blocks plus small border temporaries |

> Why not use `nanmedian` for the whole frame? NaN-aware median has to check masks and
> partition NaNs for every pixel and is much slower. On this kind of sequence, the vast
> majority of pixels are covered by all 124 frames; only a narrow warped border lacks
> samples. “Compute the fast median everywhere, then repair only the invalid border” is
> the best trade-off.

For fully covered pixels, `np.median` is identical to the old implementation. For border
pixels, the new path does not treat `0` as data; it uses only frames that actually cover
the pixel. The final TIFF md5 is `aebb14498eb2845c940e71e67cd53687`, matching the
full-frame `nanmedian` reference path.

![Median row bands](images/median-bands.png)

*Figure 2: Median row-band parallelism and the fast-median + border-nanmedian path.*

### 4.2 Hot pixels: per-channel parallelism + `dilate` instead of eight rolls

Before, `hotpixels.py` computed residuals for three channels serially, detected/repaired
serially, and computed the 8-neighbour maximum with eight `np.roll` calls (wrapping at
the border).

```python
# Before: scripts/astro/hotpixels.py
def _neighbor_max(a):
    m = np.full_like(a, -1.0)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            m = np.maximum(m, np.roll(np.roll(a, dy, 0), dx, 1))
    return m

for c in range(3):
    resid[..., c] = ...
for c in range(3):
    ...
```

After, the three channels are independent and run in parallel; the neighbour maximum is
one `cv2.dilate` with a zero-centre 3×3 kernel:

```python
# After: scripts/astro/hotpixels.py
_NEIGH_KERNEL = np.ones((3, 3), np.uint8)
_NEIGH_KERNEL[1, 1] = 0

def _neighbor_max(a):
    return cv2.dilate(a.astype(np.float32), _NEIGH_KERNEL)

def one(c):
    ...
    cand &= r > neigh_ratio * np.maximum(_neighbor_max(r), 0.0)
    ...
```

#### Algorithmic differences

- **Parallel layer:** R/G/B are independent tasks, 3 threads by default.
- **Neighbour implementation:** eight whole-image `np.roll` operations become one
  `cv2.dilate`; interior “max of the 8 neighbours” semantics are unchanged.
- **Border semantics:** `np.roll` wraps; `dilate` does not. The outer image ring no
  longer treats a bright pixel on the opposite edge as a neighbour, which is more
  sensible.
- **Memory:** explicit `(c+1)%3, (c+2)%3` indexing replaces `np.delete`, avoiding a
  whole-image copy.
- **Detection criteria unchanged:** MAD threshold + absolute threshold + 8-neighbour
  local maximum + channel dominance. The test frame has 8233 hot pixels
  (`hot_metrics.json`).

### 4.3 Auto flat: independent fits per colour channel

The polynomial background fit is already independent per channel. Before, it was a
serial `for c in range(3)`; after, `one(c)` runs in a thread pool:

```python
# After: scripts/astro/flat.py (structure)
def one(c):
    ...  # tile percentile -> least squares -> 2-D polynomial -> out[..., c]
    return c, (model.min(), model.max(), model.mean())

workers = max(1, min(int(workers), 3))
if workers == 1:
    results = (one(c) for c in range(3))
else:
    ex = ThreadPoolExecutor(max_workers=3)
    results = ex.map(one, range(3))
```

This is an **execution change, not an algorithm change**: the grid, percentile, low-order
polynomial, pedestal and strength blend are unchanged. The measured output matches the
serial per-channel version.

### 4.4 Stage images: parallel per stage / panel

Before, each stage was written by a serial `write_stage`, and montage/flat-comparison
panels were built serially. After:

- `pipeline.py` writes stage TIFF + JPEG files in parallel with `render_workers`;
- `render.write_montage` builds stage panels in parallel;
- `render.write_flat_compare` builds no-flat / strength-sweep preview panels in parallel.

Each stage file is separate, so there is no shared write. Montage panels are still
assembled in a fixed order, so the output remains deterministic.

---

## 5. Frame-level and I/O concurrency (supporting work)

### 5.1 Registration: one task per frame

`_register_task` packages the per-frame “high-pass → ORB → match → RANSAC → star-centroid
refinement → quality gate” into a pure task. The main thread dispatches frames through a
thread pool:

```python
# After: scripts/astro/pipeline.py
tasks = [(k, idx, cleaned[k], ...) for k, (idx, _) in enumerate(files)]
prev_threads = cv2.getNumThreads()
if workers > 1:
    cv2.setNumThreads(1)
try:
    with ThreadPoolExecutor(max_workers=workers) as ex:
        reg_results = list(ex.map(_register_task, tasks))
finally:
    cv2.setNumThreads(prev_threads)
```

OpenCV’s internal thread count is deliberately set to 1: the outer pool already has up to
72 frame tasks, and letting ORB/filtering start another layer of OpenCV threads would
cause severe oversubscription. The previous value is restored afterwards.

### 5.2 Full-resolution decode: bounded prefetch

Before, each frame called `load_linear` synchronously inside the loop, so the CPU waited
on disk while decoding CR2. After, the main loop consumes futures while background
threads decode later frames:

```python
# After: scripts/astro/pipeline.py (structure)
pending = {}
io_limit = max(1, io_workers * 2)

def submit_frames():
    while next_submit < len(files) and len(pending) < io_limit:
        pending[next_submit] = io_pool.submit(imgio.load_linear, files[next_submit][1])
        next_submit += 1

def take_frame(k):
    return pending.pop(k).result()
```

The queue is capped at about `2×io_workers` frames. At roughly 240 MB per full-res RGB
float32 frame, eight workers hold about 4 GB in flight; memory does not grow linearly with
the 124-frame sequence. Prefetch changes only *when* decoding happens, not *what* is
decoded, so it introduces no numerical difference.

---

## 6. Performance

### 6.1 End-to-end

| Run | Frames | Total | Per frame |
| --- | ---: | ---: | ---: |
| Old baseline | 160 | 2785 s | 17.4 s |
| First pass: frame + intra-image parallelism | 124 | 362.7 s | 2.9 s |
| Second pass: small bands + high concurrency | 124 | 343.9 s | 2.8 s |

> The frame counts differ, so total times are not a direct A/B comparison; the table uses
> **per-frame normalization** to show the order of magnitude. Absolute numbers depend on
> disk cache, CPU frequency and whether CR2 files are decoded for the first time. The
> second pass cut total time by only ~5% because its frame loop rose from 95.6 s to
> 134.3 s (disk/IO variance); the reproducible gains are median stacking and peak memory,
> covered in 6.5.

The first-pass `timing.json` wall time is 427.6 s. The difference from `total` = 362.7 s
is about 65 s, mainly Windows memmap deletion/flush during shutdown.

### 6.2 Stage breakdown

![Stage timings](images/stage-timings.png)

*Figure 3: Stage timing breakdown for the first-pass 124-frame run.*

| Stage | Seconds | Share of 362.7 s (pass 1) |
| --- | ---: | ---: |
| Median stacking | 115.8 | 31.9% |
| Frame loop (decode wait + warp + accumulate) | 95.6 | 26.4% |
| Registration | 29.4 | 8.1% |
| Grayscale preview decode | 27.0 | 7.4% |
| Stage TIFF/JPEG writes | 16.7 | 4.6% |
| hotfix (including comparison figures) | 14.5 | 4.0% |
| Flat comparison panels | 14.1 | 3.9% |
| Sky/ground composite | 12.7 | 3.5% |
| Final TIFF/JPEG writes | 12.1 | 3.3% |
| Other (stretch/neutralize/WB/star reduce/montage/audit) | 24.8 | 6.8% |

Note: these are **sequential stage timings**, not overlapping wall-clock phases.
`frame_loop` also includes waiting for prefetched futures. See 6.5 for the second-pass stage comparison.

### 6.3 Representative before/after

| Item | Before (profiling/baseline) | After (pass 1 / pass 2) | Note |
| --- | ---: | ---: | --- |
| Median stack (124 frames, pass 1) | 228.6 s | 115.8 s | Eight row bands + fast median |
| Median stack (124 frames, pass 2) | 115.8 s | 55.3 s | 16 MB bands + 72 workers |
| Hot-pixel core | 19.6 s | 3.8 s | Three channels + `dilate` |
| Auto flat | 8.8 s | 3.3 s | Three channels |
| Stage image writes | ~44 s | 16.7 s | Four render workers |
| Grayscale preview decode | ~171 s | 27.0 s | Eight I/O workers |
| End-to-end | 17.4 s/frame | 2.9 s/frame | Different frame counts; per-frame normalized |

> The “before” numbers come from profiling/microbenchmarks during this optimization pass,
> not from a side-by-side run of the old and new code under the same wall clock. They are
> useful for locating bottlenecks and judging magnitude, not as a formal benchmark. The second-pass full run is in 6.5.

### 6.4 Thread count is not the only knob

- **Median bands:** the default band target is 16 MB, with workers chosen from CPU count
  and RAM; on this 72-core machine with 69 GB available it picks 72. Smaller bands cut each worker's
  working set from ~1 GB to ~8 MB and expose more parallel tasks. On the same 124-frame
  aligned data: 16×1 GB took 115.8 s, 16×16 MB took 47.9 s, and 72×16 MB took 43.1 s;
  peak working set fell from ~63 GB to ~18 GB, with bit-identical output. Smaller bands
  add task/sort overhead, so 16 MB (one row per band for this sequence) is the compromise.
- **CR2 decode:** auto mode scales with CPU and RAM up to 16; on this machine 16 workers
  decoded 124 frames in ~16 s versus ~27–32 s with 8. More workers only increase
  in-flight memory.
- **Stage writes:** auto mode scales with CPU up to 8; disk writes and JPEG encoding
  limit the benefit, and 8 workers cover most of the wait.
- **Registration:** 72 frame tasks already saturate the CPU; more threads only add
  context switching.

---

### 6.5 Second performance pass: small bands + high concurrency

The first pass used ~1 GB median bands with a 16-worker cap. That combination had two
problems:

- Each worker's `block` is `N × band_rows × W × 3 × 4`; a 1 GB band means one worker holds
  all 124 frames for a full slab. Sixteen workers already hold more than ten GB of float32
  temporaries.
- Adding workers without shrinking the band increases memory linearly. In the first pass,
  24/32 workers with 1 GB bands gave no further speedup, and 24 concurrent bands once hit
  an allocation failure during repeated benchmark runs.
- Large bands produce few tasks, so tail tasks also lengthen the wall time.

The second pass changes both knobs together:

1. **Shrink the band.** `--stack-band-mb` defaults to a 16 MB target, and the `step` floor
   changed from 16 rows to 1 row. For 124 frames the 16 MB target resolves to **one row
   per band** (~8.2 MB).
2. **Raise concurrency.** `--stack-workers` now defaults to the CPU count; this 72-core
   machine selects 72. `median_stack` also caps workers from available RAM (about 50% of
   available memory, assuming 3x the band size per worker) so large frame counts do not
   OOM.
3. **Keep the numerical path unchanged.** Fully covered pixels still use plain
   `np.median`; only warp-border invalid pixels use `np.nanmedian`. Smaller bands change
   task partitioning, not pixel semantics.

#### Test method

- Same 124-frame real aligned data: 5496×3670, uint16 memmap (~15 GB).
- Each configuration ran in a separate process to avoid allocator-retention effects from
  the previous run.
- Peak working set was read with Windows `GetProcessMemoryInfo`. It includes memmap page
  cache, so absolute values exceed the Python heap, but relative comparisons are useful.
- All configurations produced identical checksums; `16 × 16 MB` was also compared
  bit-for-bit against `stage1_stacked.tif`.

#### Results

| Configuration | Median time | Peak working set* |
| --- | ---: | ---: |
| 16 workers × 1 GB band | 115.8 s | ~62.6 GB |
| 8 × 16 MB | 80.6 s | ~15.8 GB |
| 16 × 16 MB | 47.9 s | ~16.0 GB |
| 24 × 16 MB | 45.0 s | ~16.4 GB |
| 48 × 16 MB | 43.7 s | ~17.6 GB |
| **72 × 16 MB** | **43.1 s** | **~18.1 GB** |
| 64 × 32 MB | 44.3 s | ~24.7 GB |
| 64 × 64 MB | 52.1 s | ~36.1 GB |
| 32 × 128 MB | 56.2 s | ~36.6 GB |

* Includes memmap page cache, not just the Python heap.

Observations:

- **16 MB / one row is the sweet spot:** about **2.7×** faster than the 1 GB band and
  about **71%** lower peak working set.
- **Smaller is not always better:** 32/64 MB bands were slower or used more memory than
  16 MB; one row is already the minimum granularity.
- **72 workers is best; 48 is close:** 72×16 MB took 43.1 s and 48×16 MB took 43.7 s. On
  an 8-core machine the auto rule selects 8×16 MB, which still took 80.6 s versus
  108.3 s for 8×1 GB.
- The gain comes from better cache locality, smaller `bad`/`nanmedian` temporaries, and
  more parallel tasks. The task count rises from 31 to 3670, so Python/thread scheduling
  overhead increases, but the parallel gain dominates at this resolution.

#### Full-pipeline check

Re-running the same 124-frame workflow with the second-pass defaults:

```bash
./.venv/Scripts/python.exe scripts/align_stack.py \
    --dir "D:\pic\DCIM-2023\0812-liuxingyu\02\serial" \
    --start 4744 --end 4869 --ref 4780 --method median \
    --hot-post --flat --flat-strength 0.5 --flat-sweep 0.4,0.5,0.6,0.7 \
    --neutralize --wb --star-reduce 1 \
    --composite --ground-mask out_4744_4869/ground_mask.png \
    --out out_4744_4869
```

| Item | First pass | Second pass |
| --- | ---: | ---: |
| Median stacking | 115.8 s | 55.3 s |
| Grayscale preview decode | 27.0 s | 16.8 s |
| Registration | 29.4 s | 35.2 s |
| Frame loop | 95.6 s | 134.3 s |
| Stage writes | 16.7 s | 16.5 s |
| Total | 362.7 s | 343.9 s |

Final products:

- `stack_median.tif` md5: `cdf95e2881aef92c34e87d10d31b0b5b` (identical in both passes)
- `merge_final_4744_4869.tif` md5: `aebb14498eb2845c940e71e67cd53687` (identical)

Total time fell only ~5% because the frame loop grew from 95.6 s to 134.3 s due to
disk/IO variance. Median stacking itself fell from 115.8 s to 55.3 s; that is the stable,
reproducible gain from the second pass.

#### Conclusion and limits

- Final defaults: `--stack-band-mb 0` → 16 MB; `--stack-workers 0` → CPU count (RAM-capped).
- This is still an **exact median**, not an approximate quantile.
- For very large frame counts, the one-row `step` floor means a band grows with N again;
  the RAM cap then reduces workers automatically. To keep memory strictly O(1) in N for
  any N, the next step would be a multi-pass radix/histogram select, which is not the
  default path in this pass.

---

## 7. Correctness checks

| Check | Result |
| --- | --- |
| Registration | 124/124 |
| Final TIFF md5 | `aebb14498eb2845c940e71e67cd53687` |
| Stack md5 | `cdf95e2881aef92c34e87d10d31b0b5b` |
| Fast median vs full-frame `nanmedian` reference | Final TIFF identical |
| Hot-pixel test frame | 8233 hot pixels (`hot_metrics.json`) |
| Auto flat | Matches serial per-channel version |
| Smoke test | 4744–4760 completed end-to-end (composite/flat/WB/star reduction) |

The outputs and stage images in `out_4744_4869/` were reviewed manually: composite,
flat comparison, white-balance comparison and stage montage all look correct.

---

## 8. Tuning and memory estimate

```bash
./.venv/Scripts/python.exe scripts/align_stack.py \
    --dir "D:\pic\DCIM-2023\0812-liuxingyu\02\serial" \
    --start 4744 --end 4869 --ref 4780 --method median \
    --hot-post --flat --flat-strength 0.5 --flat-sweep 0.4,0.5,0.6,0.7 \
    --neutralize --wb --star-reduce 1 \
    --composite --ground-mask out_4744_4869/ground_mask.png \
    --io-workers 8 --stack-workers 8 --render-workers 4 --workers 72 \
    --out out_4744_4869
```

> The command above is the historical benchmark and pins 8/8/4; the current defaults
> auto-select from CPU count and available RAM.

| Option | Default / recommendation | Effect |
| --- | --- | --- |
| `--workers` | 0 = up to CPU count/frame count | Per-frame registration threads |
| `--io-workers` | 0 = auto (cap 16) | Preview/full-res decode threads and prefetch queue |
| `--stack-workers` | 0 = auto (CPU count, RAM-capped) | Median row-band threads |
| `--stack-band-mb` | 0 = 16 MB | Target working set per median row band |
| `--render-workers` | 0 = auto (cap 8) | Parallel stage TIFF/JPEG writes |

Rough memory model:

```text
Full-res aligned stack (uint16 memmap) ≈ N × H × W × 3 × 2 B
                                      = 124 × 3670 × 5496 × 3 × 2
                                      ≈ 15.0 GB (on disk, not required to fit in RAM)

Median band (target, float32)         ≈ 16 MB
Median budget                         ≈ 3 × stack_workers × 16 MB
Decode prefetch queue                 ≈ io_workers × 2 × 240 MB
```

If memory is tight, reduce `--stack-band-mb` first, then `--stack-workers` and
`--io-workers`. Do not reduce `--workers` to save memory: registration mainly holds
half-resolution grayscale images.

---

## 9. Limits and next steps

- **Median is still the largest single item:** 55.3 s for median stacking in the tuned
  full run, 43.1 s in the microbenchmark; close to the cost of reading the
  memmap and sorting. Further gains require less data movement (for example, a different
  storage/layout scheme), not just more threads.
- **The frame loop is still assembled single-threaded:** warp and accumulation happen in
  frame order; prefetch hides only part of the wait. A bounded warp future is possible,
  but must respect memory and output order.
- **Python/NumPy boundary:** this design relies on NumPy/OpenCV releasing the GIL. Not
  every third-party function is thread-safe; new steps must check this first.
- **No process pool or GPU:** a process pool would multiply the cost of 15 GB arrays; a
  GPU median/channel-filter implementation is a separate cost/benefit decision.
- **“More threads” is not free:** 72 cores do not imply 72 full-image median tasks. Here
  the bands are cut to ~16 MB (one row per band for this sequence), so 72 workers each hold
  only a small slice; peak memory is far below 72 full-image tasks.
- **Strict O(1)-in-N memory:** the current path still reads N frames per band. For much
  larger N, a 4-bit radix select (four passes, memory independent of N) would be the next
  step, at the cost of extra I/O.

---

## 10. Related files

| File | Change |
| --- | --- |
| `scripts/astro/stacking.py` | Adaptive bands (`band_mb`), thread pool, RAM cap, fast median + border `nanmedian` |
| `scripts/astro/hotpixels.py` | Per-channel parallelism, `cv2.dilate` neighbours, fewer copies |
| `scripts/astro/flat.py` | Per-channel parallelism |
| `scripts/astro/pipeline.py` | Registration pool, CR2 prefetch, parallel stage writes, `timing.json` |
| `scripts/astro/render.py` | Parallel montage / flat-comparison panels |
| `scripts/astro/registration.py` | Frame tasks; homography fitting and quality gate |
| `scripts/astro/util.py` | CPU / available-memory probes, automatic worker selection |
| `scripts/align_stack.py` | `--io-workers`, `--stack-workers`, `--stack-band-mb`, `--render-workers` |

For full CLI usage see [`README-cn.md`](README-cn.md) (Chinese) or the root
[`README.md`](../README.md) (English). For the geometry behind registration see
[`wide-field-registration.md`](wide-field-registration.md).
