# AGENTS.md

Guidance for AI coding agents working in this repository.

## Development policy

- **Python only.** `scripts/align_stack.py` is a thin command-line front end; the
  implementation lives in the `scripts/astro/` package (one module per pipeline
  step). Add features and fix bugs in the relevant module.
- A C++ port briefly existed and was removed (see `docs/python-vs-cpp-report.md`
  for the benchmark/decision rationale: functionally equivalent, performance a
  wash, Python far cheaper to build, maintain and extend). Do not reintroduce one
  unless explicitly asked.

## Environment

- **Always use the virtual environment at the repository root: `.venv/`.**
  Do not install packages into the system/global Python and do not create a
  second virtualenv.
- On Windows (this repo's primary platform) run Python via:
  `./.venv/Scripts/python.exe`
  On POSIX: `./.venv/bin/python`
- If `.venv/` is missing, create it and install dependencies:
  ```bash
  python -m venv .venv
  ./.venv/Scripts/python.exe -m pip install -r requirements.txt \
      -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
  ```
  (The Aliyun mirror is much faster in mainland China.)

## Layout

- `scripts/align_stack.py` — the CLI/argparse entry point (kept thin); it just
  parses arguments and calls `astro.pipeline.run`.
- `scripts/astro/` — the implementation, one module per pipeline step:
  - `util.py` logging + array helpers; `imgio.py` CR2 decode / TIFF write
  - `registration.py` ORB + star-centroid similarity registration
  - `stacking.py` mean / block-wise median accumulation
  - `flat.py` auto flat (ABE); `stretch.py` auto MTF; `neutralize.py` background
    neutralization; `star_reduce.py` star shrink
  - `composite.py` sky/ground ("天地") stitch; `horizon.py` sky-ground boundary
    detection
  - `render.py` per-stage TIFF/JPEG, montage, comparison figures
  - `pipeline.py` the orchestration that sequences the steps
- `requirements.txt` — pinned-free dependency list (`numpy`, `opencv-python`, `rawpy`).
- `out/` — **the default output directory** for everything the pipeline produces
  (processed frames, every intermediate stage, and the final images). It is
  **not** versioned (see `.gitignore`); never commit generated stacks or large
  binaries.

## Running

```bash
./.venv/Scripts/python.exe scripts/align_stack.py --help
./.venv/Scripts/python.exe scripts/align_stack.py --method median
```

- Prefer the repo venv interpreter for every command.
- Input CR2 frames live outside the repo (e.g. `D:\pic\...`); do not copy them in.
- **Processed images, intermediate products, and final products all default to the
  `out/` directory** (`--out`, default `out/`). `out/` is git-ignored.

## Conventions

- No comments unless they explain non-obvious *why* (the alignment pipeline has
  several: static-pattern removal, BGR vs RGB on `cv2.imwrite`, etc.).
- `cv2.imwrite` expects **BGR**; convert RGB arrays before writing TIFFs/JPEGs
  (use the `write_tiff16` helper in `scripts/align_stack.py`).
- Keep changes validated against the real frame sequence before committing.
