# AGENTS.md

Guidance for AI coding agents working in this repository.

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

- `scripts/` — all source scripts (e.g. `scripts/align_stack.py`).
  Put new code here; keep the repo root for docs/config only.
- `requirements.txt` — pinned-free dependency list (`numpy`, `opencv-python`, `rawpy`).
- `out/`, `*.tif`, `.venv/` — **not** versioned (see `.gitignore`); never commit
  generated stacks or large binaries.

## Running

```bash
./.venv/Scripts/python.exe scripts/align_stack.py --help
./.venv/Scripts/python.exe scripts/align_stack.py --method median
```

- Prefer the repo venv interpreter for every command.
- Input CR2 frames live outside the repo (e.g. `D:\pic\...`); do not copy them in.
- Outputs go to `--out` (default `out/`), which is git-ignored.

## C++ port

- `cpp/` re-implements `scripts/align_stack.py` (same algorithm, same CLI) with
  no OpenCV dependency; the only external dep is the vendored LibRaw under
  `cpp/third_party/` (LibRaw + its CMake wrapper).
- Build (Windows, MSVC + Ninja from `.venv`):
  ```bat
  cpp\build.bat            :: Release (default)
  cpp\build.bat Debug
  ```
  Produces `output/astro_align.exe`. It is a portable CMake project, so on
  other platforms: `cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release` then
  `cmake --build cpp/build -j`.
- Run (arguments mirror the Python script):
  ```bat
  output\astro_align.exe --dir "D:\pic\DCIM-2026\08-shangduhu-2" --start 5994 --end 6012 --out out_cpp --ref 6003 --method median
  ```
- `output/`, `cpp/build/` are build artifacts (git-ignored); never commit them.

## Conventions

- No comments unless they explain non-obvious *why* (the alignment pipeline has
  several: static-pattern removal, BGR vs RGB on `cv2.imwrite`, etc.).
- `cv2.imwrite` expects **BGR**; convert RGB arrays before writing TIFFs/JPEGs
  (use the `write_tiff16` helper in `scripts/align_stack.py`).
- Keep changes validated against the real frame sequence before committing.
