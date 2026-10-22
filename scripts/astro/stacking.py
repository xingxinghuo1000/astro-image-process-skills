"""Frame accumulation: weighted mean and block-wise median stacking."""
from concurrent.futures import ThreadPoolExecutor
import warnings
import cv2
import numpy as np


def _warp(img, M, W, H, interp):
    if np.asarray(M).shape == (3, 3):
        return cv2.warpPerspective(img, M, (W, H), flags=interp,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return cv2.warpAffine(img, M, (W, H), flags=interp,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def warp_into(lin, M_full, W, H):
    """Warp a full-res linear frame into the reference frame."""
    return _warp(lin, M_full, W, H, cv2.INTER_LINEAR)


def valid_mask(M_full, W, H, ones):
    """Warp a coverage mask so the mean ignores the borders a rotation exposes."""
    return _warp(ones, M_full, W, H, cv2.INTER_NEAREST)


def mean_stack(acc_sum, acc_w):
    return acc_sum / np.maximum(acc_w[..., None], 1)


def make_median_store(path, n, H, W):
    """On-disk uint16 ring used to hold every aligned frame for a median stack
    (a median over the full-res float stack would not fit comfortably in RAM)."""
    return np.memmap(path, dtype=np.uint16, mode="w+", shape=(n, H, W, 3))


def median_stack(store, H, W, step=None, workers=1):
    """Block-wise median over the aligned uint16 frames.

    The median is the expensive step: a full-res 124-frame stack is ~15 GB and
    ``np.nanmedian`` takes ~20 min single-threaded.  Numpy releases the GIL in
    the sort/partition kernels, so independent row bands can run in threads.
    ``step`` is chosen adaptively to keep one band near 1 GB; the caller caps
    ``workers`` so peak RAM stays bounded (each worker also needs a temporary
    copy for the nanmedian; budget roughly 2-3x the band size)."""
    n = store.shape[0]
    if step is None:
        # N * rows * W * 3 channels * float32 bytes -> rows for ~1 GB.
        step = int(1_000_000_000 / max(1, n * W * 3 * 4))
        step = max(16, min(256, step))
    bands = [(r0, min(H, r0 + step)) for r0 in range(0, H, step)]
    stack = np.empty((H, W, 3), np.float32)

    def one(band):
        r0, r1 = band
        block = store[:, r0:r1, :, :].astype(np.float32) / 65535.0
        # Warped frames leave exact zeros outside their footprint.  A plain
        # median would turn that uncovered border black even though the
        # reference frame covers it; ignore the invalid pixels instead.
        # NaN-aware median is ~5x slower than plain median, so first take the
        # fast median (zeros included) and recompute only the pixels that
        # actually have an uncovered sample.  On a tracked sequence that is a
        # thin border, not the whole frame.
        med = np.median(block, axis=0)
        bad = block <= 0
        invalid = bad.any(axis=0)
        if invalid.any():
            n = block.shape[0]
            flat = block.reshape(n, -1)
            badflat = bad.reshape(n, -1)
            idx = np.flatnonzero(invalid.reshape(-1))
            sub = flat[:, idx]
            sub[badflat[:, idx]] = np.nan
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                med.flat[idx] = np.nanmedian(sub, axis=0)
        stack[r0:r1] = med

    workers = max(1, min(int(workers), len(bands)))
    if workers == 1:
        for band in bands:
            one(band)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(one, bands))
    return np.nan_to_num(stack, nan=0.0)


def mean_ground(frames_acc, count):
    """Plain (un-registered) average of the frames -> sharp static foreground,
    trailed sky.  Used as the 'ground' layer of the sky/ground composite."""
    return frames_acc / max(count, 1)
