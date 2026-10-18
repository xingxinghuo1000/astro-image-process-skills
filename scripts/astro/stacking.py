"""Frame accumulation: weighted mean and block-wise median stacking."""
import cv2
import numpy as np


def warp_into(lin, M_full, W, H):
    """Warp a full-res linear frame into the reference frame."""
    return cv2.warpAffine(lin, M_full, (W, H), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def valid_mask(M_full, W, H, ones):
    """Warp a coverage mask so the mean ignores the borders a rotation exposes."""
    return cv2.warpAffine(ones, M_full, (W, H), flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def mean_stack(acc_sum, acc_w):
    return acc_sum / np.maximum(acc_w[..., None], 1)


def make_median_store(path, n, H, W):
    """On-disk uint16 ring used to hold every aligned frame for a median stack
    (a median over the full-res float stack would not fit comfortably in RAM)."""
    return np.memmap(path, dtype=np.uint16, mode="w+", shape=(n, H, W, 3))


def median_stack(store, H, W, step=128):
    stack = np.empty((H, W, 3), np.float32)
    for r0 in range(0, H, step):
        r1 = min(H, r0 + step)
        block = store[:, r0:r1, :, :].astype(np.float32) / 65535.0
        stack[r0:r1] = np.median(block, axis=0)
    return stack


def mean_ground(frames_acc, count):
    """Plain (un-registered) average of the frames -> sharp static foreground,
    trailed sky.  Used as the 'ground' layer of the sky/ground composite."""
    return frames_acc / max(count, 1)
