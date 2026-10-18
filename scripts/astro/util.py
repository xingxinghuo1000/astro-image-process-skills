"""Small shared helpers: logging and array <-> image conversions."""
import numpy as np


def log(*a):
    print(*a, flush=True)


def clip01(x):
    return np.clip(x, 0.0, 1.0)


def to_u16(rgb):
    """float [0,1] RGB -> little-endian uint16 (0..65535)."""
    return np.clip(rgb * 65535.0, 0, 65535).astype(np.uint16)


def to_u8(img):
    """float [0,1] image -> uint8 (0..255), rounded."""
    return (np.clip(img, 0, 1) * 255.0 + 0.5).astype(np.uint8)
