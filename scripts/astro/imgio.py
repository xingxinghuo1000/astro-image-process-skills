"""Frame discovery, CR2 decoding and TIFF writing."""
import os
import cv2
import numpy as np
import rawpy

from .util import log


def find_files(d, start, end):
    """Existing IMG_<n>.CR2 paths for n in [start, end]; missing ones are logged."""
    names = []
    for i in range(start, end + 1):
        n = f"IMG_{i}.CR2"
        p = os.path.join(d, n)
        if os.path.exists(p):
            names.append((i, p))
        else:
            log(f"  (!) {n} not found, skipped")
    return names


def load_linear(path):
    """Decode CR2 -> float32 linear RGB in [0,1] (camera WB, no tone curve)."""
    raw = rawpy.imread(path)
    rgb16 = raw.postprocess(use_camera_wb=True, no_auto_bright=True,
                            output_bps=16, gamma=(1, 1),
                            output_color=rawpy.ColorSpace.sRGB,
                            demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
                            half_size=False)
    return rgb16.astype(np.float32) / 65535.0


def load_gray8(path):
    """Decode a half-size 8-bit luminance image, used for registration."""
    raw = rawpy.imread(path)
    rgb = raw.postprocess(half_size=True, use_camera_wb=True,
                          no_auto_bright=True, output_bps=8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)


def write_tiff16(path, rgb_uint16):
    """Write a TIFF from an RGB (not BGR) array.  cv2.imwrite expects BGR, so
    convert first -- otherwise the red and blue channels end up swapped."""
    cv2.imwrite(path, cv2.cvtColor(rgb_uint16, cv2.COLOR_RGB2BGR))
