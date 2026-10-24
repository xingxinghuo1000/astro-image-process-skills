"""Small shared helpers: logging, CPU/memory-aware worker sizing, array conversions."""
import os
import numpy as np


def log(*a):
    print(*a, flush=True)


def cpu_count():
    """Usable logical CPUs (honours CPU affinity where the OS exposes it)."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def available_memory_gb():
    """Best-effort available physical RAM in GB; None if the probe fails."""
    try:
        if os.name == "nt":
            import ctypes

            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _MemoryStatusEx()
            status.dwLength = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return status.ullAvailPhys / 1e9
        else:
            pages = os.sysconf("SC_AVPHYS_PAGES")
            size = os.sysconf("SC_PAGE_SIZE")
            return pages * size / 1e9
    except (AttributeError, OSError, ValueError):
        pass
    return None


def auto_workers(kind, n_items):
    """Pick a worker count from CPU count and available RAM.

    ``kind`` is one of ``registration``, ``io``, ``stack`` or ``render``.
    The caps keep the pipeline from oversubscribing small machines or holding
    too many decoded frames / median bands in memory.
    """
    n_items = max(1, int(n_items))
    cpus = cpu_count()
    mem = available_memory_gb()

    if kind == "registration":
        return max(1, min(n_items, cpus))

    if kind == "io":
        # Cap 8, not 16: at 8 workers decoding already outruns the stacking
        # loop ~4x, while 16 doubles the in-flight queue (~8 GB) and gets the
        # run killed by the OS under memory pressure with no measured gain.
        workers = min(n_items, 8, max(2, cpus // 2))
        if mem is not None:
            workers = min(workers, max(1, int(mem * 0.25 / 0.24)))
        return max(1, workers)

    if kind == "stack":
        workers = min(n_items, cpus)
        if mem is not None:
            # 16 MB bands with ~3x temporary space per worker.
            workers = min(workers, max(1, int(mem * 0.5e9 / (16e6 * 3))))
        return max(1, workers)

    if kind == "render":
        return max(1, min(n_items, 8, max(2, cpus // 2)))

    raise ValueError(f"unknown worker kind: {kind}")


def clip01(x):
    return np.clip(x, 0.0, 1.0)


def to_u16(rgb):
    """float [0,1] RGB -> little-endian uint16 (0..65535)."""
    return np.clip(rgb * 65535.0, 0, 65535).astype(np.uint16)


def to_u8(img):
    """float [0,1] image -> uint8 (0..255), rounded."""
    return (np.clip(img, 0, 1) * 255.0 + 0.5).astype(np.uint8)
