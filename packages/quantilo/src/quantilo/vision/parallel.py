"""R-VIS-ING-07: how many processes read an image set at the import (study § 10, c3).

Decoding, measuring and describing an image is CPU work that one Python process does at about
75 ms per 500 x 500 image; four processes divide that. Memory bounds them: a 100-megapixel image
(the R-IMG-06 limit) decoded costs about 0.4 GiB, so each process is given 1 GiB and 2 GiB stay
free for the rest of the machine. The browser version (Pyodide: no process) and small sets
(starting the processes costs more than it saves) stay sequential; the reason is always said.

No psutil: the free memory is read from the operating system directly (Windows
`GlobalMemoryStatusEx`, Linux `/proc/meminfo`), and treated as unknown elsewhere.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

MAX_WORKERS = 4  # § 10 of the study
MIN_IMAGES = 200  # below, starting the processes costs more than it saves
GIB = 2**30
PER_WORKER = 1 * GIB
RESERVE = 2 * GIB
IN_FLIGHT = 2  # images handed to each process at a time (their bytes are in memory)


@dataclass(frozen=True)
class Workers:
    count: int  # 1: sequential
    reason: str  # the sentence shown with the import's progress


def free_memory() -> int | None:
    """Bytes of memory available now, or None when it cannot be read here."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", wintypes.DWORD),
                ("dwMemoryLoad", wintypes.DWORD),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        try:
            ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))  # type: ignore[attr-defined]
        except (AttributeError, OSError):
            return None
        return int(status.ullAvailPhys) if ok else None
    try:
        with open("/proc/meminfo", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def import_workers(
    n_images: int, *, requested: int | None = None, free: int | None = -1, cpus: int | None = None
) -> Workers:
    """R-VIS-ING-07 for a set of `n_images`. `requested` forces a count (tests, scripts), still
    within the memory bound; `free` and `cpus` stand for the machine's (tests)."""
    from quantilo.vision import browser

    if browser.active():
        return Workers(1, "Read in one process: the browser version has no other (R-VIS-ING-07).")
    if requested is not None and requested <= 1:
        return Workers(1, "Read in one process, as asked (R-VIS-ING-07).")
    if requested is None and n_images < MIN_IMAGES:
        return Workers(
            1,
            f"Read in one process: under {MIN_IMAGES} images, starting others costs more than it "
            "saves (R-VIS-ING-07).",
        )
    cores = cpus if cpus is not None else (os.cpu_count() or 1)
    wanted = requested if requested is not None else MAX_WORKERS
    count = max(1, min(wanted, cores - 1 if requested is None else wanted))
    available = free_memory() if free == -1 else free
    if available is not None:
        by_memory = int((available - RESERVE) // PER_WORKER)
        if by_memory < 2:
            return Workers(
                1,
                f"Read in one process: {available / GIB:.1f} GiB of memory free, 3 GiB are needed "
                "for two (R-VIS-ING-07).",
            )
        count = min(count, by_memory)
    if count <= 1:
        return Workers(1, "Read in one process: a single core is free here (R-VIS-ING-07).")
    memory = "" if available is None else f", {available / GIB:.0f} GiB of memory free"
    return Workers(
        count,
        f"Read with {count} processes ({cores} cores{memory}; 1 GiB each, 2 GiB left free) "
        "(R-VIS-ING-07).",
    )


__all__ = ["IN_FLIGHT", "MAX_WORKERS", "MIN_IMAGES", "Workers", "free_memory", "import_workers"]
