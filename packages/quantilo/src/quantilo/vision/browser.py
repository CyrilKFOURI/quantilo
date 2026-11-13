"""What vision does differently in the browser version (119; study 2026-11-11 §§ 2, 10, 15).

Python runs in the page (Pyodide, WebAssembly): one core, the page's memory, no torch. The vision
module is the same; these rules, each with its code and reason, are read where the module would
otherwise assume a computer. The switch is the one of machine learning (`quantilo.ml.browser`, 102).

- R-VIS-WEB-01: at most 2,000 images and 300 MB per image set; beyond, the first ones are taken
  (path order) and the import says how many were left out.
- R-VIS-WEB-02: the measures are taken at 256 pixels instead of 512, and the blur threshold of
  R-VQ-04 is moved with them (100 at 512 pixels is about 300 at 256, measured on Beans and MVTec
  `bottle` images blurred near the threshold: median ratio 3.3).
- R-VIS-WEB-03: classic models only. DINOv2, PatchCore, PaDiM, the autoencoders and the
  explanations by gradients need torch, which the page does not load: each is greyed with the
  same reason, "Desktop only".
- R-VIS-WEB-04: Pillow (about 5 MB) is loaded when Vision is opened or images are imported, not
  when the page starts (`lite/boot.py`, `frontend/src/lite/worker.ts`).

Nothing here imports Pillow or torch.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quantilo.vision.layout import RuleNote
    from quantilo.vision.quality import QualityThresholds

MAX_IMAGES = 2_000  # R-VIS-WEB-01
MAX_BYTES = 300 * 2**20  # R-VIS-WEB-01
MEASURE_SIDE = 256  # R-VIS-WEB-02 (512 on a computer)
BLUR_SHARPNESS = 300.0  # R-VIS-WEB-02: R-VQ-04's absolute threshold at 256 pixels (100 at 512)
DESKTOP_REASON = (
    "Desktop only: deep models need torch, which the browser version does not load "
    "(R-VIS-WEB-03, R-VDP-01)."
)
# R-VIS-WEB-03: what the browser version leaves to the desktop app, in the order the page says it
DESKTOP_ONLY = (
    "DINOv2",
    "PatchCore",
    "PaDiM",
    "the autoencoders",
    "explanations by gradients",
)
LOADED_ON_DEMAND = "Pillow"  # R-VIS-WEB-04


def active() -> bool:
    """True when the engine runs in the browser: the switch of 102 (`quantilo.ml.browser`, which
    tests can set), or Pyodide itself."""
    from quantilo.ml import browser

    return browser.active() or sys.platform == "emscripten"


def measure_side(side: int | None = None) -> int:
    """R-VIS-WEB-02: the side of the copy measured, 256 pixels in the browser (512 elsewhere)."""
    from quantilo.vision.measures import MEASURE_SIDE as DESKTOP_SIDE

    if side is not None:
        return side
    return MEASURE_SIDE if active() else DESKTOP_SIDE


def quality_thresholds(thresholds: QualityThresholds | None, side: int) -> QualityThresholds | None:
    """R-VIS-WEB-02: at 256 pixels the absolute blur threshold of R-VQ-04 is 300, unless it was
    set by hand (a value other than the default is kept as it is)."""
    from quantilo.vision.measures import MEASURE_SIDE as DESKTOP_SIDE
    from quantilo.vision.quality import QualityThresholds

    if side >= DESKTOP_SIDE:
        return thresholds
    base = thresholds if thresholds is not None else QualityThresholds()
    if base.blur_sharpness != QualityThresholds().blur_sharpness:
        return base
    return replace(base, blur_sharpness=BLUR_SHARPNESS)


@dataclass(frozen=True)
class Budget:
    """R-VIS-WEB-01 applied to the files found: how many are kept, and the note when some were
    left out."""

    kept: int
    note: RuleNote | None


def budget(sizes: Sequence[int]) -> Budget:
    """R-VIS-WEB-01: the first files (in the order found) within 2,000 images and 300 MB."""
    from quantilo.vision.layout import RuleNote

    if not active():
        return Budget(len(sizes), None)
    total = kept = 0
    for size in sizes:
        if kept >= MAX_IMAGES or total + size > MAX_BYTES:
            break
        total += size
        kept += 1
    if kept == len(sizes):
        return Budget(kept, None)
    left = len(sizes) - kept
    return Budget(
        kept,
        RuleNote(
            "R-VIS-WEB-01",
            f"Only the first {kept:,} of {len(sizes):,} images were taken: the browser version "
            f"holds at most {MAX_IMAGES:,} images and {MAX_BYTES // 2**20} MB.",
            "Python runs in the page, on one core and in the page's memory: beyond this an import "
            "takes many minutes and the tab can run out of memory. The desktop app takes 50,000.",
            left,
        ),
    )


def measure_note(side: int) -> RuleNote | None:
    """R-VIS-WEB-02, said at the import when the measures were taken at 256 pixels."""
    from quantilo.vision.layout import RuleNote
    from quantilo.vision.measures import MEASURE_SIDE as DESKTOP_SIDE

    if side >= DESKTOP_SIDE or not active():
        return None
    return RuleNote(
        "R-VIS-WEB-02",
        f"Measured at {side} pixels (512 in the desktop app); the blur threshold of R-VQ-04 is "
        f"{BLUR_SHARPNESS:g} at this scale.",
        "Half the side is a quarter of the pixels to decode and measure on one core. The variance "
        "of the Laplacian changes with the scale (about 3 times higher near the blur threshold), "
        "so the threshold moves with it.",
    )


def reason() -> str:
    """R-VIS-WEB-03: the reason every deep option gives in the browser."""
    return DESKTOP_REASON


__all__ = [
    "BLUR_SHARPNESS",
    "DESKTOP_ONLY",
    "DESKTOP_REASON",
    "MAX_BYTES",
    "MAX_IMAGES",
    "MEASURE_SIDE",
    "Budget",
    "active",
    "budget",
    "measure_note",
    "measure_side",
    "quality_thresholds",
    "reason",
]
