"""Classic image descriptors in numpy (103, § 4.4 of the study): HSV histogram, HOG, uniform LBP.

They are the first-plan embedding of the Map tab (109) and of the browser models (112): no network,
no weights, the same numbers in Pyodide. Sizes are fixed so that every image gives a vector of the
same length:

- HSV histogram, 72 values: the working copy, H 8 x S 3 x V 3 bins, summing to 1.
- HOG (Dalal and Triggs), 1,764 values: grey 128 x 128, cells of 16 px, 9 unsigned orientations,
  blocks of 2 x 2 cells.
- Uniform LBP (P = 8, R = 1), 236 values: grey 128 x 128, 58 uniform patterns + 1 other, on a
  2 x 2 grid.

LBP uses the 8 pixels of the 3 x 3 neighbourhood (the square approximation of the circle of radius
1, without interpolating the diagonals).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from quantilo.vision.measures import working_copy

DESCRIPTOR_SIDE = 128
HSV_BINS = (8, 3, 3)
HOG_CELL = 16
HOG_ORIENTATIONS = 9
HOG_BLOCK = 2
LBP_GRID = 2

HSV_SIZE = HSV_BINS[0] * HSV_BINS[1] * HSV_BINS[2]  # 72
_CELLS = DESCRIPTOR_SIDE // HOG_CELL  # 8
HOG_SIZE = (_CELLS - HOG_BLOCK + 1) ** 2 * HOG_BLOCK**2 * HOG_ORIENTATIONS  # 1,764
LBP_CLASSES = 59
LBP_SIZE = LBP_GRID**2 * LBP_CLASSES  # 236
CLASSIC_SIZE = HSV_SIZE + HOG_SIZE + LBP_SIZE  # 2,072


def _grey_square(image: Image.Image, side: int = DESCRIPTOR_SIDE) -> np.ndarray:
    grey = image if image.mode == "L" else image.convert("L")
    smaller = max(grey.size) <= side
    resample = Image.Resampling.BILINEAR if smaller else Image.Resampling.BOX
    return np.asarray(grey.resize((side, side), resample), dtype=np.float64)


def hsv_histogram(image: Image.Image) -> np.ndarray:
    """The joint H x S x V histogram (8 x 3 x 3 = 72 bins) of the working copy, summing to 1."""
    hsv = np.asarray(working_copy(image).convert("HSV"), dtype=np.int64).reshape(-1, 3)
    if len(hsv) == 0:
        return np.zeros(HSV_SIZE, dtype=np.float32)
    h = hsv[:, 0] * HSV_BINS[0] // 256
    s = hsv[:, 1] * HSV_BINS[1] // 256
    v = hsv[:, 2] * HSV_BINS[2] // 256
    index = (h * HSV_BINS[1] + s) * HSV_BINS[2] + v
    counts = np.bincount(index, minlength=HSV_SIZE).astype(np.float64)
    return (counts / counts.sum()).astype(np.float32)


def hog(image: Image.Image) -> np.ndarray:
    """Histogram of oriented gradients of the 128 x 128 grey image (1,764 values).

    Centred gradients [-1, 0, 1]; 9 unsigned orientation bins of 20 degrees, each vote split
    linearly between the two nearest bins and weighted by the magnitude; 8 x 8 cells of 16 pixels;
    blocks of 2 x 2 cells (stride one cell) normalised L2-Hys (L2, clipped at 0.2, L2 again)."""
    g = _grey_square(image) / 255.0
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[:, 1:-1] = g[:, 2:] - g[:, :-2]
    gy[1:-1, :] = g[2:, :] - g[:-2, :]
    magnitude = np.hypot(gx, gy)
    angle = np.degrees(np.arctan2(gy, gx)) % 180.0
    width = 180.0 / HOG_ORIENTATIONS
    position = angle / width - 0.5
    lower = np.floor(position)
    fraction = position - lower
    b0 = lower.astype(np.int64) % HOG_ORIENTATIONS
    b1 = (b0 + 1) % HOG_ORIENTATIONS
    rows, cols = np.indices(g.shape)
    cell = (rows // HOG_CELL) * _CELLS + cols // HOG_CELL
    hist = np.zeros(_CELLS * _CELLS * HOG_ORIENTATIONS)
    np.add.at(hist, (cell * HOG_ORIENTATIONS + b0).ravel(), (magnitude * (1 - fraction)).ravel())
    np.add.at(hist, (cell * HOG_ORIENTATIONS + b1).ravel(), (magnitude * fraction).ravel())
    cells = hist.reshape(_CELLS, _CELLS, HOG_ORIENTATIONS)
    blocks = []
    eps = 1e-6
    for i in range(_CELLS - HOG_BLOCK + 1):
        for j in range(_CELLS - HOG_BLOCK + 1):
            v = cells[i : i + HOG_BLOCK, j : j + HOG_BLOCK].ravel()
            v = v / np.sqrt((v**2).sum() + eps**2)
            v = np.minimum(v, 0.2)
            v = v / np.sqrt((v**2).sum() + eps**2)
            blocks.append(v)
    return np.concatenate(blocks).astype(np.float32)


def _uniform_table() -> np.ndarray:
    """LBP code (0-255) -> class: the 58 uniform patterns (at most 2 transitions around the
    circle) numbered in increasing code order, every other pattern in class 58."""
    table = np.full(256, LBP_CLASSES - 1, dtype=np.int64)
    label = 0
    for code in range(256):
        bits = [(code >> k) & 1 for k in range(8)]
        transitions = sum(bits[k] != bits[(k + 1) % 8] for k in range(8))
        if transitions <= 2:
            table[code] = label
            label += 1
    return table


_UNIFORM = _uniform_table()
# The 8 neighbours, in circular order (right, then anticlockwise), as (row, column) offsets.
_NEIGHBOURS = ((0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1))


def lbp_codes(grey: np.ndarray) -> np.ndarray:
    """The 8-bit LBP code of every inner pixel: bit k set when neighbour k >= the centre."""
    g = np.asarray(grey, dtype=np.float64)
    h, w = g.shape
    centre = g[1 : h - 1, 1 : w - 1]
    codes = np.zeros(centre.shape, dtype=np.int64)
    for k, (dr, dc) in enumerate(_NEIGHBOURS):
        neighbour = g[1 + dr : h - 1 + dr, 1 + dc : w - 1 + dc]
        codes |= (neighbour >= centre).astype(np.int64) << k
    return codes


def lbp(image: Image.Image) -> np.ndarray:
    """Uniform LBP histograms (59 classes) of the 4 quadrants of the 128 x 128 grey image, each
    summing to 1 (236 values)."""
    classes = _UNIFORM[lbp_codes(_grey_square(image))]
    h, w = classes.shape
    parts = []
    for i in range(LBP_GRID):
        for j in range(LBP_GRID):
            tile = classes[i * h // LBP_GRID : (i + 1) * h // LBP_GRID]
            tile = tile[:, j * w // LBP_GRID : (j + 1) * w // LBP_GRID]
            counts = np.bincount(tile.ravel(), minlength=LBP_CLASSES).astype(np.float64)
            parts.append(counts / max(1.0, counts.sum()))
    return np.concatenate(parts).astype(np.float32)


@dataclass(frozen=True, eq=False)
class ClassicDescriptor:
    """The three descriptors of one image, and their concatenation (2,072 values)."""

    hsv: np.ndarray
    hog: np.ndarray
    lbp: np.ndarray

    @property
    def vector(self) -> np.ndarray:
        return np.concatenate([self.hsv, self.hog, self.lbp]).astype(np.float32)


def classic_descriptor(image: Image.Image) -> ClassicDescriptor:
    """HSV histogram, HOG and uniform LBP of an image (see the module)."""
    return ClassicDescriptor(hsv=hsv_histogram(image), hog=hog(image), lbp=lbp(image))


__all__ = [
    "CLASSIC_SIZE",
    "HOG_SIZE",
    "HSV_SIZE",
    "LBP_SIZE",
    "ClassicDescriptor",
    "classic_descriptor",
    "hog",
    "hsv_histogram",
    "lbp",
    "lbp_codes",
]
