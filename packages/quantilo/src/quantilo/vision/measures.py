"""What an image looks like, in numbers, at a fixed scale (103, § 3.4 and § 4.1 of the study).

Every measure is computed on a copy whose long side is `MEASURE_SIDE` = 512 pixels (box filter).
Without a fixed scale two images cannot be compared: the variance of the Laplacian of a
24-megapixel photo and of its thumbnail differ by orders of magnitude. A smaller image is never
enlarged (enlarging invents pixels and lowers the sharpness): it is measured as it is, and R-VQ-02
(105) flags it as too small.

    m = measure(decoded.image)
    m.brightness, m.sharpness, m.dominant_colour   # 0.47, 812.3, "#5a7d3c"

All in numpy; nothing here needs OpenCV or scikit-image.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

MEASURE_SIDE = 512
CLIP_DARK = 2 / 255  # clipped_dark: share of pixels whose luma is at most this
CLIP_LIGHT = 253 / 255  # clipped_light: share at least this
# is_grayscale: the 99th percentile of the spread between the channels (max - min) is at most this.
# A JPEG of a grey picture keeps a little colour noise; 6/255 lets it through, a pastel photo not.
GRAYSCALE_SPREAD = 6 / 255
COLOUR_SAMPLE = 4_096  # pixels sampled for the dominant colours (k-means)
COLOUR_CLUSTERS = 3

_REC601 = np.array([0.299, 0.587, 0.114])


@dataclass(frozen=True)
class ImageMeasures:
    """The measures of § 3.4. Lumas, contrast, saturation and the clipped shares are 0-1."""

    brightness: float  # mean Rec. 601 luma
    luma_p05: float
    luma_p95: float
    contrast: float  # standard deviation of the luma
    entropy: float  # bits, of the 256-level luma histogram (0-8)
    saturation: float  # mean S of HSV
    colourfulness: float  # Hasler and Suesstrunk (2003), on the 0-255 scale
    is_grayscale: bool
    sharpness: float  # variance of the 3 x 3 Laplacian of the luma (0-255 scale)
    clipped_dark: float
    clipped_light: float
    dominant_colour: str  # "#rrggbb", centre of the largest of 3 k-means clusters
    palette: tuple[tuple[str, float], ...]  # the 3 colours and their shares, largest first
    measured_width: int  # the size actually measured (long side 512, or less if smaller)
    measured_height: int


def working_copy(image: Image.Image, side: int = MEASURE_SIDE) -> Image.Image:
    """An RGB copy whose long side is `side` (box filter), never enlarged."""
    rgb = image if image.mode == "RGB" else image.convert("RGB")
    width, height = rgb.size
    longest = max(width, height)
    if longest <= side:
        return rgb.copy()
    ratio = side / longest
    size = (max(1, round(width * ratio)), max(1, round(height * ratio)))
    return rgb.resize(size, Image.Resampling.BOX)


def luma(rgb: np.ndarray) -> np.ndarray:
    """Rec. 601 luma of an H x W x 3 array, on the array's own scale."""
    return rgb[..., :3].astype(np.float64) @ _REC601


def laplacian_variance(grey: np.ndarray) -> float:
    """Variance of the 4-neighbour Laplacian [[0, 1, 0], [1, -4, 1], [0, 1, 0]] (valid region)."""
    g = np.asarray(grey, dtype=np.float32)
    if g.shape[0] < 3 or g.shape[1] < 3:
        return 0.0
    lap = g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4.0 * g[1:-1, 1:-1]
    return float(lap.var(dtype=np.float64))


def entropy_bits(grey8: np.ndarray) -> float:
    """Shannon entropy, in bits, of the 256-level histogram of an 8-bit image."""
    counts = np.bincount(np.asarray(grey8, dtype=np.uint8).ravel(), minlength=256)
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts[counts > 0] / total
    return max(0.0, float(-(p * np.log2(p)).sum()))


def colourfulness(rgb255: np.ndarray) -> float:
    """Hasler and Suesstrunk's colourfulness: sigma_rgyb + 0.3 mu_rgyb, on the 0-255 scale."""
    r, g, b = (rgb255[..., i].astype(np.float32).ravel() for i in range(3))
    rg = r - g
    yb = 0.5 * (r + g) - b
    sigma = np.sqrt(rg.var(dtype=np.float64) + yb.var(dtype=np.float64))
    mu = np.sqrt(rg.mean(dtype=np.float64) ** 2 + yb.mean(dtype=np.float64) ** 2)
    return float(sigma + 0.3 * mu)


def _hex(colour: np.ndarray) -> str:
    r, g, b = (int(v) for v in np.clip(np.rint(colour), 0, 255))
    return f"#{r:02x}{g:02x}{b:02x}"


def dominant_colours(
    rgb255: np.ndarray, k: int = COLOUR_CLUSTERS, sample: int = COLOUR_SAMPLE, seed: int = 0
) -> tuple[tuple[str, float], ...]:
    """The k main colours (k-means on `sample` pixels drawn with `seed`), with their shares,
    largest first. Deterministic: k-means++ seeding, then Lloyd iterations (at most 20) from the
    same draw."""
    pixels = rgb255.reshape(-1, 3)
    if len(pixels) == 0:
        return ()
    rng = np.random.default_rng(seed)
    if len(pixels) > sample:
        pixels = pixels[rng.choice(len(pixels), sample, replace=False)]
    pixels = pixels.astype(np.float64)
    packed = (pixels[:, 0] * 65536 + pixels[:, 1] * 256 + pixels[:, 2]).astype(np.int64)
    k = max(1, min(k, len(np.unique(packed))))
    centres = np.empty((k, 3))
    centres[0] = pixels[rng.integers(len(pixels))]
    d2 = ((pixels - centres[0]) ** 2).sum(axis=1)
    found = 1
    for j in range(1, k):
        total = d2.sum()
        if total <= 0:
            break
        centres[j] = pixels[rng.choice(len(pixels), p=d2 / total)]
        d2 = np.minimum(d2, ((pixels - centres[j]) ** 2).sum(axis=1))
        found += 1
    c = centres[:found].copy()
    assign = np.full(len(pixels), -1, dtype=np.int64)
    for _ in range(20):
        new = ((pixels[:, None, :] - c[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
        if np.array_equal(new, assign):
            break
        assign = new
        sizes = np.bincount(assign, minlength=len(c)).astype(np.float64)
        sums = np.stack(
            [np.bincount(assign, weights=pixels[:, ch], minlength=len(c)) for ch in range(3)],
            axis=1,
        )
        filled = sizes > 0
        c[filled] = sums[filled] / sizes[filled, None]
    counts = np.bincount(assign, minlength=len(c))
    order = np.argsort(-counts, kind="stable")
    return tuple((_hex(c[j]), float(counts[j] / counts.sum())) for j in order if counts[j] > 0)


def _level(histogram: np.ndarray, q: float) -> int:
    """The lowest 8-bit level whose cumulative share reaches q."""
    cumulative = np.cumsum(histogram)
    return int(np.searchsorted(cumulative, q * cumulative[-1], side="left"))


def measure(
    image: Image.Image, *, side: int = MEASURE_SIDE, seed: int = 0, prepared: bool = False
) -> ImageMeasures:
    """The § 3.4 measures of an image (any mode; RGB is used). `prepared=True` when `image` is
    already the working copy (`working_copy(image, side)`). The percentiles are read on the
    256-level histogram (steps of 1/255)."""
    copy = image if prepared and image.mode == "RGB" else working_copy(image, side)
    rgb = np.asarray(copy, dtype=np.uint8)
    y = rgb.astype(np.float32) @ _REC601.astype(np.float32)  # 0-255
    y8 = np.clip(np.rint(y), 0, 255).astype(np.uint8)
    histogram = np.bincount(y8.ravel(), minlength=256)
    n = max(1, y8.size)
    high = rgb.max(axis=2)
    spread = high - rgb.min(axis=2)  # uint8: high >= low
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(high > 0, spread.astype(np.float32) / high, 0.0)
    spreads = np.bincount(spread.ravel(), minlength=256)
    palette = dominant_colours(rgb, seed=seed)
    return ImageMeasures(
        brightness=float(y.mean(dtype=np.float64)) / 255.0,
        luma_p05=_level(histogram, 0.05) / 255.0,
        luma_p95=_level(histogram, 0.95) / 255.0,
        contrast=float(y.std(dtype=np.float64)) / 255.0,
        entropy=entropy_bits(y8),
        saturation=float(s.mean(dtype=np.float64)),
        colourfulness=colourfulness(rgb),
        is_grayscale=_level(spreads, 0.99) <= round(GRAYSCALE_SPREAD * 255),
        sharpness=laplacian_variance(y),
        clipped_dark=float(histogram[: round(CLIP_DARK * 255) + 1].sum() / n),
        clipped_light=float(histogram[round(CLIP_LIGHT * 255) :].sum() / n),
        dominant_colour=palette[0][0] if palette else "#000000",
        palette=palette,
        measured_width=copy.size[0],
        measured_height=copy.size[1],
    )


__all__ = [
    "CLIP_DARK",
    "CLIP_LIGHT",
    "GRAYSCALE_SPREAD",
    "MEASURE_SIDE",
    "ImageMeasures",
    "colourfulness",
    "dominant_colours",
    "entropy_bits",
    "laplacian_variance",
    "luma",
    "measure",
    "working_copy",
]
