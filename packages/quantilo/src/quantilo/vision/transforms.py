"""Preparing images for a model, step by step (110; study 2026-11-11 vision-et-xai § 5).

A `Recipe` holds settings only, never pixels (R-VPR-03): the original files are never changed,
the steps run on the fly at training (112, 113) and here for the previews of the *Prepare* tab.
The steps have a fixed, reasoned order (R-VPR-01):

    decode -> trim -> resize -> white_balance -> denoise -> equalise -> sharpen -> augment
    -> normalise

- `decode` (always, 103): EXIF orientation, RGB (alpha flattened on white), first frame,
  16 -> 8 bits.
- `trim`: uniform borders cropped first, so no input pixel is spent on a frame.
- `resize`: the model's input size; shorter side then centre crop, or `pad` (letterbox on the
  ImageNet mean grey). An image with an odd aspect ratio (R-VQ-03) is padded in crop mode
  (R-VPR-04). Every later step works at that size (R-VPR-02): a sigma of 1 px means the same in a
  preview and at training, and it is fast.
- `white_balance` (gray world), `denoise` (Gaussian or 3 x 3 median) before `equalise` (global or
  CLAHE on the luminance) and `sharpen` (unsharp mask), which would amplify the noise.
- `augment`: training only, seeded: flip, rotation, colour jitter, random resized crop.
- `normalise`: ImageNet means and deviations; numbers, no preview (R-VPR-10).

Pillow, numpy and scipy only (Pyodide too)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any, Literal

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

STEPS = (
    "decode",
    "trim",
    "resize",
    "white_balance",
    "denoise",
    "equalise",
    "sharpen",
    "augment",
    "normalise",
)
Step = Literal[
    "decode", "trim", "resize", "white_balance", "denoise", "equalise", "sharpen", "augment",
    "normalise",
]  # fmt: skip
GEOMETRIC: frozenset[str] = frozenset({"trim", "resize", "augment"})  # R-VPR-08
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PAD_COLOUR = tuple(round(255 * m) for m in IMAGENET_MEAN)  # (124, 116, 104): 0 once normalised
SIZES = (128, 160, 192, 224, 256, 288, 320, 384, 448, 512)
MAX_ASPECT = 4.0  # R-VQ-03
TRIM_TOLERANCE = 2.0  # a border row whose pixels vary by less than this (0-255) is uniform
TRIM_MAX = 0.25  # at most a quarter of each side is cropped
VERSION = 1


class RecipeError(ValueError):
    """A recipe with an unknown step or a setting out of its bounds."""


@dataclass(frozen=True)
class Recipe:
    """The settings of every step (see the module). `task` decides the default augmentations
    (R-VPR-07): `classification` -> flip + crop, `anomaly` -> none."""

    task: Literal["classification", "anomaly", "clusters"] = "classification"
    size: int = 224
    fit: Literal["crop", "pad"] = "crop"
    trim: bool = False
    white_balance: bool = False
    denoise: Literal["off", "gaussian", "median"] = "off"
    sigma: float = 1.0
    equalise: Literal["off", "global", "clahe"] = "off"
    clahe_clip: float = 2.0
    clahe_tiles: int = 8
    sharpen: bool = False
    sharpen_radius: float = 1.0
    sharpen_amount: float = 1.0
    augment: bool = True
    flip: bool = True
    rotate: float = 0.0  # degrees, +/-
    jitter: float = 0.0  # brightness, contrast and saturation factors in [1 - j, 1 + j]
    crop_scale: float = 0.8  # random resized crop: area kept in [crop_scale, 1] (1 = off)

    def enabled(self, step: str) -> bool:
        if step in ("decode", "resize", "normalise"):
            return True
        if step == "trim":
            return self.trim
        if step == "white_balance":
            return self.white_balance
        if step == "denoise":
            return self.denoise != "off"
        if step == "equalise":
            return self.equalise != "off"
        if step == "sharpen":
            return self.sharpen
        if step == "augment":
            return self.augment and self.has_augmentation
        raise RecipeError(f"Unknown step {step!r}: one of {', '.join(STEPS)}.")

    @property
    def has_augmentation(self) -> bool:
        return self.flip or self.rotate > 0 or self.jitter > 0 or self.crop_scale < 1

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "vision-prepare", "version": VERSION, **asdict(self)}

    @classmethod
    def default(cls, task: str = "classification") -> Recipe:
        """R-VPR-07: anomalies get no augmentation (a rotation can be the defect)."""
        if task == "anomaly":
            return cls(task="anomaly", augment=False, flip=False, crop_scale=1.0)
        if task == "clusters":
            return cls(task="clusters", augment=False)
        return cls()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Recipe:
        """Read and check a recipe (unknown keys or out-of-bounds values raise `RecipeError`)."""
        names = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - names - {"kind", "version"})
        if unknown:
            raise RecipeError(f"Unknown setting(s): {', '.join(unknown)}.")
        if data.get("kind", "vision-prepare") != "vision-prepare":
            raise RecipeError("This is not an image preparation recipe.")
        recipe = replace(cls(), **{k: v for k, v in data.items() if k in names})
        recipe.check()
        return recipe

    def check(self) -> None:
        def bounded(name: str, value: float, low: float, high: float) -> None:
            if not isinstance(value, int | float) or not low <= value <= high:
                raise RecipeError(f"{name} must be between {low:g} and {high:g}.")

        choices = {
            "task": ("classification", "anomaly", "clusters"),
            "fit": ("crop", "pad"),
            "denoise": ("off", "gaussian", "median"),
            "equalise": ("off", "global", "clahe"),
        }
        for name, allowed in choices.items():
            if getattr(self, name) not in allowed:
                raise RecipeError(f"{name} must be one of {', '.join(allowed)}.")
        if self.size not in SIZES:
            raise RecipeError(f"size must be one of {', '.join(map(str, SIZES))}.")
        bounded("sigma", self.sigma, 0.3, 5)
        bounded("clahe_clip", self.clahe_clip, 1, 10)
        bounded("clahe_tiles", self.clahe_tiles, 2, 16)
        bounded("sharpen_radius", self.sharpen_radius, 0.5, 5)
        bounded("sharpen_amount", self.sharpen_amount, 0.1, 3)
        bounded("rotate", self.rotate, 0, 45)
        bounded("jitter", self.jitter, 0, 0.5)
        bounded("crop_scale", self.crop_scale, 0.3, 1)
        for name in ("trim", "white_balance", "sharpen", "augment", "flip"):
            if not isinstance(getattr(self, name), bool):
                raise RecipeError(f"{name} must be true or false.")


# --- the steps -----------------------------------------------------------------------------------


def _rgb(image: Image.Image) -> Image.Image:
    return image if image.mode == "RGB" else image.convert("RGB")


def _array(image: Image.Image) -> np.ndarray:
    return np.asarray(_rgb(image), dtype=np.float32)


def _image(array: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(np.rint(array), 0, 255).astype(np.uint8), "RGB")


def border_box(image: Image.Image, tolerance: float = TRIM_TOLERANCE) -> tuple[int, int, int, int]:
    """(left, top, right, bottom) without the uniform borders: rows and columns whose pixels
    vary by at most `tolerance` and match the corner colour; a quarter of each side at most."""
    a = _array(image)
    h, w = a.shape[:2]
    if h < 8 or w < 8:
        return 0, 0, w, h

    def uniform(line: np.ndarray, colour: np.ndarray) -> bool:
        return bool(
            line.std(axis=0).max() <= tolerance and np.abs(line.mean(axis=0) - colour).max() <= 8
        )

    corner = a[0, 0]
    top, bottom, left, right = 0, h, 0, w
    while top < h * TRIM_MAX and uniform(a[top], corner):
        top += 1
    while h - bottom < h * TRIM_MAX and uniform(a[bottom - 1], a[h - 1, w - 1]):
        bottom -= 1
    while left < w * TRIM_MAX and uniform(a[top:bottom, left], corner):
        left += 1
    while w - right < w * TRIM_MAX and uniform(a[top:bottom, right - 1], a[h - 1, w - 1]):
        right -= 1
    return left, top, right, bottom


def has_uniform_border(image: Image.Image) -> bool:
    """R-VPR-06: at least 2 % of a side is a uniform frame."""
    left, top, right, bottom = border_box(image)
    w, h = image.size
    return max(left / w, (w - right) / w, top / h, (h - bottom) / h) >= 0.02


def trim(image: Image.Image) -> Image.Image:
    box = border_box(image)
    return image if box == (0, 0, *image.size) else image.crop(box)


def resize(image: Image.Image, size: int, fit: str = "crop") -> Image.Image:
    """`crop`: shorter side to `size` (bicubic), centre crop. `pad`: longer side to `size`, on the
    ImageNet mean grey."""
    image = _rgb(image)
    w, h = image.size
    if fit == "pad":
        scale = size / max(w, h)
        inner = (max(1, round(w * scale)), max(1, round(h * scale)))
        canvas = Image.new("RGB", (size, size), PAD_COLOUR)  # type: ignore[arg-type]
        resized = image.resize(inner, Image.Resampling.BICUBIC)
        canvas.paste(resized, ((size - inner[0]) // 2, (size - inner[1]) // 2))
        return canvas
    scale = size / min(w, h)
    big = (max(size, round(w * scale)), max(size, round(h * scale)))
    resized = image.resize(big, Image.Resampling.BICUBIC)
    left, top = (big[0] - size) // 2, (big[1] - size) // 2
    return resized.crop((left, top, left + size, top + size))


def white_balance(image: Image.Image) -> Image.Image:
    """Gray world: each channel scaled so that its mean is the mean of the three."""
    a = _array(image)
    means = a.reshape(-1, 3).mean(axis=0)
    gains = means.mean() / np.maximum(means, 1e-3)
    return _image(a * gains)


def denoise(image: Image.Image, method: str, sigma: float = 1.0) -> Image.Image:
    from scipy import ndimage

    a = _array(image)
    if method == "median":
        out = np.stack([ndimage.median_filter(a[..., c], size=3) for c in range(3)], axis=2)
    else:
        out = np.stack([ndimage.gaussian_filter(a[..., c], sigma) for c in range(3)], axis=2)
    return _image(out)


def _equalise_channel(y: np.ndarray) -> np.ndarray:
    histogram = np.bincount(y.ravel(), minlength=256).astype(np.float64)
    cdf = histogram.cumsum()
    low = cdf[cdf > 0][0] if (cdf > 0).any() else 0.0
    span = max(cdf[-1] - low, 1.0)
    lut = np.clip(np.rint((cdf - low) / span * 255), 0, 255)
    return lut[y]


def _clahe_lut(tile: np.ndarray, clip: float) -> np.ndarray:
    """The mapping of one tile: histogram clipped at `clip` x the mean bin, the excess spread
    evenly, then the cumulative distribution (Zuiderveld, 1994)."""
    histogram = np.bincount(tile.ravel(), minlength=256).astype(np.float64)
    limit = max(1.0, clip * tile.size / 256)
    excess = np.maximum(histogram - limit, 0).sum()
    histogram = np.minimum(histogram, limit) + excess / 256
    cdf = histogram.cumsum()
    return np.clip(cdf / cdf[-1] * 255, 0, 255)


def clahe(y: np.ndarray, clip: float = 2.0, tiles: int = 8) -> np.ndarray:
    """Contrast-limited adaptive histogram equalisation of a uint8 channel: one mapping per tile
    of a `tiles` x `tiles` grid, blended bilinearly between tile centres."""
    h, w = y.shape
    th, tw = max(1, math.ceil(h / tiles)), max(1, math.ceil(w / tiles))
    rows, cols = math.ceil(h / th), math.ceil(w / tw)
    luts = np.empty((rows, cols, 256))
    for i in range(rows):
        for j in range(cols):
            luts[i, j] = _clahe_lut(y[i * th : (i + 1) * th, j * tw : (j + 1) * tw], clip)
    # position of each pixel between tile centres
    fy = (np.arange(h) + 0.5) / th - 0.5
    fx = (np.arange(w) + 0.5) / tw - 0.5
    y0 = np.clip(np.floor(fy).astype(int), 0, rows - 1)
    x0 = np.clip(np.floor(fx).astype(int), 0, cols - 1)
    y1 = np.clip(y0 + 1, 0, rows - 1)
    x1 = np.clip(x0 + 1, 0, cols - 1)
    wy = np.clip(fy - y0, 0, 1)[:, None]
    wx = np.clip(fx - x0, 0, 1)[None, :]
    v = y.astype(np.int64)
    top = luts[y0[:, None], x0[None, :], v] * (1 - wx) + luts[y0[:, None], x1[None, :], v] * wx
    bottom = luts[y1[:, None], x0[None, :], v] * (1 - wx) + luts[y1[:, None], x1[None, :], v] * wx
    return top * (1 - wy) + bottom * wy


def equalise(image: Image.Image, method: str, clip: float = 2.0, tiles: int = 8) -> Image.Image:
    """On the luminance only (YCbCr), so the colours keep their hue."""
    ycc = np.asarray(_rgb(image).convert("YCbCr"), dtype=np.uint8).copy()
    y = ycc[..., 0]
    out = _equalise_channel(y) if method == "global" else clahe(y, clip, tiles)
    ycc[..., 0] = np.clip(np.rint(out), 0, 255).astype(np.uint8)
    return Image.fromarray(ycc, "YCbCr").convert("RGB")


def sharpen(image: Image.Image, radius: float = 1.0, amount: float = 1.0) -> Image.Image:
    return _rgb(image).filter(
        ImageFilter.UnsharpMask(radius=radius, percent=round(amount * 100), threshold=2)
    )


def augment(image: Image.Image, recipe: Recipe, seed: int) -> Image.Image:
    """One random draw: random resized crop, flip, rotation, colour jitter (in that order)."""
    rng = np.random.default_rng(seed)
    out = _rgb(image)
    w, h = out.size
    if recipe.crop_scale < 1:
        for _ in range(10):
            area = w * h * rng.uniform(recipe.crop_scale, 1.0)
            ratio = math.exp(rng.uniform(math.log(3 / 4), math.log(4 / 3)))
            cw, ch = round(math.sqrt(area * ratio)), round(math.sqrt(area / ratio))
            if 0 < cw <= w and 0 < ch <= h:
                left = int(rng.integers(0, w - cw + 1))
                top = int(rng.integers(0, h - ch + 1))
                out = out.crop((left, top, left + cw, top + ch)).resize(
                    (w, h), Image.Resampling.BICUBIC
                )
                break
    if recipe.flip and rng.random() < 0.5:
        out = out.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if recipe.rotate > 0:
        angle = float(rng.uniform(-recipe.rotate, recipe.rotate))
        out = out.rotate(angle, Image.Resampling.BILINEAR, fillcolor=PAD_COLOUR)  # type: ignore[arg-type]
    if recipe.jitter > 0:
        j = recipe.jitter
        for enhancer in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
            out = enhancer(out).enhance(float(rng.uniform(1 - j, 1 + j)))
    return out


def normalise(image: Image.Image) -> np.ndarray:
    """(3, h, w) float32 with the ImageNet means and deviations: what a backbone reads."""
    x = _array(image) / 255.0
    x = (x - np.asarray(IMAGENET_MEAN, dtype=np.float32)) / np.asarray(IMAGENET_STD, np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1)).astype(np.float32)


# --- running a recipe ----------------------------------------------------------------------------


@dataclass
class Applied:
    image: Image.Image
    notes: list[str] = field(default_factory=list)


def _run(
    step: str, image: Image.Image, recipe: Recipe, notes: list[str], *, seed: int, odd: bool
) -> Image.Image:
    if step == "trim":
        trimmed = trim(image)
        if trimmed.size != image.size:
            notes.append(f"Uniform borders cropped: {image.size[0]}x{image.size[1]} to "
                         f"{trimmed.size[0]}x{trimmed.size[1]}.")  # fmt: skip
        return trimmed
    if step == "resize":
        fit = recipe.fit
        if fit == "crop" and odd:
            fit = "pad"
            notes.append(
                "Odd aspect ratio (R-VQ-03): padded instead of cropped, so nothing is lost "
                "(R-VPR-04)."
            )
        return resize(image, recipe.size, fit)
    if step == "white_balance":
        return white_balance(image)
    if step == "denoise":
        return denoise(
            image, recipe.denoise if recipe.denoise != "off" else "gaussian", recipe.sigma
        )
    if step == "equalise":
        method = recipe.equalise if recipe.equalise != "off" else "clahe"
        return equalise(image, method, recipe.clahe_clip, recipe.clahe_tiles)
    if step == "sharpen":
        return sharpen(image, recipe.sharpen_radius, recipe.sharpen_amount)
    if step == "augment":
        return augment(image, recipe, seed)
    return image  # decode (done before), normalise (no picture)


def _odd(image: Image.Image) -> bool:
    w, h = image.size
    return max(w / h, h / w) > MAX_ASPECT


def apply(
    image: Image.Image,
    recipe: Recipe,
    *,
    upto: str | None = None,
    train: bool = False,
    seed: int = 0,
    force: str | None = None,
) -> Applied:
    """Run the enabled steps in order, up to and including `upto` (all of them when None).
    `train=False` skips the augmentations; `force` runs that one step even when it is off (the
    preview of a step the user is considering)."""
    if upto is not None and upto not in STEPS:
        raise RecipeError(f"Unknown step {upto!r}: one of {', '.join(STEPS)}.")
    notes: list[str] = []
    out = _rgb(image)
    odd = _odd(out)
    for step in STEPS:
        wanted = recipe.enabled(step) or step == force
        if step == "augment":
            wanted = wanted and (train or force == "augment")
        if wanted:
            out = _run(step, out, recipe, notes, seed=seed, odd=odd)
        if step == upto:
            break
    return Applied(out, notes)


def previous_step(step: str) -> str:
    return STEPS[max(0, STEPS.index(step) - 1)]


@dataclass
class Preview:
    before: Image.Image  # the image as it reaches the step
    after: Image.Image  # what the step makes of it
    notes: list[str]
    geometric: bool  # R-VPR-08: no pixel difference
    enabled: bool  # the step is on in the recipe (False: previewed with its settings anyway)


def preview(image: Image.Image, recipe: Recipe, step: str, *, seed: int = 0) -> Preview:
    """Before and after one step (the earlier steps applied as the recipe says). The augmentation
    preview draws with `seed` (one per tile of the *Prepare* tab)."""
    if step not in STEPS:
        raise RecipeError(f"Unknown step {step!r}: one of {', '.join(STEPS)}.")
    before = (
        _rgb(image) if step == "decode" else apply(image, recipe, upto=previous_step(step)).image
    )
    if step in ("decode", "normalise"):
        after = before
        notes: list[str] = []
    else:
        notes = []
        after = _run(step, before, recipe, notes, seed=seed, odd=_odd(_rgb(image)))
    return Preview(before, after, notes, step in GEOMETRIC, recipe.enabled(step))


def to_tensor(
    image: Image.Image, recipe: Recipe, *, train: bool = False, seed: int = 0
) -> np.ndarray:
    """The whole recipe, normalised: the (3, size, size) input of a model."""
    return normalise(apply(image, recipe, train=train, seed=seed).image)


__all__ = [
    "GEOMETRIC",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "PAD_COLOUR",
    "SIZES",
    "STEPS",
    "Applied",
    "Preview",
    "Recipe",
    "RecipeError",
    "apply",
    "augment",
    "border_box",
    "clahe",
    "denoise",
    "equalise",
    "has_uniform_border",
    "normalise",
    "preview",
    "resize",
    "sharpen",
    "to_tensor",
    "trim",
    "white_balance",
]
