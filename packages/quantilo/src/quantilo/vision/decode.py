"""Safe decoding of one image with Pillow (103).

The format comes from the first bytes (R-IMG-01 of 084), never from the name, and Pillow is only
allowed the decoder of that format (`Image.open(formats=[...])`): PNG, JPEG, GIF, WebP, BMP, TIFF,
and AVIF when the wheel carries it. SVG, PSD and EPS are refused (EPS would call Ghostscript), HEIC
is read from its headers only. The size is read before any pixel is decoded: beyond
`MAX_IMAGE_PIXELS` the image is refused as "too large to decode safely" (decompression bomb), and
Pillow's `DecompressionBombWarning` is turned into an error. A truncated file is an error, never a
half-grey image (`LOAD_TRUNCATED_IMAGES` stays False).

What comes back is an RGB image with the EXIF orientation applied and **no metadata at all**: the
EXIF block (GPS included) and the ICC profile are dropped, so nothing written from it (a thumbnail)
can leak a location. Camera, date and the mere presence of a location come from the header reader of
084 (`ImageInfo`), which never reads GPS values.

Every transformation that changes how the pixels are read is said, with its code (R-VIS-DEC-xx):

    decoded = decode_image("IMG_0042.jpg")
    decoded.notes   # ("R-VIS-DEC-01: EXIF orientation applied (rotated 90° clockwise) ...",)
"""

from __future__ import annotations

import io
import os
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageFile, ImageOps, UnidentifiedImageError, features

from quantilo.ingestion.errors import IngestionError
from quantilo.ingestion.images import ImageInfo, inspect_image, sniff_image

MAX_IMAGE_PIXELS = 100_000_000  # R-IMG-06, § 11 of the study: a bomb beyond this
MAX_FILE_BYTES = 256 * 1024 * 1024  # R-VIS-ING-03: a single image file over 256 MiB is not read

# Pillow's own guard follows ours (its default is ~89 Mpx); the explicit check below comes first.
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

_PIL_FORMATS: dict[str, str] = {
    "png": "PNG",
    "jpeg": "JPEG",
    "gif": "GIF",
    "webp": "WEBP",
    "bmp": "BMP",
    "tiff": "TIFF",
}
if features.check("avif"):
    _PIL_FORMATS["avif"] = "AVIF"

DECODABLE_FORMATS: tuple[str, ...] = tuple(_PIL_FORMATS)
EXTENSIONS: dict[str, str] = {
    "png": ".png",
    "jpeg": ".jpg",
    "gif": ".gif",
    "webp": ".webp",
    "bmp": ".bmp",
    "tiff": ".tif",
    "heic": ".heic",
    "heif": ".heif",
    "avif": ".avif",
}

FailureKind = Literal[
    "not_image", "svg", "unsupported", "too_large", "file_too_big", "truncated", "corrupt"
]

FAILURE_CODE = "R-VQ-01"  # a file that cannot be decoded is excluded from models, listed (105)

_ALPHA_MODES = {"RGBA", "LA", "PA", "RGBa", "La"}
_HIGH_DEPTH_MODES = {"I", "I;16", "I;16B", "I;16L", "I;16N", "F"}


@dataclass(frozen=True)
class DecodeFailure:
    """Why an image could not be decoded: a kind, a sentence for the user, the R-VQ-01 code."""

    kind: FailureKind
    reason: str
    code: str = FAILURE_CODE


class ImageDecodeError(IngestionError):
    """`decode_image` on a file that cannot be decoded; `failure` says why."""

    def __init__(self, failure: DecodeFailure) -> None:
        super().__init__(failure.reason)
        self.failure = failure


@dataclass(frozen=True, eq=False)
class DecodedImage:
    """An image decoded safely: RGB, EXIF orientation applied, no metadata kept."""

    image: Image.Image  # mode "RGB"; its `info` is empty (no EXIF, no GPS, no ICC profile)
    format: str  # "png", "jpeg"... from the signature
    width: int  # of the original, after the EXIF orientation
    height: int
    source_mode: str  # Pillow's mode before conversion ("RGB", "P", "I;16", "CMYK"...)
    bit_depth: int | None
    frames: int
    has_alpha: bool
    scale: int  # 1, or the JPEG reduction factor (R-VIS-DEC-05)
    info: ImageInfo  # header metadata from 084: camera, date, has_location (never GPS values)
    notes: tuple[str, ...] = ()

    def array(self) -> np.ndarray:
        """The pixels as an H x W x 3 uint8 array."""
        return np.asarray(self.image, dtype=np.uint8)


@contextmanager
def _strict() -> Iterator[None]:
    """Bombs are errors and truncated files are refused, whatever the caller set before."""
    previous = ImageFile.LOAD_TRUNCATED_IMAGES
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            yield
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = previous


def _read(source: str | os.PathLike[str] | bytes | bytearray | memoryview) -> bytes:
    if isinstance(source, bytes | bytearray | memoryview):
        return bytes(source)
    path = Path(source)
    size = path.stat().st_size
    if size > MAX_FILE_BYTES:
        raise ImageDecodeError(
            DecodeFailure(
                "file_too_big",
                f"The file is {size / 2**20:,.0f} MiB, over the {MAX_FILE_BYTES // 2**20} MiB "
                "limit for one image (R-VIS-ING-03): it was not read.",
            )
        )
    return path.read_bytes()


def _refuse(data: bytes, name: str | None) -> DecodeFailure:
    info = inspect_image(data[:8192], name=name)
    kind: FailureKind = "svg" if info.format == "svg" else "not_image"
    return DecodeFailure(kind, info.reason or "This file is not a recognised image.")


def _too_large(width: int, height: int, limit: int) -> DecodeFailure:
    return DecodeFailure(
        "too_large",
        f"{width:,} x {height:,} pixels ({width * height / 1e6:,.0f} megapixels) is over the "
        f"{limit / 1e6:,.0f}-megapixel limit: too large to decode safely (decompression "
        "bomb guard).",
    )


def _to_rgb(image: Image.Image, notes: list[str]) -> Image.Image:
    """RGB at 8 bits: alpha flattened on white, high bit depths scaled, other modes converted."""
    mode = image.mode
    if mode == "RGB":
        return image
    if mode in _HIGH_DEPTH_MODES:
        values = np.asarray(image, dtype=np.float64)
        top = float(values.max()) if values.size else 0.0
        if mode == "F":
            scale = top if top > 1.0 else 1.0
        else:
            scale = 65535.0 if top > 255 or mode.startswith("I;16") else 255.0
        grey = np.clip(np.rint(values / scale * 255.0), 0, 255).astype(np.uint8)
        notes.append(
            f"R-VIS-DEC-03: {mode} pixels scaled to 8 bits (0-{scale:,.0f} mapped to 0-255) to be "
            "measured like the others."
        )
        return Image.fromarray(grey, "L").convert("RGB")
    if mode in ("P", "PA") and (mode == "PA" or "transparency" in image.info):
        image = image.convert("RGBA")
        mode = "RGBA"
    if mode in _ALPHA_MODES or mode == "RGBA":
        rgba = image.convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        notes.append(
            "R-VIS-DEC-02: transparency flattened on white: transparent areas are measured as "
            "white."
        )
        return Image.alpha_composite(white, rgba).convert("RGB")
    if mode not in ("L", "P", "1"):
        notes.append(f"R-VIS-DEC-06: {mode} colours converted to RGB.")
    return image.convert("RGB")


def _bit_depth(info: ImageInfo, mode: str) -> int | None:
    if info.bit_depth:
        return info.bit_depth
    return {"1": 1, "I;16": 16, "I;16B": 16, "I;16L": 16, "I": 32, "F": 32}.get(mode, 8)


def _decode(data: bytes, name: str | None, max_pixels: int, reduce_to: int | None) -> DecodedImage:
    fmt = sniff_image(data[:64])
    if fmt is None:
        raise ImageDecodeError(_refuse(data, name))
    if fmt not in _PIL_FORMATS:
        raise ImageDecodeError(
            DecodeFailure(
                "unsupported",
                f"{fmt.upper()} images are read from their headers only: convert to JPEG to "
                "analyse the pixels.",
            )
        )
    info = inspect_image(data, name=name)
    notes: list[str] = []
    with _strict():
        try:
            opened = Image.open(io.BytesIO(data), formats=[_PIL_FORMATS[fmt]])
        except (Image.DecompressionBombError, Image.DecompressionBombWarning):
            width, height = info.width or 0, info.height or 0
            raise ImageDecodeError(_too_large(width, height, max_pixels)) from None
        except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as error:
            raise ImageDecodeError(
                DecodeFailure("corrupt", f"The {fmt.upper()} header is damaged: {error}.")
            ) from None
        with opened:
            width, height = opened.size
            if width * height > max_pixels:
                raise ImageDecodeError(_too_large(width, height, max_pixels))
            source_mode = opened.mode
            frames = int(getattr(opened, "n_frames", 1) or 1)
            scale = 1
            if fmt == "jpeg" and reduce_to:
                opened.draft("RGB", (reduce_to, reduce_to))
                scale = max(1, round(width / max(1, opened.size[0])))
                if scale > 1:
                    notes.append(
                        f"R-VIS-DEC-05: JPEG decoded at 1/{scale} scale ({opened.size[0]:,} x "
                        f"{opened.size[1]:,}), enough for the {reduce_to}-pixel measures."
                    )
            try:
                opened.load()
                oriented = ImageOps.exif_transpose(opened)
            except (Image.DecompressionBombError, Image.DecompressionBombWarning):
                raise ImageDecodeError(_too_large(width, height, max_pixels)) from None
            except (OSError, SyntaxError, ValueError, EOFError) as error:
                text = str(error)
                kind: FailureKind = "truncated" if "truncat" in text.lower() else "corrupt"
                raise ImageDecodeError(
                    DecodeFailure(
                        kind,
                        f"The file is truncated or damaged: its pixels could not be decoded "
                        f"({text}).",
                    )
                ) from None
    orientation = info.exif.orientation if info.exif else None
    if orientation and orientation != 1:
        label = info.exif.orientation_label if info.exif else None
        notes.insert(0, f"R-VIS-DEC-01: EXIF orientation applied ({label}), as a viewer shows it.")
    if frames > 1:
        notes.append(f"R-VIS-DEC-04: animated ({frames} frames): the first frame is analysed.")
    rgb = _to_rgb(oriented, notes)
    if rgb is opened:
        rgb = rgb.copy()
    rgb.info = {}  # no EXIF, no GPS, no ICC profile goes further
    swapped = bool(orientation and orientation >= 5)
    return DecodedImage(
        image=rgb,
        format=fmt,
        width=height if swapped else width,
        height=width if swapped else height,
        source_mode=source_mode,
        bit_depth=_bit_depth(info, source_mode),
        frames=frames,
        has_alpha=any(note.startswith("R-VIS-DEC-02") for note in notes),
        scale=scale,
        info=info,
        notes=tuple(notes),
    )


def decode_image(
    source: str | os.PathLike[str] | bytes | bytearray | memoryview,
    *,
    name: str | None = None,
    max_pixels: int = MAX_IMAGE_PIXELS,
    reduce_to: int | None = None,
) -> DecodedImage:
    """Decode an image safely (see the module): RGB, oriented, without metadata.

    `reduce_to` lets a JPEG be decoded at a reduced scale that still covers that many pixels on
    each side (R-VIS-DEC-05): measures at 512 pixels do not need a 24-megapixel decode. Raises
    `ImageDecodeError` (with `.failure`) on a file that is not an image, is refused, too large,
    truncated or damaged."""
    data = _read(source)
    if name is None and not isinstance(source, bytes | bytearray | memoryview):
        name = Path(source).name
    return _decode(data, name, max_pixels, reduce_to)


def try_decode(
    source: str | os.PathLike[str] | bytes | bytearray | memoryview,
    *,
    name: str | None = None,
    max_pixels: int = MAX_IMAGE_PIXELS,
    reduce_to: int | None = None,
) -> DecodedImage | DecodeFailure:
    """`decode_image`, with the failure returned instead of raised (one bad file in a set)."""
    try:
        return decode_image(source, name=name, max_pixels=max_pixels, reduce_to=reduce_to)
    except ImageDecodeError as error:
        return error.failure
    except MemoryError:
        return DecodeFailure("too_large", "Not enough memory to decode this image.")


__all__ = [
    "DECODABLE_FORMATS",
    "EXTENSIONS",
    "FAILURE_CODE",
    "MAX_FILE_BYTES",
    "MAX_IMAGE_PIXELS",
    "DecodeFailure",
    "DecodedImage",
    "ImageDecodeError",
    "decode_image",
    "try_decode",
]
