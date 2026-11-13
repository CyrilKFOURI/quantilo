"""Images: detection and metadata from the headers only (084).

An image is recognised by its first bytes (never by its name), then its headers are read with
`struct`: dimensions, colour mode, bit depth, frame count, DPI and basic EXIF (camera, date taken,
orientation). No pixel is ever decoded and Pillow is not needed: this runs in the browser
(Pyodide) as it does natively. GPS coordinates are never read, only their presence ("Contains GPS
location"). SVG is refused: it is XML and can carry scripts. Every read is bounded (R-IMG-07), and a
damaged header gives "unknown" with a note, never an exception.

    info = dx.inspect_image("IMG_0042.jpg")
    info.sentence   # "JPEG image, 4,000 × 3,000 pixels (12 megapixels, 4:3), RGB at 8 bits ..."
"""

from __future__ import annotations

import datetime as dt
import math
import os
import re
import struct
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import BinaryIO, Literal

from quantilo.errors import NotFoundError
from quantilo.ingestion.errors import IngestionError

MAX_BYTES_READ = 1024 * 1024  # R-IMG-07: at most 1 MiB read per file, whatever its size
MAX_READS = 20_000  # ... and at most this many reads (chunks and blocks are skipped by seeking)
MAX_IFD_ENTRIES = 64  # R-IMG-04
MAX_TEXT = 128  # an EXIF string is cut there
MAX_TIFF_PAGES = 1_000
PREVIEW_MAX_PIXELS = 100_000_000  # R-IMG-06
_HEAD = 64  # every signature, and the brands of an ISO-BMFF `ftyp`
_TEXT_HEAD = 4096  # to recognise an SVG or a PDF

MIME_TYPES: dict[str, str] = {
    "png": "image/png",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "webp": "image/webp",
    "bmp": "image/bmp",
    "tiff": "image/tiff",
    "heic": "image/heic",
    "heif": "image/heif",
    "avif": "image/avif",
}
_LABELS = {
    "png": "PNG",
    "jpeg": "JPEG",
    "gif": "GIF",
    "webp": "WebP",
    "bmp": "BMP",
    "tiff": "TIFF",
    "heic": "HEIC",
    "heif": "HEIF",
    "avif": "AVIF",
}
_EXTENSIONS = {
    "png": {".png", ".apng"},
    "jpeg": {".jpg", ".jpeg", ".jpe", ".jfif"},
    "gif": {".gif"},
    "webp": {".webp"},
    "bmp": {".bmp", ".dib"},
    "tiff": {".tif", ".tiff"},
    "heic": {".heic", ".heif", ".hif"},
    "heif": {".heic", ".heif", ".hif"},
    "avif": {".avif"},
}
_HEIC_BRANDS = {b"heic", b"heix", b"heim", b"heis", b"hevc", b"hevx"}
_AVIF_BRANDS = {b"avif", b"avis"}
_HEIF_BRANDS = {b"mif1", b"msf1"}
_SEQUENCE_BRANDS = {b"avis", b"msf1", b"hevc", b"hevx"}  # image sequences (animated)
_BMP_HEADERS = {12, 40, 52, 56, 64, 108, 124}

ORIENTATIONS = {
    1: "normal",
    2: "mirrored horizontally",
    3: "rotated 180°",
    4: "mirrored vertically",
    5: "mirrored horizontally and rotated 270° clockwise",
    6: "rotated 90° clockwise",
    7: "mirrored horizontally and rotated 90° clockwise",
    8: "rotated 270° clockwise",
}

SVG_REASON = "SVG is XML and can carry scripts: it is not treated as an image."
NOT_AN_IMAGE = "This file is not a recognised image (PNG, JPEG, GIF, WebP, BMP, TIFF, HEIC, AVIF)."
TRUNCATED = "The header is truncated or damaged: dimensions unknown."
LIMIT = (
    f"Reading stopped at the limit ({MAX_BYTES_READ // 1024} KiB or {MAX_READS:,} reads): "
    "some details may be missing."
)


@dataclass(frozen=True)
class VisionTool:
    """A computer-vision analysis announced for later (R-IMG-11): nothing is computed."""

    key: str
    label: str
    description: str
    status: Literal["coming_later"] = "coming_later"


VISION_TOOLS: tuple[VisionTool, ...] = (
    VisionTool("classification", "Classification", "What the image shows, as labels."),
    VisionTool("object_detection", "Object detection", "Boxes around the objects found."),
    VisionTool("ocr", "Text in image (OCR)", "The text written in the image, as searchable text."),
    VisionTool(
        "embeddings",
        "Embeddings and similar images",
        "A numeric fingerprint per image, to find similar ones and use them in a model.",
    ),
)


@dataclass(frozen=True)
class ExifSummary:
    """Basic EXIF (R-IMG-04). GPS coordinates are never read: only `has_location` (R-IMG-05)."""

    make: str | None = None
    model: str | None = None
    date_taken: str | None = None  # ISO 8601, from DateTimeOriginal, else DateTime
    orientation: int | None = None  # 1-8
    orientation_label: str | None = None
    has_location: bool = False


@dataclass(frozen=True)
class ImageInfo:
    name: str | None
    size_bytes: int
    is_image: bool
    format: str | None  # "png", "jpeg"... or "svg" when refused
    mime: str | None
    reason: str | None  # why it is not treated as an image
    width: int | None = None
    height: int | None = None
    display_width: int | None = None  # after the EXIF orientation (R-IMG-09)
    display_height: int | None = None
    megapixels: float | None = None
    aspect_ratio: str | None = None  # "4:3", or "1.78:1"
    # grayscale, grayscale+alpha, RGB, RGBA, palette, CMYK, YCbCr or Lab
    color_mode: str | None = None
    channels: int | None = None
    bit_depth: int | None = None  # bits per channel (per index for a palette)
    bits_per_pixel: int | None = None
    frames: int | None = None  # pages for a TIFF
    frames_at_least: bool = False  # counting stopped before the end
    animated: bool = False
    dpi: tuple[float, float] | None = None
    previewable: bool = True  # False beyond 100 megapixels (R-IMG-06)
    exif: ExifSummary | None = None
    bytes_read: int = 0
    notes: tuple[str, ...] = ()
    sentence: str = ""
    vision: tuple[VisionTool, ...] = ()


# --- bounded reading ----------------------------------------------------------------------------


class _Exhausted(Exception):
    """The read budget is spent: keep what was found."""


class _Source:
    """Random access to the file or the bytes, every read counted against the budget."""

    def __init__(self, data: bytes | None, handle: BinaryIO | None, size: int) -> None:
        self._data = data
        self._handle = handle
        self.size = size
        self.bytes_read = 0
        self.reads = 0

    def read(self, offset: int, n: int) -> bytes:
        if offset < 0 or n <= 0 or offset >= self.size:
            return b""
        n = min(n, self.size - offset)
        if self.reads >= MAX_READS or self.bytes_read + n > MAX_BYTES_READ:
            raise _Exhausted
        self.reads += 1
        self.bytes_read += n
        if self._data is not None:
            return self._data[offset : offset + n]
        assert self._handle is not None
        self._handle.seek(offset)
        return self._handle.read(n)


class _Window:
    """An EXIF segment already in memory: no pointer may leave it (R-IMG-04)."""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self.size = len(data)

    def read(self, offset: int, n: int) -> bytes:
        if offset < 0 or n <= 0 or offset >= self.size:
            return b""
        return self._data[offset : offset + n]


@dataclass
class _Draft:
    """What has been read so far; a parser fills it as it goes, so a stop keeps the rest."""

    format: str
    width: int | None = None
    height: int | None = None
    color_mode: str | None = None
    channels: int | None = None
    bit_depth: int | None = None
    bits_per_pixel: int | None = None
    frames: int | None = None
    frames_at_least: bool = False
    animated: bool = False
    dpi: tuple[float, float] | None = None
    exif_dpi: tuple[float, float] | None = None
    rotation_swaps: bool = False  # HEIF irot by 90° or 270°
    exif: ExifSummary | None = None
    notes: list[str] = field(default_factory=list)

    def size(self, width: int, height: int) -> None:
        if width > 0 and height > 0:
            self.width, self.height = width, height

    def note(self, text: str) -> None:
        if text not in self.notes:
            self.notes.append(text)


# --- signatures ---------------------------------------------------------------------------------


def _ftyp_brands(head: bytes) -> tuple[bytes, set[bytes]] | None:
    if len(head) < 12 or head[4:8] != b"ftyp":
        return None
    size = struct.unpack(">I", head[:4])[0]
    end = min(max(size, 16), len(head), 64)
    brands = {head[i : i + 4] for i in range(16, end - 3, 4)}
    return head[8:12], brands | {head[8:12]}


def sniff_image(head: bytes) -> str | None:
    """The image format of these first bytes (64 are enough), or None (R-IMG-01). SVG is None."""
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if head.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if head.startswith((b"GIF87a", b"GIF89a")):
        return "gif"
    if len(head) >= 12 and head.startswith(b"RIFF") and head[8:12] == b"WEBP":
        return "webp"
    # "BM" alone could start a CSV ("BMW,Audi"): the DIB header size must be a known one.
    dib = int.from_bytes(head[14:18], "little") if len(head) >= 18 else 0
    if head.startswith(b"BM") and dib in _BMP_HEADERS:
        return "bmp"
    if head.startswith((b"II*\0", b"MM\0*", b"II+\0", b"MM\0+")):
        return "tiff"
    ftyp = _ftyp_brands(head)
    if ftyp is not None:
        _, brands = ftyp
        if brands & _AVIF_BRANDS:
            return "avif"
        if brands & _HEIC_BRANDS:
            return "heic"
        if brands & _HEIF_BRANDS:
            return "heif"
    return None


def _refusal(head: bytes) -> tuple[str | None, str]:
    """Why a file that is not an image is not one (R-IMG-08), naming what was recognised."""
    if not head:
        return None, "The file is empty."
    text = head.removeprefix(b"\xef\xbb\xbf").lstrip()
    if text.startswith(b"<") and re.search(rb"<svg[\s>/]", text, re.IGNORECASE):
        return "svg", SVG_REASON
    if text.startswith(b"%PDF-"):
        return None, "PDF detected: PDF files are not images."
    if _ftyp_brands(head) is not None:
        return None, "This is a video or audio file (MP4, QuickTime), not an image."
    return None, NOT_AN_IMAGE


# --- formats ------------------------------------------------------------------------------------


def _png(src: _Source, d: _Draft) -> None:
    modes = {
        0: ("grayscale", 1),
        2: ("RGB", 3),
        3: ("palette", 1),
        4: ("grayscale+alpha", 2),
        6: ("RGBA", 4),
    }
    pos, first = 8, True
    while True:
        header = src.read(pos, 8)
        if len(header) < 8:
            if first:
                d.note(TRUNCATED)
            return
        length, kind = struct.unpack(">I4s", header)
        start = pos + 8
        if first:
            first = False
            ihdr = src.read(start, 13) if kind == b"IHDR" else b""
            if len(ihdr) < 13:
                d.note(TRUNCATED)
                return
            width, height, depth, color_type = struct.unpack(">IIBB", ihdr[:10])
            d.size(width, height)
            if color_type in modes:
                d.color_mode, d.channels = modes[color_type]
                d.bit_depth, d.bits_per_pixel = depth, depth * d.channels
            d.frames = 1
        elif kind == b"acTL":
            data = src.read(start, 8)
            if len(data) == 8:
                d.frames = struct.unpack(">I", data[:4])[0] or 1
                d.animated = d.frames > 1
        elif kind == b"pHYs":
            data = src.read(start, 9)
            if len(data) == 9 and data[8] == 1:  # unit: the metre
                x, y = struct.unpack(">II", data[:8])
                d.dpi = (x * 0.0254, y * 0.0254)
        elif kind == b"eXIf":
            _exif(_Window(src.read(start, min(length, 65_536))), d)
        elif kind == b"IEND":
            return
        pos = start + length + 4


def _jpeg(src: _Source, d: _Draft) -> None:
    pos = 2
    while True:
        marker_bytes = src.read(pos, 4)
        if len(marker_bytes) < 2 or marker_bytes[0] != 0xFF:
            break
        marker = marker_bytes[1]
        if marker == 0xFF:  # fill byte
            pos += 1
            continue
        if marker in (0x01, 0xD8) or 0xD0 <= marker <= 0xD7:
            pos += 2
            continue
        if marker in (0xD9, 0xDA) or len(marker_bytes) < 4:  # end of image, start of scan
            break
        length = struct.unpack(">H", marker_bytes[2:4])[0]
        if length < 2:
            break
        start, size = pos + 4, length - 2
        if marker == 0xE0:
            data = src.read(start, 14)
            if data.startswith(b"JFIF\0") and len(data) >= 12 and data[7] in (1, 2):
                x, y = struct.unpack(">HH", data[8:12])
                scale = 1.0 if data[7] == 1 else 2.54  # dots per inch, or per centimetre
                if x and y:
                    d.dpi = (x * scale, y * scale)
        elif marker == 0xE1 and d.exif is None and src.read(start, 6) == b"Exif\0\0":
            _exif(_Window(src.read(start + 6, size - 6)), d)
        elif 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            data = src.read(start, 6)
            if len(data) == 6:
                precision, height, width, components = struct.unpack(">BHHB", data)
                d.size(width, height)
                mode = {1: "grayscale", 3: "RGB", 4: "CMYK"}.get(components)
                d.color_mode, d.channels = mode, components
                d.bit_depth, d.bits_per_pixel = precision, precision * components
                d.frames = 1
        pos = start + size
    if d.width is None:
        d.note(TRUNCATED)


def _sub_blocks(src: _Source, pos: int) -> int | None:
    """Skip GIF data sub-blocks; the position after the terminator, or None if the file stops."""
    while True:
        n = src.read(pos, 1)
        if not n:
            return None
        if n[0] == 0:
            return pos + 1
        pos += 1 + n[0]


def _gif(src: _Source, d: _Draft) -> None:
    header = src.read(0, 13)
    if len(header) < 13:
        d.note(TRUNCATED)
        return
    width, height, packed = struct.unpack("<HHB", header[6:11])
    d.size(width, height)
    has_table, table = packed & 0x80, packed & 0x07
    d.color_mode, d.channels = "palette", 1
    d.bit_depth = table + 1 if has_table else ((packed >> 4) & 0x07) + 1
    d.bits_per_pixel = d.bit_depth
    pos = 13 + (3 * 2 ** (table + 1) if has_table else 0)
    frames, d.frames_at_least = 0, True
    while True:
        d.frames, d.animated = (frames or None), frames > 1
        block = src.read(pos, 1)
        if not block:
            break
        if block[0] == 0x3B:  # trailer
            d.frames_at_least = False
            break
        if block[0] == 0x21:  # extension: label, then sub-blocks
            after = _sub_blocks(src, pos + 2)
        elif block[0] == 0x2C:  # image descriptor, optional local table, LZW code size, data
            descriptor = src.read(pos + 1, 9)
            if len(descriptor) < 9:
                break
            local = descriptor[8]
            pos += 10 + (3 * 2 ** ((local & 0x07) + 1) if local & 0x80 else 0)
            after = _sub_blocks(src, pos + 1)
            if after is not None:
                frames += 1
        else:
            d.note("The GIF blocks are damaged: frames counted up to there.")
            break
        if after is None:
            break
        pos = after
    if d.frames_at_least and frames <= 1:
        d.frames_at_least = frames == 1  # a cut still image: one frame at least
    if d.frames is None and d.width is None:
        d.note(TRUNCATED)


def _webp(src: _Source, d: _Draft) -> None:
    riff = src.read(0, 12)
    end = min(src.size, 8 + struct.unpack("<I", riff[4:8])[0]) if len(riff) >= 12 else 0
    pos, extended, frames = 12, False, 0
    d.bit_depth = 8
    while pos + 8 <= end:
        header = src.read(pos, 8)
        if len(header) < 8:
            break
        kind, size = header[:4], struct.unpack("<I", header[4:])[0]
        start = pos + 8
        if kind == b"VP8X":
            data = src.read(start, 10)
            if len(data) == 10:
                extended = True
                flags = data[0]
                d.size(
                    1 + int.from_bytes(data[4:7], "little"),
                    1 + int.from_bytes(data[7:10], "little"),
                )
                d.color_mode, d.channels = ("RGBA", 4) if flags & 0x10 else ("RGB", 3)
                d.animated = bool(flags & 0x02)
                d.frames = None if d.animated else 1
                d.frames_at_least = d.animated
        elif kind == b"VP8 " and not extended:
            data = src.read(start, 10)
            if len(data) == 10 and data[3:6] == b"\x9d\x01\x2a":
                width, height = struct.unpack("<HH", data[6:10])
                d.size(width & 0x3FFF, height & 0x3FFF)
                d.color_mode, d.channels, d.frames = "RGB", 3, 1
        elif kind == b"VP8L" and not extended:
            data = src.read(start, 5)
            if len(data) == 5 and data[0] == 0x2F:
                bits = struct.unpack("<I", data[1:])[0]
                d.size((bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1)
                d.color_mode, d.channels = ("RGBA", 4) if (bits >> 28) & 1 else ("RGB", 3)
                d.frames = 1
        elif kind == b"ANMF":
            frames += 1
            d.frames = frames
        elif kind == b"EXIF":
            data = src.read(start, min(size, 65_536))
            _exif(_Window(data.removeprefix(b"Exif\0\0")), d)
        pos = start + size + (size & 1)
    if d.animated and pos + 8 > end:
        d.frames_at_least = False
    if d.channels is not None:
        d.bits_per_pixel = 8 * d.channels
    if d.width is None:
        d.note(TRUNCATED)


def _bmp(src: _Source, d: _Draft) -> None:
    header = src.read(0, 70)
    dib = struct.unpack("<I", header[14:18])[0] if len(header) >= 18 else 0
    if dib == 12 and len(header) >= 26:
        width, height, _, bpp = struct.unpack("<HHHH", header[18:26])
    elif dib >= 40 and len(header) >= 30:
        width, height, _, bpp = struct.unpack("<iiHH", header[18:30])
        if len(header) >= 46:
            x, y = struct.unpack("<ii", header[38:46])
            if x > 0 and y > 0:
                d.dpi = (x * 0.0254, y * 0.0254)
    else:
        d.note(TRUNCATED)
        return
    d.size(abs(width), abs(height))  # a negative height is a top-down bitmap
    d.bits_per_pixel, d.frames = bpp, 1
    if bpp <= 8:
        d.color_mode, d.channels, d.bit_depth = "palette", 1, bpp
    elif bpp == 32 and dib >= 56 and len(header) >= 70 and struct.unpack("<I", header[66:70])[0]:
        d.color_mode, d.channels, d.bit_depth = "RGBA", 4, 8
    else:
        d.color_mode, d.channels = "RGB", 3
        d.bit_depth = 8 if bpp in (24, 32) else None


# --- TIFF structure (TIFF files and EXIF segments) ----------------------------------------------

_TYPE_SIZES = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8}
_IFD0_TAGS = {256, 257, 258, 262, 271, 272, 274, 277, 282, 283, 296, 306, 338, 34665, 34853}
_Reader = _Source | _Window
_Values = tuple[int, ...] | tuple[float | None, ...] | str


def _decode(order: str, kind: int, count: int, raw: bytes) -> _Values:
    if kind == 2:
        text = raw.split(b"\0", 1)[0].decode("utf-8", errors="replace")
        return "".join(ch for ch in text if ch.isprintable()).strip()[:MAX_TEXT]
    if kind in (5, 10):
        code = "I" if kind == 5 else "i"
        pairs = [struct.unpack(order + code * 2, raw[i : i + 8]) for i in range(0, len(raw) - 7, 8)]
        return tuple(n / den if den else None for n, den in pairs)
    code = {1: "B", 3: "H", 4: "I", 6: "b", 7: "B", 8: "h", 9: "i"}[kind]
    size = _TYPE_SIZES[kind]
    count = min(count, len(raw) // size)
    return struct.unpack(order + code * count, raw[: count * size])


def _ifd(
    r: _Reader, order: str, offset: int, wanted: set[int]
) -> tuple[dict[int, _Values], set[int], int]:
    """The wanted tags of the IFD at `offset`, every tag seen, and the next IFD's offset. At most 64
    entries; a value whose pointer leaves the segment is skipped (R-IMG-04)."""
    head = r.read(offset, 2) if offset >= 8 else b""
    if len(head) < 2:
        return {}, set(), 0
    count = struct.unpack(order + "H", head)[0]
    table = r.read(offset + 2, min(count, MAX_IFD_ENTRIES) * 12)
    values: dict[int, _Values] = {}
    seen: set[int] = set()
    for i in range(0, len(table) - 11, 12):
        tag, kind, n, field_ = struct.unpack(order + "HHI4s", table[i : i + 12])
        seen.add(tag)
        if tag not in wanted or kind not in _TYPE_SIZES:
            continue
        total = _TYPE_SIZES[kind] * n
        if total <= 4:
            raw = field_[:total]
        else:
            at = struct.unpack(order + "I", field_)[0]
            if at < 8 or at + min(total, 256) > r.size:
                continue
            raw = r.read(at, min(total, 256))
        values[tag] = _decode(order, kind, n, raw)
    tail = r.read(offset + 2 + count * 12, 4)
    following = struct.unpack(order + "I", tail)[0] if len(tail) == 4 else 0
    return values, seen, following


def _tiff_header(r: _Reader) -> tuple[str, int] | None:
    head = r.read(0, 8)
    if len(head) < 8 or head[:2] not in (b"II", b"MM"):
        return None
    order = "<" if head[:2] == b"II" else ">"
    magic, first = struct.unpack(order + "HI", head[2:8])
    return (order, first) if magic == 42 else None


def _first_int(values: _Values | None) -> int | None:
    if isinstance(values, tuple) and values and isinstance(values[0], int):
        return values[0]
    return None


def _first_float(values: _Values | None) -> float | None:
    if isinstance(values, tuple) and values and values[0] is not None:
        return float(values[0])
    return None


def _date(text: object) -> str | None:
    match = re.match(r"(\d{4}):(\d{2}):(\d{2})[ T](\d{2}):(\d{2}):(\d{2})", str(text or ""))
    if match is None:
        return None
    year, month, day, hour, minute, second = (int(part) for part in match.groups())
    try:
        return dt.datetime(year, month, day, hour, minute, second).isoformat()
    except ValueError:  # 0000:00:00 00:00:00, a 31st of February
        return None


def _exif_from(r: _Reader, order: str, tags: dict[int, _Values], d: _Draft) -> None:
    """Camera, date, orientation, resolution and the presence of a GPS IFD."""
    exif_tags: dict[int, _Values] = {}
    pointer = _first_int(tags.get(34665))
    if pointer:
        exif_tags = _ifd(r, order, pointer, {36867})[0]
    has_location = False
    pointer = _first_int(tags.get(34853))
    if pointer:
        # R-IMG-05: the GPS IFD's tags are listed, never their values.
        has_location = bool({2, 4} & _ifd(r, order, pointer, set())[1])
    orientation = _first_int(tags.get(274))
    orientation = orientation if orientation in ORIENTATIONS else None
    make, model = tags.get(271), tags.get(272)
    summary = ExifSummary(
        make=make if isinstance(make, str) and make else None,
        model=model if isinstance(model, str) and model else None,
        date_taken=_date(exif_tags.get(36867)) or _date(tags.get(306)),
        orientation=orientation,
        orientation_label=ORIENTATIONS[orientation] if orientation else None,
        has_location=has_location,
    )
    if summary != ExifSummary():
        d.exif = summary
    x, y = _first_float(tags.get(282)), _first_float(tags.get(283))
    unit = _first_int(tags.get(296)) or 2  # TIFF default: inches
    if x and y and unit in (2, 3):
        scale = 1.0 if unit == 2 else 2.54
        d.exif_dpi = (x * scale, y * scale)


def _exif(r: _Window, d: _Draft) -> None:
    header = _tiff_header(r)
    if header is None:
        return
    order, first = header
    _exif_from(r, order, _ifd(r, order, first, _IFD0_TAGS)[0], d)


def _tiff(src: _Source, d: _Draft) -> None:
    header = _tiff_header(src)
    if header is None:
        big = src.read(0, 4) in (b"II+\0", b"MM\0+")
        d.note("BigTIFF files are recognised but not read yet." if big else TRUNCATED)
        return
    order, offset = header
    tags, _, following = _ifd(src, order, offset, _IFD0_TAGS)
    width, height = _first_int(tags.get(256)), _first_int(tags.get(257))
    if width and height:
        d.size(width, height)
    else:
        d.note(TRUNCATED)
    if tags:
        samples = _first_int(tags.get(277)) or 1
        bits = tags.get(258)
        bits = bits if isinstance(bits, tuple) and bits else (1,)
        photometric = _first_int(tags.get(262))
        alpha = 338 in tags
        mode = {0: "grayscale", 1: "grayscale", 3: "palette", 5: "CMYK", 6: "YCbCr", 8: "Lab"}
        mode[2] = "RGBA" if samples >= 4 or alpha else "RGB"
        d.color_mode = mode.get(photometric) if photometric is not None else None
        if d.color_mode == "grayscale" and (samples >= 2 or alpha):
            d.color_mode = "grayscale+alpha"
        d.channels = samples
        d.bit_depth = int(bits[0]) if bits[0] is not None else None
        depths = [int(b) for b in bits if b is not None]
        d.bits_per_pixel = sum(depths) if len(depths) == samples else (d.bit_depth or 0) * samples
        _exif_from(src, order, tags, d)
        d.dpi, d.exif_dpi = d.exif_dpi, None
    pages, visited = 1, {offset}
    d.frames, d.frames_at_least = 1, True
    while following and following not in visited and pages < MAX_TIFF_PAGES:
        visited.add(following)
        _, seen, following = _ifd(src, order, following, set())
        if not seen:
            break
        pages += 1
        d.frames = pages
    d.frames_at_least = pages >= MAX_TIFF_PAGES


def _boxes(src: _Source, start: int, end: int) -> list[tuple[bytes, int, int]]:
    """The ISO-BMFF boxes between `start` and `end`: (type, payload start, payload end)."""
    found: list[tuple[bytes, int, int]] = []
    pos = start
    while pos + 8 <= end and len(found) < 1_000:
        header = src.read(pos, 16)
        if len(header) < 8:
            break
        size, kind = struct.unpack(">I4s", header[:8])
        body = pos + 8
        if size == 1 and len(header) == 16:
            size, body = struct.unpack(">Q", header[8:16])[0], pos + 16
        elif size == 0:
            size = end - pos
        if size < body - pos:
            break
        found.append((kind, body, min(pos + size, end)))
        pos += size
    return found


def _uint(data: bytes, at: int, size: int) -> int | None:
    """A big-endian unsigned integer of 0, 4 or 8 bytes (an `iloc` field), None when short."""
    if size == 0:
        return 0
    if size not in (4, 8) or at + size > len(data):
        return None
    return int.from_bytes(data[at : at + size], "big")


def _exif_item(src: _Source, start: int, end: int) -> int | None:
    """C1 (R-IMG-04): the id of the `Exif` item of a HEIF / AVIF `meta` box (its `iinf`)."""
    for kind, s1, e1 in _boxes(src, start, end):
        if kind != b"iinf":
            continue
        head = src.read(s1, 8)
        if len(head) < 6:
            return None
        first = s1 + (6 if head[0] == 0 else 8)  # version, flags, then a 16- or 32-bit count
        for infe, s2, e2 in _boxes(src, first, e1):
            data = src.read(s2, min(e2 - s2, 16))
            if infe != b"infe" or len(data) < (12 if data[:1] == b"" else 14) or data[0] < 2:
                continue
            if data[0] == 2:
                item, item_type = struct.unpack(">H", data[4:6])[0], data[8:12]
            else:
                item, item_type = struct.unpack(">I", data[4:8])[0], data[10:14]
            if item_type == b"Exif":
                return item
    return None


def _item_bytes(src: _Source, start: int, end: int, wanted: int) -> bytes | None:
    """The first extent of item `wanted`, located in the file by `iloc` (construction method 0
    only: data stored elsewhere in the file), 65,536 bytes at most."""
    for kind, s1, e1 in _boxes(src, start, end):
        if kind != b"iloc":
            continue
        data = src.read(s1, min(e1 - s1, 65_536))
        if len(data) < 8:
            return None
        version = data[0]
        offset_size, length_size = data[4] >> 4, data[4] & 0x0F
        base_size, index_size = data[5] >> 4, (data[5] & 0x0F) if version in (1, 2) else 0
        wide = version == 2  # 32-bit item count and ids
        at = 10 if wide else 8
        count = int.from_bytes(data[6:at], "big")
        for _ in range(min(count, 10_000)):
            id_size = 4 if wide else 2
            if at + id_size > len(data):
                return None
            item = int.from_bytes(data[at : at + id_size], "big")
            at += id_size
            method = 0
            if version in (1, 2):
                method = struct.unpack(">H", data[at : at + 2])[0] & 0x0F
                at += 2
            at += 2  # data reference index
            base = _uint(data, at, base_size)
            at += base_size
            if at + 2 > len(data) or base is None:
                return None
            extents = struct.unpack(">H", data[at : at + 2])[0]
            at += 2
            first: tuple[int, int] | None = None
            for index in range(extents):
                at += index_size
                offset = _uint(data, at, offset_size)
                length = _uint(data, at + offset_size, length_size)
                at += offset_size + length_size
                if offset is None or length is None:
                    return None
                if index == 0:
                    first = (offset, length)
            if item == wanted:
                if method != 0 or first is None:
                    return None
                offset, length = first
                return src.read(base + offset, min(length or (src.size - base - offset), 65_536))
        return None
    return None


def _heif_exif(src: _Source, start: int, end: int, d: _Draft) -> None:
    """C1: the EXIF of a HEIF / AVIF file, without a decoder: the `Exif` item of `meta` (iinf),
    found in the file (iloc); its payload starts with the offset of the TIFF header."""
    item = _exif_item(src, start, end)
    if item is None:
        return
    payload = _item_bytes(src, start, end, item)
    if not payload or len(payload) < 12:
        d.note("The EXIF of this file could not be located: camera and date left empty.")
        return
    skip = 4 + struct.unpack(">I", payload[:4])[0]
    if payload[skip : skip + 2] not in (b"MM", b"II"):  # some writers point at "Exif\0\0"
        skip = payload.find(b"MM\x00*") if b"MM\x00*" in payload else payload.find(b"II*\x00")
    if skip < 0:
        d.note("The EXIF of this file could not be read: camera and date left empty.")
        return
    _exif(_Window(payload[skip:]), d)


def _heif(src: _Source, d: _Draft, major: bytes) -> None:
    d.animated = major in _SEQUENCE_BRANDS
    d.frames = None if d.animated else 1
    largest = 0
    for kind, start, end in _boxes(src, 0, src.size):
        if kind != b"meta":
            continue
        _heif_exif(src, start + 4, end, d)  # C1: EXIF without pillow-heif (R-IMG-04)
        for iprp, s1, e1 in _boxes(src, start + 4, end):  # meta is a full box
            if iprp != b"iprp":
                continue
            for ipco, s2, e2 in _boxes(src, s1, e1):
                if ipco != b"ipco":
                    continue
                for prop, s3, e3 in _boxes(src, s2, e2):
                    data = src.read(s3, min(e3 - s3, 20))
                    if prop == b"ispe" and len(data) >= 12:
                        width, height = struct.unpack(">II", data[4:12])
                        if width * height > largest:  # the primary image, not a thumbnail
                            largest = width * height
                            d.size(width, height)
                    elif prop == b"pixi" and len(data) >= 6 and d.bit_depth is None:
                        channels = data[4]
                        depths = list(data[5 : 5 + channels])
                        if depths:
                            d.channels, d.bit_depth = channels, depths[0]
                            d.bits_per_pixel = sum(depths)
                            d.color_mode = {1: "grayscale", 3: "RGB", 4: "RGBA"}.get(channels)
                    elif prop == b"irot" and data:
                        d.rotation_swaps = bool(data[0] & 0x01)
    if d.width is None:
        d.note(TRUNCATED)


_PARSERS: dict[str, Callable[[_Source, _Draft], None]] = {
    "png": _png,
    "jpeg": _jpeg,
    "gif": _gif,
    "webp": _webp,
    "bmp": _bmp,
    "tiff": _tiff,
}


# --- the result ---------------------------------------------------------------------------------


def _aspect(width: int, height: int) -> str:
    g = math.gcd(width, height)
    a, b = width // g, height // g
    return f"{a}:{b}" if a <= 50 and b <= 50 else f"{width / height:.2f}:1"


def _megapixels(value: float) -> str:
    """'12 megapixels, ' (nothing for an icon: under 0.01 megapixels says nothing)."""
    if value >= 10:
        return f"{value:.0f} megapixels, "
    if value >= 1:
        return f"{value:.1f}".removesuffix(".0") + " megapixels, "
    return f"{value:.2f} megapixels, " if value >= 0.01 else ""


def _camera(exif: ExifSummary) -> str | None:
    make, model = exif.make, exif.model
    if make and model:
        return model if model.lower().startswith(make.split()[0].lower()) else f"{make} {model}"
    return model or make


def _sentence(info: ImageInfo, extension: str | None) -> str:
    label = _LABELS.get(info.format or "", "Image")
    if info.width is None or info.height is None:
        first = f"{label} image; its dimensions could not be read"
    else:
        ratio = f"{_megapixels(info.megapixels or 0)}{info.aspect_ratio}"
        first = f"{label} image, {info.width:,} × {info.height:,} pixels ({ratio})"
    if info.color_mode == "palette":
        first += f", {2**info.bit_depth:,}-colour palette" if info.bit_depth else ", palette"
    elif info.color_mode:
        depth = f" at {info.bit_depth} bits per channel" if info.bit_depth else ""
        first += f", {info.color_mode}{depth}"
    if info.dpi:
        x, y = round(info.dpi[0]), round(info.dpi[1])
        first += f", {x} DPI" if x == y else f", {x} × {y} DPI"
    parts = [first + "."]
    if extension:
        parts.append(f"The name says {extension} but the content is {label}.")
    if info.format == "tiff" and (info.frames or 0) > 1:
        parts.append(f"{'At least ' if info.frames_at_least else ''}{info.frames} pages.")
    elif info.animated:
        if info.frames is None:
            parts.append("Animated (image sequence).")
        elif info.frames_at_least:
            parts.append(f"Animated: at least {info.frames:,} frames (counting stopped).")
        else:
            parts.append(f"Animated: {info.frames:,} frames.")
    exif = info.exif
    if exif is not None:
        camera = _camera(exif)
        if exif.date_taken:
            day, _, time = exif.date_taken.partition("T")
            taken = f"Taken on {day} at {time[:5]}"
            parts.append(f"{taken} with a {camera}." if camera else f"{taken}.")
        elif camera:
            parts.append(f"Camera: {camera}.")
        if exif.orientation and exif.orientation != 1:
            text = f"Orientation: {exif.orientation_label}"
            if exif.orientation >= 5 and info.display_width and info.display_height:
                text = (
                    f"{(exif.orientation_label or '').capitalize()} by the camera: shown "
                    f"{info.display_width:,} × {info.display_height:,}"
                )
            parts.append(text + ".")
        if exif.has_location:
            parts.append("Contains GPS location (not shown).")
    if not info.previewable:
        parts.append("Over 100 megapixels: no preview will be drawn.")
    return " ".join(parts)


def _finish(d: _Draft, name: str | None, size: int, bytes_read: int) -> ImageInfo:
    exif = d.exif
    width, height = d.width, d.height
    display = (width, height)
    swaps = d.rotation_swaps or (exif is not None and (exif.orientation or 0) >= 5)
    if swaps:
        display = (height, width)
    pixels = width * height if width and height else None
    dpi = d.dpi or d.exif_dpi
    extension = Path(name).suffix.lower() if name else ""
    mismatch = extension if extension and extension not in _EXTENSIONS[d.format] else None
    info = ImageInfo(
        name=name,
        size_bytes=size,
        is_image=True,
        format=d.format,
        mime=MIME_TYPES[d.format],
        reason=None,
        width=width,
        height=height,
        display_width=display[0],
        display_height=display[1],
        megapixels=round(pixels / 1e6, 2) if pixels else None,
        aspect_ratio=_aspect(width, height) if width and height else None,
        color_mode=d.color_mode,
        channels=d.channels,
        bit_depth=d.bit_depth,
        bits_per_pixel=d.bits_per_pixel,
        frames=d.frames,
        frames_at_least=d.frames_at_least,
        animated=d.animated,
        dpi=(round(dpi[0], 2), round(dpi[1], 2)) if dpi else None,
        previewable=pixels is None or pixels <= PREVIEW_MAX_PIXELS,
        exif=exif,
        bytes_read=bytes_read,
        notes=tuple(d.notes),
        vision=VISION_TOOLS,
    )
    return replace(info, sentence=_sentence(info, mismatch))


def _inspect(src: _Source, name: str | None) -> ImageInfo:
    head = src.read(0, _HEAD)
    fmt = sniff_image(head)
    if fmt is None:
        refused, reason = _refusal(head + src.read(_HEAD, _TEXT_HEAD - _HEAD))
        return ImageInfo(
            name=name,
            size_bytes=src.size,
            is_image=False,
            format=refused,
            mime=None,
            reason=reason,
            bytes_read=src.bytes_read,
            sentence=reason,
        )
    draft = _Draft(fmt)
    try:
        if fmt in _PARSERS:
            _PARSERS[fmt](src, draft)
        else:
            brands = _ftyp_brands(head)
            _heif(src, draft, brands[0] if brands else b"")
    except _Exhausted:
        draft.note(LIMIT)
    except (struct.error, ValueError, IndexError, OverflowError):
        draft.note(TRUNCATED if draft.width is None else "Part of the header is damaged.")
    return _finish(draft, name, src.size, src.bytes_read)


def inspect_image(
    source: str | os.PathLike[str] | bytes | bytearray | memoryview, *, name: str | None = None
) -> ImageInfo:
    """What an image is, from its headers only: format, dimensions, colour mode, frames, DPI and
    basic EXIF, with a sentence and the vision analyses to come. A file that is not an image (SVG
    included) gives `is_image=False` and the reason, not an exception. At most 1 MiB is read."""
    if isinstance(source, bytes | bytearray | memoryview):
        data = bytes(source)
        return _inspect(_Source(data, None, len(data)), name)
    path = Path(source)
    if not path.exists():
        raise NotFoundError(f"File {path.name!r}")
    if not path.is_file():
        raise IngestionError(f"{path.name!r} is not a file.")
    with path.open("rb") as handle:
        return _inspect(_Source(None, handle, path.stat().st_size), name or path.name)


__all__ = [
    "MAX_BYTES_READ",
    "MIME_TYPES",
    "VISION_TOOLS",
    "ExifSummary",
    "ImageInfo",
    "VisionTool",
    "inspect_image",
    "sniff_image",
]
