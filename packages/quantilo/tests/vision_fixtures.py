"""Synthetic images and hostile files for the vision tests (103, 104), all made here with Pillow,
numpy and the standard library: nothing is downloaded. The quality imagette (120 T8) closes it."""

from __future__ import annotations

import colorsys
import io
import json
import stat
import struct
import zipfile
import zlib
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, ImageFilter


def noise(size: tuple[int, int] = (96, 64), seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8), "RGB")


def flat(colour: tuple[int, int, int] = (128, 128, 128), size: tuple[int, int] = (64, 48)):
    return Image.new("RGB", size, colour)


def gradient(size: tuple[int, int] = (64, 64)) -> Image.Image:
    """Brighter to the right."""
    row = np.linspace(0, 255, size[0]).astype(np.uint8)
    grey = np.tile(row, (size[1], 1))
    return Image.fromarray(np.stack([grey] * 3, axis=2), "RGB")


def checker(size: int = 256, cell: int = 8) -> Image.Image:
    idx = np.indices((size, size)) // cell
    grey = ((idx[0] + idx[1]) % 2 * 255).astype(np.uint8)
    return Image.fromarray(np.stack([grey] * 3, axis=2), "RGB")


def scene(seed: int = 0, size: tuple[int, int] = (160, 120)) -> Image.Image:
    """Smooth shapes on a background: a photo-like picture, for the hashes."""
    rng = np.random.default_rng(seed)
    h, w = size[1], size[0]
    yy, xx = np.mgrid[0:h, 0:w]
    image = np.zeros((h, w, 3))
    for _ in range(6):
        cy, cx, r = rng.uniform(0, h), rng.uniform(0, w), rng.uniform(10, 40)
        colour = rng.uniform(0, 255, 3)
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 < r**2
        image[mask] = colour
    return Image.fromarray(image.astype(np.uint8), "RGB")


def encode(image: Image.Image, fmt: str = "PNG", **options: object) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, fmt, **options)  # type: ignore[arg-type]
    return buffer.getvalue()


def jpeg_with_exif(
    image: Image.Image, orientation: int = 6, gps: bool = True, camera: bool = True
) -> bytes:
    exif = Image.Exif()
    exif[0x0112] = orientation
    if camera:
        exif[0x010F] = "Canon"
        exif[0x0110] = "Canon EOS R5"
        exif[0x0132] = "2024:05:01 14:03:00"
    if gps:
        ifd = exif.get_ifd(0x8825)
        ifd[1] = "N"
        ifd[2] = (48.0, 51.0, 24.0)
        ifd[3] = "E"
        ifd[4] = (2.0, 21.0, 3.0)
    return encode(image, "JPEG", exif=exif.tobytes(), quality=95)


def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def png_bomb(width: int = 20_000, height: int = 20_000) -> bytes:
    """A valid 1-bit PNG of width x height black pixels: a few kilobytes that would decode to
    hundreds of megabytes (the decompression bomb of § 11)."""
    row = b"\x00" * (1 + (width + 7) // 8)
    compressor = zlib.compressobj(9)
    data = b"".join(compressor.compress(row) for _ in range(height)) + compressor.flush()
    ihdr = struct.pack(">IIBBBBB", width, height, 1, 0, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", data) + _chunk(b"IEND", b"")
    )


def truncated(data: bytes, keep: float = 0.6) -> bytes:
    return data[: int(len(data) * keep)]


SVG = (
    b'<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>'
)
HEIC = b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00mif1heic" + b"\x00" * 64


def write(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def image_folder(root: Path, layout: Mapping[str, Image.Image | bytes]) -> Path:
    """Write {relative path: image or raw bytes}; PNG unless the extension says JPEG."""
    for rel, content in layout.items():
        if isinstance(content, bytes):
            write(root / rel, content)
        else:
            fmt = "JPEG" if rel.lower().endswith((".jpg", ".jpeg")) else "PNG"
            write(root / rel, encode(content, fmt))
    return root


def zip_of(path: Path, members: dict[str, bytes], *, compression: int = zipfile.ZIP_DEFLATED):
    with zipfile.ZipFile(path, "w", compression) as archive:
        for name, data in members.items():
            archive.writestr(zipfile.ZipInfo(name), data, compress_type=compression)
    return path


def hf_parquet(path: Path, images: list[bytes], labels: list[int], names: list[str]) -> Path:
    """A Hugging Face `datasets` Parquet file: image struct<bytes, path>, ClassLabel labels."""
    image_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    column = pa.array(
        [
            {"bytes": data, "path": f"/home/someone/cache/img_{i}.png"}
            for i, data in enumerate(images)
        ],
        type=image_type,
    )
    table = pa.table(
        {
            "image_file_path": [f"/data/img_{i}.png" for i in range(len(images))],
            "image": column,
            "labels": pa.array(labels, pa.int64()),
        }
    )
    features = {
        "image_file_path": {"dtype": "string", "_type": "Value"},
        "image": {"_type": "Image"},
        "labels": {"names": names, "_type": "ClassLabel"},
    }
    metadata = {b"huggingface": json.dumps({"info": {"features": features}}).encode()}
    pq.write_table(table.replace_schema_metadata(metadata), path, row_group_size=3)
    return path


# --- the quality imagette (120 T8; § 12 of the study) -------------------------------------------
#
# A few dozen small images, all made here, where each R-VQ rule that the pixels or the files can
# trigger has its own positive cases among clean negatives: "every R-VQ finds exactly its cases".
# The clean images vary in size, brightness and colour, as a real set does, so that a tiny, a grey
# or a flat image is not also "unlike the others" (R-VQ-15) by accident. Rules that do fire
# together on one image are written down with the reason (`IMAGETTE_CASES`).

_GOLDEN = 0.6180339887498949  # low-discrepancy steps: the sizes cover their range evenly
_PLASTIC = 0.7548776662466927


def spread_size(seed: int) -> tuple[int, int]:
    """A deterministic size for `seed`: width 80 to 400 px (log-uniform), aspect 3:4 to 4:3."""
    width = round(80 * 5 ** ((seed * _GOLDEN) % 1))
    aspect = 0.75 * (16 / 9) ** ((seed * _PLASTIC) % 1)
    return width, max(1, round(width / aspect))


def _theme_colour(rng: np.random.Generator) -> np.ndarray:
    """A colour of the set's theme (hues from orange to green, as in a set of plant photos):
    the photos share a palette, as the images of one real set do."""
    hue, saturation, value = rng.uniform(0.06, 0.42), rng.uniform(0.3, 0.9), rng.uniform(0.2, 0.9)
    return 255 * np.array(colorsys.hsv_to_rgb(hue, saturation, value))


def photo(seed: int, size: tuple[int, int] | None = None, noise_level: float = 8.0) -> Image.Image:
    """A photo-like picture: a two-colour gradient, seven ellipses or boxes in the theme's colours
    (`_theme_colour`), a colour strength from muted to vivid and a little sensor noise (luma and
    chroma). Size: `spread_size(seed)`."""
    rng = np.random.default_rng(seed)
    w, h = size or spread_size(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    left, right = _theme_colour(rng), _theme_colour(rng)
    t = (xx / max(1, w - 1))[..., None]
    pixels = left * (1 - t) + right * t
    for _ in range(7):
        cy, cx = rng.uniform(0, h), rng.uniform(0, w)
        ry, rx = rng.uniform(0.08, 0.3) * h, rng.uniform(0.08, 0.3) * w
        colour = _theme_colour(rng)
        if rng.random() < 0.5:
            mask = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 < 1
        else:
            mask = (np.abs(yy - cy) < ry) & (np.abs(xx - cx) < rx)
        pixels[mask] = colour
    grey = pixels.mean(axis=2, keepdims=True)
    pixels = grey + rng.uniform(0.35, 1.0) * (pixels - grey)
    pixels += rng.normal(0, noise_level, (h, w, 1)) + rng.normal(0, noise_level / 3, (h, w, 3))
    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8), "RGB")


def two_level(
    seed: int, low: tuple[int, int, int], high: tuple[int, int, int], noise_level: float = 3.0
) -> Image.Image:
    """The shapes of `photo(seed)` drawn in two tinted levels, with noise: an exposure defect
    that keeps its contrast, entropy and sharpness (so R-VQ-05 / 06 fire, not 04 or 07)."""
    rng = np.random.default_rng(seed)
    base = np.asarray(photo(seed), dtype=np.float64).mean(axis=2)
    h, w = base.shape
    pixels = np.where((base > np.median(base))[..., None], high, low).astype(np.float64)
    pixels += rng.normal(0, noise_level, (h, w, 1))
    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8), "RGB")


def _flat_orange(seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    w, h = spread_size(seed)
    pixels = np.array([230.0, 80.0, 40.0]) + rng.normal(0, 3, (h, w, 1))
    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8), "RGB")


def _vivid(seed: int) -> Image.Image:
    """Blocks of pure red, green and blue, with noise: far more colourful than any photo."""
    rng = np.random.default_rng(seed)
    w, h = spread_size(seed)
    primaries = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.float64)
    blocks = (np.indices((h, w)) // 16).sum(axis=0) % 3
    pixels = primaries[blocks] + rng.normal(0, 8, (h, w, 3))
    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8), "RGB")


def _scanned_page(seed: int, size: tuple[int, int] = (160, 220)) -> Image.Image:
    """Lines of dark words on bluish paper lit unevenly: a document scan, the "image from another
    set" of R-VQ-15's content part (its brightness and colour stay ordinary)."""
    rng = np.random.default_rng(seed)
    w, h = size
    yy, xx = np.indices((h, w))
    light = 170 + 50 * (xx + yy) / (w + h)
    pixels = light[..., None] * np.array([0.82, 0.9, 1.0])
    for top in range(8, h - 8, 12):
        for left in range(6, w - 6, 9):
            if rng.random() < 0.8:
                pixels[top : top + 6, left : left + int(rng.integers(2, 8))] = (30, 30, 60)
    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8), "RGB")


def _cropped(image: Image.Image, share: float = 0.02) -> Image.Image:
    """`share` cut off each side, resized back: a near copy (R-VQ-10)."""
    w, h = image.size
    dx, dy = max(1, round(w * share)), max(1, round(h * share))
    return image.crop((dx, dy, w - dx, h - dy)).resize((w, h), Image.Resampling.LANCZOS)


# The defect cases: (relative path, the R-VQ codes expected on it, how it is made). Every other
# image is clean, except the three of class `c` (R-VQ-14). R-VQ-13 is a set-level finding.
IMAGETTE_CASES: tuple[tuple[str, frozenset[str], str], ...] = tuple(
    (rel, frozenset(codes), how)
    for rel, codes, how in (
        ("train/a/truncated.png", {"R-VQ-01"}, "a PNG cut at 60 %"),
        ("train/a/corrupt.png", {"R-VQ-01"}, "a PNG signature and header, then garbage"),
        ("train/a/bomb.png", {"R-VQ-01"}, "20,000 x 20,000 px in 50 kB: decompression bomb"),
        ("train/a/tiny.png", {"R-VQ-02"}, "30 x 30 px; sizes vary enough that R-VQ-15 stays"),
        ("train/a/wide.png", {"R-VQ-03"}, "400 x 90 px, 4.4:1"),
        ("train/a/blurry.png", {"R-VQ-04"}, "Gaussian blur, radius 1.6 px"),
        # a too dark (bright) image is also far from the set's brightness: R-VQ-15 says so too
        ("train/a/dark.png", {"R-VQ-05", "R-VQ-15"}, "two tinted levels, luma 5 and 36 of 255"),
        ("train/a/bright.png", {"R-VQ-06", "R-VQ-15"}, "two tinted levels, luma 227 and 250"),
        ("train/a/flat.png", {"R-VQ-07"}, "one orange, noise of 3 levels (not grey, not dark)"),
        ("train/a/grey.png", {"R-VQ-08"}, "a photo in greyscale in a colour set"),
        ("train/a/copy.bmp", {"R-VQ-09"}, "the PNG below saved as BMP: same pixels, kept"),
        ("train/a/copy.png", {"R-VQ-09"}, "the second copy, left out of models"),
        ("train/a/burst.jpg", {"R-VQ-10"}, "the PNG below recompressed (JPEG 70)"),
        ("train/a/burst.png", {"R-VQ-10"}, "the original of a burst"),
        ("train/a/burst_crop.png", {"R-VQ-10"}, "the same, 2 % cropped on each side"),
        ("train/a/leak.png", {"R-VQ-10"}, "its JPEG is in test"),
        # R-VQ-11 is read on the groups of R-VQ-09 / 10: the test copy carries both codes
        ("test/a/leak.jpg", {"R-VQ-10", "R-VQ-11"}, "a near copy of a train image, in test"),
        # R-VQ-12 is read on the groups of R-VQ-09 / 10: both copies carry both codes
        ("train/a/twice.png", {"R-VQ-09", "R-VQ-12"}, "the same image labelled a ..."),
        ("train/b/twice.png", {"R-VQ-09", "R-VQ-12"}, "... and b (this copy left out)"),
        ("train/a/huge.jpg", {"R-VQ-15"}, "4,000 x 3,000 px among 80 to 400 (size part)"),
        ("train/a/vivid.png", {"R-VQ-15"}, "pure primaries (colour part)"),
        ("train/a/page.png", {"R-VQ-15"}, "a scanned page of text (content part)"),
    )
)
IMAGETTE_DATASET = frozenset({"R-VQ-13"})  # a 56 / b 13 once R-VQ-14 drops c: ratio over 3
IMAGETTE_EXCLUDED = frozenset(
    {
        "train/a/truncated.png", "train/a/corrupt.png", "train/a/bomb.png",  # R-VQ-01
        "train/a/copy.png", "train/b/twice.png",  # R-VQ-09: the second copy of each group
        "train/c/rare_47.png", "train/c/rare_48.png", "train/c/rare_49.png",  # R-VQ-14: 3 < 5
    }
)  # fmt: skip
# The rules the imagette covers, in the order of the catalogue (R-VQ-16 and 17 need a model).
IMAGETTE_RULES = tuple(f"R-VQ-{i:02d}" for i in range(1, 16))


@dataclass(frozen=True)
class Imagette:
    files: dict[str, bytes]  # relative path -> bytes, in the folder layout train|test/<label>/
    expected: dict[str, frozenset[str]]  # relative path -> the R-VQ codes it must carry
    excluded: frozenset[str]  # relative paths the rules leave out of models
    dataset: frozenset[str]  # set-level findings (no row)


def quality_imagette() -> Imagette:
    """The imagette (see `IMAGETTE_CASES`): 56 clean photos in classes a (40), b (13) and c (3,
    too small), split train / test by folder, and 22 defect cases. Deterministic: 78 files, 8 MB
    (noisy PNGs of 80 to 400 px; the bomb is 50 kB), built in about 1.5 s."""
    files: dict[str, bytes] = {}
    for seed in range(36):
        files[f"train/a/clean_{seed:02d}.png"] = encode(photo(seed))
    for seed in range(36, 47):
        files[f"train/b/clean_{seed:02d}.png"] = encode(photo(seed))
    for seed in range(47, 50):
        files[f"train/c/rare_{seed:02d}.png"] = encode(photo(seed))
    for seed in range(50, 54):
        files[f"test/a/clean_{seed:02d}.png"] = encode(photo(seed))
    for seed in range(54, 56):
        files[f"test/b/clean_{seed:02d}.png"] = encode(photo(seed))
    files["train/a/truncated.png"] = truncated(encode(photo(100)))
    files["train/a/corrupt.png"] = encode(photo(101))[:41] + b"\x00garbage" * 64
    files["train/a/bomb.png"] = png_bomb()
    files["train/a/tiny.png"] = encode(photo(102, (30, 30)))
    files["train/a/wide.png"] = encode(photo(103, (400, 90)))
    files["train/a/blurry.png"] = encode(photo(104).filter(ImageFilter.GaussianBlur(1.6)))
    files["train/a/dark.png"] = encode(two_level(105, (8, 4, 2), (44, 34, 20)))
    files["train/a/bright.png"] = encode(two_level(106, (240, 225, 200), (255, 250, 240)))
    files["train/a/flat.png"] = encode(_flat_orange(107))
    files["train/a/grey.png"] = encode(photo(108).convert("L").convert("RGB"))
    original = photo(109)
    files["train/a/copy.bmp"] = encode(original, "BMP")
    files["train/a/copy.png"] = encode(original)
    burst = photo(110)
    files["train/a/burst.jpg"] = encode(burst, "JPEG", quality=70)
    files["train/a/burst.png"] = encode(burst)
    files["train/a/burst_crop.png"] = encode(_cropped(burst))
    leaked = photo(111)
    files["train/a/leak.png"] = encode(leaked)
    files["test/a/leak.jpg"] = encode(leaked, "JPEG", quality=75)
    twice = photo(112)
    files["train/a/twice.png"] = encode(twice)
    files["train/b/twice.png"] = encode(twice)
    big = photo(113, (500, 375)).resize((4000, 3000), Image.Resampling.NEAREST)
    files["train/a/huge.jpg"] = encode(big, "JPEG", quality=90)
    files["train/a/vivid.png"] = encode(_vivid(114))
    files["train/a/page.png"] = encode(_scanned_page(115))
    expected = {rel: frozenset[str]() for rel in files}
    expected.update({rel: codes for rel, codes, _ in IMAGETTE_CASES})
    expected.update({rel: frozenset({"R-VQ-14"}) for rel in files if rel.startswith("train/c/")})
    return Imagette(files, expected, IMAGETTE_EXCLUDED, IMAGETTE_DATASET)


def imagette_folder(root: Path, imagette: Imagette | None = None) -> Path:
    for rel, data in (imagette or quality_imagette()).files.items():
        write(root / rel, data)
    return root


# The traps of the ZIP version (§ 3.1, `zipsafe.py`): member -> the rule that skips it. The bomb
# PNG is caught here before any decoding: its 50 kB of zlib zeros deflate again, 176 times.
IMAGETTE_ZIP_SKIPPED = {
    "train/a/bomb.png": "R-VIS-ZIP-04",
    "train/a/inner.zip": "R-VIS-ZIP-05",
    "train/a/link.png": "R-VIS-ZIP-06",
    "train/a/damaged.png": "R-VIS-ZIP-07",
    "train/a/chart.svg": "R-VIS-ING-06",
}
IMAGETTE_ZIP_HIDDEN = ("__MACOSX/train/a/._clean_00.png", "train/a/.thumbs.png")  # R-VIS-ING-04


def imagette_zip(path: Path, imagette: Imagette | None = None, *, slip: bool = False) -> Path:
    """The imagette as a ZIP (members deflated, as most tools do) with traps: a bomb, an archive
    inside, a link, a member whose CRC is wrong, an SVG with a script, hidden files. `slip` adds
    `../evil.png`, which refuses the whole archive (R-VIS-ZIP-01)."""
    imagette = imagette or quality_imagette()
    small = encode(photo(116, (24, 18)))  # under 8 kB: its head read reaches the CRC check
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for rel, data in imagette.files.items():
            archive.writestr(rel, data)
        archive.writestr("train/a/inner.zip", b"PK\x03\x04" + b"\0" * 30)
        link = zipfile.ZipInfo("train/a/link.png")
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(link, "../../outside.png")
        archive.writestr("train/a/damaged.png", small, compress_type=zipfile.ZIP_STORED)
        archive.writestr("train/a/chart.svg", SVG)
        for name in IMAGETTE_ZIP_HIDDEN:
            archive.writestr(name, encode(photo(117)))
        if slip:
            archive.writestr("../evil.png", small)
    # one byte of the stored member's data flipped: its CRC no longer matches
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo("train/a/damaged.png")
    with path.open("r+b") as handle:
        handle.seek(info.header_offset + 26)
        name_length, extra_length = struct.unpack("<HH", handle.read(4))
        at = info.header_offset + 30 + name_length + extra_length + len(small) // 2
        handle.seek(at)
        byte = handle.read(1)[0]
        handle.seek(at)
        handle.write(bytes([byte ^ 0xFF]))
    return path


def rule_confusion(
    expected: Mapping[str, Collection[str]],
    found: Mapping[str, Collection[str]],
    codes: Collection[str],
) -> list[dict[str, object]]:
    """Per rule: how many images were expected and found, and which were missed (expected, not
    found) or extra (found, not expected), by relative path."""
    table: list[dict[str, object]] = []
    for code in codes:
        want = {p for p, c in expected.items() if code in c}
        got = {p for p, c in found.items() if code in c}
        table.append(
            {
                "rule": code,
                "expected": len(want),
                "found": len(got),
                "missed": sorted(want - got),
                "extra": sorted(got - want),
            }
        )
    return table
