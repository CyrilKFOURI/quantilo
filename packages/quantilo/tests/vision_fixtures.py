"""Synthetic images and hostile files for the vision tests (103, 104), all made here with Pillow,
numpy and the standard library: nothing is downloaded."""

from __future__ import annotations

import io
import json
import struct
import zipfile
import zlib
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


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


def image_folder(root: Path, layout: dict[str, Image.Image | bytes]) -> Path:
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
