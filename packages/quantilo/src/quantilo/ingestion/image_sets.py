"""Is this source an image set? (106) Told from names and the Parquet schema only, before the import
screen chooses its step: nothing is decoded and Pillow is not imported (the browser build loads it
when the Vision module opens, not at the first preview).

- R-VIS-SRC-01: a folder whose files (the first 2,000 met, 6 levels deep at most, hidden ones
  left out) are images for at least half, and at least one. A folder of photos with a
  `labels.csv` is an image set; a folder of CSV files with a logo is not.
- R-VIS-SRC-02: a ZIP (not an Office file) whose members are images for at least half.
- R-VIS-SRC-03: a Parquet file with an image column (`struct<bytes, path>`, R-VIS-HF-01): offered
  as an image set, the table import stays one click away (Hugging Face `datasets`).
- R-VIS-SRC-04: a folder with no image file whose Parquet files hold an image column, for at least
  half of them (the Hugging Face layout `data/train-...parquet`, `test-...`, `validation-...`):
  imported as one image set, each file's split read from its name (R-VIS-LAY-06).

The content is checked at the import, by signature (R-IMG-01): a name only routes the screen.
"""

from __future__ import annotations

import os
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

import pyarrow as pa
import pyarrow.parquet as pq

IMAGE_SUFFIXES = frozenset(
    {".png", ".apng", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff", ".avif"}
)
SCAN_LIMIT = 2_000
SCAN_DEPTH = 6
IMAGE_SHARE = 0.5
PARQUET_LIMIT = 200  # R-VIS-SRC-04: schemas read at most (a schema is a few kilobytes)

ImageSourceKind = Literal["folder", "zip", "parquet"]


@dataclass(frozen=True)
class ImageSource:
    kind: ImageSourceKind
    rule: str
    label: str
    reason: str
    images: int  # images counted (at most SCAN_LIMIT for a folder or a ZIP)


def _hidden(name: str) -> bool:
    return name.startswith((".", "__MACOSX")) or name.lower() == "thumbs.db"


def _scan_folder(root: Path) -> tuple[int, int, list[Path]]:
    images = files = 0
    parquets: list[Path] = []
    stack = [(root, 0)]
    while stack and files < SCAN_LIMIT:
        folder, depth = stack.pop()
        try:
            entries = sorted(os.scandir(folder), key=lambda e: e.name)
        except OSError:
            continue
        for entry in entries:
            if _hidden(entry.name) or entry.is_symlink():
                continue
            if entry.is_dir(follow_symlinks=False):
                if depth < SCAN_DEPTH:
                    stack.append((Path(entry.path), depth + 1))
            elif entry.is_file(follow_symlinks=False):
                files += 1
                images += Path(entry.name).suffix.lower() in IMAGE_SUFFIXES
                if Path(entry.name).suffix.lower() in (".parquet", ".pq"):
                    parquets.append(Path(entry.path))
                if files >= SCAN_LIMIT:
                    break
    return images, files, parquets


def _has_image_column(path: Path) -> bool:
    try:
        schema = pq.read_schema(path)
    except (OSError, pa.ArrowInvalid):
        return False
    for name in schema.names:
        kind = schema.field(name).type
        if pa.types.is_struct(kind):
            fields = {kind.field(i).name: kind.field(i).type for i in range(kind.num_fields)}
            data = fields.get("bytes")
            if data is not None and (pa.types.is_binary(data) or pa.types.is_large_binary(data)):
                return True
    return False


def image_source(path: Path) -> ImageSource | None:
    """The image-set reading of `path` by R-VIS-SRC-01 to 03, or None."""
    if path.is_dir():
        images, files, parquets = _scan_folder(path)
        if images and images >= IMAGE_SHARE * files:
            more = "+" if files >= SCAN_LIMIT else ""
            return ImageSource(
                "folder",
                "R-VIS-SRC-01",
                "folder of images",
                f"{images:,}{more} of its {files:,}{more} files are images (R-VIS-SRC-01): it is "
                "imported as an image set, one row per image.",
                images,
            )
        held = sum(_has_image_column(p) for p in parquets[:PARQUET_LIMIT]) if not images else 0
        if held and held >= IMAGE_SHARE * min(len(parquets), PARQUET_LIMIT):
            return ImageSource(
                "folder",
                "R-VIS-SRC-04",
                "folder of Parquet files with an image column",
                f"{held:,} of its {len(parquets):,} Parquet files hold an image column "
                "(R-VIS-SRC-04, the Hugging Face layout): it is imported as one image set, each "
                "file's split read from its name (R-VIS-LAY-06).",
                0,
            )
        return None
    if not path.is_file():
        return None
    suffix = path.suffix.lower()
    if suffix in (".parquet", ".pq"):
        if _has_image_column(path):
            return ImageSource(
                "parquet",
                "R-VIS-SRC-03",
                "Parquet file with an image column",
                "A column holds image bytes (R-VIS-HF-01, the Hugging Face layout): it can be "
                "imported as an image set (R-VIS-SRC-03), or as a table.",
                0,
            )
        return None
    if not zipfile.is_zipfile(path):
        return None
    try:
        with zipfile.ZipFile(path) as archive:
            names = [n for n in archive.namelist()[: SCAN_LIMIT * 2] if not n.endswith("/")]
    except (zipfile.BadZipFile, OSError):
        return None
    if any(n.startswith(("xl/", "word/", "ppt/")) or n == "[Content_Types].xml" for n in names):
        return None  # an Office file (R-DET-01 of 082 reads it)
    members = [n for n in names if not any(_hidden(p) for p in PurePosixPath(n).parts)]
    images = sum(PurePosixPath(n).suffix.lower() in IMAGE_SUFFIXES for n in members)
    if images and images >= IMAGE_SHARE * len(members):
        return ImageSource(
            "zip",
            "R-VIS-SRC-02",
            "ZIP of images",
            f"{images:,} of its {len(members):,} members are images (R-VIS-SRC-02): it is "
            "imported as an image set, read in memory, never extracted under its names.",
            images,
        )
    return None


__all__ = ["IMAGE_SUFFIXES", "ImageSource", "image_source"]
