"""Where an image set lives on disk, addressed by content (104, § 3.3 of the study).

    workspace/images/<dataset_id>/
      files/<sha[0:2]>/<sha256>.<ext>   the original bytes; an exact copy is stored once
      thumbs/<sha256>.webp               long side 256 px, quality 80, oriented, no EXIF
      masks/<sha256>.png                 the pixel mask of that image (R-VIS-LAY-04)
    workspace/datasets/<dataset_id>.parquet   the table of images, one row per image

Original names never become paths on disk: only hexadecimal digests and a known extension do, so a
name such as `../../x` or `CON.jpg` cannot write anywhere. Writes go to a temporary file renamed in
place, so an interrupted import leaves no half-written file under a real name.
"""

from __future__ import annotations

import hashlib
import io
import os
import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

THUMB_SIDE = 256
THUMB_QUALITY = 80
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_SHA = re.compile(r"^[0-9a-f]{64}$")
_EXT = re.compile(r"^\.[a-z0-9]{1,5}$")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def image_id(digest: str) -> str:
    """The image's id: the first 16 hexadecimal digits of the file's SHA-256."""
    return digest[:16]


def check_dataset_id(dataset_id: str) -> str:
    if not _ID.match(dataset_id):
        raise ValueError(
            f"Invalid dataset id {dataset_id!r}: letters, digits, '-' and '_' only (64 at most)."
        )
    return dataset_id


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_bytes(data)
    os.replace(partial, path)


@dataclass(frozen=True)
class ImageStore:
    """The folders of one image set (see the module)."""

    workspace: Path
    dataset_id: str
    table_file: Path | None = None  # where the table goes when not the default (the app's layout)

    def __post_init__(self) -> None:
        check_dataset_id(self.dataset_id)

    @property
    def root(self) -> Path:
        return self.workspace / "images" / self.dataset_id

    @property
    def table_path(self) -> Path:
        if self.table_file is not None:
            return self.table_file
        return self.workspace / "datasets" / f"{self.dataset_id}.parquet"

    def file_rel(self, digest: str, ext: str) -> str:
        if not _SHA.match(digest) or not _EXT.match(ext):
            raise ValueError("A stored file is named by its SHA-256 and a known extension.")
        return f"files/{digest[:2]}/{digest}{ext}"

    def thumb_rel(self, digest: str) -> str:
        if not _SHA.match(digest):
            raise ValueError("A thumbnail is named by the SHA-256 of its image.")
        return f"thumbs/{digest}.webp"

    def mask_rel(self, digest: str) -> str:
        if not _SHA.match(digest):
            raise ValueError("A mask is named by the SHA-256 of its image.")
        return f"masks/{digest}.png"

    def path(self, rel: str) -> Path:
        """An absolute path from a stored relative one, confined to the set's folder."""
        target = (self.root / rel).resolve()
        if not target.is_relative_to(self.root.resolve()):
            raise ValueError(f"{rel!r} is outside the image set's folder.")
        return target

    def put_file(self, data: bytes, ext: str, digest: str | None = None) -> tuple[str, str]:
        """Store the original bytes once; (sha256, relative path)."""
        digest = digest or sha256(data)
        rel = self.file_rel(digest, ext)
        target = self.root / rel
        if not target.is_file():
            _write(target, data)
        return digest, rel

    def put_thumbnail(self, digest: str, image: Image.Image) -> str:
        """A WebP thumbnail, long side 256 px, quality 80, without any metadata."""
        rel = self.thumb_rel(digest)
        target = self.root / rel
        if not target.is_file():
            thumb = image.convert("RGB")  # a copy, so the caller's image is left as it is
            thumb.thumbnail((THUMB_SIDE, THUMB_SIDE), Image.Resampling.LANCZOS)
            thumb.info = {}
            buffer = io.BytesIO()
            thumb.save(buffer, "WEBP", quality=THUMB_QUALITY, method=2, exif=b"", icc_profile=None)
            _write(target, buffer.getvalue())
        return rel

    def put_mask(self, digest: str, mask: Image.Image) -> str:
        """The image's mask as a grey PNG (any value > 0 is the defect)."""
        rel = self.mask_rel(digest)
        grey = mask.convert("L")
        grey.info = {}
        buffer = io.BytesIO()
        grey.save(buffer, "PNG", optimize=True)
        _write(self.root / rel, buffer.getvalue())
        return rel


__all__ = ["THUMB_QUALITY", "THUMB_SIDE", "ImageStore", "check_dataset_id", "image_id", "sha256"]
