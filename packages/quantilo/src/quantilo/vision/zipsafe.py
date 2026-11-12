"""Reading the images of a ZIP archive safely (104, § 3.1 and § 11 of the study).

The archive is checked from its central directory before anything is decompressed, and its members
are read **into memory, never extracted under their own names**: the store names files by their
SHA-256 (§ 3.3), so a hostile name cannot reach the disk at all. Rules, each with its code:

- R-VIS-ZIP-01: a member name is absolute (`/x`, a drive letter, a network path) or climbs out
  (`..`): the whole archive is refused (zip slip).
- R-VIS-ZIP-02: more than 50,000 members: refused.
- R-VIS-ZIP-03: more than 4 GiB once decompressed (declared sizes): refused.
- R-VIS-ZIP-04: a member decompresses more than 100 times its compressed size: that member is
  skipped (zip bomb).
- R-VIS-ZIP-05: an archive inside the archive: not opened, listed.
- R-VIS-ZIP-06: links, encrypted members, hidden files and `__MACOSX`: skipped, listed.
- R-VIS-ZIP-07: a damaged member (bad CRC, unknown compression): skipped, listed.

A member is read with a byte counter: reading stops at its declared size (plus nothing), so a
member that lies about its size cannot fill the memory.
"""

from __future__ import annotations

import os
import re
import stat
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from quantilo.ingestion.errors import IngestionError

MAX_MEMBERS = 50_000
MAX_TOTAL_BYTES = 4 * 1024**3
MAX_RATIO = 100.0
MAX_MEMBER_BYTES = 256 * 1024 * 1024  # as one image file (R-VIS-ING-03)
_CHUNK = 1024 * 1024
_DRIVE = re.compile(r"^[A-Za-z]:")


class UnsafeArchive(IngestionError):
    """A ZIP refused as a whole; `code` is the R-VIS-ZIP rule."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{message} ({code})")
        self.rule = code
        self.details = {"rule": code}


@dataclass(frozen=True)
class Member:
    name: str  # the path inside the archive, with "/"
    size: int  # declared, decompressed
    compressed: int


@dataclass(frozen=True)
class SkippedMember:
    name: str
    code: str
    reason: str


@dataclass
class ZipPlan:
    """The members that may be read, and those skipped with their rule."""

    path: Path
    members: list[Member] = field(default_factory=list)
    skipped: list[SkippedMember] = field(default_factory=list)
    total_bytes: int = 0


def _unsafe_name(name: str) -> bool:
    text = name.replace("\\", "/")
    if text.startswith("/") or _DRIVE.match(text) or ":" in text:
        return True
    return any(part == ".." for part in text.split("/"))


def _is_link(info: zipfile.ZipInfo) -> bool:
    mode = (info.external_attr >> 16) & 0xFFFF
    return stat.S_ISLNK(mode)


def _hidden(name: str) -> bool:
    parts = PurePosixPath(name).parts
    return any(p.startswith(".") or p == "__MACOSX" for p in parts)


def check_zip(path: str | os.PathLike[str]) -> ZipPlan:
    """Check an archive from its central directory (see the module); raises `UnsafeArchive`
    for R-VIS-ZIP-01 to 03, `IngestionError` when the file is not a ZIP."""
    path = Path(path)
    try:
        archive = zipfile.ZipFile(path)
    except (zipfile.BadZipFile, OSError) as error:
        raise IngestionError(f"This ZIP archive cannot be read: {error}.") from None
    with archive:
        infos = archive.infolist()
    if len(infos) > MAX_MEMBERS:
        raise UnsafeArchive(
            "R-VIS-ZIP-02",
            f"The archive holds {len(infos):,} members, over the {MAX_MEMBERS:,} limit",
        )
    for info in infos:
        if _unsafe_name(info.filename):
            raise UnsafeArchive(
                "R-VIS-ZIP-01",
                f"The archive was refused: the member {info.filename!r} would be written outside "
                "the folder (absolute path or '..')",
            )
    plan = ZipPlan(path=path)
    total = sum(info.file_size for info in infos if not info.is_dir())
    if total > MAX_TOTAL_BYTES:
        raise UnsafeArchive(
            "R-VIS-ZIP-03",
            f"The archive decompresses to {total / 2**30:,.1f} GiB, over the "
            f"{MAX_TOTAL_BYTES // 2**30} GiB limit",
        )
    plan.total_bytes = total
    for info in infos:
        name = info.filename.replace("\\", "/")
        if info.is_dir():
            continue
        if _is_link(info) or info.flag_bits & 0x1 or _hidden(name):
            what = (
                "a link"
                if _is_link(info)
                else ("encrypted" if info.flag_bits & 0x1 else "hidden or system file")
            )
            plan.skipped.append(SkippedMember(name, "R-VIS-ZIP-06", f"Skipped: {what}."))
            continue
        if name.lower().endswith((".zip", ".7z", ".rar", ".tar", ".gz", ".tgz")):
            plan.skipped.append(
                SkippedMember(name, "R-VIS-ZIP-05", "An archive inside the archive is not opened.")
            )
            continue
        ratio = info.file_size / max(1, info.compress_size)
        if info.file_size > 0 and ratio > MAX_RATIO:
            plan.skipped.append(
                SkippedMember(
                    name,
                    "R-VIS-ZIP-04",
                    f"Decompresses {ratio:,.0f} times its size (over {MAX_RATIO:.0f}): skipped as "
                    "a possible zip bomb.",
                )
            )
            continue
        if info.file_size > MAX_MEMBER_BYTES:
            plan.skipped.append(
                SkippedMember(
                    name,
                    "R-VIS-ING-03",
                    f"{info.file_size / 2**20:,.0f} MiB, over the "
                    f"{MAX_MEMBER_BYTES // 2**20} MiB limit for one image.",
                )
            )
            continue
        plan.members.append(Member(name, info.file_size, info.compress_size))
    return plan


def read_member(archive: zipfile.ZipFile, member: Member, limit: int | None = None) -> bytes:
    """The bytes of one member, reading at most its declared size (or `limit`, for a head)."""
    cap = member.size if limit is None else min(limit, member.size)
    chunks: list[bytes] = []
    got = 0
    with archive.open(member.name) as handle:
        while got < cap:
            chunk = handle.read(min(_CHUNK, cap - got))
            if not chunk:
                break
            chunks.append(chunk)
            got += len(chunk)
    return b"".join(chunks)


def iter_members(plan: ZipPlan) -> Iterator[tuple[Member, zipfile.ZipFile]]:
    """The readable members with the open archive (closed at the end)."""
    with zipfile.ZipFile(plan.path) as archive:
        for member in plan.members:
            yield member, archive


def is_zip(path: str | os.PathLike[str]) -> bool:
    """A ZIP by its signature (PK\\x03\\x04, or an empty archive's PK\\x05\\x06)."""
    with Path(path).open("rb") as handle:
        return handle.read(4) in (b"PK\x03\x04", b"PK\x05\x06")


__all__ = [
    "MAX_MEMBERS",
    "MAX_RATIO",
    "MAX_TOTAL_BYTES",
    "Member",
    "SkippedMember",
    "UnsafeArchive",
    "ZipPlan",
    "check_zip",
    "is_zip",
    "iter_members",
    "read_member",
]
