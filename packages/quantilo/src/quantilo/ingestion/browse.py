"""List folders and importable files, optionally restricted to allowed roots."""

import stat
import string
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from quantilo.ingestion.errors import IngestionError
from quantilo.ingestion.options import Format
from quantilo.ingestion.paths import is_within, to_local_path
from quantilo.ingestion.sources import detect_format


@dataclass
class Entry:
    name: str
    path: str
    kind: Literal["dir", "file"]
    format: Format | None = None
    size: int | None = None


@dataclass
class Listing:
    path: str
    parent: str | None
    roots: list[str]
    entries: list[Entry]


HIDDEN_ATTRIBUTES = stat.FILE_ATTRIBUTE_HIDDEN | stat.FILE_ATTRIBUTE_SYSTEM


def is_hidden(path: Path) -> bool:
    """Dot files, and on Windows hidden or system entries (legacy junctions like "My Documents")."""
    if path.name.startswith((".", "$")):
        return True
    attributes = getattr(path.stat(follow_symlinks=False), "st_file_attributes", 0)
    return bool(attributes & HIDDEN_ATTRIBUTES)


def default_roots() -> list[Path]:
    roots = [Path.home()]
    if sys.platform == "win32":
        roots += [Path(f"{letter}:\\") for letter in string.ascii_uppercase]
        roots = [root for root in roots if root.exists()]
    else:
        roots.append(Path("/"))
    return roots


def list_directory(path: Path | None, restricted_to: list[Path] | None) -> Listing:
    roots = (
        [to_local_path(str(root)) for root in restricted_to] if restricted_to else default_roots()
    )
    current = to_local_path(str(path or roots[0]))
    if restricted_to and not is_within(current, roots):
        raise IngestionError("This folder is outside the allowed data folder.")
    if not current.is_dir():
        raise IngestionError(f"Folder not found: {current}")

    entries: list[Entry] = []
    try:
        children = sorted(current.iterdir(), key=lambda child: child.name.lower())
    except PermissionError as error:
        raise IngestionError(f"Access denied: {current}") from error
    for child in children:
        try:
            if is_hidden(child):
                continue
            if child.is_dir():
                entries.append(Entry(child.name, str(child), "dir"))
            elif file_format := detect_format(child):
                entries.append(
                    Entry(child.name, str(child), "file", file_format, child.stat().st_size)
                )
        except OSError:
            continue

    parent: Path | None = current.parent if current.parent != current else None
    if restricted_to and parent is not None and not is_within(parent, roots):
        parent = None
    entries.sort(key=lambda entry: entry.kind != "dir")
    return Listing(
        path=str(current),
        parent=str(parent) if parent else None,
        roots=[str(root) for root in roots],
        entries=entries,
    )
