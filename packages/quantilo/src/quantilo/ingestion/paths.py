"""Turn a user-supplied path into a local absolute path, without touching the file system.

Allowlist, not denylist: on Windows only `X:\\...` drive paths are accepted, so UNC shares,
`\\\\?\\` / `\\\\.\\` device paths and any mixed-slash spelling of them are rejected. Opening
a network path would make Windows authenticate to a remote host and leak NTLM credentials,
so validation is purely lexical and happens before any file access.
"""

import os
import re
import sys
from pathlib import Path

from quantilo.ingestion.errors import IngestionError

LOCAL_ABSOLUTE = re.compile(r"^[A-Za-z]:\\" if sys.platform == "win32" else r"^/(?!/)")


def is_local_absolute(value: str) -> bool:
    return bool(LOCAL_ABSOLUTE.match(value))


def to_local_path(raw: str) -> Path:
    absolute = os.path.abspath(os.path.expanduser(raw.strip()))
    if not is_local_absolute(absolute):
        raise IngestionError("Only local paths are supported: copy network files locally first.")
    return Path(absolute)


def is_within(path: Path, roots: list[Path]) -> bool:
    """Lexical containment (paths come from `to_local_path`, so `..` is already collapsed)."""
    return any(path == root or path.is_relative_to(root) for root in roots)
