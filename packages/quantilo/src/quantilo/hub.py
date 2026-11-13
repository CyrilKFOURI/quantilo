"""Verified downloads of pretrained weights (vision 114, forecasting 028 B1): one HTTPS file of a
known size and SHA-256, resumed with Range when the connection drops, written under a temporary
name and renamed only once its digest matches. Standard library only (no torch, no
huggingface_hub): nothing but the bytes listed in a manifest ever reaches the disk."""

import hashlib
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any, BinaryIO

HF = "https://huggingface.co"
ATTEMPTS = 8  # Hugging Face often resets long connections: resume with Range
TIMEOUT = 60
CHUNK = 1 << 20

Opener = Callable[[urllib.request.Request, float], Any]
Progress = Callable[[int, int], None]


class DownloadError(RuntimeError):
    """A download that cannot be trusted or did not finish."""


class Cancelled(DownloadError):
    """The user stopped the download: the partial file is kept for a later resume."""


def size_text(size: int) -> str:
    mb = size / 1_000_000
    return f"{mb / 1000:.1f} GB" if mb >= 1000 else f"{mb:.0f} MB"


def file_sha256(file: Path) -> str:
    digest = hashlib.sha256()
    with file.open("rb") as handle:
        while block := handle.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def _open(request: urllib.request.Request, timeout: float) -> Any:
    return urllib.request.urlopen(request, timeout=timeout)


def _copy(
    response: BinaryIO,
    out: BinaryIO,
    digest: Any,
    have: int,
    limit: int,
    progress: Progress | None,
    cancelled: Callable[[], bool] | None,
) -> int:
    while block := response.read(CHUNK):
        have += len(block)
        if have > limit:
            raise DownloadError(f"The server sent more than the {limit:,} bytes expected.")
        digest.update(block)
        out.write(block)
        if progress is not None:
            progress(have, limit)
        if cancelled is not None and cancelled():
            raise Cancelled("Download cancelled: the part already fetched is kept.")
    return have


def fetch(
    url: str,
    size: int,
    sha256: str,
    file: Path,
    *,
    title: str,
    user_agent: str,
    progress: Progress | None = None,
    opener: Opener | None = None,
    attempts: int = ATTEMPTS,
    pause: float = 2.0,
    resume: bool = False,
    cancelled: Callable[[], bool] | None = None,
) -> Path:
    """Fetch `url` into `file` (its `.part` first). `resume`: a `.part` left by an earlier run
    is continued with Range instead of restarted. Raises `DownloadError` (size, digest, network)
    or `Cancelled`."""
    if not url.startswith("https://"):
        raise DownloadError(f"{title}: only HTTPS downloads are allowed.")
    file.parent.mkdir(parents=True, exist_ok=True)
    part = file.with_name(file.name + ".part")
    if not resume:
        part.unlink(missing_ok=True)
    digest = hashlib.sha256()
    have = 0
    if part.exists():
        have = part.stat().st_size
        if have > size:
            part.unlink()
            have = 0
        else:
            with part.open("rb") as kept:
                while block := kept.read(CHUNK):
                    digest.update(block)
    fetch_one = opener or _open
    last: BaseException | None = None
    for attempt in range(attempts):
        if have == size:
            break
        headers = {"User-Agent": user_agent}
        if have:
            headers["Range"] = f"bytes={have}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            with fetch_one(request, TIMEOUT) as response, part.open("ab") as out:
                status = getattr(response, "status", 200)
                if have and status != 206:  # the server ignored Range: start again
                    out.truncate(0)
                    have, digest = 0, hashlib.sha256()
                have = _copy(response, out, digest, have, size, progress, cancelled)
            if have == size:
                break
            last = DownloadError(f"Connection closed after {have:,} of {size:,} bytes.")
        except Cancelled:
            raise
        except DownloadError:
            part.unlink(missing_ok=True)
            raise
        except (OSError, urllib.error.URLError) as error:
            last = error
            # what reached the disk is kept: resume from there, with its digest
            have = part.stat().st_size if part.exists() else 0
            digest = hashlib.sha256()
            if have:
                with part.open("rb") as kept:
                    while block := kept.read(CHUNK):
                        digest.update(block)
        if attempt + 1 < attempts:
            time.sleep(pause)
    if have != size:
        part.unlink(missing_ok=True)
        raise DownloadError(
            f"Download of {title} failed after {attempts} attempts ({last}). "
            "Check the connection and try again."
        )
    if digest.hexdigest() != sha256:
        part.unlink(missing_ok=True)
        raise DownloadError(
            f"{title}: the SHA-256 of the downloaded file does not match the manifest: "
            "file deleted."
        )
    os.replace(part, file)
    return file
