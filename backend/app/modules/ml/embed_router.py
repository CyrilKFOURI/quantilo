"""Sentence embeddings for free text (091): whether they can be chosen, and the model download.

The download is never started by a training: the page asks first, with the size, then calls
`POST …/text-embeddings/download`, which runs in a background thread; the page polls the status."""

import threading
from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel
from quantilo.ml import embed

router = APIRouter(tags=["module:ml"])

_state: dict[str, str | None] = {"download": None, "error": None}
_lock = threading.Lock()


class EmbeddingsStatus(BaseModel):
    available: bool
    installed: bool
    downloaded: bool
    reason: str | None  # why "Sentence embeddings" is greyed
    model: str
    size: str
    license: str
    install: str  # the command that adds the extra
    download: Literal["idle", "running", "failed"]
    error: str | None


def _status() -> EmbeddingsStatus:
    found = embed.status()
    state = _state["download"]
    return EmbeddingsStatus(
        available=found.available,
        installed=found.installed,
        downloaded=found.downloaded,
        reason=found.reason,
        model=found.model,
        size=found.size,
        license=found.license,
        install=embed.INSTALL,
        download="running" if state == "running" else "failed" if state == "failed" else "idle",
        error=_state["error"],
    )


@router.get("/text-embeddings")
def text_embeddings() -> EmbeddingsStatus:
    """Installed? Downloaded? Otherwise the reason the option is greyed."""
    return _status()


def _download() -> None:
    try:
        embed.download()
        _state["download"], _state["error"] = None, None
    except Exception as error:
        _state["download"], _state["error"] = "failed", str(error)


@router.post("/text-embeddings/download", status_code=202)
def download_text_embeddings() -> EmbeddingsStatus:
    """Fetch the model's weights once, after the user confirmed (size shown)."""
    with _lock:
        found = embed.status()
        if found.installed and not found.downloaded and _state["download"] != "running":
            _state["download"], _state["error"] = "running", None
            threading.Thread(target=_download, daemon=True, name="embed-download").start()
        elif not found.installed:
            _state["error"] = found.reason
    return _status()
