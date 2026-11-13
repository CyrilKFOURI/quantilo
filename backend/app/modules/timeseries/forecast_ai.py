"""AI forecasting models (028 phase B1): their state on this machine, the downloads of their
weights (only after the user confirmed, in the background, cancellable, resumed from the part
already fetched) and Settings > Models (delete, "Never download").

The engine (`quantilo.timeseries.ai`) holds the manifest, the checks and the files; this module
only runs the downloads as jobs and answers the page."""

import threading
from pathlib import Path
from typing import Annotated, Literal

from fastapi import APIRouter, Body, Depends, Request
from pydantic import BaseModel
from quantilo.timeseries import ai

from app.core.errors import AppError

router = APIRouter(tags=["module:timeseries"])

Download = Literal["idle", "running", "failed", "cancelled"]


class AiModelOut(BaseModel):
    key: str
    title: str
    family: str
    pretrained: bool
    state: ai.State
    reason: str | None
    size: int
    size_text: str
    licence: str
    parameters: str
    note: str
    source: str
    folder: str
    drivers: bool
    missing: list[str]
    install: str | None
    download: Download = "idle"
    progress: float = 0.0  # 0 to 1
    error: str | None = None


class AiModelsOut(BaseModel):
    models: list[AiModelOut]
    folder: str  # where the weights live: <home>/models/forecast
    offline: bool  # "Never download"
    offline_forced: bool  # set by DATA_EXPLORER_OFFLINE: not changeable here
    install: str


class OfflineIn(BaseModel):
    offline: bool


class ForecastAiService:
    def __init__(self, models_dir: Path | None = None) -> None:
        self.models_dir = models_dir
        self._downloads: dict[str, dict] = {}
        self._cancels: dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    def one(self, key: str) -> AiModelOut:
        if key not in ai.MANIFEST:
            raise AppError("Unknown AI model.", code="not_found", status_code=404)
        found = ai.status(key)
        state = self._downloads.get(key, {})
        return AiModelOut(
            **found.__dict__,
            download=state.get("status", "idle"),
            progress=state.get("progress", 1.0 if ai.downloaded(key) else 0.0),
            error=state.get("error"),
        )

    def all(self) -> AiModelsOut:
        return AiModelsOut(
            models=[self.one(key) for key in ai.MANIFEST],
            folder=str(ai.models_root() / "forecast"),
            offline=ai.offline(),
            offline_forced=ai.offline_forced(),
            install=ai.INSTALL,
        )

    def download(self, key: str) -> AiModelOut:
        """The user confirmed in the page (name, size, source shown first): fetch in the
        background. Already there, or already running: nothing new starts."""
        found = self.one(key)
        if not found.pretrained:
            raise AppError(
                "This model is trained on your series: nothing to download.", code="trained"
            )
        if found.state in ("needs_extra", "browser", "offline"):
            raise AppError(
                found.reason or "Cannot download here.", code="unavailable", status_code=409
            )
        with self._lock:
            running = self._downloads.get(key, {}).get("status") == "running"
            if not running and not ai.downloaded(key):
                self._downloads[key] = {"status": "running", "progress": 0.0, "error": None}
                self._cancels[key] = threading.Event()
                threading.Thread(
                    target=self._run, args=(key,), daemon=True, name=f"forecast-weights-{key}"
                ).start()
        return self.one(key)

    def _run(self, key: str) -> None:
        cancel = self._cancels[key]

        def progress(have: int, total: int) -> None:
            self._downloads[key]["progress"] = round(have / max(1, total), 4)

        try:
            ai.download(key, confirm=True, progress=progress, cancelled=cancel.is_set)
            self._downloads[key] = {"status": "idle", "progress": 1.0, "error": None}
        except Exception as error:  # network, size or SHA-256: said as it is
            status = "cancelled" if cancel.is_set() else "failed"
            message = None if cancel.is_set() else str(error)
            self._downloads[key] = {"status": status, "progress": 0.0, "error": message}

    def cancel(self, key: str) -> AiModelOut:
        """Stop a download: the part fetched is kept, the next download resumes from it."""
        self.one(key)
        event = self._cancels.get(key)
        if event is not None:
            event.set()
        return self.one(key)

    def delete(self, key: str) -> AiModelOut:
        found = self.one(key)
        if found.download == "running":
            raise AppError("Stop the download first.", code="downloading", status_code=409)
        ai.delete(key)
        self._downloads.pop(key, None)
        return self.one(key)

    def set_offline(self, value: bool) -> AiModelsOut:
        if ai.offline_forced() and not value:
            raise AppError(
                "Never download is set by DATA_EXPLORER_OFFLINE: change it where the app starts.",
                code="offline_forced",
                status_code=409,
            )
        ai.set_offline(value)
        return self.all()


def get_service(request: Request) -> ForecastAiService:
    return request.app.state.forecast_ai


Service = Annotated[ForecastAiService, Depends(get_service)]


@router.get("/forecast-ai")
def models(service: Service) -> AiModelsOut:
    """The AI forecasting models: whether each can run here (extra, weights, Never download,
    browser version), its size and source, a download in progress."""
    return service.all()


@router.post("/forecast-ai/{key}/download", status_code=202)
def download(key: str, service: Service) -> AiModelOut:
    """Fetch a pretrained model's weights once, after the user confirmed (size and source shown
    first). Your data is not sent anywhere: only the model's files come in."""
    return service.download(key)


@router.delete("/forecast-ai/{key}/download")
def cancel_download(key: str, service: Service) -> AiModelOut:
    """Stop a download; the part already fetched is kept and resumed next time."""
    return service.cancel(key)


@router.delete("/forecast-ai/{key}")
def delete(key: str, service: Service) -> AiModelOut:
    """Delete a downloaded model's files (Settings > Models)."""
    return service.delete(key)


@router.put("/forecast-ai/offline")
def offline(service: Service, body: Annotated[OfflineIn, Body()]) -> AiModelsOut:
    """ "Never download": the app never contacts the network for a model."""
    return service.set_offline(body.offline)
