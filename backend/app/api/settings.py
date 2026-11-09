from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response
from quantilo.gpu import GpuStatus

from app.services.settings.service import AppSettings, SettingsService, StorageInfo

router = APIRouter(tags=["settings"])


def get_settings_service(request: Request) -> SettingsService:
    return request.app.state.settings


Service = Annotated[SettingsService, Depends(get_settings_service)]


@router.get("/settings")
def read_settings(service: Service) -> AppSettings:
    return service.get()


@router.put("/settings")
def write_settings(settings: AppSettings, service: Service) -> AppSettings:
    """Detection thresholds apply to the next detections (imports, "Re-run detection")."""
    return service.update(settings)


@router.get("/settings/gpu")
def gpu(service: Service, refresh: bool = False) -> GpuStatus:
    """The graphics card the deep models could use (`refresh`: look again)."""
    return service.gpu(refresh)


@router.get("/storage")
def storage(service: Service) -> StorageInfo:
    return service.storage()


@router.post("/storage/clear-caches", status_code=204)
def clear_caches(service: Service) -> Response:
    service.clear_caches()
    return Response(status_code=204)
