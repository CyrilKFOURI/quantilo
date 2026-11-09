from importlib.metadata import PackageNotFoundError, version

from fastapi import APIRouter, Request

from app.core import db
from app.core.storage import Workspace
from app.schemas.health import HealthOut

router = APIRouter(tags=["health"])


def app_version() -> str:
    try:
        return version("quantilo-app")
    except PackageNotFoundError:
        return "0.0.0"


@router.get("/health")
def health(request: Request) -> HealthOut:
    workspace: Workspace = request.app.state.workspace
    db_ok = db.is_healthy(workspace.db_path)
    return HealthOut(
        status="ok" if db_ok else "degraded",
        version=app_version(),
        db="ok" if db_ok else "error",
    )
