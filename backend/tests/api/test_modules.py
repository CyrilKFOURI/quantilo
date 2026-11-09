from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from app.modules.registry import MODULES, Module, mount_modules


def demo_router() -> APIRouter:
    router = APIRouter()

    @router.get("")
    def ping() -> dict[str, str]:
        return {"message": "pong"}

    return router


def test_modules_are_registered() -> None:
    assert [module.id for module in MODULES] == [
        "timeseries",
        "ml",
        "segments",
        "kpi",
        "synthetic",
        "experiments",
        "agent",
        "recipes",
    ]


def test_module_ids_are_unique() -> None:
    ids = [module.id for module in MODULES]
    assert len(ids) == len(set(ids))


def test_unregistered_module_is_not_mounted() -> None:
    app = FastAPI()
    mount_modules(app, "/api/v1", modules=[])

    assert TestClient(app).get("/api/v1/modules/demo").status_code == 404


def test_mount_uses_module_id_as_path_and_runs_setup() -> None:
    app = FastAPI()
    calls: list[FastAPI] = []
    mount_modules(app, "/api/v1", modules=[Module("demo", demo_router(), calls.append)])

    assert TestClient(app).get("/api/v1/modules/demo").json() == {"message": "pong"}
    assert calls == [app]
