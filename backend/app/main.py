"""FastAPI application: API under /api/v1, built frontend served in production."""

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from quantilo.compare.service import CompareService
from quantilo.correlations.service import CorrelationService
from quantilo.profiling.service import ProfilingService
from quantilo.regression.service import RegressionService
from quantilo.rows.service import RowService
from quantilo.text.service import TextService
from quantilo.variable.service import VariableService

from app.api import (
    changes,
    columns,
    compare,
    correlations,
    datasets,
    filters,
    health,
    images,
    jobs,
    joins,
    model,
    pins,
    profiling,
    projects,
    regression,
    report,
    rows,
    sources,
    text,
    variable,
)
from app.api import settings as settings_api
from app.api.errors import register_error_handlers
from app.api.security import protect
from app.config import Settings
from app.core import db
from app.core.jobs import JobRunner, ThreadJobRunner
from app.core.storage import Workspace
from app.modules.registry import mount_modules
from app.services.columns.service import ColumnService
from app.services.datasets.service import DatasetService
from app.services.filters.saved import SavedFilterService
from app.services.image_sets import ImageSetService
from app.services.joins import JoinService
from app.services.model.service import ModelService
from app.services.pins.service import PinService
from app.services.projects.service import ProjectService
from app.services.report.service import ReportService
from app.services.settings.service import SettingsService
from app.services.versions.service import VersionService

API_PREFIX = "/api/v1"


def create_app(settings: Settings | None = None, job_runner: JobRunner | None = None) -> FastAPI:
    settings = settings or Settings()
    workspace = Workspace(settings.home)
    runner = job_runner or ThreadJobRunner()
    dataset_service = DatasetService(workspace, settings, runner)
    column_service = ColumnService(workspace, dataset_service)
    dataset_service.after_import.append(column_service.detect_storage)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        workspace.ensure()
        connection = db.connect(workspace.db_path)
        try:
            db.apply_migrations(connection)
        finally:
            connection.close()
        dataset_service.recover_interrupted()
        yield
        runner.shutdown()

    app = FastAPI(title="Quantilo", version=health.app_version(), lifespan=lifespan)
    app.state.workspace = workspace
    app.state.datasets = dataset_service
    app.state.columns = column_service
    app.state.projects = ProjectService(workspace, dataset_service)
    app.state.image_sets = ImageSetService(workspace, dataset_service)  # 106
    # 088: types of the analysis datasets, right after the detection (before 031's snapshot)
    app.state.model = ModelService(workspace, dataset_service, column_service, app.state.projects)
    dataset_service.after_import.append(app.state.model.after_import)
    # 089: join the open dataset with another one (same import path as the analysis datasets)
    app.state.joins = JoinService(
        dataset_service, column_service, app.state.projects, app.state.model
    )
    # 091: sentence embeddings are cached in the workspace (R-EMB-03)
    os.environ["QUANTILO_EMBEDDINGS_CACHE"] = str(workspace.cache_dir / "embeddings")
    app.state.profiling = ProfilingService(column_service)
    app.state.variable = VariableService(column_service)
    app.state.saved_filters = SavedFilterService(workspace, column_service)
    app.state.compare = CompareService(column_service)
    app.state.correlations = CorrelationService(column_service)
    app.state.text = TextService(column_service)  # 059
    app.state.regression = RegressionService(column_service, app.state.correlations)
    app.state.rows = RowService(column_service)
    app.state.pins = PinService(workspace, column_service)  # 099
    app.state.reports = ReportService(
        column_service,
        app.state.profiling,
        app.state.variable,
        app.state.correlations,
        app.state.compare,
        health.app_version,
        runner,
    )
    settings_service = SettingsService(workspace, dataset_service)
    column_service.thresholds = settings_service.thresholds
    settings_service.cache_clearers += [
        app.state.profiling._overviews.clear,
        app.state.variable._cache.clear,
        app.state.compare._cache.clear,
        app.state.correlations._cache.clear,
        app.state.correlations._structures.clear,
        app.state.regression.clear,
        app.state.rows._counts.clear,
    ]
    app.state.settings = settings_service
    versions = VersionService(
        workspace,
        dataset_service,
        column_service,
        settings_service.keep_policy,
        settings.keep_version_limit_mb * 1024 * 1024,
    )
    dataset_service.after_import.append(versions.record)  # after the type detection
    dataset_service.before_reimport.append(versions.keep_previous)
    settings_service.cache_clearers.append(versions.clear)
    app.state.versions = versions
    register_error_handlers(app)
    protect(app, tuple(settings.allowed_hosts))

    api = APIRouter(prefix=API_PREFIX)
    routers = (health, datasets, sources, jobs, columns, profiling, variable, filters, compare)
    others = (
        correlations,
        regression,
        rows,
        report,
        changes,
        settings_api,
        projects,
        model,
        text,
        joins,
        images,
        pins,
    )
    for module in (*routers, *others):
        api.include_router(module.router)
    app.include_router(api)
    mount_modules(app, API_PREFIX)

    if settings.static_dir and settings.static_dir.is_dir():
        serve_frontend(app, settings.static_dir)
    return app


def serve_frontend(app: FastAPI, static_dir: Path) -> None:
    """Serve the built single-page app; unknown non-API paths fall back to index.html."""
    root = static_dir.resolve()
    if (root / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=root / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> FileResponse:
        if path.startswith("api/"):
            raise HTTPException(status_code=404)
        candidate = (root / path).resolve()
        if path and candidate.is_file() and candidate.is_relative_to(root):
            return FileResponse(candidate)
        return FileResponse(root / "index.html")


app = create_app()
