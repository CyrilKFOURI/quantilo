"""Time series analyses kept with "Save analysis" (030): a named configuration of the page, per
dataset, in SQLite. They are the time series steps of a recipe."""

import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response
from pydantic import BaseModel, Field
from quantilo.recipes.model import SeriesConfig

from app.core import db
from app.core.errors import AppError
from app.core.storage import Workspace
from app.services.datasets.repository import now
from app.services.datasets.service import DatasetService, NotFoundError


@dataclass
class SavedAnalysis:
    id: str
    name: str
    config: SeriesConfig
    origin: str | None  # the recipe that added it
    created_at: str


def _saved(row: sqlite3.Row) -> SavedAnalysis:
    return SavedAnalysis(
        row["id"],
        row["name"],
        SeriesConfig.model_validate_json(row["config"]),
        row["origin"],
        row["created_at"],
    )


class AnalysisService:
    def __init__(self, workspace: Workspace, datasets: DatasetService) -> None:
        self.workspace = workspace
        self.datasets = datasets

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        connection = db.connect(self.workspace.db_path)
        try:
            yield connection
        finally:
            connection.close()

    def all(self, dataset_id: str) -> list[SavedAnalysis]:
        self.datasets.get(dataset_id)
        with self._db() as connection:
            rows = connection.execute(
                "SELECT * FROM ts_analyses WHERE dataset_id = ? ORDER BY created_at DESC",
                (dataset_id,),
            ).fetchall()
        return [_saved(row) for row in rows]

    def save(
        self, dataset_id: str, name: str, config: SeriesConfig, origin: str | None = None
    ) -> SavedAnalysis:
        """A new analysis; one with the same name and origin (a recipe run again) is replaced."""
        self.datasets.get(dataset_id)
        name = name.strip()
        if not name:
            raise AppError("Give the analysis a name.", code="invalid_name", status_code=422)
        with self._db() as connection, connection:
            if origin is not None:
                connection.execute(
                    "DELETE FROM ts_analyses WHERE dataset_id = ? AND name = ? AND origin = ?",
                    (dataset_id, name, origin),
                )
            analysis_id = uuid.uuid4().hex[:12]
            connection.execute(
                "INSERT INTO ts_analyses (id, dataset_id, name, config, origin, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (analysis_id, dataset_id, name, config.model_dump_json(), origin, now()),
            )
            row = connection.execute(
                "SELECT * FROM ts_analyses WHERE id = ?", (analysis_id,)
            ).fetchone()
        return _saved(row)

    def delete(self, dataset_id: str, analysis_id: str) -> None:
        with self._db() as connection, connection:
            deleted = connection.execute(
                "DELETE FROM ts_analyses WHERE id = ? AND dataset_id = ?",
                (analysis_id, dataset_id),
            ).rowcount
        if not deleted:
            raise NotFoundError(f"Analysis {analysis_id!r}")


router = APIRouter(tags=["module:timeseries"])


def get_analyses(request: Request) -> AnalysisService:
    return request.app.state.ts_analyses


Analyses = Annotated[AnalysisService, Depends(get_analyses)]


class SaveAnalysis(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    config: SeriesConfig


@router.get("/datasets/{dataset_id}/analyses")
def list_analyses(dataset_id: str, service: Analyses) -> list[SavedAnalysis]:
    """Analyses kept on this dataset, newest first."""
    return service.all(dataset_id)


@router.post("/datasets/{dataset_id}/analyses", status_code=201)
def save_analysis(dataset_id: str, body: SaveAnalysis, service: Analyses) -> SavedAnalysis:
    return service.save(dataset_id, body.name, body.config)


@router.delete("/datasets/{dataset_id}/analyses/{analysis_id}", status_code=204)
def delete_analysis(dataset_id: str, analysis_id: str, service: Analyses) -> Response:
    service.delete(dataset_id, analysis_id)
    return Response(status_code=204)
