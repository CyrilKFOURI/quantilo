"""Join the open dataset with another one (089, docs/features/089-jointures/spec.md).

The engine (`quantilo.model.pair`) proposes the keys, checks them on every row and writes the
result; this service reads the two datasets, and imports the result like the model's analysis
datasets (088): types and roles copied from the sources, filed in the open dataset's project."""

import duckdb
from quantilo import model as engine
from quantilo.catalog import ColumnChange
from quantilo.ingestion.options import ImportOptions
from quantilo.model.build import Measure, OutputColumn
from quantilo.model.keys import TableInput

from app.core.errors import AppError
from app.schemas import joins as j
from app.schemas.model import CreatedOut
from app.services.columns.service import ColumnService
from app.services.datasets.service import DatasetService
from app.services.model.service import ModelService, _check_out, _reasons, _safe_name
from app.services.projects.service import ProjectService

_MODE_LABEL = {"lookup": "with", "expand": "with", "summarise": "summary of"}


class JoinService:
    def __init__(
        self,
        datasets: DatasetService,
        columns: ColumnService,
        projects: ProjectService,
        model: ModelService,
    ) -> None:
        self.datasets = datasets
        self.columns = columns
        self.projects = projects
        self.model = model

    def _table(self, dataset_id: str) -> TableInput:
        record = self.datasets.get(dataset_id)
        if record.status != "ready":
            raise AppError("This dataset is not ready yet.", code="not_ready", status_code=409)
        return TableInput(
            dataset_id, record.name, record.storage_path, self.columns.columns(dataset_id)
        )

    def _pair(self, left_id: str, other_id: str) -> tuple[TableInput, TableInput]:
        if left_id == other_id:
            raise AppError(
                "Pick another dataset: a dataset is not joined with itself.",
                code="same_dataset",
                status_code=422,
            )
        return self._table(left_id), self._table(other_id)

    def _project_of(self, dataset_id: str) -> str | None:
        for project in self.projects.list_all():
            if dataset_id in project.dataset_ids:
                return project.id
        return None

    def suggest(self, left_id: str, body: j.JoinSuggestIn) -> j.JoinSuggestOut:
        left, right = self._pair(left_id, body.other_id)
        connection = duckdb.connect()
        try:
            keys = engine.suggest(connection, left, right)
            profile = engine.profile_table(connection, right)
        finally:
            connection.close()
        linking = {c for key in keys for c in key.right_columns}
        picked = set(engine.default_columns(right, linking, profile))
        project = self._project_of(left_id)
        return j.JoinSuggestOut(
            keys=[
                j.JoinKeyOut(
                    left_columns=key.left_columns,
                    right_columns=key.right_columns,
                    mode=key.mode,
                    cardinality=key.cardinality,
                    confidence=key.confidence,
                    matched_share=key.matched_share,
                    reasons=_reasons(key.reasons),
                )
                for key in keys
            ],
            other_columns=[
                j.JoinColumnOut(name=view.name, type=view.type, default_pick=view.name in picked)
                for view in sorted(right.columns, key=lambda v: v.position)
            ],
            same_project=project is not None and project == self._project_of(body.other_id),
        )

    def _measures(self, body: j.JoinIn) -> list[Measure] | None:
        if body.measures is None:
            return None
        return [Measure(item.column, item.function) for item in body.measures]

    def _preview(
        self, connection: duckdb.DuckDBPyConnection, left_id: str, body: j.JoinIn
    ) -> tuple[engine.PairPreview, TableInput, TableInput]:
        left, right = self._pair(left_id, body.other_id)
        try:
            found = engine.preview_pair(
                connection,
                left,
                right,
                body.left_columns,
                body.right_columns,
                body.mode,
                body.join_type,
                body.columns,
                self._measures(body),
            )
        except engine.ModelError as error:
            raise AppError(error.message, code=error.code, status_code=422) from None
        return found, left, right

    @staticmethod
    def _name(found: engine.PairPreview, left: TableInput, right: TableInput) -> str:
        if found.mode == "expand":
            return f"{right.name} · with {left.name}"
        return f"{left.name} · {_MODE_LABEL[found.mode]} {right.name}"

    def preview(self, left_id: str, body: j.JoinIn) -> j.JoinPreviewOut:
        connection = duckdb.connect()
        try:
            found, left, right = self._preview(connection, left_id, body)
        finally:
            connection.close()
        return j.JoinPreviewOut(
            mode=found.mode,
            join_type=found.join_type,
            rows_before=found.rows_before,
            rows_after=found.rows_after,
            columns=[column.name for column in found.columns],
            added=found.added,
            rows=found.rows,
            check=_check_out(found.check, found.warnings),
            modes=[j.ModeOut(mode=o.mode, reason=o.reason) for o in found.modes],
            reasons=_reasons(found.reasons),
            warnings=_reasons(found.warnings),
            refused=_reasons(found.refused),
            suggested_name=self._name(found, left, right),
        )

    def create(self, left_id: str, body: j.JoinIn) -> CreatedOut:
        """Written only after the user confirmed the preview (the page asks)."""
        left, right = self._pair(left_id, body.other_id)
        connection = duckdb.connect()
        try:
            probe, _, _ = self._preview(connection, left_id, body)
            name = body.name or self._name(probe, left, right)
            target = self.datasets.new_upload_path(f"{_safe_name(name)}.parquet")
            try:
                found = engine.write_pair(
                    connection,
                    left,
                    right,
                    body.left_columns,
                    body.right_columns,
                    target,
                    body.mode,
                    body.join_type,
                    body.columns,
                    self._measures(body),
                )
            except engine.ModelError as error:
                raise AppError(error.message, code=error.code, status_code=422) from None
        finally:
            connection.close()
        tables = {left.id: left, right.id: right}
        computed = set(found.added) if found.mode == "summarise" else set()
        changes = [
            change
            for change in (
                _change(tables, column) for column in found.columns if column.name not in computed
            )
            if change is not None
        ]
        self.model._pending[str(self.datasets.resolve(str(target)))] = changes
        record, _ = self.datasets.start_import(str(target), name, ImportOptions())
        project = self._project_of(left_id)
        if project is not None:
            self.projects.move([record.id], project)
        return CreatedOut(dataset_id=record.id)


def _change(tables: dict[str, TableInput], column: OutputColumn) -> ColumnChange | None:
    """The source column's type and role for the joined column (R-BLD-04); counts and sums of
    `summarise` are left to the type detection."""
    table = tables.get(column.table_id)
    view = next((v for v in table.columns if v.name == column.column), None) if table else None
    if view is None:
        return None
    return ColumnChange(
        name=column.name,
        type=view.type,
        role=view.role or "",
        order=view.order or [],
        excluded=view.excluded,
    )
