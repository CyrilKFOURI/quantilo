"""Join the open dataset with another one (089, docs/features/089-jointures/spec.md).

The engine (`quantilo.model.pair`) proposes the keys, checks them on every row and writes the
result; this service reads the two datasets, and imports the result like the model's analysis
datasets (088): types and roles copied from the sources, filed in the open dataset's project."""

import json
from pathlib import Path

import duckdb
from quantilo import model as engine
from quantilo.catalog import ColumnChange
from quantilo.errors import NotFoundError
from quantilo.ingestion.options import ImportOptions
from quantilo.model.build import Measure, OutputColumn
from quantilo.model.keys import TableInput

from app.core.errors import AppError
from app.schemas import joins as j
from app.schemas.model import CreatedOut
from app.services.columns.service import ColumnService
from app.services.datasets.repository import now
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
            found = self._write(connection, left, right, body, target)
        finally:
            connection.close()
        self.model._pending[str(self.datasets.resolve(str(target)))] = self._changes(
            left, right, found
        )
        record, _ = self.datasets.start_import(str(target), name, ImportOptions())
        project = self._project_of(left_id)
        if project is not None:
            self.projects.move([record.id], project)
        self._record(record.id, project, left, right, body, found)
        return CreatedOut(dataset_id=record.id)

    def _write(
        self,
        connection: duckdb.DuckDBPyConnection,
        left: TableInput,
        right: TableInput,
        body: j.JoinIn,
        target: Path,
    ) -> engine.PairPreview:
        try:
            return engine.write_pair(
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

    @staticmethod
    def _changes(
        left: TableInput, right: TableInput, found: engine.PairPreview
    ) -> list[ColumnChange]:
        tables = {left.id: left, right.id: right}
        computed = set(found.added) if found.mode == "summarise" else set()
        return [
            change
            for change in (
                _change(tables, column) for column in found.columns if column.name not in computed
            )
            if change is not None
        ]

    # --- C1: the lineage of a join, out of date and rebuilt (089, R-JN-06) --------------------

    def _record(
        self,
        dataset_id: str,
        project_id: str | None,
        left: TableInput,
        right: TableInput,
        body: j.JoinIn,
        found: engine.PairPreview,
    ) -> None:
        stamps = []
        for table in (left, right):
            source = self.datasets.get(table.id)
            stamps.append(
                {
                    "id": source.id,
                    "name": source.name,
                    "imported_at": source.imported_at,
                    "types_version": source.types_version,
                }
            )
        spec = {"left_id": left.id, "join": body.model_dump(mode="json")}
        report = {"rows_before": found.rows_before, "rows_after": found.rows_after}
        stamp = now()
        with self.model._db() as connection, connection:
            connection.execute(
                "INSERT INTO model_builds (dataset_id, project_id, kind, spec, sources, report, "
                "created_at, updated_at) VALUES (?, ?, 'join', ?, ?, ?, ?, ?) "
                "ON CONFLICT (dataset_id) DO UPDATE SET spec = excluded.spec, "
                "sources = excluded.sources, report = excluded.report, "
                "updated_at = excluded.updated_at",
                (
                    dataset_id,
                    project_id,
                    json.dumps(spec),
                    json.dumps(stamps),
                    json.dumps(report),
                    stamp,
                    stamp,
                ),
            )

    def lineage(self, dataset_id: str) -> j.JoinLineageOut:
        """R-JN-06 (C1): what a joined dataset was made of, and why it is out of date: a source
        reimported or retyped since, or gone (then it cannot be rebuilt)."""
        with self.model._db() as connection:
            row = connection.execute(
                "SELECT * FROM model_builds WHERE dataset_id = ? AND kind = 'join'", (dataset_id,)
            ).fetchone()
        if row is None:
            raise NotFoundError("Join")
        spec = json.loads(row["spec"])
        why: list[str] = []
        can = True
        sources = []
        for source in json.loads(row["sources"]):
            try:
                record = self.datasets.get(source["id"])
            except NotFoundError:
                why.append(f"{source['name']} was deleted")
                can = False
                sources.append(j.JoinSourceOut(id=source["id"], name=source["name"], exists=False))
                continue
            sources.append(j.JoinSourceOut(id=record.id, name=record.name, exists=True))
            if record.imported_at != source["imported_at"]:
                why.append(f"{record.name} was reimported")
            elif record.types_version != source["types_version"]:
                why.append(f"{record.name}: column types changed")
        join = spec["join"]
        return j.JoinLineageOut(
            dataset_id=dataset_id,
            left_id=spec["left_id"],
            other_id=join["other_id"],
            left_columns=join["left_columns"],
            right_columns=join["right_columns"],
            mode=join.get("mode"),
            sources=sources,
            out_of_date=why,
            can_rebuild=can,
            updated_at=row["updated_at"],
        )

    def rebuild(self, dataset_id: str) -> CreatedOut:
        """The same join on the sources as they are now, written over the dataset and imported
        again (the previous version is kept, 031)."""
        found = self.lineage(dataset_id)
        if not found.can_rebuild:
            raise AppError(
                "A dataset it was made from was deleted: make the join again.",
                code="source_gone",
                status_code=409,
            )
        with self.model._db() as connection:
            row = connection.execute(
                "SELECT spec, project_id FROM model_builds WHERE dataset_id = ?", (dataset_id,)
            ).fetchone()
        body = j.JoinIn.model_validate(json.loads(row["spec"])["join"])
        left, right = self._pair(found.left_id, body.other_id)
        record = self.datasets.get(dataset_id)
        target = Path(record.source_path)
        partial = target.with_name(target.name + ".partial")
        connection = duckdb.connect()
        try:
            written = self._write(connection, left, right, body, partial)
        finally:
            connection.close()
        partial.replace(target)
        self.model._pending[str(self.datasets.resolve(str(target)))] = self._changes(
            left, right, written
        )
        self.datasets.reimport(dataset_id)
        self._record(dataset_id, row["project_id"], left, right, body, written)
        return CreatedOut(dataset_id=dataset_id)


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
