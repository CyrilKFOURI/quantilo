"""Rows and exports of a dataset, read through each column's effective type and the filters."""

import re
import shutil
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import duckdb

from quantilo.catalog import Catalog, DatasetRef
from quantilo.filters.model import Condition, filters_key
from quantilo.filters.scope import scoped_source, where_for
from quantilo.rows.export import Format, write_summary, write_view
from quantilo.rows.view import (
    MAX_LIMIT,
    ROW,
    RowColumn,
    RowsColumn,
    RowsPage,
    UnknownColumnError,
    View,
    count,
    page,
    view_relation,
)
from quantilo.variable.source import SEED

COUNT_CACHE = 64
EXTENSIONS: dict[str, str] = {"csv": "csv", "parquet": "parquet", "xlsx": "xlsx", "summary": "csv"}


@dataclass
class ExportFile:
    path: Path  # alone in its own temporary directory
    filename: str

    def remove(self) -> None:
        shutil.rmtree(self.path.parent, ignore_errors=True)


class RowService:
    def __init__(self, columns: Catalog) -> None:
        self.columns = columns
        self._counts: OrderedDict[tuple[Any, ...], int] = OrderedDict()

    def _scope(
        self, dataset_id: str, conditions: list[Condition]
    ) -> tuple[DatasetRef, list[RowColumn], str]:
        dataset = self.columns.ready(dataset_id)
        views = self.columns.columns(dataset_id)
        columns = [RowColumn(v.name, v.type, v.parser, v.excluded) for v in views]
        source = scoped_source(dataset.storage_path, where_for(views, conditions), numbered=True)
        return dataset, columns, source

    def rows(
        self,
        dataset_id: str,
        view: View,
        offset: int = 0,
        limit: int = 100,
        conditions: list[Condition] | None = None,
    ) -> RowsPage:
        conditions = conditions or []
        dataset, columns, source = self._scope(dataset_id, conditions)
        relation, order = view_relation(source, columns, view)
        key = (
            dataset_id,
            dataset.imported_at,
            dataset.types_version,
            tuple((c.name, c.type, c.excluded) for c in columns),
            filters_key(conditions),
            view.mode,
            view.search.strip().lower(),
            view.rows,
        )
        connection = duckdb.connect()
        try:
            if key in self._counts:
                self._counts.move_to_end(key)
                total = self._counts[key]
            else:
                total = self._counts[key] = count(connection, relation)
                if len(self._counts) > COUNT_CACHE:
                    self._counts.popitem(last=False)
            limit = max(0, min(limit, MAX_LIMIT))
            offset = max(offset, 0)
            rows = page(connection, relation, order, columns, offset, limit) if limit else []
        finally:
            connection.close()
        return RowsPage(
            [RowsColumn(c.name, c.type, c.excluded) for c in columns], rows, total, offset
        )

    def sample(
        self,
        dataset_id: str,
        names: list[str],
        limit: int,
        conditions: list[Condition] | None = None,
        spread: bool = True,
    ) -> RowsPage:
        """A few filtered rows with only the columns named (the assistant's `sample_rows`, 026):
        spread over the rows (a reproducible sample, the same for the same filters) or the first
        ones, listed in file order."""
        _, columns, source = self._scope(dataset_id, conditions or [])
        by_name = {c.name: c for c in columns}
        unknown = [name for name in names if name not in by_name]
        if unknown:
            raise UnknownColumnError(unknown[0])
        chosen = [by_name[name] for name in names]
        relation, order = view_relation(source, columns, View())
        if spread:
            order = f"hash({ROW} + {SEED}), {ROW}"
        connection = duckdb.connect()
        try:
            total = count(connection, relation)
            limit = max(0, min(limit, MAX_LIMIT))
            rows = page(connection, relation, order, chosen, 0, limit) if limit else []
        finally:
            connection.close()
        rows.sort(key=lambda row: row[0])
        return RowsPage([RowsColumn(c.name, c.type, c.excluded) for c in chosen], rows, total, 0)

    def export(
        self,
        dataset_id: str,
        file_format: Format,
        view: View,
        chosen: list[str] | None = None,
        conditions: list[Condition] | None = None,
    ) -> ExportFile:
        dataset, columns, source = self._scope(dataset_id, conditions or [])
        relation, order = view_relation(source, columns, view)
        if chosen is not None:
            by_name = {c.name: c for c in columns}
            unknown = [name for name in chosen if name not in by_name]
            if unknown:
                raise UnknownColumnError(unknown[0])
            columns = [by_name[name] for name in chosen]
        target = self._target(dataset.name, file_format)
        connection = duckdb.connect()
        try:
            rows = count(connection, relation) if file_format == "xlsx" else 0
            write_view(connection, relation, order, columns, file_format, target.path, rows)
        except BaseException:
            target.remove()
            raise
        finally:
            connection.close()
        return target

    def summary(self, dataset_id: str, conditions: list[Condition] | None = None) -> ExportFile:
        dataset, columns, source = self._scope(dataset_id, conditions or [])
        target = self._target(f"{dataset.name}-summary", "summary")
        connection = duckdb.connect()
        try:
            write_summary(connection, source, columns, target.path)
        except BaseException:
            target.remove()
            raise
        finally:
            connection.close()
        return target

    @staticmethod
    def _target(name: str, kind: str) -> ExportFile:
        """A path in a new private directory (mode 0700): nothing else can create or swap it."""
        extension = EXTENSIONS[kind]
        stem = re.sub(r"[^\w.-]+", "_", Path(name).stem).strip("_") or "dataset"
        folder = Path(tempfile.mkdtemp(prefix="de-export-"))
        filename = f"{stem}-{date.today().isoformat()}.{extension}"
        return ExportFile(folder / f"export.{extension}", filename)
