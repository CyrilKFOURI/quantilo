"""Variable analyses of a dataset's column, read through its effective type; cached in memory."""

from collections import OrderedDict
from collections.abc import Callable
from typing import Any

import duckdb

from quantilo.catalog import Catalog
from quantilo.errors import ExplorerError, NotFoundError
from quantilo.filters.model import Condition, filters_key
from quantilo.filters.scope import scoped_source, where_for
from quantilo.variable.analyze import NUMERIC, VariableProfile, analyze
from quantilo.variable.fitting import Fit, fit
from quantilo.variable.numeric import describe
from quantilo.variable.outliers import OutlierMethod, Outliers, outliers
from quantilo.variable.source import VariableInput, VariableSource

CACHE_SIZE = 32


class NotNumericError(ExplorerError):
    def __init__(self, column: str) -> None:
        super().__init__(f"{column!r} is not numeric", code="not_numeric", status_code=422)


class VariableService:
    def __init__(self, columns: Catalog) -> None:
        self.columns = columns
        self._cache: OrderedDict[tuple[Any, ...], Any] = OrderedDict()

    def _cached(self, key: tuple[Any, ...], compute: Callable[[], Any]) -> Any:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        value = compute()
        self._cache[key] = value
        if len(self._cache) > CACHE_SIZE:
            self._cache.popitem(last=False)
        return value

    def _run(
        self,
        dataset_id: str,
        column: str,
        what: str,
        task: Callable[[VariableSource], Any],
        conditions: list[Condition],
    ):
        dataset = self.columns.ready(dataset_id)
        views = self.columns.columns(dataset_id)
        view = next((c for c in views if c.name == column), None)
        if view is None:
            raise NotFoundError(f"Column {column!r}")
        where = where_for(views, conditions)
        key = (
            dataset_id,
            dataset.imported_at,
            column,
            view.type,
            view.parser.model_dump_json(),
            what,
            filters_key(conditions),
        )

        def compute() -> Any:
            connection = duckdb.connect()
            try:
                source = VariableSource(
                    connection,
                    scoped_source(dataset.storage_path, where, numbered=True),
                    VariableInput(view.name, view.type, view.parser, view.order),
                )
                return task(source)
            finally:
                connection.close()

        return self._cached(key, compute)

    @staticmethod
    def _numeric(source: VariableSource) -> None:
        if source.column.type not in NUMERIC:
            raise NotNumericError(source.column.name)

    def profile(
        self, dataset_id: str, column: str, conditions: list[Condition] | None = None
    ) -> VariableProfile:
        return self._run(dataset_id, column, "profile", analyze, conditions or [])

    def fit(self, dataset_id: str, column: str, conditions: list[Condition] | None = None) -> Fit:
        def task(source: VariableSource) -> Fit:
            self._numeric(source)
            return fit(source, discrete=source.column.type == "numeric_discrete")

        return self._run(dataset_id, column, "fit", task, conditions or [])

    def outliers(
        self,
        dataset_id: str,
        column: str,
        method: OutlierMethod,
        conditions: list[Condition] | None = None,
    ) -> Outliers:
        def task(source: VariableSource) -> Outliers:
            self._numeric(source)
            return outliers(source, describe(source), method)

        return self._run(dataset_id, column, f"outliers:{method}", task, conditions or [])
