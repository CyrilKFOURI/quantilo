"""Overview and chart data of a dataset, read through the effective column types."""

from collections import OrderedDict
from typing import Any

import duckdb

from quantilo.catalog import Catalog
from quantilo.charts.prepare import Chart, ChartParams, ColumnInput, prepare_chart
from quantilo.errors import NotFoundError
from quantilo.filters.model import Condition, filters_key
from quantilo.filters.scope import scoped_source, where_for
from quantilo.profiling.overview import Overview, dataset_overview
from quantilo.typing.derived import with_derived
from quantilo.typing.summary import summarize

CACHE_SIZE = 64


class ProfilingService:
    def __init__(self, columns: Catalog) -> None:
        self.columns = columns
        # Overviews depend on the data and the column types only: cached per types_version.
        self._overviews: OrderedDict[tuple[Any, ...], Overview] = OrderedDict()

    def overview(self, dataset_id: str, conditions: list[Condition] | None = None) -> Overview:
        conditions = conditions or []
        dataset = self.columns.ready(dataset_id)
        views = self.columns.columns(dataset_id)
        where = where_for(views, conditions)
        excluded = tuple(sorted(c.name for c in views if c.excluded))
        key = (
            dataset_id,
            dataset.imported_at,
            dataset.types_version,
            excluded,
            filters_key(conditions),
        )
        if key in self._overviews:
            self._overviews.move_to_end(key)
            return self._overviews[key]
        connection = duckdb.connect()
        try:
            result = dataset_overview(
                connection,
                scoped_source(dataset.storage_path, where, numbered=True),
                dataset.size_bytes or 0,
                views,
                total_rows=dataset.row_count,
                filtered=bool(where),
            )
        finally:
            connection.close()
        self._overviews[key] = result
        if len(self._overviews) > CACHE_SIZE:
            self._overviews.popitem(last=False)
        return result

    def chart(
        self,
        dataset_id: str,
        column: str,
        params: ChartParams,
        conditions: list[Condition] | None = None,
    ) -> Chart:
        dataset = self.columns.ready(dataset_id)
        views = with_derived(self.columns.columns(dataset_id), [column])  # 058
        view = next((c for c in views if c.name == column), None)
        if view is None:
            raise NotFoundError(f"Column {column!r}")
        where = where_for(views, conditions or [])
        connection = duckdb.connect()
        try:
            return prepare_chart(
                connection,
                scoped_source(dataset.storage_path, where),
                ColumnInput(view.name, view.type, view.parser, view.order),
                params,
            )
        finally:
            connection.close()

    def summaries(self, dataset_id: str, conditions: list[Condition]) -> dict[str, dict[str, Any]]:
        """Column summaries (as on the Columns page) over the filtered rows; types unchanged."""
        dataset = self.columns.ready(dataset_id)
        views = self.columns.columns(dataset_id)
        source = scoped_source(dataset.storage_path, where_for(views, conditions))
        connection = duckdb.connect()
        try:
            return {
                view.name: summarize(
                    connection, source, view.name, view.type, view.parser, view.order
                )
                for view in views
                if not view.excluded
            }
        finally:
            connection.close()
