"""Associations and insights of a dataset, over the filtered rows; cached in memory."""

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import duckdb

from quantilo.catalog import Catalog
from quantilo.correlations.insights import Insight, insights_for
from quantilo.correlations.matrix import (
    Matrix,
    MatrixColumn,
    Method,
    association_matrix,
)
from quantilo.filters.model import Condition, filters_key
from quantilo.filters.scope import scoped_source, where_for

CACHE_SIZE = 16


@dataclass
class Correlations:
    matrix: Matrix
    insights: list[Insight] = field(default_factory=list)


class CorrelationService:
    def __init__(self, columns: Catalog) -> None:
        self.columns = columns
        self._cache: OrderedDict[tuple[Any, ...], Correlations] = OrderedDict()

    def correlations(
        self,
        dataset_id: str,
        method: Method = "pearson",
        chosen: list[str] | None = None,
        conditions: list[Condition] | None = None,
    ) -> Correlations:
        conditions = conditions or []
        dataset = self.columns.ready(dataset_id)
        views = self.columns.columns(dataset_id)
        key = (
            dataset_id,
            dataset.imported_at,
            dataset.types_version,
            tuple((v.name, v.type, v.excluded) for v in views),
            method,
            tuple(chosen) if chosen is not None else None,
            filters_key(conditions),
        )
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        columns = [MatrixColumn(v.name, v.type, v.parser) for v in views if not v.excluded]
        source = scoped_source(dataset.storage_path, where_for(views, conditions), numbered=True)
        connection = duckdb.connect()
        try:
            matrix, loaded = association_matrix(connection, source, columns, method, chosen)
            in_matrix = {info.name for info in matrix.columns}
            used = [c for c in columns if c.name in in_matrix]
            insights = insights_for(used, matrix, loaded)
        finally:
            connection.close()
        result = Correlations(matrix, insights)
        self._cache[key] = result
        if len(self._cache) > CACHE_SIZE:
            self._cache.popitem(last=False)
        return result
