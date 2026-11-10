"""Associations and insights of a dataset, over the filtered rows; cached in memory. The structure
of its numeric columns (078) is computed on request only, from the same columns and rows."""

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import duckdb
import numpy as np

from quantilo.catalog import Catalog
from quantilo.compare.multiple import Correction
from quantilo.correlations.insights import Insight, code_columns, insights_for
from quantilo.correlations.matrix import (
    ROLES,
    Matrix,
    MatrixColumn,
    Method,
    association_matrix,
    eligible,
    load,
)
from quantilo.correlations.structure import (
    Factors,
    Structure,
    StructureLeftOut,
    StructureRule,
    factor_analysis,
    pca_structure,
    structure_rule,
)
from quantilo.filters.model import Condition, filters_key
from quantilo.filters.scope import scoped_source, where_for

CACHE_SIZE = 16


@dataclass
class Correlations:
    matrix: Matrix
    insights: list[Insight] = field(default_factory=list)
    structure: StructureRule | None = None  # whether the Structure section is proposed (078)


class CorrelationService:
    def __init__(self, columns: Catalog) -> None:
        self.columns = columns
        self._cache: OrderedDict[tuple[Any, ...], Correlations] = OrderedDict()
        self._structures: OrderedDict[tuple[Any, ...], Structure | Factors] = OrderedDict()

    def _key(
        self, dataset_id: str, chosen: list[str] | None, conditions: list[Condition]
    ) -> tuple[Any, ...]:
        dataset = self.columns.ready(dataset_id)
        views = self.columns.columns(dataset_id)
        return (
            dataset_id,
            dataset.imported_at,
            dataset.types_version,
            tuple((v.name, v.type, v.excluded) for v in views),
            tuple(chosen) if chosen is not None else None,
            filters_key(conditions),
        )

    def correlations(
        self,
        dataset_id: str,
        method: Method = "pearson",
        chosen: list[str] | None = None,
        conditions: list[Condition] | None = None,
        correction: Correction = "auto",
    ) -> Correlations:
        conditions = conditions or []
        dataset = self.columns.ready(dataset_id)
        views = self.columns.columns(dataset_id)
        key = (*self._key(dataset_id, chosen, conditions), method, correction)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        columns = [MatrixColumn(v.name, v.type, v.parser, v.role) for v in views if not v.excluded]
        source = scoped_source(dataset.storage_path, where_for(views, conditions), numbered=True)
        connection = duckdb.connect()
        try:
            matrix, loaded = association_matrix(
                connection, source, columns, method, chosen, correction
            )
            in_matrix = {info.name for info in matrix.columns}
            used = [c for c in columns if c.name in in_matrix]
            insights = insights_for(used, matrix, loaded)
        finally:
            connection.close()
        codes = set(code_columns(used, loaded))
        result = Correlations(matrix, insights, structure_rule(matrix, codes))
        self._cache[key] = result
        if len(self._cache) > CACHE_SIZE:
            self._cache.popitem(last=False)
        return result

    def _numbers(
        self, dataset_id: str, chosen: list[str] | None, conditions: list[Condition]
    ) -> tuple[dict[str, np.ndarray], int, list[StructureLeftOut]]:
        """The numeric columns of the matrix (same choice, filters and sample), read once;
        codes typed as numbers left out with their reason (R-STR-2, R-COR-01)."""
        dataset = self.columns.ready(dataset_id)
        views = self.columns.columns(dataset_id)
        columns = [MatrixColumn(v.name, v.type, v.parser, v.role) for v in views if not v.excluded]
        usable, _excluded, _truncated = eligible(columns, chosen)
        numeric = [c for c in usable if ROLES[c.type] == "num"]
        if not numeric:
            return {}, 0, []
        source = scoped_source(dataset.storage_path, where_for(views, conditions), numbered=True)
        connection = duckdb.connect()
        try:
            loaded = load(connection, source, numeric)
        finally:
            connection.close()
        codes = code_columns(numeric, loaded)
        arrays = {name: values for name, values in loaded.arrays.items() if name not in codes}
        left = [StructureLeftOut(name, why) for name, why in codes.items()]
        return arrays, loaded.total, left

    def _remember(self, key: tuple[Any, ...], value: Structure | Factors) -> None:
        self._structures[key] = value
        if len(self._structures) > CACHE_SIZE:
            self._structures.popitem(last=False)

    def structure(
        self,
        dataset_id: str,
        chosen: list[str] | None = None,
        conditions: list[Condition] | None = None,
    ) -> Structure:
        """Standardised PCA of the numeric columns, components by parallel analysis (078)."""
        conditions = conditions or []
        key = ("pca", *self._key(dataset_id, chosen, conditions))
        cached = self._structures.get(key)
        if isinstance(cached, Structure):
            self._structures.move_to_end(key)
            return cached
        arrays, total, codes = self._numbers(dataset_id, chosen, conditions)
        found = pca_structure(arrays, total)
        found.left_out = [*codes, *found.left_out]
        self._remember(key, found)
        return found

    def factors(
        self,
        dataset_id: str,
        chosen: list[str] | None = None,
        conditions: list[Condition] | None = None,
        k: int | None = None,
    ) -> Factors:
        """Varimax factor analysis, KMO and Bartlett's test (078, Advanced > More). Without `k`,
        as many factors as the parallel analysis keeps."""
        conditions = conditions or []
        key = ("fa", k, *self._key(dataset_id, chosen, conditions))
        cached = self._structures.get(key)
        if isinstance(cached, Factors):
            self._structures.move_to_end(key)
            return cached
        retained = self.structure(dataset_id, chosen, conditions).retained if k is None else None
        arrays, _total, codes = self._numbers(dataset_id, chosen, conditions)
        found = factor_analysis(arrays, k, retained)
        found.left_out = [*codes, *found.left_out]
        self._remember(key, found)
        return found
