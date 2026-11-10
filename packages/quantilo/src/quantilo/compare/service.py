"""Comparisons of a dataset's columns, read through their effective types and the active filters."""

from collections import OrderedDict
from typing import Any

import duckdb

from quantilo.catalog import Catalog
from quantilo.compare.analyze import (
    MAX_DISCRETE_CATEGORIES,
    CategoryRead,
    CompareColumn,
    Comparison,
    NotComparableError,
    compare,
)
from quantilo.compare.multiple import Correction
from quantilo.filters.model import Condition, filters_key
from quantilo.filters.scope import scoped_source, where_for
from quantilo.timeseries.series import Options

CACHE_SIZE = 32


class CompareService:
    def __init__(self, columns: Catalog) -> None:
        self.columns = columns
        self._cache: OrderedDict[tuple[Any, ...], Comparison] = OrderedDict()

    def compare(
        self,
        dataset_id: str,
        x: str,
        y: str,
        z: str | None = None,
        color: str | None = None,
        conditions: list[Condition] | None = None,
        options: Options | None = None,
        numbers: list[str] | None = None,
        correction: Correction = "auto",
    ) -> Comparison:
        """`numbers`: whole-number columns to keep as numbers (R-CMP-02)."""
        conditions = conditions or []
        kept_numbers = set(numbers or [])
        options = options or Options()
        dataset = self.columns.ready(dataset_id)
        views = {view.name: view for view in self.columns.columns(dataset_id)}

        def pick(name: str | None) -> CompareColumn | None:
            if name is None:
                return None
            view = views.get(name)
            if view is None or view.excluded:
                raise NotComparableError(f"Unknown or excluded column {name!r}.")
            distinct = (view.summary or {}).get("distinct_count")
            as_category = (
                view.type == "numeric_discrete"
                and isinstance(distinct, int)
                and distinct <= MAX_DISCRETE_CATEGORIES
                and view.name not in kept_numbers
            )
            return CompareColumn(
                view.name, view.type, view.parser, view.order, view.role, as_category
            )

        chosen = [pick(name) for name in (x, y, z, color)]
        key = (
            dataset_id,
            dataset.imported_at,
            tuple((c.name, c.type, c.parser.model_dump_json()) if c else None for c in chosen),
            filters_key(conditions),
            (options.period, options.aggregate, options.window, options.align),
            tuple(c.semantic_role if c else None for c in chosen),
            tuple(c.as_category if c else None for c in chosen),
            correction,
        )
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        where = where_for(list(views.values()), conditions)
        connection = duckdb.connect()
        try:
            first, second, third, fourth = chosen
            assert first is not None and second is not None
            result = compare(
                connection,
                scoped_source(dataset.storage_path, where, numbered=True),
                first,
                second,
                third,
                fourth,
                options,
                correction,
            )
        finally:
            connection.close()
        result.categories = [
            CategoryRead(c.name, int((views[c.name].summary or {}).get("distinct_count", 0)))
            for c in chosen
            if c is not None and c.as_category
        ]
        self._cache[key] = result
        if len(self._cache) > CACHE_SIZE:
            self._cache.popitem(last=False)
        return result
