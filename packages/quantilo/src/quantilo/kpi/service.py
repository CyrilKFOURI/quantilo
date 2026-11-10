"""KPI suggestions and computation over a catalog's dataset; results cached in memory."""

import json
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import duckdb

from quantilo.catalog import Catalog
from quantilo.filters.model import Condition, filters_key
from quantilo.filters.scope import scoped_source, where_for
from quantilo.kpi.compute import KpiResult, compute
from quantilo.kpi.definition import Definition
from quantilo.kpi.probe import measure
from quantilo.kpi.suggest import Hierarchy, hierarchies, probe_targets, split_candidates, suggest

CACHE_SIZE = 16


@dataclass
class SuggestedKpi:
    rule: str
    name: str
    definition: Definition
    score: int  # out of 100, the list is sorted by it (048, R-SUG-16)
    reasons: list[str]  # "R-SUG-xx · why", the first one says what the KPI is


@dataclass
class Suggestions:
    kpis: list[SuggestedKpi]  # best first, 12 at most
    splits: list[str]  # the best columns to split by, best first
    hierarchies: list[Hierarchy]  # parent then child splits (048, R-SUG-18)


class KpiEngine:
    def __init__(self, columns: Catalog) -> None:
        self.columns = columns
        self._cache: OrderedDict[tuple[Any, ...], KpiResult] = OrderedDict()
        self._suggested: OrderedDict[tuple[Any, ...], Suggestions] = OrderedDict()

    def clear(self) -> None:
        self._cache.clear()
        self._suggested.clear()

    def _views(self, dataset_id: str) -> list[Any]:
        return [v for v in self.columns.columns(dataset_id) if not v.excluded]

    # --- Automatic start -------------------------------------------------------------------------

    def suggestions(self, dataset_id: str) -> Suggestions:
        dataset = self.columns.ready(dataset_id)
        views = self._views(dataset_id)
        key = (
            dataset_id,
            dataset.imported_at,
            tuple((v.name, v.type, v.role, v.parser.model_dump_json()) for v in views),
        )
        if key in self._suggested:
            self._suggested.move_to_end(key)
            return self._suggested[key]
        by_name = {v.name: v for v in views}
        labels, numbers = probe_targets(views)
        connection = duckdb.connect()
        try:
            source = scoped_source(dataset.storage_path, where_for(views, []), numbered=True)
            facts = measure(
                connection, source, [by_name[n] for n in labels], [by_name[n] for n in numbers]
            )
        finally:
            connection.close()
        found = suggest(views, facts)
        result = Suggestions(
            [SuggestedKpi(s.rule, s.name, s.definition, s.score, s.reasons) for s in found],
            split_candidates(views, facts),
            hierarchies(views, facts),
        )
        self._suggested[key] = result
        if len(self._suggested) > CACHE_SIZE:
            self._suggested.popitem(last=False)
        return result

    # --- Compute ---------------------------------------------------------------------------------

    def compute(
        self, dataset_id: str, definition: Definition, conditions: list[Condition]
    ) -> KpiResult:
        dataset = self.columns.ready(dataset_id)
        views = self._views(dataset_id)
        if definition.time is not None and not definition.time.page_filters:
            conditions = []  # the KPI only sees its own period
        data = definition.model_dump(mode="json", exclude={"view"})  # display needs no recompute
        key = (
            dataset_id,
            dataset.imported_at,
            json.dumps(data, sort_keys=True),
            tuple(
                (v.name, v.type, v.role, v.parser.model_dump_json(), str(v.order)) for v in views
            ),
            filters_key(conditions),
        )
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        connection = duckdb.connect()
        try:
            source = scoped_source(dataset.storage_path, where_for(views, conditions))
            result = compute(connection, source, views, definition)
        finally:
            connection.close()
        self._cache[key] = result
        if len(self._cache) > CACHE_SIZE:
            self._cache.popitem(last=False)
        return result
