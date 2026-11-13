"""KPI suggestions and computation over a catalog's dataset; results cached in memory."""

import json
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import duckdb

from quantilo.catalog import Catalog
from quantilo.errors import ExplorerError
from quantilo.filters.model import Condition, filters_key
from quantilo.filters.scope import scoped_source, where_for
from quantilo.kpi import cohort as kpi_cohort
from quantilo.kpi.archetypes import Archetype, classify
from quantilo.kpi.cohort import Analysis, CohortGrain, CohortResult, FunnelResult
from quantilo.kpi.compute import KpiResult, compute
from quantilo.kpi.definition import Definition
from quantilo.kpi.probe import measure
from quantilo.kpi.suggest import (
    DateChoice,
    Hierarchy,
    choose_date,
    hierarchies,
    notes,
    probe_keys,
    probe_targets,
    rank_dates,
    split_candidates,
    suggest,
)
from quantilo.profiling import aggregates
from quantilo.reasons import ReasonText, details
from quantilo.typing.derived import names_in, with_derived

CACHE_SIZE = 16


def aggregate_notes(found: list[aggregates.AggregateRows]) -> list[str]:
    """R-SUG-30 (fix1, KPI-2): a category holding rows that sum up the others (World, Total):
    any total over it counts the same rows twice. Said, never filtered on its own."""
    said = []
    for item in found:
        first = item.found[0]
        said.append(
            f"R-SUG-30 · {item.column} holds {len(item.found)} "
            f"label{'s' if len(item.found) != 1 else ''} ({item.rows:,} rows) that "
            "sum up other rows "
            f"({item.shown()}; '{first.label}': {first.why}): a total over {item.column} counts "
            "them twice. Filter them out, or leave them out in Clean (B-21)."
        )
    return said


@dataclass
class SuggestedKpi:
    rule: str
    name: str
    definition: Definition
    score: int  # out of 100, the list is sorted by it (048, R-SUG-16)
    reasons: list[str]  # "R-SUG-xx · why", the first one says what the KPI is
    # M02: the reasons with their code and vars, for the interface to translate
    reason_details: list[ReasonText] = field(default_factory=list)


@dataclass
class Suggestions:
    kpis: list[SuggestedKpi]  # best first, 12 at most
    splits: list[str]  # the best columns to split by, best first
    hierarchies: list[Hierarchy]  # parent then child splits (048, R-SUG-18)
    # 093: columns the rules read differently from their role (R-SUG-25 to 27), with the reason;
    # fix1: rows that sum up other rows (R-SUG-30)
    notes: list[str] = field(default_factory=list)
    # 060: retention by cohort and funnel the columns allow (R-COH-01, R-FUN-01), computed on demand
    analyses: list[Analysis] = field(default_factory=list)
    # 061: the kind of dataset and its pack of KPIs (R-ARC-01, 02), always confirmed by the user
    archetype: Archetype | None = None
    # 048 bis: the date columns in the order R-SUG-17 prefers, for the Builder's Over time list
    dates: list[DateChoice] = field(default_factory=list)


class KpiEngine:
    def __init__(self, columns: Catalog) -> None:
        self.columns = columns
        self._cache: OrderedDict[tuple[Any, ...], KpiResult] = OrderedDict()
        self._suggested: OrderedDict[tuple[Any, ...], Suggestions] = OrderedDict()
        self._analyses: OrderedDict[tuple[Any, ...], Any] = OrderedDict()

    def clear(self) -> None:
        self._cache.clear()
        self._suggested.clear()
        self._analyses.clear()

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
        keys = probe_keys(views)
        connection = duckdb.connect()
        try:
            source = scoped_source(dataset.storage_path, where_for(views, []), numbered=True)
            facts = measure(
                connection,
                source,
                [by_name[n] for n in labels],
                [by_name[n] for n in numbers],
                [by_name[n] for n in keys],
            )
            summed_up = aggregates.find(connection, source, views)
        finally:
            connection.close()
        found = suggest(views, facts)
        date, _ = choose_date(views)
        result = Suggestions(
            [
                SuggestedKpi(s.rule, s.name, s.definition, s.score, s.reasons, details(s.reasons))
                for s in found
            ],
            split_candidates(views, facts),
            hierarchies(views, facts),
            [*notes(views), *aggregate_notes(summed_up)],
            kpi_cohort.propose(views, date),
            classify(views, found),
            rank_dates(views),
        )
        self._suggested[key] = result
        if len(self._suggested) > CACHE_SIZE:
            self._suggested.popitem(last=False)
        return result

    # --- Cohorts and funnels (060), computed only when opened -----------------------------------

    def _known(self, views: list[Any], name: str | None, what: str) -> Any:
        found = next((v for v in views if v.name == name), None)
        if found is None:
            raise ExplorerError(
                f"Column {name} no longer exists ({what}).", code="unknown_column", status_code=422
            )
        return found

    def _cached(self, key: tuple[Any, ...], run: Callable[[], Any]) -> Any:
        if key in self._analyses:
            self._analyses.move_to_end(key)
            return self._analyses[key]
        value = run()
        self._analyses[key] = value
        if len(self._analyses) > CACHE_SIZE:
            self._analyses.popitem(last=False)
        return value

    @staticmethod
    def _parsers(views: list[Any]) -> tuple[Any, ...]:
        return tuple((v.name, v.type, v.parser.model_dump_json()) for v in views)

    def cohort(
        self,
        dataset_id: str,
        customer: str,
        date: str,
        grain: CohortGrain | Literal["auto"],
        conditions: list[Condition],
    ) -> CohortResult:
        dataset = self.columns.ready(dataset_id)
        views = self._views(dataset_id)
        key_column = self._known(views, customer, "customer")
        date_column = self._known(views, date, "date")

        def run() -> CohortResult:
            connection = duckdb.connect()
            try:
                source = scoped_source(dataset.storage_path, where_for(views, conditions))
                return kpi_cohort.retention(connection, source, key_column, date_column, grain)
            finally:
                connection.close()

        key = ("cohort", dataset_id, dataset.imported_at, customer, date, grain)
        return self._cached((*key, filters_key(conditions), self._parsers(views)), run)

    def funnel(
        self,
        dataset_id: str,
        column: str,
        stages: list[str],
        key: str | None,
        conditions: list[Condition],
    ) -> FunnelResult:
        dataset = self.columns.ready(dataset_id)
        views = self._views(dataset_id)
        status = self._known(views, column, "status")
        counted = self._known(views, key, "key") if key else None

        def run() -> FunnelResult:
            connection = duckdb.connect()
            try:
                source = scoped_source(
                    dataset.storage_path, where_for(views, conditions), numbered=True
                )
                return kpi_cohort.funnel(connection, source, status, stages, counted)
            finally:
                connection.close()

        cache_key = ("funnel", dataset_id, dataset.imported_at, column, tuple(stages), key)
        return self._cached((*cache_key, filters_key(conditions), self._parsers(views)), run)

    # --- Compute ---------------------------------------------------------------------------------

    def compute(
        self, dataset_id: str, definition: Definition, conditions: list[Condition]
    ) -> KpiResult:
        dataset = self.columns.ready(dataset_id)
        views = with_derived(self._views(dataset_id), names_in(definition))  # 058
        if definition.time is not None and not definition.time.page_filters:
            conditions = []  # the KPI only sees its own period
        data = definition.model_dump(mode="json", exclude={"view"})  # display needs no recompute
        data["trend"] = definition.view.show_as == "trend"  # except the trend (053, R-SHOW-02)
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
