"""Comparisons of a dataset's columns, read through their effective types and the active filters."""

from collections import OrderedDict
from typing import Any

import duckdb

from quantilo.catalog import Catalog
from quantilo.compare import durations as du
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
from quantilo.text.tokens import language_of
from quantilo.timeseries.series import Options
from quantilo.typing.derived import with_derived

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
        listed = with_derived(self.columns.columns(dataset_id), (x, y, z, color))  # 058
        views = {view.name: view for view in listed}

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
                view.name,
                view.type,
                view.parser,
                view.order,
                view.role,
                as_category,
                language_of(view),
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

    # --- Durations (062) ----------------------------------------------------------------------

    def _duration_columns(self, dataset_id: str) -> dict[str, du.DurationColumn]:
        columns: dict[str, du.DurationColumn] = {}
        for view in self.columns.columns(dataset_id):
            if view.excluded:
                continue
            summary = view.summary or {}
            numeric = summary.get("numeric") or {}
            minimum = numeric.get("min") if isinstance(numeric, dict) else None
            distinct, rows = summary.get("distinct_count"), summary.get("rows")
            columns[view.name] = du.DurationColumn(
                view.name,
                view.type,
                view.parser,
                view.role,
                int(distinct) if isinstance(distinct, int) else None,
                int(rows) - int(summary.get("null_count") or 0) if isinstance(rows, int) else None,
                view.order,
                float(minimum) if isinstance(minimum, int | float) else None,
            )
        return columns

    def _run(self, dataset_id: str, conditions: list[Condition], compute):
        dataset = self.columns.ready(dataset_id)
        views = [v for v in self.columns.columns(dataset_id) if not v.excluded]
        connection = duckdb.connect()
        try:
            source = scoped_source(dataset.storage_path, where_for(views, conditions))
            return compute(connection, source)
        finally:
            connection.close()

    def survival(
        self,
        dataset_id: str,
        conditions: list[Condition] | None = None,
        duration: str | None = None,
        event: str | None = None,
        start: str | None = None,
        end: str | None = None,
        group: str | None = None,
        event_value: str | None = None,
    ) -> du.Survival:
        """Kaplan-Meier curves; without columns, the reading a rule detects (R-SRV-01/02), or
        nothing (the section stays hidden: the onion rule)."""
        columns = self._duration_columns(dataset_id)
        if duration or event:
            setup: du.SurvivalSetup | None = du.SurvivalSetup(
                "duration", duration=duration, event=event, detected=False, reason="Your choice."
            )
        elif start or end:
            setup = du.SurvivalSetup(
                "dates", start=start, end=end, detected=False, reason="Your choice."
            )
        else:
            setup = du.detect_survival(list(columns.values()))
        if setup is None:
            return du.Survival(None, du.survival_choices(list(columns.values())))
        chosen = setup
        return self._run(
            dataset_id,
            conditions or [],
            lambda con, src: du.survival(con, src, chosen, columns, group, event_value),
        )

    def transitions(
        self,
        dataset_id: str,
        conditions: list[Condition] | None = None,
        entity: str | None = None,
        time: str | None = None,
        status: str | None = None,
        detect_only: bool = False,
    ) -> du.Transitions:
        """The Markov chain of a status; without columns, the reading R-MKV-01 detects.
        `detect_only`: say what was found without reading the rows."""
        columns = self._duration_columns(dataset_id)
        found = du.detect_transitions(list(columns.values()))
        if entity and time and status:
            setup: du.TransitionSetup | None = du.TransitionSetup(
                entity, time, status, detected=False, reason="Your choice."
            )
        else:
            setup = found
        choices = du.transition_choices(list(columns.values()))
        if setup is None or detect_only:
            return du.Transitions(setup, choices)
        chosen = setup
        return self._run(
            dataset_id,
            conditions or [],
            lambda con, src: du.transitions(con, src, chosen, columns),
        )
