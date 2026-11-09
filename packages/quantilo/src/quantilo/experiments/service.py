"""Experiments on a catalog's dataset: the columns that can describe one, the analysis through the
active filters (results cached in memory), a metric's current value for planning."""

import json
import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Literal

import duckdb

from quantilo.catalog import Catalog, ColumnView
from quantilo.compare.analyze import label_sql, number_label_sql, number_sql
from quantilo.errors import ColumnNotFound
from quantilo.experiments.analyze import (
    MAX_LEVELS,
    MIN_LEVELS,
    SEGMENT,
    TIME,
    UNIT,
    VARIANT,
    ExperimentDesign,
    ExperimentError,
    ExperimentMetric,
    ExperimentResult,
    MetricInput,
    MetricKind,
    analyze,
    guess_control,
)
from quantilo.experiments.plan import SampleSize, SampleSizeRequest, sample_size
from quantilo.filters.model import Condition, filters_key
from quantilo.filters.scope import scoped_source, where_for
from quantilo.typing.parsers import typed_sql
from quantilo.typing.types import NUMERIC_TYPES

CACHE_SIZE = 16
MAX_SEGMENT_LEVELS = 20
VARIANT_HINT = re.compile(
    r"variant|group|arm|treatment|bucket|experiment|cohort|version|^ab$|_ab$|^ab_|test", re.I
)
UNIT_HINT = re.compile(
    r"(^|_)(id|user|customer|client|visitor|session|account|member)(_|$)|id$", re.I
)
LEVEL_TYPES = ("categorical_nominal", "categorical_ordinal", "boolean", "numeric_discrete", "text")
UNIT_TYPES = ("identifier", "numeric_discrete", "categorical_nominal", "text")


@dataclass
class ExperimentVariantCandidate:
    column: str
    levels: list[str]
    counts: list[int]
    control: str


@dataclass
class ExperimentMetricCandidate:
    column: str
    kind: MetricKind


@dataclass
class ExperimentUnitCandidate:
    column: str
    rows_per_value: float


@dataclass
class ExperimentCandidates:
    variants: list[ExperimentVariantCandidate]
    metrics: list[ExperimentMetricCandidate]
    units: list[ExperimentUnitCandidate]
    times: list[str]
    segments: list[str]
    suggested: ExperimentDesign | None  # what the page opens on


@dataclass
class MetricBaseline:
    column: str
    kind: Literal["rate", "mean"]
    value: float | None
    sd: float | None
    rows: int


def _distinct(view: ColumnView) -> int | None:
    value = (view.summary or {}).get("distinct_count")
    return value if isinstance(value, int) else None


def level_sql(view: ColumnView) -> str:
    """A value as a level: "1", not "1.0", for numbers; "True"/"False" for yes/no."""
    return number_label_sql(view) if view.type in NUMERIC_TYPES else label_sql(view)


def value_sql(view: ColumnView) -> str:
    """A metric as a number: yes/no as 1/0, infinities as missing."""
    if view.type == "boolean":
        typed = typed_sql(view.name, view.parser)
        return f"(CASE WHEN {typed} THEN 1.0 WHEN NOT {typed} THEN 0.0 END)"
    if view.type not in NUMERIC_TYPES:
        raise ExperimentError(
            f"{view.name} is {view.type.replace('_', ' ')}: a metric must be a number or yes/no.",
            "metrics",
        )
    number = number_sql(view)
    return f"(CASE WHEN isfinite({number}) THEN {number} END)"


def repeated_units(views: list[ColumnView], taken: set[str]) -> list[ExperimentUnitCandidate]:
    """Columns that name a unit (an identifier, a user_id…) and repeat over the rows."""
    found = []
    for view in views:
        if view.name in taken or view.type not in UNIT_TYPES:
            continue
        if (
            view.type != "identifier"
            and view.role != "identifier"
            and not UNIT_HINT.search(view.name)
        ):
            continue
        summary = view.summary or {}
        distinct = summary.get("distinct_count")
        valid = (summary.get("rows") or 0) - (summary.get("null_count") or 0)
        if isinstance(distinct, int) and 0 < distinct < valid:
            found.append(ExperimentUnitCandidate(view.name, valid / distinct))
    return sorted(found, key=lambda unit: -unit.rows_per_value)


class ExperimentEngine:
    def __init__(self, columns: Catalog) -> None:
        self.columns = columns
        self._cache: OrderedDict[tuple[Any, ...], ExperimentResult] = OrderedDict()

    def clear(self) -> None:
        self._cache.clear()

    def _views(self, dataset_id: str) -> dict[str, ColumnView]:
        return {v.name: v for v in self.columns.columns(dataset_id) if not v.excluded}

    @staticmethod
    def _view(views: dict[str, ColumnView], name: str) -> ColumnView:
        view = views.get(name)
        if view is None:
            raise ColumnNotFound(name, list(views))
        return view

    # --- Describe ------------------------------------------------------------------------------

    def candidates(self, dataset_id: str, conditions: list[Condition]) -> ExperimentCandidates:
        dataset = self.columns.ready(dataset_id)
        views = self._views(dataset_id)
        ordered = sorted(views.values(), key=lambda v: v.position)
        level_views = [
            v
            for v in ordered
            if v.type in LEVEL_TYPES and MIN_LEVELS <= (_distinct(v) or 0) <= MAX_LEVELS
        ]
        level_views.sort(key=lambda v: 0 if VARIANT_HINT.search(v.name) else 1)
        numbers = [
            v
            for v in ordered
            if (v.type == "boolean" or v.type in NUMERIC_TYPES) and v.role != "identifier"
        ]
        connection = duckdb.connect()
        try:
            source = scoped_source(
                dataset.storage_path, where_for(list(views.values()), conditions)
            )
            variants = []
            for view in level_views[:12]:
                rows = connection.execute(
                    f"SELECT l, count(*) FROM (SELECT {level_sql(view)} AS l FROM {source}) "
                    "WHERE l IS NOT NULL GROUP BY l ORDER BY l"
                ).fetchall()
                if not MIN_LEVELS <= len(rows) <= MAX_LEVELS:
                    continue
                levels = [str(level) for level, _ in rows]
                counts = [int(count) for _, count in rows]
                variants.append(
                    ExperimentVariantCandidate(
                        view.name, levels, counts, guess_control(levels, counts)
                    )
                )
            metrics = self._metric_kinds(connection, source, numbers)
        finally:
            connection.close()
        variant_names = {v.column for v in variants}
        units = repeated_units(ordered, variant_names)
        times = [v.name for v in ordered if v.type == "datetime"]
        segments = [
            v.name
            for v in ordered
            if v.type in ("categorical_nominal", "categorical_ordinal", "boolean")
            and 2 <= (_distinct(v) or 0) <= MAX_SEGMENT_LEVELS
        ]
        suggested = None
        variant = variants[0].column if variants else None
        usable = [m for m in metrics if m.column != variant]
        if variant and usable:
            chosen = next((m for m in usable if m.kind == "rate"), usable[0])
            suggested = ExperimentDesign(
                variant=variant,
                metrics=[ExperimentMetric(column=chosen.column)],
                unit=units[0].column if units else None,
            )
        return ExperimentCandidates(
            variants,
            metrics,
            units,
            times,
            [s for s in segments if s != variant],
            suggested,
        )

    @staticmethod
    def _metric_kinds(
        connection: duckdb.DuckDBPyConnection, source: str, views: list[ColumnView]
    ) -> list[ExperimentMetricCandidate]:
        if not views:
            return []
        parts = []
        for i, view in enumerate(views):
            value = value_sql(view)
            parts.append(
                f"bool_and({value} IN (0, 1)) AS b{i}, "
                f"bool_and({value} >= 0 AND {value} = floor({value})) AS w{i}"
            )
        row = connection.execute(f"SELECT {', '.join(parts)} FROM {source}").fetchone() or ()
        found = []
        for i, view in enumerate(views):
            binary, whole = row[2 * i], row[2 * i + 1]
            kind: MetricKind
            if view.type == "boolean" or binary:
                kind = "rate"
            elif view.type == "numeric_discrete" and whole:
                kind = "count"
            else:
                kind = "mean"
            found.append(ExperimentMetricCandidate(view.name, kind))
        return found

    # --- Analyze -------------------------------------------------------------------------------

    def analyze(
        self, dataset_id: str, design: ExperimentDesign, conditions: list[Condition]
    ) -> ExperimentResult:
        dataset = self.columns.ready(dataset_id)
        views = self._views(dataset_id)
        named = [design.variant, design.unit, design.segment, design.time]
        for metric in design.metrics:
            named += [metric.column, metric.denominator, metric.pre]
        used = {name: self._view(views, name) for name in named if name}
        key = (
            dataset_id,
            dataset.imported_at,
            json.dumps(design.model_dump(mode="json"), sort_keys=True),
            tuple((v.name, v.type, v.role, v.parser.model_dump_json()) for v in used.values()),
            filters_key(conditions),
        )
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        selected = [f"{level_sql(used[design.variant])} AS {VARIANT}"]
        if design.unit:
            selected.append(f"{level_sql(used[design.unit])} AS {UNIT}")
        if design.segment:
            selected.append(f"{level_sql(used[design.segment])} AS {SEGMENT}")
        if design.time:
            time = used[design.time]
            if time.type != "datetime":
                raise ExperimentError(f"{time.name} is not a date column.", "time")
            selected.append(f"CAST({typed_sql(time.name, time.parser)} AS TIMESTAMP) AS {TIME}")
        inputs = []
        for i, metric in enumerate(design.metrics):
            view = used[metric.column]
            selected.append(f"{value_sql(view)} AS m{i}")
            if metric.denominator:
                selected.append(f"{value_sql(used[metric.denominator])} AS d{i}")
            if metric.pre:
                selected.append(f"{value_sql(used[metric.pre])} AS p{i}")
            inputs.append(
                MetricInput(
                    metric,
                    f"m{i}",
                    f"d{i}" if metric.denominator else None,
                    f"p{i}" if metric.pre else None,
                    view.type == "boolean",
                    view.type == "numeric_discrete",
                )
            )
        connection = duckdb.connect()
        try:
            source = scoped_source(
                dataset.storage_path, where_for(list(views.values()), conditions), numbered=True
            )
            frame = connection.execute(
                f"SELECT * FROM (SELECT {', '.join(selected)}, file_row_number FROM {source}) "
                f"WHERE {VARIANT} IS NOT NULL ORDER BY file_row_number"
            ).df()
        finally:
            connection.close()
        taken = {name for name in named if name}
        repeated = (
            [(u.column, u.rows_per_value) for u in repeated_units(list(views.values()), taken)]
            if not design.unit
            else None
        )
        result = analyze(frame, design, inputs, repeated)
        self._cache[key] = result
        if len(self._cache) > CACHE_SIZE:
            self._cache.popitem(last=False)
        return result

    # --- Plan ----------------------------------------------------------------------------------

    def baseline(self, dataset_id: str, column: str, conditions: list[Condition]) -> MetricBaseline:
        """A metric's current value (rate or mean, with its spread) to start a plan from."""
        dataset = self.columns.ready(dataset_id)
        views = self._views(dataset_id)
        view = self._view(views, column)
        value = value_sql(view)
        connection = duckdb.connect()
        try:
            source = scoped_source(
                dataset.storage_path, where_for(list(views.values()), conditions)
            )
            mean, sd, rows, binary = connection.execute(
                f"SELECT avg({value}), stddev_samp({value}), count({value}), "
                f"bool_and({value} IN (0, 1)) FROM {source}"
            ).fetchone() or (None, None, 0, None)
        finally:
            connection.close()
        kind: Literal["rate", "mean"] = "rate" if view.type == "boolean" or binary else "mean"
        return MetricBaseline(
            column,
            kind,
            float(mean) if mean is not None else None,
            float(sd) if sd is not None else None,
            int(rows),
        )

    @staticmethod
    def sample_size(request: SampleSizeRequest) -> SampleSize:
        return sample_size(request)
