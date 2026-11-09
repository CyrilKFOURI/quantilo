"""Prepare the data of a column's chart, according to its analysis type and the chosen view.

Display rules (docs/VISUALS.md) live here when they need the data: Top N + Others,
missing values apart, high-cardinality guard, automatic timeline step.
"""

import math
from dataclasses import dataclass, field
from typing import Literal, TypedDict

import duckdb

from quantilo.typing.dates import (
    HAS_TIME,
    PARTS,
    STEP_FORMATS,
    STEP_SECONDS,
    TIMELINE_STEPS,
    DatePart,
    auto_step,
    part_sql,
)
from quantilo.typing.parsers import typed_sql
from quantilo.typing.summary import value_label
from quantilo.typing.types import AnalysisType, Parser

ChartKind = Literal["histogram", "bars", "timeline", "identifier", "high_cardinality", "empty"]

DEFAULT_TOP_N = 10
FULL_LIMIT = 1_000
DISCRETE_LIMIT = 60
HIGH_CARDINALITY = 0.5
MIN_BINS, MAX_BINS = 10, 60


class Bar(TypedDict):
    label: str
    count: int


class Bin(TypedDict):
    start: float
    end: float
    count: int


class Others(TypedDict):
    count: int
    categories: int


class Box(TypedDict):
    q1: float
    median: float
    q3: float
    lowerfence: float
    upperfence: float
    outliers: int
    outlier_values: list[float]


@dataclass
class ChartParams:
    view: str | None = None  # e.g. "box", "pie", "timeline", "month", "day_of_week", "hour"
    top_n: int = DEFAULT_TOP_N
    full: bool = False  # every category, for the fullscreen view
    step: str | None = None  # timeline step override
    bins: int | None = None  # histogram bin count override (Freedman-Diaconis otherwise)


@dataclass
class ColumnInput:
    name: str
    type: AnalysisType
    parser: Parser
    order: list[str] | None = None


@dataclass
class Chart:
    kind: ChartKind
    view: str
    views: list[str]
    rows: int = 0
    missing: int = 0
    distinct: int = 0
    bars: list[Bar] = field(default_factory=list)
    others: Others | None = None
    ordered: bool = False
    bins: list[Bin] = field(default_factory=list)
    stats: dict[str, float | None] | None = None
    box: Box | None = None
    step: str | None = None
    steps: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    duplicates: int = 0
    message: str | None = None


class ChartBuilder:
    def __init__(self, connection: duckdb.DuckDBPyConnection, source: str, column: ColumnInput):
        self.connection = connection
        self.source = source
        self.column = column
        self.typed = typed_sql(column.name, column.parser)
        self.values = f"(SELECT {self.typed} AS t FROM {source})"

    def one(self, sql: str) -> tuple:
        return self.connection.execute(sql).fetchone() or ()

    def counts(self) -> tuple[int, int, int]:
        rows, valid, distinct = self.one(
            f"SELECT count(*), count(t), count(DISTINCT t) FROM {self.values}"
        )
        return int(rows), int(rows - valid), int(distinct)

    # --- Numbers -------------------------------------------------------------------------------

    def numeric_stats(self, expression: str = "t") -> dict[str, float | None]:
        row = self.one(
            f"SELECT count(x), avg(x), stddev_samp(x), min(x), "
            f"quantile_cont(x, 0.25), median(x), quantile_cont(x, 0.75), max(x), "
            f"skewness(x), kurtosis(x) "
            f"FROM (SELECT CAST({expression} AS DOUBLE) AS x FROM {self.values}) "
            "WHERE x IS NOT NULL"
        )
        keys = ("count", "mean", "std", "min", "q1", "median", "q3", "max", "skewness", "kurtosis")
        return {
            key: (
                None
                if value is None or (isinstance(value, float) and math.isnan(value))
                else float(value)
            )
            for key, value in zip(keys, row, strict=True)
        }

    def histogram(
        self, expression: str, stats: dict[str, float | None], forced: int | None = None
    ) -> list[Bin]:
        low, high, count = stats["min"], stats["max"], int(stats["count"] or 0)
        if low is None or high is None or count == 0:
            return []
        if low == high:
            return [{"start": low, "end": high, "count": count}]
        iqr = (stats["q3"] or 0) - (stats["q1"] or 0)
        if forced:
            bins = forced
        elif iqr > 0:
            width = 2 * iqr / count ** (1 / 3)  # Freedman-Diaconis
            bins = min(max(math.ceil((high - low) / width), MIN_BINS), MAX_BINS)
        else:
            bins = 20
        width = (high - low) / bins
        rows = self.connection.execute(
            f"SELECT LEAST(CAST(floor((x - {low}) / {width}) AS INTEGER), {bins - 1}) AS b, "
            "count(*) "
            f"FROM (SELECT CAST({expression} AS DOUBLE) AS x FROM {self.values}) "
            f"WHERE x IS NOT NULL GROUP BY b"
        ).fetchall()
        found = {int(index): int(value) for index, value in rows}
        return [
            {"start": low + i * width, "end": low + (i + 1) * width, "count": found.get(i, 0)}
            for i in range(bins)
        ]

    def box(self, stats: dict[str, float | None]) -> Box | None:
        q1, q3 = stats["q1"], stats["q3"]
        if q1 is None or q3 is None:
            return None
        iqr = q3 - q1
        low_fence, high_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        lower, upper, outliers = self.one(
            f"SELECT min(x) FILTER (WHERE x >= {low_fence}), "
            f"max(x) FILTER (WHERE x <= {high_fence}), "
            f"count(*) FILTER (WHERE x < {low_fence} OR x > {high_fence}) "
            f"FROM (SELECT CAST(t AS DOUBLE) AS x FROM {self.values}) WHERE x IS NOT NULL"
        )
        sample = self.connection.execute(
            f"SELECT x FROM (SELECT CAST(t AS DOUBLE) AS x FROM {self.values}) "
            f"WHERE x < {low_fence} OR x > {high_fence} "
            f"ORDER BY abs(x - {stats['median']}) DESC LIMIT 200"
        ).fetchall()
        return {
            "q1": q1,
            "median": stats["median"] if stats["median"] is not None else q1,
            "q3": q3,
            "lowerfence": float(lower) if lower is not None else q1,
            "upperfence": float(upper) if upper is not None else q3,
            "outliers": int(outliers),
            "outlier_values": [float(value) for (value,) in sample],
        }

    def continuous(self, params: ChartParams) -> Chart:
        rows, missing, distinct = self.counts()
        stats = self.numeric_stats()
        view = params.view if params.view in ("histogram", "box") else "histogram"
        chart = Chart("histogram", view, ["histogram", "box"], rows, missing, distinct, stats=stats)
        chart.bins = self.histogram("t", stats, params.bins)
        if view == "box":
            chart.box = self.box(stats)
        return chart

    # --- Categories ----------------------------------------------------------------------------

    def bars(self, expression: str, order_by: str, limit: int) -> list[Bar]:
        rows = self.connection.execute(
            f"SELECT v, count(*) AS c FROM (SELECT {expression} AS v FROM {self.values}) "
            f"WHERE v IS NOT NULL GROUP BY v ORDER BY {order_by} LIMIT {limit}"
        ).fetchall()
        return [Bar(label=value_label(value), count=int(count)) for value, count in rows]

    def categorical(self, params: ChartParams) -> Chart:
        rows, missing, distinct = self.counts()
        kind = self.column.type
        present = rows - missing
        if (
            kind == "categorical_nominal"
            and present >= 50
            and distinct / max(present, 1) > HIGH_CARDINALITY
            and not params.full
        ):
            examples = self.bars("t", "c DESC, v", 5)
            return Chart(
                "high_cardinality",
                "bars",
                ["bars"],
                rows,
                missing,
                distinct,
                examples=[bar["label"] for bar in examples],
                message="Most values are unique: this looks like an identifier or free text.",
            )
        limit = FULL_LIMIT if params.full else max(1, min(params.top_n, 50))
        if kind == "categorical_ordinal":
            counts = {bar["label"]: bar["count"] for bar in self.bars("t", "v", FULL_LIMIT)}
            ordered = [
                Bar(label=item, count=counts.pop(item, 0)) for item in self.column.order or []
            ]
            shown = ordered + [Bar(label=label, count=count) for label, count in counts.items()]
            others: Others | None = None
            if not params.full and len(shown) > limit:
                rest = shown[limit:]
                shown = shown[:limit]
                others = Others(count=sum(bar["count"] for bar in rest), categories=len(rest))
            views = ["bars"]
            return Chart(
                "bars",
                "bars",
                views,
                rows,
                missing,
                distinct,
                bars=shown,
                others=others,
                ordered=True,
            )
        if kind == "numeric_discrete":
            shown = self.bars("t", "v", DISCRETE_LIMIT)
            others: Others | None = None
            if distinct > DISCRETE_LIMIT:
                others = Others(
                    count=present - sum(bar["count"] for bar in shown),
                    categories=distinct - DISCRETE_LIMIT,
                )
            return Chart(
                "bars",
                "bars",
                ["bars"],
                rows,
                missing,
                distinct,
                bars=shown,
                others=others,
                ordered=True,
            )
        shown = self.bars("t", "c DESC, v", limit)
        others: Others | None = None
        if distinct > len(shown):
            others = Others(
                count=present - sum(bar["count"] for bar in shown), categories=distinct - len(shown)
            )
        views = ["bars", "pie"] if distinct <= 5 else ["bars"]
        view = params.view if params.view in views else "bars"
        return Chart("bars", view, views, rows, missing, distinct, bars=shown, others=others)

    # --- Dates ---------------------------------------------------------------------------------

    def datetime(self, params: ChartParams) -> Chart:
        rows, missing, distinct = self.counts()
        year_only = self.column.parser.kind == "year"
        low, high, timed = self.one(
            f"SELECT epoch(min(t)), epoch(max(t)), "
            f"count(*) FILTER (WHERE {HAS_TIME}) "
            f"FROM {self.values} WHERE t IS NOT NULL"
        )
        has_time = bool(timed)
        views = ["timeline"] if year_only else ["timeline", "month", "day_of_week"]
        if has_time:
            views.append("hour")
        view = params.view if params.view in views else "timeline"
        if low is None:
            return Chart("empty", view, views, rows, missing, distinct, message="No date values.")
        if view != "timeline":
            part: DatePart = view  # type: ignore[assignment]
            expression = part_sql("t", part)
            counts = {
                int(value): int(count)
                for value, count in self.connection.execute(
                    f"SELECT {expression} AS v, count(*) FROM {self.values} "
                    f"WHERE t IS NOT NULL GROUP BY v"
                ).fetchall()
            }
            spec = PARTS[part]
            assert spec.labels is not None
            bars = [
                Bar(label=label, count=counts.get(index + spec.first, 0))
                for index, label in enumerate(spec.labels)
            ]
            return Chart("bars", view, views, rows, missing, distinct, bars=bars, ordered=True)
        span = float(high) - float(low)
        allowed = [step for step in TIMELINE_STEPS if has_time or step != "hour"]
        if year_only:
            allowed = ["year"]
        step = params.step if params.step in allowed else auto_step(span, has_time, year_only)
        allowed = [s for s in allowed if span / STEP_SECONDS[s] <= 5_000]
        rows_by_period = self.connection.execute(
            f"SELECT strftime(date_trunc('{step}', t), '{STEP_FORMATS[step]}') AS p, count(*) "
            f"FROM {self.values} WHERE t IS NOT NULL GROUP BY date_trunc('{step}', t), p "
            f"ORDER BY date_trunc('{step}', t)"
        ).fetchall()
        points = [Bar(label=str(period), count=int(count)) for period, count in rows_by_period]
        return Chart(
            "timeline",
            "timeline",
            views,
            rows,
            missing,
            distinct,
            bars=points,
            ordered=True,
            step=step,
            steps=allowed,
        )

    # --- Text and identifiers ------------------------------------------------------------------

    def text(self) -> Chart:
        rows, missing, distinct = self.counts()
        expression = "length(CAST(t AS VARCHAR))"
        stats = self.numeric_stats(expression)
        chart = Chart("histogram", "lengths", ["lengths"], rows, missing, distinct, stats=stats)
        chart.bins = self.histogram(expression, stats)
        chart.examples = [bar["label"] for bar in self.bars("t", "c DESC, v", 3)]
        return chart

    def identifier(self) -> Chart:
        rows, missing, distinct = self.counts()
        duplicated = self.one(
            f"SELECT count(*) FROM (SELECT t FROM {self.values} WHERE t IS NOT NULL "
            f"GROUP BY t HAVING count(*) > 1)"
        )
        examples = [bar["label"] for bar in self.bars("t", "c DESC, v", 3)]
        return Chart(
            "identifier",
            "summary",
            ["summary"],
            rows,
            missing,
            distinct,
            examples=examples,
            duplicates=int(duplicated[0]),
        )

    def build(self, params: ChartParams) -> Chart:
        kind = self.column.type
        if kind == "numeric_continuous":
            return self.continuous(params)
        if kind in ("numeric_discrete", "categorical_nominal", "categorical_ordinal", "boolean"):
            return self.categorical(params)
        if kind == "datetime":
            return self.datetime(params)
        if kind == "text":
            return self.text()
        if kind == "identifier":
            return self.identifier()
        rows, missing, distinct = self.counts()
        return Chart(
            "empty",
            "none",
            [],
            rows,
            missing,
            distinct,
            message="Ignored column: left out of analyses.",
        )


def prepare_chart(
    connection: duckdb.DuckDBPyConnection, source: str, column: ColumnInput, params: ChartParams
) -> Chart:
    return ChartBuilder(connection, source, column).build(params)
