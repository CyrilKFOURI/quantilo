"""The statistical photo of a dataset version, kept at every import (a few kB): schema, effective
types, rows, missing values, 101 quantiles per number, the 50 most frequent values per category,
periods per date, duplicates, and the values of the saved KPIs.

Two photos are enough to say what changed (`compare.py`), even when the old rows are gone.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any, Literal

import duckdb

from quantilo.catalog import ColumnView
from quantilo.ingestion.sql import lit
from quantilo.profiling.overview import duplicate_rows
from quantilo.typing.parsers import typed_sql
from quantilo.typing.types import NUMERIC_TYPES, AnalysisType, Parser

SNAPSHOT_VERSION = 1
LEVELS = [i / 100 for i in range(101)]  # quantiles 0 %, 1 % … 100 %
TOP_LEVELS = 50
FEW_NUMBERS = 20  # a number with this many distinct values or fewer also keeps its frequencies
EXACT_QUANTILES = 200_000  # beyond, approx_quantile (quantile_cont on 101 levels is slow)
MAX_PERIODS = 400
LEVEL_TYPES: tuple[AnalysisType, ...] = ("categorical_nominal", "categorical_ordinal", "boolean")

Grain = Literal["day", "month", "year"]
GRAIN_FORMATS: dict[Grain, str] = {"day": "%Y-%m-%d", "month": "%Y-%m", "year": "%Y"}


@dataclass
class Numbers:
    count: int  # finite values
    mean: float | None
    median: float | None
    std: float | None
    min: float | None
    max: float | None
    quantiles: list[float]  # at LEVELS, empty without values
    outliers: int  # beyond 1.5 IQR of this version's own quartiles (as on the variable page)


@dataclass
class LevelCount:
    value: str
    count: int


@dataclass
class Levels:
    top: list[LevelCount]  # most frequent first, TOP_LEVELS at most
    others: int  # rows holding a value outside the top
    complete: bool  # every distinct value is in the top


@dataclass
class PeriodCount:
    label: str  # 2026-10-01, 2026-10 or 2026, as the grain says
    count: int


@dataclass
class Dates:
    min: str | None
    max: str | None
    grain: Grain
    periods: list[PeriodCount]
    future: int  # values after the day the photo was taken


@dataclass
class ColumnPhoto:
    name: str
    position: int
    storage_dtype: str
    type: AnalysisType
    detected_type: AnalysisType
    role: str | None
    parser: dict[str, Any]
    order: list[str] | None
    excluded: bool
    present: int  # values read in the column's type
    missing: int
    distinct: int
    numbers: Numbers | None = None
    levels: Levels | None = None
    dates: Dates | None = None


@dataclass
class KpiPoint:
    label: str
    value: float | None


@dataclass
class KpiValue:
    """A saved KPI at the time of the photo: its headline measure overall and per period."""

    name: str
    measure: str
    value: float | None
    periods: list[KpiPoint] = field(default_factory=list)


@dataclass
class Snapshot:
    taken_at: str
    rows: int
    duplicates: int
    columns: list[ColumnPhoto]
    kpis: list[KpiValue] = field(default_factory=list)
    types_version: int = 0  # the column types the photo was read with
    version: int = SNAPSHOT_VERSION

    def column(self, name: str) -> ColumnPhoto | None:
        return next((c for c in self.columns if c.name == name), None)


def _float(value: object) -> float | None:
    return None if value is None else float(value)  # type: ignore[arg-type]


def grain_for(first: date, last: date) -> Grain:
    days = (last - first).days
    if days <= 92:
        return "day"
    return "month" if days <= 7305 else "year"


def take_snapshot(
    connection: duckdb.DuckDBPyConnection,
    source: str,
    views: Sequence[ColumnView],
    taken_at: str | None = None,
    types_version: int = 0,
) -> Snapshot:
    """One aggregate pass for every column, then one small query per category and per date."""
    stamp = taken_at or datetime.now(UTC).isoformat()
    today = stamp[:10]
    row = connection.execute(f"SELECT count(*) FROM {source}").fetchone()
    rows = int(row[0]) if row else 0
    quantile = "quantile_cont" if rows <= EXACT_QUANTILES else "approx_quantile"
    levels_sql = "[" + ", ".join(f"{level:g}" for level in LEVELS) + "]"

    parts: list[str] = []
    for view in views:
        typed = typed_sql(view.name, view.parser)
        parts += [f"count({typed})", f"count(DISTINCT {typed})"]
        if view.type in NUMERIC_TYPES:
            x = f"CAST({typed} AS DOUBLE)"
            finite = f"FILTER (WHERE isfinite({x}))"
            parts += [
                f"count({x}) {finite}",
                f"avg({x}) {finite}",
                f"median({x}) {finite}",
                f"stddev_samp({x}) {finite}",
                f"min({x}) {finite}",
                f"max({x}) {finite}",
                f"{quantile}({x}, {levels_sql}) {finite}",
            ]
        elif view.type == "datetime":
            parts += [
                f"min(CAST({typed} AS DATE))",
                f"max(CAST({typed} AS DATE))",
                f"count(*) FILTER (WHERE CAST({typed} AS DATE) > DATE {lit(today)})",
            ]
    values = list(connection.execute(f"SELECT {', '.join(parts)} FROM {source}").fetchone() or ())

    photos: list[ColumnPhoto] = []
    for view in views:
        present, distinct = int(values.pop(0) or 0), int(values.pop(0) or 0)
        photo = ColumnPhoto(
            name=view.name,
            position=view.position,
            storage_dtype=view.storage_dtype,
            type=view.type,
            detected_type=view.detected_type,
            role=view.role,
            parser=view.parser.model_dump(),
            order=view.order,
            excluded=view.excluded,
            present=present,
            missing=rows - present,
            distinct=distinct,
        )
        typed = typed_sql(view.name, view.parser)
        if view.type in NUMERIC_TYPES:
            count, mean, median, std, low, high, quantiles = values[:7]
            del values[:7]
            photo.numbers = Numbers(
                count=int(count or 0),
                mean=_float(mean),
                median=_float(median),
                std=_float(std),
                min=_float(low),
                max=_float(high),
                quantiles=[float(q) for q in quantiles] if quantiles else [],
                outliers=0,
            )
            if distinct <= FEW_NUMBERS:
                photo.levels = _levels(connection, source, typed, present)
        elif view.type == "datetime":
            first, last, future = values[:3]
            del values[:3]
            photo.dates = _dates(connection, source, typed, first, last, int(future or 0))
        elif view.type in LEVEL_TYPES:
            photo.levels = _levels(connection, source, typed, present)
        photos.append(photo)

    _count_outliers(connection, source, views, photos)
    return Snapshot(
        taken_at=stamp,
        rows=rows,
        duplicates=duplicate_rows(connection, source) if rows else 0,
        columns=photos,
        types_version=types_version,
    )


def _count_outliers(
    connection: duckdb.DuckDBPyConnection,
    source: str,
    views: Sequence[ColumnView],
    photos: list[ColumnPhoto],
) -> None:
    """Values beyond Q1 − 1.5 IQR or Q3 + 1.5 IQR, every number in one pass."""
    checks: list[tuple[ColumnPhoto, str]] = []
    for view, photo in zip(views, photos, strict=True):
        numbers = photo.numbers
        if numbers is None or len(numbers.quantiles) != len(LEVELS):
            continue
        q1, q3 = numbers.quantiles[25], numbers.quantiles[75]
        low, high = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        x = f"CAST({typed_sql(view.name, view.parser)} AS DOUBLE)"
        checks.append(
            (photo, f"count(*) FILTER (WHERE isfinite({x}) AND ({x} < {low!r} OR {x} > {high!r}))")
        )
    if not checks:
        return
    sql = ", ".join(expression for _, expression in checks)
    counts = connection.execute(f"SELECT {sql} FROM {source}").fetchone() or ()
    for (photo, _), count in zip(checks, counts, strict=True):
        assert photo.numbers is not None
        photo.numbers.outliers = int(count or 0)


def _levels(connection: duckdb.DuckDBPyConnection, source: str, typed: str, present: int) -> Levels:
    rows = connection.execute(
        f"SELECT value, count(*) AS n FROM (SELECT CAST({typed} AS VARCHAR) AS value "
        f"FROM {source}) WHERE value IS NOT NULL GROUP BY value ORDER BY n DESC, value "
        f"LIMIT {TOP_LEVELS + 1}"
    ).fetchall()
    top = [LevelCount(str(value), int(count)) for value, count in rows[:TOP_LEVELS]]
    return Levels(
        top=top,
        others=present - sum(level.count for level in top),
        complete=len(rows) <= TOP_LEVELS,
    )


def _dates(
    connection: duckdb.DuckDBPyConnection,
    source: str,
    typed: str,
    first: date | None,
    last: date | None,
    future: int,
) -> Dates:
    if first is None or last is None:
        return Dates(None, None, "day", [], future)
    grain = grain_for(first, last)
    label = f"strftime(CAST({typed} AS DATE), {lit(GRAIN_FORMATS[grain])})"
    rows = connection.execute(
        f"SELECT period, count(*) FROM (SELECT {label} AS period FROM {source}) "
        f"WHERE period IS NOT NULL GROUP BY period ORDER BY period LIMIT {MAX_PERIODS}"
    ).fetchall()
    return Dates(
        first.isoformat(),
        last.isoformat(),
        grain,
        [PeriodCount(str(p), int(n)) for p, n in rows],
        future,
    )


def views_of(snapshot: Snapshot) -> list[ColumnView]:
    """The columns as the photo read them: enough to read that version's rows again (row by
    row comparison, KPIs, relations) when its Parquet copy is kept."""
    return [
        ColumnView(
            name=c.name,
            position=c.position,
            storage_dtype=c.storage_dtype,
            type=c.type,
            role=c.role,
            parser=Parser.model_validate(c.parser),
            order=c.order,
            detected_type=c.detected_type,
            detected_role=c.role,
            confidence="high",
            reasons=[],
            overridden=c.type != c.detected_type,
            excluded=c.excluded,
            summary=None,
        )
        for c in snapshot.columns
    ]


def snapshot_from_dict(data: dict[str, Any]) -> Snapshot:
    """A photo read back from its JSON (`to_dict`)."""
    columns = []
    for raw in data["columns"]:
        column = dict(raw)
        numbers, levels, dates = column.pop("numbers"), column.pop("levels"), column.pop("dates")
        photo = ColumnPhoto(**column)
        if numbers:
            photo.numbers = Numbers(**{**numbers, "quantiles": numbers["quantiles"] or []})
        if levels:
            photo.levels = Levels(
                [LevelCount(**level) for level in levels["top"]],
                levels["others"],
                levels["complete"],
            )
        if dates:
            photo.dates = Dates(
                dates["min"],
                dates["max"],
                dates["grain"],
                [PeriodCount(**period) for period in dates["periods"]],
                dates["future"],
            )
        columns.append(photo)
    return Snapshot(
        taken_at=data["taken_at"],
        rows=data["rows"],
        duplicates=data["duplicates"],
        columns=columns,
        kpis=[
            KpiValue(k["name"], k["measure"], k["value"], [KpiPoint(**p) for p in k["periods"]])
            for k in data.get("kpis", [])
        ],
        types_version=data.get("types_version", 0),
        version=data.get("version", SNAPSHOT_VERSION),
    )
