"""Compute a KPI: periods, top groups, every cell and total, tiles, and the automatic choices.

The browser receives cells (value, previous period, same period last year, rows) and decides
nothing about the data: "show as", layout, shading and format are applied there from these cells.
"""

import calendar
import functools
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Protocol

import duckdb

from quantilo.errors import ExplorerError
from quantilo.kpi import mix as kpi_mix
from quantilo.kpi import query as q
from quantilo.kpi import rules
from quantilo.kpi import signals as sig
from quantilo.kpi import to_date as kpi_to_date
from quantilo.kpi import trend as kpi_trend
from quantilo.kpi.definition import (
    ADDITIVE,
    NUMERIC_AGGS,
    RATIO_ID,
    Definition,
    Grain,
    Measure,
)
from quantilo.typing.types import AnalysisType, Parser

NUMBERS = {"numeric_continuous", "numeric_discrete"}
SPLITS = {"categorical_nominal", "categorical_ordinal", "boolean"}
MAX_POINTS = 5_000  # periods by groups drawn in one chart
MAX_CELLS = 50_000  # cells of the pivot table
SMALL_MULTIPLES = 6  # panels of the first split when there are two splits over time
SPARK_PERIODS = 24
YOY_GRAINS = ("day", "week", "month", "quarter")
# 054: a cell resting on few values (R-SIG-03) and a rate the mix splits (R-SIG-07)
LOW_N_AGGS = frozenset({"mean", "median", "weighted_mean", "true_share", "share"})
RATE_AGGS = frozenset({"mean", "weighted_mean", "true_share", "share"})


class KpiError(ExplorerError):
    def __init__(
        self,
        message: str,
        code: str = "kpi",
        field: str | None = None,
        suggestion: str | None = None,
    ) -> None:
        super().__init__(message, code=code, status_code=422)
        self.field = field
        self.suggestion = suggestion
        self.details = {"field": field, "suggestion": suggestion}


class Column(Protocol):
    name: str
    type: AnalysisType
    parser: Parser
    role: str | None
    order: list[str] | None


# --- Output ----------------------------------------------------------------------------------


@dataclass
class PeriodOut:
    start: str  # ISO date of the period's first day
    label: str
    partial: bool


@dataclass
class SplitOut:
    column: str
    levels: list[str]  # in display order, "Other" last
    others: int  # labels folded into "Other"
    ordinal: bool  # natural order and a sequential palette (R-VIS-33)


@dataclass
class SeriesOut:
    measure: str
    g1: str | None  # None = every label of the split (a total)
    g2: str | None
    values: list[float | None]
    previous: list[float | None]
    last_year: list[float | None]
    rows: list[int]
    overall: float | None  # over the whole displayed range
    trend: list[float | None] | None = None  # Show as "Trend" only (053, R-SHOW-02)


@dataclass
class MeasureOut:
    id: str
    label: str
    agg: str  # an aggregate, or "ratio"
    column: str | None
    additive: bool
    approximate: bool
    format: rules.NumberFormat
    chart: rules.Decision
    share: rules.Option


@dataclass
class TopGroup:
    label: str
    share: float


@dataclass
class GoalOut:
    """061, P11: the tile's value against its target, said in words (never by colour alone)."""

    target: float | None
    attained: float | None  # actual ÷ target
    gap: float | None  # actual - target
    source: str  # "column" or "value"
    reason: str


@dataclass
class Tile:
    measure: str
    label: str
    period: str | None
    value: float | None
    previous: float | None
    change: float | None  # relative change vs the previous period
    last_year: float | None
    change_yoy: float | None
    spark: list[float | None]
    spark_index: int | None
    top: TopGroup | None = None
    to_date: kpi_to_date.ToDate | None = None  # the started period at an equal date (054)
    goal: GoalOut | None = None  # actual vs target (061)


@dataclass
class Resolved:
    grain: rules.Decision | None
    layout: rules.Decision
    shading: rules.Decision
    time: bool  # a time axis with at least two periods (R-TIME-04)
    why: list[str] = field(default_factory=list)
    trend: rules.Decision | None = None  # the trend's method when Show as is "Trend" (053)


@dataclass
class KpiResult:
    name: str
    rows: int  # rows counted in the displayed range
    periods: list[PeriodOut]
    splits: list[SplitOut]
    measures: list[MeasureOut]
    series: list[SeriesOut]
    tiles: list[Tile]
    resolved: Resolved
    warnings: list[str]
    signals: list[sig.Signal] = field(default_factory=list)  # 054: anomalies, quality, mix

    @property
    def value(self) -> float | None:
        """The headline value: the ratio when there is one, otherwise the first measure, over
        the whole displayed range and every group (a recipe's key figure, 030)."""
        measure = next(
            (m for m in self.measures if m.id == RATIO_ID),
            self.measures[0] if self.measures else None,
        )
        if measure is None:
            return None
        line = next(
            (s for s in self.series if s.measure == measure.id and s.g1 is None and s.g2 is None),
            None,
        )
        return None if line is None else line.overall


# --- Dates -----------------------------------------------------------------------------------


def add_months(start: datetime, months: int) -> datetime:
    month = start.month - 1 + months
    year = start.year + month // 12
    month = month % 12 + 1
    day = min(start.day, calendar.monthrange(year, month)[1])
    return start.replace(year=year, month=month, day=day)


def next_start(start: datetime, grain: Grain) -> datetime:
    if grain == "day":
        return start + timedelta(days=1)
    if grain == "week":
        return start + timedelta(days=7)
    return add_months(start, {"month": 1, "quarter": 3, "year": 12}[grain])


def truncate(moment: datetime, grain: Grain) -> datetime:
    day = datetime.combine(moment.date(), time())
    if grain == "day":
        return day
    if grain == "week":
        return day - timedelta(days=day.weekday())
    if grain == "month":
        return day.replace(day=1)
    if grain == "quarter":
        return day.replace(month=(day.month - 1) // 3 * 3 + 1, day=1)
    return day.replace(month=1, day=1)


def year_before(start: datetime, grain: Grain) -> datetime:
    """The period holding the same date one year earlier."""
    return truncate(add_months(start, -12), grain)


def period_label(start: datetime, grain: Grain) -> str:
    if grain == "day":
        return f"{start.day} {start:%b %Y}"
    if grain == "week":
        return f"Wk {start.day} {start:%b %Y}"
    if grain == "month":
        return f"{start:%b %Y}"
    if grain == "quarter":
        return f"Q{(start.month - 1) // 3 + 1} {start.year}"
    return str(start.year)


def as_datetime(value: date | datetime) -> datetime:
    return value if isinstance(value, datetime) else datetime.combine(value, time())


# --- Validation ------------------------------------------------------------------------------


def _check(definition: Definition, by_name: dict[str, Column]) -> None:
    def known(name: str, where: str) -> Column:
        if name not in by_name:
            raise KpiError(f"Column {name} no longer exists.", "unknown_column", where)
        return by_name[name]

    for measure in definition.measures:
        where = f"measures.{measure.id}.column"
        if measure.column is None:
            continue
        column = known(measure.column, where)
        if measure.until is not None:  # a delay (053, R-MES-07)
            for name, place in (
                (measure.column, where),
                (measure.until, f"measures.{measure.id}.until"),
            ):
                if known(name, place).type != "datetime":
                    raise KpiError(
                        f"A delay runs between two dates: {name} is not one.", "wrong_type", place
                    )
            continue
        if measure.weight is not None:
            weight = known(measure.weight, f"measures.{measure.id}.weight")
            if weight.type not in NUMBERS:
                raise KpiError(
                    f"A weight is a number: {weight.name} is not.",
                    "wrong_type",
                    f"measures.{measure.id}.weight",
                )
        if measure.key is not None:
            known(measure.key, f"measures.{measure.id}.key")
        if measure.agg == "share" and column.type not in SPLITS:
            raise KpiError(
                f"“% of values” counts the values of a category: {column.name} is {column.type}.",
                "wrong_type",
                where,
            )
        if measure.agg in NUMERIC_AGGS and column.type not in NUMBERS:
            raise KpiError(
                f"A {measure.agg} needs a numeric column: {column.name} is not.",
                "wrong_type",
                where,
            )
        if measure.agg == "true_share" and column.type != "boolean":
            raise KpiError(
                f"“% true” needs a yes/no column: {column.name} is not.", "wrong_type", where
            )
    if definition.goal is not None and definition.goal.column is not None:
        column = known(definition.goal.column, "goal.column")
        if column.type not in NUMBERS:
            raise KpiError(
                f"A target is a number: {column.name} is not.", "wrong_type", "goal.column"
            )
    if definition.time is not None:
        column = known(definition.time.column, "time.column")
        if column.type != "datetime":
            raise KpiError(f"{column.name} is not a date.", "wrong_type", "time.column")
    for index, group in enumerate(definition.groups):
        column = known(group.column, f"groups.{index}.column")
        if column.type not in SPLITS:
            raise KpiError(
                f"Split by a category: {column.name} is {column.type}.",
                "wrong_type",
                f"groups.{index}.column",
            )


# --- Compute ---------------------------------------------------------------------------------


@dataclass
class _Window:
    grain: Grain
    decision: rules.Decision
    display: list[datetime]  # period starts shown
    grid: list[datetime]  # context periods (before) + displayed ones
    where_display: str
    where_grid: str
    first: datetime  # first and last dates of the displayed rows
    last: datetime
    data_first: datetime  # first and last dates of every row (context included)
    data_last: datetime
    gap: float


def _window(
    connection: duckdb.DuckDBPyConnection, definition: Definition, warnings: list[str]
) -> _Window:
    assert definition.time is not None
    data_first, data_last = q.one(connection, f"SELECT min(t), max(t) FROM {q.TABLE}")
    if data_first is None:
        raise KpiError("No row has a date here.", "no_rows")
    (undated,) = q.one(connection, f"SELECT count(*) FROM {q.TABLE} WHERE t IS NULL")
    if undated:
        warnings.append(f"{undated:,} rows without a date are left out.")
    gap = q.median_gap(connection, "t IS NOT NULL") or rules.GRAIN_SECONDS["day"]
    typical = rules.typical_grain(gap)
    period = definition.time.period
    requested = definition.time.grain
    if period.kind == "last":
        decision, warning = rules.resolve_grain(
            requested, typical, (data_last - data_first).total_seconds()
        )
        grain: Grain = decision.value  # type: ignore[assignment]
        starts = q.period_starts(connection, grain, data_first, data_last)
        low, high = starts[-min(period.n, len(starts))], data_last
    else:
        if period.kind == "ytd":
            low, high = datetime(data_last.year, 1, 1), data_last
        elif period.kind == "range":
            low = as_datetime(period.start) if period.start else data_first
            high = (
                as_datetime(period.end) + timedelta(days=1) - timedelta(microseconds=1)
                if period.end
                else data_last
            )
        else:
            low, high = data_first, data_last
        first, last = q.one(
            connection,
            f"SELECT min(t), max(t) FROM {q.TABLE} "
            f"WHERE t >= {q.timestamp(low)} AND t <= {q.timestamp(high)}",
        )
        if first is None:
            raise KpiError("No rows in this period.", "no_rows", "time.period")
        decision, warning = rules.resolve_grain(requested, typical, (last - first).total_seconds())
        grain = decision.value  # type: ignore[assignment]
        low, high = max(low, first), min(high, last)
    if warning:
        warnings.append(warning)
    display = q.period_starts(connection, grain, low, high)
    context = max(data_first, year_before(display[0], grain) - timedelta(days=1))
    grid = q.period_starts(connection, grain, min(context, display[0]), high)
    grid = [p for p in grid if p >= truncate(data_first, grain)]
    where_display = f"t >= {q.timestamp(low)} AND t <= {q.timestamp(high)}"
    where_grid = (
        f"t <= {q.timestamp(high)} AND (t >= {q.timestamp(low)} OR "
        f"(t >= {q.timestamp(grid[0])} AND t < {q.timestamp(display[0])}))"
    )
    first, last = q.one(connection, f"SELECT min(t), max(t) FROM {q.TABLE} WHERE {where_display}")
    if first is None:
        raise KpiError("No rows in this period.", "no_rows", "time.period")
    return _Window(
        grain,
        decision,
        display,
        grid,
        where_display,
        where_grid,
        first,
        last,
        data_first,
        data_last,
        gap,
    )


def _partial(start: datetime, end: datetime, first: datetime, last: datetime, gap: float) -> bool:
    """The data starts well after the period begins, or ends well before it ends."""
    late = start < first < end and (first - start).total_seconds() >= gap / 2
    early = start <= last < end and (end - last).total_seconds() - gap >= gap / 2
    return late or early


def _change(value: float | None, before: float | None) -> float | None:
    if value is None or before is None or before == 0:
        return None
    return (value - before) / abs(before)


def compute(
    connection: duckdb.DuckDBPyConnection,
    source: str,
    columns: Sequence[Column],
    definition: Definition,
) -> KpiResult:
    by_name = {c.name: c for c in columns}
    _check(definition, by_name)
    warnings: list[str] = []
    measures = definition.measures
    time_column = by_name[definition.time.column] if definition.time else None
    split_columns = [by_name[g.column] for g in definition.groups]
    ids = [m.id for m in measures] + ([RATIO_ID] if definition.ratio else [])
    # R-SIG-06: an order key and a measure that adds up: look for rows repeated exactly
    order_key = next((c for c in columns if c.role == "order_key"), None)
    if not any(definition.additive(i) for i in ids):
        order_key = None
    q.build_table(
        connection,
        source,
        time_column,
        split_columns,
        [(m, by_name[m.column] if m.column else None) for m in measures],
        {name: by_name[name] for m in measures for name in m.columns()},
        order_key,
        by_name[definition.goal.column] if definition.goal and definition.goal.column else None,
    )

    window = _window(connection, definition, warnings) if definition.time else None
    where_display = window.where_display if window else "true"
    (rows,) = q.one(connection, f"SELECT count(*) FROM {q.TABLE} WHERE {where_display}")
    if not rows:
        raise KpiError("No rows in this period.", "no_rows")
    approximate = rows > q.EXACT_MEDIAN_ROWS
    aggregates = [
        q.aggregate_sql(
            m.agg,
            f"v{i}",
            approximate,
            f"w{i}" if m.weight else None,
            f"k{i}" if m.key else None,
        )
        for i, m in enumerate(measures, start=1)
    ]
    if approximate and any(m.agg == "median" for m in measures):
        warnings.append("Medians are approximate (≈) over 200,000 rows.")
    reads = _Counts(measures, len(aggregates))

    # Splits: top N by the main measure, the rest folded into "Other" before aggregating.
    splits: list[SplitOut] = []
    for index, column in enumerate(split_columns, start=1):
        levels = q.top_levels(connection, index, aggregates[0], where_display, None)
        if len(levels) < 2:
            raise KpiError(
                f"{column.name} has a single value here.",
                "single_level",
                f"groups.{index - 1}.column",
            )
        cap = definition.top
        if index == 1 and len(split_columns) == 2 and window and cap > SMALL_MULTIPLES:
            cap = SMALL_MULTIPLES
            if len(levels) > cap:
                warnings.append(
                    f"The first split keeps its {cap} largest groups: one small chart each."
                )
        kept = [level.label for level in levels[:cap]]
        others = len(levels) - len(kept)
        if others:
            q.fold_others(connection, index, kept)
        ordinal = bool(column.order) or column.type == "boolean"
        if column.order:
            rank = {label: i for i, label in enumerate(column.order)}
            kept.sort(key=lambda label: (label not in rank, rank.get(label, 0)))
        elif column.type == "boolean":
            kept.sort(key=lambda label: label != "True")
        elif definition.group_order == "alpha":
            kept.sort(key=str.lower)
        if q.EMPTY in kept:  # "(empty)" before "Other", after every real label
            kept.remove(q.EMPTY)
            kept.append(q.EMPTY)
        if others:
            kept.append(q.OTHER)
        splits.append(SplitOut(column.name, kept, others, ordinal))

    grain = window.grain if window else None
    display = window.display if window else []
    groups = [s.levels for s in splits]
    points = max(1, len(display)) * _product(len(levels) for levels in groups)
    if window and points > MAX_POINTS:
        raise KpiError(
            f"{points:,} points are too many to draw (at most {MAX_POINTS:,}).",
            "too_many_points",
            "time.grain",
            _coarser(window.grain),
        )
    cells_count = (
        max(1, len(display))
        * _product(len(levels) + 1 for levels in groups)
        * (len(measures) + (1 if definition.ratio else 0))
    )
    if cells_count > MAX_CELLS:
        raise KpiError(
            f"{cells_count:,} cells are too many for the table (at most {MAX_CELLS:,}).",
            "too_many_cells",
            "time.grain" if window else "top",
            _coarser(window.grain) if window else None,
        )

    every = aggregates + reads.sql  # 054: the values read per cell, after the measures
    timed = q.aggregate(connection, every, len(splits), grain, window.where_grid) if window else {}
    overall = q.aggregate(connection, every, len(splits), None, where_display)

    # Periods, with the incomplete ones at the edges marked.
    partial: dict[datetime, bool] = {}
    if window:
        for start in window.grid:
            end = next_start(start, window.grain)
            shown_part = start >= window.display[0] and _partial(
                start, end, window.first, window.last, window.gap
            )
            data_part = _partial(start, end, window.data_first, window.data_last, window.gap)
            partial[start] = shown_part or data_part
    periods = (
        [
            PeriodOut(start.date().isoformat(), period_label(start, window.grain), partial[start])
            for start in display
        ]
        if window
        else []
    )
    grid_index = {start: i for i, start in enumerate(window.grid)} if window else {}

    # Series: every combination of split labels and totals (None), for every measure.
    position = {m.id: i for i, m in enumerate(measures)}
    combos = _combos(groups)

    def raw(cells: dict, key: q.Cell, measure_id: str, filled: bool) -> float | None:
        """A cell's value; an empty period is 0 for rows and sums when `filled`."""
        if measure_id == RATIO_ID:
            assert definition.ratio is not None
            top = raw(cells, key, definition.ratio.numerator, filled)
            bottom = raw(cells, key, definition.ratio.denominator, filled)
            return rules.combine(definition.ratio.kind, top, bottom)
        row = cells.get(key)
        agg = measures[position[measure_id]].agg
        if row is None:
            return 0.0 if filled and agg in ADDITIVE else None
        value = row[position[measure_id]]
        if value is None and agg == "count":
            return 0.0
        return value

    def at(start: datetime | None, g1: str | None, g2: str | None, measure_id: str) -> float | None:
        """A period's value for context: absent before the data and for incomplete periods."""
        if start is None or start not in grid_index or partial.get(start):
            return None
        return raw(timed, (start, g1, g2), measure_id, True)

    series: list[SeriesOut] = []
    for measure_id in ids:
        for g1, g2 in combos:
            values, previous, last_year, counts = [], [], [], []
            for start in display:
                key = (start, g1, g2)
                values.append(raw(timed, key, measure_id, True))
                row = timed.get(key)
                counts.append(int(row[-1] or 0) if row else 0)
                index = grid_index[start]
                previous.append(
                    at(window.grid[index - 1] if index else None, g1, g2, measure_id)  # type: ignore[union-attr]
                )
                last_year.append(
                    at(year_before(start, window.grain), g1, g2, measure_id)  # type: ignore[union-attr]
                    if window and window.grain in YOY_GRAINS
                    else None
                )
            series.append(
                SeriesOut(
                    measure_id,
                    g1,
                    g2,
                    values,
                    previous,
                    last_year,
                    counts,
                    raw(overall, (None, g1, g2), measure_id, False),
                )
            )

    # Decisions, per measure then for the table.
    labels = {m.id: rules.measure_label(m) for m in measures}
    if definition.ratio:
        labels[RATIO_ID] = rules.ratio_label(definition.ratio, measures)
    timed_axis = bool(window) and len(display) >= 2
    measure_outs = [
        _measure_out(
            measure_id,
            definition,
            labels[measure_id],
            series,
            splits,
            timed_axis,
            len(display),
            by_name,
            approximate,
        )
        for measure_id in ids
    ]
    layout = rules.resolve_layout(timed_axis, len(splits), len(display))
    shading = rules.resolve_shading(layout.value, len(splits))  # type: ignore[arg-type]
    why = []
    if window:
        why.append(window.decision.reason)
        if not timed_axis:
            why.append("A single period: shown as values, not over time (R-TIME-04).")
    why.append(measure_outs[0].chart.reason)
    for split in splits:
        if split.others:
            why.append(
                f"Top {len(split.levels) - 1} {split.column} by {labels[ids[0]]}, "
                f"{split.others} others grouped as Other."
            )
    name_labels = [labels[RATIO_ID]] if definition.ratio else [labels[m.id] for m in measures]
    trend = None
    if window and timed_axis and definition.view.show_as == "trend":
        trend = kpi_trend.attach(series, [p.partial for p in periods], window.grain)
        why.append(trend.reason)
    name = rules.kpi_name(name_labels, grain if timed_axis else None, [s.column for s in splits])
    context = _Context(definition, measures, position, splits, periods, labels, raw)
    found = _signals(context, measure_outs, series, timed, overall, reads, window, timed_axis)
    if order_key is not None:
        keyed, copies = q.duplicated(connection, where_display)
        repeated = sig.duplicates(order_key.name, copies, keyed)
        if repeated:
            found.append(repeated)
    to_dates = _to_dates(connection, context, aggregates, window) if window and timed_axis else {}
    tiles = [_tile(m, series, periods, splits, to_dates.get(m.id)) for m in measure_outs]
    if definition.goal is not None:
        for tile in tiles:
            if tile.measure == definition.goal.measure:
                tile.goal = _goal(connection, definition, tile, window, where_display, periods)
    return KpiResult(
        name=name[:1].upper() + name[1:],
        rows=int(rows),
        periods=periods,
        splits=splits,
        measures=measure_outs,
        series=series,
        tiles=tiles,
        resolved=Resolved(
            window.decision if window else None, layout, shading, timed_axis, why, trend
        ),
        warnings=warnings,
        signals=sig.ordered(found),
    )


def _product(values: object) -> int:
    total = 1
    for value in values:  # type: ignore[attr-defined]
        total *= int(value)
    return total


def _coarser(grain: Grain) -> str | None:
    order = rules.GRAINS
    index = order.index(grain)
    return order[index + 1] if index + 1 < len(order) else None


def _combos(groups: list[list[str]]) -> list[tuple[str | None, str | None]]:
    first: list[str | None] = [None, *groups[0]] if groups else [None]
    second: list[str | None] = [None, *groups[1]] if len(groups) > 1 else [None]
    return [(g1, g2) for g1 in first for g2 in second]


def _measure_out(
    measure_id: str,
    definition: Definition,
    label: str,
    series: list[SeriesOut],
    splits: list[SplitOut],
    timed_axis: bool,
    periods: int,
    by_name: dict[str, Column],
    approximate: bool,
) -> MeasureOut:
    mine = [s for s in series if s.measure == measure_id]
    finest = [
        s
        for s in mine
        if (s.g1 is not None or not splits) and (s.g2 is not None or len(splits) < 2)
    ]
    values = [v for s in mine for v in [*s.values, s.overall] if v is not None]
    additive = definition.additive(measure_id)
    dominant = None
    if additive and splits:
        total = next((s.overall for s in mine if s.g1 is None and s.g2 is None), None)
        parts = [s.overall or 0.0 for s in mine if s.g1 is not None and s.g2 is None]
        if total and total > 0 and parts:
            dominant = max(parts) / total
    cells = [n for s in finest for n in s.rows] if timed_axis else []
    empty = sum(1 for n in cells if n == 0) / len(cells) if cells else 0.0
    facts = rules.Facts(
        time=timed_axis,
        periods=periods,
        groups=len(splits),
        levels=[len(s.levels) for s in splits],
        additive=additive,
        negative=any(v < 0 for v in values),
        dominant=dominant,
        empty=empty,
        measure=label,
        split=splits[0].column if splits else None,
    )
    if measure_id == RATIO_ID:
        assert definition.ratio is not None
        ratio = definition.ratio
        percent = ratio.kind == "margin" or (ratio.kind == "ratio" and ratio.percent)
        number = rules.resolve_format(None, values, None, percent)
        agg, column, approx = "ratio", None, False
    else:
        measure: Measure = next(m for m in definition.measures if m.id == measure_id)
        role = by_name[measure.column].role if measure.column else None
        number = (
            rules.delay_format(values)
            if measure.until
            else rules.resolve_format(measure.agg, values, role)
        )
        agg, column = measure.agg, measure.column
        approx = approximate and measure.agg == "median"
    return MeasureOut(
        measure_id,
        label,
        agg,
        column,
        additive,
        approx,
        number,
        rules.resolve_chart(facts),
        rules.share_allowed(facts),
    )


def _tile(
    measure: MeasureOut,
    series: list[SeriesOut],
    periods: list[PeriodOut],
    splits: list[SplitOut],
    to_date: kpi_to_date.ToDate | None = None,
) -> Tile:
    total = next(s for s in series if s.measure == measure.id and s.g1 is None and s.g2 is None)
    top = None
    if splits and measure.additive and total.overall:
        parts = [
            s
            for s in series
            if s.measure == measure.id and s.g1 is not None and s.g2 is None and s.g1 != q.OTHER
        ]
        if parts:
            best = max(parts, key=lambda s: s.overall or 0.0)
            top = TopGroup(best.g1 or "", (best.overall or 0.0) / total.overall)
    complete = [i for i, p in enumerate(periods) if not p.partial]
    if len(periods) < 2 or not complete:
        return Tile(
            measure.id,
            measure.label,
            None,
            total.overall,
            None,
            None,
            None,
            None,
            [],
            None,
            top,
            to_date,
        )
    index = complete[-1]
    value, before, year = total.values[index], total.previous[index], total.last_year[index]
    start = max(0, index + 1 - SPARK_PERIODS)
    return Tile(
        measure.id,
        measure.label,
        periods[index].label,
        value,
        before,
        _change(value, before),
        year,
        _change(value, year),
        total.values[start : index + 1],
        index - start,
        top,
        to_date,
    )


def _goal(
    connection: duckdb.DuckDBPyConnection,
    definition: Definition,
    tile: Tile,
    window: "_Window | None",
    where_display: str,
    periods: list[PeriodOut],
) -> GoalOut:
    """R-GOAL-01 (a target column, over the tile's rows) and R-GOAL-02 (a typed target)."""
    goal = definition.goal
    assert goal is not None
    where, span = where_display, "the displayed range"
    if window and tile.period is not None:
        index = max(i for i, p in enumerate(periods) if not p.partial)
        start = window.display[index]
        end = next_start(start, window.grain)
        where = f"{where_display} AND t >= {q.timestamp(start)} AND t < {q.timestamp(end)}"
        span = tile.period
    if goal.column is not None:
        additive = definition.additive(goal.measure)
        aggregate = "sum" if additive else "avg"
        (found,) = q.one(connection, f"SELECT {aggregate}(gv) FROM {q.TABLE} WHERE {where}")
        target = None if found is None else float(found)
        how = "Σ" if additive else "the mean of"
        reason = f"R-GOAL-01 · Target: {how} {goal.column} over the same rows ({span})"
        if not additive:
            reason += f": {tile.label} does not add up, so its target is averaged"
        source = "column"
    else:
        target = goal.value
        grain = window.grain if window else "period"
        per = f"for {span}" if tile.period is None else f"per {grain}"
        reason = f"R-GOAL-02 · Target typed: {target:,.4g} {per} ({span} shown)"
        source = "value"
    actual = tile.value
    attained = actual / target if actual is not None and target else None
    gap = actual - target if actual is not None and target is not None else None
    return GoalOut(target, attained, gap, source, reason)


# --- Signals (054) ---------------------------------------------------------------------------


class _Counts:
    """What each cell's value rests on, read in the same query as the measures: the values read
    (`count(v)`), and for a weighted mean or a share of keys its own n and weight."""

    def __init__(self, measures: Sequence[Measure], offset: int) -> None:
        self.sql: list[str] = []
        self.filled: dict[str, int] = {}
        self.n: dict[str, int] = {}
        self.mass: dict[str, int] = {}

        def add(sql: str) -> int:
            self.sql.append(sql)
            return offset + len(self.sql) - 1

        for i, m in enumerate(measures, start=1):
            if m.column is None:
                continue
            self.filled[m.id] = add(f"count(v{i})")
            if m.agg == "weighted_mean" and m.weight:
                both = f"CASE WHEN v{i} IS NOT NULL AND w{i} IS NOT NULL THEN 1 END"
                self.n[m.id] = add(f"count({both})")
                self.mass[m.id] = add(f"sum(CASE WHEN v{i} IS NOT NULL THEN w{i} END)")
            elif m.agg == "share" and m.key:
                self.n[m.id] = add(f"count(DISTINCT CASE WHEN v{i} IS NOT NULL THEN k{i} END)")


Reader = Callable[[dict, q.Cell, str, bool], float | None]


@dataclass
class _Context:
    definition: Definition
    measures: list[Measure]
    position: dict[str, int]
    splits: list[SplitOut]
    periods: list[PeriodOut]
    labels: dict[str, str]
    raw: Reader  # compute's cell reader: (cells, key, measure id, filled) -> value

    def value(self, cells: dict, key: q.Cell, measure_id: str) -> float | None:
        return self.raw(cells, key, measure_id, True)

    def measure(self, measure_id: str) -> Measure:
        return self.measures[self.position[measure_id]]


def _n(context: _Context, counts: _Counts, cells: dict, key: q.Cell, measure_id: str) -> float:
    """R-SIG-03: the values a cell rests on; a ratio's is the smaller of its two measures'."""
    if measure_id == RATIO_ID:
        ratio = context.definition.ratio
        assert ratio is not None
        return min(
            _n(context, counts, cells, key, ratio.numerator),
            _n(context, counts, cells, key, ratio.denominator),
        )
    row = cells.get(key)
    if row is None:
        return 0.0
    measure = context.measure(measure_id)
    if measure.agg == "distinct":
        return float(row[context.position[measure_id]] or 0)
    index = counts.n.get(measure_id, counts.filled.get(measure_id))
    if measure.agg == "count" or index is None:
        return float(row[-1] or 0)
    return float(row[index] or 0)


def _low_n_applies(context: _Context, measure_id: str) -> bool:
    if measure_id == RATIO_ID:
        ratio = context.definition.ratio
        return ratio is not None and ratio.kind in ("ratio", "margin")
    return context.measure(measure_id).agg in LOW_N_AGGS


def _mass(
    context: _Context, counts: _Counts, cells: dict, key: q.Cell, measure_id: str
) -> float | None:
    """R-SIG-07: a group's weight in a rate (its denominator), None when it has none."""
    if measure_id == RATIO_ID:
        ratio = context.definition.ratio
        assert ratio is not None
        if ratio.kind == "difference":
            return None
        side = ratio.denominator if ratio.kind == "ratio" else ratio.numerator
        if context.measure(side).agg not in {*ADDITIVE, "distinct"}:
            return None
        return context.value(cells, key, side) or 0.0
    measure = context.measure(measure_id)
    if measure.agg not in RATE_AGGS:
        return None
    row = cells.get(key)
    if row is None:
        return 0.0
    index = counts.mass.get(measure_id, counts.n.get(measure_id, counts.filled.get(measure_id)))
    return None if index is None else float(row[index] or 0)


def _signals(
    context: _Context,
    measure_outs: list[MeasureOut],
    series: list[SeriesOut],
    timed: dict,
    overall: dict,
    counts: _Counts,
    window: _Window | None,
    timed_axis: bool,
) -> list[sig.Signal]:
    found: list[sig.Signal] = []
    periods = context.periods
    combos = _combos([s.levels for s in context.splits])
    several = len(measure_outs) > 1
    totals = {s.measure: s for s in series if s.g1 is None and s.g2 is None}
    over_time = window if window and timed_axis else None
    for out in measure_outs:
        label = context.labels[out.id]
        if over_time:
            # P12: the total series
            found += sig.series_signals(
                out.id,
                label if several else None,
                totals[out.id].values,
                periods,
                over_time.grain,
                totals[out.id].rows if out.additive else None,
            )
            # P14: the rate split by the first group, between the two last complete periods
            if context.splits:
                change = _mix(context, counts, out, timed, over_time)
                if change:
                    found.append(change)
        # P13: the measure's column, then its cells
        row = overall.get((None, None, None))
        if out.id in counts.filled and row:
            filled, rows = int(row[counts.filled[out.id]] or 0), int(row[-1] or 0)
            empty = sig.empty_measure(out.id, label, filled, rows)
            if empty:
                found.append(empty)
        if _low_n_applies(context, out.id):
            cells = []
            for g1, g2 in combos:
                if window:
                    for start, period in zip(window.display, periods, strict=True):
                        n = _n(context, counts, timed, (start, g1, g2), out.id)
                        cells.append(sig.SignalCell(period.start, g1, g2, n))
                n = _n(context, counts, overall, (None, g1, g2), out.id)
                cells.append(sig.SignalCell(None, g1, g2, n))
            low = sig.low_n(out.id, label, cells)
            if low:
                found.append(low)
    if over_time and measure_outs:
        thin = sig.thin_periods(periods, totals[measure_outs[0].id].rows)
        if thin:
            found.append(thin)
    return found


def _mix(
    context: _Context, counts: _Counts, out: MeasureOut, timed: dict, window: _Window
) -> sig.Signal | None:
    periods = context.periods
    complete = [i for i, p in enumerate(periods) if not p.partial]
    if not complete or complete[-1] < 1 or periods[complete[-1] - 1].partial:
        return None
    after = complete[-1]
    before = after - 1
    starts = window.display[before], window.display[after]
    if _mass(context, counts, timed, (starts[1], None, None), out.id) is None:
        return None
    split = context.splits[0]
    parts = []
    for level in split.levels:
        early, late = (starts[0], level, None), (starts[1], level, None)
        parts.append(
            kpi_mix.Part(
                level,
                context.value(timed, early, out.id),
                context.value(timed, late, out.id),
                _mass(context, counts, timed, early, out.id) or 0.0,
                _mass(context, counts, timed, late, out.id) or 0.0,
            )
        )
    return kpi_mix.mix_signal(
        out.id,
        context.labels[out.id],
        out.format.percent,
        split.column,
        periods[before].label,
        periods[after].start,
        context.value(timed, (starts[0], None, None), out.id),
        context.value(timed, (starts[1], None, None), out.id),
        parts,
    )


def _to_dates(
    connection: duckdb.DuckDBPyConnection,
    context: _Context,
    aggregates: list[str],
    window: _Window,
) -> dict[str, kpi_to_date.ToDate]:
    """R-TILE-03 · 04: the started last period at an equal date, for every measure."""
    grain = window.grain
    start = window.display[-1]
    end = next_start(start, grain)
    ends_early = (
        start <= window.last < end
        and (end - window.last).total_seconds() - window.gap >= window.gap / 2
    )
    if grain not in kpi_to_date.GRAINS or not ends_early:
        return {}
    current, previous, last_year, elapsed = kpi_to_date.windows(
        start, window.last, window.gap, grain, year_before(start, grain)
    )
    spans = (current, previous, last_year)
    rows = kpi_to_date.values(connection, aggregates, spans, window.data_first)
    definition = context.definition

    def read(index: int, measure_id: str) -> float | None:
        row = rows[index]
        if row is None:
            return None
        if measure_id == RATIO_ID:
            ratio = definition.ratio
            assert ratio is not None
            return rules.combine(
                ratio.kind, read(index, ratio.numerator), read(index, ratio.denominator)
            )
        agg = context.measure(measure_id).agg
        if not row:
            return 0.0 if agg in ADDITIVE else None
        value = row[context.position[measure_id]]
        return 0.0 if value is None and agg == "count" else value

    label = period_label(start, grain)
    return {
        measure_id: kpi_to_date.to_date(
            label,
            spans,
            elapsed,
            functools.partial(read, measure_id=measure_id),
            definition.additive(measure_id),
        )
        for measure_id in context.labels
    }
