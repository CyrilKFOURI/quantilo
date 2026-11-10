"""Chart data and tests for a pair (or triple) of columns, chosen from their analysis types."""

import math
from dataclasses import dataclass, field
from itertools import combinations
from typing import Literal

import duckdb
import numpy as np
from scipy import stats as scipy_stats

from quantilo.compare import bootstrap as bs
from quantilo.compare.multiple import Correction, MultipleTesting, correct
from quantilo.compare.stats import (
    BivariateTest,
    GroupStats,
    contingency_tests,
    correlation_effect,
    group_tests,
    pearson,
    rank_correlations,
    welch_interval,
    welch_t,
)
from quantilo.errors import ExplorerError
from quantilo.filters.scope import TypedColumn
from quantilo.ingestion.sql import lit
from quantilo.timeseries import builder
from quantilo.timeseries import series as ts
from quantilo.timeseries.trend import trend as trend_of
from quantilo.typing.dates import STEP_FORMATS, auto_step
from quantilo.typing.parsers import typed_sql
from quantilo.typing.types import AnalysisType, Parser
from quantilo.variable.source import SEED, clean

Role = Literal["num", "cat", "date"]
Kind = Literal["scatter", "groups", "contingency", "timeline", "date_counts"]

ROLES: dict[str, Role] = {
    "numeric_continuous": "num",
    "numeric_discrete": "num",
    "categorical_nominal": "cat",
    "categorical_ordinal": "cat",
    "boolean": "cat",
    "datetime": "date",
}
MAX_POINTS = 20_000
TEST_SAMPLE = 5_000
VIOLIN_SAMPLE = 1_000
MAX_BOXES = 15  # R-CMP-05: boxes on a groups chart, "Others" included (axis * colours)
MAX_COLORS = 5  # R-CMP-05: colour levels that stay readable
SWAP_ABOVE = 5  # R-CMP-04: a colour with more values than this (and than the axis) goes on the axis
MAX_TABLE = 15
MAX_SERIES = 8
MAX_DISCRETE_CATEGORIES = (
    12  # R-CMP-01: a whole number with this many values or fewer reads as levels
)
DENSITY_BINS = 40
OTHERS = "Others"


class NotComparableError(ExplorerError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="not_comparable", status_code=422)


@dataclass
class CompareColumn:
    name: str
    type: AnalysisType
    parser: Parser
    order: list[str] | None = None
    semantic_role: str | None = None  # amount, quantity, age… (column roles)
    as_category: bool = False  # a whole number with few values, read as levels (R-CMP-01)

    @property
    def role(self) -> Role:
        if self.as_category:
            return "cat"
        role = ROLES.get(self.type)
        if role is None:
            raise NotComparableError(f"{self.name!r} ({self.type}) can't be compared.")
        return role


@dataclass
class Points:
    x: list[float]
    y: list[float]
    z: list[float] | None = None
    color: list[str] | None = None


@dataclass
class Regression:
    slope: float
    intercept: float
    r2: float | None


@dataclass
class Lowess:
    """LOWESS curve of a scatter (064, R-SM-12): drawn by default when it bends away from the
    straight line, i.e. explains clearly more of Y than the line on the same points."""

    x: list[float]
    y: list[float]
    r2: float  # share of Y's variance the curve explains
    line_r2: float  # the same for the least-squares line, on the same points
    suggested: bool
    reason: str


LOWESS_MIN_POINTS = 30
LOWESS_POINTS = 2_000  # points fitted at most (cost)
LOWESS_GAIN = 0.05  # R² gained over the line to call the relation non-linear


def lowess_curve(xs: np.ndarray, ys: np.ndarray) -> Lowess | None:
    """Cleveland's robust LOWESS (frac 2/3, 3 passes) on at most LOWESS_POINTS points, on a grid
    of 100 x values; None under LOWESS_MIN_POINTS points or for a constant column."""
    from statsmodels.nonparametric.smoothers_lowess import lowess

    xs, ys = xs[:LOWESS_POINTS], ys[:LOWESS_POINTS]
    n = len(xs)
    if n < LOWESS_MIN_POINTS or np.ptp(xs) == 0 or np.ptp(ys) == 0:
        return None
    delta = 0.01 * float(np.ptp(xs))
    fitted = np.asarray(
        lowess(ys, xs, frac=2 / 3, it=3, delta=delta, return_sorted=False), dtype=float
    )
    total = float(((ys - ys.mean()) ** 2).sum())
    slope, intercept = np.polyfit(xs, ys, 1)
    line_r2 = 1 - float(((ys - (slope * xs + intercept)) ** 2).sum()) / total
    usable = np.isfinite(fitted)
    r2 = 1 - float(((ys[usable] - fitted[usable]) ** 2).sum()) / total
    order = np.argsort(xs[usable])
    grid = np.linspace(float(xs.min()), float(xs.max()), 100)
    curve = np.interp(grid, xs[usable][order], fitted[usable][order])
    suggested = r2 - line_r2 >= LOWESS_GAIN
    shares = f"LOWESS explains {r2:.0%} of Y's variance, the line {line_r2:.0%}"
    reason = (
        f"{shares}: the relation bends." if suggested else f"{shares}: a straight line is enough."
    )
    return Lowess(
        [float(v) for v in grid], [float(v) for v in curve], r2, line_r2, suggested, reason
    )


@dataclass
class Density:
    x_edges: list[float]
    y_edges: list[float]
    counts: list[list[int]]  # [y bin][x bin]


@dataclass
class Group:
    label: str
    n: int
    mean: float | None
    std: float | None
    q1: float | None
    median: float | None
    q3: float | None
    low: float | None  # whisker ends: Tukey fences clipped to the data range
    high: float | None
    ci_low: float | None  # 95% confidence interval of the mean
    ci_high: float | None
    sample: list[float] = field(default_factory=list)
    color: str | None = None  # the colour level of this box (None without a colour)
    ci_method: bs.IntervalMethod | None = None  # what gave the mean's interval (R-BS-01)
    ci_reason: str | None = None
    median_low: float | None = None  # 95% interval of the median, by bootstrap (R-BS-02)
    median_high: float | None = None
    median_reason: str | None = None


@dataclass
class GroupDifference:
    """Mean of b minus mean of a, between two groups of the axis (051)."""

    a: str
    b: str
    n_a: int
    n_b: int
    diff: float
    low: float | None  # 95% interval of the difference
    high: float | None
    interval: Literal["welch", "bootstrap"]  # R-BS-03
    reason: str
    p_value: float | None  # Welch's t on every row
    p_adjusted: float | None = None  # over every difference of the view (R-MT-01/02)


@dataclass
class Tested:
    """A group entering the pairwise differences."""

    label: str
    stats: GroupStats
    replicates: bs.Replicates | None
    method: bs.IntervalMethod
    reason: str


def set_intervals(
    groups: list[Group], samples: list[np.ndarray], budget_groups: int
) -> list[bs.Replicates | None]:
    """Each group's mean interval (bootstrap or Student t) and median interval, with the reason
    (R-BS-01, 02, 05, 06). Returns the bootstrap replicates, for the differences."""
    held = bs.held_size(budget_groups, max((len(values) for values in samples), default=0))
    found: list[bs.Replicates | None] = []
    for stream, (group, values) in enumerate(zip(groups, samples, strict=True)):
        method, reason = bs.mean_rule(group.n, values)
        group.ci_method, group.ci_reason = method, reason
        replicates = None
        if len(values) >= bs.MIN_VALUES and group.mean is not None and group.median is not None:
            replicates = bs.replicates(values[:held], group.n, stream, (group.mean, group.median))
            note = bs.sample_note(replicates.m, group.n)
            if method == "bootstrap":
                group.ci_low, group.ci_high = bs.interval(replicates.means, group.n)
                group.ci_reason = reason + note
            group.median_low, group.median_high = bs.interval(replicates.medians, group.n)
            group.median_reason = "bootstrap: a median has no simple formula" + note
        else:
            group.median_reason = f"{group.n} values: too few for an interval"
        found.append(replicates)
    return found


def intervals_sentence(groups: list[Group]) -> str | None:
    """Which method gave the intervals, in one sentence (R-BS-06)."""
    if not groups:
        return None
    boxes = any(g.color is not None for g in groups)
    noun, single = ("boxes", "box") if boxes else ("groups", "group")
    resampled = sum(1 for g in groups if g.ci_method == "bootstrap")
    why = "fewer than 30 values or skewness beyond ±1"
    if resampled == 0:
        mean = f"Mean intervals: Student t for every {single}."
    elif resampled == len(groups):
        mean = f"Mean intervals: bootstrap for all {len(groups)} {noun} ({why})."
    else:
        others = "the other" if len(groups) - resampled == 1 else "the others"
        mean = (
            f"Mean intervals: bootstrap for {resampled} of {len(groups)} {noun} ({why}), "
            f"Student t for {others}."
        )
    return f"{mean} Median intervals: bootstrap ({bs.DRAWS:,} resamples, fixed seed)."


def pairwise(
    tested: list[Tested], correction: Correction = "auto", alpha: float = 0.05
) -> tuple[list[GroupDifference], MultipleTesting | None]:
    """Every pair of groups: difference of means, its interval (R-BS-03) and Welch's p-value,
    adjusted over the pairs (R-MT-01/02)."""
    found: list[GroupDifference] = []
    for a, b in combinations(tested, 2):
        test = welch_t(a.stats, b.stats)
        triggers = [t for t in (a, b) if t.method == "bootstrap"]
        interval: Literal["welch", "bootstrap"] = "welch"
        if triggers and a.replicates is not None and b.replicates is not None:
            low, high = bs.interval(
                b.replicates.means - a.replicates.means, min(a.stats.n, b.stats.n), alpha
            )
            interval = "bootstrap"
            reason = "bootstrap, " + "; ".join(f"{t.label}: {t.reason}" for t in triggers)
        else:
            low, high = welch_interval(a.stats, b.stats, alpha)
            reason = "Welch interval: both means are close to normal"
        found.append(
            GroupDifference(
                a.label,
                b.label,
                a.stats.n,
                b.stats.n,
                b.stats.mean - a.stats.mean,
                clean(low),
                clean(high),
                interval,
                reason,
                test.p_value,
            )
        )
    if not found:
        return [], None
    adjusted, summary = correct([d.p_value for d in found], correction, alpha, noun="differences")
    for difference, value in zip(found, adjusted, strict=True):
        difference.p_adjusted = value
    return found, summary


@dataclass
class Contingency:
    rows: list[str]  # categories of Y
    columns: list[str]  # categories of X
    counts: list[list[int]]


@dataclass
class Series:
    label: str | None
    periods: list[str]
    values: list[float | None]
    rolling_mean: list[float | None] = field(default_factory=list)
    rolling_std: list[float | None] = field(default_factory=list)
    counts: list[int] = field(default_factory=list)  # rows aggregated in each period


@dataclass
class TimeView:
    """How a date and a number was turned into a series (each setting chosen or defaulted)."""

    period: ts.Period
    aggregate: ts.Aggregate
    window: int | None  # None: too few periods for rolling statistics
    align: ts.Align
    periods: int
    period_auto: bool
    aggregate_auto: bool
    coarsened: bool  # the period was raised to keep the series readable
    # The data starts after the first period begins (or ends before the last one ends): a sum or
    # count there is too low, so that period is left out of the trend test.
    first_partial: bool = False
    last_partial: bool = False


@dataclass
class TrendSummary:
    slope: float  # Sen's slope, in value units per period
    slope_low: float
    slope_high: float
    per_year: float | None  # change over a year, relative to the mean level
    lag1: float | None


@dataclass
class GroupLayout:
    """How a number and one or two categories were laid out (R-CMP-04, R-CMP-05)."""

    axis: str  # the category on the axis of the groups
    color: str | None  # the category in colour
    swapped: bool  # the chosen colour went on the axis (it had more values)
    axis_values: int
    color_values: int
    axis_kept: int  # slots shown, "Others" included
    color_kept: int
    sentence: str


def plan_layout(axis_values: int, color_values: int) -> tuple[bool, int, int]:
    """(swapped, axis slots, colour slots), "Others" included in the slots. The category with
    more values goes on the axis when the colour has more than 5; at most 5 colours and 15 boxes,
    the colour reduced first, then the axis."""
    swapped = color_values > SWAP_ABOVE and color_values > axis_values
    axis, color = (color_values, axis_values) if swapped else (axis_values, color_values)
    color_slots = min(color, MAX_COLORS) if color else 1
    axis_slots = min(axis, MAX_BOXES // color_slots)
    return swapped, axis_slots, color_slots


def keep_slots(labels: list[str], slots: int) -> list[str]:
    """The most frequent labels (given most frequent first): all of them if they fit, otherwise
    one slot fewer, the last slot being "Others"."""
    return labels if len(labels) <= slots else labels[: slots - 1]


@dataclass
class CategoryRead:
    """A whole-number column read as categories (R-CMP-01)."""

    column: str
    values: int


@dataclass
class Comparison:
    kind: Kind
    x: str
    y: str
    z: str | None
    color: str | None
    roles: dict[str, Role]
    rows: int  # rows where every chosen column has a value
    total: int  # rows after the global filters
    sample: int | None = None  # points drawn, when fewer than `rows`
    group_axis: Literal["x", "y"] | None = None
    points: Points | None = None
    regression: Regression | None = None
    lowess: Lowess | None = None  # scatter: LOWESS curve and whether to draw it (064)
    density: Density | None = None
    groups: list[Group] = field(default_factory=list)
    contingency: Contingency | None = None
    series: list[Series] = field(default_factory=list)
    step: str | None = None
    time: TimeView | None = None
    trend: TrendSummary | None = None
    tests: list[BivariateTest] = field(default_factory=list)
    message: str | None = None
    categories: list[CategoryRead] = field(default_factory=list)
    color_levels: list[str] = field(default_factory=list)  # groups: the colours, in legend order
    layout: GroupLayout | None = None  # groups: what is on the axis and in colour, and why
    intervals: str | None = None  # groups: which method gave the intervals (051)
    differences: list[GroupDifference] = field(default_factory=list)  # groups, pairwise (051)
    multiple: MultipleTesting | None = None  # the correction over the differences


def label_sql(column: TypedColumn) -> str:
    """Category label, as the charts and filters show it."""
    typed = typed_sql(column.name, column.parser)
    if column.type == "boolean":
        return f"(CASE WHEN {typed} THEN 'True' WHEN NOT {typed} THEN 'False' END)"
    return f"CAST({typed} AS VARCHAR)"


def number_sql(column: TypedColumn) -> str:
    return f"CAST({typed_sql(column.name, column.parser)} AS DOUBLE)"


def number_label_sql(column: TypedColumn) -> str:
    """A whole number as a level: "1", not "1.0"."""
    value = number_sql(column)
    return (
        f"(CASE WHEN {value} = floor({value}) THEN CAST(CAST({value} AS BIGINT) AS VARCHAR) "
        f"ELSE CAST({value} AS VARCHAR) END)"
    )


def ordered(labels: list[str], column: "CompareColumn") -> list[str]:
    """Levels in their natural order: an ordinal's order, numbers by value; else as given."""
    if column.type == "categorical_ordinal" and column.order:
        rank = {label: i for i, label in enumerate(column.order)}
        return sorted(labels, key=lambda label: rank.get(label, len(rank)))
    if column.as_category:

        def value(label: str) -> float:
            try:
                return float(label)
            except ValueError:
                return float("inf")

        return sorted(labels, key=value)
    return labels


class Comparer:
    def __init__(
        self,
        connection: duckdb.DuckDBPyConnection,
        source: str,
        x: CompareColumn,
        y: CompareColumn,
        z: CompareColumn | None = None,
        color: CompareColumn | None = None,
        options: ts.Options | None = None,
        correction: Correction = "auto",
    ) -> None:
        self.connection = connection
        self.options = options or ts.Options()
        self.correction: Correction = correction
        self.source = source
        self.x, self.y, self.z, self.color = x, y, z, color
        if x.name == y.name:
            raise NotComparableError("Pick two different columns.")
        if color is not None and color.role != "cat":
            raise NotComparableError(f"Color needs a category: {color.name!r} is {color.type}.")
        if z is not None and not (x.role == y.role == z.role == "num"):
            raise NotComparableError("A Z axis needs three numeric columns.")

    def one(self, sql: str) -> tuple:
        return self.connection.execute(sql).fetchone() or ()

    def all(self, sql: str) -> list[tuple]:
        return self.connection.execute(sql).fetchall()

    def value_sql(self, column: CompareColumn) -> str:
        role = column.role
        if role == "num":
            return number_sql(column)
        if role == "date":
            return f"CAST({typed_sql(column.name, column.parser)} AS TIMESTAMP)"
        if column.as_category:
            return number_label_sql(column)
        return label_sql(column)

    def rows_of(self, columns: dict[str, CompareColumn]) -> str:
        """The rows where every given column has a usable value, as named columns."""
        selected = ", ".join(f"{self.value_sql(c)} AS {alias}" for alias, c in columns.items())
        conditions = " AND ".join(
            f"{alias} IS NOT NULL" + (f" AND isfinite({alias})" if c.role == "num" else "")
            for alias, c in columns.items()
        )
        return (
            f"(SELECT * FROM (SELECT {selected}, file_row_number AS rn FROM {self.source}) "
            f"WHERE {conditions})"
        )

    def total(self) -> int:
        return int(self.one(f"SELECT count(*) FROM {self.source}")[0])

    def sample(self, relation: str, columns: str, size: int) -> list[tuple]:
        return self.all(f"SELECT {columns} FROM {relation} ORDER BY hash(rn + {SEED}) LIMIT {size}")

    # --- Dispatch -----------------------------------------------------------------------------

    def compare(self) -> Comparison:
        roles: dict[str, Role] = {self.x.name: self.x.role, self.y.name: self.y.role}
        pair = (self.x.role, self.y.role)
        if pair == ("num", "num"):
            result = self.scatter()
        elif "date" in pair and "num" in pair:
            result = self.timeline()
        elif "date" in pair and "cat" in pair:
            result = self.date_counts()
        elif pair == ("cat", "cat"):
            result = self.contingency()
        elif "num" in pair and "cat" in pair:
            result = self.groups()
        else:
            raise NotComparableError(
                "Two dates can't be compared: pick a numeric or category column."
            )
        result.roles = roles
        if self.z:
            result.roles[self.z.name] = "num"
        if self.color:
            result.roles[self.color.name] = "cat"
        return result

    def base(self, kind: Kind, rows: int) -> Comparison:
        return Comparison(
            kind,
            self.x.name,
            self.y.name,
            self.z.name if self.z else None,
            self.color.name if self.color else None,
            {},
            rows,
            self.total(),
        )

    # --- num × num -------------------------------------------------------------------------------

    def scatter(self) -> Comparison:
        columns = {"x": self.x, "y": self.y}
        if self.z:
            columns["z"] = self.z
        if self.color:
            columns["c"] = self.color
        relation = self.rows_of(columns)
        n, r, slope, intercept, r2, x_low, x_high, y_low, y_high = self.one(
            f"SELECT count(*), corr(x, y), regr_slope(y, x), regr_intercept(y, x), regr_r2(y, x), "
            f"min(x), max(x), min(y), max(y) FROM {relation}"
        )
        result = self.base("scatter", int(n))
        if n < 3:
            result.message = "Fewer than 3 rows have both values."
            return result
        names = ", ".join(columns)
        drawn = self.sample(relation, names, MAX_POINTS)
        result.sample = len(drawn) if len(drawn) < n else None
        values = list(zip(*drawn, strict=True)) if drawn else [[] for _ in columns]
        by_alias = dict(zip(columns, values, strict=True))
        result.points = Points(
            [float(v) for v in by_alias["x"]],
            [float(v) for v in by_alias["y"]],
            [float(v) for v in by_alias["z"]] if "z" in by_alias else None,
            self.top_labels([str(v) for v in by_alias["c"]], MAX_SERIES)
            if "c" in by_alias
            else None,
        )
        if slope is not None and intercept is not None:
            result.regression = Regression(float(slope), float(intercept), clean(r2))
        result.density = self.density(relation, x_low, x_high, y_low, y_high)
        tested = self.sample(relation, "x, y", TEST_SAMPLE)
        xs = np.array([row[0] for row in tested], dtype=float)
        ys = np.array([row[1] for row in tested], dtype=float)
        result.tests = [pearson(clean(r), int(n)), *rank_correlations(xs, ys)]
        result.lowess = lowess_curve(xs, ys)
        if result.tests[0].p_value is not None and r2 is not None:
            result.tests[
                0
            ].note = f"R² = {float(r2):.3f}: share of Y's variance a straight line explains."
        return result

    @staticmethod
    def top_labels(labels: list[str], limit: int) -> list[str]:
        """Keep the most frequent labels, the rest becomes "Others" (fixed palette order)."""
        counts: dict[str, int] = {}
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
        kept = set(sorted(counts, key=lambda label: (-counts[label], label))[:limit])
        return [label if label in kept else OTHERS for label in labels]

    def density(self, relation: str, x_low, x_high, y_low, y_high) -> Density | None:
        if None in (x_low, x_high, y_low, y_high) or x_low == x_high or y_low == y_high:
            return None
        bins = DENSITY_BINS
        x_width = (x_high - x_low) / bins
        y_width = (y_high - y_low) / bins
        rows = self.all(
            f"SELECT LEAST(CAST(floor((x - {x_low}) / {x_width}) AS INTEGER), {bins - 1}) AS i, "
            f"LEAST(CAST(floor((y - {y_low}) / {y_width}) AS INTEGER), {bins - 1}) AS j, count(*) "
            f"FROM {relation} GROUP BY i, j"
        )
        counts = [[0] * bins for _ in range(bins)]
        for i, j, count in rows:
            counts[int(j)][int(i)] = int(count)
        return Density(
            [x_low + k * x_width for k in range(bins + 1)],
            [y_low + k * y_width for k in range(bins + 1)],
            counts,
        )

    # --- num × cat -------------------------------------------------------------------------------

    def groups(self) -> Comparison:
        numeric, category = (self.x, self.y) if self.x.role == "num" else (self.y, self.x)
        relation = self.rows_of({"v": numeric, "g": category})
        (n,) = self.one(f"SELECT count(*) FROM {relation}")
        result = self.base("groups", int(n))
        result.group_axis = "x" if category is self.x else "y"
        if n < 3:
            result.message = "Fewer than 3 rows have both values."
            return result
        if self.color is not None:
            return self.colored_groups(numeric, category, self.color, result)
        counted = self.all(f"SELECT g, count(*) AS c FROM {relation} GROUP BY g ORDER BY c DESC, g")
        _, slots, _ = plan_layout(len(counted), 0)
        kept = ordered(keep_slots([str(label) for label, _ in counted], slots), category)
        listed = ", ".join(lit(label) for label in kept)
        grouped = (
            f"(SELECT v, CASE WHEN g IN ({listed}) THEN g ELSE '{OTHERS}' END AS g, rn "
            f"FROM {relation})"
        )
        summaries = {
            str(label): row
            for label, *row in self.all(
                f"SELECT g, count(*), avg(v), var_samp(v), min(v), max(v), "
                f"quantile_cont(v, [0.25, 0.5, 0.75]) FROM {grouped} GROUP BY g"
            )
        }
        samples = {
            str(label): [float(v) for v in values]
            for label, values in self.all(
                f"SELECT g, list(v ORDER BY k) FROM (SELECT g, v, row_number() OVER "
                f"(PARTITION BY g ORDER BY hash(rn + {SEED})) AS k FROM {grouped}) "
                f"WHERE k <= {max(VIOLIN_SAMPLE, TEST_SAMPLE // max(len(kept), 1))} GROUP BY g"
            )
        }
        order = kept + ([OTHERS] if OTHERS in summaries else [])
        result.groups = [
            self.group(label, summaries[label], samples.get(label, [])) for label in order
        ]
        tested = [g for g in result.groups if g.label != OTHERS]
        group_stats = [GroupStats(g.n, g.mean or 0.0, (g.std or 0.0) ** 2) for g in tested]
        group_samples = [np.array(g.sample, dtype=float) for g in tested]
        every_sample = [np.array(g.sample, dtype=float) for g in result.groups]
        replicates = set_intervals(result.groups, every_sample, len(result.groups))
        result.intervals = intervals_sentence(result.groups)
        if len(tested) >= 2:
            by_label = dict(zip((g.label for g in result.groups), replicates, strict=True))
            result.differences, result.multiple = pairwise(
                [
                    Tested(g.label, s, by_label[g.label], g.ci_method or "t", g.ci_reason or "")
                    for g, s in zip(tested, group_stats, strict=True)
                ],
                self.correction,
            )
        for group in result.groups:
            group.sample = group.sample[:VIOLIN_SAMPLE]
        result.tests = group_tests(group_stats, group_samples) if len(tested) >= 2 else []
        hidden = len(counted) - len(kept)
        result.layout = GroupLayout(category.name, None, False, len(counted), 0, len(order), 1, "")
        if hidden > 0:
            result.message = (
                f'{hidden} smaller groups are shown together as "{OTHERS}" '
                "and left out of the tests."
            )
            result.layout.sentence = result.message
        return result

    def colored_groups(
        self,
        numeric: CompareColumn,
        category: CompareColumn,
        color: CompareColumn,
        result: Comparison,
    ) -> Comparison:
        """One box per colour level inside each group. Whatever the places chosen, the category
        with more values goes on the axis (R-CMP-04), at most 5 colours and 15 boxes (R-CMP-05);
        the tests still compare the groups of the axis, all colours together (R-CMP-06)."""
        relation = self.rows_of({"v": numeric, "a": category, "b": color})
        (n,) = self.one(f"SELECT count(*) FROM {relation}")
        result.rows = int(n)
        if n < 3:
            result.message = "Fewer than 3 rows have all three values."
            return result
        first = self.all(f"SELECT a, count(*) AS n FROM {relation} GROUP BY a ORDER BY n DESC, a")
        second = self.all(f"SELECT b, count(*) AS n FROM {relation} GROUP BY b ORDER BY n DESC, b")
        swapped, axis_slots, color_slots = plan_layout(len(first), len(second))
        axis, hue = (color, category) if swapped else (category, color)
        axis_counts, hue_counts = (second, first) if swapped else (first, second)
        g, c = ("b", "a") if swapped else ("a", "b")
        kept = ordered(keep_slots([str(x) for x, _ in axis_counts], axis_slots), axis)
        levels = ordered(keep_slots([str(x) for x, _ in hue_counts], color_slots), hue)
        if len(levels) < len(hue_counts):
            levels = [*levels, OTHERS]
        axis_list = ", ".join(lit(label) for label in kept)
        hue_list = ", ".join(lit(label) for label in levels if label != OTHERS)
        grouped = (
            f"(SELECT v, CASE WHEN {g} IN ({axis_list}) THEN {g} ELSE '{OTHERS}' END AS g, "
            f"CASE WHEN {c} IN ({hue_list}) THEN {c} ELSE '{OTHERS}' END AS c, rn FROM {relation})"
        )
        summaries = {
            (str(label), str(level)): row
            for label, level, *row in self.all(
                f"SELECT g, c, count(*), avg(v), var_samp(v), min(v), max(v), "
                f"quantile_cont(v, [0.25, 0.5, 0.75]) FROM {grouped} GROUP BY g, c"
            )
        }
        samples = {
            (str(label), str(level)): [float(v) for v in values]
            for label, level, values in self.all(
                f"SELECT g, c, list(v ORDER BY k) FROM (SELECT g, c, v, row_number() OVER "
                f"(PARTITION BY g, c ORDER BY hash(rn + {SEED})) AS k FROM {grouped}) "
                f"WHERE k <= {VIOLIN_SAMPLE} GROUP BY g, c"
            )
        }
        order = kept + ([OTHERS] if any(label == OTHERS for label, _ in summaries) else [])
        for label in order:
            for level in levels:
                if (label, level) in summaries:
                    group = self.group(
                        label, summaries[(label, level)], samples.get((label, level), [])
                    )
                    group.color = level
                    result.groups.append(group)
        result.color_levels = levels
        # the categories now sit on the axis that is not the number's
        result.group_axis = "y" if numeric is self.x else "x"
        pooled = self.all(
            f"SELECT g, count(*), avg(v), var_samp(v), list(v ORDER BY hash(rn + {SEED}))"
            f"[1:{TEST_SAMPLE}] FROM {grouped} WHERE g <> '{OTHERS}' GROUP BY g"
        )
        rank = {label: i for i, label in enumerate(kept)}
        pooled.sort(key=lambda row: rank.get(str(row[0]), len(rank)))
        stats = [
            GroupStats(int(k), float(m or 0.0), float(var or 0.0)) for _, k, m, var, _ in pooled
        ]
        values = [np.array(sample, dtype=float) for *_, sample in pooled]
        result.tests = group_tests(stats, values) if len(pooled) >= 2 else []
        budget = len(result.groups) + len(pooled)
        set_intervals(
            result.groups, [np.array(g.sample, dtype=float) for g in result.groups], budget
        )
        result.intervals = intervals_sentence(result.groups)
        if len(pooled) >= 2:
            result.differences, result.multiple = pairwise(
                self.pooled_tested(pooled, stats, values, budget, len(result.groups)),
                self.correction,
            )
        boxes = len(result.groups)
        notes = [
            f"{axis.name} on the axis ({len(axis_counts)} values), {hue.name} in colour "
            f"({len(hue_counts)} values)"
            + (": the category with more values goes on the axis." if swapped else ".")
        ]
        if len(order) < len(axis_counts) or len(levels) < len(hue_counts) or OTHERS in levels:
            notes.append(
                f'{boxes} boxes at most {MAX_BOXES}: the smaller values are shown as "{OTHERS}".'
            )
        notes.append(f"Tests compare the {axis.name} groups, all {hue.name} values together.")
        result.message = " ".join(notes)
        result.layout = GroupLayout(
            axis.name,
            hue.name,
            swapped,
            len(axis_counts),
            len(hue_counts),
            len(order),
            len(levels),
            result.message,
        )
        return result

    @staticmethod
    def pooled_tested(
        pooled: list[tuple],
        stats: list[GroupStats],
        values: list[np.ndarray],
        budget: int,
        first_stream: int,
    ) -> list[Tested]:
        """The axis groups, all colours together, ready for the pairwise differences."""
        held = bs.held_size(budget, max((len(v) for v in values), default=0))
        tested: list[Tested] = []
        for stream, (row, s, v) in enumerate(zip(pooled, stats, values, strict=True)):
            method, reason = bs.mean_rule(s.n, v)
            replicates = None
            if len(v) >= bs.MIN_VALUES:
                center = (s.mean, float(np.median(v)))
                replicates = bs.replicates(v[:held], s.n, first_stream + stream, center)
                if method == "bootstrap":
                    reason += bs.sample_note(replicates.m, s.n)
            tested.append(Tested(str(row[0]), s, replicates, method, reason))
        return tested

    @staticmethod
    def group(label: str, row: list, sample: list[float]) -> Group:
        count, mean, var, low, high, quartiles = row
        q1, median, q3 = (clean(v) for v in quartiles) if quartiles else (None, None, None)
        std = math.sqrt(var) if var is not None and var >= 0 else None
        whisker_low = whisker_high = None
        if q1 is not None and q3 is not None:
            spread = q3 - q1
            whisker_low = max(float(low), q1 - 1.5 * spread)
            whisker_high = min(float(high), q3 + 1.5 * spread)
        ci_low = ci_high = None
        if std is not None and count > 1:
            margin = float(scipy_stats.t.ppf(0.975, count - 1)) * std / math.sqrt(count)
            ci_low, ci_high = mean - margin, mean + margin
        return Group(
            label,
            int(count),
            clean(mean),
            clean(std),
            q1,
            median,
            q3,
            whisker_low,
            whisker_high,
            clean(ci_low),
            clean(ci_high),
            sample,
        )

    # --- cat × cat -------------------------------------------------------------------------------

    def contingency(self) -> Comparison:
        relation = self.rows_of({"a": self.x, "b": self.y})
        (n,) = self.one(f"SELECT count(*) FROM {relation}")
        result = self.base("contingency", int(n))
        if n < 3:
            result.message = "Fewer than 3 rows have both values."
            return result
        columns = self.kept_labels(relation, "a", self.x)
        rows = self.kept_labels(relation, "b", self.y)
        cells = self.all(
            f"SELECT {self.bucket('a', columns)} AS a, {self.bucket('b', rows)} AS b, count(*) "
            f"FROM {relation} GROUP BY 1, 2"
        )
        column_labels = columns + ([OTHERS] if any(c == OTHERS for c, _, _ in cells) else [])
        row_labels = rows + ([OTHERS] if any(r == OTHERS for _, r, _ in cells) else [])
        table = np.zeros((len(row_labels), len(column_labels)), dtype=int)
        for a, b, count in cells:
            table[row_labels.index(b)][column_labels.index(a)] = int(count)
        result.contingency = Contingency(row_labels, column_labels, table.tolist())
        result.tests = contingency_tests(table)
        return result

    def kept_labels(self, relation: str, alias: str, column: CompareColumn) -> list[str]:
        counted = self.all(
            f"SELECT {alias}, count(*) AS c FROM {relation} GROUP BY 1 ORDER BY c DESC, 1 "
            f"LIMIT {MAX_TABLE}"
        )
        return ordered([str(label) for label, _ in counted], column)

    @staticmethod
    def bucket(alias: str, kept: list[str]) -> str:
        listed = ", ".join(lit(label) for label in kept)
        return f"CASE WHEN {alias} IN ({listed}) THEN {alias} ELSE '{OTHERS}' END"

    # --- Dates -----------------------------------------------------------------------------------

    def date_and_other(self) -> tuple[CompareColumn, CompareColumn]:
        return (self.x, self.y) if self.x.role == "date" else (self.y, self.x)

    def step_for(self, relation: str) -> tuple[str, str]:
        span, timed = self.one(
            f"SELECT epoch(max(t)) - epoch(min(t)), "
            f"count(*) FILTER (WHERE t <> CAST(CAST(t AS DATE) AS TIMESTAMP)) FROM {relation}"
        )
        step = auto_step(float(span or 0), bool(timed))
        return step, STEP_FORMATS[step]

    def timeline(self) -> Comparison:
        """One aggregated value per period, rolling mean ± standard deviation, trend test."""
        date, numeric = self.date_and_other()
        columns = {"t": date, "v": numeric}
        if self.color:
            columns["c"] = self.color
        relation = self.rows_of(columns)
        (n,) = self.one(f"SELECT count(*) FROM {relation}")
        result = self.base("timeline", int(n))
        if n < 3:
            result.message = "Fewer than 3 rows have both values."
            return result

        options = self.options
        groups = self.kept_labels(relation, "c", self.color)[:MAX_SERIES] if self.color else None
        built = builder.aggregate(self.connection, relation, numeric.semantic_role, options, groups)
        chosen = built.groups if self.color else [built.overall]
        series = [
            Series(item.label, built.periods, item.values, counts=item.counts) for item in chosen
        ]
        periods = len(built.periods)
        window = None
        if periods >= ts.MIN_PERIODS:
            window = max(
                2, min(options.window or ts.default_window(built.period, periods), periods)
            )
            for item in series:
                item.rolling_mean, item.rolling_std = ts.rolling(item.values, window, options.align)
        result.step = built.period
        result.series = series
        result.time = TimeView(
            built.period,
            built.aggregate,
            window,
            options.align,
            periods,
            built.period_auto,
            built.aggregate_auto,
            built.coarsened,
            built.first_partial,
            built.last_partial,
        )
        if periods < ts.MIN_PERIODS:
            result.message = f"Too few periods for a trend test ({ts.MIN_PERIODS} needed)."
            return result

        found = trend_of(builder.trend_input(built, built.overall.values))
        if found is None:
            result.message = "The series doesn't vary: no trend to test."
            return result
        level = float(np.mean([v for v in built.overall.values if v is not None]))
        per_year = found.slope * ts.PER_YEAR[built.period] / abs(level) if level else None
        result.trend = TrendSummary(
            found.slope, found.slope_low, found.slope_high, clean(per_year), clean(found.lag1)
        )
        notes = ["Mann-Kendall test on one value per period"]
        notes[0] += " (all groups combined)." if self.color else "."
        if found.sampled:
            notes.append(f"Tested on {found.n:,} evenly spaced periods.")
        if found.lag1 is not None and found.lag1 > 0.5:
            notes.append(
                "Successive values are strongly linked; the p-value may be optimistic. "
                "The time series page tests this properly."
            )
        result.tests = [
            BivariateTest(
                "mann_kendall",
                clean(found.z),
                clean(found.p_value),
                found.n,
                correlation_effect("tau", clean(found.tau)),
                recommended=True,
                note=" ".join(notes),
            )
        ]
        return result

    def date_counts(self) -> Comparison:
        date, category = self.date_and_other()
        relation = self.rows_of({"t": date, "c": category})
        (n,) = self.one(f"SELECT count(*) FROM {relation}")
        result = self.base("date_counts", int(n))
        if n < 3:
            result.message = "Fewer than 3 rows have both values."
            return result
        step, fmt = self.step_for(relation)
        result.step = step
        kept = [
            str(v)
            for (v,) in self.all(
                f"SELECT c FROM {relation} GROUP BY c ORDER BY count(*) DESC, c LIMIT {MAX_SERIES}"
            )
        ]
        rows = self.all(
            f"SELECT {self.bucket('c', kept)} AS g, "
            f"strftime(date_trunc('{step}', t), '{fmt}') AS p, "
            f"count(*) FROM {relation} GROUP BY g, date_trunc('{step}', t), p "
            f"ORDER BY date_trunc('{step}', t)"
        )
        periods = list(dict.fromkeys(str(p) for _, p, _ in rows))
        counts: dict[str, dict[str, int]] = {}
        for g, period, count in rows:
            counts.setdefault(str(g), {})[str(period)] = int(count)
        order = kept + ([OTHERS] if OTHERS in counts else [])
        result.series = [
            Series(label, periods, [float(counts[label].get(p, 0)) for p in periods])
            for label in order
            if label in counts
        ]
        return result


def compare(
    connection: duckdb.DuckDBPyConnection,
    source: str,
    x: CompareColumn,
    y: CompareColumn,
    z: CompareColumn | None = None,
    color: CompareColumn | None = None,
    options: ts.Options | None = None,
    correction: Correction = "auto",
) -> Comparison:
    return Comparer(connection, source, x, y, z, color, options, correction).compare()
