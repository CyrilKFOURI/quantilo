"""The automatic choices of a KPI, one pure function per decision (spec 022, section 9 bis).

Every decision carries the rule that made it, a one-sentence reason, and for the choices the user
may change, every alternative with whether it is allowed and why not. A *constraint* is a choice
that would draw a wrong chart or a wrong number: it is refused, never forced.
Rules read facts computed on the result; they never touch the data.
"""

import math
from dataclasses import dataclass, field

from quantilo.kpi.definition import GRAINS, Agg, Chart, Grain, Layout, Measure, Ratio

GRAIN_SECONDS: dict[Grain, float] = {
    "day": 86_400,
    "week": 7 * 86_400,
    "month": 30.44 * 86_400,
    "quarter": 91.31 * 86_400,
    "year": 365.25 * 86_400,
}
GRAIN_NAMES: dict[Grain, str] = {
    "day": "day",
    "week": "week",
    "month": "month",
    "quarter": "quarter",
    "year": "year",
}
AUTO_MAX_PERIODS = 60  # R-TIME-01
FORCED_MAX_PERIODS = 2_000  # R-TIME-03
FEW_PERIODS = 3  # R-VIS-11
BARS_MAX_PERIODS = 24  # bars up to two years of months; areas beyond
DOMINANT_SHARE = 0.7  # R-VIS-12
SPARSE_SHARE = 0.4  # R-VIS-13
HEATMAP_LEVELS = 8  # R-VIS-14
GROUPED_MAX = 6  # R-VIS-21: side-by-side bars stay readable up to 6 groups
COLUMNS_MAX_PERIODS = 13  # R-TAB-02: a year of months (and one) across the table


@dataclass
class Option:
    value: str
    allowed: bool = True
    reason: str | None = None


@dataclass
class Decision:
    value: str
    rule: str
    reason: str
    overridable: bool = True
    options: list[Option] = field(default_factory=list)


def plural(word: str, count: float) -> str:
    return word if round(count) == 1 else f"{word}s"


def describe_span(seconds: float) -> str:
    """'3 years', '7 months', '5 weeks', '12 days'."""
    for grain, minimum in (("year", 2), ("month", 2), ("week", 2)):
        count = seconds / GRAIN_SECONDS[grain]  # type: ignore[index]
        if count >= minimum:
            return f"{count:.0f} {plural(grain, count)}"
    days = max(1.0, seconds / GRAIN_SECONDS["day"])
    return f"{days:.0f} {plural('day', days)}"


# --- Time ------------------------------------------------------------------------------------


def typical_grain(median_gap: float) -> Grain:
    """The grain matching the usual gap between two consecutive dates (no hours in a KPI)."""
    if median_gap <= 1.5 * 86_400:
        return "day"
    if median_gap <= 10 * 86_400:
        return "week"
    if median_gap <= 45 * 86_400:
        return "month"
    if median_gap <= 120 * 86_400:
        return "quarter"
    return "year"


def count_periods(span: float, grain: Grain) -> int:
    return int(span // GRAIN_SECONDS[grain]) + 1


def grain_options(typical: Grain, span: float) -> list[Option]:
    options = []
    for grain in GRAINS:
        if GRAINS.index(grain) < GRAINS.index(typical):
            options.append(
                Option(
                    grain,
                    False,
                    f"The dates come about once a {typical}: "
                    f"a {grain} grain would mostly show empty periods.",
                )
            )
        elif count_periods(span, grain) > FORCED_MAX_PERIODS:
            options.append(
                Option(grain, False, f"Over {FORCED_MAX_PERIODS:,} periods: too many to draw.")
            )
        else:
            options.append(Option(grain))
    return options


def resolve_grain(
    requested: Grain | str, typical: Grain, span: float
) -> tuple[Decision, str | None]:
    """The grain to use and a warning when the requested one could not be kept."""
    options = grain_options(typical, span)
    allowed = [o.value for o in options if o.allowed] or ["year"]
    if requested != "auto":
        if requested in allowed:
            return Decision(requested, "user", "Chosen by you.", options=options), None
        if GRAINS.index(requested) < GRAINS.index(typical):  # type: ignore[arg-type]
            warning = (
                f"The dates come about once a {typical}: shown by {typical} instead of "
                f"by {requested} (R-TIME-02)."
            )
            return Decision(typical, "R-TIME-02", warning, False, options), warning
        coarser = allowed[0]
        for grain in GRAINS[GRAINS.index(requested) :]:  # type: ignore[arg-type]
            if grain in allowed:
                coarser = grain
                break
        warning = (
            f"By {requested} would draw over {FORCED_MAX_PERIODS:,} periods: shown by "
            f"{coarser} instead (R-TIME-03)."
        )
        return Decision(coarser, "R-TIME-03", warning, False, options), warning
    for grain in GRAINS[GRAINS.index(typical) :]:
        periods = count_periods(span, grain)
        if periods <= AUTO_MAX_PERIODS:
            reason = (
                f"{grain.capitalize()}s because the data covers {describe_span(span)} "
                f"({periods} {plural(grain, periods)})."
            )
            if grain == typical and periods > 1:
                reason = f"{grain.capitalize()}s: the dates come about once a {grain}."
            return Decision(grain, "R-TIME-01", reason, options=options), None
    return (
        Decision(
            "year",
            "R-TIME-01",
            f"Years because the data covers {describe_span(span)}.",
            options=options,
        ),
        None,
    )


# --- Chart -----------------------------------------------------------------------------------


@dataclass
class Facts:
    """What the chart rules need to know about one measure's result."""

    time: bool  # a time axis with at least two periods
    periods: int
    groups: int  # 0, 1 or 2 splits
    levels: list[int]  # labels per split, "Other" included
    additive: bool
    negative: bool  # a cell below 0
    dominant: float | None  # share of the largest group in the total (additive, 1st split)
    empty: float  # share of cells without any row
    measure: str  # the measure's name, for the reasons
    split: str | None = None  # the first split's column


TIME_CHARTS: tuple[Chart, ...] = ("bars", "line", "area")
GROUPED_TIME_CHARTS: tuple[Chart, ...] = (
    "grouped_bars",
    "stacked_bars",
    "line",
    "stacked_area",
    "heatmap",
)
STACKED: frozenset[Chart] = frozenset({"stacked_bars", "stacked_area", "hbar_stacked"})


def chart_options(f: Facts) -> list[Option]:
    """Every chart type with whether it fits this result (R-VIS-20, R-VIS-21)."""
    if not f.time:
        candidates: tuple[Chart, ...] = (
            ("none",)
            if f.groups == 0
            else ("hbar",)
            if f.groups == 1
            else ("hbar", "hbar_stacked", "heatmap")
        )
    else:
        candidates = TIME_CHARTS if f.groups == 0 else GROUPED_TIME_CHARTS
    options = []
    for chart in candidates:
        if chart in STACKED and not f.additive:
            options.append(
                Option(
                    chart,
                    False,
                    f"Stacking needs an additive measure: {f.measure} doesn't add up "
                    "across groups (R-VIS-20).",
                )
            )
        elif chart in STACKED and f.negative:
            options.append(
                Option(
                    chart,
                    False,
                    f"{f.measure} has negative values: stacked parts would hide each other "
                    "(R-VIS-21).",
                )
            )
        else:
            options.append(Option(chart))
    return options


def _grouped_time(f: Facts) -> tuple[Chart, str, str]:
    """Chart of a split over time (also each panel of the small multiples)."""
    by = f"by {f.split}" if f.split else "by group"
    if f.periods <= FEW_PERIODS:
        return (
            "grouped_bars",
            "R-VIS-11",
            f"Side-by-side bars: with {f.periods} periods, groups compare better next to "
            "each other.",
        )
    if not f.additive:
        return (
            "line",
            "R-VIS-05",
            f"One line per group: {f.measure} doesn't add up across groups, so it can't be "
            "stacked.",
        )
    if f.negative:
        if f.levels[0] <= GROUPED_MAX and f.periods <= BARS_MAX_PERIODS:
            return (
                "grouped_bars",
                "R-VIS-21",
                f"Side-by-side bars: {f.measure} has negative values, which can't be stacked.",
            )
        return (
            "line",
            "R-VIS-21",
            f"One line per group: {f.measure} has negative values, which can't be stacked.",
        )
    if f.dominant is not None and f.dominant > DOMINANT_SHARE:
        return (
            "line",
            "R-VIS-12",
            f"One line per group: the largest group makes {f.dominant:.0%} of the total and "
            "would crush the others in a stack. Try “% of total” to compare shares.",
        )
    if f.empty > SPARSE_SHARE:
        return (
            "stacked_bars",
            "R-VIS-13",
            f"Bars rather than areas: {f.empty:.0%} of the cells are empty and a line would "
            "join the gaps.",
        )
    if f.periods <= BARS_MAX_PERIODS:
        return "stacked_bars", "R-VIS-04", f"Stacked bars because {f.measure} adds up {by}."
    return (
        "stacked_area",
        "R-VIS-04",
        f"Stacked areas because {f.measure} adds up {by} over {f.periods} periods.",
    )


def resolve_chart(f: Facts) -> Decision:
    options = chart_options(f)
    if not f.time:
        if f.groups == 0:
            return Decision(
                "none", "R-VIS-07", "A single number: the tiles say it all.", False, options
            )
        if f.groups == 1:
            return Decision(
                "hbar",
                "R-VIS-08",
                "Horizontal bars, largest first, to rank the groups.",
                options=options,
            )
        if all(n > HEATMAP_LEVELS for n in f.levels[:2]):
            return Decision(
                "heatmap",
                "R-VIS-14",
                f"A heatmap: {f.levels[0]} by {f.levels[1]} groups are too many for bars.",
                options=options,
            )
        if f.additive and not f.negative:
            return Decision(
                "hbar_stacked",
                "R-VIS-09",
                f"Stacked horizontal bars because {f.measure} adds up across the second split.",
                options=options,
            )
        return Decision(
            "heatmap",
            "R-VIS-10",
            f"A heatmap: {f.measure} can't be stacked, so each cell gets its own shade.",
            options=options,
        )
    if f.groups == 0:
        if not f.additive:
            return Decision(
                "line",
                "R-VIS-03",
                f"A line because {f.measure} is a level, not a volume that adds up.",
                options=options,
            )
        if f.empty > SPARSE_SHARE:
            return Decision(
                "bars",
                "R-VIS-13",
                f"Bars: {f.empty:.0%} of the periods are empty and a line would join the gaps.",
                options=options,
            )
        if f.periods <= BARS_MAX_PERIODS:
            return Decision(
                "bars", "R-VIS-01", f"Bars: {f.measure} is a volume per period.", options=options
            )
        return Decision(
            "area",
            "R-VIS-02",
            f"An area: {f.periods} periods are too many for readable bars.",
            options=options,
        )
    chart, rule, reason = _grouped_time(f)
    if f.groups == 2:
        reason = f"Small multiples, one panel per first split. {reason}"
    return Decision(chart, rule if f.groups == 1 else f"R-VIS-06 + {rule}", reason, options=options)


def share_allowed(f: Facts) -> Option:
    """The '% of total' chart under the main one (R-VIS-20, R-VIS-21)."""
    if not f.time or f.groups == 0:
        return Option("share", False, "A share needs a split over time.")
    if not f.additive:
        return Option("share", False, f"{f.measure} doesn't add up: it has no share of a total.")
    if f.negative:
        return Option("share", False, f"{f.measure} has negative values: shares would mislead.")
    return Option("share")


# --- Table -----------------------------------------------------------------------------------


def resolve_layout(time: bool, groups: int, periods: int) -> Decision:
    if not time:
        if groups == 0:
            return Decision("single", "R-TAB-00", "One value per measure.", False)
        if groups == 1:
            return Decision(
                "groups_rows", "R-TAB-05", "One row per group, with its share of the total.", False
            )
        return Decision(
            "crosstab", "R-TAB-06", "First split in rows, second split in columns.", False
        )
    options = [Option("periods_rows"), Option("periods_columns")]
    if groups == 0:
        return Decision(
            "periods_rows",
            "R-TAB-01",
            "One row per period, one column per measure.",
            options=options,
        )
    if periods <= COLUMNS_MAX_PERIODS:
        rule = "R-TAB-02" if groups == 1 else "R-TAB-04"
        return Decision(
            "periods_columns",
            rule,
            f"Periods across, like a report: {periods} periods fit in the width.",
            options=options,
        )
    if groups == 2:
        return Decision(
            "periods_columns",
            "R-TAB-04",
            f"Periods across, the last {COLUMNS_MAX_PERIODS} of {periods}: switch to periods in "
            "rows to see them all.",
            options=options,
        )
    return Decision(
        "periods_rows",
        "R-TAB-03",
        f"One row per period: {periods} periods are too many to read across.",
        options=options,
    )


def resolve_shading(layout: Layout, groups: int) -> Decision:
    crossed = layout == "crosstab" or (layout.startswith("periods") and groups >= 1)
    if crossed:
        return Decision("on", "R-TAB-08", "Shaded from light to dark: the table crosses two axes.")
    return Decision("off", "R-TAB-08", "No shading: a simple list reads best plain.")


# --- Format ----------------------------------------------------------------------------------


@dataclass
class NumberFormat:
    decimals: int
    scale: float = 1.0  # values are multiplied by this before display (0.25 → 25 %)
    prefix: str = ""
    suffix: str = ""
    percent: bool = False
    rule: str = "R-FMT-01"
    reason: str = ""


def magnitude_decimals(values: list[float]) -> int:
    """2 decimals under 10, 1 under 1,000, none beyond (R-FMT-02)."""
    largest = max((abs(v) for v in values), default=0.0)
    if largest < 10:
        return 2
    if largest < 1_000:
        return 1
    return 0


def resolve_format(
    agg: Agg | None, values: list[float], role: str | None, percent_ratio: bool = False
) -> NumberFormat:
    """`agg` None is a ratio. Values are the result's cells, used for the magnitude."""
    finite = [v for v in values if v is not None and math.isfinite(v)]
    if agg == "true_share" or (agg is None and percent_ratio):
        return NumberFormat(
            1,
            100.0,
            suffix=" %",
            percent=True,
            rule="R-FMT-03",
            reason="A share, shown in percent.",
        )
    if role == "percentage" and agg in ("sum", "mean", "median", "min", "max"):
        within = bool(finite) and all(0 <= v <= 1 for v in finite)
        return NumberFormat(
            1,
            100.0 if within else 1.0,
            suffix=" %",
            percent=True,
            rule="R-FMT-04",
            reason="A percentage column" + (" stored between 0 and 1." if within else "."),
        )
    if agg in ("count", "distinct"):
        return NumberFormat(0, rule="R-FMT-01", reason="Whole numbers.")
    if agg == "sum" and all(float(v).is_integer() for v in finite):
        return NumberFormat(0, rule="R-FMT-01", reason="Whole numbers.")
    decimals = magnitude_decimals(finite)
    return NumberFormat(
        decimals, rule="R-FMT-02", reason=f"{decimals} decimals for values of this size."
    )


# --- Names -----------------------------------------------------------------------------------

AGG_WORDS: dict[Agg, str] = {
    "count": "Rows",
    "sum": "Total {}",
    "mean": "Average {}",
    "median": "Median {}",
    "min": "Min {}",
    "max": "Max {}",
    "distinct": "Distinct {}",
    "true_share": "% {}",
}


def measure_label(measure: Measure) -> str:
    if measure.label:
        return measure.label
    return AGG_WORDS[measure.agg].format(measure.column or "")


def ratio_label(ratio: Ratio, measures: list[Measure]) -> str:
    if ratio.label:
        return ratio.label
    by_id = {m.id: m for m in measures}

    def side(measure_id: str, fallback: str) -> str:
        measure = by_id[measure_id]
        return measure.column or fallback

    return f"{side(ratio.numerator, 'rows')} per {side(ratio.denominator, 'row')}"


def kpi_name(labels: list[str], grain: Grain | None, splits: list[str]) -> str:
    name = " & ".join(labels)
    if grain is not None:
        name += f" by {GRAIN_NAMES[grain]}"
    for split in splits:
        name += f" by {split}"
    return name
