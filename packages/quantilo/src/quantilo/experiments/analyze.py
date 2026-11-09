"""Analyze an experiment already run: variants, units, metrics, then one comparison per metric and
variant against the control.

The input is a frame of typed values (service.py reads it from the dataset): the variant label,
optionally the unit, a date and a subgroup, and per metric its value (`m0`, `m1`…), its
denominator (`d0`…) and its pre-period value (`p0`…). Rows are aggregated per unit first when a
unit is given, so that each unit counts once.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy import stats

from quantilo.errors import ExplorerError
from quantilo.experiments import stats as st

MetricKind = Literal["rate", "mean", "count", "ratio"]
Verdict = Literal["better", "worse", "unclear", "underpowered", "not_tested"]

MIN_LEVELS, MAX_LEVELS = 2, 5
SRM_ALARM = 0.001
SKEWED = 2.0  # |skewness| beyond which Mann–Whitney and a bootstrap check the mean
POISSON_DISPERSION = (0.8, 1.25)  # variance ÷ mean read as "about 1"
BOOTSTRAP_MAX_UNITS = 100_000  # per group: beyond, the normal intervals are reliable
UNDERPOWERED = 0.5
SEGMENT_LEVELS = 10
MIN_SEGMENT_UNITS = 20  # per variant
CONTROL_NAMES = (
    "control",
    "ctrl",
    "a",
    "0",
    "baseline",
    "original",
    "old",
    "holdout",
    "champion",
    "default",
    "false",
)
MAX_DAYS = 120  # cumulative points per day up to this; weeks, then months beyond
MAX_WEEKS = 120

VARIANT, UNIT, TIME, SEGMENT = "__variant", "__unit", "__time", "__segment"
# A pandas DataFrame or Series: its stubs type each indexing as a wide union, so the frames of
# this module are left untyped.
Frame = Any


class ExperimentError(ExplorerError):
    def __init__(self, message: str, field: str | None = None) -> None:
        super().__init__(message, code="experiment", status_code=422)
        self.details = {"field": field}


# --- What the user describes --------------------------------------------------------------------


class ExperimentMetric(BaseModel):
    column: str
    kind: MetricKind | Literal["auto"] = "auto"
    denominator: str | None = Field(
        default=None, description="Ratio of sums: column ÷ denominator."
    )
    pre: str | None = Field(
        default=None, description="The same measure before the experiment (CUPED)."
    )


class ExperimentDesign(BaseModel):
    variant: str = Field(description="The column that names each unit's variant (2 to 5).")
    metrics: list[ExperimentMetric] = Field(min_length=1, max_length=10)
    primary: str | None = Field(
        default=None, description="The main metric's column; the first if unset."
    )
    control: str | None = Field(default=None, description="The control variant; guessed if unset.")
    unit: str | None = Field(
        default=None, description="One row per event: the column naming the unit."
    )
    expected: dict[str, float] | None = Field(
        default=None, description="Planned share of each variant (equal if unset)."
    )
    segment: str | None = Field(default=None, description="Effect per level of this column.")
    time: str | None = Field(default=None, description="Date column for the cumulative difference.")
    alpha: float = Field(default=0.05, gt=0, lt=0.5)


# --- What the analysis returns -----------------------------------------------------------------


@dataclass
class ExperimentVariant:
    label: str
    control: bool
    rows: int
    units: int
    share: float  # observed share of the units
    expected: float  # planned share


@dataclass
class ExperimentSrm:
    statistic: float | None
    p_value: float | None
    alarm: bool
    message: str


@dataclass
class ExperimentCheck:
    key: str  # mann_whitney, bootstrap, welch_t
    p_value: float | None
    low: float | None
    high: float | None
    note: str


@dataclass
class ExperimentComparison:
    metric: str
    variant: str
    control: str
    test: str  # z_proportions, welch_t, poisson, delta_ratio
    n_control: int
    n_variant: int
    control_value: float | None
    variant_value: float | None
    diff: float | None
    diff_low: float | None
    diff_high: float | None
    lift: float | None
    lift_low: float | None
    lift_high: float | None
    statistic: float | None
    p_value: float | None
    p_adjusted: float | None
    significant: bool
    power: float | None  # reached for the observed difference
    mde: float | None  # smallest difference detectable 80 % of the time
    mde_relative: float | None
    verdict: Verdict
    sentence: str
    checks: list[ExperimentCheck] = field(default_factory=list)


@dataclass
class ExperimentCuped:
    pre: str
    theta: float
    correlation: float
    reduction: float  # share of the variance removed
    filled: int  # units without a pre-period value, given its mean
    sentence: str


@dataclass
class ExperimentMetricResult:
    column: str
    label: str
    kind: MetricKind
    reason: str
    primary: bool
    aggregation: str | None  # how the rows of a unit were combined
    values: list[float | None]  # per variant, in the order of `variants`
    comparisons: list[ExperimentComparison]
    cuped: ExperimentCuped | None = None
    skewness: float | None = None


@dataclass
class ExperimentSegment:
    level: str
    units: int
    comparisons: list[ExperimentComparison]


@dataclass
class ExperimentSegments:
    column: str
    metric: str
    levels: list[ExperimentSegment]
    skipped: int  # levels left out (too small or beyond the first ten)
    warning: str


@dataclass
class ExperimentCumulativeLine:
    variant: str
    periods: list[str]
    diff: list[float | None]
    low: list[float | None]
    high: list[float | None]
    p_value: list[float | None]


@dataclass
class ExperimentCumulative:
    column: str
    metric: str
    kind: MetricKind
    grain: Literal["day", "week", "month"]
    lines: list[ExperimentCumulativeLine]
    warning: str


@dataclass
class ExperimentResult:
    design: ExperimentDesign  # as resolved: control, primary and kinds filled in
    rows: int
    units: int
    variants: list[ExperimentVariant]
    srm: ExperimentSrm
    metrics: list[ExperimentMetricResult]
    comparisons: int  # how many p-values the Holm correction covers
    correction: str
    warnings: list[str]
    suggested_unit: str | None
    segments: ExperimentSegments | None = None
    cumulative: ExperimentCumulative | None = None


# --- Columns as the service read them ------------------------------------------------------------


@dataclass
class MetricInput:
    spec: ExperimentMetric
    value: str  # frame column of the values
    denominator: str | None  # frame column of the denominator
    pre: str | None  # frame column of the pre-period values
    boolean: bool  # yes/no column
    whole: bool  # whole-number column


# --- Words --------------------------------------------------------------------------------------


def format_p(p: float | None) -> str:
    if p is None:
        return "p not available"
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}" if p < 0.01 else f"p = {p:.2f}"


def _number(value: float) -> str:
    """Three significant digits, at least: 1,234 · 123 · 12.3 · 1.23 · 0.012."""
    magnitude = abs(value)
    if magnitude >= 100:
        return f"{value:,.0f}"
    if magnitude >= 10:
        return f"{value:.1f}"
    if magnitude >= 1 or magnitude == 0:
        return f"{value:.2f}"
    return f"{value:.2g}"


def _signed(value: float, text: str) -> str:
    return f"+{text}" if value >= 0 and not text.startswith("-") else text.replace("-", "−")


def _value(kind: MetricKind, value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.1f} %" if kind == "rate" else _number(value)


def _gap(kind: MetricKind, value: float | None) -> str:
    """A difference: points for a rate, the metric's unit otherwise; always signed."""
    if value is None:
        return "—"
    if kind == "rate":
        return _signed(value, f"{value * 100:.1f}")
    return _signed(value, _number(value))


def _percent(value: float) -> str:
    return _signed(value, f"{value * 100:.1f} %")


def _lift(value: float) -> str:
    """±4.2 % · ±36 % · ±120 %: two significant digits, never an exponent."""
    shown = abs(value) * 100
    return f"±{shown:.0f} %" if shown >= 10 else f"±{shown:.2g} %"


def _sentence(c: ExperimentComparison, kind: MetricKind, alpha: float, holm: int) -> str:
    if c.verdict == "not_tested":
        return c.sentence
    verb = {"rate": "reaches", "ratio": "has"}.get(kind, "averages")
    points = " points" if kind == "rate" else ""
    lift = f" ({_percent(c.lift)})" if c.lift is not None else ""
    interval = (
        f", {(1 - alpha) * 100:.0f} % CI [{_gap(kind, c.diff_low)} ; {_gap(kind, c.diff_high)}]"
        if c.diff_low is not None and c.diff_high is not None
        else ""
    )
    head = (
        f"{c.variant} {verb} {_value(kind, c.variant_value)} vs {_value(kind, c.control_value)} "
        f"for {c.control}: {_gap(kind, c.diff)}{points}{lift}{interval}."
    )
    p = format_p(c.p_adjusted if holm > 1 else c.p_value)
    adjusted = f", adjusted for {holm} comparisons" if holm > 1 else ""
    if c.verdict in ("better", "worse"):
        return f"{head} Unlikely to be chance ({p}{adjusted})."
    if c.mde_relative is not None:
        detectable = _lift(c.mde_relative)
    elif c.mde is not None:
        detectable = f"±{_value(kind, c.mde)}" + points
    else:
        detectable = None
    if c.verdict == "underpowered":
        reach = (
            f": with this sample you could only detect a lift of {detectable}" if detectable else ""
        )
        return f"{head} Not enough data to conclude{reach} ({p}{adjusted})."
    reach = (
        f" With this sample, a lift of {detectable} would have been detected 80 % of the time."
        if detectable
        else ""
    )
    return f"{head} Could be chance ({p}{adjusted}).{reach}"


# --- Reading the variants and units -------------------------------------------------------------


def guess_control(levels: list[str], counts: list[int]) -> str:
    """ "control", "A", "0", "baseline"…, else the most frequent variant."""
    lowered = {level.strip().lower(): level for level in levels}
    for name in CONTROL_NAMES:
        if name in lowered:
            return lowered[name]
    return levels[int(np.argmax(counts))]


def _natural(levels: list[str]) -> list[str]:
    def key(label: str) -> tuple[int, float, str]:
        try:
            return (0, float(label), label)
        except ValueError:
            return (1, 0.0, label.lower())

    return sorted(levels, key=key)


def _kind(metric: MetricInput, values: Frame) -> tuple[MetricKind, str]:
    spec = metric.spec
    if spec.denominator:
        if spec.kind not in ("auto", "ratio"):
            raise ExperimentError(
                f"{spec.column} has a denominator: it is a ratio, not a {spec.kind}.", "metrics"
            )
        return "ratio", (
            f"{spec.column} ÷ {spec.denominator}: a ratio of sums, each unit counted once "
            "(delta method)."
        )
    present = values.dropna()
    binary = metric.boolean or bool(len(present) and present.isin([0.0, 1.0]).all())
    whole = bool(len(present) and (present >= 0).all() and (present == present.round()).all())
    if spec.kind == "rate" and not binary:
        raise ExperimentError(
            f"{spec.column} is not 0/1 or yes/no: it cannot be a rate.", "metrics"
        )
    if spec.kind == "count" and not whole:
        raise ExperimentError(
            f"{spec.column} has negative or decimal values: it cannot be a count.", "metrics"
        )
    if spec.kind == "ratio":
        raise ExperimentError(f"A ratio needs a denominator for {spec.column}.", "metrics")
    if spec.kind == "mean":
        return "mean", "Chosen as a mean: Welch's t, which does not assume equal variances."
    if spec.kind == "rate" or (spec.kind == "auto" and binary):
        return "rate", "Only 0 and 1 (or yes and no): a rate, z-test of two proportions."
    if spec.kind == "count" or (spec.kind == "auto" and metric.whole and whole):
        return "count", "Whole numbers from 0: a count per unit."
    return "mean", "Numbers: a mean, Welch's t (it does not assume equal variances)."


def _per_unit(frame: Frame, metrics: list[MetricInput], kinds: list[MetricKind]):
    """One row per unit: a rate is 1 if any row is, counts and amounts are summed, both sides of
    a ratio are summed, pre-period values averaged; the first date and subgroup are kept."""
    rules: dict[str, str] = {}
    for metric, kind in zip(metrics, kinds, strict=True):
        rules[metric.value] = "max" if kind == "rate" else "sum"
        if metric.denominator:
            rules[metric.denominator] = "sum"
        if metric.pre:
            rules[metric.pre] = "mean"
    if TIME in frame:
        rules[TIME] = "min"
    if SEGMENT in frame:
        rules[SEGMENT] = "first"
    grouped = frame.groupby([UNIT, VARIANT], sort=False, observed=True)
    aggregated = grouped.agg(rules) if rules else grouped.size().to_frame("rows")
    # an all-missing metric of a unit stays missing (a sum would make it 0)
    for column, rule in rules.items():
        if rule == "sum":
            aggregated.loc[grouped[column].count() == 0, column] = np.nan
    return aggregated.reset_index()


def _aggregation(kind: MetricKind, unit: str, has_pre: bool) -> str:
    text = {
        "rate": f"1 if any row of the {unit} is 1",
        "mean": f"summed per {unit}",
        "count": f"summed per {unit}",
        "ratio": f"both sums per {unit}",
    }[kind]
    return text + ("; pre-period averaged" if has_pre else "")


# --- One comparison ------------------------------------------------------------------------------


def _not_tested(metric: str, variant: str, control: str, nc: int, nv: int, why: str):
    return ExperimentComparison(
        metric,
        variant,
        control,
        "none",
        nc,
        nv,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        False,
        None,
        None,
        None,
        "not_tested",
        why,
    )


def _compare(
    name: str,
    kind: MetricKind,
    control_label: str,
    variant_label: str,
    control: np.ndarray,
    variant: np.ndarray,
    alpha: float,
    denominators: tuple[np.ndarray, np.ndarray] | None = None,
    adjusted: bool = False,
) -> ExperimentComparison:
    nc, nv = len(control), len(variant)
    if nc < 2 or nv < 2:
        small = control_label if nc < 2 else variant_label
        return _not_tested(
            name,
            variant_label,
            control_label,
            nc,
            nv,
            f"Too few units in {small} to compare {name}.",
        )
    checks: list[ExperimentCheck] = []
    if kind == "ratio":
        assert denominators is not None
        dc, dv = denominators
        if dc.sum() == 0 or dv.sum() == 0:
            return _not_tested(
                name,
                variant_label,
                control_label,
                nc,
                nv,
                f"A denominator sums to 0: {name} cannot be compared.",
            )
        estimate = st.delta_ratio((control, dc), (variant, dv), alpha)
        if max(nc, nv) <= BOOTSTRAP_MAX_UNITS:
            low, high = st.bootstrap_ratio((control, dc), (variant, dv), alpha)
            checks.append(
                ExperimentCheck(
                    "bootstrap",
                    None,
                    low,
                    high,
                    f"Bootstrap interval ({st.BOOTSTRAP_DRAWS:,} draws, seeded): a check of the "
                    "delta method.",
                )
            )
    elif kind == "rate" and not adjusted:
        estimate = st.two_proportions(int(control.sum()), nc, int(variant.sum()), nv, alpha)
    elif kind == "count" and not adjusted:
        spread = st.dispersion(np.concatenate([control, variant]))
        low_d, high_d = POISSON_DISPERSION
        if spread is not None and low_d <= spread <= high_d:
            estimate = st.poisson(control, variant, alpha)
            other = st.welch(st.group_stats(control), st.group_stats(variant), alpha)
            checks.append(
                ExperimentCheck(
                    "welch_t",
                    other.p_value,
                    other.low,
                    other.high,
                    f"Variance ≈ mean (dispersion {spread:.2f}): Poisson test; Welch's t agrees "
                    "when the counts are large.",
                )
            )
        else:
            estimate = st.welch(st.group_stats(control), st.group_stats(variant), alpha)
            if spread is not None:
                checks.append(
                    ExperimentCheck(
                        "dispersion",
                        None,
                        None,
                        None,
                        f"Variance is {spread:.2f}× the mean: Welch's t on the counts (a Poisson "
                        "test would be too confident).",
                    )
                )
    else:
        estimate = st.welch(st.group_stats(control), st.group_stats(variant), alpha)
    if kind == "mean" and not adjusted:
        skew = st.skewness(np.concatenate([control, variant]))
        if skew is not None and abs(skew) > SKEWED:
            checks.append(
                ExperimentCheck(
                    "mann_whitney",
                    st.rank_check(control, variant),
                    None,
                    None,
                    f"Very skewed (skewness {skew:.1f}): Mann–Whitney compares the whole "
                    "distributions.",
                )
            )
            if max(nc, nv) <= BOOTSTRAP_MAX_UNITS:
                low, high = st.bootstrap_diff(control, variant, alpha)
                checks.append(
                    ExperimentCheck(
                        "bootstrap",
                        None,
                        low,
                        high,
                        f"Bootstrap interval of the difference ({st.BOOTSTRAP_DRAWS:,} draws, "
                        "seeded).",
                    )
                )
    relative, relative_low, relative_high = st.lift(estimate, alpha)
    power, mde = st.power_and_mde(estimate, nc, nv, alpha)
    control_value = estimate.control
    mde_relative = mde / abs(control_value) if mde is not None and control_value else None
    return ExperimentComparison(
        name,
        variant_label,
        control_label,
        estimate.test,
        nc,
        nv,
        estimate.control,
        estimate.variant,
        estimate.diff,
        estimate.low,
        estimate.high,
        relative,
        relative_low,
        relative_high,
        estimate.statistic,
        estimate.p_value,
        None,
        False,
        power,
        mde,
        mde_relative,
        "unclear",
        "",
        checks,
    )


def _decide(c: ExperimentComparison, kind: MetricKind, alpha: float, holm: int) -> None:
    if c.verdict == "not_tested":
        return
    p = c.p_adjusted if c.p_adjusted is not None else c.p_value
    c.significant = p is not None and p < alpha
    if c.significant:
        c.verdict = "better" if (c.diff or 0) > 0 else "worse"
    elif c.power is not None and c.power < UNDERPOWERED:
        c.verdict = "underpowered"
    else:
        c.verdict = "unclear"
    c.sentence = _sentence(c, kind, alpha, holm)


# --- The analysis -------------------------------------------------------------------------------


@dataclass
class _Prepared:
    data: Frame
    kind: MetricKind
    reason: str
    cuped: ExperimentCuped | None
    values: str  # the column compared (CUPED-adjusted when a pre-period is given)
    skewness: float | None


def _prepare(frame: Frame, metric: MetricInput, kind: MetricKind, reason: str) -> _Prepared:
    needed = [metric.value] + ([metric.denominator] if metric.denominator else [])
    data = frame.dropna(subset=needed)
    values = metric.value
    cuped = None
    if metric.pre:
        pre = data[metric.pre]
        filled = int(pre.isna().sum())
        pre = pre.fillna(pre.mean()) if filled < len(pre) else pre
        adjusted = (
            st.cuped(data[metric.value].to_numpy(float), pre.to_numpy(float))
            if kind != "ratio" and filled < len(pre)
            else None
        )
        if adjusted is not None:
            data = data.assign(__cuped=adjusted.values)
            values = "__cuped"
            sentence = (
                f"CUPED reduced the variance by {max(adjusted.reduction, 0) * 100:.0f} % "
                f"(correlation with {metric.spec.pre}: {adjusted.correlation:.2f})"
                + (f"; {filled:,} units without it got its mean." if filled else ".")
            )
            cuped = ExperimentCuped(
                metric.spec.pre or "",
                adjusted.theta,
                adjusted.correlation,
                adjusted.reduction,
                filled,
                sentence,
            )
    skew = st.skewness(data[metric.value].to_numpy(float)) if kind == "mean" else None
    return _Prepared(data, kind, reason, cuped, values, skew)


def _sides(prepared: _Prepared, metric: MetricInput, label: str):
    rows = prepared.data[prepared.data[VARIANT] == label]
    values = rows[prepared.values].to_numpy(float)
    denominator = rows[metric.denominator].to_numpy(float) if metric.denominator else None
    return values, denominator


def _comparisons(
    prepared: _Prepared,
    metric: MetricInput,
    control: str,
    others: list[str],
    alpha: float,
) -> list[ExperimentComparison]:
    control_values, control_den = _sides(prepared, metric, control)
    found = []
    for label in others:
        values, den = _sides(prepared, metric, label)
        found.append(
            _compare(
                metric.spec.column if not metric.denominator else _label(metric.spec),
                prepared.kind,
                control,
                label,
                control_values,
                values,
                alpha,
                (control_den, den) if control_den is not None and den is not None else None,
                adjusted=prepared.cuped is not None,
            )
        )
    return found


def _label(spec: ExperimentMetric) -> str:
    return f"{spec.column} ÷ {spec.denominator}" if spec.denominator else spec.column


def _value_of(prepared: _Prepared, metric: MetricInput, label: str) -> float | None:
    rows = prepared.data[prepared.data[VARIANT] == label]
    if not len(rows):
        return None
    if metric.denominator:
        total = float(rows[metric.denominator].sum())
        return float(rows[metric.value].sum()) / total if total else None
    return float(rows[metric.value].mean())


def analyze(
    frame: Frame,
    design: ExperimentDesign,
    metrics: list[MetricInput],
    repeated: list[tuple[str, float]] | None = None,
) -> ExperimentResult:
    """`repeated`: columns that look like a unit and repeat (name, rows per value), to warn about
    when no unit is chosen."""
    frame = frame[frame[VARIANT].notna()]
    rows = len(frame)
    counts = frame[VARIANT].value_counts()
    levels = _natural([str(level) for level in counts.index])
    if not MIN_LEVELS <= len(levels) <= MAX_LEVELS:
        raise ExperimentError(
            f"{design.variant} has {len(levels)} values: an experiment needs 2 to 5 variants.",
            "variant",
        )
    row_counts = [int(counts[level]) for level in levels]
    control = design.control or guess_control(levels, row_counts)
    if control not in levels:
        raise ExperimentError(
            f"{control!r} is not a value of {design.variant} ({', '.join(levels)}).", "control"
        )
    order = [control, *(level for level in levels if level != control)]
    others = order[1:]
    warnings: list[str] = []

    kinds: list[MetricKind] = []
    reasons: list[str] = []
    for metric in metrics:
        kind, reason = _kind(metric, frame[metric.value])
        kinds.append(kind)
        reasons.append(reason)

    suggested_unit = None
    if design.unit:
        mixed = frame.groupby(UNIT, observed=True)[VARIANT].nunique()
        crossed = mixed[mixed > 1].index
        if len(crossed):
            frame = frame[~frame[UNIT].isin(crossed)]
            warnings.append(
                f"{len(crossed):,} {design.unit} values appear in more than one variant: their "
                "rows are left out (check how users were assigned)."
            )
        frame = frame[frame[UNIT].notna()]
        data = _per_unit(frame, metrics, kinds)
    else:
        data = frame
        if repeated:
            name, per_value = repeated[0]
            suggested_unit = name
            warnings.append(
                f"{name} repeats ({per_value:.1f} rows per value): if a {name} can have several "
                f"rows, choose it as the unit, otherwise rows are not independent and the "
                "intervals are too narrow."
            )

    units = data[VARIANT].value_counts()
    observed = [int(units.get(level, 0)) for level in order]
    shares = _expected(design, order)
    statistic, p = st.srm(observed, shares)
    alarm = p is not None and p < SRM_ALARM
    planned = " / ".join(f"{s * 100:.0f}" for s in shares)
    srm = ExperimentSrm(
        statistic,
        p,
        alarm,
        "The groups are not the size you planned: the results may be biased. Check how users "
        "were assigned."
        if alarm
        else f"The groups match the planned split {planned} (χ² {format_p(p)}).",
    )
    total_units = sum(observed)
    variants = [
        ExperimentVariant(
            label,
            label == control,
            int(counts.get(label, 0)),
            observed[i],
            observed[i] / total_units if total_units else 0.0,
            shares[i],
        )
        for i, label in enumerate(order)
    ]

    primary = design.primary or design.metrics[0].column
    if primary not in [m.column for m in design.metrics]:
        raise ExperimentError(
            f"The primary metric {primary!r} is not among the metrics.", "primary"
        )
    results: list[ExperimentMetricResult] = []
    prepared_by_column: dict[str, _Prepared] = {}
    for metric, kind, reason in zip(metrics, kinds, reasons, strict=True):
        prepared = _prepare(data, metric, kind, reason)
        prepared_by_column[metric.spec.column] = prepared
        results.append(
            ExperimentMetricResult(
                metric.spec.column,
                _label(metric.spec),
                kind,
                reason,
                metric.spec.column == primary,
                _aggregation(kind, design.unit, bool(metric.pre)) if design.unit else None,
                [_value_of(prepared, metric, label) for label in order],
                _comparisons(prepared, metric, control, others, design.alpha),
                prepared.cuped,
                prepared.skewness,
            )
        )

    every = [c for r in results for c in r.comparisons]
    tested = [c for c in every if c.p_value is not None]
    for c, adjusted in zip(every, st.holm([c.p_value for c in every]), strict=True):
        c.p_adjusted = adjusted
    holm = len(tested)
    for r in results:
        for c in r.comparisons:
            _decide(c, r.kind, design.alpha, holm)
    correction = (
        f"p adjusted for {holm} comparisons (Holm)"
        if holm > 1
        else "One comparison: no adjustment needed."
    )

    by_column = {m.spec.column: m for m in metrics}
    main = next(r for r in results if r.primary)
    segments = (
        _segments(data, design, by_column[primary], main.kind, control, others)
        if design.segment
        else None
    )
    cumulative = (
        _cumulative(data, design, by_column[primary], main.kind, control, others)
        if design.time
        else None
    )
    resolved = design.model_copy(
        update={
            "control": control,
            "primary": primary,
            "metrics": [
                m.spec.model_copy(update={"kind": kind})
                for m, kind in zip(metrics, kinds, strict=True)
            ],
        }
    )
    return ExperimentResult(
        resolved,
        rows,
        total_units,
        variants,
        srm,
        results,
        holm,
        correction,
        warnings,
        suggested_unit,
        segments,
        cumulative,
    )


def _expected(design: ExperimentDesign, order: list[str]) -> list[float]:
    if not design.expected:
        return [1 / len(order)] * len(order)
    unknown = set(design.expected) - set(order)
    if unknown:
        raise ExperimentError(
            f"No variant named {', '.join(sorted(unknown))} in {design.variant}.", "expected"
        )
    shares = [max(float(design.expected.get(label, 0.0)), 0.0) for label in order]
    total = sum(shares)
    if total <= 0 or min(shares) == 0:
        raise ExperimentError("Give every variant a planned share above 0.", "expected")
    return [share / total for share in shares]


# --- Subgroups ----------------------------------------------------------------------------------


def _segments(
    data: Frame,
    design: ExperimentDesign,
    metric: MetricInput,
    kind: MetricKind,
    control: str,
    others: list[str],
) -> ExperimentSegments:
    sizes = data[SEGMENT].fillna("(empty)").astype(str).value_counts()
    levels: list[ExperimentSegment] = []
    skipped = 0
    labels = data[SEGMENT].fillna("(empty)").astype(str)
    for level, size in sizes.items():
        part = data[labels == level]
        per_variant = part[VARIANT].value_counts()
        if len(levels) >= SEGMENT_LEVELS or any(
            per_variant.get(v, 0) < MIN_SEGMENT_UNITS for v in [control, *others]
        ):
            skipped += 1
            continue
        prepared = _prepare(part, metric, kind, "")
        comparisons = _comparisons(prepared, metric, control, others, design.alpha)
        for c in comparisons:
            _decide(c, kind, design.alpha, 1)
        levels.append(ExperimentSegment(str(level), int(size), comparisons))
    hypotheses = len(levels) * len(others)
    warning = "Subgroups are hypotheses to confirm, not results: " + (
        f"with {hypotheses} comparisons, about {hypotheses * design.alpha:.1f} may look "
        "significant by chance alone (no correction here)."
        if hypotheses
        else "none is large enough to compare."
    )
    if skipped:
        warning += (
            f" {skipped} level{'s' if skipped > 1 else ''} left out (fewer than "
            f"{MIN_SEGMENT_UNITS} units in a variant, or beyond the {SEGMENT_LEVELS} largest)."
        )
    return ExperimentSegments(design.segment or "", _label(metric.spec), levels, skipped, warning)


# --- Over time ----------------------------------------------------------------------------------


def _grain(dates: Frame) -> Literal["day", "week", "month"]:
    days = dates.dt.normalize().nunique()
    if days <= MAX_DAYS:
        return "day"
    if dates.dt.to_period("W").nunique() <= MAX_WEEKS:
        return "week"
    return "month"


def _period(dates: Frame, grain: str) -> Frame:
    if grain == "day":
        return dates.dt.strftime("%Y-%m-%d")
    if grain == "week":
        return dates.dt.to_period("W").dt.start_time.dt.strftime("%Y-%m-%d")
    return dates.dt.to_period("M").dt.start_time.dt.strftime("%Y-%m-%d")


def _cumulative(
    data: Frame,
    design: ExperimentDesign,
    metric: MetricInput,
    kind: MetricKind,
    control: str,
    others: list[str],
) -> ExperimentCumulative:
    """The difference to the control on all units seen so far, period after period, from running
    sums (normal intervals; the delta method for a ratio)."""
    needed = [metric.value, TIME] + ([metric.denominator] if metric.denominator else [])
    rows = data.dropna(subset=needed)
    dates: Frame = pd.to_datetime(rows[TIME])
    grain = _grain(dates) if len(dates) else "day"
    y = rows[metric.value].astype(float)
    x = rows[metric.denominator].astype(float) if metric.denominator else pd.Series(1.0, rows.index)
    sums: Frame = pd.DataFrame(
        {
            "period": _period(dates, grain),
            VARIANT: rows[VARIANT],
            "n": 1.0,
            "y": y,
            "y2": y * y,
            "x": x,
            "x2": x * x,
            "xy": x * y,
        }
    )
    table = sums.groupby(["period", VARIANT]).sum().unstack(VARIANT, fill_value=0.0).cumsum()
    periods = [str(p) for p in table.index]
    z = st.z_value(design.alpha)

    def side(label: str) -> tuple[np.ndarray, np.ndarray]:
        if ("n", label) not in table:
            empty = np.full(len(periods), np.nan)
            return empty, empty
        n = table[("n", label)].to_numpy()
        sy, sy2 = table[("y", label)].to_numpy(), table[("y2", label)].to_numpy()
        sx, sx2, sxy = (table[(k, label)].to_numpy() for k in ("x", "x2", "xy"))
        with np.errstate(divide="ignore", invalid="ignore"):
            if kind == "ratio":
                ratio = sy / sx
                var_y = (sy2 - sy**2 / n) / (n - 1)
                var_x = (sx2 - sx**2 / n) / (n - 1)
                cov = (sxy - sx * sy / n) / (n - 1)
                mean_x = sx / n
                variance = (var_y - 2 * ratio * cov + ratio**2 * var_x) / (mean_x**2 * n)
                return ratio, np.sqrt(np.maximum(variance, 0))
            mean = sy / n
            variance = (sy2 - sy**2 / n) / (n - 1)
            return mean, np.sqrt(np.maximum(variance, 0) / n)

    base, base_se = side(control)
    lines = []
    for label in others:
        value, se = side(label)
        diff = value - base
        spread = np.hypot(se, base_se)
        with np.errstate(divide="ignore", invalid="ignore"):
            statistic = diff / spread
        p_values = 2 * stats.norm.sf(np.abs(statistic))

        def listed(values: np.ndarray) -> list[float | None]:
            return [float(v) if math.isfinite(v) else None for v in values]

        lines.append(
            ExperimentCumulativeLine(
                label,
                periods,
                listed(diff),
                listed(diff - z * spread),
                listed(diff + z * spread),
                listed(p_values),
            )
        )
    warning = (
        "Watching the difference every day and stopping as soon as p < 0.05 inflates false "
        "positives: decide the sample size first (Plan), then read the result once."
    )
    return ExperimentCumulative(design.time or "", _label(metric.spec), kind, grain, lines, warning)
