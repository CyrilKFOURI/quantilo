"""The automatic choices of a regression (067), each with its reason: the structure of the rows
(one dated series, a panel), the standard errors (R-REG-SE-01…05), the target and columns when
none are given (R-REG-20, R-REG-21)."""

from dataclasses import dataclass, field
from typing import Literal, cast

import numpy as np
import pandas as pd

from quantilo.correlations.matrix import Matrix
from quantilo.regression.covariance import LABELS, Kind, newey_west_lag

SeRequest = Literal[
    "auto", "classic", "hc0", "hc1", "hc2", "hc3", "hac", "cluster", "cluster2", "dk"
]
MIN_SERIES = 20  # rows of a dated series before HAC is worth it
FEW_CLUSTERS = 40  # R-REG-SE-03
TWO_WAY = 20  # R-REG-SE-01: both dimensions at least this many levels
SUGGESTED = 5  # R-REG-21
REDUNDANT = 0.95  # between two columns picked
SAME_AS_TARGET = 0.99  # the target in other units (price before and after tax)
TARGET_ROLES = ("amount", "cost", "target", "quantity")
NUMERIC = ("numeric_continuous", "numeric_discrete")
RANK = {"strong": 3, "moderate": 2, "weak": 1, "none": 0}
SE_KINDS: tuple[Kind, ...] = (
    "classic",
    "hc0",
    "hc1",
    "hc2",
    "hc3",
    "hac",
    "cluster",
    "cluster2",
    "dk",
)


@dataclass
class RowStructure:
    """What the rows are: independent, one series in time, or a panel (entities × periods)."""

    kind: Literal["none", "series", "panel"] = "none"
    date: str | None = None  # series: the date the rows are ordered by
    entity: str | None = None
    period: str | None = None
    periods: int = 0
    entities: int = 0
    message: str | None = None


def series_date(frame: pd.DataFrame, dates: list[str], rows: int) -> str | None:
    """A date with one value per row (R-REG-SE-02): the rows form one series in time."""
    for name in dates:
        values = cast(pd.Series, frame[name]).dropna()
        if len(values) >= max(MIN_SERIES, 0.9 * rows) and values.is_unique:
            return name
    return None


def structure(
    frame: pd.DataFrame,
    dates: list[str],
    periods: list[str],
    entities: list[str],
    rows: int,
    distinct: dict[str, int],
) -> RowStructure:
    date = series_date(frame, dates, rows)
    if date is not None:
        return RowStructure(
            "series",
            date=date,
            periods=rows,
            message=f"One row per `{date}`: the rows form one series in time, sorted by date.",
        )
    if not periods:
        return RowStructure()
    from quantilo.ml import panel

    found = panel.recognise(frame, periods, entities, rows, distinct)
    if not found.panel or found.entity is None or found.period is None:
        return RowStructure()
    entity_count = found.entities or 0
    return RowStructure(
        "panel",
        entity=found.entity,
        period=found.period,
        periods=found.periods,
        entities=entity_count,
        message=(
            f"Panel data: {entity_count:,} `{found.entity}` seen over {found.periods:,} "
            f"`{found.period}`."
        ),
    )


@dataclass
class SeChoice:
    kind: Kind
    label: str
    rule: str | None  # the rule that chose it, None when the user did
    reason: str
    lag: int | None = None
    groups: list[str] = field(default_factory=list)  # the columns clustered on
    clusters: int | None = None  # the smallest number of groups
    warning: str | None = None


@dataclass
class Grouping:
    """The arrays a choice of standard errors may need, aligned with the rows used."""

    cluster: str | None = None  # "Grouped by", chosen by the user
    cluster_codes: np.ndarray | None = None
    entity_codes: np.ndarray | None = None
    period_codes: np.ndarray | None = None


def _count(codes: np.ndarray | None) -> int:
    return len(np.unique(codes)) if codes is not None else 0


def available(kind: Kind, shape: RowStructure, grouping: Grouping) -> str | None:
    """Why `kind` cannot be computed here, or None when it can."""
    if kind == "hac" and shape.kind != "series":
        return "needs rows in time order: one row per date"
    if kind == "cluster" and grouping.cluster is None and shape.kind != "panel":
        return "needs a column in Grouped by, or panel data"
    if kind == "cluster2" and shape.kind != "panel":
        return "needs panel data (an entity and a period)"
    if kind == "dk" and shape.kind != "panel":
        return "needs panel data (periods to sum over)"
    return None


def choose(requested: SeRequest, shape: RowStructure, grouping: Grouping, n: int) -> SeChoice:
    """R-REG-SE-01…05. A kind asked for but not available falls back to the rule, saying why."""
    note = ""
    if requested != "auto":
        kind = cast(Kind, requested)
        missing = available(kind, shape, grouping)
        if missing is None:
            return _chosen(
                kind, shape, grouping, n, None, f"{LABELS[kind]} chosen in the settings."
            )
        note = f"{LABELS[kind]} {missing}: "
    if grouping.cluster is not None:
        return _chosen(
            "cluster",
            shape,
            grouping,
            n,
            "R-REG-SE-03",
            note
            + f"rows that share a `{grouping.cluster}` share its shocks: errors clustered on it.",
        )
    if shape.kind == "panel":
        return _chosen(
            "cluster",
            shape,
            grouping,
            n,
            "R-REG-SE-01",
            note
            + f"each `{shape.entity}` comes back every `{shape.period}`, its errors look alike: "
            f"clustered by `{shape.entity}`.",
        )
    if shape.kind == "series":
        lag = newey_west_lag(n)
        return _chosen(
            "hac",
            shape,
            grouping,
            n,
            "R-REG-SE-02",
            note + f"errors close in time look alike, classic intervals would be too narrow: "
            f"Newey–West with {lag} lags (⌊4(T/100)^(2/9)⌋).",
        )
    return _chosen(
        "hc3",
        shape,
        grouping,
        n,
        "R-REG-SE-05",
        note
        + "HC3 always: choosing the standard errors from a pre-test distorts the intervals, and "
        "HC3 stays right when the variances are equal (Long & Ervin, 2000).",
    )


def _chosen(
    kind: Kind, shape: RowStructure, grouping: Grouping, n: int, rule: str | None, reason: str
) -> SeChoice:
    choice = SeChoice(kind, LABELS[kind], rule, reason)
    if kind == "hac":
        choice.lag = newey_west_lag(n)
    elif kind == "dk":
        choice.lag = newey_west_lag(_count(grouping.period_codes))
        choice.groups = [shape.period or ""]
        choice.clusters = _count(grouping.period_codes)
    elif kind == "cluster":
        if grouping.cluster is not None:
            choice.groups = [grouping.cluster]
            choice.clusters = _count(grouping.cluster_codes)
        else:
            choice.groups = [shape.entity or ""]
            choice.clusters = _count(grouping.entity_codes)
    elif kind == "cluster2":
        choice.groups = [shape.entity or "", shape.period or ""]
        choice.clusters = min(_count(grouping.entity_codes), _count(grouping.period_codes))
    if (
        choice.clusters is not None
        and kind in ("cluster", "cluster2")
        and choice.clusters < FEW_CLUSTERS
    ):
        choice.warning = (
            f"Only {choice.clusters} groups: clustered intervals are too narrow below "
            f"{FEW_CLUSTERS} (a wild cluster bootstrap will come with resampling)."
        )
    return choice


def two_way_hint(shape: RowStructure) -> str | None:
    """R-REG-SE-01: both dimensions large enough → two-way clustering is proposed, not imposed."""
    if shape.kind != "panel" or shape.entities < TWO_WAY or shape.periods < TWO_WAY:
        return None
    return (
        f"`{shape.entity}` ({shape.entities}) and `{shape.period}` ({shape.periods}) both have "
        f"at least {TWO_WAY} levels: two-way clustering is worth a look in Standard errors."
    )


# --- Target and columns when none are given (R-REG-20, R-REG-21) ---------------------------------


@dataclass
class RegressionPick:
    column: str
    reason: str


@dataclass
class RegressionSuggestion:
    target: RegressionPick | None
    columns: list[RegressionPick]
    candidates: list[str]  # every column that can be a target (numbers, yes/no)
    regressors: list[str]  # every column that can enter the design


@dataclass
class ColumnInfo:
    name: str
    type: str
    role: str | None


TARGET_TYPES = (*NUMERIC, "boolean")
REGRESSOR_TYPES = (*NUMERIC, "boolean", "categorical_nominal", "categorical_ordinal")


def _links(matrix: Matrix, target: str) -> list[tuple[str, float, str, str]]:
    """(column, value, metric, evidence) of every pair with the target, best evidence first."""
    found = []
    for pair in matrix.pairs:
        if target not in (pair.a, pair.b):
            continue
        other = pair.b if pair.a == target else pair.a
        found.append((other, pair.value, pair.metric, pair.evidence))
    found.sort(key=lambda item: (-RANK[item[3]], -abs(item[1])))
    return found


def pick_target(columns: list[ColumnInfo], matrix: Matrix | None) -> RegressionPick | None:
    numbers = [c for c in columns if c.type in NUMERIC]
    for role in TARGET_ROLES:
        for column in numbers:
            if column.role == role:
                return RegressionPick(
                    column.name, f"R-REG-20: a numeric column with the role {role}."
                )
    if matrix is not None and numbers:

        def linked(name: str) -> int:
            return sum(
                1 for _, _, _, level in _links(matrix, name) if level in ("strong", "moderate")
            )

        best = max(numbers, key=lambda column: linked(column.name))
        count = linked(best.name)
        return RegressionPick(
            best.name, f"R-REG-20: the number with the most clear links ({count})."
        )
    return (
        RegressionPick(numbers[0].name, "R-REG-20: the first numeric column.") if numbers else None
    )


METRIC_NAMES = {"pearson": "r", "spearman": "ρ", "cramers_v": "V", "eta": "η"}


def pick_columns(target: str, matrix: Matrix | None, allowed: set[str]) -> list[RegressionPick]:
    """The target's strongest links (evidence at least weak), without near-duplicates."""
    if matrix is None:
        return []
    names = [c.name for c in matrix.columns]
    index = {name: i for i, name in enumerate(names)}
    picked: list[RegressionPick] = []
    for other, value, metric, level in _links(matrix, target):
        if other not in allowed or level == "none":
            continue
        if metric in ("pearson", "spearman") and abs(value) > SAME_AS_TARGET:
            continue
        twin = False
        for chosen in picked:
            i, j = index.get(chosen.column), index.get(other)
            if i is None or j is None:
                continue
            between = matrix.values[i][j]
            kind = matrix.metrics[i][j]
            if between is not None and kind in ("pearson", "spearman") and abs(between) > REDUNDANT:
                twin = True
        if twin:
            continue
        symbol = METRIC_NAMES.get(metric, metric)
        picked.append(
            RegressionPick(other, f"R-REG-21: {symbol} = {value:.2f} with `{target}`, {level}.")
        )
        if len(picked) == SUGGESTED:
            break
    return picked
