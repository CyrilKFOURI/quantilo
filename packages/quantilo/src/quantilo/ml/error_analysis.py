# ruff: noqa: RUF001, RUF002, RUF003  (≤, ≥, ≠, ×, – are the app's symbols)
"""Where the model is wrong (036), on its held-out predictions only: the test rows of a split, the
out-of-fold predictions of a cross-validation or of a panel walk-forward.

- Subgroups notably worse than average, found among every category and quintile of the columns
  that matter most for the error, their pairs, and the nodes of a depth-3 tree fitted on the
  error. Each is tested against the other rows (Welch, one-sided); Benjamini–Hochberg keeps the
  rate of false discoveries at 5 %. The tree is fitted on one half of the rows and its leaves are
  tested on the other half, so its search does not make its own leaves look significant.
- The error along each column, the worst predictions, rule-based hints, two runs compared
  subgroup by subgroup, and metrics side by side for the groups of a sensitive column.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, cast

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.tree import DecisionTreeRegressor
from statsmodels.stats.multitest import multipletests

from quantilo.filters.model import Condition
from quantilo.ml.data import NUMERIC, Field
from quantilo.ml.target import Task

Measure = Literal["absolute_error", "misclassified", "log_loss"]

SEED = 42
MIN_ROWS = 30  # a subgroup has at least 30 rows…
MIN_SHARE = 0.01  # …and 1 % of the rows
MAX_COLUMNS = 30  # columns searched: those that matter most for the error
PAIR_COLUMNS = 10  # pairs of conditions among the first 10 only
CATEGORIES = 30  # most frequent categories of a column tried on their own
PAIR_BINS = 10  # bins of a column tried in pairs
FEW_VALUES = 10  # a number with at most 10 values is read as categories
QUANTILES = (0.2, 0.4, 0.6, 0.8)
RANKING_ROWS = 20_000  # columns are ranked on a sample beyond
TREE_DEPTH = 3
TREE_ROWS = 50_000  # the tree learns on at most 50 000 rows of its half
ALPHA = 0.05
NOTABLY = 1.2  # a subgroup is notably worse with an error at least 20 % above average
MAX_SLICES = 10
OVERLAP = 0.7  # a subgroup sharing 70 % of its rows with one listed above is left out
MAX_CHANGED = 5
CHANGE = 0.1  # two runs: a change of at least 10 % of the first run's average error
WORST = 50
EXAMPLES = 10
SMALL_SHARE = 0.05  # "few examples" below 5 % of the rows
GROUPS = 20  # groups of a sensitive column

MEASURE_LABELS: dict[str, str] = {
    "absolute_error": "absolute error",
    "misclassified": "share of wrong predictions",
    "log_loss": "log-loss",
}


@dataclass
class HeldOut:
    """Every held-out prediction of a run (rows the model was not trained on)."""

    rows: np.ndarray  # file row numbers
    actual: np.ndarray  # numbers, or class labels as text
    predicted: np.ndarray  # numbers, or class labels as text
    probabilities: np.ndarray | None = None  # classes: one column per class of `classes`
    classes: list[str] = field(default_factory=list)


@dataclass
class ErrorPrediction:
    """One held-out row: its prediction and how wrong it is."""

    row: int
    actual: str
    predicted: str
    error: float | None = None  # numbers: |predicted − actual|
    probability: float | None = None  # classes: the probability given to the actual class


@dataclass
class Slice:
    """A subgroup where the model is notably worse than on average."""

    where: str  # "region = South and age < 25"
    conditions: list[Condition]  # the same, as filters (all of them apply)
    columns: list[str]
    rows: int
    share: float
    error: float  # mean error in the subgroup
    ratio: float  # error / overall error
    excess: float  # rows × (error − overall): the error the subgroup adds
    p_value: float  # Welch, one-sided, against the other rows
    q_value: float  # after Benjamini–Hochberg
    found_by: Literal["groups", "tree"]
    sentence: str
    examples: list[ErrorPrediction]  # its worst predictions


@dataclass
class ColumnBin:
    label: str
    rows: int
    error: float | None


@dataclass
class ColumnErrors:
    """The mean error per category or per quintile of one column."""

    column: str
    kind: Literal["number", "date", "category"]
    bins: list[ColumnBin]
    spread: float  # worst bin − best bin (bins of 30 rows or more)
    worst: str | None  # the worst bin
    excluded: bool  # the preparation left the column out of the model


@dataclass
class Hint:
    kind: Literal["few_rows", "excluded_column", "log_target", "rare_class"]
    text: str
    column: str | None = None


@dataclass
class ErrorAnalysis:
    """Where a model is wrong: subgroups, error by column, worst predictions, hints."""

    task: str
    measure: str
    measures: list[str]  # the measures this task offers
    label: str
    source: str  # "test rows" or "out-of-fold predictions"
    rows: int  # held-out predictions analysed
    overall: float  # mean error on every one of them
    columns: int  # columns searched
    tested: int  # candidate subgroups tested
    slices: list[Slice]
    message: str | None  # when no subgroup is notably worse
    by_column: list[ColumnErrors]
    worst: list[ErrorPrediction]
    hints: list[Hint]
    conditions: list[Condition] = field(default_factory=list)  # the filters the run trained on

    def to_pandas(self) -> pd.DataFrame:
        """One row per subgroup, the worst first."""
        return pd.DataFrame(
            [
                {
                    "where": s.where,
                    "rows": s.rows,
                    "share": s.share,
                    "error": s.error,
                    "overall": self.overall,
                    "ratio": s.ratio,
                    "excess": s.excess,
                    "p_value": s.p_value,
                    "q_value": s.q_value,
                    "found_by": s.found_by,
                }
                for s in self.slices
            ],
            columns=[
                "where",
                *("rows", "share", "error", "overall", "ratio", "excess"),
                *("p_value", "q_value", "found_by"),
            ],
        )


@dataclass
class ChangedSlice:
    """A subgroup where the second run changes the error more than elsewhere."""

    where: str
    conditions: list[Condition]
    rows: int
    share: float
    first: float  # mean error of the first run in the subgroup
    second: float
    p_value: float
    q_value: float
    sentence: str


@dataclass
class ErrorComparison:
    measure: str
    label: str
    rows: int  # held-out rows both runs predicted
    first: float  # overall mean errors
    second: float
    better: list[ChangedSlice]  # where the second run fixes errors of the first
    worse: list[ChangedSlice]  # where it makes them worse
    message: str | None


@dataclass
class GroupMetrics:
    group: str
    rows: int
    values: dict[str, float | None]
    few: bool  # under 30 rows: shown, left out of the gaps


@dataclass
class Fairness:
    """Metrics side by side for each group of a sensitive column (display only)."""

    column: str
    metrics: list[str]
    labels: dict[str, str]
    groups: list[GroupMetrics]
    gaps: dict[str, float | None]  # largest − smallest, over groups of 30 rows or more
    positive: str | None  # two classes: the class whose rate and recall are shown


# --- Errors row by row ---------------------------------------------------------------------------


def measures_for(task: Task | str) -> list[Measure]:
    return ["absolute_error"] if task == "regression" else ["misclassified", "log_loss"]


def _true_class_probability(held: HeldOut) -> np.ndarray:
    assert held.probabilities is not None
    codes = pd.Index(held.classes).get_indexer(held.actual.astype(str))
    found = held.probabilities[np.arange(len(codes)), np.clip(codes, 0, None)]
    return np.where(codes >= 0, found, 0.0).astype(float)


def row_errors(held: HeldOut, measure: Measure) -> np.ndarray:
    if measure == "absolute_error":
        return np.abs(held.predicted.astype(float) - held.actual.astype(float))
    if measure == "misclassified":
        return (held.predicted.astype(str) != held.actual.astype(str)).astype(float)
    return -np.log(np.clip(_true_class_probability(held), 1e-15, 1.0))


def _severity(held: HeldOut) -> np.ndarray:
    """How badly each row is predicted: its absolute error, or how little probability the actual
    class received."""
    if held.probabilities is None:
        return row_errors(held, "absolute_error")
    return row_errors(held, "log_loss")


def _label(value: object) -> str:
    if isinstance(value, float | np.floating):
        return f"{float(value):.6g}"
    return str(value)


def _prediction(held: HeldOut, i: int) -> ErrorPrediction:
    if held.probabilities is None:
        actual, predicted = float(held.actual[i]), float(held.predicted[i])
        return ErrorPrediction(
            int(held.rows[i]), _label(actual), _label(predicted), abs(predicted - actual)
        )
    probability = float(_true_class_probability(held)[i])
    return ErrorPrediction(
        int(held.rows[i]), str(held.actual[i]), str(held.predicted[i]), probability=probability
    )


def _worst_positions(severity: np.ndarray, count: int) -> np.ndarray:
    return np.argsort(-severity, kind="stable")[:count]


def worst_rows(held: HeldOut, count: int = WORST) -> list[int]:
    """File rows of the worst predictions (the explanation keeps them, for their waterfall)."""
    return [int(held.rows[i]) for i in _worst_positions(_severity(held), count)]


# --- Columns cut into bins -----------------------------------------------------------------------


def number(value: float) -> str:
    """A number as a sentence shows it: 3 significant digits, thousands separated."""
    if not math.isfinite(value):
        return "—"
    if abs(value) >= 10_000:
        return f"{value:,.0f}"
    if abs(value) >= 1_000:
        return f"{value:,.1f}".removesuffix(".0")
    return f"{value:.3g}"


def _percent(share: float, digits: int = 1) -> str:
    return f"{share * 100:.{digits}f} %"


DAY = 86_400 * 10**9


@dataclass
class _Bins:
    """A column cut into bins: categories, or quintiles of a number or a date."""

    column: str
    kind: Literal["number", "date", "category"]
    codes: np.ndarray  # the bin of each row
    labels: list[str]  # on a chart's axis
    wheres: list[str | None]  # in a sentence; None: not a subgroup of its own ("Other")
    conditions: list[list[Condition]]
    paired: list[int]  # bins tried in pairs

    @property
    def size(self) -> int:
        return len(self.labels)


def _as_numbers(values: pd.Series, kind: str) -> np.ndarray:
    if kind == "date":
        moments = pd.to_datetime(values, errors="coerce")
        nanoseconds = moments.to_numpy(dtype="datetime64[ns]").astype("int64").astype(float)
        return np.where(moments.isna().to_numpy(), np.nan, nanoseconds)
    return cast(pd.Series, pd.to_numeric(values, errors="coerce")).to_numpy(dtype=float)


def _show(value: float, kind: str, whole_days: bool) -> str:
    if kind != "date":
        return number(value)
    moment = pd.Timestamp(int(value))
    return moment.strftime("%Y-%m-%d" if whole_days else "%Y-%m-%d %H:%M")


def _bound(value: float, kind: str) -> float | str:
    """A bin limit as a filter takes it: a number, or an ISO date."""
    return pd.Timestamp(int(value)).isoformat() if kind == "date" else float(value)


def _quintiles(name: str, x: np.ndarray, kind: Literal["number", "date"]) -> _Bins | None:
    present = ~np.isnan(x)
    if present.sum() < 2:
        return None
    seen = x[present]
    whole_days = kind == "date" and bool(np.all(seen % DAY == 0))
    edges = np.quantile(seen, QUANTILES)
    if kind == "date":
        edges = np.round(edges / DAY) * DAY if whole_days else np.round(edges)
    else:  # round limits a sentence can say: 28.43 → 28.4
        edges = np.array([float(f"{e:.3g}") for e in edges])
    edges = np.unique(edges[edges > seen.min()])
    if not len(edges):
        return None
    codes = np.searchsorted(edges, np.where(present, x, 0.0), side="right")
    labels: list[str] = []
    wheres: list[str | None] = []
    conditions: list[list[Condition]] = []
    for i in range(len(edges) + 1):
        low = _show(edges[i - 1], kind, whole_days) if i else None
        high = _show(edges[i], kind, whole_days) if i < len(edges) else None
        found: list[Condition] = []
        if i:
            found.append(Condition(column=name, op="gte", value=_bound(edges[i - 1], kind)))
        if i < len(edges):
            found.append(Condition(column=name, op="lt", value=_bound(edges[i], kind)))
        if low is None:
            labels.append(f"< {high}")
            wheres.append(f"{name} < {high}")
        elif high is None:
            labels.append(f"≥ {low}")
            wheres.append(f"{name} ≥ {low}")
        else:
            labels.append(f"{low} – {high}")
            wheres.append(f"{low} ≤ {name} < {high}")
        conditions.append(found)
    if not present.all():
        codes = np.where(present, codes, len(edges) + 1)
        labels.append("(missing)")
        wheres.append(f"{name} is empty")
        conditions.append([Condition(column=name, op="is_null")])
    paired = list(range(len(labels)))
    return _Bins(name, kind, codes.astype(int), labels, wheres, conditions, paired)


def _categories(name: str, values: pd.Series, field_: Field) -> _Bins | None:
    numeric = field_.type in NUMERIC
    exact: dict[str, float] = {}
    if numeric:
        numbers = cast(pd.Series, pd.to_numeric(values, errors="coerce"))
        present = numbers.notna().to_numpy()
        for v in numbers.dropna().unique():
            exact[str(int(v)) if float(v).is_integer() else f"{float(v):.6g}"] = float(v)
        label_of = {value: label for label, value in exact.items()}
        text = cast(pd.Series, numbers.map(lambda v: "" if pd.isna(v) else label_of[float(v)]))
    else:
        present = values.notna().to_numpy()
        text = cast(pd.Series, values.astype(str).where(values.notna(), ""))
    counts = cast(pd.Series, text[present]).value_counts()
    if len(counts) < 2:
        return None
    top = [str(v) for v in counts.index[:CATEGORIES]]
    if numeric:
        top.sort(key=float)
    elif field_.order:
        rank = {label: i for i, label in enumerate(field_.order)}
        top.sort(key=lambda label: rank.get(label, len(rank)))
    codes = np.asarray(pd.Index(top).get_indexer(text.to_numpy()), dtype=int)
    codes = np.where(codes < 0, len(top), codes)
    labels = list(top)
    wheres: list[str | None] = [f"{name} = {label}" for label in top]
    conditions: list[list[Condition]] = []
    for label in top:
        if numeric:
            conditions.append([Condition(column=name, op="eq", value=exact[label])])
        else:
            conditions.append([Condition(column=name, op="in", values=[label])])
    sizes = {str(label): int(count) for label, count in counts.items()}
    by_size = sorted(range(len(top)), key=lambda i: -sizes[top[i]])
    paired = sorted(by_size[:PAIR_BINS])
    if len(counts) > len(top):
        labels.append(f"Other ({len(counts) - len(top)})")
        wheres.append(None)
        conditions.append([])
    if not present.all():
        codes = np.where(present, codes, len(labels))
        labels.append("(missing)")
        wheres.append(f"{name} is empty")
        conditions.append([Condition(column=name, op="is_null")])
        paired.append(len(labels) - 1)
    return _Bins(name, "category", codes, labels, wheres, conditions, paired)


def _bins(values: pd.Series, field_: Field) -> _Bins | None:
    if field_.type in ("identifier", "text", "ignored"):
        return None
    if field_.type == "datetime":
        return _quintiles(field_.name, _as_numbers(values, "date"), "date")
    if field_.type in NUMERIC:
        x = _as_numbers(values, "number")
        if len(np.unique(x[~np.isnan(x)])) > FEW_VALUES:
            return _quintiles(field_.name, x, "number")
    return _categories(field_.name, values, field_)


# --- Candidate subgroups and their test ----------------------------------------------------------


@dataclass
class _Stats:
    rows: np.ndarray
    total: np.ndarray
    squares: np.ndarray

    def mean(self) -> np.ndarray:
        return self.total / np.maximum(self.rows, 1)


def _stats(codes: np.ndarray, values: np.ndarray, size: int) -> _Stats:
    return _Stats(
        np.bincount(codes, minlength=size).astype(float),
        np.bincount(codes, weights=values, minlength=size),
        np.bincount(codes, weights=values * values, minlength=size),
    )


def _eta(bins: _Bins, values: np.ndarray) -> float:
    """Share of the error's variance its bins explain (η²)."""
    found = _stats(bins.codes, values, bins.size)
    spread = float(((values - values.mean()) ** 2).sum())
    if spread <= 0:
        return 0.0
    used = found.rows > 0
    between = (found.rows[used] * (found.mean()[used] - values.mean()) ** 2).sum()
    return float(between / spread)


def _welch(
    rows: np.ndarray,
    total: np.ndarray,
    squares: np.ndarray,
    all_rows: float,
    all_total: float,
    all_squares: float,
    two_sided: bool = False,
) -> np.ndarray:
    """p-values of Welch's test, a subgroup's mean against the other rows' (one-sided: the
    subgroup is worse)."""
    rest = all_rows - rows
    inside = total / np.maximum(rows, 1)
    outside = (all_total - total) / np.maximum(rest, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        var_in = np.maximum(squares - rows * inside**2, 0) / np.maximum(rows - 1, 1)
        var_out = np.maximum((all_squares - squares) - rest * outside**2, 0) / np.maximum(
            rest - 1, 1
        )
        a, b = var_in / np.maximum(rows, 1), var_out / np.maximum(rest, 1)
        se = np.sqrt(a + b)
        t = (inside - outside) / se
        df = (a + b) ** 2 / (a**2 / np.maximum(rows - 1, 1) + b**2 / np.maximum(rest - 1, 1))
    df = np.where(np.isfinite(df) & (df > 0), df, 1.0)
    difference = inside - outside
    t = np.where(se > 0, t, np.where(difference > 0, np.inf, np.where(difference < 0, -np.inf, 0)))
    if two_sided:
        return np.asarray(2 * stats.t.sf(np.abs(t), df), dtype=float).clip(0, 1)
    return np.asarray(stats.t.sf(t, df), dtype=float)


@dataclass
class _Group:
    """A candidate subgroup: bins of one or two columns, or a leaf of the tree."""

    parts: tuple[tuple[int, int], ...]  # (column, bin); empty for a tree leaf
    conditions: list[Condition]
    where: str
    columns: list[str]
    rows: int
    mean: float
    p_value: float = 1.0
    mask: np.ndarray | None = None  # a tree leaf's rows
    found_by: Literal["groups", "tree"] = "groups"
    column_masks: list[np.ndarray] | None = None  # a tree leaf: the rows each column keeps


def _mask_of(group: _Group, columns: list[_Bins]) -> np.ndarray:
    if group.mask is not None:
        return group.mask
    found = np.ones(len(columns[0].codes), dtype=bool)
    for position, code in group.parts:
        found &= columns[position].codes == code
    return found


def _grid(
    columns: list[_Bins],
    values: np.ndarray,
    minimum: int,
    two_sided: bool = False,
) -> tuple[list[_Group], dict[tuple[int, int], float]]:
    """Every bin of every column, and every pair of bins of the first columns, with the p-value
    of its test. Also each bin's mean, for the rule that a pair must add to its parts."""
    n, total, squares = float(len(values)), float(values.sum()), float((values**2).sum())
    groups: list[_Group] = []
    means: dict[tuple[int, int], float] = {}

    def add(found: _Stats, keys: list[tuple[tuple[int, int], ...]], cells: list[int]) -> None:
        rows, sums, sq = found.rows[cells], found.total[cells], found.squares[cells]
        keep = np.flatnonzero((rows >= minimum) & (n - rows >= MIN_ROWS))
        if not len(keep):
            return
        p = _welch(rows[keep], sums[keep], sq[keep], n, total, squares, two_sided)
        for k, at in enumerate(keep):
            key = keys[at]
            conditions = [c for column, code in key for c in columns[column].conditions[code]]
            wheres = [cast(str, columns[column].wheres[code]) for column, code in key]
            groups.append(
                _Group(
                    key,
                    conditions,
                    " and ".join(wheres),
                    [columns[column].column for column, _ in key],
                    int(rows[at]),
                    float(sums[at] / rows[at]),
                    float(p[k]),
                )
            )

    for position, bins in enumerate(columns):
        found = _stats(bins.codes, values, bins.size)
        for code in range(bins.size):
            means[(position, code)] = float(found.mean()[code])
        cells = [code for code in range(bins.size) if bins.wheres[code] is not None]
        add(found, [((position, code),) for code in cells], cells)
    paired = columns[:PAIR_COLUMNS]
    for first in range(len(paired)):
        for second in range(first + 1, len(paired)):
            a, b = paired[first], paired[second]
            combined = a.codes * b.size + b.codes
            found = _stats(combined, values, a.size * b.size)
            keys = [((first, i), (second, j)) for i in a.paired for j in b.paired]
            add(found, keys, [i * b.size + j for (_, i), (_, j) in keys])
    return groups, means


# --- Leaves of a tree fitted on the error --------------------------------------------------------


@dataclass
class _Feature:
    column: int  # position in the columns
    kind: Literal["number", "date", "category"]
    value: str | None = None  # a category: its one-hot input


def _tree_inputs(
    columns: list[_Bins], data: pd.DataFrame
) -> tuple[np.ndarray, list[_Feature], dict[int, np.ndarray]]:
    """Numbers and dates as they are (missing: the median), one input per category."""
    inputs: list[np.ndarray] = []
    features: list[_Feature] = []
    numbers: dict[int, np.ndarray] = {}
    for position, bins in enumerate(columns):
        if bins.kind in ("number", "date"):
            x = _as_numbers(cast(pd.Series, data[bins.column]), bins.kind)
            numbers[position] = x
            middle = float(np.nanmedian(x)) if (~np.isnan(x)).any() else 0.0
            inputs.append(np.where(np.isnan(x), middle, x))
            features.append(_Feature(position, bins.kind))
            continue
        for code in bins.paired:
            if bins.wheres[code] is None or bins.conditions[code][0].op == "is_null":
                continue
            inputs.append((bins.codes == code).astype(float))
            features.append(_Feature(position, "category", bins.labels[code]))
    matrix = np.column_stack(inputs) if inputs else np.zeros((len(data), 0))
    return matrix, features, numbers


def _leaf(
    path: list[tuple[_Feature, bool, float]], columns: list[_Bins], numbers: dict[int, np.ndarray]
) -> tuple[list[Condition], str, list[str], list[np.ndarray]]:
    """One leaf's path as conditions (bounds of one column merged), its sentence, and the rows
    each of its columns keeps (as SQL reads the conditions: an empty value never passes)."""
    order: list[int] = []
    lower: dict[int, float] = {}
    upper: dict[int, float] = {}
    equal: dict[int, str] = {}
    unequal: dict[int, list[str]] = {}
    for feature, left, threshold in path:
        if feature.column not in order:
            order.append(feature.column)
        if feature.kind == "category":
            label = cast(str, feature.value)
            if left:
                unequal.setdefault(feature.column, []).append(label)
            else:
                equal[feature.column] = label
        elif left:
            upper[feature.column] = min(upper.get(feature.column, math.inf), threshold)
        else:
            lower[feature.column] = max(lower.get(feature.column, -math.inf), threshold)
    conditions: list[Condition] = []
    wheres: list[str] = []
    masks: list[np.ndarray] = []
    for position in order:
        bins = columns[position]
        name = bins.column
        if bins.kind != "category":
            x = numbers[position]
            whole = bins.kind == "date" and bool(np.all(x[~np.isnan(x)] % DAY == 0))
            low, high = lower.get(position), upper.get(position)
            kept = ~np.isnan(x)
            if low is not None:
                conditions.append(Condition(column=name, op="gt", value=_bound(low, bins.kind)))
                kept &= x > low
            if high is not None:
                conditions.append(Condition(column=name, op="lte", value=_bound(high, bins.kind)))
                kept &= x <= high
            masks.append(kept)
            if low is not None and high is not None:
                shown = _show(low, bins.kind, whole), _show(high, bins.kind, whole)
                wheres.append(f"{shown[0]} < {name} ≤ {shown[1]}")
            elif low is not None:
                wheres.append(f"{name} > {_show(low, bins.kind, whole)}")
            elif high is not None:
                wheres.append(f"{name} ≤ {_show(high, bins.kind, whole)}")
            continue
        code_of = {label: code for code, label in enumerate(bins.labels)}
        made = bins.conditions
        if position in equal:
            code = code_of[equal[position]]
            conditions.extend(made[code])
            wheres.append(cast(str, bins.wheres[code]))
            masks.append(bins.codes == code)
            continue
        labels = unequal[position]
        codes = [code_of[label] for label in labels]
        missing = [c for c, m in enumerate(made) if m and m[0].op == "is_null"]
        masks.append(~np.isin(bins.codes, codes + missing))
        if made[codes[0]][0].op == "eq":  # a number with few values
            conditions += [made[code][0].model_copy(update={"negate": True}) for code in codes]
        else:
            conditions.append(Condition(column=name, op="in", values=labels, negate=True))
        wheres.append(
            f"{name} ≠ {labels[0]}"
            if len(labels) == 1
            else f"{name} is none of {', '.join(labels)}"
        )
    return conditions, " and ".join(wheres), [columns[p].column for p in order], masks


def _tree(
    columns: list[_Bins], data: pd.DataFrame, values: np.ndarray, minimum: int
) -> list[_Group]:
    """Nodes of a depth-3 tree (on the first 10 columns) fitted on the error of one half of the
    rows, tested on the other half: an honest test of what the tree's search found. Inner nodes
    count too, for when a leaf's last split adds nothing."""
    columns = columns[:PAIR_COLUMNS]
    x, features, numbers = _tree_inputs(columns, data)
    if not features or len(values) < 4 * MIN_ROWS:
        return []
    order = np.random.default_rng(SEED).permutation(len(values))
    halves = np.zeros(len(values), dtype=bool)
    halves[order[: len(values) // 2]] = True
    learn = np.sort(order[: min(len(values) // 2, TREE_ROWS)])
    share = len(learn) / len(values)
    model = DecisionTreeRegressor(
        max_depth=TREE_DEPTH, min_samples_leaf=max(1, int(minimum * share)), random_state=SEED
    )
    model.fit(x[learn], values[learn])
    tree = model.tree_
    left, right = tree.children_left, tree.children_right
    found: list[_Group] = []

    other = ~halves
    confirm = values[other]

    def add(path: list[tuple[_Feature, bool, float]]) -> None:
        conditions, where, names, masks = _leaf(path, columns, numbers)
        mask = np.logical_and.reduce(masks)
        rows = int(mask.sum())
        inside = mask & other
        if rows < minimum or len(values) - rows < MIN_ROWS or inside.sum() < 2:
            return
        p = _welch(
            np.array([float(inside.sum())]),
            np.array([float(values[inside].sum())]),
            np.array([float((values[inside] ** 2).sum())]),
            float(len(confirm)),
            float(confirm.sum()),
            float((confirm**2).sum()),
        )[0]
        mean = float(values[mask].mean())
        found.append(
            _Group((), conditions, where, names, rows, mean, float(p), mask, "tree", masks)
        )

    def walk(node: int, path: list[tuple[_Feature, bool, float]]) -> None:
        if path:  # every node but the root: a leaf, or a node whose split adds nothing
            add(path)
        if left[node] == -1:
            return
        feature = features[int(tree.feature[node])]
        threshold = float(tree.threshold[node])
        walk(int(left[node]), [*path, (feature, True, threshold)])
        walk(int(right[node]), [*path, (feature, False, threshold)])

    walk(0, [])
    return found


# --- The analysis --------------------------------------------------------------------------------


def _columns(
    data: pd.DataFrame, fields: list[Field], values: np.ndarray
) -> list[tuple[_Bins, float]]:
    """The columns cut into bins, those that matter most for the error first (η²), at most 30;
    ranked on a sample of 20 000 rows when there are more columns than that."""
    usable = [f for f in fields if f.name in data.columns]
    if len(usable) > MAX_COLUMNS and len(values) > RANKING_ROWS:
        sample = np.sort(
            np.random.default_rng(SEED).choice(len(values), RANKING_ROWS, replace=False)
        )
        ranked = []
        for f in usable:
            found = _bins(cast(pd.Series, data[f.name].iloc[sample]), f)
            if found is not None:
                ranked.append((f, _eta(found, values[sample])))
        ranked.sort(key=lambda item: -item[1])
        usable = [f for f, _ in ranked[:MAX_COLUMNS]]
    built = []
    for f in usable:
        found = _bins(cast(pd.Series, data[f.name]), f)
        if found is not None:
            built.append((found, _eta(found, values)))
    built.sort(key=lambda item: -item[1])
    return built[:MAX_COLUMNS]


def _sentence(where: str, share: float, error: float, overall: float, measure: str) -> str:
    head = f"When {where} ({_percent(share)} of rows),"
    if measure == "misclassified":
        return (
            f"{head} the model is wrong {_percent(error, 0)} of the time vs "
            f"{_percent(overall, 0)} overall."
        )
    if measure == "log_loss":
        return f"{head} the log-loss is {number(error)} vs {number(overall)} overall."
    return f"{head} the model is off by {number(error)} on average vs {number(overall)} overall."


def _adds_to_its_parts(
    group: _Group, means: dict[tuple[int, int], float], values: np.ndarray
) -> bool:
    """A subgroup of several conditions must be notably worse than what its conditions give
    without one of the columns (else "South and income ≥ 40,000" would repeat "South")."""
    if len(group.columns) < 2:
        return True
    if group.parts:
        return all(group.mean >= NOTABLY * means[part] for part in group.parts)
    masks = group.column_masks or []
    for left_out in range(len(masks)):
        rest = np.logical_and.reduce([m for i, m in enumerate(masks) if i != left_out])
        if rest.any() and group.mean < NOTABLY * float(values[rest].mean()):
            return False
    return True


def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    union = int((a | b).sum())
    return int((a & b).sum()) / union if union else 0.0


def _select(
    groups: list[_Group],
    columns: list[_Bins],
    keep: Callable[[_Group], bool],
    order: Callable[[_Group], float],
    limit: int,
) -> list[tuple[_Group, np.ndarray]]:
    """The first groups by `order` that pass `keep` and do not repeat one already chosen."""
    chosen: list[tuple[_Group, np.ndarray]] = []
    for group in sorted(groups, key=order):
        if not keep(group):
            continue
        mask = _mask_of(group, columns)
        if any(_jaccard(mask, other) >= OVERLAP for _, other in chosen):
            continue
        chosen.append((group, mask))
        if len(chosen) == limit:
            break
    return chosen


def _hints(
    slices: list[Slice],
    excluded: set[str],
    held: HeldOut,
    task: Task | str,
    target: str,
    positive: str | None,
) -> list[Hint]:
    """Simple, careful rules: what might help, never applied by the app."""
    found: list[Hint] = []
    small = [s for s in slices if s.share < SMALL_SHARE]
    for s in small[:2]:
        found.append(
            Hint(
                "few_rows",
                f"Few examples where {s.where} ({s.rows:,} rows): more data for this group "
                "would help.",
            )
        )
    named: set[str] = set()
    for s in slices:
        for column in s.columns:
            if column in excluded and column not in named:
                named.add(column)
                found.append(
                    Hint(
                        "excluded_column",
                        f"{column} was excluded by the preparation: try including it "
                        "(Preparation › Columns › Use).",
                        column,
                    )
                )
    if task == "regression":
        hint = _log_hint(held, target)
        if hint is not None:
            found.append(hint)
    else:
        found += _rare_class_hints(held, task, positive)
    return found


def _log_hint(held: HeldOut, target: str) -> Hint | None:
    """Errors that grow with the value: a log target (or a Poisson / Gamma regression) fits
    better. Read against the prediction, so that regression to the mean does not trigger it."""
    actual = held.actual.astype(float)
    predicted = held.predicted.astype(float)
    if len(actual) < MIN_ROWS or actual.min() < 0:
        return None
    errors = np.abs(predicted - actual)
    rho, p = stats.spearmanr(predicted, errors)
    rho, p = float(cast(float, rho)), float(cast(float, p))
    if not (math.isfinite(rho) and rho >= 0.3 and p < 0.01):
        return None
    low, high = np.quantile(predicted, [0.2, 0.8])
    bottom, top = errors[predicted <= low], errors[predicted >= high]
    if not len(bottom) or not len(top) or top.mean() < 2 * max(bottom.mean(), 1e-12):
        return None
    return Hint(
        "log_target",
        f"Errors grow with the value of {target} (×{top.mean() / max(bottom.mean(), 1e-12):.1f} "
        "from the lowest to the highest fifth of predictions): try predicting its log, or a "
        "Poisson or Gamma regression.",
        target,
    )


def _rare_class_hints(held: HeldOut, task: Task | str, positive: str | None) -> list[Hint]:
    actual = held.actual.astype(str)
    predicted = held.predicted.astype(str)
    found: list[Hint] = []
    counts = pd.Series(actual).value_counts(normalize=True)
    for label, share in counts.items():
        label = str(label)
        if share >= 0.2 or (task == "binary" and positive is not None and label != positive):
            continue
        rows = actual == label
        if rows.sum() < 5:
            continue
        recall = float((predicted[rows] == label).mean())
        if recall >= 0.5:
            continue
        what = (
            "class weights or a lower decision threshold" if task == "binary" else "class weights"
        )
        found.append(
            Hint(
                "rare_class",
                f"The rare class {label} ({_percent(float(share))} of rows) is often missed "
                f"(recall {_percent(recall, 0)}): try {what}.",
            )
        )
    return found[:3]


def analyse(
    data: pd.DataFrame,
    fields: list[Field],
    held: HeldOut,
    task: Task | str,
    measure: Measure | None = None,
    excluded: set[str] | None = None,
    target: str = "target",
    positive: str | None = None,
    source: str = "test rows",
    conditions: list[Condition] | None = None,
) -> ErrorAnalysis:
    """Where the model is wrong. `data`: the columns of the held-out rows, in the order of
    `held.rows`; `fields`: the columns to search (not the target); `excluded`: the columns the
    preparation left out (that the user may use anyway)."""
    offered = measures_for(task)
    measure = measure or offered[0]
    if measure not in offered:
        raise ValueError(f"{measure} is not a measure of this problem: {', '.join(offered)}.")
    values = row_errors(held, measure)
    n = len(values)
    overall = float(values.mean()) if n else 0.0
    excluded = excluded or set()
    severity = _severity(held)
    data = data.reset_index(drop=True)
    built = _columns(data, fields, values) if n else []
    columns = [bins for bins, _ in built]
    minimum = max(MIN_ROWS, math.ceil(MIN_SHARE * n))
    groups: list[_Group] = []
    means: dict[tuple[int, int], float] = {}
    if columns and overall > 0:
        groups, means = _grid(columns, values, minimum)
        groups += _tree(columns, data, values, minimum)
    slices: list[Slice] = []
    if groups:
        rejected, q_values, _, _ = multipletests([g.p_value for g in groups], ALPHA, "fdr_bh")
        q_of = {id(g): float(q) for g, q in zip(groups, q_values, strict=True)}
        significant = [g for g, r in zip(groups, rejected, strict=True) if r]

        def keep(group: _Group) -> bool:
            return group.mean >= NOTABLY * overall and _adds_to_its_parts(group, means, values)

        chosen = _select(
            significant, columns, keep, lambda g: -g.rows * (g.mean - overall), MAX_SLICES
        )
        for group, mask in chosen:
            inside = np.flatnonzero(mask)
            examples = inside[_worst_positions(severity[inside], EXAMPLES)]
            share = group.rows / n
            slices.append(
                Slice(
                    group.where,
                    group.conditions,
                    group.columns,
                    group.rows,
                    share,
                    group.mean,
                    group.mean / overall,
                    group.rows * (group.mean - overall),
                    group.p_value,
                    q_of[id(group)],
                    group.found_by,
                    _sentence(group.where, share, group.mean, overall, measure),
                    [_prediction(held, int(i)) for i in examples],
                )
            )
    by_column = [_column_errors(bins, values, bins.column in excluded) for bins in columns]
    by_column.sort(key=lambda c: -c.spread)
    return ErrorAnalysis(
        str(task),
        measure,
        list(offered),
        MEASURE_LABELS[measure],
        source,
        n,
        overall,
        len(columns),
        len(groups),
        slices,
        None if slices else "No subgroup is notably worse than average.",
        by_column,
        [_prediction(held, int(i)) for i in _worst_positions(severity, WORST)],
        _hints(slices, excluded, held, task, target, positive),
        list(conditions or []),
    )


def _column_errors(bins: _Bins, values: np.ndarray, excluded: bool) -> ColumnErrors:
    found = _stats(bins.codes, values, bins.size)
    means = found.mean()
    shown = [
        ColumnBin(label, int(found.rows[code]), float(means[code]) if found.rows[code] else None)
        for code, label in enumerate(bins.labels)
    ]
    solid = [b for b in shown if b.rows >= MIN_ROWS and b.error is not None]
    judged = solid or [b for b in shown if b.error is not None]
    if not judged:
        return ColumnErrors(bins.column, bins.kind, shown, 0.0, None, excluded)
    worst = max(judged, key=lambda b: cast(float, b.error))
    best = min(judged, key=lambda b: cast(float, b.error))
    spread = cast(float, worst.error) - cast(float, best.error)
    return ColumnErrors(bins.column, bins.kind, shown, spread, worst.label, excluded)


# --- Two runs compared ---------------------------------------------------------------------------


def compare(
    data: pd.DataFrame,
    fields: list[Field],
    first: HeldOut,
    second: HeldOut,
    task: Task | str,
    measure: Measure | None = None,
) -> ErrorComparison:
    """Subgroups where the second run's error changes notably more than on average: where it
    fixes the first run's errors, and where it makes them worse. `data`: the columns of the first
    run's held-out rows, in its order."""
    offered = measures_for(task)
    measure = measure or offered[0]
    if measure not in offered:
        raise ValueError(f"{measure} is not a measure of this problem: {', '.join(offered)}.")
    _, at_first, at_second = np.intersect1d(first.rows, second.rows, return_indices=True)
    a = row_errors(first, measure)[at_first]
    b = row_errors(second, measure)[at_second]
    change = b - a
    n = len(change)
    label = MEASURE_LABELS[measure]
    if n < 2 * MIN_ROWS:
        first_mean = float(a.mean()) if n else 0.0
        second_mean = float(b.mean()) if n else 0.0
        message = "Too few rows predicted by both runs to compare them."
        return ErrorComparison(measure, label, n, first_mean, second_mean, [], [], message)
    aligned = data.reset_index(drop=True).iloc[at_first].reset_index(drop=True)
    columns = [bins for bins, _ in _columns(aligned, fields, change)]
    minimum = max(MIN_ROWS, math.ceil(MIN_SHARE * n))
    groups, means = _grid(columns, change, minimum, two_sided=True) if columns else ([], {})
    average = float(change.mean())
    smallest = CHANGE * max(float(a.mean()), 1e-12)
    better: list[ChangedSlice] = []
    worse: list[ChangedSlice] = []
    if groups:
        rejected, q_values, _, _ = multipletests([g.p_value for g in groups], ALPHA, "fdr_bh")
        q_of = {id(g): float(q) for g, q in zip(groups, q_values, strict=True)}
        significant = [g for g, r in zip(groups, rejected, strict=True) if r]

        def adds(group: _Group) -> bool:
            """A pair must change more than each of its two bins alone."""
            if len(group.parts) < 2:
                return True
            return all(abs(group.mean - means[part]) >= smallest for part in group.parts)

        def made(group: _Group, mask: np.ndarray) -> ChangedSlice:
            share = group.rows / n
            before, after = float(a[mask].mean()), float(b[mask].mean())
            relative = f" ({(after - before) / before * 100:+.0f} %)" if before > 0 else ""
            return ChangedSlice(
                group.where,
                group.conditions,
                group.rows,
                share,
                before,
                after,
                group.p_value,
                q_of[id(group)],
                f"When {group.where} ({_percent(share)} of rows), the {label} goes from "
                f"{_shown(before, measure)} to {_shown(after, measure)}{relative}.",
            )

        for sign, out in ((-1, better), (1, worse)):
            chosen = _select(
                significant,
                columns,
                lambda g, sign=sign: sign * (g.mean - average) >= smallest and adds(g),
                lambda g, sign=sign: -sign * g.rows * (g.mean - average),
                MAX_CHANGED,
            )
            out.extend(made(group, mask) for group, mask in chosen)
    return ErrorComparison(
        measure,
        label,
        n,
        float(a.mean()),
        float(b.mean()),
        better,
        worse,
        None if better or worse else "No subgroup changes notably more than the rest.",
    )


def _shown(value: float, measure: str) -> str:
    return _percent(value, 0) if measure == "misclassified" else number(value)


# --- Fairness ------------------------------------------------------------------------------------


FAIRNESS_LABELS = {
    "mae": "Mean absolute error",
    "mean_predicted": "Mean prediction",
    "mean_actual": "Mean actual value",
    "error_rate": "Error rate",
    "positive_rate": "Predicted positive rate",
    "recall": "Recall",
}


def fairness(
    values: pd.Series, field_: Field, held: HeldOut, task: Task | str, positive: str | None = None
) -> Fairness:
    """Metrics side by side for each group of a sensitive column (its categories, or the
    quintiles of a number), with the largest gap. `values`: the column on the held-out rows."""
    bins = _bins(values.reset_index(drop=True), field_)
    if bins is None:
        raise ValueError(f"{field_.name} cannot be cut into groups (an identifier or free text).")
    if task == "regression":
        metrics = ["mae", "mean_predicted", "mean_actual"]
    elif task == "binary":
        metrics = ["error_rate", "positive_rate", "recall"]
    else:
        metrics = ["error_rate", "recall"]
    if task == "binary" and positive is None and held.classes:
        counts = pd.Series(held.actual.astype(str)).value_counts()
        positive = str(counts.index[-1])
    groups: list[GroupMetrics] = []
    for code, label in enumerate(bins.labels):
        rows = bins.codes == code
        count = int(rows.sum())
        if not count:
            continue
        groups.append(
            GroupMetrics(label, count, _group_metrics(held, rows, task, positive), count < MIN_ROWS)
        )
    gaps: dict[str, float | None] = {}
    for metric in metrics:
        seen = [g.values[metric] for g in groups if not g.few and g.values[metric] is not None]
        numbers = [cast(float, v) for v in seen]
        gaps[metric] = max(numbers) - min(numbers) if len(numbers) >= 2 else None
    return Fairness(
        field_.name,
        metrics,
        {m: FAIRNESS_LABELS[m] for m in metrics},
        groups[: GROUPS + 2],
        gaps,
        positive if task == "binary" else None,
    )


def _group_metrics(
    held: HeldOut, rows: np.ndarray, task: Task | str, positive: str | None
) -> dict[str, float | None]:
    if task == "regression":
        actual, predicted = held.actual.astype(float)[rows], held.predicted.astype(float)[rows]
        return {
            "mae": float(np.abs(predicted - actual).mean()),
            "mean_predicted": float(predicted.mean()),
            "mean_actual": float(actual.mean()),
        }
    actual, predicted = held.actual.astype(str)[rows], held.predicted.astype(str)[rows]
    found: dict[str, float | None] = {"error_rate": float((actual != predicted).mean())}
    if task == "binary":
        found["positive_rate"] = float((predicted == positive).mean())
        truly = actual == positive
        found["recall"] = float((predicted[truly] == positive).mean()) if truly.any() else None
    else:
        recalls = [
            float((predicted[actual == label] == label).mean()) for label in np.unique(actual)
        ]
        found["recall"] = float(np.mean(recalls)) if recalls else None
    return found
