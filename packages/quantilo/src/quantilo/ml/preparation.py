# ruff: noqa: RUF001, RUF002, RUF003  (−, ρ, · are the app's symbols)
"""Automatic data preparation by named rules (docs/features/017-ml-preparation-donnees/spec.md).

Each decision comes from the column's profile (validated type, role, statistics) and the model's
profile (does it need scaling? handle missing values?). Every decision carries the id of the rule
that made it and a sentence with the numbers behind it. The user's choices (`Choices`) are applied
on top of the rules: per column, or globally; a choice the column or the model cannot take is
refused with its reason and the automatic one is kept. The scikit-learn pipeline is built from the
decisions, so everything that learns from data (medians, scales, encodings, selection) is fitted on
the training rows only.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    StandardScaler,
    TargetEncoder,
)

from quantilo.ml import codes
from quantilo.ml import embed as embedding
from quantilo.ml import encoding as enc
from quantilo.ml import text as textual
from quantilo.ml.data import CATEGORICAL, NUMERIC
from quantilo.ml.data import Field as DataField
from quantilo.ml.steps import Clipper, JointImputer, RobustScale, Selector
from quantilo.ml.target import Task

TARGET_ENCODING_LIMIT = 100
MISSING = "Missing"
DISTANCE_CATEGORIES = 20  # one-hot columns kept per category column for distances (032)
DISTANCE_EMPTY = 0.95  # a column this empty is left out of distances
LAGS = ("past_mean", "past_ewm")  # causal smoothed lags (R-LAG-01)
# numbers that go round (R-DATE-04): role → (cycle length, first value), by the range of values
CYCLES: dict[str, list[tuple[float, float, float, float]]] = {
    # role: [(lowest, highest, period, start), …] — the first range that holds every value
    "hour": [(0, 23, 24, 0)],
    "month": [(1, 12, 12, 1)],
    "weekday": [(0, 6, 7, 0), (1, 7, 7, 1)],
    "angle": [(0, 360, 360, 0)],
}
NOT_CYCLIC = (
    "Only for a column that goes round: hour 0 to 23, month 1 to 12, weekday, angle 0 to 360 "
    "(its role in Columns)."
)
NEEDS_ORDER = "Needs ordered rows: choose panel walk-forward or a time-based split in Validation."

# --- The user's choices --------------------------------------------------------------------------

NumericImpute = Literal["auto", "mean", "median", "constant", "knn", "iterative"]
Outliers = Literal["none", "clip_percentiles", "clip_iqr", "drop_rows"]
Imbalance = Literal["auto", "none", "weights", "oversample", "undersample"]
SelectionMethod = Literal[
    "variance", "correlation", "vif", "mutual_info", "univariate", "l1", "rfe", "permutation"
]
CHOICE_KEYS = ("impute", "indicator", "transform", "encoding", "scaling")


class ColumnChoice(BaseModel):
    """What the user set for one column; None = automatic."""

    use: bool | None = None
    impute: str | None = Field(default=None, max_length=32)
    indicator: bool | None = None
    transform: str | None = Field(default=None, max_length=32)
    encoding: str | None = Field(default=None, max_length=32)
    scaling: str | None = Field(default=None, max_length=32)


class SelectionChoice(BaseModel):
    method: SelectionMethod
    value: float | None = Field(default=None, ge=0, le=1_000)  # threshold or k; None = default


class Choices(BaseModel):
    """Everything the preparation editor can change. The defaults are the automatic rules."""

    columns: dict[str, ColumnChoice] = Field(default_factory=dict, max_length=5_000)
    numeric_impute: NumericImpute = "auto"
    indicator_share: float = Field(default=0.05, ge=0, le=1)
    rare_share: float = Field(default=0.01, ge=0, le=0.5)
    rare_rows: int = Field(default=10, ge=1, le=100_000)
    one_hot_limit: int = Field(default=15, ge=2, le=200)
    target_smoothing: float = Field(default=10.0, ge=0, le=10_000)
    # 063 bis (R-LAG-03): the target's own past, shifted, as one more input (ordered rows only)
    target_lag: Literal["none", "past_mean", "past_ewm"] = "none"
    outliers: Outliers = "none"
    imbalance: Imbalance = "auto"
    selection: list[SelectionChoice] = Field(default_factory=list, max_length=8)


GLOBAL_DEFAULTS = Choices().model_dump(exclude={"columns"})


@dataclass(frozen=True)
class ModelProfile:
    """What a model needs from its inputs (017 spec, « Profil du modèle »)."""

    needs_scaling: bool = False
    handles_missing: bool = False
    native_categorical: bool = False
    sensitive_to_skew: bool = False
    sensitive_to_collinearity: bool = False
    tree_based: bool = False
    nonnegative_inputs: bool = False
    # Distances between rows (segments and unusual rows, 032 / 033): no target, robust scales on
    # every number, one-hot categories (at most DISTANCE_CATEGORIES columns each).
    distance: bool = False


DISTANCE = ModelProfile(needs_scaling=True, sensitive_to_skew=True, distance=True)


@dataclass
class ColumnProfile:
    name: str
    type: str
    role: str | None
    missing_share: float
    distinct: int
    dominant_share: float  # share of the most frequent value
    skewness: float | None
    outlier_share: float | None  # beyond 1.5 IQR
    all_nonnegative: bool | None
    distinct_share: float  # distinct values / non-missing rows
    text: textual.TextProfile | None = None  # free text: its texts and words (052)
    median_rows: float = 0.0  # categories: rows of the median category
    minimum: float | None = None  # numbers
    maximum: float | None = None
    integer: bool = False  # numbers: every value is a whole number
    span_days: float = 0.0  # dates: from the first to the last
    month_days: int = 0  # dates: how many different days of the month
    step_days: float = 0.0  # dates: median gap between two successive distinct dates


@dataclass
class Option:
    value: str
    reason: str | None = None  # why it cannot be chosen here; None = available


@dataclass
class Decision:
    column: str
    type: str
    use: bool
    rules: list[str] = field(default_factory=list)  # rule ids, in the order they applied
    why: list[str] = field(default_factory=list)  # one sentence per rule
    impute: str | None = None  # median, mean, constant, knn, iterative, most_frequent, …
    indicator: bool = False
    transform: str | None = None  # log1p, yeojohnson, date_parts
    encoding: str | None = None  # binary, onehot, target, ordinal, native
    scaling: str | None = None  # standard, robust, minmax
    order: list[str] | None = None  # categories of an ordinal column
    auto: dict[str, str] = field(default_factory=dict)  # what the rules chose ("none" = nothing)
    manual: list[str] = field(default_factory=list)  # what the user changed
    options: dict[str, list[Option]] = field(default_factory=dict)
    reactivable: bool = True  # an excluded column the user may use anyway
    leak: bool = False  # excluded because it seems to contain the answer
    missing: float = 0.0  # share of empty values
    cycle: list[float] | None = None  # a number that goes round: [period, first value] (063)

    def apply(self, rule: str, sentence: str) -> None:
        self.rules.append(rule)
        self.why.append(sentence)


@dataclass
class SelectionReport:
    method: str
    value: float | None
    kept: int
    of: int


@dataclass
class Balance:
    strategy: str  # none, weights, oversample, undersample
    rule: str  # R-IMB-01/02/03, or USER
    why: str
    before: dict[str, int]
    after: dict[str, int]


@dataclass
class Order:
    """Ordered rows (panel walk-forward or time-based split): what the lags follow."""

    period: str
    entity: str | None
    shift: int  # observations of the entity: 1 + the embargo
    window: int
    grain: str  # day, week, month, quarter, year, period


@dataclass
class Preparation:
    decisions: list[Decision]
    inputs: int = 0  # model inputs once prepared (filled after fitting)
    excluded: int = 0
    choices: Choices = field(default_factory=Choices)
    custom: bool = False  # the user changed something
    warnings: list[str] = field(default_factory=list)
    selection: list[SelectionReport] = field(default_factory=list)
    balance: Balance | None = None
    dropped_rows: int = 0  # training rows removed as extreme outliers
    distance: bool = False  # prepared for distances between rows (no target)
    folds: enc.TargetFolds | None = None  # how target encodings are cross-fitted (R-CAT-08)
    order: "Order | None" = None  # ordered rows: what the lags follow (R-LAG-01)
    target: str | None = None  # 063 bis: the target, read by its own lag (R-LAG-03)

    def __post_init__(self) -> None:
        # each row's fold key, set by the training (not a field: never sent nor saved)
        self.fold_keys: pd.Series | None = None

    @property
    def used(self) -> list[Decision]:
        return [d for d in self.decisions if d.use]


def _col(frame: pd.DataFrame, name: str) -> pd.Series:
    """One column of the frame, typed as a series."""
    return cast(pd.Series, frame[name])


# --- Profiles ------------------------------------------------------------------------------------


def profile(series: pd.Series, fld: DataField) -> ColumnProfile:
    n = len(series)
    present = series.dropna()
    missing = 1 - len(present) / n if n else 0.0
    # numbers are counted as they are: as text, 200 000 values take 0.3 s per column
    labels = present if fld.type in NUMERIC else present.astype(str)
    counts = labels.value_counts() if len(present) else pd.Series(dtype=int)
    distinct = int(counts.size)
    dominant = float(counts.iloc[0] / len(present)) if distinct else 1.0
    skewness = outliers = nonnegative = minimum = maximum = None
    integer = False
    span_days, month_days, step_days = 0.0, 0, 0.0
    if fld.type in NUMERIC and len(present):
        numbers = present.to_numpy(dtype=float)
        minimum, maximum = float(numbers.min()), float(numbers.max())
        integer = bool(np.all(np.mod(numbers, 1) == 0))
    if fld.type == "datetime" and len(present):
        moments = cast(pd.Series, pd.to_datetime(present, errors="coerce")).dropna()
        if len(moments):
            span_days = float((moments.max() - moments.min()).total_seconds() / 86_400)
            month_days = int(moments.dt.day.nunique())
            gaps = pd.Series(moments.unique()).sort_values().diff().dropna()
            step_days = float(gaps.dt.total_seconds().median() / 86_400) if len(gaps) else 0.0
    if fld.type in NUMERIC and len(present) > 2:
        values = present.to_numpy(dtype=float)
        if np.ptp(values) > 0:
            skewness = float(stats.skew(values))
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        outliers = (
            float(np.mean((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)))
            if iqr > 0
            else 0.0
        )
        nonnegative = bool(values.min() >= 0)
    return ColumnProfile(
        fld.name,
        fld.type,
        fld.role,
        missing,
        distinct,
        dominant,
        skewness,
        outliers,
        nonnegative,
        distinct / len(present) if len(present) else 0.0,
        textual.profile(series) if fld.type == "text" else None,
        float(counts.median()) if distinct and fld.type in CATEGORICAL else 0.0,
        minimum,
        maximum,
        integer,
        span_days,
        month_days,
        step_days,
    )


def _association(
    feature: pd.Series, target: pd.Series, feature_numeric: bool, target_numeric: bool
) -> float:
    """|r| (numbers), Cramér's V (categories) or the correlation ratio η (mixed)."""
    pair = pd.DataFrame({"x": feature, "y": target}).dropna()
    if len(pair) < 10:
        return 0.0
    x, y = cast(pd.Series, pair["x"]), cast(pd.Series, pair["y"])
    if feature_numeric and target_numeric:
        if x.nunique() < 2 or y.nunique() < 2:  # a constant: no correlation
            return 0.0
        return abs(float(np.corrcoef(x, y)[0, 1]))
    if not feature_numeric and not target_numeric:
        table = pd.crosstab(x.astype(str), y.astype(str))
        if min(table.shape) < 2:
            return 0.0
        chi2 = float(np.asarray(stats.chi2_contingency(table, correction=False)[0], dtype=float))
        return float(math.sqrt(chi2 / (len(pair) * (min(table.shape) - 1))))
    numbers, groups = (x, y) if feature_numeric else (y, x)
    mean = float(numbers.mean())
    total = float(((numbers - mean) ** 2).sum())
    if total == 0:
        return 0.0
    grouped = numbers.groupby(groups.astype(str)).agg(["count", "mean"])
    counts, means = cast(pd.Series, grouped["count"]), cast(pd.Series, grouped["mean"])
    between = float((counts * (means - mean) ** 2).sum())
    return math.sqrt(between / total)


def _separation(feature: pd.Series, target: pd.Series) -> float:
    """How well a single threshold on a number separates two classes (AUC, either direction)."""
    pair = pd.DataFrame({"x": feature, "y": target.astype(str)}).dropna()
    x, y = cast(pd.Series, pair["x"]), cast(pd.Series, pair["y"])
    if len(pair) < 10 or y.nunique() != 2 or x.nunique() < 2:
        return 0.0
    positive = pair["y"] == pair["y"].iloc[0]
    ranks = stats.rankdata(pair["x"])
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    auc = (ranks[positive.to_numpy()].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(max(auc, 1 - auc))


# --- Rules ---------------------------------------------------------------------------------------

USABLE = NUMERIC | CATEGORICAL | {"datetime"}
NAMES = {
    "mean": "mean",
    "median": "median",
    "constant": "0",
    "knn": "KNN imputer (5 nearest rows)",
    "iterative": "iterative imputer (each column predicted from the others)",
}


def decide(
    frame: pd.DataFrame,
    fields: list[DataField],
    target: DataField,
    task: Task,
    model: ModelProfile,
    excluded_columns: set[str] | None = None,
    choices: Choices | None = None,
    source_numeric: bool = False,
    panel_columns: dict[str, str] | None = None,
    folds: enc.TargetFolds | None = None,
    order: Order | None = None,
) -> Preparation:
    """One decision per column: the rules of the 017 spec, then the user's choices.
    `panel_columns`: the period and entity of a panel (035), {column: "period" | "entity"};
    `folds`: how target encodings are cross-fitted (R-CAT-08, random by default); `order`: the
    rows are ordered (panel or time split), so lags are possible (R-LAG-01)."""
    folds = folds or enc.target_folds("split", None, None, None)
    choices = choices or Choices()
    decisions: list[Decision] = []
    warnings: list[str] = []
    forced: set[str] = set()
    excluded_columns = excluded_columns or set()
    profiles: dict[str, ColumnProfile] = {}
    for fld in fields:
        prof = profile(_col(frame, fld.name), fld)
        profiles[fld.name] = prof

        def exclude(decision: Decision, fld: DataField = fld, prof: ColumnProfile = prof) -> None:
            _exclusion(
                decision,
                fld,
                prof,
                frame,
                target,
                task,
                model,
                excluded_columns,
                source_numeric,
                (panel_columns or {}).get(fld.name),
            )

        decision = _column(fld, prof, model, choices, exclude, warnings, folds, order)
        decisions.append(decision)
        choice = choices.columns.get(fld.name)
        if decision.use and choice and choice.use is True:
            forced.add(fld.name)
    _text_budget(decisions, profiles, forced)
    if task == "regression" or source_numeric or task == "binary":
        _identities(frame, decisions, target, task == "regression" or source_numeric, forced)
    if model.sensitive_to_collinearity:
        _collinearity(frame, decisions, target, task == "regression" or source_numeric, forced)
    custom = choices.model_dump(exclude={"columns"}) != GLOBAL_DEFAULTS or any(
        d.manual for d in decisions
    )
    if choices.target_lag != "none":
        warnings.append(target_lag_sentence(choices.target_lag, target.name, order))
    return Preparation(
        decisions,
        excluded=sum(not d.use for d in decisions if d.column != target.name),
        choices=choices,
        custom=custom,
        warnings=warnings,
        folds=folds,
        order=order,
        target=target.name,
    )


def target_lag_sentence(kind: str, target: str, order: "Order | None") -> str:
    """R-LAG-03 said in the preparation: what the target's lag reads, or why it is refused."""
    if order is None:
        return (
            "R-LAG-02 · Past values of the target need ordered rows: choose panel walk-forward "
            "or a time-based split in Validation. Not applied."
        )
    what = "mean" if kind == "past_mean" else "exponentially weighted mean"
    per = f", per {order.entity}" if order.entity else ""
    unit = "previous periods" if order.grain == "period" else f"previous {order.grain}s"
    return (
        f"R-LAG-03 · Adds the {what} of the {order.window} {unit} of {target}{per}, shifted by "
        f"{order.shift}: each row reads the target of periods already known, never its own. The "
        "held-out scores are then one-step-ahead, and new data needs the target column, filled "
        "for the past periods and empty for the ones to predict."
    )


def _column(
    fld: DataField,
    prof: ColumnProfile,
    model: ModelProfile,
    choices: Choices,
    exclude: Callable[[Decision], None],
    warnings: list[str],
    folds: enc.TargetFolds | None = None,
    order: Order | None = None,
) -> Decision:
    """One column: is it left out (`exclude`), then the rules of its type, then the user's
    choices on top, then what the chosen encoding needs said (R-CAT-08, R-LAG-01)."""
    decision = Decision(fld.name, fld.type, True)
    choice = choices.columns.get(fld.name)
    decision.missing = prof.missing_share
    exclude(decision)
    decision.auto["use"] = "yes" if decision.use else "no"
    wants = choice.use if choice else None
    if not decision.use and wants is True:
        if decision.reactivable:
            decision.use = True
            decision.manual.append("use")
            decision.apply("USER", "Used anyway: your choice.")
        else:
            warnings.append(f"{fld.name} cannot be used: {decision.why[0]}")
    if decision.use:
        _type_rules(decision, fld, prof, model, choices)
        _snapshot(decision)
        if wants is False:
            decision.use = False
            decision.manual.append("use")
            decision.apply("USER", "Left out: your choice.")
        elif choice:
            _apply(decision, choice, prof, model, warnings, order is not None)
    if decision.use:
        _encoding_notes(decision, folds, order)
    decision.options = _options(decision, prof, model, order is not None)
    return decision


def _encoding_notes(decision: Decision, folds: enc.TargetFolds | None, order: Order | None) -> None:
    """R-CAT-08: which folds a target encoding is cross-fitted on; R-LAG-01: what a lag reads."""
    if decision.encoding == "target" and folds is not None:
        decision.apply("R-CAT-08", folds.why)
    if decision.transform in LAGS and order is not None:
        what = "mean" if decision.transform == "past_mean" else "exponentially weighted mean"
        whose = f"per {order.entity}" if order.entity else f"over the rows of each {order.period}"
        decision.apply(
            "R-LAG-01",
            f"Adds the {what} of the {order.window} previous {order.grain}s of "
            f"{decision.column}, {whose}, shifted by {order.shift} "
            f"(1 + embargo {order.shift - 1}): only values known before the row's period, "
            "never the target.",
        )


def decide_distance(
    frame: pd.DataFrame, fields: list[DataField], choices: Choices | None = None
) -> Preparation:
    """The same rules for distances between rows (segments and unusual rows, 032 / 033): no
    target, so no leakage check; identifiers, free text, raw dates, constants and nearly empty
    columns are left out; every number is scaled robustly and every category one-hot encoded."""
    choices = choices or Choices()
    decisions: list[Decision] = []
    warnings: list[str] = []
    for fld in fields:
        prof = profile(_col(frame, fld.name), fld)

        def exclude(decision: Decision, fld: DataField = fld, prof: ColumnProfile = prof) -> None:
            _distance_exclusion(decision, fld, prof, frame)

        decisions.append(_column(fld, prof, DISTANCE, choices, exclude, warnings))
    forced = {
        name for name, choice in choices.columns.items() if choice.use is True
    }  # a twin the user asked for stays
    _twins(frame, decisions, forced)
    custom = choices.model_dump(exclude={"columns"}) != GLOBAL_DEFAULTS or any(
        d.manual for d in decisions
    )
    return Preparation(
        decisions,
        excluded=sum(not d.use for d in decisions),
        choices=choices,
        custom=custom,
        warnings=warnings,
        distance=True,
    )


def _distance_exclusion(
    decision: Decision, fld: DataField, prof: ColumnProfile, frame: pd.DataFrame
) -> None:
    """R-SEL-02 to R-SEL-04, R-DIST-03, R-DIST-04, R-DIST-06 and R-DIST-07: what a distance
    between rows leaves out."""

    def out(rule: str, sentence: str, reactivable: bool = True) -> None:
        decision.use, decision.reactivable = False, reactivable
        decision.apply(rule, sentence)

    if fld.type == "identifier":
        return out(
            "R-SEL-02", f"Identifier ({prof.distinct:,} distinct values).", reactivable=False
        )
    if fld.type in CATEGORICAL and (
        (prof.distinct_share > 0.5 and prof.distinct > 20) or prof.distinct > TARGET_ENCODING_LIMIT
    ):
        return out("R-SEL-02", f"Behaves like an identifier ({prof.distinct:,} distinct values).")
    if fld.type == "text":
        return out(
            "R-SEL-03",
            "Free text is left out: a distance between rows does not read text.",
            reactivable=False,
        )
    if fld.type == "datetime":
        return out("R-DIST-03", "Raw date: it says when, not what kind of row.")
    if fld.type not in USABLE:
        return out("R-SEL-03", "This type is not used here.", reactivable=False)
    if prof.distinct <= 1 or prof.dominant_share > 0.99:
        return out("R-SEL-04", f"Almost always the same value ({prof.dominant_share:.1%}).")
    if prof.missing_share > DISTANCE_EMPTY:
        return out("R-DIST-04", f"{prof.missing_share:.0%} empty (more than 95 %).")
    if codes.flag_word(fld.name):
        return out(
            "R-DIST-06",
            "A quality flag or an attribute of another column: it says how a value was "
            "recorded, not what the row is. Use it anyway if it matters.",
        )
    if fld.type in NUMERIC and prof.integer:
        word = codes.code_word(fld.name)
        values = _col(frame, fld.name).to_numpy(dtype=float)
        if word is not None or codes.sequence(values, _row_order(frame)) is not None:
            said = f"Its name says a code ({word})" if word else "A row or sequence number"
            return out(
                "R-DIST-07",
                f"{said}: a distance would read the code as an amount. Check its type in "
                "Columns, or use it anyway if it is a real quantity.",
            )
    return None


TWIN_ROWS = 20_000  # rows compared to find two columns with the same content
MISSING_KEY = "<empty cell>"  # an empty cell, for the comparison


def _same_content(
    sample: pd.DataFrame,
    first: Decision,
    second: Decision,
    labels: dict[str, pd.Series],
    distinct: int,
) -> bool:
    """The same numbers row by row; or categories (at most one value per two rows) in
    one-to-one correspondence."""
    if (first.type in NUMERIC) != (second.type in NUMERIC):
        return False
    if first.type in NUMERIC:
        a = _col(sample, first.column).to_numpy(dtype=float)
        b = _col(sample, second.column).to_numpy(dtype=float)
        return bool(np.array_equal(a, b, equal_nan=True))
    if distinct > 0.5 * len(sample):
        return False
    pair = pd.DataFrame({"a": labels[first.column], "b": labels[second.column]})
    return int(pair.drop_duplicates().shape[0]) == distinct


def _twins(frame: pd.DataFrame, decisions: list[Decision], forced: set[str]) -> None:
    """R-DIST-05 (094, UNU-2): two columns with the same content, row by row (the same numbers,
    or categories in one-to-one correspondence: borough and park_borough), count twice in a
    distance and give the same reason twice: the second is left out, reactivable."""
    sample = frame.iloc[:TWIN_ROWS]
    used = [d for d in decisions if d.use and d.type in USABLE]
    keys: dict[str, tuple[int, int]] = {}
    labels: dict[str, pd.Series] = {}
    for d in used:
        column = _col(sample, d.column)
        labels[d.column] = column.astype("string").fillna(MISSING_KEY)
        keys[d.column] = (int(labels[d.column].nunique()), int(column.isna().sum()))
    for i, first in enumerate(used):
        if not first.use:
            continue
        for second in used[i + 1 :]:
            if (
                not second.use
                or second.column in forced
                or keys[first.column] != keys[second.column]
            ):
                continue
            if not _same_content(sample, first, second, labels, keys[first.column][0]):
                continue
            second.use, second.reactivable = False, True
            second.auto["use"] = "no"
            second.apply(
                "R-DIST-05",
                f"Same content as {first.column}, row by row: counted once (it would give the "
                "same reason twice).",
            )


def _exclusion(
    decision: Decision,
    fld: DataField,
    prof: ColumnProfile,
    frame: pd.DataFrame,
    target: DataField,
    task: Task,
    model: ModelProfile,
    excluded_columns: set[str],
    source_numeric: bool = False,
    panel_role: str | None = None,
) -> None:
    """R-SEL-01 to R-SEL-06, R-SEL-08 and R-SEL-09 (R-SEL-03: too few texts, 052): is the
    column left out, and may it come back? With a number predicted as classes (018), leakage
    is measured against the number."""

    def out(rule: str, sentence: str, reactivable: bool = True, leak: bool = False) -> None:
        decision.use, decision.reactivable, decision.leak = False, reactivable, leak
        decision.apply(rule, sentence)

    if fld.name == target.name:
        return out("R-SEL-01", "This is the target.", reactivable=False)
    if fld.name in excluded_columns:
        return out("R-SEL-08", "Ignored in the Columns page.", reactivable=False)
    if panel_role == "period":
        return out(
            "R-SEL-09",
            "The period of the panel: each model is judged on periods it never saw.",
            reactivable=False,
        )
    if panel_role == "entity":
        return out(
            "R-SEL-09",
            "The entity of the panel: an id learnt by heart does not carry over to new periods.",
            reactivable=False,
        )
    if fld.type == "identifier":
        return out(
            "R-SEL-02",
            f"Behaves like an identifier ({prof.distinct:,} distinct values).",
            reactivable=False,
        )
    if fld.type in CATEGORICAL and prof.distinct_share > 0.5 and prof.distinct > 20:
        # beyond TARGET_ENCODING_LIMIT categories without being an id: frequency (R-CAT-09)
        return out("R-SEL-02", f"Behaves like an identifier ({prof.distinct:,} distinct values).")
    if fld.type in NUMERIC and prof.integer:
        word = codes.code_word(fld.name)
        if word is not None:
            return out(
                "R-SEL-10",
                f"Its name says a code or an identifier ({word}) and its values are whole "
                "numbers: a model would read the code as an amount. Check its type in Columns, "
                "or use it anyway if it is a real quantity.",
            )
        rho = codes.sequence(_col(frame, fld.name).to_numpy(dtype=float), _row_order(frame))
        if rho is not None:
            return out(
                "R-SEL-11",
                f"Whole numbers rising with the row order (ρ = {rho:.2f}), {prof.distinct:,} "
                "distinct values: a row or sequence number, not a measure. Use it anyway if it "
                "is one.",
            )
    if fld.type == "text" and prof.text is not None and prof.text.texts < textual.MIN_TEXTS:
        return out(
            "R-SEL-03",
            f"Only {prof.text.texts} non-empty texts (at least {textual.MIN_TEXTS} needed): "
            "no vocabulary emerges.",
            reactivable=False,
        )
    if fld.type not in USABLE | {"text"}:
        return out("R-SEL-03", "This type is not used by models yet.", reactivable=False)
    if prof.distinct <= 1 or prof.dominant_share > 0.99:
        return out("R-SEL-04", f"Almost always the same value ({prof.dominant_share:.1%}).")
    if prof.missing_share > 0.8:
        return out("R-SEL-05", f"{prof.missing_share:.0%} empty.")
    if fld.type in NUMERIC | CATEGORICAL:
        strength = _association(
            _col(frame, fld.name),
            _col(frame, target.name),
            fld.type in NUMERIC,
            task == "regression" or source_numeric,
        )
        if strength >= 0.98:
            return out(
                "R-SEL-06",
                f"Looks like it contains the answer (association {strength:.3f} with the target).",
                leak=True,
            )
        separation = (
            _separation(_col(frame, fld.name), _col(frame, target.name))
            if task == "binary" and fld.type in NUMERIC and not source_numeric
            else 0.0
        )
        if separation >= 0.995:
            return out(
                "R-SEL-06",
                "Looks like it contains the answer: one threshold separates the classes "
                f"(AUC {separation:.3f}).",
                leak=True,
            )
    return None


def _row_order(frame: pd.DataFrame) -> np.ndarray:
    """Each row's position in the file ("__row"), else in the frame."""
    if "__row" in frame.columns:
        return _col(frame, "__row").to_numpy(dtype=float)
    return np.arange(len(frame), dtype=float)


# --- R-SEL-06 (094): the answer hidden in a sum, a difference or a linear function -------------

IDENTITY_ROWS = 5_000  # rows the search reads (a reproducible sample beyond)
IDENTITY_TERMS = 3  # columns in one identity, at most
IDENTITY_FOUND = 4  # identities looked for per target, at most
IDENTITY_STARTS = 5  # forward searches, each from one of the columns closest to the target
IDENTITY_MIN_ROWS = 30
EXACT_SHARE = 0.9  # a number: the identity holds on this share of the rows…
EXACT_INFORMATIVE = 0.8  # …and of the rows where the target is not its most frequent value
SEPARATES = 0.995  # two classes: AUC of the combination


def _auc(score: np.ndarray, positive: np.ndarray) -> float:
    """AUC of a score for two classes, either direction."""
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if not n_pos or not n_neg:
        return 0.0
    ranks = stats.rankdata(score)
    auc = (ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(max(auc, 1 - auc))


def _combination(names: list[str], signs: np.ndarray) -> str:
    """'a + b − c' from the names and the signs of their coefficients."""
    text = names[0] if signs[0] > 0 else f"−{names[0]}"
    for name, sign in zip(names[1:], signs[1:], strict=True):
        text += f" + {name}" if sign > 0 else f" − {name}"
    return text


def _linear(names: list[str], beta: np.ndarray, intercept: float) -> str:
    """'2·a + 0.5·b + 3' from a least-squares fit."""
    text = " + ".join(f"{b:.3g}·{n}" for n, b in zip(names, beta, strict=True))
    if abs(intercept) > 1e-9:
        text += f" + {intercept:.3g}"
    return text.replace("+ -", "− ")


def _lstsq(a: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Least squares with an intercept: coefficients, intercept, fitted values."""
    design = np.column_stack([a, np.ones(len(a))])
    solution = np.linalg.lstsq(design, y, rcond=None)[0]
    return solution[:-1], float(solution[-1]), design @ solution


def _complete(columns: dict[str, np.ndarray], y: np.ndarray, names: list[str]) -> np.ndarray:
    """Rows where the target and every named column have a value."""
    rows = ~np.isnan(y)
    for name in names:
        rows &= ~np.isnan(columns[name])
    return rows


def _quality(
    columns: dict[str, np.ndarray], y: np.ndarray, names: list[str], numeric: bool
) -> float | None:
    """How well a least-squares fit on these columns follows the target: R², or for two
    classes the AUC of the fit (None: too few rows, or a constant target)."""
    rows = _complete(columns, y, names)
    if rows.sum() < IDENTITY_MIN_ROWS:
        return None
    values = y[rows]
    spread = float(np.var(values))
    if spread == 0:
        return None
    fitted = _lstsq(np.column_stack([columns[c][rows] for c in names]), values)[2]
    if numeric:
        return 1 - float(np.mean((values - fitted) ** 2)) / spread
    return _auc(fitted, values > 0.5)


def _identity(
    columns: dict[str, np.ndarray], y: np.ndarray, numeric: bool, target: str
) -> tuple[list[str], str] | None:
    """Forward search of the columns whose sum, difference or linear combination gives the
    target, from each of the IDENTITY_STARTS columns that follow it best alone: the column that
    best improves the least-squares fit is added, up to IDENTITY_TERMS; the first identity found
    is returned, without the columns it does not need, with its sentence."""
    alone = [(q, name) for name in columns if (q := _quality(columns, y, [name], numeric))]
    starts = [name for _, name in sorted(alone, reverse=True)[:IDENTITY_STARTS]]
    for first in starts:
        chosen = [first]
        while True:
            ordered = [c for c in columns if c in chosen]  # the dataset's order, for the sentence
            found = _holds(columns, y, ordered, numeric, target)
            if found is not None:
                return _pruned(columns, y, ordered, found, numeric, target)
            if len(chosen) >= IDENTITY_TERMS:
                break
            scored = [
                (q, name)
                for name in columns
                if name not in chosen
                and (q := _quality(columns, y, [*chosen, name], numeric)) is not None
            ]
            if not scored:
                break
            chosen.append(max(scored)[1])
    return None


def _pruned(
    columns: dict[str, np.ndarray],
    y: np.ndarray,
    chosen: list[str],
    found: str,
    numeric: bool,
    target: str,
) -> tuple[list[str], str]:
    """The identity without the columns it does not need: the forward search may have taken a
    column first (a pre-match forecast) that the identity holds without."""
    for name in list(chosen):
        if len(chosen) <= 1:
            break
        rest = [c for c in chosen if c != name]
        without = _holds(columns, y, rest, numeric, target)
        if without is not None:
            chosen, found = rest, without
    return chosen, found


def _holds(
    columns: dict[str, np.ndarray], y: np.ndarray, chosen: list[str], numeric: bool, target: str
) -> str | None:
    """Does the target follow these columns? The sentence that says how, or None."""
    rows = _complete(columns, y, chosen)
    a = np.column_stack([columns[c][rows] for c in chosen])
    values = y[rows]
    beta, intercept, fitted = _lstsq(a, values)
    signs = np.where(beta >= 0, 1.0, -1.0)
    combined = a @ signs
    if not numeric:
        if len(chosen) < 2:  # one column alone: the per-column rule reads it
            return None
        signs = signs if signs[0] > 0 else -signs  # either direction separates two classes
        combined = a @ signs
        auc = max(_auc(combined, values > 0.5), _auc(fitted, values > 0.5))
        if auc < SEPARATES:
            return None
        return f"{_combination(chosen, signs)} separates the classes of {target}, AUC {auc:.3f}"
    tolerance = max(1e-9, 1e-3 * float(np.std(values)))
    modes, counts = np.unique(values, return_counts=True)
    informative = values != modes[np.argmax(counts)]
    if informative.sum() < IDENTITY_MIN_ROWS:
        return None
    for guess, text in (
        (combined, _combination(chosen, signs)),
        (fitted, _linear(chosen, beta, intercept)),
    ):
        exact = np.abs(values - guess) <= tolerance
        share = float(exact.mean())
        if share >= EXACT_SHARE and float(exact[informative].mean()) >= EXACT_INFORMATIVE:
            return f"{target} = {text} on {share:.0%} of the rows"
    return None


def _identities(
    frame: pd.DataFrame,
    decisions: list[Decision],
    target: DataField,
    numeric: bool,
    forced: set[str],
) -> None:
    """R-SEL-06 (094, ML-2): no single column gives the answer, but a sum, a difference or a
    linear function of two or three does (the final score of both teams, the Elo after the
    match, the components of a total). Every column of such an identity is left out as a leak,
    reactivable, its sentence saying the identity; up to IDENTITY_FOUND identities."""
    candidates = [d for d in decisions if d.use and d.type in NUMERIC and d.column not in forced]
    if not candidates or target.name not in frame.columns:
        return
    rows = np.arange(len(frame))
    if len(rows) > IDENTITY_ROWS:
        rows = np.sort(np.random.default_rng(0).choice(len(frame), IDENTITY_ROWS, replace=False))
    sample = frame.iloc[rows]
    raw = _col(sample, target.name)
    y: np.ndarray
    if numeric:
        y = np.asarray(pd.to_numeric(raw, errors="coerce"), dtype=float)
    else:
        labels = raw.dropna().astype(str)
        if labels.nunique() != 2:
            return
        first = sorted(labels.unique())[0]
        y = np.where(raw.isna(), np.nan, (raw.astype(str) == first).to_numpy(dtype=float))
    columns = {d.column: _col(sample, d.column).to_numpy(dtype=float) for d in candidates}
    by_name = {d.column: d for d in candidates}
    for _ in range(IDENTITY_FOUND):
        found = _identity(columns, y, numeric, target.name)
        if found is None:
            return
        chosen, sentence = found
        for name in chosen:
            others = [c for c in chosen if c != name]
            joined = f"with {' and '.join(others)}, it gives" if others else "it gives"
            _leave_out(
                by_name[name],
                "R-SEL-06",
                f"Looks like it contains the answer: {joined} the target ({sentence}). "
                "Use it anyway only if it is known before the outcome.",
            )
            del columns[name]


MAX_TEXTS = 3  # R-TEXT-06: free-text columns a model uses by default


def _text_budget(
    decisions: list[Decision], profiles: dict[str, ColumnProfile], forced: set[str]
) -> None:
    """R-TEXT-06 (094, ML-6): each free-text column adds up to 50 inputs (R-TEXT-04); a survey
    with 15 of them (lists of answers "a;b;c") made the first model take minutes. The
    MAX_TEXTS richest (texts × median words) are kept, the others left out, reactivable."""
    texts = [d for d in decisions if d.use and d.type == "text" and d.column not in forced]
    if len(texts) <= MAX_TEXTS:
        return

    def richness(decision: Decision) -> float:
        text = profiles[decision.column].text
        return text.texts * max(text.median_words, 1.0) if text is not None else 0.0

    ranked = sorted(texts, key=lambda d: -richness(d))
    kept = ", ".join(d.column for d in ranked[:MAX_TEXTS])
    for decision in ranked[MAX_TEXTS:]:
        decision.use, decision.reactivable = False, True
        decision.auto["use"] = "no"
        decision.apply(
            "R-TEXT-06",
            f"Left out to keep the model fast: {len(texts)} free-text columns, each adding up "
            f"to {textual.COMPONENTS} inputs; the {MAX_TEXTS} richest are used ({kept}). Use it "
            "anyway if it matters.",
        )


def _leave_out(decision: Decision, rule: str, sentence: str) -> None:
    """A column the rules had kept, left out as a leak after all (its type rules undone)."""
    decision.use, decision.reactivable, decision.leak = False, True, True
    decision.rules, decision.why = [], []
    decision.impute = decision.transform = decision.encoding = decision.scaling = None
    decision.indicator, decision.cycle, decision.order = False, None, None
    decision.auto = {"use": "no"}
    decision.apply(rule, sentence)


def _type_rules(
    decision: Decision, fld: DataField, prof: ColumnProfile, model: ModelProfile, choices: Choices
) -> None:
    if fld.type in NUMERIC:
        decision.cycle = _cycle(fld, prof)
        _numeric(decision, prof, model, choices)
    elif fld.type == "boolean":
        _boolean(decision, prof, model, choices)
    elif fld.type == "categorical_ordinal":
        _ordinal(decision, fld, prof, model, choices)
    elif fld.type == "categorical_nominal":
        _nominal(decision, prof, model, choices)
    elif fld.type == "datetime":
        _date(decision, prof, model)
    elif fld.type == "text":
        _text(decision, fld, prof, model)


def _cycle(fld: DataField, prof: ColumnProfile) -> list[float] | None:
    """[period, first value] of a number that goes round (R-DATE-04): its role says which
    cycle, and every value must lie in the cycle's range (whole numbers, except an angle)."""
    if fld.role not in CYCLES or prof.minimum is None or prof.maximum is None:
        return None
    if fld.role != "angle" and not prof.integer:
        return None
    for lowest, highest, period, start in CYCLES[fld.role]:
        if lowest <= prof.minimum and prof.maximum <= highest and prof.distinct >= 3:
            return [period, start]
    return None


def _numeric(
    decision: Decision, prof: ColumnProfile, model: ModelProfile, choices: Choices
) -> None:
    if model.handles_missing:
        decision.impute = "none"
        decision.apply("R-NUM-01", "Missing values are handled natively by the model.")
    elif choices.numeric_impute != "auto":
        decision.impute = choices.numeric_impute
        decision.apply(
            "R-NUM-13",
            f"Missing values → {NAMES[choices.numeric_impute]} (set in Missing & outliers).",
        )
    elif prof.skewness is not None and abs(prof.skewness) <= 0.5:
        decision.impute = "mean"
        decision.apply(
            "R-NUM-02", f"Missing values → mean (nearly symmetric, skewness {prof.skewness:.2f})."
        )
    else:
        decision.impute = "median"
        skew = "unknown" if prof.skewness is None else f"{prof.skewness:.2f}"
        decision.apply("R-NUM-03", f"Missing values → median (skewness {skew}).")
    if prof.missing_share > choices.indicator_share and not model.handles_missing:
        if decision.impute in ("knn", "iterative"):
            decision.apply(
                "R-NUM-04",
                f"No “was missing” indicator with the {NAMES[decision.impute]} "
                f"({prof.missing_share:.0%} empty).",
            )
        else:
            decision.indicator = True
            decision.apply(
                "R-NUM-04",
                f"Adds a “was missing” indicator ({prof.missing_share:.0%} empty, above "
                f"{choices.indicator_share:.0%}: often informative).",
            )
    if model.tree_based:
        decision.apply(
            "R-NUM-10", "Trees are unaffected by monotonic transformations and scales: kept as is."
        )
        return
    if decision.cycle is not None:
        period, start = decision.cycle
        decision.transform = "cycle"
        decision.apply(
            "R-DATE-04",
            f"Goes round every {period:g} (from {start:g}): replaced by its sine and cosine, so "
            f"that {start + period - 1:g} sits next to {start:g}.",
        )
        if model.nonnegative_inputs:
            decision.scaling = "minmax"
            decision.apply("R-NUM-09", "Scaled to [0, 1]: this model needs inputs ≥ 0.")
        return
    if model.sensitive_to_skew and prof.skewness is not None and abs(prof.skewness) > 1:
        if prof.all_nonnegative and prof.skewness > 1:
            decision.transform = "log1p"
            decision.apply(
                "R-NUM-05", f"log1p: skewness {prof.skewness:.2f} > 1 and every value ≥ 0."
            )
        else:
            decision.transform = "yeojohnson"
            decision.apply(
                "R-NUM-06",
                f"Yeo-Johnson: |skewness| {abs(prof.skewness):.2f} > 1 with negative values.",
            )
    if model.distance:
        decision.scaling = "robust"
        decision.apply(
            "R-DIST-01",
            "Robust scaling (median, IQR): otherwise the column with the largest numbers "
            "decides every distance.",
        )
    elif model.nonnegative_inputs:
        decision.scaling = "minmax"
        decision.apply("R-NUM-09", "Scaled to [0, 1]: this model needs inputs ≥ 0.")
    elif model.needs_scaling:
        if prof.outlier_share is not None and prof.outlier_share > 0.05:
            decision.scaling = "robust"
            decision.apply(
                "R-NUM-07",
                f"Robust scaling (median, IQR): {prof.outlier_share:.0%} of values are outliers.",
            )
        else:
            decision.scaling = "standard"
            decision.apply("R-NUM-08", "Standardised (mean 0, standard deviation 1).")


def _boolean(
    decision: Decision, prof: ColumnProfile, model: ModelProfile, choices: Choices
) -> None:
    decision.encoding = "binary"
    decision.apply("R-BOOL-01", "Encoded 0 / 1.")
    if prof.missing_share > choices.indicator_share and not model.handles_missing:
        decision.impute = "missing_category"
        decision.apply(
            "R-BOOL-02", f"Missing values are their own category ({prof.missing_share:.0%} empty)."
        )
    else:
        decision.impute = "most_frequent"
        decision.apply("R-BOOL-02", "The few missing values take the most frequent value.")


def _ordinal(
    decision: Decision,
    fld: DataField,
    prof: ColumnProfile,
    model: ModelProfile,
    choices: Choices,
) -> None:
    if not fld.order:
        decision.apply("R-ORD-02", "No order set in Columns: treated as unordered categories.")
        _nominal(decision, prof, model, choices)
        return
    decision.impute = "most_frequent"
    decision.encoding = "ordinal"
    decision.order = list(fld.order)
    decision.apply(
        "R-ORD-01",
        "Encoded 0 to k-1 in the order validated in Columns: "
        + " < ".join(fld.order[:6])
        + ("…" if len(fld.order) > 6 else ""),
    )
    if model.distance:
        decision.scaling = "robust"
        decision.apply("R-DIST-01", "Codes scaled robustly, like a number.")
    elif model.nonnegative_inputs:
        decision.scaling = "minmax"
        decision.apply("R-NUM-09", "Codes scaled to [0, 1]: this model needs inputs ≥ 0.")
    elif model.needs_scaling:
        decision.scaling = "standard"
        decision.apply("R-NUM-08", "Standardised like a number.")


def _nominal(
    decision: Decision, prof: ColumnProfile, model: ModelProfile, choices: Choices
) -> None:
    decision.impute = "missing_category"
    decision.apply(
        "R-CAT-01",
        f"Missing values become “{MISSING}”; categories under {choices.rare_share:.0%} "
        f"(and {choices.rare_rows} rows) are grouped as Other.",
    )
    if prof.distinct == 2:
        decision.encoding = "binary"
        decision.apply("R-CAT-02", "Two categories: one 0 / 1 column.")
    elif model.distance:
        decision.encoding = "onehot"
        decision.apply(
            "R-DIST-02",
            f"One-hot: {prof.distinct} categories"
            + (
                f", the {DISTANCE_CATEGORIES} most frequent kept and the rest grouped as Other."
                if prof.distinct > DISTANCE_CATEGORIES
                else "."
            ),
        )
    elif prof.distinct > TARGET_ENCODING_LIMIT:
        decision.encoding = "frequency"
        decision.apply(
            "R-CAT-09",
            f"Frequency encoding: {prof.distinct:,} categories, too many to encode each one; how "
            "common a category is (share of the training rows) reads no target, so it cannot "
            "leak. A category never seen counts 0.",
        )
        if model.nonnegative_inputs:
            decision.scaling = "minmax"
        elif model.needs_scaling:
            decision.scaling = "standard"
    elif model.native_categorical:
        decision.encoding = "native"
        decision.apply("R-CAT-03", "Passed as a category: the model handles categories natively.")
    elif model.tree_based:
        decision.encoding = "ordinal" if prof.distinct <= 50 else "target"
        decision.apply(
            "R-CAT-04",
            "Ordinal codes for trees."
            if prof.distinct <= 50
            else "Target encoding (more than 50 categories).",
        )
    elif prof.distinct <= choices.one_hot_limit or model.nonnegative_inputs:
        decision.encoding = "onehot"
        rule = (
            "R-CAT-07"
            if model.nonnegative_inputs and prof.distinct > choices.one_hot_limit
            else "R-CAT-05"
        )
        decision.apply(rule, f"One-hot: {prof.distinct} categories.")
    else:
        decision.encoding = "target"
        decision.apply(
            "R-CAT-06",
            f"Target encoding, cross-fitted on 5 folds, smoothing {choices.target_smoothing:g} "
            f"({prof.distinct} categories, above the one-hot limit of {choices.one_hot_limit}).",
        )


def _date(decision: Decision, prof: ColumnProfile, model: ModelProfile) -> None:
    decision.transform = "date_parts" if model.tree_based else "date_cycles"
    decision.impute = "median"
    decision.apply(
        "R-DATE-01",
        "Replaced by its parts (year, month, day, weekday, weekend, hour) "
        "and days since the start.",
    )
    if prof.span_days >= 365:
        decision.apply(
            "R-DATE-06",
            f"The dates cover {prof.span_days / 365.25:.1f} years: week of the year and day of "
            "the year added, as sine / cosine or as integers for trees (the position in the "
            "year repeats).",
        )
    if prof.month_days >= 28 and 0 < prof.step_days <= 1.5:
        decision.apply(
            "R-DATE-07",
            f"Daily dates ({prof.month_days} days of the month): the day of the month goes "
            "round too (the 31st is next to the 1st) when parts are cycles.",
        )
    if not model.tree_based:
        decision.apply(
            "R-DATE-03", "Month, weekday and hour as sine / cosine (December is next to January)."
        )
        if model.distance:
            decision.scaling = "robust"
            decision.apply("R-DIST-01", "Parts scaled robustly, like numbers.")
        elif model.nonnegative_inputs:
            decision.scaling = "minmax"
            decision.apply("R-NUM-09", "Parts scaled to [0, 1]: this model needs inputs ≥ 0.")
        elif model.needs_scaling:
            decision.scaling = "standard"
    else:
        decision.apply("R-DATE-02", "Parts as integers for trees.")


def _text(decision: Decision, fld: DataField, prof: ColumnProfile, model: ModelProfile) -> None:
    """R-TEXT-01 to R-TEXT-05: free text read through TF-IDF then a truncated SVD (052)."""
    text = prof.text or textual.TextProfile(0, 0.0)
    decision.impute = "empty_text"
    decision.transform = "tfidf_svd"
    decision.apply(
        "R-TEXT-01",
        f"Free text used as an input ({text.texts:,} texts): the model reads it as a few "
        "numbers; empty values count as an empty text.",
    )
    decision.encoding = textual.analyzer_for(text, fld.role)
    words = f"median {text.median_words:g} word{'s' if text.median_words != 1 else ''}"
    if decision.encoding == "chars":
        why = "a short label" if fld.role == "label" else f"short labels ({words})"
        decision.apply(
            "R-TEXT-02",
            f"Cut into pieces of 3 to 5 characters: {why} share few whole words, and pieces "
            "bring spelling variants together.",
        )
    else:
        decision.apply("R-TEXT-02", f"Cut into words and pairs of words ({words}).")
    decision.apply(
        "R-TEXT-03",
        f"TF-IDF on at most {textual.TERMS:,} terms seen in at least {textual.MIN_ROWS} rows and "
        f"fewer than {textual.MAX_SHARE:.0%} of them, learnt on at most "
        f"{textual.FIT_ROWS:,} training rows.",
    )
    decision.apply(
        "R-TEXT-04",
        f"Reduced to at most {textual.COMPONENTS} numbers by a truncated SVD: the main themes "
        "of the vocabulary, a bounded width for every model.",
    )
    if model.nonnegative_inputs:
        decision.scaling = "minmax"
        decision.apply("R-TEXT-05", "Themes scaled to [0, 1]: this model needs inputs ≥ 0.")
    elif model.needs_scaling:
        decision.scaling = "standard"
        decision.apply("R-TEXT-05", "Themes standardised (mean 0, standard deviation 1).")
    elif model.tree_based:
        decision.apply("R-TEXT-05", "Themes kept as they are for trees.")


# --- Choices on top of the rules -----------------------------------------------------------------

NO_MISSING = "This model cannot handle missing values: they must be filled."
ONLY_ORDINAL = "Only for ordinal codes or frequencies."


def _value(decision: Decision, key: str) -> str:
    if key == "indicator":
        return "yes" if decision.indicator else "no"
    return getattr(decision, key) or "none"


def _snapshot(decision: Decision) -> None:
    for key in CHOICE_KEYS:
        decision.auto[key] = _value(decision, key)


def _options(
    decision: Decision, prof: ColumnProfile, model: ModelProfile, ordered: bool = False
) -> dict[str, list[Option]]:
    """The choices each setting offers for this column, with the reason when one is impossible;
    `ordered`: the rows have an order (panel or time split), so lags are possible."""
    options = {
        "use": [
            Option("yes", None if decision.reactivable else (decision.why or [""])[0]),
            Option(
                "no",
                "The target is never an input." if decision.rules[:1] == ["R-SEL-01"] else None,
            ),
        ]
    }
    no_fill = None if model.handles_missing else NO_MISSING
    if decision.type in NUMERIC:
        fills = ("mean", "median", "constant")
        options |= {
            "impute": [*(Option(v) for v in (*fills, "knn", "iterative")), Option("none", no_fill)],
            "indicator": [
                Option(
                    "yes",
                    None if decision.impute in fills else "Only with mean, median or 0 imputation.",
                ),
                Option("no"),
            ],
            "transform": [
                Option("none"),
                Option("log1p", None if prof.all_nonnegative else "Needs every value ≥ 0."),
                Option("yeojohnson"),
                Option("cycle", None if decision.cycle is not None else NOT_CYCLIC),
                *(Option(v, None if ordered else NEEDS_ORDER) for v in LAGS),
            ],
            "scaling": [Option(v) for v in ("none", "standard", "robust", "minmax")],
        }
    elif decision.type == "datetime":
        options["transform"] = [Option("date_parts"), Option("date_cycles")]
        options["scaling"] = [Option(v) for v in ("none", "standard", "minmax")]
    elif decision.type == "text":
        # 091: sentence embeddings, greyed with the reason when the `embed` extra is missing
        options["encoding"] = [
            Option("words"),
            Option("chars"),
            Option("embeddings", embedding.status().reason),
        ]
        options["scaling"] = [Option(v) for v in ("none", "standard", "minmax")]
    elif decision.type in CATEGORICAL:
        options["impute"] = [
            Option("missing_category"),
            Option("most_frequent"),
            Option("none", no_fill),
        ]
        if decision.type != "boolean":
            options["encoding"] = [
                Option("onehot"),
                Option("binary", None if prof.distinct == 2 else "Only for two categories."),
                Option("ordinal"),
                Option("frequency"),
                Option(
                    "target",
                    "No target here: distances only."
                    if model.distance
                    else (
                        "Can be negative: this model needs inputs ≥ 0."
                        if model.nonnegative_inputs
                        else None
                    ),
                ),
                Option(
                    "native",
                    None
                    if model.native_categorical
                    else "This model does not take categories as they are.",
                ),
            ]
            ordinal = None if decision.encoding in ("ordinal", "frequency") else ONLY_ORDINAL
            options["scaling"] = [
                Option("none"),
                Option("standard", ordinal),
                Option("minmax", ordinal),
            ]
    return options


def _set(decision: Decision, key: str, value: str) -> None:
    if key == "indicator":
        decision.indicator = value == "yes"
    elif key in ("transform", "scaling"):
        setattr(decision, key, None if value == "none" else value)
    elif key == "impute":
        decision.impute = value
        if value not in ("mean", "median", "constant"):
            decision.indicator = False
    elif key == "encoding":
        decision.encoding = value
        if value == "embeddings":
            decision.apply(
                "R-FUS-01",
                f"Sentence embeddings ({embedding.MODEL}), reduced to {embedding.DIMENSIONS} "
                "numbers by a PCA; kept only if they beat TF-IDF in cross-validation (R-FUS-02).",
            )
        if value not in ("ordinal", "frequency") and decision.type != "text":
            decision.scaling = None


def _apply(
    decision: Decision,
    choice: ColumnChoice,
    prof: ColumnProfile,
    model: ModelProfile,
    warnings: list[str],
    ordered: bool = False,
) -> None:
    """The user's per-column choices, in an order where each one sees the previous ones."""
    for key in CHOICE_KEYS:
        wanted = getattr(choice, key)
        if wanted is None:
            continue
        text = ("yes" if wanted else "no") if key == "indicator" else str(wanted)
        offered = _options(decision, prof, model, ordered).get(key, [])
        available = {o.value: o.reason for o in offered}
        if text not in available:
            warnings.append(
                f"{decision.column}: “{text}” is not a possible {key} here; automatic choice kept."
            )
            continue
        if available[text]:
            warnings.append(f"{decision.column}: {text} — {available[text]} Automatic choice kept.")
            continue
        if _value(decision, key) == text:
            continue
        _set(decision, key, text)
        decision.manual.append(key)
        decision.apply("USER", f"{key.capitalize()} set to {text}: your choice.")


def _timeline(series: pd.Series) -> pd.Series:
    """Dates as seconds since the earliest one; numbers as they are."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return (series - series.min()).dt.total_seconds()
    return series.astype(float)


def _collinearity(
    frame: pd.DataFrame,
    decisions: list[Decision],
    target: DataField,
    target_numeric: bool,
    forced: set[str],
) -> None:
    """R-SEL-07: of two numbers (or dates, as time) with |r| > 0.95, drop the one less linked to
    the target (never one the user asked to keep). An age and a birth date are the same
    information: kept together, their coefficients explode in opposite directions."""
    numeric = [d for d in decisions if d.use and (d.type in NUMERIC or d.type == "datetime")]
    as_number = {d.column: _timeline(_col(frame, d.column)) for d in numeric}
    for i, first in enumerate(numeric):
        for second in numeric[i + 1 :]:
            if not (first.use and second.use):
                continue
            pair = pd.DataFrame(
                {first.column: as_number[first.column], second.column: as_number[second.column]}
            ).dropna()
            if len(pair) < 10 or float(np.min(np.asarray(pair.std(), dtype=float))) == 0:
                continue
            r = abs(float(np.corrcoef(pair[first.column], pair[second.column])[0, 1]))
            if r <= 0.95:
                continue
            link = {
                d.column: _association(
                    as_number[d.column], _col(frame, target.name), True, target_numeric
                )
                for d in (first, second)
            }
            drop, keep = (
                (first, second) if link[first.column] < link[second.column] else (second, first)
            )
            if drop.column in forced:
                drop, keep = keep, drop
            if drop.column in forced:
                continue
            drop.use = False
            drop.auto["use"] = "no"
            drop.apply(
                "R-SEL-07",
                f"Nearly duplicates {keep.column} (r = {r:.3f}); unstable coefficients otherwise.",
            )


# --- Pipeline ------------------------------------------------------------------------------------


class DateParts(BaseEstimator, TransformerMixin):
    """Date → year, month, day, weekday, weekend, hour (when they vary) and days since the start;
    `yearly`: + week and day of the year (R-DATE-06); `monthly`: the day of the month as a cycle
    when `cyclical` (R-DATE-07). Models saved before 063 have neither: the class defaults say so."""

    yearly = False
    monthly = False

    def __init__(self, cyclical: bool = True, yearly: bool = False, monthly: bool = False) -> None:
        self.cyclical = cyclical
        self.yearly = yearly
        self.monthly = monthly

    def fit(self, X, y=None):
        values = pd.to_datetime(pd.Series(np.asarray(X).ravel()))
        self.start_ = values.min()
        self.hour_ = bool(values.dt.hour.nunique() > 1)
        self.names_ = self._names()
        return self

    def _names(self) -> list[str]:
        day_cycle = self.cyclical and self.monthly
        names = ["year", *([] if day_cycle else ["day"]), "weekend", "days_since_start"]
        cycles = (
            ["month", "weekday"]
            + (["hour"] if self.hour_ else [])
            + (["day"] if day_cycle else [])
            + (["week", "day_of_year"] if self.yearly else [])
        )
        if self.cyclical:
            names += [f"{c}_{f}" for c in cycles for f in ("sin", "cos")]
        else:
            names += cycles
        return names

    def transform(self, X):
        values = pd.to_datetime(pd.Series(np.asarray(X).ravel()))
        weekday = values.dt.day_of_week
        out = {
            "year": values.dt.year,
            "day": values.dt.day,
            "weekend": (weekday >= 5).astype(float),
            "days_since_start": (values - self.start_).dt.total_seconds() / 86_400,
        }
        cycles = {
            "month": (values.dt.month - 1, 12),
            "weekday": (weekday, 7),
            "hour": (values.dt.hour, 24),
            "week": (values.dt.isocalendar().week.astype(float) - 1, 52.1775),
            "day_of_year": (values.dt.day_of_year - 1, 365.25),
        }
        for name, (value, period) in cycles.items():
            out[f"{name}_sin"] = np.sin(2 * np.pi * value / period)
            out[f"{name}_cos"] = np.cos(2 * np.pi * value / period)
            if name not in out:
                out[name] = value + (1 if name in ("week", "day_of_year") else 0)
        days = values.dt.days_in_month
        out["day_sin"] = np.sin(2 * np.pi * (values.dt.day - 1) / days)
        out["day_cos"] = np.cos(2 * np.pi * (values.dt.day - 1) / days)
        return pd.DataFrame(out)[self.names_].to_numpy(dtype=float)

    def get_feature_names_out(self, input_features=None):
        prefix = input_features[0] if input_features is not None and len(input_features) else "date"
        return np.array([f"{prefix}_{name}" for name in self.names_], dtype=object)


def _scaler(kind: str | None):
    return {
        "standard": StandardScaler(),
        "robust": RobustScale(),
        "minmax": MinMaxScaler(clip=True),
    }.get(kind or "", "passthrough")


def _rare(n: int, choices: Choices) -> int:
    return max(choices.rare_rows, math.ceil(choices.rare_share * n))


def _numeric_steps(
    d: Decision, clip: str | None, order: "Order | None"
) -> tuple[list[tuple[str, object]], list[str]]:
    """The steps of a number, and the columns they read (a lag also reads the order, R-LAG-01)."""
    steps: list[tuple[str, object]] = []
    columns = [d.column]
    if d.transform in LAGS and order is not None:
        kind = "mean" if d.transform == "past_mean" else "ewm"
        steps.append(
            (
                "past",
                enc.PastSmooth(kind, order.window, order.shift, entity=order.entity is not None),
            )
        )
        columns += [*([order.entity] if order.entity else []), order.period]
    if d.impute in ("mean", "median"):
        steps.append(("impute", SimpleImputer(strategy=d.impute, add_indicator=d.indicator)))
    elif d.impute == "constant":
        steps.append(
            (
                "impute",
                SimpleImputer(strategy="constant", fill_value=0, add_indicator=d.indicator),
            )
        )
    if clip and d.transform != "cycle":
        steps.append(("clip", Clipper(clip)))
    if d.transform == "log1p":
        steps.append(("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")))
    elif d.transform == "yeojohnson":
        steps.append(("yeojohnson", PowerTransformer(method="yeo-johnson", standardize=False)))
    elif d.transform == "cycle":
        period, start = d.cycle or [24.0, 0.0]
        steps.append(("cycle", enc.Cycle(period, start)))
    steps.append(("scale", _scaler(d.scaling)))
    return steps, columns


def _target_encoder(preparation: Preparation, task: Task) -> TargetEncoder:
    """Cross-fitted on folds that follow the validation (R-CAT-08)."""
    target_type = (
        "continuous" if task == "regression" else ("binary" if task == "binary" else "multiclass")
    )
    folds = preparation.folds
    cv: Any = (
        enc.FoldPlan(folds.kind, getattr(preparation, "fold_keys", None))  # None: unpickled
        if folds is not None and folds.kind != "random"
        else enc.random_folds(task != "regression")
    )
    return TargetEncoder(
        target_type=target_type,  # type: ignore[arg-type]
        smooth=preparation.choices.target_smoothing,  # type: ignore[arg-type]
        cv=cv,
    )


def _category_steps(
    d: Decision, preparation: Preparation, rows: int, task: Task, drop_first: bool
) -> list[tuple[str, object]]:
    choices = preparation.choices
    steps: list[tuple[str, object]] = []
    imputer = None
    if d.impute == "missing_category":
        imputer = SimpleImputer(strategy="constant", fill_value=MISSING)
    elif d.impute == "most_frequent":
        imputer = SimpleImputer(strategy="most_frequent")
    if imputer is not None:
        if d.encoding == "target":  # the folds read the rows' index (R-CAT-08)
            imputer.set_output(transform="pandas")
        steps.append(("impute", imputer))
    rare = enc.RareGrouper(choices.rare_rows, choices.rare_share)
    if d.encoding == "ordinal":
        if not d.order:  # a validated order is never regrouped
            steps.append(("rare", rare))
        steps.append(
            (
                "ordinal",
                OrdinalEncoder(
                    categories=[d.order] if d.order else "auto",  # type: ignore[arg-type]
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            )
        )
        steps.append(("scale", _scaler(d.scaling)))
    elif d.encoding == "target":
        steps += [
            ("rare", rare),
            ("target", _target_encoder(preparation, task)),
            ("scale", StandardScaler()),
        ]
    elif d.encoding == "frequency":
        steps += [("frequency", enc.FrequencyEncoder()), ("scale", _scaler(d.scaling))]
    else:  # binary, onehot
        steps.append(
            (
                "onehot",
                OneHotEncoder(
                    drop="if_binary"
                    if d.encoding == "binary"
                    else ("first" if drop_first else None),
                    handle_unknown="infrequent_if_exist" if d.encoding == "onehot" else "ignore",
                    min_frequency=_rare(rows, choices) if d.encoding == "onehot" else None,
                    max_categories=DISTANCE_CATEGORIES
                    if preparation.distance and d.encoding == "onehot"
                    else None,
                    sparse_output=False,
                ),
            )
        )
    return steps


def build(
    preparation: Preparation, rows: int, task: Task, drop_first: bool = False
) -> ColumnTransformer:
    """The scikit-learn ColumnTransformer that applies the per-column decisions. Its steps are
    named c0, c1… (094, ML-1): scikit-learn refuses a step name holding "__" ("bin__") or naming
    one of its own settings ("verbose"); a step's column is the first of its column list
    (`fitted_step`)."""
    choices = preparation.choices
    clip = choices.outliers if choices.outliers in ("clip_percentiles", "clip_iqr") else None
    transformers = []
    for d in preparation.used:
        columns = [d.column]
        if d.type in NUMERIC:
            steps, columns = _numeric_steps(d, clip, preparation.order)
        elif d.type == "datetime":
            steps = [
                (
                    "parts",
                    DateParts(
                        cyclical=d.transform == "date_cycles",
                        yearly="R-DATE-06" in d.rules,
                        monthly="R-DATE-07" in d.rules,
                    ),
                ),
                ("impute", SimpleImputer(strategy="median")),
                ("scale", _scaler(d.scaling)),
            ]
        elif d.type == "text":
            analyzer = cast(textual.Analyzer, "chars" if d.encoding == "chars" else "words")
            text_step: Any = (
                embedding.EmbeddingFeatures()
                if d.encoding == "embeddings"
                else textual.TextFeatures(analyzer)
            )
            steps = [("text", text_step), ("scale", _scaler(d.scaling))]
        else:
            steps = _category_steps(d, preparation, rows, task, drop_first)
        transformers.append((f"c{len(transformers)}", Pipeline(steps), columns))
    order = preparation.order
    if lags_target(preparation) and order is not None and preparation.target is not None:
        kind = "mean" if choices.target_lag == "past_mean" else "ewm"
        lag = enc.PastSmooth(
            kind, order.window, order.shift, entity=order.entity is not None, keep_value=False
        )
        columns = [preparation.target, *([order.entity] if order.entity else []), order.period]
        steps = [("past", lag), ("impute", SimpleImputer(strategy="median"))]
        if any(d.scaling not in (None, "none") for d in preparation.used if d.type in NUMERIC):
            steps.append(("scale", StandardScaler()))
        transformers.append((f"c{len(transformers)}", Pipeline(steps), columns))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def fitted_step(transformer: Any, column: str) -> Any:
    """The fitted per-column pipeline of `column` in a fitted ColumnTransformer, or None."""
    for name, step, columns in getattr(transformer, "transformers_", []):
        if name != "remainder" and step != "drop" and len(columns) and str(columns[0]) == column:
            return step
    return None


def lags_target(preparation: Preparation) -> bool:
    """063 bis (R-LAG-03): the target's own lag is asked for and the rows are ordered."""
    return (
        preparation.choices.target_lag != "none"
        and preparation.order is not None
        and preparation.target is not None
        and not preparation.distance
    )


def context_columns(preparation: Preparation) -> list[str]:
    """Columns the pipeline reads without making them inputs: the order of a lag (R-LAG-01), and
    the target itself when its past is an input (R-LAG-03, only shifted)."""
    order = preparation.order
    own = lags_target(preparation)
    if order is None or not (own or any(d.transform in LAGS for d in preparation.used)):
        return []
    used = {d.column for d in preparation.used}
    wanted = [*([order.entity] if order.entity else []), order.period]
    if own and preparation.target is not None:
        wanted.insert(0, preparation.target)
    return [c for c in wanted if c not in used]


def steps(
    preparation: Preparation, rows: int, task: Task, drop_first: bool = False
) -> list[tuple[str, Any]]:
    """Every preparation step, in order: joint imputation, per-column preparation, selection."""
    found: list[tuple[str, Any]] = []
    for kind in ("knn", "iterative"):
        columns = tuple(
            d.column for d in preparation.used if d.type in NUMERIC and d.impute == kind
        )
        if columns:
            found.append((f"impute_{kind}", JointImputer(columns, kind)))
    found.append(("prepare", build(preparation, rows, task, drop_first)))
    for i, selection in enumerate(preparation.choices.selection):
        found.append(
            (f"select_{i}_{selection.method}", Selector(selection.method, selection.value, task))
        )
    return found
