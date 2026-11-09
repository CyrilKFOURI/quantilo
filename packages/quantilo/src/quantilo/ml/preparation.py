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
from sklearn.model_selection import KFold, StratifiedKFold
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

from quantilo.ml.data import CATEGORICAL, NUMERIC
from quantilo.ml.data import Field as DataField
from quantilo.ml.steps import Clipper, JointImputer, RobustScale, Selector
from quantilo.ml.target import Task

TARGET_ENCODING_LIMIT = 100
MISSING = "Missing"
DISTANCE_CATEGORIES = 20  # one-hot columns kept per category column for distances (032)
DISTANCE_EMPTY = 0.95  # a column this empty is left out of distances

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
    skewness = outliers = nonnegative = None
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
) -> Preparation:
    """One decision per column: the rules of the 017 spec, then the user's choices.
    `panel_columns`: the period and entity of a panel (035), {column: "period" | "entity"}."""
    choices = choices or Choices()
    decisions: list[Decision] = []
    warnings: list[str] = []
    forced: set[str] = set()
    excluded_columns = excluded_columns or set()
    for fld in fields:
        prof = profile(_col(frame, fld.name), fld)

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

        decision = _column(fld, prof, model, choices, exclude, warnings)
        decisions.append(decision)
        choice = choices.columns.get(fld.name)
        if decision.use and choice and choice.use is True:
            forced.add(fld.name)
    if model.sensitive_to_collinearity:
        _collinearity(frame, decisions, target, task == "regression" or source_numeric, forced)
    custom = choices.model_dump(exclude={"columns"}) != GLOBAL_DEFAULTS or any(
        d.manual for d in decisions
    )
    return Preparation(
        decisions,
        excluded=sum(not d.use for d in decisions if d.column != target.name),
        choices=choices,
        custom=custom,
        warnings=warnings,
    )


def _column(
    fld: DataField,
    prof: ColumnProfile,
    model: ModelProfile,
    choices: Choices,
    exclude: Callable[[Decision], None],
    warnings: list[str],
) -> Decision:
    """One column: is it left out (`exclude`), then the rules of its type, then the user's
    choices on top."""
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
            _apply(decision, choice, prof, model, warnings)
    decision.options = _options(decision, prof, model)
    return decision


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
            _distance_exclusion(decision, fld, prof)

        decisions.append(_column(fld, prof, DISTANCE, choices, exclude, warnings))
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


def _distance_exclusion(decision: Decision, fld: DataField, prof: ColumnProfile) -> None:
    """R-SEL-02 to R-SEL-04, R-DIST-03 and R-DIST-04: what a distance between rows leaves out."""

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
        return out("R-SEL-03", "Free text is left out.", reactivable=False)
    if fld.type == "datetime":
        return out("R-DIST-03", "Raw date: it says when, not what kind of row.")
    if fld.type not in USABLE:
        return out("R-SEL-03", "This type is not used here.", reactivable=False)
    if prof.distinct <= 1 or prof.dominant_share > 0.99:
        return out("R-SEL-04", f"Almost always the same value ({prof.dominant_share:.1%}).")
    if prof.missing_share > DISTANCE_EMPTY:
        return out("R-DIST-04", f"{prof.missing_share:.0%} empty (more than 95 %).")
    return None


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
    """R-SEL-01 to R-SEL-06, R-SEL-08 and R-SEL-09: is the column left out, and may it come
    back? With a number predicted as classes (018), leakage is measured against the number."""

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
    if fld.type in CATEGORICAL and (
        (prof.distinct_share > 0.5 and prof.distinct > 20) or prof.distinct > TARGET_ENCODING_LIMIT
    ):
        return out("R-SEL-02", f"Behaves like an identifier ({prof.distinct:,} distinct values).")
    if fld.type == "text":
        return out(
            "R-SEL-03", "Free text is left out (text features come later).", reactivable=False
        )
    if fld.type not in USABLE:
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


def _type_rules(
    decision: Decision, fld: DataField, prof: ColumnProfile, model: ModelProfile, choices: Choices
) -> None:
    if fld.type in NUMERIC:
        _numeric(decision, prof, model, choices)
    elif fld.type == "boolean":
        _boolean(decision, prof, model, choices)
    elif fld.type == "categorical_ordinal":
        _ordinal(decision, fld, prof, model, choices)
    elif fld.type == "categorical_nominal":
        _nominal(decision, prof, model, choices)
    elif fld.type == "datetime":
        _date(decision, model)


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


def _date(decision: Decision, model: ModelProfile) -> None:
    decision.transform = "date_parts"
    decision.impute = "median"
    decision.apply(
        "R-DATE-01",
        "Replaced by its parts (year, month, day, weekday, weekend, hour) "
        "and days since the start.",
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


# --- Choices on top of the rules -----------------------------------------------------------------

NO_MISSING = "This model cannot handle missing values: they must be filled."
ONLY_ORDINAL = "Only for ordinal codes."


def _value(decision: Decision, key: str) -> str:
    if key == "indicator":
        return "yes" if decision.indicator else "no"
    return getattr(decision, key) or "none"


def _snapshot(decision: Decision) -> None:
    for key in CHOICE_KEYS:
        decision.auto[key] = _value(decision, key)


def _options(
    decision: Decision, prof: ColumnProfile, model: ModelProfile
) -> dict[str, list[Option]]:
    """The choices each setting offers for this column, with the reason when one is impossible."""
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
            ],
            "scaling": [Option(v) for v in ("none", "standard", "robust", "minmax")],
        }
    elif decision.type == "datetime":
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
            ordinal = None if decision.encoding == "ordinal" else ONLY_ORDINAL
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
        if value != "ordinal":
            decision.scaling = None


def _apply(
    decision: Decision,
    choice: ColumnChoice,
    prof: ColumnProfile,
    model: ModelProfile,
    warnings: list[str],
) -> None:
    """The user's per-column choices, in an order where each one sees the previous ones."""
    for key in CHOICE_KEYS:
        wanted = getattr(choice, key)
        if wanted is None:
            continue
        text = ("yes" if wanted else "no") if key == "indicator" else str(wanted)
        available = {o.value: o.reason for o in _options(decision, prof, model).get(key, [])}
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
    """Date → year, month, day, weekday, weekend, hour (when they vary) and days since the start."""

    def __init__(self, cyclical: bool = True) -> None:
        self.cyclical = cyclical

    def fit(self, X, y=None):
        values = pd.to_datetime(pd.Series(np.asarray(X).ravel()))
        self.start_ = values.min()
        self.hour_ = bool(values.dt.hour.nunique() > 1)
        self.names_ = self._names()
        return self

    def _names(self) -> list[str]:
        names = ["year", "day", "weekend", "days_since_start"]
        cycles = ["month", "weekday"] + (["hour"] if self.hour_ else [])
        if self.cyclical:
            names += [f"{c}_{f}" for c in cycles for f in ("sin", "cos")]
        else:
            names += cycles
        return names

    def transform(self, X):
        values = pd.to_datetime(pd.Series(np.asarray(X).ravel()))
        out = {
            "year": values.dt.year,
            "day": values.dt.day,
            "weekend": (values.dt.weekday >= 5).astype(float),
            "days_since_start": (values - self.start_).dt.total_seconds() / 86_400,
        }
        cycles = {"month": (values.dt.month - 1, 12), "weekday": (values.dt.weekday, 7)}
        if self.hour_:
            cycles["hour"] = (values.dt.hour, 24)
        for name, (value, period) in cycles.items():
            if self.cyclical:
                out[f"{name}_sin"] = np.sin(2 * np.pi * value / period)
                out[f"{name}_cos"] = np.cos(2 * np.pi * value / period)
            else:
                out[name] = value
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


def build(
    preparation: Preparation, rows: int, task: Task, drop_first: bool = False
) -> ColumnTransformer:
    """The scikit-learn ColumnTransformer that applies the per-column decisions."""
    choices = preparation.choices
    clip = choices.outliers if choices.outliers in ("clip_percentiles", "clip_iqr") else None
    transformers = []
    for d in preparation.used:
        steps: list[tuple[str, object]] = []
        if d.type in NUMERIC:
            if d.impute in ("mean", "median"):
                steps.append(
                    ("impute", SimpleImputer(strategy=d.impute, add_indicator=d.indicator))
                )
            elif d.impute == "constant":
                steps.append(
                    (
                        "impute",
                        SimpleImputer(strategy="constant", fill_value=0, add_indicator=d.indicator),
                    )
                )
            if clip:
                steps.append(("clip", Clipper(clip)))
            if d.transform == "log1p":
                steps.append(
                    ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one"))
                )
            elif d.transform == "yeojohnson":
                steps.append(
                    ("yeojohnson", PowerTransformer(method="yeo-johnson", standardize=False))
                )
            steps.append(("scale", _scaler(d.scaling)))
        elif d.type == "datetime":
            steps += [
                ("parts", DateParts(cyclical="R-DATE-03" in d.rules)),
                ("impute", SimpleImputer(strategy="median")),
                ("scale", _scaler(d.scaling)),
            ]
        else:
            if d.impute == "missing_category":
                steps.append(("impute", SimpleImputer(strategy="constant", fill_value=MISSING)))
            elif d.impute == "most_frequent":
                steps.append(("impute", SimpleImputer(strategy="most_frequent")))
            if d.encoding == "ordinal":
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
                target_type = (
                    "continuous"
                    if task == "regression"
                    else ("binary" if task == "binary" else "multiclass")
                )
                steps.append(
                    (
                        "target",
                        TargetEncoder(
                            target_type=target_type,  # type: ignore[arg-type]
                            smooth=choices.target_smoothing,  # type: ignore[arg-type]
                            cv=(KFold if task == "regression" else StratifiedKFold)(  # type: ignore[arg-type]
                                5, shuffle=True, random_state=42
                            ),
                        ),
                    )
                )
                steps.append(("scale", StandardScaler()))
            else:  # binary, onehot
                steps.append(
                    (
                        "onehot",
                        OneHotEncoder(
                            drop="if_binary"
                            if d.encoding == "binary"
                            else ("first" if drop_first else None),
                            handle_unknown="infrequent_if_exist"
                            if d.encoding == "onehot"
                            else "ignore",
                            min_frequency=_rare(rows, choices) if d.encoding == "onehot" else None,
                            max_categories=DISTANCE_CATEGORIES
                            if preparation.distance and d.encoding == "onehot"
                            else None,
                            sparse_output=False,
                        ),
                    )
                )
        transformers.append((d.column, Pipeline(steps), [d.column]))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


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
