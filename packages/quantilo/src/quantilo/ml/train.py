"""Train a model on a frame: split, prepare (fitted on training rows only), fit, evaluate."""

import math
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn import metrics as skm
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

from quantilo.ml import classes as binning
from quantilo.ml import codes, conformal, encoding, error_analysis, evaluate, steps
from quantilo.ml import models as catalogue
from quantilo.ml import panel as panels
from quantilo.ml import preparation as prep
from quantilo.ml.data import NUMERIC, Field
from quantilo.ml.target import MAX_CLASSES, Task
from quantilo.variable.source import clean

SEED = 42
TEST_SHARE = 0.2
POINTS = 2_000  # predicted-vs-actual points drawn
PROBABILITIES = 5_000  # test probabilities sent for the threshold slider
PREDICTION_ROWS = 500
OTHER = "Other"
MAX_VIF_FEATURES = 60
MIN_CLASS_ROWS = 5  # rows per class: split in train / test, 5-fold target encoding


@dataclass
class Coefficient:
    feature: str
    estimate: float
    std_error: float | None = None
    low: float | None = None
    high: float | None = None
    p_value: float | None = None
    vif: float | None = None
    per_unit: float | None = None  # in the column's own unit (standardised numbers only)
    odds_ratio: float | None = None  # logistic regression


@dataclass
class ClassReport:
    label: str
    precision: float
    recall: float
    f1: float
    support: int


@dataclass
class Curve:
    x: list[float]
    y: list[float]
    thresholds: list[float] = field(default_factory=list)


@dataclass
class ClassCurves:
    """One class against all the others (multiclass): its ROC, precision-recall and calibration."""

    label: str
    share: float  # of the test rows: the precision-recall chance level
    auc: float | None
    roc: Curve
    pr: Curve
    calibration: Curve | None


@dataclass
class RegressionResult:
    r2: float | None
    mae: float | None
    rmse: float | None
    mape: float | None
    baseline_mae: float | None  # always predicting the training mean
    baseline_rmse: float | None
    skill: float | None  # 1 - MAE / baseline MAE
    actual: list[float]
    predicted: list[float]
    residual_bins: list[dict[str, float]]
    heteroscedasticity_p: float | None = None  # Breusch-Pagan on the training residuals


@dataclass
class ClassificationResult:
    labels: list[str]
    positive: str | None  # binary: the class whose probability is shown (the rarer one)
    accuracy: float | None
    balanced_accuracy: float | None
    f1_macro: float | None
    roc_auc: float | None
    log_loss: float | None
    baseline_accuracy: float | None  # always the majority class
    majority: str
    confusion: list[list[int]]  # [actual][predicted], in `labels` order
    report: list[ClassReport]
    roc: Curve | None = None
    pr: Curve | None = None
    calibration: Curve | None = None
    threshold: float | None = None  # binary: the probability above which `positive` is said
    truth: list[int] = field(default_factory=list)  # binary: 1 = positive (sample)
    probability: list[float] = field(default_factory=list)
    within_one: float | None = None  # ordered classes: share right or off by one class
    mean_class_error: float | None = None  # ordered classes: mean distance in classes
    class_curves: list[ClassCurves] = field(default_factory=list)  # 3 classes or more


@dataclass
class PredictionRow:
    row: int
    actual: str
    predicted: str
    error: float | None = None  # regression
    probability: float | None = None  # classification: probability of the predicted class
    correct: bool | None = None


@dataclass
class LearningCurve:
    """How the model improved iteration after iteration (iterative models only)."""

    label: str  # what is measured
    iterations: list[int]
    train: list[float] | None
    validation: list[float] | None  # on rows held out of the training rows
    best: int | None  # the iteration kept by early stopping
    lower_is_better: bool


@dataclass
class Importance:
    feature: str
    importance: float  # share of the total (impurity or gain)


@dataclass
class CrossValidation:
    method: str
    folds: int
    mean: dict[str, float | None]
    std: dict[str, float | None]


@dataclass
class FoldPredictions:
    """The out-of-fold predictions of a cross-validated run, row by row: what the Auto's ensemble
    (038) is chosen on, without refitting anything."""

    rows: np.ndarray  # file row numbers of the rows a fold held out
    folds: np.ndarray  # the fold (1, 2…) that held each row out
    actual: np.ndarray
    values: np.ndarray  # (rows, 1) predicted numbers, or (rows, classes) probabilities
    labels: list[str]  # classification: the probability columns, in order
    ranks: np.ndarray | None = None  # panel: each row's period rank


@dataclass
class Trained:
    task: Task
    model: str
    settings: dict[str, Any]
    preparation: prep.Preparation
    train_rows: int
    test_rows: int
    rows: int
    sampled: bool
    seconds: float
    regression: RegressionResult | None
    classification: ClassificationResult | None
    coefficients: list[Coefficient]
    predictions: list[PredictionRow]
    primary: str  # name of the main score
    score: float | None
    classes: binning.Classes | None = None  # a number predicted as classes (018)
    scores: dict[str, float | None] = field(default_factory=dict)  # every metric of the problem
    cross_validation: CrossValidation | None = None
    importances: list[Importance] = field(default_factory=list)
    validation: evaluate.Validation = field(default_factory=evaluate.Validation)
    evaluation_rows: list[int] = field(default_factory=list)  # file rows the scores come from
    learning_curve: LearningCurve | None = None
    pipeline: Any = field(default=None, repr=False)
    panel: panels.PanelResult | None = None  # walk-forward by period (035)
    out_of_fold: panels.OutOfFold | None = field(default=None, repr=False)
    # every held-out prediction: test rows, or out-of-fold (036); kept beside the run
    held_out: error_analysis.HeldOut | None = field(default=None, repr=False)
    # cross-validation only: what the Auto's ensemble is chosen on (038); not kept
    fold_predictions: FoldPredictions | None = field(default=None, repr=False)
    interval: conformal.ConformalResult | None = None  # a number's prediction interval (057)


class TrainingError(ValueError):
    pass


class TrainingCancelled(Exception):
    """The user stopped the training (raised at the next checkpoint)."""


def _quiet(compute):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return compute()


def prepare_target(series: pd.Series, task: Task) -> pd.Series:
    if task == "regression":
        return series.astype(float)
    labels = series.astype(str)
    counts = labels.value_counts()
    if len(counts) > MAX_CLASSES:
        kept = {str(label) for label in list(counts.index)[: MAX_CLASSES - 1]}
        labels = labels.where(labels.isin(kept), OTHER)
    return labels


def _thin(
    curve_x: np.ndarray,
    curve_y: np.ndarray,
    thresholds: np.ndarray | None = None,
    points: int = 200,
) -> Curve:
    index = np.unique(
        np.linspace(0, len(curve_x) - 1, min(points, len(curve_x))).round().astype(int)
    )
    return Curve(
        [float(v) for v in curve_x[index]],
        [float(v) for v in curve_y[index]],
        []
        if thresholds is None
        else [clean(v) or 0.0 for v in thresholds[np.minimum(index, len(thresholds) - 1)]],
    )


@dataclass
class _Setup:
    """Everything shared by training and the preparation preview."""

    spec: catalogue.ModelSpec
    context: catalogue.Context
    values: dict[str, Any]
    preparation: prep.Preparation
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    row_test: pd.Series
    minority: float | None
    unregularized: bool
    task: Task
    target: Field  # with the class order when the classes are ordered
    classes: binning.Classes | None
    validation: evaluate.Validation
    choices: prep.Choices
    groups: pd.Series | None  # group k-fold
    frame: pd.DataFrame  # the rows used (a panel's rows without a period are left out)
    periods: "_Periods | None" = None  # panel walk-forward


@dataclass
class _Periods:
    ranks: np.ndarray  # each row's period rank
    labels: dict[int, str]
    plan: panels.Plan


def panel_roles(validation: evaluate.Validation) -> dict[str, str]:
    """The panel's period and entity, left out of the features (R-SEL-09)."""
    if not validation.panel or validation.period_column is None:
        return {}
    roles = {validation.period_column: "period"}
    if validation.entity_column:
        roles[validation.entity_column] = "entity"
    return roles


def target_folds(validation: evaluate.Validation) -> encoding.TargetFolds:
    """How target encodings are cross-fitted under this validation (R-CAT-08)."""
    return encoding.target_folds(
        validation.method, validation.period_column, validation.group, validation.time_column
    )


GRAINS = [(1.5, "day", 7), (8.0, "week", 13), (32.0, "month", 12), (95.0, "quarter", 4)]


def order_for(frame: pd.DataFrame, validation: evaluate.Validation) -> prep.Order | None:
    """Ordered rows (panel walk-forward or time-based split): the period and entity a lag
    follows, its shift (1 + the embargo) and its window by the grain of the period (R-LAG-01):
    7 days, 13 weeks, 12 months, 4 quarters, 3 years, else 5 periods."""
    if validation.panel and validation.period_column in frame.columns:
        period, entity, embargo = (
            validation.period_column,
            validation.entity_column,
            validation.embargo,
        )
    elif validation.method == "time_split" and validation.time_column in frame.columns:
        period, entity, embargo = validation.time_column, None, 0
    else:
        return None
    assert period is not None
    grain, window = "period", 5
    values = cast(pd.Series, frame[period])
    if pd.api.types.is_datetime64_any_dtype(values):
        moments = pd.Series(pd.to_datetime(values).dropna().unique()).sort_values()
        gaps = moments.diff().dt.total_seconds().dropna() / 86_400
        if len(gaps):
            step = float(gaps.median())
            grain, window = next(
                ((name, size) for limit, name, size in GRAINS if step <= limit),
                ("year", 3),
            )
            if step < 0.9:
                grain, window = "period", 5
    return prep.Order(period, entity, 1 + embargo, window, grain)


def _counts(y: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in y.value_counts().items()}


def _imbalance(
    choices: prep.Choices, minority: float | None, values: dict[str, Any], warnings: list[str]
) -> tuple[str, str, str]:
    """R-IMB-01/02/03 or the user's strategy → (strategy, rule, why). Weights go through the
    model's own class_weight setting, so the run's settings show what was used."""
    weights = "class_weight" in values
    share = f"{minority:.0%}" if minority is not None else "?"
    if choices.imbalance == "auto":
        if weights and values["class_weight"] == "balanced":
            rule = "R-IMB-02" if minority is not None and minority < 0.2 else "USER"
            return (
                "weights",
                rule,
                f"The rarest class is {share} of rows: class weights balance it.",
            )
        if minority is None or minority >= 0.2:
            return "none", "R-IMB-01", f"The rarest class is {share} of rows: nothing to rebalance."
        if weights:
            return "none", "USER", "Class weights are turned off in the model settings."
        return (
            "none",
            "R-IMB-03",
            f"The rarest class is {share} of rows and this model takes no weights: "
            "judge it on balanced accuracy.",
        )
    strategy = choices.imbalance
    if strategy == "weights" and not weights:
        warnings.append("This model takes no class weights: classes left as they are.")
        return "none", "R-IMB-03", "This model takes no class weights."
    if weights:
        values["class_weight"] = "balanced" if strategy == "weights" else "none"
    why = {
        "none": "Classes left as they are: your choice.",
        "weights": "Class weights balance the classes: your choice.",
        "oversample": "Training rows of the smaller classes repeated at random up to the largest "
        "class: your choice (the test rows are untouched).",
        "undersample": "Training rows of the larger classes drawn at random down to the smallest "
        "class: your choice (the test rows are untouched).",
    }[strategy]
    return strategy, "USER", why


def _setup(
    frame: pd.DataFrame,
    fields: list[Field],
    target: Field,
    task: Task,
    model_key: str,
    given: dict[str, Any] | None,
    excluded_columns: set[str] | None,
    choices: prep.Choices | None,
    classes: binning.Classes | None = None,
    validation: evaluate.Validation | None = None,
) -> _Setup:
    choices = choices or prep.Choices()
    validation = validation or evaluate.Validation()
    if validation.panel and validation.period_column in frame.columns:
        frame = cast(pd.DataFrame, frame[frame[validation.period_column].notna()])
    spec = catalogue.CATALOGUE[model_key]
    if not spec.available:
        raise TrainingError(spec.missing())
    if classes is not None:
        y, target, classes = _as_classes(cast(pd.Series, frame[target.name]), target, classes)
        task = "binary" if y.nunique() == 2 else "multiclass"
    else:
        y = prepare_target(cast(pd.Series, frame[target.name]), task)
    if task not in spec.tasks:
        raise TrainingError(f"{spec.label} does not handle {task} problems.")
    if task != "regression" and y.nunique() < 2:
        raise TrainingError("The target has a single class in these rows.")
    if task != "regression":
        counts = y.value_counts()
        if counts.min() < MIN_CLASS_ROWS:
            raise TrainingError(
                f"Class “{counts.idxmin()}” has only {int(counts.min())} "
                f"row{'s' if int(counts.min()) != 1 else ''} "
                f"(at least {MIN_CLASS_ROWS} needed): use fewer classes or move the limits."
            )
    try:
        evaluate.check(validation, task, len(frame), set(frame.columns))
    except evaluate.ValidationError as error:
        raise TrainingError(str(error)) from None
    minority = float(y.value_counts(normalize=True).min()) if task != "regression" else None
    training_rows = (
        len(frame) if validation.cross else round(len(frame) * (1 - validation.test_size))
    )
    context = catalogue.Context(training_rows, task, minority, features=len(fields) - 1)
    reason = spec.limit(context)
    if reason:
        raise TrainingError(reason)
    values = catalogue.resolve(spec, context, given)
    model_profile = catalogue.profile_for(spec, values)
    preparation = prep.decide(
        frame,
        fields,
        target,
        task,
        model_profile,
        excluded_columns,
        choices,
        source_numeric=classes is not None,
        panel_columns=panel_roles(validation),
        folds=target_folds(validation),
        order=order_for(frame, validation),
    )
    if not preparation.used:
        raise TrainingError("No usable feature: every column was left out (see Preparation).")
    repeated = repeated_identifier(frame, fields, validation)
    if repeated is not None:
        preparation.warnings.append(repeated)
    if preparation.folds is not None:
        preparation.fold_keys = encoding.fold_keys(frame, preparation.folds)
    # a lag also reads the period and entity, which stay out of the inputs (R-SEL-09)
    features = [d.column for d in preparation.used] + prep.context_columns(preparation)
    positions = np.arange(len(frame))
    periods = None
    if validation.panel:
        ranks, labels = panels.period_ranks(frame, cast(str, validation.period_column))
        try:
            found = panels.plan(
                ranks,
                labels,
                validation.folds,
                validation.embargo,
                validation.window,
                validation.window_periods,
                validation.sample_periods,
            )
        except evaluate.ValidationError as error:
            raise TrainingError(str(error)) from None
        periods = _Periods(ranks, labels, found)
    if validation.cross:
        train_at, test_at = positions, positions[:0]
    elif validation.method == "time_split":
        train_at, test_at = evaluate.time_split(
            cast(pd.Series, frame[validation.time_column]), validation.test_size
        )
    else:
        stratify = y if task != "regression" and y.value_counts().min() >= 2 else None
        train_at, test_at = train_test_split(
            positions,
            test_size=validation.test_size,
            random_state=validation.seed,
            stratify=stratify,
        )
    x = cast(pd.DataFrame, frame[features])
    x_train, x_test = x.iloc[train_at], x.iloc[test_at]
    y_train, y_test = y.iloc[train_at], y.iloc[test_at]
    row_test = cast(pd.Series, frame["__row"]).iloc[test_at]
    if len(x_train) < 10 or (not validation.cross and len(x_test) < 3):
        raise TrainingError("Too few rows to train and test (at least 15 needed).")
    if not validation.cross:
        x_train, y_train = _fit_rows(
            x_train, y_train, preparation, choices, minority, values, task, record=True
        )
    groups = (
        cast(pd.Series, frame[validation.group])
        if validation.method == "group_kfold" and validation.group
        else None
    )
    return _Setup(
        spec,
        context,
        values,
        preparation,
        x_train,
        x_test,
        y_train,
        y_test,
        row_test,
        minority,
        model_key == "linear_regression" and values.get("regularization") == "none",
        task,
        target,
        classes,
        validation,
        choices,
        groups,
        frame,
        periods,
    )


REPEATED_MIN_VALUES = 20  # R-VAL-01: an identifier with at least this many values…
REPEATED_ROWS = 1.5  # …and at least this many rows per value on average
REPEATED_SHARE = 0.9  # …and at most this share of distinct values
RANDOM_METHODS = ("split", "kfold", "repeated_kfold")


def repeated_identifier(
    frame: pd.DataFrame, fields: list[Field], validation: evaluate.Validation
) -> str | None:
    """R-VAL-01 (094, ML-3): a random split with an identifier that repeats (each match on two
    mirror rows, a patient seen several times) puts the same unit in training and test, which
    flatters the score. Says so, and proposes Group k-fold on it; nothing is changed."""
    if validation.method not in RANDOM_METHODS:
        return None
    best: tuple[int, str, float] | None = None
    for fld in fields:
        if fld.name not in frame.columns:
            continue
        values = cast(pd.Series, frame[fld.name])
        if fld.type != "identifier" and not (
            fld.type in NUMERIC and codes.numeric_code(fld.name, values.to_numpy(dtype=float))
        ):
            continue
        present = values.dropna()
        distinct = int(present.nunique())
        if distinct < REPEATED_MIN_VALUES or not len(present):
            continue
        per_value = len(present) / distinct
        if per_value < REPEATED_ROWS or distinct > REPEATED_SHARE * len(present):
            continue
        if best is None or distinct > best[0]:
            best = (distinct, fld.name, per_value)
    if best is None:
        return None
    _, name, per_value = best
    return (
        f"{name} repeats ({per_value:.1f} rows per value): a random split puts rows of the same "
        f"{name} in training and test, which flatters the score. Choose Group k-fold on {name} "
        "in Validation (R-VAL-01)."
    )


PERFECT = {"r2": 0.995, "roc_auc": 0.999, "balanced_accuracy": 0.995}  # R-LEAK-01


def near_perfect(
    scores: dict[str, float | None],
    cross_validation: "CrossValidation | None",
    preparation: prep.Preparation,
    pipeline: Pipeline,
    names: list[str],
    coefficients: list[Coefficient],
) -> str | None:
    """R-LEAK-01 (094, ML-2): a near-perfect score on real data is more often a column known
    after the outcome than a perfect model. Says so, with the columns the model leans on."""
    found = dict(scores)
    if cross_validation is not None and cross_validation.mean:
        found.update(cross_validation.mean)
    hit = next(
        (
            (metric, value)
            for metric, limit in PERFECT.items()
            if (value := found.get(metric)) is not None and value >= limit
        ),
        None,
    )
    if hit is None:
        return None
    metric, value = hit
    ranked = [i.feature for i in _importances(pipeline, names)] or [
        c.feature for c in coefficients if c.feature != "const"
    ]
    used = sorted((d.column for d in preparation.used), key=len, reverse=True)
    top: list[str] = []
    for feature in ranked:
        origin = next((c for c in used if feature == c or feature.startswith(f"{c}_")), feature)
        if origin not in top:
            top.append(origin)
        if len(top) == 3:
            break
    label = {"r2": "R²", "roc_auc": "ROC AUC", "balanced_accuracy": "balanced accuracy"}[metric]
    leaning = f" It leans most on {', '.join(top)}:" if top else ""
    return (
        f"Near-perfect score ({label} {value:.3f}): real data rarely allows it.{leaning} check "
        "that none of them is known only after the outcome (R-LEAK-01)."
    )


def _fit_rows(
    x: pd.DataFrame,
    y: pd.Series,
    preparation: prep.Preparation,
    choices: prep.Choices,
    minority: float | None,
    values: dict[str, Any],
    task: Task,
    record: bool,
) -> tuple[pd.DataFrame, pd.Series]:
    """The training rows of one fit: extreme rows dropped, classes rebalanced (017), never the
    held-out rows. `record` writes what was done into the preparation (final fit only)."""
    if choices.outliers == "drop_rows":
        numbers = [d.column for d in preparation.used if d.type in NUMERIC]
        extreme = steps.extreme_rows(x, numbers)
        if (~extreme).sum() < 10:
            if record:
                preparation.warnings.append(
                    "Dropping extreme rows would leave too few: none dropped."
                )
        elif extreme.any():
            x, y = cast(pd.DataFrame, x[~extreme]), cast(pd.Series, y[~extreme])
            if record:
                preparation.dropped_rows = int(extreme.sum())
    if task != "regression":
        before = _counts(y)
        warnings: list[str] = []
        strategy, rule, why = _imbalance(choices, minority, values, warnings)
        if strategy in ("oversample", "undersample"):
            x, y = steps.rebalance(x, y, strategy)
        if record:
            preparation.warnings += warnings
            preparation.balance = prep.Balance(strategy, rule, why, before, _counts(y))
    return x, y


def _as_classes(
    values: pd.Series, target: Field, classes: binning.Classes
) -> tuple[pd.Series, Field, binning.Classes]:
    """The number cut into the given classes; empty classes are left out of the order."""
    labels = binning.apply(values, classes)
    counts = [int((labels == name).sum()) for name in classes.names]
    used = [name for name, count in zip(classes.names, counts, strict=True) if count]
    if len(used) < 2:
        raise TrainingError("Fewer than two classes have rows: move the limits.")
    ordered = Field(target.name, "categorical_ordinal", target.parser, target.role, used)
    return labels.astype(str), ordered, classes.model_copy(update={"counts": counts})


def _selection(pipeline: Pipeline) -> list[prep.SelectionReport]:
    return [
        prep.SelectionReport(
            step.method,
            step.value if step.value is not None else steps.SELECTION_DEFAULTS[step.method],
            int(step.support_.sum()),
            int(step.n_features_in_),
        )
        for _, step in pipeline.steps
        if isinstance(step, steps.Selector)
    ]


def train(
    frame: pd.DataFrame,
    fields: list[Field],
    target: Field,
    task: Task,
    model_key: str,
    given: dict[str, Any] | None,
    rows: int,
    sampled: bool,
    excluded_columns: set[str] | None = None,
    choices: prep.Choices | None = None,
    classes: binning.Classes | None = None,
    validation: evaluate.Validation | None = None,
    should_stop: Callable[[], bool] | None = None,
    build: Callable[[int], Pipeline] | None = None,
) -> Trained:
    """`build(rows)`, when given, replaces the catalogue's pipeline of `model_key` (the Auto's
    ensemble or constant model, 038): `model_key` still decides the preparation of the rows."""
    started = time.monotonic()

    def checkpoint() -> None:
        if should_stop is not None and should_stop():
            raise TrainingCancelled

    setup = _setup(
        frame,
        fields,
        target,
        task,
        model_key,
        given,
        excluded_columns,
        choices,
        classes,
        validation,
    )
    task, target, validation = setup.task, setup.target, setup.validation
    spec, context, values, preparation = setup.spec, setup.context, setup.values, setup.preparation
    minority, unregularized = setup.minority, setup.unregularized
    regression_task = task == "regression"

    def make(n_rows: int) -> Pipeline:
        if build is not None:
            return build(n_rows)
        return model_pipeline(spec, values, context, preparation, task, n_rows, unregularized)

    monotonic = _monotonic(values, preparation)

    def fit(pipeline: Pipeline, x: pd.DataFrame, y: pd.Series) -> None:
        checkpoint()
        _quiet(lambda: fit_pipeline(pipeline, x, y, monotonic))

    cross_validation = fold_predictions = None
    evaluated: np.ndarray | None = None  # rows a fold held out (walk-forward: not the first)
    fold_of = np.zeros(0, dtype=int)
    if validation.cross:
        x_all, y_all = setup.x_train, setup.y_train
        labels = sorted(str(v) for v in y_all.unique()) if not regression_task else []
        totals = np.zeros((len(y_all), max(1, len(labels))))
        seen = np.zeros(len(y_all))
        fold_of = np.zeros(len(y_all), dtype=int)
        per_fold: list[dict[str, float | None]] = []
        splits = (
            setup.periods.plan.folds
            if setup.periods is not None
            else evaluate.folds(validation, y_all, setup.groups, not regression_task)
        )
        for number, (fit_at, held_at) in enumerate(splits, start=1):
            x_fit, y_fit = _fit_rows(
                x_all.iloc[fit_at],
                y_all.iloc[fit_at],
                preparation,
                setup.choices,
                minority,
                values,
                task,
                record=False,
            )
            fold = make(len(x_fit))
            fit(fold, x_fit, y_fit)
            held = x_all.iloc[held_at]
            actual_held = y_all.iloc[held_at].to_numpy()
            if regression_task:
                guess = np.asarray(fold.predict(held), dtype=float)
                totals[held_at, 0] += guess
                fold_scores = evaluate.regression_metrics(actual_held.astype(float), guess)
            else:
                chances = _aligned(fold, held, labels)
                totals[held_at] += chances
                guess_labels = np.asarray(labels)[chances.argmax(axis=1)]
                positive = _positive(y_all) if len(labels) == 2 else None
                fold_scores = evaluate.classification_metrics(
                    actual_held.astype(str), guess_labels, chances, labels, positive
                )
            seen[held_at] += 1
            fold_of[held_at] = number
            if validation.method != "loo":
                per_fold.append(fold_scores)
        evaluated = seen > 0
        out_of_fold = totals[evaluated] / seen[evaluated][:, None]
        final_at = setup.periods.plan.final if setup.periods is not None else np.arange(len(y_all))
        x_train, y_train = _fit_rows(
            cast(pd.DataFrame, x_all.iloc[final_at]),
            cast(pd.Series, y_all.iloc[final_at]),
            preparation,
            setup.choices,
            minority,
            values,
            task,
            record=True,
        )
        pipeline = make(len(x_train))
        fit(pipeline, x_train, y_train)
        y_eval = cast(pd.Series, y_all[evaluated])
        rows_eval = cast(pd.Series, setup.frame["__row"][evaluated])
        mean, spread = evaluate.summarise(per_fold)
        cross_validation = CrossValidation(
            validation.method, len(per_fold) or len(y_all), mean, spread
        )
        predicted = out_of_fold[:, 0] if regression_task else None
        probabilities = None if regression_task else out_of_fold
        class_names = labels
        test_rows = len(y_eval)
        fold_predictions = FoldPredictions(
            rows_eval.to_numpy(dtype=int),
            fold_of[evaluated],
            y_eval.to_numpy(),
            out_of_fold,
            labels,
            setup.periods.ranks[evaluated] if setup.periods is not None else None,
        )
    else:
        x_train, y_train = setup.x_train, setup.y_train
        pipeline = make(len(x_train))
        fit(pipeline, x_train, y_train)
        y_eval, rows_eval = setup.y_test, setup.row_test
        class_names = [] if regression_task else [str(c) for c in pipeline.classes_]
        predicted = (
            np.asarray(pipeline.predict(setup.x_test), dtype=float) if regression_task else None
        )
        probabilities = (
            None
            if regression_task
            else np.asarray(pipeline.predict_proba(setup.x_test), dtype=float)
        )
        test_rows = len(setup.x_test)
    names = [str(n) for n in pipeline[:-1].get_feature_names_out()]  # type: ignore[union-attr]
    preparation.inputs = len(names)
    preparation.selection = _selection(pipeline)
    threshold, positive = None, None
    if validation.threshold != "default" and task == "binary":
        checkpoint()
        positive = _positive(y_train)
        threshold = _threshold(make, x_train, y_train, positive, validation, monotonic)
    checkpoint()

    regression = classification = None
    interval: conformal.ConformalResult | None = None
    coefficients: list[Coefficient] = []
    predictions: list[PredictionRow] = []
    rng = np.random.default_rng(SEED)
    if predicted is not None:
        actual = y_eval.to_numpy(dtype=float)
        regression = _regression_scores(actual, predicted, y_train.to_numpy(dtype=float))
        scores = evaluate.regression_metrics(actual, predicted)
        if build is not None:
            pass  # not a catalogue model: no coefficients to read
        elif model_key == "linear_regression":
            coefficients, regression.heteroscedasticity_p = _linear_coefficients(
                pipeline, x_train, y_train.to_numpy(dtype=float), names, preparation, unregularized
            )
        elif spec.linear:
            coefficients = _model_coefficients(pipeline, names)
        pick = rng.permutation(len(actual))[:PREDICTION_ROWS]
        predictions = [
            PredictionRow(
                int(rows_eval.iloc[i]),
                _fmt(float(actual[i])),
                _fmt(float(predicted[i])),
                clean(predicted[i] - actual[i]),
            )
            for i in pick
        ]
        held_out = error_analysis.HeldOut(rows_eval.to_numpy(dtype=int), actual, predicted)
        checkpoint()
        interval = _interval(
            setup,
            pipeline,
            x_train,
            y_train,
            actual,
            predicted,
            setup.periods.ranks[evaluated]
            if setup.periods is not None and evaluated is not None
            else None,
        )
    else:
        assert probabilities is not None
        classification, predicted_labels = _classification_scores(
            probabilities, class_names, y_eval, target, threshold, positive
        )
        scores = evaluate.classification_metrics(
            y_eval.to_numpy().astype(str),
            predicted_labels,
            probabilities,
            class_names,
            classification.positive,
        )
        if spec.linear and build is None:
            coefficients = _model_coefficients(pipeline, names)
        actual_labels = y_eval.to_numpy()
        pick = rng.permutation(len(actual_labels))[:PREDICTION_ROWS]
        column = {label: i for i, label in enumerate(class_names)}
        predictions = [
            PredictionRow(
                int(rows_eval.iloc[i]),
                str(actual_labels[i]),
                str(predicted_labels[i]),
                probability=clean(probabilities[i, column[str(predicted_labels[i])]]),
                correct=bool(str(actual_labels[i]) == str(predicted_labels[i])),
            )
            for i in pick
        ]
        held_out = error_analysis.HeldOut(
            rows_eval.to_numpy(dtype=int),
            actual_labels.astype(str),
            np.asarray(predicted_labels, dtype=str),
            probabilities,
            class_names,
        )
    primary = validation.metric or evaluate.default_metric(task, minority, validation.panel)
    score = (
        cross_validation.mean.get(primary)
        if cross_validation is not None and cross_validation.mean
        else scores.get(primary)
    )
    panel = out_of_fold_kept = None
    if setup.periods is not None and evaluated is not None:
        checkpoint()
        positive_class = classification.positive if classification is not None else None
        panel, out_of_fold_kept = _panel(
            setup,
            evaluated,
            fold_of[evaluated],
            primary,
            y_eval,
            predicted,
            probabilities,
            class_names,
            positive_class,
            rows_eval,
        )
        score = panel.value
    perfect = near_perfect(scores, cross_validation, preparation, pipeline, names, coefficients)
    if perfect is not None:
        preparation.warnings.append(perfect)
    return Trained(
        task,
        model_key,
        values,
        preparation,
        len(x_train),
        test_rows,
        rows,
        sampled,
        time.monotonic() - started,
        regression,
        classification,
        coefficients,
        predictions,
        primary,
        score,
        setup.classes,
        scores,
        cross_validation,
        _importances(pipeline, names),
        validation,
        _evaluation_rows(
            rows_eval, [p.row for p in predictions] + error_analysis.worst_rows(held_out)
        ),
        _learning_curve(pipeline.named_steps["model"]),
        pipeline,
        panel,
        out_of_fold_kept,
        held_out,
        fold_predictions,
        interval,
    )


def _interval(
    setup: "_Setup",
    pipeline: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    actual: np.ndarray,
    predicted: np.ndarray,
    ranks: np.ndarray | None,
) -> conformal.ConformalResult | None:
    """The prediction interval of a number (057), set on the held-out rows: split conformal ★,
    or CQR when asked and possible (R-ML-CONF-04, 05)."""
    validation = setup.validation
    level = validation.interval_level
    order = None
    if ranks is not None:  # panel: out-of-fold rows put back in period order (R-ML-CONF-02)
        order = np.argsort(ranks, kind="stable")
        actual, predicted = actual[order], predicted[order]
    chronological = order is not None or validation.method == "time_split"
    fallback: str | None = None
    if validation.interval_method == "cqr":
        if validation.cross:
            fallback = conformal.fallback_rule("cross")
        else:
            prepare = pipeline[:-1]
            train_inputs = prepare.transform(x_train)
            if train_inputs.shape[1] > conformal.MAX_CQR_INPUTS:
                fallback = conformal.fallback_rule("inputs")
            else:
                lower, upper = _quiet(
                    lambda: conformal.fit_bands(
                        train_inputs, y_train.to_numpy(dtype=float), level, validation.seed
                    )
                )
                found = conformal.cqr(
                    lower,
                    upper,
                    prepare.transform(setup.x_test),
                    actual,
                    level,
                    chronological,
                    validation.seed,
                )
                if found is not None:
                    result, bands = found
                    setattr(pipeline, conformal.BANDS_ATTRIBUTE, bands)
                    return result
    result = conformal.split(actual, predicted, level, chronological, validation.seed)
    if result is not None and fallback is not None:
        result.rules.append(fallback)
    return result


def _panel(
    setup: _Setup,
    evaluated: np.ndarray,
    folds: np.ndarray,
    metric: str,
    actual: pd.Series,
    predicted: np.ndarray | None,
    probabilities: np.ndarray | None,
    classes: list[str],
    positive: str | None,
    rows: pd.Series,
) -> tuple[panels.PanelResult, panels.OutOfFold]:
    """Walk-forward on a panel (035): every test period scored on its own, the summary and the
    run's score (the chosen criterion), the columns the predictions follow, the panel leaks."""
    validation, periods = setup.validation, cast(_Periods, setup.periods)
    period = cast(str, validation.period_column)
    entity = validation.entity_column
    ranks = periods.ranks[evaluated]
    if predicted is not None:
        truth = actual.to_numpy(dtype=float)
        scorer = panels.regression_scorer(metric, truth, predicted)
        score_vector: np.ndarray | None = predicted
    else:
        assert probabilities is not None
        labels_of = np.asarray(classes)[probabilities.argmax(axis=1)].astype(str)
        scorer = panels.classification_scorer(
            metric, actual.to_numpy().astype(str), labels_of, probabilities, classes, positive
        )
        score_vector = (
            probabilities[:, classes.index(positive)]
            if positive is not None and len(classes) == 2
            else None
        )
    scored = panels.score_periods(ranks, folds, periods.labels, scorer)
    summary = panels.summarise(scored, metric, validation.period_threshold)
    value = panels.criterion_value(summary, validation.criterion)
    numbers = [d.column for d in setup.preparation.used if d.type in NUMERIC]
    evaluated_frame = cast(pd.DataFrame, setup.frame[evaluated])
    shown, largest = (
        panels.top_exposures(score_vector, evaluated_frame, numbers, ranks)
        if score_vector is not None
        else ([], None)
    )
    alerts: list[panels.PanelAlert] = []
    target = panels.numeric_target(cast(pd.Series, setup.frame[setup.target.name]))
    if target is not None:
        alerts += panels.future_columns(
            setup.frame, numbers, target, periods.ranks, entity, period, periods.labels
        )
        baseline = panels.baseline_score(target[evaluated], evaluated_frame, numbers, ranks)
        if baseline is not None and score_vector is not None and metric in SCORE_METRICS:
            reference = panels.summarise(
                panels.score_periods(
                    ranks, folds, periods.labels, _score_only(metric, actual, baseline, positive)
                ),
                metric,
            )
            found = panels.suspicious(metric, summary.mean, reference.mean, period)
            if found is not None:
                alerts.append(found)
    suggested = reason = None
    if entity is not None and target is not None:
        ahead = panels.horizon(cast(pd.Series, setup.frame[entity]), periods.ranks, target, entity)
        if ahead is not None:
            suggested, reason = ahead
            if validation.embargo < suggested:
                alerts.append(
                    panels.PanelAlert(
                        "embargo",
                        f"The target looks about {suggested} periods ahead: an embargo of "
                        f"{suggested} (now {validation.embargo}) keeps overlapping targets out "
                        "of training.",
                    )
                )
    result = panels.PanelResult(
        period,
        entity,
        validation.window,
        validation.embargo,
        periods.plan.step,
        validation.criterion,
        value,
        scored,
        summary,
        periods.plan.spans,
        shown,
        largest,
        len(numbers),
        alerts,
        suggested,
        reason,
    )
    binary = score_vector is not None and predicted is None
    kept = panels.OutOfFold(
        rows.to_numpy(dtype=int),
        ranks,
        folds,
        score_vector if score_vector is not None else np.zeros(len(ranks)),
        actual.to_numpy(),
        positive if binary else None,
        next((c for c in classes if c != positive), None) if binary else None,
    )
    return result, kept


# metrics computed from a score alone: the 3-best-columns reference is scored on them
SCORE_METRICS = {"r2", "pearson", "spearman", "roc_auc"}


def _score_only(
    metric: str, actual: pd.Series, score: np.ndarray, positive: str | None
) -> panels.Scorer:
    if positive is None:
        return panels.regression_scorer(metric, actual.to_numpy(dtype=float), score)
    truth = actual.to_numpy().astype(str)
    chances = np.column_stack([-score, score])
    return panels.classification_scorer(
        metric, truth, truth, chances, ["(other)", positive], positive
    )


EVALUATION_ROWS = 10_000


def _evaluation_rows(rows: pd.Series, shown: list[int]) -> list[int]:
    """The rows the scores come from (at most 10 000, drawn with a seed), always including the
    rows of the predictions table and the worst predictions (036) so each can be explained."""
    every = rows.to_numpy(dtype=int)
    shown = list(dict.fromkeys(shown))
    if len(every) <= EVALUATION_ROWS:
        return [int(v) for v in every]
    rest = np.setdiff1d(every, shown)
    extra = np.random.default_rng(SEED).choice(rest, EVALUATION_ROWS - len(shown), replace=False)
    return [*shown, *(int(v) for v in extra)]


def _monotonic(values: dict[str, Any], preparation: prep.Preparation) -> dict[str, int]:
    """ "size:+1, mileage:-1" → {"size": 1, "mileage": -1} (numeric columns that are used)."""
    text = str(values.get("monotonic") or "").strip()
    if not text:
        return {}
    usable = {d.column for d in preparation.used if d.type in NUMERIC}
    found: dict[str, int] = {}
    for part in text.replace(";", ",").split(","):
        if not part.strip():
            continue
        column, _, sign = part.rpartition(":")
        column, sign = column.strip(), sign.strip()
        if sign not in ("+1", "1", "-1", "+", "-") or not column:
            raise TrainingError(
                f"Monotonic constraints: “{part.strip()}” should read column:+1 or column:-1."
            )
        if column not in usable:
            preparation.warnings.append(
                f"Monotonic constraint on {column} ignored: not a number used by the model."
            )
            continue
        found[column] = -1 if sign.startswith("-") else 1
    return found


def fit_pipeline(
    pipeline: Pipeline, x: pd.DataFrame, y: pd.Series, monotonic: dict[str, int] | None = None
) -> Pipeline:
    """Fit the pipeline; with monotonic constraints, the preparation first, then the model
    with one constraint per prepared input (0 = free)."""
    if not monotonic:
        return pipeline.fit(x, y)
    prepared = pipeline[:-1].fit_transform(x, y)
    names = [str(n) for n in pipeline[:-1].get_feature_names_out()]  # type: ignore[union-attr]
    vector = [monotonic.get(n, 0) for n in names]
    model = pipeline.named_steps["model"]
    if hasattr(model, "monotonic_cst"):
        model.set_params(monotonic_cst=vector)
    elif getattr(model, "library", None) == "xgboost":
        model.estimator.set_params(monotone_constraints="(" + ",".join(map(str, vector)) + ")")
    elif getattr(model, "library", None) == "lightgbm":
        model.estimator.set_params(monotone_constraints=vector)
    model.fit(prepared, y)
    return pipeline


THRESHOLDS = np.round(np.linspace(0.02, 0.98, 49), 2)


def _threshold(
    make: Callable[[int], Pipeline],
    x: pd.DataFrame,
    y: pd.Series,
    positive: str,
    validation: evaluate.Validation,
    monotonic: dict[str, int],
) -> float:
    """The probability threshold that maximises F1 or balanced accuracy on out-of-fold
    predictions of the training rows (3 folds): the test rows never choose it."""
    folds = StratifiedKFold(3, shuffle=True, random_state=validation.seed)
    chance = np.zeros(len(y))
    truth = (y.astype(str) == positive).to_numpy()
    for fit_at, held_at in folds.split(x, y):
        model = make(len(fit_at))
        _quiet(
            lambda model=model, fit_at=fit_at: fit_pipeline(
                model, x.iloc[fit_at], y.iloc[fit_at], monotonic
            )
        )
        found = np.asarray(model.predict_proba(x.iloc[held_at]), dtype=float)
        chance[held_at] = found[:, [str(c) for c in model.classes_].index(positive)]
    score = skm.f1_score if validation.threshold == "f1" else skm.balanced_accuracy_score
    results = [float(score(truth, chance >= t)) for t in THRESHOLDS]
    return float(THRESHOLDS[int(np.argmax(results))])


def _learning_curve(model: Any) -> LearningCurve | None:
    """The iteration history the fitted model kept, if any."""
    name = model.__class__.__name__
    if name.startswith("HistGradientBoosting") and len(getattr(model, "train_score_", [])) > 1:
        train = np.asarray(model.train_score_, dtype=float)
        valid = np.asarray(getattr(model, "validation_score_", []), dtype=float)
        # scikit-learn keeps negative losses for numbers: shown as positive losses
        loss = bool(train.max() <= 0)
        if loss:
            train, valid = -train, -valid
        best = (int(np.argmin(valid)) if loss else int(np.argmax(valid))) if len(valid) else None
        return LearningCurve(
            "Loss (lower is better)" if loss else "Score (higher is better)",
            list(range(len(train))),
            [float(v) for v in train],
            [float(v) for v in valid] if len(valid) else None,
            best,
            loss,
        )
    if name.startswith("GradientBoosting") and len(getattr(model, "train_score_", [])) > 1:
        train = [float(v) for v in model.train_score_]
        return LearningCurve(
            "Training loss", list(range(1, len(train) + 1)), train, None, len(train), True
        )
    if name.startswith("MLP") and len(getattr(model, "loss_curve_", [])) > 1:
        valid = getattr(model, "validation_scores_", None)
        best = int(np.argmax(valid)) + 1 if valid else None
        return LearningCurve(
            "Loss (training) · score (held out)",
            list(range(1, len(model.loss_curve_) + 1)),
            [float(v) for v in model.loss_curve_],
            [float(v) for v in valid] if valid else None,
            best,
            True,
        )
    library = getattr(model, "library", None)
    inner = getattr(model, "model_", None)
    if library in ("xgboost", "lightgbm") and inner is not None:
        history = inner.evals_result() if library == "xgboost" else inner.evals_result_
        if history:
            metric, values = next(iter(next(iter(history.values())).items()))
            lower = not any(k in metric for k in ("auc", "ndcg", "map"))
            return LearningCurve(
                f"{metric} on the held-out training rows",
                list(range(1, len(values) + 1)),
                None,
                [float(v) for v in values],
                getattr(model, "best_iteration_", None),
                lower,
            )
    return None


def model_pipeline(
    spec: catalogue.ModelSpec,
    values: dict[str, Any],
    context: catalogue.Context,
    preparation: prep.Preparation,
    task: Task,
    rows: int,
    drop_first: bool = False,
) -> Pipeline:
    """Preparation steps then the model (calibrated when it gives no probabilities)."""
    estimator = spec.build(values, context)
    if task != "regression" and not spec.has_probabilities(values):
        estimator = CalibratedClassifierCV(estimator, cv=3)
    return Pipeline(
        [*prep.steps(preparation, rows, task, drop_first=drop_first), ("model", estimator)]
    )


def _positive(y: pd.Series) -> str:
    """The rarer class of two: the one whose probability is shown."""
    return str(y.value_counts().index[-1])


def _aligned(pipeline: Pipeline, x: pd.DataFrame, labels: list[str]) -> np.ndarray:
    """Probabilities in the order of `labels` (a fold may miss a rare class)."""
    found = np.asarray(pipeline.predict_proba(x), dtype=float)
    out = np.zeros((len(x), len(labels)))
    for j, label in enumerate(str(c) for c in pipeline.classes_):
        out[:, labels.index(label)] = found[:, j]
    return out


def _importances(pipeline: Pipeline, names: list[str]) -> list[Importance]:
    model = pipeline.named_steps["model"]
    raw = getattr(model, "feature_importances_", None)
    if raw is None:
        return []
    values = np.asarray(raw, dtype=float)
    if len(values) != len(names) or not np.isfinite(values).all() or values.sum() <= 0:
        return []
    values = values / values.sum()
    order = np.argsort(-values)[:50]
    return [Importance(names[i], float(values[i])) for i in order]


def _fmt(value: float) -> str:
    return f"{value:.6g}"


def _regression_scores(
    actual: np.ndarray, predicted: np.ndarray, train_y: np.ndarray
) -> RegressionResult:
    baseline = np.full_like(actual, train_y.mean())
    mae = float(skm.mean_absolute_error(actual, predicted))
    baseline_mae = float(skm.mean_absolute_error(actual, baseline))
    positive = bool((actual > 0).all())
    residuals = actual - predicted
    counts, edges = np.histogram(residuals, bins=min(30, max(5, len(residuals) // 5)))
    pick = np.random.default_rng(SEED).permutation(len(actual))[:POINTS]
    return RegressionResult(
        clean(skm.r2_score(actual, predicted)),
        clean(mae),
        clean(math.sqrt(skm.mean_squared_error(actual, predicted))),
        clean(skm.mean_absolute_percentage_error(actual, predicted) * 100) if positive else None,
        clean(baseline_mae),
        clean(math.sqrt(skm.mean_squared_error(actual, baseline))),
        clean(1 - mae / baseline_mae) if baseline_mae else None,
        [float(v) for v in actual[pick]],
        [float(v) for v in predicted[pick]],
        [
            {"start": float(edges[i]), "end": float(edges[i + 1]), "count": int(c)}
            for i, c in enumerate(counts)
        ],
    )


MAX_BREUSCH_PAGAN_INPUTS = 200  # beyond, the test of constant variance is not run (094, ML-6)
NORMAL_EQUATIONS_ABOVE = 2e9  # rows x inputs²: beyond, standard errors from (XᵀX)⁻¹ (094)


def _normal_inference(coefficients: list[Coefficient], exog: np.ndarray, y: np.ndarray) -> None:
    """Standard errors, 95 % intervals and p-values of an OLS fit from the normal equations,
    σ² (XᵀX)⁻¹: the same as statsmodels on a full-rank design, without the SVD of the whole
    design (40 s on 52 000 rows by 400 inputs, R-ML-06)."""
    from scipy import stats as st

    gram = exog.T @ exog
    inverse = np.linalg.pinv(gram)
    beta = inverse @ (exog.T @ y)
    residuals = y - exog @ beta
    rank = int(np.linalg.matrix_rank(gram))
    freedom = max(exog.shape[0] - rank, 1)
    sigma2 = float(residuals @ residuals) / freedom
    errors = np.sqrt(np.maximum(np.diag(inverse) * sigma2, 0))
    critical = float(st.t.ppf(0.975, freedom))
    for i, coefficient in enumerate(coefficients, start=1):
        error = float(errors[i])
        coefficient.std_error = clean(error)
        coefficient.low = clean(float(beta[i]) - critical * error)
        coefficient.high = clean(float(beta[i]) + critical * error)
        if error > 0:
            coefficient.p_value = clean(float(2 * st.t.sf(abs(beta[i]) / error, freedom)))


def _linear_coefficients(
    pipeline: Pipeline,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    names: list[str],
    preparation: prep.Preparation,
    unregularized: bool,
) -> tuple[list[Coefficient], float | None]:
    design = np.asarray(pipeline[:-1].transform(x_train), dtype=float)
    model = pipeline.named_steps["model"]
    coefficients = [
        Coefficient(name, float(value))
        for name, value in zip(names, np.ravel(model.coef_), strict=True)
    ]
    per_unit = _unit_scales(pipeline, preparation)
    for coefficient in coefficients:
        if coefficient.feature in per_unit:
            coefficient.per_unit = coefficient.estimate / per_unit[coefficient.feature]
    het_p = None
    if unregularized and design.shape[0] > design.shape[1] + 2:
        exog = sm.add_constant(design, has_constant="add")
        if exog.shape[0] * exog.shape[1] ** 2 > NORMAL_EQUATIONS_ABOVE:
            _normal_inference(coefficients, exog, y_train)
            preparation.warnings.append(
                f"Constant-variance test (Breusch-Pagan) not run: {design.shape[0]:,} rows by "
                f"{design.shape[1]} inputs would take as long as the model (R-ML-06)."
            )
            return sorted(coefficients, key=lambda c: -abs(c.estimate)), None
        fit = _quiet(lambda: sm.OLS(y_train, exog).fit())
        intervals = np.asarray(fit.conf_int(), dtype=float)
        for i, coefficient in enumerate(coefficients, start=1):
            coefficient.std_error = clean(fit.bse[i])
            coefficient.low, coefficient.high = clean(intervals[i, 0]), clean(intervals[i, 1])
            coefficient.p_value = clean(fit.pvalues[i])
        if len(coefficients) <= MAX_VIF_FEATURES:
            for i, coefficient in enumerate(coefficients, start=1):
                coefficient.vif = clean(_quiet(lambda i=i: variance_inflation_factor(exog, i)))
        if design.shape[1] <= MAX_BREUSCH_PAGAN_INPUTS:  # its auxiliary fit costs as much again
            het_p = clean(_quiet(lambda: het_breuschpagan(fit.resid, exog)[1]))
        else:
            preparation.warnings.append(
                f"Constant-variance test (Breusch-Pagan) not run: {design.shape[1]} inputs, more "
                f"than {MAX_BREUSCH_PAGAN_INPUTS}; it would take as long as the model (R-ML-06)."
            )
    coefficients.sort(key=lambda c: -abs(c.estimate))
    return coefficients, het_p


def _unit_scales(pipeline: Pipeline, preparation: prep.Preparation) -> dict[str, float]:
    """Standardised numbers without transformation: the scale that turns a coefficient back into
    'per unit of the column'."""
    scales = {}
    fitted = pipeline.named_steps["prepare"]
    for decision in preparation.used:
        if (
            decision.type in NUMERIC
            and decision.scaling == "standard"
            and decision.transform is None
        ):
            scaler = prep.fitted_step(fitted, decision.column).named_steps["scale"]
            scale = float(scaler.scale_[0]) if hasattr(scaler, "scale_") else 0.0
            if scale > 0:
                scales[decision.column] = scale
    return scales


def _classification_scores(
    probabilities: np.ndarray,
    classes: list[str],
    y_test: pd.Series,
    target: Field,
    threshold: float | None = None,
    chosen_positive: str | None = None,
) -> tuple["ClassificationResult", np.ndarray]:
    predicted = np.asarray(classes)[probabilities.argmax(axis=1)].astype(str)
    if threshold is not None and chosen_positive in classes and len(classes) == 2:
        other = next(c for c in classes if c != chosen_positive)
        chance = probabilities[:, classes.index(chosen_positive)]
        predicted = np.where(chance >= threshold, chosen_positive, other).astype(str)
    actual = y_test.to_numpy().astype(str)
    labels = classes
    if target.order:
        rank = {label: i for i, label in enumerate(target.order)}
        labels = sorted(classes, key=lambda c: rank.get(c, len(rank)))
    counts = y_test.value_counts()
    majority = str(counts.index[0])
    report = np.asarray(
        skm.precision_recall_fscore_support(actual, predicted, labels=labels, zero_division=0.0),  # type: ignore[arg-type]
        dtype=float,
    )
    result = ClassificationResult(
        labels,
        None,
        clean(skm.accuracy_score(actual, predicted)),
        clean(skm.balanced_accuracy_score(actual, predicted)),
        clean(skm.f1_score(actual, predicted, average="macro", zero_division=0.0)),  # type: ignore[arg-type]
        None,
        clean(_quiet(lambda: skm.log_loss(actual, probabilities, labels=classes))),
        clean(float(counts.iloc[0] / counts.sum())),
        majority,
        skm.confusion_matrix(actual, predicted, labels=labels).tolist(),
        [
            ClassReport(
                label,
                float(report[0][i]),
                float(report[1][i]),
                float(report[2][i]),
                int(report[3][i]),
            )
            for i, label in enumerate(labels)
        ],
    )
    if len(classes) == 2:
        positive = chosen_positive or str(counts.index[-1])  # the rarer class
        column = classes.index(positive)
        truth = (actual == positive).astype(int)
        scores = probabilities[:, column]
        if 0 < truth.sum() < len(truth):
            result.roc_auc = clean(skm.roc_auc_score(truth, scores))
            fpr, tpr, thresholds = skm.roc_curve(truth, scores)
            result.roc = _thin(fpr, tpr, thresholds)
            precision, recall, _ = skm.precision_recall_curve(truth, scores)
            result.pr = _thin(recall[::-1], precision[::-1])
            bins = min(10, max(2, len(truth) // 20))
            observed, expected = calibration_curve(truth, scores, n_bins=bins, strategy="quantile")
            result.calibration = Curve([float(v) for v in expected], [float(v) for v in observed])
        result.positive = positive
        result.threshold = threshold if threshold is not None else 0.5
        pick = np.random.default_rng(SEED).permutation(len(truth))[:PROBABILITIES]
        result.truth = [int(v) for v in truth[pick]]
        result.probability = [float(v) for v in scores[pick]]
    else:
        if len(set(actual)) == len(classes):
            result.roc_auc = clean(
                _quiet(
                    lambda: skm.roc_auc_score(
                        actual, probabilities, multi_class="ovr", labels=classes
                    )
                )
            )
        result.class_curves = _class_curves(actual, probabilities, classes, labels)
    if target.order and len(labels) > 2:
        rank = {label: i for i, label in enumerate(target.order)}
        distance = np.array(
            [abs(rank.get(a, 0) - rank.get(p, 0)) for a, p in zip(actual, predicted, strict=True)]
        )
        result.within_one = clean(float(np.mean(distance <= 1)))
        result.mean_class_error = clean(float(np.mean(distance)))
    return result, predicted


def _class_curves(
    actual: np.ndarray, probabilities: np.ndarray, classes: list[str], labels: list[str]
) -> list[ClassCurves]:
    """One-vs-rest curves of every class present among the test rows, in display order."""
    curves = []
    for label in labels:
        if label not in classes:
            continue
        truth = (actual == label).astype(int)
        if not 0 < truth.sum() < len(truth):
            continue  # a class absent from the test rows (or alone) has no curve
        scores = probabilities[:, classes.index(label)]
        fpr, tpr, thresholds = skm.roc_curve(truth, scores)
        precision, recall, _ = skm.precision_recall_curve(truth, scores)
        calibration = None
        if truth.sum() >= 10:
            bins = min(10, max(2, int(truth.sum()) // 10))
            observed, expected = _quiet(
                lambda t=truth, s=scores, b=bins: calibration_curve(
                    t, s, n_bins=b, strategy="quantile"
                )
            )
            calibration = Curve([float(v) for v in expected], [float(v) for v in observed])
        curves.append(
            ClassCurves(
                label,
                float(truth.mean()),
                clean(skm.roc_auc_score(truth, scores)),
                _thin(fpr, tpr, thresholds),
                _thin(recall[::-1], precision[::-1]),
                calibration,
            )
        )
    return curves


def _model_coefficients(pipeline: Pipeline, names: list[str]) -> list[Coefficient]:
    """The weights of a linear model on the prepared inputs (odds ratios for a logistic one)."""
    model = pipeline.named_steps["model"]
    raw = getattr(model, "coef_", None)
    if raw is None:
        return []
    coef = np.atleast_2d(np.asarray(raw, dtype=float))
    if coef.shape[1] != len(names):
        return []
    logistic = model.__class__.__name__ == "LogisticRegression"
    if coef.shape[0] == 1:
        found = [
            Coefficient(n, float(v), odds_ratio=clean(math.exp(min(v, 50))) if logistic else None)
            for n, v in zip(names, coef[0], strict=True)
        ]
    else:  # multiclass: the largest weight across classes
        strongest = coef[np.abs(coef).argmax(axis=0), np.arange(coef.shape[1])]
        found = [Coefficient(n, float(v)) for n, v in zip(names, strongest, strict=True)]
    found.sort(key=lambda c: -abs(c.estimate))
    return found


PREVIEW_ROWS = 20


@dataclass
class PreparationPreview:
    """The matrix the model really receives: the first training rows once prepared."""

    preparation: prep.Preparation
    columns: list[str]
    rows: list[list[float | None]]
    row_numbers: list[int]  # row numbers in the file
    train_rows: int


def preview(
    frame: pd.DataFrame,
    fields: list[Field],
    target: Field,
    task: Task,
    model_key: str,
    given: dict[str, Any] | None,
    choices: prep.Choices | None = None,
    excluded_columns: set[str] | None = None,
    classes: binning.Classes | None = None,
    validation: evaluate.Validation | None = None,
) -> PreparationPreview:
    """`prepare.fit(train).transform(train)[:20]`, with the decisions behind it (the panel's
    period and entity left out as the training will, R-SEL-09)."""
    setup = _setup(
        frame,
        fields,
        target,
        task,
        model_key,
        given,
        excluded_columns,
        choices,
        classes,
        validation,
    )
    task = setup.task
    preparation, x_train = setup.preparation, setup.x_train
    prepare = Pipeline(prep.steps(preparation, len(x_train), task, drop_first=setup.unregularized))
    _quiet(lambda: prepare.fit(x_train, setup.y_train))
    names = [str(n) for n in prepare.get_feature_names_out()]  # type: ignore[union-attr]
    preparation.inputs = len(names)
    preparation.selection = _selection(prepare)
    first = x_train.iloc[:PREVIEW_ROWS]
    matrix = np.asarray(_quiet(lambda: prepare.transform(first)), dtype=float)
    return PreparationPreview(
        preparation,
        names,
        [[clean(v) for v in row] for row in matrix],
        [int(v) for v in frame.loc[first.index, "__row"]],
        len(x_train),
    )


@dataclass
class Estimate:
    seconds: float  # the whole training, every fit included
    fits: int
    rows: int  # training rows


def estimate(
    frame: pd.DataFrame,
    fields: list[Field],
    target: Field,
    task: Task,
    model_key: str,
    given: dict[str, Any] | None,
    choices: prep.Choices | None = None,
    classes: binning.Classes | None = None,
    validation: evaluate.Validation | None = None,
) -> Estimate:
    """Time the model on 500 then 2 000 training rows and extrapolate t = a·n^b (1 ≤ b ≤ 3) to
    the real number of rows, times the number of fits the validation needs."""
    setup = _setup(
        frame, fields, target, task, model_key, given, None, choices, classes, validation
    )
    x, y = setup.x_train, setup.y_train
    validation = setup.validation

    def timed(n: int) -> float:
        rows = np.sort(np.random.default_rng(SEED).choice(len(x), n, replace=False))
        model = model_pipeline(
            setup.spec,
            setup.values,
            setup.context,
            setup.preparation,
            setup.task,
            n,
            setup.unregularized,
        )
        begin = time.perf_counter()
        _quiet(lambda: model.fit(x.iloc[rows], y.iloc[rows]))
        return max(time.perf_counter() - begin, 1e-3)

    n = len(x)
    if n <= 2_000:
        one = timed(n)
    else:
        small, large = timed(500), timed(2_000)
        power = min(3.0, max(1.0, math.log(large / small) / math.log(4)))
        one = large * (n / 2_000) ** power
    folds = validation.folds * (validation.repeats if validation.method == "repeated_kfold" else 1)
    fits = (n if validation.method == "loo" else folds) + 1 if validation.cross else 1
    if validation.threshold != "default":
        fits += 3
    return Estimate(one * fits, fits, n)
