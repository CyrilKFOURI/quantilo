"""How a model is validated (019): split, cross-validation or walk-forward by period on a panel
(035), and every metric of the problem."""

import math
from collections.abc import Iterator
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, model_validator
from scipy import stats
from sklearn import metrics as skm
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    LeaveOneOut,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    StratifiedGroupKFold,
    StratifiedKFold,
)

from quantilo.ml import browser
from quantilo.ml.target import Task
from quantilo.variable.source import clean

Method = Literal[
    "split", "kfold", "repeated_kfold", "group_kfold", "time_split", "loo", "panel_walk_forward"
]
CROSS = ("kfold", "repeated_kfold", "group_kfold", "loo", "panel_walk_forward")
PANEL = "panel_walk_forward"
LOO_LIMIT = 1_000

REGRESSION_METRICS = ["r2", "mae", "rmse", "mape", "median_ae", "max_error", "pearson", "spearman"]
CLASSIFICATION_METRICS = [
    "accuracy",
    "balanced_accuracy",
    "f1_macro",
    "f1_weighted",
    "precision",
    "recall",
    "roc_auc",
    "pr_auc",
    "log_loss",
    "brier",
    "mcc",
    "kappa",
]
ZERO: Any = 0.0  # zero_division (the stubs expect a string)
LOWER_IS_BETTER = {"mae", "rmse", "mape", "median_ae", "max_error", "log_loss", "brier"}


class Validation(BaseModel):
    """How the model is judged, and on how many rows it is trained."""

    method: Method = "split"
    test_size: float = Field(default=0.2, ge=0.1, le=0.4)
    folds: int = Field(default=5, ge=3, le=10)
    repeats: int = Field(default=3, ge=2, le=10)
    group: str | None = Field(default=None, max_length=500)  # group k-fold
    time_column: str | None = Field(default=None, max_length=500)  # time-based split
    metric: str | None = Field(default=None, max_length=40)  # None: the default for the problem
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    max_rows: int | None = Field(default=200_000, ge=100, le=100_000_000)  # None: every row
    # two classes: the probability above which the positive class is predicted
    threshold: Literal["default", "f1", "balanced_accuracy"] = "default"
    # panel walk-forward (035): periods in order, an embargo of periods before each test block
    period_column: str | None = Field(default=None, max_length=500)
    entity_column: str | None = Field(default=None, max_length=500)
    embargo: int = Field(default=0, ge=0, le=1_000)  # periods left out before each test block
    window: Literal["expanding", "rolling"] = "expanding"
    window_periods: int | None = Field(default=None, ge=1, le=100_000)  # rolling: None = fold 1's
    sample_periods: int | None = Field(default=None, ge=1, le=100)  # train on 1 period in N
    criterion: Literal["mean", "sharpe", "worst"] = "mean"  # the score over the periods
    period_threshold: float | None = None  # a "good" period scores above it (None: by metric)

    @model_validator(mode="after")
    def _browser_rows(self) -> "Validation":
        """R-WEB-01: in the browser version, at most 50 000 training rows (one core, little
        memory); the page says so where the rows are chosen."""
        self.max_rows = browser.max_rows(self.max_rows)
        return self

    @property
    def cross(self) -> bool:
        return self.method in CROSS

    @property
    def panel(self) -> bool:
        return self.method == PANEL


def metrics_for(task: Task) -> list[str]:
    return REGRESSION_METRICS if task == "regression" else CLASSIFICATION_METRICS


def default_metric(task: Task, minority: float | None, panel: bool = False) -> str:
    """On a panel, numbers are judged by their rank correlation within each period (035). Under
    1 % of positives (fraud), ROC AUC stays high while most alerts are false: precision-recall AUC
    shows it (040)."""
    if task == "regression":
        return "spearman" if panel else "r2"
    if task == "binary":
        return "pr_auc" if minority is not None and minority < 0.01 else "roc_auc"
    return "balanced_accuracy" if minority is not None and minority < 0.2 else "accuracy"


class ValidationError(ValueError):
    pass


def check(validation: Validation, task: Task, rows: int, columns: set[str]) -> None:
    if validation.metric is not None and validation.metric not in metrics_for(task):
        raise ValidationError(f"{validation.metric} is not a metric of this problem.")
    if validation.method == "loo" and rows > LOO_LIMIT:
        raise ValidationError(f"Leave-one-out needs ≤ {LOO_LIMIT:,} rows: choose k-fold.")
    if validation.method == "group_kfold" and validation.group not in columns:
        raise ValidationError("Group k-fold needs a group column.")
    if validation.method == "time_split" and validation.time_column not in columns:
        raise ValidationError("A time-based split needs a date column.")
    if validation.panel:
        if validation.period_column not in columns:
            raise ValidationError("Panel walk-forward needs a period column.")
        if validation.entity_column is not None and validation.entity_column not in columns:
            raise ValidationError(f"Unknown entity column {validation.entity_column!r}.")
        if validation.entity_column == validation.period_column:
            raise ValidationError("The period and the entity must be two different columns.")
        metric = validation.metric or default_metric(task, None, panel=True)
        if validation.criterion == "sharpe" and metric in LOWER_IS_BETTER:
            raise ValidationError(
                "Mean / std needs a metric where higher is better: choose another criterion."
            )
    if validation.threshold != "default" and task != "binary":
        raise ValidationError("A decision threshold is chosen for two classes only.")
    if validation.threshold != "default" and validation.cross:
        raise ValidationError(
            "With cross-validation the threshold would be chosen on the rows that judge it: "
            "use a train / test split."
        )


def time_split(when: pd.Series, test_size: float) -> tuple[np.ndarray, np.ndarray]:
    """Test = the most recent rows: no test row is earlier than a training row (rows without a
    date count as the oldest)."""
    if pd.api.types.is_numeric_dtype(when):
        key = when.astype(float).to_numpy()
    else:
        moments = pd.to_datetime(when, errors="coerce")
        key = (moments - moments.min()).dt.total_seconds().to_numpy(dtype=float)
    order = np.argsort(np.nan_to_num(key, nan=-np.inf), kind="stable")
    cut = round(len(order) * (1 - test_size))
    return order[:cut], order[cut:]


def folds(
    validation: Validation, y: pd.Series, groups: pd.Series | None, classification: bool
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Training and held-out positions of every fold."""
    n = len(y)
    stratify = classification and int(y.value_counts().min()) >= validation.folds
    if validation.method == "loo":
        splitter = LeaveOneOut()
    elif validation.method == "group_kfold":
        labels = cast(pd.Series, groups).astype(str).fillna("(empty)").to_numpy()
        k = min(validation.folds, len(set(labels)))
        if k < 2:
            raise ValidationError("Group k-fold needs at least two groups.")
        splitter = (
            StratifiedGroupKFold(k, shuffle=True, random_state=validation.seed)
            if stratify
            else GroupKFold(k)
        )
        yield from splitter.split(np.zeros(n), y.to_numpy(), labels)
        return
    elif validation.method == "repeated_kfold":
        splitter = (
            RepeatedStratifiedKFold(
                n_splits=validation.folds,
                n_repeats=validation.repeats,
                random_state=validation.seed,
            )
            if stratify
            else RepeatedKFold(
                n_splits=validation.folds,
                n_repeats=validation.repeats,
                random_state=validation.seed,
            )
        )
    else:
        splitter = (
            StratifiedKFold(validation.folds, shuffle=True, random_state=validation.seed)
            if stratify
            else KFold(validation.folds, shuffle=True, random_state=validation.seed)
        )
    yield from splitter.split(np.zeros(n), y.to_numpy() if stratify else None)


# --- Metrics --------------------------------------------------------------------------------------


def _safe(compute) -> float | None:
    try:
        value = float(compute())
    except (ValueError, ZeroDivisionError, IndexError):
        return None
    return clean(value) if math.isfinite(value) else None


def _correlation(actual: np.ndarray, predicted: np.ndarray, ranked: bool) -> float:
    if len(actual) < 3 or np.ptp(actual) == 0 or np.ptp(predicted) == 0:
        raise ValueError("constant")
    if ranked:
        return float(cast(Any, stats.spearmanr(actual, predicted)).statistic)
    return float(np.corrcoef(actual, predicted)[0, 1])


def regression_metric(name: str, actual: np.ndarray, predicted: np.ndarray) -> float | None:
    """One metric of a regression (None when it does not apply, e.g. MAPE with zeros)."""
    if name == "r2":
        return _safe(lambda: skm.r2_score(actual, predicted))
    if name == "mae":
        return _safe(lambda: skm.mean_absolute_error(actual, predicted))
    if name == "rmse":
        return _safe(lambda: math.sqrt(skm.mean_squared_error(actual, predicted)))
    if name == "mape":
        if not bool((actual > 0).all()):
            return None
        return _safe(lambda: skm.mean_absolute_percentage_error(actual, predicted) * 100)
    if name == "median_ae":
        return _safe(lambda: skm.median_absolute_error(actual, predicted))
    if name == "max_error":
        return _safe(lambda: skm.max_error(actual, predicted))
    if name in ("pearson", "spearman"):
        return _safe(lambda: _correlation(actual, predicted, name == "spearman"))
    raise ValidationError(f"{name} is not a metric of this problem.")


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | None]:
    return {name: regression_metric(name, actual, predicted) for name in REGRESSION_METRICS}


def classification_metric(
    name: str,
    actual: np.ndarray,
    predicted: np.ndarray,
    probabilities: np.ndarray,
    classes: list[str],
    positive: str | None,
) -> float | None:
    """One metric of a classification (AUC, PR AUC and Brier need the positive class)."""
    if name == "accuracy":
        return _safe(lambda: skm.accuracy_score(actual, predicted))
    if name == "balanced_accuracy":
        return _safe(lambda: skm.balanced_accuracy_score(actual, predicted))
    if name in ("f1_macro", "f1_weighted"):
        average = "macro" if name == "f1_macro" else "weighted"
        return _safe(lambda: skm.f1_score(actual, predicted, average=average, zero_division=ZERO))
    if name == "precision":
        return _safe(
            lambda: skm.precision_score(actual, predicted, average="macro", zero_division=ZERO)
        )
    if name == "recall":
        return _safe(
            lambda: skm.recall_score(actual, predicted, average="macro", zero_division=ZERO)
        )
    if name == "log_loss":
        return _safe(lambda: skm.log_loss(actual, probabilities, labels=classes))
    if name == "mcc":
        return _safe(lambda: skm.matthews_corrcoef(actual, predicted))
    if name == "kappa":
        return _safe(lambda: skm.cohen_kappa_score(actual, predicted))
    if name not in ("roc_auc", "pr_auc", "brier"):
        raise ValidationError(f"{name} is not a metric of this problem.")
    if positive is not None and len(classes) == 2:
        truth = (actual == positive).astype(int)
        scores = probabilities[:, classes.index(positive)]
        if name == "brier":
            return _safe(lambda: skm.brier_score_loss(truth, scores))
        if not 0 < truth.sum() < len(truth):
            return None
        if name == "roc_auc":
            return _safe(lambda: skm.roc_auc_score(truth, scores))
        return _safe(lambda: skm.average_precision_score(truth, scores))
    if name == "roc_auc" and len(set(actual)) == len(classes):
        return _safe(
            lambda: skm.roc_auc_score(actual, probabilities, multi_class="ovr", labels=classes)
        )
    return None


def classification_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    probabilities: np.ndarray,
    classes: list[str],
    positive: str | None,
) -> dict[str, float | None]:
    return {
        name: classification_metric(name, actual, predicted, probabilities, classes, positive)
        for name in CLASSIFICATION_METRICS
    }


def summarise(
    per_fold: list[dict[str, float | None]],
) -> tuple[dict[str, float | None], dict[str, float | None]]:
    """Mean and standard deviation of every metric over the folds."""
    keys = per_fold[0].keys() if per_fold else []
    mean: dict[str, float | None] = {}
    spread: dict[str, float | None] = {}
    for key in keys:
        values = [v for fold in per_fold if (v := fold.get(key)) is not None]
        mean[key] = clean(float(np.mean(values))) if values else None
        spread[key] = clean(float(np.std(values, ddof=1))) if len(values) > 1 else None
    return mean, spread
