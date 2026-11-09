"""Does the synthetic data keep the real data's predictive power?

Two identical models (histogram gradient boosting): TRTR trained on the real training rows, TSTR
trained on the synthetic rows; both tested on the same real held-out rows.
"""

from dataclasses import dataclass, field
from typing import Literal, cast

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)

from quantilo.synthetic.encode import Learned, column_of, labels, numbers
from quantilo.synthetic.features import features

MAX_ROWS = 50_000


class UtilityError(ValueError):
    pass


@dataclass
class Metric:
    name: str
    real: float | None  # TRTR
    synthetic: float | None  # TSTR


@dataclass
class Utility:
    target: str
    task: Literal["regression", "classification"]
    primary: str  # R² or balanced accuracy
    metrics: list[Metric] = field(default_factory=list)
    ratio: float | None = None  # TSTR / TRTR on the primary metric
    train_rows: int = 0
    synthetic_rows: int = 0
    test_rows: int = 0
    sentence: str = ""


def _classification(
    fit: np.ndarray,
    y: np.ndarray,
    test: np.ndarray,
    truth: np.ndarray,
    categorical: np.ndarray,
    seed: int,
) -> dict[str, float | None]:
    if len(np.unique(y)) < 2:
        return {"Balanced accuracy": None, "F1 macro": None, "AUC": None}
    model = HistGradientBoostingClassifier(
        # the scikit-learn stubs type these two arguments as str
        categorical_features=categorical if categorical.any() else None,  # pyright: ignore[reportArgumentType]
        early_stopping=True,  # pyright: ignore[reportArgumentType]
        random_state=seed,
    )
    model.fit(fit, y)
    predicted = model.predict(test)
    probabilities = model.predict_proba(test)
    classes = list(np.asarray(model.classes_))
    auc: float | None = None
    try:
        present = np.unique(truth)
        if len(present) == 2 and len(classes) >= 2:
            positive = present[1]
            column = classes.index(positive) if positive in classes else None
            if column is not None:
                auc = float(roc_auc_score(truth == positive, probabilities[:, column]))
        elif len(present) > 2:
            full = np.zeros((len(truth), len(present)))
            for j, level in enumerate(present):
                if level in classes:
                    full[:, j] = probabilities[:, classes.index(level)]
            full = full / np.clip(full.sum(axis=1, keepdims=True), 1e-12, None)
            auc = float(roc_auc_score(truth, full, multi_class="ovr", labels=present))
    except ValueError:
        auc = None
    return {
        "Balanced accuracy": float(balanced_accuracy_score(truth, predicted)),
        "F1 macro": float(f1_score(truth, predicted, average="macro")),
        "AUC": auc,
    }


def _regression(
    fit: np.ndarray,
    y: np.ndarray,
    test: np.ndarray,
    truth: np.ndarray,
    categorical: np.ndarray,
    seed: int,
) -> dict[str, float | None]:
    model = HistGradientBoostingRegressor(
        # the scikit-learn stubs type these two arguments as str
        categorical_features=categorical if categorical.any() else None,  # pyright: ignore[reportArgumentType]
        early_stopping=True,  # pyright: ignore[reportArgumentType]
        random_state=seed,
    )
    model.fit(fit, y)
    predicted = model.predict(test)
    return {
        "R²": float(r2_score(truth, predicted)),
        "MAE": float(mean_absolute_error(truth, predicted)),
    }


def utility(
    train: pd.DataFrame,
    synthetic: pd.DataFrame,
    holdout: pd.DataFrame,
    learned: list[Learned],
    target: str,
    seed: int = 42,
) -> Utility:
    by_name = {item.name: item for item in learned}
    if target not in by_name:
        raise UtilityError(f"{target!r} is not a modelled column.")
    item = by_name[target]
    if item.kind == "date":
        raise UtilityError(f"{target!r} is a date: pick a number or a category.")
    inputs = [i for i in learned if i.name != target]
    if not inputs:
        raise UtilityError("Needs at least one other modelled column to predict from.")
    task: Literal["regression", "classification"] = (
        "classification" if item.categorical else "regression"
    )

    def rows(frame: pd.DataFrame) -> pd.DataFrame:
        frame = cast(pd.DataFrame, frame[column_of(frame, target).notna()])
        return (
            frame.sample(min(len(frame), MAX_ROWS), random_state=seed)
            if len(frame) > MAX_ROWS
            else frame
        )

    train, synthetic, holdout = rows(train), rows(synthetic), rows(holdout)
    if len(holdout) < 10 or len(train) < 10 or len(synthetic) < 10:
        raise UtilityError("Too few rows with a target value to train and test.")
    found = features([train, synthetic, holdout], inputs)
    categorical = np.array(found.categorical)

    def outcome(frame: pd.DataFrame) -> np.ndarray:
        if task == "classification":
            return labels(column_of(frame, target)).astype(str).to_numpy()
        return numbers(column_of(frame, target))

    truth = outcome(holdout)
    run = _classification if task == "classification" else _regression
    real_scores = run(found.frames[0], outcome(train), found.frames[2], truth, categorical, seed)
    synthetic_scores = run(
        found.frames[1], outcome(synthetic), found.frames[2], truth, categorical, seed
    )
    metrics = [Metric(name, real_scores[name], synthetic_scores[name]) for name in real_scores]
    primary = "R²" if task == "regression" else "Balanced accuracy"
    real_value, synthetic_value = real_scores[primary], synthetic_scores[primary]
    ratio = None
    if real_value is not None and synthetic_value is not None and real_value > 0:
        ratio = max(0.0, synthetic_value / real_value)
    if ratio is None:
        sentence = "The real data has no predictive power here to compare with."
    else:
        sentence = f"The synthetic data keeps {ratio:.0%} of the real data's predictive power."
    return Utility(
        target, task, primary, metrics, ratio, len(train), len(synthetic), len(holdout), sentence
    )
