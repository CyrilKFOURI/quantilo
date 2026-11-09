"""Can a classifier tell real from synthetic rows? A measure of fidelity, not of privacy.

Two detectors in stratified 3-fold cross-validation: logistic regression (as SDMetrics'
LogisticDetection, so comparable) and histogram gradient boosting (stronger, sees interactions).
Score = 1 - (max(AUC, 0.5) * 2 - 1); the more severe of the two is kept.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from quantilo.synthetic.encode import Learned
from quantilo.synthetic.features import features

MAX_PER_SIDE = 10_000


@dataclass
class Giveaway:
    column: str
    importance: float  # AUC lost when the column is shuffled


@dataclass
class Detector:
    name: str
    auc: float
    score: float


@dataclass
class Detection:
    score: float  # the more severe detector
    detectors: list[Detector] = field(default_factory=list)
    giveaways: list[Giveaway] = field(default_factory=list)  # strongest first
    rows: int = 0  # rows per side


def score_from_auc(auc: float) -> float:
    return 1 - (max(auc, 0.5) * 2 - 1)


def verdict(score: float) -> str:
    if score >= 0.8:
        return "Hard to tell apart"
    if score >= 0.5:
        return "Somewhat distinguishable"
    return "Easy to tell apart"


def detect(
    real: pd.DataFrame, synthetic: pd.DataFrame, learned: list[Learned], seed: int = 42
) -> Detection:
    rows = min(len(real), len(synthetic), MAX_PER_SIDE)
    if rows < 20 or not learned:
        return Detection(1.0, rows=rows)
    real = real.sample(rows, random_state=seed)
    synthetic = synthetic.sample(rows, random_state=seed)
    found = features([real, synthetic], learned)
    data = np.vstack(found.frames)
    labels = np.r_[np.zeros(rows), np.ones(rows)].astype(int)
    categorical = np.array(found.categorical)
    folds = StratifiedKFold(3, shuffle=True, random_state=seed)

    numeric_index = np.flatnonzero(~categorical)
    category_index = np.flatnonzero(categorical)
    logistic = make_pipeline(
        ColumnTransformer(
            [
                (
                    "num",
                    make_pipeline(SimpleImputer(strategy="median"), RobustScaler()),
                    numeric_index,
                ),
                (
                    "cat",
                    make_pipeline(
                        SimpleImputer(strategy="constant", fill_value=-1),
                        OneHotEncoder(handle_unknown="ignore", max_categories=30),
                    ),
                    category_index,
                ),
            ]
        ),
        LogisticRegression(max_iter=1_000),
    )
    boosting = HistGradientBoostingClassifier(
        # the scikit-learn stubs type these two arguments as str
        categorical_features=categorical if categorical.any() else None,  # pyright: ignore[reportArgumentType]
        max_iter=200,
        early_stopping=True,  # pyright: ignore[reportArgumentType]
        random_state=seed,
    )
    detectors = []
    for name, model in (("Logistic regression", logistic), ("Gradient boosting", boosting)):
        auc = float(np.mean(cross_val_score(model, data, labels, cv=folds, scoring="roc_auc")))
        detectors.append(Detector(name, auc, score_from_auc(auc)))
    score = min(d.score for d in detectors)

    fit_x, test_x, fit_y, test_y = train_test_split(
        data, labels, test_size=1 / 3, stratify=labels, random_state=seed
    )
    boosting.fit(fit_x, fit_y)
    importance = permutation_importance(
        boosting, test_x, test_y, scoring="roc_auc", n_repeats=3, random_state=seed
    )
    means = np.asarray(importance["importances_mean"])
    giveaways = [
        Giveaway(found.names[i], float(means[i])) for i in np.argsort(-means) if means[i] > 0.005
    ]
    return Detection(score, detectors, giveaways[:8], rows)
