"""Pipeline steps the preparation editor can add (017): joint imputation, clipping, statistical
selection and class rebalancing. Every step learns from the training rows only."""

from typing import cast

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (registers the class)
from sklearn.feature_selection import (
    RFECV,
    VarianceThreshold,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LassoCV, LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

SEED = 42
KNN_DONORS = 5_000  # rows the KNN imputer searches for neighbours
ITERATIVE_ROWS = 20_000  # rows the iterative imputer is fitted on
SELECTION_ROWS = 20_000  # rows the statistical selection is fitted on
COSTLY_ROWS = 5_000  # mutual information, RFE
CHUNK = 10_000
EXTREME_IQR = 3.0  # rows beyond 3 IQR are dropped by « drop rows »

SELECTION_DEFAULTS: dict[str, float | None] = {
    "variance": 0.01,  # keep inputs whose variance exceeds this
    "correlation": 0.95,  # drop the second of two inputs with |r| above this
    "vif": 10.0,  # drop the input with the largest VIF while it exceeds this
    "mutual_info": 20,  # keep the k inputs with the most mutual information
    "univariate": 0.05,  # keep inputs whose F test p-value is below this
    "l1": 1.0,  # classification: inverse strength C of the L1 penalty (regression: by CV)
    "rfe": None,  # recursive elimination, the number kept chosen by 3-fold CV
    "permutation": 0.0,  # keep inputs whose permutation importance exceeds this
}


class RobustScale(RobustScaler):
    """Median and IQR. A column whose IQR is 0 (three quarters of the rows on one value, like a
    12-month tenure) is divided by its standard deviation instead of by 1: left as it was, it
    decided every distance between rows (credit-card segments all named after it, 040)."""

    def fit(self, X, y=None):
        super().fit(X, y)
        values = np.asarray(X, dtype=float)
        if values.ndim == 2 and len(values):
            q1, q3 = np.nanpercentile(values, [25, 75], axis=0)
            flat = (q3 - q1) == 0
            if flat.any():
                spread = np.nanstd(values[:, flat], axis=0)
                scale = np.asarray(self.scale_, dtype=float)
                scale[flat] = np.where(spread > 0, spread, 1.0)
                self.scale_ = scale
        return self


def _sample(rows: int, limit: int) -> np.ndarray:
    if rows <= limit:
        return np.arange(rows)
    return np.sort(np.random.default_rng(SEED).choice(rows, limit, replace=False))


class JointImputer(BaseEstimator, TransformerMixin):
    """KNN or iterative imputation over several numeric columns at once (each column is filled
    from the others), before the per-column preparation. Keeps the frame's columns."""

    def __init__(self, columns: tuple[str, ...] = (), kind: str = "knn") -> None:
        self.columns = columns
        self.kind = kind

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        values = X[list(self.columns)].to_numpy(dtype=float, copy=True)
        if self.kind == "knn":
            imputer = KNNImputer(n_neighbors=5, keep_empty_features=True)
            values = values[_sample(len(values), KNN_DONORS)]
        else:
            imputer = IterativeImputer(max_iter=10, random_state=SEED, keep_empty_features=True)
            values = values[_sample(len(values), ITERATIVE_ROWS)]
        self.imputer_ = imputer.fit(values)
        return self

    def transform(self, X):
        out = X.copy()
        values = X[list(self.columns)].to_numpy(dtype=float, copy=True)
        filled = (
            np.vstack(
                [
                    np.asarray(self.imputer_.transform(values[i : i + CHUNK]), dtype=float)
                    for i in range(0, len(values), CHUNK)
                ]
            )
            if len(values)
            else values
        )
        for i, column in enumerate(self.columns):
            out[column] = filled[:, i]
        return out

    def get_feature_names_out(self, input_features=None):
        names = self.feature_names_in_ if input_features is None else input_features
        return np.asarray(names, dtype=object)


class Clipper(BaseEstimator, TransformerMixin):
    """Clip each column to its 1st to 99th percentiles or to 1.5 IQR beyond the quartiles,
    the limits learnt on the training rows."""

    def __init__(self, kind: str = "clip_percentiles") -> None:
        self.kind = kind

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        if self.kind == "clip_percentiles":
            self.low_, self.high_ = np.nanpercentile(values, [1, 99], axis=0)
        else:
            q1, q3 = np.nanpercentile(values, [25, 75], axis=0)
            self.low_, self.high_ = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.low_, self.high_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def extreme_rows(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Rows with a value more than 3 IQR beyond the quartiles in any of the columns."""
    mask = np.zeros(len(frame), dtype=bool)
    for column in columns:
        values = frame[column].to_numpy(dtype=float)
        if np.isnan(values).all():
            continue
        q1, q3 = np.nanpercentile(values, [25, 75])
        iqr = q3 - q1
        if iqr <= 0:
            continue
        with np.errstate(invalid="ignore"):
            mask |= (values < q1 - EXTREME_IQR * iqr) | (values > q3 + EXTREME_IQR * iqr)
    return mask


class Selector(BaseEstimator, TransformerMixin):
    """One statistical selection method on the prepared inputs; keeps at least one input."""

    def __init__(
        self, method: str = "variance", value: float | None = None, task: str = "regression"
    ) -> None:
        self.method = method
        self.value = value
        self.task = task

    def fit(self, X, y):
        x = np.nan_to_num(np.asarray(X, dtype=float))
        target = np.asarray(y)
        rows = _sample(len(x), SELECTION_ROWS)
        x, target = x[rows], target[rows]
        value = self.value if self.value is not None else SELECTION_DEFAULTS[self.method]
        support = self._support(x, target, value)
        if not support.any():  # keep the most variable input rather than nothing
            support[int(np.argmax(x.var(axis=0)))] = True
        self.support_ = support
        self.n_features_in_ = x.shape[1]
        return self

    def _support(self, x: np.ndarray, y: np.ndarray, value: float | None) -> np.ndarray:
        n = x.shape[1]
        regression = self.task == "regression"
        y_fit = y.astype(float) if regression else y.astype(str)
        if self.method == "variance":
            return cast(np.ndarray, VarianceThreshold(value or 0.0).fit(x).get_support())
        if self.method == "correlation":
            return _correlation_filter(x, float(value or 0.95))
        if self.method == "vif":
            return _vif_filter(x, float(value or 10.0))
        if self.method == "mutual_info":
            rows = _sample(len(x), COSTLY_ROWS)
            score = (mutual_info_regression if regression else mutual_info_classif)(
                x[rows], y_fit[rows], random_state=SEED
            )
            k = max(1, min(n, int(value or 20)))
            keep = np.zeros(n, dtype=bool)
            keep[np.argsort(-np.asarray(score))[:k]] = True
            return keep
        if self.method == "univariate":
            with np.errstate(all="ignore"):
                _, p = (f_regression if regression else f_classif)(x, y_fit)
            return np.nan_to_num(np.asarray(p, dtype=float), nan=1.0) < float(value or 0.05)
        if self.method == "l1":
            if regression:
                model = LassoCV(cv=3, random_state=SEED, max_iter=5_000).fit(x, y_fit)
                coef = np.abs(np.asarray(model.coef_))
            else:
                model = LogisticRegression(
                    solver="saga", l1_ratio=1.0, C=float(value or 1.0), max_iter=2_000
                ).fit(x, y_fit)
                coef = np.abs(np.atleast_2d(model.coef_)).max(axis=0)
            return coef > 1e-10
        if self.method == "rfe":
            rows = _sample(len(x), COSTLY_ROWS)
            estimator = LinearRegression() if regression else LogisticRegression(max_iter=1_000)
            rfe = RFECV(estimator, step=max(1, n // 10), cv=3, min_features_to_select=1)
            return cast(np.ndarray, rfe.fit(x[rows], y_fit[rows]).get_support())
        if self.method == "permutation":
            stratify = (
                None if regression or min(np.unique(y_fit, return_counts=True)[1]) < 2 else y_fit
            )
            x_fit, x_check, y_a, y_b = train_test_split(
                x, y_fit, test_size=0.25, random_state=SEED, stratify=stratify
            )
            estimator = Ridge() if regression else LogisticRegression(max_iter=1_000)
            estimator.fit(x_fit, y_a)
            found = permutation_importance(estimator, x_check, y_b, n_repeats=5, random_state=SEED)
            return np.asarray(found["importances_mean"], dtype=float) > float(value or 0.0)
        raise ValueError(f"Unknown selection method {self.method!r}.")

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.support_]

    def get_feature_names_out(self, input_features=None):
        names = (
            np.asarray(input_features, dtype=object)
            if input_features is not None
            else np.asarray([f"x{i}" for i in range(self.n_features_in_)], dtype=object)
        )
        return names[self.support_]


def _correlation_filter(x: np.ndarray, threshold: float) -> np.ndarray:
    with np.errstate(all="ignore"):
        corr = np.nan_to_num(np.abs(np.corrcoef(x, rowvar=False)))
    corr = np.atleast_2d(corr)
    keep = np.ones(x.shape[1], dtype=bool)
    for j in range(x.shape[1]):
        if keep[j] and np.any(corr[j, :j][keep[:j]] > threshold):
            keep[j] = False
    return keep


def _vif_filter(x: np.ndarray, threshold: float) -> np.ndarray:
    keep = x.std(axis=0) > 0
    while keep.sum() > 1:
        index = np.flatnonzero(keep)
        with np.errstate(all="ignore"):
            corr = np.nan_to_num(np.corrcoef(x[:, index], rowvar=False))
        vif = np.diag(np.linalg.pinv(np.atleast_2d(corr)))
        worst = int(np.argmax(vif))
        if vif[worst] <= threshold:
            break
        keep[index[worst]] = False
    return keep


def rebalance(x: pd.DataFrame, y: pd.Series, strategy: str) -> tuple[pd.DataFrame, pd.Series]:
    """Random over-sampling of the smaller classes up to the largest, or under-sampling of the
    larger ones down to the smallest (training rows only)."""
    rng = np.random.default_rng(SEED)
    counts = y.value_counts()
    goal = int(counts.max() if strategy == "oversample" else counts.min())
    picked: list[np.ndarray] = []
    labels = y.to_numpy()
    for label in counts.index:
        rows = np.flatnonzero(labels == label)
        if len(rows) == goal:
            picked.append(rows)
        elif len(rows) < goal:
            picked.append(np.concatenate([rows, rng.choice(rows, goal - len(rows), replace=True)]))
        else:
            picked.append(rng.choice(rows, goal, replace=False))
    order = np.sort(np.concatenate(picked))
    return x.iloc[order], y.iloc[order]
