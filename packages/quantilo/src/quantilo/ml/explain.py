# ruff: noqa: RUF002, RUF003  (φ, β, −, × are the usual notation)
# pyright: reportAttributeAccessIssue=false, reportArgumentType=false, reportCallIssue=false
# (fitted scikit-learn objects and pandas groupbys: the stubs cannot follow them)
"""Why a model predicts what it predicts (020): permutation importance, SHAP values (exact closed
form for linear models, native TreeSHAP for XGBoost / LightGBM), partial dependence, the decision
tree itself, and the strongly linked inputs whose importance is shared. No `shap` package."""

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.metrics import cohen_kappa_score, make_scorer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

from quantilo.ml import evaluate
from quantilo.ml import preparation as prep
from quantilo.ml.target import Task
from quantilo.variable.source import clean

SEED = 42
PERMUTATION_ROWS = 10_000
REPEATS = 5
SHAP_ROWS = 2_000
TOP_FEATURES = 15
DEPENDENCE_FEATURES = 8
DEPENDENCE_ROWS = 1_000
GRID = 30
TOP_CATEGORIES = 20
PAIR_ROWS = 5_000
PAIR_COLUMNS = 40
STRONG_LINK = 0.8
TREE_DEPTH = 4
WATERFALL = 10


def _pearson(actual: np.ndarray, predicted: np.ndarray) -> float:
    return evaluate.regression_metric("pearson", np.asarray(actual), np.asarray(predicted)) or 0.0


def _spearman(actual: np.ndarray, predicted: np.ndarray) -> float:
    return evaluate.regression_metric("spearman", np.asarray(actual), np.asarray(predicted)) or 0.0


SCORERS: dict[str, Any] = {
    "r2": "r2",
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
    "mape": "neg_mean_absolute_percentage_error",
    "median_ae": "neg_median_absolute_error",
    "max_error": "neg_max_error",
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "f1_macro": "f1_macro",
    "f1_weighted": "f1_weighted",
    "precision": "precision_macro",
    "recall": "recall_macro",
    "log_loss": "neg_log_loss",
    "mcc": "matthews_corrcoef",
    "kappa": make_scorer(cohen_kappa_score),
    "pearson": make_scorer(_pearson),
    "spearman": make_scorer(_spearman),
}


@dataclass
class PermutationItem:
    column: str
    mean: float  # how much the score drops when the column is shuffled
    std: float


@dataclass
class Permutation:
    metric: str
    rows: int
    sampled: bool
    items: list[PermutationItem]


@dataclass
class ShapFeature:
    column: str
    mean_abs: float
    values: list[float]  # one SHAP value per row of `Shap.rows`
    colors: list[float | None]  # the column's value as a 0–1 rank (numbers only)
    reading: str | None


@dataclass
class Shap:
    method: str
    scale: str  # "prediction", "log-odds" or "log"
    base: float  # the average raw prediction
    positive: str | None  # binary: the class whose log-odds are explained
    rows: list[int]  # file rows of the points
    features: list[ShapFeature]  # the strongest columns, then "Other features"
    columns: list[str]  # every column, for the waterfall of one row
    matrix: list[list[float]]  # rows × columns
    labels: list[list[str]]  # the value shown next to each contribution


@dataclass
class Dependence:
    column: str
    kind: str  # "number" or "category"
    grid: list[str | float]
    curves: dict[str, list[float]]  # one curve per output (prediction, or class)
    counts: list[int]  # rows near each grid point / in each category
    output: str  # what the curve shows


@dataclass
class TreeNode:
    id: int
    depth: int
    feature: str | None  # None: a leaf (or the depth limit)
    threshold: float | None
    value: str  # mean prediction, or majority class and its share
    samples: int
    left: int | None
    right: int | None


@dataclass
class Pair:
    first: str
    second: str
    strength: float
    measure: str  # |r|, Cramér's V or η


@dataclass
class Explanation:
    permutation: Permutation | None
    shap: Shap | None
    shap_unavailable: str | None
    dependence: list[Dependence]
    tree: list[TreeNode] | None
    pairs: list[Pair]
    warnings: list[str] = field(default_factory=list)


def _quiet(compute):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return compute()


def _sample(n: int, limit: int, first: list[int] | None = None) -> np.ndarray:
    """At most `limit` positions, drawn with a seed, always keeping the `first` positions."""
    keep = list(dict.fromkeys(first or []))[:limit]
    if n <= limit:
        return np.arange(n)
    rest = np.setdiff1d(np.arange(n), keep)
    extra = np.random.default_rng(SEED).choice(rest, limit - len(keep), replace=False)
    return np.sort(np.concatenate([np.asarray(keep, dtype=int), extra]))


# --- Permutation importance ----------------------------------------------------------------------


def permutation(
    pipeline: Pipeline, x: pd.DataFrame, y: pd.Series, task: Task, metric: str
) -> Permutation:
    """Shuffle each original column of the evaluation rows and measure how much the score drops.
    Columns are the dataset's, not the prepared inputs: one-hot columns go together."""
    rows = _sample(len(x), PERMUTATION_ROWS)
    xs, ys = x.iloc[rows], y.iloc[rows]
    scorer = SCORERS.get(metric)
    if metric == "roc_auc":
        scorer = "roc_auc" if task == "binary" else "roc_auc_ovr"
    if scorer is None:  # pr_auc, brier: need a positive label; fall back to the plain score
        metric, scorer = ("r2", "r2") if task == "regression" else ("accuracy", "accuracy")
    try:
        found = _quiet(
            lambda: permutation_importance(
                pipeline, xs, ys, scoring=scorer, n_repeats=REPEATS, random_state=SEED
            )
        )
    except (ValueError, TypeError):
        metric, scorer = ("r2", "r2") if task == "regression" else ("accuracy", "accuracy")
        found = _quiet(
            lambda: permutation_importance(
                pipeline, xs, ys, scoring=scorer, n_repeats=REPEATS, random_state=SEED
            )
        )
    means = np.asarray(found["importances_mean"], dtype=float)
    stds = np.asarray(found["importances_std"], dtype=float)
    items = [
        PermutationItem(str(column), clean(float(means[i])) or 0.0, clean(float(stds[i])) or 0.0)
        for i, column in enumerate(x.columns)
    ]
    items.sort(key=lambda item: -item.mean)
    return Permutation(metric, len(rows), len(rows) < len(x), items)


# --- SHAP ---------------------------------------------------------------------------------------


def origins(pipeline: Pipeline) -> list[str]:
    """For each prepared input the model receives, the dataset column it comes from."""
    transformer = pipeline.named_steps["prepare"]
    owner: dict[str, str] = {}
    for name, step, columns in transformer.transformers_:
        if name == "remainder" or step == "drop":
            continue
        for produced in step.get_feature_names_out(list(columns)):
            owner[str(produced)] = str(columns[0])
    names = [str(n) for n in pipeline[:-1].get_feature_names_out()]  # type: ignore[union-attr]
    return [owner.get(n, n) for n in names]


def _raw_linear(model: Any, z: np.ndarray, task: Task) -> tuple[np.ndarray, float, str] | None:
    """φ = β (z − E[z]) for a linear model on its link scale, or None."""
    if isinstance(model, CalibratedClassifierCV | OneVsRestClassifier):
        return None
    coef = getattr(model, "coef_", None)
    intercept = getattr(model, "intercept_", None)
    if coef is None or intercept is None:
        return None
    beta = np.atleast_2d(np.asarray(coef, dtype=float))
    if beta.shape[0] != 1 or beta.shape[1] != z.shape[1]:
        return None  # several classes: one explanation per class (not shown)
    beta = beta[0]
    bias = float(np.ravel(np.asarray(intercept, dtype=float))[0])
    mean = z.mean(axis=0)
    phi = beta * (z - mean)
    name = model.__class__.__name__
    if task != "regression":
        scale = "log-odds"
    elif name in ("PoissonRegressor", "GammaRegressor", "TweedieRegressor"):
        scale = "log"
    else:
        scale = "prediction"
    return phi, bias + float(beta @ mean), scale


def _raw_boosting(model: Any, z: np.ndarray) -> tuple[np.ndarray, float, str] | None:
    """Native TreeSHAP of XGBoost / LightGBM (exact, on the raw margin)."""
    inner = getattr(model, "model_", None)
    library = getattr(model, "library", None)
    if inner is None or library not in ("xgboost", "lightgbm"):
        return None
    if library == "xgboost":
        import xgboost  # type: ignore[import-not-found]  (optional package)

        # early stopping: the model predicts with its best iteration, so must the explanation
        best = getattr(inner, "best_iteration", None)
        stop = (0, int(best) + 1) if best is not None else (0, 0)
        values = np.asarray(
            inner.get_booster().predict(
                xgboost.DMatrix(z), pred_contribs=True, iteration_range=stop
            ),
            dtype=float,
        )
    else:
        values = np.asarray(
            inner.predict(z, pred_contrib=True, num_iteration=inner.best_iteration_ or None),
            dtype=float,
        )
    if values.ndim != 2 or values.shape[1] != z.shape[1] + 1:
        return None  # several classes
    return values[:, :-1], float(values[0, -1]), "margin"


def _as_number(raw: pd.Series) -> pd.Series | None:
    """Numbers as they are, dates as seconds since the earliest one; None for categories."""
    if pd.api.types.is_datetime64_any_dtype(raw):
        return (raw - raw.min()).dt.total_seconds()
    if pd.api.types.is_numeric_dtype(raw) and not pd.api.types.is_bool_dtype(raw):
        return raw.astype(float)
    return None


def _reading(column: str, phi: np.ndarray, raw: pd.Series) -> str | None:
    """One sentence on the direction of the effect."""
    number = _as_number(raw)
    higher = "Later" if pd.api.types.is_datetime64_any_dtype(raw) else "Higher"
    if number is not None:
        values = number.to_numpy(dtype=float)
        ok = np.isfinite(values) & np.isfinite(phi)
        if ok.sum() < 10 or np.std(values[ok]) == 0 or np.std(phi[ok]) == 0:
            return None
        r = float(np.corrcoef(pd.Series(values[ok]).rank(), pd.Series(phi[ok]).rank())[0, 1])
        if r > 0.3:
            return f"{higher} {column} pushes the prediction up."
        if r < -0.3:
            return f"{higher} {column} pushes the prediction down."
        return f"The effect of {column} goes both ways."
    labels = raw.astype(str).fillna("(empty)")
    means = pd.Series(phi, index=labels.index).groupby(labels).mean()
    if means.empty:
        return None
    return (
        f"{column} = {means.idxmax()} pushes the prediction up most; "
        f"{column} = {means.idxmin()} pulls it down most."
    )


def _label(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "(empty)"
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, pd.Timestamp):
        return (
            value.strftime("%Y-%m-%d")
            if value == value.normalize()
            else value.strftime("%Y-%m-%d %H:%M")
        )
    return str(value)[:40]


def shap_values(
    pipeline: Pipeline,
    x: pd.DataFrame,
    task: Task,
    rows: np.ndarray,
    shown: list[int],
    positive: str | None,
) -> tuple[Shap | None, str | None]:
    """SHAP values of the evaluation rows, summed per dataset column."""
    model = pipeline.named_steps["model"]
    picked = _sample(len(x), SHAP_ROWS, [int(i) for i in np.flatnonzero(np.isin(rows, shown))])
    xs = x.iloc[picked]
    z = np.asarray(_quiet(lambda: pipeline[:-1].transform(xs)), dtype=float)
    linear = _raw_linear(model, z, task)
    found = linear or _raw_boosting(model, z)
    method = "Exact (linear model)" if linear else "TreeSHAP (native)"
    # the log-odds of a binary model are those of its second class
    classes = [str(c) for c in getattr(pipeline, "classes_", [])]
    positive = classes[1] if task == "binary" and len(classes) == 2 else positive
    if found is None:
        name = model.__class__.__name__
        why = (
            "SHAP values are computed exactly for linear models (and XGBoost / LightGBM): "
            f"not available for {name}. Permutation importance and partial dependence apply to "
            "every model."
        )
        if task == "multiclass":
            why = "SHAP values are shown for numbers and two classes; with several classes, use "
            why += "permutation importance and partial dependence."
        return None, why
    phi, base, scale = found
    owner = origins(pipeline)
    columns = list(dict.fromkeys(owner))
    grouped = np.zeros((len(xs), len(columns)))
    for j, column in enumerate(owner):
        grouped[:, columns.index(column)] += phi[:, j]
    strength = np.abs(grouped).mean(axis=0)
    order = np.argsort(-strength)
    features: list[ShapFeature] = []
    for j in order[:TOP_FEATURES]:
        column = columns[j]
        raw = cast(pd.Series, xs[column]) if column in xs.columns else pd.Series(dtype=float)
        colors: list[float | None] = [None] * len(xs)
        number = _as_number(raw) if len(raw) else None
        if number is not None:
            ranks = number.rank(pct=True)
            colors = [None if pd.isna(v) else float(v) for v in ranks]
        features.append(
            ShapFeature(
                column,
                float(strength[j]),
                [float(v) for v in grouped[:, j]],
                colors,
                _reading(column, grouped[:, j], raw) if len(raw) else None,
            )
        )
    if len(order) > TOP_FEATURES:
        rest = order[TOP_FEATURES:]
        other = grouped[:, rest].sum(axis=1)
        features.append(
            ShapFeature(
                f"Other features ({len(rest)})",
                float(np.abs(other).mean()),
                [float(v) for v in other],
                [None] * len(xs),
                None,
            )
        )
    labels = [
        [_label(xs.iloc[i][c]) if c in xs.columns else "" for c in columns] for i in range(len(xs))
    ]
    return (
        Shap(
            method,
            scale,
            base,
            positive if task == "binary" else None,
            [int(v) for v in rows[picked]],
            features,
            columns,
            [[float(v) for v in row] for row in grouped],
            labels,
        ),
        None,
    )


# --- Partial dependence --------------------------------------------------------------------------


def dependence(
    pipeline: Pipeline, x: pd.DataFrame, task: Task, columns: list[str]
) -> list[Dependence]:
    """Average prediction when one column takes each value of a grid, the others as they are."""
    rows = _sample(len(x), DEPENDENCE_ROWS)
    xs = x.iloc[rows].copy()
    found = []
    classes = [str(c) for c in getattr(pipeline, "classes_", [])]
    for column in columns:
        raw = cast(pd.Series, xs[column])
        if pd.api.types.is_datetime64_any_dtype(raw):
            continue
        numeric = pd.api.types.is_numeric_dtype(raw) and not pd.api.types.is_bool_dtype(raw)
        try:
            if numeric:
                values = raw.to_numpy(dtype=float)
                if np.isnan(values).all():
                    continue
                low, high = np.nanpercentile(values, [5, 95])
                grid = np.unique(np.linspace(low, high, GRID))
                if len(grid) < 2:
                    continue
                result = _quiet(
                    lambda grid=grid, column=column: partial_dependence(
                        pipeline, xs, [column], custom_values={column: grid}, kind="average"
                    )
                )
                edges = np.concatenate([[-np.inf], (grid[1:] + grid[:-1]) / 2, [np.inf]])
                counts = np.histogram(values[np.isfinite(values)], bins=edges)[0]
                axis: list[str | float] = [float(v) for v in grid]
            else:
                labels = raw.astype(object).where(raw.notna(), None)
                top = [
                    str(v)
                    for v in list(raw.dropna().astype(str).value_counts().index)[:TOP_CATEGORIES]
                ]
                if len(top) < 2:
                    continue
                xs[column] = labels.astype(object)
                grid_values = np.array(top, dtype=object)
                result = _quiet(
                    lambda grid_values=grid_values, column=column: partial_dependence(
                        pipeline,
                        xs,
                        [column],
                        categorical_features=[column],
                        custom_values={column: grid_values},
                        kind="average",
                    )
                )
                counts = np.array([int((raw.astype(str) == v).sum()) for v in top])
                axis = list(top)
        except (ValueError, TypeError, KeyError):
            continue
        average = np.atleast_2d(np.asarray(result["average"], dtype=float))
        if task == "regression":
            curves, output = {"prediction": [float(v) for v in average[0]]}, "Average prediction"
        elif average.shape[0] == 1 and len(classes) == 2:
            curves = {classes[1]: [float(v) for v in average[0]]}
            output = f"Probability of “{classes[1]}”"
        else:
            curves = {classes[k]: [float(v) for v in average[k]] for k in range(average.shape[0])}
            output = "Probability of each class"
        found.append(
            Dependence(
                column,
                "number" if numeric else "category",
                axis,
                curves,
                [int(c) for c in counts],
                output,
            )
        )
    return found


# --- The tree itself, linked inputs ---------------------------------------------------------------


def tree(pipeline: Pipeline) -> list[TreeNode] | None:
    model = pipeline.named_steps["model"]
    structure = getattr(model, "tree_", None)
    if structure is None or model.__class__.__name__ not in (
        "DecisionTreeRegressor",
        "DecisionTreeClassifier",
        "ExtraTreeRegressor",
        "ExtraTreeClassifier",
    ):
        return None
    names = [str(n) for n in pipeline[:-1].get_feature_names_out()]  # type: ignore[union-attr]
    classes = [str(c) for c in getattr(model, "classes_", [])]
    nodes: list[TreeNode] = []
    queue = [(0, 0)]
    while queue:
        node, depth = queue.pop(0)
        left, right = int(structure.children_left[node]), int(structure.children_right[node])
        split = left != -1 and depth < TREE_DEPTH
        value = np.asarray(structure.value[node], dtype=float)[0]
        if classes:
            share = value / value.sum() if value.sum() else value
            best = int(np.argmax(share))
            text = f"{classes[best]} ({share[best]:.0%})"
        else:
            text = f"{value[0]:.4g}"
        nodes.append(
            TreeNode(
                node,
                depth,
                names[int(structure.feature[node])] if split else None,
                float(structure.threshold[node]) if split else None,
                text,
                int(structure.n_node_samples[node]),
                left if split else None,
                right if split else None,
            )
        )
        if split:
            queue += [(left, depth + 1), (right, depth + 1)]
    return nodes


def pairs(x: pd.DataFrame) -> list[Pair]:
    """Columns linked so strongly (|r|, Cramér's V or η > 0.8) that their importance is shared.
    Dates count as numbers (an age and a birth date are the same information)."""
    rows = _sample(len(x), PAIR_ROWS)
    xs = x.iloc[rows]
    columns = [str(c) for c in xs.columns][:PAIR_COLUMNS]
    values = {c: _as_number(cast(pd.Series, xs[c])) for c in columns}
    numeric = {c: values[c] is not None for c in columns}
    series = {c: values[c] if values[c] is not None else cast(pd.Series, xs[c]) for c in columns}
    found = []
    for i, first in enumerate(columns):
        for second in columns[i + 1 :]:
            strength = _quiet(
                lambda first=first, second=second: prep._association(
                    cast(pd.Series, series[first]),
                    cast(pd.Series, series[second]),
                    numeric[first],
                    numeric[second],
                )
            )
            if strength > STRONG_LINK:
                measure = (
                    "|r|"
                    if numeric[first] and numeric[second]
                    else "Cramér's V"
                    if not numeric[first] and not numeric[second]
                    else "η"
                )
                found.append(Pair(first, second, float(strength), measure))
    found.sort(key=lambda p: -p.strength)
    return found


def explain(
    pipeline: Pipeline,
    x: pd.DataFrame,
    y: pd.Series,
    rows: np.ndarray,
    task: Task,
    metric: str,
    shown: list[int],
    positive: str | None,
) -> Explanation:
    found_permutation = permutation(pipeline, x, y, task, metric)
    shap, unavailable = shap_values(pipeline, x, task, rows, shown, positive)
    strongest = [i.column for i in found_permutation.items if i.mean > 0][:DEPENDENCE_FEATURES]
    notes = []
    if found_permutation.sampled:
        notes.append(
            f"Permutation importance on {found_permutation.rows:,} of {len(x):,} rows (sample)."
        )
    return Explanation(
        found_permutation,
        shap,
        unavailable,
        dependence(pipeline, x, task, strongest),
        tree(pipeline),
        pairs(x),
        notes,
    )
