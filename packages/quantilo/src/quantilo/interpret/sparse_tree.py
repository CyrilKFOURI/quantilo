"""A sparse decision tree learnt on the target (122), the readable fallback of GOSDT everywhere,
the browser version included (scikit-learn only).

R-XAI-09: at most 4 levels (1 to 6) and 12 leaves (2 to 20), each leaf with at least 1 % of the
rows; the cost-complexity pruning is chosen by 5-fold cross-validation on the learnt rows with
the 1-SE rule: the simplest tree whose error is within one standard error of the best one. It is
CART: greedy, not optimal (GOSDT, under More, finds the optimal one when its extra is there)."""

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from quantilo.interpret import compare
from quantilo.ml.surrogate import SurrogateNode, _inputs, _nodes
from quantilo.ml.target import Task

SEED = 42
DEPTH = 4
MAX_DEPTH = 6
LEAVES = 12
MAX_LEAVES = 20
MIN_LEAF = 0.01
FOLDS = 5
ALPHAS = 30  # pruning strengths tried at most


class SparseTreeError(ValueError):
    pass


@dataclass
class Pruning:
    alpha: float
    leaves: int
    error: float  # mean cross-validated error (1 − accuracy, or mean squared error)
    se: float


@dataclass
class SparseTree:
    depth: int  # as asked
    leaves: int  # as found
    max_leaves: int  # as asked
    alpha: float  # the pruning kept (R-XAI-09)
    measure: str  # "ROC AUC", "accuracy" or "R²", on the checked rows
    score: float
    nodes: list[SurrogateNode]
    pruning: list[Pruning]
    gap: compare.Gap | None  # two classes: against the run's model (R-XAI-07)
    learnt: int
    checked: int
    sentence: str


def _model(task: Task, depth: int, leaves: int, leaf: int, alpha: float = 0.0) -> Any:
    common = {
        "max_depth": depth,
        "max_leaf_nodes": leaves,
        "min_samples_leaf": leaf,
        "random_state": SEED,
        "ccp_alpha": alpha,
    }
    return (
        DecisionTreeRegressor(**common)
        if task == "regression"
        else DecisionTreeClassifier(**common)
    )


def _error(task: Task, tree: Any, x: np.ndarray, y: np.ndarray) -> float:
    guessed = tree.predict(x)
    if task == "regression":
        return float(np.mean((guessed - y) ** 2))
    return float(np.mean(guessed != y))


def prune(
    x: np.ndarray, y: np.ndarray, task: Task, depth: int, leaves: int, leaf: int
) -> tuple[float, list[Pruning]]:
    """The pruning strength kept by the 1-SE rule, and every strength tried."""
    path = _model(task, depth, leaves, leaf).cost_complexity_pruning_path(x, y)
    alphas = np.unique(np.maximum(np.asarray(path.ccp_alphas, dtype=float), 0.0))
    if len(alphas) > ALPHAS:
        alphas = np.unique(np.quantile(alphas, np.linspace(0, 1, ALPHAS)))
    splitter = (
        KFold(FOLDS, shuffle=True, random_state=SEED)
        if task == "regression"
        else StratifiedKFold(FOLDS, shuffle=True, random_state=SEED)
    )
    folds = list(splitter.split(x, y if task != "regression" else None))
    tried: list[Pruning] = []
    for alpha in alphas:
        errors = []
        for learn, check in folds:
            tree = _model(task, depth, leaves, leaf, float(alpha)).fit(x[learn], y[learn])
            errors.append(_error(task, tree, x[check], y[check]))
        whole = _model(task, depth, leaves, leaf, float(alpha)).fit(x, y)
        tried.append(
            Pruning(
                float(alpha),
                int(whole.get_n_leaves()),
                float(np.mean(errors)),
                float(np.std(errors, ddof=1) / np.sqrt(len(errors))),
            )
        )
    best = min(tried, key=lambda p: p.error)
    simplest = max((p for p in tried if p.error <= best.error + best.se), key=lambda p: p.alpha)
    return simplest.alpha, tried


def sparse_tree(
    learn: pd.DataFrame,
    y_learn: np.ndarray,
    check: pd.DataFrame,
    y_check: np.ndarray,
    task: Task,
    depth: int = DEPTH,
    leaves: int = LEAVES,
    skip: set[str] | None = None,
    positive: str | None = None,
    model_score: np.ndarray | None = None,
    model_label: str = "model",
) -> SparseTree:
    """`y_*`: numbers, or class labels as text; with two classes, `positive` and the model's
    probability of it on the checked rows (`model_score`) give the comparison; for a number,
    `model_score` is the model's held-out prediction (R-XAI-19)."""
    depth = int(min(max(depth, 1), MAX_DEPTH))
    leaves = int(min(max(leaves, 2), MAX_LEAVES))
    x_learn, items = _inputs(learn, skip or set())
    if not items:
        raise SparseTreeError("No column a tree can split (numbers, dates, flags, categories).")
    x_check = _aligned(check, learn, items)
    target = y_learn.astype(float) if task == "regression" else y_learn.astype(str)
    truth = y_check.astype(float) if task == "regression" else y_check.astype(str)
    if task != "regression" and len(np.unique(target)) < 2:
        raise SparseTreeError("A tree needs at least two classes among the rows it learns from.")
    leaf = max(5, int(MIN_LEAF * len(target)))
    alpha, tried = prune(x_learn, target, task, depth, leaves, leaf)
    tree = _model(task, depth, leaves, leaf, alpha).fit(x_learn, target)
    nodes = _nodes(tree, items, task)
    found = int(tree.get_n_leaves())
    gap = None
    if task == "regression":
        guessed = tree.predict(x_check)
        score = compare.r2(guessed, truth)
        measure = "R²"
        if model_score is not None:  # 122 bis: the run's held-out predictions (R-XAI-19)
            gap = compare.r2_gap(
                np.asarray(model_score, dtype=float), guessed, truth, f"A {found}-leaf tree",
                model_label,
            )  # fmt: skip
    elif positive is not None and len(np.unique(target)) == 2:
        classes = [str(c) for c in tree.classes_]
        risk = tree.predict_proba(x_check)[:, classes.index(positive)]
        y01 = (truth == positive).astype(float)
        score = compare.auc(risk, y01)
        measure = "ROC AUC"
        if model_score is not None:
            gap = compare.gap(
                np.asarray(model_score, dtype=float),
                risk,
                y01,
                f"A {found}-leaf tree",
                model_label,
                np.asarray(model_score, dtype=float),
                risk,
            )
    else:
        score = float(np.mean(tree.predict(x_check) == truth))
        measure = "accuracy"
    shown = f"{score:.3f}" if measure != "accuracy" else f"{score:.0%}"
    sentence = (
        f"A tree of {tree.get_depth()} level{'s' if tree.get_depth() != 1 else ''} and {found} "
        f"leaves (greedy CART, not optimal): {measure} {shown} on {len(truth):,} rows it did not "
        f"learn from. Pruned by {FOLDS}-fold cross-validation, the simplest tree within one "
        "standard error of the best (R-XAI-09)."
    )
    return SparseTree(
        depth,
        found,
        leaves,
        float(alpha),
        measure,
        float(score),
        nodes,
        tried,
        gap,
        len(target),
        len(truth),
        sentence,
    )


def _aligned(check: pd.DataFrame, learn: pd.DataFrame, items: list[Any]) -> np.ndarray:
    """The checked rows on the learnt inputs: the same categories, the same columns, the
    learnt medians for the empty numbers."""
    blocks = []
    for item in items:
        raw = cast(pd.Series, check[item.column])
        if item.kind == "category":
            labels = raw.astype(str).where(raw.notna(), "(empty)")
            blocks.append((labels == item.category).to_numpy(dtype=float))
        elif item.kind == "flag":
            blocks.append(raw.fillna(False).astype(float).to_numpy())
        elif item.kind == "date":
            seconds = (raw - pd.Timestamp("1970-01-01")).dt.total_seconds()
            learnt = (cast(pd.Series, learn[item.column]) - pd.Timestamp("1970-01-01")).dt
            blocks.append(seconds.fillna(learnt.total_seconds().median()).to_numpy(dtype=float))
        else:
            median = float(cast(pd.Series, learn[item.column]).astype(float).median())
            blocks.append(raw.astype(float).fillna(median).to_numpy())
    return np.nan_to_num(np.column_stack(blocks)) if blocks else np.zeros((len(check), 0))
