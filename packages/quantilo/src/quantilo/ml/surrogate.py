"""A global surrogate (125): a shallow tree trained on the model's own predictions, read on the
dataset's columns in their units (one-hot of the frequent categories, dates as dates). Its
fidelity says how far to trust it: R² against the model's predictions (a number) or the share of
rows where it gives the model's class (classes), measured on rows the surrogate did not see.

R-XAI-04 (Rudin, 2019): below 80 % fidelity the tree explains itself more than the model, and
the page says so instead of letting it pass for the model's logic."""

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from quantilo.ml.target import Task

SEED = 42
ROWS = 5_000  # rows the surrogate learns from and is checked on
HELD_OUT = 0.3
DEPTH = 3
MAX_DEPTH = 5
TOP_CATEGORIES = 12  # one-hot of the most frequent categories of a column
FIDELITY = 0.8  # R-XAI-04


@dataclass
class SurrogateNode:
    id: int
    depth: int
    value: str  # mean prediction, or the majority class and its share
    samples: int
    left: int | None
    right: int | None
    left_rule: str | None  # the question, as it reads on the left branch
    right_rule: str | None


@dataclass
class Surrogate:
    depth: int
    measure: str  # "R²" or "agreement"
    fidelity: float
    rows: int  # rows it learnt from
    checked: int  # rows it was checked on
    leaves: int
    nodes: list[SurrogateNode]
    sentence: str
    warning: str | None  # R-XAI-04


@dataclass
class _Input:
    column: str
    kind: str  # "number", "date", "flag" or "category"
    category: str | None = None


def _inputs(x: pd.DataFrame, skip: set[str]) -> tuple[np.ndarray, list[_Input]]:
    """The columns as numbers a tree can split, keeping their units."""
    blocks: list[np.ndarray] = []
    found: list[_Input] = []
    for column in [str(c) for c in x.columns]:
        if column in skip:
            continue
        raw = cast(pd.Series, x[column])
        if pd.api.types.is_bool_dtype(raw):
            blocks.append(raw.fillna(False).astype(float).to_numpy()[:, None])
            found.append(_Input(column, "flag"))
        elif pd.api.types.is_datetime64_any_dtype(raw):
            seconds = (raw - pd.Timestamp("1970-01-01")).dt.total_seconds()
            blocks.append(seconds.fillna(seconds.median()).to_numpy(dtype=float)[:, None])
            found.append(_Input(column, "date"))
        elif pd.api.types.is_numeric_dtype(raw):
            values = raw.astype(float)
            blocks.append(values.fillna(values.median()).to_numpy()[:, None])
            found.append(_Input(column, "number"))
        else:
            labels = raw.astype(str).where(raw.notna(), "(empty)")
            for category in list(labels.value_counts().index)[:TOP_CATEGORIES]:
                blocks.append((labels == category).to_numpy(dtype=float)[:, None])
                found.append(_Input(column, "category", str(category)))
    matrix = np.hstack(blocks) if blocks else np.zeros((len(x), 0))
    return np.nan_to_num(matrix), found


def _rules(item: _Input, threshold: float) -> tuple[str, str]:
    if item.kind == "category":
        return f"{item.column} ≠ {item.category}", f"{item.column} = {item.category}"
    if item.kind == "flag":
        return f"{item.column} = False", f"{item.column} = True"
    if item.kind == "date":
        moment = pd.Timestamp("1970-01-01") + pd.Timedelta(seconds=float(threshold))
        shown = moment.strftime("%Y-%m-%d")
        return f"{item.column} ≤ {shown}", f"{item.column} > {shown}"
    return f"{item.column} ≤ {threshold:.4g}", f"{item.column} > {threshold:.4g}"


def _fit(pipeline: Pipeline, x: pd.DataFrame, task: Task, depth: int, skip: set[str]):
    rows = (
        np.arange(len(x))
        if len(x) <= ROWS
        else np.sort(np.random.default_rng(SEED).choice(len(x), ROWS, replace=False))
    )
    xs = x.iloc[rows]
    matrix, items = _inputs(xs, skip)
    if task == "regression":
        target: Any = np.asarray(pipeline.predict(xs), dtype=float)
    else:
        target = np.asarray(pipeline.predict(xs)).astype(str)
    order = np.random.default_rng(SEED).permutation(len(xs))
    cut = max(1, int(len(xs) * (1 - HELD_OUT)))
    learn, check = order[:cut], order[cut:]
    leaf = max(5, len(learn) // 100)
    tree = (
        DecisionTreeRegressor(max_depth=depth, min_samples_leaf=leaf, random_state=SEED)
        if task == "regression"
        else DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf, random_state=SEED)
    )
    tree.fit(matrix[learn], target[learn])
    return tree, matrix, items, target, learn, check


def surrogate(
    pipeline: Pipeline,
    x: pd.DataFrame,
    task: Task,
    depth: int = DEPTH,
    skip: set[str] | None = None,
) -> Surrogate:
    """`skip`: the free-text columns (a tree cannot split them)."""
    depth = int(min(max(depth, 1), MAX_DEPTH))
    tree, matrix, items, target, learn, check = _fit(pipeline, x, task, depth, skip or set())
    if not len(check):
        check = learn
    guessed = tree.predict(matrix[check])
    if task == "regression":
        actual = target[check]
        spread = float(((actual - actual.mean()) ** 2).sum())
        fidelity = 1 - float(((actual - guessed) ** 2).sum()) / spread if spread > 0 else 1.0
        measure = "R²"
    else:
        fidelity = float((guessed == target[check]).mean())
        measure = "agreement"
    nodes = _nodes(tree, items, task)
    leaves = int(tree.get_n_leaves())
    reading = (
        f"explains {fidelity:.0%} of the variation of the model's predictions"
        if measure == "R²"
        else f"gives the model's class on {fidelity:.0%} of the rows"
    )
    sentence = (
        f"A tree of {depth} levels ({leaves} leaves) {reading}, on {len(check):,} rows it did not "
        "learn from."
    )
    warning = None
    if fidelity < FIDELITY:
        warning = (
            f"R-XAI-04: fidelity {fidelity:.0%}, under {FIDELITY:.0%}: this tree explains itself "
            "more than the model. Do not read the model's logic in it; a deeper tree, or the "
            "SHAP values and the effects above, say more."
        )
    return Surrogate(
        depth, measure, float(fidelity), len(learn), len(check), leaves, nodes, sentence, warning
    )


def _nodes(tree: Any, items: list[_Input], task: Task) -> list[SurrogateNode]:
    structure = tree.tree_
    classes = [str(c) for c in getattr(tree, "classes_", [])]
    found = []
    queue = [(0, 0)]
    while queue:
        node, depth = queue.pop(0)
        left, right = int(structure.children_left[node]), int(structure.children_right[node])
        value = np.asarray(structure.value[node], dtype=float)[0]
        if task == "regression":
            text = f"≈ {value[0]:.4g}"
        else:
            share = value / value.sum() if value.sum() else value
            best = int(np.argmax(share))
            text = f"{classes[best]} ({share[best]:.0%})"
        split = left != -1
        rules = (
            _rules(items[int(structure.feature[node])], float(structure.threshold[node]))
            if split
            else (None, None)
        )
        found.append(
            SurrogateNode(
                node,
                depth,
                text,
                int(structure.n_node_samples[node]),
                left if split else None,
                right if split else None,
                rules[0],
                rules[1],
            )
        )
        if split:
            queue += [(left, depth + 1), (right, depth + 1)]
    return found
