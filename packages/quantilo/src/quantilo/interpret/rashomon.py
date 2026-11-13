"""Optimal sparse trees (GOSDT) and the Rashomon set of near-optimal trees (TreeFARMS), behind
the optional `interpret` extra (122). Neither package is imported before a call (R-XAI-10).

GOSDT with threshold guessing (McTavish, Zhong, Achermann, Karimalis, Chen, Rudin, Seltzer,
AAAI 2022) and TreeFARMS (Xin, Zhong, Chen, Takagi, Seltzer, Rudin, NeurIPS 2022), BSD-3. Both
learn on the conditions of R-XAI-06 (true / false), at most 5 000 rows (R-XAI-11).

A tree comes back from either solver as its JSON source ({"feature", "true", "false"} for a
split, {"prediction"} for a leaf); when a version keeps it elsewhere, the tree is read back from
its predictions on the learnt rows (a scikit-learn tree fitted to them, exact on those rows)."""

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from quantilo.interpret import compare, extras
from quantilo.interpret.conditions import Condition, conditions_for, matrix

SEED = 42
ROWS = 5_000  # R-XAI-11
DEPTH = 4
REGULARIZATION = 0.01  # GOSDT's penalty per leaf
TIME_LIMIT = 60  # seconds
RASHOMON = 0.05  # trees within 5 % of the best objective
COUNTED = 10_000
READ = 50  # trees of the set read one by one


class RashomonError(ValueError):
    pass


@dataclass
class OptimalNode:
    id: int
    depth: int
    value: str  # the leaf's class and share, or the split's majority
    samples: int
    left: int | None  # the branch where the condition does not hold
    right: int | None  # the branch where it holds
    left_rule: str | None
    right_rule: str | None


@dataclass
class OptimalTree:
    depth: int
    leaves: int
    score: float  # ROC AUC on the checked rows
    nodes: list[OptimalNode]
    gap: compare.Gap | None
    learnt: int
    checked: int
    read_back: bool  # the tree was read back from its predictions
    sentence: str


@dataclass
class RashomonColumn:
    column: str
    trees: int  # trees of the set that use it
    share: float


@dataclass
class RashomonSet:
    trees: int  # trees within the bound (counted up to 10 000)
    read: int  # trees read one by one
    bound: float  # the tolerance on the objective
    accuracy_low: float  # on the checked rows, among the trees read
    accuracy_high: float
    columns: list[RashomonColumn]  # used by the trees read, most used first
    always: list[str]  # used by every tree read
    sentence: str


def _binary(
    learn: pd.DataFrame, skip: set[str] | None, types: dict[str, str] | None
) -> tuple[list[Condition], pd.DataFrame]:
    conditions = conditions_for(learn, skip, types)
    if not conditions:
        raise RashomonError("No column gives a condition a tree can use (R-XAI-06).")
    frame = pd.DataFrame(
        matrix(learn, conditions).astype(int), columns=[c.text for c in conditions]
    )
    return conditions, frame


def _sample(n: int) -> np.ndarray:
    if n <= ROWS:
        return np.arange(n)
    return np.sort(np.random.default_rng(SEED).choice(n, ROWS, replace=False))


def _source(tree: Any) -> dict[str, Any] | None:
    """The JSON tree a solver keeps, wherever this version keeps it."""
    for candidate in (tree, *getattr(tree, "trees_", [])[:1]):
        for name in ("source", "tree", "tree_"):
            found = getattr(candidate, name, None)
            if callable(found):
                continue
            if isinstance(found, dict) and ("feature" in found or "prediction" in found):
                return cast(dict[str, Any], found)
    return None


def _nodes(
    source: dict[str, Any], conditions: list[Condition], x: np.ndarray, y: np.ndarray
) -> list[OptimalNode]:
    found: list[OptimalNode] = []
    queue: list[tuple[dict[str, Any], int, np.ndarray]] = [(source, 0, np.ones(len(y), bool))]
    while queue:
        node, depth, rows = queue.pop(0)
        index = len(found)
        labels, counts = np.unique(y[rows], return_counts=True)
        value = (
            f"{labels[int(np.argmax(counts))]} ({counts.max() / counts.sum():.0%})"
            if len(counts)
            else str(node.get("prediction", ""))
        )
        found.append(OptimalNode(index, depth, value, int(rows.sum()), None, None, None, None))
        if "feature" in node and "true" in node and "false" in node:
            j = int(node["feature"])
            holds = x[:, j] == 1
            text = conditions[j].text
            found[index].left_rule = f"not {text}"
            found[index].right_rule = text
            found[index].left = -1  # set when the child is numbered
            found[index].right = -1
            queue.append((cast(dict[str, Any], node["false"]), depth + 1, rows & ~holds))
            queue.append((cast(dict[str, Any], node["true"]), depth + 1, rows & holds))
    # breadth first: children are numbered in the order they were queued
    nxt = 1
    for node in found:
        if node.left == -1:
            node.left, node.right = nxt, nxt + 1
            nxt += 2
    return found


def _read_back(predicted: np.ndarray, x: np.ndarray, depth: int) -> dict[str, Any]:
    """A tree that gives the solver's predictions on the learnt rows, as JSON."""
    tree = DecisionTreeClassifier(max_depth=depth + 2, random_state=SEED).fit(x, predicted)
    structure = tree.tree_
    classes = [str(c) for c in tree.classes_]

    def build(node: int) -> dict[str, Any]:
        left, right = int(structure.children_left[node]), int(structure.children_right[node])
        if left == -1:
            return {"prediction": classes[int(np.argmax(structure.value[node][0]))]}
        # x ≤ 0.5 is "does not hold" (left), > 0.5 "holds" (right)
        return {"feature": int(structure.feature[node]), "false": build(left), "true": build(right)}

    return build(0)


def optimal_tree(
    learn: pd.DataFrame,
    y_learn: np.ndarray,
    check: pd.DataFrame,
    y_check: np.ndarray,
    positive: str,
    depth: int = DEPTH,
    skip: set[str] | None = None,
    types: dict[str, str] | None = None,
    model_score: np.ndarray | None = None,
    model_label: str = "model",
) -> OptimalTree:
    """GOSDT's optimal sparse tree for two classes (labels as text, `positive` the event)."""
    gosdt = extras.require("gosdt")
    conditions, frame = _binary(learn, skip, types)
    rows = _sample(len(frame))
    x = frame.to_numpy()[rows]
    y01 = (np.asarray(y_learn).astype(str)[rows] == positive).astype(int)
    model = gosdt.GOSDTClassifier(
        regularization=REGULARIZATION, depth_budget=depth + 1, time_limit=TIME_LIMIT, verbose=False
    )
    model.fit(pd.DataFrame(x, columns=frame.columns), pd.Series(y01))
    source = _source(model)
    read_back = source is None
    if source is None:
        predicted = np.asarray(model.predict(pd.DataFrame(x, columns=frame.columns))).astype(int)
        source = _read_back(predicted, x, depth)
    labels = np.where(y01 == 1, positive, "other")
    nodes = _nodes(source, conditions, x, labels)
    checked = matrix(check, conditions)
    risk = _leaf_risk(source, x, y01, checked)
    truth = (np.asarray(y_check).astype(str) == positive).astype(float)
    score = compare.auc(risk, truth)
    leaves = sum(1 for n in nodes if n.left is None)
    gap = None
    if model_score is not None:
        gap = compare.gap(
            np.asarray(model_score, float),
            risk,
            truth,
            f"An optimal {leaves}-leaf tree",
            model_label,
            np.asarray(model_score, float),
            risk,
        )
    sentence = (
        f"The optimal tree of {max(n.depth for n in nodes)} levels and {leaves} leaves (GOSDT, "
        f"penalty {REGULARIZATION} per leaf): ROC AUC {score:.3f} on {len(truth):,} rows it did "
        "not learn from."
    )
    return OptimalTree(depth, leaves, score, nodes, gap, len(rows), len(truth), read_back, sentence)


def _leaf_risk(
    source: dict[str, Any], x: np.ndarray, y01: np.ndarray, checked: np.ndarray
) -> np.ndarray:
    """Each checked row's share of positives in its leaf, as learnt (smoothed by one row)."""

    def walk(node: dict[str, Any], learnt: np.ndarray, rows: np.ndarray, out: np.ndarray) -> None:
        if "feature" not in node:
            share = (y01[learnt].sum() + 0.5) / (learnt.sum() + 1.0)
            out[rows] = share
            return
        j = int(node["feature"])
        walk(
            cast(dict[str, Any], node["true"]),
            learnt & (x[:, j] == 1),
            rows & (checked[:, j] == 1),
            out,
        )
        walk(
            cast(dict[str, Any], node["false"]),
            learnt & (x[:, j] != 1),
            rows & (checked[:, j] != 1),
            out,
        )

    out = np.zeros(len(checked))
    walk(source, np.ones(len(x), bool), np.ones(len(checked), bool), out)
    return out


def _features(source: dict[str, Any]) -> set[int]:
    if "feature" not in source:
        return set()
    return (
        {int(source["feature"])}
        | _features(cast(dict[str, Any], source["true"]))
        | _features(cast(dict[str, Any], source["false"]))
    )


def rashomon_set(
    learn: pd.DataFrame,
    y_learn: np.ndarray,
    check: pd.DataFrame,
    y_check: np.ndarray,
    positive: str,
    depth: int = DEPTH,
    skip: set[str] | None = None,
    types: dict[str, str] | None = None,
) -> RashomonSet:
    """TreeFARMS: every tree within 5 % of the best objective; the columns they rely on."""
    treefarms = extras.require("treefarms")
    conditions, frame = _binary(learn, skip, types)
    rows = _sample(len(frame))
    x = frame.iloc[rows].reset_index(drop=True)
    y01 = pd.Series((np.asarray(y_learn).astype(str)[rows] == positive).astype(int))
    model = treefarms.TREEFARMS(
        {
            "regularization": REGULARIZATION,
            "rashomon_bound_multiplier": RASHOMON,
            "depth_budget": depth + 1,
            "time_limit": TIME_LIMIT,
            "verbose": False,
        }
    )
    model.fit(x, y01)
    count = int(model.get_tree_count())
    if not count:
        raise RashomonError("TreeFARMS found no tree within the bound.")
    checked = pd.DataFrame(matrix(check, conditions).astype(int), columns=frame.columns)
    truth = (np.asarray(y_check).astype(str) == positive).astype(int)
    used: dict[str, int] = {}
    accuracies = []
    every: set[str] | None = None
    read = min(count, READ)
    for i in range(read):
        tree = model[i]
        source = _source(tree)
        columns = {conditions[j].column for j in _features(source)} if source is not None else set()
        for column in columns:
            used[column] = used.get(column, 0) + 1
        every = columns if every is None else every & columns
        accuracies.append(float(np.mean(np.asarray(tree.predict(checked)).astype(int) == truth)))
    ranked = sorted(used.items(), key=lambda item: (-item[1], item[0]))
    always = sorted(every or set())
    columns = [RashomonColumn(c, n, n / read) for c, n in ranked]
    shown = ", ".join(always) if always else "none"
    counted = f"{count:,}" if count < COUNTED else f"at least {COUNTED:,}"
    sentence = (
        f"{counted} trees are within {RASHOMON:.0%} of the best objective (TreeFARMS, at most "
        f"{depth} levels); on the {read} read, accuracy {min(accuracies):.1%} to "
        f"{max(accuracies):.1%} on {len(truth):,} rows they did not learn from. Columns used by "
        f"all of them: {shown}."
    )
    return RashomonSet(
        count,
        read,
        RASHOMON,
        min(accuracies),
        max(accuracies),
        columns,
        always,
        sentence,
    )
