"""Describe segments with a tree (124): one sparse decision tree that predicts each row's segment
from the columns in their units, the sparse CART of 122 (pruned by 5-fold cross-validation, 1-SE
rule), so the whole segmentation reads as a few questions.

R-SGT-01: the tree learns on at most TREE_ROWS rows of the segmentation and is checked on rows it
did not learn from; its fidelity is the share of those rows it puts in their own segment.
R-SGT-02: 3 levels by default, more when there are more than 8 segments (a tree of d levels has
at most 2^d leaves); leaves 8 by default, at least one per segment.
R-SGT-03: each segment's rules are the paths to the leaves that name it, with the share of the
segment they find (recall) and the share of the rows they find that belong to it (precision).
R-SGT-04: below 80 % fidelity, the tree is a rough summary and the page says so (Rudin, 2019:
a readable model only explains what it reproduces)."""

import math
import sys
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import pandas as pd

from quantilo.filters.model import Condition
from quantilo.interpret import sparse_tree as sparse
from quantilo.ml.surrogate import SurrogateNode, _inputs, _nodes
from quantilo.segments.portrait import SegmentRule, _Feature, _simplify, percent

IN_BROWSER = sys.platform == "emscripten"
TREE_ROWS = 5_000 if IN_BROWSER else 10_000
HELD_OUT = 0.3
DEPTH = 3
LEAVES = 8
FIDELITY = 0.8
SEED = 42


@dataclass
class RuleFilter:
    """R-SGT-05 (124 bis): one rule of a segment as the app's filter conditions (all must hold)."""

    rule: str
    conditions: list[Condition]
    note: str | None  # numbers read as their median by the tree, left out by a filter


@dataclass
class SegmentTreeRules:
    segment: int
    name: str
    rules: list[SegmentRule]  # one per leaf that names the segment, the largest first
    recall: float | None  # share of the segment's checked rows the tree puts in it
    precision: float | None  # share of the rows it puts there that belong to it
    text: str
    filters: list[RuleFilter] = field(default_factory=list)  # R-SGT-05, one per rule


def rule_filter(rule: SegmentRule, empty: set[str] | None = None) -> RuleFilter:
    """R-SGT-05: "≤ t" → lte, "> t" → gt, "= a" → in [a], "≠ a" → not in [a]. A number the
    tree read with its empty values set to the median keeps only the rows with a value: said."""
    conditions: list[Condition] = []
    for part in rule.parts:
        if part.op == "<=":
            conditions.append(Condition(column=part.column, op="lte", value=float(part.value)))
        elif part.op == ">":
            conditions.append(Condition(column=part.column, op="gt", value=float(part.value)))
        else:
            conditions.append(
                Condition(
                    column=part.column, op="in", values=[str(part.value)], negate=part.op == "!="
                )
            )
    gaps = sorted(
        {p.column for p in rule.parts if p.op in ("<=", ">") and p.column in (empty or set())}
    )
    note = (
        f"R-SGT-05: the tree read the empty values of {', '.join(gaps)} as the median; the "
        "filter keeps only the rows with a value."
        if gaps
        else None
    )
    return RuleFilter(rule.text, conditions, note)


@dataclass
class SegmentTree:
    depth: int  # as asked (or chosen by R-SGT-02)
    found_depth: int
    leaves: int  # as found
    max_leaves: int  # as asked
    alpha: float  # the pruning kept (1-SE rule)
    fidelity: float  # share of the checked rows put in their own segment
    learnt: int
    checked: int
    nodes: list[SurrogateNode]
    pruning: list[sparse.Pruning]
    segments: list[SegmentTreeRules]
    sentence: str
    warning: str | None  # R-SGT-04


def defaults(segments: int) -> tuple[int, int]:
    """R-SGT-02: the depth and the leaves a segmentation of this many segments starts with."""
    depth = max(DEPTH, math.ceil(math.log2(max(segments, 2))))
    leaves = max(LEAVES, segments)
    return min(depth, sparse.MAX_DEPTH), min(leaves, sparse.MAX_LEAVES)


def _feature(item: Any, left: bool) -> tuple[_Feature, bool]:
    """The tree's question on `item`, as a part of a segment's rule (left: ≤ the threshold)."""
    if item.kind == "category":
        return _Feature(item.column, item.category), left
    if item.kind == "flag":
        return _Feature(item.column, "False" if left else "True"), False
    return _Feature(item.column, None), left


def describe(
    frame: pd.DataFrame,
    segments: np.ndarray,
    names: dict[int, str],
    columns: list[str],
    depth: int | None = None,
    leaves: int | None = None,
    seed: int = SEED,
) -> SegmentTree:
    """`frame`: the rows (the columns in their units); `segments`: each row's segment id."""
    ids = sorted(int(s) for s in np.unique(segments))
    if len(ids) < 2:
        raise sparse.SparseTreeError("A tree needs at least two segments.")
    chosen_depth, chosen_leaves = defaults(len(ids))
    depth = int(min(max(depth or chosen_depth, 1), sparse.MAX_DEPTH))
    leaves = int(min(max(leaves or chosen_leaves, 2), sparse.MAX_LEAVES))
    rng = np.random.default_rng(seed)
    order = np.arange(len(frame))
    if len(order) > TREE_ROWS:
        order = np.sort(rng.choice(len(order), TREE_ROWS, replace=False))
    order = rng.permutation(order)
    cut = max(2, int(len(order) * (1 - HELD_OUT)))
    learn_rows, check_rows = order[:cut], order[cut:] if cut < len(order) else order[:cut]
    labels = np.array([names.get(int(s), str(s)) for s in segments], dtype=object)
    data = frame[columns].reset_index(drop=True)
    learn = cast(pd.DataFrame, data.iloc[learn_rows]).reset_index(drop=True)
    check = cast(pd.DataFrame, data.iloc[check_rows]).reset_index(drop=True)
    target = labels[learn_rows].astype(str)
    truth = labels[check_rows].astype(str)
    if len(np.unique(target)) < 2:
        raise sparse.SparseTreeError("A tree needs at least two segments among its rows.")
    x_learn, items = _inputs(learn, set())
    if not items:
        raise sparse.SparseTreeError("No column a tree can split (numbers, flags, categories).")
    x_check = sparse._aligned(check, learn, items)
    leaf = max(5, int(sparse.MIN_LEAF * len(target)))
    alpha, tried = sparse.prune(x_learn, target, "multiclass", depth, leaves, leaf)
    tree = sparse._model("multiclass", depth, leaves, leaf, alpha).fit(x_learn, target)
    guessed = np.asarray(tree.predict(x_check)).astype(str)
    fidelity = float(np.mean(guessed == truth)) if len(truth) else 0.0
    landed = np.asarray(tree.apply(x_check))
    inner = tree.tree_
    classes = [str(c) for c in tree.classes_]

    # R-SGT-03: each leaf's path, read as a rule of the segment it names
    paths: dict[str, list[tuple[int, list[tuple[_Feature, bool, float]]]]] = {}

    def walk(node: int, path: list[tuple[_Feature, bool, float]]) -> None:
        left, right = int(inner.children_left[node]), int(inner.children_right[node])
        if left == -1:
            named = classes[int(np.argmax(inner.value[node][0]))]
            paths.setdefault(named, []).append((node, path))
            return
        item = items[int(inner.feature[node])]
        threshold = float(inner.threshold[node])
        feature, upper = _feature(item, True)
        walk(left, [*path, (feature, upper, threshold)])
        feature, upper = _feature(item, False)
        walk(right, [*path, (feature, upper, threshold)])

    walk(0, [])
    described: list[SegmentTreeRules] = []
    for segment in ids:
        name = names.get(segment, str(segment))
        mine = truth == name
        rules: list[SegmentRule] = []
        for node, path in paths.get(name, []):
            inside = landed == node
            parts = _simplify(path)
            text = " and ".join(p.text for p in parts) or "every row"
            members = int(inside.sum())
            found = int((inside & mine).sum())
            rules.append(SegmentRule(parts, text, found, found / members if members else 0.0))
        rules.sort(key=lambda r: -r.rows)
        given = guessed == name
        recall = float((given & mine).sum() / mine.sum()) if mine.any() else None
        precision = float((given & mine).sum() / given.sum()) if given.any() else None
        if not rules:
            text = (
                f"No leaf names {name}: the tree puts its rows in other segments. A deeper tree "
                "or more leaves may find it."
            )
        else:
            text = f"These rules find {percent(recall or 0.0)} of {name}" + (
                f" ({percent(precision)} of the rows they find belong to it)."
                if precision is not None
                else "."
            )
        empty = {c for c in columns if bool(cast(pd.Series, frame[c]).isna().any())}
        filters = [rule_filter(r, empty) for r in rules]
        described.append(SegmentTreeRules(segment, name, rules, recall, precision, text, filters))

    found_leaves = int(tree.get_n_leaves())
    found_depth = int(tree.get_depth())
    sentence = (
        f"A tree of {found_depth} level{'s' if found_depth != 1 else ''} and {found_leaves} leaves "
        f"(greedy CART, not optimal) puts {percent(fidelity)} of {len(truth):,} rows it did not "
        f"learn from in their own segment. Pruned by {sparse.FOLDS}-fold cross-validation, the "
        "simplest tree within one standard error of the best (R-SGT-01)."
    )
    warning = None
    if fidelity < FIDELITY:
        warning = (
            f"R-SGT-04: fidelity {percent(fidelity)}, under {percent(FIDELITY)}: the segments do "
            "not follow a few simple questions on the columns. Read the rules as a rough summary; "
            "the portraits above describe each segment better."
        )
    return SegmentTree(
        depth,
        found_depth,
        found_leaves,
        leaves,
        float(alpha),
        fidelity,
        len(target),
        len(truth),
        _nodes(tree, items, "multiclass"),
        tried,
        described,
        sentence,
        warning,
    )


def optimal_segment(
    frame: pd.DataFrame,
    segments: np.ndarray,
    names: dict[int, str],
    columns: list[str],
    segment: int,
    depth: int = DEPTH,
    seed: int = SEED,
) -> Any:
    """R-SGT-06 (124 bis): GOSDT's optimal tree for one segment against the others, on the rows
    and columns of the greedy tree (R-SGT-01), 70 / 30 as there. GOSDT learns two classes; it
    is an optional extra (R-XAI-10), refused with its reason when absent."""
    from quantilo.interpret import rashomon

    if segment not in names:
        raise sparse.SparseTreeError(f"Segment {segment} is not in this segmentation.")
    rng = np.random.default_rng(seed)
    order = np.arange(len(frame))
    if len(order) > TREE_ROWS:
        order = np.sort(rng.choice(len(order), TREE_ROWS, replace=False))
    order = rng.permutation(order)
    cut = max(2, int(len(order) * (1 - HELD_OUT)))
    learn_rows, check_rows = order[:cut], order[cut:]
    name = names[segment]
    labels = np.where(segments == segment, name, "other").astype(str)
    data = frame[columns].reset_index(drop=True)
    learn = cast(pd.DataFrame, data.iloc[learn_rows]).reset_index(drop=True)
    check = cast(pd.DataFrame, data.iloc[check_rows]).reset_index(drop=True)
    return rashomon.optimal_tree(
        learn, labels[learn_rows], check, labels[check_rows], name, min(depth, 5)
    )
