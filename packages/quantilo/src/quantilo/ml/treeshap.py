# ruff: noqa: RUF002, RUF003  (φ, Π, Σ, × are the usual notation)
# pyright: reportAttributeAccessIssue=false
# (fitted scikit-learn trees: the stubs cannot follow their private arrays)
"""TreeSHAP for the trees of scikit-learn (125): decision trees, random forests, extra trees,
bagged trees, gradient boosting and histogram gradient boosting, in numpy only (so it runs in the
browser version, 102). Path-dependent and exact, as Lundberg et al. (2020) define it: the values
of one row add up to its raw prediction minus the cover-weighted average of the leaves.

How: every root-to-leaf path is one small game. Along a path, a feature k has a "zero fraction"
z_k (the share of the training rows that follow the path at its splits, i.e. the product of the
cover ratios) and a "one fraction" o_k(x) ∈ {0, 1} (does row x satisfy every split of the path on
k?). The leaf's part of v(S) is v_leaf × Π_{k∈S} o_k × Π_{k∉S} z_k, so the Shapley value of i is
v_leaf (o_i − z_i) Σ_s w(s, d) [t^s] Π_{k≠i} (z_k + o_k t), with w(s, d) = s! (d − s − 1)! / d!.
The polynomial is built once per path and divided by (z_i + o_i t) for each i, every path and
every row at once (numpy arrays rows × paths × features). Padding a path with features whose
o = z = 1 adds null players, which leave the Shapley values unchanged: paths of different
lengths share one array.

R-XAI-02 (cost): the work grows with rows × Σ over the leaves of (path length)²; the rows
explained are reduced to stay under a budget (fewer in the browser), the shown rows first, and
the explanation says so. A forest too large for even 50 rows gets no SHAP values, with the reason
and what to change (limit the tree depth)."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from quantilo.ml import browser

BUDGET = 1.2e9  # rows × Σ (splits + 1)(distinct inputs + 1) over the paths: about 10 s on a laptop
BROWSER_BUDGET = 2.5e8  # one core, WebAssembly (R-WEB-02)
MIN_ROWS = 50  # fewer and the beeswarm says nothing: no SHAP values then
CHUNK = 2_500_000  # elements of one rows × paths × features array


@dataclass
class Tree:
    """One tree as flat arrays (leaves: left = right = −1)."""

    left: np.ndarray
    right: np.ndarray
    feature: np.ndarray
    threshold: np.ndarray
    missing_left: np.ndarray  # a missing value goes left
    cover: np.ndarray  # training rows (or weights) at each node
    value: np.ndarray  # the leaf's value on the explained output
    float32: bool = False  # scikit-learn's trees compare float32 values
    columns: np.ndarray | None = None  # bagging: the inputs this tree sees
    depth: int = 0


@dataclass
class Ensemble:
    trees: list[Tree]
    weights: list[float]  # each tree's multiplier (1 / trees, the learning rate…)
    offset: float  # added to every prediction (the boosting's starting value)
    scale: str  # "prediction", "probability", "log-odds" or "log"
    label: str  # the model's family, for the sentence
    leaves: int = 0
    depth: int = 0


@dataclass
class Paths:
    """Every root-to-leaf path of an ensemble, padded to one length."""

    value: np.ndarray  # (L,) leaf value × the tree's weight
    feature: np.ndarray  # (L, D) inputs along the path, −1 = padding
    threshold: np.ndarray  # (L, D)
    left: np.ndarray  # (L, D) the path goes left at this split
    missing_left: np.ndarray  # (L, D)
    ratio: np.ndarray  # (L, D) cover of the child / cover of the node
    float32: np.ndarray  # (L,) the tree compares float32 values
    length: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))


# --- From scikit-learn's models to flat trees ----------------------------------------------------


def _sk_tree(estimator: Any, positive: int | None, columns: np.ndarray | None = None) -> Tree:
    """A scikit-learn tree; `positive`: the class whose probability is explained (None: the
    regression value)."""
    structure = estimator.tree_
    values = np.asarray(structure.value, dtype=float)[:, 0, :]
    if positive is None:
        leaf = values[:, 0]
    else:
        totals = values.sum(axis=1)
        leaf = np.where(totals > 0, values[:, positive] / np.where(totals > 0, totals, 1), 0.0)
    missing = getattr(structure, "missing_go_to_left", None)
    left = np.asarray(structure.children_left, dtype=np.int64)
    return Tree(
        left,
        np.asarray(structure.children_right, dtype=np.int64),
        np.asarray(structure.feature, dtype=np.int64),
        np.asarray(structure.threshold, dtype=float),
        np.zeros(len(left), dtype=bool) if missing is None else np.asarray(missing, dtype=bool),
        np.asarray(structure.weighted_n_node_samples, dtype=float),
        leaf,
        float32=True,
        columns=columns,
        depth=int(structure.max_depth),
    )


def _hist_tree(predictor: Any) -> Tree | None:
    nodes = predictor.nodes
    if bool(np.asarray(nodes["is_categorical"]).any()):
        return None  # native categorical splits: bitsets, not thresholds
    leaf = np.asarray(nodes["is_leaf"], dtype=bool)
    return Tree(
        np.where(leaf, -1, np.asarray(nodes["left"], dtype=np.int64)),
        np.where(leaf, -1, np.asarray(nodes["right"], dtype=np.int64)),
        np.asarray(nodes["feature_idx"], dtype=np.int64),
        np.asarray(nodes["num_threshold"], dtype=float),
        np.asarray(nodes["missing_go_to_left"], dtype=bool),
        np.asarray(nodes["count"], dtype=float),
        np.asarray(nodes["value"], dtype=float),
        depth=int(np.asarray(nodes["depth"]).max()),
    )


def _binary_class(model: Any) -> int | None:
    """The index of the explained class (the second, like the log-odds elsewhere)."""
    classes = getattr(model, "classes_", None)
    return 1 if classes is not None and len(classes) == 2 else None


def ensemble(model: Any, task: str) -> Ensemble | None:
    """The trees of a scikit-learn tree model, or None (another model, several classes,
    native categorical splits, a starting value that is not a constant)."""
    if task == "multiclass":
        return None
    name = model.__class__.__name__
    classifier = task != "regression"
    positive = _binary_class(model) if classifier else None
    if classifier and positive is None:
        return None
    if name in (
        "DecisionTreeRegressor",
        "ExtraTreeRegressor",
        "DecisionTreeClassifier",
        "ExtraTreeClassifier",
    ):
        if getattr(model, "n_outputs_", 1) != 1:
            return None
        scale = "probability" if classifier else "prediction"
        return Ensemble([_sk_tree(model, positive)], [1.0], 0.0, scale, "decision tree")
    if name in (
        "RandomForestRegressor",
        "RandomForestClassifier",
        "ExtraTreesRegressor",
        "ExtraTreesClassifier",
    ):
        trees = [_sk_tree(e, positive) for e in model.estimators_]
        scale = "probability" if classifier else "prediction"
        return Ensemble(trees, [1 / len(trees)] * len(trees), 0.0, scale, "forest")
    if name in ("BaggingRegressor", "BaggingClassifier"):
        members = list(model.estimators_)
        if not members or not all(hasattr(e, "tree_") for e in members):
            return None
        if classifier and not all(len(getattr(e, "classes_", [])) == 2 for e in members):
            return None  # a member that saw one class only: its probabilities do not line up
        trees = [
            _sk_tree(e, positive, np.asarray(f, dtype=np.int64))
            for e, f in zip(members, model.estimators_features_, strict=True)
        ]
        scale = "probability" if classifier else "prediction"
        return Ensemble(trees, [1 / len(trees)] * len(trees), 0.0, scale, "bagging")
    if name in ("GradientBoostingRegressor", "GradientBoostingClassifier"):
        stages = np.asarray(model.estimators_)
        if stages.ndim != 2 or stages.shape[1] != 1:
            return None
        start = np.ravel(np.asarray(model._raw_predict_init(np.zeros((2, model.n_features_in_)))))
        if len(start) != 2 or start[0] != start[1]:
            return None  # a starting model that depends on the row
        trees = [_sk_tree(e, None) for e in stages[:, 0]]
        rate = float(model.learning_rate)
        scale = "log-odds" if classifier else "prediction"
        return Ensemble(trees, [rate] * len(trees), float(start[0]), scale, "gradient boosting")
    if name in ("HistGradientBoostingRegressor", "HistGradientBoostingClassifier"):
        if getattr(model, "_preprocessor", None) is not None:
            return None
        predictors = model._predictors
        if any(len(p) != 1 for p in predictors):
            return None
        found = [_hist_tree(p[0]) for p in predictors]
        if any(t is None for t in found):
            return None
        loss = str(getattr(model, "loss", ""))
        scale = (
            "log-odds" if classifier else "log" if loss in ("poisson", "gamma") else "prediction"
        )
        offset = float(np.ravel(np.asarray(model._baseline_prediction, dtype=float))[0])
        trees = [t for t in found if t is not None]
        return Ensemble(trees, [1.0] * len(trees), offset, scale, "histogram gradient boosting")
    return None


# --- Paths ---------------------------------------------------------------------------------------


def _tree_paths(tree: Tree, weight: float) -> dict[str, list[np.ndarray]]:
    """The root-to-leaf paths of one tree, level by level (every node of a level at once)."""
    out: dict[str, list[np.ndarray]] = {k: [] for k in ("value", "f", "t", "l", "m", "r")}
    nodes = np.array([0], dtype=np.int64)
    path: dict[str, np.ndarray] = {k: np.zeros((1, 0)) for k in ("f", "t", "l", "m", "r")}
    path["f"] = np.zeros((1, 0), dtype=np.int64)

    while nodes.size:
        leaf = tree.left[nodes] == -1
        if leaf.any():
            out["value"].append(tree.value[nodes[leaf]] * weight)
            for k in ("f", "t", "l", "m", "r"):
                out[k].append(path[k][leaf])
        inner = nodes[~leaf]
        if not inner.size:
            break
        kept = {k: path[k][~leaf] for k in path}
        lefts, rights = tree.left[inner], tree.right[inner]
        feature = tree.feature[inner]
        if tree.columns is not None:
            feature = tree.columns[feature]
        cover = np.where(tree.cover[inner] > 0, tree.cover[inner], 1.0)
        added = {
            "f": np.concatenate([feature, feature]),
            "t": np.concatenate([tree.threshold[inner]] * 2),
            "l": np.concatenate([np.ones(len(inner)), np.zeros(len(inner))]),
            "m": np.concatenate([tree.missing_left[inner]] * 2).astype(float),
            "r": np.concatenate([tree.cover[lefts] / cover, tree.cover[rights] / cover]),
        }
        path = {
            k: np.concatenate([np.concatenate([kept[k], kept[k]]), added[k][:, None]], axis=1)
            for k in kept
        }
        nodes = np.concatenate([lefts, rights])
    return out


def paths(found: Ensemble) -> Paths:
    parts: dict[str, list[np.ndarray]] = {k: [] for k in ("value", "f", "t", "l", "m", "r")}
    flags: list[np.ndarray] = []
    for tree, weight in zip(found.trees, found.weights, strict=True):
        one = _tree_paths(tree, weight)
        for k in parts:
            parts[k] += one[k]
        flags += [np.full(len(v), tree.float32) for v in one["value"]]
    depth = max((p.shape[1] for p in parts["f"]), default=0)

    def pad(blocks: list[np.ndarray], fill: float) -> np.ndarray:
        return np.concatenate(
            [
                np.pad(b.astype(float), ((0, 0), (0, depth - b.shape[1])), constant_values=fill)
                for b in blocks
            ]
        )

    feature = pad(parts["f"], -1).astype(np.int64)
    return Paths(
        np.concatenate(parts["value"]),
        feature,
        pad(parts["t"], 0.0),
        pad(parts["l"], 0.0).astype(bool),
        pad(parts["m"], 0.0).astype(bool),
        pad(parts["r"], 1.0),
        np.concatenate(flags).astype(bool),
        (feature >= 0).sum(axis=1),
    )


def _slots(feature: np.ndarray) -> tuple[np.ndarray, int]:
    """For each split of each path, the index of its feature among the path's distinct features
    (−1 for padding), and the largest number of distinct features."""
    big = np.iinfo(np.int64).max
    key = np.where(feature < 0, big, feature)
    order = np.argsort(key, axis=1, kind="stable")
    ranked = np.take_along_axis(key, order, axis=1)
    new = np.ones_like(ranked, dtype=bool)
    new[:, 1:] = ranked[:, 1:] != ranked[:, :-1]
    slot_sorted = np.cumsum(new, axis=1) - 1
    slot = np.empty_like(slot_sorted)
    np.put_along_axis(slot, order, slot_sorted, axis=1)
    slot = np.where(feature < 0, -1, slot)
    return slot, int(slot.max()) + 1 if slot.size else 0


def cost(found: Paths, inputs: int) -> float:
    """R-XAI-02: the work for one row (each path: its splits × its distinct inputs)."""
    return float(((found.length + 1.0) * (np.minimum(found.length, inputs) + 1.0)).sum())


def budget() -> float:
    return BROWSER_BUDGET if browser.active() else BUDGET


def rough_cost(found: Ensemble, inputs: int) -> float:
    """An upper bound of `cost` read before the paths are built (a huge forest stops here)."""
    total = 0.0
    for tree in found.trees:
        leaves = int((tree.left == -1).sum())
        total += leaves * (tree.depth + 1) * (min(tree.depth, inputs) + 1)
    return total


def _size(rows: int, length: int) -> int:
    """Elements of the largest array of one path: rows × splits × quadrature points."""
    return max(1, rows * (int(length) + 1) * (int(length) // 2 + 2))


def _chunk(x: np.ndarray, part: Paths, start: int, stop: int, phi: np.ndarray) -> float:
    """Add to `phi` the contributions of paths start … stop − 1; returns their part of the base."""
    length = int(part.length[start:stop].max()) if stop > start else 0
    feature = part.feature[start:stop, :length]
    value = part.value[start:stop]
    count = stop - start
    if length == 0:  # a tree that is a single leaf
        return float(value.sum())
    slot, players = _slots(feature)
    valid = feature >= 0
    leaf_index = np.broadcast_to(np.arange(count)[:, None], feature.shape)
    zlog = np.zeros((count, players))
    ratio = np.maximum(part.ratio[start:stop, :length][valid], 1e-12)  # an empty child
    np.add.at(zlog, (leaf_index[valid], slot[valid]), np.log(ratio))
    z = np.exp(zlog)  # padded players: z = 1
    owner = np.full((count, players), -1, dtype=np.int64)
    owner[leaf_index[valid], slot[valid]] = feature[valid]
    # o_k(x): every split of the path on k is satisfied
    values = x[:, np.where(valid, feature, 0)]  # rows × paths × splits
    as32 = part.float32[start:stop][None, :, None]
    values = np.where(as32, values.astype(np.float32).astype(float), values)
    with np.errstate(invalid="ignore"):
        goes_left = np.where(
            np.isnan(values),
            part.missing_left[start:stop, :length][None],
            values <= part.threshold[start:stop, :length][None],
        )
    satisfied = (goes_left == part.left[start:stop, :length][None]) | ~valid[None]
    rows = x.shape[0]
    o = np.ones((rows, count, players))
    paths_at = np.arange(count)
    for d in range(length):  # a player holds when each of its splits does
        o[:, paths_at, np.maximum(slot[:, d], 0)] *= satisfied[:, :, d]
    # Σ_s w(s, d) [t^s] Π_{k≠i} (z_k + o_k t) = ∫_0^1 Π_{k≠i} (z_k + u (o_k − z_k)) du: a
    # polynomial of degree < d in u, exact with ⌈d / 2⌉ Gauss-Legendre points (every factor > 0
    # inside (0, 1), so the product over k ≠ i is the product over all k divided by the i-th)
    points, weights = np.polynomial.legendre.leggauss(max(1, (players + 1) // 2))
    u, g = (points + 1) / 2, weights / 2
    factors = (o[..., None] - z[None, :, :, None]) * u  # rows × paths × players × points
    factors += z[None, :, :, None]
    product = factors.prod(axis=2)
    np.divide(product[:, :, None, :], factors, out=factors)
    shares = factors @ g
    contribution = value[None, :, None] * (o - z[None]) * shares  # rows × paths × players
    flat_owner = owner.ravel()
    kept = flat_owner >= 0
    flat = contribution.reshape(rows, -1)[:, kept]
    order = np.argsort(flat_owner[kept], kind="stable")
    sorted_owner = flat_owner[kept][order]
    if sorted_owner.size:
        starts = np.flatnonzero(np.r_[True, sorted_owner[1:] != sorted_owner[:-1]])
        phi[:, sorted_owner[starts]] += np.add.reduceat(flat[:, order], starts, axis=1)
    return float((value * z.prod(axis=1)).sum())


def shap(found: Ensemble, x: np.ndarray, built: Paths | None = None) -> tuple[np.ndarray, float]:
    """SHAP values (rows × inputs) and the base value: base + Σ φ = the raw prediction."""
    part = built or paths(found)
    order = np.argsort(part.length, kind="stable")  # similar lengths in one array
    part = Paths(
        part.value[order],
        part.feature[order],
        part.threshold[order],
        part.left[order],
        part.missing_left[order],
        part.ratio[order],
        part.float32[order],
        part.length[order],
    )
    x = np.asarray(x, dtype=float)
    phi = np.zeros(x.shape)
    base = found.offset
    total = len(part.value)
    start = 0
    while start < total:
        stop = min(total, start + max(1, CHUNK // _size(x.shape[0], part.length[start])))
        # the chunk is sized on its first (shortest) path: hold it to the longest one
        while (
            stop - start > 1 and (stop - start) * _size(x.shape[0], part.length[stop - 1]) > CHUNK
        ):
            stop = start + max(1, (stop - start) // 2)
        base += _chunk(x, part, start, stop, phi)
        start = stop
    return phi, base
