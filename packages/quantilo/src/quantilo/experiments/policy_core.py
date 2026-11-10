"""What "who should get which variant" (B) and "evaluate a policy from logged decisions" (C)
share (docs/features/055-experiments-politiques/spec.md, R-PC1 to R-PC6): readable covariates
chosen by the ML preparation rules (017), cross-fitted outcome and propensity models, doubly
robust (AIPW) scores, a seeded Poisson bootstrap, a greedy policy tree and the Qini curve.

Everything is seeded: the same data gives the same numbers on every call (cache, library = app).
No thread, no subprocess: it runs in the browser version (Pyodide) too.
"""

import math
import sys
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from quantilo.ml import preparation as prep
from quantilo.ml.data import CATEGORICAL, NUMERIC
from quantilo.ml.data import Field as DataField

IN_BROWSER = sys.platform == "emscripten"  # the lite version: one core, little memory
SEED = 55
FOLDS = 5
DRAWS = 500 if IN_BROWSER else 1_000
MAX_UNITS = 20_000 if IN_BROWSER else 50_000
FLOOR = 0.01  # propensities below are raised to it: weights stay ≤ 100 (R-PC3)
LEVELS = 10  # indicators kept per category column (R-PC5)
MIN_TRAIN = 30  # units of an action below which its outcome model is its mean (R-PC2)
QUANTILES = 16  # thresholds tried per number (R-PL4)
MAX_DEPTH = 3
QINI_POINTS = 21
TREE_PROFILE = prep.ModelProfile(tree_based=True)
NO_EARLY_STOP: Any = False  # the stubs only know "auto"; False keeps every iteration (seeded)
# A pandas DataFrame or Series: its stubs type each indexing as a wide union (as in analyze.py).
Frame = Any

FeatureKind = Literal["number", "level"]


# --- Covariates (R-PC5) -------------------------------------------------------------------------


@dataclass
class Feature:
    name: str  # "age", "device = mobile"
    column: str
    kind: FeatureKind
    level: str | None = None


@dataclass
class PolicyCovariate:
    column: str
    used: bool
    reason: str | None = None  # why it was left out


@dataclass
class Covariates:
    x: np.ndarray  # units × features, no missing value
    features: list[Feature]
    columns: list[PolicyCovariate]

    @property
    def used(self) -> list[str]:
        return [c.column for c in self.columns if c.used]


def _labels(series: Frame) -> Frame:
    return series.astype("string").fillna(prep.MISSING).astype(str)


def covariates(
    frame: Frame, fields: list[DataField], target: DataField, binary: bool
) -> Covariates:
    """The columns the ML preparation rules keep (profile "tree": identifiers, constants, nearly
    empty columns and leaks are left out, each with its sentence), dates and free text left out,
    then a readable matrix: numbers as they are (missing → median, or the mean when the rules
    say so), each category as indicators of its 10 most frequent values."""
    columns: list[PolicyCovariate] = []
    usable: list[DataField] = []
    for fld in fields:
        if fld.type == "datetime":
            columns.append(PolicyCovariate(fld.name, False, "A date says when, not who."))
        elif fld.type == "text":
            columns.append(
                PolicyCovariate(fld.name, False, "Free text cannot be read as a simple rule.")
            )
        elif fld.type not in NUMERIC | CATEGORICAL:
            reason = "An identifier." if fld.type == "identifier" else "Not usable here."
            columns.append(PolicyCovariate(fld.name, False, reason))
        else:
            usable.append(fld)
    preparation = (
        prep.decide(
            frame, [*usable, target], target, "binary" if binary else "regression", TREE_PROFILE
        )
        if usable
        else None
    )
    decisions = {d.column: d for d in preparation.decisions} if preparation else {}
    blocks: list[np.ndarray] = []
    features: list[Feature] = []
    for fld in usable:
        decision = decisions.get(fld.name)
        if decision is None or not decision.use:
            reason = decision.why[-1] if decision and decision.why else "Left out."
            columns.append(PolicyCovariate(fld.name, False, reason))
            continue
        series: Frame = frame[fld.name]
        if fld.type in NUMERIC:
            values: Frame = pd.Series(pd.to_numeric(series, errors="coerce"), dtype=float)
            fill = values.mean() if decision.impute == "mean" else values.median()
            filled = values.fillna(0.0 if math.isnan(fill) else fill).to_numpy(float)
            blocks.append(filled[:, None])
            features.append(Feature(fld.name, fld.name, "number"))
        else:
            labels = _labels(series)
            counts = labels.value_counts()
            kept = [str(level) for level in counts.index[:LEVELS]]
            if len(counts) == 2:
                kept = kept[1:]  # two values: one indicator says it all
            for level in kept:
                blocks.append((labels == level).to_numpy(float)[:, None])
                features.append(Feature(f"{fld.name} = {level}", fld.name, "level", level))
        columns.append(PolicyCovariate(fld.name, True))
    x = np.hstack(blocks) if blocks else np.zeros((len(frame), 0))
    order = {f.name: i for i, f in enumerate(fields)}
    columns.sort(key=lambda c: order.get(c.column, len(order)))
    return Covariates(x, features, columns)


BALANCE_ALARM = 0.001
BALANCE_GAP = 0.05  # standardized difference beyond which a covariate is imbalanced


def imbalanced(frame: Frame, fields: list[DataField], variant: str) -> dict[str, str]:
    """Covariates that differ between the variants of a randomized test (R-PL9): a test says
    the difference is not chance (Kruskal–Wallis for a number, χ² for a category, p < 0.001)
    and it is large enough to matter (standardized difference above 0.05).
    Measured before the test, a column is balanced by the random assignment; one that differs
    was probably changed by the test."""
    found: dict[str, str] = {}
    groups = frame[variant]
    labels = [g for g in groups.unique() if g == g]
    for fld in fields:
        series: Frame = frame[fld.name]
        try:
            if fld.type in NUMERIC:
                values: Frame = pd.Series(pd.to_numeric(series, errors="coerce"), dtype=float)
                parts = [values[groups == g].dropna().to_numpy() for g in labels]
                parts = [part for part in parts if len(part)]
                spread = float(values.std())
                if len(parts) < 2 or not spread > 0:
                    continue
                means = [float(part.mean()) for part in parts]
                gap = (max(means) - min(means)) / spread
                p = float(stats.kruskal(*parts).pvalue)
            elif fld.type in CATEGORICAL:
                table = pd.crosstab(_labels(series), groups)
                if table.shape[0] < 2 or table.shape[1] < 2:
                    continue
                shares = table.to_numpy(float) / table.to_numpy(float).sum(axis=0, keepdims=True)
                overall = table.to_numpy(float).sum(axis=1) / table.to_numpy(float).sum()
                scale = np.sqrt(np.maximum(overall * (1 - overall), 1e-12))
                gap = float(((shares.max(axis=1) - shares.min(axis=1)) / scale).max())
                chi2: Any = stats.chi2_contingency(table.to_numpy())
                p = float(chi2[1])
            else:
                continue
        except ValueError:
            continue
        if p < BALANCE_ALARM and gap > BALANCE_GAP:
            found[fld.name] = (
                f"Differs between the variants (p < 0.001, standardized difference {gap:.2f}): "
                "the random assignment balances a column measured before the test, so this one "
                "was probably changed by it."
            )
    return found


# --- Cross-fitted nuisance models (R-PC2) --------------------------------------------------------


def folds(n: int, seed: int = SEED) -> np.ndarray:
    """A fold (0 to FOLDS-1) per unit, balanced and seeded."""
    rng = np.random.default_rng(seed)
    return rng.permutation(np.arange(n) % FOLDS)


def _regressor() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=60, max_leaf_nodes=15, min_samples_leaf=40, early_stopping=NO_EARLY_STOP
    )


def outcomes(
    x: np.ndarray, actions: np.ndarray, y: np.ndarray, k: int, fold: np.ndarray, binary: bool
) -> np.ndarray:
    """μ̂ₐ(xᵢ) for every unit and action, each unit predicted by models fitted without its fold;
    one model per action (an action with fewer than MIN_TRAIN training units: its mean)."""
    mu = np.zeros((len(y), k))
    overall = float(y.mean()) if len(y) else 0.0
    for f in range(FOLDS):
        test = fold == f
        if not test.any():
            continue
        train = ~test
        for a in range(k):
            rows = train & (actions == a)
            count = int(rows.sum())
            if count < MIN_TRAIN or x.shape[1] == 0:
                mu[test, a] = float(y[rows].mean()) if count else overall
                continue
            model = _regressor().fit(x[rows], y[rows])
            mu[test, a] = model.predict(x[test])
    return np.clip(mu, 0.0, 1.0) if binary else mu


PROPENSITY_MODELS = {
    "logistic regression": lambda: make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1_000)
    ),
    "gradient boosting": lambda: HistGradientBoostingClassifier(
        max_iter=40,
        max_leaf_nodes=7,
        min_samples_leaf=200,
        early_stopping=NO_EARLY_STOP,
    ),
}


@dataclass
class Propensities:
    values: np.ndarray  # units × actions
    model: str  # the classifier kept
    log_loss: float  # its cross-validated log loss


def propensities(x: np.ndarray, actions: np.ndarray, k: int, fold: np.ndarray) -> Propensities:
    """êₐ(xᵢ) = P(action a | xᵢ), cross-fitted like the outcomes (R-PO2): a logistic regression
    and a regularized gradient boosting, the one with the lower cross-validated log loss kept
    (a noisy propensity inflates the weights, so calibration matters more than flexibility)."""
    shares = np.bincount(actions, minlength=k) / max(len(actions), 1)
    rows = np.arange(len(actions))
    found: list[Propensities] = []
    for name, make in PROPENSITY_MODELS.items():
        e = np.tile(shares, (len(actions), 1)).astype(float)
        if x.shape[1]:
            for f in range(FOLDS):
                test = fold == f
                train = ~test
                if not test.any() or len(np.unique(actions[train])) < 2:
                    continue
                model: Any = make().fit(x[train], actions[train])
                chances = model.predict_proba(x[test])
                block = np.zeros((int(test.sum()), k))
                for j, a in enumerate(model.classes_):
                    block[:, int(a)] = chances[:, j]
                e[test] = block
        loss = float(-np.log(np.clip(e[rows, actions], 1e-6, 1.0)).mean())
        found.append(Propensities(e, name, loss))
        if not x.shape[1]:
            break
    return min(found, key=lambda chosen: chosen.log_loss)


def aipw(mu: np.ndarray, p: np.ndarray, actions: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Γᵢ(a) = μ̂ₐ(xᵢ) + 1{Aᵢ = a}(Yᵢ − μ̂ₐ(xᵢ)) / pᵢ, with pᵢ the propensity of the action taken
    (floored at FLOOR, R-PC3)."""
    rows = np.arange(len(y))
    gamma = mu.copy()
    gamma[rows, actions] += (y - mu[rows, actions]) / np.maximum(p, FLOOR)
    return gamma


# --- Seeded Poisson bootstrap (R-PC4) ------------------------------------------------------------


def bootstrap(columns: np.ndarray, draws: int = DRAWS, seed: int = SEED) -> np.ndarray:
    """Weighted means of each column for `draws` resamples (each unit weighted by a Poisson(1)
    draw): draws × columns. One matrix product per block of draws."""
    values = np.asarray(columns, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    n, m = values.shape
    out = np.empty((draws, m))
    if n == 0:
        out.fill(np.nan)
        return out
    rng = np.random.default_rng(seed)
    block = max(1, min(draws, 4_000_000 // max(n, 1)))
    for start in range(0, draws, block):
        size = min(block, draws - start)
        weights = rng.poisson(1.0, (size, n)).astype(float)
        totals = weights.sum(axis=1, keepdims=True)
        totals[totals == 0] = 1.0
        out[start : start + size] = (weights @ values) / totals
    return out


def interval(samples: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Percentile interval of each column of bootstrap samples."""
    low, high = np.nanpercentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=0)
    return np.asarray(low), np.asarray(high)


# --- Policy tree (R-PL4) -------------------------------------------------------------------------


@dataclass
class Node:
    action: int
    units: int
    feature: int | None = None
    threshold: float | None = None  # left: x < threshold; right: x ≥ threshold
    left: "Node | None" = None
    right: "Node | None" = None

    @property
    def leaf(self) -> bool:
        return self.feature is None


def _thresholds(x: np.ndarray, features: list[Feature]) -> list[np.ndarray]:
    found = []
    for j, feature in enumerate(features):
        column = x[:, j]
        if feature.kind == "level":
            found.append(np.array([0.5]) if 0 < column.mean() < 1 else np.array([]))
            continue
        cut = np.unique(np.quantile(column, np.linspace(0, 1, QUANTILES + 2)[1:-1]))
        found.append(cut[cut > column.min()])
    return found


def _grow(
    x: np.ndarray,
    gamma: np.ndarray,
    rows: np.ndarray,
    depth: int,
    min_leaf: int,
    cuts: list[np.ndarray],
) -> Node:
    totals = gamma[rows].sum(axis=0)
    node = Node(int(np.argmax(totals)), len(rows))
    if depth == 0 or len(rows) < 2 * min_leaf:
        return node
    best = float(totals.max())
    tolerance = 1e-9 * (abs(best) + 1.0)
    choice: tuple[int, float] | None = None
    part = gamma[rows]
    for j, thresholds in enumerate(cuts):
        if not len(thresholds):
            continue
        values = x[rows, j]
        order = np.argsort(values, kind="stable")
        ordered = values[order]
        running = np.cumsum(part[order], axis=0)
        position = np.searchsorted(ordered, thresholds, side="left")
        valid = (position >= min_leaf) & (len(rows) - position >= min_leaf)
        if not valid.any():
            continue
        left = running[np.maximum(position - 1, 0)]
        left[position == 0] = 0.0
        value = left.max(axis=1) + (totals - left).max(axis=1)
        value[~valid] = -np.inf
        i = int(np.argmax(value))
        if value[i] > best + tolerance:
            best, choice = float(value[i]), (j, float(thresholds[i]))
    if choice is None:
        return node
    j, threshold = choice
    goes_left = x[rows, j] < threshold
    node.feature, node.threshold = j, threshold
    node.left = _grow(x, gamma, rows[goes_left], depth - 1, min_leaf, cuts)
    node.right = _grow(x, gamma, rows[~goes_left], depth - 1, min_leaf, cuts)
    return _merged(node)


def _merged(node: Node) -> Node:
    """Two sibling leaves giving the same action become one leaf."""
    left, right = node.left, node.right
    if left and right and left.leaf and right.leaf and left.action == right.action:
        return Node(left.action, node.units)
    return node


def min_leaf(n: int) -> int:
    return max(50, math.ceil(0.05 * n))


def grow(x: np.ndarray, gamma: np.ndarray, features: list[Feature], depth: int) -> Node:
    """The greedy tree of depth ≤ `depth` maximizing Σ_leaves maxₐ Σᵢ Γᵢ(a)."""
    rows = np.arange(len(gamma))
    return _grow(x, gamma, rows, depth, min_leaf(len(rows)), _thresholds(x, features))


def predict(node: Node, x: np.ndarray) -> np.ndarray:
    out = np.empty(len(x), dtype=int)

    def walk(current: Node, rows: np.ndarray) -> None:
        if current.leaf or current.left is None or current.right is None:
            out[rows] = current.action
            return
        assert current.feature is not None and current.threshold is not None
        goes_left = x[rows, current.feature] < current.threshold
        walk(current.left, rows[goes_left])
        walk(current.right, rows[~goes_left])

    walk(node, np.arange(len(x)))
    return out


def out_of_fold(
    x: np.ndarray, gamma: np.ndarray, features: list[Feature], depth: int, fold: np.ndarray
) -> np.ndarray:
    """Each unit's action by the tree grown without its fold (R-PL5)."""
    actions = np.zeros(len(gamma), dtype=int)
    for f in range(FOLDS):
        test = fold == f
        if not test.any():
            continue
        tree = grow(x[~test], gamma[~test], features, depth)
        actions[test] = predict(tree, x[test])
    return actions


@dataclass
class DepthChoice:
    depth: int
    values: list[float]  # out-of-fold value of the rule at depth 0, 1, 2, 3
    actions: np.ndarray = field(repr=False)  # out-of-fold actions at the chosen depth


def choose_depth(
    x: np.ndarray, gamma: np.ndarray, features: list[Feature], fold: np.ndarray
) -> DepthChoice:
    """Depth 0 (one action for all) to 3 by cross-validation, one-standard-error rule: the
    shallowest depth whose value is within one standard error (of the paired difference) of
    the best."""
    rows = np.arange(len(gamma))
    found = [out_of_fold(x, gamma, features, d, fold) for d in range(0, MAX_DEPTH + 1)]
    per_unit = [gamma[rows, actions] for actions in found]
    values = [float(v.mean()) for v in per_unit]
    best = int(np.argmax(values))
    chosen = best
    for d in range(best):
        gap = per_unit[best] - per_unit[d]
        se = float(gap.std(ddof=1) / math.sqrt(len(gap))) if len(gap) > 1 else 0.0
        if float(gap.mean()) <= se:
            chosen = d
            break
    return DepthChoice(chosen, values, found[chosen])


# --- Readable rules ------------------------------------------------------------------------------


def nice(value: float) -> str:
    """A threshold as people write it: 30 · 2.5 · 0.12 · 1,250."""
    if value == int(value) and abs(value) < 1e15:
        return f"{int(value):,}"
    magnitude = abs(value)
    if magnitude >= 100:
        return f"{value:,.0f}"
    if magnitude >= 1:
        return f"{value:.3g}"
    return f"{value:.2g}"


@dataclass
class Condition:
    feature: int
    goes_left: bool
    threshold: float


def _path_text(conditions: list[Condition], features: list[Feature]) -> str:
    """One leaf's path, simplified: the tightest bounds per number ("30 ≤ age < 50"), one
    "= level" per category (or its "≠ levels")."""
    bounds: dict[str, list[float | None]] = {}
    equal: dict[str, str] = {}
    different: dict[str, list[str]] = {}
    order: list[str] = []
    for c in conditions:
        feature = features[c.feature]
        if feature.column not in order:
            order.append(feature.column)
        if feature.kind == "number":
            low, high = bounds.setdefault(feature.column, [None, None])
            if c.goes_left:
                bounds[feature.column][1] = c.threshold if high is None else min(high, c.threshold)
            else:
                bounds[feature.column][0] = c.threshold if low is None else max(low, c.threshold)
        elif c.goes_left:
            different.setdefault(feature.column, []).append(feature.level or "")
        else:
            equal[feature.column] = feature.level or ""
    parts = []
    for column in order:
        if column in bounds:
            low, high = bounds[column]
            if low is not None and high is not None:
                parts.append(f"{nice(low)} ≤ {column} < {nice(high)}")
            elif high is not None:
                parts.append(f"{column} < {nice(high)}")
            elif low is not None:
                parts.append(f"{column} ≥ {nice(low)}")
        elif column in equal:
            parts.append(f"{column} = {equal[column]}")
        else:
            parts.append(f"{column} ≠ {', '.join(different[column])}")
    return " and ".join(parts)


@dataclass
class Leaf:
    conditions: str  # "device = mobile and age < 30"; "" for everyone
    action: int
    units: int


def leaves(node: Node, features: list[Feature]) -> list[Leaf]:
    found: list[Leaf] = []

    def walk(current: Node, path: list[Condition]) -> None:
        if current.leaf or current.left is None or current.right is None:
            found.append(Leaf(_path_text(path, features), current.action, current.units))
            return
        assert current.feature is not None and current.threshold is not None
        walk(current.left, [*path, Condition(current.feature, True, current.threshold)])
        walk(current.right, [*path, Condition(current.feature, False, current.threshold)])

    walk(node, [])
    return found


def rule_sentence(found: list[Leaf], labels: list[str], verb: str = "Give", to: str = "to") -> str:
    """ "Give new_checkout to device = mobile and age < 30; control elsewhere." """
    actions = {leaf.action for leaf in found}
    if len(actions) == 1:
        return (
            f"{verb} {labels[found[0].action]} {to} everyone: no group of units does better with "
            "another choice."
        )
    sizes: dict[int, int] = {}
    for leaf in found:
        sizes[leaf.action] = sizes.get(leaf.action, 0) + leaf.units
    default = max(sizes, key=lambda a: sizes[a])
    parts = []
    for action in sorted(actions - {default}, key=lambda a: -sizes[a]):
        where = " or ".join(
            f"({leaf.conditions})"
            if " and " in leaf.conditions and len(found) > 2
            else leaf.conditions
            for leaf in found
            if leaf.action == action
        )
        parts.append(f"{verb} {labels[action]} {to} {where}")
    return f"{'; '.join(parts)}; {labels[default]} elsewhere."


# --- Qini curve (R-PL7) --------------------------------------------------------------------------


@dataclass
class QiniCurve:
    variant: str
    control: str
    shares: list[float]  # share of the units treated, 0 to 1
    gains: list[float]  # cumulative gain per unit of the population
    random: list[float]  # random targeting
    auuc: float  # area between the curve and random targeting
    sentence: str


def qini(
    gamma: np.ndarray, uplift: np.ndarray, treated: int, control: int
) -> tuple[list[float], list[float], list[float], float]:
    """Units ranked by predicted uplift; at each share q of them treated, the AIPW gain of
    treating them, Σ_top q (Γᵢ(treated) − Γᵢ(control)) / n, against q × the average effect."""
    n = len(gamma)
    diff = gamma[:, treated] - gamma[:, control]
    order = np.argsort(-uplift, kind="stable")
    running = np.concatenate([[0.0], np.cumsum(diff[order])]) / max(n, 1)
    shares = np.linspace(0.0, 1.0, QINI_POINTS)
    gains = running[np.round(shares * n).astype(int)]
    random = shares * running[-1]
    gap = gains - random
    auuc = float(np.sum((gap[1:] + gap[:-1]) / 2 * np.diff(shares)))
    return shares.tolist(), gains.tolist(), random.tolist(), auuc


def finite(value: Any) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None
