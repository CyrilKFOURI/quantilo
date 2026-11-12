"""125 — TreeSHAP in numpy for scikit-learn's trees: equal to the Shapley values of the
path-dependent game computed by brute force (every subset of the inputs), and the values of each
row add up to its raw prediction."""

import itertools
import math

import numpy as np
import pytest
from sklearn.ensemble import (
    BaggingRegressor,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from quantilo.ml import treeshap

RNG = np.random.default_rng(7)
X = RNG.normal(size=(400, 5))
X[::11, 2] = np.nan  # missing values follow the tree's own direction
Y = 2 * X[:, 0] + np.where(np.isnan(X[:, 2]), 1.0, X[:, 2]) * X[:, 1] + RNG.normal(0, 0.3, 400)
LABELS = (Y > 0).astype(int)
FILLED = np.nan_to_num(X)


def _expected(tree: treeshap.Tree, x: np.ndarray, known: set[int], node: int = 0) -> float:
    """E[f(x) | x_S] as TreeSHAP defines it (Lundberg et al. 2020, algorithm 1)."""
    if tree.left[node] == -1:
        return float(tree.value[node])
    feature = int(tree.feature[node])
    if tree.columns is not None:
        feature = int(tree.columns[feature])
    left, right = int(tree.left[node]), int(tree.right[node])
    if feature in known:
        value = float(np.float32(x[feature])) if tree.float32 else float(x[feature])
        goes_left = (
            bool(tree.missing_left[node]) if math.isnan(value) else value <= tree.threshold[node]
        )
        return _expected(tree, x, known, left if goes_left else right)
    total = tree.cover[node]
    return (
        tree.cover[left] * _expected(tree, x, known, left)
        + tree.cover[right] * _expected(tree, x, known, right)
    ) / total


def _brute_force(found: treeshap.Ensemble, x: np.ndarray) -> np.ndarray:
    """Exact Shapley values of the path-dependent game, every subset enumerated."""
    p = len(x)
    phi = np.zeros(p)
    for tree, weight in zip(found.trees, found.weights, strict=True):
        cache: dict[frozenset[int], float] = {}

        def value(subset: frozenset[int], tree=tree, cache=cache) -> float:
            if subset not in cache:
                cache[subset] = _expected(tree, x, set(subset))
            return cache[subset]

        for i in range(p):
            others = [k for k in range(p) if k != i]
            for size in range(p):
                w = math.factorial(size) * math.factorial(p - size - 1) / math.factorial(p)
                for subset in itertools.combinations(others, size):
                    s = frozenset(subset)
                    phi[i] += weight * w * (value(s | {i}) - value(s))
    return phi


@pytest.mark.parametrize(
    "model",
    [
        DecisionTreeRegressor(max_depth=6, random_state=0),  # splits on one input several times
        RandomForestRegressor(n_estimators=4, max_depth=5, random_state=0),
        HistGradientBoostingRegressor(max_iter=5, max_leaf_nodes=8, random_state=0),
    ],
    ids=["tree", "forest", "hist"],
)
def test_equal_to_the_brute_force_shapley_values(model) -> None:
    model.fit(X, Y)
    found = treeshap.ensemble(model, "regression")
    assert found is not None
    phi, _ = treeshap.shap(found, X[:6])
    for i in range(6):
        assert np.allclose(phi[i], _brute_force(found, X[i]), atol=1e-9)


def _raw(model, x: np.ndarray, kind: str) -> np.ndarray:
    if kind == "probability":
        return model.predict_proba(x)[:, 1]
    if kind == "log-odds":
        return model.decision_function(x)
    return model.predict(x)


@pytest.mark.parametrize(
    ("model", "task", "scale", "data"),
    [
        (DecisionTreeRegressor(random_state=0), "regression", "prediction", X),
        (DecisionTreeClassifier(max_depth=8, random_state=0), "binary", "probability", X),
        (RandomForestRegressor(n_estimators=30, random_state=0), "regression", "prediction", X),
        (RandomForestClassifier(n_estimators=30, random_state=0), "binary", "probability", X),
        (ExtraTreesRegressor(n_estimators=20, random_state=0), "regression", "prediction", X),
        (
            BaggingRegressor(
                DecisionTreeRegressor(), n_estimators=10, max_features=3, random_state=0
            ),
            "regression",
            "prediction",
            FILLED,
        ),
        (
            GradientBoostingRegressor(n_estimators=40, random_state=0),
            "regression",
            "prediction",
            FILLED,
        ),
        (GradientBoostingClassifier(n_estimators=40, random_state=0), "binary", "log-odds", FILLED),
        (HistGradientBoostingRegressor(max_iter=60, random_state=0), "regression", "prediction", X),
        (HistGradientBoostingClassifier(max_iter=60, random_state=0), "binary", "log-odds", X),
        (
            HistGradientBoostingRegressor(loss="poisson", max_iter=30, random_state=0),
            "regression",
            "log",
            X,
        ),
    ],
    ids=[
        "tree",
        "tree-binary",
        "forest",
        "forest-binary",
        "extra",
        "bagging",
        "gb",
        "gb-binary",
        "hist",
        "hist-binary",
        "hist-poisson",
    ],
)
def test_values_add_up_to_the_raw_prediction(model, task, scale, data) -> None:
    target = np.exp(Y / 4) if scale == "log" else (LABELS if task == "binary" else Y)
    model.fit(data, target)
    found = treeshap.ensemble(model, task)
    assert found is not None and found.scale == scale
    phi, base = treeshap.shap(found, data[:120])
    raw = _raw(model, data[:120], scale) if scale != "log" else np.log(model.predict(data[:120]))
    assert np.allclose(base + phi.sum(axis=1), raw, atol=1e-6)


def test_the_base_is_the_cover_weighted_average_and_unused_inputs_get_nothing() -> None:
    model = DecisionTreeRegressor(max_depth=3, random_state=0).fit(X[:, :2], Y)
    found = treeshap.ensemble(model, "regression")
    assert found is not None
    phi, base = treeshap.shap(found, np.c_[X[:50, :2], np.zeros((50, 1))])
    assert np.isclose(base, Y.mean())  # every training row in some leaf
    assert np.allclose(phi[:, 2], 0.0)


def test_several_classes_and_other_models_are_left_out() -> None:
    three = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, np.digitize(Y, [-1, 1]))
    assert treeshap.ensemble(three, "multiclass") is None
    from sklearn.linear_model import LinearRegression

    assert treeshap.ensemble(LinearRegression().fit(FILLED, Y), "regression") is None


def test_the_cost_rule_reads_the_paths() -> None:
    model = RandomForestRegressor(n_estimators=10, random_state=0).fit(X, Y)
    found = treeshap.ensemble(model, "regression")
    assert found is not None
    built = treeshap.paths(found)
    assert len(built.value) == sum(int((t.left == -1).sum()) for t in found.trees)
    assert 0 < treeshap.cost(built, X.shape[1]) <= treeshap.rough_cost(found, X.shape[1])
