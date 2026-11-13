# ruff: noqa: RUF003  (the usual notation: ≤, −)
"""122 — the sparse CART tree (R-XAI-09), and the seams of the optional `interpret` extra
(R-XAI-10, R-XAI-11): GOSDT, TreeFARMS, PaCMAP and TriMAP are never installed here, so fake
modules stand in for them to test the calls, and their absence is said with its reason."""

import sys
import types

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from quantilo.interpret import extras, maps, rashomon
from quantilo.interpret import sparse_tree as st
from quantilo.ml import browser


def _step(n: int = 3_000, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """y depends on one condition (x > 0.5), plus noise and an unrelated column."""
    rng = np.random.default_rng(seed)
    x = rng.random(n)
    noise = rng.normal(size=n)
    colour = rng.choice(["red", "blue", "green"], n)
    y = np.where(rng.random(n) < np.where(x > 0.5, 0.8, 0.15), "yes", "no")
    return pd.DataFrame({"x": x, "noise": noise, "colour": colour}), y


def test_the_sparse_tree_finds_the_one_condition_and_prunes_the_rest() -> None:
    frame, y = _step()
    found = st.sparse_tree(
        frame.iloc[:2_000], y[:2_000], frame.iloc[2_000:], y[2_000:], "binary", positive="yes"
    )
    root = found.nodes[0]
    assert root.left_rule is not None and root.left_rule.startswith("x ≤")
    threshold = float(root.left_rule.split("≤")[1])
    assert abs(threshold - 0.5) < 0.05
    assert found.leaves <= 4  # the 1-SE rule keeps a small tree
    assert found.measure == "ROC AUC" and found.score > 0.75
    assert "R-XAI-09" in found.sentence and "not optimal" in found.sentence


def test_the_sparse_tree_holds_its_limits_and_more_pruning_means_fewer_leaves() -> None:
    rng = np.random.default_rng(1)
    frame = pd.DataFrame(rng.normal(size=(2_000, 6)), columns=list("abcdef"))
    y = np.where(frame.sum(axis=1) + rng.normal(size=2_000) > 0, "yes", "no")
    for depth, leaves in [(1, 2), (3, 6), (6, 20)]:
        found = st.sparse_tree(
            frame, y, frame, y, "binary", depth=depth, leaves=leaves, positive="yes"
        )
        assert max(n.depth for n in found.nodes) <= depth
        assert found.leaves <= leaves
    x = frame.to_numpy()
    counts = [
        DecisionTreeClassifier(max_depth=6, max_leaf_nodes=20, ccp_alpha=alpha, random_state=0)
        .fit(x, y)
        .get_n_leaves()
        for alpha in (0.0, 0.002, 0.01, 0.05)
    ]
    assert counts == sorted(counts, reverse=True) and counts[-1] < counts[0]
    _, tried = st.prune(x, y, "binary", 6, 20, 20)
    by_alpha = sorted(tried, key=lambda p: p.alpha)
    assert [p.leaves for p in by_alpha] == sorted((p.leaves for p in by_alpha), reverse=True)


def test_the_sparse_tree_also_reads_a_number() -> None:
    frame, _ = _step()
    target = np.where(frame["x"] > 0.5, 10.0, 0.0) + np.random.default_rng(2).normal(
        size=len(frame)
    )
    found = st.sparse_tree(frame, target, frame, target, "regression")
    assert found.measure == "R²" and found.score > 0.8 and found.gap is None


def test_the_comparison_with_the_model_says_the_gap_and_its_noise() -> None:
    frame, y = _step()
    truth = (y[2_000:] == "yes").astype(float)
    model = truth * 0.2 + np.random.default_rng(3).random(len(truth)) * 0.8
    found = st.sparse_tree(
        frame.iloc[:2_000],
        y[:2_000],
        frame.iloc[2_000:],
        y[2_000:],
        "binary",
        positive="yes",
        model_score=model,
        model_label="random forest",
    )
    assert found.gap is not None and found.gap.rows == 1_000
    assert (
        found.gap.low <= found.gap.gap <= found.gap.high and "random forest" in found.gap.sentence
    )


# --- The optional extra (R-XAI-10) -------------------------------------------------------------


def test_an_absent_extra_is_greyed_with_its_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(extras, "_installed", lambda package: False)
    found = {a.key: a for a in extras.available()}
    assert set(found) == {"gosdt", "treefarms", "pacmap", "trimap"}
    assert not any(a.available for a in found.values())
    assert "not installed" in (found["pacmap"].reason or "") and found["pacmap"].install
    if sys.version_info[:2] > (3, 12):
        assert "no wheel for Python" in (found["gosdt"].reason or "")
    with pytest.raises(extras.ExtraMissing, match="R-XAI-10"):
        maps.embed(np.zeros((10, 3)), "pacmap")
    monkeypatch.setattr(browser, "IN_BROWSER", True)
    assert "desktop app" in (extras.check("treefarms").reason or "")


def _module(name: str, **attributes: object) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


class _FakeGosdt:
    """Stands in for gosdt.GOSDTClassifier: a small scikit-learn tree, no JSON kept (the tree
    is then read back from its predictions)."""

    def __init__(self, **config: object) -> None:
        self.config = config

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "_FakeGosdt":
        self.tree = DecisionTreeClassifier(max_depth=2, random_state=0).fit(x, y)
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return self.tree.predict(x)


def test_the_optimal_tree_calls_gosdt_and_reads_its_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "gosdt", _module("gosdt", GOSDTClassifier=_FakeGosdt))
    frame, y = _step()
    model = (y[2_000:] == "yes") * 0.5 + 0.25
    found = rashomon.optimal_tree(
        frame.iloc[:2_000], y[:2_000], frame.iloc[2_000:], y[2_000:], "yes", model_score=model
    )
    assert found.read_back and found.leaves >= 2 and found.score > 0.7
    assert found.nodes[0].right_rule and found.nodes[0].right_rule.startswith("x")
    assert found.gap is not None and "optimal" in found.gap.sentence


class _FakeTree:
    def __init__(self, feature: int) -> None:
        self.feature = feature
        self.source = {"feature": feature, "true": {"prediction": 1}, "false": {"prediction": 0}}

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return x.iloc[:, self.feature].to_numpy()


class _FakeFarms:
    def __init__(self, config: dict[str, object]) -> None:
        self.config = config

    def fit(self, x: pd.DataFrame, y: pd.Series) -> None:
        columns = list(x.columns)
        self.trees = [_FakeTree(i) for i, c in enumerate(columns) if c.startswith("x ")][:3]

    def get_tree_count(self) -> int:
        return len(self.trees)

    def __getitem__(self, i: int) -> _FakeTree:
        return self.trees[i]


def test_the_rashomon_set_counts_its_trees_and_the_columns_they_share(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "treefarms", _module("treefarms", TREEFARMS=_FakeFarms))
    frame, y = _step()
    found = rashomon.rashomon_set(
        frame.iloc[:2_000], y[:2_000], frame.iloc[2_000:], y[2_000:], "yes"
    )
    assert found.trees == 3 and found.read == 3 and found.always == ["x"]
    assert found.columns[0].column == "x" and found.columns[0].share == 1.0
    assert 0 <= found.accuracy_low <= found.accuracy_high <= 1 and "TreeFARMS" in found.sentence


def test_the_maps_call_pacmap_and_trimap(monkeypatch: pytest.MonkeyPatch) -> None:
    class Reducer:
        def __init__(self, **config: object) -> None:
            self.config = config

        def fit_transform(self, x: np.ndarray, init: str | None = None) -> np.ndarray:
            return x[:, :2] * 2

    monkeypatch.setitem(sys.modules, "pacmap", _module("pacmap", PaCMAP=Reducer))
    monkeypatch.setitem(sys.modules, "trimap", _module("trimap", TRIMAP=Reducer))
    x = np.arange(30, dtype=float).reshape(10, 3)
    assert maps.embed(x, "pacmap").shape == (10, 2)
    assert np.allclose(maps.embed(x, "trimap"), x[:, :2] * 2)


def test_the_segments_map_draws_with_pacmap_under_more(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    from quantilo.segments.engine import SegmentsError
    from tests.test_segments import as_dataset, blobs

    frame, _ = blobs(300)
    found = as_dataset(frame, tmp_path).segments().fit()
    monkeypatch.setattr(extras, "_installed", lambda package: False)
    with pytest.raises(SegmentsError, match="R-XAI-10"):
        found.extra_map("pacmap")

    class Reducer:
        def __init__(self, **config: object) -> None:
            pass

        def fit_transform(self, x: np.ndarray, init: str | None = None) -> np.ndarray:
            return x[:, :2]

    monkeypatch.setattr(extras, "_installed", lambda package: True)
    monkeypatch.setitem(sys.modules, "pacmap", _module("pacmap", PaCMAP=Reducer))
    drawn = found.extra_map("pacmap")
    assert drawn.projection == "pacmap" and len(drawn.x) == 300
    assert {c.segment for c in drawn.centres} == {0, 1, 2} and "pacmap" in found.result.maps
