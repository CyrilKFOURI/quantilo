# ruff: noqa: RUF001, RUF002  (the usual notation)
# pyright: reportArgumentType=false
# (small fake models stand in for a fitted Pipeline)
"""126 — interpretability under Advanced › More: SHAP by permutations (exact additivity, the
closed form of a linear model), Friedman's H (0 without interaction), Anchors (precision at the
threshold on fresh samples), prototypes and criticisms (MMD-critic)."""

from typing import cast

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from quantilo.interpret import anchors as anchoring
from quantilo.interpret import interactions as pairs
from quantilo.interpret import kernel_shap as sampling
from quantilo.interpret import tabular_prototypes as examples
from quantilo.ml.data import Field
from quantilo.ml.train import train
from tests.test_ml import FIELDS, FRAME, RAW, by_name
from tests.test_ml_explain import FOLDS

RNG = np.random.default_rng(11)
N = 600
STEP = {"low": 0.0, "mid": 5.0, "high": 7.0}


def _frame(n: int = N) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a": RNG.normal(0, 1, n),
            "b": RNG.uniform(-1, 2, n),
            "c": RNG.choice(list(STEP), n),
            "d": RNG.normal(5, 2, n),
        }
    )


DATA = _frame()
ROWS = np.arange(1000, 1000 + N)


class Additive:
    """a + b² + step(c): no interaction."""

    classes_: tuple[str, ...] = ()

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        a, b = (frame[c].to_numpy(dtype=float) for c in ("a", "b"))
        return a + b**2 + frame["c"].map(STEP).to_numpy(dtype=float)


class Product:
    """a·b + c: a and b interact, c only adds."""

    classes_: tuple[str, ...] = ()

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        a, b = (frame[c].to_numpy(dtype=float) for c in ("a", "b"))
        return a * b + frame["c"].map(STEP).to_numpy(dtype=float)


class Rule:
    """“yes” exactly when a > 0.5 and c = high."""

    classes_ = np.array(["no", "yes"])

    def _p(self, frame: pd.DataFrame) -> np.ndarray:
        a = frame["a"].to_numpy(dtype=float)
        return ((a > 0.5) & (frame["c"].astype(str).to_numpy() == "high")).astype(float)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.where(self._p(frame) == 1, "yes", "no")

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        p = self._p(frame)
        return np.stack([1 - p, p], axis=1)


# --- SHAP by permutations -----------------------------------------------------------------------


def test_permutation_shap_is_exactly_additive_for_any_model() -> None:
    found = sampling.permutation_shap(Product(), DATA, ROWS, "regression", [1003], pairs=3)  # type: ignore[arg-type]
    shap = found.shap
    assert shap.rows[0] == 1003  # the shown row first
    assert shap.method.startswith("Permutation sampling")
    matrix = np.array(shap.matrix)
    picked = [int(np.flatnonzero(r == ROWS)[0]) for r in shap.rows]
    exact = Product().predict(DATA.iloc[picked])
    assert np.allclose(shap.base + matrix.sum(axis=1), exact, atol=1e-9)
    assert found.additivity < 1e-9
    assert np.array(found.half_width).shape == matrix.shape
    assert found.widest > 0  # a·b: the order matters, so the estimate has a spread


def test_permutation_shap_of_a_linear_model_is_its_closed_form() -> None:
    x = DATA[["a", "b", "d"]]
    y = 2 * x["a"] - 3 * x["b"] + 0.5 * x["d"]
    model = LinearRegression().fit(x, y)
    found = sampling.permutation_shap(model, x, ROWS, "regression", [], pairs=2, background=40)  # type: ignore[arg-type]
    shap = found.shap
    rng = np.random.default_rng(sampling.SEED)
    reference = x.iloc[np.sort(rng.choice(len(x), 40, replace=False))]
    picked = [int(np.flatnonzero(r == ROWS)[0]) for r in shap.rows]
    expected = (x.iloc[picked] - reference.mean()).to_numpy() * model.coef_
    order = [shap.columns.index(c) for c in ("a", "b", "d")]
    assert np.allclose(np.array(shap.matrix)[:, order], expected, atol=1e-9)
    assert found.widest < 1e-9  # additive: every order gives the same credits


def test_permutation_shap_budget_and_refusals(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sampling, "BUDGET", 2 * 2 * 4 * 50 * 30)  # 30 rows
    found = sampling.permutation_shap(Additive(), DATA, ROWS, "regression", [1001], pairs=2)  # type: ignore[arg-type]
    assert len(found.shap.rows) == 30 and 1001 in found.shap.rows
    assert found.shap.note and "R-XAI-12" in found.shap.note
    monkeypatch.setattr(sampling, "BUDGET", 1_000)
    with pytest.raises(sampling.PermutationShapError, match="R-XAI-12"):
        sampling.permutation_shap(Additive(), DATA, ROWS, "regression", [], pairs=2)  # type: ignore[arg-type]
    with pytest.raises(sampling.PermutationShapError, match="several classes"):
        sampling.permutation_shap(Additive(), DATA, ROWS, "multiclass", [])  # type: ignore[arg-type]


def test_permutation_shap_on_a_trained_pipeline_with_categories() -> None:
    result = train(
        FRAME,
        [f for f in FIELDS if f.name in ("size", "skewed", "city")],
        by_name["price"],
        "regression",
        "knn",
        None,
        1,
        False,
        validation=FOLDS,
    )
    x = cast(pd.DataFrame, FRAME[[str(c) for c in result.pipeline.feature_names_in_]])
    rows = FRAME["__row"].to_numpy(dtype=int)
    found = sampling.permutation_shap(result.pipeline, x.iloc[:300], rows[:300], "regression", [])
    predicted = result.pipeline.predict(
        x.iloc[[int(np.flatnonzero(rows == r)[0]) for r in found.shap.rows]]
    )
    matrix = np.array(found.shap.matrix)
    assert np.allclose(found.shap.base + matrix.sum(axis=1), predicted, atol=1e-9)
    assert set(found.shap.columns) == {"size", "skewed", "city"}


# --- Friedman's H -------------------------------------------------------------------------------


def test_h_is_zero_without_interaction() -> None:
    frame = DATA.assign(e=RNG.normal(0, 1, N))
    found = pairs.interactions(Additive(), frame, "regression", ["a", "b", "c", "d", "e"])  # type: ignore[arg-type]
    assert len(found.pairs) == 10
    assert all(p.h < 1e-6 for p in found.pairs)
    # d and e are not used by the model: their pair has no joint effect to judge (R-XAI-13)
    weak = [p for p in found.pairs if p.weak]
    assert [(p.first, p.second) for p in weak] == [("d", "e")] and found.pairs[-1] is weak[0]


def test_h_finds_the_pair_that_multiplies() -> None:
    found = pairs.interactions(Product(), DATA, "regression", ["a", "b", "c"])  # type: ignore[arg-type]
    by_pair = {frozenset((p.first, p.second)): p for p in found.pairs}
    assert by_pair[frozenset(("a", "b"))].h > 0.5
    assert by_pair[frozenset(("a", "c"))].h < 1e-6
    assert found.pairs[0].first in ("a", "b") and found.pairs[0].second in ("a", "b")
    assert "a × b" in found.sentence or "b × a" in found.sentence


def test_h_takes_the_most_important_columns_and_says_what_it_left_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    found = pairs.interactions(
        Additive(),
        DATA,
        "regression",
        ["d", "a", "b", "c"],
        {"d": "a date"},
        count=2,  # type: ignore[arg-type]
    )
    assert found.columns == ["a", "b"] and len(found.pairs) == 1
    assert found.note and "d (a date)" in found.note
    monkeypatch.setattr(pairs, "BUDGET", 100)
    with pytest.raises(pairs.InteractionError, match="R-XAI-13"):
        pairs.interactions(Additive(), DATA, "regression", ["a", "b"])  # type: ignore[arg-type]


# --- Anchors ------------------------------------------------------------------------------------

FIELDS_ABCD = [
    Field("a", "numeric_continuous", RAW),
    Field("b", "numeric_continuous", RAW),
    Field("c", "categorical_nominal", RAW),
    Field("d", "numeric_continuous", RAW),
]


def _positive_row() -> int:
    where = np.flatnonzero((DATA["a"] > 1.0) & (DATA["c"] == "high"))
    return int(ROWS[where[0]])


def test_anchor_finds_the_rule_and_holds_its_precision_on_fresh_samples() -> None:
    row = _positive_row()
    found = anchoring.anchor(Rule(), DATA, ROWS, "binary", row, FIELDS_ABCD)  # type: ignore[arg-type]
    assert found.reached and found.predicted == "yes"
    assert {c.column for c in found.conditions} == {"a", "c"}
    assert found.precision_low >= found.threshold == 0.95
    assert found.rule.startswith("IF ") and "THEN the class “yes”" in found.rule
    # the precision holds on samples drawn afresh (another seed), not only on the search's own
    conditions = [
        anchoring._condition(
            cast(pd.Series, DATA[c.column]), c.column, DATA.loc[row == ROWS, c.column].iloc[0]
        )
        for c in found.conditions
    ]
    fresh = anchoring._Search(
        DATA,
        [c for c in conditions if c],
        anchoring._Outcome(Rule(), "binary", DATA.loc[row == ROWS], DATA, None, None),
        seed=7,
    )  # type: ignore[arg-type]
    rule = tuple(range(len(conditions)))
    fresh.measure([rule], 0, anchoring.SAMPLES + anchoring.MORE_SAMPLES)
    assert fresh.precision(rule) >= 0.95
    # coverage: the share of the rows that meet both conditions
    a_text = next(c.text for c in found.conditions if c.column == "a")
    assert ">" in a_text
    assert 0 < found.coverage < 0.5


def test_anchor_says_when_no_short_rule_reaches_the_threshold() -> None:
    row = _positive_row()
    found = anchoring.anchor(Rule(), DATA, ROWS, "binary", row, FIELDS_ABCD, max_conditions=1)  # type: ignore[arg-type]
    assert not found.reached and len(found.conditions) == 1
    assert "No rule of at most 1 conditions" in found.sentence


def test_anchor_for_a_number_and_left_out_columns() -> None:
    fields = [*FIELDS_ABCD[:3], Field("d", "identifier", RAW)]
    row = int(ROWS[0])
    found = anchoring.anchor(Additive(), DATA, ROWS, "regression", row, fields)  # type: ignore[arg-type]
    assert "a prediction within" in found.outcome
    assert "d" not in {c.column for c in found.conditions}
    assert found.left_out == ["d (an identifier)"]
    with pytest.raises(anchoring.AnchorError, match="not among"):
        anchoring.anchor(Additive(), DATA, ROWS, "regression", 5, fields)  # type: ignore[arg-type]


def test_wilson_lower_bound() -> None:
    assert anchoring.wilson_low(400, 400) == pytest.approx(0.9905, abs=1e-3)
    assert anchoring.wilson_low(0, 0) == 0.0


# --- prototypes and criticisms ------------------------------------------------------------------


def _clusters() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(5)
    parts, labels = [], []
    for label, centres in (
        ("left", [(-6, 0, 80), (-6, 6, 80), (-6, 12, 15)]),
        ("right", [(6, 0, 80), (6, 6, 80)]),
    ):
        for cx, cy, n in centres:
            parts.append(np.c_[rng.normal(cx, 0.3, n), rng.normal(cy, 0.3, n)])
            labels += [label] * n
    z = np.vstack(parts)
    return pd.DataFrame({"x": z[:, 0], "y": z[:, 1]}), np.array(labels)


def test_prototypes_cover_each_cluster_and_the_outlier_is_a_criticism() -> None:
    x, y = _clusters()
    rows = np.arange(len(x))
    fields = [Field("x", "numeric_continuous", RAW), Field("y", "numeric_continuous", RAW)]
    found = examples.prototypes(None, x, y, rows, "binary", fields, count=2, criticisms=1)
    left = next(g for g in found.groups if g.label == "left")
    ys = sorted(float(x.iloc[p.row]["y"]) for p in left.prototypes)
    assert abs(ys[0] - 0) < 1.5 and abs(ys[1] - 6) < 1.5  # one prototype per cluster
    assert all(float(x.iloc[p.row]["x"]) < 0 for p in left.prototypes)
    # two prototypes cannot stand for three groups: the small one gives the criticism
    assert [abs(float(x.iloc[c.row]["y"]) - 12) < 1.5 for c in left.criticisms] == [True]
    assert left.criticisms[0].witness is not None and left.criticisms[0].witness > 0
    assert all(b < a for a, b in zip(left.mmd, left.mmd[1:], strict=False))  # MMD² falls
    assert abs(sum(p.share or 0 for p in left.prototypes) - 1) < 1e-9
    assert found.columns == ["x", "y"] and len(found.groups) == 2


def test_prototypes_encode_categories_and_leave_text_out() -> None:
    frame = pd.DataFrame(
        {"n": [1.0, 2.0, np.nan, 4.0], "c": ["a", "b", "a", None], "t": list("wxyz")}
    )
    z = examples.encode(frame, ["n", "c"])
    assert z.shape == (4, 4)  # one number, three indicators (a, b, (empty))
    assert np.isfinite(z).all()
    fields = [Field("t", "text", RAW)]
    found = examples.prototypes(
        None, frame, np.zeros(4), np.arange(4), "regression", fields, count=1, criticisms=0
    )
    assert found.groups[0].label == "All rows" and "t (free text)" in found.left_out
