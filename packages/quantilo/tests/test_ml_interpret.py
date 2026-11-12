# ruff: noqa: RUF002  (the usual notation)
"""125 — tabular interpretability under Advanced: accumulated local effects (against their closed
form), the monotonic check, the global surrogate and its fidelity, counterfactuals."""

from typing import cast

import numpy as np
import pandas as pd
import pytest

from quantilo.ml import ale as effects
from quantilo.ml import counterfactual as whatif
from quantilo.ml import surrogate as imitation
from quantilo.ml.data import Field
from quantilo.ml.train import train
from tests.test_ml import FIELDS, FRAME, RAW, by_name
from tests.test_ml_explain import FOLDS

RNG = np.random.default_rng(3)
N = 800
STEP = {"low": 0.0, "mid": 5.0, "high": 7.0}


class Formula:
    """A model whose prediction is known: 2a + 3b² + step(c) + a·d (no fitting)."""

    classes_: tuple[str, ...] = ()

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        a, b, d = (frame[c].to_numpy(dtype=float) for c in ("a", "b", "d"))
        return 2 * a + 3 * b**2 + frame["c"].map(STEP).to_numpy(dtype=float) + a * d


def _frame() -> pd.DataFrame:
    a = RNG.normal(0, 1, N)
    return pd.DataFrame(
        {
            "a": a,
            "b": RNG.uniform(-1, 2, N),
            "c": RNG.choice(list(STEP), N, p=[0.5, 0.3, 0.2]),
            "d": a + RNG.normal(0, 0.1, N),  # strongly linked to a
        }
    )


DATA = _frame()


def test_ale_of_a_number_matches_its_closed_form() -> None:
    model = cast(object, Formula())
    found = effects.ale(model, DATA, "regression", "b", bins=10)  # type: ignore[arg-type]
    edges = np.array(found.grid, dtype=float)
    counts = np.array(found.counts)
    exact = 3 * (edges**2 - edges[0] ** 2)  # the effect of 3b², accumulated from the first edge
    middle = (exact[1:] + exact[:-1]) / 2
    exact -= (counts * middle).sum() / counts.sum()
    assert found.kind == "number" and len(edges) == 11 and counts.sum() == N
    assert np.allclose(found.curves["prediction"], exact, atol=1e-9)


def test_ale_reads_real_neighbours_where_the_partial_dependence_extrapolates() -> None:
    """a and d move together. Along a, at fixed d, 2a + a·d has slope 2 + d: in each interval
    the local effect is (2 + the mean d of its rows) × its width, the closed form on real rows
    (the partial dependence would use the mean d of every row instead)."""
    found = effects.ale(Formula(), DATA, "regression", "a", bins=20)  # type: ignore[arg-type]
    edges = np.array(found.grid, dtype=float)
    values = DATA["a"].to_numpy()
    k = np.clip(np.searchsorted(edges, values, side="left"), 1, len(edges) - 1)
    mean_d = np.array([DATA["d"].to_numpy()[k == j].mean() for j in range(1, len(edges))])
    local = (2 + mean_d) * np.diff(edges)
    expected = np.r_[0.0, np.cumsum(local)]
    middle = (expected[1:] + expected[:-1]) / 2
    expected -= (np.array(found.counts) * middle).sum() / sum(found.counts)
    assert np.allclose(found.curves["prediction"], expected, atol=1e-9)


def test_ale_of_ordered_categories_matches_the_step_function() -> None:
    found = effects.ale(
        Formula(),
        DATA,
        "regression",
        "c",
        orders={"c": ["low", "mid", "high"]},  # type: ignore[arg-type]
    )
    counts = np.array(found.counts)
    steps = np.array([STEP[c] for c in found.grid])
    assert found.kind == "category" and found.grid == ["low", "mid", "high"]
    assert found.order == "the order set in Columns"
    assert np.allclose(found.curves["prediction"], steps - (counts * steps).sum() / counts.sum())


def test_ale_needs_an_order_and_numbers() -> None:
    with pytest.raises(effects.AleError, match="no order"):
        effects.ale(Formula(), DATA, "regression", "c")  # type: ignore[arg-type]
    with pytest.raises(effects.AleError, match="not an input"):
        effects.ale(Formula(), DATA, "regression", "zzz")  # type: ignore[arg-type]


def test_the_monotonic_check_reads_every_rows_curve() -> None:
    held = effects.monotonic(Formula(), DATA.assign(a=DATA["a"].abs()), "regression", "b", -1)  # type: ignore[arg-type]
    assert held.violations > 0 and "not held" in held.sentence  # 3b² falls then rises
    rising = effects.monotonic(Formula(), DATA.assign(d=0.0), "regression", "a", 1)  # type: ignore[arg-type]
    assert rising.violations == 0 and "only rises" in rising.sentence


def test_a_monotonic_constraint_trained_with_hgb_is_held() -> None:
    fields = [f for f in FIELDS if f.name in ("size", "skewed", "city")]
    result = train(
        FRAME,
        fields,
        by_name["price"],
        "regression",
        "hist_gradient_boosting",
        {"monotonic": "skewed:-1", "max_iter": 40},
        1,
        False,
        validation=FOLDS,
    )
    x = cast(pd.DataFrame, FRAME[[str(c) for c in result.pipeline.feature_names_in_]])
    found = effects.ale(result.pipeline, x, "regression", "skewed", sign=-1)
    assert found.monotonic is not None and found.monotonic.violations == 0


def _churn(n: int = 900) -> tuple[pd.DataFrame, list[Field]]:
    tenure = RNG.integers(0, 72, n)
    monthly = RNG.uniform(20, 120, n)
    contract = RNG.choice(["month", "year", "two years"], n, p=[0.5, 0.3, 0.2])
    age = RNG.integers(18, 90, n)
    score = -0.06 * tenure + 0.03 * monthly + np.where(contract == "month", 1.5, -1.0)
    churn = np.where(score + RNG.normal(0, 0.5, n) > 0.5, "yes", "no")
    frame = pd.DataFrame(
        {
            "tenure": tenure,
            "monthly": monthly,
            "contract": contract,
            "age": age,
            "customer": [f"C{i}" for i in range(n)],
            "churn": churn,
            "__row": np.arange(n),
        }
    )
    fields = [
        Field("tenure", "numeric_discrete", RAW),
        Field("monthly", "numeric_continuous", RAW),
        Field("contract", "categorical_nominal", RAW),
        Field("age", "numeric_discrete", RAW, role="age"),
        Field("customer", "identifier", RAW),
        Field("churn", "categorical_nominal", RAW),
    ]
    return frame, fields


CHURN, CHURN_FIELDS = _churn()
CHURN_RESULT = train(
    CHURN,
    CHURN_FIELDS,
    CHURN_FIELDS[-1],
    "binary",
    "logistic_regression",
    None,
    1,
    False,
    validation=FOLDS,
)
CHURN_X = cast(pd.DataFrame, CHURN[[str(c) for c in CHURN_RESULT.pipeline.feature_names_in_]])
CHURN_ROWS = CHURN["__row"].to_numpy(dtype=int)


def test_counterfactual_flips_the_class_with_few_changes_on_free_columns() -> None:
    pipeline = CHURN_RESULT.pipeline
    predicted = pipeline.predict(CHURN_X)
    row = int(np.flatnonzero(predicted == "yes")[0])
    found = whatif.counterfactual(pipeline, CHURN_X, CHURN_ROWS, "binary", row, CHURN_FIELDS, set())
    locked = {c.column: c.locked for c in found.columns}
    assert locked["age"] and "age" in str(locked["age"])  # R-XAI-03: personal attribute
    assert locked["tenure"] is None
    assert found.reached and 1 <= len(found.changes) <= 3
    assert found.target_class == "no" and found.after > found.before  # probability of "no"
    assert {c.column for c in found.changes} <= {"tenure", "monthly", "contract"}
    changed = CHURN_X.iloc[[row]].copy()
    for step in found.changes:
        value = step.after if step.column == "contract" else float(step.after)
        changed[step.column] = [value]
    assert pipeline.predict(changed.astype({"tenure": float}))[0] == "no"
    assert found.nearest_row is not None and found.plausibility
    assert found.table[0].column in {c.column for c in found.changes}


def test_counterfactual_respects_the_chosen_columns_and_says_when_it_cannot() -> None:
    pipeline = CHURN_RESULT.pipeline
    predicted = pipeline.predict(CHURN_X)
    row = int(np.flatnonzero(predicted == "yes")[0])
    only = whatif.counterfactual(
        pipeline,
        CHURN_X,
        CHURN_ROWS,
        "binary",
        row,
        CHURN_FIELDS,
        set(),
        editable=["monthly"],
        max_changes=1,
    )
    assert all(c.column == "monthly" for c in only.changes) and len(only.changes) <= 1
    if not only.reached:
        assert only.sentence.startswith("No change")
    locked = whatif.counterfactual(
        pipeline, CHURN_X, CHURN_ROWS, "binary", row, CHURN_FIELDS, set(), editable=[]
    )
    assert not locked.changes and "unlock" in locked.sentence
    with pytest.raises(whatif.CounterfactualError, match="not among"):
        whatif.counterfactual(pipeline, CHURN_X, CHURN_ROWS, "binary", 10**6, CHURN_FIELDS, set())


def test_counterfactual_for_a_number_reaches_the_target_value() -> None:
    result = train(
        FRAME,
        [f for f in FIELDS if f.name in ("size", "skewed", "city")],
        by_name["price"],
        "regression",
        "linear_regression",
        None,
        1,
        False,
        validation=FOLDS,
    )
    x = cast(pd.DataFrame, FRAME[[str(c) for c in result.pipeline.feature_names_in_]])
    rows = FRAME["__row"].to_numpy(dtype=int)
    now = float(result.pipeline.predict(x.iloc[[0]])[0])
    found = whatif.counterfactual(
        result.pipeline, x, rows, "regression", 0, FIELDS, set(), target_value=now + 40
    )
    assert found.reached and found.after >= now + 40 and found.target_value == now + 40


def test_the_surrogate_imitates_the_model_and_says_how_well() -> None:
    result = train(
        FRAME,
        [f for f in FIELDS if f.name in ("size", "skewed", "city", "when")],
        by_name["price"],
        "regression",
        "random_forest",
        {"n_estimators": 20},
        1,
        False,
        validation=FOLDS,
    )
    x = cast(pd.DataFrame, FRAME[[str(c) for c in result.pipeline.feature_names_in_]])
    found = imitation.surrogate(result.pipeline, x, "regression", depth=3)
    assert found.measure == "R²" and found.fidelity > 0.6
    assert found.nodes[0].left_rule and found.nodes[0].left_rule.startswith("size ≤")
    assert max(n.depth for n in found.nodes) <= 3 and found.leaves <= 8
    shallow = imitation.surrogate(result.pipeline, x, "regression", depth=1)
    assert shallow.fidelity < found.fidelity


def test_a_surrogate_of_low_fidelity_is_flagged() -> None:
    found = imitation.surrogate(CHURN_RESULT.pipeline, CHURN_X, "binary", depth=1)
    assert found.measure == "agreement" and 0 <= found.fidelity <= 1
    if found.fidelity < imitation.FIDELITY:
        assert found.warning and found.warning.startswith("R-XAI-04")
    rules = {r for n in found.nodes for r in (n.left_rule, n.right_rule) if r}
    assert rules  # categories read as "contract = month", numbers as "tenure ≤ 12"
