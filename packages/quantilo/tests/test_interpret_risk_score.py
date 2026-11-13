# ruff: noqa: RUF002, RUF003  (the usual notation: ≤, −, σ)
"""121 — FasterRisk re-implemented: a known card is found again, the loss is within 1 % of a
brute-force optimum, and every card is sparse with small integer points (R-XAI-06 to 08)."""

import itertools

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from quantilo.interpret import compare
from quantilo.interpret import risk_score as fr
from quantilo.interpret.conditions import Condition, conditions_for, matrix


def _known(n: int = 20_000, seed: int = 0):
    """y drawn from a card: 3·x0 − 2·x1 + 1·x2 − 1, divided by a multiplier of 1.5."""
    rng = np.random.default_rng(seed)
    x = (rng.random((n, 8)) < rng.uniform(0.2, 0.6, 8)).astype(float)
    total = 3 * x[:, 0] - 2 * x[:, 1] + 1 * x[:, 2]
    y = (rng.random(n) < expit((total - 1) / 1.5)).astype(float)
    return x, y, total


def test_a_known_card_is_found_again() -> None:
    x, y, total = _known()
    cards = fr.fit(x, y, k=3, bound=5)
    best = cards[0]
    found = dict(zip(best.support, best.points, strict=True))
    assert set(found) == {0, 1, 2}
    assert found[0] > 0 and found[1] < 0 and found[2] > 0
    # the same ratio 3 : −2 : 1, up to one point (the multiplier is free)
    unit = found[0] / 3
    assert abs(found[1] - (-2 * unit)) <= 1 and abs(found[2] - unit) <= 1
    learnt = x[:, best.support] @ np.array(best.points)
    assert compare.auc(learnt, y) >= compare.auc(total, y) - 0.005


def _brute_force(x: np.ndarray, y: np.ndarray, size: int, bound: int) -> float:
    best = np.inf
    for support in itertools.combinations(range(x.shape[1]), size):
        for points in itertools.product(range(-bound, bound + 1), repeat=size):
            if 0 in points:
                continue
            total = x[:, list(support)] @ np.array(points, dtype=float)
            best = min(best, fr.calibrate(total, y)[2])
    return float(best)


def test_the_loss_is_within_one_percent_of_the_brute_force_optimum() -> None:
    rng = np.random.default_rng(5)
    n = 3_000
    x = (rng.random((n, 5)) < [0.3, 0.5, 0.4, 0.2, 0.6]).astype(float)
    y = (rng.random(n) < expit(1.3 * x[:, 1] - 0.8 * x[:, 3] + 0.4 * x[:, 4] - 0.5)).astype(float)
    optimum = _brute_force(x, y, 2, 3)
    cards = fr.fit(x, y, k=2, bound=3)
    assert cards[0].loss <= optimum * 1.01


def test_every_card_is_sparse_with_small_integer_points() -> None:
    x, y, _ = _known(5_000, seed=2)
    for k, bound in [(1, 3), (2, 5), (4, 10)]:
        cards = fr.fit(x, y, k=k, bound=bound)
        assert cards
        supports = set()
        for card in cards:
            assert 1 <= len(card.support) <= k
            assert all(isinstance(p, int) and p != 0 and abs(p) <= bound for p in card.points)
            assert card.scale >= 0
            supports.add(tuple(sorted(card.support)))
        assert len(supports) == len(cards)  # one card per set of conditions


def test_calibration_recovers_a_known_intercept_and_scale() -> None:
    rng = np.random.default_rng(1)
    total = rng.integers(-5, 6, 50_000).astype(float)
    y = (rng.random(len(total)) < expit(-0.7 + 0.4 * total)).astype(float)
    a, b, _ = fr.calibrate(total, y)
    assert a == pytest.approx(-0.7, abs=0.05) and b == pytest.approx(0.4, abs=0.02)


def test_auc_counts_ties_half_like_mann_whitney() -> None:
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(4)
    score = rng.integers(0, 5, 400).astype(float)
    truth = (rng.random(400) < 0.3).astype(float)
    assert compare.auc(score, truth) == pytest.approx(roc_auc_score(truth, score), abs=1e-12)


# --- Conditions (R-XAI-06) and the card as read -------------------------------------------------


def _frame(n: int = 4_000, seed: int = 7) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    tenure = rng.integers(0, 73, n).astype(float)
    tenure[rng.random(n) < 0.03] = np.nan
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.5, 0.25, 0.25])
    senior = rng.random(n) < 0.15
    charges = rng.normal(65, 30, n)
    risk = (
        -1.0
        - 0.04 * np.nan_to_num(tenure, nan=30)
        + 1.6 * (contract == "Month-to-month")
        + 0.6 * senior
    )
    y = (rng.random(n) < expit(risk + 1.5)).astype(float)
    frame = pd.DataFrame(
        {
            "tenure": tenure,
            "Contract": contract,
            "Senior": np.where(senior, "True", "False"),
            "charges": charges,
            "signed": pd.date_range("2020-01-01", periods=n, freq="h"),
        }
    )
    return frame, y


def test_conditions_are_round_thresholds_frequent_categories_and_empty_cells() -> None:
    frame, _ = _frame()
    found = conditions_for(frame)
    texts = [c.text for c in found]
    assert "Contract = Month-to-month" in texts and "Senior = True" in texts
    assert "Senior = False" not in texts  # two values: one condition
    assert "tenure is empty" in texts
    assert not [c for c in found if c.column == "signed"]  # dates left out
    thresholds = [c.number for c in found if c.kind == "below" and c.column == "charges"]
    assert thresholds and all(float(f"{t:.2g}") == t for t in thresholds)
    held = matrix(frame, found)
    assert held.shape == (len(frame), len(found)) and set(np.unique(held)) <= {0.0, 1.0}
    below = Condition("tenure", "below", "12", 12.0)
    assert not below.holds(frame)[frame["tenure"].isna().to_numpy()].any()


def test_risk_scores_reads_like_a_card_and_is_compared_with_the_model() -> None:
    frame, y = _frame()
    learn, check = frame.iloc[:3_000], frame.iloc[3_000:]
    y_learn, y_check = y[:3_000], y[3_000:]
    # a "model" that knows the truth a little better than any card
    model = np.clip(y_check * 0.3 + np.random.default_rng(0).random(len(y_check)) * 0.7, 0, 1)
    found = fr.risk_scores(
        learn, y_learn, check, y_check, model, "Yes", "random forest", "split", k=4, bound=5
    )
    card = found.best
    assert 1 <= len(card.lines) <= 4
    assert any(line.column == "Contract" for line in card.lines)
    thresholds = [line for line in card.lines if " > " in line.condition or " ≤ " in line.condition]
    assert all(line.points > 0 for line in thresholds)  # read so that their points add risk
    risks = [s.risk for s in card.steps]
    assert risks == sorted(risks)  # the risk grows with the total
    assert sum(s.rows for s in card.steps) == len(y_check)
    assert found.gap.rows == len(y_check) and found.gap.low <= found.gap.gap <= found.gap.high
    assert "R-XAI-07" in found.gap.sentence
    shown = {line.condition for line in card.lines}
    assert all({line.condition for line in o.lines} != shown for o in found.others)
    assert found.candidates > 10 and found.learnt == 3_000


def test_a_card_needs_both_classes() -> None:
    frame, _ = _frame(200)
    with pytest.raises(fr.RiskScoreError, match="both classes"):
        fr.risk_scores(
            frame, np.zeros(200), frame, np.zeros(200), np.zeros(200), "Yes", "m", "split"
        )


def test_rows_are_split_as_r_xai_08_says() -> None:
    rows = np.arange(100, 1100)
    split = compare.split(rows, rows[800:], limit=10_000)
    assert len(split.learn) == 800 and len(split.check) == 200 and "test rows" in split.source
    every = compare.split(rows, rows[::-1], limit=10_000)  # cross-validation: every row held out
    assert len(every.learn) == 700 and len(every.check) == 300
    assert not set(every.learn) & set(every.check) and "out-of-fold" in every.source
    assert (rows[::-1][every.held] == rows[every.check]).all()


# --- Through the library: a trained run, its held-out rows (R-XAI-05, R-XAI-08) -----------------


@pytest.fixture(scope="module")
def churn(tmp_path_factory: pytest.TempPathFactory):
    import quantilo as dx

    frame, y = _frame(3_000, seed=11)
    frame = frame.drop(columns=["signed"])
    frame["churn"] = np.where(y == 1, "Yes", "No")
    return dx.from_pandas(frame, "churn", cache_dir=tmp_path_factory.mktemp("churn"))


def test_the_library_compares_a_scorecard_with_the_run(churn) -> None:
    model = churn.ml("churn").train("random_forest", {"n_estimators": 50, "max_depth": 6})
    found = model.scorecard(conditions=4)
    assert found.positive == model.classification.positive
    assert found.checked == model.test_rows and found.learnt == model.train_rows
    assert "test rows" in found.source
    assert 1 <= len(found.best.lines) <= 4
    assert found.gap.model == pytest.approx(model.classification.roc_auc, abs=1e-9)
    tree = model.sparse_tree(depth=3)
    assert tree.leaves <= 12 and tree.gap is not None and tree.checked == model.test_rows


def test_cross_validation_keeps_30_percent_of_the_out_of_fold_rows(churn) -> None:
    model = churn.ml("churn", validation={"method": "kfold", "folds": 3}).train()
    found = model.scorecard(conditions=3)
    assert found.learnt + found.checked == model.rows
    assert "out-of-fold" in found.source


def test_a_number_has_no_scorecard(churn) -> None:
    from quantilo.ml.engine import MlError

    model = churn.ml("charges").train()
    with pytest.raises(MlError, match="R-XAI-05"):
        model.scorecard()


# --- 121 bis --------------------------------------------------------------------------------


def _hump(n: int = 6_000, seed: int = 3) -> tuple[pd.DataFrame, np.ndarray]:
    """Risk rises with age overall, with a dip in the middle (a shape a card can copy)."""
    rng = np.random.default_rng(seed)
    age = rng.uniform(18, 80, n)
    plan = rng.choice(["basic", "pro"], n)
    logit = 0.06 * (age - 50) - 1.2 * ((age > 40) & (age <= 55)) + 0.8 * (plan == "pro")
    y = (rng.random(n) < expit(logit)).astype(float)
    return pd.DataFrame({"age": age, "plan": plan}), y


def test_monotone_points_keep_one_direction_along_each_number() -> None:
    """R-XAI-16: free points may copy the dip; monotone ones never go up and down."""
    frame, y = _hump()
    learn, check = frame.iloc[:4000], frame.iloc[4000:]
    probability = np.full(len(check), y.mean())
    free = fr.risk_scores(learn, y[:4000], check, y[4000:], probability, "1", "model", "test", 6)
    tight = fr.risk_scores(
        learn, y[:4000], check, y[4000:], probability, "1", "model", "test", 6, monotone=True
    )
    assert tight.monotone and not tight.shape
    assert any("R-XAI-16" in n for n in tight.notes)
    ages = [line for line in tight.best.lines if line.column == "age"]
    # the risk rises with age: "age > t" adds points, "age ≤ t" never does
    assert all((line.kind == "above") == (line.points > 0) for line in ages)
    if free.shape:
        assert free.shape[0].startswith("R-XAI-16 · age: the points go up and down")
    lo, hi = fr.monotone_bounds(conditions_for(learn), learn, y[:4000], 5)
    assert ((lo == 0) | (hi == 0) | ((lo < 0) & (hi > 0))).all()


def test_one_card_per_class_for_three_classes() -> None:
    """R-XAI-17: each class against the others; a row goes to the highest risk."""
    rng = np.random.default_rng(1)
    n = 6_000
    region = rng.choice(["north", "south", "east"], n)
    income = rng.normal(50, 15, n)
    label = np.where(region == "north", "a", np.where(income > 55, "b", "c"))
    flip = rng.random(n) < 0.1
    label[flip] = rng.choice(["a", "b", "c"], flip.sum())
    frame = pd.DataFrame({"region": region, "income": income})
    classes = ["a", "b", "c"]
    model = np.eye(3)[[classes.index(c) for c in label[4000:]]] * 0.8 + 0.2 / 3
    found = fr.class_cards(
        frame.iloc[:4000], label[:4000], frame.iloc[4000:], label[4000:], model, classes, "model",
        "test", 3,
    )  # fmt: skip
    assert [c.label for c in found.classes] == classes
    assert found.accuracy > 0.8 and found.model_accuracy > 0.99
    assert all(c.auc > 0.8 for c in found.classes)
    north = next(c for c in found.classes if c.label == "a").card.lines
    assert {line.column for line in north} == {"region"}  # class a is the north
    assert "R-XAI-17" in found.sentence
    with pytest.raises(fr.RiskScoreError, match="R-XAI-17"):
        fr.class_cards(
            frame, label, frame, label, np.ones((n, 9)) / 9, [str(i) for i in range(9)], "m", "t"
        )


def test_the_card_reads_a_new_row_from_its_lines_alone() -> None:
    """R-XAI-18: the lines carry their condition, so new rows are scored without the learnt rows."""
    frame, y = _hump()
    learn, check = frame.iloc[:4000], frame.iloc[4000:]
    found = fr.risk_scores(
        learn, y[:4000], check, y[4000:], np.full(2000, 0.5), "1", "model", "test", 4
    )
    totals = fr.apply_lines(found.best.lines, check)
    assert np.allclose(totals, fr.apply(found.best.lines, check, conditions_for(learn)))
    row = fr.card_row(found, check.iloc[[0]].reset_index(drop=True), 4000)
    assert row.total == int(totals[0])
    assert row.risk == pytest.approx(expit(found.best.intercept + found.best.scale * row.total))
    assert "R-XAI-18" in row.sentence and row.positive == "1"


def test_a_number_gets_a_sparse_tree_against_the_model(churn) -> None:
    """122 bis, R-XAI-19: the readable tree of a regression, judged by R² against the run."""
    model = churn.ml("charges").train()
    tree = model.sparse_tree(depth=3)
    assert tree.measure == "R²" and tree.gap is not None and tree.gap.metric == "R²"
    assert tree.gap.model == pytest.approx(model.regression.r2, abs=1e-6)  # type: ignore[union-attr]
    assert "R-XAI-19" in tree.gap.sentence


def test_three_classes_get_one_card_each_through_the_library(churn) -> None:
    """121 bis, R-XAI-17: a three-class target, one card per class, against the run."""
    import quantilo as dx

    frame = churn.to_pandas()
    bands = pd.cut(frame["charges"], 3, labels=["low", "mid", "high"])
    frame["band"] = pd.Series(bands).astype(str)
    data = dx.from_pandas(frame.drop(columns=["churn", "charges"]), "bands")
    model = data.ml("band").train("random_forest", {"n_estimators": 30, "max_depth": 5})
    found = model.class_scorecards(conditions=3)
    assert {c.label for c in found.classes} <= {"low", "mid", "high"}
    assert found.checked == model.test_rows
    assert 0 < found.accuracy <= 1 and 0 < found.model_accuracy <= 1
    with pytest.raises(Exception, match="R-XAI-05"):
        model.scorecard()


def test_new_data_gets_the_card_and_a_counterfactual(churn) -> None:
    """121 bis (R-XAI-18) and 125 bis (R-XAI-21): one row of another dataset."""
    model = churn.ml("churn").train("random_forest", {"n_estimators": 40, "max_depth": 6})
    new, _ = _frame(300, seed=99)
    new = new.drop(columns=["signed"])
    row = model.scorecard_row(new, 5, conditions=4)
    assert row.row == 5 and "R-XAI-18" in row.sentence
    assert row.total == sum(line.points for line in row.lines)
    predicted = model.predict(new)["predicted"].iloc[5]
    wanted = next(c for c in model.card.labels if c != predicted)
    found = model.what_would_change(new, 5, target_class=wanted)
    assert found.row == 5 and len(found.changes) <= 3
    assert "R-XAI-21" in found.plausibility
