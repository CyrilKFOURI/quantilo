"""036 — where the model is wrong: failing subgroups (Benjamini-Hochberg), error by column, worst
predictions, hints, two runs compared, metrics per group, on held-out predictions only."""

import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

import quantilo as dx
from quantilo.ml import error_analysis as ea
from quantilo.ml import evaluate
from quantilo.ml.data import Field
from quantilo.ml.train import Trained, train
from quantilo.typing.types import Parser

RAW = Parser(kind="raw")
FIELDS = [
    Field("region", "categorical_nominal", RAW),
    Field("age", "numeric_continuous", RAW),
    Field("income", "numeric_continuous", RAW),
    Field("plan", "categorical_nominal", RAW),
]


def customers(n: int = 6_000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "region": rng.choice(["North", "South", "East", "West"], n),
            "age": rng.uniform(18, 70, n).round(1),
            "income": rng.normal(40_000, 8_000, n).round(0),
            "plan": rng.choice(["basic", "plus", "pro"], n),
        }
    )


def regression_held(frame: pd.DataFrame, noise: np.ndarray, seed: int = 1) -> ea.HeldOut:
    actual = np.random.default_rng(seed).normal(100, 10, len(frame))
    return ea.HeldOut(np.arange(len(frame)), actual, actual + noise)


# --- Subgroups ------------------------------------------------------------------------------------


def test_a_noisy_subgroup_is_found_first() -> None:
    frame = customers()
    rng = np.random.default_rng(2)
    noisy = (frame["region"] == "South") & (frame["age"] < 25)
    noise = rng.normal(0, 1, len(frame)) * np.where(noisy, 4.0, 1.0)

    found = ea.analyse(frame, FIELDS, regression_held(frame, noise), "regression")

    first = found.slices[0]
    assert set(first.columns) == {"region", "age"}
    assert "region = South" in first.where
    upper = [c for c in first.conditions if c.column == "age" and c.op in ("lt", "lte")]
    assert upper and 22 <= cast(float, upper[0].value) <= 30
    assert first.ratio > 2 and first.q_value < 0.05
    assert first.sentence.startswith("When ") and "of rows" in first.sentence
    assert found.message is None
    # every listed subgroup is notably worse than average and significant after BH
    assert all(s.error >= ea.NOTABLY * found.overall and s.q_value < ea.ALPHA for s in found.slices)
    # "South" and a second condition that adds nothing are not listed apart
    assert not any(s.columns == ["region", "income"] for s in found.slices)


def test_uniform_errors_give_no_false_subgroup() -> None:
    frame = customers()
    noise = np.random.default_rng(3).normal(0, 1, len(frame))

    found = ea.analyse(frame, FIELDS, regression_held(frame, noise), "regression")

    assert found.slices == []
    assert found.message == "No subgroup is notably worse than average."
    assert found.tested > 100  # categories, quintiles, pairs and tree leaves were all tried


def test_a_subgroup_of_12_rows_is_not_listed() -> None:
    frame = customers(3_000)
    frame.loc[:11, "plan"] = "legacy"  # 12 rows, badly predicted
    noise = np.random.default_rng(4).normal(0, 1, len(frame))
    noise[:12] *= 30

    found = ea.analyse(frame, FIELDS, regression_held(frame, noise), "regression")

    assert not any("legacy" in s.where for s in found.slices)
    assert all(s.rows >= max(ea.MIN_ROWS, 0.01 * len(frame)) for s in found.slices)


def test_a_classifier_is_judged_on_wrong_predictions_or_log_loss() -> None:
    frame = customers(4_000)
    rng = np.random.default_rng(5)
    actual = rng.choice(["yes", "no"], len(frame))
    hard = frame["plan"] == "pro"
    right = rng.random(len(frame)) > np.where(hard, 0.45, 0.1)
    predicted = np.where(right, actual, np.where(actual == "yes", "no", "yes"))
    chance = np.where(right, 0.8, 0.3)
    p_yes = np.where(actual == "yes", chance, 1 - chance)
    held = ea.HeldOut(
        np.arange(len(frame)), actual, predicted, np.column_stack([1 - p_yes, p_yes]), ["no", "yes"]
    )

    wrong = ea.analyse(frame, FIELDS, held, "binary")
    loss = ea.analyse(frame, FIELDS, held, "binary", "log_loss")

    assert wrong.measures == ["misclassified", "log_loss"] and wrong.measure == "misclassified"
    assert wrong.slices[0].where == "plan = pro"
    assert "wrong" in wrong.slices[0].sentence and "% of the time" in wrong.slices[0].sentence
    assert loss.slices[0].where == "plan = pro" and "log-loss" in loss.slices[0].sentence
    assert wrong.worst[0].probability is not None and wrong.worst[0].probability <= 0.3
    with pytest.raises(ValueError, match="not a measure"):
        ea.analyse(frame, FIELDS, held, "binary", "absolute_error")


def test_subgroup_filters_select_the_same_rows_as_sql(tmp_path: Path) -> None:
    frame = customers(3_000)
    rng = np.random.default_rng(6)
    noisy = (frame["age"] > 60) | ((frame["plan"] == "pro") & (frame["region"] != "North"))
    noise = rng.normal(0, 1, len(frame)) * np.where(noisy, 5.0, 1.0)
    found = ea.analyse(frame, FIELDS, regression_held(frame, noise), "regression")
    ds = dx.from_pandas(frame, "customers", cache_dir=tmp_path)

    assert len(found.slices) >= 2
    for s in found.slices:
        view = ds
        for condition in s.conditions:
            view = view.where(condition)
        assert view.rows(limit=1).total == s.rows, s.where


# --- Error by column, worst predictions -----------------------------------------------------------


def test_error_by_column_and_the_worst_predictions() -> None:
    frame = customers(3_000)
    noise = np.random.default_rng(7).normal(0, 1, len(frame)) * np.where(
        frame["region"] == "East", 3.0, 1.0
    )
    held = regression_held(frame, noise)

    found = ea.analyse(frame, FIELDS, held, "regression")

    first = found.by_column[0]
    assert first.column == "region" and first.worst == "East" and first.kind == "category"
    spreads = [c.spread for c in found.by_column]
    assert spreads == sorted(spreads, reverse=True)
    age = next(c for c in found.by_column if c.column == "age")
    assert age.kind == "number" and len(age.bins) == 5 and age.bins[0].label.startswith("< ")
    assert len(found.worst) == ea.WORST
    errors = [cast(float, w.error) for w in found.worst]
    assert errors == sorted(errors, reverse=True)
    assert errors[0] == pytest.approx(float(np.abs(noise).max()))


# --- Hints ----------------------------------------------------------------------------------------


def test_errors_growing_with_the_value_suggest_its_log() -> None:
    frame = customers(3_000)
    rng = np.random.default_rng(8)
    predicted = np.exp(rng.normal(3, 1, len(frame)))
    actual = predicted * np.exp(rng.normal(0, 0.3, len(frame)))
    held = ea.HeldOut(np.arange(len(frame)), actual, predicted)

    found = ea.analyse(frame, FIELDS, held, "regression", target="price")

    assert any(h.kind == "log_target" and "log" in h.text for h in found.hints)
    flat = regression_held(frame, rng.normal(0, 1, len(frame)))
    assert not any(
        h.kind == "log_target" for h in ea.analyse(frame, FIELDS, flat, "regression").hints
    )


def test_hints_for_a_small_group_an_excluded_column_and_a_rare_class() -> None:
    frame = customers(6_000)
    rng = np.random.default_rng(9)
    noise = rng.normal(0, 1, len(frame)) * np.where(
        (frame["region"] == "West") & (frame["age"] > 64), 6.0, 1.0
    )  # about 3 % of the rows
    found = ea.analyse(
        frame,
        FIELDS,
        regression_held(frame, noise),
        "regression",
        excluded={"region"},
    )
    kinds = {h.kind for h in found.hints}
    assert {"few_rows", "excluded_column"} <= kinds
    assert any("region was excluded" in h.text for h in found.hints)

    actual = np.where(rng.random(len(frame)) < 0.08, "fraud", "ok")
    predicted = np.where((actual == "fraud") & (rng.random(len(frame)) < 0.7), "ok", actual)
    p = np.where(predicted == "fraud", 0.7, 0.2)
    held = ea.HeldOut(
        np.arange(len(frame)), actual, predicted, np.column_stack([p, 1 - p]), ["fraud", "ok"]
    )
    rare = ea.analyse(frame, FIELDS, held, "binary", positive="fraud")
    hint = next(h for h in rare.hints if h.kind == "rare_class")
    assert "fraud" in hint.text and "lower decision threshold" in hint.text


# --- Two runs, groups of a sensitive column -------------------------------------------------------


def test_two_runs_compared_subgroup_by_subgroup() -> None:
    frame = customers(5_000)
    rng = np.random.default_rng(10)
    base = rng.normal(0, 1, len(frame))
    first = regression_held(frame, base * np.where(frame["region"] == "North", 4.0, 1.0))
    second = regression_held(frame, base * np.where(frame["plan"] == "basic", 3.0, 1.0))

    found = ea.compare(frame, FIELDS, first, second, "regression")

    assert found.rows == len(frame)
    assert found.better[0].where == "region = North"
    assert found.better[0].second < found.better[0].first
    assert found.worse[0].where == "plan = basic"
    assert "goes from" in found.worse[0].sentence
    same = ea.compare(frame, FIELDS, first, first, "regression")
    assert same.better == [] and same.worse == [] and same.message


def test_metrics_side_by_side_for_a_sensitive_column() -> None:
    frame = customers(2_000)
    rng = np.random.default_rng(11)
    actual = rng.choice(["yes", "no"], len(frame))
    missed = (frame["region"] == "South").to_numpy() & (actual == "yes")
    predicted = np.where(missed & (rng.random(len(frame)) < 0.6), "no", actual)
    p = np.where(predicted == "yes", 0.8, 0.2)
    held = ea.HeldOut(
        np.arange(len(frame)), actual, predicted, np.column_stack([1 - p, p]), ["no", "yes"]
    )

    found = ea.fairness(cast(pd.Series, frame["region"]), FIELDS[0], held, "binary", "yes")

    assert found.metrics == ["error_rate", "positive_rate", "recall"]
    by_group = {g.group: g for g in found.groups}
    assert set(by_group) == {"North", "South", "East", "West"}
    assert cast(float, by_group["South"].values["recall"]) < cast(
        float, by_group["North"].values["recall"]
    )
    gap = cast(float, found.gaps["recall"])
    assert gap > 0.4 and found.positive == "yes"
    ages = ea.fairness(cast(pd.Series, frame["age"]), FIELDS[1], held, "binary", "yes")
    assert len(ages.groups) == 5


# --- Held-out rows only ---------------------------------------------------------------------------


def _trained_on(validation: evaluate.Validation) -> Trained:
    frame = customers(600, seed=12)
    rng = np.random.default_rng(12)
    frame["spend"] = frame["income"] / 100 + rng.normal(0, 30, len(frame))
    frame["__row"] = np.arange(len(frame))
    fields = [*FIELDS, Field("spend", "numeric_continuous", RAW)]
    return train(
        frame,
        fields,
        fields[-1],
        "regression",
        "decision_tree",
        None,
        600,
        False,
        validation=validation,
    )


def test_a_split_keeps_only_the_test_rows() -> None:
    trained = _trained_on(evaluate.Validation())
    held = cast(ea.HeldOut, trained.held_out)
    assert len(held.rows) == trained.test_rows == 120
    assert set(held.rows) <= set(trained.evaluation_rows)
    worst = ea.worst_rows(held)
    assert set(worst) <= set(trained.evaluation_rows)  # explainable one by one


def test_a_cross_validation_keeps_the_out_of_fold_prediction_of_every_row() -> None:
    trained = _trained_on(evaluate.Validation(method="kfold", folds=5))
    held = cast(ea.HeldOut, trained.held_out)
    assert sorted(held.rows) == list(range(600))
    # a deep tree refits its training rows almost exactly; out of fold it cannot
    assert float(np.abs(held.predicted - held.actual).mean()) > 5


# --- Speed ----------------------------------------------------------------------------------------


def test_two_hundred_thousand_predictions_in_under_five_seconds() -> None:
    n = 200_000
    rng = np.random.default_rng(13)
    columns: dict[str, np.ndarray] = {}
    fields = []
    for i in range(20):
        columns[f"x{i}"] = rng.normal(0, 1, n)
        fields.append(Field(f"x{i}", "numeric_continuous", RAW))
    for i in range(10):
        columns[f"c{i}"] = rng.choice([f"v{k}" for k in range(25)], n)
        fields.append(Field(f"c{i}", "categorical_nominal", RAW))
    frame = pd.DataFrame(columns)
    noise = rng.normal(0, 1, n) * np.where(frame["c0"] == "v3", 3.0, 1.0)

    started = time.perf_counter()
    found = ea.analyse(frame, fields, regression_held(frame, noise), "regression")
    seconds = time.perf_counter() - started

    assert found.slices and found.slices[0].where == "c0 = v3"
    assert found.columns == ea.MAX_COLUMNS
    assert seconds < 5, f"{seconds:.1f} s"


# --- The library ----------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def shop(tmp_path_factory: pytest.TempPathFactory) -> dx.Dataset:
    frame = customers(3_000, seed=14)
    rng = np.random.default_rng(14)
    noisy = (frame["region"] == "South") & (frame["age"] < 25)
    frame["spend"] = (
        frame["income"] / 100
        + 2 * frame["age"]
        + rng.normal(0, 10, len(frame)) * np.where(noisy, 4.0, 1.0)
    ).round(2)
    return dx.from_pandas(frame, "shop", cache_dir=tmp_path_factory.mktemp("shop"))


def test_the_library_finds_where_a_model_is_wrong(shop: dx.Dataset) -> None:
    model = shop.ml("spend", validation={"test_size": 0.4}).train()

    found = model.errors()

    assert found.source == "test rows" and found.rows == model.test_rows
    first = found.slices[0]
    assert "region = South" in first.where and "age" in first.where
    table = dx.to_pandas(found)
    assert list(table["where"]) == [s.where for s in found.slices]
    assert found.to_pandas().equals(table)
    assert {c.column for c in found.by_column} == {"region", "age", "income", "plan"}
    assert len(found.worst) == ea.WORST
    pytest.importorskip("plotly")
    figure: Any = dx.plot(found, column="age")
    assert figure.data[0].type == "bar" and len(figure.data[0].x) == 5
    with pytest.raises(dx.ExplorerError, match="No error by"):
        dx.plot(found, column="nope")
    # the worst predictions are explained row by row
    shap = model.explain().shap
    assert shap is not None and {w.row for w in found.worst} <= set(shap.rows)


def test_the_library_compares_two_models_and_shows_groups(shop: dx.Dataset) -> None:
    ml = shop.ml("spend", validation={"test_size": 0.4})
    linear, tree = ml.train(), ml.train("decision_tree")

    compared = linear.compare_errors(tree)
    assert compared.rows == linear.test_rows
    assert compared.first == pytest.approx(linear.regression.mae, rel=1e-6)
    assert not dx.to_pandas(compared).empty or compared.message

    groups = linear.fairness("region")
    assert groups.metrics == ["mae", "mean_predicted", "mean_actual"]
    assert {g.group for g in groups.groups} == {"North", "South", "East", "West"}
    assert len(dx.to_pandas(groups)) == 4
    with pytest.raises(dx.ExplorerError):
        linear.fairness("nope")
    with pytest.raises(dx.ExplorerError, match="not a measure"):
        linear.errors("log_loss")


def test_a_panel_run_uses_its_out_of_fold_predictions(tmp_path: Path) -> None:
    rng = np.random.default_rng(15)
    era = np.repeat(np.arange(1, 31), 40)
    frame = pd.DataFrame({"era": era, "id": np.tile(np.arange(40), 30)})
    frame["signal"] = rng.normal(0, 1, len(frame))
    frame["target"] = frame["signal"] + rng.normal(0, 1, len(frame))
    ds = dx.from_pandas(frame, "panel", cache_dir=tmp_path)
    validation = {
        "method": "panel_walk_forward",
        "period_column": "era",
        "entity_column": "id",
        "folds": 3,
        "embargo": 1,
    }
    model = ds.ml("target", validation=validation).train()

    found = model.errors()

    assert model.out_of_fold is not None
    assert found.source == "out-of-fold predictions"
    assert found.rows == len(model.out_of_fold.rows)
    assert sorted(cast(ea.HeldOut, model.held_out).rows) == sorted(model.out_of_fold.rows)
