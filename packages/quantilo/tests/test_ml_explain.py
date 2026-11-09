"""020 — explanations: permutation importance, SHAP, partial dependence, tree, linked inputs."""

from typing import cast

import numpy as np
import pandas as pd
import pytest

from quantilo.ml import evaluate
from quantilo.ml import explain as explainer
from quantilo.ml.data import Field
from quantilo.ml.target import Task
from quantilo.ml.train import train
from tests.test_ml import FIELDS, FRAME, RAW, by_name

PRICE, EXPENSIVE = by_name["price"], by_name["expensive"]
FEATURES = [f for f in FIELDS if f.name not in ("price_copy", "expensive")]
FOLDS = evaluate.Validation(method="kfold", folds=3)  # explained on every row


def run_and_explain(model: str, task: Task = "regression", settings=None, frame=FRAME, fields=None):
    target = PRICE if task == "regression" else EXPENSIVE
    if fields is None:
        fields = (
            FEATURES
            if task == "regression"
            else [f for f in FIELDS if f.name not in ("price_copy", "price")]
        )
    result = train(frame, fields, target, task, model, settings, 1, False, validation=FOLDS)
    x = cast(pd.DataFrame, frame[[str(c) for c in result.pipeline.feature_names_in_]])
    y = cast(pd.Series, frame[target.name]).astype(float if task == "regression" else str)
    rows = frame["__row"].to_numpy(dtype=int)
    found = explainer.explain(
        result.pipeline, x, y, rows, task, result.primary, [p.row for p in result.predictions], None
    )
    return result, found, x


def test_linear_shap_values_add_up_to_the_prediction_and_group_one_hot_columns() -> None:
    result, found, x = run_and_explain("linear_regression")

    shap = found.shap
    assert shap is not None and shap.method.startswith("Exact") and shap.scale == "prediction"
    assert "city" in shap.columns and not any(c.startswith("city_") for c in shap.columns)
    positions = [int(np.flatnonzero(FRAME["__row"].to_numpy() == r)[0]) for r in shap.rows]
    predicted = result.pipeline.predict(x.iloc[positions])
    totals = shap.base + np.array(shap.matrix).sum(axis=1)
    assert np.allclose(totals, predicted, atol=1e-5)
    assert set(p.row for p in result.predictions) <= set(shap.rows)  # every shown row explained
    size = next(f for f in shap.features if f.column == "size")
    assert size.reading == "Higher size pushes the prediction up."


def test_logistic_shap_is_in_log_odds() -> None:
    result, found, x = run_and_explain("logistic_regression", "binary")

    shap = found.shap
    assert shap is not None and shap.scale == "log-odds"
    assert shap.positive == str(result.pipeline.classes_[1])
    positions = [int(np.flatnonzero(FRAME["__row"].to_numpy() == r)[0]) for r in shap.rows[:50]]
    margin = result.pipeline.decision_function(x.iloc[positions])
    totals = shap.base + np.array(shap.matrix[:50]).sum(axis=1)
    assert np.allclose(totals, margin, atol=1e-5)


def test_permutation_importance_and_partial_dependence() -> None:
    _, found, _ = run_and_explain("linear_regression")

    permutation = found.permutation
    assert permutation is not None and permutation.metric == "r2" and not permutation.sampled
    assert permutation.items[0].column == "size"  # price = 3 * size + ...
    assert all(item.std >= 0 for item in permutation.items)
    size = next(d for d in found.dependence if d.column == "size")
    curve = size.curves["prediction"]
    assert size.kind == "number" and len(curve) == len(size.grid) <= 30
    assert curve[-1] > curve[0] and sum(size.counts) > 0
    city = next((d for d in found.dependence if d.column == "city"), None)
    if city is not None:
        assert city.kind == "category" and "Paris" in city.grid


def test_models_without_shap_say_why_and_trees_show_themselves() -> None:
    _, forest, _ = run_and_explain("random_forest", settings={"n_estimators": 30})
    assert (
        forest.shap is None
        and forest.shap_unavailable
        and "RandomForest" in forest.shap_unavailable
    )
    assert forest.permutation is not None and forest.dependence

    _, tree, _ = run_and_explain("decision_tree")
    assert tree.tree is not None and tree.tree[0].feature is not None
    assert max(node.depth for node in tree.tree) <= explainer.TREE_DEPTH


def test_strongly_linked_inputs_are_flagged() -> None:
    frame = FRAME.copy()
    frame["size_twin"] = frame["size"] * 2 + np.random.default_rng(1).normal(0, 0.5, len(frame))
    fields = [*FEATURES, Field("size_twin", "numeric_continuous", RAW)]

    _, found, _ = run_and_explain(
        "random_forest", settings={"n_estimators": 20}, frame=frame, fields=fields
    )

    assert any(
        {p.first, p.second} == {"size", "size_twin"} and p.measure == "|r|" for p in found.pairs
    )


@pytest.mark.parametrize("library", ["xgboost", "lightgbm"])
def test_boosting_native_shap_adds_up(library: str) -> None:
    pytest.importorskip(library)
    result, found, x = run_and_explain(library, settings={"n_estimators": 60})
    shap = found.shap
    assert shap is not None
    positions = [int(np.flatnonzero(FRAME["__row"].to_numpy() == r)[0]) for r in shap.rows]
    predicted = result.pipeline.predict(x.iloc[positions])
    # float32 inside the libraries: equal to 1e-5 relative
    assert np.allclose(
        shap.base + np.array(shap.matrix).sum(axis=1), predicted, rtol=1e-5, atol=1e-3
    )


def test_origins_map_prepared_inputs_to_columns() -> None:
    result = train(FRAME, FEATURES, PRICE, "regression", "linear_regression", None, 1, False)
    owners = explainer.origins(result.pipeline)
    names = list(result.pipeline[:-1].get_feature_names_out())
    assert len(owners) == len(names)
    assert {owners[i] for i, n in enumerate(names) if str(n).startswith("when_")} == {"when"}
    assert pd.Series(owners).nunique() < len(owners)


def test_dates_read_as_time() -> None:
    raw = pd.Series(pd.date_range("2024-01-01", periods=50, freq="D"))
    phi = np.linspace(-1, 1, 50)

    assert explainer._reading("when", phi, raw) == "Later when pushes the prediction up."


def test_a_date_and_a_number_carrying_the_same_information_are_linked() -> None:
    born = pd.Series(pd.date_range("1940-01-01", periods=300, freq="45D"))
    x = pd.DataFrame(
        {"birth": born, "age": 2025 - born.dt.year + np.random.default_rng(0).normal(0, 0.3, 300)}
    )

    found = explainer.pairs(x)

    assert found and {found[0].first, found[0].second} == {"birth", "age"}
