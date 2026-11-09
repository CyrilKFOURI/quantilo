"""019 — the model catalogue, settings levels and rules, validation methods."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression

from quantilo.ml import catalog, evaluate
from quantilo.ml.catalog.base import Context, SettingsError, resolve
from quantilo.ml.catalog.other import neighbours
from quantilo.ml.train import TrainingError, train
from tests.test_ml import FIELDS, FRAME, by_name

SMALL = FRAME.iloc[:400].reset_index(drop=True)
PRICE, EXPENSIVE = by_name["price"], by_name["expensive"]
FEATURES = [f for f in FIELDS if f.name not in ("price_copy",)]


def settings_of(key: str, context: Context) -> dict:
    return {s.key: s.default for s in catalog.CATALOGUE[key].settings(context)}


# --- catalogue -----------------------------------------------------------------------------------


@pytest.mark.parametrize("key", [k for k, s in catalog.CATALOGUE.items() if s.installed])
def test_every_installed_model_trains_on_its_problems(key: str) -> None:
    spec = catalog.CATALOGUE[key]
    for task, target in (("regression", PRICE), ("binary", EXPENSIVE)):
        if task not in spec.tasks:
            continue
        fields = [
            f for f in FEATURES if f.name != ("expensive" if task == "regression" else "price")
        ]
        result = train(SMALL, fields, target, task, key, None, len(SMALL), False)
        assert result.score is not None and np.isfinite(result.score), (key, task)
        assert result.scores, key
        if task == "binary":
            assert result.classification is not None and result.classification.roc_auc is not None


def test_optional_packages_are_listed_but_refused_when_missing() -> None:
    missing = [s for s in catalog.CATALOGUE.values() if not s.installed]
    for spec in missing:
        assert spec.package in ("xgboost", "lightgbm", "catboost")
        with pytest.raises(TrainingError, match="not installed"):
            train(SMALL, FEATURES, PRICE, "regression", spec.key, None, len(SMALL), False)


def test_families_cover_the_catalogue() -> None:
    assert {s.family for s in catalog.CATALOGUE.values()} <= set(catalog.FAMILIES)
    assert len(catalog.CATALOGUE) >= 35


# --- adaptive defaults, rules, limits -----------------------------------------------------------


def test_adaptive_defaults() -> None:
    assert settings_of("xgboost", Context(3_000, "regression"))["max_depth"] == 4
    assert settings_of("xgboost", Context(10_000, "regression"))["max_depth"] == 6
    assert settings_of("xgboost", Context(500_000, "regression"))["learning_rate"] == 0.05
    assert neighbours(Context(10_000, "regression")) == 50
    assert neighbours(Context(10_000, "binary")) == 49  # odd for two classes
    assert neighbours(Context(400, "binary")) == 21
    assert settings_of("random_forest", Context(200_000, "regression"))["min_samples_leaf"] == 5


def test_rules_pick_an_allowed_default_and_refuse_an_impossible_choice() -> None:
    logistic = catalog.CATALOGUE["logistic_regression"]
    context = Context(1_000, "binary", 0.5)

    assert resolve(logistic, context, {"penalty": "l1"})["solver"] in ("saga", "liblinear")
    assert resolve(logistic, context, {"penalty": "elasticnet"})["solver"] == "saga"
    with pytest.raises(SettingsError, match="L1 penalty needs"):
        resolve(logistic, context, {"penalty": "l1", "solver": "lbfgs"})
    mlp = resolve(catalog.CATALOGUE["mlp"], context, {"solver": "lbfgs"})
    assert mlp["early_stopping"] is False
    with pytest.raises(SettingsError, match="whole number"):
        resolve(catalog.CATALOGUE["knn"], context, {"n_neighbors": 2.5})


def test_settings_have_a_level_a_group_and_shown_when_dependencies() -> None:
    mlp = {s.key: s for s in catalog.CATALOGUE["mlp"].settings(Context(1_000, "binary"))}
    assert mlp["momentum"].shown_when == {"solver": ["sgd"]}
    assert mlp["beta_1"].shown_when == {"solver": ["adam"]}
    assert mlp["solver"].level == "expert" and mlp["hidden_layer_sizes"].level == "essential"
    essentials = [
        s
        for s in catalog.CATALOGUE["random_forest"].settings(Context(1_000, "binary"))
        if s.level == "essential"
    ]
    assert 2 <= len(essentials) <= 4


def test_slow_models_are_limited_on_large_data() -> None:
    gp = catalog.CATALOGUE["gaussian_process"]
    assert gp.limit(Context(6_000, "regression")) is not None
    assert gp.limit(Context(1_000, "regression")) is None
    assert "Slow" in " ".join(catalog.CATALOGUE["kernel_svm"].badges(Context(20_000, "binary")))


def test_recommendations() -> None:
    small = dict(catalog.recommend(Context(500, "regression")))
    assert set(small) == {"linear_regression", "random_forest"}
    large = dict(catalog.recommend(Context(500_000, "binary")))
    assert "sgd" in large and "kernel_svm" not in large
    assert any("boosting" in key or key in ("xgboost", "lightgbm") for key in large)
    messy = catalog.recommend(Context(20_000, "binary", 0.3, 20, 0.5, 0.1))
    assert "categories" in messy[0][1]


# --- exactness ---------------------------------------------------------------------------------


def test_results_match_a_direct_library_call() -> None:
    """With cross-validation the saved model is refitted on every row: refitting the library
    directly on the same prepared matrix with the same settings gives the same predictions."""
    folds = evaluate.Validation(method="kfold", folds=3)
    forest = train(
        SMALL,
        FEATURES,
        PRICE,
        "regression",
        "random_forest",
        {"n_estimators": 40, "random_state": 7},
        1,
        False,
        validation=folds,
    )
    x = SMALL[[d.column for d in forest.preparation.used]]
    prepared = forest.pipeline[:-1].transform(x)
    direct = RandomForestRegressor(n_estimators=40, max_features=1.0, n_jobs=-1, random_state=7)
    direct.fit(prepared, SMALL["price"])
    assert np.allclose(direct.predict(prepared), forest.pipeline.predict(x))
    assert forest.importances and abs(sum(i.importance for i in forest.importances) - 1) < 1e-6

    # no target-encoded column here: its training values are cross-fitted, not transform()
    fields = [f for f in FEATURES if f.name not in ("price", "many")]
    logistic = train(
        SMALL,
        fields,
        EXPENSIVE,
        "binary",
        "logistic_regression",
        {"C": 0.3},
        1,
        False,
        validation=folds,
    )
    x = SMALL[[d.column for d in logistic.preparation.used]]
    prepared = logistic.pipeline[:-1].transform(x)
    direct = LogisticRegression(C=0.3, max_iter=1_000, random_state=42)
    direct.fit(prepared, SMALL["expensive"])
    assert np.allclose(direct.predict_proba(prepared), logistic.pipeline.predict_proba(x))
    assert logistic.coefficients and logistic.coefficients[0].odds_ratio is not None


# --- validation -----------------------------------------------------------------------------------


def test_time_split_never_tests_on_earlier_rows() -> None:
    when = (
        pd.Series(pd.date_range("2024-01-01", periods=100, freq="D"))
        .sample(frac=1, random_state=1)
        .reset_index(drop=True)
    )

    train_at, test_at = evaluate.time_split(when, 0.2)

    assert len(test_at) == 20 and when.iloc[train_at].max() < when.iloc[test_at].min()


def test_group_folds_never_share_a_group() -> None:
    groups = pd.Series([f"g{i % 17}" for i in range(340)])
    y = pd.Series(np.random.default_rng(0).choice(["a", "b"], 340))
    validation = evaluate.Validation(method="group_kfold", group="g", folds=5)

    for fit_at, held_at in evaluate.folds(validation, y, groups, True):
        assert not set(groups.iloc[fit_at]) & set(groups.iloc[held_at])


def test_cross_validation_scores_out_of_fold_and_refits_on_everything() -> None:
    validation = evaluate.Validation(method="kfold", folds=4, metric="mae")

    result = train(
        SMALL,
        FEATURES,
        PRICE,
        "regression",
        "linear_regression",
        None,
        len(SMALL),
        False,
        validation=validation,
    )

    cv = result.cross_validation
    assert cv is not None and cv.folds == 4 and cv.mean["mae"] is not None
    assert cv.std["mae"] is not None and result.primary == "mae" and result.score == cv.mean["mae"]
    assert result.test_rows == len(SMALL) and result.train_rows == len(SMALL)


def test_time_split_and_bad_validation() -> None:
    validation = evaluate.Validation(method="time_split", time_column="when", test_size=0.25)
    result = train(
        SMALL,
        FEATURES,
        PRICE,
        "regression",
        "linear_regression",
        None,
        1,
        False,
        validation=validation,
    )
    assert result.test_rows == 100

    with pytest.raises(TrainingError, match="not a metric"):
        train(
            SMALL,
            FEATURES,
            PRICE,
            "regression",
            "linear_regression",
            None,
            1,
            False,
            validation=evaluate.Validation(metric="accuracy"),
        )
    with pytest.raises(TrainingError, match="Leave-one-out"):
        train(
            FRAME,
            FEATURES,
            PRICE,
            "regression",
            "linear_regression",
            None,
            1,
            False,
            validation=evaluate.Validation(method="loo"),
        )


def test_every_metric_is_computed() -> None:
    fields = [f for f in FEATURES if f.name != "price"]
    result = train(SMALL, fields, EXPENSIVE, "binary", "logistic_regression", None, 1, False)
    assert set(result.scores) == set(evaluate.CLASSIFICATION_METRICS)
    assert all(result.scores[m] is not None for m in ("mcc", "kappa", "pr_auc", "brier"))
