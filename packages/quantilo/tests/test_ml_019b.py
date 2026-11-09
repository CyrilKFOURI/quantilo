"""019b — cancellation, duration estimate, learning curves, monotonic constraints, decision
threshold, radius neighbours and categorical Naive Bayes."""

import numpy as np
import pandas as pd
import pytest

from quantilo.ml import evaluate
from quantilo.ml.catalog.other import RadiusRegressor
from quantilo.ml.train import TrainingCancelled, TrainingError, estimate, train
from tests.test_ml import FIELDS, FRAME, by_name

PRICE, EXPENSIVE = by_name["price"], by_name["expensive"]
FEATURES = [f for f in FIELDS if f.name not in ("price_copy", "expensive")]
CLASS_FIELDS = [f for f in FIELDS if f.name not in ("price_copy", "price")]


def test_a_training_stops_at_the_next_checkpoint() -> None:
    with pytest.raises(TrainingCancelled):
        train(
            FRAME,
            FEATURES,
            PRICE,
            "regression",
            "random_forest",
            None,
            1,
            False,
            should_stop=lambda: True,
        )


@pytest.mark.parametrize("model", ["hist_gradient_boosting", "xgboost", "lightgbm"])
def test_monotonic_constraints_hold(model: str) -> None:
    result = train(
        FRAME, FEATURES, PRICE, "regression", model, {"monotonic": "size:+1, skewed:-1"}, 1, False
    )
    used = [str(c) for c in result.pipeline.feature_names_in_]
    probe = pd.concat([FRAME[used].iloc[[0]]] * 40, ignore_index=True)
    probe["size"] = np.linspace(40, 160, 40)
    rising = result.pipeline.predict(probe)
    assert np.all(np.diff(rising) >= -1e-6)
    probe = pd.concat([FRAME[used].iloc[[1]]] * 40, ignore_index=True)
    probe["skewed"] = np.linspace(1, 200, 40)
    assert np.all(np.diff(result.pipeline.predict(probe)) <= 1e-6)


def test_monotonic_text_is_checked() -> None:
    with pytest.raises(TrainingError, match="column:\\+1"):
        train(
            FRAME,
            FEATURES,
            PRICE,
            "regression",
            "hist_gradient_boosting",
            {"monotonic": "size up"},
            1,
            False,
        )
    result = train(
        FRAME,
        FEATURES,
        PRICE,
        "regression",
        "hist_gradient_boosting",
        {"monotonic": "city:+1"},
        1,
        False,
    )
    assert any("city ignored" in w for w in result.preparation.warnings)


def test_the_decision_threshold_is_chosen_on_training_rows() -> None:
    frame = FRAME.copy()
    frame["expensive"] = np.where(frame["price"] > np.quantile(frame["price"], 0.85), "yes", "no")
    validation = evaluate.Validation(threshold="f1")

    result = train(
        frame,
        CLASS_FIELDS,
        EXPENSIVE,
        "binary",
        "logistic_regression",
        None,
        1,
        False,
        validation=validation,
    )

    c = result.classification
    assert c is not None and c.threshold is not None and 0.02 <= c.threshold <= 0.98
    assert c.positive == "yes"
    default = train(frame, CLASS_FIELDS, EXPENSIVE, "binary", "logistic_regression", None, 1, False)
    assert default.classification is not None and default.classification.threshold == 0.5
    with pytest.raises(TrainingError, match="cross-validation"):
        train(
            frame,
            CLASS_FIELDS,
            EXPENSIVE,
            "binary",
            "logistic_regression",
            None,
            1,
            False,
            validation=evaluate.Validation(method="kfold", threshold="f1"),
        )


@pytest.mark.parametrize(
    ("model", "has_validation"),
    [
        ("hist_gradient_boosting", True),
        ("mlp", True),
        ("gradient_boosting", False),
        ("xgboost", True),
    ],
)
def test_iterative_models_keep_their_learning_curve(model: str, has_validation: bool) -> None:
    result = train(FRAME, FEATURES, PRICE, "regression", model, None, 1, False)

    curve = result.learning_curve
    assert curve is not None and len(curve.iterations) > 1
    assert (curve.validation is not None) == has_validation
    assert (
        train(
            FRAME, FEATURES, PRICE, "regression", "linear_regression", None, 1, False
        ).learning_curve
        is None
    )


def test_duration_estimate_counts_every_fit() -> None:
    split = estimate(FRAME, FEATURES, PRICE, "regression", "random_forest", {"n_estimators": 20})
    assert split.seconds > 0 and split.fits == 1 and split.rows == 960

    folds = estimate(
        FRAME,
        FEATURES,
        PRICE,
        "regression",
        "random_forest",
        {"n_estimators": 20},
        validation=evaluate.Validation(method="kfold", folds=4),
    )
    assert folds.fits == 5 and folds.rows == len(FRAME)


def test_radius_neighbours_fall_back_to_the_mean_and_categorical_nb_trains() -> None:
    model = RadiusRegressor(radius=0.1).fit(np.array([[0.0], [1.0]]), np.array([10.0, 20.0]))
    assert model.predict(np.array([[100.0]]))[0] == pytest.approx(15.0)

    result = train(FRAME, CLASS_FIELDS, EXPENSIVE, "binary", "categorical_nb", None, 1, False)
    assert result.classification is not None and (result.classification.accuracy or 0) > 0.6
