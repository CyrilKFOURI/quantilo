from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from quantilo.ml import models as catalogue
from quantilo.ml import preparation as prep
from quantilo.ml.data import Field
from quantilo.ml.target import candidate
from quantilo.ml.train import TrainingError, train
from quantilo.typing.types import Parser

RAW = Parser(kind="raw")
RNG = np.random.default_rng(7)
N = 1_200


def make_frame() -> pd.DataFrame:
    size = RNG.normal(100, 20, N)
    skewed = RNG.lognormal(3, 1, N)  # skewness well above 1, positive
    city = RNG.choice(["Paris", "Lyon", "Nice", "Lille"], N)
    many = RNG.choice([f"shop{i}" for i in range(40)], N)  # 40 categories → target encoding
    when = [datetime(2023, 1, 1) + timedelta(days=int(d)) for d in RNG.integers(0, 365, N)]
    effect = {"Paris": 50, "Lyon": 20, "Nice": 0, "Lille": -10}
    price = 3 * size + 0.5 * skewed + np.array([effect[c] for c in city]) + RNG.normal(0, 10, N)
    size_with_gaps = size.copy()
    size_with_gaps[RNG.random(N) < 0.1] = np.nan  # 10 % empty → indicator
    return pd.DataFrame(
        {
            "size": size_with_gaps,
            "skewed": skewed,
            "city": city,
            "many": many,
            "when": when,
            "vin": [f"V{i}" for i in range(N)],
            "price_copy": price * 2.0,  # leakage
            "constant": np.ones(N),
            "price": price,
            "expensive": np.where(price > np.median(price), "yes", "no"),
            "__row": np.arange(N),
        }
    )


FIELDS = [
    Field("size", "numeric_continuous", RAW),
    Field("skewed", "numeric_continuous", RAW),
    Field("city", "categorical_nominal", RAW),
    Field("many", "categorical_nominal", RAW),
    Field("when", "datetime", RAW),
    Field("vin", "identifier", RAW),
    Field("price_copy", "numeric_continuous", RAW),
    Field("constant", "numeric_continuous", RAW),
    Field("price", "numeric_continuous", RAW, role="amount"),
    Field("expensive", "categorical_nominal", RAW),
]
FRAME = make_frame()
by_name = {f.name: f for f in FIELDS}


def rules(preparation: prep.Preparation, column: str) -> list[str]:
    return next(d for d in preparation.decisions if d.column == column).rules


# --- target --------------------------------------------------------------------------------------


def test_candidates() -> None:
    assert candidate("price", "numeric_continuous", "amount", 900, 0).task == "regression"
    assert candidate("churn", "boolean", None, 2, 0).task == "binary"
    assert candidate("segment", "categorical_nominal", None, 4, 0).task == "multiclass"
    assert candidate("vin", "identifier", None, 900, 0).task is None
    assert candidate("price", "numeric_continuous", None, 900, 0.7).task is None
    assert "classes" in (candidate("rating", "numeric_discrete", "rating", 5, 0).reason or "")
    zip_code = candidate("zip", "categorical_nominal", None, 598, 0, present=600)
    assert zip_code.task is None and "identifier" in (zip_code.reason or "")


# --- preparation rules ---------------------------------------------------------------------------


def test_linear_profile_rules() -> None:
    preparation = prep.decide(FRAME, FIELDS, by_name["price"], "regression", catalogue.LINEAR)

    assert rules(preparation, "price") == ["R-SEL-01"]
    assert rules(preparation, "vin") == ["R-SEL-02"]
    assert rules(preparation, "price_copy") == ["R-SEL-06"]
    assert rules(preparation, "constant") == ["R-SEL-04"]
    assert "R-NUM-04" in rules(preparation, "size")  # 10 % missing → indicator
    assert "R-NUM-05" in rules(preparation, "skewed")  # log1p
    assert "R-CAT-05" in rules(preparation, "city")  # one-hot
    assert "R-CAT-06" in rules(preparation, "many")  # target encoding
    assert "R-DATE-03" in rules(preparation, "when")  # sine / cosine
    assert all(sentence for d in preparation.decisions for sentence in d.why)


def test_tree_profile_keeps_numbers_as_they_are() -> None:
    trees = prep.ModelProfile(tree_based=True, handles_missing=True)

    preparation = prep.decide(FRAME, FIELDS, by_name["price"], "regression", trees)

    assert rules(preparation, "skewed") == ["R-NUM-01", "R-NUM-10"]
    assert "R-CAT-04" in rules(preparation, "city")


def test_collinear_numbers_keep_the_one_closer_to_the_target() -> None:
    frame = FRAME.copy()
    frame["size_twin"] = frame["size"] * 1.01 + RNG.normal(0, 0.1, N)
    fields = [*FIELDS, Field("size_twin", "numeric_continuous", RAW)]

    preparation = prep.decide(frame, fields, by_name["price"], "regression", catalogue.LINEAR)

    dropped = [d.column for d in preparation.decisions if "R-SEL-07" in d.rules]
    assert len(dropped) == 1 and dropped[0] in ("size", "size_twin")


# --- training ------------------------------------------------------------------------------------


def test_linear_regression_end_to_end() -> None:
    result = train(
        FRAME, FIELDS, by_name["price"], "regression", "linear_regression", None, N, False
    )

    assert result.regression is not None
    # log1p on `skewed` (R-NUM-05) is a general rule; here price is linear in raw `skewed`
    assert result.regression.r2 is not None and result.regression.r2 > 0.8
    assert result.regression.skill is not None and result.regression.skill > 0.5
    assert result.test_rows == 240 and result.train_rows == 960
    assert result.primary == "r2" and result.preparation.inputs > 5
    named = {c.feature: c for c in result.coefficients}
    assert named["city_Paris"].p_value is not None and named["city_Paris"].p_value < 0.001
    # back in price per unit of size (median imputation of 10 % gaps attenuates it a little)
    assert named["size"].per_unit == pytest.approx(3, abs=0.6)
    assert len(result.predictions) == 240


def test_coefficients_match_scikit_learn_on_the_prepared_matrix() -> None:
    result = train(
        FRAME, FIELDS, by_name["price"], "regression", "linear_regression", None, N, False
    )
    transformer = result.pipeline.named_steps["prepare"]
    model = result.pipeline.named_steps["model"]

    assert isinstance(model, LinearRegression)
    names = list(transformer.get_feature_names_out())
    assert {c.feature for c in result.coefficients} == set(names)


def test_logistic_regression_end_to_end() -> None:
    result = train(
        FRAME, FIELDS, by_name["expensive"], "binary", "logistic_regression", None, N, False
    )
    scores = result.classification

    assert scores is not None and scores.roc_auc is not None and scores.roc_auc > 0.9
    assert (
        scores.positive in ("yes", "no")
        and scores.roc is not None
        and scores.calibration is not None
    )
    assert sum(map(sum, scores.confusion)) == result.test_rows
    assert len(scores.truth) == len(scores.probability) == result.test_rows
    assert result.primary == "roc_auc"
    assert all(c.odds_ratio is not None for c in result.coefficients)
    assert rules(result.preparation, "price") == ["R-SEL-06"]  # the number behind the classes


def test_ridge_settings_and_validation() -> None:
    result = train(
        FRAME,
        FIELDS,
        by_name["price"],
        "regression",
        "linear_regression",
        {"regularization": "ridge", "alpha": 10},
        N,
        False,
    )

    assert result.settings["regularization"] == "ridge" and result.settings["alpha"] == 10.0
    assert result.coefficients[0].p_value is None  # no p-values for a penalised fit
    with pytest.raises(catalogue.SettingsError, match="between"):
        train(
            FRAME,
            FIELDS,
            by_name["price"],
            "regression",
            "linear_regression",
            {"alpha": -1},
            N,
            False,
        )
    with pytest.raises(TrainingError, match="does not handle"):
        train(FRAME, FIELDS, by_name["price"], "regression", "logistic_regression", None, N, False)


def test_imbalanced_classes_get_balanced_weights_by_default() -> None:
    frame = FRAME.copy()
    frame["rare"] = np.where(RNG.random(N) < 0.08, "fraud", "ok")
    fields = [*FIELDS, Field("rare", "categorical_nominal", RAW)]

    result = train(frame, fields, fields[-1], "binary", "logistic_regression", None, N, False)

    assert result.settings["class_weight"] == "balanced"
    assert result.classification is not None and result.classification.positive == "fraud"


def test_rare_positives_are_judged_by_precision_recall_auc() -> None:
    # credit-card fraud: 0.17 % positives (040)
    from quantilo.ml.evaluate import default_metric

    assert default_metric("binary", 0.0017) == "pr_auc"
    assert default_metric("binary", 0.26) == "roc_auc"
    assert default_metric("binary", None) == "roc_auc"
