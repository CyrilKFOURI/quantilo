"""018 — a number predicted as ordered classes."""

import duckdb
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from quantilo.ml import classes as binning
from quantilo.ml.data import Field
from quantilo.ml.train import train
from tests.test_ml import FIELDS, FRAME, RAW, by_name

PRICE = by_name["price"]


def three(edges: list[float]) -> binning.Classes:
    return binning.Classes(method="custom", edges=edges, names=["Low", "Medium", "High"])


def test_limits_are_validated() -> None:
    with pytest.raises(ValidationError, match="One more name"):
        binning.Classes(edges=[1, 2], names=["a", "b"])
    with pytest.raises(ValidationError, match="increase"):
        binning.Classes(edges=[2, 1], names=["a", "b", "c"])
    with pytest.raises(ValidationError, match="distinct"):
        binning.Classes(edges=[1], names=["a", "a"])


def test_intervals_include_the_upper_limit() -> None:
    labels = binning.apply(pd.Series([0.0, 10.0, 10.5, 20.0, 99.0, np.nan]), three([10, 20]))

    assert labels.tolist()[:5] == ["Low", "Low", "Medium", "Medium", "High"]
    assert pd.isna(labels.iloc[5])


def test_training_on_classes_is_an_ordered_classification() -> None:
    edges = list(np.quantile(FRAME["price"], [1 / 3, 2 / 3]))

    result = train(
        FRAME,
        FIELDS,
        PRICE,
        "regression",
        "logistic_regression",
        None,
        len(FRAME),
        False,
        classes=three(edges),
    )

    assert result.task == "multiclass" and result.classification is not None
    assert result.classification.labels == ["Low", "Medium", "High"]  # in the class order
    assert result.classes is not None and sum(result.classes.counts) == len(FRAME)
    assert result.classification.within_one is not None
    assert result.classification.within_one >= (result.classification.accuracy or 0)
    assert result.classification.mean_class_error is not None
    # one ROC, precision-recall and calibration curve per class, each against the others
    curves = result.classification.class_curves
    assert [c.label for c in curves] == ["Low", "Medium", "High"]
    assert all(c.roc.x[0] == 0 and c.roc.x[-1] == 1 and c.auc is not None for c in curves)
    assert sum(c.share for c in curves) == pytest.approx(1)
    assert result.classification.roc is None  # the single binary curve stays for two classes
    decisions = {d.column: d for d in result.preparation.decisions}
    assert decisions["price"].rules == ["R-SEL-01"]  # the source number is never an input
    assert decisions["price_copy"].rules == ["R-SEL-06"]  # leak measured against the number


def test_two_classes_make_a_binary_problem_and_empty_classes_are_dropped() -> None:
    median = float(np.median(FRAME["price"]))
    top = float(np.max(FRAME["price"].to_numpy())) + 1
    classes = binning.Classes(edges=[median, top], names=["Cheap", "Dear", "Never"])

    result = train(
        FRAME,
        FIELDS,
        PRICE,
        "regression",
        "logistic_regression",
        None,
        len(FRAME),
        False,
        classes=classes,
    )

    assert result.task == "binary"
    assert result.classes is not None and result.classes.counts[2] == 0


def test_distribution_gives_quantiles_histogram_sample_and_values() -> None:
    rng = np.random.default_rng(2)
    values = np.round(rng.uniform(0, 100, 3_000), 2)
    frame = pd.DataFrame({"v": values})
    connection = duckdb.connect()
    connection.register("t", frame)
    source = "(SELECT *, row_number() OVER () - 1 AS file_row_number FROM t)"

    found = binning.distribution(connection, source, Field("v", "numeric_continuous", RAW))

    assert found.rows == 3_000 and len(found.quantiles) == 1_001
    assert found.quantiles[0] == pytest.approx(values.min())
    assert found.quantiles[500] == pytest.approx(np.quantile(values, 0.5, method="lower"), abs=0.1)
    assert sum(found.histogram) == 3_000 and len(found.histogram) == 80
    assert len(found.sample) == 3_000 and found.sample == sorted(found.sample)
    assert found.values is None
    assert len(found.equal_counts) == 9 and len(found.equal_counts[1]) == 2
    low, high = found.equal_counts[1]  # 3 classes
    counts = [
        (values <= low).sum(),
        ((values > low) & (values <= high)).sum(),
        (values > high).sum(),
    ]
    assert all(abs(c - 1_000) <= 1 for c in counts)

    small = pd.DataFrame({"v": [1.0, 2.0, 2.0, 3.0, 3.0, 3.0]})
    connection.register("s", small)
    few = binning.distribution(
        connection,
        "(SELECT *, row_number() OVER () - 1 AS file_row_number FROM s)",
        Field("v", "numeric_discrete", RAW),
    )
    assert few.values is not None
    assert [(v.value, v.count) for v in few.values] == [(1, 1), (2, 2), (3, 3)]


def test_a_class_too_small_to_learn_is_refused_clearly() -> None:
    from quantilo.ml.train import TrainingError

    values = np.sort(FRAME["price"].to_numpy())
    classes = binning.Classes(edges=[values[600], values[601]], names=["L", "M", "H"])

    with pytest.raises(TrainingError, match="“M” has only 1 row "):
        train(
            FRAME,
            FIELDS,
            PRICE,
            "regression",
            "logistic_regression",
            None,
            1,
            False,
            classes=classes,
        )
