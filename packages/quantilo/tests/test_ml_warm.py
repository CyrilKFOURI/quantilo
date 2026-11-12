"""066: warm start of a search from settings already found, by meta-features."""

import math
from typing import Any

import pytest

from quantilo.ml import warm
from quantilo.ml.tune import SearchParam

SPACE = [
    SearchParam(key="max_depth", kind="int", low=2, high=12),
    SearchParam(key="learning_rate", kind="float", low=0.01, high=0.3, log=True),
    SearchParam(key="loss", kind="categorical", choices=["squared_error", "absolute_error"]),
]
HERE = warm.Meta(10_000, 0.2, "number")


def candidate(
    source: warm.Source, score: float, meta: warm.Meta = HERE, **settings: Any
) -> warm.Candidate:
    return warm.Candidate("hgb", settings or {"max_depth": 6}, score, True, meta, source)


def test_distance_on_log_rows_and_category_share() -> None:
    assert warm.distance(HERE, warm.Meta(100_000, 0.2, "number")) == pytest.approx(1.0)
    assert warm.distance(HERE, warm.Meta(10_000, 0.45, "number")) == pytest.approx(0.5)
    assert warm.distance(HERE, warm.Meta(10_000, 0.2, "classes")) == math.inf  # R-ML-WARM-01


def test_this_dataset_first_then_versions_then_the_closest_similar() -> None:
    found = warm.starts(
        "hgb",
        HERE,
        [
            candidate("a similar dataset", 0.99, warm.Meta(20_000, 0.2, "number"), max_depth=9),
            candidate("a cleaned version", 0.80, max_depth=4),
            candidate("this dataset", 0.70, max_depth=3),
            candidate("this dataset", 0.75, max_depth=5),
        ],
        SPACE,
    )
    assert [s.params["max_depth"] for s in found] == [5, 3, 4]  # at most 3 (R-ML-WARM-03)
    assert found[0].source == "this dataset" and "this dataset" in found[0].reason


def test_far_or_other_kind_or_other_model_is_never_used() -> None:
    far = candidate("a similar dataset", 0.9, warm.Meta(1_000_000, 0.2, "number"))
    other_kind = candidate("a similar dataset", 0.9, warm.Meta(10_000, 0.2, "classes"))
    other_model = warm.Candidate("rf", {"max_depth": 6}, 0.9, True, HERE, "this dataset")
    assert warm.starts("hgb", HERE, [far, other_kind, other_model], SPACE) == []


def test_settings_are_brought_back_into_the_space() -> None:
    found = warm.starts(
        "hgb",
        HERE,
        [candidate("this dataset", 0.5, max_depth=40, learning_rate=0.1, loss="huber", other=1)],
        SPACE,
    )
    assert found[0].params == {"max_depth": 12, "learning_rate": 0.1}
    assert found[0].changed == ["max_depth"]


def test_lower_is_better_scores_rank_the_smallest_first() -> None:
    a = warm.Candidate("hgb", {"max_depth": 3}, 10.0, False, HERE, "this dataset")
    b = warm.Candidate("hgb", {"max_depth": 4}, 5.0, False, HERE, "this dataset")
    assert warm.starts("hgb", HERE, [a, b], SPACE)[0].params["max_depth"] == 4


def test_meta_features_from_column_types() -> None:
    types = {
        "y": "numeric_continuous",
        "a": "numeric_continuous",
        "b": "categorical_nominal",
        "c": "boolean",
        "d": "text",
        "e": "numeric_discrete",
    }
    meta = warm.meta_of(types, "y", 500, "regression")
    assert meta == warm.Meta(500, 0.5, "number")
