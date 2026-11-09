"""017 — the user's preparation choices on top of the rules, the steps they add, the preview."""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from quantilo.ml import models as catalogue
from quantilo.ml import preparation as prep
from quantilo.ml import steps
from quantilo.ml.data import Field
from quantilo.ml.train import _setup, preview, train
from tests.test_ml import FIELDS, FRAME, RAW, by_name

PRICE = by_name["price"]


def decision(preparation: prep.Preparation, column: str) -> prep.Decision:
    return next(d for d in preparation.decisions if d.column == column)


def choose(**columns: dict) -> prep.Choices:
    return prep.Choices(columns={k: prep.ColumnChoice(**v) for k, v in columns.items()})


def linear(choices: prep.Choices | None = None) -> prep.Preparation:
    return prep.decide(FRAME, FIELDS, PRICE, "regression", catalogue.LINEAR, None, choices)


# --- choices on top of the rules -----------------------------------------------------------------


def test_automatic_values_options_and_no_custom_flag() -> None:
    preparation = linear()
    skewed = decision(preparation, "skewed")

    assert not preparation.custom and not skewed.manual
    assert skewed.auto == {
        "use": "yes",
        "impute": "median",
        "indicator": "no",
        "transform": "log1p",
        "encoding": "none",
        "scaling": skewed.auto["scaling"],
    }
    none_fill = next(o for o in skewed.options["impute"] if o.value == "none")
    assert none_fill.reason and "cannot handle missing" in none_fill.reason


def test_a_choice_replaces_the_rule_and_is_marked_manual() -> None:
    preparation = linear(choose(skewed={"transform": "none", "scaling": "minmax"}))
    skewed = decision(preparation, "skewed")

    assert skewed.transform is None and skewed.scaling == "minmax"
    assert skewed.auto["transform"] == "log1p"  # what the rules would have done
    assert skewed.manual == ["transform", "scaling"] and preparation.custom
    assert skewed.rules[-1] == "USER"


def test_impossible_choices_are_refused_with_the_reason() -> None:
    frame = FRAME.copy()
    frame["delta"] = np.random.default_rng(1).normal(0, 1, len(frame))
    fields = [*FIELDS, Field("delta", "numeric_continuous", RAW)]
    choices = choose(
        delta={"transform": "log1p"},  # negative values
        size={"impute": "none"},  # linear regression needs filled values
        city={"encoding": "binary"},  # four categories
        skewed={"encoding": "weird"},
    )

    preparation = prep.decide(frame, fields, PRICE, "regression", catalogue.LINEAR, None, choices)

    assert decision(preparation, "delta").transform is None
    assert decision(preparation, "size").impute == "mean"  # symmetric: R-NUM-02
    assert decision(preparation, "city").encoding == "onehot"
    assert len(preparation.warnings) == 4
    assert any("≥ 0" in w for w in preparation.warnings)
    assert any("cannot handle missing" in w for w in preparation.warnings)


def test_excluded_columns_can_come_back_except_identifiers_and_the_target() -> None:
    preparation = linear(choose(price_copy={"use": True}, vin={"use": True}, price={"use": True}))

    copy = decision(preparation, "price_copy")
    assert copy.use and copy.leak and copy.manual == ["use"]
    assert copy.rules[:2] == ["R-SEL-06", "USER"] and copy.auto["use"] == "no"
    assert copy.scaling is not None  # numeric rules applied once it is used
    assert not decision(preparation, "vin").use and not decision(preparation, "price").use
    assert len(preparation.warnings) == 2


def test_a_column_can_be_left_out() -> None:
    preparation = linear(choose(city={"use": False}))

    city = decision(preparation, "city")
    assert not city.use and city.manual == ["use"] and city.auto["use"] == "yes"


def test_forced_columns_survive_the_collinearity_rule() -> None:
    frame = FRAME.copy()
    frame["size_twin"] = frame["size"] * 1.01 + np.random.default_rng(3).normal(0, 0.1, len(frame))
    fields = [*FIELDS, Field("size_twin", "numeric_continuous", RAW)]

    automatic = prep.decide(frame, fields, PRICE, "regression", catalogue.LINEAR)
    dropped = next(d.column for d in automatic.decisions if "R-SEL-07" in d.rules)
    forced = prep.decide(
        frame,
        fields,
        PRICE,
        "regression",
        catalogue.LINEAR,
        None,
        choose(**{dropped: {"use": True}}),
    )

    assert decision(forced, dropped).use
    assert sum("R-SEL-07" in d.rules for d in forced.decisions) == 1  # the other one goes


def test_global_settings_feed_the_rules() -> None:
    preparation = linear(
        prep.Choices(numeric_impute="knn", one_hot_limit=3, indicator_share=0.5, target_smoothing=2)
    )

    size = decision(preparation, "size")
    assert size.impute == "knn" and "R-NUM-13" in size.rules and not size.indicator
    assert decision(preparation, "city").encoding == "target"  # 4 categories > limit 3
    assert "smoothing 2" in decision(preparation, "many").why[-1]
    assert preparation.custom


# --- steps ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["knn", "iterative"])
def test_joint_imputation_fills_from_the_other_columns(kind: str) -> None:
    rng = np.random.default_rng(5)
    a = rng.normal(0, 1, 400)
    frame = pd.DataFrame({"a": a, "b": 2 * a + rng.normal(0, 0.05, 400)})
    gaps = frame.copy()
    gaps.loc[:39, "b"] = np.nan

    filled = steps.JointImputer(("a", "b"), kind).fit(gaps).transform(gaps)

    error = np.abs(filled.loc[:39, "b"] - frame.loc[:39, "b"]).mean()
    assert error < 0.3 and not filled.isna().any().any()


def test_clipping_limits_are_learnt_on_training_rows() -> None:
    values = np.arange(100, dtype=float).reshape(-1, 1)

    clipper = steps.Clipper("clip_percentiles").fit(values)
    assert clipper.transform(np.array([[-50.0], [500.0]])).ravel().tolist() == pytest.approx(
        [0.99, 98.01]
    )
    iqr = steps.Clipper("clip_iqr").fit(values)
    assert iqr.high_[0] == pytest.approx(74.25 + 1.5 * 49.5)


def test_extreme_rows_and_rebalancing() -> None:
    frame = pd.DataFrame({"x": [*range(20), 1_000.0]})
    assert steps.extreme_rows(frame, ["x"]).tolist() == [False] * 20 + [True]

    x = pd.DataFrame({"v": range(10)})
    y = pd.Series(["a"] * 8 + ["b"] * 2)
    up_x, up_y = steps.rebalance(x, y, "oversample")
    down_x, down_y = steps.rebalance(x, y, "undersample")
    assert up_y.value_counts().to_dict() == {"a": 8, "b": 8} and len(up_x) == 16
    assert down_y.value_counts().to_dict() == {"a": 2, "b": 2} and len(down_x) == 4


@pytest.mark.parametrize("method", list(steps.SELECTION_DEFAULTS))
def test_every_selection_method_keeps_the_informative_inputs(method: str) -> None:
    rng = np.random.default_rng(11)
    signal = rng.normal(0, 1, (600, 2))
    noise = rng.normal(0, 1, (600, 8))
    x = np.column_stack([signal, signal[:, 0] * 1.0001 + rng.normal(0, 1e-3, 600), noise])
    y = 3 * signal[:, 0] - 2 * signal[:, 1] + rng.normal(0, 0.5, 600)
    value = {"mutual_info": 3, "variance": 0.5}.get(method)

    selector = steps.Selector(method, value, "regression").fit(x, y)

    kept = set(np.flatnonzero(selector.support_))
    assert kept and (0 in kept or 2 in kept)
    if method in ("univariate", "l1", "rfe", "permutation", "mutual_info"):
        assert 1 in kept
    if method in ("correlation", "vif"):
        assert not {0, 2} <= kept  # one of the twins goes
    names = selector.get_feature_names_out(np.array([f"f{i}" for i in range(11)], dtype=object))
    assert len(names) == len(kept)


def test_selection_runs_in_the_pipeline_and_is_reported() -> None:
    choices = prep.Choices(selection=[prep.SelectionChoice(method="univariate", value=0.01)])

    result = train(
        FRAME,
        FIELDS,
        PRICE,
        "regression",
        "linear_regression",
        None,
        len(FRAME),
        False,
        choices=choices,
    )

    report = result.preparation.selection[0]
    assert report.method == "univariate" and report.kept < report.of
    assert result.preparation.inputs == report.kept
    assert {c.feature for c in result.coefficients} <= set(
        result.pipeline[:-1].get_feature_names_out()
    )


# --- imbalance -----------------------------------------------------------------------------------


def _imbalanced() -> tuple[pd.DataFrame, Field]:
    frame = FRAME.copy()
    cut = np.quantile(frame["price"], 0.9)
    frame["rare"] = np.where(frame["price"] > cut, "yes", "no")
    return frame, Field("rare", "categorical_nominal", RAW)


@pytest.mark.parametrize(
    ("strategy", "expected"),
    [
        ("auto", "weights"),
        ("none", "none"),
        ("oversample", "oversample"),
        ("undersample", "undersample"),
    ],
)
def test_imbalance_strategies(strategy: str, expected: str) -> None:
    frame, rare = _imbalanced()
    fields = [f for f in FIELDS if f.name not in ("price_copy", "expensive")] + [rare]

    setup = _setup(
        frame,
        fields,
        rare,
        "binary",
        "logistic_regression",
        None,
        None,
        prep.Choices(imbalance=strategy),  # type: ignore[arg-type]
    )

    balance = setup.preparation.balance
    assert balance is not None and balance.strategy == expected
    assert balance.before == {"no": 864, "yes": 96}
    if expected == "weights":
        assert balance.rule == "R-IMB-02" and setup.values["class_weight"] == "balanced"
    if expected == "oversample":
        assert balance.after == {"no": 864, "yes": 864} and setup.values["class_weight"] == "none"
    if expected == "undersample":
        assert balance.after == {"no": 96, "yes": 96}
    assert len(setup.x_test) == 240  # the test rows are never touched


def test_drop_rows_only_touches_training_rows() -> None:
    frame = FRAME.copy()
    frame.loc[:9, "skewed"] = 1e7

    setup = _setup(
        frame,
        FIELDS,
        PRICE,
        "regression",
        "linear_regression",
        None,
        None,
        prep.Choices(outliers="drop_rows"),
    )

    assert setup.preparation.dropped_rows > 0
    assert len(setup.x_test) == 240
    assert len(setup.x_train) == 960 - setup.preparation.dropped_rows


# --- preview -------------------------------------------------------------------------------------


def test_preview_equals_fit_on_train_then_transform() -> None:
    choices = prep.Choices(outliers="clip_percentiles", numeric_impute="iterative")

    shown = preview(FRAME, FIELDS, PRICE, "regression", "linear_regression", None, choices)

    setup = _setup(FRAME, FIELDS, PRICE, "regression", "linear_regression", None, None, choices)
    pipeline = Pipeline(prep.steps(setup.preparation, len(setup.x_train), "regression", True))
    pipeline.fit(setup.x_train, setup.y_train)
    expected = np.asarray(pipeline.transform(setup.x_train.iloc[:20]), dtype=float)
    assert np.allclose(np.array(shown.rows, dtype=float), expected, equal_nan=True)
    assert shown.columns == [str(n) for n in pipeline.get_feature_names_out()]  # type: ignore[union-attr]
    assert len(shown.rows) == 20 and len(shown.row_numbers) == 20
    assert shown.preparation.inputs == len(shown.columns) and shown.train_rows == 960


def test_statistics_are_learnt_on_the_training_rows_only() -> None:
    setup = _setup(FRAME, FIELDS, PRICE, "regression", "linear_regression", None, None, None)
    pipeline = Pipeline(prep.steps(setup.preparation, len(setup.x_train), "regression"))
    pipeline.fit(setup.x_train, setup.y_train)

    imputer = pipeline.named_steps["prepare"].named_transformers_["size"].named_steps["impute"]
    assert imputer.statistics_[0] == pytest.approx(setup.x_train["size"].mean())
    assert imputer.statistics_[0] != pytest.approx(FRAME["size"].mean(), abs=1e-9)


def test_choices_are_saved_with_the_run_and_reproduce_it() -> None:
    choices = choose(skewed={"transform": "none"})
    first = train(
        FRAME, FIELDS, PRICE, "regression", "linear_regression", None, 1, False, choices=choices
    )
    again = train(
        FRAME,
        FIELDS,
        PRICE,
        "regression",
        "linear_regression",
        None,
        1,
        False,
        choices=first.preparation.choices,
    )

    assert first.preparation.choices == choices
    assert first.regression is not None and again.regression is not None
    assert first.regression.r2 == again.regression.r2


def test_a_number_that_duplicates_a_date_is_dropped_for_plain_linear_models() -> None:
    frame = FRAME.copy()
    days = (frame["when"] - frame["when"].min()).dt.days
    frame["age_days"] = 400 - days + np.random.default_rng(2).normal(0, 0.5, len(frame))
    fields = [*FIELDS, Field("age_days", "numeric_continuous", RAW)]

    found = prep.decide(frame, fields, PRICE, "regression", catalogue.LINEAR)

    dropped = [d.column for d in found.decisions if "R-SEL-07" in d.rules]
    assert len(dropped) == 1 and dropped[0] in ("when", "age_days")
