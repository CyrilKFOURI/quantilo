"""063 — encodings: target-encoding folds aligned with the validation (L1), frequency, cycles and
rare categories (L2), causal smoothed lags for ordered data (L7)."""

import io
from datetime import datetime, timedelta
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from quantilo.ml import encoding as enc
from quantilo.ml import evaluate
from quantilo.ml import preparation as prep
from quantilo.ml.data import Field
from quantilo.ml.train import _setup, train
from quantilo.typing.types import Parser

RAW = Parser(kind="raw")


def panel(periods: int = 30, entities: int = 40, seed: int = 5) -> pd.DataFrame:
    """Entities by eras: a target made of a shop effect, a strong era shock and a signal."""
    rng = np.random.default_rng(seed)
    era = np.repeat(np.arange(1, periods + 1), entities)
    ids = np.tile(np.arange(entities), periods)
    shops = np.array([f"shop{i}" for i in range(30)])
    shop = shops[rng.integers(0, 30, len(era))]
    effect = {s: rng.normal(0, 1) for s in shops}
    shock = rng.normal(0, 5, periods + 1)[era]
    signal = rng.normal(0, 1, len(era))
    target = np.array([effect[s] for s in shop]) + shock + signal + rng.normal(0, 0.3, len(era))
    start = datetime(2020, 1, 3)
    return pd.DataFrame(
        {
            "era": era,
            "id": ids,
            "when": [start + timedelta(weeks=int(e)) for e in era],
            "shop": shop,
            "signal": signal,
            "target": target,
            "__row": np.arange(len(era)),
        }
    )


FIELDS = [
    Field("era", "numeric_discrete", RAW),
    Field("id", "identifier", RAW),
    Field("when", "datetime", RAW),
    Field("shop", "categorical_nominal", RAW),
    Field("signal", "numeric_continuous", RAW),
    Field("target", "numeric_continuous", RAW),
]
TARGET = FIELDS[-1]
WALK = evaluate.Validation(
    method="panel_walk_forward", period_column="era", entity_column="id", folds=3
)


def decision(preparation: prep.Preparation, column: str) -> prep.Decision:
    return next(d for d in preparation.decisions if d.column == column)


def encodings(frame: pd.DataFrame, validation: evaluate.Validation, y: pd.Series) -> np.ndarray:
    """The cross-fitted training encodings of `shop` (before the scaler), for these targets."""
    setup = _setup(
        frame, FIELDS, TARGET, "regression", "linear_regression", None, None, None, None, validation
    )
    assert decision(setup.preparation, "shop").encoding == "target"
    built = prep.build(setup.preparation, len(setup.x_train), "regression")
    shop = next(step for _, step, columns in built.transformers if columns[0] == "shop")
    unscaled = Pipeline(shop.steps[:-1])
    rows = setup.x_train.index
    return np.asarray(unscaled.fit_transform(setup.x_train[["shop"]], y.loc[rows]), dtype=float)


# --- L1: target-encoding folds follow the validation ---------------------------------------------


def test_panel_encoding_never_reads_the_rows_own_period() -> None:
    frame = panel()
    y = cast(pd.Series, frame["target"])
    before = encodings(frame, WALK, y)
    period = (frame["era"] == 7).to_numpy()
    moved = y.copy()
    moved[period] += 100.0  # only the targets of era 7 change

    after = encodings(frame, WALK, moved)

    np.testing.assert_allclose(after[period], before[period])  # era 7 is encoded from other eras
    assert not np.allclose(after[~period], before[~period])  # the others do read era 7


def test_group_kfold_encoding_never_reads_the_rows_own_group() -> None:
    frame = panel()
    validation = evaluate.Validation(method="group_kfold", group="id", folds=3)
    y = cast(pd.Series, frame["target"])
    before = encodings(frame, validation, y)
    group = (frame["id"] == 3).to_numpy()
    moved = y.copy()
    moved[group] += 100.0

    after = encodings(frame, validation, moved)

    np.testing.assert_allclose(after[group], before[group])


def test_time_split_encoding_never_reads_the_rows_own_moment() -> None:
    frame = panel()
    validation = evaluate.Validation(method="time_split", time_column="when")
    setup = _setup(
        frame, FIELDS, TARGET, "regression", "linear_regression", None, None, None, None, validation
    )
    y = cast(pd.Series, frame["target"])
    before = encodings(frame, validation, y)
    trained = frame.loc[setup.x_train.index]
    moment = (trained["era"] == 5).to_numpy()
    moved = y.copy()
    moved[frame["era"] == 5] += 100.0

    after = encodings(frame, validation, moved)

    np.testing.assert_allclose(after[moment], before[moment])


def test_the_rule_says_which_folds_and_why() -> None:
    walk = _setup(
        panel(), FIELDS, TARGET, "regression", "linear_regression", None, None, None, None, WALK
    )
    shop = decision(walk.preparation, "shop")
    assert "R-CAT-08" in shop.rules
    sentence = shop.why[shop.rules.index("R-CAT-08")]
    assert "era" in sentence and "period" in sentence

    split = _setup(
        panel(), FIELDS, TARGET, "regression", "linear_regression", None, None, None, None, None
    )
    shop = decision(split.preparation, "shop")
    assert "random" in shop.why[shop.rules.index("R-CAT-08")]


def test_a_random_split_keeps_the_former_folds() -> None:
    setup = _setup(
        panel(), FIELDS, TARGET, "regression", "linear_regression", None, None, None, None, None
    )
    built = prep.build(setup.preparation, len(setup.x_train), "regression")
    shop = next(step for _, step, columns in built.transformers if columns[0] == "shop")
    cv = shop.named_steps["target"].cv
    assert isinstance(cv, KFold) and cv.shuffle and cv.random_state == 42


def test_a_saved_panel_model_loads_without_the_fold_keys_and_predicts_the_same() -> None:
    frame = panel()
    result = train(
        frame,
        FIELDS,
        TARGET,
        "regression",
        "linear_regression",
        None,
        len(frame),
        False,
        validation=WALK,
    )
    pipeline = cast(Any, result.pipeline)
    buffer = io.BytesIO()
    joblib.dump(pipeline, buffer)
    loaded = joblib.load(io.BytesIO(buffer.getvalue()))

    encoder = prep.fitted_step(loaded.named_steps["prepare"], "shop").named_steps["target"]
    assert encoder.cv.keys is None  # the keys stayed behind: the file holds no period of the data
    x = frame[[str(c) for c in pipeline.feature_names_in_]]
    np.testing.assert_allclose(loaded.predict(x), pipeline.predict(x))


# --- L2: frequency, rare categories, cycles ------------------------------------------------------

LINEAR = prep.ModelProfile(needs_scaling=True, sensitive_to_skew=True)
TREES = prep.ModelProfile(tree_based=True)


def mixed(n: int = 6_000, seed: int = 2) -> tuple[pd.DataFrame, list[Field]]:
    """300 products (too many to encode each), 40 shops of which 5 rare, an hour, two years of
    daily dates and a price."""
    rng = np.random.default_rng(seed)
    weights = np.linspace(1, 3, 300)
    products = rng.choice([f"p{i}" for i in range(300)], n, p=weights / weights.sum())
    shop = rng.choice([f"s{i}" for i in range(35)], n).astype(object)
    shop[:25] = np.repeat([f"rare{i}" for i in range(5)], 5)  # 5 rows each: under 10 and 1 %
    hour = rng.integers(0, 24, n)
    day = [datetime(2021, 1, 1) + timedelta(days=int(d)) for d in rng.integers(0, 730, n)]
    price = np.cos(2 * np.pi * hour / 24) * 5 + rng.normal(0, 1, n)
    frame = pd.DataFrame(
        {
            "product": products,
            "shop": shop,
            "hour": hour,
            "day": day,
            "price": price,
            "__row": np.arange(n),
        }
    )
    fields = [
        Field("product", "categorical_nominal", RAW),
        Field("shop", "categorical_nominal", RAW),
        Field("hour", "numeric_discrete", RAW, role="hour"),
        Field("day", "datetime", RAW),
        Field("price", "numeric_continuous", RAW),
    ]
    return frame, fields


MIXED, MIXED_FIELDS = mixed()
PRICE = MIXED_FIELDS[-1]


def prepared(
    model: prep.ModelProfile,
    choices: prep.Choices | None = None,
    frame: pd.DataFrame = MIXED,
    fields: list[Field] = MIXED_FIELDS,
) -> prep.Preparation:
    return prep.decide(frame, fields, PRICE, "regression", model, None, choices)


def fitted(preparation: prep.Preparation) -> tuple[Any, list[str], np.ndarray]:
    built = prep.build(preparation, len(MIXED), "regression")
    x = MIXED[[d.column for d in preparation.used]]
    matrix = np.asarray(built.fit_transform(x, MIXED["price"]), dtype=float)
    return built, [str(n) for n in built.get_feature_names_out()], matrix


def option(found: prep.Decision, key: str, value: str) -> prep.Option:
    return next(o for o in found.options[key] if o.value == value)


def test_many_categories_are_encoded_by_their_frequency() -> None:
    product = decision(prepared(LINEAR), "product")
    assert product.use and product.encoding == "frequency" and "R-CAT-09" in product.rules

    preparation = prepared(TREES)
    built, names, matrix = fitted(preparation)
    counts = cast(pd.Series, MIXED["product"]).value_counts(normalize=True)
    shares = cast(pd.Series, MIXED["product"]).map(counts.to_dict())
    np.testing.assert_allclose(matrix[:, names.index("product")], shares.to_numpy())
    unseen = MIXED.iloc[:1][[d.column for d in preparation.used]].assign(product="never seen")
    assert built.transform(unseen)[0, names.index("product")] == 0.0


def test_rare_categories_share_one_ordinal_code_and_one_target_encoding() -> None:
    _, names, matrix = fitted(prepared(TREES))  # 40 shops ≤ 50: ordinal codes for trees
    assert len(set(matrix[:25, names.index("shop")])) == 1  # the 5 rare shops are one "Other"

    target = prepared(LINEAR, prep.Choices(columns={"shop": prep.ColumnChoice(encoding="target")}))
    built = prep.build(target, len(MIXED), "regression")
    shop = next(step for _, step, columns in built.transformers if columns[0] == "shop")
    grouped = Pipeline(shop.steps[:2]).fit_transform(MIXED[["shop"]])
    assert set(np.asarray(grouped)[:25, 0]) == {"Other"}


def test_a_number_that_goes_round_becomes_a_sine_and_a_cosine() -> None:
    linear = prepared(LINEAR)
    hour = decision(linear, "hour")
    assert hour.transform == "cycle" and hour.cycle == [24, 0] and "R-DATE-04" in hour.rules
    _, names, matrix = fitted(linear)
    assert "hour_sin" in names and "hour_cos" in names and "hour" not in names
    at = MIXED["hour"].to_numpy()
    np.testing.assert_allclose(matrix[:, names.index("hour_cos")], np.cos(2 * np.pi * at / 24))

    trees = decision(prepared(TREES), "hour")
    assert trees.transform is None and option(trees, "transform", "cycle").reason is None

    counted = MIXED.assign(count=MIXED["hour"])
    fields = [*MIXED_FIELDS, Field("count", "numeric_discrete", RAW)]
    plain = decision(prepared(LINEAR, frame=counted, fields=fields), "count")
    reason = option(plain, "transform", "cycle").reason
    assert plain.transform != "cycle" and reason and "goes round" in reason


def test_dates_add_the_cycles_of_the_year_and_of_the_month() -> None:
    linear = prepared(LINEAR)
    day = decision(linear, "day")
    assert day.transform == "date_cycles"
    assert {"R-DATE-03", "R-DATE-06", "R-DATE-07"} <= set(day.rules)
    _, names, _ = fitted(linear)
    for part in ("week_sin", "day_of_year_cos", "day_sin", "month_sin"):
        assert f"day_{part}" in names
    assert "day_day" not in names  # the day of the month is a cycle here

    trees = prepared(TREES)
    assert decision(trees, "day").transform == "date_parts"
    _, names, _ = fitted(trees)
    assert {"day_week", "day_day_of_year", "day_day", "day_month"} <= set(names)

    weeks = [datetime(2021, 1, 1) + timedelta(weeks=i % 30) for i in range(len(MIXED))]
    short = decision(prepared(LINEAR, frame=MIXED.assign(day=weeks)), "day")
    assert "R-DATE-06" not in short.rules and "R-DATE-07" not in short.rules
    weekly = [datetime(2021, 1, 1) + timedelta(weeks=i % 120) for i in range(len(MIXED))]
    fridays = decision(prepared(LINEAR, frame=MIXED.assign(day=weekly)), "day")
    assert "R-DATE-06" in fridays.rules  # 2.3 years
    assert "R-DATE-07" not in fridays.rules  # every day of the month seen, but weekly dates


def test_the_user_can_choose_the_new_encodings() -> None:
    choices = prep.Choices(
        columns={
            "shop": prep.ColumnChoice(encoding="frequency"),
            "day": prep.ColumnChoice(transform="date_parts"),
        }
    )
    preparation = prepared(LINEAR, choices)
    assert decision(preparation, "shop").encoding == "frequency"
    assert decision(preparation, "day").transform == "date_parts"
    _, names, _ = fitted(preparation)
    assert "day_month" in names and "day_month_sin" not in names


def _dumped(value: Any) -> bytes:
    buffer = io.BytesIO()
    joblib.dump(value, buffer)  # our own object, loaded back at once
    return buffer.getvalue()


def test_a_date_parts_step_saved_before_063_still_transforms() -> None:
    step = prep.DateParts(cyclical=True).fit(MIXED[["day"]])
    del step.__dict__["yearly"], step.__dict__["monthly"]  # as unpickled from an older model
    restored = joblib.load(io.BytesIO(_dumped(step)))
    assert restored.transform(MIXED[["day"]].iloc[:3]).shape == (3, 8)
    assert restored.get_params()["yearly"] is False


def test_the_model_card_says_what_an_unknown_category_becomes() -> None:
    from quantilo.ml.predict import _encoders

    built, _, _ = fitted(prepared(TREES))
    found = _encoders(Pipeline([("prepare", built)]))
    assert found["product"][1] == "counted as never seen (frequency 0)"
    assert found["shop"][1] == "grouped with the rare categories (Other)"
    assert "rare0" in found["shop"][0]


# --- L7: causal smoothed lags ------------------------------------------------------------------


def lagged(values: pd.DataFrame, kind: str = "mean", shift: int = 1) -> np.ndarray:
    step = enc.PastSmooth(kind, window=3, shift=shift, entity=True)
    return step.fit_transform(values[["temp", "id", "era"]])[:, 1]


def test_a_lag_never_reads_the_rows_period_nor_the_embargo_nor_later() -> None:
    frame = panel().assign(temp=lambda f: np.random.default_rng(1).normal(0, 1, len(f)))
    for kind in ("mean", "ewm"):
        before = lagged(frame, kind, shift=3)  # embargo 2
        for t in (5, 12, 20):
            moved = frame.copy()
            moved.loc[moved["era"] >= t - 2, "temp"] += 1_000.0  # the embargo and after
            after = lagged(moved, kind, shift=3)
            at = (frame["era"] == t).to_numpy()
            np.testing.assert_allclose(after[at], before[at])
    first = lagged(frame)[(frame["era"] == 1).to_numpy()]
    assert np.isnan(first).all()  # nothing before the first period


def test_a_lag_is_the_mean_of_the_previous_periods_of_the_entity() -> None:
    frame = panel().assign(temp=lambda f: f["era"] * 10.0 + f["id"])
    found = lagged(frame)  # window 3, shift 1
    row = int(np.flatnonzero((frame["era"] == 10).to_numpy() & (frame["id"] == 4).to_numpy())[0])
    assert found[row] == np.mean([90 + 4, 80 + 4, 70 + 4])


def test_the_test_block_reads_the_history_learnt_at_fitting() -> None:
    frame = panel().assign(temp=lambda f: f["era"] * 1.0)
    columns = ["temp", "id", "era"]
    step = enc.PastSmooth("mean", window=2, shift=1).fit(frame.loc[frame["era"] <= 20, columns])
    later = step.transform(frame.loc[frame["era"] > 20, columns])
    assert not np.isnan(later[:, 1]).any()
    assert later[0, 1] == np.mean([20.0, 19.0])  # era 21 reads eras 20 and 19 from history
    restored = joblib.load(io.BytesIO(_dumped(step)))  # the history travels with the model
    np.testing.assert_allclose(restored.transform(frame.loc[frame["era"] > 20, columns]), later)


LAG_FIELDS = [*FIELDS[:-1], Field("echo", "numeric_continuous", RAW), TARGET]


def echoed() -> pd.DataFrame:
    """`echo` is the target plus noise; targets are independent from one era to the next."""
    rng = np.random.default_rng(9)
    frame = panel()
    frame["target"] = rng.normal(0, 1, len(frame))
    frame["echo"] = frame["target"] + rng.normal(0, 0.5, len(frame))
    return frame


def test_a_lag_of_a_column_that_echoes_the_target_does_not_leak() -> None:
    frame = echoed()
    choices = prep.Choices(columns={"echo": prep.ColumnChoice(transform="past_mean")})
    validation = evaluate.Validation(
        method="panel_walk_forward", period_column="era", entity_column="id", folds=3, embargo=1
    )
    setup = _setup(
        frame,
        LAG_FIELDS,
        TARGET,
        "regression",
        "linear_regression",
        None,
        None,
        choices,
        None,
        validation,
    )
    echo = decision(setup.preparation, "echo")
    assert echo.transform == "past_mean" and "R-LAG-01" in echo.rules
    assert "shifted by 2" in echo.why[echo.rules.index("R-LAG-01")]
    assert {"id", "era"} <= set(setup.x_train.columns)  # the order is read, not used as an input
    built = prep.build(setup.preparation, len(setup.x_train), "regression")
    matrix = np.asarray(built.fit_transform(setup.x_train, setup.y_train), dtype=float)
    names = [str(n) for n in built.get_feature_names_out()]
    assert "era" not in names and "id" not in names
    past = matrix[:, names.index("echo_past_mean")]
    now = matrix[:, names.index("echo")]
    y = setup.y_train.to_numpy()
    assert abs(np.corrcoef(past, y)[0, 1]) < 0.1  # only earlier eras: nothing of today's target
    assert np.corrcoef(now, y)[0, 1] > 0.8  # the value of the row itself, as before


def test_a_lag_without_ordered_rows_is_refused_with_the_reason() -> None:
    choices = prep.Choices(columns={"echo": prep.ColumnChoice(transform="past_ewm")})
    setup = _setup(
        echoed(),
        LAG_FIELDS,
        TARGET,
        "regression",
        "linear_regression",
        None,
        None,
        choices,
        None,
        None,
    )
    echo = decision(setup.preparation, "echo")
    assert echo.transform != "past_ewm" and "R-LAG-01" not in echo.rules
    assert any("Needs ordered rows" in w for w in setup.preparation.warnings)
    assert option(echo, "transform", "past_ewm").reason == prep.NEEDS_ORDER


def test_a_lagged_model_trains_walks_forward_and_needs_the_order_to_predict() -> None:
    frame = echoed()
    choices = prep.Choices(columns={"signal": prep.ColumnChoice(transform="past_ewm")})
    result = train(
        frame,
        LAG_FIELDS,
        TARGET,
        "regression",
        "linear_regression",
        None,
        len(frame),
        False,
        choices=choices,
        validation=WALK,
    )
    pipeline = cast(Any, result.pipeline)
    assert {"era", "id"} <= {str(c) for c in pipeline.feature_names_in_}
    assert "signal_past_ewm" in [str(n) for n in pipeline[:-1].get_feature_names_out()]
    assert result.panel is not None


def test_the_window_follows_the_grain_of_the_period() -> None:
    from quantilo.ml.train import order_for

    weekly = evaluate.Validation(
        method="panel_walk_forward", period_column="when", entity_column="id"
    )
    order = order_for(panel(), weekly)
    assert order is not None and (order.grain, order.window, order.shift) == ("week", 13, 1)
    numbered = order_for(panel(), WALK)
    assert numbered is not None and (numbered.grain, numbered.window) == ("period", 5)
    assert order_for(panel(), evaluate.Validation()) is None
