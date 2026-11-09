"""035 — machine learning on panel data: recognition, walk-forward by period with an embargo,
scores per period, panel leaks, exposure and neutralization, whole-period sampling."""

from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

import quantilo as dx
from quantilo.ml import evaluate
from quantilo.ml import panel as panels
from quantilo.ml.data import Field
from quantilo.ml.facade import Failed
from quantilo.ml.train import TrainingError, train
from quantilo.typing.types import Parser

RAW = Parser(kind="raw")


def simulated(periods: int = 40, entities: int = 60, seed: int = 3) -> pd.DataFrame:
    """ids by eras: a target made of an id effect, an era shock and one real signal; `future` is
    the id's target of the next era (a column from the future)."""
    rng = np.random.default_rng(seed)
    era = np.repeat(np.arange(1, periods + 1), entities)
    ids = np.tile(np.arange(entities), periods)
    signal = rng.normal(0, 1, len(era))
    noise = rng.normal(0, 1, (len(era), 3))
    shock = rng.normal(0, 3, periods + 1)[era - 1]
    effect = rng.normal(0, 1, entities)[ids]
    target = 0.6 * signal + shock + effect + rng.normal(0, 0.5, len(era))
    frame = pd.DataFrame({"era": era, "id": ids, "signal": signal, "target": target})
    for i in range(3):
        frame[f"noise{i}"] = noise[:, i]
    following = frame.groupby("id")["target"].shift(-1)
    frame["future"] = following.fillna(frame["target"])
    frame["__row"] = np.arange(len(frame))
    return frame


PANEL_FIELDS = [
    Field("era", "numeric_discrete", RAW),
    Field("id", "identifier", RAW),
    Field("signal", "numeric_continuous", RAW),
    Field("noise0", "numeric_continuous", RAW),
    Field("noise1", "numeric_continuous", RAW),
    Field("noise2", "numeric_continuous", RAW),
    Field("future", "numeric_continuous", RAW),
    Field("target", "numeric_continuous", RAW),
]
TARGET = PANEL_FIELDS[-1]
WALK = evaluate.Validation(
    method="panel_walk_forward", period_column="era", entity_column="id", folds=3, embargo=1
)


# --- Walk-forward folds --------------------------------------------------------------------------


def test_no_test_period_is_closer_than_the_embargo_to_training() -> None:
    ranks = np.repeat(np.arange(30), 5)
    labels = {r: str(r + 1) for r in range(30)}
    found = panels.plan(ranks, labels, folds=3, embargo=2)
    assert len(found.folds) == 3
    previous_end = -1
    for fit_at, held_at in found.folds:
        trained, tested = ranks[fit_at], ranks[held_at]
        assert tested.min() - trained.max() > 2  # 2 whole periods between them
        assert tested.min() > previous_end  # blocks follow each other in time
        previous_end = int(tested.max())
    assert previous_end == 29
    spans = found.spans
    assert [s.fold for s in spans] == [1, 2, 3]
    assert spans[0].test_start - spans[0].train_end == 3
    assert spans[2].train_from == "1" and spans[2].test_to == "30"


def test_rolling_window_keeps_the_last_periods_only() -> None:
    ranks = np.repeat(np.arange(40), 2)
    found = panels.plan(ranks, {}, folds=3, embargo=0, window="rolling", window_periods=6)
    for fit_at, held_at in found.folds:
        trained = np.unique(ranks[fit_at])
        assert len(trained) == 6
        assert trained.max() == ranks[held_at].min() - 1


def test_too_few_periods_or_too_long_an_embargo_is_refused() -> None:
    with pytest.raises(evaluate.ValidationError, match="at least 6 periods"):
        panels.plan(np.repeat(np.arange(4), 3), {}, folds=5, embargo=0)
    with pytest.raises(evaluate.ValidationError, match="embargo"):
        panels.plan(np.repeat(np.arange(8), 3), {}, folds=3, embargo=5)


def test_one_period_in_four_keeps_each_training_period_whole_or_absent() -> None:
    ranks = np.repeat(np.arange(240), np.random.default_rng(1).integers(3, 9, 240))
    rows_of = pd.Series(ranks).value_counts()
    found = panels.plan(ranks, {}, folds=3, embargo=4)
    assert found.step == 4  # more than 200 periods: 1 in 4 by default
    for fit_at, _ in [*found.folds, (found.final, None)]:
        used = pd.Series(ranks[fit_at]).value_counts()
        for rank, count in used.items():
            assert count == rows_of[rank]  # a whole period, never part of one
        kept = np.unique(ranks[fit_at])
        assert set(np.diff(kept)) == {4}
    few = panels.plan(np.repeat(np.arange(50), 2), {}, folds=3, embargo=0)
    assert few.step == 1
    chosen = panels.plan(np.repeat(np.arange(50), 2), {}, folds=3, embargo=0, sample_periods=2)
    assert chosen.step == 2


def test_periods_are_put_in_order_whatever_their_form() -> None:
    frame = pd.DataFrame({"era": ["0010", "0002", "0001", "0010", None]})
    ranks, labels = panels.period_ranks(frame, "era")
    assert list(ranks) == [2, 1, 0, 2, -1]
    assert labels == {0: "0001", 1: "0002", 2: "0010"}
    dates = pd.DataFrame({"when": pd.to_datetime(["2024-02-01", "2024-01-01"])})
    assert list(panels.period_ranks(dates, "when")[0]) == [1, 0]
    assert panels.period_ranks(dates, "when")[1][0] == "2024-01-01"


# --- Scores per period ---------------------------------------------------------------------------


def test_summary_of_the_periods() -> None:
    values = [0.02, -0.01, -0.03, 0.04, -0.02, 0.05]
    scores = [panels.PeriodScore(str(i), i, 1, 10, v) for i, v in enumerate(values)]
    summary = panels.summarise(scores, "spearman")
    assert summary.mean == pytest.approx(np.mean(values))
    assert summary.std == pytest.approx(np.std(values, ddof=1))
    assert summary.sharpe == pytest.approx(np.mean(values) / np.std(values, ddof=1))
    assert summary.share_good == pytest.approx(0.5) and summary.threshold == 0.0
    assert summary.worst == -0.03 and summary.worst_period == "2"
    assert summary.longest_bad_streak == 2
    assert panels.criterion_value(summary, "worst") == -0.03
    errors = panels.summarise(scores, "mae", threshold=0.0)
    assert errors.lower_is_better and errors.worst == 0.05


def test_correlation_metrics_join_the_regression_metrics() -> None:
    actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    predicted = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
    found = evaluate.regression_metrics(actual, predicted)
    assert found["pearson"] == pytest.approx(0.8)
    assert found["spearman"] == pytest.approx(0.8)
    assert evaluate.regression_metric("spearman", actual, np.ones(5)) is None


# --- Training ------------------------------------------------------------------------------------


def test_walk_forward_scores_every_test_period_and_names_the_future_column() -> None:
    frame = simulated()
    result = train(
        frame,
        PANEL_FIELDS,
        TARGET,
        "regression",
        "linear_regression",
        None,
        1,
        False,
        validation=WALK,
    )
    panel = result.panel
    assert panel is not None and result.primary == "spearman"
    assert [f.fold for f in panel.folds] == [1, 2, 3]
    assert len(panel.periods) == sum(f.test_periods for f in panel.folds)
    assert result.test_rows == sum(p.rows for p in panel.periods)
    assert result.score == panel.value == panel.summary.mean
    assert all(p.pearson is not None and p.spearman is not None for p in panel.periods)
    future = [a for a in panel.alerts if a.kind == "future"]
    assert [a.column for a in future] == ["future"]
    assert "column from the future" in future[0].message
    decisions = {d.column: d for d in result.preparation.decisions}
    assert decisions["era"].rules == ["R-SEL-09"] and not decisions["era"].use
    assert result.out_of_fold is not None and len(result.out_of_fold.rows) == result.test_rows
    assert panel.exposures and panel.max_exposure is not None


def test_a_shuffled_k_fold_overestimates_the_score_of_a_panel() -> None:
    frame = simulated(periods=30, entities=50)
    fields = [f for f in PANEL_FIELDS if f.name != "future"]
    shuffled = train(
        frame,
        fields,
        TARGET,
        "regression",
        "random_forest",
        {"n_estimators": 40},
        1,
        False,
        validation=evaluate.Validation(method="kfold", folds=3),
    )
    forward = train(
        frame,
        fields,
        TARGET,
        "regression",
        "random_forest",
        {"n_estimators": 40},
        1,
        False,
        validation=WALK.model_copy(update={"metric": "r2"}),
    )
    # random folds learn each era's shock from its own rows; walk-forward cannot
    assert shuffled.scores["r2"] is not None and forward.scores["r2"] is not None
    assert shuffled.scores["r2"] > forward.scores["r2"] + 0.3


def test_criterion_and_suspicious_score() -> None:
    frame = simulated()
    frame["copy"] = frame["target"] * 2 + 1  # the answer, but not perfect enough for R-SEL-06
    frame["copy"] += np.random.default_rng(0).normal(0, 0.8, len(frame))
    fields = [*PANEL_FIELDS[:-1], Field("copy", "numeric_continuous", RAW), TARGET]
    worst = train(
        frame,
        fields,
        TARGET,
        "regression",
        "random_forest",
        {"n_estimators": 30},
        1,
        False,
        validation=WALK.model_copy(update={"criterion": "worst"}),
    )
    assert worst.panel is not None and worst.score == worst.panel.summary.worst
    honest = train(
        frame,
        PANEL_FIELDS,
        TARGET,
        "regression",
        "linear_regression",
        None,
        1,
        False,
        validation=WALK,
    )
    assert honest.panel is not None
    assert not [a for a in honest.panel.alerts if a.kind == "suspicious"]
    with pytest.raises(TrainingError, match="higher is better"):
        train(
            frame,
            PANEL_FIELDS,
            TARGET,
            "regression",
            "linear_regression",
            None,
            1,
            False,
            validation=WALK.model_copy(update={"criterion": "sharpe", "metric": "mae"}),
        )


def test_suspicious_score_needs_a_wide_gap() -> None:
    alert = panels.suspicious("spearman", 0.9, 0.1, "era")
    assert (
        alert is not None and "Check that no column is computed from the target." in alert.message
    )
    assert panels.suspicious("spearman", 0.04, 0.02, "era") is None
    assert panels.suspicious("roc_auc", 0.62, 0.55, "era") is None


def test_a_panel_classifier_is_scored_per_period() -> None:
    frame = simulated()
    frame["up"] = np.where(
        frame["target"] > frame.groupby("era")["target"].transform("median"), "up", "down"
    )
    fields = [*[f for f in PANEL_FIELDS if f.name not in ("target", "future")]]
    fields.append(Field("up", "categorical_nominal", RAW))
    result = train(
        frame, fields, fields[-1], "binary", "logistic_regression", None, 1, False, validation=WALK
    )
    assert result.panel is not None and result.primary == "roc_auc"
    assert result.panel.summary.threshold == 0.5
    assert result.out_of_fold is not None and result.out_of_fold.positive is not None


# --- Embargo suggested by the target -------------------------------------------------------------


def test_a_target_four_periods_ahead_calls_for_an_embargo_of_four() -> None:
    rng = np.random.default_rng(5)
    periods, entities = 80, 40
    moves = rng.normal(0, 1, (entities, periods + 4))
    rows = [
        {"era": t + 1, "id": e, "target": moves[e, t + 1 : t + 5].sum()}
        for t in range(periods)
        for e in range(entities)
    ]
    frame = pd.DataFrame(rows)
    ranks, _ = panels.period_ranks(frame, "era")
    found = panels.horizon(cast(pd.Series, frame["id"]), ranks, frame["target"].to_numpy(), "id")
    assert found is not None
    embargo, reason = found
    assert embargo == 4 and "embargo of 4" in reason
    # a persistent target (an id effect that never fades) is no horizon
    persistent = frame.assign(target=rng.normal(0, 1, entities)[frame["id"]] * 3)
    ids = cast(pd.Series, persistent["id"])
    assert panels.horizon(ids, ranks, persistent["target"].to_numpy(), "id") is None


# --- Exposure and neutralization -----------------------------------------------------------------


def test_neutralizing_on_a_column_removes_the_exposure_to_it() -> None:
    rng = np.random.default_rng(2)
    ranks = np.repeat(np.arange(20), 100)
    column = rng.normal(0, 1, len(ranks))
    other = rng.normal(0, 1, len(ranks))
    scores = 0.9 * column + np.sqrt(1 - 0.81) * other
    features = column[:, None]
    before = panels.exposures(scores, features, ranks)[0]
    assert before == pytest.approx(0.9, abs=0.05)
    neutral = panels.neutralize(scores, features, ranks, 1.0)
    assert abs(panels.exposures(neutral, features, ranks)[0]) < 0.05
    half = panels.neutralize(scores, features, ranks, 0.5)
    assert 0.05 < panels.exposures(half, features, ranks)[0] < before
    # each period keeps its mean and spread
    first = ranks == 0
    assert neutral[first].std() == pytest.approx(scores[first].std())

    actual = scores + rng.normal(0, 1, len(ranks))
    oof = panels.OutOfFold(
        np.arange(len(ranks)), ranks, np.ones(len(ranks), dtype=int), scores, actual
    )
    done = panels.neutralized(oof, pd.DataFrame({"c": column}), ["c"], 1.0, "spearman", {}, "mean")
    assert abs(done.after[0].mean) < 0.05 and done.before[0].mean > 0.8
    assert len(done.periods) == 20 and done.summary.mean is not None
    raw = panels.summarise(
        panels.score_periods(
            ranks, oof.folds, {}, panels.regression_scorer("spearman", actual, scores)
        ),
        "spearman",
    )
    assert done.summary.mean != pytest.approx(raw.mean)  # the periods were scored again


# --- Recognition ---------------------------------------------------------------------------------


def test_an_integer_era_and_a_repeated_id_are_recognised() -> None:
    frame = simulated()
    check = panels.recognise(
        frame, ["era", "signal"], ["id", "era"], len(frame), {"era": 40}, "target"
    )
    assert check.panel and check.period == "era" and check.entity == "id"
    assert check.periods == 40 and check.per_period == 60
    assert check.message == (
        "This looks like panel data: 40 eras × ~60 ids. Random splits would overestimate the "  # noqa: RUF001
        "score."
    )
    assert check.duplicates == 0
    # an unnamed whole number without an entity coming back is no period
    plain = pd.DataFrame({"age": np.repeat(np.arange(20, 60), 10), "x": np.arange(400)})
    assert not panels.recognise(plain, ["age"], ["x"], 400, {"age": 40}).panel


# --- Library and loader --------------------------------------------------------------------------


@pytest.fixture
def numerai_like(tmp_path: Path) -> dx.Dataset:
    frame = simulated(periods=30, entities=40)
    frame["era"] = frame["era"].map(lambda e: f"{e:04d}")
    path = tmp_path / "panel.csv"
    frame.drop(columns=["__row"]).to_csv(path, index=False)
    return dx.open(path, cache_dir=tmp_path / "cache")


def test_the_library_trains_walk_forward_by_era(numerai_like: dx.Dataset) -> None:
    ml = numerai_like.ml(
        "target",
        validation={
            "method": "panel_walk_forward",
            "period_column": "era",
            "entity_column": "id",
            "embargo": 4,
            "folds": 3,
        },
    )
    check = ml.panel()
    assert check.panel and check.period == "era"
    model = ml.train()
    assert model.panel is not None and model.panel.embargo == 4
    assert model.panel.periods[0].period == "10"  # "0010" is read as a number
    for span in model.panel.folds:
        assert span.test_start - span.train_end > 4
    neutral = model.neutralize(["signal"], 1.0)
    assert abs(neutral.after[0].mean) < 0.05


def test_a_sample_of_a_panel_keeps_whole_periods(numerai_like: dx.Dataset) -> None:
    validation = {
        "method": "panel_walk_forward",
        "period_column": "era",
        "folds": 3,
        "max_rows": 400,
    }
    model = numerai_like.ml("target", validation=validation).train()
    assert model.sampled
    assert model.panel is not None
    assert {p.rows for p in model.panel.periods} == {40}  # every era read whole
    ranks = [p.rank for p in model.panel.periods]
    assert len(set(np.diff(ranks))) == 1 and ranks[-1] == 29  # one era in 3, the last kept


def test_compare_and_tune_follow_the_panel_criterion(numerai_like: dx.Dataset) -> None:
    validation = {
        "method": "panel_walk_forward",
        "period_column": "era",
        "entity_column": "id",
        "folds": 3,
        "criterion": "worst",
    }
    ml = numerai_like.ml("target", validation=validation)
    found = ml.compare(["linear_regression", "decision_tree"])
    scores = [m.result.score for m in found if not isinstance(m, Failed)]
    assert len(scores) == 2 and None not in scores
    assert cast(float, scores[0]) >= cast(float, scores[1])
    for model in found:
        assert not isinstance(model, Failed) and model.panel is not None
        assert model.result.score == model.panel.summary.worst
    progress = ml.tune("decision_tree", budget_seconds=None, trials=4)
    assert progress.criterion == "worst" and progress.metric == "spearman"
    assert progress.best_value is not None and len(progress.trials) == 4


def test_the_panel_charts_are_drawn_as_in_the_app(numerai_like: dx.Dataset) -> None:
    pytest.importorskip("plotly")
    validation = {"method": "panel_walk_forward", "period_column": "era", "folds": 3}
    model = numerai_like.ml("target", validation=validation).train()
    bars: Any = dx.plot(model, "periods")
    assert bars.data[0].type == "bar" and len(bars.data[0].x) == len(model.panel.periods)
    line: Any = dx.plot(model, "cumulative")
    assert line.data[0].y[-1] == pytest.approx(sum(p.score or 0 for p in model.panel.periods))
    schema: Any = dx.plot(model, "folds")
    assert [t.name for t in schema.data] == ["trained on", "embargo", "tested on"]


def test_the_preparation_preview_leaves_the_period_out_as_training_does(
    numerai_like: dx.Dataset,
) -> None:
    # the S&P 500 panel showed the date "used" in the preview but dropped by the training (040)
    validation = {"method": "panel_walk_forward", "period_column": "era", "entity_column": "id"}
    decisions = numerai_like.ml("target", validation=validation).preview().preparation.decisions
    era = next(d for d in decisions if d.column == "era")

    assert not era.use
    assert "R-SEL-09" in era.rules
