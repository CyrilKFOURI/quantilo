"""038 — AutoML within a time budget: baselines, one model per family, successive elimination,
Optuna on the 3 best families, and an honest ensemble chosen on out-of-fold predictions."""

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import VotingClassifier, VotingRegressor

import quantilo as dx
from quantilo.errors import ExplorerError
from quantilo.ml import auto as automl
from quantilo.ml import train as trainer
from quantilo.ml import tune as tuner
from tests.test_ml_panel import simulated

pytest.importorskip("optuna")


def regression(rows: int, seed: int = 0, nonlinear: float = 1.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, (rows, 5))
    city = rng.choice(["a", "b", "c"], rows)
    y = 2 * x[:, 0] - x[:, 1] + nonlinear * np.sin(3 * x[:, 2]) * x[:, 3]
    y = y + np.where(city == "a", 1.0, 0.0) + rng.normal(0, 0.5, rows)
    frame = pd.DataFrame(x, columns=[f"x{i}" for i in range(5)])
    frame["city"] = city
    frame["y"] = y
    return frame


def weakly_correlated(rows: int = 3_000, seed: int = 1, task: str = "binary") -> pd.DataFrame:
    """Twenty small linear effects (a linear model's strength: boosting draws them as steps) plus
    a wave and an interaction (boosting's strength): the two models' errors are weakly
    correlated. Two classes (log-odds = 2 x the signal) or a number."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, (rows, 23))
    signal = x[:, :20] @ np.full(20, 0.5) + np.sin(2 * x[:, 20]) + x[:, 21] * x[:, 22]
    frame = pd.DataFrame(x, columns=[f"x{i}" for i in range(23)])
    if task == "regression":
        frame["label"] = signal + rng.normal(0, 0.3, rows)
    else:
        chance = 1 / (1 + np.exp(-2 * signal))
        frame["label"] = np.where(rng.random(rows) < chance, "yes", "no")
    return frame


def dataset(frame: pd.DataFrame, tmp_path: Path, name: str = "data") -> dx.Dataset:
    path = tmp_path / f"{name}.csv"
    frame.to_csv(path, index=False)
    return dx.open(path, cache_dir=tmp_path / "cache")


KFOLD = {"method": "kfold", "folds": 5}


# --- Pieces ---------------------------------------------------------------------------------------


def test_greedy_selection_adds_the_model_whose_errors_differ() -> None:
    rng = np.random.default_rng(0)
    truth = rng.normal(0, 1, 2_000)
    first = truth + rng.normal(0, 0.5, 2_000)  # two models with independent errors
    second = truth + rng.normal(0, 0.5, 2_000)
    bad = truth + rng.normal(0, 2.0, 2_000)

    def score(values: np.ndarray) -> float:
        return -float(np.mean((values[:, 0] - truth) ** 2))

    counts = automl.greedy([v.reshape(-1, 1) for v in (first, second, bad)], score, True)
    assert counts[0] > 0 and counts[1] > 0 and counts[2] == 0
    alone = automl.greedy([first.reshape(-1, 1), first.reshape(-1, 1) + 0.0], score, True)
    assert alone == [1, 0]  # a copy of the same model adds nothing


def test_the_gain_is_said_against_the_linear_model() -> None:
    assert automl.gain_sentence("mae", None, 77, 100, "linear model") == (
        "23% lower MAE than the linear model"
    )
    assert automl.gain_sentence("r2", None, 0.85, 0.80, "linear model") == (
        "25% lower error than the linear model"
    )
    assert automl.gain_sentence("spearman", "sharpe", 1.2, 0.8, "linear model") == (
        "+0.4 spearman against the linear model"
    )
    assert automl.gain_sentence("r2", None, None, 0.8, "linear model") is None


def test_unknown_models_are_refused(tmp_path: Path) -> None:
    ml = dataset(regression(200), tmp_path).ml("y")
    with pytest.raises(ExplorerError, match="Unknown model"):
        ml.auto(minutes=0.1, models=["nope"])
    with pytest.raises(ExplorerError, match="Invalid Auto request"):
        ml.auto(minutes=0)


# --- Acceptance scenarios -------------------------------------------------------------------------


def test_a_short_budget_is_respected_with_baselines_fast_families_and_a_result(
    tmp_path: Path,
) -> None:
    """The 2-minute scenario at a test's scale: 20 000 rows in 20 seconds."""
    ml = dataset(regression(20_000), tmp_path).ml("y")
    budget = 20.0
    began = time.monotonic()
    found = ml.auto(minutes=budget / 60)
    spent = time.monotonic() - began
    # The budget is a ceiling (+10 %, plus 3 s for the closing steps, which do not shrink with a
    # test's short budget: 22.2 s seen on a CI runner for 20 s). Auto may stop earlier when no
    # model left fits the time remaining (15 to 19 s seen locally), so the floor checks it was used.
    assert budget * 0.7 <= spent <= budget * 1.1 + 3
    progress = found.progress
    assert progress.finished and not progress.stopped
    by_id = {e.id: e for e in progress.entries}
    assert by_id["constant"].status == "done" and by_id["linear_regression"].status == "done"
    families = [e for e in progress.entries if e.step != "baselines" and e.status == "done"]
    assert families, "at least one family model"
    assert found.model is not None and progress.best is not None
    assert [s.key for s in progress.steps] == [
        "baselines",
        "families",
        "reduction",
        "tuning",
        "ensemble",
    ]
    assert progress.log and progress.log[0].message.startswith("Your validation is a train")
    skipped = [e for e in progress.entries if e.status == "skipped"]
    assert all(e.note for e in skipped)  # every skipped family says why


def test_the_ensemble_beats_the_best_single_model_when_errors_differ(tmp_path: Path) -> None:
    """LightGBM and a logistic regression with weakly correlated errors."""
    ml = dataset(weakly_correlated(), tmp_path).ml("label", validation=KFOLD)
    found = ml.auto(minutes=5, models=["lightgbm"], trials=2)
    ensemble = found.progress.ensemble
    assert ensemble is not None and ensemble.score is not None
    assert ensemble.best_score is not None
    assert ensemble.score > ensemble.best_score  # AUC: higher is better, on the same folds
    models = {m.model for m in ensemble.members}
    assert {"lightgbm", "logistic_regression"} <= models
    if ensemble.kept:  # beaten by more than one standard deviation: the run is the ensemble
        model = found.model
        assert model is not None and model.result.model == automl.ENSEMBLE
        assert isinstance(model.pipeline.named_steps["model"], VotingClassifier)
        predicted = model.predict(weakly_correlated(rows=20, seed=9).drop(columns=["label"]))
        assert list(predicted.columns[:3]) == ["predicted", "p(False)", "p(True)"]


def test_the_kept_ensemble_is_a_normal_run(tmp_path: Path) -> None:
    frame = weakly_correlated(rows=2_000, task="regression")
    ml = dataset(frame, tmp_path).ml("label", validation=KFOLD)
    found = ml.auto(minutes=5, models=["lightgbm"], trials=2)
    ensemble = found.progress.ensemble
    assert ensemble is not None and ensemble.kept, ensemble and ensemble.reason
    assert {m.model for m in ensemble.members} >= {"lightgbm", "linear_regression"}
    assert found.progress.best == automl.ENSEMBLE
    assert found.leaderboard[0].id == automl.ENSEMBLE
    assert found.progress.gain is not None and "than the linear model" in found.progress.gain
    model = found.model
    assert model is not None and model.result.model == automl.ENSEMBLE
    assert isinstance(model.pipeline.named_steps["model"], VotingRegressor)
    # judged on the same folds as the single models: refitting it fold by fold gives the score
    # the greedy selection measured on their out-of-fold predictions
    assert model.result.cross_validation is not None
    assert model.result.score == pytest.approx(ensemble.score, abs=1e-6)
    explanation = model.explain()
    assert explanation.permutation is not None and explanation.permutation.items
    assert explanation.shap is None and explanation.shap_unavailable
    predicted = model.predict(weakly_correlated(rows=20, seed=9, task="regression"))
    assert len(predicted) == 20


def test_the_single_model_is_recommended_when_the_ensemble_is_not_clearly_better(
    tmp_path: Path,
) -> None:
    linear = regression(800, seed=4, nonlinear=0.0)
    ml = dataset(linear, tmp_path).ml("y", validation=KFOLD)
    found = ml.auto(minutes=2, models=["decision_tree", "knn"], trials=2)
    ensemble = found.progress.ensemble
    assert ensemble is not None and not ensemble.kept
    assert "single model is recommended" in ensemble.reason or "alone" in ensemble.reason
    assert found.model is not None and found.model.result.model != automl.ENSEMBLE
    assert found.progress.best == found.model.result.model


def test_the_rule_needs_more_than_one_standard_deviation() -> None:
    assert automl.kept_ensemble(0.83, 0.80, 0.02, True)
    assert not automl.kept_ensemble(0.81, 0.80, 0.02, True)
    assert automl.kept_ensemble(9.0, 10.0, 0.5, False)  # an error: lower is better
    assert not automl.kept_ensemble(None, 0.80, 0.02, True)


def test_stopping_during_tuning_keeps_the_best_model_so_far(tmp_path: Path) -> None:
    ml = dataset(regression(1_000, seed=2), tmp_path).ml("y", validation=KFOLD)
    seen: dict[str, Any] = {}
    original = tuner.tune

    def spy(*args: Any, **kwargs: Any) -> Any:
        seen["tuning"] = True
        return original(*args, **kwargs)

    tuner.tune = spy
    try:
        found = ml.auto(
            minutes=2,
            models=["random_forest", "decision_tree"],
            should_stop=lambda: bool(seen.get("tuning")),
        )
    finally:
        tuner.tune = original
    progress = found.progress
    assert progress.stopped and progress.finished
    assert next(s for s in progress.steps if s.key == "tuning").status == "stopped"
    assert found.model is not None
    best = next(e for e in progress.entries if e.id == progress.best)
    assert best.status == "done" and found.model.result.score == best.score
    assert any("Stopped" in line.message for line in progress.log)


def test_panel_validation_is_used_at_every_step(tmp_path: Path, monkeypatch: Any) -> None:
    frame = simulated(periods=30, entities=40).drop(columns=["__row"])
    validation = {
        "method": "panel_walk_forward",
        "period_column": "era",
        "entity_column": "id",
        "folds": 3,
        "embargo": 1,
        "criterion": "worst",
    }
    ml = dataset(frame, tmp_path, "panel").ml("target", validation=validation)
    methods: list[str] = []
    panels_seen: list[bool] = []
    train, tune = trainer.train, tuner.tune

    def train_spy(*args: Any, **kwargs: Any) -> Any:
        methods.append(kwargs["validation"].method)
        return train(*args, **kwargs)

    def tune_spy(*args: Any, **kwargs: Any) -> Any:
        panels_seen.append(args[11] is not None)
        return tune(*args, **kwargs)

    monkeypatch.setattr(trainer, "train", train_spy)
    monkeypatch.setattr(tuner, "tune", tune_spy)
    found = ml.auto(minutes=2, models=["decision_tree", "random_forest"], trials=3)
    assert methods and set(methods) == {"panel_walk_forward"}
    assert panels_seen and all(panels_seen)
    progress = found.progress
    assert progress.criterion == "worst" and progress.metric == "spearman"
    assert "Walk-forward" in progress.compared_on
    model = found.model
    assert model is not None and model.panel is not None
    assert model.panel.criterion == "worst"
    assert model.result.score == model.panel.summary.worst
    ranked = [e.score for e in found.leaderboard if e.score is not None and e.status == "done"]
    assert ranked == sorted(ranked, reverse=True)


def test_a_split_is_compared_on_five_folds_and_the_test_rows_judge_the_kept_model(
    tmp_path: Path,
) -> None:
    frame = weakly_correlated(rows=900, seed=3)
    ml = dataset(frame, tmp_path).ml("label", validation={"threshold": "f1"})
    found = ml.auto(minutes=2, models=["decision_tree"], trials=2)
    progress = found.progress
    assert "5 folds of the training rows" in progress.compared_on
    model = found.model
    assert model is not None and model.result.validation.method == "split"
    assert model.result.test_rows == 180  # the 20 % test rows judge it
    assert progress.test_score == model.result.score
    classification = model.result.classification
    assert classification is not None and classification.threshold is not None
    trained = [e for e in progress.entries if e.status == "done" and e.model != "constant"]
    assert all(e.rows <= 720 for e in trained)  # the search never saw a test row


def test_stacking_is_an_option(tmp_path: Path) -> None:
    ml = dataset(regression(600, seed=5), tmp_path).ml("y", validation=KFOLD)
    found = ml.auto(minutes=2, models=["random_forest"], trials=2, stacking=True)
    ensemble = found.progress.ensemble
    assert ensemble is not None and ensemble.stacking_score is not None
