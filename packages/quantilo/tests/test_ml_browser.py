"""102 — machine learning in the browser version: what is offered there, and its rules R-WEB-01
to 04. The browser is simulated (`browser.IN_BROWSER`); the heavy models are made "installed", so
that only the browser rule keeps them out."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import quantilo as dx
from quantilo.ml import browser, catalog, evaluate
from quantilo.ml import tune as tuner
from quantilo.ml.catalog.base import Context
from quantilo.ml.train import TrainingError, train
from tests.test_ml import FIELDS, FRAME, by_name

HEAVY = ("xgboost", "lightgbm", "catboost")
SMALL = FRAME.iloc[:300].reset_index(drop=True)


@pytest.fixture
def in_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(browser, "IN_BROWSER", True)


@pytest.fixture
def heavy_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in HEAVY:
        monkeypatch.setattr(catalog.CATALOGUE[key], "package", None)


def regression(rows: int) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(0, 1, (rows, 4)), columns=["x0", "x1", "x2", "x3"])
    frame["city"] = rng.choice(["a", "b", "c"], rows)
    frame["y"] = 2 * frame["x0"] - frame["x1"] + rng.normal(0, 0.5, rows)
    return frame


def dataset(frame: pd.DataFrame, tmp_path: Path) -> dx.Dataset:
    path = tmp_path / "data.csv"
    frame.to_csv(path, index=False)
    return dx.open(path, cache_dir=tmp_path / "cache")


def build(key: str, task: str = "binary") -> Any:
    spec = catalog.CATALOGUE[key]
    context = Context(1_000, task)  # type: ignore[arg-type]
    return spec.build({s.key: s.default for s in spec.settings(context)}, context)


# --- R-WEB-02: one core ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["random_forest", "extra_trees", "bagging"])
def test_models_use_every_core_on_a_computer_and_one_in_the_browser(
    key: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert build(key).n_jobs == -1
    monkeypatch.setattr(browser, "IN_BROWSER", True)
    assert build(key).n_jobs == 1


def test_forests_grow_fewer_trees_by_default_in_the_browser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = Context(1_000, "regression")
    for key in ("random_forest", "extra_trees"):
        trees = next(s for s in catalog.CATALOGUE[key].settings(context) if s.key == "n_estimators")
        assert trees.default == 300
    monkeypatch.setattr(browser, "IN_BROWSER", True)
    for key in ("random_forest", "extra_trees"):
        trees = next(s for s in catalog.CATALOGUE[key].settings(context) if s.key == "n_estimators")
        assert trees.default == 100 and "R-WEB-02" in trees.why


def test_a_loaded_model_predicts_on_one_core_in_the_browser(in_browser: None) -> None:
    pipeline = Pipeline([("scale", StandardScaler()), ("model", RandomForestClassifier(n_jobs=-1))])
    assert browser.single_core(pipeline).named_steps["model"].n_jobs == 1


def test_a_loaded_model_is_left_alone_on_a_computer() -> None:
    model = RandomForestClassifier(n_jobs=-1)
    assert browser.single_core(model).n_jobs == -1


# --- R-WEB-04: what is offered --------------------------------------------------------------------


def test_the_heavy_models_are_left_to_the_desktop_app_in_the_browser(
    in_browser: None, heavy_installed: None
) -> None:
    for key, spec in catalog.CATALOGUE.items():
        if key in HEAVY:
            assert spec.desktop_only and spec.elsewhere and not spec.available
            assert spec.missing() == f"{spec.label} is available in the desktop app."
        else:
            assert spec.package is None and spec.available, key  # pure scikit-learn


def test_on_a_computer_nothing_is_left_to_the_desktop_app(heavy_installed: None) -> None:
    assert all(spec.available and not spec.elsewhere for spec in catalog.CATALOGUE.values())


def test_a_missing_package_on_a_computer_still_says_not_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(catalog.CATALOGUE["xgboost"], "package", "no_such_package_102")
    spec = catalog.CATALOGUE["xgboost"]
    assert not spec.available and not spec.elsewhere
    assert spec.missing() == "XGBoost is not installed in this build."


def test_recommendations_only_name_available_models(
    in_browser: None, heavy_installed: None
) -> None:
    for rows in (500, 5_000, 500_000):
        for task in ("regression", "binary"):
            keys = [key for key, _ in catalog.recommend(Context(rows, task))]  # type: ignore[arg-type]
            assert all(catalog.CATALOGUE[key].available for key in keys), (rows, task, keys)
    large = [key for key, _ in catalog.recommend(Context(500_000, "regression"))]
    assert "hist_gradient_boosting" in large


def test_the_catalogue_says_what_the_browser_holds_back(
    tmp_path: Path, in_browser: None, heavy_installed: None
) -> None:
    found = dataset(regression(300), tmp_path).ml("y").catalogue()
    assert found.browser is not None
    assert (found.browser.max_rows, found.browser.auto_minutes, found.browser.tuning) == (
        50_000,
        5.0,
        False,
    )
    views = {m.key: m for m in found.models}
    assert {k for k, m in views.items() if m.desktop_only} == set(HEAVY) & set(views)
    assert all(not views[k].installed for k in HEAVY if k in views)
    assert views["hist_gradient_boosting"].installed and views["random_forest"].installed
    assert found.compare and not set(found.compare) & set(HEAVY)
    assert "hist_gradient_boosting" in found.compare
    assert not {r.key for r in found.recommended} & set(HEAVY)


def test_the_catalogue_holds_nothing_back_on_a_computer(tmp_path: Path) -> None:
    found = dataset(regression(300), tmp_path).ml("y").catalogue()
    assert found.browser is None
    assert not any(m.desktop_only for m in found.models)


def test_training_a_heavy_model_in_the_browser_says_where_it_is(
    in_browser: None, heavy_installed: None
) -> None:
    fields = [f for f in FIELDS if f.name not in ("price_copy", "expensive")]
    with pytest.raises(TrainingError, match="XGBoost is available in the desktop app"):
        train(SMALL, fields, by_name["price"], "regression", "xgboost", None, len(SMALL), False)


def test_tuning_is_left_to_the_desktop_app_in_the_browser(in_browser: None) -> None:
    assert not tuner.available()
    with pytest.raises(tuner.TuneError, match="Tuning is available in the desktop app"):
        tuner._optuna()


# --- R-WEB-01: training rows ----------------------------------------------------------------------


def test_training_rows_are_held_to_50_000_in_the_browser(in_browser: None) -> None:
    assert evaluate.Validation().max_rows == 50_000
    assert evaluate.Validation(max_rows=None).max_rows == 50_000
    assert evaluate.Validation(max_rows=200_000).max_rows == 50_000
    assert evaluate.Validation(max_rows=10_000).max_rows == 10_000


def test_training_rows_are_unchanged_on_a_computer() -> None:
    assert evaluate.Validation().max_rows == 200_000
    assert evaluate.Validation(max_rows=None).max_rows is None


# --- R-WEB-03 and R-WEB-04: the Auto --------------------------------------------------------------


def test_the_auto_in_the_browser_is_short_picks_available_models_and_skips_tuning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, in_browser: None, heavy_installed: None
) -> None:
    monkeypatch.setattr(browser, "AUTO_MINUTES", 0.1)  # 6 seconds at a test's scale
    found = dataset(regression(1_500), tmp_path).ml("y").auto(minutes=1)
    progress = found.progress
    assert progress.budget_seconds == pytest.approx(6.0)
    assert progress.finished and found.model is not None
    messages = [line.message for line in progress.log]
    assert any("R-WEB-03" in message for message in messages)
    tuning = next(s for s in progress.steps if s.key == "tuning")
    assert tuning.status == "skipped" and "R-WEB-04" in tuning.note
    tried = {e.model for e in progress.entries if e.status != "skipped"}
    assert not tried & set(HEAVY)
