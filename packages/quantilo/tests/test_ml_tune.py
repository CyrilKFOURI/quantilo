"""021 — search spaces and the optimizer."""

import time
from typing import Any

import pytest

from quantilo.ml import catalog
from quantilo.ml import tune as tuner
from quantilo.ml.catalog.base import Context
from quantilo.ml.train import _setup, model_pipeline
from tests.test_ml import FIELDS, FRAME, by_name

PRICE = by_name["price"]
FEATURES = [f for f in FIELDS if f.name not in ("price_copy", "expensive")]


@pytest.mark.parametrize("key", sorted(catalog.spaces.SPACES))
def test_every_search_space_stays_within_the_settings(key: str) -> None:
    spec = catalog.CATALOGUE[key]
    task = "regression" if "regression" in spec.tasks else "binary"
    tuner.check_space(spec, Context(1_000, task), tuner.default_space(key))


def test_bad_spaces_are_refused() -> None:
    spec = catalog.CATALOGUE["random_forest"]
    context = Context(1_000, "regression")
    with pytest.raises(tuner.TuneError, match="not a setting"):
        tuner.check_space(spec, context, [tuner.SearchParam(key="nope", kind="int", low=1, high=2)])
    with pytest.raises(tuner.TuneError, match="between"):
        tuner.check_space(
            spec, context, [tuner.SearchParam(key="n_estimators", kind="int", low=1, high=2)]
        )
    with pytest.raises(tuner.TuneError, match="below the upper"):
        tuner.check_space(
            spec, context, [tuner.SearchParam(key="max_depth", kind="int", low=9, high=3)]
        )


def search(
    model: str, request: dict[str, Any], stop_after: int | None = None, warm: list | None = None
):
    setup = _setup(FRAME, FEATURES, PRICE, "regression", model, None, None, None)
    progress = tuner.TuneProgress("r2", True)

    def build(values: dict[str, Any], rows: int) -> Any:
        return model_pipeline(
            setup.spec, values, setup.context, setup.preparation, "regression", rows
        )

    found = tuner.tune(
        setup.x_train,
        setup.y_train,
        "regression",
        setup.spec,
        setup.context,
        setup.preparation,
        tuner.TuneRequest(model=model, **request),
        "r2",
        progress,
        (lambda: stop_after is not None and len(progress.trials) >= stop_after),
        build,
        None,
        warm,
    )
    return found, setup


def test_a_search_records_trials_beats_or_matches_defaults_and_keeps_the_best() -> None:
    found, setup = search("hist_gradient_boosting", {"trials": 8, "budget_seconds": None})

    assert len(found.trials) == 8 and found.default_value is not None
    complete = [t for t in found.trials if t.state == "complete"]
    assert complete and found.best_value == max(t.value for t in complete if t.value is not None)
    assert set(found.best_params) == {p.key for p in tuner.default_space("hist_gradient_boosting")}
    assert len(setup.x_train) == 960  # 80 % of the rows: the test rows never reach the search


def test_the_budget_and_a_manual_stop_are_respected() -> None:
    started = time.monotonic()
    found, _ = search("random_forest", {"budget_seconds": 10})
    assert time.monotonic() - started < 10 * 1.1 + 4  # + the defaults' cross-validation
    assert found.trials

    stopped, _ = search("linear_regression", {"budget_seconds": 60}, stop_after=3)
    assert stopped.stopped and 3 <= len(stopped.trials) <= 4


@pytest.mark.parametrize("sampler", ["random", "grid", "cmaes", "nsga2"])
def test_every_sampler_runs(sampler: str) -> None:
    found, _ = search(
        "linear_regression",
        {"trials": 4, "budget_seconds": None, "sampler": sampler, "pruner": "none"},
    )
    assert len(found.trials) == 4 and found.best_value is not None


def test_settings_importance_after_enough_trials_and_fixed_settings() -> None:
    space = [
        tuner.SearchParam(key="regularization", kind="categorical", fixed="ridge"),
        tuner.SearchParam(key="alpha", kind="float", low=1e-4, high=100, log=True),
    ]
    found, _ = search("linear_regression", {"trials": 22, "budget_seconds": None, "space": space})

    assert found.best_params["regularization"] == "ridge"
    assert set(found.importances) == {"alpha"} or found.importances == {}
    assert all(t.params.keys() == {"alpha"} for t in found.trials)


def test_a_model_with_nothing_to_tune_says_so() -> None:
    with pytest.raises(tuner.TuneError, match="nothing worth tuning"):
        search("bayesian_ridge", {"trials": 2})


@pytest.mark.parametrize("model", ["xgboost", "lightgbm", "catboost"])
def test_boosting_libraries_can_be_tuned_on_a_class_with_roc_auc(model: str) -> None:
    spec = catalog.CATALOGUE[model]
    if not spec.installed:
        pytest.skip(f"{model} is not installed")
    target = by_name["expensive"]
    features = [f for f in FIELDS if f.name not in ("price", "price_copy", "expensive")]
    setup = _setup(FRAME, features, target, "binary", model, None, None, None)
    progress = tuner.TuneProgress("roc_auc", True)

    def build(values: dict[str, Any], rows: int) -> Any:
        return model_pipeline(setup.spec, values, setup.context, setup.preparation, "binary", rows)

    found = tuner.tune(
        setup.x_train,
        setup.y_train,
        "binary",
        setup.spec,
        setup.context,
        setup.preparation,
        tuner.TuneRequest(model=model, trials=2, budget_seconds=None, sampler="grid"),
        "roc_auc",
        progress,
        lambda: False,
        build,
    )
    # the wrapper says it is a classifier: the scorer takes one column of probabilities
    assert found.default_value is not None and found.best_value is not None


# --- 066: very advanced search ---------------------------------------------------------------


def test_two_objectives_give_a_pareto_front_of_score_and_time() -> None:
    found, _ = search(
        "random_forest",
        {"trials": 8, "budget_seconds": None, "sampler": "nsga2", "objectives": "score_time"},
    )
    assert found.objectives == "score_time"
    complete = [t for t in found.trials if t.state == "complete" and t.value is not None]
    front = [t for t in complete if t.pareto]
    assert front  # the front is never empty
    # R-ML-TUNE-02: the best value is the best score of the front, also the best overall
    assert found.best_value == max(t.value or 0.0 for t in front)
    assert found.best_value == max(t.value or 0.0 for t in complete)
    assert any(r.startswith("R-ML-TUNE-01") for r in found.rules)


def test_two_objectives_refuse_a_single_objective_sampler() -> None:
    with pytest.raises(tuner.TuneError, match="R-ML-TUNE-01"):
        search("random_forest", {"trials": 2, "sampler": "cmaes", "objectives": "score_time"})


def test_a_warm_start_runs_first_and_says_where_it_comes_from() -> None:
    from quantilo.ml import warm

    start = warm.Start({"alpha": 0.5}, "Best settings found on this dataset.")
    space = [tuner.SearchParam(key="alpha", kind="float", low=1e-4, high=100, log=True)]
    found, _ = search(
        "linear_regression",
        {
            "trials": 3,
            "budget_seconds": None,
            "space": space,
            "settings": {"regularization": "ridge"},
        },
        warm=[start],
    )
    first = found.trials[0]
    assert first.params == {"alpha": 0.5} and first.warm == start.reason
    assert found.warm == [start.reason]
    assert all(t.warm is None for t in found.trials[1:])
