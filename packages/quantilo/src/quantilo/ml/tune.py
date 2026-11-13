"""Search the best settings of a model within a budget (021), with Optuna. Every trial is judged by
3-fold cross-validation on the training rows only: the test rows are never seen here. On a panel
(035), the 3 folds walk forward by period and a trial is judged on the panel criterion (mean of
the periods, mean / std, or worst period).

Very advanced (066), in Advanced > More: a warm start from settings already found (`warm.py`), and
two objectives, the score and the seconds a trial takes, whose Pareto front is shown (NSGA-II).
- R-ML-TUNE-01: with two objectives no trial is pruned (Optuna cannot compare a trial halfway
  through on two objectives), and CMA-ES or a grid are refused (single objective only).
- R-ML-TUNE-02: the best trial of a two-objective search is the best score of the front; the
  other points of the front trade score for speed."""

from __future__ import annotations

import importlib.util
import math
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.base import clone
from sklearn.metrics import get_scorer
from sklearn.model_selection import KFold, StratifiedKFold

from quantilo.ml import browser, catalog, evaluate
from quantilo.ml import panel as panels
from quantilo.ml import preparation as prep
from quantilo.ml.explain import SCORERS
from quantilo.ml.target import Task

if TYPE_CHECKING:  # Optuna is the `ml` extra: imported when a search starts
    import optuna

    from quantilo.ml.warm import Start

FOLDS = 3
GRID_POINTS = 4
IMPORTANCE_TRIALS = 20
SEED = 42

Sampler = Literal["tpe", "random", "grid", "cmaes", "nsga2"]
Pruner = Literal["median", "halving", "hyperband", "none"]
Objectives = Literal["score", "score_time"]
SINGLE_OBJECTIVE = ("grid", "cmaes")  # R-ML-TUNE-01


class SearchParam(BaseModel):
    """One setting of the search space, as the user may change it."""

    key: str = Field(max_length=64)
    kind: Literal["float", "int", "categorical"]
    low: float | None = None
    high: float | None = None
    log: bool = False
    choices: list[str] = Field(default_factory=list, max_length=50)
    fixed: Any = None  # a value: the setting is not searched


class TuneRequest(BaseModel):
    model: str = Field(max_length=64)
    budget_seconds: int | None = Field(default=60, ge=10, le=3_600)
    trials: int | None = Field(default=None, ge=2, le=2_000)
    sampler: Sampler = "tpe"
    pruner: Pruner = "median"
    space: list[SearchParam] | None = None  # None: the catalogue's
    settings: dict[str, Any] = Field(default_factory=dict)  # other settings, held fixed
    # very advanced (066): start from settings already found; optimise the score and the time
    warm_start: bool = False
    objectives: Objectives = "score"


@dataclass
class Trial:
    number: int
    value: float | None  # in the metric's own direction (an MAE stays positive)
    state: str  # complete, pruned, failed
    params: dict[str, Any]
    seconds: float
    warm: str | None = None  # a warm start (066): where these settings come from
    pareto: bool = False  # two objectives: on the front (no other trial is better and faster)


@dataclass
class TuneProgress:
    metric: str
    greater_is_better: bool
    trials: list[Trial] = field(default_factory=list)
    best_value: float | None = None
    best_params: dict[str, Any] = field(default_factory=dict)
    default_value: float | None = None
    importances: dict[str, float] = field(default_factory=dict)
    stopped: bool = False
    seconds: float = 0.0
    criterion: str | None = None  # panel: how the periods' scores are summed up
    objectives: Objectives = "score"
    warm: list[str] = field(default_factory=list)  # why each warm start was chosen
    rules: list[str] = field(default_factory=list)  # the rules applied (066)


@dataclass
class PanelSearch:
    """Walk-forward folds of a panel: each trial is scored on every test period, then judged on
    the criterion over those periods."""

    folds: list[tuple[np.ndarray, np.ndarray]]
    ranks: np.ndarray  # period rank of each training row
    criterion: str
    threshold: float | None = None


class TuneError(ValueError):
    pass


def available() -> bool:
    """Tuning can run here: never in the browser version (R-WEB-04), else when Optuna is there."""
    return not browser.active() and importlib.util.find_spec("optuna") is not None


def _optuna() -> Any:
    if browser.active():
        raise TuneError(browser.desktop_only("Tuning"))
    try:
        import optuna
    except ModuleNotFoundError:
        raise TuneError("Tuning needs Optuna: pip install 'quantilo[ml]'.") from None
    return optuna


def default_space(model: str) -> list[SearchParam]:
    return [
        SearchParam(key=p.key, kind=p.kind, low=p.low, high=p.high, log=p.log, choices=p.choices)
        for p in catalog.spaces.space(model)
    ]


def check_space(
    spec: catalog.ModelSpec, context: catalog.Context, space: list[SearchParam]
) -> None:
    """Every searched setting exists and its bounds stay within what the model accepts."""
    settings = {s.key: s for s in spec.settings(context)}
    for param in space:
        setting = settings.get(param.key)
        if setting is None:
            raise TuneError(f"{param.key} is not a setting of {spec.label}.")
        if param.fixed is not None:
            continue
        if param.kind == "categorical":
            text = setting.kind == "text"  # e.g. hidden layers: any value the model accepts
            if not param.choices or (
                not text and any(c not in setting.options for c in param.choices)
            ):
                raise TuneError(f"{setting.label}: choose among {', '.join(setting.options)}.")
            continue
        if param.low is None or param.high is None or param.low >= param.high:
            raise TuneError(f"{setting.label}: the lower bound must be below the upper one.")
        if (setting.minimum is not None and param.low < setting.minimum) or (
            setting.maximum is not None and param.high > setting.maximum
        ):
            raise TuneError(
                f"{setting.label}: bounds must stay between {setting.minimum:g} and "
                f"{setting.maximum:g}."
            )
        if param.log and param.low <= 0:
            raise TuneError(f"{setting.label}: a log scale needs a lower bound above 0.")


def _scorer(metric: str, task: Task) -> Any:
    name = SCORERS.get(metric)
    if metric == "roc_auc":
        name = "roc_auc" if task == "binary" else "roc_auc_ovr"
    if name is None:
        raise TuneError(f"{metric} cannot guide the search: choose another primary metric.")
    return get_scorer(name) if isinstance(name, str) else name


def _sampler(kind: Sampler, space: list[SearchParam]) -> optuna.samplers.BaseSampler:
    optuna = _optuna()
    if kind == "random":
        return optuna.samplers.RandomSampler(seed=SEED)
    if kind == "cmaes":
        return optuna.samplers.CmaEsSampler(seed=SEED, warn_independent_sampling=False)
    if kind == "nsga2":
        return optuna.samplers.NSGAIISampler(seed=SEED)
    if kind == "grid":
        grid: dict[str, list[Any]] = {}
        for p in space:
            if p.fixed is not None:
                continue
            if p.kind == "categorical":
                grid[p.key] = list(p.choices)
            else:
                low, high = float(p.low or 0), float(p.high or 1)
                points = (
                    np.geomspace(low, high, GRID_POINTS)
                    if p.log
                    else np.linspace(low, high, GRID_POINTS)
                )
                grid[p.key] = (
                    sorted({round(float(v)) for v in points})
                    if p.kind == "int"
                    else [float(v) for v in points]
                )
        return optuna.samplers.GridSampler(grid, seed=SEED)
    return optuna.samplers.TPESampler(seed=SEED)


def _pruner(kind: Pruner) -> optuna.pruners.BasePruner:
    optuna = _optuna()
    return {
        "median": optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0),
        "halving": optuna.pruners.SuccessiveHalvingPruner(),
        "hyperband": optuna.pruners.HyperbandPruner(min_resource=1, max_resource=FOLDS),
        "none": optuna.pruners.NopPruner(),
    }[kind]


def _suggest(trial: optuna.Trial, param: SearchParam) -> Any:
    if param.fixed is not None:
        return param.fixed
    if param.kind == "categorical":
        return trial.suggest_categorical(param.key, param.choices)
    if param.kind == "int":
        return trial.suggest_int(
            param.key, int(param.low or 0), int(param.high or 1), log=param.log
        )
    return trial.suggest_float(
        param.key, float(param.low or 0), float(param.high or 1), log=param.log
    )


def tune(
    x: pd.DataFrame,
    y: pd.Series,
    task: Task,
    spec: catalog.ModelSpec,
    context: catalog.Context,
    preparation: prep.Preparation,
    request: TuneRequest,
    metric: str,
    progress: TuneProgress,
    should_stop: Callable[[], bool],
    build: Callable[[dict[str, Any], int], Any],
    panel: PanelSearch | None = None,
    warm: list[Start] | None = None,
) -> TuneProgress:
    """Optimise `request.space` on 3-fold cross-validation of the training rows `x`, `y`.
    `build(values, rows)` returns the full pipeline for a set of settings; `progress` is updated
    after every trial (the caller reads it while the search runs)."""
    space = request.space if request.space is not None else default_space(spec.key)
    if not [p for p in space if p.fixed is None]:
        raise TuneError(f"{spec.label} has nothing worth tuning.")
    check_space(spec, context, space)
    two = request.objectives == "score_time"
    if two and request.sampler in SINGLE_OBJECTIVE:
        raise TuneError("R-ML-TUNE-01: score and time together need NSGA-II, TPE or random search.")
    optuna = _optuna()  # after the checks: a wrong request says so even without Optuna
    progress.objectives = request.objectives
    if two:
        progress.rules.append(
            "R-ML-TUNE-01: two objectives (score and seconds per trial): no trial is pruned."
        )
    scorer = _scorer(metric, task) if panel is None else None
    sign = 1.0 if metric not in evaluate.LOWER_IS_BETTER else -1.0
    if panel is not None:
        folds = panel.folds
    else:
        splitter = (
            StratifiedKFold(FOLDS, shuffle=True, random_state=SEED)
            if task != "regression" and int(y.value_counts().min()) >= FOLDS
            else KFold(FOLDS, shuffle=True, random_state=SEED)
        )
        folds = list(splitter.split(x, y if task != "regression" else None))
    positive = str(y.value_counts().index[-1]) if task == "binary" else None

    def periods(model: Any, at: np.ndarray) -> list[panels.PeriodScore]:
        held, truth = x.iloc[at], y.iloc[at]
        if task == "regression":
            guess = np.asarray(model.predict(held), dtype=float)
            score = panels.regression_scorer(metric, truth.to_numpy(dtype=float), guess)
        else:
            classes = [str(c) for c in model.classes_]
            chances = np.asarray(model.predict_proba(held), dtype=float)
            labels = np.asarray(classes)[chances.argmax(axis=1)]
            score = panels.classification_scorer(
                metric, truth.to_numpy().astype(str), labels, chances, classes, positive
            )
        ranks = cast(PanelSearch, panel).ranks[at]
        return panels.score_periods(ranks, np.zeros(len(at), dtype=int), {}, score)

    def judged(found: list[panels.PeriodScore]) -> float:
        search = cast(PanelSearch, panel)
        summary = panels.summarise(found, metric, search.threshold)
        value = panels.criterion_value(summary, search.criterion)
        return float("nan") if value is None else value

    started = time.monotonic()
    budget = request.budget_seconds

    def over() -> bool:
        return should_stop() or (budget is not None and time.monotonic() - started > budget)

    def cross_validate(values: dict[str, Any], trial: optuna.Trial | None) -> float:
        scores: list[float] = []
        found: list[panels.PeriodScore] = []
        for step, (fit_at, held_at) in enumerate(folds):
            if trial is not None and over():
                raise optuna.TrialPruned()
            model: Any = clone(build(values, len(fit_at)))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(x.iloc[fit_at], y.iloc[fit_at])
                if panel is not None:
                    found += periods(model, held_at)
                    scores = [sign * judged(found)]
                else:
                    rate = cast(Callable[..., float], scorer)
                    scores.append(float(rate(model, x.iloc[held_at], y.iloc[held_at])))
            if trial is not None and not two:
                trial.report(float(np.mean(scores)), step)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        return float(np.mean(scores))

    def resolved(params: dict[str, Any]) -> dict[str, Any]:
        return catalog.resolve(spec, context, {**request.settings, **params})

    # the defaults, judged on the same folds: the gain is measured against them
    default = sign * cross_validate(resolved({}), None)
    progress.default_value = default if math.isfinite(default) else None

    def objective(trial: optuna.Trial) -> float | tuple[float, float]:
        params = {p.key: _suggest(trial, p) for p in space}
        try:
            values = resolved(params)
        except catalog.SettingsError as error:
            raise optuna.TrialPruned(str(error)) from None
        begun = time.monotonic()
        found = cross_validate(values, trial)
        return (found, time.monotonic() - begun) if two else found

    reasons: dict[int, str] = {}

    def record(study: optuna.Study, frozen: optuna.trial.FrozenTrial) -> None:
        raw = frozen.values[0] if frozen.values else None
        value = None if raw is None or not math.isfinite(raw) else sign * raw
        state = frozen.state.name.lower()
        seconds = (
            (frozen.datetime_complete - frozen.datetime_start).total_seconds()
            if frozen.datetime_complete and frozen.datetime_start
            else 0.0
        )
        progress.trials.append(
            Trial(
                frozen.number,
                value,
                state,
                dict(frozen.params),
                seconds,
                reasons.get(frozen.number),
            )
        )
        complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if complete:
            if two:  # R-ML-TUNE-02: the best score of the front
                front = study.best_trials
                on_front = {t.number for t in front}
                for t in progress.trials:
                    t.pareto = t.number in on_front
                best = max(front, key=lambda t: t.values[0])
            else:
                best = study.best_trial
            progress.best_value = sign * float(best.values[0] if best.values else 0.0)
            progress.best_params = {
                **{p.key: p.fixed for p in space if p.fixed is not None},
                **best.params,
            }
        progress.seconds = time.monotonic() - started
        if over():
            progress.stopped = should_stop()
            study.stop()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        directions=["maximize", "minimize"] if two else ["maximize"],
        sampler=_sampler(request.sampler, space),
        pruner=_pruner("none" if two else request.pruner),
    )
    if warm and request.sampler == "grid":
        progress.rules.append("R-ML-WARM-04: the grid walks its own points: no warm start.")
    elif warm:
        for number, start in enumerate(warm):
            study.enqueue_trial(start.params, skip_if_exists=True)
            reasons[number] = start.reason
            progress.warm.append(start.reason)
    study.optimize(
        objective,
        n_trials=request.trials,
        timeout=budget,
        callbacks=[record],
        catch=(ValueError, ArithmeticError),
    )
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(complete) >= IMPORTANCE_TRIALS:
        try:
            found = optuna.importance.get_param_importances(
                study, target=(lambda t: t.values[0]) if two else None
            )
            progress.importances = {k: float(v) for k, v in found.items()}
        except (RuntimeError, ValueError):
            progress.importances = {}
    progress.seconds = time.monotonic() - started
    return progress
