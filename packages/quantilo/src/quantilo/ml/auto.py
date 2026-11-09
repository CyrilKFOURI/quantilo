"""AutoML within a time budget (038): "the best model you can find in 10 minutes", in steps said
one by one, every model judged on the same folds, the ensemble included.

1. Baselines: a constant model (mean or majority class) and the linear reference.
2. Families: one model per family with its defaults, cheapest first by the estimated duration
   (019b); a family too slow for the budget is skipped, and the journal says why.
3. Reduction: successive elimination (the weak families stop early, on a quarter then half of the
   rows), then the 3 best families are trained on every row.
4. Tuning: Optuna (TPE, median pruning) on those 3 families, the time left shared by rank.
5. Ensemble: greedy selection with replacement on out-of-fold predictions (Caruana et al.,
   2004), or a regularized linear stack (option). It is kept only if it beats the best single
   model by more than that model's standard deviation between folds.

No leakage: the preparation is refitted inside every fold (it is part of every pipeline), the
ensemble only sees out-of-fold predictions, and with a train / test split every model is compared
on 5 folds of the training rows: the test rows only judge the model kept (and a decision
threshold is chosen on training folds, as in 019b). The result is a normal run: one scikit-learn
pipeline (a weighted `VotingRegressor` / `VotingClassifier`, or a `Stacking…` estimator)."""

from __future__ import annotations

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
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    StackingClassifier,
    StackingRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from quantilo.ml import classes as binning
from quantilo.ml import evaluate
from quantilo.ml import models as catalogue
from quantilo.ml import panel as panels
from quantilo.ml import preparation as prep
from quantilo.ml import train as trainer
from quantilo.ml import tune as tuner
from quantilo.ml.data import Field as DataField
from quantilo.ml.target import Task

if TYPE_CHECKING:
    from quantilo.ml.engine import MlEngine

ENSEMBLE = "ensemble"  # the run's model key when the ensemble is kept
CONSTANT = "constant"
ENSEMBLE_LABEL = "Ensemble"
ENSEMBLE_FOLDS = 5  # a train / test split is compared on 5 folds of the training rows
TOP_FAMILIES = 3
TUNE_SHARES = (0.5, 0.3, 0.2)  # the time left for tuning, by rank
RUNG_ROWS = 2_000  # successive elimination from 4 x 2 000 rows: a quarter, half, then all
PANEL_RUNG_PERIODS = 80  # … on a panel, from 80 periods: 1 period in 4, in 2, then all
FAMILY_SHARE = 0.4  # no new family starts after 40 % of the budget
REDUCTION_SHARE = 0.6
FINAL_SHARE = 0.15  # the 3 best are trained on every row up to 85 % of the budget
SLOW_SHARE = 0.25  # a family estimated above a quarter of the budget is skipped
RESERVE = 1.5  # time kept for the ensemble and the final run: their estimate x 1.5…
RESERVE_SHARE = 0.03  # … plus 3 % of the budget (preparing the members, saving)
GRACE = 0.08  # the last fit may run this share of the budget past it (the run stays in +10 %)
ROUNDS = 20  # greedy selection rounds
POOL = 8  # models the ensemble may draw from (the best ones trained on every row)
STACK_FOLDS = 3
ALPHAS = np.logspace(-3, 3, 13)
BOUNDED = {
    "r2",
    "pearson",
    "spearman",
    "roc_auc",
    "pr_auc",
    "accuracy",
    "balanced_accuracy",
    "f1_macro",
    "f1_weighted",
    "precision",
    "recall",
    "mcc",
    "kappa",
}
ERROR_NAMES = {
    "mae": "MAE",
    "rmse": "RMSE",
    "mape": "MAPE",
    "median_ae": "median absolute error",
    "max_error": "maximum error",
    "log_loss": "log loss",
    "brier": "Brier score",
}

StepKey = Literal["baselines", "families", "reduction", "tuning", "ensemble"]
STEPS: list[tuple[StepKey, str]] = [
    ("baselines", "Baselines"),
    ("families", "One model per family"),
    ("reduction", "Reduction"),
    ("tuning", "Tuning"),
    ("ensemble", "Ensemble"),
]


class AutoRequest(BaseModel):
    """What the Auto may spend and try. `trials`: a fixed number of Optuna trials per tuned
    family (the ranking is then reproducible at a fixed seed); None: as many as the time allows."""

    minutes: float = Field(default=5, gt=0, le=240)
    models: list[str] | None = Field(default=None, min_length=1, max_length=40)
    trials: int | None = Field(default=None, ge=2, le=2_000)
    stacking: bool = False
    preparation: prep.Choices | None = None


@dataclass
class AutoStep:
    key: str
    title: str
    status: Literal["waiting", "running", "done", "skipped", "stopped"] = "waiting"
    seconds: float = 0.0
    note: str = ""


@dataclass
class AutoEntry:
    """One model the Auto tried: a row of the final leaderboard."""

    id: str
    model: str  # catalogue key, "constant" or "ensemble"
    label: str
    family: str
    step: str  # where it was last trained
    settings: dict[str, Any] = field(default_factory=dict)  # changed from the defaults
    status: Literal["running", "done", "failed", "skipped", "stopped"] = "running"
    score: float | None = None  # on the Auto's folds (panel: the criterion)
    std: float | None = None  # between folds
    seconds: float = 0.0
    rows: int = 0  # training rows of its last fit
    share: float = 1.0  # of the rows (a panel: of the periods) it was compared on
    note: str | None = None
    estimate: float | None = None  # seconds, before it ran


@dataclass
class LogLine:
    at: float  # seconds since the start
    step: str
    message: str


@dataclass
class Member:
    entry: str
    model: str
    label: str
    weight: float  # greedy: times chosen; stacking: the linear stack's share
    settings: dict[str, Any] = field(default_factory=dict)


@dataclass
class EnsembleResult:
    method: Literal["greedy", "stacking"]
    members: list[Member]
    score: float | None  # on the same folds as the single models
    std: float | None
    best_single: str | None  # the entry it is measured against
    best_score: float | None
    best_std: float | None
    kept: bool
    reason: str
    stacking_score: float | None = None


@dataclass
class AutoProgress:
    """The Auto's state, read while it runs and kept with the run it saves (its journal)."""

    budget_seconds: float
    metric: str = ""
    greater_is_better: bool = True
    criterion: str | None = None  # panel: how the periods' scores are summed up
    compared_on: str = ""  # how every model is judged, in a sentence
    step: str | None = None
    steps: list[AutoStep] = field(default_factory=lambda: [AutoStep(k, t) for k, t in STEPS])
    entries: list[AutoEntry] = field(default_factory=list)
    log: list[LogLine] = field(default_factory=list)
    ensemble: EnsembleResult | None = None
    best: str | None = None  # the entry kept
    gain: str | None = None  # "23% lower error than the linear model"
    test_score: float | None = None  # a train / test split: the kept model on the test rows
    seconds: float = 0.0
    trials: int = 0  # Optuna trials, every family together
    stopped: bool = False
    finished: bool = False

    def ranking(self) -> list[AutoEntry]:
        """Every model tried, best first; those without a score last."""
        sign = 1.0 if self.greater_is_better else -1.0
        scored = [e for e in self.entries if e.score is not None and e.status == "done"]
        scored.sort(key=lambda e: -sign * cast(float, e.score))
        return [*scored, *(e for e in self.entries if e not in scored)]


@dataclass
class AutoOutcome:
    progress: AutoProgress
    trained: trainer.Trained | None  # the run to save (None: nothing could be trained)


class AutoError(ValueError):
    pass


class _Stopped(Exception):
    """The user stopped the Auto."""


@dataclass
class _Candidate:
    """A model trained on every row of the Auto's folds: it may join the ensemble."""

    entry: AutoEntry
    key: str
    settings: dict[str, Any]
    held: trainer.FoldPredictions
    seconds: float
    score: float | None  # the run's own score
    rows: int  # training rows


def identity() -> FunctionTransformer:
    """The ensemble's "prepare" step: every member prepares the rows itself."""
    return FunctionTransformer(feature_names_out="one-to-one")


def constant_model(task: Task) -> Callable[[int], Pipeline]:
    model = DummyRegressor() if task == "regression" else DummyClassifier(strategy="prior")
    return lambda rows: Pipeline([("prepare", identity()), ("model", clone(model))])


def greedy(
    values: list[np.ndarray],
    score: Callable[[np.ndarray], float | None],
    greater: bool,
    rounds: int = ROUNDS,
) -> list[int]:
    """Ensemble selection (Caruana et al., 2004): start from the best model, then add, with
    replacement, the one whose addition scores best; keep the best of these averages. Returns how
    many times each model was chosen (its weight)."""
    sign = 1.0 if greater else -1.0

    def judged(found: np.ndarray) -> float:
        value = score(found)
        return -math.inf if value is None or not math.isfinite(value) else sign * value

    singles = [judged(v) for v in values]
    first = int(np.argmax(singles))
    counts = [0] * len(values)
    counts[first] = 1
    total, n = values[first].astype(float).copy(), 1
    best, kept = singles[first], list(counts)
    for _ in range(rounds - 1):
        tried = [judged((total + v) / (n + 1)) for v in values]
        pick = int(np.argmax(tried))
        total, n = total + values[pick], n + 1
        counts[pick] += 1
        if tried[pick] > best + 1e-12:
            best, kept = tried[pick], list(counts)
    return kept


def kept_ensemble(
    score: float | None, best: float | None, best_std: float | None, greater: bool
) -> bool:
    """The ensemble is kept only if it beats the best single model by more than that model's
    standard deviation between folds; otherwise the single model is simpler and as good."""
    if score is None or best is None:
        return False
    gap = (score - best) * (1.0 if greater else -1.0)
    return gap > (best_std or 0.0)


def gain_sentence(
    metric: str, criterion: str | None, best: float | None, reference: float | None, label: str
) -> str | None:
    """The kept model against the linear reference, in words."""
    if best is None or reference is None:
        return None
    if metric in evaluate.LOWER_IS_BETTER:
        if reference <= 0:
            return None
        change = (reference - best) / reference
        name = ERROR_NAMES.get(metric, metric)
    elif metric in BOUNDED and criterion != "sharpe" and reference < 1:
        change = ((1 - reference) - (1 - best)) / (1 - reference)
        name = "error"
    else:
        diff = best - reference
        return f"{diff:+.3g} {metric} against the {label}"
    direction = "lower" if change >= 0 else "higher"
    return f"{abs(change):.0%} {direction} {name} than the {label}"


class _Auto:
    def __init__(
        self,
        engine: MlEngine,
        fields: list[DataField],
        target: DataField,
        source: str,
        request: AutoRequest,
        classes: binning.Classes | None,
        validation: evaluate.Validation,
        should_stop: Callable[[], bool],
        clock: Callable[[], float],
    ) -> None:
        self.engine, self.fields, self.target, self.source = engine, fields, target, source
        self.request, self.classes, self.validation = request, classes, validation
        self.should_stop, self.clock = should_stop, clock
        self.choices = request.preparation
        self.budget = request.minutes * 60
        self.start = clock()
        self.deadline = self.start + self.budget
        self.progress = AutoProgress(self.budget)
        self.pool: list[_Candidate] = []
        # the last run of every model (and the share of rows it saw): the best is kept
        self.latest: dict[str, tuple[AutoEntry, trainer.Trained, float]] = {}
        self.final: trainer.Trained | None = None
        self.unit = "rows"

    # --- Small helpers ---------------------------------------------------------------------------

    def now(self) -> float:
        return self.clock() - self.start

    def log(self, message: str) -> None:
        self.progress.log.append(LogLine(round(self.now(), 2), self.progress.step or "", message))
        self.progress.seconds = self.now()

    def check(self) -> None:
        if self.should_stop():
            raise _Stopped

    def step(self, key: StepKey) -> AutoStep:
        found = next(s for s in self.progress.steps if s.key == key)
        found.status, self.progress.step = "running", key
        return found

    def close(self, step: AutoStep, began: float, status: str = "done", note: str = "") -> None:
        step.status = cast(Any, status)
        step.seconds = round(self.now() - began, 2)
        if note:
            step.note = note
        self.progress.seconds = self.now()

    def sign(self) -> float:
        return 1.0 if self.progress.greater_is_better else -1.0

    def order(self, score: float | None) -> float:
        """Sort key: the best score first, a missing one last."""
        return math.inf if score is None else -self.sign() * score

    def better(self, a: float | None, b: float | None) -> bool:
        if a is None:
            return False
        return b is None or self.sign() * (a - b) > 0

    def train(
        self,
        key: str,
        settings: dict[str, Any] | None,
        data: pd.DataFrame,
        validation: evaluate.Validation,
        limit: float,
        build: Callable[[int], Pipeline] | None = None,
    ) -> trainer.Trained:
        """One run; stopped at its next checkpoint by the user or past `limit`."""
        return trainer.train(
            data,
            self.fields,
            self.target,
            self.task,
            key,
            settings,
            self.rows,
            self.sampled,
            choices=self.choices,
            classes=self.classes,
            validation=validation,
            should_stop=lambda: self.should_stop() or self.clock() > limit,
            build=build,
        )

    # --- Scores on the Auto's folds ---------------------------------------------------------------

    def _score_at(
        self, held: trainer.FoldPredictions, values: np.ndarray, at: np.ndarray
    ) -> float | None:
        metric = self.progress.metric
        if self.task == "regression":
            actual = held.actual[at].astype(float)
            if held.ranks is not None:
                scorer = panels.regression_scorer(metric, actual, values[at, 0])
                return self._criterion(held.ranks[at], held.folds[at], scorer)
            return evaluate.regression_metric(metric, actual, values[at, 0])
        chances = values[at]
        labels = np.asarray(held.labels)[chances.argmax(axis=1)]
        truth = held.actual[at].astype(str)
        if held.ranks is not None:
            scorer = panels.classification_scorer(
                metric, truth, labels, chances, held.labels, self.positive
            )
            return self._criterion(held.ranks[at], held.folds[at], scorer)
        return evaluate.classification_metric(
            metric, truth, labels, chances, held.labels, self.positive
        )

    def _criterion(
        self, ranks: np.ndarray, folds: np.ndarray, scorer: panels.Scorer
    ) -> float | None:
        found = panels.score_periods(ranks, folds, {}, scorer)
        summary = panels.summarise(found, self.progress.metric, self.validation.period_threshold)
        return panels.criterion_value(summary, self.validation.criterion)

    def fold_scores(self, held: trainer.FoldPredictions, values: np.ndarray) -> list[float]:
        found = []
        for fold in np.unique(held.folds):
            value = self._score_at(held, values, np.flatnonzero(held.folds == fold))
            if value is not None and math.isfinite(value):
                found.append(float(value))
        return found

    def overall(self, held: trainer.FoldPredictions, values: np.ndarray) -> float | None:
        """The score as a run gives it: the mean over the folds (a panel: the criterion over
        every test period)."""
        if held.ranks is not None:
            return self._score_at(held, values, np.arange(len(held.folds)))
        scores = self.fold_scores(held, values)
        return float(np.mean(scores)) if scores else None

    @staticmethod
    def spread(scores: list[float]) -> float | None:
        return float(np.std(scores, ddof=1)) if len(scores) > 1 else None

    # --- Entries ---------------------------------------------------------------------------------

    def entry(
        self, id_: str, key: str, step: str, settings: dict[str, Any] | None = None
    ) -> AutoEntry:
        found = next((e for e in self.progress.entries if e.id == id_), None)
        if found is None:
            spec = catalogue.CATALOGUE.get(key)
            label = spec.label if spec else ("Constant" if key == CONSTANT else ENSEMBLE_LABEL)
            family = spec.family if spec else ("Baseline" if key == CONSTANT else "Ensemble")
            if id_.endswith(":tuned"):
                label += " · tuned"
            found = AutoEntry(id_, key, label, family, step, settings or {})
            self.progress.entries.append(found)
        found.step, found.status = step, "running"
        return found

    def best(self, fitted: bool = False) -> tuple[AutoEntry, trainer.Trained, float] | None:
        """The best model so far, on any share of the rows (`fitted`: one whose pipeline is
        still in memory)."""
        found = [v for v in self.latest.values() if not fitted or v[1].pipeline is not None]
        found.sort(key=lambda v: self.order(v[1].score))
        return found[0] if found else None

    def _keep(self, entry: AutoEntry, trained: trainer.Trained, share: float) -> None:
        """Remember the run; only the 3 best keep their fitted pipeline in memory."""
        self.latest[entry.id] = (entry, trained, share)
        ranked = sorted(self.latest.values(), key=lambda v: self.order(v[1].score))
        for _, other, _ in ranked[TOP_FAMILIES:]:
            other.pipeline = None

    def record(
        self,
        entry: AutoEntry,
        trained: trainer.Trained,
        share: float,
        full: bool,
        track: bool = True,
    ) -> None:
        held = trained.fold_predictions
        entry.status, entry.share, entry.rows = "done", share, trained.train_rows
        entry.seconds = round(entry.seconds + trained.seconds, 3)
        entry.score = trained.score
        entry.std = self.spread(self.fold_scores(held, held.values)) if held is not None else None
        if not self.progress.metric:
            self.progress.metric = trained.primary
        if entry.model == CONSTANT:
            return
        if track:
            self._keep(entry, trained, share)
        if not full or held is None:
            return
        self.pool.append(
            _Candidate(
                entry,
                trained.model,
                entry.settings,
                held,
                trained.seconds,
                trained.score,
                trained.train_rows,
            )
        )

    def attempt(
        self,
        entry: AutoEntry,
        key: str,
        settings: dict[str, Any] | None,
        data: pd.DataFrame,
        validation: evaluate.Validation,
        limit: float,
        share: float,
        full: bool,
        build: Callable[[int], Pipeline] | None = None,
        track: bool = True,
    ) -> trainer.Trained | None:
        """Train one entry; a failure is noted and the Auto goes on."""
        self.check()
        began = self.clock()
        entry.status = "running"
        try:
            trained = self.train(key, settings, data, validation, limit, build)
        except trainer.TrainingCancelled:
            self.check()
            entry.status, entry.note = "stopped", "Stopped by the time budget."
            entry.seconds = round(entry.seconds + self.clock() - began, 3)
            self.log(f"{entry.label}: stopped by the time budget.")
            return None
        except Exception as error:  # one model failing never stops the others
            entry.status, entry.note = "failed", str(error) or error.__class__.__name__
            self.log(f"{entry.label} failed: {entry.note}")
            return None
        self.record(entry, trained, share, full, track)
        on = "" if share >= 1 else f" on {share:.0%} of the {self.unit}"
        self.log(f"{entry.label}: {self.progress.metric} {_fmt(entry.score)}{on}.")
        return trained

    # --- The run ---------------------------------------------------------------------------------

    def run(self) -> AutoOutcome:
        try:
            self.prepare()
            self.baselines()
            ranked = self.families()
            top = self.reduction(ranked)
            self.tuning(top)
            self.finish()
        except _Stopped:
            self.progress.stopped = True
            step = next((s for s in self.progress.steps if s.status == "running"), None)
            if step is not None:
                step.status = "stopped"
            for entry in self.progress.entries:
                if entry.status == "running":
                    entry.status, entry.note = "stopped", "Stopped."
            best = self.best(fitted=True)
            kept = best[0].label if best else None
            self.log(
                f"Stopped: the best model so far ({kept}) is kept as a run."
                if kept
                else "Stopped before any model was trained on every row."
            )
            if best is not None:
                self.progress.best = best[0].id
                self._gain(best[1].score)
            self.final = best[1] if best else None
        self.progress.seconds = self.now()
        self.progress.step = None
        self.progress.finished = True
        return AutoOutcome(self.progress, self.final)

    def prepare(self) -> None:
        validation = self.validation
        loaded, task = self.engine._load(
            self.fields,
            self.target,
            self.source,
            validation.max_rows,
            _period(validation, self.fields),
        )
        if self.classes is not None:
            task = "binary" if len(self.classes.names) == 2 else "multiclass"
        self.task: Task = task
        self.rows, self.sampled = loaded.rows, loaded.sampled
        self.data = loaded.data
        series = cast(pd.Series, self.data[self.target.name])
        minority = None
        if task != "regression":
            labels = (
                binning.apply(series, self.classes).astype(str)
                if self.classes is not None
                else trainer.prepare_target(series, task)
            )
            shares = labels.value_counts(normalize=True)
            minority = float(shares.min())
            self.positive: str | None = str(labels.value_counts().index[-1])
        else:
            self.positive = None
        self.minority = minority
        self.reference = catalogue.default_model(task)
        metric = validation.metric or evaluate.default_metric(task, minority, validation.panel)
        self.progress.metric = metric
        self.progress.greater_is_better = metric not in evaluate.LOWER_IS_BETTER
        self.progress.criterion = validation.criterion if validation.panel else None
        self.unit = "periods" if validation.panel else "rows"
        self.inner = not validation.cross
        if validation.method == "loo":
            # one fit per row for every model: the Auto compares on 5 folds instead
            self.search = validation.model_copy(
                update={"method": "kfold", "folds": ENSEMBLE_FOLDS, "threshold": "default"}
            )
            self.final_validation = self.search
            self.inner = False
            self.progress.compared_on = (
                "Leave-one-out would fit every model once per row: every model is compared, "
                "and the kept one judged, on 5 folds."
            )
            self.search_data = self.data
        elif self.inner:
            self.search = validation.model_copy(
                update={"method": "kfold", "folds": ENSEMBLE_FOLDS, "threshold": "default"}
            )
            self.final_validation = validation
            setup = trainer._setup(
                self.data,
                self.fields,
                self.target,
                task,
                self.reference,
                None,
                None,
                self.choices,
                self.classes,
                validation,
            )
            held = set(int(v) for v in setup.row_test)
            row = cast(pd.Series, self.data["__row"])
            self.search_data = cast(pd.DataFrame, self.data[~row.isin(held)])
            split = (
                "time-based split" if validation.method == "time_split" else "train / test split"
            )
            self.progress.compared_on = (
                f"Your validation is a {split}: every model, the ensemble included, is compared "
                f"on {ENSEMBLE_FOLDS} folds of the training rows (out-of-fold predictions are "
                "needed for the ensemble); the test rows only judge the model kept."
            )
            if validation.method == "time_split":
                self.progress.compared_on += " Those folds do not follow the time order."
        else:
            self.search = validation
            self.final_validation = validation
            self.search_data = self.data
            if validation.panel:
                self.progress.compared_on = (
                    f"Walk-forward by period on {validation.folds} folds (embargo "
                    f"{validation.embargo}), ranked by the {validation.criterion} criterion "
                    "over the test periods, at every step."
                )
            else:
                self.progress.compared_on = (
                    f"Your validation ({validation.method.replace('_', ' ')}, "
                    f"{validation.folds} folds) at every step."
                )
        self.log(self.progress.compared_on)

    def baselines(self) -> None:
        step, began = self.step("baselines"), self.now()
        limit = self.deadline
        constant = self.entry(CONSTANT, CONSTANT, "baselines")
        self.attempt(
            constant,
            self.reference,
            None,
            self.search_data,
            self.search,
            limit,
            1.0,
            True,
            constant_model(self.task),
        )
        linear = self.entry(self.reference, self.reference, "baselines")
        self.reference_entry = linear
        self.attempt(linear, self.reference, None, self.search_data, self.search, limit, 1.0, True)
        self.close(step, began)

    def _rungs(self) -> list[tuple[float, pd.DataFrame, evaluate.Validation]]:
        """The rows (or periods) of each round of the successive elimination: a quarter, half,
        then all of them; a single round on small data."""
        full = (1.0, self.search_data, self.search)
        if self.validation.panel:
            ranks, _ = panels.period_ranks(
                self.search_data, cast(str, self.validation.period_column)
            )
            periods = len(np.unique(ranks[ranks >= 0]))
            base = self.validation.sample_periods or (
                panels.DEFAULT_STEP if periods > panels.MANY_PERIODS else 1
            )
            if periods < PANEL_RUNG_PERIODS * base:
                return [full]
            return [
                (
                    1 / f,
                    self.search_data,
                    self.search.model_copy(update={"sample_periods": base * f}),
                )
                for f in (4, 2)
            ] + [full]
        n = len(self.search_data)
        if n < 4 * RUNG_ROWS:
            return [full]
        return [
            (1 / f, self.search_data.sample(n // f, random_state=self.validation.seed), self.search)
            for f in (4, 2)
        ] + [full]

    def families(self) -> list[AutoEntry]:
        step, began = self.step("families"), self.now()
        self.rungs = self._rungs()
        share, data, validation = self.rungs[0]
        keys = self._candidates()
        context = catalogue.Context(
            len(self.search_data), self.task, self.minority, features=len(self.fields) - 1
        )
        ready: list[tuple[float, str]] = []
        for key in keys:
            spec = catalogue.CATALOGUE[key]
            reason = None
            if self.task not in spec.tasks:
                reason = f"It does not handle {self.task} problems."
            elif not spec.installed:
                reason = "Not installed in this build."
            else:
                reason = spec.limit(context)
            if reason:
                skipped = self.entry(key, key, "families")
                skipped.status, skipped.note = "skipped", reason
                self.log(f"{skipped.label} skipped: {reason}")
                continue
            self.check()
            try:
                found = trainer.estimate(
                    data,
                    self.fields,
                    self.target,
                    self.task,
                    key,
                    None,
                    self.choices,
                    self.classes,
                    validation,
                ).seconds
            except Exception as error:
                failed = self.entry(key, key, "families")
                failed.status, failed.note = "failed", str(error)
                self.log(f"{failed.label} failed: {error}")
                continue
            ready.append((found, key))
        ready.sort()
        self.log(
            "Order by estimated duration: "
            + ", ".join(f"{catalogue.CATALOGUE[k].label} ~{_seconds(s)}" for s, k in ready)
            + "."
        )
        allowance = self.start + FAMILY_SHARE * self.budget
        reference_entry = self.reference_entry
        tried: list[AutoEntry] = []
        if share < 1:  # the reference on the same rows as the others, to rank it among them
            self.attempt(
                reference_entry,
                self.reference,
                None,
                data,
                validation,
                self.deadline,
                share,
                False,
                track=False,  # its run on every row (a baseline) stays the one kept
            )
            if reference_entry.score is not None:
                tried.append(reference_entry)
        elif reference_entry.status == "done":
            tried.append(reference_entry)
        for estimate, key in ready:
            entry = self.entry(key, key, "families")
            entry.estimate = round(estimate, 2)
            if estimate > SLOW_SHARE * self.budget:
                entry.status = "skipped"
                entry.note = (
                    f"Too slow for the budget: about {_seconds(estimate)} on these rows, more "
                    "than a quarter of it."
                )
                self.log(f"{entry.label} skipped: {entry.note}")
                continue
            if self.clock() + estimate > allowance:
                entry.status = "skipped"
                entry.note = f"No time left in this step (about {_seconds(estimate)} needed)."
                self.log(f"{entry.label} skipped: {entry.note}")
                continue
            limit = min(self.deadline, max(allowance, self.clock() + 2 * estimate))
            if self.attempt(entry, key, None, data, validation, limit, share, share >= 1):
                tried.append(entry)
        self.close(step, began, note=f"{len(tried)} families compared.")
        return self._ranked(tried)

    def _candidates(self) -> list[str]:
        if self.request.models is not None:
            keys = list(dict.fromkeys(self.request.models))
        else:
            context = catalogue.Context(
                len(self.search_data), self.task, self.minority, features=len(self.fields) - 1
            )
            keys = self.engine.catalogue(context).compare
        return [k for k in keys if k != self.reference]

    def _ranked(self, entries: list[AutoEntry]) -> list[AutoEntry]:
        scored = [e for e in entries if e.status == "done" and e.score is not None]
        return sorted(scored, key=lambda e: -self.sign() * cast(float, e.score))

    def reduction(self, ranked: list[AutoEntry]) -> list[AutoEntry]:
        """Successive elimination over the rounds, then the best 3 on every row."""
        step, began = self.step("reduction"), self.now()
        allowance = self.start + REDUCTION_SHARE * self.budget
        alive = ranked
        for index, (share, data, validation) in enumerate(self.rungs[1:], start=1):
            last = index == len(self.rungs) - 1
            if last:  # every row for the 3 best: what a result needs, so later in the budget
                allowance = self.deadline - FINAL_SHARE * self.budget
            keep = TOP_FAMILIES if last else max(TOP_FAMILIES, math.ceil(len(alive) / 2))
            for dropped in alive[keep:]:
                dropped.note = f"Eliminated on {self.rungs[index - 1][0]:.0%} of the {self.unit}."
            if alive[keep:]:
                self.log("Eliminated: " + ", ".join(e.label for e in alive[keep:]) + ".")
            alive = alive[:keep]
            survivors: list[AutoEntry] = []
            for entry in alive:
                if last and entry is self.reference_entry and entry in self.pool_entries():
                    self._restore(entry)  # already trained on every row as a baseline
                    survivors.append(entry)
                    continue
                cost = entry.seconds * (share / self.rungs[index - 1][0]) * 1.2
                if self.clock() + cost > allowance:
                    entry.note = f"No time left to train it on {share:.0%} of the {self.unit}."
                    self.log(f"{entry.label}: {entry.note}")
                    survivors.append(entry)
                    continue
                limit = min(self.deadline, max(allowance, self.clock() + 2 * cost))
                self.attempt(entry, entry.model, None, data, validation, limit, share, last)
                survivors.append(entry)
            alive = self._ranked(survivors) + [e for e in survivors if e.score is None]
        if self.reference_entry in self.pool_entries() and self.reference_entry.share < 1:
            self._restore(self.reference_entry)
        top = alive[:TOP_FAMILIES]
        if top:
            self.log("Kept for tuning: " + ", ".join(e.label for e in top) + ".")
        self.close(step, began, note=", ".join(e.label for e in top))
        return top

    def pool_entries(self) -> list[AutoEntry]:
        return [c.entry for c in self.pool]

    def _restore(self, entry: AutoEntry) -> None:
        """The reference was trained on every row as a baseline: show that result again."""
        found = next(c for c in self.pool if c.entry is entry)
        entry.status, entry.share, entry.rows = "done", 1.0, found.rows
        entry.score = found.score
        entry.std = self.spread(self.fold_scores(found.held, found.held.values))

    # --- Tuning ----------------------------------------------------------------------------------

    def _ensemble_cost(self, members: list[_Candidate], stacking: bool) -> float:
        """Seconds the ensemble's run would take: every member's fits under the user's
        validation (a split: one fit each instead of the Auto's 6)."""
        threshold = 4 if self.final_validation.threshold != "default" else 1
        stack = STACK_FOLDS + 1 if stacking else 1
        single = sum(c.seconds for c in members) / (self._fits() if self.inner else 1)
        return single * stack * threshold

    def _single_cost(self) -> float:
        """Seconds the kept single model's final run takes: with a train / test split, one fit
        on the training rows; a model compared on part of the rows, its run on all of them."""
        best = self.best()
        if best is None:
            return 0.0
        _, trained, share = best
        full = trained.seconds / max(share, 0.25)
        if not self.inner:
            return 0.0 if share >= 1 and trained.pipeline is not None else full
        threshold = 4 if self.final_validation.threshold != "default" else 1
        return full / self._fits() * threshold

    def _fits(self) -> int:
        return self.search.folds + 1

    def tuning(self, top: list[AutoEntry]) -> None:
        step, began = self.step("tuning"), self.now()
        members = sorted(self.pool, key=lambda c: self.order(c.entry.score))
        reserve = RESERVE_SHARE * self.budget + RESERVE * (
            self._ensemble_cost(members[:TOP_FAMILIES], self.request.stacking) + self._single_cost()
        )
        end = self.deadline - reserve
        tuned = 0
        tunable: list[tuple[AutoEntry, float]] = []
        for entry in top:
            if not tuner.default_space(entry.model):
                self.log(f"{entry.label}: nothing to tune.")
                continue
            tunable.append((entry, self._refit(entry)))
        # a family whose share would not even pay for its run on every row leaves its time to
        # the others (the worst ranked first)
        while tunable and self.request.trials is None:
            left = end - self.clock()
            weights = TUNE_SHARES[: len(tunable)]
            short = [
                i
                for i, (_, refit) in enumerate(tunable)
                if left * weights[i] / sum(weights) - refit < max(1.0, 0.75 * refit)
            ]
            if not short:
                break
            dropped, _ = tunable.pop(short[-1])
            self.log(f"{dropped.label}: too little time left to tune it.")
        shares = list(TUNE_SHARES[: len(tunable)])
        for index, (entry, refit) in enumerate(tunable):
            self.check()
            key = entry.model
            left = end - self.clock()
            search = left * shares[index] / sum(shares[index:]) - refit
            if self.request.trials is None and search < 1.0:
                self.log(f"{entry.label}: too little time left to tune it.")
                continue
            if self.request.trials is not None and left < refit:
                self.log(f"{entry.label}: too little time left to tune it.")
                continue
            stop_at = self.clock() + search if self.request.trials is None else end - refit
            progress = self._search(key, stop_at)
            if progress is None:
                continue
            self.progress.trials += len(progress.trials)
            if progress.best_value is None or not self.better(
                progress.best_value, progress.default_value
            ):
                self.log(
                    f"{entry.label}: the defaults stayed best after {len(progress.trials)} trials."
                )
                continue
            settings = dict(progress.best_params)
            found = self.entry(f"{key}:tuned", key, "tuning", settings)
            found.note = f"{len(progress.trials)} trials."
            self.log(
                f"{entry.label}: {len(progress.trials)} trials, best {_fmt(progress.best_value)} "
                f"against {_fmt(progress.default_value)} with the defaults (3 folds)."
            )
            if self.attempt(
                found,
                key,
                settings,
                self.search_data,
                self.search,
                end + GRACE * self.budget,
                1.0,
                True,
            ):
                tuned += 1
        self.close(
            step,
            began,
            "done" if tuned or top else "skipped",
            f"{self.progress.trials} trials, {tuned} improved.",
        )

    def _refit(self, entry: AutoEntry) -> float:
        """Seconds a run of this model on every row of the Auto's folds takes."""
        found = self.latest.get(entry.id)
        if found is None:
            return entry.seconds
        _, trained, share = found
        return trained.seconds / max(share, 0.25)

    def _search(self, key: str, stop_at: float) -> tuner.TuneProgress | None:
        """Optuna on one family, on the Auto's training rows (walk-forward on a panel)."""
        try:
            setup = trainer._setup(
                self.search_data,
                self.fields,
                self.target,
                self.task,
                key,
                None,
                None,
                self.choices,
                self.classes,
                self.search,
            )
            metric = self.progress.metric
            progress = tuner.TuneProgress(metric, self.progress.greater_is_better)
            search_panel = None
            if setup.periods is not None:
                folds = panels.plan(
                    setup.periods.ranks,
                    setup.periods.labels,
                    tuner.FOLDS,
                    self.search.embargo,
                    self.search.window,
                    self.search.window_periods,
                    self.search.sample_periods,
                )
                search_panel = tuner.PanelSearch(
                    folds.folds,
                    setup.periods.ranks,
                    self.search.criterion,
                    self.search.period_threshold,
                )
                progress.criterion = self.search.criterion

            def build(values: dict[str, Any], rows: int) -> Any:
                return trainer.model_pipeline(
                    setup.spec,
                    values,
                    setup.context,
                    setup.preparation,
                    setup.task,
                    rows,
                    setup.unregularized,
                )

            request = tuner.TuneRequest(model=key, budget_seconds=None, trials=self.request.trials)
            tuner.tune(
                setup.x_train,
                setup.y_train,
                setup.task,
                setup.spec,
                setup.context,
                setup.preparation,
                request,
                metric,
                progress,
                lambda: self.should_stop() or self.clock() > stop_at,
                build,
                search_panel,
            )
        except (tuner.TuneError, trainer.TrainingError, evaluate.ValidationError) as error:
            self.log(f"{catalogue.CATALOGUE[key].label}: not tuned ({error}).")
            return None
        self.check()
        return progress

    # --- Ensemble and the kept model -------------------------------------------------------------

    def _ensemble_pool(self) -> list[_Candidate]:
        if not self.pool:
            return []
        ordered = sorted(self.pool, key=lambda c: self.order(c.entry.score))
        first = ordered[0].held
        same = [
            c
            for c in ordered
            if len(c.held.rows) == len(first.rows)
            and np.array_equal(c.held.rows, first.rows)
            and np.array_equal(c.held.folds, first.folds)
            and c.held.labels == first.labels
        ]
        return same[:POOL]

    def finish(self) -> None:
        step, began = self.step("ensemble"), self.now()
        pool = self._ensemble_pool()
        result = None
        if len(pool) >= 2:
            result = self._ensemble(pool)
            self.progress.ensemble = result
        else:
            self.log("The ensemble needs at least two models trained on every row: skipped.")
        best = self.best()
        if (
            result is not None
            and result.kept
            and best is not None
            and not kept_ensemble(
                result.score, best[1].score, None, self.progress.greater_is_better
            )
        ):
            result.kept = False
            result.reason = (
                f"{best[0].label} (on {best[2]:.0%} of the {self.unit}) scores "
                f"{_fmt(best[1].score)}, above the ensemble: the single model is kept."
            )
            self.log(result.reason)
        kept = result is not None and result.kept
        if kept and result is not None:
            trained = self._final_ensemble(pool, result)
            if trained is not None:
                entry = self.entry(ENSEMBLE, ENSEMBLE, "ensemble")
                entry.status, entry.score, entry.std = "done", result.score, result.std
                entry.seconds, entry.rows = round(trained.seconds, 3), trained.train_rows
                entry.settings = {m.label: m.weight for m in result.members}
                self.final = trained
                self.progress.best = ENSEMBLE
                self._gain(result.score)
                self._test_score(trained)
                self.close(step, began, note="Ensemble kept.")
                return
            result.kept = False
            result.reason = "No time left to train the ensemble: the best single model is kept."
            self.log(result.reason)
        self.close(
            step,
            began,
            "done" if result is not None else "skipped",
            "Single model recommended."
            if result is not None
            else "Needs two models trained on every row.",
        )
        self._final_single()

    def _ensemble(self, pool: list[_Candidate]) -> EnsembleResult:
        held = pool[0].held
        values = [c.held.values.astype(float) for c in pool]
        counts = greedy(values, lambda v: self.overall(held, v), self.progress.greater_is_better)
        total = sum(counts)
        blended = sum(v * c for v, c in zip(values, counts, strict=True)) / total
        members = [
            Member(c.entry.id, c.key, c.entry.label, float(n), c.settings)
            for c, n in zip(pool, counts, strict=True)
            if n
        ]
        score = self.overall(held, cast(np.ndarray, blended))
        spread = self.spread(self.fold_scores(held, cast(np.ndarray, blended)))
        best = pool[0]
        best_score = self.overall(held, best.held.values)
        best_std = self.spread(self.fold_scores(held, best.held.values))
        method: Literal["greedy", "stacking"] = "greedy"
        stacking_score = None
        if self.request.stacking and len(pool) >= 2:
            if self.validation.panel:
                self.log(
                    "Stacking needs shuffled folds: on a panel, only the greedy ensemble runs."
                )
            else:
                stacked, weights = self._stacked(pool)
                stacking_score = self.overall(held, stacked)
                self.log(f"Linear stack of {len(pool)} models: {_fmt(stacking_score)}.")
                if self.better(stacking_score, score):
                    method, score = "stacking", stacking_score
                    spread = self.spread(self.fold_scores(held, stacked))
                    members = [
                        Member(c.entry.id, c.key, c.entry.label, round(w, 4), c.settings)
                        for c, w in zip(pool, weights, strict=True)
                    ]
        names = describe(members, method, " + ")
        greater = self.progress.greater_is_better
        if len(members) < 2 and method == "greedy":
            kept = False
            reason = (
                f"The greedy selection kept {best.entry.label} alone: the single model is kept."
            )
        elif not kept_ensemble(score, best_score, best_std, greater):
            kept = False
            spread_text = f"±{_fmt(best_std)}" if best_std is not None else "its spread"
            reason = (
                f"The ensemble ({names}) scores {_fmt(score)} against {_fmt(best_score)} for "
                f"{best.entry.label} alone: not better by more than one standard deviation "
                f"({spread_text}), so the single model is recommended, simpler and as good."
            )
        else:
            kept = True
            reason = (
                f"The ensemble ({names}) scores {_fmt(score)} against {_fmt(best_score)} for "
                f"{best.entry.label} alone, more than one standard deviation better: kept."
            )
        self.log(reason)
        return EnsembleResult(
            method,
            members,
            score,
            spread,
            best.entry.id,
            best_score,
            best_std,
            kept,
            reason,
            stacking_score,
        )

    def _meta_features(self, pool: list[_Candidate]) -> np.ndarray:
        columns = []
        for c in pool:
            values = c.held.values.astype(float)
            if self.task == "binary":
                columns.append(values[:, 1:2])
            else:
                columns.append(values)
        return np.hstack(columns)

    def _stacked(self, pool: list[_Candidate]) -> tuple[np.ndarray, list[float]]:
        """A regularized linear model on the out-of-fold predictions, cross-fitted: each fold's
        stacked predictions come from a stack learnt on the other folds."""
        held = pool[0].held
        features = self._meta_features(pool)
        out = np.zeros_like(held.values, dtype=float)
        for fold in np.unique(held.folds):
            at, rest = held.folds == fold, held.folds != fold
            meta = self._meta()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                meta.fit(features[rest], self._meta_target(held, rest))
            out[at] = self._meta_predict(meta, features[at], held.labels)
        final = self._meta()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            final.fit(features, self._meta_target(held, np.ones(len(held.folds), dtype=bool)))
        # each member's share: the size of its coefficients (its columns of the features)
        coef = np.abs(np.atleast_2d(np.asarray(final.coef_, dtype=float))).sum(axis=0)
        coef = coef.reshape(len(pool), -1).sum(axis=1)
        shares = coef / coef.sum() if coef.sum() > 0 else np.full(len(pool), 1 / len(pool))
        return out, [float(v) for v in shares]

    def _meta(self) -> Any:
        if self.task == "regression":
            return RidgeCV(alphas=ALPHAS)
        return LogisticRegression(max_iter=1_000)

    def _meta_target(self, held: trainer.FoldPredictions, at: np.ndarray) -> np.ndarray:
        actual = held.actual[at]
        return actual.astype(float) if self.task == "regression" else actual.astype(str)

    def _meta_predict(self, meta: Any, features: np.ndarray, labels: list[str]) -> np.ndarray:
        if self.task == "regression":
            return np.asarray(meta.predict(features), dtype=float).reshape(-1, 1)
        found = np.asarray(meta.predict_proba(features), dtype=float)
        out = np.zeros((len(features), len(labels)))
        for j, label in enumerate(str(c) for c in meta.classes_):
            out[:, labels.index(label)] = found[:, j]
        return out

    def _member_factories(
        self, data: pd.DataFrame, members: list[Member]
    ) -> list[tuple[str, Callable[[int], Pipeline]]]:
        found = []
        for i, member in enumerate(members):
            setup = trainer._setup(
                data,
                self.fields,
                self.target,
                self.task,
                member.model,
                member.settings or None,
                None,
                self.choices,
                self.classes,
                self.final_validation,
            )

            def factory(rows: int, setup: Any = setup) -> Pipeline:
                return trainer.model_pipeline(
                    setup.spec,
                    setup.values,
                    setup.context,
                    setup.preparation,
                    setup.task,
                    rows,
                    setup.unregularized,
                )

            found.append((f"m{i + 1}_{member.model}", factory))
        return found

    def _ensemble_build(
        self, data: pd.DataFrame, result: EnsembleResult
    ) -> Callable[[int], Pipeline]:
        factories = self._member_factories(data, result.members)
        weights = [m.weight for m in result.members]
        seed = self.validation.seed
        regression, stacking = self.task == "regression", result.method == "stacking"

        def build(rows: int) -> Pipeline:
            estimators = [(name, make(rows)) for name, make in factories]
            model: Any
            if stacking and regression:
                model = StackingRegressor(
                    estimators,
                    final_estimator=RidgeCV(alphas=ALPHAS),
                    cv=KFold(STACK_FOLDS, shuffle=True, random_state=seed),
                )
            elif stacking:
                model = StackingClassifier(
                    estimators,
                    final_estimator=LogisticRegression(max_iter=1_000),
                    cv=StratifiedKFold(STACK_FOLDS, shuffle=True, random_state=seed),
                    stack_method="predict_proba",
                )
            elif regression:
                model = VotingRegressor(estimators, weights=weights)
            else:
                model = VotingClassifier(estimators, voting="soft", weights=weights)
            return Pipeline([("prepare", identity()), ("model", model)])

        return build

    def _final_ensemble(
        self, pool: list[_Candidate], result: EnsembleResult
    ) -> trainer.Trained | None:
        """The ensemble as a run, judged by the user's own validation."""
        self.check()
        by_entry = {c.entry.id: c for c in pool}
        members = [by_entry[m.entry] for m in result.members]
        cost = self._ensemble_cost(members, result.method == "stacking")
        limit = self.deadline + GRACE * self.budget
        if self.clock() + cost > limit:
            return None
        data = self.data if self.inner else self.search_data
        top = result.members[0]
        try:
            trained = self.train(
                top.model,
                top.settings or None,
                data,
                self.final_validation,
                limit,
                self._ensemble_build(data, result),
            )
        except trainer.TrainingCancelled:
            self.check()
            return None
        except Exception as error:
            self.log(f"The ensemble could not be trained: {error}")
            return None
        trained.model = ENSEMBLE
        trained.settings = {
            "method": result.method,
            "members": describe(result.members, result.method, ", "),
        }
        self.log(f"Ensemble trained as a run in {_seconds(trained.seconds)}.")
        return trained

    def _final_single(self) -> None:
        """The best single model as the run: as it is when it was judged on every row of the
        user's own validation, else trained again (a split: on the training rows, judged once on
        the test rows; part of the rows: on all of them)."""
        found = self.best()
        if found is None:
            self.log("No model could be trained.")
            return
        entry, best, share = found
        self.progress.best = entry.id
        self._gain(best.score)
        if not self.inner and share >= 1 and best.pipeline is not None:
            self.final = best
            self.log(f"Kept: {entry.label}.")
            return
        data, validation = (
            (self.data, self.final_validation) if self.inner else (self.search_data, self.search)
        )
        limit = self.deadline + GRACE * self.budget
        try:
            trained = self.train(best.model, entry.settings or None, data, validation, limit)
        except Exception as error:
            self.check()
            fallback = self.best(fitted=True)
            self.final = fallback[1] if fallback else None
            if fallback is not None:
                self.progress.best = fallback[0].id
                self._gain(fallback[1].score)
            why = "no time left" if isinstance(error, trainer.TrainingCancelled) else str(error)
            kept = fallback[0].label if fallback else "nothing"
            self.log(f"Kept: {kept}, as judged by the Auto ({why} for its final run).")
            return
        if not self.inner:
            self.record(entry, trained, 1.0, True)
            self._gain(trained.score)
        self.final = trained
        self._test_score(trained)
        self.log(f"Kept: {entry.label}.")

    def _test_score(self, trained: trainer.Trained) -> None:
        if self.inner:
            self.progress.test_score = trained.score
            self.log(f"On the test rows: {trained.primary} {_fmt(trained.score)}.")

    def _gain(self, best: float | None) -> None:
        reference = self.reference_entry
        candidate = next((c for c in self.pool if c.entry is reference), None)
        value = (
            self.overall(candidate.held, candidate.held.values) if candidate is not None else None
        )
        label = "linear model" if self.task == "regression" else "logistic regression"
        self.progress.gain = gain_sentence(
            self.progress.metric, self.progress.criterion, best, value, label
        )


def describe(members: list[Member], method: str, separator: str) -> str:
    """The members in words: each with its weight (greedy selection), or the stacked models."""
    times = chr(0xD7)  # the multiplication sign
    return separator.join(
        f"{m.label} {times}{m.weight:g}" if method == "greedy" else m.label for m in members
    )


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4g}"


def _seconds(value: float) -> str:
    return f"{value:.1f} s" if value < 90 else f"{value / 60:.1f} min"


def _period(validation: evaluate.Validation, fields: list[DataField]) -> str | None:
    if not validation.panel or validation.period_column is None:
        return None
    return (
        validation.period_column if validation.period_column in {f.name for f in fields} else None
    )


def run(
    engine: MlEngine,
    fields: list[DataField],
    target: DataField,
    source: str,
    request: AutoRequest,
    classes: binning.Classes | None,
    validation: evaluate.Validation,
    should_stop: Callable[[], bool] = lambda: False,
    started: Callable[[AutoProgress], None] = lambda _progress: None,
    clock: Callable[[], float] = time.monotonic,
) -> AutoOutcome:
    """Run the Auto within `request.minutes`. `started` receives the progress object, updated
    step after step; `should_stop` stops it, the best model so far being kept."""
    unknown = [k for k in request.models or [] if k not in catalogue.CATALOGUE]
    if unknown:
        raise AutoError(f"Unknown model(s): {', '.join(unknown)}.")
    auto = _Auto(engine, fields, target, source, request, classes, validation, should_stop, clock)
    started(auto.progress)
    return auto.run()
