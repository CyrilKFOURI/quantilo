"""Machine learning on one target, for the library: the app's ML page as methods.

ml = ds.ml("Survived")
model = ml.train()                 # preparation and model chosen by the app's rules
model.score, model.classification  # held-out results
model.explain()                    # permutation importance, SHAP, partial dependence
model.check(new_rows)              # a Dataset or a pandas DataFrame: Ready, Check or Blocked
model.predict(new_rows)            # predictions, intervals or probabilities, reliability (037)
model.export("model.zip")          # reloaded with dx.load_model("model.zip")

Panel data (035): `ds.ml("target", validation={"method": "panel_walk_forward", "period_column":
"era", "embargo": 4})`, then `model.panel` (scores per period, summary, folds, exposures, alerts)
and `model.neutralize(["feature_1"], 1.0)`.

Where it is wrong (036): `model.errors()` (subgroups, error by column, worst predictions, hints),
`model.compare_errors(other)`, `model.fairness("sex")`.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from quantilo.filters.model import Condition
from quantilo.ml import auto as automl
from quantilo.ml import classes as binning
from quantilo.ml import error_analysis as mistakes
from quantilo.ml import evaluate
from quantilo.ml import explain as explainer
from quantilo.ml import models as catalogue
from quantilo.ml import panel as panels
from quantilo.ml import predict as use
from quantilo.ml import preparation as prep
from quantilo.ml import train as trainer
from quantilo.ml import tune as tuner
from quantilo.ml.catalog import Context
from quantilo.ml.data import NUMERIC, Field
from quantilo.ml.engine import (
    Catalogue,
    MlEngine,
    MlError,
    NeutralizeRequest,
    TuneStart,
)
from quantilo.ml.saved import Predicting

if TYPE_CHECKING:
    pass


class TrainedModel(Predicting):
    """A trained model: the app's run (scores, predictions, coefficients, importances…) with its
    fitted pipeline. Every field of the result is an attribute (`model.score`, `model.task`).
    On new data (037): `check`, `predict`, `prediction`, `explain_row`, `export`."""

    def __init__(
        self,
        engine: MlEngine,
        dataset_id: str,
        target: str,
        conditions: list[Condition],
        fields: list[Field],
        trained: trainer.Trained,
    ) -> None:
        self._engine = engine
        self._id = dataset_id
        self.target = target
        self._conditions = conditions
        self._fields = fields
        self.result = trained
        self._explanation: explainer.Explanation | None = None
        self._card: use.ModelCard | None = None

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.result, name)

    def __repr__(self) -> str:
        score = f"{self.result.score:.4g}" if self.result.score is not None else "n/a"
        return (
            f"<TrainedModel {self.result.model} -> {self.target!r} ({self.result.task}): "
            f"{self.result.primary} {score}>"
        )

    @property
    def pipeline(self) -> Any:
        """The fitted scikit-learn pipeline (preparation + model)."""
        return self.result.pipeline

    @property
    def inputs(self) -> list[str]:
        return [str(c) for c in self.result.pipeline.feature_names_in_]

    def explain(self) -> explainer.Explanation:
        """Permutation importance, SHAP values, partial dependence and the tree (when there is
        one), on the rows the model was scored on."""
        if self._explanation is None:
            self._explanation = self._engine._explanation(
                self._id, self.result.pipeline, self, self._conditions, self.result.held_out
            )
        return self._explanation

    def _held_out(self) -> mistakes.HeldOut:
        if self.result.held_out is None:
            raise MlError("This model kept no held-out predictions: train it again.")
        return self.result.held_out

    def errors(self, measure: mistakes.Measure | None = None) -> mistakes.ErrorAnalysis:
        """Where the model is wrong, on its held-out predictions (036): `slices` (subgroups
        notably worse than average, in sentences, with their filter conditions), `by_column`,
        `worst` (50 worst predictions), `hints`. `measure`: absolute_error (numbers),
        misclassified or log_loss (classes)."""
        return self._engine._errors(self._id, self, self._held_out(), self._conditions, measure)

    def compare_errors(
        self, other: "TrainedModel", measure: mistakes.Measure | None = None
    ) -> mistakes.ErrorComparison:
        """Subgroups where `other` fixes this model's errors, and where it makes them worse."""
        return self._engine._errors_compared(
            self._id,
            self,
            self._held_out(),
            other,
            other._held_out(),
            self._conditions,
            measure,
        )

    def fairness(self, column: str) -> mistakes.Fairness:
        """Metrics side by side for each group of a sensitive column (error rate, predicted
        positive rate, recall; numbers: mean error, prediction and actual), with the largest gap.
        Display only: nothing is corrected."""
        return self._engine._fairness(self._id, self, self._held_out(), self._conditions, column)

    def neutralize(self, columns: list[str], proportion: float = 1.0) -> panels.Neutralized:
        """Remove from the held-out predictions the part `columns` explain linearly, period by
        period (`proportion` from 0 to 1), and score every period again (panel runs only)."""
        if self.result.out_of_fold is None:
            raise MlError("Neutralization needs a run validated by period (panel walk-forward).")
        try:
            request = NeutralizeRequest(columns=columns, proportion=proportion)
        except ValueError as error:
            raise MlError(f"Invalid neutralization: {error}") from None
        return self._engine._neutralize(
            self._id, self, self.result.out_of_fold, request, self._conditions
        )

    @property
    def card(self) -> use.ModelCard:
        """What the model expects and how it was validated (037): `check`, `predict`,
        `prediction`, `explain_row` and `export` come with it (`ml/saved.py`)."""
        if self._card is None:
            try:
                self._card = use.card_from(
                    self._engine, self._id, self.result.pipeline, self, self._conditions
                )
            except ValueError as error:
                raise MlError(str(error)) from None
        return self._card


@contextmanager
def _readable() -> Iterator[None]:
    """The engine's own errors (bad settings, validation, tuning) as `MlError`, like the app."""
    try:
        yield
    except (
        trainer.TrainingError,
        catalogue.SettingsError,
        evaluate.ValidationError,
        tuner.TuneError,
    ) as error:
        raise MlError(str(error)) from error


Preparation = prep.Choices | dict[str, Any] | None


def _choices(preparation: Preparation) -> prep.Choices | None:
    """Preparation choices given as the model or as a plain dict (validated the same way)."""
    if preparation is None or isinstance(preparation, prep.Choices):
        return preparation
    try:
        return prep.Choices.model_validate(preparation)
    except ValueError as error:
        raise MlError(f"Invalid preparation: {error}") from None


@dataclass
class Failed:
    """A model of a comparison that could not be trained."""

    model: str
    error: str


@dataclass
class AutoResult:
    """What `.auto()` found: the journal of its steps and every model tried (`progress`), and
    the model it kept (the ensemble when it beats the best single model by more than one std)."""

    progress: automl.AutoProgress
    model: TrainedModel | None

    @property
    def leaderboard(self) -> list[automl.AutoEntry]:
        """Every model tried, best first (family, settings, score +/- std between folds)."""
        return self.progress.ranking()

    def __repr__(self) -> str:
        kept = self.progress.best or "nothing"
        return (
            f"<AutoResult {len(self.progress.entries)} models tried in "
            f"{self.progress.seconds:.0f} s, kept {kept}>"
        )


class MachineLearning:
    """Models predicting one column of a dataset (`classes`: a number cut into classes;
    `validation`: split, k-fold… and the metric)."""

    def __init__(
        self,
        engine: MlEngine,
        dataset_id: str,
        target: str,
        conditions: list[Condition],
        classes: binning.Classes | dict[str, Any] | None = None,
        validation: evaluate.Validation | dict[str, Any] | None = None,
    ) -> None:
        self._engine = engine
        self._id = dataset_id
        self.target = target
        self._conditions = conditions
        try:
            self.classes = (
                binning.Classes.model_validate(classes) if isinstance(classes, dict) else classes
            )
            self.validation = (
                evaluate.Validation.model_validate(validation)
                if isinstance(validation, dict)
                else validation or evaluate.Validation()
            )
        except ValueError as error:
            raise MlError(f"Invalid classes or validation: {error}") from None

    def _request(self, model: str | None = None) -> tuple[list[Field], Field, str]:
        return self._engine._request(self._id, self.target, model, self._conditions, self.classes)

    def task(self) -> str:
        """regression, binary or multiclass."""
        if self.classes is not None:
            return "binary" if len(self.classes.names) == 2 else "multiclass"
        fields, target, source = self._request()
        return self._engine._load(fields, target, source, self.validation.max_rows)[1]

    def panel(self, period: str | None = None, entity: str | None = None) -> panels.PanelCheck:
        """Whether the rows look like a panel (entities by periods), its period and entity (found,
        or the ones given), and the embargo this target calls for."""
        return self._engine.panel(self._id, self._conditions, self.target, period, entity)

    def catalogue(self) -> Catalogue:
        """The models for this problem, their settings with defaults for this data, and the
        recommended ones."""
        found = self._engine.targets(self._id, self._conditions)
        usable = [c for c in found.candidates if c.name != self.target and c.task is not None]
        categorical = [c for c in usable if c.type not in NUMERIC]
        context = Context(
            rows=found.rows,
            task=self.task(),  # type: ignore[arg-type]
            features=len(usable),
            categorical_share=len(categorical) / len(usable) if usable else 0.0,
            missing_share=float(np.mean([c.missing_share for c in usable])) if usable else 0.0,
        )
        return self._engine.catalogue(context)

    def preview(
        self,
        model: str | None = None,
        settings: dict[str, Any] | None = None,
        preparation: Preparation = None,
    ) -> trainer.PreparationPreview:
        """The preparation the training would use (per column: selection, imputation, encoding,
        scaling) and the first prepared rows."""
        with _readable():
            return self._engine.preview(
                self._id,
                self.target,
                model,
                settings,
                self._conditions,
                _choices(preparation),
                self.classes,
                self.validation,
            )

    def estimate(
        self,
        model: str | None = None,
        settings: dict[str, Any] | None = None,
        preparation: Preparation = None,
    ) -> trainer.Estimate:
        """How long the training would take."""
        with _readable():
            return self._engine.estimate(
                self._id,
                self.target,
                model,
                settings,
                self._conditions,
                _choices(preparation),
                self.classes,
                self.validation,
            )

    def train(
        self,
        model: str | None = None,
        settings: dict[str, Any] | None = None,
        preparation: Preparation = None,
    ) -> TrainedModel:
        """Train a model (by default the app's first model for the problem: linear or
        logistic) with the automatic, explained preparation."""
        fields, target, source = self._request(model)
        with _readable():
            trained = self._engine._fit(
                fields,
                target,
                source,
                model,
                settings,
                _choices(preparation),
                self.classes,
                self.validation,
            )
        return TrainedModel(self._engine, self._id, self.target, self._conditions, fields, trained)

    def compare(
        self,
        models: list[str] | None = None,
        preparation: Preparation = None,
        progress: Callable[[str], None] | None = None,
        settings: dict[str, dict[str, Any]] | None = None,
    ) -> list[TrainedModel | Failed]:
        """Train several models with the same validation, seed and preparation (by default one
        per suitable family); best first, failures last. `settings`: per model key, e.g.
        `{"random_forest": {"n_estimators": 500}}`."""
        keys = models or self.catalogue().compare
        unknown = [k for k in keys if k not in catalogue.CATALOGUE]
        if unknown:
            raise MlError(f"Unknown model(s): {', '.join(unknown)}.", "unknown_model")
        found: list[TrainedModel | Failed] = []
        for key in dict.fromkeys(keys):
            if progress is not None:
                progress(key)
            try:
                found.append(self.train(key, (settings or {}).get(key), preparation))
            except MlError as error:
                found.append(Failed(key, str(error)))
        trained = [m for m in found if isinstance(m, TrainedModel)]
        lower = bool(trained) and trained[0].result.primary in evaluate.LOWER_IS_BETTER
        trained.sort(
            key=lambda m: (m.result.score is None, (m.result.score or 0) * (1 if lower else -1))
        )
        return [*trained, *(m for m in found if isinstance(m, Failed))]

    def tune(
        self,
        model: str,
        budget_seconds: int | None = 60,
        trials: int | None = None,
        sampler: tuner.Sampler = "tpe",
        settings: dict[str, Any] | None = None,
        preparation: Preparation = None,
        pruner: tuner.Pruner = "median",
        space: list[tuner.SearchParam | dict[str, Any]] | None = None,
    ) -> tuner.TuneProgress:
        """Search the model's best settings with Optuna on the training rows (best value, best
        settings, every trial, importance of each setting). `settings` stay fixed; `space`
        replaces the searched settings, e.g. `[{"key": "max_depth", "kind": "int", "low": 2,
        "high": 12}]` (the catalogue's space by default: `default_space(model)`)."""
        try:
            searched = (
                None
                if space is None
                else [
                    tuner.SearchParam.model_validate(p) if isinstance(p, dict) else p for p in space
                ]
            )
            request = TuneStart(
                model=model,
                budget_seconds=budget_seconds,
                trials=trials,
                sampler=sampler,
                pruner=pruner,
                space=searched,
                settings=settings or {},
                preparation=_choices(preparation),
            )
        except ValueError as error:
            raise MlError(f"Invalid search: {error}") from None
        fields, target, source = self._request(model)
        spec = catalogue.CATALOGUE[model]
        if spec.elsewhere:
            raise MlError(spec.missing())
        if not spec.installed:
            raise MlError(f"{spec.label} is not installed: pip install quantilo[ml].")
        with _readable():
            return self._engine._search(
                fields,
                target,
                source,
                request,
                request.space if request.space is not None else tuner.default_space(spec.key),
                self.classes,
                self.validation,
                lambda: False,
            )

    def auto(
        self,
        minutes: float = 5,
        models: list[str] | None = None,
        trials: int | None = None,
        stacking: bool = False,
        preparation: Preparation = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> AutoResult:
        """The app's Auto (038): baselines, one model per family (cheapest first), successive
        elimination, Optuna on the 3 best families, then a greedy ensemble on out-of-fold
        predictions, within `minutes`. `trials`: a fixed number of trials per tuned family (the
        same ranking as the app at a fixed seed); `stacking`: also try a linear stack."""
        try:
            request = automl.AutoRequest(
                minutes=minutes,
                models=models,
                trials=trials,
                stacking=stacking,
                preparation=_choices(preparation),
            )
        except ValueError as error:
            raise MlError(f"Invalid Auto request: {error}") from None
        fields, target, source = self._request()
        try:
            with _readable():
                outcome = automl.run(
                    self._engine,
                    fields,
                    target,
                    source,
                    request,
                    self.classes,
                    self.validation,
                    should_stop or (lambda: False),
                )
        except automl.AutoError as error:
            raise MlError(str(error), "unknown_model") from None
        model = (
            TrainedModel(
                self._engine, self._id, self.target, self._conditions, fields, outcome.trained
            )
            if outcome.trained is not None
            else None
        )
        return AutoResult(outcome.progress, model)
