"""Machine learning on a catalog's dataset: models, target candidates, preparation preview,
training, tuning, explanations, panel data (035) and where a model is wrong (036). Synchronous:
the app adds jobs and saved runs around it."""

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, cast

import duckdb
import numpy as np
import pandas as pd
from pydantic import BaseModel
from pydantic import Field as PydanticField

from quantilo.catalog import Catalog
from quantilo.errors import ExplorerError
from quantilo.filters.model import Condition
from quantilo.filters.scope import scoped_source, where_for
from quantilo.ml import classes as binning
from quantilo.ml import data as ml_data
from quantilo.ml import error_analysis as mistakes
from quantilo.ml import evaluate
from quantilo.ml import explain as explainer
from quantilo.ml import models as catalogue
from quantilo.ml import panel as panels
from quantilo.ml import preparation as prep
from quantilo.ml import train as trainer
from quantilo.ml import tune as tuner
from quantilo.ml.catalog import Context as ModelContext
from quantilo.ml.catalog.spaces import Param
from quantilo.ml.data import NUMERIC, Field
from quantilo.ml.target import Candidate, Task, candidate


class MlError(ExplorerError):
    def __init__(self, message: str, code: str = "ml", status_code: int = 422) -> None:
        super().__init__(message, code=code, status_code=status_code)


@dataclass
class SettingView:
    key: str
    label: str
    kind: str
    default: Any
    level: str
    options: list[str]
    minimum: float | None
    maximum: float | None
    log: bool
    why: str
    shown_when: dict[str, list[Any]] | None
    group: str
    param: str
    option_labels: dict[str, str]


@dataclass
class RuleView:
    when: dict[str, list[Any]]
    setting: str
    allowed: list[Any]
    reason: str


@dataclass
class ModelView:
    key: str
    label: str
    family: str
    description: str
    tasks: list[str]
    settings: list[SettingView]
    rules: list[RuleView]
    badges: list[str]
    installed: bool
    unavailable: str | None  # why it cannot run on this data
    probabilities: bool
    space: list[Param]  # where the optimizer searches (empty: nothing to tune)


@dataclass
class Recommendation:
    key: str
    why: str


@dataclass
class Catalogue:
    families: list[str]
    models: list[ModelView]
    recommended: list[Recommendation]
    metrics: list[str]  # every metric of the problem, the default first
    default_metric: str
    compare: list[str]  # models compared by default: one per suitable family


@dataclass
class Targets:
    candidates: list[Candidate]
    rows: int  # rows after the filters


class CompareRequest(BaseModel):
    models: list[str] = PydanticField(min_length=1, max_length=40)
    preparation: prep.Choices | None = None


class TuneStart(tuner.TuneRequest):
    preparation: prep.Choices | None = None


class NeutralizeRequest(BaseModel):
    """Remove from a panel run's predictions the part these columns explain (035)."""

    columns: list[str] = PydanticField(min_length=1, max_length=200)
    proportion: float = PydanticField(default=1.0, ge=0, le=1)


PERIOD_TYPES = {"datetime", "numeric_discrete", "numeric_continuous", "categorical_ordinal"}
PERIOD_TYPES |= {"categorical_nominal", "identifier"}
ENTITY_TYPES = {"identifier", "categorical_nominal", "categorical_ordinal", "numeric_discrete"}
ENTITY_TYPES |= {"numeric_continuous", "text"}
MAX_PANEL_CANDIDATES = 8


def period_of(validation: evaluate.Validation, fields: list[Field]) -> str | None:
    """The period column a panel validation reads by whole periods, if any."""
    if not validation.panel or validation.period_column is None:
        return None
    names = {f.name for f in fields}
    return validation.period_column if validation.period_column in names else None


class MlEngine:
    def __init__(self, columns: Catalog) -> None:
        self.columns = columns

    # --- Catalogue and targets -----------------------------------------------------------------

    def catalogue(self, context: catalogue.Context) -> Catalogue:
        """The models that handle a task, their settings with data-dependent defaults, the rules
        between settings, and two or three recommendations for this data."""
        task = context.task
        models = []
        for spec in catalogue.CATALOGUE.values():
            if task not in spec.tasks:
                continue
            settings = spec.settings(context)
            defaults = {s.key: s.default for s in settings}
            models.append(
                ModelView(
                    spec.key,
                    spec.label,
                    spec.family,
                    spec.description,
                    list(spec.tasks),
                    [SettingView(**asdict(setting)) for setting in settings],
                    [RuleView(**asdict(rule)) for rule in spec.rules],
                    spec.badges(context),
                    spec.installed,
                    spec.limit(context),
                    task == "regression" or spec.has_probabilities(defaults),
                    catalogue.spaces.space(spec.key),
                )
            )
        default = evaluate.default_metric(task, context.minority_share)
        metrics = evaluate.metrics_for(task)
        return Catalogue(
            [f for f in catalogue.FAMILIES if any(m.family == f for m in models)],
            models,
            [Recommendation(key, why) for key, why in catalogue.recommend(context)],
            [default, *(m for m in metrics if m != default)],
            default,
            self._compare_default(models, context),
        )

    @staticmethod
    def _compare_default(models: list[ModelView], context: ModelContext) -> list[str]:
        """One model per family that suits the problem and the data."""
        ready = {m.key for m in models if m.installed and m.unavailable is None}
        wanted = [
            catalogue.default_model(context.task),
            "random_forest",
            "xgboost" if "xgboost" in ready else "hist_gradient_boosting",
            "knn",
            "mlp",
            "decision_tree",
        ]
        if context.task != "regression":
            wanted.append("gaussian_nb")
        if context.rows <= 10_000:
            wanted.append("kernel_svm")
        return [key for key in wanted if key in ready]

    def _fields(self, dataset_id: str) -> tuple[list[Field], Any]:
        dataset = self.columns.ready(dataset_id)
        views = [v for v in self.columns.columns(dataset_id) if not v.excluded]
        fields = [Field(v.name, v.type, v.parser, v.role, v.order) for v in views]
        return fields, (dataset, views)

    def _counts(
        self, dataset_id: str, conditions: list[Condition]
    ) -> tuple[list[Field], int, dict[str, tuple[int, int]]]:
        """Rows after the filters, and per column: distinct values and values present."""
        fields, (dataset, views) = self._fields(dataset_id)
        connection = duckdb.connect()
        try:
            source = scoped_source(
                dataset.storage_path, where_for(views, conditions), numbered=True
            )
            parts = ", ".join(
                f"count(DISTINCT {ml_data._expression(f)}), count({ml_data._expression(f)})"
                for f in fields
            )
            row = connection.execute(f"SELECT count(*), {parts} FROM {source}").fetchone() or ()
        finally:
            connection.close()
        total = int(row[0]) if row else 0
        counts = {
            fld.name: (int(row[1 + 2 * i]), int(row[2 + 2 * i])) for i, fld in enumerate(fields)
        }
        return fields, total, counts

    def targets(self, dataset_id: str, conditions: list[Condition]) -> Targets:
        fields, total, counts = self._counts(dataset_id, conditions)
        found = []
        for fld in fields:
            distinct, present = counts[fld.name]
            missing = 1 - present / total if total else 1.0
            found.append(candidate(fld.name, fld.type, fld.role, distinct, missing, present))
        found.sort(key=lambda c: (c.task is None, not c.suggested))
        return Targets(found, total)

    def panel(
        self,
        dataset_id: str,
        conditions: list[Condition],
        target: str | None = None,
        period: str | None = None,
        entity: str | None = None,
    ) -> panels.PanelCheck:
        """Is this a panel (entities by periods)? Its period and entity, found or as declared, how
        many periods and entities, and the embargo the target calls for."""
        fields, total, counts = self._counts(dataset_id, conditions)
        by_name = {f.name: f for f in fields}
        for name in (target, period, entity):
            if name is not None and name not in by_name:
                raise MlError(f"Unknown or ignored column {name!r}.", "unknown_column")
        types = {f.name: f.type for f in fields}
        others = [f for f in fields if f.name != target]
        periods = panels.rank_period_candidates(
            [
                f.name
                for f in others
                if f.type in PERIOD_TYPES
                and counts[f.name][0] >= panels.MIN_PERIODS
                and counts[f.name][0] * panels.MIN_ROWS_PER_PERIOD <= total
            ],
            types,
        )[:MAX_PANEL_CANDIDATES]
        entities = panels.rank_entity_candidates(
            [
                f.name
                for f in others
                if f.type in ENTITY_TYPES and 2 <= counts[f.name][0] < counts[f.name][1]
            ],
            types,
        )[:MAX_PANEL_CANDIDATES]
        wanted = {*periods, *entities, *(n for n in (target, period, entity) if n)}
        if not wanted:
            return panels.PanelCheck(False, None, None, 0, 0, None, 0, None, None)
        chosen = [f for f in fields if f.name in wanted]
        _, (dataset, views) = self._fields(dataset_id)
        source = scoped_source(dataset.storage_path, where_for(views, conditions), numbered=True)
        connection = duckdb.connect()
        try:
            frame = ml_data.load(
                connection, source, chosen, target or period or chosen[0].name, ml_data.MAX_ROWS
            )
        finally:
            connection.close()
        return panels.recognise(
            frame.data,
            periods,
            entities,
            total,
            {name: distinct for name, (distinct, _) in counts.items()},
            target,
            period,
            entity,
        )

    def _request(
        self,
        dataset_id: str,
        target: str,
        model: str | None,
        conditions: list[Condition],
        classes: binning.Classes | None = None,
    ) -> tuple[list[Field], Field, str]:
        fields, (dataset, views) = self._fields(dataset_id)
        by_name = {f.name: f for f in fields}
        if target not in by_name:
            raise MlError(f"Unknown or ignored column {target!r}.", "unknown_column")
        field_ = by_name[target]
        if field_.type in ("identifier", "text", "datetime"):
            raise MlError(f"{target!r} cannot be predicted: it is {field_.type}.", "bad_target")
        if model is not None and model not in catalogue.CATALOGUE:
            raise MlError(f"Unknown model {model!r}.", "unknown_model")
        if classes is not None and field_.type not in NUMERIC:
            raise MlError(
                f"{target!r} is not a number: it cannot be cut into classes.", "bad_classes"
            )
        source = scoped_source(dataset.storage_path, where_for(views, conditions), numbered=True)
        return fields, field_, source

    def _load(
        self,
        fields: list[Field],
        target: Field,
        source: str,
        limit: int | None = ml_data.MAX_ROWS,
        period: str | None = None,
    ) -> tuple[ml_data.Frame, Task]:
        connection = duckdb.connect()
        try:
            frame = ml_data.load(connection, source, fields, target.name, limit, period)
        finally:
            connection.close()
        if frame.rows < 15:
            raise MlError("Fewer than 15 rows have a target value.")
        distinct = len(set(frame.data[target.name].dropna().astype(str)))
        task = candidate(target.name, target.type, target.role, distinct, 0).task
        if task is None:
            raise MlError(f"{target.name!r} cannot be predicted.")
        return frame, task

    def preview(
        self,
        dataset_id: str,
        target: str,
        model: str | None,
        settings: dict[str, Any] | None,
        conditions: list[Condition],
        choices: prep.Choices | None,
        classes: binning.Classes | None = None,
        validation: evaluate.Validation | None = None,
    ) -> trainer.PreparationPreview:
        """The preparation the next training would use, and its first prepared rows."""
        fields, field_, source = self._request(dataset_id, target, model, conditions, classes)
        frame, task = self._load(fields, field_, source)
        if classes is not None:
            task = "binary" if len(classes.names) == 2 else "multiclass"
        try:
            return trainer.preview(
                frame.data,
                fields,
                field_,
                task,
                model or catalogue.default_model(task),
                settings,
                choices,
                classes=classes,
                validation=validation,
            )
        except (trainer.TrainingError, catalogue.SettingsError) as error:
            raise MlError(str(error)) from None

    def estimate(
        self,
        dataset_id: str,
        target: str,
        model: str | None,
        settings: dict[str, Any] | None,
        conditions: list[Condition],
        choices: prep.Choices | None,
        classes: binning.Classes | None = None,
        validation: evaluate.Validation | None = None,
    ) -> trainer.Estimate:
        """How long the training would take, from fits on 500 and 2 000 rows."""
        fields, field_, source = self._request(dataset_id, target, model, conditions, classes)
        validation = validation or evaluate.Validation()
        frame, task = self._load(
            fields, field_, source, validation.max_rows, period_of(validation, fields)
        )
        if classes is not None:
            task = "binary" if len(classes.names) == 2 else "multiclass"
        try:
            return trainer.estimate(
                frame.data,
                fields,
                field_,
                task,
                model or catalogue.default_model(task),
                settings,
                choices,
                classes,
                validation,
            )
        except (trainer.TrainingError, catalogue.SettingsError) as error:
            raise MlError(str(error)) from None

    # --- Synchronous cores of training, tuning and explaining ------------------------------------

    def _fit(
        self,
        fields: list[Field],
        target: Field,
        source: str,
        model: str | None,
        settings: dict[str, Any] | None,
        choices: prep.Choices | None = None,
        classes: binning.Classes | None = None,
        validation: evaluate.Validation | None = None,
        should_stop: Callable[[], bool] | None = None,
        stage: Callable[[str], None] = lambda _stage: None,
    ) -> trainer.Trained:
        """Load the rows, prepare them and train the model (with its pipeline)."""
        validation = validation or evaluate.Validation()
        frame, task = self._load(
            fields, target, source, validation.max_rows, period_of(validation, fields)
        )
        if classes is not None:
            task = "binary" if len(classes.names) == 2 else "multiclass"
        stage("preparing")
        key = model or catalogue.default_model(task)
        stage("training")
        return trainer.train(
            frame.data,
            fields,
            target,
            task,
            key,
            settings,
            frame.rows,
            frame.sampled,
            choices=choices,
            classes=classes,
            validation=validation,
            should_stop=should_stop,
        )

    def _search(
        self,
        fields: list[Field],
        target: Field,
        source: str,
        request: "TuneStart",
        space: list[tuner.SearchParam],
        classes: binning.Classes | None,
        validation: evaluate.Validation,
        should_stop: Callable[[], bool],
        started: Callable[[tuner.TuneProgress], None] = lambda _progress: None,
    ) -> tuner.TuneProgress:
        """Search a model's best settings on the training rows; `started` receives the progress
        object, updated trial after trial."""
        frame, task = self._load(
            fields, target, source, validation.max_rows, period_of(validation, fields)
        )
        if classes is not None:
            task = "binary" if len(classes.names) == 2 else "multiclass"
        setup = trainer._setup(
            frame.data,
            fields,
            target,
            task,
            request.model,
            request.settings,
            None,
            request.preparation,
            classes,
            validation,
        )
        metric = validation.metric or evaluate.default_metric(
            setup.task, setup.minority, validation.panel
        )
        progress = tuner.TuneProgress(metric, metric not in evaluate.LOWER_IS_BETTER)
        search_panel = None
        if setup.periods is not None:
            # walk-forward on 3 folds, judged on the panel criterion (035)
            try:
                folds = panels.plan(
                    setup.periods.ranks,
                    setup.periods.labels,
                    tuner.FOLDS,
                    validation.embargo,
                    validation.window,
                    validation.window_periods,
                    validation.sample_periods,
                )
            except evaluate.ValidationError as error:
                raise MlError(str(error)) from None
            search_panel = tuner.PanelSearch(
                folds.folds,
                setup.periods.ranks,
                validation.criterion,
                validation.period_threshold,
            )
            progress.criterion = validation.criterion
        started(progress)

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

        tuner.tune(
            setup.x_train,
            setup.y_train,
            setup.task,
            setup.spec,
            setup.context,
            setup.preparation,
            request.model_copy(update={"space": space}),
            metric,
            progress,
            should_stop,
            build,
            search_panel,
        )
        return progress

    def _explanation(
        self,
        dataset_id: str,
        pipeline: Any,
        trained: Any,
        conditions: list[Condition],
        held_out: mistakes.HeldOut | None = None,
    ) -> explainer.Explanation:
        """Explain a trained model on the rows it was scored on. `trained` is a saved run or a
        `TrainedModel`: target, task, primary, classes, classification, predictions, validation
        and evaluation_rows. With its held-out predictions, the worst ones get a waterfall too."""
        fields, target, source = self._request(dataset_id, trained.target, None, conditions)
        connection = duckdb.connect()
        try:
            frame = ml_data.load(
                connection,
                source,
                fields,
                target.name,
                trained.validation.max_rows,
                period_of(trained.validation, fields),
            )
        finally:
            connection.close()
        data = frame.data[frame.data["__row"].isin(list(trained.evaluation_rows))].reset_index(
            drop=True
        )
        inputs = [str(c) for c in pipeline.feature_names_in_]
        missing = [c for c in inputs if c not in data.columns]
        if missing or data.empty:
            raise MlError("The data changed since this run: train it again to explain it.")
        series = cast(pd.Series, data[target.name])
        if trained.classes is not None:
            y = binning.apply(series, trained.classes).astype(str)
        else:
            y = trainer.prepare_target(series, trained.task)
        positive = trained.classification.positive if trained.classification else None
        return explainer.explain(
            pipeline,
            cast(pd.DataFrame, data[inputs]),
            y,
            cast(pd.Series, data["__row"]).to_numpy(dtype=int),
            trained.task,
            trained.primary,
            [p.row for p in trained.predictions]
            + (mistakes.worst_rows(held_out) if held_out is not None else []),
            positive,
        )

    def _neutralize(
        self,
        dataset_id: str,
        trained: Any,
        oof: panels.OutOfFold,
        request: NeutralizeRequest,
        conditions: list[Condition],
    ) -> panels.Neutralized:
        """A panel run's held-out predictions neutralized on some columns, scored again period by
        period. `trained`: a saved run or a `TrainedModel` (target, primary, validation, panel)."""
        if trained.panel is None:
            raise MlError("Neutralization needs a run validated by period (panel walk-forward).")
        fields, target, source = self._request(dataset_id, trained.target, None, conditions)
        by_name = {f.name: f for f in fields}
        unknown = [c for c in request.columns if c not in by_name]
        if unknown:
            raise MlError(f"Unknown or ignored column(s): {', '.join(unknown)}.", "unknown_column")
        wrong = [c for c in request.columns if by_name[c].type not in NUMERIC]
        if wrong:
            raise MlError(f"Only numbers can be neutralized: {', '.join(wrong)}.", "bad_columns")
        period = period_of(trained.validation, fields)
        wanted = {target.name, *request.columns, *([period] if period else [])}
        chosen = [f for f in fields if f.name in wanted]
        connection = duckdb.connect()
        try:
            frame = ml_data.load(
                connection, source, chosen, target.name, trained.validation.max_rows, period
            )
        finally:
            connection.close()
        data = frame.data.set_index("__row")
        if not np.isin(oof.rows, data.index.to_numpy()).all():
            raise MlError("The data changed since this run: train it again to neutralize it.")
        aligned = cast(pd.DataFrame, data.loc[oof.rows, request.columns])
        labels = {p.rank: p.period for p in trained.panel.periods}
        try:
            return panels.neutralized(
                oof,
                aligned,
                list(request.columns),
                request.proportion,
                trained.primary,
                labels,
                trained.validation.criterion,
                trained.validation.period_threshold,
            )
        except panels.PanelError as error:
            raise MlError(str(error)) from None

    # --- Where the model is wrong (036) --------------------------------------------------------

    def _held_out_columns(
        self,
        dataset_id: str,
        trained: Any,
        held_out: mistakes.HeldOut,
        conditions: list[Condition],
    ) -> tuple[list[Field], pd.DataFrame]:
        """Every column but the target, on a run's held-out rows and in their order."""
        fields, target, source = self._request(dataset_id, trained.target, None, conditions)
        connection = duckdb.connect()
        try:
            frame = ml_data.load(
                connection,
                source,
                fields,
                target.name,
                trained.validation.max_rows,
                period_of(trained.validation, fields),
            )
        finally:
            connection.close()
        data = frame.data.set_index("__row")
        if not np.isin(held_out.rows, data.index.to_numpy()).all():
            raise MlError("The data changed since this run: train it again to see its errors.")
        others = [f for f in fields if f.name != target.name]
        return others, cast(pd.DataFrame, data.loc[held_out.rows, [f.name for f in others]])

    def _errors(
        self,
        dataset_id: str,
        trained: Any,
        held_out: mistakes.HeldOut,
        conditions: list[Condition],
        measure: mistakes.Measure | None = None,
    ) -> mistakes.ErrorAnalysis:
        """Where a run is wrong, on its held-out predictions. `trained`: a saved run or a
        `TrainedModel` (target, task, preparation, classification, validation)."""
        fields, data = self._held_out_columns(dataset_id, trained, held_out, conditions)
        excluded = {
            d.column
            for d in trained.preparation.decisions
            if not d.use and d.reactivable and not d.leak
        }
        positive = trained.classification.positive if trained.classification else None
        try:
            return mistakes.analyse(
                data,
                fields,
                held_out,
                trained.task,
                measure,
                excluded,
                trained.target,
                positive,
                "out-of-fold predictions" if trained.validation.cross else "test rows",
                conditions,
            )
        except ValueError as error:
            raise MlError(str(error), "bad_measure") from None

    def _errors_compared(
        self,
        dataset_id: str,
        first: Any,
        first_held: mistakes.HeldOut,
        second: Any,
        second_held: mistakes.HeldOut,
        conditions: list[Condition],
        measure: mistakes.Measure | None = None,
    ) -> mistakes.ErrorComparison:
        """Subgroups where the second run fixes or worsens the first run's errors."""
        if first.target != second.target or first.task != second.task:
            raise MlError("Both runs must predict the same target the same way.", "incomparable")
        fields, data = self._held_out_columns(dataset_id, first, first_held, conditions)
        try:
            return mistakes.compare(data, fields, first_held, second_held, first.task, measure)
        except ValueError as error:
            raise MlError(str(error), "bad_measure") from None

    def _fairness(
        self,
        dataset_id: str,
        trained: Any,
        held_out: mistakes.HeldOut,
        conditions: list[Condition],
        column: str,
    ) -> mistakes.Fairness:
        """A run's metrics side by side for each group of a sensitive column."""
        fields, data = self._held_out_columns(dataset_id, trained, held_out, conditions)
        by_name = {f.name: f for f in fields}
        if column not in by_name:
            raise MlError(f"Unknown or ignored column {column!r}.", "unknown_column")
        positive = trained.classification.positive if trained.classification else None
        try:
            return mistakes.fairness(
                cast(pd.Series, data[column]), by_name[column], held_out, trained.task, positive
            )
        except ValueError as error:
            raise MlError(str(error), "bad_column") from None
