"""Machine learning on a catalog's dataset: models, target candidates, preparation preview,
training, tuning, explanations (and, on demand, the interpretability of 125: ALE, surrogate,
monotonic check, counterfactuals), panel data (035) and where a model is wrong (036).
Synchronous: the app adds jobs and saved runs around it."""

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
from quantilo.interpret import anchors as anchoring
from quantilo.interpret import compare as readable
from quantilo.interpret import extras
from quantilo.interpret import interactions as pairing
from quantilo.interpret import kernel_shap as sampling
from quantilo.interpret import rashomon as optimal
from quantilo.interpret import risk_score as interpretable
from quantilo.interpret import sparse_tree as sparse
from quantilo.interpret import tabular_prototypes as examples
from quantilo.ml import ale as effects
from quantilo.ml import browser, evaluate
from quantilo.ml import classes as binning
from quantilo.ml import counterfactual as whatif
from quantilo.ml import data as ml_data
from quantilo.ml import error_analysis as mistakes
from quantilo.ml import explain as explainer
from quantilo.ml import models as catalogue
from quantilo.ml import panel as panels
from quantilo.ml import preparation as prep
from quantilo.ml import surrogate as imitation
from quantilo.ml import text as textual
from quantilo.ml import train as trainer
from quantilo.ml import tune as tuner
from quantilo.ml import warm as warming
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
    installed: bool  # it can be trained in this build
    unavailable: str | None  # why it cannot run on this data
    probabilities: bool
    space: list[Param]  # where the optimizer searches (empty: nothing to tune)
    desktop_only: bool = False  # R-WEB-04: not in the browser version, in the desktop app (102)


@dataclass
class Recommendation:
    key: str
    why: str


@dataclass
class BrowserLimits:
    """What the browser version holds back (102), so that the page can say it."""

    max_rows: int  # R-WEB-01
    auto_minutes: float  # R-WEB-03
    tuning: bool  # R-WEB-04: Optuna is left to the desktop app


@dataclass
class Catalogue:
    families: list[str]
    models: list[ModelView]
    recommended: list[Recommendation]
    metrics: list[str]  # every metric of the problem, the default first
    default_metric: str
    compare: list[str]  # models compared by default: one per suitable family
    browser: BrowserLimits | None = None  # None: the desktop app, nothing held back


@dataclass
class Targets:
    candidates: list[Candidate]
    rows: int  # rows after the filters


class CompareRequest(BaseModel):
    models: list[str] = PydanticField(min_length=1, max_length=40)
    preparation: prep.Choices | None = None


class TuneStart(tuner.TuneRequest):
    preparation: prep.Choices | None = None


class CounterfactualRequest(BaseModel):
    """What would change one row's prediction (125)."""

    row: int = PydanticField(ge=0, description="The file row (as on the Predictions tab).")
    editable: list[str] | None = PydanticField(
        default=None,
        max_length=500,
        description="Columns the search may change (default: all but those locked, R-XAI-03).",
    )
    target_class: str | None = PydanticField(default=None, max_length=500)
    target_value: float | None = None
    max_changes: int = PydanticField(default=whatif.MAX_CHANGES, ge=1, le=whatif.MAX_CHANGES)


@dataclass
class Explained:
    """The rows a run was scored on, as its explanations read them."""

    x: pd.DataFrame
    y: pd.Series
    rows: np.ndarray  # file row of each
    fields: list[Field]
    positive: str | None


@dataclass
class Readable:
    """The rows a readable model learns on and is judged on (R-XAI-08), with the run's model's
    probability of the positive class on the judged rows."""

    learn: pd.DataFrame
    y_learn: np.ndarray  # class labels as text (numbers for a regression, 122 bis)
    check: pd.DataFrame
    y_check: np.ndarray
    probability: np.ndarray  # two classes: of the positive; a number: the model's prediction
    positive: str
    label: str  # the run's model, as the sentences name it
    source: str
    types: dict[str, str]
    task: str = "binary"
    classes: list[str] | None = None  # 121 bis: three classes or more, as the model orders them
    probabilities: np.ndarray | None = None  # checked rows by classes


def model_label(model: str) -> str:
    """The run's model, as the comparison sentences name it."""
    spec = catalogue.CATALOGUE.get(model)
    label = spec.label if spec else ("ensemble" if model == "ensemble" else model)
    return f"run's model ({label})"


def constraints(text: str | None) -> dict[str, int]:
    """ "size:+1, age:-1" → {"size": 1, "age": -1} (the monotonic setting of 019, read back)."""
    found: dict[str, int] = {}
    for part in str(text or "").replace(";", ",").split(","):
        column, _, sign = part.rpartition(":")
        if column.strip() and sign.strip() in ("+1", "1", "-1", "+", "-"):
            found[column.strip()] = -1 if sign.strip().startswith("-") else 1
    return found


class NeutralizeRequest(BaseModel):
    """Remove from a panel run's predictions the part these columns explain (035)."""

    columns: list[str] = PydanticField(min_length=1, max_length=200)
    proportion: float = PydanticField(default=1.0, ge=0, le=1)


PERIOD_TYPES = {"datetime", "numeric_discrete", "numeric_continuous", "categorical_ordinal"}
PERIOD_TYPES |= {"categorical_nominal", "identifier"}
ENTITY_TYPES = {"identifier", "categorical_nominal", "categorical_ordinal", "numeric_discrete"}
ENTITY_TYPES |= {"numeric_continuous", "text"}
MAX_PANEL_CANDIDATES = 8
DOMINANT_DISTINCT = 50  # R-TGT-04: numbers with at most this many values get their mode share


def period_of(validation: evaluate.Validation, fields: list[Field]) -> str | None:
    """The period column a panel validation reads by whole periods, if any."""
    if not validation.panel or validation.period_column is None:
        return None
    names = {f.name for f in fields}
    return validation.period_column if validation.period_column in names else None


class MlEngine:
    def __init__(self, columns: Catalog) -> None:
        self.columns = columns
        # the original of a cleaned version (065), when the app knows it: a warm start (066)
        self.lineage: Callable[[str], str | None] | None = None

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
                    spec.available,
                    spec.limit(context),
                    task == "regression" or spec.has_probabilities(defaults),
                    catalogue.spaces.space(spec.key),
                    spec.elsewhere,
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
            BrowserLimits(browser.MAX_ROWS, browser.AUTO_MINUTES, tuner.available())
            if browser.active()
            else None,
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
        dominant = self._dominant(dataset_id, conditions, fields, counts)
        found = []
        for fld in fields:
            distinct, present = counts[fld.name]
            missing = 1 - present / total if total else 1.0
            found.append(
                candidate(
                    fld.name, fld.type, fld.role, distinct, missing, present, dominant.get(fld.name)
                )
            )
        found.sort(key=lambda c: (c.task is None, not c.suggested))
        return Targets(found, total)

    def _dominant(
        self,
        dataset_id: str,
        conditions: list[Condition],
        fields: list[Field],
        counts: dict[str, tuple[int, int]],
    ) -> dict[str, float]:
        """R-TGT-04 (094): for the numbers with few distinct values (counts, mostly), the share of
        the most frequent one among the values present."""
        wanted = [
            f
            for f in fields
            if f.type in NUMERIC
            and 1 <= counts[f.name][0] <= DOMINANT_DISTINCT
            and counts[f.name][1]
        ]
        if not wanted:
            return {}
        _, (dataset, views) = self._fields(dataset_id)
        source = scoped_source(dataset.storage_path, where_for(views, conditions), numbered=True)
        found: dict[str, float] = {}
        connection = duckdb.connect()
        try:
            for fld in wanted:
                value = ml_data._expression(fld)
                (top,) = connection.execute(
                    f"SELECT max(n) FROM (SELECT count(*) AS n FROM {source} "
                    f"WHERE {value} IS NOT NULL GROUP BY {value})"
                ).fetchone() or (0,)
                found[fld.name] = int(top or 0) / counts[fld.name][1]
        finally:
            connection.close()
        return found

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
        candidates: list[warming.Candidate] | None = None,
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
        warm = None
        if candidates is not None:  # a warm start (066)
            meta = warming.meta_of(
                {f.name: f.type for f in fields}, target.name, len(frame.data), setup.task
            )
            warm = warming.starts(request.model, meta, candidates, space)
            if not warm:
                progress.rules.append(
                    "R-ML-WARM-03: no settings found yet for this model on this dataset, its "
                    "versions or a similar one: the search starts from scratch."
                )
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
            warm,
        )
        return progress

    def meta(self, dataset_id: str, target: str, task: str) -> warming.Meta:
        """The meta-features a warm start compares datasets on (066), from the column types and
        the row count, without reading the data."""
        fields, (dataset, _views) = self._fields(dataset_id)
        rows = int(getattr(dataset, "row_count", None) or 0)
        return warming.meta_of({f.name: f.type for f in fields}, target, rows, task)

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
        found = self._explained(dataset_id, pipeline, trained, conditions)
        return explainer.explain(
            pipeline,
            found.x,
            found.y,
            found.rows,
            trained.task,
            trained.primary,
            [p.row for p in trained.predictions]
            + (mistakes.worst_rows(held_out) if held_out is not None else []),
            found.positive,
        )

    def _explained(
        self, dataset_id: str, pipeline: Any, trained: Any, conditions: list[Condition]
    ) -> Explained:
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
        return Explained(
            cast(pd.DataFrame, data[inputs]),
            y,
            cast(pd.Series, data["__row"]).to_numpy(dtype=int),
            fields,
            positive,
        )

    # --- Interpretability on demand (125) --------------------------------------------------------

    def _ale(
        self,
        dataset_id: str,
        pipeline: Any,
        trained: Any,
        conditions: list[Condition],
        column: str,
        bins: int = effects.BINS,
        rule: str | None = None,
    ) -> effects.Ale:
        """Accumulated local effects of one column; its monotonic check when it has one."""
        found = self._explained(dataset_id, pipeline, trained, conditions)
        orders = {d.column: list(d.order) for d in trained.preparation.decisions if d.order}
        sign = constraints(trained.settings.get("monotonic")).get(column)
        try:
            return effects.ale(pipeline, found.x, trained.task, column, bins, orders, rule, sign)
        except effects.AleError as error:
            raise MlError(str(error), "no_ale") from None

    def _ale_2d(
        self,
        dataset_id: str,
        pipeline: Any,
        trained: Any,
        conditions: list[Condition],
        first: str,
        second: str,
    ) -> effects.Ale2d:
        """The two-column ALE of two numbers (125 bis, R-XAI-20): their interaction alone."""
        found = self._explained(dataset_id, pipeline, trained, conditions)
        try:
            return effects.ale_2d(pipeline, found.x, trained.task, first, second)
        except effects.AleError as error:
            raise MlError(str(error), "no_ale") from None

    def _monotonic(
        self, dataset_id: str, pipeline: Any, trained: Any, conditions: list[Condition]
    ) -> list[effects.Monotonic]:
        """Each monotonic constraint of the run, checked row by row."""
        wanted = constraints(trained.settings.get("monotonic"))
        if not wanted:
            return []
        found = self._explained(dataset_id, pipeline, trained, conditions)
        return [
            effects.monotonic(pipeline, found.x, trained.task, column, sign)
            for column, sign in wanted.items()
            if column in found.x.columns
        ]

    def _surrogate(
        self,
        dataset_id: str,
        pipeline: Any,
        trained: Any,
        conditions: list[Condition],
        depth: int = imitation.DEPTH,
    ) -> imitation.Surrogate:
        found = self._explained(dataset_id, pipeline, trained, conditions)
        texts = set(textual.steps_of(pipeline))
        return imitation.surrogate(pipeline, found.x, trained.task, depth, texts)

    def _counterfactual(
        self,
        dataset_id: str,
        pipeline: Any,
        trained: Any,
        conditions: list[Condition],
        request: CounterfactualRequest,
    ) -> whatif.Counterfactual:
        found = self._explained(dataset_id, pipeline, trained, conditions)
        classification = trained.classification
        try:
            return whatif.counterfactual(
                pipeline,
                found.x,
                found.rows,
                trained.task,
                request.row,
                found.fields,
                set(textual.steps_of(pipeline)),
                request.editable,
                request.target_class,
                request.target_value,
                request.max_changes,
                classification.threshold if classification else None,
                found.positive,
            )
        except whatif.CounterfactualError as error:
            raise MlError(str(error), "no_counterfactual") from None

    # --- Interpretability under Advanced, More (126) ----------------------------------------------

    def _left_out(self, pipeline: Any, found: Explained) -> dict[str, str]:
        """Inputs that a method reading values between rows cannot use, and why."""
        texts = set(textual.steps_of(pipeline))
        by_name = {f.name: f for f in found.fields}
        left: dict[str, str] = {}
        for column in map(str, found.x.columns):
            info = by_name.get(column)
            if column in texts or (info is not None and info.type == "text"):
                left[column] = "free text"
            elif info is not None and info.type == "datetime":
                left[column] = "a date"
            elif info is not None and info.type == "identifier":
                left[column] = "an identifier"
        return left

    def _permutation_shap(
        self,
        dataset_id: str,
        pipeline: Any,
        trained: Any,
        conditions: list[Condition],
        pairs: int = sampling.PAIRS,
        background: int = sampling.BACKGROUND,
    ) -> sampling.PermutationShap:
        """SHAP values for any model, by sampling permutations (R-XAI-12)."""
        found = self._explained(dataset_id, pipeline, trained, conditions)
        shown = [p.row for p in trained.predictions]
        try:
            return sampling.permutation_shap(
                pipeline,
                found.x,
                found.rows,
                trained.task,
                shown,
                set(textual.steps_of(pipeline)),
                pairs,
                background,
            )
        except sampling.PermutationShapError as error:
            raise MlError(str(error), "no_permutation_shap") from None

    def _interactions(
        self,
        dataset_id: str,
        pipeline: Any,
        trained: Any,
        conditions: list[Condition],
        ranked: list[str],
        count: int = pairing.COLUMNS,
    ) -> pairing.Interactions:
        """Friedman's H on the most important columns (R-XAI-13)."""
        found = self._explained(dataset_id, pipeline, trained, conditions)
        try:
            return pairing.interactions(
                pipeline, found.x, trained.task, ranked, self._left_out(pipeline, found), count
            )
        except pairing.InteractionError as error:
            raise MlError(str(error), "no_interactions") from None

    def _anchor(
        self,
        dataset_id: str,
        pipeline: Any,
        trained: Any,
        conditions: list[Condition],
        row: int,
        ranked: list[str],
        threshold: float = anchoring.THRESHOLD,
        max_conditions: int = anchoring.MAX_CONDITIONS,
    ) -> anchoring.Anchor:
        """One row's anchor: a rule that holds on most similar rows (R-XAI-14)."""
        found = self._explained(dataset_id, pipeline, trained, conditions)
        classification = trained.classification
        try:
            return anchoring.anchor(
                pipeline,
                found.x,
                found.rows,
                trained.task,
                row,
                found.fields,
                set(textual.steps_of(pipeline)),
                ranked,
                threshold,
                max_conditions,
                classification.threshold if classification else None,
                found.positive,
            )
        except anchoring.AnchorError as error:
            raise MlError(str(error), "no_anchor") from None

    def _prototypes(
        self,
        dataset_id: str,
        pipeline: Any,
        trained: Any,
        conditions: list[Condition],
        ranked: list[str],
        count: int = examples.PROTOTYPES,
        criticisms: int = examples.CRITICISMS,
    ) -> examples.Prototypes:
        """Prototypes and criticisms of each class (R-XAI-15)."""
        found = self._explained(dataset_id, pipeline, trained, conditions)
        try:
            return examples.prototypes(
                pipeline,
                found.x,
                found.y,
                found.rows,
                trained.task,
                found.fields,
                set(textual.steps_of(pipeline)),
                ranked,
                count,
                criticisms,
            )
        except examples.PrototypeError as error:
            raise MlError(str(error), "no_prototypes") from None

    # --- Models readable by construction (121, 122) ----------------------------------------------

    def _readable_rows(
        self,
        dataset_id: str,
        pipeline: Any,
        trained: Any,
        conditions: list[Condition],
        held_out: mistakes.HeldOut | None,
        tasks: frozenset[str] = frozenset({"binary"}),
    ) -> Readable:
        """R-XAI-05 and R-XAI-08: a run's rows to learn a card or a tree on, and its held-out
        rows, where the model's probability (or prediction, for a number) was honest. `tasks`:
        two classes for one card; 121 bis and 122 bis open more classes and numbers."""
        classification = trained.classification
        positive = classification.positive if classification else None
        if trained.task not in tasks or (trained.task == "binary" and positive is None):
            raise MlError(
                "R-XAI-05: a scorecard gives the risk of one event: it needs a run with two "
                "classes.",
                "not_binary",
            )
        number = trained.task == "regression"
        if held_out is None or (held_out.probabilities is None and not number):
            raise MlError(
                "This run kept no held-out predictions: train it again to compare a readable "
                "model with it.",
                "no_held_out",
            )
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
        data = frame.data.reset_index(drop=True)
        inputs = [str(c) for c in pipeline.feature_names_in_]
        if [c for c in inputs if c not in data.columns] or data.empty:
            raise MlError("The data changed since this run: train it again.")
        series = cast(pd.Series, data[target.name])
        if trained.classes is not None:
            labels = binning.apply(series, trained.classes).astype(str)
        else:
            labels = trainer.prepare_target(series, trained.task).astype(str)
        limit = interpretable.ROWS if not browser.active() else interpretable.BROWSER_ROWS
        rows = readable.split(
            cast(pd.Series, data["__row"]).to_numpy(dtype=int), held_out.rows, limit
        )
        if not len(rows.check) or not len(rows.learn):
            raise MlError("Too few held-out rows to judge a readable model on.")
        x = cast(pd.DataFrame, data[inputs])
        types = {f.name: str(f.type) for f in fields}
        if number:  # 122 bis: the model's held-out prediction stands for its probability
            values = np.asarray(pd.to_numeric(series, errors="coerce"), dtype=float)
            return Readable(
                x.iloc[rows.learn].reset_index(drop=True),
                values[rows.learn],
                x.iloc[rows.check].reset_index(drop=True),
                values[rows.check],
                np.asarray(held_out.predicted, dtype=float)[rows.held],
                "",
                model_label(trained.model),
                rows.source,
                types,
                "regression",
            )
        classes = list(held_out.classes)
        y = labels.to_numpy()
        if trained.task != "binary":  # 121 bis: one card per class
            matrix = np.asarray(held_out.probabilities)[rows.held]
            return Readable(
                x.iloc[rows.learn].reset_index(drop=True),
                y[rows.learn],
                x.iloc[rows.check].reset_index(drop=True),
                y[rows.check],
                matrix[:, 0],
                "",
                model_label(trained.model),
                rows.source,
                types,
                trained.task,
                classes,
                matrix,
            )
        if positive not in classes:
            raise MlError("The held-out predictions do not name the positive class: train again.")
        probability = np.asarray(held_out.probabilities)[rows.held, classes.index(positive)]
        return Readable(
            x.iloc[rows.learn].reset_index(drop=True),
            y[rows.learn],
            x.iloc[rows.check].reset_index(drop=True),
            y[rows.check],
            probability,
            positive,
            model_label(trained.model),
            rows.source,
            types,
        )

    def _risk_scores(
        self,
        dataset_id: str,
        pipeline: Any,
        trained: Any,
        conditions: list[Condition],
        held_out: mistakes.HeldOut | None,
        k: int = interpretable.CONDITIONS,
        bound: int = interpretable.BOUND,
        monotone: bool = False,
    ) -> interpretable.RiskScores:
        """FasterRisk's scorecard (121) against the run's model, on the same held-out rows;
        `monotone`: one direction of points per number (R-XAI-16)."""
        found = self._readable_rows(dataset_id, pipeline, trained, conditions, held_out)
        y_learn = (found.y_learn == found.positive).astype(float)
        y_check = (found.y_check == found.positive).astype(float)
        try:
            return interpretable.risk_scores(
                found.learn,
                y_learn,
                found.check,
                y_check,
                found.probability,
                found.positive,
                found.label,
                found.source,
                k,
                bound,
                set(textual.steps_of(pipeline)),
                found.types,
                monotone,
            )
        except interpretable.RiskScoreError as error:
            raise MlError(str(error), "no_scorecard") from None

    def _class_cards(
        self,
        dataset_id: str,
        pipeline: Any,
        trained: Any,
        conditions: list[Condition],
        held_out: mistakes.HeldOut | None,
        k: int = interpretable.CONDITIONS,
        bound: int = interpretable.BOUND,
        monotone: bool = False,
    ) -> interpretable.ClassCards:
        """Three classes or more (121 bis, R-XAI-17): one card per class, against the model."""
        found = self._readable_rows(
            dataset_id, pipeline, trained, conditions, held_out, frozenset({"multiclass"})
        )
        assert found.classes is not None and found.probabilities is not None
        try:
            return interpretable.class_cards(
                found.learn,
                found.y_learn,
                found.check,
                found.y_check,
                found.probabilities,
                found.classes,
                found.label,
                found.source,
                k,
                bound,
                set(textual.steps_of(pipeline)),
                found.types,
                monotone,
            )
        except interpretable.RiskScoreError as error:
            raise MlError(str(error), "no_scorecard") from None

    def _sparse_tree(
        self,
        dataset_id: str,
        pipeline: Any,
        trained: Any,
        conditions: list[Condition],
        held_out: mistakes.HeldOut | None,
        depth: int = sparse.DEPTH,
        leaves: int = sparse.LEAVES,
    ) -> sparse.SparseTree:
        """A sparse CART tree learnt on the target (122, R-XAI-09), against the run's model; for
        three classes or more and for a number too (122 bis, R-XAI-19)."""
        found = self._readable_rows(
            dataset_id,
            pipeline,
            trained,
            conditions,
            held_out,
            frozenset({"binary", "multiclass", "regression"}),
        )
        skip = set(textual.steps_of(pipeline)) | {
            c for c, kind in found.types.items() if kind in {"text", "identifier"}
        }
        try:
            return sparse.sparse_tree(
                found.learn,
                found.y_learn,
                found.check,
                found.y_check,
                cast(Any, found.task),
                depth,
                leaves,
                skip,
                found.positive or None,
                found.probability if found.task in ("binary", "regression") else None,
                found.label,
            )
        except sparse.SparseTreeError as error:
            raise MlError(str(error), "no_tree") from None

    def _optimal(
        self,
        dataset_id: str,
        pipeline: Any,
        trained: Any,
        conditions: list[Condition],
        held_out: mistakes.HeldOut | None,
        kind: str,
        depth: int = optimal.DEPTH,
    ) -> optimal.OptimalTree | optimal.RashomonSet:
        """Advanced, More (122): GOSDT's optimal tree, or TreeFARMS's Rashomon set; refused
        with R-XAI-10 when the `interpret` extra cannot run here."""
        found = self._readable_rows(dataset_id, pipeline, trained, conditions, held_out)
        skip = set(textual.steps_of(pipeline))
        try:
            if kind == "gosdt":
                return optimal.optimal_tree(
                    found.learn,
                    found.y_learn,
                    found.check,
                    found.y_check,
                    found.positive,
                    depth,
                    skip,
                    found.types,
                    found.probability,
                    found.label,
                )
            return optimal.rashomon_set(
                found.learn,
                found.y_learn,
                found.check,
                found.y_check,
                found.positive,
                depth,
                skip,
                found.types,
            )
        except extras.ExtraMissing as error:
            raise MlError(str(error), "extra_missing", 409) from None
        except optimal.RashomonError as error:
            raise MlError(str(error), "no_tree") from None

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
