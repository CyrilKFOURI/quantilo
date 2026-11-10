"""What the machine-learning module lets the AI agent do (spec 026b): pick a target, train,
compare, tune and explain models, through this module's own service; on panel data (035), check
the panel and validate walk-forward by period; see where a model is wrong (036, `errors.py`)."""

from typing import Any, ClassVar

from fastapi import FastAPI
from pydantic import BaseModel, Field, ValidationError
from quantilo.ml import catalog as catalogue
from quantilo.ml import evaluate

from app.core.tools import AppLink, Tool, ToolContext, ToolError, ToolOutput, compact, wait_for
from app.modules.ml.errors import ModelErrors
from app.modules.ml.service import CompareRequest, MlError, MlService, Run, TuneStart

TRAIN_TIMEOUT = 600.0


def _r(value: float | None, digits: int = 4) -> float | None:
    if value is None or value != value:
        return None
    return float(f"{value:.{digits}g}")


def _link(target: str, run_id: str | None = None) -> AppLink:
    params = {"target": target} | ({"run": run_id} if run_id else {})
    return AppLink("ml", params, "Open in Machine learning")


def run_content(run: Run) -> dict[str, Any]:
    """A trained model as the agent reads it: what was fitted, how well, and what mattered."""
    content: dict[str, Any] = {
        "run_id": run.id,
        "target": run.target,
        "task": run.task,
        "model": run.model,
        "model_label": spec.label if (spec := catalogue.CATALOGUE.get(run.model)) else run.model,
        "rows": {"train": run.train_rows, "test": run.test_rows, "sampled": run.sampled},
        "primary_metric": run.primary,
        "score": _r(run.score),
        "scores": {k: _r(v) for k, v in run.scores.items()},
        "validation": compact(run.validation),
        "seconds": _r(run.seconds, 3),
    }
    if run.cross_validation is not None:
        content["cross_validation"] = compact(run.cross_validation)
    if run.importances:
        content["top_features"] = [
            {"feature": i.feature, "share": _r(i.importance, 3)} for i in run.importances[:10]
        ]
    if run.coefficients:
        content["coefficients"] = compact(run.coefficients[:10])
    if run.settings:
        content["settings"] = run.settings
    if run.panel is not None:
        p = run.panel
        content["panel"] = {
            "period": p.period_column,
            "entity": p.entity_column,
            "criterion": p.criterion,
            "score": _r(p.value),
            "summary": compact(p.summary),
            "folds": len(p.folds),
            "embargo": p.embargo,
            "one_period_in": p.step,
            "max_exposure": _r(p.max_exposure, 3),
            "exposures": [{"column": e.column, "mean": _r(e.mean, 3)} for e in p.exposures],
            "alerts": [a.message for a in p.alerts],
        }
    return content


VALIDATION_HELP = (
    "How the model is judged; empty for the app's default (80 / 20 split). On panel data (the "
    "same entities at successive periods, see check_panel) use walk-forward by period, e.g. "
    '{"method": "panel_walk_forward", "period_column": "era", "entity_column": "id", '
    '"embargo": 4, "folds": 5, "criterion": "sharpe"}. Other keys: metric, window '
    '("expanding" or "rolling"), sample_periods, max_rows.'
)


def _validation(given: dict[str, Any] | None) -> evaluate.Validation | None:
    if not given:
        return None
    try:
        return evaluate.Validation.model_validate(given)
    except ValidationError as error:
        message = error.errors()[0].get("msg", "invalid")
        raise ToolError(f"Validation: {message}.") from None


def _finished(job: Any) -> int:
    return sum(1 for i in job.items if i.status in ("done", "failed"))


def _failed(error: Exception) -> ToolError:
    return ToolError(str(getattr(error, "message", None) or error))


class NoInput(BaseModel):
    pass


class ListTargets(Tool[NoInput]):
    name: ClassVar[str] = "list_ml_targets"
    description: ClassVar[str] = (
        "Columns that can be predicted, each with the problem it gives (regression, binary or "
        "multiclass classification), why some can't, and which ones the app suggests. Use it "
        "before training when the target is not obvious."
    )
    Input: ClassVar[type[BaseModel]] = NoInput

    def __init__(self, service: MlService) -> None:
        self.service = service

    def label(self, args: NoInput) -> str:
        return "Finding what can be predicted"

    def run(self, ctx: ToolContext, args: NoInput) -> ToolOutput:
        found = self.service.targets(ctx.dataset_id, ctx.conditions)
        targets = [compact(c) for c in found.candidates]
        usable = sum(1 for c in found.candidates if c.task)
        return ToolOutput(
            f"{usable} columns can be predicted", {"rows": found.rows, "targets": targets}
        )


class ModelsInput(BaseModel):
    task: str = Field(description="regression, binary or multiclass")


class ListModels(Tool[ModelsInput]):
    name: ClassVar[str] = "list_ml_models"
    description: ClassVar[str] = (
        "The models available for a problem type, with their key (used by train_model), family "
        "and what they are good at. Only installed models are listed."
    )
    Input: ClassVar[type[BaseModel]] = ModelsInput

    def label(self, args: ModelsInput) -> str:
        return f"Listing {args.task} models"

    def run(self, ctx: ToolContext, args: ModelsInput) -> ToolOutput:
        models = [
            {"key": s.key, "label": s.label, "family": s.family, "description": s.description}
            for s in catalogue.CATALOGUE.values()
            if args.task in s.tasks and s.available
        ]
        if not models:
            raise ToolError("Unknown task: use regression, binary or multiclass.")
        return ToolOutput(f"{len(models)} {args.task} models", {"models": models})


class TrainInput(BaseModel):
    target: str = Field(description="The column to predict.")
    model: str | None = Field(
        None, description="Model key from list_ml_models; empty for the readable reference model."
    )
    settings: dict[str, Any] | None = Field(
        None, description="Model settings to change, e.g. tuned values from tune_model."
    )
    validation: dict[str, Any] | None = Field(None, description=VALIDATION_HELP)


class TrainModel(Tool[TrainInput]):
    name: ClassVar[str] = "train_model"
    kind = "compute"
    description: ClassVar[str] = (
        "Train one model to predict a target, with the app's automatic data preparation and its "
        "validation (held-out rows), and save it as a run. Returns its scores on unseen rows, "
        "cross-validation when available and the features that mattered. Takes seconds to "
        "minutes. Start with the default (reference) model, then a stronger one to compare."
    )
    Input: ClassVar[type[BaseModel]] = TrainInput

    def __init__(self, service: MlService) -> None:
        self.service = service

    def label(self, args: TrainInput) -> str:
        model = (
            catalogue.CATALOGUE[args.model].label
            if args.model in catalogue.CATALOGUE
            else "a model"
        )
        return f"Training {model} to predict {args.target}"

    def run(self, ctx: ToolContext, args: TrainInput) -> ToolOutput:
        try:
            job = self.service.start(
                ctx.dataset_id,
                args.target,
                args.model,
                args.settings,
                ctx.conditions,
                validation=_validation(args.validation),
            )
        except MlError as error:
            raise _failed(error) from error
        job = wait_for(
            lambda: self.service.job(job.id),
            lambda j: j.status != "running",
            ctx,
            describe=lambda j: (f"Training: {j.stage}", None),
            cancel=lambda: self.service.cancel(job.id),
            timeout=TRAIN_TIMEOUT,
        )
        if job.status != "done" or not job.run_id:
            raise ToolError(f"Training {job.status}: {job.error or 'no detail'}")
        run = self.service.run(ctx.dataset_id, job.run_id)
        content = run_content(run)
        summary = f"{content['model_label']}: {run.primary} = {_r(run.score, 3)}"
        return ToolOutput(summary, content, _link(run.target, run.id))


class CompareInput(BaseModel):
    target: str = Field(description="The column to predict.")
    models: list[str] | None = Field(
        None,
        description="Model keys to compare (2 to 6); empty for the app's recommendation.",
        max_length=6,
    )
    validation: dict[str, Any] | None = Field(None, description=VALIDATION_HELP)


class CompareModels(Tool[CompareInput]):
    name: ClassVar[str] = "compare_models"
    kind = "compute"
    description: ClassVar[str] = (
        "Train several models on the same rows, preparation and validation, and rank them by the "
        "primary metric. Each one is saved as a run. Use it to pick the best family before "
        "tuning. Can take minutes on large data."
    )
    Input: ClassVar[type[BaseModel]] = CompareInput

    def __init__(self, service: MlService) -> None:
        self.service = service

    def label(self, args: CompareInput) -> str:
        return f"Comparing models to predict {args.target}"

    def _recommended(self, ctx: ToolContext, target: str) -> list[str]:
        candidate = next(
            (
                c
                for c in self.service.targets(ctx.dataset_id, ctx.conditions).candidates
                if c.name == target
            ),
            None,
        )
        if candidate is None or candidate.task is None:
            reason = f": {candidate.reason}" if candidate and candidate.reason else ""
            raise ToolError(f"{target!r} cannot be predicted{reason}.")
        context = catalogue.Context(rows=1_000, task=candidate.task)
        keys = [key for key, _ in catalogue.recommend(context)]
        return list(dict.fromkeys([catalogue.default_model(candidate.task), *keys]))

    def run(self, ctx: ToolContext, args: CompareInput) -> ToolOutput:
        models = args.models or self._recommended(ctx, args.target)
        try:
            job = self.service.compare(
                ctx.dataset_id,
                args.target,
                CompareRequest(models=models),
                ctx.conditions,
                validation=_validation(args.validation),
            )
        except MlError as error:
            raise _failed(error) from error
        job = wait_for(
            lambda: self.service.comparison(job.id),
            lambda j: j.status == "done",
            ctx,
            describe=lambda j: (
                f"{_finished(j)} of {len(j.items)} models",
                _finished(j) / len(j.items),
            ),
            timeout=TRAIN_TIMEOUT * 2,
        )
        lower = job.metric in evaluate.LOWER_IS_BETTER
        done = [i for i in job.items if i.status == "done" and i.score is not None]
        done.sort(key=lambda i: i.score if lower else -(i.score or 0))  # type: ignore[arg-type,operator]
        ranking = [
            {
                "model": i.key,
                "label": i.label,
                "score": _r(i.score),
                "run_id": i.run_id,
                "seconds": _r(i.seconds, 3),
            }
            for i in done
        ]
        failed = [{"model": i.key, "error": i.error} for i in job.items if i.status == "failed"]
        content = {
            "metric": job.metric,
            "lower_is_better": lower,
            "ranking": ranking,
            "failed": failed,
        }
        best = done[0] if done else None
        summary = (
            f"Best: {best.label} ({job.metric} {_r(best.score, 3)})"
            if best
            else "No model finished"
        )
        return ToolOutput(summary, content, _link(args.target, best.run_id if best else None))


class TuneInput(BaseModel):
    target: str = Field(description="The column to predict.")
    model: str = Field(description="Model key to tune, usually the best of compare_models.")
    budget_seconds: int = Field(60, ge=10, le=600, description="Time to spend searching.")
    validation: dict[str, Any] | None = Field(None, description=VALIDATION_HELP)


class TuneModel(Tool[TuneInput]):
    name: ClassVar[str] = "tune_model"
    kind = "compute"
    description: ClassVar[str] = (
        "Search a model's best settings (Optuna, cross-validated on the training rows) within a "
        "time budget. Returns the best settings and score against the default settings' score. "
        "Then call train_model with these settings to get the tuned run on unseen rows."
    )
    Input: ClassVar[type[BaseModel]] = TuneInput

    def __init__(self, service: MlService) -> None:
        self.service = service

    def label(self, args: TuneInput) -> str:
        return f"Tuning {args.model} for {args.budget_seconds} s"

    def run(self, ctx: ToolContext, args: TuneInput) -> ToolOutput:
        try:
            job = self.service.tune(
                ctx.dataset_id,
                args.target,
                TuneStart(model=args.model, budget_seconds=args.budget_seconds),
                ctx.conditions,
                validation=_validation(args.validation),
            )
        except (MlError, ValueError) as error:
            raise _failed(error) from error
        job = wait_for(
            lambda: self.service.tuning(job.id),
            lambda j: j.status != "running",
            ctx,
            describe=lambda j: (
                f"{len(j.progress.trials) if j.progress else 0} trials",
                min((j.progress.seconds if j.progress else 0) / args.budget_seconds, 1.0),
            ),
            cancel=lambda: self.service.stop_tuning(job.id),
            timeout=args.budget_seconds + 300,
        )
        if job.status == "failed" or job.progress is None:
            raise ToolError(f"Tuning failed: {job.error or 'no trial finished'}")
        p = job.progress
        content = {
            "model": job.model,
            "metric": p.metric,
            "greater_is_better": p.greater_is_better,
            "best_value": _r(p.best_value),
            "default_value": _r(p.default_value),
            "best_settings": p.best_params,
            "trials": len(p.trials),
            "setting_importance": {k: _r(v, 3) for k, v in p.importances.items()},
        }
        best, default = _r(p.best_value, 3), _r(p.default_value, 3)
        summary = f"{job.label}: best {p.metric} {best} (defaults {default})"
        return ToolOutput(summary, content, _link(args.target))


class ExplainInput(BaseModel):
    run_id: str = Field(description="A run id from train_model or compare_models.")


class ExplainModel(Tool[ExplainInput]):
    name: ClassVar[str] = "explain_model"
    kind = "compute"
    description: ClassVar[str] = (
        "Why a trained model predicts what it does: permutation importance (score drop when a "
        "column is shuffled), SHAP contributions with how each feature pushes the prediction, "
        "and warnings (e.g. a feature that leaks the target)."
    )
    Input: ClassVar[type[BaseModel]] = ExplainInput

    def __init__(self, service: MlService) -> None:
        self.service = service

    def label(self, args: ExplainInput) -> str:
        return "Explaining the model"

    def run(self, ctx: ToolContext, args: ExplainInput) -> ToolOutput:
        try:
            run = self.service.run(ctx.dataset_id, args.run_id)
            state = wait_for(
                lambda: self.service.explanation(ctx.dataset_id, args.run_id),
                lambda s: s.status != "running",
                ctx,
                describe=lambda s: ("Computing importances and SHAP values", None),
                timeout=TRAIN_TIMEOUT,
            )
        except MlError as error:
            raise _failed(error) from error
        if state.status != "done" or state.explanation is None:
            raise ToolError(f"Explanation failed: {state.error or 'no detail'}")
        e = state.explanation
        content: dict[str, Any] = {"run_id": run.id, "model": run.model, "target": run.target}
        if e.permutation is not None:
            content["permutation"] = {
                "metric": e.permutation.metric,
                "items": [
                    {"column": i.column, "drop": _r(i.mean, 3), "std": _r(i.std, 3)}
                    for i in e.permutation.items[:12]
                ],
            }
        if e.shap is not None:
            content["shap"] = {
                "scale": e.shap.scale,
                "positive_class": e.shap.positive,
                "features": [
                    {"column": f.column, "mean_abs": _r(f.mean_abs, 3), "reading": f.reading}
                    for f in e.shap.features[:12]
                ],
            }
        elif e.shap_unavailable:
            content["shap_unavailable"] = e.shap_unavailable
        if e.words:  # free text (052): the terms that move the predictions
            content["words"] = [
                {
                    "column": w.column,
                    "method": w.method,
                    "output": w.output,
                    "sentence": w.sentence,
                    "up": [{"term": x.term, "effect": _r(x.effect, 3)} for x in w.up[:8]],
                    "down": [{"term": x.term, "effect": _r(x.effect, 3)} for x in w.down[:8]],
                }
                for w in e.words
            ]
        if e.warnings:
            content["warnings"] = e.warnings
        top = e.permutation.items[0].column if e.permutation and e.permutation.items else None
        summary = f"Most important: {top}" if top else "Explanation ready"
        return ToolOutput(summary, content, _link(run.target, run.id))


class PanelInput(BaseModel):
    target: str | None = Field(None, description="The column to predict, for the embargo.")
    period: str | None = Field(None, description="The period column, if you know it.")
    entity: str | None = Field(None, description="The entity column (an id), if you know it.")


class CheckPanel(Tool[PanelInput]):
    name: ClassVar[str] = "check_panel"
    description: ClassVar[str] = (
        "Whether the rows are panel data (the same entities at successive periods: ids by eras, "
        "shops by weeks): the period and entity columns, how many periods, and the embargo the "
        "target calls for. On a panel, random splits overestimate the score: train with the "
        "validation it returns (walk-forward by period)."
    )
    Input: ClassVar[type[BaseModel]] = PanelInput

    def __init__(self, service: MlService) -> None:
        self.service = service

    def label(self, args: PanelInput) -> str:
        return "Checking for panel data"

    def run(self, ctx: ToolContext, args: PanelInput) -> ToolOutput:
        try:
            found = self.service.panel(
                ctx.dataset_id, ctx.conditions, args.target, args.period, args.entity
            )
        except MlError as error:
            raise _failed(error) from error
        content: dict[str, Any] = compact(found)
        if found.panel and found.period:
            content["validation"] = {
                "method": "panel_walk_forward",
                "period_column": found.period,
                "entity_column": found.entity,
                "embargo": found.embargo or 0,
            }
        summary = found.message or "Not panel data"
        return ToolOutput(summary, content, _link(args.target) if args.target else None)


class RunsInput(BaseModel):
    target: str | None = Field(None, description="Only the runs predicting this column.")


class ListRuns(Tool[RunsInput]):
    name: ClassVar[str] = "list_ml_runs"
    description: ClassVar[str] = (
        "Models already trained on this dataset (newest first) with their metric and score, to "
        "reuse or compare them instead of training again."
    )
    Input: ClassVar[type[BaseModel]] = RunsInput

    def __init__(self, service: MlService) -> None:
        self.service = service

    def label(self, args: RunsInput) -> str:
        return "Listing trained models"

    def run(self, ctx: ToolContext, args: RunsInput) -> ToolOutput:
        runs = self.service.runs(ctx.dataset_id, args.target)
        items = [
            {
                "run_id": r.id,
                "target": r.target,
                "model": r.model,
                "label": r.label,
                "metric": r.metric,
                "score": _r(r.score),
                "created_at": r.created_at,
            }
            for r in runs[:30]
        ]
        return ToolOutput(f"{len(runs)} trained models", {"runs": items})


def agent_tools(app: FastAPI) -> list[Tool[Any]]:
    from app.modules.ml.agent_auto import AutoMl  # the Auto (038), which reuses this file's help

    service: MlService = app.state.ml
    return [
        ListTargets(service),
        ListModels(),
        CheckPanel(service),
        TrainModel(service),
        CompareModels(service),
        TuneModel(service),
        ExplainModel(service),
        ModelErrors(service),
        ListRuns(service),
        AutoMl(service, app.state.ml_auto),
    ]
