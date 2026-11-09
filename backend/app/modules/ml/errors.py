"""Where a model is wrong (036), on every held-out prediction kept beside the run
(`<run>.heldout.npz`, saved by the service): the analysis computed on demand and cached with the
run, two runs compared, metrics per group of a sensitive column; its routes and its agent tool."""

from pathlib import Path
from typing import Annotated, Any, ClassVar

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel, Field, TypeAdapter
from quantilo.ml import error_analysis as mistakes

from app.core.tools import AppLink, Tool, ToolContext, ToolError, ToolOutput
from app.modules.ml.service import MlError, MlService, load_held_out

ANALYSIS = TypeAdapter(mistakes.ErrorAnalysis)
NO_HELD_OUT = (
    "This run was trained before error analysis existed: train it again to see its errors."
)


def held_out_path(service: MlService, dataset_id: str, run_id: str) -> Path:
    return service._folder(dataset_id) / f"{run_id}.heldout.npz"


def _held_out(service: MlService, dataset_id: str, run_id: str) -> mistakes.HeldOut:
    path = held_out_path(service, dataset_id, run_id)
    if not path.is_file():
        raise MlError(NO_HELD_OUT, "no_held_out")
    return load_held_out(path)


def errors(
    service: MlService, dataset_id: str, run_id: str, measure: mistakes.Measure | None = None
) -> mistakes.ErrorAnalysis:
    """The run's error analysis, cached beside it (the held-out rows and filters never change)."""
    run = service.run(dataset_id, run_id)
    measure = measure or mistakes.measures_for(run.task)[0]
    cached = service._folder(dataset_id) / f"{run_id}.errors-{measure}.json"
    if cached.is_file():
        return ANALYSIS.validate_json(cached.read_bytes())
    found = service._errors(
        dataset_id,
        run,
        _held_out(service, dataset_id, run_id),
        service._conditions(run_id),
        measure,
    )
    cached.write_bytes(ANALYSIS.dump_json(found))
    return found


def compared(
    service: MlService,
    dataset_id: str,
    run_id: str,
    other_id: str,
    measure: mistakes.Measure | None = None,
) -> mistakes.ErrorComparison:
    first, second = service.run(dataset_id, run_id), service.run(dataset_id, other_id)
    return service._errors_compared(
        dataset_id,
        first,
        _held_out(service, dataset_id, run_id),
        second,
        _held_out(service, dataset_id, other_id),
        service._conditions(run_id),
        measure,
    )


def fairness(service: MlService, dataset_id: str, run_id: str, column: str) -> mistakes.Fairness:
    run = service.run(dataset_id, run_id)
    held = _held_out(service, dataset_id, run_id)
    return service._fairness(dataset_id, run, held, service._conditions(run_id), column)


# --- Routes --------------------------------------------------------------------------------------

router = APIRouter()


def get_service(request: Request) -> MlService:
    return request.app.state.ml


Service = Annotated[MlService, Depends(get_service)]


@router.get("/datasets/{dataset_id}/runs/{run_id}/errors")
def run_errors(
    dataset_id: str, run_id: str, service: Service, measure: mistakes.Measure | None = None
) -> mistakes.ErrorAnalysis:
    """Where the model is wrong, on its held-out predictions: subgroups notably worse than
    average (Benjamini-Hochberg), error by column, worst predictions, hints."""
    return errors(service, dataset_id, run_id, measure)


@router.get("/datasets/{dataset_id}/runs/{run_id}/errors/compare/{other_id}")
def run_errors_compared(
    dataset_id: str,
    run_id: str,
    other_id: str,
    service: Service,
    measure: mistakes.Measure | None = None,
) -> mistakes.ErrorComparison:
    """Subgroups where the other run fixes this run's errors, and where it makes them worse."""
    return compared(service, dataset_id, run_id, other_id, measure)


@router.get("/datasets/{dataset_id}/runs/{run_id}/errors/fairness")
def run_fairness(
    dataset_id: str,
    run_id: str,
    service: Service,
    column: Annotated[str, Query(min_length=1, max_length=500)],
) -> mistakes.Fairness:
    """Metrics side by side for each group of a sensitive column (display only)."""
    return fairness(service, dataset_id, run_id, column)


# --- Agent tool ----------------------------------------------------------------------------------


def _r(value: float | None, digits: int = 4) -> float | None:
    if value is None or value != value:
        return None
    return float(f"{value:.{digits}g}")


class ErrorsInput(BaseModel):
    run_id: str = Field(description="The trained model (see list_ml_runs or train_model).")
    measure: mistakes.Measure | None = Field(
        None, description="absolute_error (numbers); misclassified or log_loss (classes)."
    )
    compare_to: str | None = Field(
        None, description="Another run of the same target: where it fixes or worsens errors."
    )
    sensitive: str | None = Field(
        None, description="A column (sex, age…) to show the metrics of each of its groups."
    )


class ModelErrors(Tool[ErrorsInput]):
    name: ClassVar[str] = "model_errors"
    kind = "compute"
    description: ClassVar[str] = (
        "Where a trained model is wrong, on its held-out predictions only: subgroups where the "
        "error is notably worse than average (found automatically, false discoveries controlled "
        "by Benjamini-Hochberg), the columns along which the error varies most, the worst "
        "predictions and careful hints (more data, an excluded column, predicting the log, class "
        "weights). Optionally another run compared subgroup by subgroup, or the metrics of each "
        "group of a sensitive column (display only)."
    )
    Input: ClassVar[type[BaseModel]] = ErrorsInput

    def __init__(self, service: MlService) -> None:
        self.service = service

    def label(self, args: ErrorsInput) -> str:
        return "Looking for where the model is wrong"

    def run(self, ctx: ToolContext, args: ErrorsInput) -> ToolOutput:
        try:
            run = self.service.run(ctx.dataset_id, args.run_id)
            found = errors(self.service, ctx.dataset_id, args.run_id, args.measure)
            content: dict[str, Any] = {
                "run_id": run.id,
                "target": run.target,
                "measure": found.label,
                "rows": found.rows,
                "source": found.source,
                "overall": _r(found.overall),
                "subgroups": [
                    {
                        "sentence": s.sentence,
                        "rows": s.rows,
                        "error": _r(s.error),
                        "q_value": _r(s.q_value, 2),
                    }
                    for s in found.slices
                ],
                "by_column": [
                    {"column": c.column, "spread": _r(c.spread), "worst": c.worst}
                    for c in found.by_column[:8]
                ],
                "worst": [
                    {"row": w.row + 1, "actual": w.actual, "predicted": w.predicted}
                    for w in found.worst[:5]
                ],
                "hints": [h.text for h in found.hints],
            }
            if found.message:
                content["message"] = found.message
            if args.compare_to:
                other = compared(
                    self.service, ctx.dataset_id, args.run_id, args.compare_to, args.measure
                )
                content["compared"] = {
                    "other_run": args.compare_to,
                    "overall": {"this": _r(other.first), "other": _r(other.second)},
                    "better": [c.sentence for c in other.better],
                    "worse": [c.sentence for c in other.worse],
                    "message": other.message,
                }
            if args.sensitive:
                groups = fairness(self.service, ctx.dataset_id, args.run_id, args.sensitive)
                content["groups"] = {
                    "column": groups.column,
                    "positive_class": groups.positive,
                    "groups": [
                        {
                            "group": g.group,
                            "rows": g.rows,
                            **{k: _r(v, 3) for k, v in g.values.items()},
                        }
                        for g in groups.groups
                    ],
                    "largest_gap": {k: _r(v, 3) for k, v in groups.gaps.items()},
                }
        except MlError as error:
            raise ToolError(str(getattr(error, "message", None) or error)) from error
        summary = found.slices[0].sentence if found.slices else (found.message or "No subgroup")
        link = AppLink("ml", {"target": run.target, "run": run.id}, "Open in Machine learning")
        return ToolOutput(summary, content, link)
