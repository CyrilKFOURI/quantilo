"""What the AI agent can do with a trained model on new data (037): check and predict another
dataset with a run, then, once the user confirms, save the predictions as a new dataset."""

from typing import Any, ClassVar

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.core.errors import AppError
from app.core.tools import AppLink, Tool, ToolContext, ToolError, ToolOutput, compact, wait_for
from app.modules.ml.predictions import PredictionService

PREDICT_TIMEOUT = 900.0


def _r(value: float | None, digits: int = 4) -> float | None:
    if value is None or value != value:
        return None
    return float(f"{value:.{digits}g}")


class PredictInput(BaseModel):
    run_id: str = Field(description="A run of this dataset (train_model, list_ml_runs).")
    data: str | None = Field(
        None,
        description="The dataset to predict: its name or id (another imported dataset); empty "
        "for the current one.",
    )
    keep: list[str] | None = Field(
        None, max_length=20, description="Columns of that data to keep next to the predictions."
    )


class Predict(Tool[PredictInput]):
    name: ClassVar[str] = "predict"
    kind = "compute"
    description: ClassVar[str] = (
        "Use a trained run on new data: checks the data first (missing columns, types, "
        "categories the model never saw, values out of the training range: ready, check or "
        "blocked), then predicts every row. Returns the check, the share of less reliable rows, "
        "the predictions' distribution against the validation's and, when the data has the "
        "target, its scores next to the validation scores. Save them with save_predictions."
    )
    Input: ClassVar[type[BaseModel]] = PredictInput

    def __init__(self, service: PredictionService) -> None:
        self.service = service

    def label(self, args: PredictInput) -> str:
        return f"Predicting {args.data or 'this dataset'} with the model"

    def _dataset(self, ctx: ToolContext, data: str | None) -> str:
        if not data:
            return ctx.dataset_id
        found = self.service.columns.datasets.list_all()
        match = [d for d in found if data in (d.id, d.name) and d.status == "ready"]
        if not match:
            names = ", ".join(d.name for d in found if d.status == "ready")[:500]
            raise ToolError(f"No ready dataset named {data!r}. Datasets: {names}.")
        return match[0].id

    def run(self, ctx: ToolContext, args: PredictInput) -> ToolOutput:
        source = self._dataset(ctx, args.data)
        try:
            checked = self.service.check(ctx.dataset_id, args.run_id, source)
            if checked.level == "blocked":
                raise ToolError(f"{checked.message} Nothing was predicted.")
            job = self.service.start(ctx.dataset_id, args.run_id, source, args.keep)
        except AppError as error:
            raise ToolError(error.message) from error
        job = wait_for(
            lambda: self.service.job(job.id),
            lambda j: j.status != "running",
            ctx,
            describe=lambda j: (f"{j.rows:,} of {j.total:,} rows", j.rows / (j.total or 1)),
            cancel=lambda: self.service.cancel(job.id),
            timeout=PREDICT_TIMEOUT,
        )
        if job.status != "done" or not job.prediction_id:
            raise ToolError(f"Prediction {job.status}: {job.error or 'no detail'}")
        found = self.service.get(ctx.dataset_id, job.prediction_id)
        summary = found.summary
        assert summary is not None
        content: dict[str, Any] = {
            "prediction_id": found.id,
            "data": found.source_name,
            "rows": found.rows,
            "less_reliable_rows": found.less_reliable,
            "check": {
                "level": checked.level,
                "message": checked.message,
                "problems": checked.problems,
                "columns": [
                    compact(c) for c in checked.columns if c.note is not None or c.converted
                ][:15],
                "ignored_columns": checked.ignored[:30],
            },
            "distribution": compact(summary.distribution),
        }
        if summary.interval is not None:
            content["interval"] = compact(summary.interval)
        if summary.scores is not None:
            content["scores_on_this_data"] = {k: _r(v) for k, v in summary.scores.items()}
            content["validation_scores"] = {k: _r(v) for k, v in summary.validation_scores.items()}
            content["primary_metric"] = summary.primary
        run = self.service.ml.run(ctx.dataset_id, args.run_id)
        link = AppLink(
            "ml",
            {"target": run.target, "run": run.id, "prediction": found.id},
            "Open the predictions",
        )
        share = found.less_reliable / (found.rows or 1)
        line = f"{found.rows:,} rows predicted ({checked.level}; {share:.0%} less reliable)"
        return ToolOutput(line, content, link)


class SaveInput(BaseModel):
    prediction_id: str = Field(description="A prediction from the predict tool.")


class SavePredictions(Tool[SaveInput]):
    name: ClassVar[str] = "save_predictions"
    kind = "write"
    description: ClassVar[str] = (
        "Save the rows of a prediction (every column of the predicted data plus the "
        "predictions) as a new dataset, to explore them. The user confirms first."
    )
    Input: ClassVar[type[BaseModel]] = SaveInput

    def __init__(self, service: PredictionService) -> None:
        self.service = service

    def label(self, args: SaveInput) -> str:
        return "Saving the predictions as a dataset"

    def confirmation(self, args: SaveInput) -> str:
        return "Create a new dataset with the predicted rows and their predictions."

    def run(self, ctx: ToolContext, args: SaveInput) -> ToolOutput:
        try:
            created = self.service.create_dataset(ctx.dataset_id, args.prediction_id)
        except AppError as error:
            raise ToolError(error.message) from error
        return ToolOutput(
            "New dataset created",
            {"dataset_id": created},
            AppLink("dataset", {"dataset": created}, "Open the new dataset"),
        )


def prediction_tools(app: FastAPI) -> list[Tool[Any]]:
    service: PredictionService = app.state.ml_predictions
    return [Predict(service), SavePredictions(service)]
