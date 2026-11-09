"""Predict on new data with a run, and export its model (037). Importing a model file is not
offered: a joblib file runs code when it is opened (the library's `dx.load_model` does that,
after checking the model card)."""

from typing import Annotated

from fastapi import APIRouter, Query, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field
from quantilo.ml.predict import ModelCard, ModelCheck, RowExplanation

from app.modules.ml.predictions import (
    Prediction,
    PredictionJob,
    PredictionRows,
    PredictionService,
    PredictionSummary,
)

router = APIRouter(tags=["module:ml"])


def _service(request: Request) -> PredictionService:
    return request.app.state.ml_predictions


Source = Annotated[str, Query(min_length=1, description="The dataset to predict.")]


class PredictStart(BaseModel):
    source: str = Field(min_length=1, description="The dataset to predict.")
    keep: list[str] = Field(
        default_factory=list, max_length=200, description="Columns kept next to the predictions."
    )


class CreatedDataset(BaseModel):
    dataset_id: str


@router.get("/datasets/{dataset_id}/runs/{run_id}/card")
def card(dataset_id: str, run_id: str, request: Request) -> ModelCard:
    """What the run's model expects (columns, types, categories, ranges) and its validation."""
    return _service(request).card(dataset_id, run_id)


@router.get("/datasets/{dataset_id}/runs/{run_id}/check")
def check(dataset_id: str, run_id: str, source: Source, request: Request) -> ModelCheck:
    """Can the model predict this dataset? Ready, Check or Blocked, column by column."""
    return _service(request).check(dataset_id, run_id, source)


@router.post("/datasets/{dataset_id}/runs/{run_id}/predictions", status_code=202)
def predict(dataset_id: str, run_id: str, body: PredictStart, request: Request) -> PredictionJob:
    """Check, then predict every row in the background, 100 000 at a time (refused if blocked)."""
    return _service(request).start(dataset_id, run_id, body.source, body.keep)


@router.get("/datasets/{dataset_id}/runs/{run_id}/predictions")
def predictions(dataset_id: str, run_id: str, request: Request) -> list[PredictionSummary]:
    """The predictions made with this run, most recent first."""
    return _service(request).list(dataset_id, run_id)


@router.get("/predict-jobs/{job_id}")
def prediction_job(job_id: str, request: Request) -> PredictionJob:
    return _service(request).job(job_id)


@router.delete("/predict-jobs/{job_id}")
def cancel_prediction(job_id: str, request: Request) -> PredictionJob:
    """Stop after the current block of rows; nothing is saved."""
    return _service(request).cancel(job_id)


@router.get("/datasets/{dataset_id}/predictions/{prediction_id}")
def prediction(dataset_id: str, prediction_id: str, request: Request) -> Prediction:
    """A prediction: its check, distribution against the validation, scores if the data had the
    answer."""
    return _service(request).get(dataset_id, prediction_id)


@router.get("/datasets/{dataset_id}/predictions/{prediction_id}/rows")
def prediction_rows(
    dataset_id: str,
    prediction_id: str,
    request: Request,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    less_reliable: bool = False,
) -> PredictionRows:
    """A page of the predicted rows (only the less reliable ones on demand)."""
    return _service(request).rows(dataset_id, prediction_id, offset, limit, less_reliable)


@router.get("/datasets/{dataset_id}/predictions/{prediction_id}/rows/{row}/explanation")
def prediction_row_explanation(
    dataset_id: str, prediction_id: str, row: int, request: Request
) -> RowExplanation:
    """Why the model predicts what it does for one predicted row (0-based)."""
    return _service(request).explain_row(dataset_id, prediction_id, row)


@router.get("/datasets/{dataset_id}/predictions/{prediction_id}/predictions.csv")
def prediction_csv(dataset_id: str, prediction_id: str, request: Request) -> StreamingResponse:
    """Every predicted row with its kept columns, as CSV (formulas neutralised)."""
    service = _service(request)
    service.get(dataset_id, prediction_id)  # an unknown prediction: 404 before streaming
    return StreamingResponse(
        service.csv(dataset_id, prediction_id),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="predictions-{prediction_id}.csv"'},
    )


@router.post("/datasets/{dataset_id}/predictions/{prediction_id}/dataset", status_code=201)
def save_as_dataset(dataset_id: str, prediction_id: str, request: Request) -> CreatedDataset:
    """The predicted rows with their predictions, imported as a new dataset."""
    return CreatedDataset(dataset_id=_service(request).create_dataset(dataset_id, prediction_id))


@router.get("/datasets/{dataset_id}/runs/{run_id}/model.zip")
def export_model(dataset_id: str, run_id: str, request: Request) -> Response:
    """The model as a zip: model.joblib, model_card.json, predict.py (no data rows)."""
    name, content = _service(request).export(dataset_id, run_id)
    return Response(
        content,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )
