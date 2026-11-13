from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from quantilo.vision.task import Task

from app.modules.vision.service import (
    MislabelledOut,
    TaskOut,
    VisionJob,
    VisionRun,
    VisionRunRequest,
    VisionRunSummary,
    VisionService,
    WeightOut,
)

router = APIRouter(tags=["module:vision"])

# A served map is data, never a page (as the image files of 106).
FILE_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Content-Security-Policy": "default-src 'none'; sandbox",
    "Cache-Control": "private, max-age=86400, immutable",
    "Cross-Origin-Resource-Policy": "same-origin",
}


def get_service(request: Request) -> VisionService:
    return request.app.state.vision


Service = Annotated[VisionService, Depends(get_service)]


@router.get("/datasets/{dataset_id}/task")
def task(
    dataset_id: str,
    service: Service,
    task: Annotated[Task | None, Query(description="Impose another possible task.")] = None,
    target: Annotated[
        str | None, Query(max_length=200, description="The column a regression predicts.")
    ] = None,
) -> TaskOut:
    """The task the rules read in the set (R-VML), the split a run would use (R-VSP), the models
    it would train, and whether the deep vectors are available (with the reason when not)."""
    return service.task(dataset_id, task, target)


@router.post("/datasets/{dataset_id}/runs", status_code=202)
def train(dataset_id: str, service: Service, request: VisionRunRequest | None = None) -> VisionJob:
    """Train and evaluate every first-plan model in the background."""
    return service.start(dataset_id, request or VisionRunRequest())


@router.get("/jobs/{job_id}")
def job(job_id: str, service: Service) -> VisionJob:
    return service.job(job_id)


@router.delete("/jobs/{job_id}")
def cancel(job_id: str, service: Service) -> VisionJob:
    """Stop at the next checkpoint; nothing is kept."""
    return service.cancel(job_id)


@router.get("/datasets/{dataset_id}/runs")
def runs(dataset_id: str, service: Service) -> list[VisionRunSummary]:
    return service.runs(dataset_id)


@router.get("/datasets/{dataset_id}/runs/{run_id}", response_model=VisionRun)
def run(dataset_id: str, run_id: str, service: Service) -> Any:
    """The whole result: task, split, every model's report (a metric that cannot be computed,
    such as an AUROC without defects, is null)."""
    return JSONResponse(service.run(dataset_id, run_id))


@router.get("/datasets/{dataset_id}/mislabelled")
def mislabelled(dataset_id: str, service: Service) -> MislabelledOut:
    """R-VQ-16: the images whose label the latest classification run finds unlikely (out of
    fold: their label under 10 %, another class over 80 %). A suggestion; nothing is excluded."""
    return service.mislabelled(dataset_id)


@router.get("/datasets/{dataset_id}/runs/{run_id}/predictions")
def predictions(dataset_id: str, run_id: str, service: Service) -> list[dict[str, Any]]:
    """One row per scored image: its row and image id, its fold, the truth, and each model's
    prediction and probability (classification) or score (anomalies), or its cluster."""
    return service.predictions(dataset_id, run_id)


@router.get("/datasets/{dataset_id}/runs/{run_id}/maps/{model}/{row}", response_class=FileResponse)
def anomaly_map(
    dataset_id: str, run_id: str, model: str, row: int, service: Service
) -> FileResponse:
    """A kept anomaly map of one table row (8-bit grey PNG on the model's `map_range`)."""
    path = service.image(dataset_id, run_id, "maps", model, row)
    return FileResponse(path, media_type="image/png", headers=FILE_HEADERS)


@router.get(
    "/datasets/{dataset_id}/runs/{run_id}/reconstructions/{model}/{row}",
    response_class=FileResponse,
)
def reconstruction(
    dataset_id: str, run_id: str, model: str, row: int, service: Service
) -> FileResponse:
    """What an autoencoder expected for one table row (RGB PNG at the model's side)."""
    path = service.image(dataset_id, run_id, "recon", model, row)
    return FileResponse(path, media_type="image/png", headers=FILE_HEADERS)


@router.get("/weights")
def weights(service: Service) -> list[WeightOut]:
    """Every pretrained weights file of the vision models: size, licence, on this machine?"""
    return service.weights()


@router.post("/weights/{key}/download", status_code=202)
def download(key: str, service: Service) -> WeightOut:
    """Fetch one weights file once, after the user confirmed (size and licence shown first)."""
    return service.download(key)


@router.delete("/datasets/{dataset_id}/runs/{run_id}", status_code=204)
def delete(dataset_id: str, run_id: str, service: Service) -> Response:
    service.delete(dataset_id, run_id)
    return Response(status_code=204)
