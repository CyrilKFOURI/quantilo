from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Query, Request, Response
from pydantic import BaseModel, Field
from quantilo.synthetic.compare import ColumnDetail, PairDetail
from quantilo.synthetic.plan import Plan
from quantilo.synthetic.privacy import Disclosure
from quantilo.synthetic.utility import Utility

from app.modules.synthetic.service import (
    Origin,
    RunRequest,
    SyntheticError,
    SyntheticJob,
    SyntheticPreview,
    SyntheticRun,
    SyntheticRunSummary,
    SyntheticService,
)

router = APIRouter(tags=["module:synthetic"])


def get_service(request: Request) -> SyntheticService:
    return request.app.state.synthetic


Service = Annotated[SyntheticService, Depends(get_service)]


class UtilityRequest(BaseModel):
    target: str = Field(min_length=1)


class DisclosureRequest(BaseModel):
    known: list[str] = Field(min_length=1, max_length=10)
    sensitive: str = Field(min_length=1)


class CreatedDataset(BaseModel):
    dataset_id: str


@router.get("/datasets/{dataset_id}/plan")
def plan(dataset_id: str, service: Service) -> Plan:
    """What happens to each column (rule and reason) and the recommended method."""
    return service.plan(dataset_id)


@router.post("/datasets/{dataset_id}/runs", status_code=202)
def generate(dataset_id: str, service: Service, request: RunRequest | None = None) -> SyntheticJob:
    """Generate in the background: learning, generating, evaluating, saving."""
    return service.start(dataset_id, request or RunRequest())


@router.get("/jobs/{job_id}")
def job(job_id: str, service: Service) -> SyntheticJob:
    return service.job(job_id)


@router.delete("/jobs/{job_id}")
def cancel(job_id: str, service: Service) -> SyntheticJob:
    """Stop at the next step; nothing is saved."""
    return service.cancel(job_id)


@router.get("/datasets/{dataset_id}/runs")
def runs(dataset_id: str, service: Service) -> list[SyntheticRunSummary]:
    return service.runs(dataset_id)


@router.get("/datasets/{dataset_id}/runs/compare")
def compare(
    dataset_id: str,
    service: Service,
    runs: Annotated[str, Query(description="2 to 5 run ids, comma-separated.")],
) -> list[SyntheticRunSummary]:
    return service.compare(dataset_id, [r for r in runs.split(",") if r])


@router.get("/datasets/{dataset_id}/runs/{run_id}")
def run(dataset_id: str, run_id: str, service: Service) -> SyntheticRun:
    return service.run(dataset_id, run_id)


@router.delete("/datasets/{dataset_id}/runs/{run_id}", status_code=204)
def delete(dataset_id: str, run_id: str, service: Service) -> Response:
    service.delete(dataset_id, run_id)
    return Response(status_code=204)


@router.get("/datasets/{dataset_id}/runs/{run_id}/preview")
def preview(dataset_id: str, run_id: str, service: Service) -> SyntheticPreview:
    return service.preview(dataset_id, run_id)


@router.get("/datasets/{dataset_id}/runs/{run_id}/download")
def download(
    dataset_id: str, run_id: str, service: Service, format: Literal["csv", "parquet"] = "csv"
) -> Response:
    content = service.download(dataset_id, run_id, format)
    media = "text/csv" if format == "csv" else "application/vnd.apache.parquet"
    return Response(
        content,
        media_type=media,
        headers={"Content-Disposition": f'attachment; filename="synthetic-{run_id}.{format}"'},
    )


@router.post("/datasets/{dataset_id}/runs/{run_id}/dataset")
def create_dataset(dataset_id: str, run_id: str, service: Service) -> CreatedDataset:
    """Save the synthetic rows as a new dataset with the source's column types."""
    return CreatedDataset(dataset_id=service.create_dataset(dataset_id, run_id))


@router.post("/datasets/{dataset_id}/runs/{run_id}/utility")
def utility(dataset_id: str, run_id: str, body: UtilityRequest, service: Service) -> Utility:
    """Trained on synthetic vs on real, both tested on the same held-out real rows."""
    return service.utility(dataset_id, run_id, body.target)


@router.post("/datasets/{dataset_id}/runs/{run_id}/disclosure")
def disclosure(
    dataset_id: str, run_id: str, body: DisclosureRequest, service: Service
) -> Disclosure:
    """How well known columns let an attacker guess a sensitive one (CAP)."""
    return service.disclosure(dataset_id, run_id, body.known, body.sensitive)


@router.get("/datasets/{dataset_id}/runs/{run_id}/columns/{column}")
def column(
    dataset_id: str,
    run_id: str,
    column: str,
    service: Service,
    bins: Annotated[int | None, Query(ge=2, le=100)] = None,
    target: str | None = None,
) -> ColumnDetail:
    """Real and synthetic side by side for one column (target overlay optional)."""
    return service.column(dataset_id, run_id, column, bins, target)


@router.get("/datasets/{dataset_id}/runs/{run_id}/pairs")
def pair(
    dataset_id: str,
    run_id: str,
    service: Service,
    a: Annotated[str, Query(min_length=1)],
    b: Annotated[str, Query(min_length=1)],
) -> PairDetail:
    return service.pair(dataset_id, run_id, a, b)


@router.get("/origin/{dataset_id}")
def origin(dataset_id: str, service: Service) -> Origin | None:
    """The source and run of a dataset created from synthetic data (null otherwise)."""
    return service.origin(dataset_id)


__all__ = ["SyntheticError", "router"]
