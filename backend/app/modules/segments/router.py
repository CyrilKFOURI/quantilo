from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request, Response
from pydantic import BaseModel, Field
from quantilo.filters.model import Condition, parse_filters
from quantilo.segments.engine import RowValue, SegmentPlan, SegmentRequest
from quantilo.text.distinctive import DistinctiveWords

from app.modules.segments.service import (
    MarkChange,
    SegmentComparison,
    SegmentJob,
    SegmentRun,
    SegmentRunSummary,
    SegmentsError,
    SegmentsService,
    UnusualMark,
    UnusualRun,
    UnusualRunSummary,
    UnusualStart,
)

router = APIRouter(tags=["module:segments"])


def get_service(request: Request) -> SegmentsService:
    return request.app.state.segments


def active_filters(
    filters: Annotated[str | None, Query(max_length=100_000)] = None,
) -> list[Condition]:
    return parse_filters(filters)


Service = Annotated[SegmentsService, Depends(get_service)]
Filters = Annotated[list[Condition], Depends(active_filters)]


class CreatedDataset(BaseModel):
    dataset_id: str


class Names(BaseModel):
    names: dict[int, str] = Field(max_length=200)


class Threshold(BaseModel):
    threshold: float | None = None  # None: the proposed one


def _csv(content: bytes, name: str) -> Response:
    return Response(
        content,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )


# --- Segments --------------------------------------------------------------------------------


@router.post("/datasets/{dataset_id}/plan")
def plan(
    dataset_id: str, service: Service, filters: Filters, request: SegmentRequest | None = None
) -> SegmentPlan:
    """The columns a segmentation would use (rule and reason for each) and the method Auto
    takes."""
    return service.plan(dataset_id, filters, request)


@router.post("/datasets/{dataset_id}/runs", status_code=202)
def segment(
    dataset_id: str, service: Service, filters: Filters, request: SegmentRequest | None = None
) -> SegmentJob:
    """Segment in the background: prepare, choose k, check stability, describe."""
    return service.start(dataset_id, request or SegmentRequest(), filters)


@router.get("/jobs/{job_id}")
def job(job_id: str, service: Service) -> SegmentJob:
    return service.job(job_id)


@router.delete("/jobs/{job_id}")
def cancel(job_id: str, service: Service) -> SegmentJob:
    """Stop at the next step; nothing is saved."""
    return service.cancel(job_id)


@router.get("/datasets/{dataset_id}/runs")
def runs(dataset_id: str, service: Service) -> list[SegmentRunSummary]:
    return service.runs(dataset_id)


@router.get("/datasets/{dataset_id}/runs/compare")
def compare(
    dataset_id: str,
    service: Service,
    runs: Annotated[str, Query(description="2 to 5 run ids, comma-separated.")],
) -> SegmentComparison:
    """The runs side by side, and how much their segments agree (adjusted Rand index)."""
    return service.compare(dataset_id, [r for r in runs.split(",") if r])


@router.get("/datasets/{dataset_id}/runs/{run_id}")
def run(dataset_id: str, run_id: str, service: Service) -> SegmentRun:
    return service.run(dataset_id, run_id)


@router.delete("/datasets/{dataset_id}/runs/{run_id}", status_code=204)
def delete(dataset_id: str, run_id: str, service: Service) -> Response:
    service.delete(dataset_id, run_id)
    return Response(status_code=204)


@router.patch("/datasets/{dataset_id}/runs/{run_id}/names")
def rename(dataset_id: str, run_id: str, body: Names, service: Service) -> SegmentRun:
    """Rename segments (empty names keep the proposed ones)."""
    return service.rename(dataset_id, run_id, body.names)


@router.post("/datasets/{dataset_id}/runs/{run_id}/tsne", status_code=202)
def tsne(dataset_id: str, run_id: str, service: Service) -> SegmentJob:
    """Draw the map with t-SNE (at most 5,000 rows), in the background."""
    return service.start_tsne(dataset_id, run_id)


@router.get("/datasets/{dataset_id}/runs/{run_id}/words")
def words(
    dataset_id: str,
    run_id: str,
    service: Service,
    column: Annotated[str, Query(min_length=1)],
) -> DistinctiveWords:
    """059: the words over-represented in each segment's texts (computed when asked)."""
    return service.words(dataset_id, run_id, column)


@router.get("/datasets/{dataset_id}/runs/{run_id}/profiles.csv")
def profiles(dataset_id: str, run_id: str, service: Service) -> Response:
    return _csv(service.profiles_csv(dataset_id, run_id), f"segments-{run_id}.csv")


@router.post("/datasets/{dataset_id}/runs/{run_id}/dataset")
def create_dataset(dataset_id: str, run_id: str, service: Service) -> CreatedDataset:
    """The rows with a `segment` column, as a new dataset (the source is not changed)."""
    return CreatedDataset(dataset_id=service.create_dataset(dataset_id, run_id))


# --- Unusual rows ----------------------------------------------------------------------------


@router.post("/datasets/{dataset_id}/unusual", status_code=202)
def find_unusual(
    dataset_id: str, service: Service, filters: Filters, request: UnusualStart | None = None
) -> SegmentJob:
    """Score every row in the background, propose a threshold, explain the most unusual rows."""
    return service.start_unusual(dataset_id, request or UnusualStart(), filters)


@router.get("/datasets/{dataset_id}/unusual")
def unusual_runs(dataset_id: str, service: Service) -> list[UnusualRunSummary]:
    return service.unusual_runs(dataset_id)


@router.get("/datasets/{dataset_id}/unusual/{run_id}")
def unusual_run(dataset_id: str, run_id: str, service: Service) -> UnusualRun:
    return service.unusual_run(dataset_id, run_id)


@router.get("/datasets/{dataset_id}/unusual/{run_id}/rows/{row}")
def unusual_row(dataset_id: str, run_id: str, row: int, service: Service) -> list[RowValue]:
    """Every value of the row, placed on its column (percentile, or share of its category)."""
    return service.row(dataset_id, run_id, row)


@router.get("/datasets/{dataset_id}/unusual/{run_id}/export.csv")
def export_unusual(
    dataset_id: str, run_id: str, service: Service, threshold: float | None = None
) -> Response:
    return _csv(service.unusual_csv(dataset_id, run_id, threshold), f"unusual-{run_id}.csv")


@router.post("/datasets/{dataset_id}/unusual/{run_id}/dataset")
def create_unusual_dataset(
    dataset_id: str, run_id: str, service: Service, body: Threshold | None = None
) -> CreatedDataset:
    """The rows with `unusual_score` and `is_unusual`, as a new dataset."""
    threshold = body.threshold if body else None
    return CreatedDataset(dataset_id=service.create_unusual_dataset(dataset_id, run_id, threshold))


@router.get("/datasets/{dataset_id}/marks")
def marks(dataset_id: str, service: Service) -> list[UnusualMark]:
    return service.marks(dataset_id)


@router.put("/datasets/{dataset_id}/marks")
def set_mark(dataset_id: str, body: MarkChange, service: Service) -> list[UnusualMark]:
    """Mark a row "expected" (it leaves the next list) or "error"; null removes the mark."""
    return service.set_mark(dataset_id, body)


__all__ = ["SegmentsError", "router"]
