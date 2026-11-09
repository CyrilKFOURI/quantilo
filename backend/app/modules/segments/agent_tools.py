"""What the Segments module lets the AI agent do (specs 032 and 033): find segments and describe
them, save them as a dataset (confirmed), and find the rows that are unusual across columns."""

from typing import Any, ClassVar, Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field
from quantilo.segments.engine import SegmentRequest

from app.core.errors import AppError
from app.core.tools import AppLink, Tool, ToolContext, ToolError, ToolOutput, compact, wait_for
from app.modules.segments.service import SegmentJob, SegmentsService, UnusualStart

TIMEOUT = 600.0
LISTED = 10  # unusual rows the model reads


def _link(params: dict[str, str]) -> AppLink:
    return AppLink("segments", params, "Open in Segments")


def _wait(service: SegmentsService, job: SegmentJob, ctx: ToolContext) -> SegmentJob:
    found = wait_for(
        lambda: service.job(job.id),
        lambda j: j.status != "running",
        ctx,
        describe=lambda j: (j.message or f"Stage: {j.stage}", j.progress),
        cancel=lambda: service.cancel(job.id),
        timeout=TIMEOUT,
    )
    if found.status != "done" or not found.run_id:
        raise ToolError(f"{found.kind.capitalize()} {found.status}: {found.error or 'no detail'}")
    return found


class SegmentsInput(BaseModel):
    columns: list[str] | None = Field(
        None, description="Columns to group rows on; empty: the app chooses (recommended)."
    )
    method: Literal["auto", "kmeans", "kprototypes", "gmm", "ward", "hdbscan"] = Field(
        "auto", description="Auto: k-means, or k-prototypes with many category columns."
    )
    k: int | None = Field(None, ge=2, le=20, description="Number of segments; empty: chosen.")


class FindSegments(Tool[SegmentsInput]):
    name: ClassVar[str] = "find_segments"
    kind = "compute"
    description: ClassVar[str] = (
        "Group the rows into segments of similar rows (clustering), with the active filters: "
        "the columns used and why, how the number of segments was chosen (silhouette, elbow…), "
        "whether the segments are stable, and each segment's size, proposed name, distinctive "
        "traits and simple rules. Saved as a run. Use it for 'what kinds of customers do I "
        "have?'; not to predict a column (that is machine learning)."
    )
    Input: ClassVar[type[BaseModel]] = SegmentsInput

    def __init__(self, service: SegmentsService) -> None:
        self.service = service

    def label(self, args: SegmentsInput) -> str:
        return "Finding segments"

    def run(self, ctx: ToolContext, args: SegmentsInput) -> ToolOutput:
        request = SegmentRequest(columns=args.columns, method=args.method, k=args.k)
        try:
            job = self.service.start(ctx.dataset_id, request, ctx.conditions)
        except AppError as error:
            raise ToolError(error.message) from error
        done = _wait(self.service, job, ctx)
        assert done.run_id is not None
        result = self.service.run(ctx.dataset_id, done.run_id).result
        content: dict[str, Any] = {
            "run_id": done.run_id,
            "method": result.method_label,
            "why_method": result.method_why,
            "segments_count": result.k,
            "why_count": result.choice.why,
            "stability": compact(result.stability),
            "rows": result.rows,
            "columns_used": [d.column for d in result.preparation.used],
            "columns_left_out": {
                d.column: d.why[0] for d in result.preparation.decisions if not d.use and d.why
            },
            "warnings": result.warnings,
            "segments": [
                {
                    "id": s.id,
                    "name": s.name,
                    "rows": s.size,
                    "share": round(s.share, 3),
                    "traits": [t.text for t in s.traits],
                    "rules": s.rules.text if s.rules else None,
                    "rule": s.rules.rules[0].text if s.rules and s.rules.rules else None,
                }
                for s in result.segments
            ],
        }
        return ToolOutput(
            f"{result.k} segments ({result.method_label})",
            compact(content, max_items=12),
            _link({"run": done.run_id}),
        )


class SaveInput(BaseModel):
    run_id: str = Field(description="The run of find_segments to save.")


class SaveSegments(Tool[SaveInput]):
    name: ClassVar[str] = "save_segments"
    kind = "write"
    description: ClassVar[str] = (
        "Save a segmentation as a new dataset: the same (filtered) rows with a `segment` column "
        "holding each row's segment name, so the segments can be compared, used in KPIs or ML. "
        "The source is not changed. The user confirms first."
    )
    Input: ClassVar[type[BaseModel]] = SaveInput

    def __init__(self, service: SegmentsService) -> None:
        self.service = service

    def label(self, args: SaveInput) -> str:
        return "Saving the segments as a dataset"

    def confirmation(self, args: SaveInput) -> str:
        return "Create a new dataset with a segment column (the source stays as it is)."

    def run(self, ctx: ToolContext, args: SaveInput) -> ToolOutput:
        try:
            created = self.service.create_dataset(ctx.dataset_id, args.run_id)
        except AppError as error:
            raise ToolError(error.message) from error
        return ToolOutput(
            "New dataset created",
            {"dataset_id": created},
            AppLink("dataset", {"dataset": created}, "Open the new dataset"),
        )


class UnusualInput(BaseModel):
    columns: list[str] | None = Field(None, description="Columns to read; empty: the app's.")
    method: Literal["auto", "isolation_forest", "mahalanobis", "lof"] = Field(
        "auto", description="Auto: robust Mahalanobis on a few numbers, else Isolation Forest."
    )


class FindUnusualRows(Tool[UnusualInput]):
    name: ClassVar[str] = "find_unusual_rows"
    kind = "compute"
    description: ClassVar[str] = (
        "Find rows that are unusual across several columns (normal on each column, odd "
        "together), with the active filters: how many rows stand out (threshold where the "
        "scores break), and for the most unusual ones, the columns that make them so. Rows the "
        "user marked as expected are left out. Not for outliers of a single column (use the "
        "variable's outliers)."
    )
    Input: ClassVar[type[BaseModel]] = UnusualInput

    def __init__(self, service: SegmentsService) -> None:
        self.service = service

    def label(self, args: UnusualInput) -> str:
        return "Looking for unusual rows"

    def run(self, ctx: ToolContext, args: UnusualInput) -> ToolOutput:
        request = UnusualStart(columns=args.columns, method=args.method)
        try:
            job = self.service.start_unusual(ctx.dataset_id, request, ctx.conditions)
        except AppError as error:
            raise ToolError(error.message) from error
        done = _wait(self.service, job, ctx)
        assert done.run_id is not None
        result = self.service.unusual_run(ctx.dataset_id, done.run_id).result
        content = {
            "run_id": done.run_id,
            "method": result.method_label,
            "why_method": result.method_why,
            "rows_scored": result.rows,
            "unusual_rows": result.threshold.rows,
            "why_threshold": result.threshold.why,
            "hidden_as_expected": result.hidden_expected,
            "columns_used": result.used,
            "most_unusual": [
                {
                    "row": r.row,
                    "score": round(r.score, 4),
                    "why": [x for x in [r.combination, *(t.text for t in r.reasons)] if x],
                }
                for r in result.listed[:LISTED]
            ],
        }
        return ToolOutput(
            f"{result.threshold.rows:,} unusual rows of {result.rows:,}",
            compact(content, max_items=LISTED),
            _link({"section": "unusual", "unusual": done.run_id}),
        )


def agent_tools(app: FastAPI) -> list[Tool[Any]]:
    service: SegmentsService = app.state.segments
    return [FindSegments(service), SaveSegments(service), FindUnusualRows(service)]
