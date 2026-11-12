"""Image sets (106): preview and import a folder, a ZIP, an image or a Hugging Face Parquet file as
a dataset with `format = 'images'`; its report; its files and thumbnails by image id."""

from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import FileResponse

from app.api.datasets import to_out
from app.api.deps import Datasets, Projects
from app.schemas.datasets import ImportResponse, JobOut
from app.schemas.images import (
    ImageImportRequest,
    ImagePreviewOut,
    ImagePreviewRequest,
    ImageReportOut,
    ImageSetOptions,
    QualityRuleOut,
)
from app.services.image_sets import FileKind, ImageImportOptions, ImageSetService

router = APIRouter(tags=["images"])

# Served images are data, never a page: no script, no plugin, no sniffing (§ 11 of the study).
FILE_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Content-Security-Policy": "default-src 'none'; sandbox",
    "Cache-Control": "private, max-age=3600",
    "Cross-Origin-Resource-Policy": "same-origin",
}


def get_image_sets(request: Request) -> ImageSetService:
    return request.app.state.image_sets


ImageSets = Annotated[ImageSetService, Depends(get_image_sets)]


def _options(body: ImageSetOptions) -> ImageImportOptions:
    return ImageImportOptions(
        labels=body.labels,
        label_column=body.label_column,
        ignore_rules=list(body.ignore_rules),
        thresholds=dict(body.thresholds),
    )


@router.post("/images/preview")
def preview_image_set(body: ImagePreviewRequest, service: ImageSets) -> ImagePreviewOut:
    """What the import would read, without decoding a pixel (rules R-VIS-* with their reasons)."""
    return ImagePreviewOut.model_validate(service.preview(body.path, _options(body.options)))


@router.post("/images/import", status_code=201)
def import_image_set(
    body: ImageImportRequest, service: ImageSets, datasets: Datasets, projects: Projects
) -> ImportResponse:
    if body.project_id is not None:
        projects.get(body.project_id)  # 404 before anything is imported
    record, job = service.start_import(body.path, body.name, _options(body.options))
    if body.project_id is not None:
        projects.move([record.id], body.project_id)
    return ImportResponse(
        dataset=to_out(datasets, datasets.get(record.id)),
        job=JobOut.of(datasets.job(job.id)) if job else None,
        reused=job is None,
    )


@router.get("/images/rules")
def quality_rules() -> list[QualityRuleOut]:
    """The R-VQ catalogue (105): title, reason, effect and default thresholds of each rule."""
    from quantilo.vision.quality import RULES, QualityThresholds

    defaults = QualityThresholds()
    return [
        QualityRuleOut(
            code=rule.code,
            title=rule.title,
            reason=rule.reason,
            effect=rule.effect,
            parameters={name: float(getattr(defaults, name)) for name in rule.parameters},
        )
        for rule in RULES.values()
    ]


@router.get("/datasets/{dataset_id}/images/report")
def image_set_report(dataset_id: str, service: ImageSets) -> ImageReportOut:
    return ImageReportOut.model_validate(service.report(dataset_id))


def _serve(service: ImageSetService, dataset_id: str, image_id: str, kind: FileKind):
    path, media_type = service.file(dataset_id, image_id, kind)
    return FileResponse(path, media_type=media_type, headers=FILE_HEADERS)


@router.get("/datasets/{dataset_id}/images/{image_id}/thumb", response_class=FileResponse)
def image_thumbnail(dataset_id: str, image_id: str, service: ImageSets) -> FileResponse:
    """The WebP thumbnail (long side 256 px, no EXIF)."""
    return _serve(service, dataset_id, image_id, "thumb")


@router.get("/datasets/{dataset_id}/images/{image_id}/file", response_class=FileResponse)
def image_file(dataset_id: str, image_id: str, service: ImageSets) -> FileResponse:
    """The original bytes, typed by their signature."""
    return _serve(service, dataset_id, image_id, "file")


@router.get("/datasets/{dataset_id}/images/{image_id}/mask", response_class=FileResponse)
def image_mask(dataset_id: str, image_id: str, service: ImageSets) -> FileResponse:
    """The pixel mask of a defect (R-VIS-LAY-04), a grey PNG."""
    return _serve(service, dataset_id, image_id, "mask")
