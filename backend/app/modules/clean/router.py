from typing import Annotated
from urllib.parse import quote

from fastapi import APIRouter, Depends, Request, Response
from quantilo.cleaning.checks import FailingRows
from quantilo.cleaning.dismiss import Dismissal
from quantilo.cleaning.duplicates import DuplicateGroups
from quantilo.cleaning.entities import EntityClusters
from quantilo.cleaning.expert import ExpertMethod, ExpertOptions, MethodId
from quantilo.cleaning.facet import Facet
from quantilo.cleaning.journal import CleanJournal, JournalEntry
from quantilo.cleaning.model import Steps
from quantilo.cleaning.preview import CleanPreview
from quantilo.cleaning.replay import CleanRecipe, ReplayReport

from app.core.errors import AppError
from app.modules.clean.checks import ChecksRequest, CleanChecks
from app.modules.clean.dismissed import CleanDismissal, Dismissed
from app.modules.clean.service import (
    CleanDraft,
    CleanFacetRequest,
    CleanFamily,
    CleanJob,
    CleanLineageItem,
    CleanPreviewRequest,
    CleanService,
    CleanSettings,
    CleanSuggestions,
    DuplicatesRequest,
    EntitiesRequest,
    ExpertJob,
    ReplayRequest,
    SaveRequest,
)

router = APIRouter(tags=["module:clean"])


def get_service(request: Request) -> CleanService:
    """404 while `QUANTILO_CLEAN_ENABLED` is off (S8)."""
    service: CleanService | None = request.app.state.clean
    if service is None:
        raise AppError("Cleaning is turned off.", code="clean_disabled", status_code=404)
    return service


Service = Annotated[CleanService, Depends(get_service)]


@router.get("/settings")
def settings(service: Service) -> CleanSettings:
    return service.settings()


@router.put("/settings")
def update_settings(settings: CleanSettings, service: Service) -> CleanSettings:
    return service.update_settings(settings)


@router.get("/lineage")
def lineage(service: Service) -> list[CleanLineageItem]:
    """Every cleaned version and the dataset it was made from."""
    return service.lineage()


@router.get("/datasets/{dataset_id}/suggestions")
def suggestions(dataset_id: str, service: Service) -> CleanSuggestions:
    """The Basic suggestions, each with its reason, numbers and step (cached per import)."""
    return service.suggestions(dataset_id)


@router.get("/datasets/{dataset_id}/suggestions/advanced")
def advanced(dataset_id: str, service: Service) -> CleanSuggestions:
    """097: the Advanced layer, computed when the page opens it (cached per import)."""
    return service.advanced(dataset_id)


@router.get("/datasets/{dataset_id}/dismissed")
def dismissals(dataset_id: str, service: Service) -> list[CleanDismissal]:
    """N05: the dismissals that apply here (this family's, then every dataset's)."""
    return service.dismissals(dataset_id)


@router.post("/datasets/{dataset_id}/dismissed", status_code=201)
def dismiss(dataset_id: str, body: Dismissal, service: Service) -> Dismissed:
    """N05: "Don't suggest again": what it hides is never ticked nor applied, and stays listed."""
    return service.dismiss(dataset_id, body)


@router.delete("/dismissed/{dismissal_id}", status_code=204)
def restore(dismissal_id: str, service: Service) -> Response:
    """N05: suggest it again (nothing is ticked again)."""
    service.restore(dismissal_id)
    return Response(status_code=204)


@router.get("/datasets/{dataset_id}/checks")
def checks(dataset_id: str, service: Service) -> CleanChecks:
    """N31: the family's declared checks and what they find here (run again if the data changed)."""
    return service.checks(dataset_id)


@router.put("/datasets/{dataset_id}/checks")
def save_checks(dataset_id: str, body: ChecksRequest, service: Service) -> CleanChecks:
    """N31: declare the checks (kept on the original) and run them here; nothing is fixed."""
    return service.save_checks(dataset_id, body.checks)


@router.get("/datasets/{dataset_id}/checks/{check_id}/rows")
def check_rows(dataset_id: str, check_id: str, service: Service) -> FailingRows:
    """N31: the first rows failing a check, its columns first."""
    return service.check_rows(dataset_id, check_id)


@router.get("/datasets/{dataset_id}/expert")
def expert_methods(dataset_id: str, service: Service) -> list[ExpertMethod]:
    """101: the *More checks* and whether each applies here (read from the columns only)."""
    return service.expert_methods(dataset_id)


@router.post("/datasets/{dataset_id}/expert/{method}", status_code=202)
def run_expert(
    dataset_id: str, method: MethodId, body: ExpertOptions, service: Service
) -> ExpertJob:
    """101: run one check in the background; its suggestions are never ticked (C-30)."""
    return service.run_expert(dataset_id, method, body)


@router.get("/expert-jobs/{job_id}")
def expert_job(job_id: str, service: Service) -> ExpertJob:
    return service.expert_job(job_id)


@router.delete("/expert-jobs/{job_id}")
def cancel_expert(job_id: str, service: Service) -> ExpertJob:
    """101 (C-31): stop a check; nothing it found is kept."""
    return service.cancel_expert(job_id)


@router.get("/versions/{dataset_id}/recipe")
def recipe(dataset_id: str, service: Service) -> CleanRecipe:
    """101 (C-39): a cleaned version's steps as a recipe file, to replay on another file."""
    return service.recipe(dataset_id)


def _attachment(name: str, content: str, media_type: str) -> Response:
    return Response(
        content,
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(name)}"},
    )


@router.get("/versions/{dataset_id}/sql", response_class=Response)
def export_sql(dataset_id: str, service: Service) -> Response:
    """N45: the cleaning as one DuckDB script, step by step with its rule and reason."""
    name, content = service.export(dataset_id, "sql")
    return _attachment(name, content, "application/sql; charset=utf-8")


@router.get("/versions/{dataset_id}/python", response_class=Response)
def export_python(dataset_id: str, service: Service) -> Response:
    """N45: the cleaning as a Python script on the quantilo library."""
    name, content = service.export(dataset_id, "python")
    return _attachment(name, content, "text/x-python; charset=utf-8")


@router.get("/datasets/{dataset_id}/journal")
def draft_journal(dataset_id: str, service: Service) -> list[JournalEntry]:
    """N46: who changed what in this dataset's draft since its last version, and when."""
    return service.draft_journal(dataset_id)


@router.get("/versions/{dataset_id}/journal")
def journal(dataset_id: str, service: Service) -> CleanJournal:
    """N46: a version's cleaning log: its steps (automatic, ticked, by hand), what they changed,
    and the changes that led to it."""
    return service.journal(dataset_id)


@router.get("/versions/{dataset_id}/journal/markdown", response_class=Response)
def journal_markdown(dataset_id: str, service: Service) -> Response:
    name, content = service.journal_file(dataset_id, "markdown")
    return _attachment(name, content, "text/markdown; charset=utf-8")


@router.get("/versions/{dataset_id}/journal/html", response_class=Response)
def journal_html(dataset_id: str, service: Service) -> Response:
    name, content = service.journal_file(dataset_id, "html")
    return _attachment(name, content, "text/html; charset=utf-8")


@router.post("/versions/{dataset_id}/rebuild", status_code=202)
def rebuild(dataset_id: str, service: Service) -> CleanJob:
    """B1 (C-47): Rebuild from original: the version's steps on the dataset it was made from, as
    it is now, saved as the next cleaned version."""
    return service.rebuild(dataset_id)


@router.post("/datasets/{dataset_id}/replay")
def replay(dataset_id: str, body: ReplayRequest, service: Service) -> ReplayReport:
    """101 (M5): what a cleaning does on this dataset; nothing is saved (C-43)."""
    return service.replay(dataset_id, body)


@router.get("/datasets/{dataset_id}/draft")
def draft(dataset_id: str, service: Service) -> CleanDraft:
    """The saved draft, or else the mode's default: the reliable fixes applied (N03), or none."""
    return service.draft(dataset_id)


@router.put("/datasets/{dataset_id}/draft")
def save_draft(dataset_id: str, body: Steps, service: Service) -> CleanDraft:
    return service.save_draft(dataset_id, body.steps)


@router.delete("/datasets/{dataset_id}/draft")
def reset_draft(dataset_id: str, service: Service) -> CleanDraft:
    """Back to the mode's default (the reliable fixes, or none)."""
    return service.reset_draft(dataset_id)


@router.post("/datasets/{dataset_id}/preview")
def preview(dataset_id: str, body: CleanPreviewRequest, service: Service) -> CleanPreview:
    """Before / after of every step (or one): counts on the whole file, first rows touched."""
    return service.preview(dataset_id, body)


@router.post("/datasets/{dataset_id}/duplicates")
def duplicates(dataset_id: str, body: DuplicatesRequest, service: Service) -> DuplicateGroups:
    """N35: the groups of a removal of duplicates (B-14, B-23, A-15, A-16), side by side."""
    return service.duplicates(dataset_id, body)


@router.post("/datasets/{dataset_id}/entities")
def entities(dataset_id: str, body: EntitiesRequest, service: Service) -> EntityClusters:
    """N36: the clusters of an entity number (A-20), side by side."""
    return service.entities(dataset_id, body)


@router.post("/datasets/{dataset_id}/facet")
def facet(dataset_id: str, body: CleanFacetRequest, service: Service) -> Facet:
    """N40: a column of the draft faceted, and the first rows a selection of it picks."""
    return service.facet(dataset_id, body)


@router.post("/datasets/{dataset_id}/versions", status_code=202)
def save(dataset_id: str, body: SaveRequest, service: Service) -> CleanJob:
    """Save the steps as a new cleaned version (a dataset); the original is not changed."""
    return service.save(dataset_id, body.steps)


@router.get("/datasets/{dataset_id}/versions")
def family(dataset_id: str, service: Service) -> CleanFamily:
    """The original of this dataset and its cleaned versions, v1 first."""
    return service.family(dataset_id)


@router.get("/jobs/{job_id}")
def job(job_id: str, service: Service) -> CleanJob:
    return service.job(job_id)


@router.delete("/jobs/{job_id}")
def cancel(job_id: str, service: Service) -> CleanJob:
    return service.cancel(job_id)
