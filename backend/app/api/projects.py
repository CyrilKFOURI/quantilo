"""Projects (087): create, rename, delete, and move datasets between them."""

from fastapi import APIRouter, Response

from app.api.deps import Projects
from app.schemas.projects import (
    ItemsRequest,
    MoveRequest,
    ProjectImagesOut,
    ProjectIn,
    ProjectOut,
)

router = APIRouter(prefix="/projects", tags=["projects"])


@router.get("")
def list_projects(service: Projects) -> list[ProjectOut]:
    """By name; each with its datasets, cleaned versions included."""
    return service.list_all()


@router.post("", status_code=201)
def create_project(body: ProjectIn, service: Projects) -> ProjectOut:
    return service.create(body.name)


@router.post("/move")
def move_datasets(body: MoveRequest, service: Projects) -> list[ProjectOut]:
    """Put datasets (and their cleaned versions) in a project, or in none; returns all."""
    return service.move(body.dataset_ids, body.project_id, body.from_project_id)


@router.get("/{project_id}")
def get_project(project_id: str, service: Projects) -> ProjectOut:
    """C1: one project (its own page)."""
    return service.get(project_id)


@router.post("/{project_id}/add")
def add_datasets(project_id: str, body: ItemsRequest, service: Projects) -> list[ProjectOut]:
    """C1: also in this project, their other projects kept (R-PRJ-02 lifted)."""
    return service.add(body.dataset_ids, project_id)


@router.post("/{project_id}/remove")
def remove_datasets(project_id: str, body: ItemsRequest, service: Projects) -> list[ProjectOut]:
    """C1: out of this project only; nothing is deleted (R-PRJ-03)."""
    return service.remove(body.dataset_ids, project_id)


@router.get("/{project_id}/images")
def project_images(project_id: str, service: Projects) -> ProjectImagesOut:
    """C1 (R-PRJ-11): its tables and image sets, and the columns that name its images."""
    return service.images(project_id)


@router.patch("/{project_id}")
def rename_project(project_id: str, body: ProjectIn, service: Projects) -> ProjectOut:
    return service.rename(project_id, body.name)


@router.delete("/{project_id}", status_code=204)
def delete_project(project_id: str, service: Projects) -> Response:
    """Its datasets are kept: they are in no project any more."""
    service.delete(project_id)
    return Response(status_code=204)
