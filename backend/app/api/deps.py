from typing import Annotated

from fastapi import Depends, Request

from app.services.datasets.service import DatasetService
from app.services.projects.service import ProjectService


def get_dataset_service(request: Request) -> DatasetService:
    return request.app.state.datasets


Datasets = Annotated[DatasetService, Depends(get_dataset_service)]


def get_project_service(request: Request) -> ProjectService:
    return request.app.state.projects


Projects = Annotated[ProjectService, Depends(get_project_service)]
