from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

# R-PRJ-08: 1 to 80 characters once the spaces around are removed
ProjectName = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=80)]


class ProjectOut(BaseModel):
    id: str
    name: str
    created_at: str
    updated_at: str
    # Its datasets, cleaned versions included (they follow their original, R-PRJ-04), in import
    # order. `DatasetOut` has no project field: the home page and the top bar join on this list.
    dataset_ids: list[str]


class ProjectIn(BaseModel):
    name: ProjectName


class MoveRequest(BaseModel):
    dataset_ids: list[str] = Field(min_length=1, max_length=1000)
    # None: out of every project ("Not in a project")
    project_id: str | None
    # C1: moved out of this project only (a dataset may be in several, R-PRJ-02 lifted); None:
    # out of every other project
    from_project_id: str | None = None


class ItemsRequest(BaseModel):
    """C1: datasets added to (or removed from) one project, their other projects kept."""

    dataset_ids: list[str] = Field(min_length=1, max_length=1000)


class ImageRefOut(BaseModel):
    """C1 (R-PRJ-11): a column of a table of the project that names the images of an image set
    of the same project."""

    dataset_id: str
    column: str
    image_set_id: str
    by: str  # path, file_name or stem
    found: int
    distinct: int
    refused: int
    examples: list[str]
    rule: str
    reason: str
    applied: bool  # the column already has the role `image_ref`


class ProjectImagesOut(BaseModel):
    tables: list[str]  # its datasets that are tables
    image_sets: list[str]  # its image sets (format "images")
    references: list[ImageRefOut]
