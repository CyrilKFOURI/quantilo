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
