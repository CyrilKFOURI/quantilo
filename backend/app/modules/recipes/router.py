from typing import Annotated
from urllib.parse import quote

from fastapi import APIRouter, Depends, Request, Response
from fastapi import Path as PathParam
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from quantilo.recipes.model import MAX_BYTES

from app.modules.recipes.service import (
    CompatibleRecipe,
    RecipeDetail,
    RecipeDraft,
    RecipePlan,
    RecipeRunDetail,
    RecipeRunSummary,
    RecipeService,
    RecipeSummary,
)

router = APIRouter(tags=["module:recipes"])

Id = Annotated[str, Field(pattern=r"^[\w-]{1,64}$")]
Mapping = dict[Annotated[str, Field(max_length=500)], Annotated[str, Field(max_length=500)] | None]


def get_service(request: Request) -> RecipeService:
    return request.app.state.recipes


Service = Annotated[RecipeService, Depends(get_service)]


def attachment(name: str, content: str, media_type: str) -> Response:
    return Response(
        content,
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(name)}"},
    )


class CreateRecipe(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    dataset_id: Id
    keep: list[Annotated[str, Field(max_length=8)]] | None = Field(default=None, max_length=1000)
    active_filter: str | None = Field(default=None, max_length=8)
    report: bool = False
    csv: bool = False


class EditRecipe(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    enabled: dict[Annotated[str, Field(max_length=8)], bool] | None = None
    remove: list[Annotated[str, Field(max_length=8)]] = Field(default_factory=list, max_length=500)


class ImportRecipe(BaseModel):
    content: str = Field(min_length=2, max_length=MAX_BYTES)


class RecipeRunRequest(BaseModel):
    dataset_id: Id
    mapping: Mapping | None = None  # recipe column → file column (None: missing)


@router.get("/recipes")
def list_recipes(service: Service) -> list[RecipeSummary]:
    return service.all()


@router.get("/datasets/{dataset_id}/draft")
def draft(dataset_id: str, service: Service) -> RecipeDraft:
    """Everything kept on the dataset, as the steps of a new recipe (all of them, to tick)."""
    return service.draft(dataset_id)


@router.get("/datasets/{dataset_id}/compatible")
def compatible(dataset_id: str, service: Service) -> list[CompatibleRecipe]:
    """Recipes made on other files that this dataset fits (≥ 80 % of their columns by name)."""
    return service.compatible(dataset_id)


@router.post("/recipes", status_code=201)
def create(body: CreateRecipe, service: Service) -> RecipeSummary:
    return service.create(
        body.dataset_id, body.name, body.keep, body.active_filter, body.report, body.csv
    )


@router.post("/recipes/import", status_code=201)
def import_recipe(body: ImportRecipe, service: Service) -> RecipeSummary:
    """A `.recipe.json` file, validated against its schema (nothing in it is run)."""
    return service.import_recipe(body.content)


@router.get("/recipes/{recipe_id}")
def detail(recipe_id: str, service: Service) -> RecipeDetail:
    return service.detail(recipe_id)


@router.patch("/recipes/{recipe_id}")
def edit(recipe_id: str, body: EditRecipe, service: Service) -> RecipeDetail:
    """Rename, turn steps on or off, delete steps: a new version of the recipe."""
    return service.edit(recipe_id, body.name, body.enabled, body.remove)


@router.delete("/recipes/{recipe_id}", status_code=204)
def delete(recipe_id: str, service: Service) -> Response:
    service.delete(recipe_id)
    return Response(status_code=204)


@router.get("/recipes/{recipe_id}/export", response_class=Response)
def export_file(recipe_id: str, service: Service) -> Response:
    name, content = service.export_json(recipe_id)
    return attachment(name, content, "application/json")


@router.get("/recipes/{recipe_id}/python", response_class=Response)
def export_python(recipe_id: str, service: Service) -> Response:
    name, content = service.export_python(recipe_id)
    return attachment(name, content, "text/x-python; charset=utf-8")


@router.get("/recipes/{recipe_id}/history.csv", response_class=Response)
def history_csv(recipe_id: str, service: Service) -> Response:
    name, content = service.history_csv(recipe_id)
    return attachment(name, content, "text/csv; charset=utf-8")


@router.post("/recipes/{recipe_id}/plan")
def plan(recipe_id: str, body: RecipeRunRequest, service: Service) -> RecipePlan:
    """How the dataset's columns pair with the recipe's, and the steps that will be skipped."""
    return service.plan(recipe_id, body.dataset_id, body.mapping)


@router.post("/recipes/{recipe_id}/runs", status_code=202)
def start_run(recipe_id: str, body: RecipeRunRequest, service: Service) -> RecipeRunSummary:
    """Replay the recipe on the dataset in the background (step by step, cancellable)."""
    return service.start(recipe_id, body.dataset_id, body.mapping)


@router.get("/recipe-runs/{run_id}")
def run_detail(run_id: str, service: Service) -> RecipeRunDetail:
    """Progress, then the report: steps, key figures next to the previous run, what changed."""
    return service.run_detail(run_id)


@router.delete("/recipe-runs/{run_id}")
def cancel_run(run_id: str, service: Service) -> RecipeRunSummary:
    """Stop after the step in progress: the steps done stay, the others are cancelled."""
    return service.cancel(run_id)


@router.get("/recipe-runs/{run_id}/files/{name}", response_class=FileResponse)
def run_file(
    run_id: str, name: Annotated[str, PathParam(max_length=300)], service: Service
) -> FileResponse:
    path = service.file(run_id, name)
    return FileResponse(path, filename=path.name)
