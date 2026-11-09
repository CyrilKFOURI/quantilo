"""What the recipes module lets the AI agent do (030): list the recipes (with how well the open
dataset fits each), and replay one on the open dataset, which the user confirms first."""

from typing import Any, ClassVar

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.core.errors import AppError
from app.core.tools import AppLink, Tool, ToolContext, ToolError, ToolOutput, wait_for
from app.modules.recipes.service import RecipeService


def _done(run: Any) -> int:
    steps = run.report.steps if run.report else []
    return sum(1 for s in steps if s.status not in ("waiting", "running"))


class NoInput(BaseModel):
    pass


class ListRecipes(Tool[NoInput]):
    name: ClassVar[str] = "list_recipes"
    description: ClassVar[str] = (
        "The recipes kept in the app (work saved on a dataset to replay on a new file: column "
        "corrections, filters, KPIs, forecasts, models, outputs): name, steps, last run, and how "
        "many of each recipe's columns the open dataset has. Use it before run_recipe."
    )
    Input: ClassVar[type[BaseModel]] = NoInput

    def __init__(self, service: RecipeService) -> None:
        self.service = service

    def label(self, args: NoInput) -> str:
        return "Listing the recipes"

    def run(self, ctx: ToolContext, args: NoInput) -> ToolOutput:
        recipes = self.service.all()
        content = []
        for recipe in recipes:
            try:
                fit = self.service.plan(recipe.id, ctx.dataset_id, None)
                matched = sum(1 for m in fit.matching.columns if m.column is not None)
                skipped = [s.sentence for s in fit.steps if s.skip]
                fits: dict[str, Any] = {
                    "columns_found": f"{matched} of {len(fit.matching.columns)}",
                    "steps_skipped": skipped[:5],
                }
            except AppError as error:
                fits = {"problem": error.message}
            last = recipe.last_run
            content.append(
                {
                    "id": recipe.id,
                    "name": recipe.name,
                    "version": recipe.version,
                    "steps": recipe.steps,
                    "made_on": recipe.source,
                    "last_run": f"{last.created_at[:10]} on {last.dataset_name}: {last.status}"
                    if last
                    else None,
                    "on_this_dataset": fits,
                }
            )
        return ToolOutput(
            f"{len(recipes)} recipe(s)", {"recipes": content}, AppLink("recipes", {}, "Recipes")
        )


class RunInput(BaseModel):
    recipe: str = Field(min_length=1, max_length=200, description="The recipe's id or name.")


class RunRecipe(Tool[RunInput]):
    name: ClassVar[str] = "run_recipe"
    kind = "write"
    description: ClassVar[str] = (
        "Replay a recipe on the open dataset: columns are paired automatically (same name, or the "
        "same once case, accents and underscores are ignored; close names are not used), steps "
        "on missing columns are skipped. Adds KPIs, analyses and models to the dataset. The user "
        "confirms first. Returns each step's status and the key figures next to the last run."
    )
    Input: ClassVar[type[BaseModel]] = RunInput

    def __init__(self, service: RecipeService) -> None:
        self.service = service

    def _find(self, key: str) -> Any:
        recipes = self.service.all()
        found = next((r for r in recipes if r.id == key), None) or next(
            (r for r in recipes if r.name.casefold() == key.strip().casefold()), None
        )
        if found is None:
            names = ", ".join(r.name for r in recipes) or "none"
            raise ToolError(f"No recipe {key!r}. Recipes: {names}.")
        return found

    def label(self, args: RunInput) -> str:
        return f"Running the recipe “{args.recipe}”"

    def confirmation(self, args: RunInput) -> str:
        return (
            f"Run the recipe “{args.recipe}” on this dataset: it corrects columns and adds its "
            "KPIs, analyses and models here."
        )

    def run(self, ctx: ToolContext, args: RunInput) -> ToolOutput:
        recipe = self._find(args.recipe)
        try:
            started = self.service.start(recipe.id, ctx.dataset_id, None)
        except AppError as error:
            raise ToolError(f"The recipe can't run here: {error.message}") from error
        detail = wait_for(
            lambda: self.service.run_detail(started.id),
            lambda run: run.status != "running",
            ctx,
            lambda run: (f"{_done(run)} steps done", None),
            cancel=lambda: self.service.cancel(started.id),
            timeout=1800,
        )
        steps = [
            {"step": s.sentence, "status": s.status, "why": s.message, "warnings": s.warnings}
            for s in (detail.report.steps if detail.report else [])
        ]
        figures = [
            {
                "figure": c.label,
                "value": c.value,
                "previous": c.previous,
                "change_pct": None if c.change_pct is None else round(100 * c.change_pct, 2),
            }
            for c in detail.compared
        ]
        content: dict[str, Any] = {"status": detail.status, "steps": steps, "key_figures": figures}
        if detail.changes is not None:
            content["what_changed"] = detail.changes.headline
        if detail.error:
            content["error"] = detail.error
        return ToolOutput(
            f"“{recipe.name}”: {detail.status}",
            content,
            AppLink("recipe", {"recipe": recipe.id, "run": started.id}, "Open the report"),
        )


def agent_tools(app: FastAPI) -> list[Tool[Any]]:
    service: RecipeService = app.state.recipes
    return [ListRecipes(service), RunRecipe(service)]
