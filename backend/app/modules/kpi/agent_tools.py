"""What the KPI module lets the AI agent do (spec 026b): suggest, compute and save KPIs, through
this module's own service."""

from typing import Any, ClassVar

from fastapi import FastAPI
from pydantic import BaseModel, Field
from quantilo.kpi.definition import Definition

from app.core.errors import AppError
from app.core.tools import AppLink, Tool, ToolContext, ToolError, ToolOutput, compact
from app.modules.kpi.service import KpiService

DEFINITION_HELP = (
    "A KPI definition: measures (each {id: 'm1', agg: count|sum|mean|median|min|max|"
    "distinct|true_share, column}), an optional ratio {numerator: 'm1', denominator: 'm2'}, "
    "an optional time {column, grain: day|week|month|quarter|year}, optional groups [{column}] "
    "to split by. Start "
    "from a suggestion of suggest_kpis and adapt it."
)


class NoInput(BaseModel):
    pass


class SuggestKpis(Tool[NoInput]):
    name: ClassVar[str] = "suggest_kpis"
    description: ClassVar[str] = (
        "KPIs the app suggests from the column types and roles (amounts, quantities, orders, "
        "customers, dates), best score first, each with its reasons and its full definition "
        "ready for compute_kpi, plus the best columns to split by and parent > child hierarchies."
    )
    Input: ClassVar[type[BaseModel]] = NoInput

    def __init__(self, service: KpiService) -> None:
        self.service = service

    def label(self, args: NoInput) -> str:
        return "Finding KPIs worth tracking"

    def run(self, ctx: ToolContext, args: NoInput) -> ToolOutput:
        found = self.service.suggestions(ctx.dataset_id)
        content = {
            "kpis": [
                {
                    "name": k.name,
                    "rule": k.rule,
                    "score": k.score,
                    "reasons": k.reasons,
                    "definition": k.definition.model_dump(mode="json", exclude_defaults=True),
                }
                for k in found.kpis
            ],
            "split_by": found.splits,
            "hierarchies": [
                {"parent": h.parent, "child": h.child, "reason": h.reason}
                for h in found.hierarchies
            ],
        }
        return ToolOutput(
            f"{len(found.kpis)} KPIs suggested", content, AppLink("kpi", {}, "Open the KPI builder")
        )


class DefinitionInput(BaseModel):
    definition: Definition = Field(description=DEFINITION_HELP)


class ComputeKpi(Tool[DefinitionInput]):
    name: ClassVar[str] = "compute_kpi"
    description: ClassVar[str] = (
        "Compute a KPI over the filtered rows: its value per period and per group, the headline "
        "tiles (total, last period, change) and warnings. Does not save it."
    )
    Input: ClassVar[type[BaseModel]] = DefinitionInput

    def __init__(self, service: KpiService) -> None:
        self.service = service

    def label(self, args: DefinitionInput) -> str:
        return "Computing the KPI"

    def run(self, ctx: ToolContext, args: DefinitionInput) -> ToolOutput:
        try:
            result = self.service.compute(ctx.dataset_id, args.definition, ctx.conditions)
        except (AppError, ValueError) as error:
            raise ToolError(
                f"This KPI can't be computed: {getattr(error, 'message', error)}"
            ) from error
        content = compact(result, max_items=24)
        return ToolOutput(
            f"{result.name}: {len(result.periods)} periods",
            content,
            AppLink("kpi", {}, "Open the KPI builder"),
        )


class SaveInput(DefinitionInput):
    name: str = Field(min_length=1, max_length=120, description="A short name for the KPI.")


class SaveKpi(Tool[SaveInput]):
    name: ClassVar[str] = "save_kpi"
    kind = "write"
    description: ClassVar[str] = (
        "Save a KPI definition under a name, so it appears in the KPI builder's saved list. "
        "The user confirms first. Compute it before saving."
    )
    Input: ClassVar[type[BaseModel]] = SaveInput

    def __init__(self, service: KpiService) -> None:
        self.service = service

    def label(self, args: SaveInput) -> str:
        return f"Saving the KPI “{args.name}”"

    def confirmation(self, args: SaveInput) -> str:
        return f"Save the KPI “{args.name}” in the KPI builder."

    def run(self, ctx: ToolContext, args: SaveInput) -> ToolOutput:
        saved = self.service.create(ctx.dataset_id, args.name, args.definition)
        return ToolOutput(
            f"Saved “{saved.name}”",
            {"kpi_id": saved.id, "name": saved.name},
            AppLink("kpi", {"kpi": saved.id}, "Open the saved KPI"),
        )


def agent_tools(app: FastAPI) -> list[Tool[Any]]:
    service: KpiService = app.state.kpi
    return [SuggestKpis(service), ComputeKpi(service), SaveKpi(service)]
