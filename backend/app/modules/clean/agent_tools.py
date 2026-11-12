"""What the Clean module lets the AI agent do (065): read the suggestions, and save a cleaned
version (confirmed by the user; the original is never changed)."""

from typing import Any, ClassVar

from fastapi import FastAPI
from pydantic import BaseModel, Field
from quantilo.cleaning.suggest import default_steps, with_suggestion

from app.core.errors import AppError
from app.core.tools import AppLink, Tool, ToolContext, ToolError, ToolOutput, compact, wait_for
from app.modules.clean.service import CleanService

TIMEOUT = 600.0


class NoInput(BaseModel):
    pass


class CleaningSuggestions(Tool[NoInput]):
    name: ClassVar[str] = "cleaning_suggestions"
    kind = "read"
    description: ClassVar[str] = (
        "List what looks dirty in the dataset and the fix proposed for each: extra spaces, case "
        "variants, wrong encoding, 'N/A' words, numbers or dates stored as text, -999 codes, "
        "lost leading zeros, duplicate rows, empty columns… Each has its reason with numbers and "
        "a mode: A (lossless, ticked by default), S (suggested, check first). Whole dataset; "
        "filters do not apply."
    )
    Input: ClassVar[type[BaseModel]] = NoInput

    def __init__(self, service: CleanService) -> None:
        self.service = service

    def label(self, args: NoInput) -> str:
        return "Looking for data to clean"

    def run(self, ctx: ToolContext, args: NoInput) -> ToolOutput:
        try:
            found = self.service.suggestions(ctx.dataset_id).suggestions
        except AppError as error:
            raise ToolError(error.message) from error
        content = {
            "suggestions": [
                {
                    "id": s.id,
                    "mode": s.mode,
                    "impact": s.impact,  # 096: format, values, types, rows, columns
                    "risk": s.risk,  # review: needs the user's decision
                    "title": s.title,
                    "why": s.reason,
                    "cells": s.cells,
                    "rows": s.rows,
                }
                for s in found
            ]
        }
        automatic = sum(1 for s in found if s.mode == "A")
        review = sum(1 for s in found if s.risk == "review")
        return ToolOutput(
            f"{len(found)} cleaning suggestions ({automatic} automatic, {review} need a decision)",
            compact(content, max_items=25),
            AppLink("clean", {}, "Open in Clean"),
        )


class SaveInput(BaseModel):
    add: list[str] = Field(
        default_factory=list,
        description="Ids of suggested (S) suggestions to apply too; automatic ones always are.",
    )
    skip: list[str] = Field(
        default_factory=list, description="Ids of automatic suggestions to leave out."
    )


class SaveCleanedVersion(Tool[SaveInput]):
    name: ClassVar[str] = "save_cleaned_version"
    kind = "write"
    description: ClassVar[str] = (
        "Save a cleaned version of the dataset ('Cleaned vN'), a new dataset made by the "
        "automatic suggestions plus the ones named in `add`, minus `skip`. The original stays "
        "as it is. The user confirms first. Use cleaning_suggestions before."
    )
    Input: ClassVar[type[BaseModel]] = SaveInput

    def __init__(self, service: CleanService) -> None:
        self.service = service

    def label(self, args: SaveInput) -> str:
        return "Saving a cleaned version"

    def confirmation(self, args: SaveInput) -> str:
        extra = f", plus {len(args.add)} suggested" if args.add else ""
        return f"Create a cleaned version (automatic fixes{extra}); the original stays as it is."

    def run(self, ctx: ToolContext, args: SaveInput) -> ToolOutput:
        try:
            found = self.service.suggestions(ctx.dataset_id).suggestions
            steps = [s for s in default_steps(found) if s.suggestion not in args.skip]
            by_id = {s.id: s for s in found}
            for suggestion_id in args.add:
                if suggestion_id not in by_id:
                    raise ToolError(f"Unknown suggestion {suggestion_id!r}.")
                steps = with_suggestion(steps, by_id[suggestion_id], found)
            if not steps:
                raise ToolError("Nothing to clean with these choices.")
            job = self.service.save(ctx.dataset_id, steps)
        except AppError as error:
            raise ToolError(error.message) from error
        done = wait_for(
            lambda: self.service.job(job.id),
            lambda j: j.status != "running",
            ctx,
            describe=lambda j: (f"Stage: {j.stage}", j.progress),
            cancel=lambda: self.service.cancel(job.id),
            timeout=TIMEOUT,
        )
        if done.status != "done" or not done.created_dataset_id:
            raise ToolError(f"Cleaning {done.status}: {done.error or 'no detail'}")
        return ToolOutput(
            "Cleaned version created",
            {"dataset_id": done.created_dataset_id, "steps": len(steps)},
            AppLink("dataset", {"dataset": done.created_dataset_id}, "Open the cleaned version"),
        )


def agent_tools(app: FastAPI) -> list[Tool[Any]]:
    service: CleanService | None = app.state.clean
    if service is None:
        return []
    return [CleaningSuggestions(service), SaveCleanedVersion(service)]
