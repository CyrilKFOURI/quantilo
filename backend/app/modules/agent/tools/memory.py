"""The assistant's memory and method (spec 026, § 13.1): dataset notes and playbooks."""

from pathlib import Path
from typing import ClassVar, Literal, get_args

from pydantic import BaseModel, Field

from app.modules.agent.engine.tools import Tool, ToolContext, ToolError, ToolOutput
from app.modules.agent.repository import MAX_NOTE_CHARS, NoteError, NoteRepository

PLAYBOOKS = Path(__file__).resolve().parent.parent / "playbooks"
Topic = Literal["eda", "statistics", "ml", "panel", "kpi", "timeseries", "synthetic", "experiments"]
TOPICS: tuple[str, ...] = get_args(Topic)


class RememberInput(BaseModel):
    text: str = Field(
        description=(
            "One short, lasting fact about this dataset, in the user's language: what a column "
            "means, the target, a unit, a known data problem. Example: 'Fare is in 1912 pounds'."
        ),
        min_length=3,
        max_length=MAX_NOTE_CHARS,
    )


class Remember(Tool[RememberInput]):
    name: ClassVar[str] = "remember"
    kind = "write"
    description: ClassVar[str] = (
        "Save a note on this dataset that every future conversation starts with (R-AG-M1). The "
        "user confirms it first. Use it for facts the user tells you or confirms (the target, "
        "units, what a code means, a known problem), never for analysis results, which can be "
        "recomputed. Existing notes are in <dataset_notes>: don't repeat one."
    )
    Input: ClassVar[type[BaseModel]] = RememberInput

    def __init__(self, notes: NoteRepository) -> None:
        self.notes = notes

    def label(self, args: RememberInput) -> str:
        return "Saving a note on this dataset"

    def confirmation(self, args: RememberInput) -> str:
        return f"Remember for this dataset: “{args.text}”"

    def run(self, ctx: ToolContext, args: RememberInput) -> ToolOutput:
        try:
            note = self.notes.add(ctx.dataset_id, args.text, "agent")
        except NoteError as error:
            raise ToolError(str(error)) from error
        return ToolOutput(f"Noted: {note.text}", {"saved": note.text})


class PlaybookInput(BaseModel):
    topic: Topic = Field(description="The kind of work: " + ", ".join(TOPICS) + ".")


class GetPlaybook(Tool[PlaybookInput]):
    name: ClassVar[str] = "get_playbook"
    description: ClassVar[str] = (
        "The method to follow for a broad request, written by a statistician: eda (first look "
        "at a dataset), statistics (reading tests), ml (preparing and judging a model), panel "
        "(a model on entities seen period after period: walk-forward, per-period scores), kpi "
        "(indicators over time), timeseries (trend, seasonality, forecasts), synthetic (fake "
        "data that stays faithful and private), experiments (A/B tests and their sample size). "
        "Read it once at the start of such a request."
    )
    Input: ClassVar[type[BaseModel]] = PlaybookInput

    def label(self, args: PlaybookInput) -> str:
        return f"Reading the {args.topic} method"

    def run(self, ctx: ToolContext, args: PlaybookInput) -> ToolOutput:
        text = (PLAYBOOKS / f"{args.topic}.md").read_text(encoding="utf-8")
        return ToolOutput(f"Method: {args.topic}", text)
