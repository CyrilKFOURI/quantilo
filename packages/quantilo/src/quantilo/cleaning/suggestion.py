"""A suggestion: what a rule found, why (with its numbers), and the step that fixes it."""

from typing import Literal

from pydantic import BaseModel, Field

from quantilo.cleaning.model import Step

Mode = Literal["A", "S", "I"]  # automatic (ticked), suggested (unticked), information
Layer = Literal["basic", "advanced", "expert"]


class Example(BaseModel):
    before: str | None
    after: str | None


class CleanSuggestion(BaseModel):
    id: str  # "B-03:city": the rule and what it is about
    rule: str
    layer: Layer = "basic"
    mode: Mode
    order: int = 0  # where its step goes among the others (C-06)
    title: str  # "Merge 3 spellings of 'New York'"
    reason: str  # why, with the numbers that trigger it (C-04)
    columns: list[str] = Field(default_factory=list)
    cells: int = 0  # cells it changes
    rows: int = 0  # rows it removes (row rules)
    examples: list[Example] = Field(default_factory=list)
    step: Step | None = None  # None for an information
