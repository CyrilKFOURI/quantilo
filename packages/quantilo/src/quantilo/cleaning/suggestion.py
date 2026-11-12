"""A suggestion: what a rule found, why (with its numbers), and the step that fixes it."""

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from quantilo.cleaning.model import OP_IMPACT, Impact, Risk, Step

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
    # 096: how it changes the dataset, and whether it needs a decision (`review`: never ticked by
    # default). Left out, they come from the step's op and the mode (A → safe).
    impact: Impact = "values"
    risk: Risk = "review"
    order: int = 0  # where its step goes among the others (C-06)
    title: str  # "Merge 3 spellings of 'New York'"
    reason: str  # why, with the numbers that trigger it (C-04)
    columns: list[str] = Field(default_factory=list)
    cells: int = 0  # cells it changes
    rows: int = 0  # rows it removes (row rules)
    examples: list[Example] = Field(default_factory=list)
    step: Step | None = None  # None for an information

    @model_validator(mode="before")
    @classmethod
    def _effect(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        if data.get("impact") is None:
            step = data.get("step")
            op = step.get("op") if isinstance(step, dict) else getattr(step, "op", None)
            data["impact"] = OP_IMPACT.get(op or "", "values")
        if data.get("risk") is None:
            data["risk"] = "safe" if data.get("mode") == "A" else "review"
        return data
