"""A suggestion: what a rule found, why (with its numbers), and the step that fixes it."""

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from quantilo.cleaning.model import OP_IMPACT, Impact, Risk, RowCondition, Step
from quantilo.reasons import Reason

Mode = Literal["A", "S", "I"]  # automatic (ticked), suggested (unticked), information
Layer = Literal["basic", "advanced", "expert"]


class Example(BaseModel):
    before: str | None
    after: str | None


class MergeValue(BaseModel):
    """N38: one value of a merge group: as shown (its most frequent spelling up to case and
    accents), its rows, and the keys of the step's map it stands for (raw and normalized, C-23)."""

    value: str
    rows: int
    spellings: list[str] = Field(default_factory=list)


class MergeGroup(BaseModel):
    """N38: one group of a merge of spellings (A-01 to A-03), as OpenRefine's cluster dialog
    lists it: the value its members become (the most frequent, `rows` of its own) and them."""

    target: str
    rows: int
    values: list[MergeValue] = Field(default_factory=list)


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
    # M02: the rule's sentence as a template code and the vars it cites, for the interface to
    # write it in its language; taken from a `Reason` given as `reason`. They say the rule's own
    # sentence: the catalogue's note an automatic suggestion's reason ends with is told by
    # `measured`. None when the reason has no template (the interface shows `reason`).
    reason_code: str | None = None
    reason_vars: dict[str, str] = Field(default_factory=dict)
    columns: list[str] = Field(default_factory=list)
    cells: int = 0  # cells it changes
    rows: int = 0  # rows it removes (row rules)
    examples: list[Example] = Field(default_factory=list)
    step: Step | None = None  # None for an information
    # N02: the catalogue entry it falls under ("B-04:sure"), and whether that entry is reliable
    # enough to be applied without asking (measured right on 99 % or more, `cleaning.catalogue`)
    catalogue: str | None = None
    auto: bool = False
    # N04: for an `auto` suggestion, the cases its entry was judged right on (0: lossless by
    # construction, B-02), so the page can say why it was applied in the user's language
    measured: int | None = None
    # N38: a merge of spellings, group by group (empty for any other rule): the page lets the user
    # untick a value or a group and change a target; the step keeps what remains ticked
    clusters: list[MergeGroup] = Field(default_factory=list)
    # N32: the rows an information is about, as conditions on the dataset's columns (the page
    # opens them as a Data filter, where they can be pinned); empty when they cannot be written so
    where: list[RowCondition] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _effect(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        reason = data.get("reason")
        if isinstance(reason, Reason) and reason.code and data.get("reason_code") is None:
            data["reason_code"], data["reason_vars"] = reason.code, dict(reason.vars)
        if data.get("impact") is None:
            step = data.get("step")
            op = step.get("op") if isinstance(step, dict) else getattr(step, "op", None)
            data["impact"] = OP_IMPACT.get(op or "", "values")
        if data.get("risk") is None:
            data["risk"] = "safe" if data.get("mode") == "A" else "review"
        return data
