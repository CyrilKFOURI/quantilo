"""N05: "Don't suggest again". A dismissal hides the suggestions it matches: they are never ticked,
never applied for the user, and stay listed with the reason they are hidden (*Dismissed*), until
the user restores them. Its reach (`scope`):

- `suggestion`: this suggestion (this rule on this column), in this dataset;
- `rule`: the rule on every column of this dataset (`B-19` on Open Food Facts' 60 empty columns);
- `column`: the rule on the columns of this name, in every dataset;
- `everywhere`: the rule in every dataset (Settings, Cleaning: rules turned off).

A cleaning recipe carries the dismissals of its dataset (`suggestion`, `rule`): replayed on the
file of next month, the same suggestions stay hidden. Matching is a coded rule on the suggestion's
rule, id and columns, nothing else.
"""

from collections.abc import Iterable, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from quantilo.cleaning.suggestion import CleanSuggestion

Scope = Literal["suggestion", "rule", "column", "everywhere"]
SCOPES: tuple[Scope, ...] = ("suggestion", "rule", "column", "everywhere")
HERE: tuple[Scope, ...] = ("suggestion", "rule")  # kept by the dataset (and its recipe)
MAX_DISMISSED = 2_000


class Dismissal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scope: Scope
    rule: str = Field(min_length=1, max_length=20)  # "B-19"
    column: str | None = Field(default=None, max_length=500)
    suggestion: str | None = Field(default=None, max_length=600)  # "B-19:notes" (scope suggestion)
    title: str | None = Field(default=None, max_length=300)  # what it hid, as written then
    note: str | None = Field(default=None, max_length=300)  # the user's own reason, if any

    @model_validator(mode="after")
    def _reach(self) -> "Dismissal":
        if self.scope == "suggestion" and not self.suggestion:
            raise ValueError("Dismissing one suggestion needs its id (`suggestion`).")
        if self.scope == "column" and not self.column:
            raise ValueError("Dismissing a rule on a column needs the column's name.")
        if self.scope in ("rule", "everywhere"):
            self.column, self.suggestion = None, None
        if self.scope == "column":
            self.suggestion = None
        return self

    def key(self) -> tuple[str, str, str, str]:
        """Two dismissals with one key hide the same suggestions."""
        return (self.scope, self.rule, self.column or "", self.suggestion or "")


def matches(dismissal: Dismissal, suggestion: CleanSuggestion) -> bool:
    """Whether the dismissal hides the suggestion: same rule, and the same suggestion (scope
    `suggestion`) or a column of that name among those it touches (scope `column`)."""
    if dismissal.rule != suggestion.rule:
        return False
    if dismissal.scope == "suggestion":
        return dismissal.suggestion == suggestion.id
    if dismissal.scope == "column":
        return dismissal.column in suggestion.columns
    return True


def first_match(dismissals: Sequence[Dismissal], suggestion: CleanSuggestion) -> int | None:
    """The index of the first dismissal that hides the suggestion (None: shown)."""
    for index, dismissal in enumerate(dismissals):
        if matches(dismissal, suggestion):
            return index
    return None


def split(
    suggestions: Iterable[CleanSuggestion], dismissals: Sequence[Dismissal]
) -> tuple[list[CleanSuggestion], list[tuple[CleanSuggestion, int]]]:
    """The suggestions shown, and those hidden with the index of the dismissal that hides each."""
    shown: list[CleanSuggestion] = []
    hidden: list[tuple[CleanSuggestion, int]] = []
    for suggestion in suggestions:
        index = first_match(dismissals, suggestion) if dismissals else None
        if index is None:
            shown.append(suggestion)
        else:
            hidden.append((suggestion, index))
    return shown, hidden


def reason(dismissal: Dismissal) -> str:
    """Why a suggestion is hidden, in a sentence (the page writes its own, in its language)."""
    if dismissal.scope == "suggestion":
        text = f"You dismissed this suggestion ({dismissal.suggestion}) for this dataset."
    elif dismissal.scope == "rule":
        text = f"You turned off rule {dismissal.rule} for this dataset."
    elif dismissal.scope == "column":
        text = f"You turned off rule {dismissal.rule} for columns named '{dismissal.column}'."
    else:
        text = f"You turned off rule {dismissal.rule} in every dataset."
    return f"{text} {dismissal.note}" if dismissal.note else text
