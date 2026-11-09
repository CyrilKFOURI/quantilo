"""Pair the columns a recipe expects with the columns of a new file.

In this order: the same name; the same name once case, spaces, accents, `_` and `-` are ignored
(matched, flagged); the closest name at most 2 edits away with the same detected type (proposed:
the user confirms it); otherwise missing. A column of the file is used once at most.
"""

import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from quantilo.recipes.model import RecipeColumn

MAX_DISTANCE = 2
Status = Literal["same", "matched", "proposed", "missing"]


@dataclass
class ColumnMatch:
    recipe: str  # the column the recipe expects
    type: str  # its detected type when the recipe was made
    column: str | None  # the file's column paired with it
    status: Status
    distance: int | None = None  # edits between the names (proposed)
    file_type: str | None = None  # the paired column's detected type


@dataclass
class Matching:
    columns: list[ColumnMatch]
    new_columns: list[str] = field(default_factory=list)  # in the file, unknown to the recipe

    def mapping(self, accept_proposals: bool = False) -> dict[str, str | None]:
        """Recipe column → file column (None: missing). Proposals count only once accepted."""
        return {
            match.recipe: match.column if match.status != "proposed" or accept_proposals else None
            for match in self.columns
        }

    @property
    def matched(self) -> int:
        """Columns paired on their name (same, or the same once normalized)."""
        return sum(1 for m in self.columns if m.status in ("same", "matched"))


def normalize(name: str) -> str:
    """Lower case, no accents, no spaces, `_` or `-`: "Montant_TTC" and "montant ttc" agree."""
    decomposed = unicodedata.normalize("NFKD", name)
    plain = "".join(c for c in decomposed if not unicodedata.combining(c))
    return "".join(c for c in plain.casefold() if not c.isspace() and c not in "_-")


def distance(a: str, b: str, limit: int = MAX_DISTANCE) -> int:
    """Levenshtein distance (insert, delete, substitute), stopped early above `limit`."""
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            cost = 0 if char_a == char_b else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        if min(current) > limit:
            return limit + 1
        previous = current
    return previous[-1]


def match_columns(expected: Sequence[RecipeColumn], actual: Sequence[tuple[str, str]]) -> Matching:
    """`actual`: the file's columns as (name, detected type), in file order."""
    types = dict(actual)
    free = [name for name, _ in actual]
    found: dict[str, ColumnMatch] = {}

    def take(column: RecipeColumn, name: str, status: Status, edits: int | None = None) -> None:
        found[column.name] = ColumnMatch(
            column.name, column.type, name, status, edits, types.get(name)
        )
        free.remove(name)

    for column in expected:
        if column.name in free:
            take(column, column.name, "same")
    for column in expected:
        if column.name in found:
            continue
        key = normalize(column.name)
        same = next((name for name in free if normalize(name) == key), None)
        if same is not None:
            take(column, same, "matched")
    for column in expected:
        if column.name in found:
            continue
        key = normalize(column.name)
        close = [
            (distance(key, normalize(name)), position, name)
            for position, name in enumerate(free)
            if types[name] == column.type
        ]
        close = [entry for entry in close if entry[0] <= MAX_DISTANCE]
        if close:
            edits, _, name = min(close)
            take(column, name, "proposed", edits)
    columns = [
        found.get(column.name) or ColumnMatch(column.name, column.type, None, "missing")
        for column in expected
    ]
    return Matching(columns, list(free))
