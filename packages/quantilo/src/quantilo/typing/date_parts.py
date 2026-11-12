"""The parts of one date held in separate columns (fix1, rules R-TYP-12 and B-20).

`an, mois, jour`, `year, month, day`, `Yr Sold, Mo Sold`: a year and a month (and maybe a day,
an hour, a minute) whose names say the same thing once the part word is taken out. Read from the
names only (`vocabulary.toml`, `[[date_part]]`); Clean checks the values before proposing the
assembled date, and the column detection only points at it. Nothing is assembled silently.
"""

from collections.abc import Iterable
from dataclasses import dataclass

from quantilo.typing.knowledge_base import load, name_tokens

PARTS = ("year", "month", "day", "hour", "minute")


@dataclass(frozen=True)
class PartGroup:
    """One date in pieces: part → column name, and what the names share (`""` for `year, month,
    day`; `"sold"` for `Yr Sold, Mo Sold`)."""

    stem: str
    parts: dict[str, str]

    @property
    def columns(self) -> list[str]:
        return [self.parts[part] for part in PARTS if part in self.parts]

    @property
    def listed(self) -> str:
        """'an (year), mois (month) and jour (day)'."""
        items = [f"{self.parts[part]} ({part})" for part in PARTS if part in self.parts]
        return ", ".join(items[:-1]) + f" and {items[-1]}"

    @property
    def name(self) -> str:
        """The assembled column's name: 'date', or 'sold date'."""
        return f"{self.stem} date" if self.stem else "date"


def part_of(name: str) -> tuple[str, str] | None:
    """The date part a column name holds and the rest of the name, or None (no part word, or
    two: 'year_month' is a period, not a part)."""
    kb = load()
    words = name_tokens(name)
    found = [
        (index, part)
        for index, word in enumerate(words)
        for part, part_words in kb.date_parts
        if word in part_words
    ]
    if len(found) != 1:
        return None
    index, part = found[0]
    rest = [w for i, w in enumerate(words) if i != index and w not in kb.function_words]
    return part, " ".join(rest)


def groups(names: Iterable[str]) -> list[PartGroup]:
    """Every set of columns holding at least a year and a month of one date, by shared stem; the
    first column of a part wins, and a minute only counts with an hour."""
    by_stem: dict[str, dict[str, str]] = {}
    for name in names:
        hit = part_of(name)
        if hit is None:
            continue
        part, stem = hit
        by_stem.setdefault(stem, {}).setdefault(part, name)
    found = []
    for stem, parts in by_stem.items():
        if "year" not in parts or "month" not in parts:
            continue
        if "day" not in parts:  # a month without a day: no hour either
            parts = {p: c for p, c in parts.items() if p in ("year", "month")}
        if "hour" not in parts:
            parts.pop("minute", None)
        found.append(PartGroup(stem, parts))
    return found
