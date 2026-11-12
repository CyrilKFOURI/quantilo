"""The kind of dataset and its KPI pack (061, P17): "Looks like sales data · 5 KPIs".

Written rules on the column roles and names (`kpi_templates.toml`), no model: R-ARC-01 picks the
kind whose needs are all met with the most clues, R-ARC-02 builds its pack from the suggestions
already made (so every KPI keeps its own reasons). Nothing is applied on its own: the page says
what it found and the user confirms before a KPI is saved.
"""

import tomllib
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Protocol

from quantilo.typing.knowledge_base import name_tokens

TEMPLATES = Path(__file__).with_name("kpi_templates.toml")
PACK_MAX = 5
PACK_MIN = 3


class Column(Protocol):
    name: str
    role: str | None


class Suggested(Protocol):
    rule: str
    name: str
    score: int


@dataclass(frozen=True)
class Template:
    id: str
    label: str
    needs: tuple[tuple[str, ...], ...]
    signals: tuple[str, ...]
    min_signals: int
    rules: tuple[str, ...]


@dataclass
class Archetype:
    id: str
    label: str
    message: str  # "Looks like sales data · 5 KPIs"
    kpis: list[str]  # names of the suggested KPIs in the pack, in order
    reasons: list[str] = field(default_factory=list)


@cache
def templates() -> tuple[Template, ...]:
    raw = tomllib.loads(TEMPLATES.read_text(encoding="utf-8"))
    return tuple(
        Template(
            item["id"],
            item["label"],
            tuple(tuple(group) for group in item["needs"]),
            tuple(item.get("signals", [])),
            int(item.get("min_signals", 1)),
            tuple(item["rules"]),
        )
        for item in raw["archetype"]
    )


def _meets(clue: str, column: Column) -> bool:
    kind, _, value = clue.partition(":")
    if kind == "role":
        return column.role == value
    return value in set(name_tokens(column.name))


def _said(clue: str, column: Column) -> str:
    kind, _, value = clue.partition(":")
    if kind == "role":
        return f"{column.name} ({value.replace('_', ' ')})"
    return f"{column.name} (name says {value})"


def _match(template: Template, columns: Sequence[Column]) -> tuple[list[str], list[str]] | None:
    """What meets each need, then the clues found; None when a need is not met."""
    needs = []
    for group in template.needs:
        hit = next(((c, column) for c in group for column in columns if _meets(c, column)), None)
        if hit is None:
            return None
        needs.append(_said(*hit))
    clues = []
    for clue in template.signals:
        column = next((column for column in columns if _meets(clue, column)), None)
        if column is not None:
            clues.append(_said(clue, column))
    if len(clues) < template.min_signals:
        return None
    return needs, list(dict.fromkeys(clues))


def classify(columns: Sequence[Column], suggested: Sequence[Suggested]) -> Archetype | None:
    """R-ARC-01 and R-ARC-02: the best-fitting kind, with a pack of 3 to 5 suggested KPIs."""
    best: tuple[int, Template, list[str], list[str]] | None = None
    for template in templates():
        found = _match(template, columns)
        if found is None:
            continue
        needs, clues = found
        if best is None or len(clues) > best[0]:
            best = (len(clues), template, needs, clues)
    if best is None:
        return None
    _, template, needs, clues = best
    rank = {rule: index for index, rule in enumerate(template.rules)}
    pack = sorted(
        (s for s in suggested if s.rule in rank),
        key=lambda s: (rank[s.rule], -s.score),
    )[:PACK_MAX]
    if len(pack) < PACK_MIN:
        return None
    extra = [c for c in clues if c not in needs][:4]
    reasons = [
        f"R-ARC-01 · Looks like {template.label}: {', '.join(needs)}"
        + (f"; also {', '.join(extra)}" if extra else ""),
        f"R-ARC-02 · {len(pack)} KPIs usual for {template.label}, from the suggestions above. "
        "A guess from the column names and roles: nothing is saved until you confirm",
    ]
    return Archetype(
        template.id,
        template.label,
        f"Looks like {template.label} · {len(pack)} KPIs",
        [s.name for s in pack],
        reasons,
    )
