"""A-21 (audit 2026-11-12, N30): business synonyms in a column's values, from the coded lists of
`typing/knowledge_base/synonyms.toml` ('Not Rated' / 'Unrated', 'Homme' / 'Male', 'CB' / 'Carte
bancaire'). Advanced, mode S only: merged into the most frequent spelling if the user ticks it,
group by group (the review of N38), never on its own.

A domain counts in a column only when the column looks like it: two members of one group, and,
for the domains that say so, at least `min_context` of the domain's other values ('NR' beside
'PG-13' and 'R' is a rating). A one-letter member counts only beside its pair ('M' beside 'F').
Spellings that only differ by case or accents are B-03's; those A-01 to A-03 merged are left to
them.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from quantilo.cleaning import phonetic as ph
from quantilo.cleaning.rules import basic
from quantilo.typing.knowledge_base.synonyms import ValueDomain, load_synonyms

if TYPE_CHECKING:
    from quantilo.cleaning.rules.advanced import Advanced, Unit

RULE = "A-21"


def _present(domain: ValueDomain, keys: dict[str, int]) -> list[list[int]]:
    """The units of each group of `domain` the column holds (two or more), by group."""
    if domain.min_context:
        context = sum(1 for value in domain.context if ph.fold(value) in keys)
        if context < domain.min_context:
            return []
    found = []
    for group in domain.groups:
        members: list[int] = []
        for member in group:
            index = keys.get(ph.fold(member.value))
            if index is None or index in members:
                continue
            if member.beside and not any(ph.fold(other) in keys for other in member.beside):
                continue
            members.append(index)
        if len(members) >= 2:
            found.append(members)
    return found


def _how(domains: list[ValueDomain]) -> str:
    said = "; ".join(f"{domain.reason} ({domain.label})" for domain in domains)
    return f"{said}, from the coded list of business synonyms. Merged only if you tick it"


def business(advanced: Advanced) -> None:
    from quantilo.cleaning.rules import advanced as adv  # it imports this module

    domains = load_synonyms().values
    for column, stages in advanced.context.texts():
        if (
            not basic._is_category(column)
            or basic._converted(column)
            or basic._person_names(column)
        ):
            continue
        units: list[Unit] = adv.units_of(stages)
        if len(units) < 2:
            continue
        merged = advanced.merges.get(column.name, {})  # A-01 to A-03 already joined these
        keys = {unit.key: index for index, unit in enumerate(units) if unit.key not in merged}
        groups = adv.Groups(units)
        used: list[ValueDomain] = []
        for domain in domains:
            found = _present(domain, keys)
            for members in found:
                for index in members[1:]:
                    groups.union(members[0], index)
            if found:
                used.append(domain)
        roots: dict[int, list[int]] = defaultdict(list)
        for index in range(len(units)):
            roots[groups.find(index)].append(index)
        joined = {root: members for root, members in roots.items() if len(members) > 1}
        if joined:
            adv._merge_suggestion(advanced, RULE, column, groups, joined, _how(used), False)
