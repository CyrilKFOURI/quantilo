"""N30: business synonyms (`synonyms.toml`), for the roles (column names, R-SYN-01) and for Clean
(values, A-21). Extending them = editing the TOML file; every use gives its reason."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

FILE = Path(__file__).parent / "synonyms.toml"
CODE = "R-SYN-01"


def _tokens(text: str) -> tuple[str, ...]:
    from quantilo.typing.knowledge_base import name_tokens  # it imports this module

    return name_tokens(text)


@dataclass(frozen=True)
class NameSynonym:
    """`words` of a column name read as `means` before the role is matched."""

    words: tuple[tuple[str, ...], ...]
    means: tuple[str, ...]
    reason: str
    alone: bool = True  # False: only in a name of two words or more
    beside: frozenset[str] = frozenset()  # ... one of them one of these words, or a number

    def fits(self, tokens: tuple[str, ...], start: int, size: int) -> bool:
        if self.alone:
            return True
        others = tokens[:start] + tokens[start + size :]
        if not others:
            return False
        return not self.beside or any(word in self.beside or word.isdigit() for word in others)


@dataclass(frozen=True)
class Member:
    value: str  # as written in the file
    beside: tuple[str, ...] = ()  # counts only when the column holds one of these too


@dataclass(frozen=True)
class ValueDomain:
    id: str
    label: str
    reason: str
    groups: tuple[tuple[Member, ...], ...]
    context: tuple[str, ...] = ()
    min_context: int = 0


@dataclass(frozen=True)
class Synonyms:
    names: tuple[NameSynonym, ...]
    values: tuple[ValueDomain, ...]

    def expand(self, tokens: tuple[str, ...]) -> tuple[tuple[str, ...], NameSynonym | None, int]:
        """The name's words with the first synonym found replaced, the synonym, and how many words
        longer the name became (0 when nothing matched). The longest phrase wins."""
        best: tuple[int, int, NameSynonym, tuple[str, ...]] | None = None
        for synonym in self.names:
            for phrase in synonym.words:
                size = len(phrase)
                if not size:
                    continue
                for start in range(len(tokens) - size + 1):
                    if (
                        tokens[start : start + size] == phrase
                        and (best is None or size > best[1])
                        and synonym.fits(tokens, start, size)
                    ):
                        best = (start, size, synonym, phrase)
        if best is None:
            return tokens, None, 0
        start, size, synonym, _ = best
        expanded = tokens[:start] + synonym.means + tokens[start + size :]
        return expanded, synonym, len(synonym.means) - size

    def reason(self, name: str) -> str | None:
        """The reason a column name's synonym gives, None without one."""
        _, synonym, _ = self.expand(_tokens(name))
        return None if synonym is None else f"{synonym.reason} ({CODE})"


def _member(raw: Any) -> Member:
    if isinstance(raw, str):
        return Member(raw)
    return Member(str(raw["value"]), tuple(str(v) for v in raw.get("with", [])))


@cache
def load_synonyms() -> Synonyms:
    data = tomllib.loads(FILE.read_text(encoding="utf-8"))
    names = tuple(
        NameSynonym(
            words=tuple(_tokens(word) for word in item["words"]),
            means=_tokens(item["means"]),
            reason=item["reason"],
            alone=bool(item.get("alone", True)),
            beside=frozenset(token for word in item.get("beside", []) for token in _tokens(word)),
        )
        for item in data.get("name", [])
    )
    values = tuple(
        ValueDomain(
            id=item["id"],
            label=item["label"],
            reason=item["reason"],
            groups=tuple(tuple(_member(raw) for raw in group) for group in item["groups"]),
            context=tuple(item.get("context", [])),
            min_context=int(item.get("min_context", 0)),
        )
        for item in data.get("values", [])
    )
    return Synonyms(names, values)
