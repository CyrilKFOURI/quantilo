"""Knowledge base: semantic roles, null tokens, boolean vocabularies, known orders, stop words.

Extending detection = editing the TOML files next to this module (no code change).
"""

import re
import tomllib
import unicodedata
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, cast

from quantilo.typing.types import ANALYSIS_TYPES, AnalysisType

DIRECTORY = Path(__file__).parent


def name_tokens(name: str) -> tuple[str, ...]:
    """'customerAge' / 'Customer_AGE' / 'âge client' → ('customer', 'age') / ('age', 'client')."""
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
    ascii_name = unicodedata.normalize("NFKD", spaced).encode("ascii", "ignore").decode()
    return tuple(token for token in re.split(r"[^a-z0-9]+", ascii_name.lower()) if token)


@dataclass(frozen=True)
class Role:
    id: str
    label: str
    names: tuple[tuple[str, ...], ...]
    type: AnalysisType | None = None
    min: float | None = None
    max: float | None = None
    pattern: str | None = None
    # A key (identifier, customer_key, order_key) is detected like an identifier. With `suffixes`,
    # the role only names a column whose whole name is one of its words, alone or with a suffix.
    key: bool = False
    suffixes: tuple[str, ...] = ()

    def whole_name(self, tokens: tuple[str, ...]) -> bool:
        """'customer' · 'customer id' · 'id customer' · 'customerid' (048, R-ROLE-01)."""
        words = {name[0] for name in self.names if len(name) == 1}
        if tokens in self.names:
            return True
        if len(tokens) == 2:
            first, last = tokens
            return (first in words and last in self.suffixes) or (
                first in self.suffixes and last in words
            )
        if len(tokens) == 1:
            return any(
                tokens[0] == word + suffix or tokens[0] == suffix + word
                for word in words
                for suffix in self.suffixes
            )
        return False

    def accepts_range(self, low: float | None, high: float | None) -> bool:
        if low is None or high is None:
            return True
        too_low = self.min is not None and low < self.min
        too_high = self.max is not None and high > self.max
        return not (too_low or too_high)


@dataclass(frozen=True)
class Language:
    """A language known by its stop words (`stopwords.toml`)."""

    id: str
    label: str
    words: frozenset[str]


@dataclass(frozen=True)
class KnowledgeBase:
    roles: tuple[Role, ...]
    null_tokens: tuple[str, ...]
    boolean_pairs: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...]
    orders: tuple[tuple[str, ...], ...]
    languages: tuple[Language, ...] = ()
    street_words: frozenset[str] = frozenset()
    street_suffixes: tuple[str, ...] = ()

    def language(self, language_id: str | None) -> Language | None:
        return next((item for item in self.languages if item.id == language_id), None)

    def role(self, role_id: str | None) -> Role | None:
        return next((role for role in self.roles if role.id == role_id), None)

    def match_role(self, column_name: str) -> Role | None:
        """Best role whose name appears in the column name.

        Prefer the longest match, then the one ending on the last word: in "long_description"
        the head noun is "description", so the column is free text, not a longitude.
        """
        tokens = name_tokens(column_name)
        best: tuple[int, int, int] | None = None
        best_role: Role | None = None
        for index, role in enumerate(self.roles):
            if role.suffixes:
                # whole-name roles: the match spans the name ("customer_id" beats "id")
                if role.whole_name(tokens):
                    score = (len(tokens), 1, -index)
                    if best is None or score > best:
                        best, best_role = score, role
                continue
            for name in role.names:
                for start in range(len(tokens) - len(name) + 1):
                    if tokens[start : start + len(name)] != name:
                        continue
                    ends_last = int(start + len(name) == len(tokens))
                    score = (len(name), ends_last, -index)
                    if best is None or score > best:
                        best, best_role = score, role
        return best_role

    def boolean_pair_for(self, values: set[str]) -> tuple[list[str], list[str]] | None:
        """The vocabulary pair covering all values, with both sides present."""
        for true_words, false_words in self.boolean_pairs:
            trues, falses = set(true_words), set(false_words)
            if values <= trues | falses and values & trues and values & falses:
                return sorted(values & trues), sorted(values & falses)
        return None

    def order_for(self, values: set[str]) -> tuple[str, ...] | None:
        """The known order containing all values (at least 3 of them), smallest list first."""
        if len(values) < 3:
            return None
        matches = [order for order in self.orders if values <= set(order)]
        return min(matches, key=len) if matches else None


def _role(raw: dict[str, Any]) -> Role:
    forced = raw.get("type")
    if forced is not None and forced not in ANALYSIS_TYPES:
        raise ValueError(f"Unknown analysis type {forced!r} for role {raw['id']!r}")
    return Role(
        id=raw["id"],
        label=raw["label"],
        names=tuple(name_tokens(name) for name in raw.get("names", [])),
        type=cast(AnalysisType | None, forced),
        min=raw.get("min"),
        max=raw.get("max"),
        pattern=raw.get("pattern"),
        key=bool(raw.get("key", raw["id"] == "identifier")),
        suffixes=tuple(raw.get("suffixes", [])),
    )


@cache
def load() -> KnowledgeBase:
    roles = tomllib.loads((DIRECTORY / "roles.toml").read_text(encoding="utf-8"))
    vocabulary = tomllib.loads((DIRECTORY / "vocabulary.toml").read_text(encoding="utf-8"))
    stopwords = tomllib.loads((DIRECTORY / "stopwords.toml").read_text(encoding="utf-8"))
    return KnowledgeBase(
        roles=tuple(_role(role) for role in roles["role"]),
        null_tokens=tuple(vocabulary["null_tokens"]),
        boolean_pairs=tuple(
            (tuple(pair["true"]), tuple(pair["false"])) for pair in vocabulary["boolean_pair"]
        ),
        orders=tuple(tuple(order["values"]) for order in vocabulary["order"]),
        languages=tuple(
            Language(item["id"], item["label"], frozenset(word.lower() for word in item["words"]))
            for item in stopwords["language"]
        ),
        street_words=frozenset(word.lower() for word in vocabulary.get("street_words", [])),
        street_suffixes=tuple(vocabulary.get("street_suffixes", [])),
    )
