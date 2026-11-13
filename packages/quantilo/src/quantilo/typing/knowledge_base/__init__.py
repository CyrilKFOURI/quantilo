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

from quantilo.typing.knowledge_base.synonyms import load_synonyms
from quantilo.typing.types import ANALYSIS_TYPES, AnalysisType

DIRECTORY = Path(__file__).parent

# Roles saying a measure or a part of a date (092): their zero-padded numbers are no codes
# (R-TYP-03), and a code word beside them does not make a code ("YearsCode", R-TYP-04).
MEASURE_ROLES = ("month", "weekday", "hour", "year", "age", "duration", "latitude", "longitude",
                 "angle", "amount", "cost", "unit_price", "percentage", "quantity", "rating",
                 "propensity", "weight")  # fmt: skip
# "BIN #", "Job #" (exported as "bin__"), "N° dossier": a number sign says a number (R-TYP-04).
NUMBER_SIGN = re.compile(r"(#|n[°º]\.?|__)\s*$|^\s*(#|n[°º])", re.IGNORECASE)
# R-TYP-13 (fix3): a note in brackets closing the name ("Spending Score (1-100)", "Income (k$)")
# gives a range or a unit; the name ends before it.
NAME_NOTE = re.compile(r"\s*(\([^()]*\)|\[[^\[\]]*\])\s*$")
# R-TYP-14 (fix3): the amount word that says no money by itself ("total ads" counts ads).
GENERIC_AMOUNT_WORD = "total"


def name_tokens(name: str) -> tuple[str, ...]:
    """'customerAge' / 'Customer_AGE' / 'âge client' → ('customer', 'age') / ('age', 'client')."""
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
    ascii_name = unicodedata.normalize("NFKD", spaced).encode("ascii", "ignore").decode()
    return tuple(token for token in re.split(r"[^a-z0-9]+", ascii_name.lower()) if token)


def name_note(name: str) -> str | None:
    """R-TYP-13: the bracketed notes closing a name ('(1-100)'), None without any."""
    core = name
    while (shorter := NAME_NOTE.sub("", core)) != core:
        core = shorter
    note = name[len(core) :].strip()
    return note if note and name_tokens(core) else None


def name_length(tokens: tuple[str, ...], name: str) -> int:
    """How many of the name's words come before its closing bracketed notes (R-TYP-13)."""
    note = name_note(name)
    if note is None:
        return len(tokens)
    core = name_tokens(name.rstrip()[: -len(note)])
    return len(core) if tokens[: len(core)] == core else len(tokens)


def head_word(name: str, function_words: frozenset[str] | None = None) -> str | None:
    """The word a column name is about (092, R-TYP-04): the last one ("patient_nbr" -> "nbr"), or
    the first one when a Romance function word follows it ("Code du département" -> "code")."""
    tokens = name_tokens(name)
    if not tokens:
        return None
    words = load().function_words if function_words is None else function_words
    return tokens[0] if any(token in words for token in tokens[1:]) else tokens[-1]


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
    # 053: (tracked side, other side) of a status, and column names that say an outcome
    outcomes: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = ()
    outcome_names: frozenset[str] = frozenset()
    # 092: head words saying a code (R-TYP-04), and words saying a date (R-TYP-05)
    code_words: frozenset[str] = frozenset()
    short_code_words: frozenset[str] = frozenset()
    category_words: frozenset[str] = frozenset()
    category_stems: tuple[str, ...] = ()
    unit_words: frozenset[str] = frozenset()
    function_words: frozenset[str] = frozenset()
    date_words: frozenset[str] = frozenset()
    # fix1 (R-TYP-12): words saying a name holds one part of a date (year, month, day…)
    date_parts: tuple[tuple[str, frozenset[str]], ...] = ()
    # fix1 (B-21): codes and label words of rows that sum up other rows (World, Total…)
    aggregate_prefixes: tuple[str, ...] = ()
    aggregate_words: frozenset[str] = frozenset()
    aggregate_group_words: frozenset[str] = frozenset()
    # 060 (R-FUN-01): known funnels, each stage with its spellings, in order
    funnels: tuple[tuple[tuple[str, ...], ...], ...] = ()
    # N12 (B-13): country name (lowercased) → length of its postal codes, digits only
    postal_lengths: tuple[tuple[str, int], ...] = ()

    def language(self, language_id: str | None) -> Language | None:
        return next((item for item in self.languages if item.id == language_id), None)

    def role(self, role_id: str | None) -> Role | None:
        return next((role for role in self.roles if role.id == role_id), None)

    def match_role(self, column_name: str, notes: bool = True) -> Role | None:
        """Best role whose name appears in the column name.

        Prefer the longest match, then the one ending on the last word: in "long_description"
        the head noun is "description", so the column is free text, not a longitude. Then the one
        starting the name (092, R-TYP-08): "PercentSalaryHike" is a percentage, not a salary.
        The last word is the one before a closing note in brackets (R-TYP-13): "Spending Score
        (1-100)" is a score, not a spending; `notes=False` reads the whole name.
        """
        tokens = name_tokens(column_name)
        length = name_length(tokens, column_name) if notes else len(tokens)
        # N30 (R-SYN-01): a business synonym read as the word the roles know ('Qté', 'CA HT')
        tokens, _, longer = load_synonyms().expand(tokens)
        length += longer
        best: tuple[int, int, int, int] | None = None
        best_role: Role | None = None
        for index, role in enumerate(self.roles):
            if role.suffixes:
                # whole-name roles: the match spans the name ("customer_id" beats "id")
                if role.whole_name(tokens):
                    score = (len(tokens), 1, 1, -index)
                    if best is None or score > best:
                        best, best_role = score, role
                continue
            for name in role.names:
                for start in range(len(tokens) - len(name) + 1):
                    if tokens[start : start + len(name)] != name:
                        continue
                    ends_last = int(start + len(name) == length)
                    score = (len(name), ends_last, int(start == 0), -index)
                    if best is None or score > best:
                        best, best_role = score, role
        return best_role

    def only_total(self, name: str) -> bool:
        """R-TYP-14: 'total' is the name's only amount word, beside other words ("total ads",
        "Total Bags"): the name says a sum of something, not money."""
        tokens = name_tokens(name)
        tokens = tokens[: name_length(tokens, name)]
        amount = self.role("amount")
        if amount is None or GENERIC_AMOUNT_WORD not in tokens or len(tokens) < 2:
            return False
        found = {
            phrase
            for phrase in amount.names
            for start in range(len(tokens) - len(phrase) + 1)
            if tokens[start : start + len(phrase)] == phrase
        }
        return found == {(GENERIC_AMOUNT_WORD,)}

    def name_code(self, name: str) -> tuple[str, str] | None:
        """("code" | "category", word) when the name's head word says that whole numbers name
        or classify things (092, R-TYP-04): "patient_nbr", "Code du département", "MS SubClass",
        "council_district". Not when a measure role comes from another word ("YearsCode" counts
        years); "item_nbr", whose quantity word is the head word itself, is an item number.
        Used by the detection and by `quantilo.semantics` (S-01), so both agree."""
        if NUMBER_SIGN.search(name):
            return "code", "#"
        tokens = name_tokens(name)
        head = head_word(name, self.function_words)
        if head is None or set(tokens) & self.unit_words:
            return None  # a unit says a measure: "sqft_lot" is an area, not a lot number
        found: tuple[str, str] | None = None
        if head in self.code_words and (len(tokens) > 1 or head not in self.short_code_words):
            found = ("code", head)
        elif head in self.category_words or head.startswith(self.category_stems):
            found = ("category", head)
        if found is None:
            return None
        named = self.match_role(name)
        if named and named.id in MEASURE_ROLES and head not in {w[0] for w in named.names}:
            return None
        return found

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
        outcomes=tuple(
            (
                tuple(word.lower() for word in item["share"]),
                tuple(word.lower() for word in item["other"]),
            )
            for item in vocabulary.get("outcome", [])
        ),
        outcome_names=frozenset(word.lower() for word in vocabulary.get("outcome_names", [])),
        code_words=_words(vocabulary, "code_words"),
        short_code_words=_words(vocabulary, "short_code_words"),
        category_words=_words(vocabulary, "category_words"),
        category_stems=tuple(sorted(_words(vocabulary, "category_stems"))),
        unit_words=_words(vocabulary, "unit_words"),
        function_words=_words(vocabulary, "function_words"),
        date_words=_words(vocabulary, "date_words"),
        date_parts=tuple(
            (item["part"], frozenset(word.lower() for word in item["words"]))
            for item in vocabulary.get("date_part", [])
        ),
        aggregate_prefixes=tuple(
            word.lower() for word in vocabulary.get("aggregates", {}).get("code_prefixes", [])
        ),
        aggregate_words=frozenset(
            word.lower() for word in vocabulary.get("aggregates", {}).get("words", [])
        ),
        aggregate_group_words=frozenset(
            word.lower() for word in vocabulary.get("aggregates", {}).get("group_words", [])
        ),
        funnels=tuple(
            tuple(tuple(word.lower() for word in stage) for stage in item["stages"])
            for item in vocabulary.get("funnel", [])
        ),
        postal_lengths=tuple(
            (country.strip().lower(), int(item["length"]))
            for item in vocabulary.get("postal_length", [])
            for country in item["countries"]
        ),
    )


def _words(vocabulary: dict[str, Any], key: str) -> frozenset[str]:
    return frozenset(token for word in vocabulary.get(key, []) for token in name_tokens(word))
