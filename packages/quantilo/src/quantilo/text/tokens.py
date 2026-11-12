"""The words 059 reads in a text (R-WRD-01), and the stop words left out of them.

A word is a run of letters (any script) of at least 2 letters, in lower case with its accents
removed. `quantilo.typing.derived.topic_tokens_sql` reads the same words in SQL, so the Topic
rule (R-TOP-05) gives in DuckDB what it gives here.
"""

import re
import unicodedata
from collections.abc import Iterable
from typing import Any

from quantilo.typing.knowledge_base import load
from quantilo.typing.text import MIXED

WORD = re.compile(r"[^\W\d_]{2,}")


def plain(text: str) -> str:
    """Lower case, accents removed (é → e), as DuckDB's `strip_accents(lower(...))`."""
    decomposed = unicodedata.normalize("NFD", text.lower())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def words(text: str, stop: frozenset[str] = frozenset()) -> list[str]:
    """The words of a text, in order, stop words left out (R-WRD-01)."""
    return [word for word in WORD.findall(plain(text)) if word not in stop]


def distinct_words(text: str, stop: frozenset[str] = frozenset()) -> set[str]:
    """Each word once (R-WRD-02)."""
    return set(words(text, stop))


def stop_words(language: str | None) -> frozenset[str]:
    """The stop words of the column's language (`summary.text.language`, 050), of every known
    language when it has none or several; written plain, with and without an elision."""
    kb = load()
    chosen = [lang for lang in kb.languages if lang.id == language] if language != MIXED else []
    found: Iterable[str] = (word for lang in (chosen or kb.languages) for word in lang.words)
    bare: set[str] = set()
    for word in found:  # "didn't" also as "didn", the word before its apostrophe
        bare.add(plain(word))
        bare.add(plain(word.split("'")[0]))
    return frozenset(bare)


def language_of(column: Any) -> str | None:
    """The column's language from its summary (050), if any."""
    summary = getattr(column, "summary", None) or {}
    text = summary.get("text") or {}
    language = text.get("language")
    return str(language) if language else None
