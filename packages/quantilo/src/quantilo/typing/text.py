"""Measures of a text column: words, sentence punctuation, stop words, language, addresses (050).

Pure Python on at most a few thousand distinct values, each weighted by its count: it runs
the same in the browser (Pyodide). The detection rules that read these measures live in
`engine.py`; the word lists in the knowledge base (`stopwords.toml`, `vocabulary.toml`).

A text's length in words counts what spaces separate ("CA-2017-100111" is one word, an id);
stop words read its letters, an elision keeping its apostrophe ("l'avis" → l', avis; a
lone "t" of "T-LIGHT" is no French "t'"); expressions and addresses read letters and digits.
"""

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field

from quantilo.typing.knowledge_base import KnowledgeBase

WORD = re.compile(r"[^\W_]+")  # letters or digits, any script
RIGHT_QUOTE = chr(0x2019)  # the typographic apostrophe
LETTERS = re.compile(r"[^\W\d_]+(?:['" + RIGHT_QUOTE + r"](?=[^\W\d_]))?")
# ". ! ? …" closing a sentence: followed by a space or the end ("2.5" and URLs do not count).
SENTENCE = re.compile(r"[.!?…](?:\s|$)")
NUMBER = re.compile(r"\d+[a-z]?|\d+(?:bis|ter)")
# 5-digit (FR, DE, ES, IT, US) or 4-digit (BE, CH, AT…) codes, or a UK postcode.
POSTAL = re.compile(r"\b\d{4,5}\b|\b[A-Z]{1,2}\d[A-Z\d]? ?\d[A-Z]{2}\b")

DOMINANT_LANGUAGE = 0.8  # below this share of the texts, the column's language is "Mixed"
MIN_LANGUAGE_SHARE = 0.1  # texts with a language needed to name one (and at least 5)
MIN_LANGUAGE_HITS = 2  # stop words a text needs to vote: "SET OF 6" or "DOVE" tell nothing
MIXED = "mixed"


@dataclass
class TextMeasures:
    values: int  # texts measured (weighted)
    median_words: float = 0
    p90_words: float = 0
    multiword_share: float = 0  # texts of at least 3 words
    sentence_share: float = 0  # texts with sentence punctuation
    stopword_share: float = 0  # stop words of the dominant language among all words
    language: str | None = None  # language id, MIXED, or None
    language_share: float | None = None  # share of the texts with a language that use it
    languages: list[tuple[str, float]] = field(default_factory=list)  # id, share of those texts
    address_share: float = 0  # texts with a number and a street word
    postal_share: float = 0  # texts with a postal code


def words(text: str) -> list[str]:
    return WORD.findall(text.lower())


def letter_tokens(text: str) -> list[str]:
    return [token.replace(RIGHT_QUOTE, "'") for token in LETTERS.findall(text.lower())]


def is_stop(token: str, words: frozenset[str]) -> bool:
    """A stop word, or one with an apostrophe after 2+ letters ("don't" → "don'" → "don")."""
    return token in words or (len(token) > 2 and token.endswith("'") and token[:-1] in words)


def weighted_quantile(counts: dict[int, int], q: float) -> float:
    total = sum(counts.values())
    if not total:
        return 0
    target = q * total
    seen = 0
    for value in sorted(counts):
        seen += counts[value]
        if seen >= target:
            return float(value)
    return float(max(counts))


def _is_street(token: str, kb: KnowledgeBase) -> bool:
    if token in kb.street_words:
        return True
    return any(len(token) > len(end) + 2 and token.endswith(end) for end in kb.street_suffixes)


def text_language(tokens: list[str], kb: KnowledgeBase) -> str | None:
    """The language whose stop words the text holds most; none on a tie or without any."""
    hits = [
        (sum(is_stop(token, language.words) for token in tokens), language.id)
        for language in kb.languages
    ]
    hits.sort(reverse=True)
    if not hits or hits[0][0] < MIN_LANGUAGE_HITS or (len(hits) > 1 and hits[1][0] == hits[0][0]):
        return None
    return hits[0][1]


def measure(values: Iterable[tuple[str, int]], kb: KnowledgeBase) -> TextMeasures:
    """Measures of `(text, count)` pairs (distinct texts with their number of rows)."""
    total = 0
    word_counts: Counter[int] = Counter()
    multiword = sentences = addresses = postal = 0
    all_tokens = 0
    stop_hits: Counter[str] = Counter()
    voted: Counter[str] = Counter()
    for text, count in values:
        if not text:
            continue
        total += count
        tokens = words(text)
        length = len(text.split())
        word_counts[length] += count
        multiword += count if length >= 3 else 0
        sentences += count if SENTENCE.search(text) else 0
        letters = letter_tokens(text)
        all_tokens += len(letters) * count
        for language in kb.languages:
            hits = sum(is_stop(token, language.words) for token in letters)
            stop_hits[language.id] += hits * count
        if language_id := text_language(letters, kb):
            voted[language_id] += count
        if any(NUMBER.fullmatch(token) for token in tokens) and any(
            _is_street(token, kb) for token in tokens
        ):
            addresses += count
        postal += count if POSTAL.search(text) else 0
    measures = TextMeasures(values=total)
    if not total:
        return measures
    measures.median_words = weighted_quantile(word_counts, 0.5)
    measures.p90_words = weighted_quantile(word_counts, 0.9)
    measures.multiword_share = multiword / total
    measures.sentence_share = sentences / total
    measures.address_share = addresses / total
    measures.postal_share = postal / total
    if all_tokens:
        measures.stopword_share = max(stop_hits.values(), default=0) / all_tokens
    with_language = sum(voted.values())
    if with_language >= max(5, MIN_LANGUAGE_SHARE * total):
        ranked = voted.most_common()
        measures.languages = [(language, count / with_language) for language, count in ranked]
        top, share = measures.languages[0]
        measures.language = top if share >= DOMINANT_LANGUAGE else MIXED
        measures.language_share = share
    return measures


def language_label(measures: TextMeasures, kb: KnowledgeBase) -> str | None:
    if measures.language is None:
        return None
    if measures.language == MIXED:
        return "Mixed"
    language = kb.language(measures.language)
    return language.label if language else measures.language


def language_reason(measures: TextMeasures, kb: KnowledgeBase) -> str | None:
    """'language: English (96% of texts)' or 'mixed languages: French 55%, English 41%'."""
    if measures.language is None:
        return None
    if measures.language != MIXED:
        return f"language: {language_label(measures, kb)} ({measures.language_share:.0%} of texts)"
    shares = ", ".join(
        f"{(kb.language(language_id) or kb.languages[0]).label} {share:.0%}"
        for language_id, share in measures.languages[:3]
    )
    return f"mixed languages: {shares}"


def stopwords_for(measures: TextMeasures | None, kb: KnowledgeBase) -> frozenset[str]:
    """Stop words left out at the ends of frequent expressions: the column's language(s), or all."""
    if measures and measures.languages:
        if measures.language != MIXED:
            ids = [measures.language]
        else:
            ids = [language_id for language_id, share in measures.languages if share >= 0.1]
        chosen = [language for language in kb.languages if language.id in ids]
        if chosen:
            return _bare(frozenset().union(*(language.words for language in chosen)))
    return _bare(frozenset().union(*(language.words for language in kb.languages)))


def _bare(words: frozenset[str]) -> frozenset[str]:
    """With elisions also without their apostrophe: expressions split "l'avis" into l, avis."""
    return words | {word.rstrip("'") for word in words}
