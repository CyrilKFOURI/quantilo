# ruff: noqa: RUF001, RUF002, RUF003  (the separator of derived names is "›", as for dates)
"""Derived columns (058): parts of a text column, read at query time like the parts of a date.

`review › Length`, `review › Has question`, `skills › has Python`, `review › Topic`: each one is an
SQL expression over the source column's cleaned text (`clean_text`), carried by a `derived`
parser, so every analysis reads it through `typed_sql` like a stored column. Nothing is stored.

A name is resolved from the columns alone (R-DER-03): a known source and a known part. The
catalog never lists them (R-DER-07): a service adds the derived columns a request names
(`with_derived`), and `derived_columns` lists what the pickers offer (R-DER-01, 02, 05, 06).
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from typing import Any

import duckdb
from pydantic import BaseModel

from quantilo.catalog import ColumnView
from quantilo.ingestion.sql import lit
from quantilo.typing.parsers import VALUE_PATTERNS, clean_text
from quantilo.typing.types import AnalysisType, DerivedSpec, Parser, TopicRule

SEPARATOR = " › "  # "review › Length", as "order_date › Month" (docs/DATES.md)
CHOICE_SEPARATOR = ";"  # R-TYP-09
MAX_CHOICE_COLUMNS = 10  # R-DER-05: "has <option>" columns offered at most...
MIN_CHOICE_SHARE = 0.01  # ...each picked in at least this share of the non-empty answers
HAS = "has "
NO_TOPIC = "No topic"
FREE_TEXT_ROLES = (None, "free_text")  # R-DER-02
CHECK = "Check that your data was read correctly."


@dataclass(frozen=True)
class Part:
    key: str
    label: str
    type: AnalysisType
    dtype: str  # what DuckDB returns
    rule: str
    meaning: str  # what the value is, for the reason shown next to the column
    free_text_only: bool = False


TEXT_PARTS: tuple[Part, ...] = (
    Part("length", "Length", "numeric_discrete", "BIGINT", "R-DER-01", "number of characters"),
    Part("words", "Words", "numeric_discrete", "BIGINT", "R-DER-01",
         "number of words (what spaces separate)"),
    Part("empty", "Empty", "boolean", "BOOLEAN", "R-DER-01",
         "True when the text is empty or a missing value"),
    Part("has_question", "Has question", "boolean", "BOOLEAN", "R-DER-02",
         "True when the text contains a question mark", True),
    Part("exclamations", "Exclamations", "numeric_discrete", "BIGINT", "R-DER-02",
         "number of exclamation marks", True),
    Part("uppercase_share", "Uppercase share", "numeric_continuous", "DOUBLE", "R-DER-02",
         "share of its letters in upper case (0 to 1)"),
    Part("contains_url", "Contains URL", "boolean", "BOOLEAN", "R-DER-02",
         "True when the text contains a web address", True),
    Part("contains_email", "Contains e-mail", "boolean", "BOOLEAN", "R-DER-02",
         "True when the text contains an e-mail address", True),
    Part("contains_number", "Contains number", "boolean", "BOOLEAN", "R-DER-02",
         "True when the text contains a digit"),
)  # fmt: skip
COUNT = Part("count", "Count", "numeric_discrete", "BIGINT", "R-DER-05", "number of options picked")
HAS_PART = Part("has", "has", "boolean", "BOOLEAN", "R-DER-05", "an option of the list")
TOPIC = Part(
    "topic", "Topic", "categorical_nominal", "VARCHAR", "R-TOP-05",
    "the topic whose words the text uses most, from the topics kept on its column",
)  # fmt: skip
PARTS = {part.key: part for part in (*TEXT_PARTS, COUNT, HAS_PART, TOPIC)}
BY_LABEL = {part.label.lower(): part for part in (*TEXT_PARTS, COUNT, TOPIC)}


# --- SQL ---------------------------------------------------------------------------------------


def _items(text: str) -> str:
    """The options of a list of choices, trimmed (R-TYP-09)."""
    return f"list_transform(string_split({text}, {lit(CHOICE_SEPARATOR)}), lambda q_i: trim(q_i))"


def topic_tokens_sql(text: str) -> str:
    """The words a topic reads (R-WRD-01 before stop words): letters of any script, 2 or more,
    lower case, accents removed, in order. `quantilo.text.tokens.words` reads the same."""
    return f"regexp_extract_all(strip_accents(lower({text})), '\\pL{{2,}}')"


def _number(value: float) -> str:
    return f"{value:.6e}"


def topic_sql(text: str, topics: Sequence[TopicRule]) -> str:
    """R-TOP-05: the topic whose weighted words the text uses most (each time it uses one);
    "No topic" without any of them, the first topic on a tie, empty for an empty text."""
    if not topics:
        return "CAST(NULL AS VARCHAR)"
    every = sorted({word for topic in topics for word in topic.words})
    known = ", ".join(lit(word) for word in every)
    # a word list and list_contains: 3x faster on 117,000 reviews than one regular expression
    # of every topic word (DuckDB 1.4), and whole words in any script by construction
    kept = f"list_filter({topic_tokens_sql(text)}, lambda q_w: list_contains([{known}], q_w))"
    scores = []
    for topic in topics:
        weights = " ".join(
            f"WHEN {lit(word)} THEN {_number(weight)}" for word, weight in topic.words.items()
        )
        scores.append(
            f"coalesce(list_sum(list_transform(q_k, lambda q_x: CASE q_x {weights} ELSE 0.0 END))"
            ", 0.0)"
        )
    per_topic = f"list_transform([{kept}], lambda q_k: [{', '.join(scores)}])[1]"
    best = (
        f"list_transform([{per_topic}], lambda q_s: "
        "CASE WHEN list_max(q_s) > 0 THEN list_position(q_s, list_max(q_s)) END)[1]"
    )
    labels = ", ".join(lit(topic.label) for topic in topics)
    return (
        f"(CASE WHEN {text} IS NULL THEN NULL "
        f"ELSE coalesce(list_extract([{labels}], {best}), {lit(NO_TOPIC)}) END)"
    )


def derived_sql(spec: DerivedSpec) -> str:
    """The derived column's value (R-DER-04)."""
    text = clean_text(spec.source)
    part = spec.part
    if part == "length":
        return f"length({text})"
    if part == "words":
        return f"len(string_split_regex({text}, '\\s+'))"
    if part == "empty":
        return f"({text} IS NULL)"
    if part == "has_question":
        return f"contains({text}, '?')"
    if part == "exclamations":
        return f"(length({text}) - length(replace({text}, '!', '')))"
    if part == "uppercase_share":
        letters = f"length(regexp_replace({text}, '[^\\pL]', '', 'g'))"
        upper = f"length(regexp_replace({text}, '[^\\p{{Lu}}]', '', 'g'))"
        return f"(CAST({upper} AS DOUBLE) / nullif({letters}, 0))"
    if part == "contains_url":
        return f"regexp_matches({text}, '(?i)(?:https?://|www\\.)\\S')"
    if part == "contains_email":
        email = VALUE_PATTERNS["email"].strip("^$")
        return f"regexp_matches({text}, {lit(email)})"
    if part == "contains_number":
        return f"regexp_matches({text}, '[0-9]')"
    if part == "count":
        return f"len(list_filter({_items(text)}, lambda q_i: q_i <> ''))"
    if part == "has":
        return f"list_contains({_items(text)}, {lit(spec.option or '')})"
    if part == "topic":
        return topic_sql(text, spec.topics)
    raise ValueError(f"Unknown derived part {part!r}")


# --- Resolution --------------------------------------------------------------------------------


def is_derived(column: Any) -> bool:
    parser = getattr(column, "parser", None)
    return isinstance(parser, Parser) and parser.kind == "derived"


def _source_ok(source: Any) -> bool:
    return source.type == "text" and not getattr(source, "excluded", False)


def _view(source: Any, spec: DerivedSpec, part: Part, name: str, reason: str) -> ColumnView:
    return ColumnView(
        name=name,
        position=source.position,
        storage_dtype=part.dtype,
        type=part.type,
        role=None,
        parser=Parser(kind="derived", derived=spec),
        order=None,
        detected_type=part.type,
        detected_role=None,
        confidence="high",
        reasons=[reason],
        overridden=False,
        excluded=False,
        summary=None,
    )


def _reason(source: str, meaning: str, code: str) -> str:
    return f"derived from {source!r}: {meaning} ({code})"


def kept_topics(source: Any) -> list[TopicRule]:
    """The topics kept on a text column (059, R-TOP-06), from its summary."""
    summary = getattr(source, "summary", None) or {}
    found = summary.get("topics") or {}
    return [TopicRule.model_validate(topic) for topic in found.get("rules", [])]


def _resolve_part(source: Any, label: str) -> ColumnView | None:
    name = f"{source.name}{SEPARATOR}{label}"
    if source.role == "choices":
        if label.startswith(HAS) and label[len(HAS) :].strip():
            option = label[len(HAS) :]
            spec = DerivedSpec(source=source.name, part="has", option=option)
            meaning = f"True when {option!r} is among the options picked"
            return _view(source, spec, HAS_PART, name, _reason(source.name, meaning, "R-DER-05"))
        if label.lower() == COUNT.label.lower():
            spec = DerivedSpec(source=source.name, part="count")
            return _view(source, spec, COUNT, name, _reason(source.name, COUNT.meaning, COUNT.rule))
        return None
    part = BY_LABEL.get(label.lower())
    if part is None or part is COUNT:
        return None
    if part is TOPIC:
        topics = kept_topics(source)
        if not topics:
            return None
        spec = DerivedSpec(source=source.name, part="topic", topics=topics)
        return _view(source, spec, TOPIC, name, _reason(source.name, TOPIC.meaning, TOPIC.rule))
    if part.free_text_only and source.role not in FREE_TEXT_ROLES:
        return None
    spec = DerivedSpec(source=source.name, part=part.key)
    return _view(source, spec, part, name, _reason(source.name, part.meaning, part.rule))


def resolve(columns: Mapping[str, Any], name: str) -> ColumnView | None:
    """The derived column `name` names, from the columns alone (R-DER-03); None if none."""
    start = name.find(SEPARATOR)
    while start > 0:
        source = columns.get(name[:start])
        if source is not None and _source_ok(source):
            found = _resolve_part(source, name[start + len(SEPARATOR) :])
            if found is not None:
                return found
        start = name.find(SEPARATOR, start + 1)
    return None


def with_derived[C](columns: Sequence[C], names: Iterable[str | None]) -> list[C]:
    """The columns, plus the derived ones `names` asks for (R-DER-07: only those)."""
    known: dict[str, Any] = {getattr(c, "name"): c for c in columns}  # noqa: B009
    extra: list[Any] = []
    for name in names:
        if not name or name in known or SEPARATOR not in name:
            continue
        found = resolve(known, name)
        if found is not None:
            known[name] = found
            extra.append(found)
    return [*columns, *extra]


def names_in(value: Any) -> list[str]:
    """Every string naming a derived column inside a request (models, dataclasses, lists)."""
    found: list[str] = []

    def walk(item: Any) -> None:
        if isinstance(item, str):
            if SEPARATOR in item:
                found.append(item)
        elif isinstance(item, BaseModel):
            walk(item.model_dump())
        elif is_dataclass(item) and not isinstance(item, type):
            for f in fields(item):
                walk(getattr(item, f.name))
        elif isinstance(item, Mapping):
            for key, inner in item.items():
                walk(key)
                walk(inner)
        elif isinstance(item, (list, tuple, set, frozenset)):
            for inner in item:
                walk(inner)

    walk(value)
    return found


# --- What the pickers offer --------------------------------------------------------------------


def choice_options(
    connection: duckdb.DuckDBPyConnection, source: str, column: str
) -> tuple[list[tuple[str, int]], int]:
    """Every option of a list of choices with the answers that pick it, most picked first, and
    the number of non-empty answers (read on the distinct answers, R-DER-05)."""
    text = clean_text(column)
    answers = (
        f"(SELECT {text} AS v, count(*) AS c FROM {source} WHERE {text} IS NOT NULL GROUP BY 1)"
    )
    rows = connection.execute(
        f"SELECT o, sum(c) AS n FROM (SELECT trim(unnest(string_split(v, "
        f"{lit(CHOICE_SEPARATOR)}))) AS o, c FROM {answers}) WHERE o <> '' "
        "GROUP BY o ORDER BY n DESC, o"
    ).fetchall()
    total = connection.execute(f"SELECT coalesce(sum(c), 0) FROM {answers}").fetchone()
    return [(str(o), int(n)) for o, n in rows], int(total[0] if total else 0)


def derived_columns(
    connection: duckdb.DuckDBPyConnection, source: str, columns: Sequence[Any]
) -> list[ColumnView]:
    """The derived columns offered for each text column (R-DER-01, 02, 05, R-TOP-06), each right
    after the previous ones of its source, in the order of the sources."""
    offered: list[ColumnView] = []
    for column in sorted(columns, key=lambda c: c.position):
        if not _source_ok(column) or is_derived(column):
            continue
        if column.role == "choices":
            offered.extend(_choice_columns(connection, source, column))
            continue
        for part in TEXT_PARTS:
            found = _resolve_part(column, part.label)
            if found is not None:
                offered.append(found)
        if kept_topics(column):
            topic = _resolve_part(column, TOPIC.label)
            if topic is not None:
                offered.append(topic)
    return offered


def _choice_columns(
    connection: duckdb.DuckDBPyConnection, source: str, column: Any
) -> list[ColumnView]:
    options, answers = choice_options(connection, source, column.name)
    least = MIN_CHOICE_SHARE * answers
    kept = [(o, n) for o, n in options if n >= least][:MAX_CHOICE_COLUMNS]
    left = len(options) - len(kept)
    count = _resolve_part(column, COUNT.label)
    assert count is not None
    if left:
        count.reasons.append(
            f"{left} rarer option{'s' if left > 1 else ''} not offered as columns (picked in "
            f"fewer than {MIN_CHOICE_SHARE:.0%} of the answers, or beyond the "
            f"{MAX_CHOICE_COLUMNS} most picked): filter them with 'contains' (R-DER-05)"
        )
    found = [count]
    for option, picked in kept:
        view = _resolve_part(column, f"{HAS}{option}")
        if view is None:
            continue
        share = picked / answers if answers else 0
        view.reasons = [
            _reason(
                column.name,
                f"True when {option!r} is among the options picked ({share:.0%} of the answers)",
                "R-DER-05",
            )
        ]
        found.append(view)
    return found
