"""Rows that sum up other rows (fix1, rules B-21 and R-SUG-30): OWID's "World", "Europe" or
"High-income countries" beside the countries, a "Total" row under the departments.

Such a row counts the others a second time in any total. Three written signals find one, each
given as its reason, on the labels of a category column (the entity):

- its code starts with a prefix given to aggregates (`OWID_WRL`, `vocabulary.toml` `[aggregates]`);
- its label is made of words saying "every row" (`World`, `Grand Total`, `All countries`; not
  `All-Terrain Vehicle`, nor `C (all)`, a zoning code), or says a group of rows (`Europe`,
  `High-income countries`) while it has no code and nearly every other label has one;
- its value of a measure is the sum of the other labels' values, period by period.

Aggregates stay a minority of the labels. Nothing is removed here: Clean proposes to leave them
out (B-21, mode S) and the KPI page warns that its totals count them twice (R-SUG-30).
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import duckdb

from quantilo import semantics
from quantilo.ingestion.sql import ident, lit
from quantilo.typing.knowledge_base import load, name_tokens
from quantilo.typing.parsers import typed_sql

MIN_LABELS = 5  # labels left beside the aggregates
MAX_SHARE = 0.3  # aggregates are at most this share of the labels
MAX_LABELS = 5_000  # category columns with more labels are not read
MAX_COLUMNS = 20  # category columns read
CODED_SHARE = 0.7  # labels with a code: a missing code then says something
SUM_TOLERANCE = 0.02  # a value within 2 % of the sum of the others
SUM_SHARE = 0.8  # of the periods where the label has a value
MEASURES = 2  # measures tried for the sum
ENTITY_ROLES = frozenset(["country", "region", "city", "state", "continent"])
CATEGORY = "categorical_nominal"
NUMBERS = ("numeric_continuous", "numeric_discrete")


@dataclass
class Aggregate:
    label: str
    rows: int
    why: str  # "its iso_code is OWID_WRL"


@dataclass
class AggregateRows:
    column: str
    found: list[Aggregate]
    labels: int  # distinct labels of the column
    code: str | None = None  # the code column beside it, when there is one
    measure: str | None = None  # the measure of the sum check, when it ran
    notes: list[str] = field(default_factory=list)

    @property
    def names(self) -> list[str]:
        return [item.label for item in self.found]

    @property
    def rows(self) -> int:
        return sum(item.rows for item in self.found)

    def shown(self, count: int = 3) -> str:
        """ "'World', 'Europe', 'Asia' and 33 more"."""
        quoted = ", ".join(f"'{label}'" for label in self.names[:count])
        more = len(self.found) - count
        return quoted + (f" and {more} more" if more > 0 else "")


FILLERS = frozenset(["the", "of", "and", "et", "y", "und", "e", "grand", "sum"])


def every_row_words(label: str) -> set[str]:
    """The words saying "every row" of a label made only of such words, of group words and of
    fillers, outside brackets: 'World', 'Grand Total', 'All countries', 'Total (GCP)' say every
    row; 'C (all)', 'All-Terrain Vehicle' or 'No, not at all' do not."""
    kb = load()
    words = name_tokens(re.sub(r"\([^)]*\)|\[[^\]]*\]", " ", label))
    allowed = kb.aggregate_words | kb.aggregate_group_words | kb.function_words | FILLERS
    if not words or any(word not in allowed and not word.isdigit() for word in words):
        return set()
    return set(words) & kb.aggregate_words


def _label(view: Any) -> str:
    return f"NULLIF(trim(CAST({ident(view.name)} AS VARCHAR)), '')"


def _labels(connection: duckdb.DuckDBPyConnection, source: str, view: Any) -> list[tuple[str, int]]:
    label = _label(view)
    return [
        (str(value), int(count))
        for value, count in connection.execute(
            f"SELECT {label}, count(*) FROM {source} WHERE {label} IS NOT NULL "
            f"GROUP BY 1 ORDER BY 2 DESC, 1 LIMIT {MAX_LABELS + 1}"
        ).fetchall()
    ]


def _code_column(
    connection: duckdb.DuckDBPyConnection, source: str, entity: Any, views: Sequence[Any]
) -> tuple[Any, dict[str, str]] | None:
    """A column naming a code (iso_code, country_code) with at most one code per label."""
    for view in views:
        if view.name == entity.name or view.type in NUMBERS or view.type == "datetime":
            continue
        if semantics.code_by_name(view.name) is None and "iso" not in name_tokens(view.name):
            continue
        label, code = _label(entity), _label(view)
        rows = connection.execute(
            f"SELECT {label}, max({code}), count(DISTINCT {code}) FROM {source} "
            f"WHERE {label} IS NOT NULL GROUP BY 1"
        ).fetchall()
        if rows and all(int(k) <= 1 for _, _, k in rows):
            return view, {str(lab): str(c) for lab, c, _ in rows if c is not None}
    return None


def _measures(views: Sequence[Any], entity: Any) -> list[Any]:
    found = []
    for view in views:
        if view.type not in NUMBERS or view.name == entity.name:
            continue
        if semantics.code_by_name(view.name, view.role, view.type) is not None:
            continue
        if semantics.is_coordinate(view.name, view.role) or semantics.is_ratio(view.name):
            continue
        if view.role in ("year", "month", "weekday", "hour") or semantics.is_date_part(view.name):
            continue
        found.append(view)
    return found[:MEASURES]


def _sums(
    connection: duckdb.DuckDBPyConnection,
    source: str,
    entity: Any,
    measure: Any,
    period: Any | None,
    left_out: list[str],
) -> dict[str, tuple[int, int]]:
    """label → (periods with a value, periods where it is the sum of the other labels)."""
    label = _label(entity)
    value = f"TRY_CAST({typed_sql(measure.name, measure.parser)} AS DOUBLE)"
    when = f"CAST({typed_sql(period.name, period.parser)} AS VARCHAR)" if period else "'all'"
    kept = f"l NOT IN ({', '.join(lit(name) for name in left_out)})" if left_out else "true"
    rows = connection.execute(
        f"WITH g AS (SELECT {when} AS p, {label} AS l, sum({value}) AS v FROM {source} "
        f"WHERE {label} IS NOT NULL AND isfinite({value}) GROUP BY 1, 2), "
        f"k AS (SELECT * FROM g WHERE {kept} AND p IS NOT NULL), "
        "t AS (SELECT p, sum(v) AS total, count(*) AS n FROM k GROUP BY p) "
        "SELECT l, count(*), count(*) FILTER (WHERE t.n > ? AND v <> 0 "
        "AND abs(v - (total - v)) <= ? * abs(total - v)) FROM k JOIN t USING (p) GROUP BY l",
        [MIN_LABELS, SUM_TOLERANCE],
    ).fetchall()
    return {str(lab): (int(n), int(equal)) for lab, n, equal in rows}


def find(
    connection: duckdb.DuckDBPyConnection,
    source: str,
    views: Sequence[Any],
    labels: dict[str, list[tuple[str, int]]] | None = None,
) -> list[AggregateRows]:
    """The category columns holding rows that sum up other rows. `views`: typed columns (name,
    type, role, parser); `labels`: their values and rows when already read (Clean)."""
    kb = load()
    usable = [v for v in views if not getattr(v, "excluded", False)]
    entities = sorted(
        (v for v in usable if v.type == CATEGORY),
        key=lambda v: (v.role not in ENTITY_ROLES, v.position if hasattr(v, "position") else 0),
    )[:MAX_COLUMNS]
    period = next((v for v in usable if v.type == "datetime"), None)
    found: list[AggregateRows] = []
    for entity in entities:
        values = (labels or {}).get(entity.name)
        if values is None:
            values = _labels(connection, source, entity)
        values = [(lab, k) for lab, k in values if lab.strip()]  # as stored: a filter needs them
        if not MIN_LABELS < len(values) <= MAX_LABELS:
            continue
        rows_of = dict(values)
        hinted = kb.aggregate_words | kb.aggregate_group_words
        needs_codes = entity.role in ENTITY_ROLES or any(
            set(name_tokens(lab)) & hinted for lab, _ in values
        )  # a code column is looked for only when a label or the role calls for it
        coded = _code_column(connection, source, entity, usable) if needs_codes else None
        code_view, codes = coded if coded else (None, {})
        has_codes = bool(codes) and len(codes) >= CODED_SHARE * len(values)
        why: dict[str, str] = {}
        for lab, _ in values:
            words = set(name_tokens(lab))
            ends = every_row_words(lab)
            code = codes.get(lab.strip(), "")
            if code and code.lower().startswith(kb.aggregate_prefixes):
                why[lab] = f"its {code_view.name} is {code}"  # type: ignore[union-attr]
            elif ends and not (has_codes and code):
                why[lab] = f"its name says every row ('{sorted(ends)[0]}')"
            elif words & kb.aggregate_group_words and has_codes and not code:
                group = sorted(words & kb.aggregate_group_words)[0]
                why[lab] = (
                    f"its name says a group of rows ('{group}') and it has no "
                    f"{code_view.name}"  # type: ignore[union-attr]
                )
        measure_used = None
        if why or entity.role in ENTITY_ROLES:
            for measure in _measures(usable, entity):
                sums = _sums(connection, source, entity, measure, period, [x.strip() for x in why])
                raw = {lab.strip(): lab for lab, _ in values}
                added = {
                    raw[lab]: f"its {measure.name} is the sum of the other rows' "
                    f"({equal} of {n} {'periods' if period else 'checks'})"
                    for lab, (n, equal) in sums.items()
                    if lab in raw and equal >= 1 and equal >= SUM_SHARE * n
                }
                if added:
                    why.update(added)
                    measure_used = measure.name
                    break
        if not why:
            continue
        left = len(values) - len(why)
        if len(why) > MAX_SHARE * len(values) or left < MIN_LABELS:
            continue
        ordered = [lab for lab, _ in sorted(values, key=lambda item: item[0])]
        found.append(
            AggregateRows(
                entity.name,
                [Aggregate(lab, rows_of[lab], why[lab]) for lab in ordered if lab in why],
                len(values),
                code_view.name if code_view else None,
                measure_used,
            )
        )
    return found
