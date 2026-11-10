"""Profiles of the columns that can link tables, and the candidate keys of a table (088, R-KEY).

One pass of aggregates per table (counts, approximate distinct counts, ranges, lengths), then an
exact distinct count only for the columns that look unique. Everything is compared on a canonical
form (`canonical`): an integer without decimals, a trimmed text without null tokens, a date
without time, so that a code read as text on one side and as a number on the other still meets
(R-REL-03)."""

import itertools
from dataclasses import dataclass, field
from typing import Literal

import duckdb

from quantilo.catalog import ColumnView
from quantilo.ingestion.sql import ident, lit
from quantilo.model import names
from quantilo.typing.parsers import clean_text, typed_sql

Kind = Literal["integer", "text", "date"]
Role = Literal["key", "measure", "attribute", "date", "text", "other"]

# R-KEY-02: roles that measure something; never a key, never a link.
MEASURE_ROLES = frozenset(
    {"amount", "quantity", "percentage", "cost", "unit_price", "rating", "duration"}
)
KEY_ROLES = frozenset({"identifier", "customer_key", "order_key"})
KEY_TYPES = frozenset(
    {"identifier", "categorical_nominal", "categorical_ordinal", "numeric_discrete", "text"}
)
NUMERIC = frozenset({"numeric_continuous", "numeric_discrete"})
MAX_KEY_LENGTH = 64  # R-KEY-02: a longer text is prose, not a code
PAIR_COLUMNS = 12  # R-KEY-03: pairs searched among the first 12 eligible columns (66 pairs)
_INTEGERS = ("TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT", "UTINYINT", "USMALLINT")
_INTEGERS += ("UINTEGER", "UBIGINT", "UHUGEINT")
_FLOATS = ("DOUBLE", "FLOAT", "REAL", "DECIMAL")


@dataclass
class Reason:
    """Why the app reads something this way: a rule code and a sentence."""

    code: str
    text: str


@dataclass
class TableInput:
    """A dataset as the model sees it."""

    id: str
    name: str
    storage_path: str
    columns: list[ColumnView]


@dataclass
class ColumnProfile:
    name: str
    type: str
    role: str | None
    kind: Kind | None  # how its values compare; None = never a key nor a link
    excluded_by: Reason | None  # why it can't link (R-KEY-02, R-REL-06)
    category: Role
    present: int = 0  # non-empty values
    distinct: int = 0
    unique: bool = False
    minimum: int | None = None
    maximum: int | None = None
    max_length: int | None = None
    position: int = 0

    @property
    def linkable(self) -> bool:
        return self.kind is not None


@dataclass
class CandidateKey:
    columns: list[str]
    primary: bool = False
    reasons: list[Reason] = field(default_factory=list)


@dataclass
class TableProfile:
    id: str
    name: str
    rows: int
    columns: list[ColumnProfile]
    keys: list[CandidateKey]

    def column(self, name: str) -> ColumnProfile:
        return next(column for column in self.columns if column.name == name)

    @property
    def primary_key(self) -> CandidateKey | None:
        return next((key for key in self.keys if key.primary), None)

    def count(self, category: Role) -> int:
        return sum(1 for column in self.columns if column.category == category)


def source(table: TableInput) -> str:
    return f"read_parquet({lit(table.storage_path)})"


def _family(dtype: str) -> str:
    upper = dtype.upper()
    if upper.startswith(_INTEGERS):
        return "integer"
    if upper.startswith(_FLOATS):
        return "float"
    if upper.startswith(("TIMESTAMP", "DATE")):
        return "date"
    if upper.startswith("BOOLEAN"):
        return "boolean"
    if upper.startswith("VARCHAR"):
        return "text"
    return "other"


def category_of(column: ColumnView, is_key: bool) -> Role:
    """What the column is to a reader of the model card (R-ROLE-03)."""
    if is_key or column.type == "identifier" or (column.role in KEY_ROLES):
        return "key"
    if column.type in NUMERIC and column.role not in KEY_ROLES:
        return "measure"
    if column.type == "datetime":
        return "date"
    if column.type == "text" and column.role == "free_text":
        return "text"
    if column.type in ("categorical_nominal", "categorical_ordinal", "boolean", "text"):
        return "attribute"
    return "other"


def _first_check(column: ColumnView) -> tuple[Kind | None, Reason | None]:
    """R-KEY-02 and R-REL-06 from the type and the role alone (no data read)."""
    if column.excluded or column.type == "ignored":
        return None, Reason("R-KEY-02", "left out of the analysis")
    if column.type == "boolean":
        return None, Reason("R-REL-06", "a yes/no flag links nothing")
    if column.role in MEASURE_ROLES:
        return None, Reason("R-KEY-02", f"a measure ({column.role.replace('_', ' ')})")
    if column.role == "free_text":
        return None, Reason("R-REL-06", "free text")
    if column.role == "choices":
        return None, Reason("R-REL-06", "a list of choices in one cell")
    family = _family(column.storage_dtype)
    parser = column.parser.kind
    if family == "integer" or parser == "year":
        return "integer", None
    if family == "float" or parser == "number":
        return "integer", None  # whole numbers only: checked on the values (`_measure`)
    if family == "date" or column.type == "datetime" or parser == "date":
        return "date", None
    if family == "text":
        if column.type == "numeric_continuous":
            return None, Reason("R-KEY-02", "a continuous number")
        if column.type in KEY_TYPES or column.type == "datetime":
            return "text", None
    return None, Reason("R-KEY-02", "not a stable value")


def canonical(column: ColumnView, kind: Kind) -> str:
    """The value as compared between tables (R-REL-03): NULL when empty."""
    family = _family(column.storage_dtype)
    if kind == "integer":
        if family == "integer":
            return f"CAST({ident(column.name)} AS VARCHAR)"
        typed = typed_sql(column.name, column.parser) if family == "text" else ident(column.name)
        number = f"TRY_CAST({typed} AS DOUBLE)"
        whole = f"CAST(CAST({number} AS BIGINT) AS VARCHAR)"
        return f"CASE WHEN {number} = floor({number}) THEN {whole} END"
    if kind == "date":
        typed = typed_sql(column.name, column.parser) if family == "text" else ident(column.name)
        if family == "text" and column.parser.kind != "date":
            return clean_text(column.name)
        return f"strftime(CAST({typed} AS DATE), '%Y-%m-%d')"
    return clean_text(column.name)


def key_expression(table: TableInput, columns: list[str], kinds: dict[str, Kind]) -> str:
    """One canonical value for one or two columns (NULL when any part is empty)."""
    views = {view.name: view for view in table.columns}
    parts = [canonical(views[name], kinds[name]) for name in columns]
    if len(parts) == 1:
        return parts[0]
    return " || chr(31) || ".join(f"({part})" for part in parts)


def profile_table(connection: duckdb.DuckDBPyConnection, table: TableInput) -> TableProfile:
    """Counts of every column that can link, then the candidate keys (R-KEY-01 to 04)."""
    relation = source(table)
    row = connection.execute(f"SELECT count(*) FROM {relation}").fetchone()
    rows = int(row[0]) if row else 0
    profiles: list[ColumnProfile] = []
    checks: list[tuple[ColumnProfile, ColumnView]] = []
    for view in sorted(table.columns, key=lambda item: item.position):
        kind, reason = _first_check(view)
        profile = ColumnProfile(
            name=view.name,
            type=view.type,
            role=view.role,
            kind=kind,
            excluded_by=reason,
            category=category_of(view, False),
            position=view.position,
        )
        profiles.append(profile)
        if kind is not None:
            checks.append((profile, view))
    if checks and rows:
        _measure(connection, relation, checks)
    single = [p for p in profiles if p.linkable and p.unique and rows > 0]
    keys = [CandidateKey([p.name]) for p in single]
    if not keys and rows:
        keys = _pairs(connection, table, relation, [p for p in profiles if p.linkable], rows)
    profile_out = TableProfile(table.id, table.name, rows, profiles, keys)
    _choose_primary(profile_out, table)
    primary = profile_out.primary_key
    for profile in profiles:
        if primary and profile.name in primary.columns:
            profile.category = "key"
    return profile_out


def _measure(
    connection: duckdb.DuckDBPyConnection,
    relation: str,
    checks: list[tuple[ColumnProfile, ColumnView]],
) -> None:
    parts: list[str] = []
    for index, (profile, view) in enumerate(checks):
        value = canonical(view, profile.kind or "text")
        parts += [
            f"count({value}) AS p{index}",
            f"approx_count_distinct({value}) AS a{index}",
            f"max(length({value})) AS l{index}",
        ]
        family = _family(view.storage_dtype)
        if profile.kind == "integer":
            parts += [
                f"min(TRY_CAST({value} AS BIGINT)) AS n{index}",
                f"max(TRY_CAST({value} AS BIGINT)) AS x{index}",
            ]
            if family != "integer":
                # whole numbers only: a value that is not one makes the canonical form NULL
                raw = typed_sql(view.name, view.parser) if family == "text" else ident(view.name)
                parts.append(f"count({raw}) AS r{index}")
        if profile.kind == "date" and view.storage_dtype.upper().startswith("TIMESTAMP"):
            time = f"CAST({ident(view.name)} AS TIME)"
            parts.append(f"count(*) FILTER (WHERE {time} <> TIME '00:00:00') AS t{index}")
    result = connection.execute(f"SELECT {', '.join(parts)} FROM {relation}").fetchone()
    names_out = [description[0] for description in connection.description or []]
    values = dict(zip(names_out, result or (), strict=False))
    exact: list[int] = []
    for index, (profile, _view) in enumerate(checks):
        profile.present = int(values.get(f"p{index}") or 0)
        profile.distinct = int(values.get(f"a{index}") or 0)
        profile.max_length = values.get(f"l{index}")
        profile.minimum = values.get(f"n{index}")
        profile.maximum = values.get(f"x{index}")
        if f"r{index}" in values and int(values[f"r{index}"] or 0) > profile.present:
            profile.kind, profile.excluded_by = None, Reason("R-KEY-02", "numbers with decimals")
            continue
        if int(values.get(f"t{index}") or 0) > 0:
            profile.kind, profile.excluded_by = None, Reason("R-KEY-02", "a time of day")
            continue
        if profile.kind == "text" and (profile.max_length or 0) > MAX_KEY_LENGTH:
            profile.kind = None
            profile.excluded_by = Reason(
                "R-KEY-02", f"text longer than {MAX_KEY_LENGTH} characters"
            )
            continue
        if profile.present == 0:
            profile.kind, profile.excluded_by = None, Reason("R-KEY-02", "empty")
            continue
        if profile.distinct >= profile.present * 0.9:
            exact.append(index)
    if not exact:
        return
    for index in exact:
        profile, view = checks[index]
        if profile.kind is None:
            continue
        value = canonical(view, profile.kind)
        found = connection.execute(
            f"SELECT count(DISTINCT {value}), count(*) FILTER (WHERE {value} IS NULL) "
            f"FROM {relation}"
        ).fetchone()
        distinct, empty = (int(found[0]), int(found[1])) if found else (0, 1)
        profile.distinct = distinct
        # R-KEY-01: unique and never empty
        profile.unique = empty == 0 and distinct == profile.present


def _pairs(
    connection: duckdb.DuckDBPyConnection,
    table: TableInput,
    relation: str,
    linkable: list[ColumnProfile],
    rows: int,
) -> list[CandidateKey]:
    """R-KEY-03: a key of two columns when no single column is one, among the columns whose name,
    type or role says key (an order line: `orderID` + `productID`)."""
    pool = [
        p
        for p in linkable
        if p.present == rows
        and p.distinct > 1
        and (names.has_key_word(p.name) or p.type == "identifier" or p.role in KEY_ROLES)
    ][:PAIR_COLUMNS]
    pairs = list(itertools.combinations(pool, 2))
    if not pairs:
        return []
    kinds: dict[str, Kind] = {p.name: p.kind for p in pool if p.kind}
    parts = [
        f"count(DISTINCT {key_expression(table, [a.name, b.name], kinds)}) AS c{index}"
        for index, (a, b) in enumerate(pairs)
    ]
    result = connection.execute(f"SELECT {', '.join(parts)} FROM {relation}").fetchone() or ()
    return [
        CandidateKey([a.name, b.name])
        for (a, b), distinct in zip(pairs, result, strict=False)
        if int(distinct or 0) == rows
    ]


def _choose_primary(profile: TableProfile, table: TableInput) -> None:
    """R-KEY-04: name says key or names the table, then identifier type or key role, then the
    leftmost column."""
    if not profile.keys:
        return

    def rank(key: CandidateKey) -> tuple[int, int, int, int]:
        columns = [profile.column(name) for name in key.columns]
        named = sum(
            1
            for column in columns
            if names.has_key_word(column.name) or names.names_table(column.name, table.name)
        )
        typed = sum(1 for c in columns if c.type == "identifier" or c.role in KEY_ROLES)
        return (-named, -typed, len(columns), min(c.position for c in columns))

    best = min(profile.keys, key=rank)
    best.primary = True
    columns = [profile.column(name) for name in best.columns]
    label = " + ".join(f"`{c.name}`" for c in columns)
    text = (
        f"{label}: unique and never empty on all {profile.rows:,} rows".replace(",", " ")
        if profile.rows
        else f"{label}: unique"
    )
    best.reasons.append(Reason("R-KEY-01", text))
    if len(columns) == 2:
        best.reasons.append(Reason("R-KEY-03", "no single column is unique: a key of two columns"))
    elif any(names.has_key_word(c.name) or names.names_table(c.name, table.name) for c in columns):
        best.reasons.append(Reason("R-KEY-04", "its name says it is the table's key"))
    elif any(c.type == "identifier" for c in columns):
        best.reasons.append(Reason("R-KEY-04", "detected as an identifier"))
    else:
        best.reasons.append(Reason("R-KEY-04", "the leftmost unique column"))


def kind_of(column: ColumnView) -> Kind | None:
    """How the column's values compare, from its type alone (`profile_table` also reads them)."""
    return _first_check(column)[0]


def link_kinds(
    child: TableInput, child_columns: list[str], parent: TableInput, parent_columns: list[str]
) -> tuple[dict[str, Kind], dict[str, Kind]] | Reason:
    """The kinds to compare a child's columns with a parent's key (R-REL-03), or why not."""
    if len(child_columns) != len(parent_columns) or not 1 <= len(child_columns) <= 2:
        return Reason("R-KEY-03", "a relation joins one or two columns on each side")
    child_views = {view.name: view for view in child.columns}
    parent_views = {view.name: view for view in parent.columns}
    child_kinds: dict[str, Kind] = {}
    parent_kinds: dict[str, Kind] = {}
    for child_name, parent_name in zip(child_columns, parent_columns, strict=True):
        if child_name not in child_views:
            return Reason("R-REL-01", f"`{child_name}` is not a column of {child.name}")
        if parent_name not in parent_views:
            return Reason("R-REL-01", f"`{parent_name}` is not a column of {parent.name}")
        child_view, parent_view = child_views[child_name], parent_views[parent_name]
        left, right = kind_of(child_view), kind_of(parent_view)
        if left is None or right is None:
            column = child_view if left is None else parent_view
            reason = _first_check(column)[1]
            text = reason.text if reason else "not a stable value"
            return Reason(
                reason.code if reason else "R-KEY-02", f"`{column.name}` can't link: {text}"
            )
        if not (left == right or {left, right} == {"integer", "text"}):
            return Reason(
                "R-REL-03", f"`{child_name}` ({left}) and `{parent_name}` ({right}) don't compare"
            )
        # a code read as text on one side and as a number on the other meets as a number
        shared: Kind = "integer" if left != right else left
        child_kinds[child_name] = shared
        parent_kinds[parent_name] = shared
    return child_kinds, parent_kinds
