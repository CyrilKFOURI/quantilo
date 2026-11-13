"""N31: declared checks, as dbt tests, Great Expectations or Soda checks: what must hold in a
dataset (`order_id` is unique, `city` is never empty, `status` is one of a list, `created_date` is
before `closed_date`, `city` determines `country`, a condition every row meets…), declared by the
user and run again on each import, each new version and each replay. A check never fixes anything:
it says how many rows fail, which ones, and why.

Each check is one coded test on each column's typed value (its parser, as the filters read it): a
boolean `failing` per row, counted on the whole file. Columns named in a check that the data does
not have make it an `error` (said, never a silent pass).
"""

import time
from dataclasses import dataclass, field
from typing import Annotated, Literal

import duckdb
from pydantic import BaseModel, ConfigDict, Field, model_validator

from quantilo.catalog import ColumnView
from quantilo.cleaning import model as m
from quantilo.cleaning.compile import Stage, as_text, check_pattern, is_number, is_temporal
from quantilo.cleaning.compile import _condition as row_condition  # pyright: ignore[reportPrivateUsage]
from quantilo.ingestion.sql import ident, lit
from quantilo.typing.parsers import typed_sql

Kind = Literal[
    "not_empty",
    "unique",
    "in_list",
    "between",
    "pattern",
    "compare",
    "sum",
    "determines",
    "condition",
]
KINDS: tuple[Kind, ...] = (
    "not_empty",
    "unique",
    "in_list",
    "between",
    "pattern",
    "compare",
    "sum",
    "determines",
    "condition",
)
Compare = Literal["lt", "le", "eq", "ne", "ge", "gt"]
SYMBOLS: dict[str, str] = {"lt": "<", "le": "<=", "eq": "=", "ne": "<>", "ge": ">=", "gt": ">"}
MAX_CHECKS = 100
SHOWN_ROWS = 50
EXAMPLES = 5
ROW = "__quantilo_row"
# columns a kind takes: (at least, at most)
ARITY: dict[str, tuple[int, int]] = {
    "not_empty": (1, 50),
    "unique": (1, 4),
    "in_list": (1, 1),
    "between": (1, 1),
    "pattern": (1, 1),
    "compare": (2, 2),
    "sum": (3, 21),
    "determines": (2, 2),
    "condition": (0, 0),
}


class CleanCheck(BaseModel):
    """One declared check. `columns` per kind: the column(s) tested; `compare` and `determines`:
    [A, B]; `sum`: [the total, its parts…]. `condition`: the rows must all meet `conditions`
    (`expect = "all"`) or none of them may (`"none"`), written with the editor of `filter_rows`."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^c[0-9]{1,4}$")
    kind: Kind
    name: str | None = Field(default=None, max_length=120)
    enabled: bool = True
    columns: list[m.Name] = Field(default_factory=list, max_length=50)
    values: list[m.Value] = Field(default_factory=list, max_length=m.MAX_VALUES)
    min: m.Value | None = None
    max: m.Value | None = None
    pattern: Annotated[str, Field(max_length=500)] | None = None
    op: Compare = "le"
    tolerance: float = Field(default=0.0, ge=0)
    conditions: list[m.RowCondition] = Field(default_factory=list, max_length=20)
    combine: Literal["all", "any"] = "all"
    expect: Literal["all", "none"] = "all"
    # where it came from: written by the user, made from a rule's suggestion, or from a step
    origin: Literal["user", "rule", "step"] = "user"
    rule: str | None = Field(default=None, max_length=20)

    @model_validator(mode="after")
    def _fields(self) -> "CleanCheck":
        least, most = ARITY[self.kind]
        if not least <= len(self.columns) <= most:
            wanted = f"{least}" if least == most else f"{least} to {most}"
            raise ValueError(f"A `{self.kind}` check takes {wanted} columns.")
        if self.kind == "in_list" and not self.values:
            raise ValueError("A list check needs its values.")
        if self.kind == "between" and self.min is None and self.max is None:
            raise ValueError("A range check needs a lower or an upper bound.")
        if self.kind == "pattern" and not self.pattern:
            raise ValueError("A pattern check needs its pattern.")
        if self.kind == "condition" and not self.conditions:
            raise ValueError("A condition check needs at least one condition.")
        return self


class Checks(BaseModel):
    model_config = ConfigDict(extra="forbid")

    checks: list[CleanCheck] = Field(default_factory=list, max_length=MAX_CHECKS)

    @model_validator(mode="after")
    def _ids(self) -> "Checks":
        ids = [c.id for c in self.checks]
        if len(set(ids)) != len(ids):
            raise ValueError("Two checks have the same id.")
        return self


def describe(check: CleanCheck) -> str:
    """The check in a sentence ("order_id is unique")."""
    if check.name:
        return check.name
    cols = check.columns
    if check.kind == "not_empty":
        return f"{', '.join(cols)} {'is' if len(cols) == 1 else 'are'} never empty"
    if check.kind == "unique":
        return f"{' + '.join(cols)} is unique"
    if check.kind == "in_list":
        shown = ", ".join(check.values[:5]) + ("…" if len(check.values) > 5 else "")
        return f"{cols[0]} is one of: {shown}"
    if check.kind == "between":
        if check.min is not None and check.max is not None:
            return f"{cols[0]} is between {check.min} and {check.max}"
        bound = f"at least {check.min}" if check.min is not None else f"at most {check.max}"
        return f"{cols[0]} is {bound}"
    if check.kind == "pattern":
        return f"{cols[0]} matches {check.pattern}"
    if check.kind == "compare":
        return f"{cols[0]} {SYMBOLS[check.op].replace('<=', '≤').replace('>=', '≥')} {cols[1]}"
    if check.kind == "sum":
        within = f" (within {check.tolerance:g})" if check.tolerance else ""
        return f"{cols[0]} = {' + '.join(cols[1:])}{within}"
    if check.kind == "determines":
        return f"{cols[0]} determines {cols[1]}"
    word = "Every row" if check.expect == "all" else "No row"
    joiner = " and " if check.combine == "all" else " or "
    parts = [
        f"{c.column} {c.op} {c.value if c.value is not None else ', '.join(c.values or [])}"
        for c in check.conditions
    ]
    return f"{word} meets: {joiner.join(parts)}"


# --- SQL -----------------------------------------------------------------------------------------


def typed_source(
    connection: duckdb.DuckDBPyConnection, parquet_path: str, views: list[ColumnView]
) -> tuple[str, Stage]:
    """The rows with each column read as its type (its parser, as the filters do) and the row's
    number in the file; and their columns with DuckDB's types."""
    raw = f"read_parquet({lit(parquet_path)}, file_row_number = true)"
    names = [
        str(row[0])
        for row in connection.execute(f"DESCRIBE SELECT * FROM {raw}").fetchall()
        if row[0] != "file_row_number"
    ]
    by_name = {v.name: v for v in views}
    parts = [
        f"{typed_sql(name, by_name[name].parser) if name in by_name else ident(name)} "
        f"AS {ident(name)}"
        for name in names
    ]
    sql = f"SELECT {', '.join(parts)}, file_row_number AS {ident(ROW)} FROM {raw}"
    rows = connection.execute(f"DESCRIBE {sql}").fetchall()
    return sql, Stage("t0", [(str(r[0]), str(r[1])) for r in rows if r[0] != ROW])


def _empty(column: str, dtype: str) -> str:
    return f"({ident(column)} IS NULL OR trim({as_text(column, dtype)}) = '')"


def _typed_literal(value: str, dtype: str) -> str:
    if is_number(dtype):
        return f"TRY_CAST({lit(value)} AS DOUBLE)"
    if is_temporal(dtype):
        return f"TRY_CAST({lit(value)} AS {dtype})"
    return lit(value)


def failing_sql(check: CleanCheck, stage: Stage, connection: duckdb.DuckDBPyConnection) -> str:
    """A boolean per row, true when the row fails the check (`stage`: the typed columns)."""
    for column in check.columns:
        if column not in stage.names:
            raise m.CleanError(f"'{column}' is not a column of this dataset.")
    for condition in check.conditions:
        if condition.column not in stage.names:
            raise m.CleanError(f"'{condition.column}' is not a column of this dataset.")
    cols = check.columns
    if check.kind == "not_empty":
        return " OR ".join(_empty(c, stage.dtype(c)) for c in cols)
    if check.kind == "unique":
        keys = ", ".join(ident(c) for c in cols)
        filled = " AND ".join(f"{ident(c)} IS NOT NULL" for c in cols)
        return f"({filled}) AND count(*) OVER (PARTITION BY {keys}) > 1"
    if check.kind == "in_list":
        column, dtype = cols[0], stage.dtype(cols[0])
        listed = ", ".join(lit(v) for v in check.values)
        return f"{ident(column)} IS NOT NULL AND {as_text(column, dtype)} NOT IN ({listed})"
    if check.kind == "between":
        column, dtype = cols[0], stage.dtype(cols[0])
        name = ident(column)
        target = name if is_number(dtype) or is_temporal(dtype) else as_text(column, dtype)
        out = []
        if check.min is not None:
            out.append(f"{target} < {_typed_literal(check.min, dtype)}")
        if check.max is not None:
            out.append(f"{target} > {_typed_literal(check.max, dtype)}")
        return f"{name} IS NOT NULL AND ({' OR '.join(out)})"
    if check.kind == "pattern":
        check_pattern(connection, check.pattern or "")
        column, dtype = cols[0], stage.dtype(cols[0])
        text = as_text(column, dtype)
        pattern = lit(check.pattern or "")
        return f"{ident(column)} IS NOT NULL AND NOT regexp_full_match({text}, {pattern})"
    if check.kind == "compare":
        a, b = (ident(c) for c in cols)
        kinds = {
            ("n" if is_number(t) else "d" if is_temporal(t) else "t")
            for t in (stage.dtype(cols[0]), stage.dtype(cols[1]))
        }
        if len(kinds) > 1:
            raise m.CleanError(
                f"'{cols[0]}' and '{cols[1]}' are not of the same type: read them as numbers or "
                "dates first."
            )
        return f"{a} IS NOT NULL AND {b} IS NOT NULL AND NOT ({a} {SYMBOLS[check.op]} {b})"
    if check.kind == "sum":
        for column in cols:
            if not is_number(stage.dtype(column)):
                raise m.CleanError(f"'{column}' is not read as numbers: a sum check needs numbers.")
        parts = " + ".join(f"coalesce({ident(c)}, 0)" for c in cols[1:])
        total = ident(cols[0])
        return (
            f"{total} IS NOT NULL AND abs({total} - ({parts})) > {float(check.tolerance)!r} + 1e-9"
        )
    if check.kind == "determines":
        a, b = cols
        text = as_text(b, stage.dtype(b))
        window = f"OVER (PARTITION BY {ident(a)})"
        filled = f"{ident(a)} IS NOT NULL AND {ident(b)} IS NOT NULL"
        return f"{filled} AND min({text}) {window} <> max({text}) {window}"
    joined = (" AND " if check.combine == "all" else " OR ").join(
        row_condition(c, stage) for c in check.conditions
    )
    test = f"coalesce(({joined}), false)"
    return f"NOT {test}" if check.expect == "all" else test


# --- Run -----------------------------------------------------------------------------------------


@dataclass
class CheckResult:
    id: str
    status: Literal["pass", "fail", "error"]  # error: it cannot run on these data, and says why
    failing: int = 0
    rows: int = 0
    message: str | None = None
    examples: list[int] = field(default_factory=list)  # the first failing rows (numbers from 0)
    description: str = ""


@dataclass
class CheckReport:
    rows: int
    results: list[CheckResult]
    passed: int = 0
    failed: int = 0
    errors: int = 0
    failing_rows: int = 0  # rows failing one check or more
    seconds: float = 0.0

    def summary(self) -> str:
        """'3 checks: 2 pass · 1 fails, 9 rows'."""
        parts = [f"{self.passed} pass"]
        if self.failed:
            parts.append(
                f"{self.failed} fail{'s' if self.failed == 1 else ''}: {self.failing_rows:,} rows"
            )
        if self.errors:
            parts.append(f"{self.errors} cannot run")
        return f"{len(self.results)} checks: {' · '.join(parts)}"


def _flags(
    check: CleanCheck, stage: Stage, source: str, connection: duckdb.DuckDBPyConnection
) -> str:
    test = failing_sql(check, stage, connection)
    return f"SELECT {ident(ROW)} AS r, coalesce(({test}), false) AS f FROM ({source})"


def _count(connection: duckdb.DuckDBPyConnection, sql: str) -> int:
    row = connection.execute(sql).fetchone()
    return int(row[0]) if row else 0


def run_checks(
    connection: duckdb.DuckDBPyConnection,
    parquet_path: str,
    views: list[ColumnView],
    checks: list[CleanCheck],
) -> CheckReport:
    """Every enabled check on the whole file: pass, fail (with its count and first rows), or
    error (a column it names is missing, a type that does not fit), never a silent pass."""
    started = time.monotonic()
    source, stage = typed_source(connection, parquet_path, views)
    total = _count(connection, f"SELECT count(*) FROM read_parquet({lit(parquet_path)})")
    results: list[CheckResult] = []
    failing: list[str] = []
    for check in checks:
        if not check.enabled:
            continue
        result = CheckResult(check.id, "pass", rows=total, description=describe(check))
        try:
            flags = _flags(check, stage, source, connection)
            count = _count(connection, f"SELECT count(*) FROM ({flags}) WHERE f")
            if count:
                result.status, result.failing = "fail", count
                result.examples = [
                    int(r[0])
                    for r in connection.execute(
                        f"SELECT r FROM ({flags}) WHERE f ORDER BY r LIMIT {EXAMPLES}"
                    ).fetchall()
                ]
                failing.append(flags)
        except (m.CleanError, duckdb.Error) as error:
            result.status = "error"
            result.message = (
                error.message
                if isinstance(error, m.CleanError)
                else str(error).strip().splitlines()[0][:300]
            )
        results.append(result)
    report = CheckReport(rows=total, results=results)
    report.passed = sum(1 for r in results if r.status == "pass")
    report.failed = sum(1 for r in results if r.status == "fail")
    report.errors = sum(1 for r in results if r.status == "error")
    if len(failing) == 1:
        report.failing_rows = next(r.failing for r in results if r.status == "fail")
    elif failing:
        union = " UNION ".join(f"SELECT r FROM ({flags}) WHERE f" for flags in failing)
        report.failing_rows = _count(connection, f"SELECT count(*) FROM ({union})")
    report.seconds = round(time.monotonic() - started, 2)
    return report


@dataclass
class FailingRow:
    row: int  # its number in the file (from 0)
    values: list[str | None]  # as text, in the order of `columns`


@dataclass
class FailingRows:
    """The rows failing a check (the first ones), with the check's columns first."""

    check: str
    failing: int
    columns: list[str]
    rows: list[FailingRow]


def failing_rows(
    connection: duckdb.DuckDBPyConnection,
    parquet_path: str,
    views: list[ColumnView],
    check: CleanCheck,
    limit: int = SHOWN_ROWS,
    others: int = 8,
) -> FailingRows:
    source, stage = typed_source(connection, parquet_path, views)
    flags = failing_sql(check, stage, connection)
    named = [*check.columns, *(c.column for c in check.conditions)]
    columns = list(dict.fromkeys(named))
    columns += [n for n in stage.names if n not in columns][:others]
    shown = ", ".join(f"{as_text(c, stage.dtype(c))} AS {ident(c)}" for c in columns)
    flagged = f"SELECT {ident(ROW)} AS r, coalesce(({flags}), false) AS f, {shown} FROM ({source})"
    query = f"SELECT * FROM ({flagged}) WHERE f"
    count = _count(connection, f"SELECT count(*) FROM ({query})")
    rows = connection.execute(
        f"SELECT * EXCLUDE (f) FROM ({query}) ORDER BY r LIMIT {int(limit)}"
    ).fetchall()
    return FailingRows(
        check=check.id,
        failing=count,
        columns=columns,
        rows=[FailingRow(int(r[0]), [None if v is None else str(v) for v in r[1:]]) for r in rows],
    )
