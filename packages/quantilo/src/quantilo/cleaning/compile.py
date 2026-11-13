"""Steps → a chain of DuckDB CTEs on the parent's Parquet (the parent is only read).

`t0` reads the Parquet with its row number (`__quantilo_row`, the order kept to the end, C-08);
each enabled step is one CTE selecting from the previous one: an explicit column list (a value
replaced, renamed, dropped, added), a `WHERE`, or a `QUALIFY`. Each stage's columns and types come
from DuckDB itself (`DESCRIBE`), so a step sees the types the previous steps left.

A step that cannot run (a column that no longer exists…) is reported in `problems` and skipped,
so the draft stays editable; `apply` refuses a cleaning with problems (`strict`).
"""

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import duckdb

from quantilo.cleaning import formula as fx
from quantilo.cleaning import model as m
from quantilo.cleaning.text import SQL_INVISIBLE, SQL_SPACE_RUN, normalize_spaces_sql
from quantilo.ingestion.sql import ident, lit
from quantilo.typing.conversion import default_boolean_parser
from quantilo.typing.knowledge_base import load as load_knowledge
from quantilo.typing.parsers import ALL_DATE_FORMATS, clean_text, number_sql, strptime_sql

ROW = "__quantilo_row"
INTEGERS = ("TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT", "UTINYINT", "USMALLINT")
INTEGERS += ("UINTEGER", "UBIGINT", "UHUGEINT")
FLOATS = ("FLOAT", "DOUBLE", "REAL")
# steps whose formulas or conditions can fail on a value (a text read as a number): tried on the
# first rows when compiled, so the step is reported instead of breaking the preview
FORMULA_OPS = frozenset({"set_where", "conditional_column", "computed_column", "swap_values"})
PROBE_ROWS = 2_000


def is_text(dtype: str) -> bool:
    return dtype == "VARCHAR"


def is_integer(dtype: str) -> bool:
    return dtype in INTEGERS


def is_number(dtype: str) -> bool:
    return dtype in INTEGERS or dtype in FLOATS or dtype.startswith("DECIMAL")


def is_temporal(dtype: str) -> bool:
    return dtype == "DATE" or dtype.startswith("TIMESTAMP")


def as_text(column: str, dtype: str) -> str:
    """The value as the user reads it: an integral float without its '.0' (17850.0 → '17850')."""
    name = ident(column)
    if is_text(dtype):
        return name
    if dtype in FLOATS or dtype.startswith("DECIMAL"):
        return (
            f"CASE WHEN {name} = trunc({name}) AND abs({name}) < 1e15 "
            f"THEN CAST(CAST({name} AS BIGINT) AS VARCHAR) ELSE CAST({name} AS VARCHAR) END"
        )
    return f"CAST({name} AS VARCHAR)"


def literal_list(values: list[str]) -> str:
    return "(" + ", ".join(lit(value) for value in values) + ")"


@dataclass
class Stage:
    """The table after a step: its CTE, columns (name, DuckDB type) and where each column came
    from in the previous stage (`None`: added by the step)."""

    name: str
    columns: list[tuple[str, str]]
    step: Any = None
    origin: dict[str, str | None] = field(default_factory=dict)
    removes_rows: bool = False
    reshapes: bool = False  # C-59: its rows no longer come one by one from the previous stage

    @property
    def names(self) -> list[str]:
        return [name for name, _ in self.columns]

    def dtype(self, column: str) -> str:
        for name, dtype in self.columns:
            if name == column:
                return dtype
        raise m.CleanError(f"Column '{column}' does not exist at this step.")


@dataclass
class Compiled:
    source: str
    ctes: list[str]
    stages: list[Stage]  # stages[0] = the parent; then one per step that runs
    problems: dict[str, str] = field(default_factory=dict)  # step id → why it was skipped

    def sql(self, index: int = -1, columns: str = "*") -> str:
        stage = self.stages[index]
        count = self.stages.index(stage) + 1
        return f"WITH {', '.join(self.ctes[:count])} SELECT {columns} FROM {stage.name}"

    def stage_of(self, step_id: str) -> int | None:
        for index, stage in enumerate(self.stages):
            if stage.step is not None and stage.step.id == step_id:
                return index
        return None

    @property
    def final(self) -> Stage:
        return self.stages[-1]

    def reshaped(self, start: int = 0, end: int = -1) -> Stage | None:
        """C-59: the first stage after `start`, up to `end`, that reshapes the table."""
        end = end % len(self.stages)
        return next((s for s in self.stages[start + 1 : end + 1] if s.reshapes), None)

    def output_columns(self) -> str:
        return ", ".join(ident(name) for name in self.final.names)


# --- Step bodies -----------------------------------------------------------------------------


@dataclass
class Body:
    sql: str
    origin: dict[str, str | None]
    removes_rows: bool = False
    reshapes: bool = False


@dataclass
class Context:
    """What a step that reads data at compile time needs (C1): the connection, the CTEs so far,
    and the Parquet of the other datasets a step names (`append_rows`, `merge_dataset`)."""

    connection: duckdb.DuckDBPyConnection
    ctes: list[str]
    sources: Mapping[str, str]

    def rows(self, sql: str) -> list[tuple[Any, ...]]:
        return self.connection.execute(f"WITH {', '.join(self.ctes)} {sql}").fetchall()


def _select(
    prev: Stage,
    replace: dict[str, str] | None = None,
    rename: dict[str, str] | None = None,
    drop: set[str] | None = None,
    after: dict[str, list[tuple[str, str]]] | None = None,
) -> Body:
    replace, rename, drop, after = replace or {}, rename or {}, drop or set(), after or {}
    parts: list[str] = []
    origin: dict[str, str | None] = {}
    for name in prev.names:
        if name in drop:
            continue
        out = rename.get(name, name)
        parts.append(f"{replace.get(name, ident(name))} AS {ident(out)}")
        origin[out] = name
        for new_name, expr in after.get(name, []):
            parts.append(f"{expr} AS {ident(new_name)}")
            origin[new_name] = None
    if len(set(origin)) != len(origin) or len(origin) != len(parts):
        raise m.CleanError("Two columns would have the same name.")
    parts.append(ident(ROW))
    return Body(f"SELECT {', '.join(parts)} FROM {prev.name}", origin)


def _where(prev: Stage, condition: str) -> Body:
    origin: dict[str, str | None] = {name: name for name in prev.names}
    return Body(f"SELECT * FROM {prev.name} WHERE {condition}", origin, removes_rows=True)


def _require(prev: Stage, *columns: str) -> None:
    for column in columns:
        prev.dtype(column)


def _texts(prev: Stage, columns: list[str]) -> list[str]:
    _require(prev, *columns)
    return [c for c in columns if is_text(prev.dtype(c))]


def _normalize_spaces(step: m.NormalizeSpaces, prev: Stage) -> Body:
    return _select(
        prev,
        {c: normalize_spaces_sql(ident(c), step.keep_newlines) for c in _texts(prev, step.columns)},
    )


def _remove_invisible(step: m.RemoveInvisible, prev: Stage) -> Body:
    pattern = lit(SQL_INVISIBLE)
    return _select(
        prev,
        {c: f"regexp_replace({ident(c)}, {pattern}, '', 'g')" for c in _texts(prev, step.columns)},
    )


def _change_case(step: m.ChangeCase, prev: Stage) -> Body:
    def expr(column: str) -> str:
        name = ident(column)
        if step.case == "lower":
            return f"lower({name})"
        if step.case == "upper":
            return f"upper({name})"
        words = f"string_split({name}, ' ')"
        title = f"list_transform({words}, lambda w: upper(w[1]) || lower(w[2:]))"
        return f"array_to_string({title}, ' ')"

    return _select(prev, {c: expr(c) for c in _texts(prev, step.columns)})


def _replace_values(step: m.ReplaceValues, prev: Stage) -> Body:
    _require(prev, step.column)
    dtype = prev.dtype(step.column)
    text = as_text(step.column, dtype)
    if step.match == "exact":
        found = (
            f"{text} = {lit(step.find)}"
            if step.case_sensitive
            else f"lower({text}) = lower({lit(step.find)})"
        )
        new = lit(step.replace) if is_text(dtype) else f"TRY_CAST({lit(step.replace)} AS {dtype})"
        return _select(
            prev, {step.column: f"CASE WHEN {found} THEN {new} ELSE {ident(step.column)} END"}
        )
    if step.match == "contains" and step.case_sensitive:
        return _select(
            prev, {step.column: f"replace({text}, {lit(step.find)}, {lit(step.replace)})"}
        )
    pattern = step.find if step.match == "regex" else _escape_regex(step.find)
    if not step.case_sensitive:
        pattern = "(?i)" + pattern
    replacement = step.replace if step.match == "regex" else step.replace.replace("\\", "\\\\")
    return _select(
        prev,
        {step.column: f"regexp_replace({text}, {lit(pattern)}, {lit(replacement)}, 'g')"},
    )


def _escape_regex(text: str) -> str:
    return "".join(f"\\{char}" if char in r".^$*+?()[]{}|\-" else char for char in text)


def _map_values(step: m.MapValues, prev: Stage) -> Body:
    """A correspondence table joined on the value (5,000 entries at most)."""
    _require(prev, step.column)
    dtype = prev.dtype(step.column)
    rows = ", ".join(
        f"({lit(old)}, {'NULL' if new is None else lit(new)})" for old, new in step.map.items()
    )
    table = f"(VALUES {rows}) AS map_table(old_value, new_value)"
    key = as_text(step.column, dtype).replace(ident(step.column), f"p.{ident(step.column)}")
    new = "map_table.new_value" if is_text(dtype) else f"TRY_CAST(map_table.new_value AS {dtype})"
    replaced = f"CASE WHEN map_table.old_value IS NULL THEN p.{ident(step.column)} ELSE {new} END"
    parts = [
        f"{replaced if name == step.column else 'p.' + ident(name)} AS {ident(name)}"
        for name in prev.names
    ]
    sql = (
        f"SELECT {', '.join(parts)}, p.{ident(ROW)} FROM {prev.name} AS p "
        f"LEFT JOIN {table} ON {key} = map_table.old_value"
    )
    return Body(sql, {name: name for name in prev.names})


def _fill(step: m.Fill, prev: Stage) -> Body:
    _require(prev, step.column)
    name = ident(step.column)
    value = f"nullif({name}, '')" if is_text(prev.dtype(step.column)) else name
    if step.direction == "down":
        frame = "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW"
        expr = f"last_value({value} IGNORE NULLS) OVER (ORDER BY {ident(ROW)} {frame})"
    else:
        frame = "ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING"
        expr = f"first_value({value} IGNORE NULLS) OVER (ORDER BY {ident(ROW)} {frame})"
    return _select(prev, {step.column: expr})


def _fill_empty(step: m.FillEmpty, prev: Stage) -> Body:
    _require(prev, step.column)
    dtype = prev.dtype(step.column)
    name = ident(step.column)
    if is_text(dtype):
        expr = (
            f"CASE WHEN {name} IS NULL OR trim({name}) = '' THEN {lit(step.value)} ELSE {name} END"
        )
    else:
        expr = f"coalesce({name}, TRY_CAST({lit(step.value)} AS {dtype}))"
    return _select(prev, {step.column: expr})


def _null_values(step: m.NullValues, prev: Stage) -> Body:
    _require(prev, step.column)
    text = as_text(step.column, prev.dtype(step.column))
    if step.match == "token":
        test = f"lower(trim({text})) IN {literal_list([v.strip().lower() for v in step.values])}"
    else:
        test = f"{text} IN {literal_list(step.values)}"
    return _select(prev, {step.column: f"CASE WHEN {test} THEN NULL ELSE {ident(step.column)} END"})


def _null_outside(step: m.NullOutside, prev: Stage) -> Body:
    _require(prev, step.column)
    if not is_number(prev.dtype(step.column)):
        raise m.CleanError(f"'{step.column}' is not a number column.")
    name = ident(step.column)
    tests = []
    if step.min is not None:
        tests.append(f"{name} < {float(step.min)!r}")
    if step.max is not None:
        tests.append(f"{name} > {float(step.max)!r}")
    return _select(prev, {step.column: f"CASE WHEN {' OR '.join(tests)} THEN NULL ELSE {name} END"})


def _pad_zeros(step: m.PadZeros, prev: Stage) -> Body:
    _require(prev, step.column)
    text = as_text(step.column, prev.dtype(step.column))
    only = "".join(f" AND coalesce({_condition(c, prev)}, false)" for c in step.conditions)
    expr = (
        f"CASE WHEN regexp_full_match({text}, '[0-9]+') AND length({text}) < {step.length}{only} "
        f"THEN lpad({text}, {step.length}, '0') ELSE {text} END"
    )
    return _select(prev, {step.column: expr})


def _date_expr(text: str, formats: list[str], excel: bool) -> str:
    # N19: month names read in English and French (strptime_sql)
    parts = [strptime_sql(text, f) for f in formats or ALL_DATE_FORMATS]
    if excel:
        number = f"TRY_CAST({text} AS DOUBLE)"
        parts.append(
            f"CASE WHEN regexp_full_match({text}, '[0-9]{{5}}(\\.[0-9]+)?') AND {number} "
            f"BETWEEN 20000 AND 60000 THEN CAST(DATE '1899-12-30' + "
            f"CAST(floor({number}) AS INTEGER) AS TIMESTAMP) END"
        )
    return parts[0] if len(parts) == 1 else f"coalesce({', '.join(parts)})"


def _change_type(step: m.ChangeType, prev: Stage) -> Body:
    _require(prev, step.column)
    dtype = prev.dtype(step.column)
    name = ident(step.column)
    cleaned = clean_text(step.column)  # trimmed, null tokens removed
    if step.to == "text":
        expr = as_text(step.column, dtype)
    elif step.to in ("number", "integer"):
        if is_text(dtype):
            number = number_sql(cleaned, step.decimal)
        elif is_number(dtype):
            number = f"CAST({name} AS DOUBLE)"
        else:
            number = f"TRY_CAST({name} AS DOUBLE)"
        if step.to == "number":
            expr = number
        elif is_integer(dtype):
            expr = f"CAST({name} AS BIGINT)"
        else:
            expr = f"CASE WHEN {number} = trunc({number}) THEN TRY_CAST({number} AS BIGINT) END"
    elif step.to in ("date", "datetime"):
        if is_text(dtype):
            stamp = _date_expr(cleaned, step.formats, step.excel_serial)
        elif is_number(dtype) and step.excel_serial:
            stamp = (
                f"CASE WHEN {name} BETWEEN 20000 AND 60000 THEN CAST(DATE '1899-12-30' + "
                f"CAST(floor({name}) AS INTEGER) AS TIMESTAMP) END"
            )
        else:
            stamp = f"TRY_CAST({name} AS TIMESTAMP)"
        expr = f"CAST({stamp} AS DATE)" if step.to == "date" else stamp
    else:
        if is_text(dtype):
            default = default_boolean_parser()
            trues = [v.lower() for v in step.true_values] or default.true_values
            falses = [v.lower() for v in step.false_values] or default.false_values
            low = f"lower({cleaned})"
            # N27: an empty word among the false ones makes a blank cell false (a checkbox)
            blank = (
                f" WHEN {low} IS NULL OR trim({low}) = '' THEN false"
                if any(not v.strip() for v in step.false_values)
                else ""
            )
            expr = (
                f"CASE WHEN {low} IN {literal_list(trues)} THEN true "
                f"WHEN {low} IN {literal_list(falses)} THEN false{blank} END"
            )
        elif is_number(dtype):
            expr = f"{name} <> 0"
        else:
            expr = f"TRY_CAST({name} AS BOOLEAN)"
    return _select(prev, {step.column: expr})


def _rename(step: m.RenameColumn, prev: Stage) -> Body:
    _require(prev, step.column)
    if step.to != step.column and step.to in prev.names:
        raise m.CleanError(f"A column '{step.to}' already exists.")
    return _select(prev, rename={step.column: step.to})


def _drop(step: m.DropColumns, prev: Stage) -> Body:
    _require(prev, *step.columns)
    if len(set(step.columns)) >= len(prev.names):
        raise m.CleanError("At least one column must remain.")
    return _select(prev, drop=set(step.columns))


def _duplicate(step: m.DuplicateColumn, prev: Stage) -> Body:
    _require(prev, step.column)
    if step.to in prev.names:
        raise m.CleanError(f"A column '{step.to}' already exists.")
    return _select(prev, after={step.column: [(step.to, ident(step.column))]})


def _whole(column: str, dtype: str) -> str:
    """A whole number from a column of any type ('2024', 2024, 2024.0), else NULL."""
    name = ident(column)
    if is_number(dtype):
        value = f"CAST({name} AS DOUBLE)"
    elif is_text(dtype):
        value = f"TRY_CAST(trim({name}) AS DOUBLE)"
    else:
        value = f"TRY_CAST({name} AS DOUBLE)"
    return f"CASE WHEN {value} = trunc({value}) THEN CAST({value} AS INTEGER) END"


def month_names() -> dict[str, int]:
    """Month names and abbreviations of the known orders (Jan, janvier…) → 1 to 12."""
    names: dict[str, int] = {}
    for order in load_knowledge().orders:
        if len(order) == 12:
            for number, word in enumerate(order, start=1):
                names.setdefault(word.lower(), number)
    return names


def month_sql(column: str, dtype: str) -> str:
    number = _whole(column, dtype)
    if not is_text(dtype):
        return number
    cases = " ".join(f"WHEN {lit(word)} THEN {k}" for word, k in month_names().items())
    return f"coalesce({number}, CASE lower(trim({ident(column)})) {cases} END)"


TIME_TEXT = "[0-9]{1,2}[:hH.][0-9]{2}([:.][0-9]{2})?"  # N19: 18.00.00 (UCI Air Quality)


def time_of_day_sql(column: str, dtype: str) -> tuple[str, str]:
    """C1 (fix1): the hour and the minute of a time of day: a time or date-time, '07:40',
    '7h40', '07.40', '07:40:00', or digits read as hhmm ('0740', 740, '5' = 00:05). NULL when it is
    none of these."""
    name = ident(column)
    if dtype.startswith("TIME") or is_temporal(dtype):
        return f"hour({name})", f"minute({name})"
    text = f"trim({as_text(column, dtype)})"
    digits = f"regexp_full_match({text}, '[0-9]{{1,4}}')"
    written = f"regexp_full_match({text}, {lit(TIME_TEXT)})"
    hour = (
        f"CASE WHEN {written} THEN "
        f"TRY_CAST(regexp_extract({text}, '^([0-9]{{1,2}})', 1) AS INTEGER) "
        f"WHEN {digits} THEN TRY_CAST({text} AS INTEGER) // 100 END"
    )
    minute = (
        f"CASE WHEN {written} THEN "
        f"TRY_CAST(regexp_extract({text}, '[:hH.]([0-9]{{2}})', 1) AS INTEGER) "
        f"WHEN {digits} THEN TRY_CAST({text} AS INTEGER) % 100 END"
    )
    return hour, minute


def _date_part(column: str, dtype: str, part: str, number: str) -> str:
    """N19 (B-20): the year, month or day of a date column (a DATE, a TIMESTAMP or an ISO date in
    text), so that a date and a time of day held apart make one moment; else the number given."""
    if is_temporal(dtype):
        return f"{part}({ident(column)})"
    if is_text(dtype):
        return f"coalesce({number}, {part}(TRY_CAST(trim({ident(column)}) AS TIMESTAMP)))"
    return number


def assembled_date_sql(step: m.AssembleDate, prev: Stage) -> str:
    """B-20: the date its parts make, NULL when they make none (out of range or impossible). A date
    column given as the year, the month and the day gives its own (N19: `Date` + `Time`)."""
    dtype = prev.dtype(step.year)
    year = _date_part(step.year, dtype, "year", _whole(step.year, dtype))
    dtype = prev.dtype(step.month)
    month = _date_part(step.month, dtype, "month", month_sql(step.month, dtype))
    day = "1"
    if step.day:
        dtype = prev.dtype(step.day)
        day = _date_part(step.day, dtype, "day", _whole(step.day, dtype))
    checks = [
        f"{year} BETWEEN 1000 AND 2200",
        f"{month} BETWEEN 1 AND 12",
        f"{day} BETWEEN 1 AND 31",
    ]
    if step.hour is None and step.time is None:
        text = f"printf('%04d-%02d-%02d', {year}, {month}, {day})"
        return f"CASE WHEN {' AND '.join(checks)} THEN TRY_CAST({text} AS DATE) END"
    if step.time is not None:
        hour, minute = time_of_day_sql(step.time, prev.dtype(step.time))
    else:
        assert step.hour is not None
        hour = _whole(step.hour, prev.dtype(step.hour))
        minute = _whole(step.minute, prev.dtype(step.minute)) if step.minute else "0"
    checks += [f"{hour} BETWEEN 0 AND 23", f"{minute} BETWEEN 0 AND 59"]
    text = f"printf('%04d-%02d-%02d %02d:%02d:00', {year}, {month}, {day}, {hour}, {minute})"
    return f"CASE WHEN {' AND '.join(checks)} THEN TRY_CAST({text} AS TIMESTAMP) END"


def _assemble_date(step: m.AssembleDate, prev: Stage) -> Body:
    parts = [c for c in (step.year, step.month, step.day, step.hour, step.minute, step.time) if c]
    _require(prev, *parts)
    if step.to in prev.names:
        raise m.CleanError(f"A column '{step.to}' already exists.")
    if step.minute and not step.hour:
        raise m.CleanError("A minute needs an hour.")
    last = max(parts, key=prev.names.index)  # the new date goes after its last part
    return _select(prev, after={last: [(step.to, assembled_date_sql(step, prev))]})


def _label_rows(step: m.LabelRows, prev: Stage) -> Body:
    """099 (R-PIN-15): a correspondence table joined on the row number, which every stage keeps."""
    if step.to in prev.names:
        raise m.CleanError(f"A column '{step.to}' already exists.")
    pairs = ", ".join(f"({int(row)}, {lit(label)})" for row, label in step.labels.items())
    table = f"(VALUES {pairs}) AS pin_table(pin_row, pin_label)"
    other = "NULL" if step.other is None else lit(step.other)
    parts = [f"p.{ident(name)} AS {ident(name)}" for name in prev.names]
    parts.append(f"CAST(coalesce(pin_table.pin_label, {other}) AS VARCHAR) AS {ident(step.to)}")
    sql = (
        f"SELECT {', '.join(parts)}, p.{ident(ROW)} FROM {prev.name} AS p "
        f"LEFT JOIN {table} ON p.{ident(ROW)} = pin_table.pin_row"
    )
    origin: dict[str, str | None] = {name: name for name in prev.names}
    origin[step.to] = None
    return Body(sql, origin)


def _set_cells(step: m.SetCells, prev: Stage) -> Body:
    """101 (C-33): a table (row number, new value) joined on the row number, as `label_rows`; the
    value is cast to the column's type at this stage (text stays text)."""
    _require(prev, step.column)
    dtype = prev.dtype(step.column)
    pairs = ", ".join(
        f"({int(row)}, {'NULL' if value is None else lit(value)})"
        for row, value in step.cells.items()
    )
    table = f"(VALUES {pairs}) AS cell_table(cell_row, cell_value)"
    value = "cell_table.cell_value"
    new = value if is_text(dtype) else f"TRY_CAST({value} AS {dtype})"
    column = ident(step.column)
    replaced = f"CASE WHEN cell_table.cell_row IS NULL THEN p.{column} ELSE {new} END"
    parts = [
        f"{replaced if name == step.column else 'p.' + ident(name)} AS {ident(name)}"
        for name in prev.names
    ]
    sql = (
        f"SELECT {', '.join(parts)}, p.{ident(ROW)} FROM {prev.name} AS p "
        f"LEFT JOIN {table} ON p.{ident(ROW)} = cell_table.cell_row"
    )
    return Body(sql, {name: name for name in prev.names})


def _drop_rows(step: m.DropRows, prev: Stage) -> Body:
    """101 (C-33): rows removed by their number in the file."""
    rows = ", ".join(str(int(row)) for row in sorted(set(step.rows)))
    return _where(prev, f"{ident(ROW)} NOT IN ({rows})")


def _condition(condition: m.RowCondition, prev: Stage) -> str:
    _require(prev, condition.column)
    dtype = prev.dtype(condition.column)
    name = ident(condition.column)
    text = as_text(condition.column, dtype)
    if condition.op == "empty":
        return f"({name} IS NULL OR trim({text}) = '')"
    if condition.op == "not_empty":
        return f"({name} IS NOT NULL AND trim({text}) <> '')"
    if condition.op == "in":
        return f"{text} IN {literal_list(condition.values or [])}"
    value = condition.value or ""
    if condition.op == "contains":
        return f"contains(lower({text}), lower({lit(value)}))"
    symbol = {"eq": "=", "ne": "<>", "lt": "<", "le": "<=", "gt": ">", "ge": ">="}[condition.op]
    if is_number(dtype):
        return f"{name} {symbol} TRY_CAST({lit(value)} AS DOUBLE)"
    if is_temporal(dtype):
        return f"{name} {symbol} TRY_CAST({lit(value)} AS {dtype})"
    return f"{text} {symbol} {lit(value)}"


def _filter_rows(step: m.FilterRows, prev: Stage) -> Body:
    joined = (" AND " if step.combine == "all" else " OR ").join(
        _condition(c, prev) for c in step.conditions
    )
    test = f"coalesce(({joined}), false)"
    return _where(prev, test if step.action == "keep" else f"NOT {test}")


def _drop_duplicates(step: m.DropDuplicates, prev: Stage) -> Body:
    if step.groups is not None or step.survivors or step.not_duplicates or step.merge:
        return _reviewed_duplicates(step, prev)
    keys = step.keys or prev.names
    _require(prev, *keys)
    partition = ", ".join(ident(key) for key in keys)
    order = _keep_order(step, prev)
    body = (
        f"SELECT * FROM {prev.name} "
        f"QUALIFY row_number() OVER (PARTITION BY {partition} ORDER BY {order}) = 1"
    )
    return Body(body, {name: name for name in prev.names}, removes_rows=True)


DUP = "__quantilo_dup"  # N35: the table of explicit groups (row, group, place in it)


def duplicate_groups_sql(step: m.DropDuplicates, prev: Stage) -> tuple[str, str, str]:
    """N35: the rows of `prev` with their group (`partition`) and their place in it by the rule
    (`order`, the first kept), as `FROM` clause, partition and order expressions."""
    if step.groups is None:
        keys = step.keys or prev.names
        _require(prev, *keys)
        partition = ", ".join(ident(key) for key in keys)
        return f"{prev.name}", partition, _keep_order(step, prev)
    triples = ", ".join(
        f"({int(row)}, {group}, {place})"
        for group, rows in enumerate(step.groups)
        for place, row in enumerate(rows)
    )
    names = f"{DUP}_row, {DUP}_group, {DUP}_place"
    table = f"(VALUES {triples}) AS {DUP}({names})"
    source = f"{prev.name} LEFT JOIN {table} ON {ident(ROW)} = {DUP}.{DUP}_row"
    # a row in no group is its own: never removed
    partition = f"coalesce({DUP}.{DUP}_group, -1 - {ident(ROW)})"
    return source, partition, f"{DUP}.{DUP}_place, {ident(ROW)}"


def _reviewed_duplicates(step: m.DropDuplicates, prev: Stage) -> Body:
    """N35: the rule's groups, a row the user chose kept first in its own, the groups said not
    duplicates kept whole, and (`merge`) the kept row's empty cells filled from its group."""
    source, partition, order = duplicate_groups_sql(step, prev)
    if step.survivors:
        chosen = ", ".join(str(int(row)) for row in sorted(set(step.survivors)))
        order = f"CASE WHEN {ident(ROW)} IN ({chosen}) THEN 0 ELSE 1 END, {order}"
    apart = sorted({int(row) for group in step.not_duplicates or [] for row in group})
    whole = (
        f"bool_or({ident(ROW)} IN ({', '.join(map(str, apart))})) OVER (PARTITION BY {partition})"
        if apart
        else "false"
    )
    window = f"PARTITION BY {partition} ORDER BY {order}"
    merged = []
    for index, (name, dtype) in enumerate(prev.columns):
        if not step.merge:
            break
        column = ident(name)
        value = f"CASE WHEN {column} IS NULL OR trim({as_text(name, dtype)}) = '' THEN NULL "
        value += f"ELSE {column} END"
        merged.append(
            f"first_value({value} IGNORE NULLS) OVER ({window} ROWS BETWEEN UNBOUNDED PRECEDING "
            f"AND UNBOUNDED FOLLOWING) AS __quantilo_fill{index}"
        )
    inner = (
        f"SELECT {prev.name}.*, row_number() OVER ({window}) AS __quantilo_rank, {whole} AS "
        f"__quantilo_whole{''.join(', ' + part for part in merged)} FROM {source}"
    )
    parts = []
    for index, name in enumerate(prev.names):
        column = ident(name)
        if step.merge:
            parts.append(
                f"CASE WHEN __quantilo_whole THEN {column} ELSE coalesce(__quantilo_fill{index}, "
                f"{column}) END AS {column}"
            )
        else:
            parts.append(column)
    body = (
        f"SELECT {', '.join(parts)}, {ident(ROW)} FROM ({inner}) "
        "WHERE __quantilo_rank = 1 OR __quantilo_whole"
    )
    return Body(body, {name: name for name in prev.names}, removes_rows=True)


ENTITY = "__quantilo_entity"  # N36: the columns an entity number is computed with


def entity_name_sql(text: str) -> str:
    """N36: a name as `entity_column` reads a record: case, accents and runs of spaces folded,
    empty when blank (`text`: the column as text)."""
    spaced = f"regexp_replace(trim({text}), {lit(SQL_SPACE_RUN)}, ' ', 'g')"
    return f"nullif(lower(strip_accents({spaced})), '')"


def entity_anchor_sql(text: str, kind: str) -> str:
    """N36: an anchor normalized by its kind: a postcode without spaces or signs, in capitals; a
    phone's last 9 digits (7 at least); an e-mail in small letters; a key as written."""
    if kind == "postal_code":
        return f"nullif(upper(regexp_replace({text}, '[^0-9A-Za-z]', '', 'g')), '')"
    if kind == "phone":
        digits = f"regexp_replace({text}, '[^0-9]', '', 'g')"
        return f"CASE WHEN length({digits}) >= 7 THEN right({digits}, 9) END"
    if kind == "email":
        return f"nullif(lower(trim({text})), '')"
    return f"nullif(trim({text}), '')"


def entity_keys_sql(step: m.EntityColumn, prev: Stage) -> tuple[str, str]:
    """N36: the record of a row at this stage: its name and anchor keys, as SQL."""
    _require(prev, step.column, *([step.anchor] if step.anchor else []))
    name = entity_name_sql(as_text(step.column, prev.dtype(step.column)))
    if step.anchor is None:
        return name, "NULL"
    return name, entity_anchor_sql(as_text(step.anchor, prev.dtype(step.anchor)), step.anchor_kind)


def entity_cluster_keys(step: m.EntityColumn) -> list[tuple[str, str, str]]:
    """N36: (anchor, name, entity) of each record of a cluster the review kept: the records of a
    cluster share its entity, those taken out of it (`split`) another one; a cluster said apart
    gives none (each record its own entity)."""
    apart = set(step.not_entities or [])
    taken = {index: set(records) for index, records in (step.split or {}).items()}
    found = []
    for index, records in enumerate(step.groups):
        if index in apart:
            continue
        for at, record in enumerate(records):
            part = "b" if at in taken.get(index, set()) else "a"
            found.append((record.anchor, record.name, f"{chr(30)}{index}{part}"))
    return found


def _entity_column(step: m.EntityColumn, prev: Stage) -> Body:
    """N36 (A-20): the entity number of each row, by its record (name and anchor folded) and the
    clusters of records the step lists; the entities numbered in the order they first appear."""
    _new_names(prev, [step.to])
    name, anchor = entity_keys_sql(step, prev)
    row = ident(ROW)
    inner = f"SELECT {prev.name}.*, {name} AS {ENTITY}_name, {anchor} AS {ENTITY}_anchor "
    inner += f"FROM {prev.name}"
    own = f"coalesce({ENTITY}_anchor, '') || chr(31) || {ENTITY}_name"
    clusters = entity_cluster_keys(step)
    if clusters:
        values = ", ".join(f"({lit(a)}, {lit(n)}, {lit(c)})" for a, n, c in clusters)
        table = f"(VALUES {values}) AS {ENTITY}_map({ENTITY}_a, {ENTITY}_n, {ENTITY}_c)"
        key = f"coalesce({ENTITY}_map.{ENTITY}_c, {own})"
        source = (
            f"({inner}) AS {ENTITY}_rows LEFT JOIN {table} ON {ENTITY}_anchor = {ENTITY}_a "
            f"AND {ENTITY}_name = {ENTITY}_n"
        )
    else:
        key, source = own, f"({inner}) AS {ENTITY}_rows"
    keyed = (
        f"SELECT *, CASE WHEN {ENTITY}_name IS NULL THEN NULL ELSE min({row}) OVER "
        f"(PARTITION BY {key}) END AS {ENTITY}_first FROM {source}"
    )
    parts = [ident(name) for name in prev.names]
    parts.append(
        f"CAST(CASE WHEN {ENTITY}_first IS NULL THEN NULL ELSE dense_rank() OVER (ORDER BY "
        f"{ENTITY}_first NULLS LAST) END AS BIGINT) AS {ident(step.to)}"
    )
    sql = f"SELECT {', '.join(parts)}, {row} FROM ({keyed})"
    origin: dict[str, str | None] = {name: name for name in prev.names}
    origin[step.to] = None
    return Body(sql, origin)


def _keep_order(step: m.DropDuplicates, prev: Stage) -> str:
    """The rows of a group in the order `keep` reads them: the first is kept."""
    if step.keep == "first":
        order = f"{ident(ROW)}"
    elif step.keep == "last":
        order = f"{ident(ROW)} DESC"
    elif step.keep == "latest":
        assert step.order_by is not None  # the model refuses `latest` without it
        _require(prev, step.order_by)
        order = f"{ident(step.order_by)} DESC NULLS LAST, {ident(ROW)}"
    else:
        filled = " + ".join(f"CASE WHEN {ident(n)} IS NULL THEN 0 ELSE 1 END" for n in prev.names)
        order = f"({filled}) DESC, {ident(ROW)}"
    return order


def _blank(column: str, dtype: str) -> str:
    return f"coalesce(trim({as_text(column, dtype)}), '') = ''"


def _drop_empty_rows(step: m.DropEmptyRows, prev: Stage) -> Body:
    if step.share is None:
        every = " AND ".join(_blank(name, dtype) for name, dtype in prev.columns)
        return _where(prev, f"NOT ({every})")
    empty = " + ".join(f"CASE WHEN {_blank(n, d)} THEN 1 ELSE 0 END" for n, d in prev.columns)
    return _where(prev, f"({empty}) <= {float(step.share)!r} * {len(prev.columns)}")


def _drop_header_rows(_step: m.DropHeaderRows, prev: Stage) -> Body:
    if len(prev.columns) < 2:
        raise m.CleanError("A repeated header needs at least two columns.")
    every = " AND ".join(
        f"coalesce(lower(trim({as_text(name, dtype)})) = {lit(name.strip().lower())}, false)"
        for name, dtype in prev.columns
    )
    return _where(prev, f"NOT ({every})")


# --- M5: numbers, splitting, extracting, calculating (study § 3) --------------------------------


def _number_column(prev: Stage, column: str) -> str:
    _require(prev, column)
    if not is_number(prev.dtype(column)):
        raise m.CleanError(f"'{column}' is not a number column at this step.")
    return ident(column)


def _transform_number(step: m.TransformNumber, prev: Stage) -> Body:
    name = _number_column(prev, step.column)
    if step.how == "round":
        expr = f"round({name}, {int(step.decimals)})"
    elif step.how == "abs":
        expr = f"abs({name})"
    elif step.how == "multiply":
        expr = f"{name} * {float(step.factor)!r}"
    elif step.how == "divide":
        expr = f"{name} / {float(step.factor)!r}"
    elif step.how == "clip":
        expr = name
        if step.min is not None:
            expr = f"greatest({expr}, {float(step.min)!r})"
        if step.max is not None:
            expr = f"least({expr}, {float(step.max)!r})"
        expr = f"CASE WHEN {name} IS NULL THEN NULL ELSE {expr} END"
    else:
        width = float(step.width or 1.0)
        expr = f"floor({name} / {width!r}) * {width!r}"
    return _select(prev, {step.column: expr})


def _new_names(prev: Stage, names: list[str]) -> None:
    for name in names:
        if name in prev.names:
            raise m.CleanError(f"A column '{name}' already exists.")


def _split_column(step: m.SplitColumn, prev: Stage) -> Body:
    _require(prev, step.column)
    _new_names(prev, step.to)
    text = as_text(step.column, prev.dtype(step.column))
    count = len(step.to)
    parts: list[str] = []
    if step.by == "positions":
        cuts = [1, *step.positions]
        for index, start in enumerate(cuts):
            if index + 1 < len(cuts):
                parts.append(f"substr({text}, {start}, {cuts[index + 1] - start})")
            else:
                parts.append(f"substr({text}, {start})")
    else:
        if step.by == "delimiter":
            pieces = f"string_split({text}, {lit(step.delimiter)})"
            glue = lit(step.delimiter)
        else:
            pieces = f"regexp_extract_all({text}, '[0-9]+|[^0-9]+')"
            glue = "''"
        for index in range(1, count):
            parts.append(f"{pieces}[{index}]")
        parts.append(f"array_to_string({pieces}[{count}:], {glue})")
    exprs = [(name, f"nullif({part}, '')") for name, part in zip(step.to, parts, strict=True)]
    return _select(prev, after={step.column: exprs})


def _merge_columns(step: m.MergeColumns, prev: Stage) -> Body:
    _require(prev, *step.columns)
    _new_names(prev, [step.to])
    values = [f"nullif(trim({as_text(column, prev.dtype(column))}), '')" for column in step.columns]
    expr = f"nullif(concat_ws({lit(step.separator)}, {', '.join(values)}), '')"
    last = max(step.columns, key=prev.names.index)
    return _select(prev, after={last: [(step.to, expr)]})


def check_pattern(connection: duckdb.DuckDBPyConnection, pattern: str) -> None:
    """A regular expression DuckDB can use (RE2), else a CleanError saying why."""
    try:
        connection.execute("SELECT regexp_matches('', ?)", [pattern]).fetchall()
    except duckdb.Error as error:
        raise m.CleanError(f"Invalid pattern: {_short(error)}") from None


def _extract_text(step: m.ExtractText, prev: Stage) -> Body:
    _require(prev, step.column)
    text = as_text(step.column, prev.dtype(step.column))
    if step.how == "first":
        expr = f"left({text}, {int(step.count)})"
    elif step.how == "last":
        expr = f"right({text}, {int(step.count)})"
    elif step.how == "between":
        pattern = f"{_escape_regex(step.start)}(.*?){_escape_regex(step.end) if step.end else '$'}"
        if not step.start:
            pattern = f"^(.*?){_escape_regex(step.end)}"
        expr = f"regexp_extract({text}, {lit('(?s)' + pattern)}, 1)"
    else:
        pattern = step.pattern or ""
        group = 1 if re.search(r"\((?!\?)", pattern) else 0
        expr = f"regexp_extract({text}, {lit(pattern)}, {group})"
    expr = f"nullif({expr}, '')"
    if step.to is None:
        return _select(prev, {step.column: expr})
    _new_names(prev, [step.to])
    return _select(prev, after={step.column: [(step.to, expr)]})


def _format_text(step: m.FormatText, prev: Stage) -> Body:
    _require(prev, step.column)
    text = as_text(step.column, prev.dtype(step.column))
    value = text
    if step.pad_to:
        value = (
            f"CASE WHEN length({text}) < {int(step.pad_to)} "
            f"THEN lpad({text}, {int(step.pad_to)}, {lit(step.pad_with)}) ELSE {text} END"
        )
    expr = f"CASE WHEN {text} IS NULL OR {text} = '' THEN {text} ELSE "
    expr += f"{lit(step.prefix)} || {value} || {lit(step.suffix)} END"
    return _select(prev, {step.column: expr})


def _parse_quantity(step: m.ParseQuantity, prev: Stage) -> Body:
    """N23: the number and the unit read by the rule's own pattern (F-05), the number brought to
    the base unit of its kind; another kind, or an unknown unit, gives an empty cell."""
    from quantilo.cleaning.rules.formats import QUANTITY_PATTERN, UNITS

    _require(prev, step.column)
    targets = [
        (name, kind)
        for name, kind in (
            (step.to_grams, "grams"),
            (step.to_litres, "litres"),
            (step.to_metres, "metres"),
        )
        if name is not None
    ]
    _new_names(prev, [name for name, _ in targets] + ([step.unit_to] if step.unit_to else []))
    text = as_text(step.column, prev.dtype(step.column))
    pattern = lit(QUANTITY_PATTERN)

    def group(index: int) -> str:
        return f"regexp_extract({text}, {pattern}, {index})"

    unit = f"lower(regexp_replace({group(3)}, '[ .]', '', 'g'))"
    number = (
        f"TRY_CAST(replace({group(2)}, ',', '.') AS DOUBLE) * "
        f"coalesce(TRY_CAST(nullif({group(1)}, '') AS DOUBLE), 1)"
    )
    exprs: list[tuple[str, str]] = []
    for name, kind in targets:
        cases = " ".join(
            f"WHEN {lit(key)} THEN {factor!r}" for key, (of, factor) in UNITS.items() if of == kind
        )
        exprs.append((name, f"round(CASE {unit} {cases} END * {number}, 9)"))
    if step.unit_to is not None:
        exprs.append((step.unit_to, f"nullif({group(3)}, '')"))
    return _select(prev, after={step.column: exprs})


def _case_test(case: m.Case, prev: Stage) -> str:
    tests: list[str] = []
    if case.conditions:
        joined = (" AND " if case.combine == "all" else " OR ").join(
            _condition(c, prev) for c in case.conditions
        )
        tests.append(f"({joined})")
    if case.when is not None:
        tests.append(fx.to_sql(case.when, prev.names))
    return f"coalesce({' AND '.join(tests)}, false)"


def _case_value(case: m.Case, prev: Stage, dtype: str | None) -> str:
    """The value a case gives, in `dtype` (a column's type), or as written (a new column)."""
    if case.formula is not None:
        value = fx.to_sql(case.formula, prev.names)
        if dtype is None:
            return value
        return f"CAST({value} AS VARCHAR)" if is_text(dtype) else f"TRY_CAST({value} AS {dtype})"
    if case.value is None:
        return "NULL"
    if dtype is None or is_text(dtype):
        return lit(case.value)
    return f"TRY_CAST({lit(case.value)} AS {dtype})"


def _cases(cases: list[m.Case], prev: Stage, dtype: str | None, otherwise: str) -> str:
    whens = " ".join(
        f"WHEN {_case_test(case, prev)} THEN {_case_value(case, prev, dtype)}" for case in cases
    )
    return f"CASE {whens} ELSE {otherwise} END"


def _set_where(step: m.SetWhere, prev: Stage) -> Body:
    _require(prev, step.column)
    dtype = prev.dtype(step.column)
    return _select(prev, {step.column: _cases(step.cases, prev, dtype, ident(step.column))})


def _conditional_column(step: m.ConditionalColumn, prev: Stage) -> Body:
    _new_names(prev, [step.to])
    otherwise = "NULL" if step.otherwise is None else lit(step.otherwise)
    if step.otherwise is None and all(case.formula is None for case in step.cases):
        otherwise = "CAST(NULL AS VARCHAR)"
    expr = _cases(step.cases, prev, None, otherwise)
    return _select(prev, after={prev.names[-1]: [(step.to, expr)]})


def _computed_column(step: m.ComputedColumn, prev: Stage) -> Body:
    _new_names(prev, [step.to])
    expr = fx.to_sql(step.formula, prev.names)
    return _select(prev, after={prev.names[-1]: [(step.to, expr)]})


def _swap_values(step: m.SwapValues, prev: Stage) -> Body:
    first, second = step.columns
    _require(prev, first, second)
    types = prev.dtype(first), prev.dtype(second)
    alike = types[0] == types[1] or all(is_temporal(t) for t in types)
    if not (alike or all(is_number(t) for t in types)):
        raise m.CleanError(
            f"'{first}' and '{second}' have different types at this step ({types[0]}, "
            f"{types[1]}): read both as dates (or numbers) first."
        )
    test = f"coalesce({fx.to_sql(step.when, prev.names)}, false)"
    return _select(
        prev,
        {
            first: f"CASE WHEN {test} THEN CAST({ident(second)} AS {types[0]}) "
            f"ELSE {ident(first)} END",
            second: f"CASE WHEN {test} THEN CAST({ident(first)} AS {types[1]}) "
            f"ELSE {ident(second)} END",
        },
    )


# --- C1: dates, rows and header, structure (rest of M5) -----------------------------------------

UNITS = {
    "years": "to_years",
    "months": "to_months",
    "days": "to_days",
    "hours": "to_hours",
    "minutes": "to_minutes",
    "seconds": "to_seconds",
}


def _renumbered(order: str) -> str:
    """C-59: the rows numbered again from 0, in `order`."""
    return f"CAST(row_number() OVER (ORDER BY {order}) - 1 AS BIGINT) AS {ident(ROW)}"


def _date_column(prev: Stage, column: str) -> str:
    _require(prev, column)
    dtype = prev.dtype(column)
    if not is_temporal(dtype):
        raise m.CleanError(
            f"'{column}' is not a date at this step: read it as a date first (Change type)."
        )
    return dtype


def _shift_date(step: m.ShiftDate, prev: Stage) -> Body:
    dtype = _date_column(prev, step.column)
    if dtype == "DATE" and step.unit in ("hours", "minutes", "seconds"):
        raise m.CleanError(
            f"'{step.column}' holds dates without a time: shift it by days or more, or read it "
            "as a date-time first."
        )
    name = ident(step.column)
    expr = f"CAST({name} + {UNITS[step.unit]}({int(step.amount)}) AS {dtype})"
    return _select(prev, {step.column: expr})


def _convert_timezone(step: m.ConvertTimezone, prev: Stage, context: Context) -> Body:
    dtype = _date_column(prev, step.column)
    if dtype == "DATE":
        raise m.CleanError(f"'{step.column}' holds dates without a time of day: no zone applies.")
    for zone in (step.from_zone, step.to_zone):
        try:
            found = context.connection.execute(
                "SELECT count(*) FROM pg_timezone_names() WHERE name = ?", [zone]
            ).fetchone()
        except duckdb.Error:
            raise m.CleanError("Time zones are not available here (DuckDB without ICU).") from None
        if not found or not found[0]:
            raise m.CleanError(
                f"Unknown time zone '{zone}': give an IANA name such as Europe/Paris, or UTC."
            )
    name = ident(step.column)
    if dtype.startswith("TIMESTAMP WITH TIME ZONE"):  # an instant: only the zone it is read in
        expr = f"timezone({lit(step.to_zone)}, {name})"
    else:
        expr = f"timezone({lit(step.to_zone)}, timezone({lit(step.from_zone)}, {name}))"
    return _select(prev, {step.column: expr})


def _keep_rows(step: m.KeepRows, prev: Stage) -> Body:
    order = f"{ident(ROW)}" if step.how in ("first", "remove_first") else f"{ident(ROW)} DESC"
    test = "<=" if step.how in ("first", "last") else ">"
    body = (
        f"SELECT * FROM {prev.name} "
        f"QUALIFY row_number() OVER (ORDER BY {order}) {test} {int(step.count)}"
    )
    return Body(body, {name: name for name in prev.names}, removes_rows=True)


def _promote_header(step: m.PromoteHeader, prev: Stage, context: Context) -> Body:
    """C-54: the names read on the first row (as text, trimmed); an empty one keeps the current
    name, a repeated one gets ' 2', ' 3'… (case ignored). N47: on row `step.row`, the rows above it
    leaving the table with it."""
    texts = ", ".join(as_text(name, dtype) for name, dtype in prev.columns)
    rows = context.rows(
        f"SELECT {ident(ROW)}, {texts} FROM {prev.name} ORDER BY {ident(ROW)} "
        f"LIMIT 1 OFFSET {int(step.row) - 1}"
    )
    if not rows:
        raise m.CleanError(f"The table has no row {step.row} to read the names from.")
    first, values = int(rows[0][0]), rows[0][1:]
    seen: set[str] = set()
    rename: dict[str, str] = {}
    for old, value in zip(prev.names, values, strict=True):
        base = (str(value).strip() if value is not None else "")[:500] or old
        name, count = base, 2
        while name.lower() in seen:
            name, count = f"{base} {count}", count + 1
        seen.add(name.lower())
        rename[old] = name
    body = _select(prev, rename=rename)
    body.sql += f" WHERE {ident(ROW)} > {first}"
    body.removes_rows = True
    return body


def _split_rows(step: m.SplitRows, prev: Stage) -> Body:
    """C-55: one row per part; an empty cell gives one row with an empty part."""
    _require(prev, step.column)
    text = as_text(step.column, prev.dtype(step.column))
    parts = f"coalesce(string_split({text}, {lit(step.delimiter)}), [NULL::VARCHAR])"
    inner = (
        f"SELECT *, unnest({parts}) AS __q_part, unnest(range(len({parts}))) AS __q_pos "
        f"FROM {prev.name}"
    )
    value = "nullif(trim(__q_part), '')" if step.trim else "nullif(__q_part, '')"
    columns = [f"{value if n == step.column else ident(n)} AS {ident(n)}" for n in prev.names]
    sql = f"SELECT {', '.join(columns)}, {_renumbered(f'{ident(ROW)}, __q_pos')} FROM ({inner})"
    return Body(sql, {name: name for name in prev.names}, reshapes=True)


def _unpivot(step: m.Unpivot, prev: Stage) -> Body:
    """C-56: one row per (row, column turned); values of mixed types are written as text."""
    _require(prev, *step.columns)
    melted = list(dict.fromkeys(step.columns))
    kept = [n for n in prev.names if n not in set(melted)]
    _new_names(Stage(prev.name, [(n, "") for n in kept]), [step.names_to, step.values_to])
    types = {prev.dtype(c) for c in melted}
    values = [ident(c) if len(types) == 1 else as_text(c, prev.dtype(c)) for c in melted]
    inner = (
        f"SELECT *, unnest([{', '.join(lit(c) for c in melted)}]) AS __q_name, "
        f"unnest([{', '.join(values)}]) AS __q_value, unnest(range({len(melted)})) AS __q_pos "
        f"FROM {prev.name}"
    )
    where = " WHERE __q_value IS NOT NULL" if step.drop_empty else ""
    columns = [f"{ident(n)} AS {ident(n)}" for n in kept]
    columns += [f"__q_name AS {ident(step.names_to)}", f"__q_value AS {ident(step.values_to)}"]
    sql = (
        f"SELECT {', '.join(columns)}, {_renumbered(f'{ident(ROW)}, __q_pos')} "
        f"FROM ({inner}){where}"
    )
    origin: dict[str, str | None] = {name: name for name in kept}
    origin[step.names_to] = origin[step.values_to] = None
    return Body(sql, origin, reshapes=True)


def _aggregate(fn: str, column: str | None, prev: Stage, where: str = "") -> str:
    """An aggregate of a column at this stage (rows counted when there is none)."""
    filtered = f" FILTER (WHERE {where})" if where else ""
    if column is None:
        return f"count(*){filtered}"
    _require(prev, column)
    dtype = prev.dtype(column)
    name = ident(column)
    if fn in ("sum", "mean", "median") and not is_number(dtype):
        raise m.CleanError(f"'{column}' is not a number column at this step: `{fn}` needs one.")
    sql = {
        "count": f"count({name})",
        "count_distinct": f"count(DISTINCT {name})",
        "sum": f"sum({name})",
        "mean": f"avg({name})",
        "median": f"median({name})",
        "min": f"min({name})",
        "max": f"max({name})",
        "first": f"arg_min({name}, {ident(ROW)})",
        "last": f"arg_max({name}, {ident(ROW)})",
    }[fn]
    return sql + filtered


def _grouped(prev: Stage, keys: list[str], aggregates: list[tuple[str, str]]) -> str:
    """One row per group of `keys`, the groups in the order they first appear."""
    selected = [*(ident(k) for k in keys), *(f"{expr} AS {ident(to)}" for to, expr in aggregates)]
    group = f" GROUP BY {', '.join(ident(k) for k in keys)}" if keys else ""
    inner = f"SELECT {', '.join(selected)}, min({ident(ROW)}) AS __q_first FROM {prev.name}{group}"
    return f"SELECT * EXCLUDE (__q_first), {_renumbered('__q_first')} FROM ({inner})"


def _pivot(step: m.Pivot, prev: Stage, context: Context) -> Body:
    _require(prev, step.column)
    text = as_text(step.column, prev.dtype(step.column))
    headers: list[str | None] = list(step.headers)
    if not headers:
        found = context.rows(
            f"SELECT {text}, min({ident(ROW)}) AS first FROM {prev.name} GROUP BY 1 "
            f"ORDER BY first LIMIT {m.MAX_HEADERS + 1}"
        )
        if len(found) > m.MAX_HEADERS:
            raise m.CleanError(
                f"'{step.column}' has more than {m.MAX_HEADERS} values: too many columns. "
                "List the values to keep in Headers."
            )
        headers = [None if row[0] is None else str(row[0]) for row in found]
    others = [n for n in prev.names if n not in (step.column, step.value_column)]
    taken = {n.lower() for n in others}
    aggregates: list[tuple[str, str]] = []
    for header in headers:
        name = header if header else "(empty)"
        if name.lower() in taken:
            raise m.CleanError(f"'{name}' would be two columns: rename the column first.")
        taken.add(name.lower())
        where = f"{text} IS NULL OR {text} = ''" if not header else f"{text} = {lit(header)}"
        aggregates.append((name, _aggregate(step.aggregate, step.value_column, prev, where)))
    origin: dict[str, str | None] = {name: name for name in others}
    origin.update({name: None for name, _ in aggregates})
    return Body(_grouped(prev, others, aggregates), origin, reshapes=True)


def _group_by(step: m.GroupBy, prev: Stage) -> Body:
    _require(prev, *step.keys)
    aggregates = [(a.to, _aggregate(a.fn, a.column, prev)) for a in step.aggregations]
    origin: dict[str, str | None] = {key: key for key in step.keys}
    origin.update({a.to: None for a in step.aggregations})
    return Body(_grouped(prev, list(step.keys), aggregates), origin, reshapes=True)


def _other(step: m.AppendRows | m.MergeDataset, context: Context) -> tuple[str, str]:
    """The other dataset's Parquet and the name to say."""
    label = step.label or step.dataset
    path = context.sources.get(step.dataset)
    if not path:
        raise m.CleanError(
            f"The dataset '{label}' is not available: it was deleted, or is not ready."
        )
    return path, label


def _append_rows(step: m.AppendRows, prev: Stage, context: Context) -> Body:
    """C-57: `UNION ALL BY NAME`; this table's rows first, then the other's, each in file order."""
    path, label = _other(step, context)
    side = ""
    if step.source_column:
        _new_names(prev, [step.source_column])
        side = f", {{}} AS {ident(step.source_column)}"
    this = lit(step.this_label or "this dataset")
    mine = f"SELECT *, 0 AS __q_side{side.format(this)} FROM {prev.name}"
    theirs = f"SELECT *, 1 AS __q_side{side.format(lit(label))} FROM ({source_sql(path)})"
    union = f"({mine}) UNION ALL BY NAME ({theirs})"
    sql = (
        f"SELECT * EXCLUDE (__q_side, {ident(ROW)}), {_renumbered(f'__q_side, {ident(ROW)}')} "
        f"FROM ({union})"
    )
    columns = _describe(context.connection, f"WITH {', '.join(context.ctes)} {sql}")
    origin: dict[str, str | None] = {
        name: (name if name in prev.names else None) for name, _ in columns
    }
    return Body(sql, origin, reshapes=True)


def _merge_dataset(step: m.MergeDataset, prev: Stage, context: Context) -> Body:
    """C-58: a lookup, never a multiplication: the other side must have one row per key."""
    path, label = _other(step, context)
    _require(prev, *step.keys)
    source = f"read_parquet({lit(path)})"
    theirs = _describe(context.connection, f"SELECT * FROM {source}")
    other = Stage("o", theirs)
    for key in step.other_keys:
        if key not in other.names:
            raise m.CleanError(f"'{key}' is not a column of '{label}'.")
    add = step.add if step.add is not None else [n for n in other.names if n not in step.other_keys]
    for name in add:
        if name not in other.names:
            raise m.CleanError(f"'{name}' is not a column of '{label}'.")
    if not add:
        raise m.CleanError("Choose at least one column to add.")
    names = [f"{step.prefix}{name}" for name in add]
    for name in names:
        if name in prev.names:
            raise m.CleanError(f"A column '{name}' already exists here: give a prefix.")
    if len(set(names)) != len(names):
        raise m.CleanError("Two added columns share a name.")
    there = [as_text(k, other.dtype(k)) for k in step.other_keys]
    repeated = context.connection.execute(
        f"SELECT count(*), any_value(k) FROM (SELECT concat_ws(' · ', {', '.join(there)}) AS k "
        f"FROM {source} WHERE {' AND '.join(f'({t}) IS NOT NULL' for t in there)} "
        "GROUP BY ALL HAVING count(*) > 1)"
    ).fetchone()
    if repeated and repeated[0]:
        raise m.CleanError(
            f"'{label}' has {int(repeated[0]):,} keys on several rows (such as {repeated[1]!r}): "
            "rows would multiply. Group it by its key first, or choose other keys (R-JOIN-04)."
        )
    here = [as_text(k, prev.dtype(k)).replace(ident(k), f"p.{ident(k)}") for k in step.keys]
    there_o = [
        t.replace(ident(k), f"o.{ident(k)}") for t, k in zip(there, step.other_keys, strict=True)
    ]
    on = " AND ".join(f"{a} = {b}" for a, b in zip(here, there_o, strict=True))
    columns = [f"p.{ident(n)} AS {ident(n)}" for n in prev.names]
    columns += [f"o.{ident(a)} AS {ident(n)}" for a, n in zip(add, names, strict=True)]
    join = "LEFT JOIN" if step.how == "left" else "JOIN"
    sql = (
        f"SELECT {', '.join(columns)}, p.{ident(ROW)} FROM {prev.name} AS p "
        f"{join} {source} AS o ON {on}"
    )
    origin: dict[str, str | None] = {name: name for name in prev.names}
    origin.update({name: None for name in names})
    return Body(sql, origin, removes_rows=step.how == "inner")


# steps that read data while they are compiled (names, values, another dataset)
CONTEXT_BUILDERS: dict[str, Callable[[Any, Stage, Context], Body]] = {
    "convert_timezone": _convert_timezone,
    "promote_header": _promote_header,
    "pivot": _pivot,
    "append_rows": _append_rows,
    "merge_dataset": _merge_dataset,
}

BUILDERS: dict[str, Callable[[Any, Stage], Body]] = {
    "rename_column": _rename,
    "drop_columns": _drop,
    "duplicate_column": _duplicate,
    "assemble_date": _assemble_date,
    "label_rows": _label_rows,
    "change_type": _change_type,
    "normalize_spaces": _normalize_spaces,
    "remove_invisible": _remove_invisible,
    "change_case": _change_case,
    "replace_values": _replace_values,
    "map_values": _map_values,
    "set_cells": _set_cells,
    "fill": _fill,
    "fill_empty": _fill_empty,
    "null_values": _null_values,
    "null_outside": _null_outside,
    "pad_zeros": _pad_zeros,
    "transform_number": _transform_number,
    "split_column": _split_column,
    "merge_columns": _merge_columns,
    "extract_text": _extract_text,
    "format_text": _format_text,
    "parse_quantity": _parse_quantity,
    "set_where": _set_where,
    "conditional_column": _conditional_column,
    "computed_column": _computed_column,
    "swap_values": _swap_values,
    "filter_rows": _filter_rows,
    "drop_duplicates": _drop_duplicates,
    "entity_column": _entity_column,
    "drop_rows": _drop_rows,
    "drop_empty_rows": _drop_empty_rows,
    "drop_header_rows": _drop_header_rows,
    "shift_date": _shift_date,
    "keep_rows": _keep_rows,
    "split_rows": _split_rows,
    "unpivot": _unpivot,
    "group_by": _group_by,
}


# --- Chain -----------------------------------------------------------------------------------


def source_sql(parquet_path: str) -> str:
    return (
        f"SELECT * EXCLUDE (file_row_number), file_row_number AS {ident(ROW)} "
        f"FROM read_parquet({lit(parquet_path)}, file_row_number = true)"
    )


def _describe(connection: duckdb.DuckDBPyConnection, sql: str) -> list[tuple[str, str]]:
    rows = connection.execute(f"DESCRIBE {sql}").fetchall()
    return [(str(row[0]), str(row[1])) for row in rows if row[0] != ROW]


def compile_steps(
    connection: duckdb.DuckDBPyConnection,
    parquet_path: str,
    steps: list[Any],
    strict: bool = False,
    sources: Mapping[str, str] | None = None,
) -> Compiled:
    """The CTE chain of the enabled steps. `strict`: a step that cannot run is an error.
    `sources`: the Parquet of the other datasets the steps name (C-57, C-58), by their id."""
    ctes = [f"t0 AS ({source_sql(parquet_path)})"]
    stages = [Stage("t0", _describe(connection, f"WITH {ctes[0]} SELECT * FROM t0"))]
    compiled = Compiled(parquet_path, ctes, stages)
    context = Context(connection, ctes, sources or {})
    reshaped_by: str | None = None
    for step in steps:
        if not step.enabled:
            continue
        prev = stages[-1]
        name = f"t{len(stages)}"
        try:
            if step.op == "extract_text" and step.how == "regex":
                check_pattern(connection, step.pattern or "")
            if reshaped_by and (m.names_rows(step) or m.reviewed(step)):
                raise m.CleanError(
                    f"Step {reshaped_by} reshaped the rows: row numbers no longer name the "
                    "file's rows (C-59)."
                )
            if step.op in CONTEXT_BUILDERS:
                body = CONTEXT_BUILDERS[step.op](step, prev, context)
            else:
                body = BUILDERS[step.op](step, prev)
            cte = f"{name} AS ({body.sql})"
            columns = _describe(connection, f"WITH {', '.join([*ctes, cte])} SELECT * FROM {name}")
            if step.op in FORMULA_OPS:  # a formula fails on the values, not on the types: probe
                connection.execute(
                    f"WITH {', '.join([*ctes, cte])} SELECT * FROM {name} LIMIT {PROBE_ROWS}"
                ).fetchall()
        except (m.CleanError, duckdb.Error) as error:
            message = error.message if isinstance(error, m.CleanError) else _short(error)
            if strict:
                raise m.CleanError(f"Step {step.id} ({step.op}): {message}") from None
            compiled.problems[step.id] = message
            continue
        ctes.append(cte)
        stages.append(Stage(name, columns, step, body.origin, body.removes_rows, body.reshapes))
        if body.reshapes and reshaped_by is None:
            reshaped_by = step.id
    return compiled


def _short(error: duckdb.Error) -> str:
    text = str(error).strip().splitlines()[0]
    return text[:300]
