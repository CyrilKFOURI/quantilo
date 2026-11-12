"""Steps → a chain of DuckDB CTEs on the parent's Parquet (the parent is only read).

`t0` reads the Parquet with its row number (`__quantilo_row`, the order kept to the end, C-08);
each enabled step is one CTE selecting from the previous one: an explicit column list (a value
replaced, renamed, dropped, added), a `WHERE`, or a `QUALIFY`. Each stage's columns and types come
from DuckDB itself (`DESCRIBE`), so a step sees the types the previous steps left.

A step that cannot run (a column that no longer exists…) is reported in `problems` and skipped,
so the draft stays editable; `apply` refuses a cleaning with problems (`strict`).
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import duckdb

from quantilo.cleaning import model as m
from quantilo.cleaning.text import SQL_INVISIBLE, SQL_SPACE_RUN
from quantilo.ingestion.sql import ident, lit
from quantilo.typing.conversion import default_boolean_parser
from quantilo.typing.knowledge_base import load as load_knowledge
from quantilo.typing.parsers import DATE_FORMATS, clean_text, number_sql

ROW = "__quantilo_row"
INTEGERS = ("TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT", "UTINYINT", "USMALLINT")
INTEGERS += ("UINTEGER", "UBIGINT", "UHUGEINT")
FLOATS = ("FLOAT", "DOUBLE", "REAL")


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

    def output_columns(self) -> str:
        return ", ".join(ident(name) for name in self.final.names)


# --- Step bodies -----------------------------------------------------------------------------


@dataclass
class Body:
    sql: str
    origin: dict[str, str | None]
    removes_rows: bool = False


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
    pattern = lit(SQL_SPACE_RUN)
    return _select(
        prev,
        {
            c: f"trim(regexp_replace({ident(c)}, {pattern}, ' ', 'g'), ' ')"
            for c in _texts(prev, step.columns)
        },
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
    expr = (
        f"CASE WHEN regexp_full_match({text}, '[0-9]+') AND length({text}) < {step.length} "
        f"THEN lpad({text}, {step.length}, '0') ELSE {text} END"
    )
    return _select(prev, {step.column: expr})


def _date_expr(text: str, formats: list[str], excel: bool) -> str:
    parts = [f"TRY_STRPTIME({text}, {lit(f)})" for f in formats or DATE_FORMATS]
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
            expr = (
                f"CASE WHEN {low} IN {literal_list(trues)} THEN true "
                f"WHEN {low} IN {literal_list(falses)} THEN false END"
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


def assembled_date_sql(step: m.AssembleDate, prev: Stage) -> str:
    """B-20: the date its parts make, NULL when they make none (out of range or impossible)."""
    year = _whole(step.year, prev.dtype(step.year))
    month = month_sql(step.month, prev.dtype(step.month))
    day = _whole(step.day, prev.dtype(step.day)) if step.day else "1"
    checks = [
        f"{year} BETWEEN 1000 AND 2200",
        f"{month} BETWEEN 1 AND 12",
        f"{day} BETWEEN 1 AND 31",
    ]
    if step.hour is None:
        text = f"printf('%04d-%02d-%02d', {year}, {month}, {day})"
        return f"CASE WHEN {' AND '.join(checks)} THEN TRY_CAST({text} AS DATE) END"
    hour = _whole(step.hour, prev.dtype(step.hour))
    minute = _whole(step.minute, prev.dtype(step.minute)) if step.minute else "0"
    checks += [f"{hour} BETWEEN 0 AND 23", f"{minute} BETWEEN 0 AND 59"]
    text = f"printf('%04d-%02d-%02d %02d:%02d:00', {year}, {month}, {day}, {hour}, {minute})"
    return f"CASE WHEN {' AND '.join(checks)} THEN TRY_CAST({text} AS TIMESTAMP) END"


def _assemble_date(step: m.AssembleDate, prev: Stage) -> Body:
    parts = [c for c in (step.year, step.month, step.day, step.hour, step.minute) if c]
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
    keys = step.keys or prev.names
    _require(prev, *keys)
    partition = ", ".join(ident(key) for key in keys)
    if step.keep == "first":
        order = f"{ident(ROW)}"
    elif step.keep == "last":
        order = f"{ident(ROW)} DESC"
    else:
        filled = " + ".join(f"CASE WHEN {ident(n)} IS NULL THEN 0 ELSE 1 END" for n in prev.names)
        order = f"({filled}) DESC, {ident(ROW)}"
    body = (
        f"SELECT * FROM {prev.name} "
        f"QUALIFY row_number() OVER (PARTITION BY {partition} ORDER BY {order}) = 1"
    )
    return Body(body, {name: name for name in prev.names}, removes_rows=True)


def _blank(column: str, dtype: str) -> str:
    return f"coalesce(trim({as_text(column, dtype)}), '') = ''"


def _drop_empty_rows(_step: m.DropEmptyRows, prev: Stage) -> Body:
    every = " AND ".join(_blank(name, dtype) for name, dtype in prev.columns)
    return _where(prev, f"NOT ({every})")


def _drop_header_rows(_step: m.DropHeaderRows, prev: Stage) -> Body:
    if len(prev.columns) < 2:
        raise m.CleanError("A repeated header needs at least two columns.")
    every = " AND ".join(
        f"coalesce(lower(trim({as_text(name, dtype)})) = {lit(name.strip().lower())}, false)"
        for name, dtype in prev.columns
    )
    return _where(prev, f"NOT ({every})")


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
    "fill": _fill,
    "fill_empty": _fill_empty,
    "null_values": _null_values,
    "null_outside": _null_outside,
    "pad_zeros": _pad_zeros,
    "filter_rows": _filter_rows,
    "drop_duplicates": _drop_duplicates,
    "drop_empty_rows": _drop_empty_rows,
    "drop_header_rows": _drop_header_rows,
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
) -> Compiled:
    """The CTE chain of the enabled steps. `strict`: a step that cannot run is an error."""
    ctes = [f"t0 AS ({source_sql(parquet_path)})"]
    stages = [Stage("t0", _describe(connection, f"WITH {ctes[0]} SELECT * FROM t0"))]
    compiled = Compiled(parquet_path, ctes, stages)
    for step in steps:
        if not step.enabled:
            continue
        prev = stages[-1]
        name = f"t{len(stages)}"
        try:
            body = BUILDERS[step.op](step, prev)
            cte = f"{name} AS ({body.sql})"
            columns = _describe(connection, f"WITH {', '.join([*ctes, cte])} SELECT * FROM {name}")
        except (m.CleanError, duckdb.Error) as error:
            message = error.message if isinstance(error, m.CleanError) else _short(error)
            if strict:
                raise m.CleanError(f"Step {step.id} ({step.op}): {message}") from None
            compiled.problems[step.id] = message
            continue
        ctes.append(cte)
        stages.append(Stage(name, columns, step, body.origin, body.removes_rows))
    return compiled


def _short(error: duckdb.Error) -> str:
    text = str(error).strip().splitlines()[0]
    return text[:300]
