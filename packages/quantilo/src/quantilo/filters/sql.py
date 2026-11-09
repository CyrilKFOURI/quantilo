"""Conditions → SQL over each column's typed value (its parser), every value escaped or cast.

Comparisons follow SQL: a negated condition never keeps empty values, except `is_null` negated
(= "is not empty").
"""

import math
from datetime import datetime

from quantilo.filters.model import ROW_OPERATORS, Condition, InvalidFilterError
from quantilo.ingestion.sql import lit
from quantilo.typing.dates import part_sql
from quantilo.typing.parsers import typed_sql
from quantilo.typing.types import AnalysisType, Parser

NUMERIC = ("numeric_continuous", "numeric_discrete")
CATEGORICAL = ("categorical_nominal", "categorical_ordinal")
TEXTUAL = ("text", "identifier", *CATEGORICAL)

ALLOWED: dict[str, tuple[str, ...]] = {
    "numeric_continuous": ("eq", "gt", "gte", "lt", "lte", "between", "is_null"),
    "numeric_discrete": ("eq", "gt", "gte", "lt", "lte", "between", "in", "is_null"),
    "categorical_nominal": ("in", "eq", "contains", "starts_with", "is_null"),
    "categorical_ordinal": ("in", "eq", "contains", "starts_with", "is_null"),
    "boolean": ("in", "is_null"),
    "text": ("eq", "contains", "starts_with", "in", "is_null"),
    "identifier": ("eq", "contains", "starts_with", "in", "is_null"),
    "datetime": ("eq", "gt", "gte", "lt", "lte", "between", "part_in", "is_null"),
}
COMPARISONS = {"eq": "=", "gt": ">", "gte": ">=", "lt": "<", "lte": "<="}


class FilterColumn:
    """What a condition needs to know about its column."""

    def __init__(self, name: str, analysis_type: AnalysisType, parser: Parser) -> None:
        self.name = name
        self.type = analysis_type
        self.parser = parser


def _number(value: object, column: str) -> str:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise InvalidFilterError(f"{column!r}: {value!r} is not a number.") from None
    if not math.isfinite(number):
        raise InvalidFilterError(f"{column!r}: {value!r} is not a finite number.")
    return repr(number)


def _integer(value: object, column: str) -> str:
    try:
        return str(int(str(value)))
    except ValueError:
        raise InvalidFilterError(f"{column!r}: {value!r} is not a whole number.") from None


def _timestamp(value: object, column: str) -> str:
    try:
        moment = datetime.fromisoformat(str(value))
    except ValueError:
        raise InvalidFilterError(f"{column!r}: {value!r} is not an ISO date.") from None
    return f"TIMESTAMP {lit(moment.isoformat(sep=' '))}"


def _text(value: object, column: str) -> str:
    if value is None:
        raise InvalidFilterError(f"{column!r}: a value is required.")
    return lit(str(value))


def _like(value: object, column: str) -> str:
    """Escaped LIKE pattern body (no wildcards from the user)."""
    text = str(value if value is not None else "")
    if not text:
        raise InvalidFilterError(f"{column!r}: a value is required.")
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _label(typed: str, analysis_type: str) -> str:
    """The value as the charts label it (value_label): what `in` compares against."""
    if analysis_type == "boolean":
        return f"(CASE WHEN {typed} THEN 'True' WHEN NOT {typed} THEN 'False' END)"
    return f"CAST({typed} AS VARCHAR)"


TRUE_WORDS = {"true", "1", "yes", "y", "oui", "vrai"}
FALSE_WORDS = {"false", "0", "no", "n", "non", "faux"}


def _boolean_label(value: str) -> str:
    """ "True" or "False" as the charts label them, however the value is written."""
    word = value.strip().lower()
    if word in TRUE_WORDS:
        return "True"
    if word in FALSE_WORDS:
        return "False"
    return value


def condition_sql(condition: Condition, column: FilterColumn) -> str:
    kind = column.type
    if kind == "ignored":
        raise InvalidFilterError(f"{column.name!r} is ignored: it can't be filtered.")
    if condition.op not in ALLOWED[kind]:
        hint = ' with values "True" / "False"' if kind == "boolean" else ""
        raise InvalidFilterError(
            f"{column.name!r} ({kind}) can't use {condition.op!r}; "
            f"use {', '.join(ALLOWED[kind])}{hint}."
        )
    typed = typed_sql(column.name, column.parser)
    if condition.op == "is_null":
        return f"({typed}) IS NOT NULL" if condition.negate else f"({typed}) IS NULL"

    if kind in NUMERIC:
        value_of = _number
        subject = f"CAST({typed} AS DOUBLE)"
    elif kind == "datetime":
        value_of = _timestamp
        subject = f"CAST({typed} AS TIMESTAMP)"
    else:
        value_of = _text
        subject = f"CAST({typed} AS VARCHAR)"

    op = condition.op
    if op in COMPARISONS:
        sql = f"{subject} {COMPARISONS[op]} {value_of(condition.value, column.name)}"
    elif op == "between":
        low, high = (value_of(v, column.name) for v in (condition.low, condition.high))
        sql = f"{subject} BETWEEN {low} AND {high}"
    elif op in ("contains", "starts_with"):
        body = _like(condition.value, column.name)
        pattern = f"%{body}%" if op == "contains" else f"{body}%"
        sql = f"lower({subject}) LIKE lower({lit(pattern)}) ESCAPE '\\'"
    elif op == "in":
        if not condition.values:
            raise InvalidFilterError(f"{column.name!r}: pick at least one value.")
        if kind == "numeric_discrete":
            listed = ", ".join(_number(v, column.name) for v in condition.values)
            sql = f"{subject} IN ({listed})"
        else:
            values = condition.values
            if kind == "boolean":
                values = [_boolean_label(v) for v in values]
            listed = ", ".join(lit(v) for v in values)
            sql = f"{_label(typed, kind)} IN ({listed})"
    else:  # part_in
        if condition.part is None or not condition.values:
            raise InvalidFilterError(f"{column.name!r}: pick a date part and at least one value.")
        listed = ", ".join(
            lit(v) if condition.part == "year_month" else _integer(v, column.name)
            for v in condition.values
        )
        sql = f"{part_sql(typed, condition.part)} IN ({listed})"
    # Negation keeps SQL's three-valued logic explicit: empty values are never kept.
    return f"(({typed}) IS NOT NULL AND NOT ({sql}))" if condition.negate else f"({sql})"


def where_sql(conditions: list[Condition], columns: dict[str, FilterColumn]) -> str:
    """AND of every value condition; "" when there is none (row conditions: see RowStep)."""
    parts = []
    for condition in conditions:
        if condition.op in ROW_OPERATORS:
            continue
        parts.append(condition_sql(condition, _known(condition.column, columns)))
    return " AND ".join(parts)


def _known(name: str, columns: dict[str, FilterColumn]) -> FilterColumn:
    column = columns.get(name)
    if column is None:
        raise InvalidFilterError(f"Unknown column {name!r}.")
    if column.type == "ignored":
        raise InvalidFilterError(f"{name!r} is ignored: it can't be filtered.")
    return column


ROW = "file_row_number"


class RowStep:
    """Unique values or Duplicates only: a window over the rows already filtered by value.

    Values compare in their analysis type (parser applied); empty values form one group.
    """

    def __init__(self, condition: Condition, columns: dict[str, FilterColumn]) -> None:
        names = list(dict.fromkeys([condition.column, *condition.columns]))
        keys = [_known(name, columns) for name in names]
        self.partition = ", ".join(typed_sql(k.name, k.parser) for k in keys)
        self.op = condition.op
        self.negate = condition.negate
        if condition.op == "unique" and condition.negate:
            raise InvalidFilterError("Unique values can't be negated: use Duplicates only.")
        direction = "DESC" if condition.keep == "last" else "ASC"
        self.order = f"{ROW} {direction}"
        if condition.order_by:
            by = _known(condition.order_by, columns)
            self.order = f"{typed_sql(by.name, by.parser)} {direction} NULLS LAST, {self.order}"

    def apply(self, relation: str) -> str:
        """`relation` must carry `file_row_number`."""
        if self.op == "unique":
            return (
                f"(SELECT * EXCLUDE (__de_rank) FROM (SELECT *, row_number() OVER "
                f"(PARTITION BY {self.partition} ORDER BY {self.order}) AS __de_rank "
                f"FROM {relation}) WHERE __de_rank = 1)"
            )
        test = "= 1" if self.negate else "> 1"
        return (
            f"(SELECT * EXCLUDE (__de_copies) FROM (SELECT *, count(*) OVER "
            f"(PARTITION BY {self.partition}) AS __de_copies FROM {relation}) "
            f"WHERE __de_copies {test})"
        )


def row_steps(conditions: list[Condition], columns: dict[str, FilterColumn]) -> list[RowStep]:
    return [RowStep(c, columns) for c in conditions if c.op in ROW_OPERATORS]


def filtered_source(base: str, where: str) -> str:
    """A relation of the base rows that pass the filter (the base itself without filter)."""
    return f"(SELECT * FROM {base} WHERE {where})" if where else base
