"""Read the columns of a dataset into a pandas frame, typed, through the active filters."""

import math
from dataclasses import dataclass
from typing import cast

import duckdb
import pandas as pd

from quantilo.compare.analyze import label_sql, number_sql
from quantilo.typing.parsers import typed_sql
from quantilo.typing.types import AnalysisType, Parser
from quantilo.variable.source import SEED

MAX_ROWS = 200_000  # training rows by default (a reproducible sample beyond)
NUMERIC = {"numeric_continuous", "numeric_discrete"}
CATEGORICAL = {"categorical_nominal", "categorical_ordinal", "boolean"}


@dataclass
class Field:
    name: str
    type: AnalysisType
    parser: Parser
    role: str | None = None
    order: list[str] | None = None


@dataclass
class Frame:
    data: pd.DataFrame  # one column per field, plus "__row" (row number in the file)
    rows: int  # rows after the filters with a target value
    sampled: bool


def _expression(field: Field) -> str:
    if field.type in NUMERIC:
        return number_sql(field)  # type: ignore[arg-type]
    if field.type == "datetime":
        return f"CAST({typed_sql(field.name, field.parser)} AS TIMESTAMP)"
    if field.type in CATEGORICAL:
        return label_sql(field)  # type: ignore[arg-type]
    return f"CAST({typed_sql(field.name, field.parser)} AS VARCHAR)"


def load(
    connection: duckdb.DuckDBPyConnection,
    source: str,
    fields: list[Field],
    target: str | None,
    limit: int | None = MAX_ROWS,
    period: str | None = None,
) -> Frame:
    """Rows with a target value (every row without a target, 032); at most `limit` of them,
    drawn reproducibly. With a `period` (panel data, 035): each row also gets its period's rank
    among every period (`__period`), and a sample keeps whole periods, one in k counting back
    from the most recent, never rows drawn at random inside a period."""
    aliases = {field.name: f"c{i}" for i, field in enumerate(fields)}
    selected = ", ".join(f"{_expression(f)} AS {aliases[f.name]}" for f in fields)
    where = f"WHERE {aliases[target]} IS NOT NULL" if target is not None else ""
    relation = (
        f"(SELECT * FROM (SELECT {selected}, file_row_number AS __row FROM {source}) {where})"
    )
    names = [*(f.name for f in fields), "__row"]
    if period is not None:
        key = aliases[period]
        ranked = (
            f"(SELECT {key} AS __key, row_number() OVER (ORDER BY "
            f"TRY_CAST(CAST({key} AS VARCHAR) AS DOUBLE) NULLS LAST, {key}) - 1 AS __period "
            f"FROM (SELECT DISTINCT {key} FROM {relation} WHERE {key} IS NOT NULL))"
        )
        relation = (
            f"(SELECT r.*, p.__period FROM {relation} r LEFT JOIN {ranked} p ON r.{key} = p.__key)"
        )
        names.append("__period")
    (rows,) = connection.execute(f"SELECT count(*) FROM {relation}").fetchone() or (0,)
    sampled = limit is not None and rows > limit
    order = f"ORDER BY hash(__row + {SEED}) LIMIT {limit}" if sampled else "ORDER BY __row"
    if sampled and period is not None:
        (last,) = connection.execute(f"SELECT max(__period) FROM {relation}").fetchone() or (0,)
        step = math.ceil(rows / cast(int, limit))
        order = f"WHERE __period % {step} = {int(last or 0) % step} ORDER BY __row"
    data = connection.execute(f"SELECT * FROM {relation} {order}").df()
    data.columns = names
    for field in fields:
        if field.type in NUMERIC:
            data[field.name] = pd.Series(
                pd.to_numeric(data[field.name], errors="coerce"), index=data.index, dtype=float
            )
    return Frame(data, int(rows), bool(sampled))
