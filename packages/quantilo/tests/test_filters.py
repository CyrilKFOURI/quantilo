import duckdb
import pytest

from quantilo.filters.model import Condition, InvalidFilterError, parse_filters
from quantilo.filters.sql import FilterColumn, where_sql
from quantilo.typing.types import Parser

RAW = Parser(kind="raw")
COLUMNS = {
    "price": FilterColumn("price", "numeric_continuous", Parser(kind="number", decimal=",")),
    "kids": FilterColumn("kids", "numeric_discrete", RAW),
    "country": FilterColumn("country", "categorical_nominal", RAW),
    "active": FilterColumn("active", "boolean", RAW),
    "comment": FilterColumn("comment", "text", RAW),
    "day": FilterColumn("day", "datetime", RAW),
    "junk": FilterColumn("junk", "ignored", RAW),
}


@pytest.fixture
def connection() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(
        """
        CREATE TABLE t AS SELECT * FROM (VALUES
            (1, '79,50', 0, 'FR', true, 'Great value', TIMESTAMP '2024-01-01 10:00'),
            (2, '141,62', 2, 'BE', false, 'Too long 100%', TIMESTAMP '2024-01-02 09:00'),
            (3, '38,64', 1, 'FR', NULL, NULL, TIMESTAMP '2024-01-08 12:00'),
            (4, NULL, 3, 'ES', true, 'it''s fine', NULL),
            (5, '250,00', 2, NULL, false, 'great_deal', TIMESTAMP '2024-02-05 18:30')
        ) AS v(id, price, kids, country, active, comment, day)
        """
    )
    return con


def ids(con: duckdb.DuckDBPyConnection, *conditions: Condition) -> list[int]:
    where = where_sql(list(conditions), COLUMNS)
    rows = con.execute(f"SELECT id FROM t WHERE {where} ORDER BY id").fetchall()
    return [row[0] for row in rows]


def c(column: str, op: str, **kwargs) -> Condition:
    return Condition.model_validate({"column": column, "op": op, **kwargs})


def test_numbers_compare_typed_values(connection: duckdb.DuckDBPyConnection) -> None:
    # "79,50" is read through the column's parser: 79.5 > 50, compared as numbers.
    assert ids(connection, c("price", "gt", value=100)) == [2, 5]
    assert ids(connection, c("price", "between", low=50, high=150)) == [1, 2]


def test_negation_excludes_empty_values(connection: duckdb.DuckDBPyConnection) -> None:
    outside = c("price", "between", low=50, high=150, negate=True)

    assert ids(connection, outside) == [3, 5]  # row 4 has no price: never kept


def test_conditions_combine_with_and_values_with_or(connection: duckdb.DuckDBPyConnection) -> None:
    countries = c("country", "in", values=["FR", "BE"])

    assert ids(connection, countries) == [1, 2, 3]
    assert ids(connection, countries, c("kids", "gte", value=1)) == [2, 3]
    assert ids(connection, c("country", "in", values=["FR"], negate=True)) == [2, 4]


def test_booleans_and_discrete_values(connection: duckdb.DuckDBPyConnection) -> None:
    assert ids(connection, c("active", "in", values=["True"])) == [1, 4]
    # however a boolean is written, it means the same rows
    for written in (["true"], ["1"], ["yes"], [True]):
        assert ids(connection, c("active", "in", values=written)) == [1, 4]
    assert ids(connection, c("active", "in", values=[False])) == ids(
        connection, c("active", "in", values=["False"])
    )
    assert ids(connection, c("kids", "in", values=["2", "3"])) == [2, 4, 5]


def test_empty_and_not_empty(connection: duckdb.DuckDBPyConnection) -> None:
    assert ids(connection, c("country", "is_null")) == [5]
    assert ids(connection, c("country", "is_null", negate=True)) == [1, 2, 3, 4]


def test_text_search_escapes_wildcards(connection: duckdb.DuckDBPyConnection) -> None:
    assert ids(connection, c("comment", "contains", value="GREAT")) == [1, 5]
    assert ids(connection, c("comment", "contains", value="_")) == [5]  # literal underscore
    assert ids(connection, c("comment", "contains", value="100%")) == [2]
    assert ids(connection, c("comment", "starts_with", value="it's")) == [4]


def test_dates_and_date_parts(connection: duckdb.DuckDBPyConnection) -> None:
    assert ids(connection, c("day", "lt", value="2024-01-05")) == [1, 2]
    assert ids(connection, c("day", "part_in", part="day_of_week", values=["1"])) == [1, 3, 5]
    assert ids(connection, c("day", "part_in", part="month", values=["2"])) == [5]
    assert ids(connection, c("day", "part_in", part="hour", values=["9", "10"])) == [1, 2]


def test_values_cannot_inject_sql(connection: duckdb.DuckDBPyConnection) -> None:
    sneaky = c("country", "in", values=["FR') OR 1=1 --"])

    assert ids(connection, sneaky) == []
    with pytest.raises(InvalidFilterError):
        ids(connection, c("price", "gt", value="1; DROP TABLE t"))


@pytest.mark.parametrize(
    ("condition", "message"),
    [
        (c("nope", "eq", value=1), "Unknown column"),
        (c("country", "gt", value=1), "can't use"),
        (c("junk", "is_null"), "ignored"),
        (c("country", "in", values=[]), "at least one value"),
        (c("day", "eq", value="13/45/2024"), "ISO date"),
        (c("price", "gt", value="inf"), "finite"),
    ],
)
def test_invalid_conditions_are_explained(condition: Condition, message: str) -> None:
    with pytest.raises(InvalidFilterError, match=message):
        where_sql([condition], COLUMNS)


def test_filters_parameter() -> None:
    assert parse_filters(None) == []
    assert parse_filters('[{"column": "a", "op": "is_null"}]')[0].column == "a"
    with pytest.raises(InvalidFilterError):
        parse_filters("not json")
    with pytest.raises(InvalidFilterError):
        parse_filters('[{"column": "a", "op": "drop"}]')


FLEET = {
    "vehicle": FilterColumn("vehicle", "identifier", RAW),
    "month": FilterColumn("month", "categorical_ordinal", RAW),
    "country": FilterColumn("country", "categorical_nominal", RAW),
    "seen": FilterColumn("seen", "datetime", Parser(kind="date", format="%d/%m/%Y")),
    "junk": FilterColumn("junk", "ignored", RAW),
}


@pytest.fixture
def fleet() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(
        """
        CREATE TABLE f AS SELECT * FROM (VALUES
            (0, 'V1', 'jan', 'FR', '03/01/2024', 'x'),
            (1, 'V2', 'jan', 'BE', '01/01/2024', 'x'),
            (2, 'V1', 'feb', 'BE', '20/02/2024', 'x'),
            (3, 'V1', 'feb', 'FR', '10/02/2024', 'x'),
            (4, 'V3', 'jan', 'FR', NULL, 'x'),
            (5, 'V2', 'jan', 'BE', '05/01/2024', 'x')
        ) AS v(file_row_number, vehicle, month, country, seen, junk)
        """
    )
    return con


def rows_of(con: duckdb.DuckDBPyConnection, *conditions: Condition) -> list[int]:
    from quantilo.filters.sql import filtered_source, row_steps

    relation = filtered_source("f", where_sql(list(conditions), FLEET))
    for step in row_steps(list(conditions), FLEET):
        relation = step.apply(relation)
    rows = con.execute(f"SELECT file_row_number FROM {relation} ORDER BY 1").fetchall()
    return [row[0] for row in rows]


def test_unique_keeps_first_or_last_row_per_value(fleet: duckdb.DuckDBPyConnection) -> None:
    assert rows_of(fleet, c("vehicle", "unique")) == [0, 1, 4]
    assert rows_of(fleet, c("vehicle", "unique", keep="last")) == [3, 4, 5]


def test_unique_ordered_by_a_column(fleet: duckdb.DuckDBPyConnection) -> None:
    latest = rows_of(fleet, c("vehicle", "unique", keep="last", order_by="seen"))
    earliest = rows_of(fleet, c("vehicle", "unique", keep="first", order_by="seen"))

    assert latest == [2, 4, 5]  # V1 on 20/02 (typed dates, not text), V2 on 05/01
    assert earliest == [0, 1, 4]


def test_unique_on_a_combination_of_columns(fleet: duckdb.DuckDBPyConnection) -> None:
    assert rows_of(fleet, c("vehicle", "unique", columns=["month"])) == [0, 1, 2, 4]


def test_duplicates_and_their_negation(fleet: duckdb.DuckDBPyConnection) -> None:
    assert rows_of(fleet, c("vehicle", "duplicated")) == [0, 1, 2, 3, 5]
    assert rows_of(fleet, c("vehicle", "duplicated", negate=True)) == [4]
    assert rows_of(fleet, c("vehicle", "duplicated", columns=["month"])) == [1, 2, 3, 5]


def test_value_filters_apply_first(fleet: duckdb.DuckDBPyConnection) -> None:
    # The last FR row of each vehicle, not the vehicles whose last row is FR.
    found = rows_of(fleet, c("vehicle", "unique", keep="last"), c("country", "in", values=["FR"]))

    assert found == [3, 4]


def test_row_conditions_are_validated() -> None:
    from quantilo.filters.sql import row_steps

    with pytest.raises(InvalidFilterError, match="Unknown column"):
        row_steps([c("vehicle", "unique", columns=["nope"])], FLEET)
    with pytest.raises(InvalidFilterError, match="ignored"):
        row_steps([c("vehicle", "unique", order_by="junk")], FLEET)
    with pytest.raises(InvalidFilterError, match="negated"):
        row_steps([c("vehicle", "unique", negate=True)], FLEET)
