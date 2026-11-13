"""N18 and N19 (audit 2026-11-12, § 4.3): French / EU amounts and the dates the parsers missed.

Amounts: no-break and thin spaces, an apostrophe as thousands separator, a currency sign or ISO code
before or after, accounting negatives in brackets and a trailing minus, the U+2212 minus; a dot or a
comma that could be either separator is decided on the whole column, and the reason says on what.
Dates: milliseconds, AM/PM, ISO with a final Z, 12-Jan-2024, Jan 12, 2024, French month names.
"""

import csv
from pathlib import Path
from typing import Any

import duckdb
import pytest

import quantilo as dx
from quantilo.cleaning import compile_steps, model
from quantilo.cleaning.rules import basic, date_time
from quantilo.cleaning.rules.shapes import _reads
from quantilo.typing import date_parts
from quantilo.typing.engine import detect_columns
from quantilo.typing.parsers import (
    ALL_DATE_FORMATS,
    currency_sql,
    english_months,
    number_sql,
    strptime_sql,
    typed_sql,
)
from quantilo.typing.types import ColumnDetection, Parser


def values_of(expression_of, values: list[str]) -> list[Any]:
    connection = duckdb.connect()
    connection.execute("CREATE TABLE t (x VARCHAR)")
    connection.executemany("INSERT INTO t VALUES (?)", [[v] for v in values])
    return [row[0] for row in connection.execute(f"SELECT {expression_of('x')} FROM t").fetchall()]


def detect_values(values: list[str], name: str = "x") -> ColumnDetection:
    connection = duckdb.connect()
    connection.execute(f'CREATE TABLE t ("{name}" VARCHAR)')
    connection.executemany("INSERT INTO t VALUES (?)", [[v] for v in values])
    return detect_columns(connection, "t")[0]


def write(tmp_path: Path, rows: list[list[Any]], name: str = "data") -> dx.Dataset:
    path = tmp_path / f"{name}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle, delimiter=";").writerows(rows)
    return dx.open(path, cache_dir=tmp_path / f"cache-{name}")


# --- N18: amounts --------------------------------------------------------------------------------

COMMA_AMOUNTS = {
    "1 234,56 €": 1234.56,
    "1\u202f234,56\u00a0€": 1234.56,  # thin no-break space as thousands, no-break space before €
    "(1 234,56)": -1234.56,  # accounting negative
    "(1 234,56 €)": -1234.56,
    "1 234,56-": -1234.56,  # trailing minus (SAP exports)
    "\u221212,5": -12.5,  # the minus sign U+2212
    "-12,5 €": -12.5,
    "EUR 12,50": 12.5,
    "12,50 CHF": 12.5,
    "1.234.567,8": 1234567.8,
    "12,5\u00a0%": 12.5,
    "0,500": 0.5,
    "abc": None,
    "12-20": None,
    "--12": None,
    "(12": None,
    "1 23,5": None,
}

DOT_AMOUNTS = {
    "1'234.50": 1234.5,  # Swiss apostrophe
    "1\u2019234.50": 1234.5,
    "USD 1,234.50": 1234.5,
    "(1,234.50)": -1234.5,
    "$-12.5": -12.5,
    "-€12.5": -12.5,
    "1 234.5": 1234.5,
    "1234-": -1234.0,
    "1.2e-3": 0.0012,
    "SEK 12.5": 12.5,  # a code holding an E is no exponent
    "0,500": None,  # a first group of 0 is no thousands grouping
    "12 kg": None,
}


class TestAmounts:
    @pytest.mark.parametrize(("decimal", "expected"), [(",", COMMA_AMOUNTS), (".", DOT_AMOUNTS)])
    def test_number_sql_reads_business_formats(self, decimal: str, expected: dict) -> None:
        read = values_of(lambda x: number_sql(x, decimal), list(expected))

        assert dict(zip(expected, read, strict=True)) == pytest.approx(expected)

    def test_currency_is_extracted(self) -> None:
        read = values_of(currency_sql, ["1 234,56 €", "EUR 12,50", "$12", "12,5", "(12 CHF)"])

        assert read == ["€", "EUR", "$", None, "CHF"]

    def test_one_currency_whatever_its_sign(self) -> None:
        read = values_of(lambda x: currency_sql(x, code=True), ["12 €", "EUR 12", "12 eur", "$1"])

        assert read == ["EUR", "EUR", "EUR", "$"]

    def test_french_accounting_column_is_numeric(self) -> None:
        values = ["1 234,56 €"] * 60 + ["12,50"] * 20 + ["(1 234,56)"] * 10 + ["EUR 12,50"] * 10
        detection = detect_values(values, "montant_ttc")

        assert detection.type == "numeric_continuous"
        assert detection.parser == Parser(kind="number", decimal=",")
        assert detection.role == "amount"

    def test_plain_numbers_keep_their_reasons(self) -> None:
        detection = detect_values([f"{i}.5" for i in range(40)])

        assert detection.parser.decimal == "."
        assert not any("separator" in reason for reason in detection.reasons)

    def test_comma_only_values_decide(self) -> None:
        # '1.234' alone reads both ways; '12,5' only with a decimal comma: that is the evidence
        detection = detect_values(["1.234", "2.500", "12,5", "7,25"] * 10)

        assert detection.parser.decimal == ","
        assert any("read only this way" in reason for reason in detection.reasons)

    def test_dot_groups_with_small_plain_numbers_are_thousands(self) -> None:
        # every dotted value has groups of three digits and the values without one are below 1,000
        detection = detect_values(["1.234", "12.500", "3.750", "250", "975", "40"] * 10)

        assert detection.parser.decimal == ","
        assert any("thousands" in reason for reason in detection.reasons)
        assert detection.confidence == "medium"

    def test_dot_on_every_value_stays_a_decimal_point(self) -> None:
        # every value has the dot and three digits after it: a fixed precision, not thousands
        detection = detect_values(["1.234", "2.500", "3.125", "12.750"] * 10)

        assert detection.parser.decimal == "."
        assert any("ambiguous" in reason or "every value" in reason for reason in detection.reasons)
        assert detection.confidence == "medium"

    def test_large_plain_numbers_rule_out_thousands(self) -> None:
        # '1500' without a separator: a writer grouping thousands would write '1.500'
        detection = detect_values(["1.234", "2.500", "1500", "2750"] * 10)

        assert detection.parser.decimal == "."

    def test_comma_groups_with_small_plain_numbers_are_thousands(self) -> None:
        detection = detect_values(["1,234", "12,500", "250", "975"] * 10)

        assert detection.parser.decimal == "."
        assert any("thousands" in reason for reason in detection.reasons)

    def test_b07_stores_the_french_amounts(self, tmp_path: Path) -> None:
        rows = [["n", "montant_ttc"]] + [
            [str(i), v]
            for i, v in enumerate(["1 234,56 €", "12,50", "(1 234,56)", "EUR 12,50"] * 25)
        ]
        ds = write(tmp_path, rows)
        found = next(s for s in ds.clean().suggestions if s.rule == "B-07")
        connection = duckdb.connect()
        compiled = compile_steps(connection, str(ds.path), [found.step], strict=True)
        read = connection.execute(
            f"SELECT montant_ttc FROM ({compiled.sql()}) ORDER BY __quantilo_row LIMIT 4"
        ).fetchall()

        assert found.mode == "A"
        assert [r[0] for r in read] == [1234.56, 12.5, -1234.56, 12.5]


# --- N19: dates ----------------------------------------------------------------------------------


class TestDates:
    @pytest.mark.parametrize(
        ("value", "date_format", "expected"),
        [
            ("1998-04-10 00:00:00.000", "%Y-%m-%d %H:%M:%S.%f", "1998-04-10 00:00:00"),
            ("2024-01-05T10:00:00.250", "%Y-%m-%dT%H:%M:%S.%f", "2024-01-05 10:00:00.25"),
            ("2024-01-05T10:00:00Z", "%Y-%m-%dT%H:%M:%SZ", "2024-01-05 10:00:00"),
            ("2024-01-05T10:00:00.5Z", "%Y-%m-%dT%H:%M:%S.%fZ", "2024-01-05 10:00:00.5"),
            ("01/05/2024 12:00:00 AM", "%m/%d/%Y %I:%M:%S %p", "2024-01-05 00:00:00"),
            ("01/05/2024 01:30 PM", "%m/%d/%Y %I:%M %p", "2024-01-05 13:30:00"),
            ("12-Jan-2024", "%d-%b-%Y", "2024-01-12 00:00:00"),
            ("Jan 12, 2024", "%b %d, %Y", "2024-01-12 00:00:00"),
            ("12 janv. 2024", "%d %b %Y", "2024-01-12 00:00:00"),
            ("3 février 2024", "%d %B %Y", "2024-02-03 00:00:00"),
            ("1er mars 2024", "%d %B %Y", "2024-03-01 00:00:00"),
            ("15 AOÛT 2023", "%d %B %Y", "2023-08-15 00:00:00"),
            ("7 déc. 2023", "%d %b %Y", "2023-12-07 00:00:00"),
            ("7 Sept. 2023", "%d %b %Y", "2023-09-07 00:00:00"),
            ("12 mars", "%d %B %Y", None),
        ],
    )
    def test_formats_read(self, value: str, date_format: str, expected: str | None) -> None:
        assert date_format in ALL_DATE_FORMATS
        (read,) = values_of(lambda x: f"CAST({strptime_sql(x, date_format)} AS VARCHAR)", [value])

        assert read == expected

    def test_english_months_in_python(self) -> None:
        assert english_months("12 janv. 2024") == "12 Jan 2024"
        assert english_months("1er décembre 2023") == "1 Dec 2023"
        assert _reads("12 janv. 2024", ["%d %B %Y"]) is not None

    def test_milliseconds_are_dates(self) -> None:
        values = [f"1998-04-{d:02d} 00:00:00.000" for d in range(1, 29)] * 3 + ["NULL"] * 4
        detection = detect_values(values, "shippedDate")

        assert detection.type == "datetime"
        assert detection.parser.format == "%Y-%m-%d %H:%M:%S.%f"

    def test_us_times_with_am_pm(self) -> None:
        values = [
            f"01/{d:02d}/2024 {h:02d}:15:00 {p}"
            for d in range(13, 29)
            for h, p in ((12, "AM"), (3, "PM"))
        ]
        detection = detect_values(values, "created")
        typed = values_of(lambda x: typed_sql(x, detection.parser), values[:2])

        assert detection.type == "datetime"
        assert detection.parser.format == "%m/%d/%Y %I:%M:%S %p"
        assert [str(v) for v in typed] == ["2024-01-13 00:15:00", "2024-01-13 15:15:00"]

    def test_french_month_names_are_dates(self) -> None:
        months = [
            "janv.",
            "févr.",
            "mars",
            "avr.",
            "mai",
            "juin",
            "juil.",
            "août",
            "sept.",
            "oct.",
            "nov.",
            "déc.",
        ]
        values = [f"{d} {m} 2024" for d in range(1, 15) for m in months]
        detection = detect_values(values, "date_facture")
        typed = values_of(
            lambda x: typed_sql(x, detection.parser), ["12 janv. 2024", "1 août 2024"]
        )

        assert detection.type == "datetime"
        assert [str(v) for v in typed] == ["2024-01-12", "2024-08-01"]

    @staticmethod
    def northwind_dates(tmp_path: Path) -> dx.Dataset:
        # Northwind's shippedDate: milliseconds, and 'NULL' for orders not shipped (2.5 %)
        rows = [["orderID", "shippedDate"]] + [
            [str(i), "NULL" if i % 40 == 0 else f"1998-04-{1 + i % 28:02d} 00:00:00.000"]
            for i in range(400)
        ]
        return write(tmp_path, rows)

    def test_b11_reads_milliseconds(self, tmp_path: Path) -> None:
        found = next(
            s for s in self.northwind_dates(tmp_path).clean().suggestions if s.rule == "B-11"
        )

        assert found.step.formats == ["%Y-%m-%d %H:%M:%S.%f"]  # type: ignore[union-attr]
        assert found.step.to == "datetime"  # type: ignore[union-attr]
        assert found.cells == 400  # 390 dates, and 10 'NULL' that become empty

    def test_b11_reads_milliseconds_automatically(self, tmp_path: Path) -> None:
        found = next(
            s for s in self.northwind_dates(tmp_path).clean().suggestions if s.rule == "B-11"
        )

        assert found.mode == "A"

    def test_change_type_reads_french_months(self, tmp_path: Path) -> None:
        ds = write(tmp_path, [["d"], ["12 janv. 2024"], ["3 février 2024"], ["1er mars 2024"]])
        step = model.parse_steps(
            [
                {
                    "id": "s1",
                    "op": "change_type",
                    "column": "d",
                    "to": "date",
                    "formats": ["%d %b %Y", "%d %B %Y"],
                }
            ]
        ).steps
        connection = duckdb.connect()
        compiled = compile_steps(connection, str(ds.path), step, strict=True)
        read = connection.execute(
            f"SELECT CAST(d AS VARCHAR) FROM ({compiled.sql()}) ORDER BY __quantilo_row"
        ).fetchall()

        assert [r[0] for r in read] == ["2024-01-12", "2024-02-03", "2024-03-01"]


# --- N19: a date column and a time column (B-20) ----------------------------------------------


class TestDateAndTime:
    @pytest.mark.parametrize(
        ("names", "expected"),
        [
            (["Date", "Time", "CO(GT)"], [("Date", "Time")]),
            (["crash_date", "crash_time", "borough"], [("crash_date", "crash_time")]),
            (["date_facture", "heure_facture"], [("date_facture", "heure_facture")]),
            (
                ["Date", "Time", "end_date", "end_time"],
                [("Date", "Time"), ("end_date", "end_time")],
            ),
            (["date", "amount"], []),
            (["start_date", "end_time"], []),
        ],
    )
    def test_pairs_by_name(self, names: list[str], expected: list[tuple[str, str]]) -> None:
        assert [(p.date, p.time) for p in date_parts.date_time_pairs(names)] == expected

    def test_assemble_a_date_and_a_time(self, tmp_path: Path) -> None:
        rows = [["Date", "Time"], ["2004-03-10", "18.00.00"], ["2004-03-10", "19:30"], ["", "1"]]
        ds = write(tmp_path, rows)
        step = model.parse_steps(
            [
                {
                    "id": "s1",
                    "op": "assemble_date",
                    "year": "Date",
                    "month": "Date",
                    "day": "Date",
                    "time": "Time",
                    "to": "when",
                }
            ]
        ).steps
        connection = duckdb.connect()
        compiled = compile_steps(connection, str(ds.path), step, strict=True)
        read = connection.execute(
            f'SELECT CAST("when" AS VARCHAR) FROM ({compiled.sql()}) ORDER BY __quantilo_row'
        ).fetchall()

        assert [r[0] for r in read] == ["2004-03-10 18:00:00", "2004-03-10 19:30:00", None]

    def test_b20_proposes_the_moment(self, tmp_path: Path) -> None:
        rows = [["Date", "Time", "CO(GT)"]] + [
            [f"2004-03-{10 + i // 24:02d}", f"{i % 24:02d}.00.00", "2,6"] for i in range(96)
        ]
        ds = write(tmp_path, rows)
        connection = duckdb.connect()
        context = basic.profile(connection, f"read_parquet('{ds.path.as_posix()}')", ds.types)
        date_time.b20_date_and_time(context)
        (found,) = context.found
        compiled = compile_steps(connection, str(ds.path), [found.step], strict=True)
        first = connection.execute(
            f"SELECT CAST(datetime AS VARCHAR) FROM ({compiled.sql()}) ORDER BY __quantilo_row"
        ).fetchone()

        assert (found.rule, found.mode, found.columns) == ("B-20", "S", ["Date", "Time"])
        assert found.cells == 96 and "Check your data." in found.reason
        assert first == ("2004-03-10 00:00:00",)

    def test_b20_leaves_a_date_that_has_its_time(self, tmp_path: Path) -> None:
        rows = [["Date", "Time"]] + [
            [f"2004-03-10 {h:02d}:30:00", f"{h:02d}:30"] for h in range(24)
        ]
        ds = write(tmp_path, rows)
        connection = duckdb.connect()
        context = basic.profile(connection, f"read_parquet('{ds.path.as_posix()}')", ds.types)
        date_time.b20_date_and_time(context)

        assert context.found == []
