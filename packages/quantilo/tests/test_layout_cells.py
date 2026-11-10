"""Cell signatures, numbers in text and placeholders (080: G2, T1, T2, T3, period labels)."""

import datetime as dt

import pytest

from quantilo.ingestion.layout.cells import (
    clean_text,
    is_blank,
    is_period,
    is_total_label,
    kind,
    null_token,
    parse_number,
    strip_refs,
)
from quantilo.ingestion.layout.model import ColumnPlan
from quantilo.ingestion.layout.types import convert, infer


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1 234", 1234.0),
        ("1,234.5", 1234.5),
        ("1.234,5", 1234.5),
        ("69,55%", 69.55),
        ("(123)", -123.0),
        ("$5", 5.0),
        ("-3.2", -3.2),
        ("2009-10", None),
        ("~2015", None),
        ("-", None),
        ("abc", None),
    ],
)
def test_parse_number(text: str, expected: float | None) -> None:
    assert parse_number(text) == expected


def test_blank_and_kinds() -> None:
    assert is_blank("\xa0 ") and is_blank(None) and not is_blank("x")  # G2
    assert kind(3) == "n" and kind("3,5") == "t" and kind("n/d") == "z" and kind("01") == "s"
    assert kind(dt.date(2024, 1, 1)) == "d" and kind(True) == "b"
    assert null_token("—") == "strong" and null_token("-") == "weak" and null_token("x1") is None


def test_periods_totals_refs() -> None:
    for label in (2024, "2024 (42)", "Sep.\n2023", "Fall 1990", "1869-70", "Q1 2024", "2026-01-02"):
        assert is_period(clean_text(label) if isinstance(label, str) else label), label
    assert not is_period("Alabama") and not is_period(12)
    assert is_total_label("Grand Total") and is_total_label("DO NOT DELETE THIS TOTALS ROW")
    assert is_total_label("Total operating expenses") and not is_total_label("Totality")
    assert strip_refs("Instruction\\2\\") == ("Instruction", ["2"])
    assert strip_refs("2024 (42)") == ("2024", ["42"])
    assert strip_refs("Rate (c) = (a) - (b)")[0] == "Rate (c) = (a) - (b)"


def test_types_decimal_codes_flags() -> None:
    plan = ColumnPlan("Turnout", 1)
    infer(["69,55%", "60,10%", "n/d", "28,67%"], plan)
    assert (plan.dtype, plan.decimal, plan.unit, plan.nulls) == ("number", ",", "%", 1)
    assert convert("69,55%", plan) == (69.55, None)
    codes = ColumnPlan("Code", 1)
    infer(["01", "2A", "ZX", "12", 75], codes)
    assert codes.dtype == "text"  # T1: codes keep their zeros and letters
    flags = ColumnPlan("Population", 1)
    infer(["74794 e", "75100", "75900 p", ":", "76100 e", "76500 b"], flags)
    assert flags.dtype == "number" and flags.flags
    assert convert("74794 e", flags) == (74794.0, "e")
    assert convert(":", flags) == (None, None)  # weak placeholder in a numeric column
    text = ColumnPlan("Reconcile", 1)
    infer(["R", "-", "X", "R"], text)
    assert convert("-", text) == ("-", None)  # kept in a text column (cash book marks)
