"""Column types after the layout is known (T1–T3).

T1: a column whose values read as numbers at 90 % or more is numeric, with its decimal separator
decided once for the column; codes ("01", "2A") stay text. T2: placeholders become empty (strong
tokens anywhere, weak ones in a numeric column only). T3: a flag glued to a number ("74794 e",
"2024 (2)") is split into the value and a flag column.
"""

import datetime as dt
import re
from collections import Counter

from quantilo.ingestion.layout.cells import (
    FLAGGED_NUMBER,
    LEADING_ZERO,
    SHORT_CODE,
    clean_text,
    null_token,
    parse_number,
)
from quantilo.ingestion.layout.model import ColumnPlan, Dtype

NUMERIC_SHARE = 0.9
_DECIMAL_COMMA = re.compile(r"\d,(\d{1,2}|\d{4,})\s*%?\s*$")


def decimal_of(texts: list[str]) -> str:
    commas = sum(1 for t in texts if _DECIMAL_COMMA.search(t) and "." not in t)
    dots = sum(1 for t in texts if re.search(r"\d\.\d", t) and "," not in t)
    return "," if commas > dots else "."


def infer(values: list[object], plan: ColumnPlan) -> None:
    """Fill the plan's dtype, decimal, nulls and flags from the body values of one column."""
    present = [v for v in values if v is not None]
    texts = [clean_text(v) for v in present if isinstance(v, str)]
    plan.decimal = decimal_of(texts)
    strong = [v for v in present if isinstance(v, str) and null_token(v) == "strong"]
    weak = [v for v in present if isinstance(v, str) and null_token(v) == "weak"]
    rest = [v for v in present if not (isinstance(v, str) and null_token(v))]
    if not rest:
        plan.dtype = "text"
        plan.nulls = len(strong)
        return
    kinds: Counter[str] = Counter()
    flagged = 0
    codes = 0
    fractional = False
    for v in rest:
        if isinstance(v, bool):
            kinds["bool"] += 1
        elif isinstance(v, int | float):
            kinds["number"] += 1
            fractional = fractional or not float(v).is_integer()
        elif isinstance(v, dt.date | dt.datetime | dt.time):
            kinds["date"] += 1
        else:
            text = clean_text(v)
            if LEADING_ZERO.match(text) or SHORT_CODE.match(text):
                codes += 1
            number = parse_number(text, plan.decimal)
            if number is not None and not LEADING_ZERO.match(text):
                kinds["number"] += 1
                fractional = fractional or not number.is_integer()
            elif (flag := FLAGGED_NUMBER.match(text)) and parse_number(
                flag.group(1), plan.decimal
            ) is not None:
                kinds["number"] += 1
                flagged += 1
            else:
                kinds["text"] += 1
    dtype, count = kinds.most_common(1)[0]
    if count < NUMERIC_SHARE * len(rest) or (dtype == "number" and codes and not fractional):
        dtype = "text"
    plan.dtype = dtype  # type: ignore[assignment]
    plan.flags = dtype == "number" and flagged >= 2
    if (
        dtype == "number"
        and plan.unit is None
        and texts
        and sum(t.endswith("%") for t in texts) * 2 >= len(texts)
    ):
        plan.unit = "%"
    plan.nulls = len(strong) + (len(weak) if dtype == "number" else 0)
    plan.unreadable = 0 if dtype == "text" else len(rest) - kinds[dtype]


def convert(value: object, plan: ColumnPlan) -> tuple[object, str | None]:
    """A cell as the column's type, and its flag (T3) if any."""
    if value is None:
        return None, None
    if isinstance(value, str):
        token = null_token(value)
        if token == "strong" or (token == "weak" and plan.dtype == "number"):
            return None, None
    dtype: Dtype = plan.dtype
    if dtype == "number":
        if isinstance(value, bool):
            return float(value), None
        if isinstance(value, int | float):
            return value, None
        text = clean_text(value)
        number = parse_number(text, plan.decimal)
        if number is not None:
            return number, None
        match = FLAGGED_NUMBER.match(text)
        if match:
            return parse_number(match.group(1), plan.decimal), match.group(2).strip("()")
        return None, None
    if dtype in ("date", "bool") and isinstance(value, str):
        return None, None  # a stray label in a typed column (counted in `unreadable`)
    if dtype == "text" and not isinstance(value, str):
        if isinstance(value, float) and value.is_integer():
            return str(int(value)), None
        return str(value), None
    if dtype == "text":
        text = str(value)
        return clean_text(text) if "\n" in text or "\xa0" in text else text.strip(), None
    return value, None
