"""Outliers of a numeric variable by four classic rules, and the rows they sit in."""

from dataclasses import dataclass, field
from typing import Literal

from quantilo.variable.numeric import Describe
from quantilo.variable.source import VariableSource

OutlierMethod = Literal["iqr", "iqr3", "zscore", "robust"]
METHODS: tuple[OutlierMethod, ...] = ("iqr", "iqr3", "zscore", "robust")
ROW_LIMIT = 100
Z_LIMIT = 3.0
ROBUST_LIMIT = 3.5
MAD_TO_SIGMA = 0.6745  # Iglewicz & Hoaglin's modified z-score


@dataclass
class OutlierRule:
    method: OutlierMethod
    low: float | None
    high: float | None
    count: int
    pct: float


@dataclass
class OutlierRow:
    row: int  # 1-based, as in the source file
    value: float


@dataclass
class Outliers:
    method: OutlierMethod
    rules: list[OutlierRule] = field(default_factory=list)
    rows: list[OutlierRow] = field(default_factory=list)


def bounds(method: OutlierMethod, summary: Describe) -> tuple[float | None, float | None]:
    if method in ("iqr", "iqr3"):
        if summary.q1 is None or summary.q3 is None:
            return None, None
        k = 1.5 if method == "iqr" else 3.0
        spread = summary.q3 - summary.q1
        return summary.q1 - k * spread, summary.q3 + k * spread
    if method == "zscore":
        if summary.mean is None or not summary.std:
            return None, None
        return summary.mean - Z_LIMIT * summary.std, summary.mean + Z_LIMIT * summary.std
    if summary.median is None or not summary.mad:
        return None, None
    reach = ROBUST_LIMIT * summary.mad / MAD_TO_SIGMA
    return summary.median - reach, summary.median + reach


def outliers(source: VariableSource, summary: Describe, method: OutlierMethod) -> Outliers:
    limits: dict[OutlierMethod, tuple[float | None, float | None]] = {
        m: bounds(m, summary) for m in METHODS
    }
    filters = [
        f"count(*) FILTER (WHERE x < {low} OR x > {high})" if low is not None else "0"
        for low, high in limits.values()
    ]
    counts = source.one(f"SELECT {', '.join(filters)} FROM {source.numbers}")
    total = summary.count or 1
    result = Outliers(method)
    result.rules = [
        OutlierRule(m, low, high, int(count), 100 * int(count) / total)
        for (m, (low, high)), count in zip(limits.items(), counts, strict=True)
    ]
    low, high = limits[method]
    if low is not None and summary.median is not None:
        rows = source.all(
            f"SELECT row, x FROM {source.numbers} WHERE x < {low} OR x > {high} "
            f"ORDER BY abs(x - {summary.median}) DESC, row LIMIT {ROW_LIMIT}"
        )
        result.rows = [OutlierRow(int(row) + 1, float(value)) for row, value in rows]
    return result
