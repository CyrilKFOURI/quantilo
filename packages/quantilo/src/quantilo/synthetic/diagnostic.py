"""Validity checks, as SDMetrics' Diagnostic report (expected: 100 %)."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quantilo.synthetic.encode import (
    OTHER,
    Learned,
    column_of,
    labels,
    numbers,
    seconds,
)


@dataclass
class Check:
    check: str  # "Boundary adherence", ...
    column: str | None
    score: float
    detail: str


@dataclass
class Diagnostic:
    score: float  # mean of Data validity and Data structure
    validity: float
    structure: float
    checks: list[Check] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)  # one sentence per check under 100 %


def _values(series: pd.Series, item: Learned) -> np.ndarray:
    raw = seconds(series) if item.kind == "date" else numbers(series)
    return raw[~np.isnan(raw)]


def boundary(series: pd.Series, item: Learned, low: float | None, high: float | None) -> float:
    values = _values(series, item)
    low = item.minimum if low is None else low
    high = item.maximum if high is None else high
    if values.size == 0 or low is None or high is None:
        return 1.0
    return float(((values >= low - 1e-9) & (values <= high + 1e-9)).mean())


def category(series: pd.Series, item: Learned) -> float:
    allowed = set(item.levels) | ({OTHER} if item.grouped else set())
    present = labels(series).dropna()
    return float(present.isin(allowed).mean()) if len(present) else 1.0


def uniqueness(series: pd.Series) -> float:
    present = series.dropna()
    return float(1 - present.duplicated().mean()) if len(present) else 1.0


def integer(series: pd.Series) -> float:
    values = numbers(series)
    values = values[~np.isnan(values)]
    return float((values == np.round(values)).mean()) if values.size else 1.0


def diagnose(
    synthetic: pd.DataFrame,
    learned: list[Learned],
    unique: list[str],
    bounds: dict[str, tuple[float | None, float | None]] | None = None,
    expected: list[str] | None = None,
) -> Diagnostic:
    """`expected` = the columns the synthetic data must have, in this order."""
    bounds = bounds or {}
    checks: list[Check] = []
    for item in learned:
        if item.name not in synthetic:
            continue
        values = column_of(synthetic, item.name)
        if item.kind == "category":
            checks.append(
                Check(
                    "Category adherence",
                    item.name,
                    category(values, item),
                    "Levels exist in the real data.",
                )
            )
            continue
        low, high = bounds.get(item.name, (None, None))
        checks.append(
            Check(
                "Boundary adherence",
                item.name,
                boundary(values, item, low, high),
                "Inside the real min - max.",
            )
        )
        if item.kind == "integer":
            checks.append(Check("Integer adherence", item.name, integer(values), "Whole numbers."))
    for name in unique:
        if name in synthetic:
            checks.append(
                Check(
                    "Key uniqueness",
                    name,
                    uniqueness(column_of(synthetic, name)),
                    "No repeated value.",
                )
            )
    wanted = expected if expected is not None else [i.name for i in learned] + unique
    found = list(synthetic.columns)
    shared = len(set(wanted) & set(found))
    union = len(set(wanted) | set(found))
    structure = shared / union if union else 1.0
    checks.append(
        Check("Table structure", None, structure, "Same columns as planned, nothing extra.")
    )
    column_checks = [c.score for c in checks if c.column is not None]
    validity = float(np.mean(column_checks)) if column_checks else 1.0
    failures = [
        f"{c.check}: {c.column or 'the table'} at {c.score:.1%}." for c in checks if c.score < 1
    ]
    return Diagnostic((validity + structure) / 2, validity, structure, checks, failures)
