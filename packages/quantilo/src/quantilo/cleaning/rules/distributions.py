"""Advanced rules on a number column's distribution (B1, study § 2.2), on the reproducible sample
of the Advanced layer (200,000 rows at most):

- A-06: outliers, beyond 3 × IQR of the quartiles and 5 robust deviations (1.4826 × MAD) of the
  median (on log10 for a positive column spread over an order of magnitude or more), 1 % of the
  values at most. Flagged in a new column (the study's default); the reason gives the bounds to
  cap them by hand. Never emptied by this rule (B-10 empties a value out of its role's range).
- A-07: floating-point noise (0.30000000000000004): rounded to the decimals the other values use.
- A-08: first digits against Benford's law (Nigrini: MAD above 0.015). A hint, never a correction.
- A-09: last digits heaped on 0 and 5 (estimates, rounding): a hint, never a correction.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from scipy import stats

from quantilo import semantics
from quantilo.cleaning import formula as fx
from quantilo.cleaning import model as m
from quantilo.cleaning.compile import FLOATS, is_number
from quantilo.cleaning.rules.basic import Column, n

if TYPE_CHECKING:
    from quantilo.cleaning.rules.advanced import Advanced

MIN_VALUES = 50
IQR_FENCE = 3.0  # A-06
ROBUST_Z = 5.0
OUTLIER_SHARE = 0.01  # A-06: at most this share of the values are outliers
LOG_SPREAD = 10  # A-06: p99 / p1 from which a positive column is read on log10
NOISE_SHARE = 0.99  # A-07
NOISE = 1e-9  # A-07: relative distance of a noisy value to its rounding
BENFORD_VALUES = 1_000  # A-08
BENFORD_SPREAD = 100  # A-08: two orders of magnitude between p1 and p99
BENFORD_MAD = 0.015  # Nigrini, first digit: nonconformity above this
LAST_VALUES = 200  # A-09
LAST_P = 0.001
LAST_EXCESS = 1.5  # 0 and 5 together at least 1.5 × their 20 %
AMOUNT_ROLES = ("amount", "cost", "unit_price")


def _values(advanced: Advanced, column: Column) -> np.ndarray:
    from quantilo.cleaning.rules.advanced import SAMPLE, number_expr

    expr = number_expr(column)
    rows = advanced.context.connection.execute(
        f"SELECT v FROM (SELECT {expr} AS v FROM {SAMPLE}) WHERE v IS NOT NULL AND isfinite(v)"
    ).fetchall()
    return np.array([r[0] for r in rows], dtype=float)


def _skip(advanced: Advanced, column: Column, rules: tuple[str, ...]) -> bool:
    """A column another Advanced rule explains already (A-05: values in another unit)."""
    return any(f.rule in rules and f.columns[:1] == [column.name] for f in advanced.found)


# --- A-06 ----------------------------------------------------------------------------------------


def _bounds(values: np.ndarray) -> tuple[float, float, bool] | None:
    """The outlier bounds (in the column's unit), and whether they were found on log10."""
    positive = bool(np.all(values > 0))
    low, high = np.quantile(values, [0.01, 0.99])
    logged = positive and low > 0 and high / low >= LOG_SPREAD
    scale = np.log10(values) if logged else values
    q1, median, q3 = np.quantile(scale, [0.25, 0.5, 0.75])
    iqr = q3 - q1
    mad = float(np.median(np.abs(scale - median))) * 1.4826
    if iqr <= 0 or mad <= 0:
        return None
    lower = min(q1 - IQR_FENCE * iqr, median - ROBUST_Z * mad)
    upper = max(q3 + IQR_FENCE * iqr, median + ROBUST_Z * mad)
    if logged:
        return float(10**lower), float(10**upper), True
    return float(lower), float(upper), False


def _short(value: float) -> str:
    if value == 0:
        return "0"
    digits = max(0, 3 - math.floor(math.log10(abs(value))))
    rounded = round(value, digits)
    return f"{rounded:,.{digits}f}".rstrip("0").rstrip(".") if digits else f"{rounded:,.0f}"


def outliers(advanced: Advanced) -> None:
    from quantilo.cleaning.rules.advanced import _measures, display_sql, number_expr, row_label

    context = advanced.context
    for column in _measures(advanced):
        if _skip(advanced, column, ("A-05",)):
            continue
        values = _values(advanced, column)
        if len(values) < MIN_VALUES or len(np.unique(values)) < 10:
            continue
        bounds = _bounds(values)
        if bounds is None:
            continue
        lower, upper, logged = bounds
        if lower <= float(values.min()):
            low_bound: float | None = None  # nothing below it: one bound only
        else:
            low_bound = lower
        high_bound = None if upper >= float(values.max()) else upper
        expr = number_expr(column)
        outside = f"(v < {lower!r} OR v > {upper!r})"
        count, total = (
            int(x)
            for x in context.one(
                f"SELECT count(*) FILTER (WHERE {outside}), count(v) FROM (SELECT {expr} AS v "
                f"FROM {context.source})"
            )
        )
        if not count or count > OUTLIER_SHARE * total:
            continue
        listed = context.connection.execute(
            f"SELECT file_row_number, {display_sql(column)}, v FROM (SELECT file_row_number, "
            f"{expr} AS v, * EXCLUDE (file_row_number) FROM {advanced.rows_source}) "
            f"WHERE {outside} ORDER BY abs(v) DESC LIMIT 3"
        ).fetchall()
        shown = ", ".join(f"{r[1]} ({row_label(r[0])})" for r in listed)
        flag = _free_name(context, f"{column.name}_unusual")
        scale = (
            " (read on a log scale: the values spread over orders of magnitude)" if logged else ""
        )
        if low_bound is None:
            where = f"above {_short(upper)}"
        elif high_bound is None:
            where = f"below {_short(lower)}"
        else:
            where = f"outside {_short(lower)} – {_short(upper)}"
        reason = (
            f"{n(count)} values of {column.name} lie {where}, "
            f"beyond 3 × IQR of the quartiles and 5 robust deviations of the median{scale}: "
            f"{shown}. They may be real: errors and rare cases look alike, and models rarely "
            f"gain from removing them. The step flags them in a new column, {flag}, to look at "
            "them or compare them; to cap them instead, add the step Transform numbers › Clip "
            "with these bounds."
        )
        name = fx.quoted(column.name)
        tests = [f"{name} < {lower!r}"] if low_bound is not None else []
        tests += [f"{name} > {upper!r}"] if high_bound is not None else []
        test = " or ".join(tests) or f"{name} > {upper!r}"
        advanced.add(
            "A-06",
            "S",
            [column],
            f"Flag {n(count)} unusual values of {column.name}",
            reason,
            m.ConditionalColumn(
                id="s0", to=flag, cases=[m.Case(when=test, value="unusual")], otherwise=None
            ),
            cells=count,
            examples=[(str(r[1]), "unusual") for r in listed],
        )


def _free_name(context: object, wanted: str) -> str:
    names = {c.name for c in getattr(context, "columns", [])}
    name, number = wanted, 2
    while name in names:
        name, number = f"{wanted}_{number}", number + 1
    return name


# --- A-07 ----------------------------------------------------------------------------------------


def noise(advanced: Advanced) -> None:
    context = advanced.context
    for column in context.columns:
        if column.dtype not in FLOATS or not column.nonnull:
            continue
        values = _values(advanced, column)
        if len(values) < MIN_VALUES:
            continue
        for places in range(0, 7):
            rounded = np.round(values, places)
            exact = values == rounded
            close = np.abs(values - rounded) <= NOISE * np.maximum(1.0, np.abs(values))
            noisy = close & ~exact
            if np.mean(close) < NOISE_SHARE:
                continue
            if not noisy.any() or np.mean(exact) < NOISE_SHARE:
                break
            count = int(
                context.one(
                    f"SELECT count(*) FROM {context.source} WHERE {_ident(column)} <> "
                    f"round({_ident(column)}, {places})"
                )[0]
            )
            example = values[noisy][0]
            advanced.add(
                "A-07",
                "S",
                [column],
                f"Round {n(count)} values of {column.name} to {places} decimals",
                f"{np.mean(exact):.1%} of {column.name} has at most {places} decimals; "
                f"{n(count)} values are a hair away ({float(example)!r}): floating-point noise "
                f"from a calculation. Rounded to {places} decimals, they read and compare as "
                "written.",
                m.TransformNumber(id="s0", column=column.name, how="round", decimals=places),
                cells=count,
                examples=[(repr(float(example)), f"{float(np.round(example, places)):g}")],
            )
            break


def _ident(column: Column) -> str:
    from quantilo.ingestion.sql import ident

    return ident(column.name)


# --- A-08 ----------------------------------------------------------------------------------------

BENFORD = np.log10(1 + 1 / np.arange(1, 10))


def _amount(column: Column) -> bool:
    view = column.view
    role = (view.role if view else None) or (column.role.id if column.role else None)
    return role in AMOUNT_ROLES or semantics.is_money(column.name)


def benford(advanced: Advanced) -> None:
    from quantilo.cleaning.rules.advanced import _measures

    for column in _measures(advanced):
        if not _amount(column) or _skip(advanced, column, ("A-05",)):
            continue
        values = np.abs(_values(advanced, column))
        values = values[values > 0]
        if len(values) < BENFORD_VALUES:
            continue
        low, high = np.quantile(values, [0.01, 0.99])
        if low <= 0 or high / low < BENFORD_SPREAD:
            continue
        first = (values / 10 ** np.floor(np.log10(values))).astype(int).clip(1, 9)
        shares = np.bincount(first, minlength=10)[1:10] / len(first)
        mad = float(np.mean(np.abs(shares - BENFORD)))
        if mad <= BENFORD_MAD:
            continue
        worst = int(np.argmax(np.abs(shares - BENFORD))) + 1
        advanced.add(
            "A-08",
            "I",
            [column],
            f"First digits of {column.name} depart from Benford's law",
            f"Amounts spread over orders of magnitude usually start with 1 about 30 % of the "
            f"time, 9 about 5 % (Benford's law). Here {worst} starts {shares[worst - 1]:.1%} of "
            f"the values ({BENFORD[worst - 1]:.1%} expected; mean gap {mad:.3f}, above "
            f"{BENFORD_MAD}) over {n(len(values))} values. A hint of made-up, capped or "
            "generated values, or of a mix of populations, never a proof: check how they were "
            "recorded. Information only.",
            cells=0,
            examples=[
                (f"starts with {d}", f"{shares[d - 1]:.1%} (Benford {BENFORD[d - 1]:.1%})")
                for d in (1, 2, worst)
            ],
        )


# --- A-09 ----------------------------------------------------------------------------------------


def last_digits(advanced: Advanced) -> None:
    from quantilo.cleaning.rules.advanced import _measures

    for column in _measures(advanced):
        if not is_number(column.dtype) or _skip(advanced, column, ("A-05",)):
            continue
        values = np.abs(_values(advanced, column))
        whole = values[(values == np.round(values)) & (values >= 10)]
        if len(whole) < LAST_VALUES or len(whole) < 0.9 * len(values):
            continue
        digits = (whole % 10).astype(int)
        counts = np.bincount(digits, minlength=10)
        if counts[0] >= 0.95 * len(digits):
            continue  # counted in tens: the unit, not an estimate
        heaped = (counts[0] + counts[5]) / len(digits)
        _, p = stats.chisquare(counts)
        if heaped < LAST_EXCESS * 0.2 or p >= LAST_P:
            continue
        advanced.add(
            "A-09",
            "I",
            [column],
            f"{heaped:.0%} of {column.name} end in 0 or 5",
            f"{counts[0] / len(digits):.0%} of the whole values of {column.name} end in 0 and "
            f"{counts[5] / len(digits):.0%} in 5, against 10 % each if the last digit were left "
            f"to chance (χ² p < {LAST_P}): rounded or estimated values (heaping)? They are less "
            "precise than they look. Information only.",
            cells=0,
            examples=[(f"ends in {d}", f"{counts[d] / len(digits):.1%}") for d in (0, 5)],
        )
