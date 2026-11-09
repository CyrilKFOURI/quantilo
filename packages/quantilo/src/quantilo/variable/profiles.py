"""Profiles of non-numeric variables: categories, dates, free text, identifiers."""

import math
from dataclasses import dataclass, field

from scipy import stats

from quantilo.charts.prepare import Bar
from quantilo.typing.dates import HAS_TIME, STEP_SECONDS, auto_step, classify_step
from quantilo.typing.summary import value_label
from quantilo.variable.source import VariableSource, clean

RARE_SHARE = 0.01
DOMINANT_SHARE = 0.8
LIST_LIMIT = 20
CALENDAR_DAYS = 3 * 366


@dataclass
class CategoricalProfile:
    rows: int
    missing: int
    distinct: int
    entropy: float | None  # bits
    normalized_entropy: float | None  # entropy / log2(distinct), 1 = perfectly even
    herfindahl: float | None  # sum of squared shares, 1 = a single category
    top: Bar | None
    top_share: float | None
    imbalanced: bool
    rare_count: int
    rare: list[Bar] = field(default_factory=list)  # least frequent first
    chi2: float | None = None  # goodness of fit to a uniform distribution
    chi2_p: float | None = None


@dataclass
class DateProfile:
    rows: int
    missing: int
    distinct: int
    min: str | None
    max: str | None
    has_time: bool
    granularity: str
    step: str | None
    periods: int
    gap_count: int
    gaps: list[str] = field(default_factory=list)  # first empty periods
    calendar: list[Bar] = field(default_factory=list)  # rows per day, when the range is short


@dataclass
class TextProfile:
    rows: int
    missing: int
    distinct: int
    min_length: int | None
    max_length: int | None
    mean_length: float | None
    words: list[Bar] = field(default_factory=list)


@dataclass
class IdentifierProfile:
    rows: int
    missing: int
    distinct: int
    unique_pct: float
    duplicated_values: int
    duplicates: list[Bar] = field(default_factory=list)
    min_length: int | None = None
    max_length: int | None = None
    patterns: list[Bar] = field(default_factory=list)


def _counts(source: VariableSource) -> tuple[int, int, int]:
    rows, valid, distinct = source.one(
        f"SELECT count(*), count(t), count(DISTINCT t) FROM {source.values}"
    )
    return int(rows), int(rows - valid), int(distinct)


def _bars(rows: list[tuple]) -> list[Bar]:
    return [Bar(label=value_label(value), count=int(count)) for value, count in rows]


def categorical(source: VariableSource) -> CategoricalProfile:
    rows, missing, distinct = _counts(source)
    present = rows - missing
    grouped = f"(SELECT t, count(*) AS c FROM {source.values} WHERE t IS NOT NULL GROUP BY t)"
    entropy, herfindahl = (
        source.one(
            f"SELECT -sum((c / {present}) * log2(c / {present})), sum(power(c / {present}, 2)) "
            f"FROM {grouped}"
        )
        if present
        else (None, None)
    )
    top = _bars(source.all(f"SELECT t, c FROM {grouped} ORDER BY c DESC, t LIMIT 1"))
    limit = present * RARE_SHARE
    (rare_count,) = source.one(f"SELECT count(*) FROM {grouped} WHERE c < {limit}")
    rare = _bars(
        source.all(f"SELECT t, c FROM {grouped} WHERE c < {limit} ORDER BY c, t LIMIT {LIST_LIMIT}")
    )
    top_share = top[0]["count"] / present if top and present else None
    profile = CategoricalProfile(
        rows,
        missing,
        distinct,
        clean(entropy),
        clean(entropy / math.log2(distinct)) if entropy is not None and distinct > 1 else None,
        clean(herfindahl),
        top[0] if top else None,
        top_share,
        bool(top_share is not None and top_share > DOMINANT_SHARE and distinct > 1),
        int(rare_count),
        rare,
    )
    if distinct > 1:
        observed = [int(c) for (c,) in source.all(f"SELECT c FROM {grouped}")]
        test = stats.chisquare(observed)
        profile.chi2, profile.chi2_p = clean(test.statistic), clean(test.pvalue)
    return profile


def dates(source: VariableSource, year_only: bool) -> DateProfile:
    rows, missing, distinct = _counts(source)
    low, high, span, timed = source.one(
        f"SELECT min(t), max(t), epoch(max(t)) - epoch(min(t)), "
        f"count(*) FILTER (WHERE {HAS_TIME}) FROM {source.values} WHERE t IS NOT NULL"
    )
    (gap,) = source.one(
        "SELECT median(d) FROM (SELECT epoch(t) - lag(epoch(t)) OVER (ORDER BY t) AS d "
        f"FROM (SELECT DISTINCT t FROM {source.values} WHERE t IS NOT NULL))"
    )
    has_time = bool(timed)
    profile = DateProfile(
        rows,
        missing,
        distinct,
        str(low) if low is not None else None,
        str(high) if high is not None else None,
        has_time,
        classify_step(float(gap) if gap is not None else None),
        None,
        0,
        0,
    )
    if low is None:
        return profile
    step = auto_step(float(span), has_time, year_only)
    profile.step = step
    periods = (
        f"(SELECT unnest(generate_series(date_trunc('{step}', min(t)), "
        f"date_trunc('{step}', max(t)), INTERVAL 1 {step})) AS p "
        f"FROM {source.values} WHERE t IS NOT NULL)"
    )
    used = (
        f"(SELECT DISTINCT date_trunc('{step}', t) AS p FROM {source.values} WHERE t IS NOT NULL)"
    )
    (profile.periods,) = source.one(f"SELECT count(*) FROM {periods}")
    empty = f"(SELECT p FROM {periods} EXCEPT SELECT p FROM {used})"
    (profile.gap_count,) = source.one(f"SELECT count(*) FROM {empty}")
    profile.gaps = [
        str(p.date()) if step != "hour" else str(p)
        for (p,) in source.all(f"SELECT p FROM {empty} ORDER BY p LIMIT {LIST_LIMIT}")
    ]
    if not year_only and float(span) <= CALENDAR_DAYS * STEP_SECONDS["day"]:
        profile.calendar = _bars(
            source.all(
                f"SELECT CAST(t AS DATE) AS d, count(*) FROM {source.values} "
                "WHERE t IS NOT NULL GROUP BY d ORDER BY d"
            )
        )
    return profile


def text(source: VariableSource) -> TextProfile:
    rows, missing, distinct = _counts(source)
    low, high, mean = source.one(
        f"SELECT min(n), max(n), avg(n) FROM (SELECT length(CAST(t AS VARCHAR)) AS n "
        f"FROM {source.values} WHERE t IS NOT NULL)"
    )
    words = source.all(
        "SELECT w, count(*) AS c FROM ("
        "SELECT unnest(regexp_split_to_array(lower(CAST(t AS VARCHAR)), '[^\\p{L}\\p{N}]+')) AS w "
        f"FROM {source.values} WHERE t IS NOT NULL) "
        f"WHERE length(w) >= 2 GROUP BY w ORDER BY c DESC, w LIMIT {LIST_LIMIT}"
    )
    return TextProfile(
        rows,
        missing,
        distinct,
        int(low) if low is not None else None,
        int(high) if high is not None else None,
        clean(mean),
        _bars(words),
    )


def identifier(source: VariableSource) -> IdentifierProfile:
    rows, missing, distinct = _counts(source)
    present = rows - missing
    as_text = "CAST(t AS VARCHAR)"
    duplicated = (
        f"(SELECT t, count(*) AS c FROM {source.values} WHERE t IS NOT NULL "
        "GROUP BY t HAVING c > 1)"
    )
    (duplicated_values,) = source.one(f"SELECT count(*) FROM {duplicated}")
    low, high = source.one(
        f"SELECT min(length({as_text})), max(length({as_text})) FROM {source.values} "
        "WHERE t IS NOT NULL"
    )
    pattern = f"regexp_replace(regexp_replace({as_text}, '[A-Za-z]', 'A', 'g'), '[0-9]', '9', 'g')"
    return IdentifierProfile(
        rows,
        missing,
        distinct,
        100 * distinct / present if present else 0.0,
        int(duplicated_values),
        _bars(source.all(f"SELECT t, c FROM {duplicated} ORDER BY c DESC, t LIMIT {LIST_LIMIT}")),
        int(low) if low is not None else None,
        int(high) if high is not None else None,
        _bars(
            source.all(
                f"SELECT {pattern} AS p, count(*) AS c FROM {source.values} WHERE t IS NOT NULL "
                "GROUP BY p ORDER BY c DESC, p LIMIT 5"
            )
        ),
    )
