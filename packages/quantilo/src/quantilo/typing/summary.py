"""Per-column summary on the full data, read through the column's parser.

Feeds the Columns page: counts, invalid values (present but unreadable in the chosen type),
example values and a small distribution for the sparkline.
"""

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import duckdb

from quantilo.typing.dates import date_profile
from quantilo.typing.knowledge_base import load
from quantilo.typing.parsers import present_sql, typed_sql
from quantilo.typing.text import TextMeasures, language_label, measure
from quantilo.typing.types import AnalysisType, Parser

HISTOGRAM_BINS = 12
TOP_CATEGORIES = 8
TEXT_SAMPLE = 5_000  # distinct texts measured for the language (050, C-2)


def number_label(value: float) -> str:
    return f"{value:.4g}"


EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def epoch_label(value: float) -> str:
    # Not datetime.fromtimestamp: on Windows it rejects dates before 1970.
    return (EPOCH + timedelta(seconds=value)).date().isoformat()


def _bins(
    connection: duckdb.DuckDBPyConnection,
    source: str,
    numeric: str,
    label: Callable[[float], str],
) -> list[dict[str, Any]]:
    """Equal-width histogram of a numeric expression."""
    values = f"(SELECT {numeric} AS x FROM {source}) WHERE x IS NOT NULL"
    low, high, count = connection.execute(
        f"SELECT min(x), max(x), count(x) FROM {values}"
    ).fetchone() or (None, None, 0)
    if low is None or high is None:
        return []
    if low == high:
        return [{"label": label(low), "count": int(count)}]
    width = (high - low) / HISTOGRAM_BINS
    rows = connection.execute(
        f"SELECT LEAST(CAST(floor((x - {low}) / {width}) AS INTEGER), {HISTOGRAM_BINS - 1}) AS b, "
        f"count(*) FROM {values} GROUP BY b"
    ).fetchall()
    counts = {int(bin_index): int(bin_count) for bin_index, bin_count in rows}
    return [
        {"label": label(low + index * width), "count": counts.get(index, 0)}
        for index in range(HISTOGRAM_BINS)
    ]


def value_label(value: Any) -> str:
    """30.0 → "30"; other values as text."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _top(
    connection: duckdb.DuckDBPyConnection, source: str, typed: str, order_by: str, limit: int
) -> list[dict[str, Any]]:
    rows = connection.execute(
        f"SELECT t, CAST(t AS VARCHAR) AS v, count(*) AS c "
        f"FROM (SELECT {typed} AS t FROM {source}) "
        f"WHERE t IS NOT NULL GROUP BY t ORDER BY {order_by} LIMIT {limit}"
    ).fetchall()
    return [{"label": value_label(value), "count": int(count)} for value, _text, count in rows]


def histogram(
    connection: duckdb.DuckDBPyConnection,
    source: str,
    typed: str,
    analysis_type: AnalysisType,
    order: list[str] | None,
) -> list[dict[str, Any]]:
    if analysis_type == "numeric_continuous":
        return _bins(connection, source, f"CAST({typed} AS DOUBLE)", number_label)
    if analysis_type == "datetime":
        return _bins(connection, source, f"epoch({typed})", epoch_label)
    if analysis_type == "numeric_discrete":
        return _top(connection, source, typed, "t", 30)
    if analysis_type in ("categorical_nominal", "boolean"):
        return _top(connection, source, typed, "c DESC, v", TOP_CATEGORIES)
    if analysis_type == "categorical_ordinal":
        counts = {row["label"]: row["count"] for row in _top(connection, source, typed, "v", 100)}
        ordered = [{"label": item, "count": counts.pop(item, 0)} for item in order or []]
        return ordered + [{"label": label, "count": count} for label, count in counts.items()]
    return []


def text_measures(
    connection: duckdb.DuckDBPyConnection, values: str, limit: int = TEXT_SAMPLE
) -> TextMeasures:
    """Measures of a text column (`values` yields `t`): `limit` distinct texts with their counts,
    picked by a hash of the text (reproducible)."""
    rows = connection.execute(
        f"SELECT v, c FROM (SELECT CAST(t AS VARCHAR) AS v, count(*) AS c FROM {values} "
        f"WHERE t IS NOT NULL GROUP BY v) ORDER BY hash(v), v LIMIT {limit}"
    ).fetchall()
    return measure([(str(value), int(count)) for value, count in rows], load())


def text_summary(measures: TextMeasures) -> dict[str, Any]:
    return {
        "language": measures.language,
        "language_label": language_label(measures, load()),
        "language_share": measures.language_share,
        "median_words": measures.median_words,
    }


def summarize(
    connection: duckdb.DuckDBPyConnection,
    source: str,
    column: str,
    analysis_type: AnalysisType,
    parser: Parser,
    order: list[str] | None = None,
) -> dict[str, Any]:
    typed = typed_sql(column, parser)
    present = present_sql(column, parser)
    rows, present_count, valid, distinct = connection.execute(
        f"SELECT count(*), count(r), count(t), count(DISTINCT t) "
        f"FROM (SELECT {typed} AS t, {present} AS r FROM {source})"
    ).fetchone() or (0, 0, 0, 0)
    examples = _top(connection, source, typed, "c DESC, v", 5)
    summary: dict[str, Any] = {
        "rows": int(rows),
        "null_count": int(rows - valid),
        "invalid_count": int(present_count - valid),
        "distinct_count": int(distinct),
        "examples": [example["label"] for example in examples],
        "histogram": histogram(connection, source, typed, analysis_type, order),
        "date": None,
        "text": None,
    }
    if analysis_type == "datetime" and valid:
        summary["date"] = date_profile(connection, source, typed, parser.kind == "year")
    if analysis_type == "text" and valid:
        values = f"(SELECT {typed} AS t FROM {source})"
        summary["text"] = text_summary(text_measures(connection, values))
    return summary
