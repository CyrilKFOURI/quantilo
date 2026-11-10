"""Per-column summary on the full data, read through the column's parser.

Feeds the Columns page: counts, invalid values (present but unreadable in the chosen type),
example values and a small distribution for the sparkline.
"""

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import duckdb

from quantilo.ingestion.normalize import ZONED_TYPES
from quantilo.typing.dates import date_profile
from quantilo.typing.knowledge_base import load
from quantilo.typing.parsers import present_sql, typed_sql
from quantilo.typing.text import TextMeasures, language_label, measure
from quantilo.typing.types import AnalysisType, Parser

HISTOGRAM_BINS = 12
TOP_CATEGORIES = 8
TEXT_SAMPLE = 2_000  # distinct texts measured for the language (050, C-2; 092, R-ING-06)


def number_label(value: float) -> str:
    return f"{value:.4g}"


EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def epoch_label(value: float) -> str:
    # Not datetime.fromtimestamp: on Windows it rejects dates before 1970.
    return (EPOCH + timedelta(seconds=value)).date().isoformat()


def _extent(
    connection: duckdb.DuckDBPyConnection, values: str, median: bool = False
) -> tuple[Any, Any, int, Any]:
    """min, max, count (and the median when asked) of `x` in `values`, in one pass."""
    middle = "median(x)" if median else "NULL"
    row = connection.execute(f"SELECT min(x), max(x), count(x), {middle} FROM {values}").fetchone()
    return row if row else (None, None, 0, None)  # type: ignore[return-value]


def _values(source: str, numeric: str) -> str:
    return f"(SELECT {numeric} AS x FROM {source}) WHERE x IS NOT NULL"


def numeric_range(
    connection: duckdb.DuckDBPyConnection, source: str, typed: str
) -> tuple[dict[str, float] | None, tuple[Any, Any, int]]:
    """{min, max, median} of a number (the filter editor's hint, 100 R-FLT-09), and the extent
    the histogram reuses."""
    low, high, count, median = _extent(
        connection, _values(source, f"CAST({typed} AS DOUBLE)"), median=True
    )
    if low is None or high is None:
        return None, (low, high, count)
    return {"min": float(low), "max": float(high), "median": float(median)}, (low, high, count)


def _bins(
    connection: duckdb.DuckDBPyConnection,
    source: str,
    numeric: str,
    label: Callable[[float], str],
    extent: tuple[Any, Any, int] | None = None,
) -> list[dict[str, Any]]:
    """Equal-width histogram of a numeric expression (`extent`: its min, max, count if known)."""
    values = _values(source, numeric)
    low, high, count = extent or _extent(connection, values)[:3]
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
    extent: tuple[Any, Any, int] | None = None,
) -> list[dict[str, Any]]:
    if analysis_type == "numeric_continuous":
        return _bins(connection, source, f"CAST({typed} AS DOUBLE)", number_label, extent)
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
    role: str | None = None,
) -> dict[str, Any]:
    typed = typed_sql(column, parser)
    present = present_sql(column, parser)
    kind = connection.execute(f"SELECT typeof({typed}) FROM {source} LIMIT 1").fetchone()
    if kind and str(kind[0]).upper().startswith(ZONED_TYPES):
        # stored before 092 (R-ING-01): read in UTC, DuckDB would need pytz to hand it over
        typed = f"timezone('UTC', {typed})"
    rows, present_count, valid, distinct = connection.execute(
        f"SELECT count(*), count(r), count(t), count(DISTINCT t) "
        f"FROM (SELECT {typed} AS t, {present} AS r FROM {source})"
    ).fetchone() or (0, 0, 0, 0)
    examples = _top(connection, source, typed, "c DESC, v", 5)
    numeric, extent = (
        numeric_range(connection, source, typed)
        if analysis_type in ("numeric_continuous", "numeric_discrete") and valid
        else (None, None)
    )
    summary: dict[str, Any] = {
        "rows": int(rows),
        "null_count": int(rows - valid),
        "invalid_count": int(present_count - valid),
        "distinct_count": int(distinct),
        "examples": [example["label"] for example in examples],
        "histogram": histogram(connection, source, typed, analysis_type, order, extent),
        "numeric": numeric,
        "date": None,
        "text": None,
    }
    if analysis_type == "datetime" and valid:
        summary["date"] = date_profile(connection, source, typed, parser.kind == "year")
    # A list of choices (R-TYP-09) has no language: its measures would cost time for nothing.
    if analysis_type == "text" and valid and role != "choices":
        values = f"(SELECT {typed} AS t FROM {source})"
        summary["text"] = text_summary(text_measures(connection, values))
    return summary
