"""N40: a facet of one column of the draft, as OpenRefine shows one: its values and their rows (a
text), a histogram (a number or a date), the blank cells and those its type cannot read; and the
first rows of the draft a selection of them picks. Read on the draft as it stands (every step
applied); nothing is written: the page turns a selection into a step only when asked."""

from collections.abc import Callable
from typing import Any, Literal

import duckdb
from pydantic import BaseModel, Field

from quantilo.catalog import ColumnView
from quantilo.cleaning.compile import (
    ROW,
    Compiled,
    Stage,
    as_text,
    is_number,
    is_temporal,
    is_text,
)
from quantilo.cleaning.model import MAX_VALUES, CleanError, Name, Strict, Value
from quantilo.ingestion.sql import ident, lit
from quantilo.typing.parsers import present_sql, typed_sql

FacetKind = Literal["text", "number", "date"]
MAX_LISTED = 200  # the values listed at once; a search finds the others
BINS = 24
SAMPLE_ROWS = 50
OTHER_COLUMNS = 7  # shown beside the faceted column in the rows picked
READ_KINDS = ("number", "date", "boolean", "year")

Run = Callable[[str], list[tuple[Any, ...]]]


class FacetSelection(Strict):
    """What is picked: values (as text), the blank cells, the unreadable ones (they add up: any
    of them), or a range of the number or date (alone; dates in seconds since 1970)."""

    values: list[Value] = Field(default_factory=list, max_length=MAX_VALUES)
    blank: bool = False
    unreadable: bool = False
    low: float | None = None
    high: float | None = None

    def empty(self) -> bool:
        return (
            not (self.values or self.blank or self.unreadable)
            and self.low is None
            and (self.high is None)
        )


class FacetRequest(Strict):
    column: Name
    selection: FacetSelection | None = None
    sort: Literal["count", "name"] = "count"
    search: str | None = Field(default=None, max_length=200)


class FacetValue(BaseModel):
    value: str
    rows: int


class FacetBin(BaseModel):
    low: float
    high: float
    rows: int


class FacetRow(BaseModel):
    row: int  # the parent's row number (0-based)
    values: list[str | None]


class Facet(BaseModel):
    column: str
    kind: FacetKind
    dtype: str
    rows: int  # rows of the draft
    distinct: int
    blank: int  # NULL or only spaces, as the "is empty" condition of a step reads it
    # a text column its type reads as numbers, dates…: the cells it cannot read, and those values
    # (most rows first, MAX_VALUES at most); None when the type reads nothing
    unreadable: int | None = None
    unreadable_values: list[FacetValue] = Field(default_factory=list)
    reading: str | None = None
    values: list[FacetValue] = Field(default_factory=list)  # a text's, MAX_LISTED at most
    listed_from: int = 0  # distinct values the search kept (all of them without a search)
    low: float | None = None
    high: float | None = None
    bins: list[FacetBin] = Field(default_factory=list)
    matched: int | None = None  # rows the selection picks (None: nothing picked)
    shown: list[str] = Field(default_factory=list)
    sample: list[FacetRow] = Field(default_factory=list)


def _reading(dtype: str, view: ColumnView | None) -> str | None:
    if view is None or view.excluded or not is_text(dtype):
        return None
    return view.parser.kind if view.parser.kind in READ_KINDS else None


def facet(
    connection: duckdb.DuckDBPyConnection,
    compiled: Compiled,
    request: FacetRequest,
    view: ColumnView | None = None,
) -> Facet:
    """The facet of `request.column` after every step; `view` is the dataset's typing of the
    column (its reading of a text as numbers or dates), if it still holds."""
    final = compiled.final
    column = request.column
    dtype = final.dtype(column)  # CleanError when no step leaves it
    name = ident(column)
    text = as_text(column, dtype)
    kind: FacetKind = "number" if is_number(dtype) else "date" if is_temporal(dtype) else "text"
    measure = name if kind == "number" else f"epoch({name})" if kind == "date" else None
    reading = _reading(dtype, view)
    blank = f"({name} IS NULL OR trim({text}) = '')"
    unread = None
    if reading is not None and view is not None:
        unread = (
            f"(({present_sql(column, view.parser)}) IS NOT NULL "
            f"AND ({typed_sql(column, view.parser)}) IS NULL)"
        )
    source = compiled.sql(-1, "*")

    def run(sql: str) -> list[tuple[Any, ...]]:
        try:
            return connection.execute(sql).fetchall()
        except duckdb.Error as error:
            message = str(error).splitlines()[0]
            raise CleanError(f"The facet could not be computed: {message}") from None

    counts = ["count(*)", f"count(DISTINCT {text})", f"count(*) FILTER (WHERE {blank})"]
    if unread:
        counts.append(f"count(*) FILTER (WHERE {unread})")
    if measure:
        counts += [f"min({measure})", f"max({measure})"]
    head = run(f"WITH f AS ({source}) SELECT {', '.join(counts)} FROM f")[0]
    result = Facet(
        column=column,
        kind=kind,
        dtype=dtype,
        rows=int(head[0]),
        distinct=int(head[1]),
        blank=int(head[2]),
        reading=reading,
    )
    at = 3
    if unread:
        result.unreadable = int(head[at])
        at += 1
        if result.unreadable:
            found = run(
                f"WITH f AS ({source}) SELECT {text}, count(*) AS c FROM f WHERE {unread} "
                f"GROUP BY 1 ORDER BY c DESC, 1 LIMIT {MAX_VALUES}"
            )
            result.unreadable_values = [FacetValue(value=str(v), rows=int(c)) for v, c in found]
    if measure:
        low, high = head[at], head[at + 1]
        result.low = None if low is None else float(low)
        result.high = None if high is None else float(high)
        result.bins = _bins(run, source, measure, result.low, result.high)
    else:
        _values(run, source, text, request, result)
    if request.selection is not None and not request.selection.empty():
        where = _where(request.selection, text, blank, unread, measure, result)
        _sample(run, source, where, column, final, result)
    return result


def _bins(
    run: Run, source: str, measure: str, low: float | None, high: float | None
) -> list[FacetBin]:
    """BINS bins of equal width from the minimum to the maximum (one when they are equal)."""
    if low is None or high is None:
        return []
    if high == low:
        found = run(f"WITH f AS ({source}) SELECT count(*) FROM f WHERE {measure} = {low!r}")
        return [FacetBin(low=low, high=high, rows=int(found[0][0]))]
    width = (high - low) / BINS
    bucket = f"least({BINS - 1}, CAST(floor(({measure} - {low!r}) / {width!r}) AS INTEGER))"
    found = dict(
        run(
            f"WITH f AS ({source}) SELECT {bucket} AS b, count(*) FROM f "
            f"WHERE {measure} IS NOT NULL GROUP BY b"
        )
    )
    return [
        FacetBin(
            low=low + i * width,
            high=high if i == BINS - 1 else low + (i + 1) * width,
            rows=int(found.get(i, 0)),
        )
        for i in range(BINS)
    ]


def _values(run: Run, source: str, text: str, request: FacetRequest, result: Facet) -> None:
    """A text's values with their rows, most rows first or by name; a search keeps those that
    contain it (any case)."""
    search = f"AND contains(lower(v), lower({lit(request.search)}))" if request.search else ""
    order = "c DESC, v" if request.sort == "count" else "v, c DESC"
    found = run(
        f"WITH f AS ({source}), g AS (SELECT {text} AS v, count(*) AS c FROM f "
        f"WHERE {text} IS NOT NULL GROUP BY 1) SELECT v, c, count(*) OVER () FROM g "
        f"WHERE true {search} ORDER BY {order} LIMIT {MAX_LISTED}"
    )
    result.values = [FacetValue(value=str(v), rows=int(c)) for v, c, _ in found]
    result.listed_from = int(found[0][2]) if found else 0


def _where(
    selection: FacetSelection,
    text: str,
    blank: str,
    unread: str | None,
    measure: str | None,
    result: Facet,
) -> str:
    if selection.low is not None or selection.high is not None:
        if measure is None:
            raise CleanError("A range is picked on a column of numbers or dates only.")
        parts = []
        if selection.low is not None:
            parts.append(f"{measure} >= {selection.low!r}")
        if selection.high is not None:
            parts.append(f"{measure} <= {selection.high!r}")
        return " AND ".join(parts)
    parts = []
    if selection.values:
        parts.append(f"{text} IN ({', '.join(lit(v) for v in selection.values)})")
    if selection.blank:
        parts.append(blank)
    if selection.unreadable and unread:
        parts.append(unread)
    return " OR ".join(f"({p})" for p in parts) or "false"


def _sample(run: Run, source: str, where: str, column: str, final: Stage, result: Facet) -> None:
    """The rows picked: how many, and the first ones in the file's order, the faceted column
    first, then a few others."""
    others = [n for n in final.names if n not in (column, ROW)][:OTHER_COLUMNS]
    shown = [column, *others]
    head = run(f"WITH f AS ({source}) SELECT count(*) FROM f WHERE {where}")
    result.matched = int(head[0][0])
    result.shown = shown
    if not result.matched:
        return
    values = ", ".join(as_text(n, final.dtype(n)) for n in shown)
    found = run(
        f"WITH f AS ({source}) SELECT {ident(ROW)}, {values} FROM f WHERE {where} "
        f"ORDER BY {ident(ROW)} LIMIT {SAMPLE_ROWS}"
    )
    result.sample = [
        FacetRow(row=int(r[0]), values=[None if v is None else str(v) for v in r[1:]])
        for r in found
    ]
