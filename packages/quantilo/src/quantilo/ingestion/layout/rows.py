"""What each row of a region is: title, note, header, column numbers, section, total, template,
data.

C1 titles, C2 notes, C6 column-number rows, C7 section labels, C8 totals by keyword, C9 totals
verified by arithmetic (above or below, linear time with running sums), C11 empty templates.
"""

import datetime as dt
from dataclasses import dataclass
from itertools import pairwise

from quantilo.ingestion.layout.cells import (
    clean_text,
    is_note_text,
    is_total_label,
    is_year_number,
    kind,
    numeric_value,
    parse_number,
)
from quantilo.ingestion.layout.grid import Grid

NUMERIC_KINDS = frozenset("ntdbz")
TOTAL_TOLERANCE = 0.005  # C9: 0.5 %
TITLE_LENGTH = 40
DATA_SHARE = 0.4  # a body row: 40 % of its cells past the label are numbers (StatCan: 5 of 11)
MIN_RUN = 3  # C9 without the word "total": the run summed must have 3 rows or more


@dataclass
class RowStats:
    row: int
    cols: list[int]
    kinds: dict[int, str]
    texts: list[int]

    @property
    def n(self) -> int:
        return len(self.cols)


def row_stats(grid: Grid, r: int, left: int, right: int) -> RowStats:
    cols = grid.row_cols(r, left, right)
    kinds = {c: ("u" if grid.value(r, c) is None else kind(grid.value(r, c))) for c in cols}
    return RowStats(r, cols, kinds, [c for c in cols if kinds[c] == "s"])


def is_column_numbers(grid: Grid, stats: RowStats) -> bool:
    """C6: 1, 2, 3… under the header (NCES), maybe repeated under merged pairs (1, 2, 2, 3)."""
    if stats.n < 3:
        return False
    numbers = []
    for c in stats.cols:
        value = grid.value(stats.row, c)
        number = numeric_value(value) if not isinstance(value, bool) else None
        if number is None or not float(number).is_integer():
            return False
        numbers.append(int(number))
    if numbers[0] not in (0, 1):
        return False
    return all(b - a in (0, 1) for a, b in pairwise(numbers)) and len(set(numbers)) >= 3


def _period_key(value: object) -> int | None:
    """An ordering key for a year or a date heading a column."""
    if isinstance(value, dt.datetime | dt.date):
        return value.toordinal() * 10
    if is_year_number(value):
        year = int(clean_text(value)[:4]) if isinstance(value, str) else int(value)  # type: ignore[call-overload]
        return dt.date(year, 1, 1).toordinal() * 10
    return None


def is_year_header(grid: Grid, stats: RowStats, stub: int) -> bool:
    """A header row of years or dates (2020 2021…, timesheet days): nothing in the label column,
    or 3+ of them in order beside a label (Eurostat "geo\\TIME_PERIOD | 2015 | 2016 …")."""
    labelled = grid.occupied(stats.row, stub)
    values = [grid.value(stats.row, c) for c in stats.cols if not (labelled and c == stub)]
    keys = [_period_key(v) for v in values]
    if len(values) < (3 if labelled else 2) or any(k is None for k in keys):
        return False
    if len({clean_text(v) for v in values}) != len(values):  # "2017 (40)" and "2017 (41)" differ
        return False
    return not labelled or keys == sorted(keys) or keys == sorted(keys, reverse=True)  # type: ignore[type-var]


def is_data(grid: Grid, stats: RowStats, stub: int) -> bool:
    """A body row: 40 % of its cells past the label column are numbers (or empty formulas)."""
    if stats.n == 0 or is_column_numbers(grid, stats) or is_year_header(grid, stats, stub):
        return False
    others = [c for c in stats.cols if c != stub]
    if not others:
        return False
    known = [c for c in others if stats.kinds[c] != "z"]  # placeholders say nothing either way
    if not known:
        return len(others) >= 2
    numeric = sum(stats.kinds[c] in NUMERIC_KINDS or stats.kinds[c] == "u" for c in known)
    return numeric >= DATA_SHARE * len(known)


def is_title_row(grid: Grid, stats: RowStats, stub: int, width: int) -> bool:
    """C1: one cell in the label column, a long sentence, or a cell merged across the table."""
    if stats.n != 1:
        return False
    c = stats.cols[0]
    value = grid.value(stats.row, c)
    if c <= stub or (isinstance(value, str) and len(clean_text(value)) > TITLE_LENGTH):
        return True
    return any(
        m[0] <= stats.row <= m[2] and m[1] == c and (m[3] - m[1] + 1) * 3 >= width * 2
        for m in grid.merges
    )


def is_note_row(grid: Grid, stats: RowStats) -> bool:
    """C2: under the body, a footnote, a source, a reading guide or a long sentence."""
    if stats.n == 0 or len(stats.texts) != stats.n or stats.n > 2:
        return False
    text = clean_text(grid.value(stats.row, stats.cols[0]))
    return is_note_text(text) or len(text) > TITLE_LENGTH


def first_text(grid: Grid, stats: RowStats) -> str | None:
    for c in stats.cols:
        value = grid.value(stats.row, c)
        if isinstance(value, str) and parse_number(value) is None:
            return clean_text(value)
    return None


def total_keyword(grid: Grid, stats: RowStats) -> bool:
    """C8: the first text says total, subtotal, grand total, ensemble, all…"""
    text = first_text(grid, stats)
    return bool(text) and is_total_label(text or "")


def _close(total: float, value: float) -> bool:
    return abs(total - value) <= max(TOTAL_TOLERANCE * abs(total), 1e-9)


def _match(row_values: list[float | None], sums: list[float], keyword: bool) -> bool:
    checked = hits = 0
    for value, total in zip(row_values, sums, strict=True):
        if value is None or value == 0:
            continue
        checked += 1
        hits += _close(value, total)
    if checked == 0:
        return False
    if checked == 1:
        return keyword and hits == 1  # one column: only with the word "total"
    return hits / checked >= 0.8 or (hits >= 3 and hits / checked >= 0.6)


def verify_totals(
    grid: Grid,
    body: list[int],
    cols: list[int],
    style_key: dict[int, tuple[object, ...]],
    breaks: set[int],
) -> dict[int, str]:
    """C9: rows equal (0.5 %) to the sum of the contiguous block below (top) or above (bottom).

    `breaks` are section labels: a run never crosses one. Candidate runs end at the section end, at
    the end of the run of rows styled like the next row, or before the next sibling (same style as
    the candidate). Running sums keep each check linear in the number of columns."""
    if len(body) < 3 or not cols:
        return {}
    values = [[numeric_value(grid.value(r, c)) for c in cols] for r in body]
    prefix = [[0.0] * len(cols)]
    for row in values:
        prefix.append([p + (v or 0.0) for p, v in zip(prefix[-1], row, strict=True)])

    def span(i: int, j: int) -> list[float]:  # sum of body[i:j]
        return [b - a for a, b in zip(prefix[i], prefix[j], strict=True)]

    section_of: list[int] = []
    current = 0
    for i, r in enumerate(body):
        if r in breaks:
            current = i + 1
        section_of.append(current)
    ends: list[int] = []
    end = len(body)
    for i in range(len(body) - 1, -1, -1):
        if body[i] in breaks:
            end = i
        ends.append(end)
    ends.reverse()
    keys = [style_key.get(r) for r in body]
    found: dict[int, str] = {}
    after_total = 0  # a run above never reaches back over a total already found
    for i, r in enumerate(body):
        if r in breaks:
            continue
        row = values[i]
        if sum(1 for v in row if v not in (None, 0)) < 1:
            continue
        opens, closes = i == section_of[i], i == ends[i] - 1
        label = next(
            (v for v in grid.values[r] if isinstance(v, str) and parse_number(v) is None), None
        )
        keyword = label is not None and is_total_label(label)
        if not keyword and (sum(1 for v in row if v not in (None, 0)) < 2 or not (opens or closes)):
            continue  # without the word "total", a total opens or closes its section
        # below (C9-top)
        candidates = {ends[i]}
        j = i + 1
        while j < ends[i] and keys[j] == keys[min(i + 1, len(keys) - 1)]:
            j += 1
        candidates.add(j)
        j = i + 1
        while j < ends[i] and keys[j] != keys[i]:
            j += 1
        candidates.add(j)
        shortest = 2 if keyword else MIN_RUN  # Net sales = Cost of sales + Gross margin is no total
        if (keyword or opens) and any(
            e - i - 1 >= shortest and _match(row, span(i + 1, e), keyword) for e in candidates
        ):
            found[r] = "top"
            after_total = i + 1
            continue
        # above (C9-bottom)
        start = max(section_of[i], after_total)
        candidates = {start}
        j = i - 1
        while j > start and keys[j - 1] == keys[max(i - 1, 0)]:
            j -= 1
        candidates.add(j)
        j = i - 1
        while j > start and keys[j - 1] != keys[i]:
            j -= 1
        candidates.add(j)
        if (keyword or closes) and any(
            i - s >= shortest and _match(row, span(s, i), keyword) for s in candidates
        ):
            found[r] = "bottom"
            after_total = i + 1
    return found
