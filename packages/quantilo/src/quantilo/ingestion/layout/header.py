"""Column names from one or several header rows (C4, C4b, C5, C5-ref, G3).

Parents spread over their children by merge (Census, SEC) or as a cell followed by empty cells above
a full sub-row ("Income" is not merged in the cash book). Names join the levels with " · "; a label
cut word by word over several rows ("Food and / non- / alcoholic") is joined back as one phrase.
"""

import re
from dataclasses import dataclass, field
from itertools import pairwise

from quantilo.ingestion.layout.cells import clean_text, is_unit_cell, strip_refs
from quantilo.ingestion.layout.grid import Grid

SEPARATOR = " · "
_CONTINUES = re.compile(r"(\b(and|or|of|the|in|for|to|de|et|du|des|la|le)|[,&/(-])$", re.I)


@dataclass
class Header:
    names: dict[int, str]  # sheet column -> final name
    parts: dict[int, list[str]]  # sheet column -> one part per naming row ("" when none)
    spread: dict[int, list[bool]]  # the part came from a parent spreading over the column
    units: dict[int, str] = field(default_factory=dict)
    unit_rows: list[int] = field(default_factory=list)
    refs: list[str] = field(default_factory=list)  # footnote references taken out of the names
    raw: dict[int, list[object]] = field(default_factory=dict)  # raw cell values per naming row


def is_unit_row(grid: Grid, row: int, cols: list[int]) -> bool:
    """C5: a row of units under the names ("(Quadrillion Btu)", "($/bbl)", "%")."""
    filled = [c for c in cols if grid.value(row, c) is not None]
    if not filled:
        return False
    units = sum(is_unit_cell(grid.value(row, c)) for c in filled)
    return units * 2 >= len(filled) and units >= 1


def _merge_value(grid: Grid, row: int, col: int) -> tuple[object, bool, bool]:
    """G3: the value of the merged range a header cell belongs to (copied, header only).

    Returns (value, inside a merged range, copied from its top-left cell)."""
    for top, left, bottom, right in grid.merges:
        if top <= row <= bottom and left <= col <= right and (bottom > top or right > left):
            return grid.value(top, left), True, (top, left) != (row, col)
    return grid.value(row, col), False, False


def _phrase(parts: list[str]) -> bool:
    """C4b: the parts read as one sentence cut over several rows."""
    filled = [p for p in parts if p]
    if len(filled) < 2:
        return False
    return any(_CONTINUES.search(a) or (b[:1].islower()) for a, b in pairwise(filled))


def _join_phrase(parts: list[str]) -> str:
    text = ""
    for part in (p for p in parts if p):
        if not text:
            text = part
        elif text.endswith("-"):
            text += part
        else:
            text += " " + part
    return text


def build_header(grid: Grid, rows: list[int], cols: list[int], skip: set[int]) -> Header:
    """Names of `cols` from the header `rows` (sheet indices); `skip` rows do not name columns."""
    # C5: units sit under the names, never on the first header row
    unit_rows = [r for r in rows[1:] if r not in skip and is_unit_row(grid, r, cols)]
    naming = [r for r in rows if r not in skip and r not in unit_rows]
    header = Header(names={}, parts={c: [] for c in cols}, spread={c: [] for c in cols})
    header.unit_rows = unit_rows
    header.raw = {c: [] for c in cols}
    for level, r in enumerate(naming):
        below = naming[level + 1] if level + 1 < len(naming) else None
        current = ""
        current_raw: object = None
        for c in cols:
            value, in_merge, copied = _merge_value(grid, r, c)
            text = clean_text(value) if value is not None else ""
            spread = copied
            if in_merge:
                current = ""  # a merged parent spreads by its merge only
            elif text:
                current, current_raw = text, value
            elif below is not None and current and not _starts_group(grid, r, c, cols):
                # a parent cell followed by empty cells above a filled sub-row
                if _merge_value(grid, below, c)[0] is not None:
                    text, value, spread = current, current_raw, True
                else:
                    current = ""
            else:
                current = ""
            header.parts[c].append(text)
            header.spread[c].append(spread)
            header.raw[c].append(value if text else None)
    for c in cols:
        cleaned = []
        for part in header.parts[c]:
            text, refs = strip_refs(part)
            header.refs.extend(refs)
            cleaned.append(text)
        header.parts[c] = cleaned
        if not any(header.spread[c]) and _phrase(cleaned):
            name = _join_phrase(cleaned)
        else:
            unique: list[str] = []
            for part in cleaned:
                if part and (not unique or unique[-1] != part):
                    unique.append(part)
            name = SEPARATOR.join(unique)
        header.names[c] = name
        for r in unit_rows:
            unit = grid.value(r, c)
            if unit is not None:
                header.units[c] = clean_text(unit).strip("()[] ")
    header.refs = sorted(set(header.refs), key=lambda ref: (len(ref), ref))
    return header


def _starts_group(grid: Grid, row: int, col: int, cols: list[int]) -> bool:
    """An empty column (gap) or a merged cell starting here ends the parent's spread."""
    index = cols.index(col)
    if index > 0 and cols[index - 1] != col - 1:
        return True
    return any(top <= row <= bottom and left == col for top, left, bottom, _ in grid.merges)


def unique_names(names: list[str]) -> list[str]:
    """Empty names become column_N, duplicates get their position (like `read_sheet`)."""
    result: list[str] = []
    for index, name in enumerate(names):
        name = name or f"column_{index + 1}"
        while name in result:
            name = f"{name}_{index + 1}"
        result.append(name)
    return result
