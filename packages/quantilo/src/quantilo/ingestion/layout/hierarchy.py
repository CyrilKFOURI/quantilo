"""Row hierarchy read from the layout (C10) and labels left blank under a group (C12).

C10: the level of a row comes from an explicit "Indent Level" column (BLS), a "." prefix (Census),
leading spaces or NBSP (NCES), or the cell's indent style, in that order of trust. It becomes a
`level` column, and `Level_1…k` columns (the nearest ancestor of each level) when at least two
parents have children: a typographic hierarchy whose parents are not followed by their children
(states listed after all the regions) only gets `level`.
"""

import re
from itertools import pairwise

from quantilo.ingestion.layout.cells import clean_text, indent_of, numeric_value
from quantilo.ingestion.layout.grid import Grid

_DOTS = re.compile(r"^\.+")


def level_column(names: dict[int, str]) -> int | None:
    for c, name in names.items():
        if clean_text(name).lower().replace("_", " ").startswith("indent level"):
            return c
    return None


def clean_label(value: object) -> object:
    """A hierarchy label without its "." prefix or leading spaces."""
    if not isinstance(value, str):
        return value
    return _DOTS.sub("", value.lstrip(" \xa0  ")).strip()


def detect_levels(
    grid: Grid, rows: list[int], label_col: int, names: dict[int, str]
) -> tuple[dict[int, int], str | None, int | None]:
    """Levels of the body rows, the signal they come from, and the explicit level column if any."""
    explicit = level_column(names)
    if explicit is not None:
        levels = {}
        for r in rows:
            number = numeric_value(grid.value(r, explicit))
            if number is not None and float(number).is_integer():
                levels[r] = int(number)
        if len(set(levels.values())) >= 2:
            return levels, "column", explicit
    labelled = [r for r in rows if isinstance(grid.value(r, label_col), str)]
    if len(labelled) < 3:
        return {}, None, None
    dots = {
        r: len(m.group()) for r in labelled if (m := _DOTS.match(str(grid.value(r, label_col))))
    }
    if len(dots) >= 3:
        return {r: dots.get(r, 0) for r in labelled}, "dot", None
    spaces = {r: indent_of(grid.value(r, label_col)) for r in labelled}
    if len(set(spaces.values())) >= 2:
        return spaces, "spaces", None
    styled = {r: grid.indent.get((r, label_col), 0) for r in labelled}
    if len(set(styled.values())) >= 2:
        low = min(styled.values())
        return {r: v - low for r, v in styled.items()}, "style", None
    return {}, None, None


def has_parents(levels: dict[int, int], rows: list[int]) -> bool:
    """At least two rows are directly followed by deeper rows (their children)."""
    ordered = [r for r in rows if r in levels]
    parents = sum(1 for a, b in pairwise(ordered) if levels[b] > levels[a])
    return parents >= 2


def ancestors(
    levels: dict[int, int], rows: list[int], labels: dict[int, object]
) -> dict[int, list[object]]:
    """For each row, the label of its nearest ancestor at each level above it."""
    depth = max(levels.values(), default=0)
    stack: list[object] = [None] * (depth + 1)
    result: dict[int, list[object]] = {}
    for r in rows:
        if r not in levels:
            result[r] = [None] * depth
            continue
        level = levels[r]
        result[r] = [stack[i] if i < level else None for i in range(depth)]
        stack[level] = labels.get(r)
        for i in range(level + 1, depth + 1):
            stack[i] = None
    return result


def fill_down_columns(grid: Grid, rows: list[int], key_cols: list[int]) -> list[int]:
    """C12: leading label columns left blank under each group label (pivot-table exports)."""
    found = []
    for index, c in enumerate(key_cols):
        if index > 1 or (index == 1 and key_cols[0] not in found):
            break
        values = [grid.value(r, c) for r in rows]
        filled = [v for v in values if v is not None]
        blanks = len(values) - len(filled)
        neighbour = key_cols[index + 1] if index + 1 < len(key_cols) else None
        if neighbour is None or len(filled) < 2 or blanks * 4 < len(values) or values[0] is None:
            break
        beside = sum(grid.value(r, neighbour) is not None for r in rows)
        if beside * 10 < len(rows) * 9:
            break
        found.append(c)
    return found
