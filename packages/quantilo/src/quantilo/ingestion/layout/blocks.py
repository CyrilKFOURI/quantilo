"""Regions of a sheet that may each hold one table (D rules).

D1: rows separated by at least one empty row are separate blocks. D2: an empty column splits a block
into side-by-side tables only when the rows on each side differ (Jaccard < 0.8) and no merged cell
crosses it. D1-glue: a block without a header right under a block that has one, with the same column
profile, continues it (internal blank rows). Template rows (C11) are ignored by the D2 test.
"""

from dataclasses import dataclass, field
from itertools import pairwise

from quantilo.ingestion.layout.cells import is_blank, numeric_value
from quantilo.ingestion.layout.grid import Grid
from quantilo.ingestion.layout.model import RuleHit

SPLIT_JACCARD = 0.8
GLUE_GAP = 2
GLUE_PROFILE = 0.8  # share of the lower block's columns found in the upper one


@dataclass
class Region:
    rows: list[int]  # 0-based sheet rows with at least one occupied cell in [left, right]
    left: int
    right: int
    hits: list[RuleHit] = field(default_factory=list)
    gaps: list[int] = field(default_factory=list)  # empty columns kept inside (D2-keep)

    @property
    def top(self) -> int:
        return self.rows[0]

    @property
    def bottom(self) -> int:
        return self.rows[-1]


TEMPLATE_RUN = 3


def template_rows(grid: Grid) -> set[int]:
    """C11: runs of 3+ rows holding only zeros (formulas of an empty template), no text.

    A single row of zeros is real data (day 0, value 0)."""
    zero = []
    for row in grid.values:
        cells = [v for v in row if v is not None]
        zero.append(
            bool(cells) and all(not isinstance(v, str) and numeric_value(v) == 0 for v in cells)
        )
    found: set[int] = set()
    run: list[int] = []
    for r, flag in enumerate([*zero, False]):
        if flag:
            run.append(r)
            continue
        if len(run) >= TEMPLATE_RUN:
            found.update(run)
        run = []
    return found


def _row_runs(grid: Grid, rows: list[int], left: int, right: int) -> list[list[int]]:
    runs: list[list[int]] = []
    current: list[int] = []
    previous = None
    for r in rows:
        if not grid.row_cols(r, left, right):
            continue
        if previous is not None and r != previous + 1:
            runs.append(current)
            current = []
        current.append(r)
        previous = r
    if current:
        runs.append(current)
    return runs


def _crossed(grid: Grid, rows: list[int], col: int) -> bool:
    """A merged cell spans the empty column (its value sits left of it)."""
    top, bottom = rows[0], rows[-1]
    return any(m[0] <= bottom and m[2] >= top and m[1] < col <= m[3] for m in grid.merges)


def _jaccard(a: set[int], b: set[int]) -> float:
    return len(a & b) / max(1, len(a | b))


def split_columns(
    grid: Grid, rows: list[int], left: int, right: int, template: set[int]
) -> list[Region]:
    """D2 on one block: split at an empty column only for truly side-by-side tables."""
    counted = [r for r in rows if r not in template] or rows
    used = [c for c in range(left, right + 1) if any(grid.occupied(r, c) for r in counted)]
    if not used:
        return []
    left, right = used[0], used[-1]
    gaps = [c for c in range(left, right + 1) if c not in set(used)]
    segments: list[tuple[int, int]] = []
    start = left
    for c in range(left, right + 2):
        if c > right or c in gaps:
            if c - 1 >= start:
                segments.append((start, c - 1))
            start = c + 1
    for (a0, a1), (b0, b1) in pairwise(segments):
        gap = a1 + 1
        left_rows = {r for r in counted if grid.row_cols(r, a0, a1)}
        right_rows = {r for r in counted if grid.row_cols(r, b0, b1)}
        score = _jaccard(left_rows, right_rows)
        if score < SPLIT_JACCARD and not any(_crossed(grid, rows, c) for c in range(a1 + 1, b0)):
            hit = RuleHit(
                "D2",
                0.0,
                f"Empty column {gap + 1} separates two tables side by side "
                f"(rows in common {score:.0%}).",
            )
            regions = []
            for lo, hi in ((left, a1), (b0, right)):
                for run in _row_runs(grid, rows, lo, hi):
                    for region in split_columns(grid, run, lo, hi, template):
                        region.hits.insert(0, hit)
                        regions.append(region)
            return regions
    region = Region(rows=rows, left=left, right=right, gaps=gaps)
    if gaps:
        region.hits.append(
            RuleHit(
                "D2-keep",
                -0.1,
                f"{len(gaps)} empty column{'s' if len(gaps) > 1 else ''} inside the table kept: "
                "the rows go on across them.",
            )
        )
    return [region]


def find_regions(grid: Grid, template: set[int]) -> list[Region]:
    """D1 then D2: blocks of contiguous rows, each split into side-by-side tables when needed."""
    regions: list[Region] = []
    for run in _row_runs(grid, list(range(grid.n_rows)), 0, max(grid.n_cols - 1, 0)):
        regions.extend(split_columns(grid, run, 0, grid.n_cols - 1, template))
    return sorted(regions, key=lambda region: (region.top, region.left))


def column_profile(grid: Grid, rows: list[int], left: int, right: int) -> set[int]:
    """Columns filled in at least half of the given rows."""
    if not rows:
        return set()
    return {
        c for c in range(left, right + 1) if sum(grid.occupied(r, c) for r in rows) * 2 >= len(rows)
    }


def is_single_label(grid: Grid, r: int, left: int, right: int) -> bool:
    """One text label, alone or beside a level number in the first column (BLS "0 | Region")."""
    cols = grid.row_cols(r, left, right)
    texts = [
        c
        for c in cols
        if isinstance(grid.value(r, c), str) and numeric_value(grid.value(r, c)) is None
    ]
    others = [c for c in cols if c not in texts]
    return len(texts) == 1 and (not others or (others == [left] and texts[0] > left))


def blank_row(grid: Grid, r: int, left: int, right: int) -> bool:
    return all(is_blank(grid.value(r, c)) for c in range(left, right + 1))
