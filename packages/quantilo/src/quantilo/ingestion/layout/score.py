"""The confidence of a detected table and the reasons behind it (study § 3.8).

Additive and explained: every term is a RuleHit with its weight, kept on the table so the interface
can say why a table is checked, under review, or folded in Advanced.
"""

from collections import Counter
from dataclasses import dataclass, field

from quantilo.ingestion.layout.cells import kind
from quantilo.ingestion.layout.grid import Grid
from quantilo.ingestion.layout.model import RuleHit

BASE = 0.55


@dataclass
class ScoreFacts:
    grid: Grid
    header_rows: list[int]  # naming rows, 0-based
    body: list[int]  # data rows, 0-based
    cols: list[int]  # 0-based columns
    text_table: bool
    key_value: bool
    uncached: int
    verified_totals: int
    gaps: int
    extra: list[RuleHit] = field(default_factory=list)


def consistency(grid: Grid, body: list[int], cols: list[int]) -> float:
    """Mean share of each column's dominant cell kind (numbers in text count as numbers)."""
    shares = []
    for c in cols:
        kinds = Counter(
            "n" if (k := kind(grid.value(r, c))) == "t" else k
            for r in body
            if grid.value(r, c) is not None
        )
        kinds.pop("z", None)
        total = sum(kinds.values())
        if total:
            shares.append(kinds.most_common(1)[0][1] / total)
    return sum(shares) / len(shares) if shares else 0.0


def _share(grid: Grid, cells: set[tuple[int, int]], rows: list[int], cols: list[int]) -> float:
    filled = [(r, c) for r in rows for c in cols if grid.value(r, c) is not None]
    return sum(cell in cells for cell in filled) / len(filled) if filled else 0.0


def styled_header(grid: Grid, row: int, body: list[int], cols: list[int]) -> bool:
    """C3: the header row is bold, filled or bordered where the body is not."""
    sample = body[:20]
    return any(
        _share(grid, cells, [row], cols) >= 0.6
        and _share(grid, cells, [row], cols) - _share(grid, cells, sample, cols) >= 0.3
        for cells in (grid.bold, grid.filled, grid.bordered)
    )


def score(facts: ScoreFacts) -> tuple[float, list[RuleHit]]:
    grid, body, cols = facts.grid, facts.body, facts.cols
    hits: list[RuleHit] = []
    agreement = consistency(grid, body, cols)
    hits.append(
        RuleHit(
            "S-body",
            BASE + 0.15 * agreement,
            f"Body of {len(body)} rows, column types agree at {agreement:.0%}.",
        )
    )
    header = facts.header_rows
    if header:
        last = header[-1]
        header_text = sum(
            isinstance(grid.value(r, c), str)
            for r in header
            for c in cols
            if grid.value(r, c) is not None
        )
        header_cells = sum(grid.value(r, c) is not None for r in header for c in cols)
        typed = [c for c in cols if any(kind(grid.value(r, c)) in "ntdb" for r in body[:50])]
        if not grid.styled and last == 0 and header_cells and header_text == header_cells:
            hits.append(RuleHit("S-csv", 0.1, "Names on the first row of a plain CSV."))
        if header_cells and header_text >= 0.6 * header_cells and len(typed) * 2 >= len(cols):
            hits.append(RuleHit("S-contrast", 0.1, "Text header above a typed body."))
        if grid.styled and body:
            if styled_header(grid, last, body, cols):
                hits.append(
                    RuleHit(
                        "C3",
                        0.1,
                        f"Header row {last + 1} is styled (bold, fill or border) unlike the body.",
                    )
                )
            elif facts.text_table:
                hits.append(RuleHit("C3-text", -0.2, "Text block without a styled header row."))
        if grid.freeze is not None and body and grid.freeze[0] in (last + 1, body[0]):
            hits.append(RuleHit("C3-pane", 0.1, f"Frozen pane right under row {last + 1}."))
    elif not facts.key_value:
        hits.append(RuleHit("S-noheader", -0.15, "No header row: columns are numbered."))
    if facts.text_table and not grid.styled and header:
        hits.append(
            RuleHit("C3-text", -0.2, "Text block: no types or styles to confirm the header.")
        )
    if facts.uncached:
        hits.append(
            RuleHit(
                "T4",
                -0.2,
                f"{facts.uncached} formula cell"
                f"{'s have' if facts.uncached > 1 else ' has'} no saved value: "
                "left empty, open and save the file in Excel to fill them.",
            )
        )
    if len(body) < 3:
        hits.append(
            RuleHit("S-small", -0.2, f"Only {len(body)} row{'s' if len(body) != 1 else ''}.")
        )
    if facts.verified_totals:
        hits.append(
            RuleHit(
                "C9",
                0.05,
                f"{facts.verified_totals} total row"
                f"{'s' if facts.verified_totals > 1 else ''} checked by arithmetic.",
            )
        )
    hits.extend(facts.extra)
    confidence = sum(hit.weight for hit in hits)
    return round(min(1.0, max(0.0, confidence)), 2), hits
