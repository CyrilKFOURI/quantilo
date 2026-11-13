"""Find every real table of a workbook or a verbose CSV: `detect_layout(path) -> Layout`.

Pipeline (study § 3): bounded grid (G) → regions (C11, D1, D2, D1-glue) → per region, the first
data line, the header above it (C1, C4-parent), the body below it (C2, C7, C8, D3-rep) → totals
checked by arithmetic (C9) → names (C4, C4b, C5) → hierarchy (C10, C12) → types (T) → proposals (K)
→ confidence (S). Deterministic, rules only, pure Python (Pyodide).
"""

from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

from openpyxl.utils.cell import range_boundaries

from quantilo.ingestion.layout.blocks import (
    GLUE_GAP,
    GLUE_PROFILE,
    Region,
    column_profile,
    find_regions,
    is_single_label,
    template_rows,
)
from quantilo.ingestion.layout.cells import (
    clean_text,
    indent_of,
    is_note_text,
    is_period,
    is_unit_cell,
    kind,
    strip_refs,
)
from quantilo.ingestion.layout.csv import CsvShape, load_csv_grid, stream_csv_end
from quantilo.ingestion.layout.grid import LIMITS, Grid, Limits, load_grids, stream_body_end
from quantilo.ingestion.layout.header import Header, build_header, unique_names
from quantilo.ingestion.layout.hierarchy import (
    detect_levels,
    fill_down_columns,
    has_parents,
)
from quantilo.ingestion.layout.model import (
    REVIEW,
    ColumnPlan,
    DetectedTable,
    Layout,
    RuleHit,
    SheetLayout,
    Transform,
)
from quantilo.ingestion.layout.reshape import (
    Facts,
    derived_columns,
    key_value,
    numbered_groups,
    period_columns,
    running_balance,
    sparse_groups,
    transposed,
)
from quantilo.ingestion.layout.rows import (
    RowStats,
    is_column_numbers,
    is_data,
    is_title_row,
    row_stats,
    total_keyword,
    verify_totals,
)
from quantilo.ingestion.layout.score import ScoreFacts, score
from quantilo.ingestion.layout.types import infer

MAX_HEADER_ROWS = 6
SECTION_LOOKAHEAD = 3
CSV_EXTENSIONS = (".csv", ".tsv", ".txt")


@dataclass
class Draft:
    region: Region
    stub: int
    header_rows: list[int]
    title_rows: list[int]
    roles: dict[int, str]  # body row -> data | template | section | repeated_header
    text_table: bool
    notes: list[int] = field(default_factory=list)
    hits: list[RuleHit] = field(default_factory=list)

    @property
    def body(self) -> list[int]:
        return sorted(self.roles)

    @property
    def first_data(self) -> int:
        return min(r for r, role in self.roles.items() if role in ("data", "template"))

    @property
    def last_data(self) -> int:
        return max(r for r, role in self.roles.items() if role in ("data", "template"))


def _covered(grid: Grid, row: int, col: int) -> bool:
    if grid.occupied(row, col):
        return True
    return any(t <= row <= b and lft <= col <= rgt for t, lft, b, rgt in grid.merges)


def _in_merge(grid: Grid, row: int, col: int) -> bool:
    return any(
        t <= row <= b and lft <= col <= rgt and (b > t or rgt > lft)
        for t, lft, b, rgt in grid.merges
    )


def _body_profile(grid: Grid, rows: list[int], left: int, right: int) -> set[int]:
    return {c for c in range(left, right + 1) if any(_covered(grid, r, c) for r in rows)}


def _parent_ok(
    grid: Grid,
    parent: RowStats,
    lower: list[int],
    below: list[int],
    stub: int,
    right: int,
    data_cols: set[int],
) -> bool:
    """C4-parent: every cell of a parent header row sits above a filled cell (or is merged).

    `lower` are the rows it must cover (the header row under it, or the first data rows), `below`
    every header row under it. Cells left of the first covered column are the stub zone ("State",
    "Année" above nothing); a leaf label with no header cell under it is fine over data."""
    covered = _body_profile(grid, lower, stub, right)
    if not covered:
        return False
    first = min(covered)
    for c in parent.cols:
        if c <= stub or c < first or c in covered or _in_merge(grid, parent.row, c):
            continue
        if all(not _covered(grid, r, c) for r in below) and c in data_cols:
            continue  # a leaf header: nothing under it in the header, data under it
        return False
    return True


def _same_header(grid: Grid, row: int, header_row: int, left: int, right: int) -> bool:
    """D3-rep: a row that repeats a header row (a stacked block's own header)."""
    a = {
        c: clean_text(grid.value(row, c))
        for c in range(left, right + 1)
        if grid.value(row, c) is not None
    }
    b = {
        c: clean_text(grid.value(header_row, c))
        for c in range(left, right + 1)
        if grid.value(header_row, c) is not None
    }
    if len(a) < 2 or set(a) != set(b):
        return False
    return sum(a[c] == b[c] for c in a) >= 0.9 * len(a)


def _first_data(grid: Grid, rows: list[int], data: list[int], left: int, right: int) -> int:
    """The first data line: a data row followed by data (or a section label, or nothing)."""
    for r in data:
        following = [x for x in rows if x > r][:1]
        if (
            not following
            or following[0] in data
            or is_single_label(grid, following[0], left, right)
        ):
            return r
    return data[0]


def _transposed(grid: Grid, rows: list[int], left: int, right: int) -> bool:
    """K4 (wide): distinct labels in the first column, one kind of value per row, the kinds differ
    between rows: the records run down the columns."""
    labels = [grid.value(r, left) for r in rows]
    if not all(isinstance(v, str) for v in labels) or len(set(map(str, labels))) < len(labels):
        return False
    if sum(is_period(v) for v in labels) * 2 >= len(labels):
        return False  # dates down the first column: a time series, not field names
    kinds = []
    for r in rows:
        found = {
            "n" if (k := kind(grid.value(r, c))) == "t" else k for c in range(left + 1, right + 1)
        }
        found -= {"e", "z"}
        if len(found) != 1:
            return False
        kinds.append(found.pop())
    return len(set(kinds)) >= 2 and right - left >= 2


def parse_region(
    grid: Grid, region: Region, template: set[int], csv: CsvShape | None
) -> tuple[Draft | None, list[int]]:
    """One table from the top of a region, and the rows left under it (another stacked table)."""
    left, right = region.left, region.right
    rows = region.rows
    stub = left
    width = right - left + 1
    stats = {r: row_stats(grid, r, left, right) for r in rows}
    preamble = set(csv.preamble) if csv else set()
    skipped = preamble | (set(csv.metadata) if csv else set())
    data = [
        r for r in rows if r not in skipped and (r in template or is_data(grid, stats[r], stub))
    ]
    multi = [r for r in rows if stats[r].n >= 2 and r not in skipped]
    if len(multi) >= 4 and _transposed(grid, multi[1:], left, right):
        data = multi[1:]  # K4: one record per column, the first row names the records
    text_table = not data
    if data:
        first = _first_data(grid, rows, data, left, right)
    else:
        multi = [r for r in rows if stats[r].n >= 2 and r not in skipped]
        if not multi or multi[0] == rows[-1]:
            return None, []
        first = next(r for r in rows if r > multi[0])
    upper = [r for r in rows if r < first]
    roles: dict[int, str] = {}
    # C7 right under the header: a section label before the first data line
    if (
        not text_table
        and len(upper) >= 2
        and stats[upper[-1]].n == 1
        and stats[upper[-2]].n >= 2
        and not is_note_text(_text(grid, stats[upper[-1]]))
    ):
        roles[upper[-1]] = "section"
        upper = upper[:-1]
    header: list[int] = []
    if text_table:
        header = [max(r for r in upper if stats[r].n >= 2)]
    else:
        metadata = [r for r in upper if csv and csv.metadata and r >= min(csv.metadata)]
        data_cols = _body_profile(grid, data[:10], left, right)
        for r in reversed(upper):
            row = stats[r]
            if r in preamble or len(header) == MAX_HEADER_ROWS:
                break
            if r in metadata:
                header.append(r)  # V2: column metadata between the names and the body
                continue
            if is_title_row(grid, row, stub, width) or (
                row.n == 1 and is_note_text(_text(grid, row))
            ):
                break
            named = [x for x in header if x not in metadata]
            if named:
                lower = [named[-1]]
                if stats[lower[0]].n * 2 < row.n:  # a sparse partial sub-header: look further down
                    fuller = [x for x in named[:-1] if stats[x].n * 2 >= row.n]
                    lower = fuller[-1:] or data[:5]
                if not _parent_ok(grid, row, lower, named, stub, right, data_cols):
                    break
            header.append(r)
        header.sort()
    titles = [r for r in upper if r not in header]
    body_rows = [r for r in rows if r >= first]
    stop = None
    for index, r in enumerate(body_rows):
        row = stats[r]
        if r in template:
            roles[r] = "template"
        elif text_table:
            if row.n >= 2 or (
                row.n == 1
                and index + 1 < len(body_rows)
                and stats[body_rows[index + 1]].n >= 2
                and not is_title_row(grid, row, stub, width)
            ):
                roles[r] = "data"
            else:
                stop = r
                break
        elif is_data(grid, row, stub):
            roles[r] = "data"
        elif any(_same_header(grid, r, h, left, right) for h in header) or (
            is_column_numbers(grid, row) and list(roles.values())[-1:] == ["repeated_header"]
        ):
            roles[r] = "repeated_header"
        elif row.n <= 1 or is_single_label(grid, r, left, right):
            ahead = body_rows[index + 1 : index + 1 + SECTION_LOOKAHEAD]
            follows = any(
                is_data(grid, stats[x], stub)
                or any(_same_header(grid, x, h, left, right) for h in header)
                for x in ahead
            )
            if follows and not is_note_text(_text(grid, row)):
                roles[r] = "section"
            else:
                stop = r
                break
        else:
            ahead = body_rows[index + 1 : index + 2]
            if (ahead and is_data(grid, stats[ahead[0]], stub)) or _fits_body(grid, row, roles):
                roles[r] = "data"
            else:
                stop = r
                break
    while roles and roles[max(roles)] in ("section", "repeated_header"):
        del roles[max(roles)]
    if not any(role in ("data", "template") for role in roles.values()):
        return None, []
    last = max(r for r, role in roles.items() if role in ("data", "template"))
    rest = [r for r in rows if stop is not None and r >= stop]
    if rest and not any(is_data(grid, stats[r], stub) for r in rest if r not in skipped):
        rest = []
    notes = [r for r in rows if r > last and r not in rest]
    draft = Draft(
        region, stub, header, titles, roles, text_table, notes=notes, hits=list(region.hits)
    )
    return draft, rest


def _fits_body(grid: Grid, row: RowStats, roles: dict[int, str]) -> bool:
    """A row whose cells have the kinds of the body above, column by column (a row of codes and
    text in a sales log is still a sale): at least 10 data rows seen, 80 % of its cells agree."""
    seen = [r for r, role in roles.items() if role == "data"][-50:]
    if len(seen) < 10 or row.n < 2:
        return False
    agree = 0
    for c in row.cols:
        kinds: dict[str, int] = {}
        for r in seen:
            k = row_stats(grid, r, c, c).kinds.get(c)
            if k is not None:
                k = "n" if k == "t" else k
                kinds[k] = kinds.get(k, 0) + 1
        mine = "n" if row.kinds[c] == "t" else row.kinds[c]
        agree += bool(kinds) and max(kinds, key=kinds.__getitem__) == mine
    return agree >= 0.8 * row.n


def _text(grid: Grid, row: RowStats) -> str:
    return clean_text(grid.value(row.row, row.cols[0])) if row.cols else ""


def _label_only(grid: Grid, region: Region) -> bool:
    return all(is_single_label(grid, r, region.left, region.right) for r in region.rows)


def _data_rows(grid: Grid, region: Region) -> list[int]:
    return [
        r
        for r in region.rows
        if is_data(grid, row_stats(grid, r, region.left, region.right), region.left)
    ]


def _has_header(grid: Grid, region: Region) -> bool:
    data = _data_rows(grid, region)
    first = data[0] if data else region.bottom + 1
    return any(
        len(row_stats(grid, r, region.left, region.right).texts) >= 2
        for r in region.rows
        if r < first
    )


def _profile(grid: Grid, region: Region) -> set[int]:
    data = _data_rows(grid, region)
    rows = data or [r for r in region.rows if len(grid.row_cols(r, region.left, region.right)) >= 2]
    return column_profile(grid, rows, region.left, region.right)


def _empty_between(grid: Grid, top: int, bottom: int, left: int, right: int) -> int:
    return sum(1 for r in range(top + 1, bottom) if not grid.row_cols(r, left, right))


def glue_regions(grid: Grid, regions: list[Region]) -> list[Region]:
    """D1-glue: a block without a header under a block with one, same columns, continues it."""
    result: list[Region] = []
    pending: list[Region] = []
    for region in regions:
        if result and _label_only(grid, region) and not pending:
            pending.append(region)
            continue
        target = result[-1] if result else None
        if target is not None:
            left, right = min(target.left, region.left), max(target.right, region.right)
            gap = _empty_between(grid, target.bottom, region.top, left, right)
            overlap = target.left <= region.right and region.left <= target.right
            starts_with_data = is_data(
                grid, row_stats(grid, region.top, region.left, region.right), region.left
            )
            if (
                overlap
                and gap <= GLUE_GAP + len(pending)
                and starts_with_data
                and _has_header(grid, target)
            ):
                a, b = _profile(grid, target), _profile(grid, region)
                if len(b) >= 2 and len(a & b) >= GLUE_PROFILE * len(b):
                    rows = sorted(
                        {*target.rows, *(r for p in pending for r in p.rows), *region.rows}
                    )
                    glued = Region(
                        rows=rows,
                        left=left,
                        right=right,
                        hits=[
                            *target.hits,
                            RuleHit(
                                "D1-glue",
                                0.0,
                                f"Rows {region.top + 1}+ continue the table above "
                                "across empty rows.",
                            ),
                        ],
                        gaps=sorted({*target.gaps, *region.gaps}),
                    )
                    result[-1] = glued
                    pending = []
                    continue
        result.extend(pending)
        pending = []
        if _label_only(grid, region) and result:
            pending.append(region)
        else:
            result.append(region)
    result.extend(pending)
    return result


def _style_key(grid: Grid, r: int, stub: int) -> tuple[object, ...]:
    value = grid.value(r, stub)
    dots = len(value) - len(value.lstrip(".")) if isinstance(value, str) else 0
    return ((r, stub) in grid.bold, grid.indent.get((r, stub), 0), indent_of(value), dots)


def _title_text(grid: Grid, rows: list[int]) -> tuple[str | None, list[str], str | None]:
    """Title (the nearest line above that is not a note), the other lines as notes, a unit."""
    texts: list[str] = []
    bold: list[str] = []
    for r in rows:
        line = " ".join(clean_text(v) for v in grid.values[r] if v is not None)
        if line:
            texts.append(line)
            if any((r, c) in grid.bold for c in grid.row_cols(r)):
                bold.append(line)
    candidates = [t for t in texts if not is_note_text(t)]
    styled = [t for t in candidates if t in bold]
    title = (styled or candidates or [None])[-1]
    notes = [t for t in texts if t != title]
    unit = next((t.strip("[]() ") for t in texts if is_unit_cell(t) and len(t) <= 80), None)
    return title, notes, unit


def finalize(
    grid: Grid, draft: Draft, extra_titles: list[int], extra_notes: list[int], csv: CsvShape | None
) -> DetectedTable:
    region = draft.region
    left, right = region.left, region.right
    hits = list(draft.hits)
    header_rows = draft.header_rows
    data_rows = [r for r, role in draft.roles.items() if role == "data"]
    body = draft.body
    used = [
        c for c in range(left, right + 1) if any(grid.occupied(r, c) for r in header_rows + body)
    ]
    left, right = used[0], used[-1]
    colnum = [r for r in header_rows if is_column_numbers(grid, row_stats(grid, r, left, right))]
    if colnum:
        hits.append(RuleHit("C6", 0.0, f"Row {colnum[0] + 1} numbers the columns: dropped."))
    metadata = [r for r in header_rows if csv and csv.metadata and r >= min(csv.metadata)]
    header: Header | None = None
    if header_rows:
        skip = set(colnum) | set(metadata)
        header = build_header(grid, header_rows, used, skip)
        header.unit_rows = sorted({*header.unit_rows, *metadata})
        if metadata and csv:
            for r in metadata:
                key = clean_text(grid.value(r, used[0])).rstrip(": ").lower()
                if key.startswith("unit"):
                    for c in used[1:]:
                        if grid.value(r, c) is not None:
                            header.units[c] = clean_text(grid.value(r, c))
            hits.append(
                RuleHit(
                    "V2",
                    0.0,
                    f"{len(metadata)} metadata rows ('Unit:', …) kept as column metadata.",
                )
            )
        naming = [r for r in header_rows if r not in skip and r not in header.unit_rows]
        if len(naming) > 1:
            hits.append(
                RuleHit("C4", 0.0, f"{len(naming)} header rows flattened as 'parent · child'.")
            )
        if header.unit_rows and not metadata:
            hits.append(
                RuleHit(
                    "C5",
                    0.0,
                    f"Row {header.unit_rows[0] + 1} holds units: kept as column metadata.",
                )
            )
        if header.refs:
            hits.append(
                RuleHit(
                    "C5-ref",
                    0.0,
                    f"Footnote references {', '.join(header.refs)} taken out of the names.",
                )
            )
    raw_names = [header.names.get(c, "") if header else "" for c in used]
    names = unique_names(raw_names)
    columns = [ColumnPlan(name=name, source_col=c + 1) for name, c in zip(names, used, strict=True)]
    by_col = {plan.source_col - 1: plan for plan in columns}
    if header:
        for c, unit in header.units.items():
            if c in by_col:
                by_col[c].unit = unit
    # totals (C8, C9)
    numeric_cols = [
        c
        for c in used
        if sum(
            (grid.value(r, c) is not None and not isinstance(grid.value(r, c), str))
            or _numeric_text(grid.value(r, c))
            for r in data_rows
        )
        >= 2
    ]
    breaks = {r for r, role in draft.roles.items() if role in ("section", "repeated_header")}
    style_key = {r: _style_key(grid, r, draft.stub) for r in body}
    totals = verify_totals(
        grid, [r for r in body if draft.roles[r] != "template"], numeric_cols, style_key, breaks
    )
    keyword_totals = [r for r in data_rows if total_keyword(grid, row_stats(grid, r, left, right))]
    for r, where in totals.items():
        hits.append(
            RuleHit(
                f"C9-{where}",
                0.0,
                f"Row {r + 1} is the sum of the rows "
                f"{'below' if where == 'top' else 'above'} it: removed as a total.",
            )
        )
    marked = [r for r in keyword_totals if r not in totals]
    if marked:
        hits.append(
            RuleHit(
                "C8",
                0.0,
                f"Row {marked[0] + 1} reads like a total (not checked by "
                "arithmetic): kept, marked.",
            )
        )
    clean_rows = [r for r in data_rows if r not in totals]
    for plan in columns:
        infer([grid.value(r, plan.source_col - 1) for r in clean_rows], plan)
    nulls = sum(plan.nulls for plan in columns)
    if nulls:
        hits.append(RuleHit("T2", 0.0, f"{nulls} placeholders (n/d, ---, …) read as empty."))
    unreadable = [p for p in columns if p.unreadable]
    if unreadable:
        hits.append(
            RuleHit(
                "T1-left",
                0.0,
                f"{sum(p.unreadable for p in unreadable)} value(s) of "
                "another type left empty in "
                f"{', '.join(repr(p.name) for p in unreadable[:3])}.",
            )
        )
    converted = [
        p.name
        for p in columns
        if p.dtype == "number"
        and any(isinstance(grid.value(r, p.source_col - 1), str) for r in clean_rows)
    ]
    if converted:
        hits.append(
            RuleHit(
                "T1",
                0.0,
                f"Numbers written as text read as numbers in "
                f"{len(converted)} column{'s' if len(converted) > 1 else ''}.",
            )
        )
    if any(p.flags for p in columns):
        hits.append(RuleHit("T3", 0.0, "Flags glued to the numbers split into a flag column."))
    first_number = next((i for i, p in enumerate(columns) if p.dtype == "number"), len(columns))
    for i, plan in enumerate(columns):
        if i < first_number and plan.dtype in ("text", "date"):
            plan.role = "key"
    # hierarchy (C10) and blank-filled labels (C12)
    label_col = next(
        (p.source_col - 1 for p in columns if p.role == "key" and p.dtype == "text"), draft.stub
    )
    levels, source, explicit = detect_levels(
        grid, clean_rows, label_col, header.names if header else {}
    )
    table_levels: dict[int, int] = {}
    if source:
        table_levels = levels
        if explicit is not None:
            by_col[explicit].role = "level"
        parents = has_parents(levels, clean_rows)
        hits.append(
            RuleHit(
                "C10",
                0.0,
                f"Indentation ({source}) gives {len(set(levels.values()))} "
                f"levels: 'level'{' and Level_1…' if parents else ''} columns.",
            )
        )
    transforms: list[Transform] = []
    key_cols = [p.source_col - 1 for p in columns if p.role == "key"]
    filled = fill_down_columns(grid, clean_rows, key_cols)
    if filled:
        fill_names = [by_col[c].name for c in filled]
        transforms.append(
            Transform(
                kind="fill_down",
                rule="C12",
                columns=fill_names,
                into=fill_names,
                message=f"Labels left blank under each group in "
                f"{', '.join(fill_names)}: filled down.",
                default=True,
            )
        )
    if csv and csv.combined and columns:
        transforms.append(
            Transform(
                kind="split_combined",
                rule="V3",
                columns=[columns[0].name],
                into=csv.combined,
                default=True,
                message=f"First column combines {', '.join(csv.combined)}: split on ','.",
            )
        )
    # restructurings (K)
    facts = Facts(grid, [r for r in clean_rows if r not in set(totals)], columns, header)
    hits.extend(derived_columns(facts))
    hits.extend(running_balance(facts))  # K5 (C1)
    kv = key_value(facts, bool(header_rows)) or transposed(facts)
    for proposal in (sparse_groups(facts), period_columns(facts), numbered_groups(facts), kv):
        if proposal is not None:
            transforms.append(proposal)
            hits.append(RuleHit(proposal.rule, 0.0, proposal.message))
    # titles and notes
    title, notes, unit = _title_text(grid, sorted({*extra_titles, *draft.title_rows}))
    below = [
        line
        for r in sorted({*draft.notes, *extra_notes})
        if (line := " ".join(clean_text(v) for v in grid.values[r] if v is not None))
    ]
    notes = [*notes, *below]
    if below:
        hits.append(
            RuleHit(
                "C2",
                0.0,
                f"{len(below)} note line{'s' if len(below) > 1 else ''} "
                "under the table kept as notes.",
            )
        )
    if draft.title_rows or extra_titles:
        hits.append(
            RuleHit(
                "C1",
                0.0,
                f"{len(draft.title_rows) + len(extra_titles)} title or "
                "preamble line(s) above the header kept as the name and description.",
            )
        )
    if csv and csv.preamble:
        hits.append(
            RuleHit(
                "V1", 0.0, f"{len(csv.preamble)} preamble rows narrower than the table skipped."
            )
        )
    sections: dict[int, str] = {}
    for r, role in sorted(draft.roles.items()):
        if role == "section":
            sections[r] = strip_refs(
                " ".join(clean_text(v) for v in grid.values[r] if v is not None)
            )[0]
    above = sorted({*extra_titles, *draft.title_rows})
    if any(role == "repeated_header" for role in draft.roles.values()) and above:
        # D3: the line right above the first header names the first block
        sections[draft.first_data] = strip_refs(_title_text(grid, above[-1:])[0] or "")[0]
        title, notes, unit = _title_text(grid, above[:-1])
        notes = [*notes, *below]
        hits.append(
            RuleHit(
                "D3",
                0.0,
                "Stacked blocks with the same header form one table with a 'Section' column.",
            )
        )
    elif sections:
        hits.append(
            RuleHit(
                "C7",
                0.0,
                f"{len(sections)} section label"
                f"{'s' if len(sections) > 1 else ''} inside the body: "
                "a 'Section' column.",
            )
        )
    uncached = sum(1 for (r, c) in grid.uncached if r in set(body) and left <= c <= right)
    naming_rows = [
        r
        for r in header_rows
        if r not in set(colnum) and (header is None or r not in header.unit_rows)
    ]
    facts_score = ScoreFacts(
        grid=grid,
        header_rows=naming_rows,
        body=clean_rows,
        cols=used,
        text_table=draft.text_table,
        key_value=kv is not None,
        uncached=uncached,
        verified_totals=len(totals),
        gaps=len(region.gaps),
        extra=[],
    )
    confidence, score_hits = score(facts_score)
    reasons = score_hits + hits
    confidence = round(min(1.0, max(0.0, confidence + sum(h.weight for h in hits))), 2)
    top = (header_rows or [draft.first_data])[0]
    bottom = draft.last_data
    dropped: dict[str, list[int]] = {}
    for r, role in draft.roles.items():
        if role in ("template", "section", "repeated_header"):
            dropped.setdefault(role, []).append(r + 1)
    if totals:
        dropped["total"] = sorted(r + 1 for r in totals)
    if colnum:
        dropped["column_numbers"] = [r + 1 for r in colnum]
    warnings = []
    if not data_rows and any(role == "template" for role in draft.roles.values()):
        warnings.append("Empty template: the rows hold only zeros (formulas), no data yet.")
    if uncached:
        warnings.append(
            f"{uncached} formula cells have no saved value (the file was written by a "
            "program, not Excel): they are left empty."
        )
    table = DetectedTable(
        id="",
        sheet=grid.name,
        top=top + 1,
        left=left + 1,
        bottom=bottom + 1,
        right=right + 1,
        header_rows=[r + 1 for r in header_rows],
        kind="key_value" if kv is not None else ("text" if draft.text_table else "table"),
        title=title,
        notes=notes,
        unit=unit,
        units={
            by_col[c].name: u for c, u in (header.units.items() if header else []) if c in by_col
        },
        unit_rows=[r + 1 for r in (header.unit_rows if header else [])],
        columns=columns,
        dropped_rows={k: sorted(v) for k, v in dropped.items()},
        sections={r + 1: label for r, label in sections.items()},
        levels={r + 1: level for r, level in table_levels.items()},
        hierarchy_col=label_col + 1 if table_levels else None,
        level_col=explicit + 1 if explicit is not None and table_levels else None,
        marked_totals=[r + 1 for r in marked],
        transforms=transforms,
        confidence=confidence,
        reasons=reasons,
        warnings=warnings,
        uncached=uncached,
        first_data_row=draft.first_data + 1,
    )
    return table


def _numeric_text(value: object) -> bool:
    return isinstance(value, str) and kind(value) == "t"


def merge_stacked(tables: list[DetectedTable], grid: Grid) -> list[DetectedTable]:
    """D3 across blocks: consecutive tables with the same columns form one table + 'Section'."""
    result: list[DetectedTable] = []
    for table in tables:
        previous = result[-1] if result else None
        if (
            previous is not None
            and previous.header_rows
            and table.header_rows
            and previous.left == table.left
            and previous.right == table.right
            and [c.name for c in previous.columns] == [c.name for c in table.columns]
            and previous.kind == table.kind == "table"
        ):
            if not previous.sections and previous.title:
                previous.sections[previous.first_data_row] = previous.title
                # the section title named the first block: the table takes the line above it
                earlier = [n for n in previous.notes if not is_note_text(n)]
                previous.title = earlier[-1] if earlier else None
                if previous.title:
                    previous.notes.remove(previous.title)
            previous.sections[table.first_data_row] = (
                table.title or f"Block {len(previous.sections) + 1}"
            )
            previous.dropped_rows.setdefault("repeated_header", []).extend(table.header_rows)
            between = [
                r
                for r in range(previous.bottom + 1, table.top)
                if any(v is not None for v in grid.values[r - 1])
            ]
            previous.dropped_rows.setdefault("section", []).extend(between)
            for key, rows in table.dropped_rows.items():
                previous.dropped_rows.setdefault(key, []).extend(rows)
            for key in previous.dropped_rows:
                previous.dropped_rows[key] = sorted(set(previous.dropped_rows[key]))
            previous.sections.update(table.sections)
            previous.levels.update(table.levels)
            previous.marked_totals.extend(table.marked_totals)
            previous.bottom = table.bottom
            previous.uncached += table.uncached
            if not any(hit.rule == "D3" for hit in previous.reasons):
                previous.reasons.append(
                    RuleHit(
                        "D3",
                        0.0,
                        "Stacked blocks with the same header "
                        "form one table with a 'Section' column.",
                    )
                )
            previous.confidence = round(max(previous.confidence, table.confidence), 2)
            continue
        result.append(table)
    return result


def detect_sheet(grid: Grid, csv: CsvShape | None = None) -> SheetLayout:
    sheet = SheetLayout(name=grid.name, warnings=list(grid.warnings), truncated=grid.truncated)
    sheet.first_row = next((r + 1 for r in range(grid.n_rows) if grid.row_cols(r)), 0)
    if grid.n_rows == 0:
        sheet.fallback = True
        return sheet
    template = template_rows(grid)
    parsed: list[tuple[Region, Draft | None]] = []
    for region in glue_regions(grid, find_regions(grid, template)):
        while region.rows:
            draft, rest = parse_region(grid, region, template, csv)
            used = [r for r in region.rows if r not in set(rest)]
            parsed.append(
                (Region(used, region.left, region.right, region.hits, region.gaps), draft)
            )
            if not rest or rest == region.rows:
                break
            region = Region(rest, region.left, region.right, list(region.hits), region.gaps)
    regions = [region for region, _ in parsed]
    drafts = [draft for _, draft in parsed]
    tables: list[DetectedTable] = []
    pending_titles: list[int] = []
    pending_notes: dict[int, list[int]] = {}
    last_index: int | None = None
    built: list[tuple[Draft, list[int]]] = []
    for region, draft in zip(regions, drafts, strict=True):
        if draft is None:
            lines = region.rows
            first_text = clean_text(grid.value(lines[0], grid.row_cols(lines[0])[0]))
            if built and (is_note_text(first_text) and lines[0] - built[-1][0].last_data <= 3):
                pending_notes.setdefault(len(built) - 1, []).extend(lines)
            else:
                pending_titles.extend(lines)
            continue
        built.append((draft, pending_titles))
        pending_titles = []
        last_index = len(built) - 1
    if pending_titles and last_index is not None:
        pending_notes.setdefault(last_index, []).extend(pending_titles)
    for index, (draft, titles) in enumerate(built):
        tables.append(finalize(grid, draft, titles, pending_notes.get(index, []), csv))
    tables = merge_stacked(tables, grid)
    for index, table in enumerate(tables):
        table.id = f"{grid.name}!{table.range}"
        if index and table.id in {t.id for t in tables[:index]}:
            table.id += f"#{index}"
    sheet.tables = tables
    sheet.fallback = not any(t.confidence >= REVIEW for t in tables)
    return sheet


_CACHE: OrderedDict[tuple[object, ...], Layout] = OrderedDict()
_CACHE_SIZE = 8


def detect_layout(
    path: str | Path, sheets: list[str] | None = None, limits: Limits = LIMITS
) -> Layout:
    """Every table of each sheet (or of a CSV), with its range, header, plan and confidence."""
    path = Path(path)
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns, tuple(sheets or ()), limits)
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]
    if path.suffix.lower() in CSV_EXTENSIONS:
        grid, shape = load_csv_grid(path, limits)
        layout = Layout(file=str(path), format="csv", section_name=path.stem)
        sheet = detect_sheet(grid, shape)
        sheet.name = ""
        for table in sheet.tables:
            table.sheet = ""
            table.id = table.range
        if grid.truncated and sheet.tables and sheet.tables[-1].bottom >= grid.n_rows:
            last = sheet.tables[-1]
            last.bottom = stream_csv_end(path, shape.delimiter, shape.encoding, grid.n_rows + 1)
            last.truncated = True
            last.id = last.range
            last.reasons.append(
                RuleHit(
                    "B1",
                    0.0,
                    f"File longer than {limits.max_rows} rows: the body end "
                    f"(row {last.bottom}) was found by streaming.",
                )
            )
        layout.sheets = [sheet]
        layout.delimiter = shape.delimiter
        layout.encoding = shape.encoding
    else:
        grids, notes = load_grids(path, sheets, limits)
        layout = Layout(file=str(path), format="excel", section_name=path.stem, warnings=notes)
        for grid in grids:
            sheet = detect_sheet(grid)
            if grid.truncated and sheet.tables:
                last = sheet.tables[-1]
                if last.bottom >= grid.n_rows:
                    last.bottom = stream_body_end(path, grid.name, grid.n_rows + 1, last.right)
                    last.truncated = True
                    last.reasons.append(
                        RuleHit(
                            "B1",
                            0.0,
                            f"Sheet longer than {limits.max_rows} "
                            f"rows: the body end (row {last.bottom}) was "
                            "found by streaming.",
                        )
                    )
                    last.id = f"{grid.name}!{last.range}"
            layout.sheets.append(sheet)
    _CACHE[key] = layout
    while len(_CACHE) > _CACHE_SIZE:
        _CACHE.popitem(last=False)
    return layout


def detect_range(
    path: str | Path, sheet: str, cell_range: str, limits: Limits = LIMITS
) -> DetectedTable:
    """080b: the table the user marked by hand ("A4:L14"): the same rules, on that range only.

    Cells outside the range are left out before the detection; the biggest table found in it is
    kept and its bounds widened to the range, so nothing the user marked is dropped silently."""
    try:
        min_col, min_row, max_col, max_row = range_boundaries(cell_range.replace("$", ""))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{cell_range!r} is not a range such as A4:L14.") from error
    if None in (min_col, min_row, max_col, max_row):
        raise ValueError(f"{cell_range!r} is not a range such as A4:L14.")
    assert min_col and min_row and max_col and max_row
    grids, _ = load_grids(Path(path), [sheet], limits)
    if not grids:
        raise ValueError(f"No sheet {sheet!r}.")
    grid = grids[0]
    grid.values = [
        [
            value if min_row - 1 <= r <= max_row - 1 and min_col - 1 <= c <= max_col - 1 else None
            for c, value in enumerate(row)
        ]
        for r, row in enumerate(grid.values)
    ]
    grid.uncached = {
        (r, c)
        for r, c in grid.uncached
        if min_row - 1 <= r <= max_row - 1 and min_col - 1 <= c <= max_col - 1
    }
    found = detect_sheet(grid).tables
    if not found:
        raise ValueError(f"No table found in {sheet}!{cell_range}: the range is empty.")
    table = max(found, key=lambda t: (t.bottom - t.top + 1) * (t.right - t.left + 1))
    if max_row > grid.n_rows:  # past the analysed window (B1): the user's bottom is kept
        table.bottom = max_row
        table.truncated = True
    table.reasons.insert(
        0,
        RuleHit("U1", 0.0, f"Range {cell_range.upper()} marked by hand: the rules ran on it only."),
    )
    table.id = f"{sheet}!{cell_range.upper().replace('$', '')}"
    return table


def find_table(path: str | Path, table_id: str) -> DetectedTable:
    """A table by its id ("Sheet!A4:L14"): a detected one, or else a range marked by hand (U1)."""
    sheet, _, cell_range = table_id.rpartition("!")
    layout = detect_layout(path, [sheet] if sheet else None)
    for table in layout.tables:
        if table.id == table_id:
            return table
    if not sheet or not cell_range:
        raise ValueError(f"No table {table_id!r} in {Path(path).name}.")
    return detect_range(path, sheet, cell_range.split("#")[0])
