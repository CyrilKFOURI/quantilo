"""Restructurings proposed for a detected table (K rules); `read(reshape=True)` applies them.

K1 marks a column equal to the sum of 2–12 contiguous columns as derived. K2 proposes the long
format for a sparse group of exclusive amounts (cash book accounts). K3 proposes the long format
for periods
across the columns (K3b keeps repeated sub-measures as columns), K3c for numbered groups
("Voix 1, Voix 2…"). K4 recognises a key-value block or a transposed table.
"""

import re
from collections import Counter
from dataclasses import dataclass

from quantilo.ingestion.layout.cells import clean_text, is_period, kind, numeric_value
from quantilo.ingestion.layout.grid import Grid
from quantilo.ingestion.layout.header import Header
from quantilo.ingestion.layout.model import ColumnPlan, RuleHit, Transform

DERIVED_SHARE = 0.95
DERIVED_ROWS = 10
DERIVED_ROWS_NAMED = 3
SPARSE_SHARE = 0.75
MAX_BALANCE_PAIRS = 12  # K5: columns among which a (money in, money out) pair is looked for
MIN_PERIODS = 4
_NUMBERED = re.compile(r"^(.*?\D)\s*(\d{1,3})$")
_TOTAL_NAME = re.compile(r"\b(total|sum|somme)\b", re.I)


@dataclass
class Facts:
    grid: Grid
    rows: list[int]  # body data rows, totals and templates excluded
    columns: list[ColumnPlan]
    header: Header | None


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= max(0.005 * abs(a), 1e-9)


def derived_columns(facts: Facts) -> list[RuleHit]:
    """K1: a column equal, row by row, to the sum of the contiguous columns just before it."""
    hits = []
    numeric = [p for p in facts.columns if p.dtype == "number"]
    grid = facts.grid
    values = {
        p.source_col: [numeric_value(grid.value(r, p.source_col - 1)) for r in facts.rows]
        for p in numeric
    }
    for index, plan in enumerate(numeric):
        target = values[plan.source_col]
        needed = DERIVED_ROWS_NAMED if _TOTAL_NAME.search(plan.name) else DERIVED_ROWS
        for width in range(2, min(12, index) + 1):
            group = numeric[index - width : index]
            if any(p.role == "derived" for p in group):
                continue
            checked = ok = nontrivial = 0
            for i, total in enumerate(target):
                parts = [values[p.source_col][i] for p in group]
                if total is None and all(v is None for v in parts):
                    continue
                checked += 1
                expected = sum(v or 0.0 for v in parts)
                ok += _close(total or 0.0, expected)
                nontrivial += any(v not in (None, 0) for v in parts)
            if checked and ok / checked >= DERIVED_SHARE and nontrivial >= needed:
                plan.role = "derived"
                names = ", ".join(p.name for p in group)
                hits.append(
                    RuleHit(
                        "K1",
                        0.0,
                        f"'{plan.name}' is the sum of {names} on "
                        f"{ok / checked:.0%} of the rows: marked derived.",
                    )
                )
                break
    return hits


def running_balance(facts: Facts) -> list[RuleHit]:
    """K5 (C1): a running balance. A column whose value is, row after row, the previous one plus
    an amount column (`balance = previous + amount`) or plus one column and minus another (money
    in, money out) on 95 % of the rows: marked derived, like a sum of columns (K1), and said in
    the table's reasons. It is computed from the others: a long format leaves it out."""
    hits = []
    numeric = [p for p in facts.columns if p.dtype == "number"]
    grid = facts.grid
    values = {
        p.source_col: [numeric_value(grid.value(r, p.source_col - 1)) for r in facts.rows]
        for p in numeric
    }
    for plan in numeric:
        if plan.role == "derived":  # a sum of columns (K1) is no balance
            continue
        balance = values[plan.source_col]
        others = [p for p in numeric if p is not plan]  # a total of money in counts (K1)
        pairs: list[tuple[ColumnPlan, ColumnPlan | None]] = [(p, None) for p in others]
        # money in, money out: any pair among a few columns, else the totals (K1) only
        among = (
            others
            if len(others) <= MAX_BALANCE_PAIRS
            else [p for p in others if p.role == "derived"]
        )
        pairs += [(a, b) for a in among for b in among if a is not b]
        for plus, minus in pairs:
            checked = ok = moved = 0
            for i in range(1, len(balance)):
                before, now = balance[i - 1], balance[i]
                add = values[plus.source_col][i]
                take = values[minus.source_col][i] if minus is not None else None
                if before is None or now is None or (add is None and take is None):
                    continue
                checked += 1
                change = (add or 0.0) - (take or 0.0)
                ok += _close(now, before + change)
                moved += change != 0
                if checked == DERIVED_ROWS and ok < checked - 1:  # cheap early exit
                    break
            enough = checked >= DERIVED_ROWS and moved >= DERIVED_ROWS_NAMED
            if enough and ok / checked >= DERIVED_SHARE:
                plan.role = "derived"
                how = f"plus '{plus.name}'" + (f" minus '{minus.name}'" if minus else "")
                hits.append(
                    RuleHit(
                        "K5",
                        0.0,
                        f"'{plan.name}' is a running balance (the row above {how}) on "
                        f"{ok / checked:.0%} of the rows: marked derived.",
                    )
                )
                break
    return hits


def header_groups(facts: Facts) -> list[tuple[str, list[ColumnPlan]]]:
    """Top-level header labels that span several columns (by merge or by spreading)."""
    header = facts.header
    if header is None or not any(len(parts) > 1 for parts in header.parts.values()):
        return []
    groups: list[tuple[str, list[ColumnPlan]]] = []
    for plan in facts.columns:
        parts = header.parts.get(plan.source_col - 1, [])
        top = parts[0] if len(parts) > 1 and any(parts[1:]) else ""
        if top and groups and groups[-1][0] == top:
            groups[-1][1].append(plan)
        elif top:
            groups.append((top, [plan]))
        else:
            groups.append(("", [plan]))
    return [(name, plans) for name, plans in groups if name and len(plans) >= 2]


def sparse_groups(facts: Facts) -> Transform | None:
    """K2: a header group of amounts where (totals aside) a row fills at most one column."""
    grid = facts.grid
    groups: dict[str, list[str]] = {}
    shares = []
    for name, plans in header_groups(facts):
        amounts = [
            p
            for p in plans
            if p.role != "derived"
            and (
                p.dtype == "number"
                or all(grid.value(r, p.source_col - 1) is None for r in facts.rows)
            )
        ]
        if len(amounts) < 2:
            continue
        filled_rows = []
        for r in facts.rows:
            filled = sum(
                numeric_value(grid.value(r, p.source_col - 1)) not in (None, 0) for p in amounts
            )
            if filled:
                filled_rows.append(filled)
        if len(filled_rows) < 3:
            continue
        exclusive = sum(1 for n in filled_rows if n <= 1) / len(filled_rows)
        used = sum(
            any(numeric_value(grid.value(r, p.source_col - 1)) not in (None, 0) for r in facts.rows)
            for p in amounts
        )
        if exclusive >= SPARSE_SHARE and used >= 2:
            groups[name] = [p.name for p in amounts]
            shares.append(f"{name} {exclusive:.0%}")
    if not groups:
        return None
    into = [f"{name} type" for name in groups] + ["Amount"]
    return Transform(
        kind="melt_sparse",
        rule="K2",
        columns=[column for names in groups.values() for column in names],
        into=into,
        groups=groups,
        message=(
            f"Rows fill one column of {', '.join(groups)} at most ({', '.join(shares)} of the "
            f"rows): long format {' + '.join(into)}, one row per amount."
        ),
    )


def period_columns(facts: Facts) -> Transform | None:
    """K3: at least 4 period labels across the columns at one header level (K3b: sub-measures)."""
    header = facts.header
    if header is None:
        return None
    depth = max((len(parts) for parts in header.parts.values()), default=0)
    for level in range(depth):
        periods = [
            p
            for p in facts.columns
            if p.role not in ("level", "section")
            and level < len(raw := header.raw.get(p.source_col - 1, []))
            and raw[level] is not None
            and is_period(raw[level])
        ]
        labels = {clean_text(header.parts[p.source_col - 1][level]) for p in periods}
        if len(labels) < MIN_PERIODS:
            continue
        below = [
            " · ".join(x for x in header.parts[p.source_col - 1][level + 1 :] if x) for p in periods
        ]
        measures = sorted({m for m in below if m}, key=below.index)
        if len(measures) >= 2 and all(Counter(below)[m] >= 2 for m in measures):
            rule, into = "K3b", ["Period", *measures]
        else:
            above = {tuple(header.parts[p.source_col - 1][:level]) for p in periods}
            value = " · ".join(x for x in next(iter(above)) if x) if len(above) == 1 else ""
            rule, into = "K3", ["Period", value or "Value"]
            below = [into[1]] * len(periods)
        return Transform(
            kind="melt_periods",
            rule=rule,
            columns=[p.name for p in periods],
            into=into,
            message=f"{len(labels)} periods across the columns: long format {' + '.join(into)}.",
            groups={
                p.name: [header.parts[p.source_col - 1][level], measure]
                for p, measure in zip(periods, below, strict=True)
            },
        )
    return None


def numbered_groups(facts: Facts) -> Transform | None:
    """K3c: numbered groups of columns ("Nuance candidat 1", "Voix 1", … "Voix 2")."""
    stems: dict[str, list[str]] = {}
    for plan in facts.columns:
        match = _NUMBERED.match(plan.name)
        if match:
            stems.setdefault(match.group(1).strip(), []).append(match.group(2))
    kept = {stem: numbers for stem, numbers in stems.items() if len(set(numbers)) >= 3}
    if len(kept) < 2:
        return None
    columns = [
        p.name
        for p in facts.columns
        if (m := _NUMBERED.match(p.name)) and m.group(1).strip() in kept
    ]
    count = max(len(set(n)) for n in kept.values())
    groups = {}
    for name in columns:
        match = _NUMBERED.match(name)
        if match:
            groups[name] = [match.group(2), match.group(1).strip()]
    return Transform(
        kind="melt_numbered",
        rule="K3c",
        columns=columns,
        into=["n", *kept],
        message=f"{len(kept)} column groups numbered 1 to {count}: long format with 'n'.",
        groups=groups,
    )


def key_value(facts: Facts, has_header: bool) -> Transform | None:
    """K4: labels in the first column and values of mixed types beside them: a record, not rows."""
    columns = facts.columns
    grid = facts.grid
    if not 2 <= len(columns) <= 4 or len(facts.rows) < 2:
        return None
    first = columns[0].source_col - 1
    labels = [grid.value(r, first) for r in facts.rows]
    texts = [clean_text(v) for v in labels if isinstance(v, str)]
    if len(texts) < 0.8 * len(labels) or len(set(texts)) < 0.9 * len(texts):
        return None
    if sum(is_period(t) for t in texts) * 2 >= len(texts):
        return None  # dates down the first column: a time index, not field names
    mixed = False
    for plan in columns[1:]:
        kinds = Counter(kind(grid.value(r, plan.source_col - 1)) for r in facts.rows)
        kinds.pop("e", None)
        kinds.pop("z", None)
        total = sum(kinds.values())
        if total and len([k for k, n in kinds.items() if n >= 0.2 * total]) >= 2:
            mixed = True
    if not mixed and has_header:
        return None
    return Transform(
        kind="transpose",
        rule="K4",
        columns=[p.name for p in columns],
        into=[clean_text(v) for v in texts],
        message="Labels in the first column, one value beside each: a key-value record "
        "(transpose it, never unpivot it).",
    )


def transposed(facts: Facts) -> Transform | None:
    """K4 (wide form): types agree along the rows, not down the columns: one record per column."""
    columns = facts.columns
    grid = facts.grid
    if len(columns) < 3 or len(facts.rows) < 3:
        return None
    first = columns[0].source_col - 1
    labels = [grid.value(r, first) for r in facts.rows]
    if not all(isinstance(v, str) for v in labels) or len(set(map(str, labels))) < len(labels):
        return None

    def kinds_of(cells: list[object]) -> set[str]:
        found = {kind(v) for v in cells} - {"e", "z"}
        return {"n" if k == "t" else k for k in found}

    rows_single = sum(
        len(kinds_of([grid.value(r, p.source_col - 1) for p in columns[1:]])) <= 1
        for r in facts.rows
    )
    cols_single = sum(
        len(kinds_of([grid.value(r, p.source_col - 1) for r in facts.rows])) <= 1
        for p in columns[1:]
    )
    if rows_single >= 0.8 * len(facts.rows) and cols_single <= 0.5 * (len(columns) - 1):
        return Transform(
            kind="transpose",
            rule="K4",
            columns=[p.name for p in columns],
            into=[clean_text(v) for v in labels],
            message="Each row holds one kind of value and each column mixes them: one record per "
            "column, transpose it.",
        )
    return None
