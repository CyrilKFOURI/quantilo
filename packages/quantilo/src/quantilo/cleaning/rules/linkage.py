"""Advanced rules on whole rows (B1, study § 2.2):

- A-16: near-duplicates by Fellegi–Sunter (Splink's model, in numpy): pairs of rows that share an
  anchor (C-26: an event date or a key shared by rows, the block), each other column compared at
  three levels (equal, close: Jaro–Winkler ≥ 0.92 or within 1 %, different), the columns the
  anchor determines left out; *u* (how often two rows taken at random agree) on random pairs, *m*
  (how often a true duplicate agrees) and the share of duplicates by EM on the blocked pairs.
  Pairs at a probability ≥ 0.95, with one slip at most (every column equal but one, close), and
  not equal once normalized (A-15's), form groups; the most complete row of each is kept
  (`drop_duplicates` with the groups listed by row, N35: reviewed group by group).
- A-17: rows with more than half their cells empty (not every cell: B-15), when most rows are
  nearly full.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

from quantilo import semantics
from quantilo.cleaning import model as m
from quantilo.cleaning.compile import as_text, is_number, is_temporal, is_text
from quantilo.cleaning.rules import basic
from quantilo.cleaning.rules.basic import Column, n, q
from quantilo.ingestion.sql import ident

if TYPE_CHECKING:
    from quantilo.cleaning.rules.advanced import Advanced

COMPARED = 10  # columns compared at most
BLOCK = 50  # rows of one anchor value at most (a larger block is a category, not a record)
PAIRS = 500_000
RANDOM_PAIRS = 20_000
CLOSE_TEXT = 0.92  # Jaro–Winkler
CLOSE_NUMBER = 0.01
MATCH = 0.95
TRAIT = 0.9  # a column equal on this share of the blocked pairs is a trait of the anchor
ITERATIONS = 40
DUPLICATE_SHARE = 0.05  # A-16: at most this share of the rows are dropped
SPARSE = 0.5  # A-17: more than half the cells empty
SPARSE_ROWS = 0.1  # … on at most this share of the rows
FULL_MEDIAN = 0.25  # … while the median row has at most this share empty
SEED = 97
UNIQUE = 0.95  # N17: a column with this share of distinct values names a row: a new row, a new id
TIME_WORDS = {"time", "hour", "heure", "horaire"}
PARCEL_WORDS = {"parcel", "parcelle", "parcelles", "cadastre", "cadastral"}


def _level_sql(column: Column, a: str, b: str) -> str:
    """2 equal, 1 close, 0 different, NULL when one side is empty."""
    left, right = f"{a}.{ident(column.name)}", f"{b}.{ident(column.name)}"
    if is_text(column.dtype):
        norm_a = f"lower(strip_accents(trim({left})))"
        norm_b = f"lower(strip_accents(trim({right})))"
        return (
            f"CASE WHEN nullif(trim({left}), '') IS NULL OR nullif(trim({right}), '') IS NULL "
            f"THEN NULL WHEN {norm_a} = {norm_b} THEN 2 WHEN jaro_winkler_similarity({norm_a}, "
            f"{norm_b}) >= {CLOSE_TEXT} THEN 1 ELSE 0 END"
        )
    if is_number(column.dtype):
        return (
            f"CASE WHEN {left} IS NULL OR {right} IS NULL THEN NULL WHEN {left} = {right} THEN 2 "
            f"WHEN abs({left} - {right}) <= {CLOSE_NUMBER} * greatest(abs({left}), abs({right})) "
            "THEN 1 ELSE 0 END"
        )
    return (
        f"CASE WHEN {left} IS NULL OR {right} IS NULL THEN NULL WHEN CAST({left} AS VARCHAR) = "
        f"CAST({right} AS VARCHAR) THEN 2 ELSE 0 END"
    )


def _columns(advanced: Advanced) -> tuple[Column | None, list[Column]]:
    from quantilo.cleaning.rules.advanced import _anchor, _trivial

    context = advanced.context
    usable = [
        c
        for c in context.columns
        if c.nonnull >= 0.5 * context.rows
        and not _trivial(c, context.rows)
        and (is_text(c.dtype) or is_number(c.dtype) or is_temporal(c.dtype))
    ]
    anchors = [c for c in usable if _anchor(advanced, c) and c.distinct <= 0.5 * context.rows]
    if not anchors:
        return None, []
    anchor = max(anchors, key=lambda c: c.distinct)  # the most selective block
    compared = sorted((c for c in usable if c is not anchor), key=lambda c: -c.distinct)
    return anchor, compared[:COMPARED]


def _patterns(
    advanced: Advanced, sql: str, compared: list[Column]
) -> tuple[np.ndarray, np.ndarray]:
    """The agreement patterns of the pairs (−1: empty) and how many pairs have each."""
    rows = advanced.context.connection.execute(
        f"SELECT {', '.join(f'coalesce(g{i}, -1)' for i in range(len(compared)))}, count(*) "
        f"FROM ({sql}) GROUP BY ALL"
    ).fetchall()
    if not rows:
        return np.zeros((0, len(compared)), dtype=np.int64), np.zeros(0)
    data = np.array(rows, dtype=np.int64)
    return data[:, :-1], data[:, -1].astype(float)


def _probabilities(patterns: np.ndarray, counts: np.ndarray, u: np.ndarray) -> np.ndarray:
    """EM with u fixed: the probability that each pattern is a duplicate."""
    columns = patterns.shape[1]
    mk = np.tile(np.array([0.05, 0.10, 0.85]), (columns, 1))
    share = 0.05
    probability = np.zeros(len(patterns))
    for _ in range(ITERATIONS):
        log_m = np.zeros(len(patterns))
        log_u = np.zeros(len(patterns))
        for k in range(columns):
            seen = patterns[:, k] >= 0
            levels = patterns[seen, k]
            log_m[seen] += np.log(mk[k, levels])
            log_u[seen] += np.log(u[k, levels])
        top = np.log(share) + log_m
        bottom = np.log1p(-share) + log_u
        probability = 1 / (1 + np.exp(np.clip(bottom - top, -50, 50)))
        weight = probability * counts
        share = float(np.clip(weight.sum() / counts.sum(), 1e-6, 0.5))
        for k in range(columns):
            seen = patterns[:, k] >= 0
            total = weight[seen].sum()
            if total <= 0:
                continue
            for level in range(3):
                mk[k, level] = (weight[seen & (patterns[:, k] == level)].sum() + 0.01) / (
                    total + 0.03
                )
    return probability


def fellegi_sunter(advanced: Advanced) -> None:
    from quantilo.cleaning.rules.advanced import SAMPLE, row_label, scaled

    context = advanced.context
    anchor, compared = _columns(advanced)
    if anchor is None or len(compared) < 3 or advanced.sample_rows < 20:
        return
    connection = context.connection
    levels = ", ".join(f"{_level_sql(c, 'a', 'b')} AS g{i}" for i, c in enumerate(compared))
    key = ident(anchor.name)
    blocks = (
        f"SELECT {key} FROM {SAMPLE} WHERE {key} IS NOT NULL GROUP BY 1 "
        f"HAVING count(*) BETWEEN 2 AND {BLOCK}"
    )
    pairs = (
        f"SELECT a.__row AS ra, b.__row AS rb, {levels} FROM {SAMPLE} a JOIN {SAMPLE} b "
        f"ON a.{key} = b.{key} AND a.__row < b.__row WHERE a.{key} IN ({blocks}) LIMIT {PAIRS}"
    )
    random = (
        f"SELECT {levels} FROM (SELECT *, row_number() OVER (ORDER BY hash(__row + {SEED})) AS i "
        f"FROM {SAMPLE}) a JOIN (SELECT *, row_number() OVER (ORDER BY hash(__row + "
        f"{SEED + 1})) AS i FROM {SAMPLE}) b ON a.i = b.i AND a.__row <> b.__row LIMIT "
        f"{RANDOM_PAIRS}"
    )
    connection.execute(f"CREATE OR REPLACE TEMP TABLE quantilo_pairs AS {pairs}")
    try:
        patterns, counts = _patterns(advanced, "SELECT * FROM quantilo_pairs", compared)
        if not len(patterns):
            return
        random_patterns, random_counts = _patterns(advanced, random, compared)
        u = np.full((len(compared), 3), 1.0)
        for k in range(len(compared)):
            for level in range(3):
                u[k, level] += random_counts[random_patterns[:, k] == level].sum()
            u[k] /= u[k].sum()
        # columns the anchor determines (a customer's name in the customer's block) agree on
        # every blocked pair: they say nothing about two rows being one record, leave them out
        known = patterns >= 0
        agree = [
            counts[known[:, k] & (patterns[:, k] == 2)].sum() / max(counts[known[:, k]].sum(), 1)
            for k in range(len(compared))
        ]
        model = [k for k in range(len(compared)) if agree[k] < TRAIT]
        if len(model) < 3:
            return
        projected, inverse = np.unique(patterns[:, model], axis=0, return_inverse=True)
        weights = np.bincount(inverse.ravel(), weights=counts)
        probability = _probabilities(projected, weights, u[model])[inverse.ravel()]
        fuzzy = ((patterns < 2) & known).any(axis=1)  # not equal once normalized (A-15's)
        shown = patterns[:, model]
        # one slip at most: every compared column equal but one, which is close (a typo, a
        # rounding); two order lines of related products differ in two (code and label)
        slips = (shown == 1).sum(axis=1)
        agreeing = (slips <= 1) & ~(shown == 0).any(axis=1) & ((shown == 2).sum(axis=1) >= 2)
        matched = (probability >= MATCH) & fuzzy & agreeing
        if not matched.any():
            return
        tests = [
            " AND ".join(f"coalesce(g{k}, -1) = {int(level)}" for k, level in enumerate(pattern))
            for pattern in patterns[matched]
        ]
        found = connection.execute(
            f"SELECT ra, rb FROM quantilo_pairs WHERE ({') OR ('.join(tests)}) ORDER BY ra, rb"
        ).fetchall()
        lowest = float(probability[matched].min())
    finally:
        connection.execute("DROP TABLE IF EXISTS quantilo_pairs")
    found = _one_record(advanced, [(int(a), int(b)) for a, b in found])
    if not found:
        return
    groups = _groups(found)
    filled = _filled(advanced, sorted({r for group in groups for r in group}))
    dropped: list[int] = []
    ordered: list[list[int]] = []  # N35: each group, the row kept first (the rule's order)
    examples: list[tuple[str, str]] = []
    for group in groups:
        keep = max(sorted(group), key=lambda r: filled.get(r, 0))  # the first of the fullest
        others = [r for r in sorted(group) if r != keep]
        dropped += others
        ordered.append([keep, *sorted(others, key=lambda r: (-filled.get(r, 0), r))])
        if len(examples) < 3:
            examples.append((row_label(others[0]), f"same record as {row_label(keep)}"))
    if not dropped or len(dropped) > DUPLICATE_SHARE * context.rows or len(dropped) > m.MAX_MAP:
        return
    differing = _differing(advanced, groups[0], compared)
    chance = "over 99.9 %" if lowest > 0.999 else f"at least {lowest:.1%}"
    reason = (
        f"{n(len(dropped))} rows look like another row with the same {anchor.name}, apart from "
        f"small differences ({differing}): each pair is a duplicate with a probability of "
        f"{chance} (Fellegi–Sunter: columns compared equal, close or different, "
        "weighted by how often rows taken at random agree). The most complete row of each group "
        "is kept. Check a few groups before ticking: two real records can look alike."
    )
    if advanced.sampled:
        reason += f" Found on a sample of {n(advanced.sample_rows)} rows."
    advanced.add(
        "A-16",
        "S",
        [],
        f"Remove {scaled(advanced, len(dropped))} likely duplicate rows",
        reason,
        m.DropDuplicates(id="s0", groups=ordered, keep="most_complete"),
        rows=len(dropped),
        examples=examples,
        key="linkage",
    )


def _telling(advanced: Advanced, column: Column) -> bool:
    """N17: a column on which two rows that differ are two records, never one typed twice: a code
    (`StockCode`, `code_nature_culture`), a repeated identifier (a parcel, a permit's sequence
    number), a time of day (air quality's `Time`)."""
    view = column.view
    words = set(semantics.tokens(column.name))
    if (view is not None and view.role == "hour") or (words & TIME_WORDS and "date" not in words):
        return True
    if words & PARCEL_WORDS:
        return True
    code = semantics.code_by_name(column.name)
    if code is None:
        return False
    return code.kind == "code" or column.distinct < UNIQUE * max(column.nonnull, 1)


def _one_record(advanced: Advanced, pairs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """N17: the pairs that differ on one column at most, all columns read (not only the compared
    ones; a column naming each row apart), and never on a telling column (`_telling`): air quality
    rows of two hours, dvf lines of two crops of a parcel, nyc_dob permits of two sequences,
    online_retail items of two stock codes."""
    from quantilo.cleaning.rules.advanced import SAMPLE

    if not pairs:
        return []
    context = advanced.context
    columns = [
        c
        for c in context.columns
        if c.nonnull and not (c.distinct >= UNIQUE * c.nonnull and c.distinct > 1)
    ]
    if not columns:
        return pairs
    tests = []
    for column in columns:
        left, right = f"a.{ident(column.name)}", f"b.{ident(column.name)}"
        if is_text(column.dtype):
            left = f"nullif(lower(strip_accents(trim({left}))), '')"
            right = f"nullif(lower(strip_accents(trim({right}))), '')"
        tests.append(f"CAST({left} IS DISTINCT FROM {right} AS INTEGER)")
    telling = [t for t, c in zip(tests, columns, strict=True) if _telling(advanced, c)]
    values = ", ".join(f"({a}, {b})" for a, b in pairs)
    rows = context.connection.execute(
        f"SELECT p.ra, p.rb, {' + '.join(tests)}, "
        f"{' + '.join(telling) if telling else '0'} "
        f"FROM (VALUES {values}) p(ra, rb) JOIN {SAMPLE} a ON a.__row = p.ra "
        f"JOIN {SAMPLE} b ON b.__row = p.rb"
    ).fetchall()
    return sorted((int(a), int(b)) for a, b, differ, told in rows if differ <= 1 and not told)


def _groups(pairs: list[tuple[int, int]]) -> list[list[int]]:
    parent: dict[int, int] = {}

    def find(row: int) -> int:
        parent.setdefault(row, row)
        while parent[row] != row:
            parent[row] = parent[parent[row]]
            row = parent[row]
        return row

    for a, b in pairs:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)
    groups: dict[int, list[int]] = defaultdict(list)
    for row in list(parent):
        groups[find(row)].append(row)
    return sorted((sorted(g) for g in groups.values()), key=lambda g: g[0])


def _filled(advanced: Advanced, rows: list[int]) -> dict[int, int]:
    from quantilo.cleaning.rules.advanced import SAMPLE

    if not rows:
        return {}
    context = advanced.context
    count = " + ".join(f"CASE WHEN {basic._empty(c)} THEN 0 ELSE 1 END" for c in context.columns)
    found = context.connection.execute(
        f"SELECT __row, {count} FROM {SAMPLE} WHERE __row IN ({', '.join(map(str, rows))})"
    ).fetchall()
    return {int(r): int(c) for r, c in found}


def _differing(advanced: Advanced, group: list[int], compared: list[Column]) -> str:
    from quantilo.cleaning.rules.advanced import SAMPLE

    texts = ", ".join(as_text(c.name, c.dtype) for c in compared)
    rows = advanced.context.connection.execute(
        f"SELECT {texts} FROM {SAMPLE} WHERE __row IN ({group[0]}, {group[1]}) ORDER BY __row"
    ).fetchall()
    if len(rows) < 2:
        return "a few columns"
    parts = [
        f"{c.name} {q(str(a))} / {q(str(b))}"
        for c, a, b in zip(compared, rows[0], rows[1], strict=True)
        if a != b
    ]
    return ", ".join(parts[:2]) or "a few columns"


# --- A-17 ----------------------------------------------------------------------------------------


def mostly_empty(advanced: Advanced) -> None:
    from quantilo.cleaning.rules.advanced import SAMPLE, row_label

    context = advanced.context
    columns = context.columns
    if len(columns) < 4 or context.rows < 20:
        return
    empty = " + ".join(f"CASE WHEN {basic._empty(c)} THEN 1 ELSE 0 END" for c in columns)
    width = len(columns)
    sparse = f"e > {SPARSE} * {width} AND e < {width}"
    count, median, last_full, last_sparse = context.one(
        f"SELECT count(*) FILTER (WHERE {sparse}), median(e) / {width}, "
        f"max(r) FILTER (WHERE e <= {SPARSE} * {width}), max(r) FILTER (WHERE {sparse}) "
        f"FROM (SELECT {empty} AS e, file_row_number AS r FROM {advanced.rows_source})"
    )
    count = int(count)
    if not count or count > SPARSE_ROWS * context.rows or float(median) > FULL_MEDIAN:
        return
    # N17: the rows after the last full one are data still to come (eco2mix's forecasts), not
    # rows cut short: left out, and said
    later, full = 0, int(last_full) if last_full is not None else -1
    if last_full is not None and last_sparse is not None and last_sparse > last_full:
        (later,) = context.one(
            f"SELECT count(*) FROM (SELECT {empty} AS e, file_row_number AS r "
            f"FROM {advanced.rows_source}) WHERE {sparse} AND r > {full}"
        )
        later = int(later)
    if later == count:
        return
    before = f" AND __row < {full}" if later else ""
    listed = context.connection.execute(
        f"SELECT __row, {empty} FROM {SAMPLE} WHERE ({empty}) > {SPARSE} * {width} "
        f"AND ({empty}) < {width}{before} ORDER BY __row LIMIT 3"
    ).fetchall()
    if later:
        _sparse_before(advanced, count - later, later, width, float(median), listed, full)
        return
    advanced.add(
        "A-17",
        "S",
        [],
        f"Remove {n(count)} mostly empty rows",
        f"{n(count)} rows have more than half their {width} cells empty, while the median row "
        f"has {float(median):.0%} empty: rows cut short, or notes typed in the table? Counted "
        "before the Basic fixes (a word for 'missing' is not empty yet). Removed, they are kept "
        "in the original.",
        m.DropEmptyRows(id="s0", share=SPARSE),
        rows=count,
        examples=[(row_label(int(r)), f"{int(e)} of {width} cells empty") for r, e in listed],
        key="sparse",
    )


def _sparse_before(
    advanced: Advanced,
    count: int,
    later: int,
    width: int,
    median: float,
    listed: list[tuple[int, int]],
    last_full: int,
) -> None:
    """N17: the mostly empty rows above the last full one, by number; the last rows, empty
    because their data is still to come, kept."""
    from quantilo.cleaning.rules.advanced import row_label

    context = advanced.context
    empty = " + ".join(f"CASE WHEN {basic._empty(c)} THEN 1 ELSE 0 END" for c in context.columns)
    rows = [
        int(r)
        for (r,) in context.connection.execute(
            f"SELECT r FROM (SELECT {empty} AS e, file_row_number AS r FROM "
            f"{advanced.rows_source}) WHERE e > {SPARSE} * {width} AND e < {width} "
            f"AND r < {last_full} ORDER BY r"
        ).fetchall()
    ]
    if not rows or len(rows) > m.MAX_MAP:
        return
    advanced.add(
        "A-17",
        "S",
        [],
        f"Remove {n(count)} mostly empty rows",
        f"{n(count)} rows have more than half their {width} cells empty, while the median row "
        f"has {median:.0%} empty: rows cut short, or notes typed in the table? The last "
        f"{n(later)} rows, after the last full one, are empty the same way: data still to come "
        "(forecasts, a month not over), kept. Counted before the Basic fixes. Removed, they are "
        "kept in the original.",
        m.DropRows(id="s0", rows=rows),
        rows=count,
        examples=[(row_label(int(r)), f"{int(e)} of {width} cells empty") for r, e in listed],
        key="sparse",
    )
