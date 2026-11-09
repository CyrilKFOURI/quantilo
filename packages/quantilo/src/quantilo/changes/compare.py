"""What changed between two versions, read from their photos alone (`snapshot.py`): schema
(with guessed renames), quality, dates, distributions, and an overall level.

Distributions are ranked by effect size (PSI), never by p-value: on a million rows every
difference is "significant". KS and χ² are given for those who want them.
"""

import difflib
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Literal

import numpy as np

from quantilo.changes.metrics import (
    PSI_MODERATE,
    PSI_SOURCE,
    Drift,
    cdf_at,
    chi_square,
    drift_of,
    ks_from_quantiles,
    normalised_wasserstein,
    psi,
    psi_on_deciles,
)
from quantilo.changes.snapshot import (
    GRAIN_FORMATS,
    LEVELS,
    ColumnPhoto,
    Grain,
    Levels,
    Snapshot,
)
from quantilo.typing.types import NUMERIC_TYPES, AnalysisType

Level = Literal["none", "check", "breaking"]
LEVEL_LABELS: dict[Level, str] = {
    "none": "No notable change",
    "check": "Check",
    "breaking": "Breaking",
}
Family = Literal["number", "levels", "date", "text", "identifier", "other"]

HISTOGRAM_BINS = 24
CHART_LEVELS = 12
MAX_LISTED = 20
LONG_TAIL = 0.5  # share of rows outside the top values above which new values are expected
MISSING_JUMP = 5.0  # points of % of rows
OUTLIER_JUMP = 2.0  # points of % of values, and at least twice as many
DUPLICATE_JUMP = 1.0  # points of % of rows
ROWS_DROP = 10.0  # % of rows lost
NAME_SIMILARITY = 0.75
RENAME_PSI = 0.05


@dataclass
class Side:
    label: str  # "2026-10-01" for a previous version, the dataset's name otherwise
    taken_at: str
    rows: int
    duplicates: int


@dataclass
class SchemaChange:
    kind: Literal["added", "removed", "renamed", "retyped", "constant", "empty"]
    column: str  # in the new version (in the old one for "removed")
    before: str | None = None  # the old name (renamed) or type (retyped)
    after: str | None = None
    detail: str = ""
    used_by: list[str] = field(default_factory=list)  # saved KPIs (later recipes, models)
    breaking: bool = False


@dataclass
class QualityChange:
    column: str
    missing_before: float  # % of rows
    missing_after: float
    outliers_before: float | None  # % of values beyond 1.5 IQR, each version on its quartiles
    outliers_after: float | None
    notable: bool


@dataclass
class DateChange:
    column: str
    grain: Grain
    before_min: str | None
    before_max: str | None
    after_min: str | None
    after_max: str | None
    future: int  # values after the day the new version was imported
    gaps: list[str]  # periods with no row inside the new range (that had rows before)
    periods: list[str]  # volume per period, both versions on the same grain
    before_counts: list[int]
    after_counts: list[int]
    notable: bool


@dataclass
class Moments:
    mean: float | None
    median: float | None
    std: float | None


@dataclass
class LevelShift:
    value: str
    before: float  # share of the version's values (0–1)
    after: float
    status: Literal["appeared", "disappeared", "kept"]
    certain: bool = True  # False: outside the other version's 50 most frequent, not proven absent


@dataclass
class Histogram:
    edges: list[float]  # 1st to 99th percentile of both versions
    before: list[float]  # share of each version's values per bin
    after: list[float]


@dataclass
class Shift:
    column: str
    type: AnalysisType
    kind: Literal["numeric", "categorical"]
    psi: float | None
    drift: Drift | None
    rows_before: int  # values compared
    rows_after: int
    wasserstein: float | None = None  # in units of the old interquartile range
    ks: float | None = None
    ks_p: float | None = None
    chi2: float | None = None
    chi2_p: float | None = None
    dof: int | None = None
    before: Moments | None = None
    after: Moments | None = None
    histogram: Histogram | None = None
    levels: list[LevelShift] = field(default_factory=list)  # biggest shares, for the chart
    moves: list[LevelShift] = field(default_factory=list)  # biggest changes of share
    appeared: list[LevelShift] = field(default_factory=list)
    disappeared: list[LevelShift] = field(default_factory=list)
    used_by: list[str] = field(default_factory=list)
    focus: bool = False  # asked to come first (e.g. the model's most important columns)
    # Most rows outside the top values (invoices, customers): new values are expected, the PSI
    # on the top says little, so the column is listed last and not counted as shifted (040).
    long_tail: bool = False


@dataclass
class Counts:
    added: int = 0
    removed: int = 0
    renamed: int = 0
    retyped: int = 0
    shifted: int = 0  # PSI ≥ 0.1
    new_categories: int = 0


@dataclass
class Changes:
    before: Side
    after: Side
    level: Level
    level_label: str
    headline: str  # "Compared with 2026-10-01: +4,812 rows (+3.1%), 1 column removed…"
    reasons: list[str]
    rows_change: int
    rows_change_pct: float | None
    counts: Counts
    schema: list[SchemaChange]
    quality: list[QualityChange]
    dates: list[DateChange]
    distributions: list[Shift]  # strongest shift first
    thresholds: str = PSI_SOURCE


# --- Families and renames ----------------------------------------------------------------------


def family(analysis_type: str) -> Family:
    if analysis_type in NUMERIC_TYPES:
        return "number"
    if analysis_type in ("categorical_nominal", "categorical_ordinal", "boolean"):
        return "levels"
    if analysis_type == "datetime":
        return "date"
    if analysis_type in ("text", "identifier"):
        return analysis_type  # type: ignore[return-value]
    return "other"


def _words(name: str) -> list[str]:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
    return [w for w in re.split(r"[\s_\-.]+", spaced.lower()) if w]


def names_close(old: str, new: str) -> bool:
    """Close spellings (difflib ratio ≥ 0.75), or one name is the initials of the other
    (chiffre_affaires → ca)."""
    a, b = old.lower(), new.lower()
    if difflib.SequenceMatcher(None, a, b).ratio() >= NAME_SIMILARITY:
        return True
    for long, short in ((old, b), (new, a)):
        words = _words(long)
        if len(words) >= 2 and "".join(w[0] for w in words) == short.replace("_", ""):
            return True
    return False


def _rate(count: int, total: int) -> float:
    return 100 * count / total if total else 0.0


def values_close(old: ColumnPhoto, new: ColumnPhoto, old_rows: int, new_rows: int) -> bool:
    """Same values, as far as two photos can tell."""
    if abs(_rate(old.missing, old_rows) - _rate(new.missing, new_rows)) > 2:
        return False
    if old.numbers and new.numbers:
        if not old.numbers.quantiles or not new.numbers.quantiles:
            return False
        return psi_on_deciles(old.numbers.quantiles, new.numbers.quantiles, LEVELS) < RENAME_PSI
    if old.levels and new.levels:
        old_values = {level.value for level in old.levels.top}
        new_values = {level.value for level in new.levels.top}
        shared = len(old_values & new_values) / max(len(old_values | new_values), 1)
        return shared >= 0.8 and _levels_psi(old.levels, new.levels) < RENAME_PSI
    if old.dates and new.dates:
        return (old.dates.min, old.dates.max) == (new.dates.min, new.dates.max)
    return abs(old.distinct - new.distinct) <= max(1, 0.01 * max(old.distinct, new.distinct))


def guess_renames(
    removed: Sequence[ColumnPhoto], added: Sequence[ColumnPhoto], old_rows: int, new_rows: int
) -> list[tuple[ColumnPhoto, ColumnPhoto, str]]:
    """(old, new, why) pairs, best first, each column used once. Same family of type, and the
    same values or a close name. Never applied: shown as "probably renamed"."""
    scored: list[tuple[int, ColumnPhoto, ColumnPhoto, str]] = []
    for old in removed:
        for new in added:
            if family(old.type) != family(new.type):
                continue
            same_values = values_close(old, new, old_rows, new_rows)
            close_name = names_close(old.name, new.name)
            if not (same_values or close_name):
                continue
            reasons = [
                r for r, ok in (("same values", same_values), ("close name", close_name)) if ok
            ]
            scored.append((2 * same_values + close_name, old, new, " and ".join(reasons)))
    scored.sort(key=lambda item: -item[0])
    pairs: list[tuple[ColumnPhoto, ColumnPhoto, str]] = []
    used_old: set[str] = set()
    used_new: set[str] = set()
    for _score, old, new, why in scored:
        if old.name in used_old or new.name in used_new:
            continue
        used_old.add(old.name)
        used_new.add(new.name)
        pairs.append((old, new, why))
    return pairs


# --- Distributions ------------------------------------------------------------------------------


def _level_counts(before: Levels, after: Levels) -> tuple[list[str], list[int], list[int]]:
    """Counts over the values seen in either top list, plus what is left as "Others"."""
    values = [level.value for level in before.top]
    values += [level.value for level in after.top if level.value not in set(values)]
    old = {level.value: level.count for level in before.top}
    new = {level.value: level.count for level in after.top}
    b = [old.get(value, 0) for value in values]
    a = [new.get(value, 0) for value in values]
    old_total = sum(level.count for level in before.top) + before.others
    new_total = sum(level.count for level in after.top) + after.others
    b.append(old_total - sum(b))
    a.append(new_total - sum(a))
    return [*values, "Others"], b, a


def _levels_psi(before: Levels, after: Levels) -> float:
    _values, b, a = _level_counts(before, after)
    if not sum(b) or not sum(a):
        return 0.0
    return psi(b, a)


def _categorical(old: ColumnPhoto, new: ColumnPhoto, shift: Shift) -> None:
    assert old.levels is not None and new.levels is not None
    values, b, a = _level_counts(old.levels, new.levels)
    old_total, new_total = sum(b), sum(a)
    if not old_total or not new_total:
        return
    shift.long_tail = (
        old.levels.others / old_total > LONG_TAIL or new.levels.others / new_total > LONG_TAIL
    )
    shift.psi = psi(b, a)
    shift.drift = drift_of(shift.psi)
    tested = chi_square(b, a)
    if tested:
        shift.chi2, shift.chi2_p, shift.dof = tested
    old_seen = {level.value for level in old.levels.top}
    new_seen = {level.value for level in new.levels.top}
    shifts: list[LevelShift] = []
    for value, before, after in zip(values[:-1], b[:-1], a[:-1], strict=True):
        status: Literal["appeared", "disappeared", "kept"] = "kept"
        certain = True
        if value not in old_seen:
            status, certain = "appeared", old.levels.complete
        elif value not in new_seen:
            status, certain = "disappeared", new.levels.complete
        shifts.append(LevelShift(value, before / old_total, after / new_total, status, certain))
    shift.appeared = sorted((s for s in shifts if s.status == "appeared"), key=lambda s: -s.after)[
        :MAX_LISTED
    ]
    shift.disappeared = sorted(
        (s for s in shifts if s.status == "disappeared"), key=lambda s: -s.before
    )[:MAX_LISTED]
    shift.moves = sorted(
        (s for s in shifts if s.status == "kept"), key=lambda s: -abs(s.after - s.before)
    )[:MAX_LISTED]
    shift.levels = sorted(shifts, key=lambda s: -max(s.before, s.after))[:CHART_LEVELS]


def _numeric(old: ColumnPhoto, new: ColumnPhoto, shift: Shift) -> None:
    assert old.numbers is not None and new.numbers is not None
    before, after = old.numbers, new.numbers
    shift.before = Moments(before.mean, before.median, before.std)
    shift.after = Moments(after.mean, after.median, after.std)
    shift.rows_before, shift.rows_after = before.count, after.count
    if not before.quantiles or not after.quantiles:
        return
    shift.psi = psi_on_deciles(before.quantiles, after.quantiles, LEVELS)
    shift.drift = drift_of(shift.psi)
    shift.wasserstein = normalised_wasserstein(before.quantiles, after.quantiles, LEVELS)
    shift.ks, shift.ks_p = ks_from_quantiles(
        before.quantiles, after.quantiles, LEVELS, before.count, after.count
    )
    low = min(before.quantiles[1], after.quantiles[1])
    high = max(before.quantiles[99], after.quantiles[99])
    if high > low:
        edges = np.linspace(low, high, HISTOGRAM_BINS + 1)
        old_cdf = cdf_at(before.quantiles, LEVELS, edges)
        new_cdf = cdf_at(after.quantiles, LEVELS, edges)
        shift.histogram = Histogram(
            [float(e) for e in edges],
            [float(v) for v in np.clip(np.diff(old_cdf), 0, 1)],
            [float(v) for v in np.clip(np.diff(new_cdf), 0, 1)],
        )
    if old.levels and new.levels:  # few distinct numbers: bars read better than a histogram
        values, b, a = _level_counts(old.levels, new.levels)
        if sum(b) and sum(a):
            shift.levels = [
                LevelShift(v, x / sum(b), y / sum(a), "kept")
                for v, x, y in zip(values[:-1], b[:-1], a[:-1], strict=True)
            ][:CHART_LEVELS]


def compare_distribution(old: ColumnPhoto, new: ColumnPhoto) -> Shift | None:
    kind = family(new.type)
    if family(old.type) != kind or kind not in ("number", "levels"):
        return None
    shift = Shift(
        column=new.name,
        type=new.type,
        kind="numeric" if kind == "number" else "categorical",
        psi=None,
        drift=None,
        rows_before=old.present,
        rows_after=new.present,
    )
    if kind == "number" and old.numbers and new.numbers:
        _numeric(old, new, shift)
    elif kind == "levels" and old.levels and new.levels:
        _categorical(old, new, shift)
    return shift


# --- Dates -------------------------------------------------------------------------------------

COARSER: dict[Grain, int] = {"day": 0, "month": 1, "year": 2}
LABEL_LENGTH: dict[Grain, int] = {"day": 10, "month": 7, "year": 4}


def _regroup(periods: dict[str, int], grain: Grain) -> dict[str, int]:
    grouped: dict[str, int] = {}
    for label, count in periods.items():
        key = label[: LABEL_LENGTH[grain]]
        grouped[key] = grouped.get(key, 0) + count
    return grouped


def _expected(first: str, last: str, grain: Grain) -> list[str]:
    start, end = date.fromisoformat(first), date.fromisoformat(last)
    labels: list[str] = []
    if grain == "year":
        return [str(y) for y in range(start.year, end.year + 1)]
    if grain == "month":
        year, month = start.year, start.month
        while (year, month) <= (end.year, end.month) and len(labels) < 1200:
            labels.append(f"{year:04d}-{month:02d}")
            year, month = (year + 1, 1) if month == 12 else (year, month + 1)
        return labels
    day = start
    while day <= end and len(labels) < 1200:
        labels.append(day.strftime(GRAIN_FORMATS["day"]))
        day = date.fromordinal(day.toordinal() + 1)
    return labels


def compare_dates(old: ColumnPhoto, new: ColumnPhoto) -> DateChange | None:
    if old.dates is None or new.dates is None:
        return None
    grain: Grain = max(old.dates.grain, new.dates.grain, key=lambda g: COARSER[g])
    before = _regroup({p.label: p.count for p in old.dates.periods}, grain)
    after = _regroup({p.label: p.count for p in new.dates.periods}, grain)
    periods = sorted(set(before) | set(after))
    gaps: list[str] = []
    if new.dates.min and new.dates.max and grain != "day":
        for label in _expected(new.dates.min, new.dates.max, grain):
            if not after.get(label) and before.get(label):
                gaps.append(label)
    future_new = new.dates.future > old.dates.future
    return DateChange(
        column=new.name,
        grain=grain,
        before_min=old.dates.min,
        before_max=old.dates.max,
        after_min=new.dates.min,
        after_max=new.dates.max,
        future=new.dates.future,
        gaps=gaps[:MAX_LISTED],
        periods=periods,
        before_counts=[before.get(p, 0) for p in periods],
        after_counts=[after.get(p, 0) for p in periods],
        notable=future_new or bool(gaps),
    )


# --- Everything --------------------------------------------------------------------------------


def _plural(count: int, word: str, many: str | None = None) -> str:
    return f"{count:,} {word if count == 1 else many or word + 's'}"


def _rows_part(change: int, pct: float | None) -> str:
    if change == 0:
        return "same number of rows"
    sign = "+" if change > 0 else "−"
    share = f" ({sign}{abs(pct):.1f}%)" if pct is not None else ""
    return f"{sign}{abs(change):,} rows{share}"


def _quality(old: ColumnPhoto, new: ColumnPhoto, old_rows: int, new_rows: int) -> QualityChange:
    missing_before, missing_after = _rate(old.missing, old_rows), _rate(new.missing, new_rows)
    outliers_before = outliers_after = None
    if old.numbers and new.numbers:
        outliers_before = _rate(old.numbers.outliers, old.numbers.count)
        outliers_after = _rate(new.numbers.outliers, new.numbers.count)
    notable = missing_after - missing_before >= MISSING_JUMP
    if outliers_before is not None and outliers_after is not None:
        jump = outliers_after - outliers_before
        notable = notable or (jump >= OUTLIER_JUMP and outliers_after >= 2 * outliers_before)
    return QualityChange(
        new.name, missing_before, missing_after, outliers_before, outliers_after, notable
    )


def compare_snapshots(
    before: Snapshot,
    after: Snapshot,
    *,
    before_label: str | None = None,
    after_label: str = "this version",
    used: dict[str, list[str]] | None = None,
    focus: Sequence[str] = (),
) -> Changes:
    """What changed from `before` (the reference) to `after`.

    `used`: column → what depends on it (saved KPIs; later recipes and models): such a column
    disappearing, or changing family of type, is Breaking. `focus`: columns to list first among
    distributions (a model's most important features, 037)."""
    used = used or {}
    label = before_label or before.taken_at[:10]
    old_columns = {c.name: c for c in before.columns}
    new_columns = {c.name: c for c in after.columns}
    removed = [c for name, c in old_columns.items() if name not in new_columns]
    added = [c for name, c in new_columns.items() if name not in old_columns]
    renames = guess_renames(removed, added, before.rows, after.rows)
    renamed_old = {old.name: (new.name, why) for old, new, why in renames}
    renamed_new = {new.name for _old, new, _why in renames}

    schema: list[SchemaChange] = []
    for column in removed:
        if column.name in renamed_old:
            continue  # listed once, as "probably renamed"
        users = used.get(column.name, [])
        schema.append(SchemaChange("removed", column.name, used_by=users, breaking=bool(users)))
    for old, new, why in renames:
        schema.append(
            SchemaChange(
                "renamed",
                new.name,
                before=old.name,
                after=new.name,
                detail=f"Probably renamed ({why}): not applied, the old name is gone.",
                used_by=used.get(old.name, []),
                breaking=bool(used.get(old.name)),
            )
        )
    for column in added:
        if column.name not in renamed_new:
            schema.append(SchemaChange("added", column.name, after=column.type))

    quality: list[QualityChange] = []
    dates: list[DateChange] = []
    distributions: list[Shift] = []
    pairs = [(old_columns[name], new) for name, new in new_columns.items() if name in old_columns]
    pairs += [(old, new) for old, new, _why in renames]
    for old, new in pairs:
        users = used.get(old.name, []) or used.get(new.name, [])
        if old.type != new.type or old.storage_dtype != new.storage_dtype:
            breaking = bool(users) and family(old.type) != family(new.type)
            what = (
                f"Read as {new.type} (was {old.type})."
                if old.type != new.type
                else f"Stored as {new.storage_dtype} (was {old.storage_dtype})."
            )
            if old.detected_type != new.detected_type and old.type == new.type:
                what = f"Detected as {new.detected_type} (was {old.detected_type})."
            schema.append(
                SchemaChange(
                    "retyped",
                    new.name,
                    before=old.type,
                    after=new.type,
                    detail=what,
                    used_by=users,
                    breaking=breaking,
                )
            )
        if old.present > 0 and new.present == 0:
            schema.append(SchemaChange("empty", new.name, detail="Every value is missing now."))
        elif old.distinct > 1 and new.distinct == 1:
            schema.append(SchemaChange("constant", new.name, detail="A single value is left."))
        quality.append(_quality(old, new, before.rows, after.rows))
        if (found := compare_dates(old, new)) is not None:
            dates.append(found)
        if (shift := compare_distribution(old, new)) is not None:
            shift.used_by = users
            shift.focus = new.name in focus or old.name in focus
            distributions.append(shift)

    distributions.sort(
        key=lambda s: (not s.focus, s.long_tail, -(s.psi if s.psi is not None else -1))
    )
    rows_change = after.rows - before.rows
    rows_pct = 100 * rows_change / before.rows if before.rows else None
    counts = Counts(
        added=sum(1 for s in schema if s.kind == "added"),
        removed=sum(1 for s in schema if s.kind == "removed"),
        renamed=len(renames),
        retyped=sum(1 for s in schema if s.kind == "retyped"),
        shifted=sum(1 for s in distributions if (s.psi or 0) >= PSI_MODERATE and not s.long_tail),
        new_categories=sum(len(s.appeared) for s in distributions if not s.long_tail),
    )

    reasons: list[str] = []
    for change in schema:
        if change.breaking:
            names = ", ".join(change.used_by)
            reasons.append(f"{change.column} ({change.kind}) is used by {names}.")
    breaking = bool(reasons)
    if counts.removed:
        reasons.append(f"{_plural(counts.removed, 'column')} removed.")
    if counts.renamed:
        reasons.append(f"{_plural(counts.renamed, 'column')} probably renamed.")
    if counts.retyped:
        reasons.append(f"{_plural(counts.retyped, 'column')} changed type.")
    for change in schema:
        if change.kind in ("empty", "constant"):
            reasons.append(f"{change.column}: {change.detail[:-1].lower()}.")
    if counts.shifted:
        reasons.append(f"{_plural(counts.shifted, 'distribution')} shifted (PSI ≥ 0.1).")
    if counts.new_categories:
        reasons.append(f"{_plural(counts.new_categories, 'new category', 'new categories')}.")
    for item in quality:
        if item.notable:
            reasons.append(f"{item.column}: missing or unusual values went up.")
    for item in dates:
        if item.future and item.notable:
            reasons.append(f"{item.column}: {_plural(item.future, 'date')} in the future.")
        if item.gaps:
            reasons.append(f"{item.column}: no rows in {', '.join(item.gaps[:3])}.")
    if (
        _rate(after.duplicates, after.rows) - _rate(before.duplicates, before.rows)
        >= DUPLICATE_JUMP
    ):
        reasons.append(f"{_plural(after.duplicates, 'duplicate row')} (was {before.duplicates:,}).")
    if rows_pct is not None and rows_pct <= -ROWS_DROP:
        reasons.append(f"{abs(rows_pct):.0f}% of the rows are gone.")
    level: Level = "breaking" if breaking else "check" if reasons else "none"

    parts = [_rows_part(rows_change, rows_pct)]
    for count, word, many in (
        (counts.removed, "column removed", "columns removed"),
        (counts.renamed, "column probably renamed", "columns probably renamed"),
        (counts.added, "column added", "columns added"),
        (counts.retyped, "type changed", "types changed"),
        (counts.shifted, "distribution shifted", "distributions shifted"),
        (counts.new_categories, "new category", "new categories"),
    ):
        if count:
            parts.append(_plural(count, word, many))
    if level == "none":
        parts.append("no notable change")
    headline = f"Compared with {label}: {', '.join(parts)}."

    return Changes(
        before=Side(label, before.taken_at, before.rows, before.duplicates),
        after=Side(after_label, after.taken_at, after.rows, after.duplicates),
        level=level,
        level_label=LEVEL_LABELS[level],
        headline=headline,
        reasons=reasons,
        rows_change=rows_change,
        rows_change_pct=rows_pct,
        counts=counts,
        schema=schema,
        quality=quality,
        dates=dates,
        distributions=distributions,
    )
