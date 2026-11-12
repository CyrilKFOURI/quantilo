"""R-SST-24 (fix1): records of several kinds in one file, told apart by a field such as `type`.

A GH Archive hour mixes `PushEvent`, `WatchEvent`, `IssuesEvent`… whose `payload` has another
shape for each kind: one table holds 500 columns, most of them empty on most rows. When a top-level
text field named like a discriminator (`type`, `kind`, `event_type`, `resourceType`, `@type`,
`__typename`…) has 2 to 20 values on every record, and enough fields are filled for some kinds and
never for another kind that, at their rate elsewhere, should show them on 5 rows or more, those
fields move to one table per kind
(`events.PushEvent`), one row per record of that kind, linked by `_parent_id` to the record's
`_id`. The fields every kind may have stay in the records table. Written as a rule with its reason;
`discriminator=""` keeps one table, a field name forces the split on it.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import pyarrow as pa

from quantilo.ingestion.structured.model import RuleHit, StructuredColumn, StructuredTable
from quantilo.ingestion.structured.plan import sanitize, unique

if TYPE_CHECKING:
    from quantilo.ingestion.structured.model import StructuredLayout

NAMES = frozenset(
    [
        *("type", "kind", "event", "eventtype", "eventname", "recordtype", "objecttype"),
        *("doctype", "entitytype", "resourcetype", "typename", "messagetype", "msgtype"),
        "itemtype",
    ]
)
MIN_KINDS = 2
MAX_KINDS = 20
FILLED = 0.99  # the discriminator is given on this share of the records
EXPECTED = 5  # a kind where a field should be filled on 5 rows or more, by its rate elsewhere
MIN_FIELDS = 3  # fields owned by some kinds only
MIN_SHARE = 0.2  # of the records table's data columns
KEYS = ("_id", "_parent_id", "_position", "_extra")
MAX_DEPTH = 64  # R-LIM-02


def is_discriminator_name(name: str) -> bool:
    return re.sub(r"[^a-z0-9]", "", name.lower()) in NAMES


def _counts(table: pa.Table, by: str, columns: list[str]) -> dict[str, dict[str, int]]:
    """kind → column → filled rows (and "" → rows of the kind)."""
    grouped = (
        table.select([by, *columns])
        .group_by(by)
        .aggregate([([], "count_all"), *((column, "count") for column in columns)])
    )
    found: dict[str, dict[str, int]] = {}
    data = grouped.to_pydict()
    for index, kind in enumerate(data[by]):
        if kind is None:
            continue
        row = {"": int(data["count_all"][index])}
        for column in columns:
            row[column] = int(data[f"{column}_count"][index])
        found[str(kind)] = row
    return found


def _owned(counts: dict[str, dict[str, int]], columns: list[str]) -> list[str]:
    """Fields some kinds have and another kind never has, although at the field's rate in the
    kinds that have it, that kind should show it on at least 5 rows: not chance, not its own."""
    owned = []
    for column in columns:
        having = [row for row in counts.values() if row[column] > 0]
        if not having:
            continue
        rate = sum(row[column] for row in having) / sum(row[""] for row in having)
        if any(row[column] == 0 and rate * row[""] >= EXPECTED for row in counts.values()):
            owned.append(column)
    return owned


def _paths(node: dict[str, Any], prefix: str, out: set[str], depth: int) -> None:
    for key, value in node.items():
        if value is None:
            continue
        path = f"{prefix}{key}"
        if isinstance(value, dict) and depth < MAX_DEPTH:
            _paths(value, path + ".", out, depth + 1)
        else:
            out.add(path)


def detect(sample: list[Any], requested: str | None) -> tuple[str, int] | None:
    """The discriminator and its number of kinds, read on the sample before the plan, so that the
    column limit (R-SST-18) is counted per kind and not over every kind at once."""
    if requested == "":
        return None
    records = [record for record in sample if isinstance(record, dict)]
    if not records:
        return None
    keys = [requested] if requested else [k for k in records[0] if is_discriminator_name(k)]
    for key in keys:
        values = [record.get(key) for record in records]
        texts = [v for v in values if isinstance(v, str)]
        if len(texts) < FILLED * len(records):
            continue
        counts: dict[str, dict[str, int]] = {}
        for record, value in zip(records, values, strict=True):
            if not isinstance(value, str):
                continue
            row = counts.setdefault(value, {"": 0})
            row[""] += 1
            found: set[str] = set()
            _paths({k: v for k, v in record.items() if k != key}, "", found, 1)
            for path in found:
                row[path] = row.get(path, 0) + 1
        if not MIN_KINDS <= len(counts) <= MAX_KINDS:
            continue
        paths = sorted({path for row in counts.values() for path in row if path})
        for row in counts.values():
            for path in paths:
                row.setdefault(path, 0)
        owned = _owned(counts, paths)
        if requested or (len(owned) >= MIN_FIELDS and len(owned) >= MIN_SHARE * len(paths)):
            return key, len(counts)
    return None


def _candidate(
    table: pa.Table, described: StructuredTable, forced: str | None
) -> tuple[str, dict[str, dict[str, int]], list[str]] | None:
    columns = [c.name for c in described.columns if c.name not in KEYS]
    names = [forced] if forced else [n for n in columns if "." not in n]
    for name in names:
        if name not in table.column_names or not (forced or is_discriminator_name(name)):
            continue
        if not pa.types.is_string(table.schema.field(name).type):
            continue
        data = table.column(name)
        if table.num_rows == 0 or data.null_count > (1 - FILLED) * table.num_rows:
            continue
        others = [c for c in columns if c != name]
        counts = _counts(table, name, others)
        if not MIN_KINDS <= len(counts) <= MAX_KINDS:
            continue
        owned = _owned(counts, others)
        if forced or (len(owned) >= MIN_FIELDS and len(owned) >= MIN_SHARE * len(others)):
            return name, counts, owned
    return None


def split_by_kind(layout: StructuredLayout, forced: str | None) -> None:
    """Move the fields of each kind of record to its own table (R-SST-24), in place."""
    from quantilo.ingestion.structured.model import Relation

    if not layout.tables or layout.tables[0].kind != "records":
        return
    main = layout.tables[0]
    table = layout._data[main.name]
    if "_id" not in table.column_names:
        return
    found = _candidate(table, main, forced)
    if found is None:
        return
    by, counts, owned = found
    if not owned:
        return
    specs = {c.name: c for c in main.columns}
    used = {t.name for t in layout.tables}
    kinds = sorted(counts, key=lambda kind: (-counts[kind][""], kind))
    # A field never filled (always null) goes with its filled siblings (`payload.forkee.*`).
    empty = {
        c: _parent(c)
        for c in specs
        if c not in KEYS and c != by and all(row.get(c, 0) == 0 for row in counts.values())
    }
    owners = {_parent(c) for c in owned}
    moved = [*owned, *(c for c, parent in empty.items() if parent and parent in owners)]
    moving = set(moved)
    labels = table.column(by).to_pylist()
    groups: list[tuple[str, pa.Array, dict[str, int]]] = [
        (kind, pa.array([label == kind for label in labels]), counts[kind]) for kind in kinds
    ]
    blank = pa.array([label is None for label in labels])
    if table.column(by).null_count:  # records without a kind keep their fields too
        rest = table.filter(blank)
        row = {"": rest.num_rows, **{c: rest.num_rows - rest.column(c).null_count for c in moved}}
        groups.append((f"no_{by}", blank, row))
    position = 1
    for kind, mask, row in groups:
        filled = {_parent(c) for c in owned if row.get(c, 0) > 0}
        mine = [
            c
            for c in table.column_names
            if c in moving and (row.get(c, 0) > 0 or (c in empty and empty[c] in filled))
        ]
        if not mine:
            continue
        rows = row[""]
        part = table.filter(mask).select(["_id", *mine])
        part = part.rename_columns(["_parent_id", *mine])
        name = unique(sanitize(f"{main.name}.{kind}"), used)
        shown = ", ".join(mine[:4]) + (" …" if len(mine) > 4 else "")
        child = StructuredTable(
            name=name,
            kind="object",
            locator=(
                f"{main.locator}[?(!@.{by})]"
                if mask is blank
                else f"{main.locator}[?(@.{by}=='{kind}')]"
            ),
            rule="R-SST-24",
            reason=(
                (
                    f"the {rows:,} records without a {by}"
                    if mask is blank
                    else f"the {rows:,} records whose {by} is '{kind}'"
                )
                + f", with the {_fields(len(mine))} of this kind ({shown}): one row per record"
            ),
            parent=main.name,
            rows=rows,
            columns=[
                StructuredColumn(
                    name="_parent_id",
                    path="_parent_id",
                    dtype="int",
                    rule="R-KEY-01",
                    reason=f"_id of the {main.name} row",
                    present=rows,
                ),
                *(_subset(specs[c], rows, part.column(c)) for c in mine),
            ],
        )
        layout.tables.insert(position, child)
        position += 1
        layout._data[name] = part
        layout.relations.append(
            Relation(
                main.name,
                name,
                "_id",
                "_parent_id",
                "R-SST-24",
                f"no {by}" if mask is blank else f"{by}={kind}",
            )
        )
    keep = [c for c in table.column_names if c not in moving]
    layout._data[main.name] = table.select(keep)
    main.columns = [c for c in main.columns if c.name not in moving]
    listed = ", ".join(f"{kind} {counts[kind]['']:,}" for kind in kinds[:4])
    more = f" and {len(kinds) - 4} more" if len(kinds) > 4 else ""
    main.reason += f"; the fields of each {by} in their own table (R-SST-24)"
    layout.reasons.append(
        RuleHit(
            "R-SST-24",
            0.0,
            f"Records of {len(kinds)} kinds by '{by}' ({listed}{more}): the {_fields(len(moved))} "
            "that only some kinds have moved to one table per kind, linked by _parent_id; the "
            "fields every kind may have stay in the records table. Read with discriminator='' to "
            "keep one table.",
        )
    )


def _fields(count: int) -> str:
    return f"{count:,} field{'s' if count != 1 else ''}"


def _parent(column: str) -> str:
    return column.rsplit(".", 1)[0] if "." in column else ""


def _subset(spec: StructuredColumn, rows: int, data: pa.ChunkedArray) -> StructuredColumn:
    present = rows - data.null_count
    return StructuredColumn(
        name=spec.name,
        path=spec.path,
        dtype=spec.dtype,
        rule=spec.rule,
        reason=spec.reason,
        present=present,
        nulls=0,
        missing=rows - present,
        unreadable=spec.unreadable,
        examples=list(spec.examples),
        notes=[n for n in spec.notes if not n.startswith("present in")],
    )
