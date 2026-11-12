"""`read_structured`: a JSON, JSON Lines or XML document → linked tables (083 spec).

read → find the records (R-SST-01..07) → profile a sample → plan (R-SST-10..21)
     → emit every record into Arrow → keys, relations, confidence (R-KEY, R-CONF)
"""

from __future__ import annotations

import itertools
import json
import os
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pyarrow as pa

from quantilo.ingestion.structured.build import Emitter, dumps
from quantilo.ingestion.structured.errors import StructuredError
from quantilo.ingestion.structured.kinds import detect as detect_kinds
from quantilo.ingestion.structured.kinds import split_by_kind
from quantilo.ingestion.structured.model import (
    Format,
    RuleHit,
    StructuredColumn,
    StructuredLayout,
    StructuredOptions,
    StructuredTable,
)
from quantilo.ingestion.structured.plan import EXTRA, ColumnSpec, TablePlan, make_plan, sanitize
from quantilo.ingestion.structured.plan import unique as unique_name
from quantilo.ingestion.structured.profile import INT64_MAX, INT64_MIN, profile_records
from quantilo.ingestion.structured.readers import (
    ParseNotes,
    describe_json_error,
    detect_format,
    iter_jsonl,
    load_json,
    looks_jsonl,
)
from quantilo.ingestion.structured.records import Records, as_record, select_records
from quantilo.ingestion.structured.xmltree import load_xml, normalize_lists

INVALID_LINES = 0.01  # R-LIM-05
CHILD_RATIO = 50  # R-SST-22
CHILD_ROWS = 20_000_000
KEY_NAMES = ("id", "uuid", "guid", "@id", "key", "code")
OVERRIDES = ("flatten", "separate", "json", "skip")


def read_structured(
    path: str | os.PathLike[str], options: StructuredOptions | None = None
) -> StructuredLayout:
    """A JSON, JSON Lines or XML document as linked tables, each rule explained."""
    options = options or StructuredOptions()
    for where, choice in options.overrides.items():
        if choice not in OVERRIDES:
            raise StructuredError(
                f"Unknown choice {choice!r} for {where!r}: use one of {', '.join(OVERRIDES)}.",
                rule="R-OVR",
            )
    source = Path(path)
    if not source.is_file():
        raise StructuredError(f"No such file: {source}")
    fmt: Format = options.format or detect_format(source)
    notes = ParseNotes()
    reasons: list[RuleHit] = []
    root: Any = None
    if fmt == "json":
        try:
            root = load_json(source, options.max_bytes, notes)
        except json.JSONDecodeError as error:
            if not looks_jsonl(source):
                raise StructuredError(describe_json_error(error)) from error
            fmt = "jsonl"
            notes = ParseNotes()
            reasons.append(
                RuleHit(
                    "R-DET-02", 0.0, "Not one JSON value, but one per line: read as JSON Lines."
                )
            )
    elif fmt == "xml":
        root = load_xml(source, options.max_bytes)
        wrapped = normalize_lists(root)
        if wrapped:
            reasons.append(
                RuleHit(
                    "R-SST-06",
                    0.0,
                    f"{wrapped:,} lone elements read as one-element lists: the same element is "
                    "repeated under another parent.",
                )
            )
    if fmt == "jsonl":
        return Reader(source, fmt, options, notes, reasons).jsonl()
    return Reader(source, fmt, options, notes, reasons).document(root)


class Reader:
    def __init__(
        self,
        source: Path,
        fmt: Format,
        options: StructuredOptions,
        notes: ParseNotes,
        reasons: list[RuleHit],
    ) -> None:
        self.source = source
        self.fmt: Format = fmt
        self.options = options
        self.notes = notes
        self.reasons = reasons

    # the two ways in ----------------------------------------------------------------------

    def document(self, root: Any) -> StructuredLayout:
        records = select_records(root, self.fmt, self.options.records_path)
        name = records.keys[-1] if records.keys else self.source.stem
        items = records.items
        limit = self.options.limit
        chosen = items if limit is None else items[:limit]
        return self.layout(
            records,
            name,
            sample=items[: self.options.sample],
            stream=iter(chosen),
            total=len(items),
            truncated=limit is not None and len(items) > limit,
        )

    def jsonl(self) -> StructuredLayout:
        records = Records((), [], "R-SST-04", "JSON Lines: one row per line", True, locator="$[*]")
        sample: list[dict[str, Any]] = []
        size = 0
        for _, value, length in iter_jsonl(self.source, self.notes):
            sample.append(as_record(value))
            size += length
            if len(sample) >= self.options.sample:
                break
            if size > self.options.sample_bytes:
                break
        last = self.notes.last_line
        rest = (as_record(value) for _, value, _ in iter_jsonl(self.source, self.notes, after=last))
        stream: Iterator[dict[str, Any]] = itertools.chain(sample, rest)
        limit = self.options.limit
        truncated = False
        if limit is not None:
            head = list(itertools.islice(stream, limit + 1))
            truncated = len(head) > limit
            stream = iter(head[:limit])
        layout = self.layout(records, self.source.stem, sample, stream, None, truncated)
        self.check_lines(layout)
        return layout

    def check_lines(self, layout: StructuredLayout) -> None:
        """R-LIM-05: up to 1 % of invalid lines (at least one) are skipped, beyond it's an error."""
        notes = self.notes
        if not notes.skipped:
            return
        allowed = max(1, int(notes.lines * INVALID_LINES))
        listed = ", ".join(str(line) for line in notes.skipped_lines[:10])
        if notes.skipped > allowed:
            raise StructuredError(
                f"{notes.skipped:,} of {notes.lines:,} lines are not JSON (lines {listed}…): over "
                f"the 1 % that can be skipped.",
                rule="R-LIM-05",
            )
        layout.skipped_lines = list(notes.skipped_lines)
        layout.reasons.append(
            RuleHit("R-LIM-05", -0.1, f"{notes.skipped:,} invalid lines skipped (lines {listed}).")
        )

    # the common path ----------------------------------------------------------------------

    def layout(
        self,
        records: Records,
        name: str,
        sample: list[Any],
        stream: Iterator[dict[str, Any]],
        total: int | None,
        truncated: bool,
    ) -> StructuredLayout:
        options = self.options
        flat = {path for path, choice in options.overrides.items() if choice == "flatten"}
        kinds = None if options.one_table else detect_kinds(sample, options.discriminator)
        if kinds is not None:
            # R-SST-24: the kinds go to their own tables, so the column limit is per kind
            options = replace(options, max_columns=options.max_columns * kinds[1])
        try:
            profile, maps = profile_records(sample, flat)
            root, forced = make_plan(
                profile, maps, options, name, records.locator, records.rule, records.reason
            )
            emitter = Emitter(root)
            count = 0
            for count, record in enumerate(stream, start=1):
                emitter.record(count - 1, record)
        except RecursionError as error:
            raise StructuredError("Nesting too deep for this reader.", rule="R-LIM-02") from error
        built = emitter.finish()
        layout = StructuredLayout(
            source=str(self.source),
            format=self.fmt,
            records_path=".".join(records.keys),
            records_rule=records.rule,
            records=total if total is not None and not truncated else count,
            sampled=profile.records,
            truncated=truncated,
            candidates=records.candidates,
            reasons=list(self.reasons),
        )
        layout.reasons.insert(0, RuleHit(records.rule, 0.0, f"Records: {records.reason}."))
        for plan, table, rows in built.values():
            layout.tables.append(self.describe(plan, rows, table))
            layout._data[layout.tables[-1].name] = table
        self.relations(layout, root)
        if kinds is not None:
            split_by_kind(layout, kinds[0])
        if options.one_table and layout.tables and count:
            main = layout.tables[0]
            ratio = main.rows / count
            main.rule = "R-SST-15b"
            main.reason = (
                f"One table: {count:,} {name} → {main.rows:,} rows (×{ratio:.1f}); the statistics "
                "of the records are weighted by their number of elements"
            )
        if records.meta:
            self.meta(layout, records)
        self.document_notes(layout, root, maps, forced)
        self.confidence(layout, records, root)
        return layout

    def describe(self, plan: TablePlan, rows: int, table: pa.Table) -> StructuredTable:
        columns = []
        for name, spec in plan.columns.items():
            columns.append(self.column(spec, rows, table.column(name)))
        if EXTRA in plan.columns and plan.extra_paths:
            shown = ", ".join(plan.extra_paths[:5]) + ("…" if len(plan.extra_paths) > 5 else "")
            extra = next(column for column in columns if column.name == EXTRA)
            extra.notes.append(f"holds {shown}")
        described = StructuredTable(
            name=plan.name,
            kind=plan.kind,
            locator=plan.locator,
            rule=plan.rule,
            reason=plan.reason,
            parent=plan.parent.name if plan.parent else None,
            rows=rows,
            columns=columns,
        )
        if "_id" in plan.columns:
            described.natural_key = natural_key(table, plan)
        if plan.shape_drift:
            described.warnings.append(
                f"{plan.shape_drift:,} values had another shape than in the sample: kept in "
                "_extra (R-SST-20)."
            )
        return described

    @staticmethod
    def column(spec: ColumnSpec, rows: int, data: pa.ChunkedArray) -> StructuredColumn:
        nulls = data.null_count
        if spec.tracked:
            present, explicit = spec.present, spec.nulls
        else:
            present, explicit = rows - nulls, 0
        notes = list(spec.notes)
        if spec.unreadable:
            examples = ", ".join(repr(value) for value in spec.examples)
            notes.append(
                f"{spec.unreadable:,} values are not {spec.dtype}s and were left empty "
                f"(e.g. {examples})"
            )
        return StructuredColumn(
            name=spec.name,
            path=spec.path,
            dtype=spec.dtype,
            rule=spec.rule,
            reason=spec.reason,
            present=present,
            nulls=explicit,
            missing=max(0, rows - present),
            unreadable=spec.unreadable,
            examples=list(spec.examples),
            notes=notes,
        )

    def relations(self, layout: StructuredLayout, root: TablePlan) -> None:
        from quantilo.ingestion.structured.model import Relation

        for plan in root.walk():
            if plan.parent is None or plan.source is None:
                continue
            layout.relations.append(
                Relation(
                    parent=plan.parent.name,
                    child=plan.name,
                    parent_key="_id",
                    child_key="_parent_id",
                    rule=plan.rule,
                    path=plan.source.display,
                )
            )
            child = layout.table(plan.name)
            parent = layout.table(plan.parent.name)
            if child.rows > CHILD_ROWS or (parent.rows and child.rows > CHILD_RATIO * parent.rows):
                child.selected = False
                child.warnings.append(
                    f"{child.rows:,} rows for {parent.rows:,} in {parent.name}: not selected by "
                    "default (R-SST-22)."
                )

    def meta(self, layout: StructuredLayout, records: Records) -> None:
        """R-SST-05: what surrounds the records, as a one-row table, not selected."""
        assert records.meta is not None
        flat: dict[str, Any] = {}
        flatten_meta(records.meta, "", flat, 1, self.options.max_depth)
        used: set[str] = set()
        row = {unique_name(sanitize(key), used): value for key, value in flat.items()}
        table = pa.Table.from_pylist([row])
        reason = "what surrounds the records (pagination, totals, dates): one row"
        layout.tables.append(
            StructuredTable(
                name="_meta",
                kind="meta",
                locator="$",
                rule="R-SST-05",
                reason=reason,
                rows=1,
                selected=False,
                columns=[
                    StructuredColumn(
                        name=name,
                        path=key,
                        dtype=arrow_dtype(table.schema.field(name).type),
                        rule="R-SST-05",
                        reason="around the records",
                        present=1,
                    )
                    for name, key in zip(row, flat, strict=True)
                ],
            )
        )
        layout._data["_meta"] = table
        layout.reasons.append(
            RuleHit("R-SST-05", 0.0, f"{len(row)} values around the records kept in _meta.")
        )

    def document_notes(
        self, layout: StructuredLayout, root: TablePlan, maps: dict[Any, str], forced: set[Any]
    ) -> None:
        notes = self.notes
        if notes.duplicate_keys:
            keys = ", ".join(notes.duplicate_examples)
            layout.reasons.append(
                RuleHit(
                    "R-SST-23",
                    -0.05,
                    f"{notes.duplicate_keys:,} keys given twice in one object ({keys}): the last "
                    "one wins.",
                )
            )
        if notes.constants:
            layout.reasons.append(
                RuleHit("R-LIM-04", 0.0, f"{notes.constants:,} NaN or Infinity values read.")
            )
        if notes.big_ints:
            layout.reasons.append(
                RuleHit(
                    "R-LIM-03",
                    0.0,
                    f"{notes.big_ints:,} integers longer than 4,300 digits kept as text.",
                )
            )
        if maps:
            layout.reasons.append(
                RuleHit(
                    "R-SST-12",
                    0.0,
                    f"{len(maps)} dictionar{'ies' if len(maps) > 1 else 'y'} read as long tables "
                    "(key, value).",
                )
            )
        if forced:
            layout.reasons.append(
                RuleHit(
                    "R-SST-21",
                    -0.1,
                    f"More than {self.options.max_tables} tables: {len(forced)} of the deepest "
                    "lists kept as JSON text.",
                )
            )
        drift = [(plan.name, name) for plan in root.walk() for name in plan.drift.values()]
        shape = sum(plan.shape_drift for plan in root.walk())
        if drift or shape:
            shown = ", ".join(f"{table}.{name}" for table, name in drift[:5])
            parts = []
            if drift:
                parts.append(f"{len(drift)} columns seen only after the sample ({shown})")
            if shape:
                parts.append(f"{shape:,} values of another shape kept in _extra")
            layout.reasons.append(RuleHit("R-SST-20", -0.1, "; ".join(parts) + "."))
        extra = [plan.name for plan in root.walk() if plan.extra_paths]
        if extra:
            layout.reasons.append(
                RuleHit(
                    "R-SST-18",
                    -0.1,
                    f"Values past the depth or column limit kept in _extra ({', '.join(extra)}).",
                )
            )
        if layout.truncated:
            layout.warnings.append(f"Only the first {layout.records:,} records were read (limit).")

    def confidence(self, layout: StructuredLayout, records: Records, root: TablePlan) -> None:
        """R-CONF-01: from 1, minus what makes the result worth a look."""
        hits = [hit for hit in layout.reasons if hit.weight]
        candidates = records.candidates
        if len(candidates) > 1 and candidates[1].count >= 0.5 * candidates[0].count:
            hits.append(
                RuleHit(
                    "R-CONF-01",
                    -0.2,
                    f"Another list could hold the records: {candidates[1].path} "
                    f"({candidates[1].count:,} elements).",
                )
            )
            layout.reasons.append(hits[-1])
        if records.rule == "R-SST-02" and records.meta and layout.tables:
            around = len(layout.table("_meta").columns)
            fields = sum(1 for c in layout.tables[0].columns if c.rule != "R-KEY-01")
            if around > fields:
                hits.append(
                    RuleHit(
                        "R-CONF-01",
                        -0.2,
                        f"More values around the records ({around}) than in them ({fields}): "
                        "check the records path, or read the whole document as one record.",
                    )
                )
                layout.reasons.append(hits[-1])
        mixed = sum(
            1
            for plan in root.walk()
            for spec in plan.columns.values()
            if spec.rule in ("R-SST-17", "R-SST-17b")
        )
        if mixed:
            hits.append(
                RuleHit(
                    "R-SST-17",
                    -min(0.2, 0.05 * mixed),
                    f"{mixed} columns of mixed shapes kept as JSON text.",
                )
            )
            layout.reasons.append(hits[-1])
        unreadable = sum(spec.unreadable for plan in root.walk() for spec in plan.columns.values())
        if unreadable:
            hits.append(
                RuleHit(
                    "R-SST-16",
                    -0.05,
                    f"{unreadable:,} values did not fit their column's type and were left empty.",
                )
            )
            layout.reasons.append(hits[-1])
        layout.confidence = round(max(0.0, min(1.0, 1 + sum(hit.weight for hit in hits))), 2)


def flatten_meta(
    node: dict[str, Any], prefix: str, out: dict[str, Any], level: int, max_depth: int
) -> None:
    for key, value in node.items():
        name = prefix + key
        if isinstance(value, dict) and level <= max_depth:
            flatten_meta(value, name + ".", out, level + 1, max_depth)
        elif isinstance(value, dict | list):
            out[name] = dumps(value)
        elif isinstance(value, int) and not isinstance(value, bool):
            out[name] = value if INT64_MIN <= value <= INT64_MAX else str(value)
        else:
            out[name] = value


def arrow_dtype(kind: pa.DataType) -> Any:
    if pa.types.is_boolean(kind):
        return "bool"
    if pa.types.is_integer(kind):
        return "int"
    if pa.types.is_floating(kind):
        return "float"
    return "string"


def natural_key(table: pa.Table, plan: TablePlan) -> str | None:
    """R-KEY-02: a field named like an identifier, never empty, unique over the whole table."""
    if not table.num_rows:
        return None
    names = [
        name
        for name, spec in plan.columns.items()
        if spec.rule != "R-KEY-01" and spec.dtype in ("int", "string") and is_key_name(name)
    ]
    names.sort(key=lambda name: name.lower() not in KEY_NAMES)
    for name in names:
        column = table.column(name)
        if column.null_count == 0 and len(column.unique()) == table.num_rows:
            return name
    return None


def is_key_name(name: str) -> bool:
    last = name.rsplit(".", 1)[-1]
    return (
        last.lower() in KEY_NAMES
        or last.endswith(("_id", "Id", "ID"))
        or last.lower().endswith("_key")
    ) and "." not in name
