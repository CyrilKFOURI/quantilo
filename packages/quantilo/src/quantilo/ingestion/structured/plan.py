"""The plan: what every path of the records becomes (R-SST-10 to 21), and the tables it makes.

The first rule that matches decides; each column and table keeps its code and its reason.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Literal

from quantilo.ingestion.structured.errors import StructuredError
from quantilo.ingestion.structured.model import Dtype, StructuredOptions, TableKind
from quantilo.ingestion.structured.profile import LIST, VALUES, Path, PathStat, Profile, user_path
from quantilo.ingestion.structured.records import locator_key

FieldKind = Literal[
    "column",
    "json",
    "flatten",
    "single",
    "positional",
    "scalars",
    "objects",
    "map",
    "object",
    "extra",
    "skip",
    "count",
    "explode",
]

EXTRA = "_extra"
NAME_BAD = re.compile(r"[^\w.@#\[\]]")  # R-NAME-01 (\w: Unicode letters, digits, _)
MAX_NAME = 120
MAX_POSITIONAL = 50
NUMERIC_SHARE = 0.95  # R-SST-16


def sanitize(name: str) -> str:
    return (NAME_BAD.sub("_", name) or "_")[:MAX_NAME]


def unique(name: str, used: set[str]) -> str:
    if name not in used:
        used.add(name)
        return name
    number = 2
    while True:
        suffix = f"~{number}"
        candidate = name[: MAX_NAME - len(suffix)] + suffix
        if candidate not in used:
            used.add(candidate)
            return candidate
        number += 1


@dataclass(eq=False)
class ColumnSpec:
    name: str
    path: str
    dtype: Dtype
    rule: str
    reason: str
    notes: list[str] = field(default_factory=list)
    tracked: bool = False  # presence and explicit nulls counted while emitting
    present: int = 0
    nulls: int = 0
    unreadable: int = 0
    examples: list[str] = field(default_factory=list)


@dataclass(eq=False)
class FieldPlan:
    kind: FieldKind
    path: Path  # in the record
    display: str  # dotted, relative to the table's row
    rule: str
    reason: str
    column: str | None = None
    columns: list[str] = field(default_factory=list)  # positional
    count_column: str | None = None
    joined_column: str | None = None
    position_column: str | None = None  # explode
    fields: dict[str, FieldPlan] = field(default_factory=dict)
    table: TablePlan | None = None


@dataclass(eq=False)
class TablePlan:
    name: str
    kind: TableKind
    row_path: Path
    locator: str
    rule: str
    reason: str
    max_columns: int
    parent: TablePlan | None = None
    source: FieldPlan | None = None
    fields: dict[str, FieldPlan] = field(default_factory=dict)
    columns: dict[str, ColumnSpec] = field(default_factory=dict)
    used: set[str] = field(default_factory=lambda: {EXTRA})
    value_column: str | None = None
    map_values: Literal["scalar", "object", "json"] | None = None
    children: list[TablePlan] = field(default_factory=list)
    data_columns: int = 0
    drift: dict[str, str] = field(default_factory=dict)  # display path -> column (R-SST-20)
    shape_drift: int = 0
    extra_paths: list[str] = field(default_factory=list)

    def key(self, name: str, reason: str) -> str:
        self.used.add(name)
        self.columns[name] = ColumnSpec(name, name, "int", "R-KEY-01", reason)
        return name

    def add(
        self,
        display: str,
        dtype: Dtype,
        rule: str,
        reason: str,
        notes: list[str] | None = None,
        *,
        tracked: bool = False,
    ) -> str:
        name = unique(sanitize(display), self.used)
        self.columns[name] = ColumnSpec(
            name, display, dtype, rule, reason, list(notes or []), tracked=tracked
        )
        self.data_columns += 1
        return name

    def room(self, count: int = 1) -> bool:
        return self.data_columns + count <= self.max_columns

    def extra_column(self) -> str:
        if EXTRA not in self.columns:
            self.columns[EXTRA] = ColumnSpec(
                EXTRA,
                EXTRA,
                "json",
                "R-SST-18",
                "values past the depth or column limit, as one JSON object per row (keys = paths)",
            )
        return EXTRA

    def walk(self) -> Iterator[TablePlan]:
        yield self
        for child in self.children:
            yield from child.walk()


def display(path: Path, row_path: Path) -> str:
    return ".".join(segment for segment in path[len(row_path) :] if segment != LIST)


def depth(path: Path, row_path: Path) -> int:
    return sum(1 for segment in path[len(row_path) :] if segment != LIST)


def describe_lengths(stat: PathStat) -> str:
    summary = stat.length_summary()
    if not summary:
        return "no lists"
    if summary["min"] == summary["max"]:
        return f"always {summary['max']:g}"
    return f"{summary['min']:g} to {summary['max']:g}, median {summary['median']:g}"


def infer(stat: PathStat) -> tuple[Dtype, str, str]:
    """The type of a path's scalar values (R-SST-10, R-SST-16, R-SST-16b)."""
    numbers = stat.ints + stat.big_ints + stat.floats
    if not stat.strings and not stat.bools:
        if stat.big_ints:
            return "string", "R-SST-16b", "integers beyond 64 bits: kept as text"
        if stat.floats:
            return "float", "R-SST-10", "numbers"
        if stat.ints:
            return "int", "R-SST-10", "whole numbers"
        return "string", "R-SST-19", "always null in the sample"
    if stat.bools and not numbers and not stat.strings:
        return "bool", "R-SST-10", "true / false"
    if stat.strings and not numbers and not stat.bools:
        if stat.numeric_strings == stat.strings:
            return "string", "R-SST-10", "numbers written as text: the type detection reads them"
        return "string", "R-SST-10", "text"
    if stat.strings and numbers and not stat.bools:
        share = (numbers + stat.numeric_strings) / (numbers + stat.strings)
        if share >= NUMERIC_SHARE and not stat.big_ints:
            whole = not stat.floats and stat.int_strings == stat.numeric_strings
            return (
                "int" if whole else "float",
                "R-SST-16",
                f"{numbers:,} numbers and {stat.numeric_strings:,} numeric texts "
                f"({share:.0%} numeric): read as numbers, the other values left empty",
            )
        return "string", "R-SST-16b", f"numbers and text mixed ({share:.0%} numeric): kept as text"
    return "string", "R-SST-16b", "true/false mixed with numbers or text: kept as text"


class Planner:
    def __init__(
        self,
        profile: Profile,
        maps: dict[Path, str],
        options: StructuredOptions,
        records_locator: str,
        forced_json: set[Path] | None = None,
        explode: Path | None = None,
    ) -> None:
        self.profile = profile
        self.maps = maps
        self.options = options
        self.records_locator = records_locator
        self.forced_json = forced_json or set()
        self.explode = explode
        self.exploding = False
        self.table_names = {"_meta"}

    # tables -------------------------------------------------------------------------------

    def plan(self, name: str, rule: str, reason: str) -> TablePlan:
        table = self.new_table(name, "records", (), self.records_locator, rule, reason, None)
        table.key("_id", "record number in the file (0 = the first)")
        table.fields = self.object_fields((), table)
        return table

    def new_table(
        self,
        name: str,
        kind: TableKind,
        row_path: Path,
        locator: str,
        rule: str,
        reason: str,
        parent: TablePlan | None,
    ) -> TablePlan:
        table = TablePlan(
            name=unique(sanitize(name), self.table_names),
            kind=kind,
            row_path=row_path,
            locator=locator,
            rule=rule,
            reason=reason,
            max_columns=self.options.max_columns,
            parent=parent,
        )
        if parent is not None:
            parent.children.append(table)
        return table

    def locator(self, path: Path) -> str:
        steps = []
        for segment in path:
            if segment == LIST:
                steps.append("[*]")
            elif segment == VALUES:
                steps.append(".*")
            else:
                steps.append(locator_key(segment))
        return self.records_locator + "".join(steps)

    # fields -------------------------------------------------------------------------------

    def object_fields(self, base: Path, table: TablePlan) -> dict[str, FieldPlan]:
        stat = self.profile.stats.get(base)
        if stat is None:
            return {}
        return {key: self.decide((*base, key), table) for key in stat.children}

    def decide(self, path: Path, table: TablePlan) -> FieldPlan:
        stat = self.profile.stats[path]
        shown = display(path, table.row_path)
        choice = self.options.overrides.get(user_path(path))
        if choice == "skip":
            return FieldPlan("skip", path, shown, "R-OVR", "skipped (your choice)")
        if choice == "json":
            return self.json(path, shown, table, "R-OVR", "kept as JSON text (your choice)")
        if path in self.forced_json:
            return self.json(
                path,
                shown,
                table,
                "R-SST-21",
                f"more than {self.options.max_tables} tables: this list, among the deepest, is "
                "kept as JSON text",
            )
        if stat.count == stat.nulls:
            return self.column(
                path, shown, table, "string", "R-SST-19", "always null in the sample"
            )
        shapes = [
            name
            for name, count in (
                ("values", stat.scalars),
                ("objects", stat.objects),
                ("lists", stat.lists),
            )
            if count
        ]
        if len(shapes) > 1:
            return self.json(
                path, shown, table, "R-SST-17", f"mixed shapes ({' and '.join(shapes)}): JSON text"
            )
        if stat.scalars:
            dtype, rule, reason = infer(stat)
            return self.column(path, shown, table, dtype, rule, reason)
        if stat.objects:
            return self.object(path, shown, table, choice)
        return self.list(path, stat, shown, table, choice)

    def presence_notes(self, path: Path, stat: PathStat) -> list[str]:
        notes = []
        share = self.profile.presence(path)
        if share < 1:
            notes.append(f"present in {share:.0%} of its parents in the sample: missing = NULL")
        if stat.nulls and stat.nulls < stat.count:
            notes.append(f"{stat.nulls:,} explicit nulls in the sample")
        return notes

    def column(
        self, path: Path, shown: str, table: TablePlan, dtype: Dtype, rule: str, reason: str
    ) -> FieldPlan:
        if not table.room():
            return self.extra(path, shown, table, self.full_reason())
        notes = self.presence_notes(path, self.profile.stats[path])
        name = table.add(shown, dtype, rule, reason, notes, tracked=True)
        return FieldPlan("column", path, shown, rule, reason, column=name)

    def json(self, path: Path, shown: str, table: TablePlan, rule: str, reason: str) -> FieldPlan:
        if not table.room():
            return self.extra(path, shown, table, self.full_reason())
        notes = self.presence_notes(path, self.profile.stats[path])
        name = table.add(shown, "json", rule, reason, notes, tracked=True)
        return FieldPlan("json", path, shown, rule, reason, column=name)

    def full_reason(self) -> str:
        return f"the table already has {self.options.max_columns} columns: kept in _extra"

    def extra(self, path: Path, shown: str, table: TablePlan, reason: str) -> FieldPlan:
        table.extra_column()
        table.extra_paths.append(shown)
        return FieldPlan("extra", path, shown, "R-SST-18", reason)

    def inside_explosion(self, path: Path, shown: str, table: TablePlan) -> FieldPlan:
        return self.json(
            path, shown, table, "R-SST-15b", "inside the exploded list (One table): JSON text"
        )

    def object(self, path: Path, shown: str, table: TablePlan, choice: str | None) -> FieldPlan:
        if path in self.maps:
            if self.exploding:
                return self.inside_explosion(path, shown, table)
            return self.map_table(path, shown, table, self.maps[path])
        if choice == "separate":
            if self.exploding:
                return self.inside_explosion(path, shown, table)
            return self.object_table(path, shown, table)
        if depth(path, table.row_path) > self.options.max_depth and choice != "flatten":
            return self.extra(
                path,
                shown,
                table,
                f"nested deeper than {self.options.max_depth} levels: kept in _extra as JSON",
            )
        rule, reason = (
            ("R-OVR", "flattened (your choice)")
            if choice == "flatten"
            else ("R-SST-11", f"object flattened into {shown}.… columns")
        )
        return FieldPlan(
            "flatten", path, shown, rule, reason, fields=self.object_fields(path, table)
        )

    def list(
        self, path: Path, stat: PathStat, shown: str, table: TablePlan, choice: str | None
    ) -> FieldPlan:
        items = self.profile.stats.get((*path, LIST))
        if items is None or items.count == items.nulls:
            if not table.room():
                return self.extra(path, shown, table, self.full_reason())
            rule = "R-SST-14b"
            reason = "the lists are always empty in the sample: only their length is kept"
            name = table.add(shown + ".count", "int", rule, reason)
            return FieldPlan("count", path, shown, rule, reason, count_column=name)
        kinds = [
            name
            for name, count in (
                ("values", items.scalars),
                ("objects", items.objects),
                ("lists", items.lists),
            )
            if count
        ]
        if len(kinds) > 1:
            return self.json(
                path, shown, table, "R-SST-17", f"lists mixing {' and '.join(kinds)}: JSON text"
            )
        if items.lists:
            return self.json(
                path, shown, table, "R-SST-17b", "lists of lists have no flat shape: JSON text"
            )
        if items.objects:
            return self.object_list(path, stat, shown, table, choice)
        return self.scalar_list(path, stat, items, shown, table, choice)

    def object_list(
        self, path: Path, stat: PathStat, shown: str, table: TablePlan, choice: str | None
    ) -> FieldPlan:
        if path == self.explode:
            return self.explode_field(path, shown, table)
        if max(stat.lengths) <= 1 and choice != "separate":
            return FieldPlan(
                "single",
                path,
                shown,
                "R-SST-15a",
                "lists of at most one object: read as an object",
                fields=self.object_fields((*path, LIST), table),
            )
        if choice == "flatten":
            raise StructuredError(
                f"{user_path(path)} is a list of objects: it can't be flattened. Choose "
                "'separate', 'json' or 'skip', or one_table=True.",
                rule="R-OVR",
            )
        if self.exploding:
            return self.inside_explosion(path, shown, table)
        if not table.room():
            return self.extra(path, shown, table, self.full_reason())
        reason = f"list of objects ({describe_lengths(stat)}): a linked table, one row per element"
        child = self.new_table(
            f"{table.name}.{shown}",
            "objects",
            (*path, LIST),
            self.locator((*path, LIST)),
            "R-SST-15",
            reason,
            table,
        )
        child.key("_id", "row number in this table (0 = the first)")
        child.key("_parent_id", f"_id of the {table.name} row")
        child.key("_position", f"position in {shown} (0 = the first)")
        count = table.add(shown + ".count", "int", "R-SST-15", f"number of elements of {shown}")
        plan = FieldPlan("objects", path, shown, "R-SST-15", reason, count_column=count)
        plan.table = child
        child.source = plan
        child.fields = self.object_fields((*path, LIST), child)
        return plan

    def scalar_list(
        self,
        path: Path,
        stat: PathStat,
        items: PathStat,
        shown: str,
        table: TablePlan,
        choice: str | None,
    ) -> FieldPlan:
        dtype, _, kind = infer(items)
        shortest, longest = min(stat.lengths), max(stat.lengths)
        fixed = shortest == longest and 1 <= longest <= 4 and choice != "separate"
        if choice == "flatten" or fixed:
            size = min(longest, MAX_POSITIONAL)
            if not table.room(size):
                return self.extra(path, shown, table, self.full_reason())
            rule, reason = (
                ("R-OVR", f"one column per position, {size} at most (your choice)")
                if choice == "flatten"
                else ("R-SST-13", f"lists of exactly {longest} values ({kind}): one column each")
            )
            names = [table.add(f"{shown}[{i}]", dtype, rule, reason) for i in range(size)]
            return FieldPlan("positional", path, shown, rule, reason, columns=names)
        if self.exploding:
            return self.inside_explosion(path, shown, table)
        joined = len(items.distinct) <= self.options.joined_distinct
        if not table.room(2 if joined else 1):
            return self.extra(path, shown, table, self.full_reason())
        reason = f"lists of values ({describe_lengths(stat)}): a linked table, one row per value"
        child = self.new_table(
            f"{table.name}.{shown}",
            "scalars",
            (*path, LIST),
            self.locator((*path, LIST)),
            "R-SST-14",
            reason,
            table,
        )
        child.key("_parent_id", f"_id of the {table.name} row")
        child.key("_position", f"position in {shown} (0 = the first)")
        child.value_column = child.add("value", dtype, "R-SST-10", kind, tracked=True)
        count = table.add(shown + ".count", "int", "R-SST-14", f"number of values of {shown}")
        joined_name = None
        if joined:
            joined_name = table.add(
                shown,
                "string",
                "R-SST-14",
                f"its {len(items.distinct)} distinct values joined by '; ' (30 at most)",
            )
        plan = FieldPlan(
            "scalars",
            path,
            shown,
            "R-SST-14",
            reason,
            count_column=count,
            joined_column=joined_name,
        )
        plan.table = child
        child.source = plan
        return plan

    def map_table(self, path: Path, shown: str, table: TablePlan, why: str) -> FieldPlan:
        values = self.profile.stats.get((*path, VALUES))
        reason = f"a dictionary ({why}): one row per key, not one column per key"
        child = self.new_table(
            f"{table.name}.{shown}",
            "map",
            (*path, VALUES),
            self.locator((*path, VALUES)),
            "R-SST-12",
            reason,
            table,
        )
        plan = FieldPlan("map", path, shown, "R-SST-12", reason)
        plan.table = child
        child.source = plan
        shapes = (values.scalars, values.objects, values.lists) if values else (0, 0, 0)
        if shapes[1] and not shapes[0] and not shapes[2]:
            child.map_values = "object"
            child.key("_id", "row number in this table (0 = the first)")
            child.key("_parent_id", f"_id of the {table.name} row")
            child.used.add("key")
            child.columns["key"] = ColumnSpec("key", "key", "string", "R-SST-12", "the key")
            child.fields = self.object_fields((*path, VALUES), child)
        else:
            child.key("_parent_id", f"_id of the {table.name} row")
            child.used.add("key")
            child.columns["key"] = ColumnSpec("key", "key", "string", "R-SST-12", "the key")
            if values is not None and values.scalars and not shapes[1] and not shapes[2]:
                child.map_values = "scalar"
                dtype, rule, kind = infer(values)
                child.value_column = child.add("value", dtype, rule, kind, tracked=True)
            else:
                child.map_values = "json"
                child.value_column = child.add(
                    "value", "json", "R-SST-17", "values of several shapes: JSON text", tracked=True
                )
        return plan

    def object_table(self, path: Path, shown: str, table: TablePlan) -> FieldPlan:
        reason = "a separate table (your choice): one row per parent"
        child = self.new_table(
            f"{table.name}.{shown}", "object", path, self.locator(path), "R-OVR", reason, table
        )
        child.key("_id", "row number in this table (0 = the first)")
        child.key("_parent_id", f"_id of the {table.name} row")
        plan = FieldPlan("object", path, shown, "R-OVR", reason)
        plan.table = child
        child.source = plan
        child.fields = self.object_fields(path, child)
        return plan

    def explode_field(self, path: Path, shown: str, table: TablePlan) -> FieldPlan:
        reason = f"One table: one row per element of {shown}, the record repeated"
        position = table.add(
            shown + "._position", "int", "R-SST-15b", f"position in {shown} (0 = the first)"
        )
        self.exploding = True
        fields = self.object_fields((*path, LIST), table)
        self.exploding = False
        return FieldPlan(
            "explode",
            path,
            shown,
            "R-SST-15b",
            reason,
            position_column=position,
            fields=fields,
        )


def make_plan(
    profile: Profile,
    maps: dict[Path, str],
    options: StructuredOptions,
    name: str,
    locator: str,
    rule: str,
    reason: str,
) -> tuple[TablePlan, set[Path]]:
    """The plan, with R-SST-21 (at most `max_tables` tables) and R-SST-15b (One table)."""
    explode = None
    if options.one_table:
        first = Planner(profile, maps, options, locator).plan(name, rule, reason)
        lists = [table for table in first.walk() if table.kind == "objects"]
        if len(lists) != 1 or lists[0].parent is not first or lists[0].source is None:
            names = ", ".join(table.name for table in lists) or "none"
            raise StructuredError(
                f"One table needs exactly one list of objects in the records; this document "
                f"has {len(lists)} ({names}).",
                rule="R-SST-15b",
            )
        explode = lists[0].source.path
    forced: set[Path] = set()
    while True:
        planner = Planner(profile, maps, options, locator, forced, explode)
        root = planner.plan(name, rule, reason)
        tables = list(root.walk())
        if len(tables) <= options.max_tables:
            return root, forced
        children = [(index, table) for index, table in enumerate(tables) if table.source]
        _, deepest = max(
            children,
            key=lambda item: (len(item[1].source.path), item[0]),  # type: ignore[union-attr]
        )
        assert deepest.source is not None
        forced.add(deepest.source.path)
