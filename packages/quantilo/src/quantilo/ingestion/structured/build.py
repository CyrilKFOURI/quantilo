"""Emitting the records through the plan into Arrow tables, in batches.

Every value is converted to its column's type when it is emitted; a value that doesn't fit is
left empty and counted (R-SST-16). A key the sample never showed becomes a column (R-SST-20), a
known path with an unexpected shape goes to `_extra`.
"""

from __future__ import annotations

import json
from typing import Any

import pyarrow as pa

from quantilo.ingestion.structured.plan import EXTRA, ColumnSpec, FieldPlan, TablePlan
from quantilo.ingestion.structured.profile import (
    INT64_MAX,
    INT64_MIN,
    INT_TEXT,
    MAX_NUMBER_TEXT,
    NUM_TEXT,
)

BATCH = 50_000
EXAMPLES = 3
ARROW_TYPES: dict[str, pa.DataType] = {
    "int": pa.int64(),
    "float": pa.float64(),
    "bool": pa.bool_(),
    "string": pa.string(),
    "json": pa.string(),
}


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    return dumps(value)


def convert(value: Any, spec: ColumnSpec) -> Any:
    """`value` in the column's type, or None (counted) when it doesn't fit."""
    dtype = spec.dtype
    if dtype == "string":
        return to_text(value)
    if dtype == "json":
        return dumps(value)
    kind = type(value)
    if dtype == "int":
        if kind is int and INT64_MIN <= value <= INT64_MAX:
            return value
        if kind is float and value.is_integer() and INT64_MIN <= value <= INT64_MAX:
            return int(value)
        if kind is str and len(value) <= MAX_NUMBER_TEXT and INT_TEXT.fullmatch(value):
            number = int(value)
            if INT64_MIN <= number <= INT64_MAX:
                return number
    elif dtype == "float":
        if kind is int or kind is float:
            try:
                return float(value)
            except OverflowError:
                pass
        elif kind is str and len(value) <= MAX_NUMBER_TEXT and NUM_TEXT.fullmatch(value):
            return float(value)
    elif dtype == "bool" and kind is bool:
        return value
    spec.unreadable += 1
    if len(spec.examples) < EXAMPLES:
        spec.examples.append(to_text(value)[:40])
    return None


class Builder:
    """Columns of one table as Python lists, turned into Arrow every `BATCH` rows."""

    def __init__(self, table: TablePlan) -> None:
        self.table = table
        self.columns: dict[str, list[Any]] = {}
        self.size = 0  # rows in the current batch
        self.rows = 0
        self.batches: list[tuple[int, dict[str, pa.Array]]] = []
        self.next_id = 0

    def new_id(self) -> int:
        number = self.next_id
        self.next_id += 1
        return number

    def append(self, row: dict[str, Any]) -> None:
        extra = row.pop(EXTRA, None)
        if extra:
            row[self.table.extra_column()] = dumps(extra)
        size = self.size
        for name, value in row.items():
            column = self.columns.get(name)
            if column is None:
                column = self.columns[name] = [None] * size
            elif len(column) < size:
                column.extend([None] * (size - len(column)))
            column.append(value)
        self.size += 1
        self.rows += 1
        if self.size >= BATCH:
            self.flush()

    def flush(self) -> None:
        if not self.size:
            return
        arrays = {}
        for name, values in self.columns.items():
            if len(values) < self.size:
                values.extend([None] * (self.size - len(values)))
            arrays[name] = pa.array(values, type=ARROW_TYPES[self.table.columns[name].dtype])
        self.batches.append((self.size, arrays))
        self.columns = {}
        self.size = 0

    def finish(self) -> pa.Table:
        self.flush()
        names = [name for name in self.table.columns if name != EXTRA]
        if EXTRA in self.table.columns:
            names.append(EXTRA)
        data = {}
        for name in names:
            kind = ARROW_TYPES[self.table.columns[name].dtype]
            chunks = [arrays.get(name, pa.nulls(size, kind)) for size, arrays in self.batches]
            data[name] = pa.chunked_array(chunks, type=kind)
        self.batches = []
        return pa.table(data) if data else pa.table({})


class Emitter:
    def __init__(self, root: TablePlan) -> None:
        self.root = root
        self.builders = {id(table): Builder(table) for table in root.walk()}

    def builder(self, table: TablePlan) -> Builder:
        return self.builders[id(table)]

    def record(self, number: int, record: dict[str, Any]) -> None:
        table = self.root
        row: dict[str, Any] = {"_id": number}
        pending: list[tuple[FieldPlan, Any]] = []
        self.fields(record, table.fields, row, table, number, "", pending)
        builder = self.builder(table)
        if not pending:
            builder.append(row)
            return
        plan, value = pending[0]
        items = value if isinstance(value, list) else [value]
        if not items:
            builder.append(row)
            return
        for position, item in enumerate(items):
            copy = dict(row)
            if EXTRA in row:
                copy[EXTRA] = dict(row[EXTRA])
            assert plan.position_column is not None
            copy[plan.position_column] = position
            if isinstance(item, dict):
                self.fields(item, plan.fields, copy, table, number, plan.display + ".", [])
            elif item is not None:
                self.extra(copy, plan.display, item, table)
            builder.append(copy)

    def fields(
        self,
        value: dict[str, Any],
        fields: dict[str, FieldPlan],
        row: dict[str, Any],
        table: TablePlan,
        row_id: int,
        prefix: str,
        pending: list[tuple[FieldPlan, Any]],
    ) -> None:
        for key, item in value.items():
            plan = fields.get(key)
            if plan is None:
                self.drift(row, table, prefix + key, item)
            elif plan.kind == "explode":
                if item is not None:
                    pending.append((plan, item))
            else:
                self.field(plan, item, row, table, row_id, pending)

    def extra(self, row: dict[str, Any], shown: str, value: Any, table: TablePlan) -> None:
        row.setdefault(EXTRA, {})[shown] = value
        table.shape_drift += 1

    def drift(self, row: dict[str, Any], table: TablePlan, shown: str, value: Any) -> None:
        """R-SST-20: a key the sample did not show."""
        name = table.drift.get(shown)
        if name is None:
            if not table.room():
                row.setdefault(EXTRA, {})[shown] = value
                table.extra_column()
                return
            dtype = "json" if isinstance(value, dict | list) else "string"
            name = table.add(
                shown,
                dtype,
                "R-SST-20",
                "seen only after the sample: added, empty before",
                tracked=True,
            )
            table.drift[shown] = name
        spec = table.columns[name]
        spec.present += 1
        if value is None:
            spec.nulls += 1
        else:
            row[name] = convert(value, spec)

    def field(
        self,
        plan: FieldPlan,
        value: Any,
        row: dict[str, Any],
        table: TablePlan,
        row_id: int,
        pending: list[tuple[FieldPlan, Any]],
    ) -> None:
        kind = plan.kind
        if kind in ("column", "json"):
            assert plan.column is not None
            spec = table.columns[plan.column]
            spec.present += 1
            if value is None:
                spec.nulls += 1
            elif kind == "column" and isinstance(value, dict | list):
                self.extra(row, plan.display, value, table)
            else:
                row[plan.column] = convert(value, spec)
            return
        if value is None or kind == "skip":
            return
        if kind == "extra":
            row.setdefault(EXTRA, {})[plan.display] = value
        elif kind == "flatten":
            if isinstance(value, dict):
                self.fields(value, plan.fields, row, table, row_id, plan.display + ".", pending)
            else:
                self.extra(row, plan.display, value, table)
        elif kind == "single":
            self.single(plan, value, row, table, row_id, pending)
        elif kind == "positional":
            self.positional(plan, value, row, table)
        elif kind == "count":
            if isinstance(value, list):
                assert plan.count_column is not None
                row[plan.count_column] = len(value)
                if value:
                    self.extra(row, plan.display, value, table)
            else:
                self.extra(row, plan.display, value, table)
        elif kind == "scalars":
            self.scalars(plan, value, row, table, row_id)
        elif kind == "objects":
            self.objects(plan, value, row, table, row_id)
        elif kind == "map":
            self.mapping(plan, value, row, table, row_id)
        elif kind == "object":
            self.object(plan, value, row, table, row_id)

    def single(
        self,
        plan: FieldPlan,
        value: Any,
        row: dict[str, Any],
        table: TablePlan,
        row_id: int,
        pending: list[tuple[FieldPlan, Any]],
    ) -> None:
        if isinstance(value, dict):
            value = [value]
        if not isinstance(value, list):
            self.extra(row, plan.display, value, table)
            return
        if value and isinstance(value[0], dict):
            self.fields(value[0], plan.fields, row, table, row_id, plan.display + ".", pending)
        elif value and value[0] is not None:
            self.extra(row, plan.display, value, table)
            return
        if len(value) > 1:  # more than one after the sample: kept, not lost
            self.extra(row, plan.display, value[1:], table)

    def positional(
        self, plan: FieldPlan, value: Any, row: dict[str, Any], table: TablePlan
    ) -> None:
        if not isinstance(value, list):
            self.extra(row, plan.display, value, table)
            return
        for name, item in zip(plan.columns, value, strict=False):
            if item is not None and not isinstance(item, dict | list):
                row[name] = convert(item, table.columns[name])
            elif item is not None:
                self.extra(row, plan.display, value, table)
                return
        if len(value) > len(plan.columns):
            self.extra(row, plan.display, value, table)

    def scalars(
        self, plan: FieldPlan, value: Any, row: dict[str, Any], table: TablePlan, row_id: int
    ) -> None:
        child = plan.table
        if not isinstance(value, list) or child is None or child.value_column is None:
            self.extra(row, plan.display, value, table)
            return
        assert plan.count_column is not None
        row[plan.count_column] = len(value)
        if plan.joined_column is not None:
            row[plan.joined_column] = "; ".join(to_text(item) for item in value if item is not None)
        builder = self.builder(child)
        spec = child.columns[child.value_column]
        for position, item in enumerate(value):
            line: dict[str, Any] = {"_parent_id": row_id, "_position": position}
            spec.present += 1
            if item is None:
                spec.nulls += 1
            elif isinstance(item, dict | list):
                line[EXTRA] = {"value": item}
                child.shape_drift += 1
            else:
                line[child.value_column] = convert(item, spec)
            builder.append(line)

    def objects(
        self, plan: FieldPlan, value: Any, row: dict[str, Any], table: TablePlan, row_id: int
    ) -> None:
        child = plan.table
        if isinstance(value, dict):
            value = [value]
        if not isinstance(value, list) or child is None:
            self.extra(row, plan.display, value, table)
            return
        assert plan.count_column is not None
        row[plan.count_column] = len(value)
        builder = self.builder(child)
        for position, item in enumerate(value):
            number = builder.new_id()
            line: dict[str, Any] = {"_id": number, "_parent_id": row_id, "_position": position}
            if isinstance(item, dict):
                self.fields(item, child.fields, line, child, number, "", [])
            elif item is not None:
                self.extra(line, "value", item, child)
            builder.append(line)

    def mapping(
        self, plan: FieldPlan, value: Any, row: dict[str, Any], table: TablePlan, row_id: int
    ) -> None:
        child = plan.table
        if not isinstance(value, dict) or child is None:
            self.extra(row, plan.display, value, table)
            return
        builder = self.builder(child)
        for key, item in value.items():
            line: dict[str, Any] = {"_parent_id": row_id, "key": key}
            if child.map_values == "object":
                number = builder.new_id()
                line["_id"] = number
                if isinstance(item, dict):
                    self.fields(item, child.fields, line, child, number, "", [])
                elif item is not None:
                    self.extra(line, "value", item, child)
            else:
                assert child.value_column is not None
                spec = child.columns[child.value_column]
                spec.present += 1
                if item is None:
                    spec.nulls += 1
                elif child.map_values == "scalar" and isinstance(item, dict | list):
                    self.extra(line, "value", item, child)
                else:
                    line[child.value_column] = convert(item, spec)
            builder.append(line)

    def object(
        self, plan: FieldPlan, value: Any, row: dict[str, Any], table: TablePlan, row_id: int
    ) -> None:
        child = plan.table
        if not isinstance(value, dict) or child is None:
            self.extra(row, plan.display, value, table)
            return
        builder = self.builder(child)
        number = builder.new_id()
        line: dict[str, Any] = {"_id": number, "_parent_id": row_id}
        self.fields(value, child.fields, line, child, number, "", [])
        builder.append(line)

    def finish(self) -> dict[int, tuple[TablePlan, pa.Table, int]]:
        return {
            key: (builder.table, builder.finish(), builder.rows)
            for key, builder in self.builders.items()
        }
