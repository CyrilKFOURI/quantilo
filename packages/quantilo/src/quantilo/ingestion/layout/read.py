"""A detected table as a pyarrow Table: `read_table(path, table, reshape=False)`.

Only the table's range is read again (openpyxl `read_only`, or the CSV rows). By default the
lossless
clean-ups are applied: rows set aside (totals, templates, section labels, repeated headers) are left
out, sections and hierarchy become columns, types are cleaned (T1–T3), blank group labels are filled
down (C12), a combined first column is split (V3). `reshape=True` also applies the proposals (K2,
K3, K3c, K4); derived columns (K1) are left out of a long format.
"""

import warnings
from pathlib import Path

import openpyxl
import pyarrow as pa

from quantilo.ingestion.layout.cells import clean_text, is_blank
from quantilo.ingestion.layout.csv import read_csv_rows
from quantilo.ingestion.layout.detect import CSV_EXTENSIONS
from quantilo.ingestion.layout.header import unique_names
from quantilo.ingestion.layout.hierarchy import ancestors, clean_label, has_parents
from quantilo.ingestion.layout.model import DetectedTable, Transform
from quantilo.ingestion.layout.types import convert

Row = tuple[int, list[object], str | None]


def _column(values: list[object]) -> pa.Array:
    try:
        return pa.array(values)
    except (pa.ArrowInvalid, pa.ArrowTypeError):
        return pa.array([None if v is None else str(v) for v in values], pa.string())


def source_rows(
    path: Path, table: DetectedTable, delimiter: str | None = None
) -> list[tuple[int, list[object]]]:
    """(1-based row, values of the table's columns) from the first data row to the bottom."""
    width = table.right - table.left + 1
    start = table.first_data_row or table.top
    if path.suffix.lower() in CSV_EXTENSIONS:
        rows, _, _ = read_csv_rows(path, delimiter)
        result = []
        for number in range(start, min(table.bottom, len(rows)) + 1):
            cells = rows[number - 1][table.left - 1 : table.right]
            values: list[object] = [None if is_blank(v) else v for v in cells]
            result.append((number, values + [None] * (width - len(values))))
        return result
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        sheet = workbook[table.sheet]
        result = []
        for offset, row in enumerate(
            sheet.iter_rows(
                min_row=start,
                max_row=table.bottom,  # type: ignore[union-attr]
                min_col=table.left,
                max_col=table.right,
                values_only=True,
            )
        ):
            values = [None if is_blank(v) else v for v in row]
            result.append((start + offset, values + [None] * (width - len(values))))
        return result
    finally:
        workbook.close()


def _kept_rows(table: DetectedTable, rows: list[tuple[int, list[object]]]) -> list[Row]:
    dropped = {r for key, numbers in table.dropped_rows.items() for r in numbers}
    kept: list[Row] = []
    section: str | None = None
    for number, values in rows:
        if number in table.sections:
            section = table.sections[number]
        if number in dropped or all(v is None for v in values):
            continue
        kept.append((number, values, section))
    return kept


def clean_columns(table: DetectedTable, rows: list[Row]) -> dict[str, list[object]]:
    """The table's columns after the default clean-ups, in order."""
    data: dict[str, list[object]] = {}
    names = [plan.name for plan in table.columns]
    if table.sections:
        name = unique_names([*names, "Section"])[-1]
        data[name] = [section for _, _, section in rows]
    numbers = [number for number, _, _ in rows]
    if table.levels and table.hierarchy_col is not None:
        label_index = table.hierarchy_col - table.left
        labels = {number: clean_label(values[label_index]) for number, values, _ in rows}
        levels = {n: table.levels[n] for n in numbers if n in table.levels}
        if table.level_col is None:
            data[unique_names([*names, "level"])[-1]] = [levels.get(n) for n in numbers]
        if has_parents(levels, numbers):
            parents = ancestors(levels, numbers, labels)  # type: ignore[arg-type]
            depth = max(levels.values(), default=0)
            for level in range(depth):
                data[unique_names([*names, f"Level_{level + 1}"])[-1]] = [
                    parents[n][level] for n in numbers
                ]
    for plan in table.columns:
        index = plan.source_col - table.left
        values, flags = [], []
        for _, row, _ in rows:
            raw = row[index]
            if table.hierarchy_col == plan.source_col and table.levels:
                raw = clean_label(raw)
            value, flag = convert(raw, plan)
            values.append(value)
            flags.append(flag)
        data[plan.name] = values
        if plan.flags:
            data[f"{plan.name} flag"] = flags
    for transform in table.transforms:
        if not transform.default:
            continue
        if transform.kind == "fill_down":
            for name in transform.columns:
                last: object = None
                last_section: object = None
                for i, (_, _, section) in enumerate(rows):
                    if section != last_section:
                        last, last_section = None, section
                    plan = next(p for p in table.columns if p.name == name)
                    if rows[i][1][plan.source_col - table.left] is None:  # blank, not "(blank)"
                        data[name][i] = last
                    else:
                        last = data[name][i]
        elif transform.kind == "split_combined" and transform.columns[0] in data:
            source = transform.columns[0]
            parts = [str(v).split(",") if v is not None else [] for v in data.pop(source)]
            split: dict[str, list[object]] = {
                name: [p[i].strip() if i < len(p) else None for p in parts]
                for i, name in enumerate(transform.into)
            }
            data = {**split, **data}
    return data


def _melt_sparse(
    data: dict[str, list[object]], transform: Transform, derived: set[str], numbers: list[int]
) -> dict[str, list[object]]:
    grouped = {c for cols in transform.groups.values() for c in cols}
    ids = [name for name in data if name not in grouped and name not in derived]
    types = [f"{group} type" for group in transform.groups]
    out: dict[str, list[object]] = {name: [] for name in [*ids, *types, "Amount", "row_id"]}
    for i, number in enumerate(numbers):
        for group, columns in transform.groups.items():
            for column in columns:
                value = data[column][i]
                if value in (None, 0):
                    continue
                for name in ids:
                    out[name].append(data[name][i])
                label = column.removeprefix(f"{group} · ")
                for name in types:
                    out[name].append(label if name == f"{group} type" else None)
                out["Amount"].append(value)
                out["row_id"].append(number)
    return out


def _melt_groups(
    data: dict[str, list[object]], transform: Transform, derived: set[str], key: str
) -> dict[str, list[object]]:
    """K3/K3b (periods) and K3c (numbered groups): one row per period or number."""
    melted = set(transform.groups)
    ids = [name for name in data if name not in melted and name not in derived]
    blocks: dict[str, dict[str, str]] = {}
    for column, (label, measure) in transform.groups.items():
        blocks.setdefault(label, {})[measure] = column
    measures = transform.into[1:]
    out: dict[str, list[object]] = {name: [] for name in [*ids, key, *measures]}
    count = len(next(iter(data.values()), []))
    for i in range(count):
        for label, columns in blocks.items():
            values = [data[columns[m]][i] if m in columns else None for m in measures]
            if all(v is None for v in values):
                continue
            for name in ids:
                out[name].append(data[name][i])
            out[key].append(label)
            for measure, value in zip(measures, values, strict=True):
                out[measure].append(value)
    return out


def _transpose(data: dict[str, list[object]]) -> dict[str, list[object]]:
    names = list(data)
    labels = [
        clean_text(v) if v is not None else f"field_{i + 1}" for i, v in enumerate(data[names[0]])
    ]
    labels = unique_names(labels)
    return {label: [data[column][i] for column in names[1:]] for i, label in enumerate(labels)}


def read_table(
    path: Path, table: DetectedTable, reshape: bool = False, delimiter: str | None = None
) -> pa.Table:
    rows = _kept_rows(table, source_rows(path, table, delimiter))
    data = clean_columns(table, rows)
    if reshape:
        derived = {plan.name for plan in table.columns if plan.role == "derived"}
        numbers = [number for number, _, _ in rows]
        for transform in table.transforms:
            if transform.default:
                continue
            if transform.kind == "melt_sparse":
                data = _melt_sparse(data, transform, derived, numbers)
            elif transform.kind == "melt_periods":
                data = _melt_groups(data, transform, derived, "Period")
            elif transform.kind == "melt_numbered":
                data = _melt_groups(data, transform, derived, "n")
            elif transform.kind == "transpose":
                data = _transpose(data)
            break  # one restructuring at a time: they do not compose
    return pa.table({name: _column(values) for name, values in data.items()})
