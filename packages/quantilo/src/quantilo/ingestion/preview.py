"""Quick look at a source before importing it."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import duckdb

from quantilo.ingestion.errors import readable_errors
from quantilo.ingestion.image_sets import image_source
from quantilo.ingestion.normalize import normalized_sql, note_lines
from quantilo.ingestion.options import CsvOptions, Format, SourceOptions
from quantilo.ingestion.reader import SourceReader, excel_sheets, load_npy
from quantilo.ingestion.sniff import sniff_path
from quantilo.ingestion.sources import (
    DOCUMENT_FORMATS,
    TEXT_EXTENSIONS,
    SourceFile,
    list_source_files,
)
from quantilo.ingestion.sql import lit

PREVIEW_ROWS = 20
EXACT_COUNT_MAX_BYTES = 50 * 1024 * 1024


@dataclass
class ColumnInfo:
    name: str
    dtype: str


@dataclass
class FileInfo:
    name: str
    format: Format
    size: int


@dataclass
class Detection:
    """What the first bytes say about the file (082): its kind, its format, the rule and why."""

    kind: Literal["table", "document", "image", "unsupported"]
    format: str
    label: str
    rule: str
    reason: str
    mismatch: bool = False  # R-DET-01: the name says something else (the content won)
    name_format: str | None = None


@dataclass
class RuleNote:
    rule: str
    weight: float
    message: str


@dataclass
class TableArea:
    """Where a detected table sits in its sheet (1-based, inclusive), to draw it on the original."""

    top: int
    left: int
    bottom: int
    right: int
    header_rows: list[int]
    first_data_row: int
    dropped_rows: list[int] = field(default_factory=list)  # totals, notes, templates set aside


@dataclass
class TableChoice:
    """One table the source gives (080b: a table of a sheet; 085: a linked table of a document)."""

    id: str  # what to pass back in `SourceOptions.table`
    name: str  # the dataset name proposed
    location: str  # "Sheet1!A4:L14", "$.orders[*].items[*]"
    rows: int
    columns: int
    level: Literal["high", "review", "low"]
    confidence: float
    summary: str  # the sentence above the result
    selected: bool  # ticked at first (high confidence, not metadata)
    kind: str = "table"
    sheet: str | None = None
    parent: str | None = None  # a document's child table: the table it belongs to
    area: TableArea | None = None
    reasons: list[RuleNote] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    proposals: list[str] = field(default_factory=list)  # K rules, applied only with `reshape`
    column_names: list[str] = field(default_factory=list)  # C1: the names read (S1 compares them)


@dataclass
class TableStack:
    """S1 (C1): tables of one workbook with the same column names, in the same order (case and
    spaces aside): they can be imported as one dataset, stacked, with a `source_sheet` column."""

    rule: str
    tables: list[str]  # their ids, in the workbook's order
    locations: list[str]
    columns: list[str]
    rows: int
    reason: str


MIN_STACK_COLUMNS = 2


def stack_groups(path: Path) -> list[TableStack]:
    """S1: every group of two tables or more that share their columns, across the sheets. Reads
    every sheet's layout: asked for on demand (Advanced), never when the import screen opens."""
    groups: dict[tuple[str, ...], list[TableChoice]] = {}
    for choice in sheet_choices(path):
        if choice.kind != "table" or len(choice.column_names) < MIN_STACK_COLUMNS:
            continue
        key = tuple(" ".join(name.split()).casefold() for name in choice.column_names)
        groups.setdefault(key, []).append(choice)
    stacks = []
    for members in groups.values():
        if len(members) < 2:
            continue
        names = members[0].column_names
        shown = ", ".join(f"'{name}'" for name in names[:4]) + ("…" if len(names) > 4 else "")
        places = ", ".join(choice.location for choice in members[:6])
        stacks.append(
            TableStack(
                rule="S1",
                tables=[choice.id for choice in members],
                locations=[choice.location for choice in members],
                columns=list(names),
                rows=sum(choice.rows for choice in members),
                reason=(
                    f"{len(members)} tables ({places}{'…' if len(members) > 6 else ''}) have the "
                    f"same {len(names)} columns ({shown}): they can be stacked into one dataset, "
                    "with a column 'source_sheet' saying where each row comes from (S1)."
                ),
            )
        )
    return stacks


@dataclass
class TableLink:
    """`parent.parent_key` = `child.child_key` between two tables of a document (R-KEY-03)."""

    parent: str
    child: str
    parent_key: str
    child_key: str
    rule: str


@dataclass
class Preview:
    kind: Literal["file", "folder"]
    format: Format | Literal["mixed", "other"]
    files: list[FileInfo]
    columns: list[ColumnInfo]
    rows: list[list[str | None]]
    row_estimate: int | None
    csv: CsvOptions | None = None
    sheets: list[str] | None = None
    schema_mismatch: bool = False
    suggested_name: str = ""
    warnings: list[str] = field(default_factory=list)
    # Excel sheet -> header row read (080, J1): pass it back in the import options
    header_row: dict[str, int] | None = None
    detected: Detection | None = None  # 082: a single file, told by its first bytes
    # 080b / 085: the tables the source gives (None: a plain file, a folder); `table` is the one
    # shown, `links` the keys between a document's tables
    tables: list[TableChoice] | None = None
    table: str | None = None
    links: list[TableLink] = field(default_factory=list)


def detected_header_rows(path: Path, sheets: list[str]) -> tuple[dict[str, int], list[str]]:
    """W1: the header row of each sheet whose first table is detected with confidence >= 0.8,
    when it is not the first non-empty row; with a sentence per sheet. Never raises (W4)."""
    from quantilo.ingestion.layout import detect_layout

    try:
        layout = detect_layout(path, sheets)
    except Exception:
        return {}, []
    rows: dict[str, int] = {}
    notes: list[str] = []
    for sheet in layout.sheets:
        row = sheet.preview_header()
        if row is not None:
            rows[sheet.name] = row
            above = (
                f"row {sheet.first_row}"
                if row - sheet.first_row == 1
                else (f"rows {sheet.first_row} to {row - 1}")
            )
            notes.append(
                f"Header read from row {row} of '{sheet.name}' (confidence "
                f"{sheet.tables[0].confidence:.0%}): {above} above it (title, notes) skipped."
            )
    return rows, notes


def reading_notes(reader: SourceReader, files: list[SourceFile]) -> list[str]:
    """092: a sentence per rule that chose how the files are read (R-ING-02 to R-ING-04)."""
    notes: list[str] = []
    for file in files:
        prefix = f"{file.path.name}: " if len(files) > 1 else ""
        if file.format == "csv" and file.path.suffix.lower() in TEXT_EXTENSIONS:
            delimiter = reader.csv_options_for(file).delimiter
            shown = {" ": "spaces", "\t": "tabs"}.get(delimiter, repr(delimiter))
            notes.append(
                f"{prefix}{file.path.suffix} file read as delimited text, columns separated by "
                f"{shown} (R-ING-04)."
            )
        notes += [prefix + note for note in reader.csv_notes.get(file.path, [])]
        notes += [prefix + note for note in reader.log_notes.get(file.path, [])]
    return list(dict.fromkeys(notes))


def estimate_rows(reader: SourceReader, file: SourceFile) -> int:
    connection = reader.connection
    if file.format == "parquet":
        row = connection.execute(
            f"SELECT SUM(num_rows) FROM parquet_file_metadata({lit(str(file.path))})"
        ).fetchone()
        return int(row[0]) if row and row[0] is not None else 0
    if file.format == "excel" or file.format in DOCUMENT_FORMATS:
        return reader.excel_rows.get(file.path, 0)
    if file.format == "npy":
        return int(load_npy(file.path).shape[0])
    if file.format == "log":  # one entry per line, about (090)
        with file.path.open("rb") as handle:
            sample = handle.read(1024 * 1024)
        lines = max(sample.count(b"\n"), 1)
        return max(1, int(file.size / max(len(sample), 1) * lines))
    if file.size <= EXACT_COUNT_MAX_BYTES:
        row = connection.execute(f"SELECT count(*) FROM ({reader.csv_sql(file)})").fetchone()
        return int(row[0]) if row else 0
    with file.path.open("rb") as handle:
        sample = handle.read(1024 * 1024)
    lines = max(sample.count(b"\n"), 1)
    return int(file.size / len(sample) * lines)


def detection_of(path: Path) -> Detection:
    found = sniff_path(path)
    return Detection(
        kind=found.kind,
        format=found.format,
        label=found.label,
        rule=found.rule,
        reason=found.reason,
        mismatch=found.mismatch,
        name_format=found.name_format,
    )


def sheet_choices(path: Path, sheets: list[str] | None = None) -> list[TableChoice]:
    """080b: every table of the sheets read (all when None), with its place, its rules and its
    confidence. The import screen asks sheet by sheet: nothing slow runs at its opening."""
    from quantilo.ingestion.layout import detect_layout

    try:
        layout = detect_layout(path, sheets)
    except Exception:  # W4: the detection never blocks the preview
        return []
    choices: list[TableChoice] = []
    stem = path.stem
    for sheet in layout.sheets:
        for index, table in enumerate(sheet.tables):
            body = table.bottom - table.first_data_row + 1 if table.first_data_row else 0
            title = table.title or (sheet.name if len(sheet.tables) == 1 else None)
            name = f"{stem} - {title}" if title else f"{stem} - {sheet.name} {index + 1}"
            dropped = sorted({row for rows in table.dropped_rows.values() for row in rows})
            choices.append(
                TableChoice(
                    id=table.id,
                    name=name[:200],
                    location=f"{sheet.name}!{table.range}",
                    rows=max(body, 0),
                    columns=len(table.columns) or table.right - table.left + 1,
                    level=table.level,
                    confidence=round(table.confidence, 3),
                    summary=table.summary(),
                    selected=table.level == "high",
                    kind=table.kind,
                    sheet=sheet.name,
                    area=TableArea(
                        top=table.top,
                        left=table.left,
                        bottom=table.bottom,
                        right=table.right,
                        header_rows=list(table.header_rows),
                        first_data_row=table.first_data_row,
                        dropped_rows=dropped,
                    ),
                    reasons=[RuleNote(r.rule, r.weight, r.message) for r in table.reasons],
                    warnings=list(table.warnings),
                    proposals=[t.message for t in table.transforms if not t.default],
                    column_names=[plan.name for plan in table.columns][:500],
                )
            )
    return choices


def token_notes(options: SourceOptions) -> list[str]:
    """T2u: the sentence the result shows when the user changed the placeholders."""
    from quantilo.ingestion.layout.cells import token_set

    notes = []
    empty, text = token_set(options.null_tokens), token_set(options.text_tokens)
    if empty:
        listed = ", ".join(repr(t) for t in sorted(empty))
        notes.append(f"{listed} read as empty in every column, as you asked (T2u).")
    if text:
        listed = ", ".join(repr(t) for t in sorted(text))
        notes.append(f"{listed} kept as text, never read as empty, as you asked (T2u).")
    return notes


def first_sheet_with_a_table(path: Path, look: int = 5) -> tuple[list[str], list[str]]:
    """R-ING-07 (080b): the first sheet that holds a table, among the first five; with the empty
    sheets skipped before it (a cover sheet left blank)."""
    from quantilo.ingestion.layout import detect_layout

    names = excel_sheets(path)
    for index, name in enumerate(names[:look]):
        try:
            sheet = detect_layout(path, [name]).sheets
        except Exception:  # W4
            return names[:1], []
        if sheet and sheet[0].tables:
            return [name], names[:index]
    return names[:1], []


def document_choices(
    path: Path, options: SourceOptions
) -> tuple[list[TableChoice], list[TableLink]]:
    """085: the linked tables of a JSON, JSON Lines or XML document (083), each with its rule."""
    from quantilo.ingestion.documents import read_document

    layout = read_document(path, options.records_path, bool(options.one_table))
    stem = path.stem
    choices = []
    for index, table in enumerate(layout.tables):
        reasons = [RuleNote(table.rule, 0.0, table.reason)]
        if table.natural_key:
            reasons.append(RuleNote("R-KEY-02", 0.0, f"'{table.natural_key}' identifies each row."))
        if index == 0:
            reasons = [RuleNote(r.rule, r.weight, r.message) for r in layout.reasons] + reasons
        names = ", ".join(column.name for column in table.columns[:6])
        more = f" and {len(table.columns) - 6} more" if len(table.columns) > 6 else ""
        choices.append(
            TableChoice(
                id=table.name,
                name=f"{stem} - {table.name}" if table.name != stem else stem,
                location=table.locator,
                rows=table.rows,
                columns=len(table.columns),
                level=layout.level,
                confidence=round(layout.confidence, 3),
                summary=f"{table.rows:,} rows, {len(table.columns)} columns: {names}{more}.",
                selected=table.selected and table.kind != "meta",
                kind=table.kind,
                parent=table.parent,
                reasons=reasons,
                warnings=list(table.warnings),
            )
        )
    links = [
        TableLink(r.parent, r.child, r.parent_key, r.child_key, r.rule) for r in layout.relations
    ]
    return choices, links


def preview_source(path: Path, options: SourceOptions) -> Preview:
    with readable_errors():
        detected = detection_of(path) if path.is_file() else None
        if detected is not None and detected.kind == "image":
            # 082: an image is not a table; the import screen shows its own step for it.
            # Anything else not read is refused by name below (R-DET-03, `sources.refusal`).
            return Preview(
                kind="file",
                format="other",
                files=[],
                columns=[],
                rows=[],
                row_estimate=None,
                suggested_name=path.stem,
                detected=detected,
            )
        images = image_source(path)  # 106: a folder or a ZIP of images (R-VIS-SRC-01, 02)
        if images is not None and images.kind != "parquet":
            return Preview(
                kind="folder" if path.is_dir() else "file",
                format="other",
                files=[],
                columns=[],
                rows=[],
                row_estimate=None,
                suggested_name=path.stem if path.is_file() else path.name,
                detected=Detection("image", "images", images.label, images.rule, images.reason),
            )
        if images is not None:  # R-VIS-SRC-03: read as a table, offered as an image set
            detected = Detection("image", "parquet", images.label, images.rule, images.reason)
        all_files = list_source_files(path)
        files = list_source_files(path, options.files) if options.files else all_files
        header_rows: dict[str, int] = {}
        notes: list[str] = [file.note for file in files if file.note]
        tables: list[TableChoice] | None = None
        links: list[TableLink] = []
        shown = options.table
        single = files[0] if len(files) == 1 else None
        if single is not None and single.format == "excel":
            read = options.sheets
            if read is None and options.table is not None:
                read = [options.table.rpartition("!")[0]]
            elif read is None:
                read, empty = first_sheet_with_a_table(single.path)
                if empty:
                    options = options.model_copy(update={"sheets": read})
                    notes.append(
                        f"{', '.join(repr(name) for name in empty)} "
                        f"{'is' if len(empty) == 1 else 'are'} empty: '{read[0]}' is shown "
                        "first (R-ING-07)."
                    )
            tables = sheet_choices(single.path, read)
            if options.table is not None:
                notes += token_notes(options)
            if options.header_row is None and options.table is None:
                header_rows, header_notes = detected_header_rows(single.path, read)
                notes += header_notes
                if header_rows:
                    options = options.model_copy(update={"header_row": header_rows})
        if single is not None and single.format in DOCUMENT_FORMATS:
            from quantilo.ingestion.documents import main_table, read_document

            tables, links = document_choices(single.path, options)
            if shown is None:
                document = read_document(single.path, options.records_path, bool(options.one_table))
                shown = main_table(document)
        connection = duckdb.connect()
        try:
            reader = SourceReader(connection, options)
            reader.npy_limit = PREVIEW_ROWS
            sql, changed = normalized_sql(connection, reader.combined_sql(files))
            columns = [ColumnInfo(name, dtype) for name, dtype in reader.columns(sql)]
            rows = connection.execute(
                f"SELECT COLUMNS(*)::VARCHAR FROM ({sql}) LIMIT {PREVIEW_ROWS}"
            ).fetchall()
            first = files[0]
            source_format: Format | Literal["mixed", "other"] = (
                first.format if len({file.format for file in files}) == 1 else "mixed"
            )
            return Preview(
                kind="folder" if path.is_dir() else "file",
                format=source_format,
                files=[FileInfo(f.path.name, f.format, f.size) for f in all_files],
                columns=columns,
                rows=[list(row) for row in rows],
                row_estimate=sum(estimate_rows(reader, file) for file in files),
                csv=reader.csv_options_for(first) if first.format == "csv" else None,
                sheets=excel_sheets(first.path) if first.format == "excel" else None,
                schema_mismatch=reader.schema_mismatch(files),
                suggested_name=path.stem if path.is_file() else path.name,
                warnings=[*notes, *reading_notes(reader, files), *note_lines(changed)],
                header_row=options.header_row,
                detected=detected,
                tables=tables,
                table=shown,
                links=links,
            )
        finally:
            connection.close()
