"""A bounded grid of cell values and styles (G rules, J7 bounds).

One openpyxl pass in `read_only` mode, which exposes bold, fill, border and indent cell by cell and
keeps memory flat at any file size. What `read_only` does not expose (merged cells, the frozen pane,
formulas without a cached value) is read in the sheet's XML with `zipfile` and `re`: pure Python,
so the same code runs in the browser (Pyodide). No formula is evaluated, no macro runs.
"""

import html
import posixpath
import re
import warnings
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import openpyxl
from openpyxl.utils.cell import column_index_from_string as column_index
from openpyxl.utils.cell import coordinate_from_string, range_boundaries

from quantilo.ingestion.layout.cells import is_blank

Merge = tuple[int, int, int, int]  # 0-based inclusive (top, left, bottom, right)


@dataclass(frozen=True)
class Limits:
    """Study § 7: the analysis reads at most this much; past it the body end is streamed."""

    max_rows: int = 2000
    max_cols: int = 200
    max_sheets: int = 30
    max_cells: int = 2_000_000
    max_xml_bytes: int = 200 * 1024 * 1024


LIMITS = Limits()


@dataclass
class Grid:
    """A sheet's top-left window: values (0-based rows and columns) and the styles that matter."""

    name: str
    values: list[list[object]]
    bold: set[tuple[int, int]] = field(default_factory=set)
    filled: set[tuple[int, int]] = field(default_factory=set)
    bordered: set[tuple[int, int]] = field(default_factory=set)
    indent: dict[tuple[int, int], int] = field(default_factory=dict)
    merges: list[Merge] = field(default_factory=list)
    freeze: tuple[int, int] | None = None  # first unfrozen (row, col), 0-based
    uncached: set[tuple[int, int]] = field(default_factory=set)
    truncated: bool = False
    warnings: list[str] = field(default_factory=list)
    styled: bool = True  # False for a CSV: content rules weigh more

    @property
    def n_rows(self) -> int:
        return len(self.values)

    @property
    def n_cols(self) -> int:
        return max((len(row) for row in self.values), default=0)

    def value(self, row: int, col: int) -> object:
        if 0 <= row < len(self.values) and 0 <= col < len(self.values[row]):
            return self.values[row][col]
        return None

    def occupied(self, row: int, col: int) -> bool:
        """A value, or a formula whose value was not cached (T4: the cell is part of the table)."""
        return self.value(row, col) is not None or (row, col) in self.uncached

    def row_cols(self, row: int, left: int = 0, right: int | None = None) -> list[int]:
        right = self.n_cols - 1 if right is None else right
        return [c for c in range(left, right + 1) if self.occupied(row, c)]

    def is_styled(self, row: int, col: int) -> bool:
        cell = (row, col)
        return cell in self.bold or cell in self.filled or cell in self.bordered

    @classmethod
    def from_rows(cls, name: str, rows: list[list[object]], styled: bool = False) -> "Grid":
        """A grid from plain values (CSV fields, tests): blanks become None (G2)."""
        values = [[None if is_blank(v) else v for v in row] for row in rows]
        grid = cls(name=name, values=values, styled=styled)
        grid.trim()
        return grid

    def trim(self) -> None:
        """G1: only the real extent counts (declared dimensions can run to column ALR)."""
        last_row = -1
        width = 0
        for r, row in enumerate(self.values):
            cols = [c for c, v in enumerate(row) if v is not None or (r, c) in self.uncached]
            if cols:
                last_row = r
                width = max(width, cols[-1] + 1)
        self.values = [
            row[:width] + [None] * (width - len(row[:width])) for row in self.values[: last_row + 1]
        ]


@dataclass
class SheetXml:
    merges: list[Merge] = field(default_factory=list)
    freeze: tuple[int, int] | None = None
    formulas: set[tuple[int, int]] = field(default_factory=set)  # formula cells without a value
    skipped: bool = False


_SHEET = re.compile(rb"<(?:\w+:)?sheet\b([^>]*)/?>")
_RELATIONSHIP = re.compile(rb"<(?:\w+:)?Relationship\b([^>]*)/?>")
_QATTR = re.compile(rb'([\w:]+)="([^"]*)"')
_MERGE = re.compile(rb'<(?:\w+:)?mergeCell\b[^>]*?\bref="([A-Z]+\d+(?::[A-Z]+\d+)?)"')
_PANE = re.compile(rb"<(?:\w+:)?pane\b([^>]*)>")
_ATTR = re.compile(rb'(\w+)="([^"]*)"')
_CELL = re.compile(rb"<(?:\w+:)?c\b([^>]*?)(?:/>|>(.*?)</(?:\w+:)?c>)", re.S)
_REF = re.compile(rb'\br="([A-Z]+)(\d+)"')
_FORMULA = re.compile(rb"<(?:\w+:)?f\b")
_VALUE = re.compile(rb"<(?:\w+:)?v>[^<]")  # a cached value: "<v></v>" is none


def _attrs(tag: bytes) -> dict[str, str]:
    """Attributes of one XML tag, unprefixed and unescaped (no XML parser: no entity expansion)."""
    return {
        key.decode().split(":")[-1]: html.unescape(value.decode("utf-8", "replace"))
        for key, value in _QATTR.findall(tag)
    }


def sheet_paths(archive: zipfile.ZipFile) -> dict[str, str]:
    """Sheet name -> path of its XML part, from the workbook and its relationships."""
    try:
        workbook = archive.read("xl/workbook.xml")
        rels = archive.read("xl/_rels/workbook.xml.rels")
    except KeyError:
        return {}
    targets = {}
    for tag in _RELATIONSHIP.findall(rels):
        attrs = _attrs(tag)
        targets[attrs.get("Id")] = attrs.get("Target", "")
    paths = {}
    for tag in _SHEET.findall(workbook):
        attrs = _attrs(tag)
        target = targets.get(attrs.get("id"), "")
        if target:
            path = target.lstrip("/") if target.startswith("/") else posixpath.join("xl", target)
            paths[attrs.get("name", "")] = posixpath.normpath(path)
    return paths


def scan_sheet_xml(raw: bytes, max_rows: int) -> SheetXml:
    """Merged cells, the frozen pane and formulas without a cached value, from the sheet's XML."""
    found = SheetXml()
    for match in _MERGE.finditer(raw):
        left, top, right, bottom = range_boundaries(match.group(1).decode())
        found.merges.append((top - 1, left - 1, bottom - 1, right - 1))  # type: ignore[operator]
    pane = _PANE.search(raw)
    if pane:
        attrs = {k.decode(): v.decode() for k, v in _ATTR.findall(pane.group(1))}
        if attrs.get("state", "split").startswith("frozen"):
            if "topLeftCell" in attrs:
                letters, row = coordinate_from_string(attrs["topLeftCell"])
                found.freeze = (row - 1, column_index(letters) - 1)
            else:
                found.freeze = (
                    int(float(attrs.get("ySplit", 0))),
                    int(float(attrs.get("xSplit", 0))),
                )
    for match in _CELL.finditer(raw):
        inner = match.group(2)
        ref = _REF.search(match.group(1))
        if ref is None:
            continue
        row = int(ref.group(2))
        if row > max_rows:
            break
        if inner and _FORMULA.search(inner) and not _VALUE.search(inner):
            found.formulas.add((row - 1, column_index(ref.group(1).decode()) - 1))
    return found


def read_sheet_xml(path: Path, limits: Limits = LIMITS) -> dict[str, SheetXml]:
    found: dict[str, SheetXml] = {}
    try:
        with zipfile.ZipFile(path) as archive:
            for name, part in sheet_paths(archive).items():
                try:
                    info = archive.getinfo(part)
                except KeyError:
                    continue
                if info.file_size > limits.max_xml_bytes:  # § 9: cap on decompressed XML
                    found[name] = SheetXml(skipped=True)
                    continue
                found[name] = scan_sheet_xml(archive.read(part), limits.max_rows)
    except zipfile.BadZipFile:
        return {}
    return found


def _filled(cell: object) -> bool:
    fill = getattr(cell, "fill", None)
    pattern = getattr(fill, "patternType", None)
    if pattern in (None, "none"):
        return False
    color = getattr(getattr(fill, "fgColor", None), "rgb", None)
    return not (isinstance(color, str) and color.upper() in ("FFFFFFFF", "00FFFFFF"))


def load_grids(
    path: Path, sheets: list[str] | None = None, limits: Limits = LIMITS
) -> tuple[list[Grid], list[str]]:
    """The bounded grids of a workbook's sheets (all, up to `max_sheets`, or those named)."""
    notes: list[str] = []
    xml = read_sheet_xml(path, limits)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # openpyxl warns about extensions it drops
        workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    grids: list[Grid] = []
    cells = 0
    try:
        names = sheets if sheets is not None else workbook.sheetnames
        if len(names) > limits.max_sheets:
            notes.append(
                f"Only the first {limits.max_sheets} of {len(names)} sheets were analysed."
            )
            names = names[: limits.max_sheets]
        for name in names:
            if name not in workbook.sheetnames:
                continue
            if cells > limits.max_cells:
                notes.append(f"Sheet {name!r} not analysed: the workbook is too large.")
                grids.append(Grid(name=name, values=[], truncated=True))
                continue
            grid = _read_grid(workbook[name], name, xml.get(name), limits)
            cells += grid.n_rows * grid.n_cols
            grids.append(grid)
    finally:
        workbook.close()
    return grids, notes


def _read_grid(sheet: object, name: str, xml: SheetXml | None, limits: Limits) -> Grid:
    grid = Grid(name=name, values=[])
    iter_rows = getattr(sheet, "iter_rows", None)
    if iter_rows is None:  # a chart sheet
        return grid
    reset = getattr(sheet, "reset_dimensions", None)
    if reset is not None:
        reset()  # declared dimensions can be wrong either way (G1)
    for r, row in enumerate(iter_rows(max_row=limits.max_rows + 1, max_col=limits.max_cols)):
        if r == limits.max_rows:
            grid.truncated = any(not is_blank(getattr(c, "value", None)) for c in row)
            break
        values: list[object] = []
        for c, cell in enumerate(row):
            value = getattr(cell, "value", None)
            if is_blank(value):
                values.append(None)
                continue
            values.append(value)
            font = getattr(cell, "font", None)
            if font is not None and font.b:
                grid.bold.add((r, c))
            if _filled(cell):
                grid.filled.add((r, c))
            border = getattr(cell, "border", None)
            if border is not None and border.bottom is not None and border.bottom.style:
                grid.bordered.add((r, c))
            alignment = getattr(cell, "alignment", None)
            if alignment is not None and alignment.indent:
                grid.indent[(r, c)] = int(alignment.indent)
        grid.values.append(values)
    if xml is not None:
        grid.merges = [m for m in xml.merges if m[0] < limits.max_rows and m[1] < limits.max_cols]
        grid.freeze = xml.freeze
        grid.uncached = {
            (r, c)
            for r, c in xml.formulas
            if r < len(grid.values) and c < limits.max_cols and grid.value(r, c) is None
        }
        if xml.skipped:
            grid.warnings.append("The sheet is too large to read its merged cells and panes.")
    grid.trim()
    return grid


def stream_body_end(path: Path, sheet: str, start_row: int, max_col: int) -> int:
    """B1: past the analysed window, the body runs to the first empty row (1-based, inclusive)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        last = start_row - 1
        for offset, row in enumerate(
            workbook[sheet].iter_rows(min_row=start_row, max_col=max_col, values_only=True)
        ):
            if all(is_blank(value) for value in row):
                break
            last = start_row + offset
        return last
    finally:
        workbook.close()
