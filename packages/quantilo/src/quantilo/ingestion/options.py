from typing import Literal

from pydantic import BaseModel, Field

Format = Literal["csv", "parquet", "excel", "npy", "log", "json", "jsonl", "xml"]
Encoding = Literal["utf-8", "latin-1", "windows-1252", "utf-16"]


class CsvOptions(BaseModel):
    delimiter: str = Field(default=",", min_length=1, max_length=1)
    encoding: Encoding = "utf-8"
    header: bool = True
    decimal: Literal[".", ","] = "."


class SourceOptions(BaseModel):
    """How to read a source. `None` fields mean "detect" or "everything"."""

    csv: CsvOptions | None = None
    sheets: list[str] | None = None
    files: list[str] | None = None
    folder_mode: Literal["union", "intersection"] = "union"
    # Excel: sheet -> 1-based row holding the column names, the rows above skipped (080, J1).
    # None keeps the first non-empty row; the import preview fills it when the detection is sure.
    header_row: dict[str, int] | None = None
    # One table of the source (080b, 085). Excel: a detected table's id ("Sheet!A4:L14", read with
    # its plan), or a range typed by the user (rule U1); a document: the linked table's name.
    # None keeps the current path (a whole sheet; a document's main table).
    table: str | None = None
    reshape: bool | None = None  # Excel table: apply the proposed restructurings (K1-K4)
    records_path: str | None = None  # a document: where the records are (None: found, R-SST)
    one_table: bool | None = None  # a document: explode its only list into one table (R-SST-15b)


class ImportOptions(SourceOptions):
    columns: list[str] | None = None
    sample_pct: float = Field(default=100, gt=0, le=100)
