from typing import Literal

from pydantic import BaseModel, Field

Format = Literal["csv", "parquet", "excel", "npy"]
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


class ImportOptions(SourceOptions):
    columns: list[str] | None = None
    sample_pct: float = Field(default=100, gt=0, le=100)
