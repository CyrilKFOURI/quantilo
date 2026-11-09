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


class ImportOptions(SourceOptions):
    columns: list[str] | None = None
    sample_pct: float = Field(default=100, gt=0, le=100)
