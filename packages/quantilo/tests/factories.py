"""Small source files written on the fly for ingestion tests."""

from pathlib import Path

import duckdb
import openpyxl


def write_csv(path: Path, text: str, encoding: str = "utf-8") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding=encoding)
    return path


def write_parquet(path: Path, rows: int = 100) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    duckdb.execute(
        f"COPY (SELECT range AS id, range * 1.5 AS value, 'g' || (range % 3) AS grp "
        f"FROM range({rows})) TO '{path.as_posix()}' (FORMAT parquet)"
    )
    return path


def write_excel(path: Path, sheets: dict[str, list[list[object]]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook = openpyxl.Workbook()
    workbook.remove(workbook.active)  # type: ignore[arg-type]
    for name, rows in sheets.items():
        sheet = workbook.create_sheet(name)
        for row in rows:
            sheet.append(row)
    workbook.save(path)
    return path


SEMICOLON_CSV = "nom;prix;date\nCafé;1,5;2024-01-02\nThé;2,25;2024-02-03\nJus;3,10;2024-03-04\n"
