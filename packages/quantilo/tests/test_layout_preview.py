"""J1 wiring (080, rules W1–W4): the Excel preview reads the detected header row, the import follows
the options it returns, and everything else keeps the current path."""

import os
from pathlib import Path

import duckdb
import pytest

from quantilo.ingestion.importer import import_to_parquet
from quantilo.ingestion.layout import detect_layout
from quantilo.ingestion.options import ImportOptions, SourceOptions
from quantilo.ingestion.preview import preview_source
from quantilo.ingestion.reader import read_sheet
from quantilo.ingestion.sources import list_source_files
from tests import layout_fixtures as fx
from tests.factories import write_csv

OWNER = Path(
    os.environ.get(
        "QUANTILO_OWNER_FILE",
        r"C:\Users\cyril\.claude\uploads\1d6ae763-3536-4981-9ef1-4c00d795251f"
        r"\60e26fe2-Kiwi_Market_TAM_Exits.xlsx",
    )
)


def test_preview_reads_the_detected_header(tmp_path: Path) -> None:
    path = fx.owner_like(tmp_path / "o.xlsx")
    preview = preview_source(path, SourceOptions())
    assert preview.header_row == {"Players": 4}
    assert [c.name for c in preview.columns][:2] == ["Player", "Type"]
    assert preview.warnings and "row 4" in preview.warnings[0]


def test_import_follows_the_preview(tmp_path: Path) -> None:
    """W2: the header row the preview returns, passed back, is what gets imported."""
    path = fx.owner_like(tmp_path / "o.xlsx")
    preview = preview_source(path, SourceOptions())
    options = ImportOptions(header_row=preview.header_row)
    target = tmp_path / "out.parquet"
    import_to_parquet(list_source_files(path), options, target)
    names = [
        row[0] for row in duckdb.execute(f"DESCRIBE SELECT * FROM '{target.as_posix()}'").fetchall()
    ]
    assert names[:2] == ["Player", "Type"]


def test_legacy_paths_unchanged(tmp_path: Path) -> None:
    """W3: no header_row = first non-empty row; clean files, CSV, an explicit {}: untouched."""
    path = fx.owner_like(tmp_path / "o.xlsx")
    assert read_sheet(path, "Players").column_names[0].startswith("Market attractiveness")
    assert read_sheet(path, "Players", 4).column_names[0] == "Player"
    assert preview_source(path, SourceOptions(header_row={})).columns[0].name.startswith("Market")
    assert preview_source(fx.plain(tmp_path / "p.xlsx"), SourceOptions()).header_row is None
    csv = write_csv(tmp_path / "a.csv", "a,b\n1,2\n")
    assert preview_source(csv, SourceOptions()).header_row is None


def test_detection_error_never_blocks_the_preview(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """W4."""
    import quantilo.ingestion.layout as layout

    def broken(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(layout, "detect_layout", broken)
    preview = preview_source(fx.owner_like(tmp_path / "o.xlsx"), SourceOptions())
    assert preview.header_row is None and preview.columns


@pytest.mark.skipif(not OWNER.exists(), reason="the owner's example file is not on this machine")
def test_owner_file() -> None:
    """S1 on the owner's real file (outside the repository)."""
    layout = detect_layout(OWNER)
    ranges = {
        s.name: [(t.range, t.header_rows) for t in s.tables if t.confidence >= 0.4]
        for s in layout.sheets
    }
    assert ranges["1_Market_Players"] == [("A4:L14", [4]), ("A17:C23", [17])]
    assert ranges["2_AMMC_TAM"][:2] == [("A4:C20", [4]), ("A23:F28", [23])]
    assert ranges["3_Comparable_Exits"][0] == ("A4:K24", [4])
    assert layout.sheet("2_AMMC_TAM").tables[1].warnings  # T4
    assert preview_source(OWNER, SourceOptions()).header_row == {"1_Market_Players": 4}
