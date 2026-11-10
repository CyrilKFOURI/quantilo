"""Table detection on synthetic workbooks, one per layout family (080, scenarios S1–S12)."""

import json
import zipfile
from pathlib import Path

import pytest

import quantilo as dx
from quantilo.ingestion.layout import DetectedTable, detect_layout
from quantilo.ingestion.layout.grid import Limits, load_grids, scan_sheet_xml
from tests import layout_fixtures as fx


def only(path: Path) -> DetectedTable:
    tables = detect_layout(path).tables
    assert len(tables) == 1, [t.range for t in tables]
    return tables[0]


def rules(table: DetectedTable) -> set[str]:
    return {hit.rule for hit in table.reasons}


class TestCashBook:
    """S3: two header levels, a totals row under the header, internal empty columns."""

    def test_one_table_with_flattened_names(self, tmp_path: Path) -> None:
        table = only(fx.cash_book(tmp_path / "c.xlsx"))
        assert table.range == "B3:P21" and table.header_rows == [3, 4]  # C4-parent: rows 1-2 out
        names = [c.name for c in table.columns]
        assert names[:3] == ["Details · Date", "Details · Name", "Details · Ref"]
        assert "Income · Coaching" in names and "Expenses · Phone" in names
        assert {"D2-keep", "C9-top", "K1", "K2", "C3", "C3-pane"} <= rules(table)
        assert table.dropped_rows["total"] == [5]
        derived = {c.name for c in table.columns if c.role == "derived"}
        assert derived == {"Total Money In", "Total Money Out"}
        assert table.confidence >= 0.8

    def test_sparse_group_long_format(self, tmp_path: Path) -> None:
        layout = detect_layout(fx.cash_book(tmp_path / "c.xlsx"))
        table = layout.tables[0]
        wide = layout.read(table)
        assert wide.num_rows == 16  # the totals row is out
        long = layout.read(table, reshape=True).to_pylist()
        assert {"Income type", "Expenses type", "Amount", "row_id"} <= set(long[0])
        assert "Total Money In" not in long[0]  # derived columns leave the long format
        split = [r for r in long if r["row_id"] == 8]
        assert sorted(r["Income type"] for r in split) == [
            "Coaching",
            "Travel",
        ]  # 2 amounts, 2 rows
        assert len(long) == 17


def test_multi_level_header_total_and_dots(tmp_path: Path) -> None:
    """S4, C9-top, C10 by '.' prefix (no ancestors: states do not follow their region)."""
    layout = detect_layout(fx.multi_level(tmp_path / "m.xlsx"))
    table = layout.tables[0]
    assert table.range == "A2:F14" and table.header_rows == [2, 3]
    assert [c.name for c in table.columns][2] == "Population Estimate · 2020"
    assert table.dropped_rows == {"total": [4]}
    periods = next(t for t in table.transforms if t.rule == "K3")
    assert periods.into == ["Period", "Population Estimate"]
    data = layout.read(table)
    assert "Level_1" not in data.column_names and "level" in data.column_names
    assert data.column("Geographic Area").to_pylist()[-1] == "Colorado"
    long = layout.read(table, reshape=True)
    assert long.num_rows == 4 * 10 and set(long.column("Period").to_pylist()) == {
        "2020",
        "2021",
        "2022",
        "2023",
    }
    assert table.notes and table.title == "Annual Estimates of the Resident Population"


def test_indentation_hierarchy_units_column_numbers(tmp_path: Path) -> None:
    """S5, C5 units, C6 column numbers, C5-ref footnote references, C2 notes."""
    layout = detect_layout(fx.indented(tmp_path / "i.xlsx"))
    table = layout.tables[0]
    assert table.header_rows == [2, 3, 4] and table.dropped_rows["column_numbers"] == [4]
    assert table.units == {"Earnings": "dollars", "Workers": "thousands"}
    assert [c.name for c in table.columns] == ["Level", "Earnings", "Workers"]
    data = layout.read(table).to_pylist()
    diploma = next(r for r in data if r["Level"] == "Diploma")
    assert (diploma["level"], diploma["Level_1"], diploma["Level_2"]) == (
        2,
        "All levels",
        "High school",
    )
    assert any("SOURCE" in note for note in table.notes)


def test_side_by_side_tables(tmp_path: Path) -> None:
    """S6, D2: an empty column between tables whose rows differ."""
    tables = detect_layout(fx.side_by_side(tmp_path / "s.xlsx")).tables
    assert [t.range for t in tables] == ["A1:C9", "E1:F4"]
    assert all("D2" in rules(t) for t in tables)
    assert not tables[0].dropped_rows  # an arithmetic series is not a total (mid-series row)


def test_stacked_blocks_one_table_with_section(tmp_path: Path) -> None:
    """S6, D3: same header twice, each under a section title."""
    layout = detect_layout(fx.stacked(tmp_path / "st.xlsx"))
    table = layout.tables[0]
    assert len(layout.tables) == 1 and table.title == "Median income by state"
    data = layout.read(table)
    assert data.num_rows == 8
    assert data.column("Section").to_pylist() == ["Current dollars"] * 4 + ["2025 dollars"] * 4


def test_transposed_record(tmp_path: Path) -> None:
    """S7, K4: one record per column, transposed on demand, never unpivoted."""
    layout = detect_layout(fx.transposed(tmp_path / "t.xlsx"))
    table = layout.tables[0]
    assert table.kind == "key_value" and table.header_rows == [1]
    records = layout.read(table, reshape=True).to_pylist()
    assert [r["City"] for r in records] == ["Paris", "Lyon", "Nice"]


def test_pivot_export(tmp_path: Path) -> None:
    """S8: labels filled down, '(blank)' empty, 'Grand Total' removed (checked by arithmetic)."""
    layout = detect_layout(fx.pivot_export(tmp_path / "p.xlsx"))
    table = layout.tables[0]
    assert table.dropped_rows == {"total": [10]} and table.title == "Sum of Sales"
    labels = layout.read(table).column("Row Labels").to_pylist()
    assert labels == ["East", "East", "East", "West", "West", None]


def test_timesheet_dates_across(tmp_path: Path) -> None:
    """S9: a header of dates is a header, and the long format is proposed."""
    layout = detect_layout(fx.timesheet(tmp_path / "h.xlsx"))
    table = layout.tables[0]
    assert table.header_rows == [3] and table.columns[1].name == "2026-01-05"
    long = layout.read(table, reshape=True)
    assert long.num_rows == 20 and "Period" in long.column_names


def test_owner_like_file(tmp_path: Path) -> None:
    """S1: title and note above a styled header, text body, a second table, formulas (T4)."""
    layout = detect_layout(fx.owner_like(tmp_path / "o.xlsx"))
    players, frame = layout.sheet("Players").tables
    assert (players.range, players.header_rows, frame.range) == ("A4:E9", [4], "A12:C15")
    assert players.title == "Market attractiveness — existing players"
    assert players.confidence >= 0.8 and {"C3", "C3-pane"} <= rules(players)
    assert frame.title == "Regulatory frame"
    founded = layout.read(players).column("Founded").to_pylist()
    assert founded == ["2023", None, "~2015", "2009-10", "2013"]  # 'n/d' read as empty
    inputs, levels = layout.sheet("TAM").tables
    assert inputs.range == "A3:C7" and levels.range == "A10:D13"
    assert levels.uncached == 6 and levels.warnings and "T4" in rules(levels)
    assert levels.confidence < inputs.confidence


def test_templates_and_text_sheets_fall_back(tmp_path: Path) -> None:
    """S11: an empty template and a sheet of paragraphs are not offered as tables."""
    sheet = detect_layout(fx.templates(tmp_path / "t.xlsx")).sheets[0]
    assert sheet.fallback and sheet.tables[0].dropped_rows["template"] == list(range(2, 12))
    assert sheet.tables[0].warnings
    about = detect_layout(fx.text_sheet(tmp_path / "a.xlsx")).sheets[0]
    assert about.fallback and not about.tables


def test_plain_table_unchanged(tmp_path: Path) -> None:
    """S12: a clean table keeps row 1 as its header, nothing dropped, nothing applied."""
    sheet = detect_layout(fx.plain(tmp_path / "p.xlsx")).sheets[0]
    table = sheet.tables[0]
    assert (table.range, table.header_rows, table.dropped_rows) == ("A1:C11", [1], {})
    assert sheet.preview_header() is None
    assert not [t for t in table.transforms if t.default]


def test_types_and_numbered_groups(tmp_path: Path) -> None:
    layout = detect_layout(fx.typed_text(tmp_path / "t.xlsx"))
    data = layout.read(layout.tables[0]).to_pylist()
    assert data[0]["Code"] == "01" and data[0]["Turnout"] == 69.55
    assert data[0]["Voters"] == 1234 and data[2]["Voters"] is None
    assert data[0]["Population"] == 74794 and data[0]["Population flag"] == "e"
    groups = detect_layout(fx.numbered_groups(tmp_path / "n.xlsx"))
    long = groups.read(groups.tables[0], reshape=True).to_pylist()
    assert len(long) == 15 and long[0]["n"] == "1" and long[0]["Voix"] == 300


def test_long_sheet_streams_the_body_end(tmp_path: Path) -> None:
    """B1: past the analysed window the last table runs to the first empty row."""
    path = fx.long_sheet(tmp_path / "l.xlsx", rows=120)
    layout = detect_layout(path, limits=Limits(max_rows=50))
    table = layout.tables[0]
    assert table.truncated and table.bottom == 123 and "B1" in rules(table)
    assert layout.read(table).num_rows == 120
    assert detect_layout(path).tables[0].range == "A3:B123"


def test_xml_scan_merges_pane_formulas(tmp_path: Path) -> None:
    """G3 and T4 read from the sheet XML (read_only mode exposes neither)."""
    path = fx.owner_like(tmp_path / "o.xlsx")
    with zipfile.ZipFile(path) as archive:
        raw = archive.read("xl/worksheets/sheet2.xml")
    found = scan_sheet_xml(raw, 100)
    assert (10, 1) in found.formulas
    grids, _ = load_grids(path)
    assert grids[0].freeze == (4, 1) and grids[0].bold


def test_serializable_and_facade(tmp_path: Path) -> None:
    layout = dx.detect_tables(fx.cash_book(tmp_path / "c.xlsx"))
    text = json.dumps(layout.to_dict())
    assert "B3:P21" in text and "Table B3:P21, header rows 3, 4" in layout.tables[0].summary()
    with pytest.raises(KeyError):
        layout.table("nope")


class TestCsv:
    def test_preamble(self, tmp_path: Path) -> None:
        """V1: rows narrower than the table at the top are a preamble."""
        table = only(fx.write_text(tmp_path / "v.csv", fx.VERBOSE_CSV))
        assert table.range == "A4:D8" and table.header_rows == [4] and "V1" in rules(table)

    def test_metadata_block(self, tmp_path: Path) -> None:
        """V2: 'Unit:' rows are column metadata, the names come from row 1."""
        layout = detect_layout(fx.write_text(tmp_path / "f.csv", fx.METADATA_CSV))
        table = layout.tables[0]
        assert table.header_rows == [1, 2, 3, 4, 5] and table.first_data_row == 6
        assert table.units == {"Yield 1-month": "Percent", "Yield 3-month": "Percent"}
        assert layout.read(table).column("Yield 1-month").to_pylist()[2] is None  # 'ND'

    def test_combined_first_column(self, tmp_path: Path) -> None:
        """V3 + K3 + T3: 'freq,geo\\TIME_PERIOD' split, years across, flags split."""
        layout = detect_layout(fx.write_text(tmp_path / "e.tsv", fx.COMBINED_TSV))
        table = layout.tables[0]
        assert table.header_rows == [1]
        data = layout.read(table)
        assert (
            data.column_names[:2] == ["freq", "geo"] and data.column("geo").to_pylist()[0] == "AD"
        )
        assert "2018 flag" in data.column_names
        long = layout.read(table, reshape=True)
        assert long.num_rows == 14  # 16 cells minus 2 ':' placeholders
