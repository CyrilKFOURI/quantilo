"""080b and 085, engine side: every table a source gives, each one previewed and imported alone."""

from pathlib import Path

import duckdb
import pytest

from quantilo.ingestion.errors import IngestionError
from quantilo.ingestion.importer import import_to_parquet
from quantilo.ingestion.layout import detect_range, find_table
from quantilo.ingestion.options import ImportOptions, SourceOptions
from quantilo.ingestion.preview import preview_source
from quantilo.ingestion.sources import list_source_files
from tests import layout_fixtures as fx

ORDERS = Path(__file__).parent / "fixtures" / "structured" / "orders.json"


def names(target: Path) -> list[str]:
    sql = f"DESCRIBE SELECT * FROM '{target.as_posix()}'"
    return [row[0] for row in duckdb.execute(sql).fetchall()]


def count(target: Path) -> int:
    row = duckdb.execute(f"SELECT count(*) FROM '{target.as_posix()}'").fetchone()
    return int(row[0]) if row else 0


class TestSheetTables:
    def test_every_table_of_every_sheet_is_listed(self, tmp_path: Path) -> None:
        path = fx.side_by_side(tmp_path / "s.xlsx")
        tables = preview_source(path, SourceOptions()).tables
        assert tables is not None
        assert [t.id for t in tables] == ["Two!A1:C9", "Two!E1:F4"]
        first = tables[0]
        assert first.area is not None and first.area.header_rows == [1]
        assert first.rows == 8 and first.level == "high" and first.selected
        assert first.summary.startswith("Table A1:C9")
        assert first.reasons and all(r.rule and r.message for r in first.reasons)

    def test_one_table_is_previewed_and_imported_alone(self, tmp_path: Path) -> None:
        path = fx.side_by_side(tmp_path / "s.xlsx")
        preview = preview_source(path, SourceOptions(table="Two!E1:F4"))
        assert [c.name for c in preview.columns] == ["Region", "Target"]
        assert preview.row_estimate == 3 and preview.table == "Two!E1:F4"
        target = tmp_path / "t.parquet"
        import_to_parquet(list_source_files(path), ImportOptions(table="Two!E1:F4"), target)
        assert names(target) == ["Region", "Target"] and count(target) == 3

    def test_titles_and_notes_stay_out(self, tmp_path: Path) -> None:
        path = fx.owner_like(tmp_path / "o.xlsx")
        tables = preview_source(path, SourceOptions()).tables or []
        players = next(t for t in tables if t.id.startswith("Players!A4"))
        assert players.name.endswith("Market attractiveness — existing players")
        assert players.selected
        target = tmp_path / "p.parquet"
        import_to_parquet(list_source_files(path), ImportOptions(table=players.id), target)
        assert names(target)[:2] == ["Player", "Type"]

    def test_proposals_are_applied_only_when_asked(self, tmp_path: Path) -> None:
        path = fx.timesheet(tmp_path / "t.xlsx")
        table = (preview_source(path, SourceOptions()).tables or [])[0]
        assert table.proposals, "K3: a long format is proposed for dates in columns"
        wide, long = tmp_path / "w.parquet", tmp_path / "l.parquet"
        files = list_source_files(path)
        import_to_parquet(files, ImportOptions(table=table.id), wide)
        import_to_parquet(files, ImportOptions(table=table.id, reshape=True), long)
        assert "Period" in names(long) and "Period" not in names(wide)
        assert count(long) > count(wide)

    def test_a_range_marked_by_hand(self, tmp_path: Path) -> None:
        """U1: the rules run on the range the user typed; the id keeps it for the import."""
        path = fx.side_by_side(tmp_path / "s.xlsx")
        table = detect_range(path, "Two", "A1:B5")
        assert table.id == "Two!A1:B5" and table.reasons[0].rule == "U1"
        assert [c.name for c in table.columns] == ["Product", "Units"]
        assert find_table(path, "Two!A1:B5").bottom == 5
        target = tmp_path / "r.parquet"
        import_to_parquet(list_source_files(path), ImportOptions(table="Two!A1:B5"), target)
        assert names(target) == ["Product", "Units"] and count(target) == 4

    def test_a_bad_range_is_a_clear_error(self, tmp_path: Path) -> None:
        path = fx.side_by_side(tmp_path / "s.xlsx")
        with pytest.raises(IngestionError, match="range such as"):
            preview_source(path, SourceOptions(table="Two!nonsense"))
        with pytest.raises(IngestionError, match="empty"):
            preview_source(path, SourceOptions(table="Two!K40:L50"))

    def test_the_whole_sheet_path_is_unchanged(self, tmp_path: Path) -> None:
        """W3: no table chosen = the current import (first non-empty row, or the header row)."""
        path = fx.plain(tmp_path / "p.xlsx")
        preview = preview_source(path, SourceOptions())
        assert preview.table is None and preview.header_row is None
        assert preview.tables and len(preview.tables) == 1


class TestDocumentTables:
    def test_linked_tables_and_their_keys(self) -> None:
        preview = preview_source(ORDERS, SourceOptions())
        assert preview.format == "json" and preview.table == "orders"
        tables = {t.id: t for t in preview.tables or []}
        assert set(tables) >= {"orders", "orders.items"}
        assert tables["orders.items"].parent == "orders" and tables["orders.items"].rows == 8
        assert tables["orders"].reasons[0].rule.startswith("R-")
        link = next(link for link in preview.links if link.child == "orders.items")
        assert (link.parent_key, link.child_key) == ("_id", "_parent_id")

    def test_a_child_table_is_imported_alone(self, tmp_path: Path) -> None:
        target = tmp_path / "items.parquet"
        options = ImportOptions(table="orders.items")
        import_to_parquet(list_source_files(ORDERS), options, target)
        assert count(target) == 8 and "_parent_id" in names(target)

    def test_unknown_table(self) -> None:
        with pytest.raises(IngestionError, match="No table"):
            preview_source(ORDERS, SourceOptions(table="nope"))

    def test_one_table(self) -> None:
        preview = preview_source(ORDERS, SourceOptions(one_table=True))
        assert preview.tables is not None and preview.row_estimate == preview.tables[0].rows


class TestOriginal:
    def test_sheet_cells_and_document_lines(self, tmp_path: Path) -> None:
        from quantilo.ingestion.original import sheet_cells, text_head

        path = fx.owner_like(tmp_path / "o.xlsx")
        original = sheet_cells(path, "Players")
        assert original.kind == "sheet" and original.cells[3][:2] == ["Player", "Type"]
        assert original.cells[0][0] and original.cells[0][0].startswith("Market")
        text = text_head(ORDERS)
        assert text.kind == "text" and text.text and text.text.lstrip().startswith("[")


def test_an_empty_first_sheet_is_skipped(tmp_path: Path) -> None:
    """R-ING-07: a blank cover sheet does not hide the workbook's first table."""
    from tests.factories import write_excel

    path = write_excel(tmp_path / "c.xlsx", {"Cover": [], "Data": [["a", "b"], [1, 2], [3, 4]]})
    preview = preview_source(path, SourceOptions())
    assert [c.name for c in preview.columns] == ["a", "b"]
    assert any("R-ING-07" in note for note in preview.warnings)


def test_user_null_tokens_change_the_reading_not_the_tables(tmp_path: Path) -> None:
    """T2u (b2, Advanced > More): the user's placeholder 'nil' becomes empty and the column a
    number; a token kept as text stays; where the tables are found does not change."""
    from openpyxl import Workbook

    from quantilo.ingestion.layout.cells import kind, null_token, user_null_tokens

    wb = Workbook()
    ws = wb.active
    assert ws is not None
    ws.title = "T"
    ws.append(["Item", "Amount", "Mark"])
    for i in range(12):
        ws.append([f"I{i}", "nil" if i in (3, 7) else 10 + i, "n/a" if i == 5 else "ok"])
    path = tmp_path / "tokens.xlsx"
    wb.save(path)
    plain = preview_source(path, SourceOptions())
    assert plain.tables is not None
    table = plain.tables[0].id
    before = preview_source(path, SourceOptions(table=table))
    amount = next(c for c in before.columns if c.name == "Amount")
    assert amount.dtype in ("VARCHAR", "varchar")  # 'nil' keeps the column text
    asked = SourceOptions(table=table, null_tokens=["NIL "], text_tokens=["n/a"])
    after = preview_source(path, asked)
    assert [t.id for t in after.tables or []] == [t.id for t in plain.tables]
    amount = next(c for c in after.columns if c.name == "Amount")
    assert amount.dtype.upper() in ("DOUBLE", "BIGINT", "INTEGER")
    rows = {row[0]: row for row in after.rows}
    assert rows["I3"][1] is None and rows["I5"][2] == "n/a"
    assert any("'nil' read as empty" in note and "T2u" in note for note in after.warnings)
    with user_null_tokens(["nil"], ["n/a"]):
        assert null_token("Nil") == "strong" and null_token("n/a") is None and kind("nil") == "z"
    assert null_token("nil") is None and null_token("n/a") == "strong" and kind("nil") == "s"
