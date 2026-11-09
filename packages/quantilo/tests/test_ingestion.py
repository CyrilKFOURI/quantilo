from pathlib import Path

import duckdb
import pytest

from quantilo.ingestion.csv_sniff import sniff_csv
from quantilo.ingestion.errors import IngestionError
from quantilo.ingestion.importer import import_to_parquet
from quantilo.ingestion.options import CsvOptions, ImportOptions, SourceOptions
from quantilo.ingestion.preview import preview_source
from quantilo.ingestion.sources import fingerprint, list_source_files
from tests.factories import SEMICOLON_CSV, write_csv, write_excel, write_parquet


def stored_rows(path: Path) -> list[tuple[object, ...]]:
    return duckdb.execute(f"SELECT * FROM read_parquet('{path.as_posix()}')").fetchall()


class TestSniffing:
    def test_semicolon_and_decimal_comma(self, tmp_path: Path) -> None:
        options = sniff_csv(write_csv(tmp_path / "a.csv", SEMICOLON_CSV))

        assert options == CsvOptions(delimiter=";", encoding="utf-8", header=True, decimal=",")

    def test_latin1_encoding(self, tmp_path: Path) -> None:
        options = sniff_csv(write_csv(tmp_path / "l.csv", SEMICOLON_CSV, encoding="latin-1"))

        assert options.encoding == "latin-1"

    def test_comma_delimited_keeps_dot_decimal(self, tmp_path: Path) -> None:
        options = sniff_csv(write_csv(tmp_path / "c.csv", "a,b\n1.5,x\n2.5,y\n"))

        assert (options.delimiter, options.decimal) == (",", ".")


class TestPreview:
    def test_csv_preview(self, tmp_path: Path) -> None:
        preview = preview_source(write_csv(tmp_path / "a.csv", SEMICOLON_CSV), SourceOptions())

        assert preview.kind == "file"
        assert preview.format == "csv"
        assert [c.name for c in preview.columns] == ["nom", "prix", "date"]
        assert preview.rows[0] == ["Café", "1.5", "2024-01-02"]
        assert preview.row_estimate == 3
        assert preview.suggested_name == "a"

    def test_latin1_accents_are_readable(self, tmp_path: Path) -> None:
        path = write_csv(tmp_path / "l.csv", SEMICOLON_CSV, encoding="latin-1")

        assert preview_source(path, SourceOptions()).rows[0][0] == "Café"

    def test_windows_1252_quotes_are_readable(self, tmp_path: Path) -> None:
        # Excel exports: curly quotes and no-break spaces, bytes DuckDB's latin-1 refuses (040)
        path = tmp_path / "w.csv"
        path.write_bytes("name,price\n“Bush”\xa0bookcase,261.96\n".encode("cp1252"))

        assert sniff_csv(path).encoding == "windows-1252"
        assert preview_source(path, SourceOptions()).rows[0] == ["“Bush”\xa0bookcase", "261.96"]
        target = tmp_path / "w.parquet"
        import_to_parquet(list_source_files(path), ImportOptions(), target)
        assert stored_rows(target) == [("“Bush”\xa0bookcase", 261.96)]

    def test_user_csv_options_override_detection(self, tmp_path: Path) -> None:
        path = write_csv(tmp_path / "a.csv", SEMICOLON_CSV)
        options = SourceOptions(csv=CsvOptions(delimiter=";", decimal="."))

        assert preview_source(path, options).rows[0][1] == "1,5"

    def test_excel_lists_sheets_and_previews_first(self, tmp_path: Path) -> None:
        path = write_excel(
            tmp_path / "book.xlsx",
            {"Sales": [["a", "b"], [1, "x"], [2, "y"]], "Other": [["z"], [9]]},
        )
        preview = preview_source(path, SourceOptions())

        assert preview.sheets == ["Sales", "Other"]
        assert [c.name for c in preview.columns] == ["a", "b"]
        assert preview.row_estimate == 2

    def test_parquet_row_estimate_from_metadata(self, tmp_path: Path) -> None:
        preview = preview_source(write_parquet(tmp_path / "p.parquet", 250), SourceOptions())

        assert preview.row_estimate == 250

    def test_folder_lists_supported_files_and_flags_mismatch(self, tmp_path: Path) -> None:
        folder = tmp_path / "folder"
        write_csv(folder / "a.csv", "x,y\n1,2\n")
        write_csv(folder / "b.csv", "x,z\n3,4\n")
        write_csv(folder / "notes.txt", "ignored")
        preview = preview_source(folder, SourceOptions())

        assert preview.kind == "folder"
        assert [f.name for f in preview.files] == ["a.csv", "b.csv"]
        assert preview.schema_mismatch is True

    def test_empty_file_is_a_clear_error(self, tmp_path: Path) -> None:
        with pytest.raises(IngestionError, match="empty"):
            preview_source(write_csv(tmp_path / "e.csv", ""), SourceOptions())

    def test_corrupted_excel_is_a_clear_error(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.xlsx"
        path.write_bytes(b"not a zip file at all")

        with pytest.raises(IngestionError, match="password-protected or corrupted"):
            preview_source(path, SourceOptions())

    def test_legacy_xls_is_explained(self, tmp_path: Path) -> None:
        path = tmp_path / "old.xls"
        path.write_bytes(b"x")

        with pytest.raises(IngestionError, match=r"\.xlsx"):
            preview_source(path, SourceOptions())


class TestImport:
    def test_imports_csv_to_parquet(self, tmp_path: Path) -> None:
        files = list_source_files(write_csv(tmp_path / "a.csv", SEMICOLON_CSV))
        target = tmp_path / "out" / "data.parquet"
        steps: list[str] = []

        result = import_to_parquet(files, ImportOptions(), target, lambda _f, m: steps.append(m))

        assert (result.row_count, result.col_count) == (3, 3)
        assert result.size_bytes == target.stat().st_size
        assert stored_rows(target)[0][1] == 1.5
        assert steps[-1] == "Finalizing"

    def test_column_selection(self, tmp_path: Path) -> None:
        files = list_source_files(write_csv(tmp_path / "a.csv", SEMICOLON_CSV))
        target = tmp_path / "data.parquet"

        result = import_to_parquet(files, ImportOptions(columns=["prix"]), target)

        assert result.col_count == 1

    def test_unknown_column_is_rejected(self, tmp_path: Path) -> None:
        files = list_source_files(write_csv(tmp_path / "a.csv", SEMICOLON_CSV))

        with pytest.raises(IngestionError, match="Unknown columns: nope"):
            import_to_parquet(files, ImportOptions(columns=["nope"]), tmp_path / "d.parquet")

    def test_sampling_is_reproducible(self, tmp_path: Path) -> None:
        files = list_source_files(write_parquet(tmp_path / "p.parquet", 10_000))
        options = ImportOptions(sample_pct=10)

        first = import_to_parquet(files, options, tmp_path / "one.parquet")
        import_to_parquet(files, options, tmp_path / "two.parquet")

        assert 700 < first.row_count < 1300
        assert stored_rows(tmp_path / "one.parquet") == stored_rows(tmp_path / "two.parquet")

    def test_folder_union_adds_source_file(self, tmp_path: Path) -> None:
        folder = tmp_path / "folder"
        write_csv(folder / "a.csv", "x,y\n1,2\n")
        write_csv(folder / "b.csv", "x,z\n3,4\n")
        target = tmp_path / "data.parquet"

        result = import_to_parquet(list_source_files(folder), ImportOptions(), target)
        columns = [row[0] for row in duckdb.execute(f"DESCRIBE '{target.as_posix()}'").fetchall()]

        assert result.row_count == 2
        assert set(columns) == {"x", "y", "z", "source_file"}

    def test_folder_intersection_keeps_common_columns(self, tmp_path: Path) -> None:
        folder = tmp_path / "folder"
        write_csv(folder / "a.csv", "x,y\n1,2\n")
        write_csv(folder / "b.csv", "x,z\n3,4\n")
        options = ImportOptions(folder_mode="intersection")

        result = import_to_parquet(list_source_files(folder), options, tmp_path / "d.parquet")

        assert result.col_count == 2  # x + source_file

    def test_excel_sheets_can_be_stacked(self, tmp_path: Path) -> None:
        path = write_excel(tmp_path / "book.xlsx", {"Jan": [["a"], [1], [2]], "Feb": [["a"], [3]]})
        options = ImportOptions(sheets=["Jan", "Feb"])

        result = import_to_parquet(list_source_files(path), options, tmp_path / "d.parquet")

        assert (result.row_count, result.col_count) == (3, 2)  # a + source_sheet


class TestFingerprint:
    def test_changes_with_file_content_and_options(self, tmp_path: Path) -> None:
        path = write_csv(tmp_path / "a.csv", "x\n1\n")
        before = fingerprint(list_source_files(path), {"sample_pct": 100})

        assert fingerprint(list_source_files(path), {"sample_pct": 50}) != before
        write_csv(path, "x\n1\n2\n")
        assert fingerprint(list_source_files(path), {"sample_pct": 100}) != before


def test_excel_columns_keep_types_and_names(tmp_path: Path) -> None:
    from datetime import datetime

    from quantilo.ingestion.reader import read_sheet

    path = write_excel(
        tmp_path / "mixed.xlsx",
        {
            "S": [
                ["amount", None, "amount", "when"],
                [1.5, "x", 2, datetime(2024, 1, 2)],
                [None, None, None, None],
                [3, 4, "n/a", datetime(2024, 1, 3)],
            ]
        },
    )

    table = read_sheet(path, "S")

    assert table.column_names == ["amount", "column_2", "amount_3", "when"]
    assert table.num_rows == 2  # the empty row is skipped
    assert table.column("amount").to_pylist() == [1.5, 3.0]
    assert table.column("column_2").to_pylist() == ["x", "4"]  # mixed: text
    assert table.column("amount_3").to_pylist() == ["2", "n/a"]
    assert str(table.schema.field("when").type).startswith("timestamp")


class TestNpy:
    def test_2d_array_becomes_numbered_columns(self, tmp_path: Path) -> None:
        import numpy as np

        path = tmp_path / "features.npy"
        np.save(path, np.arange(12, dtype=np.float32).reshape(4, 3))
        files = list_source_files(path)
        assert preview_source(path, SourceOptions()).row_estimate == 4
        result = import_to_parquet(files, ImportOptions(), tmp_path / "d.parquet")
        assert (result.row_count, result.col_count) == (4, 3)
        assert stored_rows(tmp_path / "d.parquet")[1] == (3.0, 4.0, 5.0)

    def test_structured_array_keeps_field_names(self, tmp_path: Path) -> None:
        import numpy as np

        path = tmp_path / "s.npy"
        np.save(path, np.array([(1, 0.5), (2, 0.25)], dtype=[("era", "i4"), ("target", "f8")]))
        import_to_parquet(list_source_files(path), ImportOptions(), tmp_path / "d.parquet")
        assert stored_rows(tmp_path / "d.parquet") == [(1, 0.5), (2, 0.25)]

    def test_object_arrays_are_refused(self, tmp_path: Path) -> None:
        import numpy as np

        path = tmp_path / "o.npy"
        np.save(path, np.array([{"a": 1}], dtype=object), allow_pickle=True)
        with pytest.raises(IngestionError, match="Python objects"):
            import_to_parquet(list_source_files(path), ImportOptions(), tmp_path / "d.parquet")

    def test_sample_and_columns_are_read_without_loading_everything(self, tmp_path: Path) -> None:
        import numpy as np

        path = tmp_path / "big.npy"
        np.save(path, np.arange(3000, dtype=np.float32).reshape(1000, 3))
        options = ImportOptions(sample_pct=10, columns=["column_2"])
        result = import_to_parquet(list_source_files(path), options, tmp_path / "d.parquet")
        assert (result.row_count, result.col_count) == (100, 1)
        values = [int(str(row[0]).split(".")[0]) for row in stored_rows(tmp_path / "d.parquet")]
        assert all(value % 3 == 1 for value in values) and values == sorted(values)

    def test_preview_reads_only_the_first_rows(self, tmp_path: Path) -> None:
        import numpy as np

        path = tmp_path / "big.npy"
        np.save(path, np.zeros((5000, 2), dtype=np.float32))
        preview = preview_source(path, SourceOptions())
        assert preview.row_estimate == 5000
        assert len(preview.rows) == 20


class TestCancel:
    def test_a_stopped_import_writes_nothing(self, tmp_path: Path) -> None:
        import threading

        from quantilo.ingestion.importer import ImportCancelled

        stop = threading.Event()
        stop.set()
        files = list_source_files(write_csv(tmp_path / "a.csv", SEMICOLON_CSV))
        with pytest.raises(ImportCancelled):
            import_to_parquet(files, ImportOptions(), tmp_path / "d.parquet", stop=stop)
        assert not (tmp_path / "d.parquet").exists()

    def test_a_stopped_npy_read_is_cancelled(self, tmp_path: Path) -> None:
        import threading

        import numpy as np

        from quantilo.ingestion.importer import ImportCancelled

        path = tmp_path / "a.npy"
        np.save(path, np.zeros((10, 2)))
        stop = threading.Event()
        stop.set()
        with pytest.raises(ImportCancelled):
            import_to_parquet(
                list_source_files(path), ImportOptions(), tmp_path / "d.parquet", stop=stop
            )
