"""092: real-data import defects (docs/features/092-corrections-multidomaine-import-types)."""

import zipfile
from pathlib import Path

import duckdb
import pytest

import quantilo as dx
from quantilo.ingestion.csv_sniff import sniff_csv
from quantilo.ingestion.errors import IngestionError
from quantilo.ingestion.importer import import_to_parquet
from quantilo.ingestion.normalize import stored_notes
from quantilo.ingestion.options import ImportOptions, SourceOptions
from quantilo.ingestion.preview import preview_source
from quantilo.ingestion.sources import list_source_files
from quantilo.typing.engine import detect_columns
from quantilo.typing.summary import summarize
from tests.factories import write_csv

ZONED = (
    "ts,value\n2026-09-01T00:00:00+02:00,1\n2026-09-01T01:00:00+02:00,2\n2026-09-01T02:00:00Z,3\n"
)
GERMAN = "\n".join(f"A1{i % 4} {6 + i % 30} A3{i % 5} {1169 + i * 13} A9{i % 3}" for i in range(60))


class TestTimeZones:
    """R-ING-01 (ING-1): no pytz, times in UTC, the rule said."""

    def test_zoned_timestamps_are_stored_in_utc(self, tmp_path: Path) -> None:
        source = write_csv(tmp_path / "z.csv", ZONED)
        target = tmp_path / "z.parquet"
        import_to_parquet(list_source_files(source), ImportOptions(), target)

        connection = duckdb.connect()
        parquet = f"read_parquet('{target.as_posix()}')"
        dtype = connection.execute(f"DESCRIBE SELECT ts FROM {parquet}").fetchall()[0][1]
        assert dtype == "TIMESTAMP"
        first = connection.execute(f"SELECT CAST(min(ts) AS VARCHAR) FROM {parquet}").fetchone()
        assert first == ("2026-08-31 22:00:00",)
        notes = stored_notes(connection, str(target))
        assert "R-ING-01" in notes["ts"][0]

        detection = next(d for d in detect_columns(connection, parquet) if d.name == "ts")
        assert detection.type == "datetime"
        assert any("R-ING-01" in reason for reason in detection.reasons)
        summary = summarize(connection, parquet, "ts", detection.type, detection.parser)
        assert summary["examples"]

    def test_open_works_end_to_end(self, tmp_path: Path) -> None:
        dataset = dx.open(write_csv(tmp_path / "z.csv", ZONED), cache_dir=tmp_path / "cache")

        assert dataset.types[0].type == "datetime"
        assert len(dataset.to_pandas()) == 3

    def test_preview_announces_the_rule(self, tmp_path: Path) -> None:
        preview = preview_source(write_csv(tmp_path / "z.csv", ZONED), SourceOptions())

        assert any("R-ING-01" in warning and "ts" in warning for warning in preview.warnings)
        assert preview.rows[0][0] == "2026-08-31 22:00:00"


class TestSpaceSeparated:
    """R-ING-02 / R-ING-03 (ING-2)."""

    def test_space_delimiter_without_header(self, tmp_path: Path) -> None:
        options = sniff_csv(write_csv(tmp_path / "g.csv", GERMAN + "\n"))

        assert (options.delimiter, options.header) == (" ", False)

    def test_preview_has_the_columns_and_the_notes(self, tmp_path: Path) -> None:
        preview = preview_source(write_csv(tmp_path / "g.csv", GERMAN + "\n"), SourceOptions())

        assert len(preview.columns) == 5
        assert preview.row_estimate == 60
        assert any("R-ING-02" in warning for warning in preview.warnings)
        assert any("R-ING-03" in warning for warning in preview.warnings)

    def test_text_with_spaces_is_not_split(self, tmp_path: Path) -> None:
        text = "comment\n" + "\n".join(f"a nice day number {i}" for i in range(30)) + "\n"
        options = sniff_csv(write_csv(tmp_path / "c.csv", text))

        assert options.delimiter != " "


class TestExtensions:
    """R-ING-04 / R-ING-05 (ING-3, ING-4)."""

    @pytest.mark.parametrize("suffix", [".txt", ".dat", ".data"])
    def test_delimited_text_extensions(self, tmp_path: Path, suffix: str) -> None:
        preview = preview_source(write_csv(tmp_path / f"g{suffix}", GERMAN + "\n"), SourceOptions())

        assert len(preview.columns) == 5
        assert any("R-ING-04" in warning for warning in preview.warnings)

    def test_a_folder_of_dat_files(self, tmp_path: Path) -> None:
        write_csv(tmp_path / "f" / "a.dat", "x,y\n1,2\n")
        write_csv(tmp_path / "f" / "b.dat", "x,y\n3,4\n")

        assert [f.path.name for f in list_source_files(tmp_path / "f")] == ["a.dat", "b.dat"]

    def test_binary_xls_is_explained(self, tmp_path: Path) -> None:
        path = tmp_path / "old.xls"
        path.write_bytes(bytes.fromhex("D0CF11E0A1B11AE1") + b"\0" * 64)

        with pytest.raises(IngestionError, match=r"Excel 97-2003.*\.xlsx"):
            preview_source(path, SourceOptions())

    def test_renamed_xlsx_is_explained(self, tmp_path: Path) -> None:
        path = tmp_path / "renamed.xls"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("[Content_Types].xml", "<x/>")

        with pytest.raises(IngestionError, match=r"renamed.*\.xlsx"):
            preview_source(path, SourceOptions())

    def test_text_xls_is_explained(self, tmp_path: Path) -> None:
        path = tmp_path / "export.xls"
        path.write_text("a\tb\n1\t2\n", encoding="utf-8")

        with pytest.raises(IngestionError, match=r"text.*\.csv"):
            preview_source(path, SourceOptions())
