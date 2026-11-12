"""082: the type of a file told by its first bytes (R-DET-01 to R-DET-04)."""

import json
import struct
import zlib
from pathlib import Path

import pytest

from quantilo.ingestion.errors import IngestionError
from quantilo.ingestion.importer import import_to_parquet
from quantilo.ingestion.options import ImportOptions, SourceOptions
from quantilo.ingestion.preview import preview_source
from quantilo.ingestion.sniff import HEAD_BYTES, sniff_bytes, sniff_path
from quantilo.ingestion.sources import list_source_files
from tests.factories import write_csv, write_excel, write_parquet


def png(width: int = 3, height: int = 2) -> bytes:
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    chunk = b"IHDR" + ihdr
    return (
        b"\x89PNG\r\n\x1a\n"
        + struct.pack(">I", len(ihdr))
        + chunk
        + struct.pack(">I", zlib.crc32(chunk))
    )


class TestSignatures:
    @pytest.mark.parametrize(
        ("head", "name", "kind", "fmt"),
        [
            (b"a,b\n1,2\n", "x.csv", "table", "csv"),
            (b"PAR1\x15\x04", "x.parquet", "table", "parquet"),
            (b"\x93NUMPY\x01\x00", "x.npy", "table", "npy"),
            (b'{"a": 1}', "x.json", "document", "json"),
            (b'[{"a": 1}, {"a": 2}]', "x.json", "document", "json"),
            (b'{"a": 1}\n{"a": 2}\n{"a": 3}\n', "x.json", "document", "jsonl"),
            (b'<?xml version="1.0"?><a><b/></a>', "x.xml", "document", "xml"),
            (b"<catalog><book/></catalog>", "x.xml", "document", "xml"),
            (b"<b>x</b>,y\n1,a\n", "x.csv", "table", "csv"),
            (b"%PDF-1.7\n", "x.pdf", "unsupported", "pdf"),
            (b"\x1f\x8b\x08\x00", "x.gz", "unsupported", "gzip"),
            (bytes.fromhex("D0CF11E0A1B11AE1") + b"\0" * 8, "x.xls", "unsupported", "xls"),
            (b"SQLite format 3\x00", "x.db", "unsupported", "sqlite"),
            (b"\x00\x01\x02\x03binary", "x.bin", "unsupported", "binary"),
            (b"<!DOCTYPE html><html><body>", "x.html", "unsupported", "html"),
        ],
    )
    def test_kind_and_format(self, head: bytes, name: str, kind: str, fmt: str) -> None:
        found = sniff_bytes(head, name)
        assert (found.kind, found.format) == (kind, fmt)
        assert found.rule.startswith(("R-DET", "R-IMG")) and found.reason

    def test_image(self) -> None:
        found = sniff_bytes(png(), "photo.png")
        assert (found.kind, found.format, found.label) == ("image", "png", "PNG image")
        assert not found.mismatch

    def test_svg_is_refused_as_an_image(self) -> None:
        """R-DET-04: an SVG is XML that can carry scripts."""
        found = sniff_bytes(b'<svg xmlns="http://www.w3.org/2000/svg"></svg>', "logo.svg")
        assert (found.kind, found.rule) == ("unsupported", "R-DET-04")

    def test_csv_starting_like_a_bmp_or_a_list_stays_a_table(self) -> None:
        assert sniff_bytes(b"BMW,Audi\n1,2\n", "cars.csv").kind == "table"
        assert sniff_bytes(b"[a],b\n1,2\n", "x.csv").format == "csv"
        assert sniff_bytes(b"ID3,ID4\n1,2\n", "x.csv").format == "csv"

    def test_utf16_text_is_text(self) -> None:
        head = "a;b\n1;2\n".encode("utf-16")
        assert sniff_bytes(head, "x.csv").format == "csv"

    def test_a_json_cut_by_the_head_is_still_json(self) -> None:
        """R-DET-02: only 64 KiB are read; a big document is cut in the middle of a value."""
        big = json.dumps([{"id": i, "text": "x" * 50} for i in range(5000)]).encode()
        assert len(big) > HEAD_BYTES
        assert sniff_bytes(big[:HEAD_BYTES], "big.json").format == "json"

    def test_json_lines_with_a_cut_last_line(self) -> None:
        lines = b"".join(json.dumps({"i": i}).encode() + b"\n" for i in range(4000))
        assert sniff_bytes(lines[:HEAD_BYTES], "events.txt").format == "jsonl"

    def test_zip_without_workbook_parts(self, tmp_path: Path) -> None:
        import zipfile

        path = tmp_path / "doc.docx"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("word/document.xml", "<x/>")
        found = sniff_path(path)
        assert (found.kind, found.format) == ("unsupported", "docx")
        assert "Word" in found.reason


class TestContentWins:
    """R-DET-01: the content decides; the difference is said at import."""

    def test_parquet_named_csv(self, tmp_path: Path) -> None:
        path = write_parquet(tmp_path / "export.csv", rows=5)
        files = list_source_files(path)
        assert files[0].format == "parquet"
        assert files[0].note and "R-DET-01" in files[0].note and "Parquet" in files[0].note
        preview = preview_source(path, SourceOptions())
        assert preview.detected is not None and preview.detected.mismatch
        assert preview.row_estimate == 5
        assert any("R-DET-01" in note for note in preview.warnings)

    def test_workbook_named_dat(self, tmp_path: Path) -> None:
        source = write_excel(tmp_path / "a.xlsx", {"S": [["x", "y"], [1, 2], [3, 4]]})
        renamed = tmp_path / "a.dat"
        renamed.write_bytes(source.read_bytes())
        preview = preview_source(renamed, SourceOptions())
        assert preview.format == "excel"
        assert [c.name for c in preview.columns] == ["x", "y"]
        target = tmp_path / "out.parquet"
        assert import_to_parquet(list_source_files(renamed), ImportOptions(), target).row_count == 2

    def test_json_named_txt(self, tmp_path: Path) -> None:
        path = write_csv(tmp_path / "api.txt", json.dumps([{"a": 1}, {"a": 2}]))
        preview = preview_source(path, SourceOptions())
        assert preview.format == "json" and preview.detected and preview.detected.mismatch
        assert preview.row_estimate == 2

    def test_a_damaged_workbook_is_not_read_as_text(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.xlsx"
        path.write_bytes(b"not a zip file at all")
        with pytest.raises(IngestionError, match="corrupted"):
            preview_source(path, SourceOptions())

    def test_a_text_file_without_a_known_name_is_refused(self, tmp_path: Path) -> None:
        path = write_csv(tmp_path / "notes.md", "# title\n")
        with pytest.raises(IngestionError, match="Unsupported file type"):
            list_source_files(path)


class TestRefusals:
    """R-DET-03: a file recognised but not read is named."""

    def test_pdf_named_csv(self, tmp_path: Path) -> None:
        path = tmp_path / "report.csv"
        path.write_bytes(b"%PDF-1.4\n%binary")
        with pytest.raises(IngestionError, match="PDF detected"):
            preview_source(path, SourceOptions())

    def test_image_gives_a_preview_without_rows(self, tmp_path: Path) -> None:
        path = tmp_path / "chart.png"
        path.write_bytes(png())
        preview = preview_source(path, SourceOptions())
        assert preview.detected is not None and preview.detected.kind == "image"
        assert preview.columns == [] and preview.rows == []
        with pytest.raises(IngestionError, match="images are not tables"):
            list_source_files(path)

    def test_a_document_in_a_folder_union_is_refused(self, tmp_path: Path) -> None:
        write_csv(tmp_path / "f" / "a.csv", "x\n1\n")
        write_csv(tmp_path / "f" / "b.csv", '{"x": 1}')
        with pytest.raises(IngestionError, match="on its own"):
            list_source_files(tmp_path / "f")
