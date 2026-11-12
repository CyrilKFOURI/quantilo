"""090: text logs → one table (docs/features/090-journaux)."""

from pathlib import Path

import duckdb

from quantilo.ingestion.importer import import_to_parquet
from quantilo.ingestion.logs import Drain, choose_format, looks_like_log, read_log
from quantilo.ingestion.options import ImportOptions, SourceOptions
from quantilo.ingestion.preview import preview_source
from quantilo.ingestion.sources import list_source_files

APACHE = "\n".join(
    f"192.168.0.{i % 7} - {'-' if i % 3 else 'bob'} [10/Oct/2026:13:{i % 60:02d}:36 +0200] "
    f'"GET /items/{i}?page=2 HTTP/1.1" {200 if i % 5 else 404} {2326 + i} '
    f'"http://example.com/start" "Mozilla/5.0 (X11)"'
    for i in range(40)
)
COMMON = "\n".join(
    f'10.0.0.1 - - [01/Jan/2026:00:00:{i:02d} +0000] "POST /api/login HTTP/1.0" 302 -'
    for i in range(10)
)
SYSLOG = "\n".join(
    f"Mar  {1 + i % 9} 08:1{i % 10}:00 web01 sshd[{1000 + i}]: Accepted password for user{i} "
    f"from 10.1.1.{i} port {22000 + i} ssh2"
    for i in range(30)
)
RFC5424 = "\n".join(
    f"<34>1 2026-10-11T22:14:{i:02d}.003Z mymachine su - ID47 - 'su root' failed for lonvick "
    "on /dev/pts/8"
    for i in range(12)
)
JAVA = (
    "2026-11-01 10:00:00,123 INFO  Server started on port 8080\n"
    "2026-11-01 10:00:01,456 ERROR Request 42 failed\n"
    "java.lang.IllegalStateException: boom\n"
    "\tat com.example.Foo.bar(Foo.java:10)\n"
    "\tat com.example.Main.main(Main.java:3)\n"
    "2026-11-01 10:00:02,000 WARN  Slow request 43 took 2500ms\n"
    "2026-11-01T10:00:03Z [DEBUG] cache hit key=abc\n"
    "2026-11-01 10:00:04,999 INFO  Server started on port 9090\n"
)


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text + "\n", encoding="utf-8")
    return path


class TestFormats:
    def test_apache_combined_is_read_into_columns(self, tmp_path: Path) -> None:
        table, reading = read_log(_write(tmp_path, "access.log", APACHE))
        assert reading.format == "apache_combined"
        assert reading.rule == "R-LOG-01"
        assert table.num_rows == 40
        row = table.slice(0, 1).to_pylist()[0]
        assert row["method"] == "GET"
        assert row["path"] == "/items/0?page=2"
        assert row["status"] == 404
        assert row["bytes"] == 2326
        assert row["user_agent"] == "Mozilla/5.0 (X11)"
        assert str(row["time"]) == "2026-10-10 11:00:36"  # R-LOG-10: in UTC
        assert row["line_no"] == 1

    def test_apache_common_and_empty_bytes(self, tmp_path: Path) -> None:
        table, reading = read_log(_write(tmp_path, "c.log", COMMON))
        assert reading.format == "apache_common"
        assert table.column("bytes").to_pylist()[0] is None
        assert set(table.column_names) >= {"method", "status", "template"}

    def test_bsd_syslog_takes_the_files_year(self, tmp_path: Path) -> None:
        table, reading = read_log(_write(tmp_path, "auth.log", SYSLOG))
        assert reading.format == "syslog_3164"
        assert reading.year is not None
        row = table.slice(0, 1).to_pylist()[0]
        assert row["program"] == "sshd" and row["process"] == "1000"
        assert row["time"].year == reading.year
        assert any("R-LOG-03" in note for note in reading.notes)

    def test_rfc5424_severity_and_facility(self, tmp_path: Path) -> None:
        table, reading = read_log(_write(tmp_path, "s.log", RFC5424))
        assert reading.format == "syslog_5424"
        row = table.slice(0, 1).to_pylist()[0]
        assert row["severity"] == "critical" and row["facility"] == 4
        assert row["app"] == "su"

    def test_dated_lines_with_a_stack_trace(self, tmp_path: Path) -> None:
        table, reading = read_log(_write(tmp_path, "app.log", JAVA))
        assert reading.format == "iso_lines"
        levels = table.column("level").to_pylist()
        assert levels == ["INFO", "ERROR", None, "WARN", "DEBUG", "INFO"]
        # R-LOG-07: the indented "at …" lines join the entry above; the exception line does not
        # start with a space and does not match: one entry of its own, counted
        assert reading.continued == 2
        assert reading.unmatched == 1
        assert table.num_rows == 6  # 5 dated entries + the exception line
        assert any("R-LOG-07" in note for note in reading.notes)


class TestDrain:
    def test_variable_tokens_become_wildcards(self) -> None:
        drain = Drain()
        a = drain.add("Server started on port 8080")
        b = drain.add("Server started on port 9090")
        c = drain.add("Connection closed by 10.0.0.1")
        assert a == b != c
        assert drain.text(a) == "Server started on port <*>"
        assert drain.text(c) == "Connection closed by <*>"

    def test_template_columns_on_the_table(self, tmp_path: Path) -> None:
        table, reading = read_log(_write(tmp_path, "auth.log", SYSLOG))
        templates = set(table.column("template").to_pylist())
        assert reading.templates == len(templates) == 1
        assert "Accepted password for <*>" in next(iter(templates))


class TestDetection:
    def test_not_a_log(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "notes.txt", "a,b\n1,2\n3,4\n5,6")
        assert not looks_like_log(path)
        assert choose_format(["hello", "world"]) is None

    def test_txt_log_is_detected_by_its_lines(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "server.txt", JAVA)
        assert looks_like_log(path)
        assert list_source_files(path)[0].format == "log"

    def test_unknown_log_is_one_row_per_line(self, tmp_path: Path) -> None:
        table, reading = read_log(_write(tmp_path, "sweep.log", "== a\n   exit 0\n== b"))
        assert reading.format == "lines" and reading.rule == "R-LOG-09"
        assert table.num_rows == 3
        assert table.column_names == ["line_no", "message", "template_id", "template"]


class TestImport:
    def test_preview_says_how_the_log_was_read(self, tmp_path: Path) -> None:
        preview = preview_source(_write(tmp_path, "access.log", APACHE), SourceOptions())
        assert preview.format == "log"
        assert any("R-LOG-01" in w and "Check that your data" in w for w in preview.warnings)
        assert len(preview.rows) <= 40
        assert (preview.row_estimate or 0) >= 30

    def test_import_writes_parquet(self, tmp_path: Path) -> None:
        source = _write(tmp_path, "access.log", APACHE)
        target = tmp_path / "out.parquet"
        result = import_to_parquet(list_source_files(source), ImportOptions(), target)
        assert result.row_count == 40
        found = (
            duckdb.connect()
            .execute(f"SELECT count(*) FROM read_parquet('{target.as_posix()}') WHERE status = 404")
            .fetchone()
        )
        assert found == (8,)
