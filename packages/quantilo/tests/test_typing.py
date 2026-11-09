from pathlib import Path

import duckdb
import pytest

from quantilo.ingestion.importer import import_to_parquet
from quantilo.ingestion.options import ImportOptions
from quantilo.ingestion.sources import list_source_files
from quantilo.typing.conversion import parser_for
from quantilo.typing.engine import detect_columns
from quantilo.typing.knowledge_base import load, name_tokens
from quantilo.typing.parsers import typed_sql
from quantilo.typing.summary import summarize
from quantilo.typing.types import ColumnDetection, Parser
from tests.factories import write_csv

SAMPLES = Path(__file__).resolve().parents[3] / "samples"

# Reference answers: (type, role). Detection must get all of them right.
TYPING_TRAPS = {
    "customer_id": ("identifier", "identifier"),
    "age": ("numeric_continuous", "age"),
    "nb_children": ("numeric_discrete", "quantity"),
    "satisfaction": ("numeric_discrete", "rating"),
    "price_eur": ("numeric_continuous", "amount"),
    "discount": ("numeric_discrete", "percentage"),
    "signup_date": ("datetime", None),
    "last_login": ("datetime", None),
    "is_active": ("boolean", None),
    "zip_code": ("categorical_nominal", "postal_code"),
    "country": ("categorical_nominal", "country"),
    "city": ("categorical_nominal", "city"),
    "segment": ("categorical_ordinal", None),
    "income": ("numeric_continuous", "amount"),
    "birth_year": ("datetime", "year"),
    "latitude": ("numeric_continuous", "latitude"),
    "longitude": ("numeric_continuous", "longitude"),
    "email": ("identifier", "email"),
    "comment": ("text", "free_text"),
    "source_system": ("ignored", None),
    "legacy_code": ("ignored", None),
}

SALES_TIMESERIES = {
    "date": ("datetime", None),
    "store": ("categorical_nominal", "entity"),
    "units": ("numeric_continuous", "quantity"),
    "revenue": ("numeric_continuous", "amount"),
    "promo": ("boolean", None),
}


def stored(source: Path, tmp_path: Path) -> str:
    target = tmp_path / f"{source.stem}.parquet"
    import_to_parquet(list_source_files(source), ImportOptions(), target)
    return f"read_parquet('{target.as_posix()}')"


def detect(source: Path, tmp_path: Path) -> dict[str, ColumnDetection]:
    connection = duckdb.connect()
    return {d.name: d for d in detect_columns(connection, stored(source, tmp_path))}


@pytest.fixture(scope="module")
def traps(tmp_path_factory: pytest.TempPathFactory) -> dict[str, ColumnDetection]:
    return detect(SAMPLES / "typing_traps.csv", tmp_path_factory.mktemp("traps"))


class TestReferenceDatasets:
    def test_typing_traps_all_columns(self, traps: dict[str, ColumnDetection]) -> None:
        found = {name: (d.type, d.role) for name, d in traps.items()}

        assert found == TYPING_TRAPS

    def test_sales_timeseries_all_columns(self, tmp_path: Path) -> None:
        detections = detect(SAMPLES / "sales_timeseries.csv", tmp_path)

        assert {name: (d.type, d.role) for name, d in detections.items()} == SALES_TIMESERIES

    def test_every_detection_is_explained(self, traps: dict[str, ColumnDetection]) -> None:
        for detection in traps.values():
            assert detection.reasons, detection.name
            assert detection.confidence in ("high", "medium", "low")

    def test_french_numbers_get_a_comma_parser(self, traps: dict[str, ColumnDetection]) -> None:
        assert traps["price_eur"].parser == Parser(kind="number", decimal=",")
        assert traps["income"].parser == Parser(kind="number", decimal=",")

    def test_ordinal_order_is_found(self, traps: dict[str, ColumnDetection]) -> None:
        assert traps["segment"].order == ["low", "medium", "high"]

    def test_boolean_vocabulary(self, traps: dict[str, ColumnDetection]) -> None:
        parser = traps["is_active"].parser
        assert (parser.true_values, parser.false_values) == (["oui"], ["non"])


class TestTextDates:
    def test_unambiguous_day_first(self, tmp_path: Path) -> None:
        rows = "\n".join(f"{day:02d}/03/2024" for day in range(1, 29))
        detection = detect(write_csv(tmp_path / "d.csv", f"when\n{rows}\n"), tmp_path)["when"]

        # DuckDB's CSV sniffer may already store it as DATE; either way it is a date.
        assert detection.type == "datetime"

    def test_compact_iso_timestamps(self, tmp_path: Path) -> None:
        # ISO 8601 basic format, as in the King County house sales (040)
        rows = "\n".join(f'"201410{day:02d}T000000",{day}' for day in range(1, 29))
        path = write_csv(tmp_path / "d.csv", f"date,n\n{rows}\n")
        detection = detect(path, tmp_path)["date"]

        assert detection.type == "datetime"
        assert detection.parser.format == "%Y%m%dT%H%M%S"

    def test_ambiguous_day_month_lowers_confidence(self, tmp_path: Path) -> None:
        rows = "\n".join(
            f"{day:02d}/{month:02d}/2024 x" for day in range(1, 13) for month in (1, 2)
        )
        path = write_csv(tmp_path / "d.csv", f"label\n{rows}\n")
        connection = duckdb.connect()
        source = stored(path, tmp_path)
        connection.execute(
            f"CREATE TABLE t AS SELECT replace(label, ' x', '') AS label FROM {source}"
        )
        detection = detect_columns(connection, "t")[0]

        assert detection.type == "datetime"
        assert detection.confidence == "medium"
        assert any("ambiguous" in reason for reason in detection.reasons)


class TestRolesAndValues:
    """A categorical role read from the name must not turn decimal numbers into categories."""

    def test_pandas_index_column_is_ignored(self, tmp_path: Path) -> None:
        # marketing_AB.csv: written by df.to_csv() with its index, header cell empty (040)
        rows = "\n".join(f"{i},{i * 3 % 7}" for i in range(300))
        detections = detect(write_csv(tmp_path / "m.csv", f",total ads\n{rows}\n"), tmp_path)

        index = next(d for name, d in detections.items() if name != "total ads")
        assert index.type == "ignored"
        assert detections["total ads"].type != "ignored"

    def test_codes_with_a_letter_prefix_stay_text(self, tmp_path: Path) -> None:
        # Online Retail II: 98 % "489434", 2 % "C489449" (cancelled); as numbers those empty (040)
        invoices = [f"C{489000 + i}" if i % 50 == 0 else str(489000 + i) for i in range(500)]
        rows = "\n".join(f"{invoice},{i}" for i, invoice in enumerate(invoices))
        detections = detect(write_csv(tmp_path / "r.csv", f"Invoice,n\n{rows}\n"), tmp_path)

        assert detections["Invoice"].type in ("identifier", "categorical_nominal")
        assert detections["Invoice"].parser.kind == "text"

    def test_quoted_numeric_ids_stay_identifiers(self, tmp_path: Path) -> None:
        # King County houses: "0007200179" (040)
        rows = "\n".join(f'"{7200179 + i * 13:010d}",{i}' for i in range(300))
        detections = detect(write_csv(tmp_path / "h.csv", f"id,n\n{rows}\n"), tmp_path)

        assert detections["id"].type == "identifier"
        assert detections["id"].parser.kind == "text"

    def test_numbered_stores_are_categories(self, tmp_path: Path) -> None:
        # Walmart: 45 stores, weekly sales (040)
        rows = "\n".join(f"{i % 45 + 1},{1000 + i * 7.5}" for i in range(450))
        detections = detect(
            write_csv(tmp_path / "w.csv", f"Store,Weekly_Sales\n{rows}\n"), tmp_path
        )

        assert detections["Store"].type == "categorical_nominal"
        assert detections["Weekly_Sales"].type == "numeric_continuous"

    def test_decimal_numbers_with_a_country_name_stay_numeric(self, tmp_path: Path) -> None:
        rows = "\n".join(f"{i / 37 - 1.3:.6f},{i % 5}" for i in range(200))
        path = write_csv(tmp_path / "f.csv", f"rsi_60d_country_ranknorm,country\n{rows}\n")

        detections = detect(path, tmp_path)

        assert detections["rsi_60d_country_ranknorm"].type == "numeric_continuous"
        assert detections["rsi_60d_country_ranknorm"].role is None
        # Whole-number country codes keep the role, as before.
        assert (detections["country"].type, detections["country"].role) == (
            "categorical_nominal",
            "country",
        )


class TestIdentifiers:
    def test_unique_ids_are_identifiers_and_repeated_ids_are_categories(
        self, tmp_path: Path
    ) -> None:
        rows = "\n".join(f"{index},{index % 3}" for index in range(200))
        path = write_csv(tmp_path / "ids.csv", f"order_id,store_id\n{rows}\n")

        detections = detect(path, tmp_path)

        assert detections["order_id"].type == "identifier"
        assert detections["store_id"].type == "categorical_nominal"


class TestKnowledgeBase:
    @pytest.mark.parametrize(
        ("name", "tokens"),
        [
            ("customerAge", ("customer", "age")),
            ("Customer_AGE", ("customer", "age")),
            ("âge client", ("age", "client")),
            ("zip-code", ("zip", "code")),
        ],
    )
    def test_name_tokens(self, name: str, tokens: tuple[str, ...]) -> None:
        assert name_tokens(name) == tokens

    @pytest.mark.parametrize(
        ("name", "role"),
        [
            ("customer_age", "age"),
            ("zip_code", "postal_code"),
            ("code postal", "postal_code"),
            ("long_description", "free_text"),
            ("longitude", "longitude"),
            ("paid", None),
            ("order_id", "identifier"),
            ("PrixTTC", "amount"),
        ],
    )
    def test_role_matching(self, name: str, role: str | None) -> None:
        match = load().match_role(name)

        assert (match.id if match else None) == role


class TestParsersAndSummary:
    def test_number_parser_reads_french_formats(self) -> None:
        connection = duckdb.connect()
        connection.execute(
            "CREATE TABLE t AS SELECT * FROM "
            "(VALUES ('1 234,56'), ('12,5 %'), ('N/A'), ('abc')) v(x)"
        )
        expression = typed_sql("x", Parser(kind="number", decimal=","))

        values = [row[0] for row in connection.execute(f"SELECT {expression} FROM t").fetchall()]

        assert values == [1234.56, 12.5, None, None]

    def test_summary_counts_invalid_values(self) -> None:
        connection = duckdb.connect()
        connection.execute(
            "CREATE TABLE t AS SELECT * FROM (VALUES ('1,5'), ('2,5'), ('oops'), (NULL)) v(x)"
        )

        summary = summarize(
            connection, "t", "x", "numeric_continuous", Parser(kind="number", decimal=",")
        )

        assert (summary["rows"], summary["null_count"], summary["invalid_count"]) == (4, 2, 1)
        assert summary["histogram"][0]["count"] + summary["histogram"][-1]["count"] == 2

    def test_switching_type_reuses_detected_candidates(
        self, traps: dict[str, ColumnDetection]
    ) -> None:
        zip_code = traps["zip_code"]

        parser = parser_for("numeric_discrete", zip_code.candidates, zip_code.type, zip_code.parser)

        assert parser.kind == "number"
