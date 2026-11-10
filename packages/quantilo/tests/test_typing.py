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
from quantilo.typing.text import measure
from quantilo.typing.types import ColumnDetection, Parser
from tests.factories import write_csv

SAMPLES = Path(__file__).resolve().parents[3] / "samples"

# Reference answers: (type, role). Detection must get all of them right.
TYPING_TRAPS = {
    "customer_id": ("identifier", "customer_key"),  # 048: a key, still an identifier
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
    # 050: texts without a telling name, read from their values
    "item": ("text", "label"),
    "ship_to": ("text", "address"),
    "survey_answer": ("text", "free_text"),
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


class TestBusinessRoles:
    """048 (P2): keys keep the identifier's types; business measures get their roles."""

    def test_keys_are_typed_like_identifiers(self, tmp_path: Path) -> None:
        rows = "\n".join(
            f'{i // 3},C{i % 40:03d},{1000 + i % 40},"{i:06d}",{i % 7 + 0.5},{i % 5 + 1},'
            f"{(i % 9) * 1.25:.2f},{['EUR', 'USD'][i % 2]},{2000 + i % 11}"
            for i in range(300)
        )
        header = "order_id,Customer ID,user,InvoiceNo,UnitPrice,Quantity,shipping_cost,currency"
        path = write_csv(tmp_path / "o.csv", f"{header},budget\n{rows}\n")
        detections = detect(path, tmp_path)

        found = {name: (d.type, d.role) for name, d in detections.items()}
        assert found == {
            "order_id": ("categorical_nominal", "order_key"),  # repeated: categories, as before
            "Customer ID": ("categorical_nominal", "customer_key"),
            "user": ("categorical_nominal", "customer_key"),
            "InvoiceNo": ("identifier", "order_key"),  # unique: an identifier
            "UnitPrice": ("numeric_continuous", "unit_price"),
            "Quantity": ("numeric_discrete", "quantity"),
            "shipping_cost": ("numeric_continuous", "cost"),
            "currency": ("categorical_nominal", "currency"),
            "budget": ("numeric_discrete", "target"),  # no forced type
        }
        assert detections["InvoiceNo"].parser.kind == "text"  # quoted zeros kept

    def test_a_numeric_unique_key_is_an_identifier(self, tmp_path: Path) -> None:
        rows = "\n".join(f"{100000 + i},{i % 4}" for i in range(300))
        detections = detect(write_csv(tmp_path / "k.csv", f"CustomerID,n\n{rows}\n"), tmp_path)

        assert (detections["CustomerID"].type, detections["CustomerID"].role) == (
            "identifier",
            "customer_key",
        )

    def test_a_key_word_on_decimal_values_is_no_key(self, tmp_path: Path) -> None:
        # an A/B test's "basket" is the basket's value, not an order key
        rows = "\n".join(f'{i * 1.37:.2f},"{i},5"' for i in range(200))
        detections = detect(write_csv(tmp_path / "b.csv", f"basket,order\n{rows}\n"), tmp_path)

        for name in ("basket", "order"):
            assert detections[name].type == "numeric_continuous", name
            assert detections[name].role is None, name

    def test_a_machine_learning_target_keeps_its_type(self, tmp_path: Path) -> None:
        rows = "\n".join(f"{i % 2},{i % 7}" for i in range(200))
        detections = detect(write_csv(tmp_path / "t.csv", f"target,label\n{rows}\n"), tmp_path)

        assert detections["target"].type == "boolean"


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
            ("order_id", "order_key"),
            ("PrixTTC", "amount"),
            # 048 (P2): keys only when their word is the whole name, with an id word or not
            ("Customer ID", "customer_key"),
            ("CustomerID", "customer_key"),
            ("user", "customer_key"),
            ("id_client", "customer_key"),
            ("InvoiceNo", "order_key"),
            ("Invoice", "order_key"),
            ("order_number", "order_key"),
            ("InvoiceDate", None),
            ("account_balance", None),
            ("order_priority", None),
            ("Customer Name", None),
            ("id", "identifier"),
            ("product_id", "identifier"),
            # business measures
            ("shipping_cost", "cost"),
            ("Total Cost", "cost"),
            ("COGS", "cost"),
            ("unit_price", "unit_price"),
            ("UnitPrice", "unit_price"),
            ("unit cost", "unit_price"),
            ("Price", "amount"),
            ("price_eur", "amount"),
            ("Sales", "amount"),
            ("sales_target", "target"),
            ("budget", "target"),
            ("currency", "currency"),
            ("Currency Code", "currency"),
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


def csv_column(tmp_path: Path, name: str, values: list[str]) -> ColumnDetection:
    """Detect one text column written to a CSV (values quoted, a row number beside)."""
    rows = "\n".join(f'"{value}",{index}' for index, value in enumerate(values))
    path = write_csv(tmp_path / "t.csv", f'"{name}",n\n{rows}\n')
    return detect(path, tmp_path)[name]


PRODUCTS = [
    f"{color} {thing} {size}"
    for color in ("WHITE", "RED", "BLUE", "PINK", "GREEN", "IVORY", "BLACK", "GOLD")
    for thing in ("HANGING HEART HOLDER", "CAKE CASES", "LUNCH BAG", "GLASS BALL", "TEA SET",
                  "PHOTO FRAME", "WALL CLOCK", "PARTY BUNTING")
    for size in ("SMALL", "LARGE", "JUMBO")
]  # fmt: skip
EN = [
    f"The {thing} {verb} {end}"
    for thing in ("delivery", "parcel", "support team", "product", "price", "refund")
    for verb in ("was really fast", "was a bit slow", "is not what I expected", "is great")
    for end in ("and I will order again.", "but the box was damaged.", "for my first order!",
                "so I am happy.", "although it took a while.")
]  # fmt: skip
FR = [
    f"{start} {middle} {end}"
    for start in ("La livraison était", "Le colis est arrivé", "Le service client a été",
                  "Le produit est", "Le remboursement a été", "Le prix est")
    for middle in ("très rapide", "un peu lent", "vraiment parfait", "assez décevant")
    for end in ("et je recommande.", "mais la boîte était abîmée.", "pour une première commande !",
                "donc je suis content.", "même si cela a pris du temps.")
]  # fmt: skip


class TestTextMeasures:
    def test_words_punctuation_and_language(self) -> None:
        measures = measure(
            [("Great value, I would buy it again.", 6), ("Price 2.5 EUR", 2)], load()
        )

        assert measures.values == 8
        assert measures.median_words == 7
        assert measures.sentence_share == 0.75  # "2.5" is not the end of a sentence
        assert measures.language == "en"

    @pytest.mark.parametrize(
        ("text", "language"),
        [
            ("Il pleut depuis le matin et nous sommes restés à la maison.", "fr"),
            ("El pedido llegó tarde pero el producto es muy bueno.", "es"),
            ("Die Lieferung war schnell und das Paket ist gut verpackt.", "de"),
            ("Il pacco è arrivato in ritardo ma il prodotto è molto bello.", "it"),
            ("The parcel was late but the product is very nice.", "en"),
        ],
    )
    def test_language_of_a_text(self, text: str, language: str) -> None:
        assert measure([(text, 10)], load()).language == language

    def test_elisions_and_labels(self) -> None:
        # "l'avis" holds the French "l'"; the "T" of "T-LIGHT" is no French "t'"
        assert measure([("J'ai lu l'avis qu'il a donné.", 10)], load()).language == "fr"
        labels = [("WHITE HANGING HEART T-LIGHT HOLDER", 10), ("SET OF 6 MEN'S T-SHIRTS", 10)]
        assert measure(labels, load()).language is None  # a single stop word votes for nothing

    def test_two_languages_are_mixed(self) -> None:
        measures = measure([(text, 1) for text in EN[:60] + FR[:60]], load())

        assert measures.language == "mixed"
        assert {language for language, _ in measures.languages} == {"en", "fr"}

    def test_addresses(self) -> None:
        values = ["12 rue de la Paix, 75002 Paris", "221B Baker Street, London NW1 6XE",
                  "Hauptstraße 5, 10115 Berlin", "WHITE HANGING HEART"]  # fmt: skip
        measures = measure([(value, 1) for value in values], load())

        assert measures.address_share == 0.75
        assert measures.postal_share == 0.75


class TestTextVerdicts:
    def test_product_descriptions_are_short_text(self, tmp_path: Path) -> None:
        # Online Retail II "Description": product names, repeated (040)
        values = [PRODUCTS[(index * 7) % len(PRODUCTS)] for index in range(600)]
        detection = csv_column(tmp_path, "Description", values)

        assert (detection.type, detection.role) == ("text", "label")
        assert "values are short labels" in detection.reasons[0]

    def test_named_product_labels_win_over_low_uniqueness(self, tmp_path: Path) -> None:
        # Superstore "Product Name": 1,850 products over 10,000 rows (18 % unique)
        values = [PRODUCTS[index % len(PRODUCTS)] for index in range(1200)]
        detection = csv_column(tmp_path, "Product Name", values)

        assert (detection.type, detection.role) == ("text", "label")

    def test_unnamed_repeated_labels_stay_categories(self, tmp_path: Path) -> None:
        values = [PRODUCTS[index % len(PRODUCTS)] for index in range(1200)]
        detection = csv_column(tmp_path, "thing", values)

        assert (detection.type, detection.role) == ("categorical_nominal", None)

    def test_reviews_without_a_name_are_free_text(self, tmp_path: Path) -> None:
        detection = csv_column(tmp_path, "answer", EN)

        assert (detection.type, detection.role, detection.confidence) == (
            "text",
            "free_text",
            "high",
        )
        assert any(reason.startswith("language: English") for reason in detection.reasons)

    def test_bilingual_reviews_are_mixed(self, tmp_path: Path) -> None:
        detection = csv_column(tmp_path, "answer", EN + FR)

        assert detection.role == "free_text"
        assert any(reason.startswith("mixed languages") for reason in detection.reasons)

    def test_languages_are_explained(self, traps: dict[str, ColumnDetection]) -> None:
        assert traps["survey_answer"].reasons[-1].startswith("language: French")
        assert any("English" in reason for reason in traps["comment"].reasons)

    def test_few_repeated_sentences_are_categories(self, tmp_path: Path) -> None:
        values = [EN[index % 30] for index in range(300)]
        detection = csv_column(tmp_path, "answer", values)

        assert detection.type == "categorical_nominal"

    def test_long_unique_tokens_are_identifiers(self, tmp_path: Path) -> None:
        values = [f"{index:064x}" for index in range(300)]
        detection = csv_column(tmp_path, "token", values)

        assert detection.type == "identifier"

    def test_addresses_by_name_only(self, tmp_path: Path) -> None:
        values = [f"Unit {index} Industrial Estate North" for index in range(300)]
        detection = csv_column(tmp_path, "Address", values)

        assert (detection.type, detection.role) == ("text", "address")

    def test_repeated_codes_with_dashes_stay_categories(self, tmp_path: Path) -> None:
        # Superstore "Order ID": CA-2017-100111, repeated per order line, named id (040)
        values = [f"CA-{2014 + index // 2 % 4}-{100000 + index // 2}" for index in range(600)]
        detection = csv_column(tmp_path, "Order ID", values)

        assert detection.type == "categorical_nominal"
