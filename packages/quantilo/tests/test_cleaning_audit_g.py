"""Audit of 2026-11-12, wave 2 agent G (formats): N21 (F-02, identifiers with a check digit) and
N22 (B-23, a key that repeats with differences).

Each test builds a small synthetic file with the pattern the audit saw on a real one (the French
probe's SIREN and IBAN, Open Food Facts' EAN codes, IMDB's repeated links, online-retail's invoice
lines); no corpus file is copied."""

import csv
import random
from pathlib import Path
from typing import Any

import duckdb
import pytest

import quantilo as dx
from quantilo.cleaning import compile_steps, parse_steps
from quantilo.cleaning import model as m
from quantilo.cleaning.replay import renamed_step, step_columns
from quantilo.cleaning.rules import formats as f
from quantilo.cleaning.suggestion import CleanSuggestion


def write(tmp_path: Path, rows: list[list[Any]], name: str = "data") -> dx.Dataset:
    path = tmp_path / f"{name}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)
    return dx.open(path, cache_dir=tmp_path / f"cache-{name}")


def basic(ds: dx.Dataset) -> dict[str, CleanSuggestion]:
    return {s.id: s for s in ds.clean().suggestions}


def advanced(ds: dx.Dataset) -> dict[str, CleanSuggestion]:
    return {s.id: s for s in ds.clean().advanced()}


def apply(ds: dx.Dataset, steps: list[Any], sql: str) -> list[tuple[Any, ...]]:
    connection = duckdb.connect()
    try:
        compiled = compile_steps(connection, str(ds.path), steps, strict=True)
        return connection.execute(sql.replace("{t}", f"({compiled.sql()})")).fetchall()
    finally:
        connection.close()


# --- Check digits, one by one --------------------------------------------------------------------


def luhn_digit(base: str) -> str:
    for digit in "0123456789":
        if f.luhn(base + digit):
            return digit
    raise AssertionError


def gtin_digit(base: str) -> str:
    return next(d for d in "0123456789" if f.gtin(base + d))


def vat_of(siren: str) -> str:
    return f"FR{(12 + 3 * (int(siren) % 97)) % 97:02d}{siren}"


def iban_fr(bban: str) -> str:
    check = 98 - int(bban + "152700") % 97  # F = 15, R = 27, then 00
    return f"FR{check:02d}{bban}"


def test_check_digits_of_known_numbers() -> None:
    assert f.siren("732829320") and not f.siren("732829321")
    assert f.siren(f.LA_POSTE)  # La Poste's SIREN breaks Luhn's rule
    assert f.siret("35600000000048")  # La Poste's head office passes Luhn
    assert f.siret("35600000000001") and not f.luhn("35600000000001")  # an establishment: sum 15
    assert f.vat_fr(vat_of("404833048")) and not f.vat_fr("FR00404833048")
    for good in ("FR1420041010050500013M02606", "GB82WEST12345698765432", "DE89370400440532013000"):
        assert f.iban(good), good
    assert not f.iban("FR1420041010050500013M02607")  # one digit off
    assert not f.iban("DE8937040044053201300")  # a German IBAN has 22 characters
    assert f.gtin("4006381333931") and f.gtin("036000291452") and f.gtin("96385074")
    assert not f.gtin("4006381333932")
    assert f.isbn("0306406152") and f.isbn("080442957X") and f.isbn("9780306406157")
    assert not f.isbn("0306406153") and not f.isbn("4006381333931")  # an EAN that is no book
    assert (
        f.upc_e("00002220") and not f.gtin("00002220") and f.barcode("00002220")
    )  # Open Food Facts
    assert f.compact(" fr76 3000-6000.0112 ") == "FR76300060000112"


# --- F-02 on columns ----------------------------------------------------------------------------


def sirens(count: int, seed: int = 3) -> list[str]:
    rng = random.Random(seed)
    found = []
    for _ in range(count):
        base = "".join(rng.choice("0123456789") for _ in range(8))
        found.append(base + luhn_digit(base))
    return found


def wrong(number: str) -> str:
    return number[:-1] + str((int(number[-1]) + 1) % 10)  # a check digit off: Luhn always fails


def _companies(bad: list[int], rows: int = 300, column: str = "siren") -> list[list[Any]]:
    table: list[list[Any]] = [["id_client", column, "nom"]]
    for i, number in enumerate(sirens(rows)):
        table.append([i + 1, wrong(number) if i in bad else number, f"Client {i}"])
    return table


def test_n21_sirens_failing_their_key_are_flagged_never_corrected(tmp_path: Path) -> None:
    bad = [11, 86, 150, 151, 290]
    ds = write(tmp_path, _companies(bad))
    item = advanced(ds)["F-02:siren:check"]
    assert item.mode == "S" and item.risk == "review" and item.layer == "advanced"
    assert item.cells == 5 and item.impact == "columns"
    assert "rows 12, 87, 151, 152, 291" in item.reason
    assert "nothing is corrected" in item.reason and item.reason.endswith("Check your data.")
    assert item.step is not None and item.step.op == "conditional_column"
    rows = apply(
        ds,
        [item.step],
        "SELECT id_client FROM {t} WHERE siren_check = 'check digit fails' ORDER BY 1",
    )
    assert [r[0] for r in rows] == [i + 1 for i in bad]
    untouched = apply(ds, [item.step], "SELECT count(*) FROM {t} WHERE siren_check IS NULL")
    assert untouched == [(295,)]


def test_n21_a_clean_column_gets_nothing_and_la_poste_is_no_error(tmp_path: Path) -> None:
    table = _companies([])
    table[5][1] = f.LA_POSTE
    found = advanced(write(tmp_path, table))
    assert not [s for s in found.values() if s.rule == "F-02"]


def test_n21_mostly_failing_numbers_are_only_an_information(tmp_path: Path) -> None:
    table = _companies(list(range(0, 300, 2)) + list(range(1, 60, 2)))  # 180 of 300 wrong
    item = advanced(write(tmp_path, table))["F-02:siren:check"]
    assert item.mode == "I" and item.step is None
    assert "more than half" in item.reason and "nothing is flagged" in item.reason


def test_n21_siret_found_by_its_values_alone(tmp_path: Path) -> None:
    rng = random.Random(5)
    table: list[list[Any]] = [["etablissement", "ville"]]
    for i in range(200):
        base = "".join(rng.choice("0123456789") for _ in range(13))
        number = base + luhn_digit(base)
        table.append([wrong(number) if i in (7, 70) else number, "Lyon"])
    item = advanced(write(tmp_path, table))["F-02:etablissement:check"]
    assert item.mode == "S" and item.cells == 2
    assert "pass the SIRET check" in item.reason and "once in ten" in item.reason


def test_n21_sirens_stored_as_numbers_keep_their_lost_zeros(tmp_path: Path) -> None:
    numbers = ["0" + n[1:-1] for n in sirens(200, seed=9)]
    numbers = [n + luhn_digit(n) for n in numbers]  # all start with 0: stored as integers
    table: list[list[Any]] = [["siren", "ca"]] + [[n, 1000 + i] for i, n in enumerate(numbers)]
    table[40][0] = wrong(numbers[39])
    ds = write(tmp_path, table)
    item = advanced(ds)["F-02:siren:check"]
    assert item.cells == 1 and item.step is not None
    rows = apply(ds, [item.step], "SELECT ca FROM {t} WHERE siren_check IS NOT NULL")
    assert rows == [(1039,)]


def test_n21_ibans_get_their_electronic_form_and_bad_keys_are_flagged(tmp_path: Path) -> None:
    rng = random.Random(1)
    table: list[list[Any]] = [["client", "iban"]]
    for i in range(120):
        number = iban_fr("".join(rng.choice("0123456789") for _ in range(23)))
        if i == 30:
            number = number[:-1] + str((int(number[-1]) + 1) % 10)
        if i % 3 == 0:
            number = " ".join(number[j : j + 4] for j in range(0, len(number), 4))
        table.append([f"c{i}", number.lower() if i == 4 else number])
    ds = write(tmp_path, table)
    found = advanced(ds)
    form, check = found["F-02:iban:format"], found["F-02:iban:check"]
    assert (
        form.mode == "S" and form.impact == "format" and form.cells == 41
    )  # 40 in groups, 1 small
    assert form.step is not None and form.step.op == "map_values"
    assert form.order <= check.order  # the electronic form first, the flag finds both writings
    assert check.cells == 1 and check.step is not None
    rows = apply(
        ds,
        [form.step, check.step],
        "SELECT client, iban, iban_check FROM {t} WHERE iban LIKE '% %' OR iban_check IS NOT NULL",
    )
    assert len(rows) == 1 and rows[0][0] == "c30" and " " not in rows[0][1]


def test_n21_ean_codes_by_their_values_and_the_failing_ones_listed(tmp_path: Path) -> None:
    rng = random.Random(4)
    table: list[list[Any]] = [["code", "product_name"]]
    for i in range(400):
        base = "".join(rng.choice("0123456789") for _ in range(12 if i % 10 else 7))
        number = base + gtin_digit(base)
        table.append([wrong(number) if i in (3, 33, 333) else number, f"product {i}"])
    item = advanced(write(tmp_path, table))["F-02:code:check"]
    assert item.cells == 3 and "EAN / UPC barcodes" in item.reason


def test_n21_french_vat_numbers_and_isbns(tmp_path: Path) -> None:
    table: list[list[Any]] = [["num_tva", "isbn"]]
    for i, number in enumerate(sirens(100, seed=8)):
        vat = vat_of(number)
        base = f"978{i:09d}"
        table.append(
            ["FR00" + number if i == 9 else vat, base + gtin_digit(base) if i else "9780306406158"]
        )
    found = advanced(write(tmp_path, table))
    assert found["F-02:num_tva:check"].cells == 1
    assert found["F-02:isbn:check"].cells == 1


def test_n21_no_false_alarm_on_postal_codes_amounts_or_row_numbers(tmp_path: Path) -> None:
    rng = random.Random(2)
    table: list[list[Any]] = [["Postal Code", "TVA", "order_number", "ref"]]
    for i in range(300):
        table.append(
            [
                f"{rng.randint(1000, 99999):05d}",
                f"{rng.uniform(1, 500):.2f}",
                100_000_000 + i,  # 9 digits, consecutive: a SIREN passes Luhn once in ten
                "".join(rng.choice("0123456789") for _ in range(13)),
            ]
        )
    found = advanced(write(tmp_path, table))
    assert not [s for s in found.values() if s.rule == "F-02"]


def test_n21_counts_mostly_zero_are_no_identifiers(tmp_path: Path) -> None:
    """Ames' porches, adult's capital gains: whole numbers mostly 0 ('000000000' passes Luhn)."""
    rng = random.Random(6)
    table: list[list[Any]] = [["Screen Porch", "capital_gain", "Misc Val"]]
    for i in range(500):
        table.append(
            [0 if i % 9 else rng.randint(40, 500), 0 if i % 11 else rng.randint(100, 99_999), 0]
        )
    found = advanced(write(tmp_path, table))
    assert not [s for s in found.values() if s.rule == "F-02"]


# --- N22: a key that repeats with differences -----------------------------------------------------


def _movies(updated: bool = False, reshuffled: bool = False) -> list[list[Any]]:
    """IMDB's shape: 600 films, 10 entered twice with their votes updated (5 with their gross lost),
    5 copied exactly. `reshuffled`: the second rows are other films drawn under the same link."""
    header = [
        "movie_imdb_link",
        "movie_title",
        "director",
        "title_year",
        "num_voted_users",
        "gross",
    ]
    table: list[list[Any]] = [header + (["updated_at"] if updated else [])]
    for i in range(600):  # unique on 600 of the 610 distinct rows (98.4 %, IMDB: 98.4 %)
        link = f"http://www.imdb.com/title/tt{i:07d}/"
        row: list[Any] = [
            link,
            f"Film {i}",
            f"Director {i % 90}",
            1990 + i % 30,
            1000 + i,
            5_000 + i,
        ]
        table.append(row + (["2016-05-01"] if updated else []))
    for i in range(10):
        link = f"http://www.imdb.com/title/tt{i:07d}/"
        if reshuffled:
            row = [link, f"Other {i}", f"Director {i + 7}", 1980 + i, 50 + i, 70 + i]
        else:
            gross = "" if i < 5 else 5_000 + i
            row = [link, f"Film {i}", f"Director {i % 90}", 1990 + i % 30, 1003 + i, gross]
        table.append(row + (["2016-06-01"] if updated else []))
    for i in range(50, 55):
        table.append(list(table[1 + i]))
    return table


def test_n22_a_key_repeated_with_differences_is_suggested_with_its_columns(tmp_path: Path) -> None:
    ds = write(tmp_path, _movies())
    found = basic(ds)
    assert found["B-14:rows"].rows == 5
    item = found["B-23:movie_imdb_link"]
    assert item.mode == "S" and item.risk == "review" and item.impact == "rows"
    assert item.rows == 10 and item.layer == "basic"
    assert "10 movie_imdb_link values repeat on 20 rows" in item.reason
    assert "num_voted_users in 10, gross in 5" in item.reason
    assert "most complete" in item.reason and "without it, this step removes 15 rows" in item.reason
    assert item.order > found["B-14:rows"].order
    assert item.step is not None and item.step.op == "drop_duplicates"
    assert item.step.keys == ["movie_imdb_link"] and item.step.keep == "most_complete"  # type: ignore[union-attr]
    assert item.examples and "num_voted_users" in (item.examples[0].before or "")
    steps = [found["B-14:rows"].step, item.step]
    rows = apply(ds, steps, "SELECT count(*), count(DISTINCT movie_imdb_link) FROM {t}")
    assert rows == [(600, 600)]
    kept = apply(ds, steps, "SELECT gross FROM {t} WHERE movie_title = 'Film 1'")
    assert kept == [(5_001,)]  # the most complete: its gross is given


def test_n22_the_latest_row_is_kept_when_a_date_says_when_it_changed(tmp_path: Path) -> None:
    ds = write(tmp_path, _movies(updated=True))
    item = basic(ds)["B-23:movie_imdb_link"]
    assert item.step is not None
    assert item.step.keep == "latest" and item.step.order_by == "updated_at"  # type: ignore[union-attr]
    assert "the latest by updated_at" in item.reason
    rows = apply(ds, [item.step], "SELECT num_voted_users FROM {t} WHERE movie_title = 'Film 7'")
    assert rows == [(1010,)]


def test_n22_the_rows_can_be_only_marked(tmp_path: Path) -> None:
    ds = write(tmp_path, _movies())
    item = basic(ds)["B-23:movie_imdb_link:mark"]
    assert item.mode == "S" and item.impact == "columns" and item.cells == 20
    assert item.step is not None and item.step.op == "conditional_column"
    rows = apply(
        ds, [item.step], "SELECT count(*) FROM {t} WHERE movie_imdb_link_repeated IS NOT NULL"
    )
    assert rows == [(20,)]  # the exact copies (films 50 to 54) are B-14's, not marked


def test_n22_rows_sharing_a_key_and_little_else_are_two_records(tmp_path: Path) -> None:
    found = basic(write(tmp_path, _movies(reshuffled=True)))  # dirty_traps: SKUs drawn twice
    assert not [s for s in found.values() if s.rule == "B-23"]


def test_n22_a_house_sold_twice_is_two_sales_not_a_duplicate(tmp_path: Path) -> None:
    """kc_house: `id` repeats on 177 rows whose sale date and price differ."""
    table: list[list[Any]] = [["id", "date", "price", "bedrooms", "zipcode"]]
    for i in range(800):
        table.append([7_129_300_520 + i, f"2014-{1 + i % 12:02d}-13", 200_000 + i, 3, 98178])
    for i in range(10):
        table.append([7_129_300_520 + i, f"2015-{1 + i % 12:02d}-20", 260_000 + i, 3, 98178])
    found = basic(write(tmp_path, table))
    assert not [s for s in found.values() if s.rule == "B-23"]


def test_n22_lines_of_an_invoice_or_a_unique_key_get_nothing(tmp_path: Path) -> None:
    table: list[list[Any]] = [["Invoice", "StockCode", "Quantity", "unique_key"]]
    for i in range(600):
        table.append([f"{489434 + i // 3}", f"SKU{i % 40}", 1 + i % 7, 10_000 + i])
    found = basic(write(tmp_path, table))
    assert not [s for s in found.values() if s.rule == "B-23"]


def test_n22_keep_latest_needs_its_date_and_replays_by_name() -> None:
    with pytest.raises(m.CleanError, match="order_by"):
        parse_steps([{"id": "s1", "op": "drop_duplicates", "keys": ["k"], "keep": "latest"}])
    step = parse_steps(
        [{"id": "s1", "op": "drop_duplicates", "keys": ["k"], "keep": "latest", "order_by": "when"}]
    ).steps[0]
    assert set(step_columns(step)) == {"when", "k"}
    assert renamed_step(step, {"when": "updated", "k": "key"}).order_by == "updated"
