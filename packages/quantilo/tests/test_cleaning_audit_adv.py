"""Audit of 2026-11-12, the remaining rule milestones (feat/clean-adv): N10, N11, N27, N29 (Basic),
N16, N17, N33, N34 (Advanced).

Each test reproduces on a small synthetic file what the audit saw on a real one (kc_house,
melbourne, chicago_beach, nyc_dob, nyc_311, législatives, air quality, dvf, eco2mix, fr_rne,
diabetes, adult); no corpus file is copied."""

import csv
import random
from pathlib import Path
from typing import Any

import duckdb

import quantilo as dx
from quantilo.cleaning import compile_steps
from quantilo.cleaning.suggestion import CleanSuggestion


def write(tmp_path: Path, rows: list[list[Any]], name: str = "data") -> dx.Dataset:
    path = tmp_path / f"{name}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)
    return dx.open(path, cache_dir=tmp_path / f"cache-{name}")


def found(ds: dx.Dataset) -> dict[str, CleanSuggestion]:
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


# --- N10: a 0 that means 'never' (B-22) ----------------------------------------------------------


def _houses(rows: int = 400, zeros: float = 0.9) -> list[list[Any]]:
    rng = random.Random(10)
    table: list[list[Any]] = [
        ["id", "yr_built", "yr_renovated", "bedrooms", "YearsAtCompany", "price"]
    ]
    for i in range(rows):
        built = rng.randint(1900, 2015)
        renovated = 0 if rng.random() < zeros else rng.randint(max(built, 1950), 2015)
        table.append(
            [i + 1, built, renovated, 0 if i in (3, 4) else rng.randint(1, 6), i % 30, 1_000 * i]
        )
    return table


def test_n10_zero_among_years_is_read_as_never(tmp_path: Path) -> None:
    ds = write(tmp_path, _houses())
    by_id = found(ds)
    never = by_id["B-22:yr_renovated"]
    assert never.mode == "S" and never.catalogue == "B-22" and not never.auto
    assert "'never'" in never.reason and never.reason.endswith("Check your data.")
    assert never.step is not None and never.step.op == "null_values"
    assert not any(s.startswith("B-22:yr_built") for s in by_id)  # no 0 among its years
    assert not any(s.startswith("B-22:YearsAtCompany") for s in by_id)  # 0 years of service
    rows = apply(ds, [never.step], "SELECT count(*), min(yr_renovated) FROM {t}")
    assert rows[0][0] == 400 and rows[0][1] >= 1950  # the rows stay, 0 is empty


def test_n10_few_zeros_in_a_year_column_are_not_said_twice(tmp_path: Path) -> None:
    """B-10 would empty the years outside 1000 to 2200: B-22 says 0, B-10 nothing more."""
    ds = write(tmp_path, _houses(zeros=0.02))
    by_id = found(ds)
    assert "B-22:yr_renovated" in by_id
    assert "B-10:yr_renovated" not in by_id


def test_n10_zero_bedrooms_is_shown_not_changed(tmp_path: Path) -> None:
    rare = found(write(tmp_path, _houses()))["B-22:bedrooms:rare"]
    assert rare.mode == "I" and rare.step is None and rare.cells == 2
    assert rare.catalogue == "B-22:rare" and "Not changed" in rare.reason


def test_n10_a_water_temperature_above_40_is_outside_its_range(tmp_path: Path) -> None:
    rng = random.Random(11)
    table: list[list[Any]] = [["beach", "water_temperature", "battery_life"]]
    for i in range(300):
        table.append(["Ohio" if i % 2 else "Calumet", 44.4 if i == 9 else rng.uniform(8, 26), 9.5])
    by_id = found(write(tmp_path, table))
    outside = by_id["B-10:water_temperature"]
    assert outside.mode == "S" and outside.cells == 1
    assert "water temperature (-2 to 40 °C)" in outside.reason
    assert "B-10:battery_life" not in by_id  # volts, not a level in %


# --- N11: impossible years -----------------------------------------------------------------------


def _built(odd: list[int], rows: int = 2_000) -> list[list[Any]]:
    rng = random.Random(12)
    table: list[list[Any]] = [["id", "YearBuilt", "year"]]
    for i in range(rows):
        built = odd[i] if i < len(odd) else rng.randint(1830, 2018)
        table.append([i + 1, built, rng.randint(1750, 2023)])
    return table


def test_n11_a_year_centuries_before_the_others_is_a_typing_error(tmp_path: Path) -> None:
    ds = write(tmp_path, _built([1196]))
    by_id = found(ds)
    year = by_id["B-10:YearBuilt:year"]
    assert year.mode == "S" and year.catalogue == "B-10:year" and year.cells == 1
    assert year.reason.startswith("1196 is 634 years before every other YearBuilt (1830 to 2018)")
    assert "1196 → 1916 or 1996" in year.reason and "never written for you" in year.reason
    assert year.step is not None and year.step.op == "null_values"
    rows = apply(ds, [year.step], "SELECT count(*), min(YearBuilt) FROM {t}")
    assert rows[0] == (2_000, 1830)
    assert not any(k.startswith("B-10:year") for k in by_id)  # owid's years from 1750: real


def test_n11_a_future_year_too_and_old_years_close_to_the_others_stay(tmp_path: Path) -> None:
    by_id = found(write(tmp_path, _built([2105])))
    assert "in the future" in by_id["B-10:YearBuilt:year"].reason
    assert "B-10:YearBuilt:year" not in found(write(tmp_path, _built([1720]), "old"))
    many = found(write(tmp_path, _built([1196] * 5, rows=1_000), "many"))
    assert "B-10:YearBuilt:year" not in many  # 0.5 % of the rows: not isolated


# --- N27: checkboxes, Oui / Non, V / F -----------------------------------------------------------


def _forms(rows: int = 300) -> list[list[Any]]:
    rng = random.Random(27)
    table: list[list[Any]] = [["id", "newsletter", "abonne", "valide", "residential", "self_cert"]]
    for i in range(rows):
        table.append(
            [
                i + 1,
                "X" if rng.random() < 0.4 else "",
                rng.choice(["Oui", "Non", "oui", "X", ""]),
                rng.choice(["V", "F"]),
                "YES" if rng.random() < 0.35 else "",
                rng.choice(["Y", "N"]),
            ]
        )
    return table


def test_n27_one_mark_and_blanks_is_a_checkbox(tmp_path: Path) -> None:
    ds = write(tmp_path, _forms())
    by_id = found(ds)
    box = by_id["B-18:newsletter:checkbox"]
    assert box.mode == "S" and box.catalogue == "B-18:checkbox" and box.cells == 300
    assert "a blank becomes false" in box.reason
    assert not any(k.startswith("B-19:newsletter") for k in by_id)  # not a constant column
    rows = apply(ds, [box.step], "SELECT count(*) FILTER (WHERE newsletter IS NULL) FROM {t}")
    assert rows[0][0] == 0  # every blank is false


def test_n27_a_mark_among_yes_and_no_words_and_v_f(tmp_path: Path) -> None:
    by_id = found(write(tmp_path, _forms()))
    mark = by_id["B-18:abonne:mark"]
    assert mark.mode == "S" and mark.catalogue == "B-18:mark"
    assert "'X' read as yes" in mark.reason and "blanks stay empty" in mark.reason
    assert mark.step is not None and "x" in mark.step.true_values  # type: ignore[attr-defined]
    pair = by_id["B-18:valide"]
    assert pair.mode == "S" and pair.catalogue == "B-18:pair"
    assert by_id["B-18:self_cert"].mode == "S"  # Y / N unchanged (decision 7 of 065)


def test_n27_a_boolean_with_only_true_and_blanks(tmp_path: Path) -> None:
    ds = write(tmp_path, _forms())
    by_id = found(ds)
    box = by_id["B-18:residential:checkbox"]
    assert box.step is not None and box.step.op == "fill_empty"
    assert not any(k.startswith("B-19:residential") for k in by_id)
    rows = apply(ds, [box.step], "SELECT count(*) FILTER (WHERE NOT residential) FROM {t}")
    assert rows[0][0] == box.cells


# --- N29: free text keeps its line breaks --------------------------------------------------------


def _notes(rows: int = 120) -> list[list[Any]]:
    table: list[list[Any]] = [["id", "notes", "location"]]
    for i in range(rows):
        note = f"  Call   back on day {i % 9}.\n\n  Second  paragraph {i % 7}.  " if i % 3 else "ok"
        place = (
            f"\n,  \n({40 + i / 1000:.3f}, -73.850)" if i % 2 else f"({40 + i / 1000:.3f}, -73.9)"
        )
        table.append([i + 1, note, place])
    return table


def test_n29_a_free_text_keeps_its_paragraphs(tmp_path: Path) -> None:
    ds = write(tmp_path, _notes())
    by_id = found(ds)
    notes = by_id["B-01:notes"]
    assert notes.mode == "A" and notes.auto
    assert notes.step is not None and notes.step.keep_newlines  # type: ignore[attr-defined]
    assert "Line breaks are kept" in notes.reason
    assert all("\n\nSecond paragraph" in (e.after or "") for e in notes.examples)
    rows = apply(ds, [notes.step], "SELECT notes FROM {t} WHERE id = 2")
    assert rows[0][0] == "Call back on day 1.\n\nSecond paragraph 1."
    location = by_id["B-01:location"]  # a layout field, not free text: one line
    assert location.step is not None and not location.step.keep_newlines  # type: ignore[attr-defined]
    assert apply(ds, [location.step], "SELECT location FROM {t} WHERE id = 2")[0][0] == (
        ", (40.001, -73.850)"
    )


SPACED = [
    "  Hello   world  \n\n  second  line \t end  ",
    "a \r\n b\u00a0\u00a0c\n",
    "\n\nx",
    "one\u2028two  ",
    "plain",
    " \t ",
    "x \n \n y",
]
KEPT = ["Hello world\n\nsecond line end", "a\r\nb c", "x", "one\u2028two", "plain", "", "x\n\ny"]


def test_n29_python_and_sql_write_the_same_and_stay_linear() -> None:
    import time

    from quantilo.cleaning import text as tx

    connection = duckdb.connect()
    try:
        for value, kept in zip(SPACED, KEPT, strict=True):
            for keep in (False, True):
                sql = tx.normalize_spaces_sql("v", keep)
                written = connection.execute(f"SELECT {sql} FROM (SELECT $1 AS v)", [value])
                python = tx.normalize_spaces(value, keep)
                assert written.fetchone() == (python,)
            assert tx.normalize_spaces(value, keep_newlines=True) == kept
        long = "a" + " " * 100_000 + "b" + "\n" * 50_000 + " " * 49_998
        start = time.perf_counter()
        assert tx.normalize_spaces(long, keep_newlines=True) == "a b"
        assert tx.normalize_spaces(long) == "a b"
        assert time.perf_counter() - start < 0.5
    finally:
        connection.close()


# --- N16: A-11, date words and an honest fix -----------------------------------------------------


def _requests(rows: int = 400) -> list[list[Any]]:
    """nyc_311: closed_date written to the minute, a few seconds 'before' created_date; orders
    shipped a few days before they were ordered (swapped), and two years before (a wrong date)."""
    import datetime as dt

    rng = random.Random(16)
    table: list[list[Any]] = [["id", "created_date", "closed_date", "order_date", "ship_date"]]
    start = dt.datetime(2024, 1, 1)
    for i in range(rows):
        created = start + dt.timedelta(minutes=rng.randint(0, 500_000), seconds=rng.randint(1, 59))
        closed = created + dt.timedelta(hours=rng.randint(1, 72))
        if i < 6:  # closed in the same minute, written without its seconds
            closed = created.replace(second=0)
        ordered = (start + dt.timedelta(days=rng.randint(0, 300))).date()
        shipped = ordered + dt.timedelta(days=rng.randint(1, 9))
        if 10 <= i < 16:
            ordered, shipped = shipped, ordered  # swapped
        if i in (20, 21):
            shipped = ordered - dt.timedelta(days=700)  # a wrong year
        table.append(
            [
                i + 1,
                created.strftime("%Y-%m-%d %H:%M:%S"),
                closed.strftime("%Y-%m-%d %H:%M:%S"),
                ordered.isoformat(),
                shipped.isoformat(),
            ]
        )
    return table


def test_n16_the_same_minute_written_two_ways_is_no_inversion(tmp_path: Path) -> None:
    by_id = advanced(write(tmp_path, _requests()))
    minute = by_id["A-11:created_date+closed_date:minute"]
    assert minute.mode == "I" and minute.step is None and minute.cells == 6
    assert "same minute" in minute.reason and "not swapped" in minute.reason
    assert "A-11:created_date+closed_date" not in by_id  # never a swap


def test_n16_only_close_dates_are_swapped(tmp_path: Path) -> None:
    ds = write(tmp_path, _requests())
    swap = advanced(ds)["A-11:order_date+ship_date"]
    assert swap.mode == "S" and swap.cells == 6
    assert "2 are 60 days or more apart (23 months, median)" in swap.reason
    assert swap.step is not None and "< 60" in swap.step.when  # type: ignore[attr-defined]


def test_n16_dates_far_apart_are_listed_not_swapped(tmp_path: Path) -> None:
    table = [row[:1] + row[3:] for row in _requests()]
    for row in table[11:17]:
        row[1], row[2] = row[2], row[1]  # put the swapped ones back in order
    far = advanced(write(tmp_path, table))["A-11:order_date+ship_date"]
    assert far.mode == "I" and far.step is None and far.cells == 2
    assert "one date is wrong there, not swapped" in far.reason


def test_n16_start_and_end_words() -> None:
    from quantilo.cleaning.rules import advanced as adv

    def column(name: str) -> Any:
        return adv.Column(name, 0, "DATE", None, 1, 1, None, None)

    assert adv._ordered(column("created_date"), column("closed_date"))
    assert adv._ordered(column("received_on"), column("resolved_on"))
    assert adv._ordered(column("ordered"), column("delivered"))


# --- N17: guards seen for A-05, A-12, A-16, A-17, A-04 --------------------------------------------


def _case(tmp_path: Path, name: str) -> dict[str, CleanSuggestion]:
    from tests.clean_corpus.cases import ADV_CASES

    case = next(c for c in ADV_CASES if c.name == name)
    path = case.write(tmp_path)
    return advanced(dx.open(path, cache_dir=tmp_path / "cache"))


def test_n17_percentages_are_not_added_to_counts_nor_rescaled(tmp_path: Path) -> None:
    by_id = _case(tmp_path, "guards_counts")
    totals = [s for s in by_id.values() if s.rule == "A-12"]
    assert totals and all("% Votants" not in s.columns for s in totals)
    assert "A-12:Inscrits+Votants+Abstentions" in by_id
    assert not any(s.rule == "A-05" for s in by_id.values())
    assert not any(s.rule == "A-04" for s in by_id.values())  # '2A', '971', '[0-10)'


def test_n17_a05_still_rescales_a_percentage_parted_by_a_gap(tmp_path: Path) -> None:
    rng = random.Random(17)
    table: list[list[Any]] = [["order", "discount_pct"]]
    for i in range(400):
        pct = rng.choice([5, 10, 15, 20, 25])
        table.append([i, pct / 100 if i % 12 == 0 else pct])
    by_id = advanced(write(tmp_path, table))
    assert by_id["A-05:discount_pct"].mode == "S"


def test_n17_rows_of_two_hours_and_forecasts_are_left(tmp_path: Path) -> None:
    by_id = _case(tmp_path, "guards_rows")
    assert not any(s.rule in ("A-16", "A-17") for s in by_id.values())


def test_n17_a17_keeps_the_last_rows_but_removes_the_others(tmp_path: Path) -> None:
    rng = random.Random(18)
    table: list[list[Any]] = [["date", "a", "b", "c", "d", "e"]]
    for i in range(300):
        cut = i in (50, 120)
        table.append([f"d{i}", *([None] * 4 if cut else [rng.randint(1, 9) for _ in range(4)]), 1])
    for i in range(10):
        table.append([f"f{i}", None, None, None, None, 1])
    sparse = advanced(write(tmp_path, table))["A-17:rows:sparse"]
    assert sparse.rows == 2 and "The last 10 rows" in sparse.reason
    assert sparse.step is not None and sparse.step.rows == [50, 120]  # type: ignore[attr-defined]


def test_n17_known_codes() -> None:
    from quantilo.cleaning.rules.shapes import _known_code

    assert _known_code("2A", "99") and _known_code("971", "99") and _known_code("2B033", "99999")
    assert _known_code("[0-10)", "[99-99)") and _known_code("[90-100)", "[99-99)")
    assert not _known_code("2C", "99") and not _known_code("A12", "999")


# --- N33: a value of the group repeated on its rows (A-18) ---------------------------------------


def test_n33_a_sale_price_on_every_line_of_its_sale(tmp_path: Path) -> None:
    by_id = _case(tmp_path, "header_values")
    header = by_id["A-18:valeur_fonciere+id_mutation:header"]
    assert header.mode == "I" and header.step is None and header.catalogue == "A-18"
    assert "a value of the id_mutation, not of the row" in header.reason
    assert "sum it once per id_mutation" in header.reason
    assert not any(s.rule == "A-18" and "Sales" in s.columns for s in by_id.values())


# --- N34: columns that say the same thing (A-19) -------------------------------------------------


def test_n34_codes_and_labels_are_said_a_copy_is_offered_for_removal(tmp_path: Path) -> None:
    by_id = _case(tmp_path, "redundant_columns")
    same = by_id["A-19:education+education-num:same"]
    assert same.mode == "I" and same.step is None and "which one to keep" in same.reason
    label = by_id["A-19:measurement_timestamp_label+measurement_timestamp:copy"]
    assert label.mode == "S" and label.step is not None
    assert label.step.columns == ["measurement_timestamp_label"]  # type: ignore[attr-defined]
    assert label.title == (
        "Remove measurement_timestamp_label: measurement_timestamp written as text"
    )
    copy = by_id["A-19:park_borough+borough:copy"]
    assert copy.title == "Remove park_borough: a copy of borough"
    assert not any("Customer Name" in s.columns for s in by_id.values() if s.rule == "A-19")
