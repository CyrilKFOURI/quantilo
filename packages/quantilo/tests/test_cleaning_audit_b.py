"""Audit of 2026-11-12, wave 1 agent B (Basic): N06, N07, N08, N09, N12, N47.

Each test reproduces on a small synthetic file what the audit saw on a real one (Ames, kiwi,
adult, diabetes, air quality, superstore); no corpus file is copied."""

import csv
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


def apply(ds: dx.Dataset, steps: list[Any], sql: str) -> list[tuple[Any, ...]]:
    connection = duckdb.connect()
    try:
        compiled = compile_steps(connection, str(ds.path), steps, strict=True)
        return connection.execute(sql.replace("{t}", f"({compiled.sql()})")).fetchall()
    finally:
        connection.close()


# --- N06: 'NA' that means "none" -----------------------------------------------------------------


def _houses(rows: int = 400) -> list[list[Any]]:
    """Ames' shape: no garage ⇔ Garage Area 0 and Garage Type 'NA'; a free note with a few 'NA'."""
    table: list[list[Any]] = [["Garage Type", "Garage Area", "Garage Cars", "Remark", "Price"]]
    kinds = ["Attchd", "Detchd", "BuiltIn"]
    for i in range(rows):
        garage = i % 10 != 0  # 40 houses without a garage
        table.append(
            [
                kinds[i % 3] if garage else "NA",
                300 + i if garage else 0,
                1 + i % 3 if garage else 0,
                "NA" if i % 50 == 1 else f"note {i % 37}",
                100_000 + 7 * i,
            ]
        )
    return table


def test_n06_na_where_the_garage_area_is_zero_means_no_garage(tmp_path: Path) -> None:
    ds = write(tmp_path, _houses())
    by_id = found(ds)
    assert "B-04:Garage Type" not in by_id  # never automatic any more
    item = by_id["B-04:Garage Type:none"]
    assert item.mode == "S" and item.risk == "review" and item.cells == 40
    assert "where Garage Area is 0" in item.reason or "where Garage Cars is 0" in item.reason
    assert "no garage" in item.reason and item.reason.endswith("Check your data.")
    assert item.step is not None and item.step.op == "map_values"
    assert item.step.map == {"NA": "None"}  # type: ignore[union-attr]
    rows = apply(ds, [item.step], "SELECT count(*) FROM {t} WHERE \"Garage Type\" = 'None'")
    assert rows == [(40,)]
    assert "B-04:Garage Type:maybe" not in by_id  # one reading of 'NA', not two


def test_n06_a_bare_na_without_a_number_beside_it_is_only_suggested(tmp_path: Path) -> None:
    by_id = found(write(tmp_path, _houses()))
    assert "B-04:Remark" not in by_id and "B-04:Remark:none" not in by_id
    assert by_id["B-04:Remark:maybe"].mode == "S"


def test_n06_n_a_stays_automatic_and_na_becomes_suggested(tmp_path: Path) -> None:
    names = ["N/A"] * 30 + ["NA"] * 10 + [f"Builder {i % 90}" for i in range(360)]
    by_id = found(write(tmp_path, [["business_name"], *([v] for v in names)]))
    assert by_id["B-04:business_name"].mode == "A"
    assert "'N/A'" in by_id["B-04:business_name"].title
    assert "'NA'" not in by_id["B-04:business_name"].title
    assert by_id["B-04:business_name:maybe"].mode == "S"
    assert "'NA'" in by_id["B-04:business_name:maybe"].title


# --- N07: French and Excel words for "missing" ---------------------------------------------------


def test_n07_french_tokens_and_excel_french_errors(tmp_path: Path) -> None:
    rows: list[list[Any]] = [["player", "ticket", "comment", "gender"]]
    for i in range(200):
        rows.append(
            [
                f"Player {i % 40}",
                ["n/d", "#VALEUR!", "N.D."][i % 3] if i % 9 == 0 else f"{100 + i % 7} pts",
                ["néant", "N.C.", "—"][i % 3] if i % 11 == 0 else f"remark {i % 23}",
                "Unknown/Invalid" if i == 5 else ["Female", "Male"][i % 2],
            ]
        )
    by_id = found(write(tmp_path, rows))
    ticket = by_id["B-04:ticket"]
    assert ticket.mode == "A" and ticket.cells == 23
    assert "French" in ticket.reason and "Excel" in ticket.reason
    comment = by_id["B-04:comment:maybe"]  # 'néant', 'N.C.', '—' can be an answer
    assert comment.mode == "S" and comment.cells == 19 and "French" in comment.reason
    assert "B-04:comment" not in by_id
    assert by_id["B-04:gender:maybe"].mode == "S"
    assert "Unknown/Invalid" in by_id["B-04:gender:maybe"].title


# --- N08: a weak token in several columns is the file's code for "missing" ----------------------


def _census(
    rows: int = 300, questioned: tuple[str, ...] = ("work", "job", "country", "note")
) -> list[list[Any]]:
    table: list[list[Any]] = [["age", "work", "job", "country", "weight", "note"]]
    for i in range(rows):
        table.append(
            [
                20 + i % 50,
                "?" if "work" in questioned and i % 20 == 0 else ["Private", "State-gov"][i % 2],
                "?" if "job" in questioned and i % 25 == 0 else ["Sales", "Tech", "Admin"][i % 3],
                "?" if "country" in questioned and i % 30 == 0 else ["US", "Mexico"][i % 2],
                "?" if i % 10 else ["[75-100)", "[50-75)"][i % 2],  # '?' on 90 % of the rows
                "?" if "note" in questioned and i == 7 else f"remark {i % 40}",
            ]
        )
    return table


def test_n08_question_mark_in_several_columns_is_read_as_missing_everywhere(
    tmp_path: Path,
) -> None:
    by_id = found(write(tmp_path, _census()))
    for name, cells in (("work", 15), ("job", 12), ("country", 10), ("weight", 270), ("note", 1)):
        item = by_id[f"B-04:{name}:file"]
        assert item.mode == "A" and item.risk == "safe" and item.cells == cells
        assert "'?' stands for 'missing' in 5 columns of this file" in item.reason
        assert item.reason.endswith("Check your data.")
        assert f"B-04:{name}:maybe" not in by_id


def test_n08_a_token_in_one_column_only_stays_suggested(tmp_path: Path) -> None:
    by_id = found(write(tmp_path, _census(questioned=("work",))))
    # 'work' is the only column where '?' is a minority: weight (90 %) does not make it a code
    assert not [k for k in by_id if k.endswith(":file")]
    assert by_id["B-04:work:maybe"].mode == "S"
    assert by_id["B-04:weight:maybe"].mode == "S"


# --- N09: one sentinel repeated over several measure columns ------------------------------------


def _sensors(columns: int, rows: int = 400) -> list[list[Any]]:
    names = ["co", "no2", "temp", "humidity"][:columns]
    table: list[list[Any]] = [["hour", *names]]
    for i in range(rows):
        table.append(
            [
                i,
                *(
                    -200 if (i + k) % 25 == 0 else round(1.5 + (i * (k + 3)) % 97 / 10, 1)
                    for k, _ in enumerate(names)
                ),
            ]
        )
    return table


def test_n09_a_sentinel_in_three_columns_is_read_as_missing_automatically(tmp_path: Path) -> None:
    by_id = found(write(tmp_path, _sensors(4)))
    items = [s for s in by_id.values() if s.rule == "B-09"]
    assert {s.id for s in items} == {
        f"B-09:{c}:-200:repeated" for c in ("co", "no2", "temp", "humidity")
    }
    assert all(s.mode == "A" and s.risk == "safe" for s in items)
    assert sum(s.cells for s in items) == 64
    assert (
        "-200 appears in 4 measure columns (16 rows in co)" in by_id["B-09:co:-200:repeated"].reason
    )
    assert by_id["B-09:co:-200:repeated"].reason.endswith("Check your data.")


def test_n09_a_sentinel_in_two_columns_stays_suggested(tmp_path: Path) -> None:
    by_id = found(write(tmp_path, _sensors(2)))
    assert by_id["B-09:co:-200"].mode == "S" and by_id["B-09:no2:-200"].mode == "S"
    assert not [k for k in by_id if k.endswith(":repeated")]


# --- N12: the country guard of the postal codes --------------------------------------------------


def _addresses(belgian: int, lost: int = 20, country: bool = True) -> list[list[Any]]:
    table: list[list[Any]] = [
        ["order", "zip_code", "country"] if country else ["order", "zip_code"]
    ]
    for i in range(400):
        if i < lost:
            row = [f"O{i}", str(1000 + i), "France"]  # 01000…: lost its zero
        elif i < lost + belgian:
            row = [f"O{i}", str(1000 + 10 * i), "Belgium"]  # 4 digits by nature
        else:
            row = [f"O{i}", str(75001 + i % 20), "France"]
        table.append(row if country else row[:2])
    return table


def test_n12_without_a_country_column_the_lost_zeros_stay_automatic(tmp_path: Path) -> None:
    item = found(write(tmp_path, _addresses(0, country=False)))["B-13:zip_code"]
    assert item.mode == "A" and item.cells == 20


def test_n12_shorter_codes_all_of_a_five_digit_country_stay_automatic(tmp_path: Path) -> None:
    item = found(write(tmp_path, _addresses(0)))["B-13:zip_code"]
    assert item.mode == "A" and item.cells == 20
    assert "20 postal codes of France rows" in item.reason


def test_n12_belgian_codes_make_it_suggested_and_are_never_padded(tmp_path: Path) -> None:
    ds = write(tmp_path, _addresses(15))
    item = found(ds)["B-13:zip_code"]
    assert item.mode == "S" and item.risk == "review" and item.cells == 20
    assert "15 rows of Belgium have 4-digit codes by nature: not padded." in item.reason
    assert item.step is not None
    rows = apply(ds, [item.step], "SELECT country, zip_code FROM {t} ORDER BY __quantilo_row")
    assert rows[0] == ("France", "01000") and rows[19] == ("France", "01019")
    assert rows[20] == ("Belgium", "1200") and rows[34] == ("Belgium", "1340")


def test_n12_only_belgian_short_codes_need_nothing(tmp_path: Path) -> None:
    assert "B-13:zip_code" not in found(write(tmp_path, _addresses(15, lost=0)))


# --- N47: the header shifted under a note (B-24) -------------------------------------------------


def _sheet(tmp_path: Path, note_rows: int) -> dx.Dataset:
    """kiwi's shape: the reader took a note for the header, the real names come under it."""
    rows: list[list[Any]] = [["Market overview (Morocco)", *(f"column_{i}" for i in range(2, 7))]]
    rows += [["n/d = not disclosed", "", "", "", "", ""]] * note_rows
    rows.append(["Player", "Type", "HQ", "Founded", "Avg ticket", "Source"])
    for i in range(12):
        rows.append([f"Player {i}", "Equity", "Dubai", str(2000 + i), f"{100 + i} MAD", "a.com"])
    return write(tmp_path, rows, f"sheet{note_rows}")


def test_n47_the_real_names_under_a_note_are_suggested_as_the_header(tmp_path: Path) -> None:
    ds = _sheet(tmp_path, 1)
    item = found(ds)["B-24:header"]
    assert item.mode == "S" and item.risk == "review" and item.impact == "columns"
    assert item.title == "Use row 2 as the column names" and item.rows == 2
    assert "Row 2 looks like the real column names (Player, Type, HQ…)" in item.reason
    assert "the row above is a note" in item.reason and item.reason.endswith("Check your data.")
    assert item.step is not None and item.step.row == 2  # type: ignore[union-attr]
    connection = duckdb.connect()
    compiled = compile_steps(connection, str(ds.path), [item.step], strict=True)
    assert compiled.final.names == ["Player", "Type", "HQ", "Founded", "Avg ticket", "Source"]
    first = connection.execute(f"SELECT count(*), min(Player) FROM ({compiled.sql()})").fetchone()
    assert first == (12, "Player 0")
    connection.close()


def test_n47_names_on_the_first_row_need_no_row_removed(tmp_path: Path) -> None:
    item = found(_sheet(tmp_path, 0))["B-24:header"]
    assert item.title == "Use row 1 as the column names" and item.rows == 1
    assert item.step is not None and item.step.row == 1  # type: ignore[union-attr]


def test_n47_a_file_without_header_whose_first_row_is_numbers_gets_nothing(tmp_path: Path) -> None:
    rows: list[list[Any]] = [[f"column{i:02d}" for i in range(1, 6)]]
    for i in range(30):
        rows.append([str(30 + i), "State-gov", str(77516 + i), "Bachelors", str(13)])
    assert "B-24:header" not in found(write(tmp_path, rows))


def test_n07_nc_and_nd_stay_states_in_a_column_of_state_codes(tmp_path: Path) -> None:
    states = ["NY"] * 300 + ["NJ"] * 80 + ["ND"] * 6 + ["NC"] * 4
    by_id = found(write(tmp_path, [["state"], *([v] for v in states)]))
    assert "B-04:state:maybe" not in by_id and "B-04:state" not in by_id
