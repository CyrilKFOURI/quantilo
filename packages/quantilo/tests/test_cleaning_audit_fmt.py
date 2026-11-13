"""Audit of 2026-11-12, wave 3 (formats): N26 (e-mail domains), N24 (F-04, countries), N20 (F-01,
phone numbers in the international form), N23 (F-05, quantities with a unit).

Each test builds a small synthetic file with the pattern the audit saw (the French probe, Open Food
Facts' quantities, Northwind's countries and phones); no corpus file is copied."""

import csv
import random
from pathlib import Path
from typing import Any

import duckdb

import quantilo as dx
from quantilo.cleaning import catalogue, compile_steps
from quantilo.cleaning.rules import formats, shapes
from quantilo.cleaning.suggestion import CleanSuggestion


def write(tmp_path: Path, rows: list[list[Any]], name: str = "data") -> dx.Dataset:
    path = tmp_path / f"{name}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)
    return dx.open(path, cache_dir=tmp_path / f"cache-{name}")


def advanced(ds: dx.Dataset) -> dict[str, CleanSuggestion]:
    return {s.id: s for s in ds.clean().advanced()}


def apply(ds: dx.Dataset, steps: list[Any], sql: str) -> list[tuple[Any, ...]]:
    connection = duckdb.connect()
    try:
        compiled = compile_steps(connection, str(ds.path), steps, strict=True)
        return connection.execute(sql.replace("{t}", f"({compiled.sql()})")).fetchall()
    finally:
        connection.close()


# --- N26: e-mails --------------------------------------------------------------------------------


def test_domain_slips_are_coded_and_the_local_part_is_never_touched() -> None:
    assert shapes._fixed_domain("gmial.com") == "gmail.com"
    assert shapes._fixed_domain("GMAIL.FR") == "gmail.com"
    assert shapes._fixed_domain("orange.con") == "orange.com"
    assert shapes._fixed_domain("exemple.cmo") == "exemple.com"
    for real in ("gmail.com", "hotmail.fr", "yahoo.fr", "outlook.fr", "exemple.co", "laposte.net"):
        assert shapes._fixed_domain(real) is None, real
    assert shapes._small_domain("Jean.Dupont@Gmail.COM") == "Jean.Dupont@gmail.com"


def emails(rng: random.Random) -> list[list[Any]]:
    rows: list[list[Any]] = [["id", "email"]]
    for i in range(300):
        domain = rng.choice(("exemple.fr", "gmail.com", "orange.fr"))
        name = f"client{i}" if i % 30 else f"Client{i}"
        if i % 50 == 7:
            domain = ("gmial.com", "hotmial.com", "orange.con", "gmail.fr")[(i // 50) % 4]
        elif i % 25 == 3:
            domain = domain.upper()
        rows.append([i, f"{name}@{domain}"])
    return rows


def test_emails_domain_in_small_letters_and_slips(tmp_path: Path) -> None:
    ds = write(tmp_path, emails(random.Random(1)))
    found = advanced(ds)
    case = found["A-04:email:case"]
    assert case.step is not None and case.mode == "S"
    assert all(k.split("@")[0] == v.split("@")[0] for k, v in case.step.map.items())  # type: ignore[union-attr]
    assert "Client0@exemple.fr" not in case.step.map  # type: ignore[union-attr]  # the name stays
    slips = found["A-04:email:domain"]
    assert slips.mode == "S" and slips.catalogue == "A-04:domain"
    mapping = slips.step.map  # type: ignore[union-attr]
    assert mapping["client7@gmial.com"] == "client7@gmail.com"
    assert mapping["client57@hotmial.com"] == "client57@hotmail.com"
    assert mapping["client107@orange.con"] == "client107@orange.com"
    assert mapping["client157@gmail.fr"] == "client157@gmail.com"
    assert slips.cells == 6 and "4 e-mail domains look mistyped" in slips.reason
    # both steps ticked, in either order: every slip fixed, no name changed
    for steps in ([case.step, slips.step], [slips.step, case.step]):
        rows = apply(ds, steps, "SELECT email FROM {t} ORDER BY id")
        values = [r[0] for r in rows]
        assert not any(d in v for v in values for d in ("gmial", "hotmial", ".con", "gmail.fr"))
        assert values[0] == "Client0@exemple.fr"
    assert catalogue.CATALOGUE["A-04:domain"].mode == "decision"


def test_no_slip_no_suggestion(tmp_path: Path) -> None:
    rows: list[list[Any]] = [["email"]] + [[f"user{i}@example.com"] for i in range(50)]
    found = advanced(write(tmp_path, rows))
    assert not any(s.rule == "A-04" for s in found.values())


# --- N24: countries (F-04) -----------------------------------------------------------------------


def test_the_country_table() -> None:
    table = formats.country_table()
    assert len(table.countries) >= 249 and not table.ambiguous  # no writing shared by two
    read = formats.read_country
    assert read("France") == ("FR", frozenset({"en", "fr"}))
    assert read("fra") == ("FR", frozenset({"a3"}))
    assert read("U.S.A.") == ("US", frozenset({"a3"}))
    assert read("United-States") == ("US", frozenset({"en"}))  # uci_adult
    assert read("UK") == ("GB", frozenset({"abbr"}))
    assert read("États-Unis")[0] == read("Etats Unis")[0] == "US"  # type: ignore[index]
    assert read("The Netherlands")[0] == "NL"  # type: ignore[index]
    for other in ("World", "Europe", "England", "West Germany", "Soviet Union", "NA", "", "?"):
        assert read(other) is None, other


def countries(values: list[str], name: str = "pays") -> list[list[Any]]:
    return [["id", name, "amount"]] + [[i, v, i % 7] for i, v in enumerate(values)]


def test_countries_written_several_ways_take_the_columns_writing(tmp_path: Path) -> None:
    rng = random.Random(3)
    values = [rng.choice(("France", "FR", "FRA", "france", "Belgique", "BE")) for _ in range(300)]
    ds = write(tmp_path, countries(values))
    found = advanced(ds)["F-04:pays:countries"]
    assert found.mode == "S" and found.step is not None
    assert found.step.map == {  # type: ignore[union-attr]
        "FR": "France",
        "FRA": "France",
        "france": "France",
        "BE": "Belgique",
    }
    assert "French names" in found.reason and "5 ways to write 2 countries" in found.reason
    rows = apply(ds, [found.step], "SELECT DISTINCT pays FROM {t} ORDER BY 1")
    assert [r[0] for r in rows] == ["Belgique", "France"]


def test_codes_among_names_and_groups_left_alone(tmp_path: Path) -> None:
    names = ["Germany", "France", "Brazil", "Spain", "Italy", "Mexico", "Canada", "Japan"]
    values = names * 10 + ["USA"] * 30 + ["UK"] * 12 + ["West Germany", "World", "England"]
    found = advanced(write(tmp_path, countries(values, "country")))["F-04:country:countries"]
    assert found.step.map == {"USA": "United States", "UK": "United Kingdom"}  # type: ignore[union-attr]
    assert "2 countries written as codes among names" in found.reason
    assert "'West Germany'" in found.reason and "never merged" in found.reason


def test_no_suggestion_when_each_country_is_written_once(tmp_path: Path) -> None:
    # OWID: groups beside countries; messy_orders: 'france' beside 'France' is B-03's (case only)
    places = ["France", "World", "High-income countries", "India", "Kosovo", "Congo"] * 10
    assert "F-04:country:countries" not in advanced(write(tmp_path, countries(places, "country")))
    cased = ["France"] * 40 + ["france"] * 5 + ["Belgium"] * 10
    assert not any(
        s.rule == "F-04" for s in advanced(write(tmp_path, countries(cased), "c")).values()
    )
    # a column of codes (ISO alpha-2 and alpha-3 mixed, nothing else) is not mixed with names
    codes = ["FR", "DE", "ES", "IT", "BE"] * 10 + ["CHE"] * 5
    found = advanced(write(tmp_path, countries(codes, "country"), "codes"))
    assert not any(s.rule == "F-04" for s in found.values())


# --- N20: phone numbers (F-01) -------------------------------------------------------------------


def test_dialled_numbers() -> None:
    plans = formats.phone_plans()
    fr, no, it, us = plans["FR"], plans["NO"], plans["IT"], plans["US"]
    dial = formats.dial
    for written in (
        "+33 (0)6 12 34 56 78",
        "0033 6 12 34 56 78",
        "+33 06 12 34 56 78",
        "+33612345678",
    ):
        assert dial(written).e164 == "+33612345678", written
    assert dial("+1 (503) 555-9573").e164 == "+15035559573"
    assert dial("+36 1 234 5678").plan is None  # Hungary: not in the coded plans
    assert dial("+33 6 12 34").e164 is None  # a known code, an impossible length
    assert dial("06.12.34.56.78", fr).e164 == "+33612345678"
    assert dial("612345678", fr).e164 == "+33612345678"  # a lost leading zero
    assert dial("0612345", fr).e164 is None
    assert dial("(1) 42.34.22.66", fr).e164 == "+33142342266"  # Paris, without its 0
    assert dial("0522-556721", it).e164 == "+390522556721"  # Italy keeps its 0
    assert dial("1 503 555 9573", us).e164 == dial("(503) 555-9573", us).e164 == "+15035559573"
    assert dial("(100) 555-4822", us).e164 is None  # an area code never starts with 1
    assert dial("07-98 92 35", no).e164 is None  # a Norwegian number never starts with 0
    assert dial("(0)2-953010", no).e164 is None  # the trunk '(0)' dropped: 7 digits
    assert dial("06 12 34 56 78").e164 is None  # no code, no country: never guessed


def french_phones(rng: random.Random) -> list[str]:
    out = []
    for _ in range(300):
        digits = "06" + "".join(rng.choice("0123456789") for _ in range(8))
        pairs = [digits[k : k + 2] for k in range(0, 10, 2)]
        out.append(
            rng.choice(
                (
                    " ".join(pairs),
                    digits,
                    "+33 " + digits[1] + " " + " ".join(pairs[1:]),
                    ".".join(pairs),
                )
            )
        )
    return out


def test_phones_of_one_country_by_the_numbers_with_a_code(tmp_path: Path) -> None:
    phones = french_phones(random.Random(5))
    places = [random.Random(i).choice(("France", "Belgique")) for i in range(300)]
    rows: list[list[Any]] = [["id", "telephone", "pays"]] + [
        [i, p, c] for i, (p, c) in enumerate(zip(phones, places, strict=True))
    ]
    ds = write(tmp_path, rows)
    found = advanced(ds)
    suggestion = found["F-01:telephone:e164"]
    assert suggestion.mode == "S" and suggestion.step is not None and suggestion.cells == 300
    assert "is not the numbers' own" in suggestion.reason  # pays is random: rejected
    assert "of France (+33)" in suggestion.reason
    fixed = apply(ds, [suggestion.step], "SELECT telephone FROM {t}")
    assert all(v.startswith("+336") and len(v) == 12 for (v,) in fixed)


def test_phones_by_the_rows_country(tmp_path: Path) -> None:
    rows: list[list[Any]] = [["id", "country", "phone"]]
    examples = [
        ("Germany", "030-0074321", "+49300074321"),
        ("UK", "(171) 555-7788", "+441715557788"),
        ("Spain", "(91) 555 22 82", "+34915552282"),
        ("USA", "(503) 555-7555", "+15035557555"),
        ("Italy", "0522-556721", "+390522556721"),
        ("France", "(1) 42.34.22.66", "+33142342266"),
    ]
    for i in range(60):
        country, phone, _ = examples[i % len(examples)]
        rows.append([i, country, phone[:-1] + str(i // len(examples))])
    rows += [[100, "France", "88.60.15.31"], [101, "Japan", "(03) 3555-5011"]]
    rows += [[102, "Norway", "07-98 92 35"]]
    found = advanced(write(tmp_path, rows))
    mapping = found["F-01:phone:e164"].step.map  # type: ignore[union-attr]
    for country, phone, e164 in examples:
        assert mapping[phone[:-1] + "0"] == e164[:-1] + "0", country
    assert "88.60.15.31" not in mapping and "(03) 3555-5011" not in mapping
    length = found["F-01:phone:length"]
    assert length.mode == "I" and length.step is None and length.cells == 2  # France, Norway
    assert "1 number is of a country whose plan" in found["F-01:phone:e164"].reason


def test_no_country_or_one_layout_nothing(tmp_path: Path) -> None:
    rng = random.Random(7)
    bare = [
        f"{rng.randint(200, 999)}{rng.randint(200, 999)}{rng.randint(0, 9999):04d}"
        for _ in range(80)
    ]
    found = advanced(write(tmp_path, [["phone"]] + [[p] for p in bare]))
    assert not any(s.rule == "F-01" for s in found.values())  # one layout (nyc_dob)
    national = [p for p in french_phones(random.Random(8)) if not p.startswith("+")]
    found = advanced(write(tmp_path, [["phone"]] + [[p] for p in national], "fr"))
    assert not any(s.rule == "F-01" for s in found.values())  # no code, no country: not guessed


# --- N23: quantities with a unit (F-05) ----------------------------------------------------------


def test_read_quantity() -> None:
    read = formats.read_quantity
    assert read("12 kg") == ("grams", 12_000.0, "kg")
    assert read("1,5 kg") == ("grams", 1_500.0, "kg")
    assert read("12 x 33 cl") == ("litres", 3.96, "cl")  # the multiplier read
    assert read("2x100g") == ("grams", 200.0, "g")
    assert read("8 fl. oz")[0] == "litres" and read("16 OZ")[0] == "grams"  # type: ignore[index]
    assert read("150 g e") == ("grams", 150.0, "g")  # the EU sign of an estimated quantity
    for other in ("3 boxes", "12", "7 oz (198 g)", "24 - 12 oz bottles", "kg", "1.5.2 kg"):
        assert read(other) is None, other


def test_quantities_become_numbers_in_new_columns(tmp_path: Path) -> None:
    rng = random.Random(4)
    weights = ("12 kg", "500 g", "1,5 kg", "750 g", "2 kg")
    packs = ("33 cl", "1 L", "12 x 33 cl", "500 ml", "250 g", "half a litre")
    rows: list[list[Any]] = [["id", "poids", "contenance"]]
    rows += [[i, rng.choice(weights), packs[i % len(packs)] if i % 20 else "?"] for i in range(300)]
    ds = write(tmp_path, rows)
    found = advanced(ds)
    weight = found["F-05:poids:units"]
    assert weight.mode == "S" and weight.catalogue == "F-05" and weight.cells == 300
    assert weight.step is not None and weight.step.op == "parse_quantity"
    assert weight.step.to_grams == "poids_g" and weight.step.to_litres is None  # type: ignore[union-attr]
    volume = found["F-05:contenance:units"]
    assert volume.step.to_litres == "contenance_l" and volume.step.to_grams == "contenance_g"  # type: ignore[union-attr]
    assert "'half a litre'" in volume.reason  # an unknown unit is said, and stays empty
    out = apply(
        ds,
        [weight.step, volume.step],
        "SELECT poids, poids_g, poids_unit, contenance, contenance_l, contenance_g FROM {t}",
    )
    grams = {"12 kg": 12_000, "500 g": 500, "1,5 kg": 1_500, "750 g": 750, "2 kg": 2_000}
    litres = {"33 cl": 0.33, "1 L": 1.0, "12 x 33 cl": 3.96, "500 ml": 0.5}
    for poids, poids_g, unit, pack, pack_l, pack_g in out:
        assert poids_g == grams[poids] and unit == poids.split()[-1]
        assert pack_l == litres.get(pack) and pack_g == (250 if pack == "250 g" else None)


def test_a_packing_is_no_quantity(tmp_path: Path) -> None:
    packs = ["24 - 12 oz bottles", "10 boxes x 20 bags", "5 kg pkg.", "1 kg pkg.", "36 boxes"]
    rows: list[list[Any]] = [["product", "quantityPerUnit"]] + [
        [i, packs[i % 5]] for i in range(60)
    ]
    assert not any(s.rule == "F-05" for s in advanced(write(tmp_path, rows)).values())
