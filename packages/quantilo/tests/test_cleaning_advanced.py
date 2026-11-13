"""Clean's Advanced layer (097): keys of spellings, each rule on a small table (what it finds, what
it must leave alone), the messy sample's recall and controls, laziness, impact and risk."""

import csv
import json
import random
from pathlib import Path
from typing import Any

import duckdb
import pytest

import quantilo as dx
from quantilo.cleaning import compile_steps, model
from quantilo.cleaning.phonetic import damerau, fingerprint, fold, ngram_fingerprint, phonetic
from quantilo.cleaning.suggestion import CleanSuggestion

SAMPLES = Path(__file__).resolve().parents[3] / "samples"


def write(tmp_path: Path, rows: list[dict[str, Any]], name: str = "data") -> dx.Dataset:
    path = tmp_path / f"{name}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return dx.open(path, cache_dir=tmp_path / "cache")


def advanced(ds: dx.Dataset) -> dict[str, CleanSuggestion]:
    return {s.id: s for s in ds.clean().advanced()}


def merged(found: dict[str, CleanSuggestion]) -> dict[str, Any]:
    """Every value an A-01 to A-03 suggestion would change → its new value."""
    out: dict[str, Any] = {}
    for suggestion in found.values():
        if suggestion.rule in ("A-01", "A-02", "A-03") and suggestion.step is not None:
            out.update(suggestion.step.map)  # type: ignore[union-attr]
    return out


# --- Keys ----------------------------------------------------------------------------------------


def test_keys_of_spellings() -> None:
    assert fingerprint("Cruise, Tom") == fingerprint("tom  CRUISE") == "cruise tom"
    assert fingerprint("Bourg-en-Bresse") == fingerprint("Bourg en Bresse")
    assert (
        ngram_fingerprint("New York")
        == ngram_fingerprint("NewYork")
        == ngram_fingerprint("new-york")
    )
    assert phonetic("Keyboard") == phonetic("Kiboard")
    assert phonetic("Philadelphia") == phonetic("Filadelfia")
    assert phonetic("Paris") == phonetic("Pariss") != phonetic("Parisot")
    assert fold("  Éric   DUPONT ") == "eric dupont" and fold(" New  York ") == "new york"


def test_damerau_counts_a_swap_once_and_stops_at_its_limit() -> None:
    assert damerau("contractor", "contractro", 2) == 1  # two neighbours swapped
    assert damerau("brooklyn", "brookyn", 1) == 1
    assert damerau("paris", "pariss", 1) == 1
    assert damerau("clinton", "clifton", 1) == 1
    assert damerau("paris", "parisot", 1) == 2  # past the limit: limit + 1
    assert damerau("abcdef", "badcfe", 1) == 2
    assert damerau("same", "same", 1) == 0


def test_spellings_swap_of_two_letters(tmp_path: Path) -> None:
    rows = [{"job": "Contractor", "n": i} for i in range(300)]
    rows += [{"job": "Contractro", "n": i} for i in range(4)]
    rows += [{"job": "?!-?", "n": i} for i in range(300)]  # no sound: never 'the same sound'
    rows += [{"job": "?!-!", "n": i} for i in range(3)]
    assert merged(advanced(write(tmp_path, rows, "jobs"))) == {"Contractro": "Contractor"}


# --- Spellings (A-01, A-02, A-03) ----------------------------------------------------------------


def cities(rng: random.Random) -> list[dict[str, Any]]:
    spellings = [("Paris", 600), ("Parisot", 120), ("Pariss", 4), ("Lyon", 300), ("Lyon.", 6)]
    spellings += [("New York", 200), ("NewYork", 5), ("Model S", 150), ("Model X", 9)]
    spellings += [("Item 12", 150), ("Item 13", 8), ("Toulouse", 200), ("Tolouse", 3)]
    rows = [{"city": v, "amount": rng.randint(1, 99)} for v, c in spellings for _ in range(c)]
    rng.shuffle(rows)
    return rows


def test_spellings_merge_rare_variants_into_the_most_frequent(tmp_path: Path) -> None:
    found = advanced(write(tmp_path, cities(random.Random(1))))
    changes = merged(found)
    assert changes["Pariss"] == "Paris"  # A-03, one letter
    assert changes["Tolouse"] == "Toulouse"
    assert changes["Lyon."] == "Lyon"  # A-01, punctuation
    assert changes["NewYork"] == "New York"  # A-02, no separator
    assert "Parisot" not in changes  # frequent and close: two places (C-21)
    assert "Model X" not in changes and "Item 13" not in changes  # one letter or number (C-22)
    for rule in ("A-01", "A-02", "A-03"):
        suggestion = found[f"{rule}:city"]
        assert suggestion.mode == "S" and suggestion.risk == "review"
        assert suggestion.layer == "advanced" and suggestion.impact == "values"
    assert "'Pariss' (4 rows)" in found["A-03:city"].reason


def test_merges_carry_their_groups_value_by_value(tmp_path: Path) -> None:
    """N38: each merge lists its groups (the target, its rows) and their values with their rows
    and the step's keys they stand for, so the page can untick one and keep the rest."""
    found = advanced(write(tmp_path, cities(random.Random(1))))
    for rule in ("A-01", "A-02", "A-03"):
        suggestion = found[f"{rule}:city"]
        keys = {
            key
            for group in suggestion.clusters
            for value in group.values
            for key in value.spellings
        }
        assert keys == set(suggestion.step.map)  # type: ignore[union-attr]
    a03 = {group.target: group for group in found["A-03:city"].clusters}
    assert a03["Paris"].rows == 600
    assert [(v.value, v.rows, v.spellings) for v in a03["Paris"].values] == [
        ("Pariss", 4, ["Pariss"])
    ]
    assert a03["Toulouse"].values[0].value == "Tolouse"
    assert all(not s.clusters for s in found.values() if s.rule not in ("A-01", "A-02", "A-03"))


def test_spellings_skip_names_of_people_by_distance(tmp_path: Path) -> None:
    rows = [{"last_name": "Martin", "score": i} for i in range(300)]
    rows += [{"last_name": "Martine", "score": i} for i in range(5)]
    found = advanced(write(tmp_path, rows))
    assert "A-03:last_name" not in found  # two spellings may be two people (CLN-5)


def test_spellings_keep_real_neighbours_apart(tmp_path: Path) -> None:
    """C-25 and the corpus: real towns one or two letters apart, names in another order, a set of
    letter pairs that 'Berge' and 'Berger' share."""
    spellings = [("Clinton", 300), ("Clifton", 6), ("Monroe", 300), ("Conroe", 5)]
    spellings += [("Arlington", 300), ("Farmington", 4), ("Hollins", 300), ("Hollis", 6)]
    spellings += [("Brooklyn", 400), ("Brookyn", 3), ("Berger", 200), ("Berge", 5)]
    rows = [{"city": v, "n": i} for v, c in spellings for i in range(c)]
    rows += [{"city": "Paris", "n": i} for i in range(300)]
    changes = merged(advanced(write(tmp_path, rows, "towns")))
    assert changes == {"Brookyn": "Brooklyn"}  # one letter missing in a long word: a typo
    people = [{"first_name": "Jean-Pierre", "n": i} for i in range(200)]
    people += [{"first_name": "Pierre-Jean", "n": i} for i in range(8)]
    people += [{"first_name": "Jean Pierre", "n": i} for i in range(8)]
    found = advanced(write(tmp_path, people, "people"))
    assert merged(found) == {"Jean Pierre": "Jean-Pierre"}  # the order of a name is kept


# --- Units (A-05) --------------------------------------------------------------------------------


def test_units_find_cents_among_euros(tmp_path: Path) -> None:
    rng = random.Random(2)
    prices = [round(rng.lognormvariate(3, 0.6), 2) for _ in range(1_500)]
    rows = [{"unit_price": p if i % 40 else round(p * 100), "id": i} for i, p in enumerate(prices)]
    found = advanced(write(tmp_path, rows))
    suggestion = found["A-05:unit_price"]
    assert suggestion.mode == "S" and suggestion.step is not None
    assert "hundred times larger" in suggestion.reason
    values = list(suggestion.step.map.items())  # type: ignore[union-attr]
    assert all(float(new) * 100 == pytest.approx(float(old)) for old, new in values if new)


def test_units_leave_a_wide_but_single_distribution(tmp_path: Path) -> None:
    rng = random.Random(3)
    rows = [{"population": round(rng.lognormvariate(9, 2.5)), "n": i} for i in range(2_000)]
    assert "A-05:population" not in advanced(write(tmp_path, rows))


# --- Dependencies (A-10) -------------------------------------------------------------------------


def places(rng: random.Random, broken: int) -> list[dict[str, Any]]:
    countries = {"Paris": "France", "Lyon": "France", "Nice": "France", "Madrid": "Spain"}
    countries |= {"Sevilla": "Spain", "Roma": "Italy", "Milano": "Italy", "Berlin": "Germany"}
    countries |= {"Köln": "Germany", "Wien": "Austria", "Porto": "Portugal", "Lisboa": "Portugal"}
    rows = []
    for _ in range(1_200):
        city = rng.choice(sorted(countries))
        rows.append({"city": city, "country": countries[city], "channel": rng.choice("ABC")})
    for row in rows[:broken]:
        row["country"] = "Belgium" if row["country"] != "Belgium" else "France"
    rng.shuffle(rows)
    return rows


def test_dependency_lists_the_rows_that_break_it(tmp_path: Path) -> None:
    ds = write(tmp_path, places(random.Random(4), broken=7))
    found = advanced(ds)
    suggestion = found["A-10:city+country"]
    # B1 (C-45): the conditional step sets country back on these rows only
    assert suggestion.mode == "S" and suggestion.cells == 7
    assert suggestion.step is not None and suggestion.step.op == "set_where"
    connection = duckdb.connect()
    try:
        compiled = compile_steps(connection, str(ds.path), [suggestion.step], strict=True)
        left = connection.execute(
            f"SELECT count(*) FROM ({compiled.sql()}) WHERE country = 'Belgium'"
        ).fetchone()
    finally:
        connection.close()
    assert left == (0,)
    assert suggestion.examples[0].before.startswith("row ")  # type: ignore[union-attr]
    assert "determines country" in suggestion.reason
    assert not any("channel" in s.columns for s in found.values())  # independent: no rule


def test_dependency_without_violation_says_nothing(tmp_path: Path) -> None:
    assert not any(
        s.rule == "A-10" for s in advanced(write(tmp_path, places(random.Random(5), 0))).values()
    )


def test_dependency_accepts_a_frequent_second_answer(tmp_path: Path) -> None:
    """C-27: 'Dominique' is a man's and a woman's name; a strong association (95 %) is not a
    dependency."""
    rng = random.Random(7)
    names = {f"Name{i:02d}": "MF"[i % 2] for i in range(30)}
    rows = [
        {"first": n, "sex": s, "team": rng.choice("ABCD")}
        for n, s in names.items()
        for _ in range(60)
    ]
    rows += [{"first": "Dominique", "sex": "M", "team": "A"} for _ in range(50)]
    rows += [{"first": "Dominique", "sex": "F", "team": "B"} for _ in range(12)]
    found = advanced(write(tmp_path, rows))
    assert "A-10:first+sex" not in found
    loose = [
        {"kind": f"K{i % 12}", "place": f"P{(i % 12) if rng.random() < 0.96 else 0}"}
        for i in range(3_000)
    ]
    assert not any(s.rule == "A-10" for s in advanced(write(tmp_path, loose, "loose")).values())


# --- Cross-column checks (A-11, A-12, A-13) and near-duplicates (A-15) ---------------------------


def orders(rng: random.Random) -> list[dict[str, Any]]:
    from datetime import date, timedelta

    rows = []
    for index in range(800):
        ordered = date(2024, 1, 1) + timedelta(days=rng.randrange(300))
        born = date(1960, 1, 1) + timedelta(days=rng.randrange(12_000))
        quantity, price = rng.randint(1, 5), rng.choice([9.5, 12.0, 30.0, 45.25, 99.0])
        age = ordered.year - born.year - ((ordered.month, ordered.day) < (born.month, born.day))
        rows.append(
            {
                "order_id": f"O-{index:05d}",
                "order_date": ordered.isoformat(),
                "ship_date": (ordered + timedelta(days=rng.randint(1, 5))).isoformat(),
                "birth_date": born.isoformat(),
                "age": age,
                "customer": rng.choice(["Ann", "Bob", "Cy", "Dee", "Eve"]) + str(index % 97),
                "product": rng.choice(["Pen", "Ink", "Pad"]),
                "quantity": quantity,
                "unit_price": price,
                "total": round(quantity * price, 2),
            }
        )
    for row in rows[:12]:
        row["ship_date"], row["order_date"] = row["order_date"], row["ship_date"]
    for row in rows[12:20]:
        row["total"] = round(row["total"] + 10, 2)
    for row in rows[20:29]:
        row["age"] += 9
    for index, row in enumerate(rows[30:40]):
        rows.append(dict(row, order_id=f"O-{9_000 + index}", product=row["product"].upper()))
    return rows


def test_cross_column_checks_and_near_duplicates(tmp_path: Path) -> None:
    found = advanced(write(tmp_path, orders(random.Random(6))))
    assert found["A-11:order_date+ship_date"].cells == 12
    total = found["A-12:total+quantity+unit_price"]
    assert total.cells == 8 and "total = quantity × unit_price" in total.reason
    assert found["A-13:age+birth_date"].cells == 9
    near = found["A-15:rows:near"]
    assert near.rows == 10 and near.impact == "rows" and near.step is not None
    assert "order_id" not in near.step.keys  # type: ignore[union-attr]


def test_near_duplicates_need_an_anchor_and_keep_symbols(tmp_path: Path) -> None:
    """C-26: customers alike (same plan, same charge) are not one customer entered twice; '<=50K'
    and '>50K' differ."""
    rng = random.Random(8)
    rows = [
        {
            "customerID": f"{i:04d}-X",
            "contract": rng.choice(["Month", "Year"]),
            "internet": rng.choice(["DSL", "Fiber", "No"]),
            "charges": rng.choice([20.2, 19.9, 70.5]),
        }
        for i in range(600)
    ]
    assert "A-15:rows:near" not in advanced(write(tmp_path, rows, "alike"))
    people = [
        {
            "day": f"2024-01-{1 + i % 28:02d}",
            "who": f"P{i % 150}",
            "income": rng.choice(["<=50K", ">50K"]),
        }
        for i in range(300)
    ]
    people.append(dict(people[0], income=">50K" if people[0]["income"] == "<=50K" else "<=50K"))
    assert "A-15:rows:near" not in advanced(write(tmp_path, people, "symbols"))


# --- The messy sample ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def messy(tmp_path_factory: pytest.TempPathFactory) -> tuple[dx.Dataset, Any]:
    ds = dx.open(SAMPLES / "messy_orders.csv", cache_dir=tmp_path_factory.mktemp("messy"))
    cleaning = ds.clean()
    cleaning.advanced()
    return ds, cleaning


def test_messy_sample_recall_modes_and_controls(messy: tuple[dx.Dataset, Any]) -> None:
    _, cleaning = messy
    expected = json.loads((SAMPLES / "messy_orders.expected.json").read_text(encoding="utf-8"))
    found = {s.id: s for s in cleaning.suggestions}
    for layer in ("basic", "advanced"):
        wanted = expected[layer]
        hits = [e for e in wanted if e["id"] in found and found[e["id"]].mode == e["mode"]]
        missing = [e["id"] for e in wanted if e not in hits]
        assert len(hits) / len(wanted) >= 0.9, f"{layer} missing: {missing}"
        assert not missing  # 100 % today; the 90 % above is the floor a rule change may not cross
    touched = {c for s in found.values() for c in s.columns}
    assert not touched & set(expected["controls"])
    steps = [s.step for s in found.values() if s.step is not None]
    for step in steps:
        named = [getattr(step, "column", None), *getattr(step, "columns", [])]
        assert not set(filter(None, named)) & set(expected["controls"])
    changes = merged(found)
    for rare, frequent in expected["never_merged"]:
        assert changes.get(rare) != frequent and changes.get(frequent) != rare


def test_messy_sample_advanced_steps_run_and_fix(
    messy: tuple[dx.Dataset, Any], tmp_path: Path
) -> None:
    ds, cleaning = messy
    picked = [s.id for s in cleaning.suggestions if s.layer == "advanced" and s.step is not None]
    cleaning.use(*picked)
    connection = duckdb.connect()
    try:
        compiled = compile_steps(connection, str(ds.path), cleaning.steps, strict=True)
        cities = {
            r[0]
            for r in connection.execute(f"SELECT DISTINCT city FROM ({compiled.sql()})").fetchall()
        }
        prices = connection.execute(f"SELECT max(unit_price) FROM ({compiled.sql()})").fetchone()
    finally:
        connection.close()
    assert not {"Pariss", "Marseile", "SaintEtienne", "Bourg en Bresse"} & cities
    assert "Parisot" in cities
    assert prices is not None and float(prices[0]) < 1_000  # no price left in cents
    cleaned = cleaning.apply(cache_dir=tmp_path)
    assert cleaning.report is not None and len(cleaned) < len(ds)


def test_advanced_is_lazy_and_computed_once(messy: tuple[dx.Dataset, Any]) -> None:
    ds, _ = messy
    cleaning = ds.clean()
    assert all(s.layer == "basic" for s in cleaning.suggestions)  # C-16
    first = cleaning.advanced()
    assert cleaning.advanced() is first
    assert {s.mode for s in first} <= {"S", "I"} and {s.risk for s in first} == {"review"}
    ids = [s.step.id for s in cleaning.suggestions if s.step is not None]
    assert len(ids) == len(set(ids))  # Advanced step ids follow the Basic ones
    assert not any(s.suggestion in {a.id for a in first} for s in cleaning.steps)  # none ticked


def test_every_suggestion_has_impact_and_risk(messy: tuple[dx.Dataset, Any]) -> None:
    _, cleaning = messy
    by_rule = {s.id: s for s in cleaning.suggestions}
    for suggestion in cleaning.suggestions:
        assert suggestion.impact in model.IMPACTS and suggestion.risk in model.RISKS
        assert suggestion.risk == ("safe" if suggestion.mode == "A" else "review")
    assert by_rule["B-01:city"].impact == "format"
    assert by_rule["B-05:city"].impact == "format"
    assert by_rule["B-03:city"].impact == "values"
    assert by_rule["B-11:order_date"].impact == "types"
    assert by_rule["B-14:rows"].impact == "rows"
    assert by_rule["B-19:currency"].impact == "columns"
