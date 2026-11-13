"""Precision of Advanced rules A-01, A-02, A-10 on the false positives the cleaning audit found on
real files (docs/etudes/2026-11-12-audit-nettoyage.md, N13 to N15): columns of lists, names of
people, repeated exceptions. Each pattern has a small table, and so has each true positive kept."""

import csv
import random
from pathlib import Path
from typing import Any

import quantilo as dx
from quantilo.cleaning.phonetic import fingerprint, items, items_key, ngram_fingerprint
from quantilo.cleaning.suggestion import CleanSuggestion


def write(tmp_path: Path, rows: list[dict[str, Any]], name: str = "data") -> dx.Dataset:
    path = tmp_path / f"{name}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return dx.open(path, cache_dir=tmp_path / "cache")


def advanced(ds: dx.Dataset) -> dict[str, CleanSuggestion]:
    return {s.id: s for s in ds.clean().advanced()}


def merged(found: dict[str, CleanSuggestion], rule: str | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for suggestion in found.values():
        wanted = suggestion.rule in ("A-01", "A-02", "A-03") and rule in (None, suggestion.rule)
        if wanted and suggestion.step is not None:
            out.update(suggestion.step.map)  # type: ignore[union-attr]
    return out


def table(spellings: list[tuple[str, int]], column: str) -> list[dict[str, Any]]:
    rows = [{column: v, "n": i} for v, c in spellings for i in range(c)]
    random.Random(13).shuffle(rows)
    return rows


# --- N13: columns of lists -----------------------------------------------------------------------


def test_list_keys_compare_items_not_pooled_words() -> None:
    assert items("en:vegan,  en:palm-oil-free ; x|") == ["en:vegan", "en:palm-oil-free", "x"]
    vegetarian = items_key("en:palm-oil, en:vegetarian", fingerprint)
    assert vegetarian == items_key("en:vegetarian,en:palm-oil", fingerprint)  # another order
    assert vegetarian != items_key("en:palm-oil, en:non-vegetarian", fingerprint)
    # one fingerprint of the whole value pools the words of the items: the audit's false merge
    assert fingerprint("en:vegetarian, en:non-vegetarian") == fingerprint("en:non-vegetarian")
    assert items_key("en:vegetarian, en:non-vegetarian", fingerprint) != items_key(
        "en:non-vegetarian", fingerprint
    )


def labels(rng: random.Random) -> list[dict[str, Any]]:
    """Open Food Facts' tags: lists of items that come back from one product to the next."""
    tags = ["en:vegan", "en:vegetarian", "en:non-vegetarian", "en:palm-oil-free", "en:organic"]
    tags += ["en:no-gluten", "en:fair-trade", "en:no-gmos", "en:green-dot", "en:kosher"]
    rows = []
    for index in range(1_500):
        chosen = rng.sample(tags, rng.randint(1, 3))
        rows.append({"labels_tags": ",".join(chosen), "n": index})
    spellings = [
        ("en:palm-oil-free,en:vegetarian", 300),
        ("en:palm-oil-free,en:non-vegetarian", 12),
        ("Snacks, Biscuits and cakes, Biscuits", 300),
        ("Snacks, Biscuits and cakes, Cakes", 14),
        ("No gluten, No GMOs", 200),
        ("No GMOs, No gluten", 9),  # the same items in another order: merged
        ("No GMOs,no  gluten", 3),
    ]
    rows += [{"labels_tags": v, "n": i} for v, c in spellings for i in range(c)]
    rng.shuffle(rows)
    return rows


def test_lists_merge_only_the_same_items(tmp_path: Path) -> None:
    found = advanced(write(tmp_path, labels(random.Random(11)), "labels"))
    changes = merged(found)
    assert "en:palm-oil-free,en:non-vegetarian" not in changes
    assert "Snacks, Biscuits and cakes, Cakes" not in changes
    assert changes["No GMOs, No gluten"] == "No gluten, No GMOs"
    assert changes["No GMOs,no  gluten"] == "No gluten, No GMOs"
    reason = found["A-01:labels_tags"].reason
    assert "a column of lists, compared item by item" in reason


def test_a_free_text_with_commas_is_not_a_list(tmp_path: Path) -> None:
    """Items that never come back: one value, fingerprinted as a whole (A-01 as before)."""
    rows = table([("Smith, John", 200), ("John Smith", 6), ("Doe, Jane", 150)], "customer")
    rng = random.Random(12)
    words = ["".join(rng.choice("abcdefghijklmnop") for _ in range(7)) for _ in range(80)]
    unique = [(f"{words[i]}, {words[40 + i]}", 1) for i in range(40)]
    rows += table(unique, "customer")
    changes = merged(advanced(write(tmp_path, rows, "text")), "A-01")
    assert changes.get("John Smith") == "Smith, John"


# --- N14: A-02 on names of people ----------------------------------------------------------------


def test_ordered_pairs_keep_the_order_of_letters() -> None:
    assert ngram_fingerprint("VAN DAM ST", ordered=True) == ngram_fingerprint(
        "VANDAM ST", ordered=True
    )
    assert ngram_fingerprint("Pierre-Philippe") == ngram_fingerprint("Philippe Pierre")
    assert ngram_fingerprint("Pierre-Philippe", ordered=True) != ngram_fingerprint(
        "Philippe Pierre", ordered=True
    )


def test_no_two_letter_merge_on_names_of_people(tmp_path: Path) -> None:
    """fr_rne_maires: 'LE ROUX' and 'LEROUX' are two families; the first-name hyphen still merges
    (A-01, order kept)."""
    people = [("LEROUX", 300), ("LE ROUX", 40), ("DUPONT", 300), ("DU PONT", 12)]
    found = advanced(write(tmp_path, table(people, "nom"), "noms"))
    assert not any(s.rule in ("A-02", "A-03") for s in found.values())
    first = [("Jean-Pierre", 300), ("Jean Pierre", 9), ("Pierre-Philippe", 200)]
    first += [("Philippe Pierre", 7), ("Marie", 300)]
    found = advanced(write(tmp_path, table(first, "prenom"), "prenoms"))
    assert merged(found) == {"Jean Pierre": "Jean-Pierre"}


def test_two_letter_merge_kept_on_other_columns(tmp_path: Path) -> None:
    """nyc_collisions: 'VANDAM ST' is 'VAN DAM ST'."""
    streets = [("VAN DAM ST", 200), ("VANDAM ST", 3), ("BROADWAY", 300), ("QUEENS BLVD", 150)]
    found = advanced(write(tmp_path, table(streets, "on_street_name"), "streets"))
    assert merged(found, "A-02") == {"VANDAM ST": "VAN DAM ST"}


# --- N15: A-10, repeated exceptions --------------------------------------------------------------


def boroughs(rng: random.Random) -> list[dict[str, Any]]:
    """ZIP → borough with a neighbourhood column: Marble Hill (ZIP 10463, in the Bronx's area but
    in Manhattan) and a typo in the borough of one row."""
    zips = {f"104{i:02d}": "BRONX" for i in range(60, 75)}
    zips |= {f"100{i:02d}": "MANHATTAN" for i in range(1, 15)}
    zips |= {f"113{i:02d}": "QUEENS" for i in range(54, 70)}
    rows = []
    for index in range(3_000):
        zip_code = rng.choice(sorted(zips))
        area = f"Area {zip_code}"
        rows.append({"zip_code": zip_code, "borough": zips[zip_code], "nta": area, "n": index})
    marble = [r for r in rows if r["zip_code"] == "10463"][:3]
    for row in marble:
        row.update(borough="MANHATTAN", nta="Marble Hill-Inwood")
    rows += [
        {"zip_code": "10001", "borough": "MANHATTAN", "nta": "Marble Hill-Inwood", "n": 9_000 + i}
        for i in range(30)
    ]
    rows[5]["borough"] = "QUEENS" if rows[5]["borough"] != "QUEENS" else "BRONX"  # a typo
    return rows


def test_dependency_leaves_a_consistent_exception_alone(tmp_path: Path) -> None:
    """nyc_dob: the Marble Hill rows have the neighbourhood of Manhattan too: kept, listed, the step
    only fixes the typo."""
    rows = boroughs(random.Random(21))
    typo = rows[5]
    found = advanced(write(tmp_path, rows, "boroughs"))
    suggestion = found["A-10:zip_code+borough"]
    assert suggestion.mode == "S" and suggestion.cells == 1
    assert "look like real exceptions" in suggestion.reason and "Not changed." in suggestion.reason
    assert "'10463' has borough 'MANHATTAN'" in suggestion.reason
    cases = suggestion.step.cases  # type: ignore[union-attr]
    assert [c.conditions[0].values for c in cases] == [[typo["zip_code"]]]
    kept = [e for e in suggestion.examples if (e.after or "").endswith("(kept)")]
    assert len(kept) == 3


def test_dependency_with_only_exceptions_is_information(tmp_path: Path) -> None:
    """Postcode 3168 is Notting Hill (Southern) and Clayton (South-Eastern): the suburb explains
    the second region; nothing to change, an information without a step."""
    rng = random.Random(22)
    regions = {f"31{i:02d}": rng.choice(["Southern", "Eastern", "Northern"]) for i in range(40)}
    rows = []
    for index in range(2_000):
        postcode = rng.choice(sorted(regions))
        rows.append(
            {
                "postcode": postcode,
                "region": regions[postcode],
                "suburb": f"Suburb {rng.choice('ABCDEFGH')}",  # shared across regions
                "n": index,
            }
        )
    rows += [{"postcode": "3168", "region": "South-Eastern", "suburb": "Clayton", "n": 0}] * 40
    rows += [{"postcode": "3168", "region": "Southern", "suburb": "Notting Hill", "n": 0}] * 3
    found = advanced(write(tmp_path, rows, "suburbs"))
    suggestion = found["A-10:postcode+region"]
    assert suggestion.mode == "I" and suggestion.step is None
    assert "on all its 3 rows with suburb 'Notting Hill'" in suggestion.reason
    assert "Information only" in suggestion.reason


def test_dependency_does_not_correct_a_value_without_a_clear_majority(tmp_path: Path) -> None:
    """nyc_dob: ZIP 07645 NY on one row, NJ on the other: no majority to correct toward."""
    rng = random.Random(23)
    states = {f"1{i:04d}": "NY" for i in range(30)} | {f"0{i:04d}": "NJ" for i in range(15)}
    rows = []
    for index in range(1_800):
        zip_code = rng.choice(sorted(states))
        rows.append({"owner_zip": zip_code, "state": states[zip_code], "n": index})
    rows += [{"owner_zip": "07645", "state": "NY", "n": 1}, {"owner_zip": "07645", "state": "NJ"}]
    rows[3]["state"] = "PA"  # a typo where the majority is clear
    found = advanced(write(tmp_path, rows, "owners"))
    suggestion = found["A-10:owner_zip+state"]
    cases = suggestion.step.cases  # type: ignore[union-attr]
    assert [c.conditions[0].values for c in cases] == [[rows[3]["owner_zip"]]]
    assert "'07645' has no clear state (1 of its 2 rows agree)" in suggestion.reason


def test_first_names_accept_a_second_sex_from_two_percent(tmp_path: Path) -> None:
    """fr_rne_maires: 'Claude' is a woman on 4 % of its rows: not an error (C-27 from 2 % on a
    first name); a 1-row slip on another name is still listed, never changed: no step corrects a
    person's sex from their first name."""
    rng = random.Random(24)
    names = {f"Name{i:02d}": "MF"[i % 2] for i in range(30)}
    rows = [
        {"prenom": k, "sexe": v, "n": rng.random()} for k, v in names.items() for _ in range(60)
    ]
    rows += [{"prenom": "Claude", "sexe": "M", "n": 0}] * 144
    rows += [{"prenom": "Claude", "sexe": "F", "n": 0}] * 6
    rows[0]["sexe"] = "F" if rows[0]["sexe"] == "M" else "M"
    found = advanced(write(tmp_path, rows, "maires"))
    suggestion = found["A-10:prenom+sexe"]
    assert suggestion.cells == 1 and suggestion.mode == "I" and suggestion.step is None
    assert all("Claude" not in (e.before or "") for e in suggestion.examples)
    assert "no step changes a person's sexe from their name" in suggestion.reason


def test_dependency_exception_explained_by_any_text_column(tmp_path: Path) -> None:
    """nyc_dob: the three Marble Hill permits share the tax block 02215, a column too fine to be a
    dependency column (a block per few rows): read on the rows of ZIP 10463 only."""
    rng = random.Random(25)
    zips = {f"104{i:02d}": "BRONX" for i in range(60, 75)}
    zips |= {f"100{i:02d}": "MANHATTAN" for i in range(1, 15)}
    rows = []
    for index in range(2_400):
        zip_code = rng.choice(sorted(zips))
        block = f"B{rng.randrange(1_500):05d}"
        rows.append({"zip_code": zip_code, "borough": zips[zip_code], "block": block, "n": index})
    marble = [r for r in rows if r["zip_code"] == "10463"][:3]
    for row in marble:
        row.update(borough="MANHATTAN", block="B02215")
    found = advanced(write(tmp_path, rows, "blocks"))
    suggestion = found["A-10:zip_code+borough"]
    assert suggestion.mode == "I" and suggestion.step is None
    assert "on all its 3 rows with block 'B02215'" in suggestion.reason


def test_brands_are_lists_when_their_items_stand_alone(tmp_path: Path) -> None:
    """Open Food Facts' brands: 'The Kroger Co.' is not 'Kroger, The Kroger Co.' (another list),
    though one fingerprint of the whole value says so; most rows hold one brand only."""
    rng = random.Random(26)
    words = ["".join(rng.choice("bcdfghjklmnpqrstvwxz") for _ in range(6)) for _ in range(40)]
    spellings = [(f"{w.title()} Foods", 30) for w in words]
    spellings += [(f"{w.title()}, {w.title()} Foods", 3) for w in words[:12]]
    spellings += [("The Kroger Co.", 80), ("Kroger, The Kroger Co.", 120), ("Kroger", 40)]
    spellings += [("Kroger,The Kroger Co.", 5)]
    found = advanced(write(tmp_path, table(spellings, "brands"), "brands"))
    changes = merged(found, "A-01")
    assert "The Kroger Co." not in changes and "Kroger" not in changes
    assert changes["Kroger,The Kroger Co."] == "Kroger, The Kroger Co."


def test_a_legal_suffix_does_not_make_a_list(tmp_path: Path) -> None:
    """brand_owner: 'Pepperidge Farm, Inc.' is one company; 'Inc.' never stands alone."""
    rng = random.Random(27)
    words = ["".join(rng.choice("bcdfghjklmnpqrstvwxz") for _ in range(6)) for _ in range(30)]
    spellings = [(f"{w.title()} Farm Inc", 20) for w in words]
    spellings += [(f"{w.title()} Mills, Inc.", 2) for w in words[:8]]
    spellings += [("PEPPERIDGE FARM INC", 60), ("Pepperidge Farm, Inc.", 4)]
    changes = merged(advanced(write(tmp_path, table(spellings, "brand_owner"), "owners")), "A-01")
    assert changes.get("Pepperidge Farm, Inc.") == "PEPPERIDGE FARM INC"


def test_dependency_leaves_another_naming_alone(tmp_path: Path) -> None:
    """nyc_311: city 'MANHATTAN' on a few rows of ZIP codes whose other rows write 'NEW YORK', and
    borough 'MANHATTAN' on the same rows: another naming, not a typo; the 'BROOKLYN' typo of a
    Manhattan row keeps its step, though a copy of the city column says 'BROOKLYN' too."""
    rng = random.Random(28)
    zips = [f"100{i:02d}" for i in range(1, 25)]
    rows = [
        {"incident_zip": rng.choice(zips), "city": "NEW YORK", "borough": "MANHATTAN", "n": i}
        for i in range(2_400)
    ]
    for index, zip_code in enumerate(zips[:6]):
        rows.append(
            {"incident_zip": zip_code, "city": "MANHATTAN", "borough": "MANHATTAN", "n": index}
        )
    rows[7]["city"] = "BROOKLYN"
    rows += [
        {"incident_zip": f"112{i:02d}", "city": "BROOKLYN", "borough": "BROOKLYN", "n": i}
        for i in range(20)
    ] * 6
    for row in rows:  # a copy of the city proves nothing (311's park_borough)
        row["mail_city"] = row["city"]
    found = advanced(write(tmp_path, rows, "naming"))
    suggestion = found["A-10:incident_zip+city"]
    assert suggestion.mode == "S" and suggestion.cells == 1
    assert "has city 'MANHATTAN', the same as its borough" in suggestion.reason
    cases = suggestion.step.cases  # type: ignore[union-attr]
    assert [c.conditions[1].values for c in cases] == [["BROOKLYN"]]
