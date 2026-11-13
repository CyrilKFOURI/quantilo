"""N28 (audit 2026-11-12, § 4.3): wider encodings, at the import (R-ING-08) and in B-05."""

import codecs
from pathlib import Path

import pytest

import quantilo as dx
from quantilo.cleaning import catalogue
from quantilo.cleaning import model as m
from quantilo.cleaning import text as tx
from quantilo.cleaning.suggestion import CleanSuggestion
from quantilo.ingestion import encodings
from quantilo.ingestion.csv_sniff import readable_path, sniff_dialect

TEXT = (
    "ville;prix;remarque\n"
    "Orléans;12 €;très bien à côté\n"
    "Besançon;7 €;déjà vu, élève\n"
    "Nîmes;3 €;fenêtre, garçon\n"
    "Sète;9 €;où est le cœur\n"
    "Évry;4 €;Été\n"
)
CITIES = ["Orléans", "Besançon", "Nîmes", "Sète", "Évry"]
PLAIN = TEXT.replace("€", "EUR").replace("œ", "oe")  # Latin-1 and CP850 have no € nor œ


@pytest.mark.parametrize(
    ("data", "expected", "reason"),
    [
        (TEXT.encode(), "utf-8", False),
        (codecs.BOM_UTF8 + TEXT.encode(), "utf-8", False),  # DuckDB skips the mark itself
        (TEXT.encode("cp1252"), "windows-1252", False),
        (PLAIN.encode("latin-1"), "latin-1", False),
        (TEXT.encode("iso8859_15"), "iso-8859-15", True),
        (TEXT.encode("mac_roman"), "mac-roman", True),
        (PLAIN.encode("cp850"), "cp850", True),
        (TEXT.encode("utf-16-le"), "utf-16", True),  # no mark: one byte in two is 0
        (codecs.BOM_UTF16_LE + TEXT.encode("utf-16-le"), "utf-16", False),
        (codecs.BOM_UTF16_BE + TEXT.encode("utf-16-be"), "utf-16-be", True),
        (TEXT.encode("utf-16-be"), "utf-16-be", True),
        (TEXT.encode("utf-32"), "utf-32", True),
    ],
)
def test_the_encoding_and_its_evidence(
    tmp_path: Path, data: bytes, expected: str, reason: bool
) -> None:
    found = encodings.detect(data)
    assert found.encoding == expected
    assert (found.reason is not None) is reason
    if reason:
        assert "R-ING-08" in (found.reason or "")
    path = tmp_path / "villes.csv"
    path.write_bytes(data)
    options, notes = sniff_dialect(path)
    assert options.encoding == expected
    assert any("R-ING-08" in note for note in notes) is reason
    frame = dx.open(path, cache_dir=tmp_path / "cache").to_pandas()
    assert list(frame["ville"]) == CITIES


def test_mac_roman_says_why() -> None:
    reason = encodings.detect(TEXT.encode("mac_roman")).reason or ""
    assert "Mac Roman" in reason and "'Orléans'" in reason and "'OrlŽans' as Windows-1252" in reason


def test_a_no_break_space_keeps_windows_1252() -> None:
    """Superstore: 'Conference\\xa0phone' is a no-break space in Windows-1252, not CP850's 'á'."""
    raw = ("Product\n" + "Conference\xa0phone – “quoted”\n" * 40).encode("cp1252")  # noqa: RUF001
    assert encodings.detect(raw).encoding == "windows-1252"


def test_the_copy_follows_the_encoding(tmp_path: Path) -> None:
    path = tmp_path / "villes.csv"
    path.write_bytes(TEXT.encode("mac_roman"))
    mac, read_as = readable_path(path, "mac-roman")
    win, _ = readable_path(path, "windows-1252")
    assert read_as == "utf-8" and mac != win  # one copy per encoding chosen
    assert "Orléans" in mac.read_text(encoding="utf-8")
    assert readable_path(path, "latin-1") == (path, "latin-1")  # DuckDB reads it itself


@pytest.mark.parametrize(
    ("value", "repaired", "way"),
    [
        ("CafÃ©", "Café", "windows-1252"),
        ("CafÃ\u0083Â©", "Café", "double"),  # the audit's 'Ã\u0083Â©' (Latin-1)
        ("CafÃƒÂ©", "Café", "double"),
        ("CafÃ© â\u0080\u0099", "Café ’", "latin-1"),  # noqa: RUF001
        ("Caf√©", "Café", "mac-roman"),
        ("l‚Äôété".replace("été", "√©t√©"), "l’été", "mac-roman"),  # noqa: RUF001
    ],
)
def test_repairs(value: str, repaired: str, way: str) -> None:
    assert tx.repair_encoding(value) == (repaired, way)
    assert tx.fix_encoding(value) == repaired


@pytest.mark.parametrize("value", ["Café", "√2 = 1.41", "Ã la carte", "¬ 3", "Prix: 3 €"])
def test_no_repair(value: str) -> None:
    assert tx.repair_encoding(value) is None


def _b05(mapping: dict[str, str] | None) -> CleanSuggestion:
    step = None if mapping is None else m.MapValues(id="s1", column="c", map=mapping)  # type: ignore[arg-type]
    return CleanSuggestion(
        id="B-05:c" + ("" if step else ":lost"),
        rule="B-05",
        mode="A" if step else "I",
        title="t",
        reason="r",
        columns=["c"],
        step=step,
    )


def test_catalogue_of_b05() -> None:
    assert catalogue.classify(_b05({"CafÃ©": "Café"})) == "B-05"
    assert catalogue.classify(_b05({"CafÃ©": "Café", "CafÃƒÂ©": "Café"})) == "B-05:wide"
    assert catalogue.classify(_b05({"Caf√©": "Café"})) == "B-05:wide"
    assert catalogue.classify(_b05(None)) == "B-05:lost"
    assert catalogue.CATALOGUE["B-05"].automatic
    assert catalogue.CATALOGUE["B-05:wide"].mode == "suggested"
    assert catalogue.CATALOGUE["B-05:lost"].mode == "information"


def test_b05_in_a_dataset(tmp_path: Path) -> None:
    twice = "Café".encode().decode("cp1252").encode().decode("cp1252")
    rows = ["name,comment"] + [f"{twice},Livr�"] * 3 + ["Caf√©,ok"] * 2 + ["Café,ok"] * 20
    path = tmp_path / "p.csv"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    found = {s.id: s for s in dx.open(path, cache_dir=tmp_path / "cache").clean().suggestions}
    b05 = found["B-05:name"]
    assert b05.catalogue == "B-05:wide" and not b05.auto
    assert b05.step is not None and b05.step.map == {twice: "Café", "Caf√©": "Café"}  # type: ignore[union-attr]
    assert "double encoding" in b05.reason and "Mac Roman" in b05.reason
    lost = found["B-05:comment:lost"]
    assert lost.mode == "I" and lost.step is None and lost.cells == 3
    assert lost.reason.startswith("3 values hold '�'")
