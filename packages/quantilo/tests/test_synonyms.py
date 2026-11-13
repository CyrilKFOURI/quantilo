"""N30 (audit 2026-11-12, § 4.3): business synonyms, for the roles (R-SYN-01) and Clean (A-21)."""

from pathlib import Path

import pandas as pd
import pytest

import quantilo as dx
from quantilo.cleaning import catalogue
from quantilo.cleaning import phonetic as ph
from quantilo.cleaning.rules import advanced
from quantilo.cleaning.rules.synonyms import _present
from quantilo.typing.knowledge_base import load
from quantilo.typing.knowledge_base.synonyms import load_synonyms


@pytest.mark.parametrize(
    ("name", "role"),
    [
        ("CA HT", "amount"),
        ("CA_2023", "amount"),
        ("chiffre d'affaires", "amount"),
        ("Chiffre_Affaires_Net", "amount"),
        ("Qté", "quantity"),
        ("qte_vendue", "quantity"),
        ("PU TTC", "unit_price"),
        ("Mnt HT", "amount"),
        ("TTC", "amount"),
        ("amt", "amount"),
        ("nbre_clients", "quantity"),
        # the short words alone, or beside a word that says something else, stay what they were
        ("ca", None),
        ("serum_ca", None),
        ("ca_code", None),
        ("pu", None),
    ],
)
def test_names(name: str, role: str | None) -> None:
    found = load().match_role(name)
    assert (found.id if found else None) == role


def test_the_reason_is_shown() -> None:
    frame = pd.DataFrame(
        {
            "Qté": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 5,
            "CA HT": [float(i) * 10.5 for i in range(50)],
            "ca": [0, 1, 2, 3, 0] * 10,
        }
    )
    types = {t.name: t for t in dx.from_pandas(frame).types}
    assert types["Qté"].role == "quantity"
    assert "'Qté' is short for quantité (quantity) (R-SYN-01)" in types["Qté"].reasons
    assert types["CA HT"].role == "amount"
    assert any("chiffre d'affaires" in reason for reason in types["CA HT"].reasons)
    assert types["ca"].role is None
    assert not any("R-SYN-01" in reason for reason in types["ca"].reasons)


def test_the_lists_load() -> None:
    synonyms = load_synonyms()
    assert {domain.id for domain in synonyms.values} >= {"film_rating", "sex", "payment"}
    for domain in synonyms.values:
        assert domain.reason and domain.label and domain.groups
        folded = [ph.fold(member.value) for group in domain.groups for member in group]
        # a spelling belongs to one group of its domain only
        assert len(folded) == len(set(folded)), domain.id


def _groups(domain_id: str, *values: str) -> list[list[str]]:
    domain = next(d for d in load_synonyms().values if d.id == domain_id)
    keys = {ph.fold(value): index for index, value in enumerate(values)}
    return [[values[i] for i in members] for members in _present(domain, keys)]


def test_values_need_their_context() -> None:
    assert _groups("film_rating", "R", "PG-13", "Not Rated", "Unrated") == [
        ["Not Rated", "Unrated"]
    ]
    assert _groups("film_rating", "Not Rated", "NR") == []  # no rating beside them
    assert _groups("sex", "M", "F", "Male", "Female") == [["Male", "M"], ["Female", "F"]]
    assert _groups("sex", "H", "M") == []  # Spanish: 'M' is mujer, not male
    assert _groups("sex", "Male", "Female") == []
    assert _groups("payment", "CB", "Carte bancaire", "Espèces", "Chèque") == [
        ["Carte bancaire", "CB"]
    ]


def test_a21_is_a_decision(tmp_path: Path) -> None:
    rows = ["title,content_rating"] + [f"m{i},{r}" for i, r in enumerate(["R", "PG"] * 20)]
    rows += [f"n{i},Not Rated" for i in range(6)] + [f"u{i},Unrated" for i in range(3)]
    path = tmp_path / "ratings.csv"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    found = [
        s for s in dx.open(path, cache_dir=tmp_path / "c").clean().advanced() if s.rule == "A-21"
    ]
    assert len(found) == 1
    merge = found[0]
    assert merge.mode == "S" and merge.risk == "review" and not merge.auto
    assert merge.step is not None and merge.step.map == {"Unrated": "Not Rated"}  # type: ignore[union-attr]
    assert merge.clusters and merge.clusters[0].target == "Not Rated"
    assert "the same rating under two names" in merge.reason
    assert catalogue.classify(merge) == "A-21"
    assert catalogue.CATALOGUE["A-21"].mode == "decision"
    assert "A-21" in advanced.ORDER
