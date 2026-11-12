"""Clean classified by effect (096): every suggestion says how it changes the dataset (`impact`) and
whether it needs a decision (`risk`); only safe ones are ticked by default."""

from pathlib import Path
from typing import get_args

import pytest

import quantilo as dx
from quantilo.cleaning import model as m
from quantilo.cleaning.suggest import default_steps
from quantilo.cleaning.suggestion import CleanSuggestion

SAMPLES = Path(__file__).resolve().parents[3] / "samples"

# The catalogue of spec 096 on the trap file: (impact, risk) of each expected suggestion.
CATALOGUE = {
    "B-15:empty": ("rows", "safe"),
    "B-02:product": ("format", "safe"),
    "B-01:description": ("format", "safe"),
    "B-01:status": ("format", "safe"),
    "B-05:brand": ("format", "safe"),
    "B-06:title": ("format", "safe"),
    "B-04:status": ("values", "safe"),
    "B-04:region:maybe": ("values", "review"),
    "B-03:city": ("values", "safe"),
    "B-12:customer_id": ("types", "safe"),
    "B-13:zip_code": ("values", "safe"),
    "B-07:price": ("types", "safe"),
    "B-07:weight": ("types", "review"),
    "B-11:signup_date": ("types", "safe"),
    "B-18:is_member": ("types", "safe"),
    "B-09:income:-999": ("values", "review"),
    "B-16:end_date:9999-12-31": ("values", "safe"),
    "B-10:age": ("values", "review"),
    "B-10:quantity": ("rows", "review"),
    "B-14:rows": ("rows", "safe"),
    "B-19:country": ("columns", "review"),
    "B-19:empty_col": ("columns", "review"),
}


@pytest.fixture(scope="module")
def traps(tmp_path_factory: pytest.TempPathFactory) -> dx.Dataset:
    return dx.open(SAMPLES / "dirty_traps.csv", cache_dir=tmp_path_factory.mktemp("traps096"))


def test_the_shared_enums() -> None:
    assert get_args(m.Impact) == ("format", "values", "types", "rows", "columns")
    assert get_args(m.Risk) == ("safe", "review")
    assert get_args(m.Impact) == m.IMPACTS
    assert set(m.OP_IMPACT) == set(m.OPS)  # every op has a default effect


def test_trap_file_effect_and_risk_of_every_suggestion(traps: dx.Dataset) -> None:
    found = {s.id: (s.impact, s.risk) for s in traps.clean().suggestions}
    assert found == CATALOGUE


def test_risk_follows_the_mode(traps: dx.Dataset) -> None:
    for suggestion in traps.clean().suggestions:
        assert suggestion.risk == ("safe" if suggestion.mode == "A" else "review"), suggestion.id


def test_defaults_from_the_step_and_the_mode() -> None:
    """A rule of another layer that leaves them out (CL-03)."""
    drop = m.DropDuplicates(id="s1")
    made = CleanSuggestion(id="X:rows", rule="X", mode="A", title="t", reason="r", step=drop)
    assert (made.impact, made.risk) == ("rows", "safe")
    asked = CleanSuggestion(
        id="X:c",
        rule="X",
        mode="S",
        title="t",
        reason="r",
        step=m.ChangeType(id="s2", column="c", to="date"),
    )
    assert (asked.impact, asked.risk) == ("types", "review")
    info = CleanSuggestion(id="X:i", rule="X", mode="I", title="t", reason="r")
    assert (info.impact, info.risk) == ("values", "review")
    told = CleanSuggestion(
        id="X:f", rule="X", mode="S", title="t", reason="r", impact="format", risk="safe", step=drop
    )
    assert (told.impact, told.risk) == ("format", "safe")  # given values are kept
    assert CleanSuggestion.model_validate(told.model_dump()).impact == "format"


def test_a_suggestion_to_review_is_never_ticked_by_default() -> None:
    step = m.DropColumns(id="s1", columns=["a"])
    risky = CleanSuggestion(
        id="X:a", rule="X", mode="A", risk="review", title="t", reason="r", step=step
    )
    other = m.DropColumns(id="s2", columns=["b"])
    safe = CleanSuggestion(id="X:b", rule="X", mode="A", title="t", reason="r", step=other)
    assert [s.id for s in default_steps([risky, safe])] == ["s2"]
