"""M02: reasons keep their English text and carry a template code and vars for the interface."""

import copy
import json
import pickle
from pathlib import Path

import pytest

import quantilo as dx
from quantilo.catalog import ColumnRecord, to_view
from quantilo.cleaning.suggestion import CleanSuggestion
from quantilo.reasons import Reason, ReasonText, details, restored, say, say_count, stored

SAMPLES = Path(__file__).resolve().parents[3] / "samples"


class TestSay:
    def test_the_text_is_the_template_filled(self) -> None:
        said = say(
            "TYP.distinct", "{count} distinct values, e.g. {examples}", count=3, examples="a"
        )
        assert said == "3 distinct values, e.g. a"
        assert (said.code, said.vars) == ("TYP.distinct", {"count": "3", "examples": "a"})

    @pytest.mark.parametrize("values", [{}, {"count": 1, "other": 2}])
    def test_the_vars_are_exactly_the_placeholders(self, values: dict[str, int]) -> None:
        with pytest.raises(ValueError):
            say("X", "{count} rows", **values)

    def test_a_count_of_one_says_the_singular(self) -> None:
        one = say_count("B-15", 1, "{count} row has no value.", "{count} rows have no value.")
        many = say_count("B-15", 1234, "{count} row has no value.", "{count} rows have no value.")
        assert (one, one.vars) == ("1 row has no value.", {"count": "1"})
        assert (many, many.vars) == ("1,234 rows have no value.", {"count": "1,234"})

    def test_parts_said_one_after_the_other_keep_their_codes(self) -> None:
        said = say("A", "{n} rows", n=2) + say("SEP", "; ") + say("B", "{n} dates", n=3)
        assert said == "2 rows; 3 dates"
        assert isinstance(said, Reason)
        # the third part cites {n} with another value: it reads n@2
        assert (said.code, said.vars) == ("A+SEP+B", {"n": "2", "n@2": "3"})

    def test_a_plain_text_added_loses_the_code(self) -> None:
        said = say("A", "{n} rows", n=2) + " more"
        assert said == "2 rows more" and not isinstance(said, Reason)
        assert details([said]) == [ReasonText(text="2 rows more")]

    def test_copies_and_pickles_keep_the_code(self) -> None:
        said = say("A", "{n} rows", n=2)
        # a reason the test itself pickles: nothing untrusted is loaded
        for again in (copy.deepcopy(said), pickle.loads(pickle.dumps(said))):
            assert (again, again.code, again.vars) == (said, "A", {"n": "2"})


class TestStored:
    def test_round_trip_and_older_records(self) -> None:
        reasons = [say("TYP.name", "name suggests {role}", role="amount"), "a plain reason"]
        kept = json.loads(json.dumps(stored(reasons)))
        assert kept[1] == "a plain reason"
        back = restored(kept)
        assert back == reasons and getattr(back[0], "code", None) == "TYP.name"
        record = ColumnRecord(
            "d", "x", 0, "VARCHAR", "text", "{}", "{}", "high", json.dumps(["old text"])
        )
        assert to_view(record).reasons == ["old text"]


def test_a_suggestion_takes_its_code_from_its_reason() -> None:
    reason = say("B-19", "Always {value}: no information.", value="'x'")
    found = CleanSuggestion(id="B-19:x", rule="B-19", mode="S", title="t", reason=reason)
    assert (found.reason, found.reason_code, found.reason_vars) == (
        reason,
        "B-19",
        {"value": "'x'"},
    )
    again = CleanSuggestion.model_validate_json(found.model_dump_json())
    assert (again.reason_code, again.reason_vars) == ("B-19", {"value": "'x'"})
    plain = CleanSuggestion(id="B-19:y", rule="B-19", mode="S", title="t", reason="text")
    assert plain.reason_code is None


def test_the_reasons_shown_on_dirty_traps_have_their_codes(tmp_path: Path) -> None:
    """Every detection reason of dirty_traps is coded; so are the Basic suggestions' and the
    KPI's; each coded one is its parts filled with its vars (M02)."""
    ds = dx.open(SAMPLES / "dirty_traps.csv", cache_dir=tmp_path)
    shown = [r for column in ds.types for r in column.reasons]
    assert shown and all(isinstance(r, Reason) and r.code for r in shown)
    again = dx.open(SAMPLES / "dirty_traps.csv", cache_dir=tmp_path)  # read back from the cache
    codes = [getattr(r, "code", None) for c in again.types for r in c.reasons]
    assert codes == [getattr(r, "code", None) for r in shown]
    suggestions = ds.clean().suggestions
    coded = [s for s in suggestions if s.reason_code]
    assert len(coded) >= 0.8 * len(suggestions)
    kpis = ds.kpi_suggestions().kpis
    assert all(d.code for k in kpis for d in k.reason_details)
    result = ds.kpi(kpis[0].definition)
    assert [d.text for d in result.warning_details] == result.warnings
