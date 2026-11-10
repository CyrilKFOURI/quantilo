"""093: fixes from the multi-domain verification (cleaning, KPI, time series, tails, overview).

Each test reproduces one corpus defect on a small synthetic file (no corpus file is copied)."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import quantilo as dx
from quantilo import semantics
from quantilo.kpi.suggest import Facts, notes, suggest
from quantilo.variable.numeric import describe
from quantilo.variable.tails import analyze_tail, tail_rule

from .test_kpi_measures import Col, summary
from .test_variable import source_of


def open_frame(tmp_path: Path, frame: pd.DataFrame, name: str = "data") -> dx.Dataset:
    path = tmp_path / f"{name}.csv"
    frame.to_csv(path, index=False)
    return dx.open(path, cache_dir=tmp_path / "cache")


# --- Semantics (S-01 to S-06) ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "kind"),
    [
        ("patient_nbr", "identifier"),
        ("permit_si_no", "identifier"),
        ("bin__", "identifier"),
        ("BIN #", "identifier"),
        ("unique_key", "identifier"),
        ("Code du département", "code"),
        ("adresse_numero", "identifier"),
        ("number_of_persons_injured", None),
        ("nb_lots", None),
        ("num_critic_for_reviews", None),
        ("total_ghg", None),
    ],
)
def test_code_by_name(name: str, kind: str | None) -> None:
    code = semantics.code_by_name(name)
    assert (code.kind if code else None) == kind


def test_code_by_values_fixed_length_whole_numbers() -> None:
    assert semantics.code_by_values(1_000_005, 5_999_999, True) is not None
    assert semantics.code_by_values(1_000, 9_999, True) is None  # 4 digits: a year, a price
    assert semantics.code_by_values(12_000, 450_000, True) is None
    assert semantics.code_by_values(1_000_005.5, 5_999_999, False) is None


def test_money_coordinates_ratios_and_dates() -> None:
    assert semantics.is_money("Total") and semantics.is_money("sales_total")
    assert not semantics.is_money("Total Bsmt SF") and not semantics.is_money("total_ghg")
    assert semantics.is_coordinate("x_coordinate_state_plane")
    assert semantics.is_coordinate("gis_latitude")
    assert semantics.is_ratio("co2_including_luc_per_unit_energy")
    assert semantics.date_tier("Yr Sold") < semantics.date_tier("Year Built")
    assert semantics.date_tier("Date de début du mandat") < semantics.date_tier("Date de naissance")


# --- VAR-1: moments that overflow ------------------------------------------------------------


def absurd(size: int = 2_000) -> list[float]:
    values = list(np.random.default_rng(1).lognormal(10, 1, size))
    values[5] = 1e150  # a typo in a survey answer
    values[9] = 3e160
    return [float(v) for v in values]


def test_describe_survives_values_whose_cube_overflows() -> None:
    summary = describe(source_of(absurd()))
    assert summary.skewness is not None and summary.skewness > 10
    assert summary.kurtosis is not None


def test_tail_of_a_column_with_absurd_values_does_not_crash() -> None:
    source = source_of(absurd())
    summary = describe(source)
    tail = analyze_tail(source, summary, tail_rule(source, summary))
    assert tail.rule.reason


def test_overview_survives_skewness_out_of_range(tmp_path: Path) -> None:
    frame = pd.DataFrame({"comp": absurd(), "age": np.arange(2_000) % 60 + 18.5})
    ds = open_frame(tmp_path, frame)
    overview = ds.overview()
    assert overview.column_stats["comp"].skewness is not None
    assert "comp" in next(a for a in overview.alerts if a.kind == "skewed").columns


# --- VAR-2: no tail on a code, an identifier or a coordinate ---------------------------------


def heavy(size: int = 3_000) -> np.ndarray:
    return np.random.default_rng(3).pareto(1.5, size) + 1


def test_tail_rule_withholds_a_code_by_name() -> None:
    source = source_of([float(v) for v in heavy() * 1_000], name="block_code")
    rule = tail_rule(source, describe(source))
    assert not rule.fires and rule.code == "R-TAIL-8"
    assert "code" in rule.reason


def test_tail_rule_withholds_fixed_length_codes() -> None:
    rng = np.random.default_rng(2)
    values = np.concatenate([rng.integers(1_000_000, 1_100_000, 2_900), [5_800_000] * 100])
    source = source_of([float(v) for v in values], name="building")
    rule = tail_rule(source, describe(source))
    assert not rule.fires and rule.code == "R-TAIL-8"


def test_tail_rule_withholds_a_coordinate() -> None:
    values = list(np.random.default_rng(4).normal(40.7, 0.05, 3_000))
    values[:60] = [0.0] * 60
    source = source_of([abs(float(v)) for v in values], name="latitude")
    rule = tail_rule(source, describe(source))
    assert not rule.fires and "coordinate" in rule.reason


def test_tail_rule_withholds_few_whole_levels() -> None:
    rng = np.random.default_rng(5)
    values = rng.choice([1, 2, 3, 19, 37, 38, 44, 150, 190], 3_000, p=[0.5, *[0.0625] * 8])
    source = source_of([float(v) for v in values], name="occupation")
    rule = tail_rule(source, describe(source))
    assert not rule.fires and "distinct values" in rule.reason


def test_tail_rule_still_fires_on_a_heavy_measure() -> None:
    source = source_of([float(v) for v in heavy()], name="claim_amount")
    rule = tail_rule(source, describe(source))
    assert rule.fires and rule.code is None


# --- Cleaning (CLN-1 to CLN-9) ---------------------------------------------------------------


def suggestions(ds: dx.Dataset) -> dict[str, Any]:
    return {s.id: s for s in ds.clean().suggestions}


def sensor_frame(rows: int = 1_000) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    co = rng.gamma(2.0, 1.0, rows).round(1)
    co[rng.random(rows) < 0.18] = -200  # the documented missing code of the device
    temperature = rng.normal(18, 3, rows).round(2)
    temperature[rng.random(rows) < 0.07] = -100000
    return pd.DataFrame({"co_gt": co, "water_temperature": temperature})


def test_b09_finds_a_round_isolated_negative_peak(tmp_path: Path) -> None:
    found = suggestions(open_frame(tmp_path, sensor_frame()))
    co = found["B-09:co_gt:-200"]
    assert co.mode == "S" and "-200" in co.reason and co.cells > 100
    assert "B-09:water_temperature:-100000" in found


def test_b09_leaves_a_real_extreme_alone(tmp_path: Path) -> None:
    rng = np.random.default_rng(12)
    values = rng.normal(100, 10, 1_000).round(1)
    values[:3] = 300  # a few real highs: rare, not a code
    found = suggestions(open_frame(tmp_path, pd.DataFrame({"weight": values})))
    assert not [k for k in found if k.startswith("B-09")]


def test_b09_reads_a_zero_zero_position_as_missing(tmp_path: Path) -> None:
    rng = np.random.default_rng(13)
    lat, lon = rng.normal(40.7, 0.05, 1_000), rng.normal(-73.9, 0.05, 1_000)
    lat[:30], lon[:30] = 0.0, 0.0
    found = suggestions(open_frame(tmp_path, pd.DataFrame({"latitude": lat, "longitude": lon})))
    for column in ("latitude", "longitude"):
        item = found[f"B-09:{column}:0,0"]
        assert item.mode == "S" and "(0, 0)" in item.reason and item.cells == 30


def test_b04_a_majority_na_in_few_levels_may_be_a_category(tmp_path: Path) -> None:
    pool = ["NA"] * 900 + ["Gd"] * 50 + ["Ex"] * 30 + ["Fa"] * 20
    note = ["NA"] * 20 + [f"note {i}" for i in range(980)]
    frame = pd.DataFrame({"pool_qc": pool, "remark": note})
    path = tmp_path / "pool.csv"
    frame.to_csv(path, index=False)  # NA written as text, as in the source file
    found = suggestions(dx.open(path, cache_dir=tmp_path / "cache"))
    assert "B-04:pool_qc" not in found
    assert found["B-04:pool_qc:maybe"].mode == "S"
    assert "most frequent" in found["B-04:pool_qc:maybe"].reason
    assert found["B-04:remark"].mode == "A"  # a minority NA among many values stays sure


def test_b04_weak_tokens_and_suppressed_numbers(tmp_path: Path) -> None:
    rng = np.random.default_rng(14)
    scores = [str(v) for v in rng.integers(300, 800, 800)] + ["s"] * 200
    kinds = ["Noise"] * 600 + ["Unspecified"] * 100 + ["Heat"] * 300
    found = suggestions(open_frame(tmp_path, pd.DataFrame({"score": scores, "kind": kinds})))
    assert found["B-04:kind:maybe"].mode == "S"
    item = found["B-04:score:maybe"]
    assert item.mode == "S" and "'s'" in item.title and "numbers" in item.reason


def test_b03_person_names_are_suggested_and_keep_the_accent(tmp_path: Path) -> None:
    names = ["MÜLLER"] * 3 + ["MULLER"] * 4 + ["DURAND"] * 50 + ["MARTIN"] * 50
    found = suggestions(open_frame(tmp_path, pd.DataFrame({"nom": names})))
    item = found["B-03:nom"]
    assert item.mode == "S"
    assert item.step.map == {"MULLER": "MÜLLER"}
    cities = ["Brooklyn"] * 30 + ["BROOKLYN"] * 5 + ["Queens"] * 40
    found = suggestions(open_frame(tmp_path, pd.DataFrame({"city": cities}), "cities"))
    assert found["B-03:city"].mode == "A"  # not names of people: still automatic


def test_b07_keeps_codes_with_leading_zeros_as_text(tmp_path: Path) -> None:
    rng = np.random.default_rng(15)
    block = [f"{v:05d}" for v in rng.integers(1, 9_000, 500)]
    found = suggestions(open_frame(tmp_path, pd.DataFrame({"block": block})))
    assert "B-07:block" not in found
    item = found["B-13:block"]
    assert item.mode == "S" and "leading" in item.reason.lower()


def test_b10_negative_values_without_refunds_outside_money(tmp_path: Path) -> None:
    rng = np.random.default_rng(16)
    ghg = rng.gamma(2, 50, 1_000)
    ghg[:20] = -rng.uniform(1, 30, 20)
    sales = rng.gamma(2, 50, 1_000)
    sales[:20] = -10
    frame = pd.DataFrame({"total_ghg": ghg, "sales": sales})
    found = suggestions(open_frame(tmp_path, frame))
    assert "Refunds" not in found["B-10:total_ghg"].reason
    assert "real" in found["B-10:total_ghg"].reason
    assert "Refunds" in found["B-10:sales"].reason


def test_b07_reads_less_than_and_more_than_as_bounds(tmp_path: Path) -> None:
    rng = np.random.default_rng(17)
    years = [str(v) for v in rng.integers(1, 50, 5_000)]
    years += ["Less than 1 year"] * 50 + ["More than 50 years"] * 30
    found = suggestions(open_frame(tmp_path, pd.DataFrame({"years_code": years})))
    item = found["B-07:years_code:bounds"]
    assert item.step.map == {"Less than 1 year": "0.5", "More than 50 years": "50"}
    assert item.mode == "S"


# --- KPI (KPI-1 to KPI-8) ----------------------------------------------------------------------


def names(columns: list[Col], facts: Facts | None = None) -> dict[str, Any]:
    return {s.name: s for s in suggest(columns, facts)}


def measured(columns: list[Col]) -> set[tuple[str, str | None]]:
    return {(m.agg, m.column) for s in suggest(columns) for m in s.definition.measures}


def test_kpi_never_sums_an_identifier_read_as_a_quantity() -> None:
    columns = [
        Col("patient_nbr", "numeric_continuous", "quantity", summary=summary(700)),
        Col("time_in_hospital", "numeric_discrete", "duration", summary=summary(14)),
    ]
    found = names(columns)
    assert "Total patient_nbr" not in found
    assert ("sum", "patient_nbr") not in measured(columns)
    assert found["Distinct patient_nbr"].rule == "R-SUG-04"
    said = notes(columns)
    assert any(n.startswith("R-SUG-25 · patient_nbr") and "'nbr'" in n for n in said)


def test_kpi_never_sums_a_ratio_and_total_is_not_money() -> None:
    columns = [
        Col("year", "datetime", summary=summary(200)),
        Col("total_ghg", "numeric_continuous", "amount", summary=summary(900)),
        Col(
            "co2_including_luc_per_unit_energy",
            "numeric_continuous",
            "quantity",
            summary=summary(900),
        ),
    ]
    found = names(columns)
    assert "Average unit price" not in found
    assert ("sum", "co2_including_luc_per_unit_energy") not in measured(columns)
    assert ("sum", "total_ghg") not in measured(columns)
    assert ("mean", "co2_including_luc_per_unit_energy") in measured(columns)
    said = " ".join(notes(columns))
    assert "R-SUG-26" in said and "R-SUG-27" in said


def test_kpi_no_basket_of_a_surface() -> None:
    columns = [
        Col("Order", "identifier", "order_key", summary=summary(1_000)),
        Col("Total Bsmt SF", "numeric_continuous", "amount", summary=summary(800)),
        Col("SalePrice", "numeric_continuous", "amount", summary=summary(900)),
    ]
    found = names(columns)
    assert "Average basket" not in found and "Total Total Bsmt SF" not in found
    assert found["Average SalePrice"].rule == "R-SUG-09"


def test_kpi_measures_without_a_role_are_averaged() -> None:
    columns = [
        Col("water_temperature", "numeric_continuous", summary=summary(900)),
        Col("station_code", "numeric_continuous", summary=summary(30)),
        Col("latitude", "numeric_continuous", "latitude", summary=summary(900)),
        Col("turbidity", "numeric_continuous", summary=summary(900)),
    ]
    found = names(columns)
    assert found["Average water_temperature"].rule == "R-SUG-28"
    assert "Average turbidity" in found
    assert not {"Average station_code", "Average latitude"} & set(found)


def test_kpi_a_value_repeated_per_key_takes_the_median() -> None:
    columns = [
        Col("id_mutation", "identifier", summary=summary(400)),
        Col("valeur_fonciere", "numeric_continuous", summary=summary(380)),
    ]
    facts = Facts(dependency={("id_mutation", "valeur_fonciere"): 1.0}, rows=1_000)
    found = names(columns, facts)
    value = found["Median valeur_fonciere"]
    assert value.rule == "R-SUG-28"
    assert any(r.startswith("R-SUG-29 · ") and "id_mutation" in r for r in value.reasons)


def test_kpi_sentences_use_the_article_and_the_percentage() -> None:
    columns = [
        Col("driver_age", "numeric_continuous", "age", summary=summary(60)),
        Col("PercentSalaryHike", "numeric_discrete", "amount", summary=summary(15)),
    ]
    reasons = " ".join(r for s in suggest(columns) for r in s.reasons)
    assert "a age" not in reasons and "driver_age is an age" in reasons
    alone = " ".join(
        r for s in suggest([Col("age", "numeric_continuous", "age")]) for r in s.reasons
    )
    assert "age is a age" not in alone and "age: averaged" in alone
    assert "a percentage" in reasons and "salary" not in reasons


def test_kpi_a_count_read_as_yes_no_is_not_a_share() -> None:
    columns = [Col("number_of_cyclist_killed", "boolean", summary=summary(2))]
    assert "% number_of_cyclist_killed" not in names(columns)
    assert any("number_of_cyclist_killed" in n for n in notes(columns))


def test_kpi_notes_reach_the_dataset_suggestions(tmp_path: Path) -> None:
    rng = np.random.default_rng(21)
    frame = pd.DataFrame(
        {
            "patient_nbr": rng.integers(100_000, 900_000, 600) // 3,
            "time_in_hospital": rng.integers(1, 14, 600),
        }
    )
    found = open_frame(tmp_path, frame).kpi_suggestions()
    assert not any(
        m.column == "patient_nbr" and m.agg == "sum"
        for k in found.kpis
        for m in k.definition.measures
    )
    assert any(n.startswith("R-SUG-25 · patient_nbr") for n in found.notes)


# --- Time series (TS-1, TS-2) ------------------------------------------------------------------


def requests_311(rows: int = 2_000) -> pd.DataFrame:
    rng = np.random.default_rng(31)
    created = pd.Timestamp("2026-11-01") + pd.to_timedelta(
        np.sort(rng.integers(0, 7 * 86_400, rows)), "s"
    )
    return pd.DataFrame(
        {
            "unique_key": np.arange(71_000_000, 71_000_000 + rows),
            "created_date": created,
            "closed_date": created + pd.to_timedelta(rng.integers(60, 86_400, rows), "s"),
            "council_district": rng.integers(1, 52, rows),
            "bbl": rng.integers(1_000_000_000, 5_999_999_999, rows),
            "latitude": rng.normal(40.7, 0.05, rows),
            "longitude": rng.normal(-73.9, 0.05, rows),
        }
    )


def test_series_without_a_measure_counts_the_rows(tmp_path: Path) -> None:
    setup = open_frame(tmp_path, requests_311()).timeseries().setup()
    assert setup.time == "created_date"
    assert setup.aggregate == "count"
    said = " ".join(setup.reasons)
    assert "R-TS-05" in said and "R-TS-06" in said and "unique_key" in said


def test_series_averages_a_price(tmp_path: Path) -> None:
    rng = np.random.default_rng(32)
    frame = pd.DataFrame(
        {
            "Date": pd.Timestamp("2016-01-01") + pd.to_timedelta(rng.integers(0, 700, 3_000), "D"),
            "Price": rng.lognormal(13.5, 0.5, 3_000).round(-3),
            "Postcode": rng.integers(3_000, 3_200, 3_000),
        }
    )
    setup = open_frame(tmp_path, frame).timeseries().setup()
    assert (setup.value, setup.aggregate) == ("Price", "mean")
    assert any(r.startswith("R-TS-03 · Price") for r in setup.reasons)


def test_series_prefers_an_event_date_to_a_birth_date(tmp_path: Path) -> None:
    rng = np.random.default_rng(33)
    rows = 1_000
    frame = pd.DataFrame(
        {
            "Code du département": rng.integers(1, 96, rows),
            "Date de naissance": pd.Timestamp("1950-01-01")
            + pd.to_timedelta(rng.integers(0, 15_000, rows), "D"),
            "Date de début du mandat": pd.Timestamp("2020-03-15")
            + pd.to_timedelta(rng.integers(0, 900, rows), "D"),
            "indemnite": rng.uniform(500, 4_000, rows).round(2),
        }
    )
    setup = open_frame(tmp_path, frame).timeseries().setup()
    assert setup.time == "Date de début du mandat"
    assert setup.value != "Code du département" or setup.aggregate == "count"


def test_series_sums_an_amount_and_averages_a_measurement(tmp_path: Path) -> None:
    rng = np.random.default_rng(34)
    days = pd.date_range("2024-01-01", periods=400, freq="D")
    sales = pd.DataFrame({"order_date": days, "sales": rng.gamma(2, 100, 400)})
    setup = open_frame(tmp_path, sales, "sales").timeseries().setup()
    assert (setup.value, setup.aggregate) == ("sales", "sum")
    weather = pd.DataFrame({"DATE": days, "TMAX": rng.normal(60, 15, 400).round(1)})
    setup = open_frame(tmp_path, weather, "weather").timeseries().setup()
    assert (setup.value, setup.aggregate) == ("TMAX", "mean")
    assert any(r.startswith("R-TS-04") for r in setup.reasons)


def test_series_keeps_the_users_choice(tmp_path: Path) -> None:
    series = open_frame(tmp_path, requests_311()).timeseries(value="council_district")
    setup = series.setup()
    assert setup.value == "council_district" and setup.aggregate == "mean"


def test_series_averages_whole_readings_taken_once_a_day(tmp_path: Path) -> None:
    rng = np.random.default_rng(35)
    days = pd.date_range("2020-01-01", periods=900, freq="D")
    frame = pd.DataFrame({"DATE": days, "TMAX": rng.normal(60, 15, 900).round().astype(int)})
    setup = open_frame(tmp_path, frame).timeseries().setup()
    assert (setup.value, setup.aggregate) == ("TMAX", "mean")


def test_sequence_numbers_and_date_parts() -> None:
    assert semantics.sequence(1, 62_541, True, 62_541) is not None  # gameorder
    assert semantics.sequence(1, 82, True, 82) is None  # too few to tell
    assert semantics.sequence(12_000, 755_000, True, 1_000) is None  # prices
    assert semantics.is_date_part("mois") and semantics.is_date_part("jour")
    assert not semantics.is_date_part("jour_ferie_count")


def test_kpi_a_total_beside_a_currency_is_money() -> None:
    columns = [
        Col("CompTotal", "numeric_continuous", "amount", summary=summary(900)),
        Col("Currency", "categorical_nominal", "currency", summary=summary(12)),
    ]
    found = names(columns)
    assert found["Total CompTotal"].rule == "R-SUG-02"
    assert not any("CompTotal" in n for n in notes(columns))


def test_kpi_codes_and_sequences_by_their_values_are_no_measure() -> None:
    columns = [
        Col("bbl", "numeric_continuous", summary=summary(900)),
        Col("gameorder", "numeric_continuous", summary=summary(900)),
        Col("temperature", "numeric_continuous", summary=summary(900)),
    ]
    facts = Facts(
        ranges={
            "bbl": (1_000_160_001, 5_080_500_013, True),
            "gameorder": (1, 950, True),
            "temperature": (-5.5, 38.2, False),
        },
        rows=1_000,
    )
    found = names(columns, facts)
    assert "Average temperature" in found
    assert not {"Average bbl", "Average gameorder"} & set(found)


def test_b07_still_reads_a_month_number_with_its_zero(tmp_path: Path) -> None:
    months = [f"{m:02d}" for m in np.random.default_rng(18).integers(1, 13, 600)]
    found = suggestions(open_frame(tmp_path, pd.DataFrame({"mois": months})))
    assert "B-13:mois" not in found
