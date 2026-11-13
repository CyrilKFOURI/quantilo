"""Fixes from the multi-domain verification (2026-11-10), defects left after 092 to 095.

Each test reproduces one corpus defect on a small synthetic file (no corpus file is copied)."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import quantilo as dx
from quantilo.cleaning.model import AssembleDate, FilterRows
from quantilo.typing import date_parts


def open_frame(tmp_path: Path, frame: pd.DataFrame, name: str = "data") -> dx.Dataset:
    path = tmp_path / f"{name}.csv"
    frame.to_csv(path, index=False)
    return dx.open(path, cache_dir=tmp_path / "cache")


# --- 1. A date and no measure: the series counts the rows (R-TS-07, fr_rne_maires) -----------


def days(start: str, offsets: np.ndarray) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.Timestamp(start) + pd.to_timedelta(offsets, "D"))


def removed(suggestion: Any) -> list[str]:
    """The labels a B-21 suggestion leaves out."""
    step = suggestion.step
    assert isinstance(step, FilterRows)
    return list(step.conditions[0].values or [])


def mayors(rows: int = 1_200) -> pd.DataFrame:
    rng = np.random.default_rng(71)
    departments = [f"{d:02d}" for d in range(1, 96)] + ["2A", "2B"]
    return pd.DataFrame(
        {
            "Code du département": rng.choice(departments, rows),
            "Nom de l'élu": rng.choice(["MARTIN", "BERNARD", "DUBOIS", "THOMAS"], rows),
            "Code sexe": rng.choice(["M", "F"], rows),
            "Date de naissance": days("1945-01-01", rng.integers(0, 20_000, rows)).strftime(
                "%d/%m/%Y"
            ),
            "Date de début du mandat": days("2020-03-15", rng.integers(0, 1_500, rows)).strftime(
                "%d/%m/%Y"
            ),
        }
    )


def test_series_without_any_numeric_column_counts_the_rows(tmp_path: Path) -> None:
    ds = open_frame(tmp_path, mayors())
    assert not [c for c in ds.types if c.type.startswith("numeric")]
    series = ds.timeseries()
    overview = series.overview()
    setup = overview.setup
    assert setup.time == "Date de début du mandat"
    assert setup.value == ""  # no column: the rows themselves
    assert setup.aggregate == "count"
    assert sum(v or 0 for v in overview.analyzed.values) == 1_200
    assert any(r.startswith("R-TS-07") for r in setup.reasons)


def test_rows_series_keeps_its_count_when_another_aggregate_is_asked(tmp_path: Path) -> None:
    ds = open_frame(tmp_path, mayors())
    setup = ds.timeseries(aggregate="mean", period="year").setup()
    assert setup.aggregate == "count"
    assert setup.periods >= 4


def test_rows_series_by_group(tmp_path: Path) -> None:
    ds = open_frame(tmp_path, mayors())
    overview = ds.timeseries(group="Code sexe", period="year").overview()
    assert overview.setup.value == ""
    assert {line.label for line in overview.lines} == {"M", "F"}
    total = sum(v or 0 for line in overview.lines for v in line.values)
    assert total == 1_200


# --- 2. Year, month, day in separate columns: one assembled date (R-TYP-12, B-20) -------------


def accidents(rows: int = 800) -> pd.DataFrame:
    rng = np.random.default_rng(72)
    when = days("2024-01-01", rng.integers(0, 366, rows))
    return pd.DataFrame(
        {
            "Num_Acc": np.arange(202_400_000_001, 202_400_000_001 + rows),
            "jour": when.day,
            "mois": when.month,
            "an": when.year,
            "hrmn": [
                f"{h:02d}:{m:02d}"
                for h, m in zip(rng.integers(0, 24, rows), rng.integers(0, 60, rows), strict=True)
            ],
            "lum": rng.integers(1, 6, rows),
        }
    )


@pytest.mark.parametrize(
    ("names", "expected"),
    [
        (["year", "month", "day"], {"": ["year", "month", "day"]}),
        (["annee", "mois", "jour", "heure"], {"": ["annee", "mois", "jour", "heure"]}),
        (["Yr Sold", "Mo Sold", "Year Built"], {"sold": ["Yr Sold", "Mo Sold"]}),
        (["yr", "mo"], {"": ["yr", "mo"]}),
        (["an", "mois", "jour", "minute"], {"": ["an", "mois", "jour"]}),  # no hour: no minute
        (["year", "day"], {}),  # no month: no date
        (["year_month", "day"], {}),
        (
            ["Année de naissance", "Mois de naissance"],
            {"naissance": ["Année de naissance", "Mois de naissance"]},
        ),
    ],
)
def test_date_part_groups_by_name(names: list[str], expected: dict[str, list[str]]) -> None:
    found = {g.stem: g.columns for g in date_parts.groups(names)}
    assert found == expected


def test_assembled_date_is_suggested_and_reversible(tmp_path: Path) -> None:
    ds = open_frame(tmp_path, accidents())
    # Columns: the reading note on the year column (R-TYP-12)
    note = [r for r in ds.column("an").reasons if "R-TYP-12" in r]
    assert note and "jour (day)" in note[0] and "B-20" in note[0]
    # Time series: no date column, and the message points at Clean
    with pytest.raises(dx.ExplorerError, match="B-20"):
        ds.timeseries().setup()
    cleaning = ds.clean()
    found = [s for s in cleaning.suggestions if s.rule == "B-20"]
    assert len(found) == 1
    suggestion = found[0]
    assert suggestion.mode == "S"  # proposed, never ticked by default
    # C1: the time of day in `hrmn` ("07:40") joins the date (it was left out by fix1)
    assert suggestion.columns == ["an", "mois", "jour", "hrmn"]
    assert isinstance(suggestion.step, AssembleDate) and suggestion.step.to == "date"
    assert suggestion.step.time == "hrmn" and suggestion.step.hour is None
    assert not any(step.rule == "B-20" for step in cleaning.steps)
    assert "800 of 800 rows" in suggestion.reason
    cleaned = cleaning.use(suggestion.id).apply(cache_dir=tmp_path / "cache")
    assert cleaned.column("date").type == "datetime"
    assert {"jour", "mois", "an"} <= set(cleaned.columns)  # the parts stay
    first = cleaned.to_pandas(["date", "hrmn"]).head(1)
    assert str(first["date"].iloc[0])[11:16] == accidents()["hrmn"].iloc[0]
    assert ds.columns == list(accidents().columns)  # the source is never changed
    setup = cleaned.timeseries().setup()
    assert setup.time == "date"


def test_impossible_dates_stay_empty(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "year": [2024] * 60 + [2023],
            "month": [1, 2, 3] * 20 + [2],
            "day": [15] * 60 + [30],
            "sales": np.arange(61.0),
        }
    )
    ds = open_frame(tmp_path, frame)
    cleaning = ds.clean()
    suggestion = next(s for s in cleaning.suggestions if s.rule == "B-20")
    assert "60 of 61 rows" in suggestion.reason
    cleaned = cleaning.use(suggestion.id).apply(cache_dir=tmp_path / "cache")
    summary = cleaned.column("date").summary or {}
    assert summary.get("null_count") == 1


def test_no_assembly_when_the_date_is_already_there(tmp_path: Path) -> None:
    days = pd.date_range("2024-01-01", periods=200, freq="D")
    frame = pd.DataFrame({"Date": days, "Year": days.year, "Month": days.month, "x": range(200)})
    ds = open_frame(tmp_path, frame)
    assert not [s for s in ds.clean().suggestions if s.rule == "B-20"]


def test_month_names_are_read(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {"year": [2023, 2024] * 30, "month": ["Jan", "February", "mars"] * 20, "v": range(60)}
    )
    ds = open_frame(tmp_path, frame)
    suggestion = next(s for s in ds.clean().suggestions if s.rule == "B-20")
    assert "60 of 60 rows" in suggestion.reason
    assert "1st of the month" in suggestion.reason


def test_series_prefers_the_assembled_date_to_a_year(tmp_path: Path) -> None:
    rng = np.random.default_rng(73)
    frame = pd.DataFrame(
        {
            "Yr Sold": rng.integers(2006, 2011, 600),
            "Mo Sold": rng.integers(1, 13, 600),
            "SalePrice": rng.lognormal(12, 0.3, 600).round(),
        }
    )
    ds = open_frame(tmp_path, frame)
    cleaning = ds.clean()
    suggestion = next(s for s in cleaning.suggestions if s.rule == "B-20")
    cleaned = cleaning.use(suggestion.id).apply(cache_dir=tmp_path / "cache")
    assert cleaned.timeseries().setup().time == "sold date"


# --- 3. Rows that sum up other rows: flagged, never removed (B-21, R-SUG-30) ------------------


COUNTRIES = {
    "France": "FRA",
    "Germany": "DEU",
    "Italy": "ITA",
    "Spain": "ESP",
    "South Africa": "ZAF",
    "Japan": "JPN",
    "Brazil": "BRA",
    "Canada": "CAN",
    "Kosovo": "",
    "Chile": "CHL",
}


def panel() -> pd.DataFrame:
    rng = np.random.default_rng(74)
    rows = []
    for year in range(1990, 2021):
        values = {c: float(rng.uniform(10, 100)) for c in COUNTRIES}
        for country, code in COUNTRIES.items():
            rows.append((country, year, code, values[country]))
        europe = sum(values[c] for c in ("France", "Germany", "Italy", "Spain", "Kosovo"))
        rows.append(("Europe", year, "", europe))
        rows.append(("High-income countries", year, "", europe + values["Japan"]))
        rows.append(("Africa (GCP)", year, "OWID_AFR", values["South Africa"]))
        rows.append(("World", year, "OWID_WRL", sum(values.values())))
    return pd.DataFrame(rows, columns=["country", "year", "iso_code", "co2"])


def test_aggregate_rows_are_proposed_to_leave_out(tmp_path: Path) -> None:
    ds = open_frame(tmp_path, panel())
    cleaning = ds.clean()
    found = [s for s in cleaning.suggestions if s.rule == "B-21"]
    assert len(found) == 1
    suggestion = found[0]
    assert suggestion.mode == "S"
    assert not any(step.rule == "B-21" for step in cleaning.steps)  # never ticked by default
    labels = sorted(removed(suggestion))
    assert labels == ["Africa (GCP)", "Europe", "High-income countries", "World"]
    assert suggestion.rows == 4 * 31
    assert "OWID_AFR" in " ".join(e.after or "" for e in suggestion.examples)
    assert len(ds) == 14 * 31  # nothing removed on its own
    cleaned = cleaning.use(suggestion.id).apply(cache_dir=tmp_path / "cache")
    assert len(cleaned) == 10 * 31  # Kosovo (no code, not a group) stays
    kpis = ds.kpi_suggestions()
    assert any(note.startswith("R-SUG-30 · country") for note in kpis.notes)


def test_total_row_found_by_its_sum(tmp_path: Path) -> None:
    rng = np.random.default_rng(75)
    sales = rng.uniform(100, 900, 8).round(2)
    frame = pd.DataFrame(
        {
            "department": [f"Dept {i}" for i in range(8)] + ["Sum of departments"],
            "sales": [*sales, sales.sum()],
        }
    )
    found = [s for s in open_frame(tmp_path, frame).clean().suggestions if s.rule == "B-21"]
    assert len(found) == 1
    assert removed(found[0]) == ["Sum of departments"]
    assert "is the sum of the other rows" in found[0].reason


def test_no_aggregate_among_plain_labels(tmp_path: Path) -> None:
    rng = np.random.default_rng(76)
    frame = pd.DataFrame(
        {
            "country": rng.choice(list(COUNTRIES), 500),
            "region": rng.choice(["Europe", "Asia", "Africa", "Oceania", "Americas", "Other"], 500),
            "sales": rng.uniform(1, 10, 500),
        }
    )
    ds = open_frame(tmp_path, frame)
    assert not [s for s in ds.clean().suggestions if s.rule == "B-21"]
    assert not [n for n in ds.kpi_suggestions().notes if n.startswith("R-SUG-30")]


def test_a_qualifier_in_brackets_is_not_an_aggregate(tmp_path: Path) -> None:
    rng = np.random.default_rng(77)
    zoning = ["RL", "RM", "FV", "RH", "C (all)", "I (all)", "A (agr)"]
    frame = pd.DataFrame(
        {"MS Zoning": rng.choice(zoning, 700), "SalePrice": rng.lognormal(12, 0.3, 700).round()}
    )
    assert not [s for s in open_frame(tmp_path, frame).clean().suggestions if s.rule == "B-21"]


# --- 4. One file, several kinds of records told apart by `type`: one table per kind (R-SST-24) -


FIXTURES = Path(__file__).parent / "fixtures" / "structured"


def write_jsonl(path: Path, records: list[dict[str, object]]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    return path


def test_events_split_by_type() -> None:
    layout = dx.read_structured(FIXTURES / "events.jsonl")
    names = [t.name for t in layout.tables]
    assert {"events.PushEvent", "events.WatchEvent", "events.IssuesEvent"} <= set(names)
    main = layout.table("events")
    assert not [c.name for c in main.columns if c.name.startswith("payload.")]
    assert {"id", "type", "actor.login", "created_at"} <= {c.name for c in main.columns}
    push = layout.table("events.PushEvent")
    assert push.rule == "R-SST-24"
    assert [c.name for c in push.columns][:1] == ["_parent_id"]
    assert "payload.size" in {c.name for c in push.columns}
    kinds = [t for t in layout.tables if t.rule == "R-SST-24"]
    assert sum(t.rows for t in kinds) == main.rows
    relation = next(r for r in layout.relations if r.child == "events.PushEvent")
    assert (relation.parent_key, relation.child_key, relation.rule) == (
        "_id",
        "_parent_id",
        "R-SST-24",
    )
    assert relation.path == "type=PushEvent"
    assert any(hit.rule == "R-SST-24" for hit in layout.reasons)
    # the list tables stay linked to the records
    assert "events.payload.commits" in names
    ids = set(layout.arrow("events").column("_id").to_pylist())
    assert set(layout.arrow("events.PushEvent").column("_parent_id").to_pylist()) <= ids


def test_split_can_be_turned_off() -> None:
    layout = dx.read_structured(FIXTURES / "events.jsonl", discriminator="")
    assert not [t for t in layout.tables if t.rule == "R-SST-24"]
    assert "payload.size" in {c.name for c in layout.table("events").columns}


def test_no_split_when_every_kind_has_the_same_fields(tmp_path: Path) -> None:
    records = [
        {"id": i, "type": ["a", "b", "c"][i % 3], "amount": i * 1.5, "user": {"name": f"u{i}"}}
        for i in range(90)
    ]
    layout = dx.read_structured(write_jsonl(tmp_path / "same.jsonl", records))
    assert not [t for t in layout.tables if t.rule == "R-SST-24"]


def test_a_field_named_otherwise_is_split_only_when_asked(tmp_path: Path) -> None:
    records = [
        {"id": i, "shape": "circle", "radius": i}
        if i % 2
        else {"id": i, "shape": "box", "w": i, "h": 2}
        for i in range(60)
    ]
    path = write_jsonl(tmp_path / "shapes.jsonl", records)
    assert not [t for t in dx.read_structured(path).tables if t.rule == "R-SST-24"]
    forced = dx.read_structured(path, discriminator="shape")
    assert {"shapes.circle", "shapes.box"} <= {t.name for t in forced.tables}


def test_column_limit_counts_per_kind(tmp_path: Path) -> None:
    kinds = ["Push", "Fork", "Issue", "Release"]
    records = [
        {
            "id": i,
            "type": kinds[i % 4],
            "payload": {kinds[i % 4].lower(): {f"field_{k}": k for k in range(150)}},
            "created_at": "2015-01-01",
        }
        for i in range(80)
    ]
    layout = dx.read_structured(write_jsonl(tmp_path / "wide.jsonl", records))
    main = layout.table("wide")
    assert "created_at" in {c.name for c in main.columns}  # not pushed into _extra (R-SST-18)
    assert len(layout.table("wide.Push").columns) == 151


def test_records_without_a_kind_keep_their_fields(tmp_path: Path) -> None:
    records: list[dict[str, object]] = [
        {"id": i, "type": "Push", "payload": {"size": i, "head": f"h{i}", "ref": "main"}}
        if i % 2
        else {"id": i, "type": "Watch", "payload": {"action": "started"}}
        for i in range(199)
    ]
    records.append({"id": 199, "payload": {"size": 7}})  # no type: 0.5 % of the records
    layout = dx.read_structured(write_jsonl(tmp_path / "loose.jsonl", records))
    lost = layout.table("loose.no_type")
    assert lost.rows == 1
    assert layout.arrow("loose.no_type").column("payload.size").to_pylist() == [7]
    kinds = [t for t in layout.tables if t.rule == "R-SST-24"]
    assert sum(t.rows for t in kinds) == 200
