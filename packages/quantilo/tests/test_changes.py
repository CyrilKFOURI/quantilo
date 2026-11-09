"""What changed between two versions (031): the measures against hand-computed values, then the
spec's scenarios on small datasets."""

import math
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest
from scipy import stats

import quantilo as dx
from quantilo.changes.compare import guess_renames, names_close
from quantilo.changes.kpis import KpiValue, compare_kpi
from quantilo.changes.metrics import (
    bin_shares,
    cdf_at,
    chi_square,
    drift_of,
    ks_from_quantiles,
    normalised_wasserstein,
    psi,
    psi_on_deciles,
    wasserstein,
)
from quantilo.changes.rows import RowKeyError, write_differences
from quantilo.changes.snapshot import (
    LEVELS,
    KpiPoint,
    snapshot_from_dict,
    views_of,
)

# --- Measures, by hand ---------------------------------------------------------------------------


def test_psi_by_hand() -> None:
    # e = (0.5, 0.5), a = (0.3, 0.7): (0.3 − 0.5)·ln(0.6) + (0.7 − 0.5)·ln(1.4) = 0.2·ln(7/3)
    assert psi([50, 50], [30, 70]) == pytest.approx(0.2 * math.log(7 / 3), abs=1e-12)
    assert psi([1, 2, 3], [1, 2, 3]) == 0


def test_psi_floors_empty_bins_at_epsilon() -> None:
    # e = (1, 0 → 1e-4), a = (0.5, 0.5)
    expected = (0.5 - 1) * math.log(0.5 / 1) + (0.5 - 1e-4) * math.log(0.5 / 1e-4)
    assert psi([100, 0], [50, 50]) == pytest.approx(expected, abs=1e-12)


def test_psi_reading() -> None:
    assert [drift_of(v) for v in (0.05, 0.1, 0.2, 0.25, 1.0, None)] == [
        "stable",
        "moderate",
        "moderate",
        "strong",
        "strong",
        None,
    ]


def uniform(low: float, high: float) -> list[float]:
    return [low + (high - low) * p for p in LEVELS]


def test_cdf_from_quantiles_and_ties() -> None:
    q = uniform(0, 10)
    assert cdf_at(q, LEVELS, np.array([-1.0, 0.0, 2.5, 10.0, 11.0])).tolist() == pytest.approx(
        [0, 0, 0.25, 1, 1]
    )
    # half zeros, half ones: F(0) = 0.5, F(1) = 1
    ties = [0.0] * 51 + [1.0] * 50
    assert cdf_at(ties, LEVELS, np.array([0.0, 1.0])).tolist() == pytest.approx([0.5, 1.0])
    assert bin_shares(ties, LEVELS, np.array([0.0])).tolist() == pytest.approx([0.5, 0.5])


def test_psi_on_deciles_of_the_old_version() -> None:
    # Old uniform on [0, 10]: deciles 1…9 hold 10 % each. New uniform on [0, 20]: the first nine
    # bins hold 5 % each, the last one (> 9) holds 55 %.
    shares_new = [0.05] * 9 + [0.55]
    expected = sum((a - 0.1) * math.log(a / 0.1) for a in shares_new)
    assert psi_on_deciles(uniform(0, 10), uniform(0, 20), LEVELS) == pytest.approx(
        expected, abs=1e-9
    )


def test_wasserstein_by_hand() -> None:
    # A shift by 2 moves every quantile by 2: W₁ = 2; the old IQR is 7.5 − 2.5 = 5.
    assert wasserstein(uniform(0, 10), uniform(2, 12), LEVELS) == pytest.approx(2)
    assert normalised_wasserstein(uniform(0, 10), uniform(2, 12), LEVELS) == pytest.approx(0.4)
    assert normalised_wasserstein([3.0] * 101, uniform(0, 1), LEVELS) is None


def test_wasserstein_matches_scipy_on_samples() -> None:
    rng = np.random.default_rng(3)
    a, b = rng.normal(0, 1, 20_000), rng.normal(0.5, 1.5, 20_000)
    found = wasserstein(np.quantile(a, LEVELS), np.quantile(b, LEVELS), LEVELS)
    assert found == pytest.approx(stats.wasserstein_distance(a, b), rel=0.05)


def test_ks_by_hand_and_against_scipy() -> None:
    # Uniform [0, 10] against [2, 12]: the CDFs are 0.2 apart on [2, 10].
    statistic, p = ks_from_quantiles(uniform(0, 10), uniform(2, 12), LEVELS, 1000, 1000)
    assert statistic == pytest.approx(0.2, abs=1e-9)
    assert p == pytest.approx(stats.kstwobign.sf(0.2 * math.sqrt(500)))
    rng = np.random.default_rng(5)
    a, b = rng.normal(0, 1, 5_000), rng.normal(0.2, 1, 5_000)
    statistic, p = ks_from_quantiles(
        np.quantile(a, LEVELS), np.quantile(b, LEVELS), LEVELS, 5000, 5000
    )
    reference = stats.ks_2samp(a, b)
    assert statistic == pytest.approx(reference.statistic, abs=0.015)  # type: ignore[attr-defined]
    assert p is not None and p < 1e-6


def test_chi_square_by_hand() -> None:
    # Expected 15 in each cell: χ² = 4 · 5² / 15 = 6.667, one degree of freedom.
    found = chi_square([10, 20], [20, 10])
    assert found is not None
    statistic, p, dof = found
    assert statistic == pytest.approx(20 / 3)
    assert dof == 1
    assert p == pytest.approx(stats.chi2.sf(20 / 3, 1))
    assert chi_square([5, 0], [7, 0]) is None


# --- Renames ------------------------------------------------------------------------------------


def test_names_close() -> None:
    assert names_close("chiffre_affaires", "ca")
    assert names_close("customer_name", "customer_names")
    assert names_close("TotalAmount", "ta")
    assert not names_close("price", "region")


# --- Scenarios ----------------------------------------------------------------------------------


def frame(rows: int = 2_000, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "order_id": [f"o{i:05d}" for i in range(rows)],
            "prix": rng.lognormal(3, 0.4, rows).round(2),
            "region": rng.choice(["North", "South", "East", "West"], rows, p=[0.4, 0.3, 0.2, 0.1]),
            "chiffre_affaires": rng.normal(1000, 150, rows).round(1),
            "day": pd.date_range("2026-01-01", periods=rows, freq="4h"),
        }
    )


def open_frame(data: pd.DataFrame, tmp_path: Path, name: str) -> dx.Dataset:
    return dx.from_pandas(data, name, cache_dir=tmp_path / name)


def test_same_file_no_notable_change(tmp_path: Path) -> None:
    data = frame()
    changes = open_frame(data, tmp_path, "v2").changes(open_frame(data, tmp_path, "v1"))
    assert changes.level == "none"
    assert changes.level_label == "No notable change"
    assert all((s.psi or 0) < 0.1 for s in changes.distributions)
    assert changes.schema == []
    assert changes.headline == "Compared with v1: same number of rows, no notable change."


def test_price_times_1_3_comes_first(tmp_path: Path) -> None:
    old = frame()
    new = old.assign(prix=old["prix"] * 1.3)
    changes = open_frame(new, tmp_path, "new").changes(open_frame(old, tmp_path, "old"))
    first = changes.distributions[0]
    assert first.column == "prix"
    assert first.psi is not None and first.psi > 0.25
    assert first.drift == "strong"
    assert first.histogram is not None
    assert len(first.histogram.before) == len(first.histogram.after) == 24
    assert first.before is not None and first.after is not None
    assert first.after.mean == pytest.approx(first.before.mean * 1.3, rel=1e-6)  # type: ignore[operator]
    assert changes.level == "check"
    assert "1 distribution shifted" in changes.headline


def test_a_new_category_is_listed_with_its_share(tmp_path: Path) -> None:
    old = frame()
    new = old.copy()
    new.loc[:99, "region"] = "Outre-mer"
    changes = open_frame(new, tmp_path, "new").changes(open_frame(old, tmp_path, "old"))
    region = next(s for s in changes.distributions if s.column == "region")
    appeared = {level.value: level for level in region.appeared}
    assert "Outre-mer" in appeared
    assert appeared["Outre-mer"].after == pytest.approx(100 / 2000)
    assert appeared["Outre-mer"].certain
    assert changes.counts.new_categories == 1
    assert region.chi2 is not None and region.chi2_p is not None


def test_a_renamed_column_is_proposed_not_counted_twice(tmp_path: Path) -> None:
    old = frame()
    new = old.rename(columns={"chiffre_affaires": "ca"})
    changes = open_frame(new, tmp_path, "new").changes(open_frame(old, tmp_path, "old"))
    kinds = [(c.kind, c.column, c.before) for c in changes.schema]
    assert kinds == [("renamed", "ca", "chiffre_affaires")]
    assert changes.counts.removed == changes.counts.added == 0
    assert changes.counts.renamed == 1
    assert "probably renamed" in changes.headline
    shift = next(s for s in changes.distributions if s.column == "ca")
    assert (shift.psi or 0) < 0.1  # the values are compared under the new name


def test_renames_need_the_same_kind_of_values() -> None:
    old = dx.from_pandas(pd.DataFrame({"a": [1.0, 2.0, 3.0] * 30})).snapshot()
    new = dx.from_pandas(pd.DataFrame({"b": ["x", "y", "z"] * 30})).snapshot()
    assert guess_renames(old.columns, new.columns, old.rows, new.rows) == []


def test_a_used_column_that_disappears_is_breaking(tmp_path: Path) -> None:
    old = frame()
    new = old.drop(columns=["prix"])
    changes = open_frame(new, tmp_path, "new").changes(
        open_frame(old, tmp_path, "old"), used={"prix": ["KPI Average price"]}
    )
    assert changes.level == "breaking"
    removed = next(c for c in changes.schema if c.kind == "removed")
    assert removed.column == "prix" and removed.breaking
    assert removed.used_by == ["KPI Average price"]
    assert any("KPI Average price" in reason for reason in changes.reasons)


def test_type_change_empty_and_constant_columns(tmp_path: Path) -> None:
    old = frame()
    new = old.assign(region="North", chiffre_affaires=np.nan, prix=old["prix"].astype(str) + " €?")
    changes = open_frame(new, tmp_path, "new").changes(open_frame(old, tmp_path, "old"))
    found = {(c.kind, c.column) for c in changes.schema}
    assert ("constant", "region") in found
    assert ("empty", "chiffre_affaires") in found
    assert ("retyped", "prix") in found
    quality = next(q for q in changes.quality if q.column == "chiffre_affaires")
    assert quality.missing_before == 0 and quality.missing_after == 100
    assert quality.notable


def test_focus_columns_come_first(tmp_path: Path) -> None:
    old = frame()
    new = old.assign(prix=old["prix"] * 2)
    changes = open_frame(new, tmp_path, "new").changes(
        open_frame(old, tmp_path, "old"), focus=["region"]
    )
    assert changes.distributions[0].column == "region"
    assert changes.distributions[0].focus


def test_dates_out_of_range_and_gaps(tmp_path: Path) -> None:
    months = pd.date_range("2025-01-01", periods=24, freq="MS")
    old = pd.DataFrame({"when": np.repeat(months[:18], 10), "x": range(180)})
    later = list(months[:6]) + list(months[8:18]) + [pd.Timestamp("2099-01-01")]
    new = pd.DataFrame({"when": np.repeat(later, 10), "x": range(170)})
    changes = open_frame(new, tmp_path, "new").changes(open_frame(old, tmp_path, "old"))
    when = next(d for d in changes.dates if d.column == "when")
    assert when.future == 10
    assert when.grain == "year"  # 2025 to 2099 is counted in years
    assert when.notable
    # On months (no date in the future), the missing months show
    later = list(months[:6]) + list(months[8:18])
    new = pd.DataFrame({"when": np.repeat(later, 10), "x": range(160)})
    changes = open_frame(new, tmp_path, "new2").changes(open_frame(old, tmp_path, "old"))
    when = next(d for d in changes.dates if d.column == "when")
    assert when.grain == "month"
    assert when.gaps == ["2025-07", "2025-08"]
    assert len(when.periods) == len(when.before_counts) == len(when.after_counts)


def test_snapshot_round_trip_and_views(tmp_path: Path) -> None:
    snapshot = open_frame(frame(), tmp_path, "v").snapshot()
    assert snapshot.rows == 2000
    prix = snapshot.column("prix")
    assert prix is not None and prix.numbers is not None and len(prix.numbers.quantiles) == 101
    region = snapshot.column("region")
    assert region is not None and region.levels is not None and region.levels.complete
    assert snapshot_from_dict(dx.to_dict(snapshot)) == snapshot
    assert [v.name for v in views_of(snapshot)] == [c.name for c in snapshot.columns]


# --- Row by row ---------------------------------------------------------------------------------


def test_rows_modified_added_removed(tmp_path: Path) -> None:
    old = frame(200)
    new = old.copy()
    new.loc[:11, "prix"] = new.loc[:11, "prix"] + 1  # 12 amounts changed
    extra = old.iloc[:2].assign(order_id=["zz1", "zz2"])
    new = pd.concat([new.iloc[:190], extra], ignore_index=True)  # 10 removed, 2 added
    after, before = open_frame(new, tmp_path, "new"), open_frame(old, tmp_path, "old")
    found = after.row_changes(before)
    assert found.key == "order_id"
    assert (found.modified, found.added, found.removed) == (12, 2, 10)
    assert found.unchanged == 190 - 12
    assert [(c.column, c.modified) for c in found.columns] == [("prix", 12)]
    assert found.added_keys == ["zz1", "zz2"]
    assert len(found.examples) == 12
    first = found.examples[0]
    assert first.key == "o00000" and first.column == "prix"
    assert float(first.after or 0) == pytest.approx(float(first.before or 0) + 1)

    target = write_differences(
        duckdb.connect(),
        str(before.path),
        before.types,
        str(after.path),
        after.types,
        "order_id",
        tmp_path / "diff.csv",
    )
    written = pd.read_csv(target)
    assert list(written.columns) == ["change", "order_id", "column", "before", "after"]
    modified = written[written["change"] == "modified"]
    assert len(modified) == 12 and set(modified["column"]) == {"prix"}
    gaps = (modified["after"].astype(float) - modified["before"].astype(float)).round(6)
    assert gaps.tolist() == [1.0] * 12
    assert written["change"].tolist().count("added") == 2
    assert written["change"].tolist().count("removed") == 10


def test_rows_need_a_unique_key(tmp_path: Path) -> None:
    data = pd.DataFrame({"k": ["a", "a", "b"], "v": [1, 1, 2]})
    after, before = open_frame(data, tmp_path, "a"), open_frame(data, tmp_path, "b")
    with pytest.raises(RowKeyError, match="repeats"):
        after.row_changes(before, key="k")
    with pytest.raises(RowKeyError, match="No column identifies"):
        after.row_changes(before)


# --- KPIs and relations -------------------------------------------------------------------------


def test_kpi_common_periods_show_revisions() -> None:
    before = KpiValue(
        "Sales", "Sum of amount", 30.0, [KpiPoint("2026-08", 10), KpiPoint("2026-09", 20)]
    )
    after = KpiValue(
        "Sales",
        "Sum of amount",
        45.0,
        [KpiPoint("2026-08", 10), KpiPoint("2026-09", 22), KpiPoint("2026-10", 13)],
    )
    change = compare_kpi("Sales", before, after)
    assert change.status == "ok"
    assert change.change == pytest.approx(0.5)
    assert [c.label for c in change.common] == ["2026-08", "2026-09"]
    assert change.common[1].change == pytest.approx(0.1)
    assert change.revised == 1
    broken = compare_kpi("Sales", before, None, "Column 'amount' not found", broken=True)
    assert broken.status == "broken" and broken.before == 30.0


def test_kpi_changes_on_two_versions(tmp_path: Path) -> None:
    old = frame()
    new = old.assign(prix=old["prix"] * 2)
    definition = {"measures": [{"id": "m1", "agg": "sum", "column": "prix"}]}
    after, before = open_frame(new, tmp_path, "new"), open_frame(old, tmp_path, "old")
    [change] = after.kpi_changes(before, {"Revenue": definition})
    assert change.status == "ok"
    assert change.change == pytest.approx(1.0)
    [gone] = open_frame(new.drop(columns=["prix"]), tmp_path, "gone").kpi_changes(
        before, {"Revenue": definition}
    )
    assert gone.status == "broken" and gone.problem


def test_relations_that_appear(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, 500)
    old = pd.DataFrame({"x": x, "y": rng.normal(0, 1, 500), "z": x + rng.normal(0, 0.1, 500)})
    new = old.assign(y=old["x"] * 2 + rng.normal(0, 0.1, 500))
    found = open_frame(new, tmp_path, "new").relation_changes(open_frame(old, tmp_path, "old"))
    appeared = [r for r in found if r.status == "appeared"]
    assert {(r.a, r.b) for r in appeared} >= {("x", "y")}
    assert found[0].status == "appeared"
    stable = next(r for r in found if {r.a, r.b} == {"x", "z"})
    assert stable.status == "stable"


def test_results_as_tables(tmp_path: Path) -> None:
    data = frame()
    changes = open_frame(data, tmp_path, "b").changes(open_frame(data, tmp_path, "a"))
    table = dx.to_pandas(changes)
    assert {"column", "psi", "drift"} <= set(table.columns)
    assert len(table) == len(changes.distributions)


def test_figures_without_plotly(tmp_path: Path) -> None:
    from quantilo.plots import PlotError, figure_data

    old = frame()
    changes = open_frame(old.assign(prix=old["prix"] * 1.3), tmp_path, "new").changes(
        open_frame(old, tmp_path, "old")
    )
    built, height = figure_data(changes)
    assert [trace["name"] for trace in built["data"]] == ["Before", "After"]
    assert built["layout"]["barmode"] == "overlay" and height == 300
    bars, _ = figure_data(changes, column="region")
    assert bars["data"][0]["orientation"] == "h"
    periods, _ = figure_data(changes, column="day")
    assert periods["data"][0]["x"] == changes.dates[0].periods
    with pytest.raises(PlotError):
        figure_data(changes, column="nope")


def test_differences_csv_cannot_carry_formulas(tmp_path: Path) -> None:
    # spreadsheet formula injection: a value "=HYPERLINK(...)" must not run when opened
    old = pd.DataFrame(
        {"order_id": ["a1", "a2", "a3"], "note": ["ok", "ok", "ok"], "qty": [1.0, 2.0, 3.0]}
    )
    new = old.assign(note=['=HYPERLINK("x")', "-2+3", "ok"], qty=[-12.0, 2.0, 3.0])
    after, before = open_frame(new, tmp_path, "new"), open_frame(old, tmp_path, "old")
    target = write_differences(
        duckdb.connect(),
        str(before.path),
        before.types,
        str(after.path),
        after.types,
        "order_id",
        tmp_path / "diff.csv",
    )
    written = pd.read_csv(target, dtype=str)
    notes = written[written["column"] == "note"]["after"].tolist()
    assert notes == ['\'=HYPERLINK("x")', "'-2+3"]
    assert written[written["column"] == "qty"]["after"].tolist() == ["-12.0"]


def test_new_invoice_numbers_are_expected_not_a_shift(tmp_path: Path) -> None:
    # Online Retail II, 2010 vs 2011: invoices repeat (lines of an invoice) but each year has new
    # ones; their PSI on the top 50 looked like the strongest shifts (040)
    rng = np.random.default_rng(5)

    def year(start: int) -> pd.DataFrame:
        invoices = [f"INV{n}" for n in rng.integers(start, start + 3_000, 6_000)]
        region = rng.choice(["North", "South"], 6_000)
        return pd.DataFrame({"invoice": invoices, "region": region})

    changes = open_frame(year(500_000), tmp_path, "2011").changes(
        open_frame(year(100_000), tmp_path, "2010")
    )
    invoice = next(s for s in changes.distributions if s.column == "invoice")
    assert invoice.long_tail
    assert changes.distributions[-1].column == "invoice"
    assert changes.counts.shifted == 0 and changes.counts.new_categories == 0
    assert changes.level == "none"
