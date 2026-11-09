"""Several series at once through a dataset (039): the library, as the app."""

from pathlib import Path

import numpy as np
import pytest

import quantilo as dx
from quantilo.timeseries import groups as gr

RNG = np.random.default_rng(39)
MONTHS = 48


def write(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


@pytest.fixture
def stores(tmp_path: Path) -> dx.Dataset:
    """30 stores in 3 regions, monthly sales (several rows per store and month)."""
    lines = ["date,montant,magasin,region"]
    for s in range(30):
        region = ("North", "South", "East")[s % 3]
        for m in range(MONTHS):
            level = 100 + 5 * s + 20 * np.sin(2 * np.pi * m / 12)
            for day in (3, 17):
                amount = level / 2 + RNG.normal(0, 4)
                lines.append(
                    f"{2021 + m // 12}-{m % 12 + 1:02d}-{day:02d},{amount:.2f},S{s:02d},{region}"
                )
    return dx.open(write(tmp_path / "stores.csv", lines), cache_dir=tmp_path / "cache")


def test_thirty_stores_give_thirty_forecasts(stores: dx.Dataset) -> None:
    series = stores.timeseries("date", "montant", group="magasin", period="month")

    setup = series.groups()
    found = series.forecast_groups(horizon=3, models=["naive", "seasonal_naive"])

    assert setup.groups == 30 and setup.summable and setup.levels == ["region"]
    region = next(c for c in setup.candidates if c.column == "region")
    assert region.parent and region.values == 3
    assert len(found.rows) == 30 and found.kept == 30 and found.pooled == 0
    assert {r.label for r in found.rows} == {f"S{s:02d}" for s in range(30)}
    assert all(len(r.forecast) == 3 and r.level == "magasin" for r in found.rows)
    assert found.rows[0].volume >= found.rows[-1].volume  # largest first
    assert found.future == ["2025-01", "2025-02", "2025-03"]
    assert sum(s.series for s in found.best_models) == 30
    assert all(r.change is not None for r in found.rows)  # vs the same months of 2024


def test_the_hierarchy_is_recognised_and_totals_add_up(stores: dx.Dataset) -> None:
    """Spec 039: store → region; after reconciliation Σ stores = region and Σ regions = total at
    every date (gap < 1e-6)."""
    series = stores.timeseries("date", "montant", group="magasin", period="month")

    found = series.forecast_groups(
        horizon=3, models=["seasonal_naive", "ses"], reconcile="mint_shrink"
    )

    assert found.hierarchy is not None and found.hierarchy.levels == ["Total", "region", "magasin"]
    assert found.hierarchy.method == "mint_shrink" and found.hierarchy.gap < 1e-6
    by_level: dict[str, list[gr.GroupRow]] = {}
    for row in found.rows:
        by_level.setdefault(row.level, []).append(row)
    assert [len(by_level[k]) for k in ("magasin", "region", "Total")] == [30, 3, 1]
    for region in by_level["region"]:
        children = np.sum(
            [r.forecast for r in by_level["magasin"] if r.parent == region.label], axis=0
        )
        assert np.max(np.abs(children - region.forecast)) < 1e-6
    regions = np.sum([r.forecast for r in by_level["region"]], axis=0)
    assert np.max(np.abs(regions - by_level["Total"][0].forecast)) < 1e-6
    assert [e.level for e in found.hierarchy.errors] == ["Total", "region", "magasin"]
    assert all(e.mase_before is not None for e in found.hierarchy.errors)


def test_a_column_that_is_not_above_the_group_is_refused(stores: dx.Dataset) -> None:
    series = stores.timeseries("date", "montant", group="region", period="month")
    with pytest.raises(dx.ExplorerError, match="not above"):
        series.forecast_groups(
            horizon=3, models=["naive"], reconcile="bottom_up", levels=["magasin"]
        )
    mean = stores.timeseries("date", "montant", group="magasin", period="month", aggregate="mean")
    with pytest.raises(dx.ExplorerError, match="sums and counts"):
        mean.forecast_groups(horizon=3, models=["naive"], reconcile="bottom_up")


def test_five_hundred_products_keep_the_two_hundred_largest(tmp_path: Path) -> None:
    """Spec 039: 500 products → the 200 largest are forecast, the 300 others form "Others",
    and it is said."""
    lines = ["date,units,product"]
    for p in range(500):
        for m in range(24):
            lines.append(
                f"{2023 + m // 12}-{m % 12 + 1:02d}-01,{1000 - p + RNG.integers(0, 5)},P{p:03d}"
            )
    ds = dx.open(write(tmp_path / "products.csv", lines), cache_dir=tmp_path / "cache")

    found = ds.timeseries("date", "units", group="product", period="month").forecast_groups(
        horizon=3, models=["naive"]
    )

    assert found.groups == 500 and found.kept == 200 and found.pooled == 300 and found.limit == 200
    assert len(found.rows) == 201
    assert {r.label for r in found.rows[:200]} == {f"P{p:03d}" for p in range(200)}
    others = found.rows[-1]
    assert others.label == "Others" and others.other
    assert any("300 smaller product values are pooled" in note for note in found.notes)


def test_short_histories_are_said(tmp_path: Path) -> None:
    lines = ["date,units,shop"]
    for m in range(36):
        lines.append(f"{2022 + m // 12}-{m % 12 + 1:02d}-01,{50 + m % 12},old")
        if m >= 30:
            lines.append(f"{2022 + m // 12}-{m % 12 + 1:02d}-01,{20 + m % 3},new")
    ds = dx.open(write(tmp_path / "shops.csv", lines), cache_dir=tmp_path / "cache")

    found = ds.timeseries("date", "units", group="shop", period="month").forecast_groups(horizon=3)

    new = next(r for r in found.rows if r.label == "new")
    old = next(r for r in found.rows if r.label == "old")
    assert new.short_history and new.best == "seasonal_naive" and not old.short_history
    assert found.short == 1


def test_csv_export_is_long(stores: dx.Dataset) -> None:
    found = stores.timeseries("date", "montant", group="magasin", period="month").forecast_groups(
        horizon=2, models=["naive"]
    )

    lines = gr.long_csv(found).splitlines()

    assert len(lines) == 1 + 30 * 2
    assert lines[1].split(",")[:3] == [found.rows[0].label, "magasin", "2025-01"]


# --- Scenarios -----------------------------------------------------------------------------------


@pytest.fixture
def shop(tmp_path: Path) -> dx.Dataset:
    """Monthly sales driven by the price (-12 per unit), with 12 future rows of planned prices."""
    lines = ["month,sales,price"]
    for i in range(72):
        price = 20 + 3 * np.sin(i / 3) + (i % 5)
        sales = "" if i >= 60 else f"{500 - 12 * price + RNG.normal(0, 3):.2f}"
        lines.append(f"{2020 + i // 12}-{i % 12 + 1:02d}-01,{sales},{price:.2f}")
    return dx.open(write(tmp_path / "shop.csv", lines), cache_dir=tmp_path / "cache")


@pytest.mark.parametrize("model", ["tslm", "sarimax"])
def test_a_december_price_cut_only_moves_december(shop: dx.Dataset, model: str) -> None:
    """Spec 039: "price -10 % in December" changes December's forecast in the direction of the
    learned coefficient (sales rise when the price falls); the other months stay the same."""
    series = shop.timeseries(time="month", value="sales", period="month")
    drivers = [dx.Driver("price", known=True)]
    base = series.scenario(drivers, model=model, horizon=12)

    cut = series.scenario(
        drivers,
        model=model,
        horizon=12,
        changes=[dx.ScenarioChange("price", "percent", -10, start="2025-12", end="2025-12")],
    )

    december = base.future.index("2025-12")
    assert cut.drivers[0].future_values[december] == pytest.approx(
        0.9 * (base.drivers[0].future_values[december] or 0)
    )
    assert cut.forecast[december] > base.forecast[december] + 10  # -12 per unit, ~2.5 units
    others = [i for i in range(12) if i != december]
    assert np.array(cut.forecast)[others] == pytest.approx(np.array(base.forecast)[others])


def test_fixed_values_and_wrong_scenarios(shop: dx.Dataset) -> None:
    series = shop.timeseries(time="month", value="sales", period="month")
    drivers = [dx.Driver("price", known=True)]

    fixed = series.scenario(
        drivers, horizon=12, changes=[{"column": "price", "how": "value", "amount": 30}]
    )

    assert fixed.drivers[0].future_values == [30.0] * 12
    with pytest.raises(dx.ExplorerError, match="not a forecast period"):
        series.scenario(drivers, horizon=12, changes=[dx.ScenarioChange("price", start="2030-01")])
    with pytest.raises(dx.ExplorerError, match="past only"):
        series.scenario(["price"], horizon=12, changes=[dx.ScenarioChange("price", amount=5)])
