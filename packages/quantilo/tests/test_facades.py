"""The library's modules on a dataset: machine learning, time series, KPIs, synthetic data."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import quantilo as dx
from quantilo.ml.evaluate import LOWER_IS_BETTER
from quantilo.ml.facade import Failed, TrainedModel

SAMPLES = Path(__file__).resolve().parents[3] / "samples"


@pytest.fixture(scope="module")
def shop(tmp_path_factory: pytest.TempPathFactory) -> dx.Dataset:
    rng = np.random.default_rng(1)
    n = 300
    frame = pd.DataFrame(
        {
            "age": rng.integers(18, 80, n),
            "income": rng.normal(40_000, 9_000, n).round(2),
            "city": rng.choice(["Paris", "Lyon", "Nice"], n),
        }
    )
    frame["bought"] = np.where(frame["income"] + rng.normal(0, 5_000, n) > 42_000, "buy", "skip")
    return dx.from_pandas(frame, "shop", cache_dir=tmp_path_factory.mktemp("shop"))


@pytest.fixture(scope="module")
def sales(tmp_path_factory: pytest.TempPathFactory) -> dx.Dataset:
    return dx.open(SAMPLES / "sales_timeseries.csv", cache_dir=tmp_path_factory.mktemp("sales"))


def test_targets_say_which_problem_each_column_makes(shop: dx.Dataset) -> None:
    tasks = {c.name: c.task for c in shop.ml_targets().candidates}
    assert tasks["bought"] == "binary"
    assert tasks["income"] == "regression"


def test_train_explain_and_predict(shop: dx.Dataset) -> None:
    model = shop.ml("bought").train()
    assert model.task == "binary" and model.model == "logistic_regression"
    assert model.score is not None and model.score > 0.7
    assert model.explain().permutation is not None
    new = pd.DataFrame({"age": [30, 60], "income": [80_000.0, 10_000.0], "city": ["Lyon", "Nice"]})
    predicted = model.predict(new)
    assert list(predicted["predicted"]) == ["buy", "skip"]
    assert {"p(buy)", "p(skip)"} <= set(predicted.columns)
    with pytest.raises(dx.ExplorerError):
        model.predict(new.drop(columns=["income"]))


def test_preparation_preview_as_a_table(shop: dx.Dataset) -> None:
    preview = shop.ml("bought").preview()

    frame = dx.to_pandas(preview)

    assert list(frame.columns) == preview.columns
    assert list(frame.index) == preview.row_numbers
    assert len(frame) == len(preview.rows) > 0


def test_compare_ranks_models(shop: dx.Dataset) -> None:
    found = shop.ml("income").compare(["linear_regression", "decision_tree"])
    trained = [m for m in found if isinstance(m, TrainedModel)]
    assert {m.model for m in trained} == {"linear_regression", "decision_tree"}
    scores = [m.score for m in trained]
    lower = trained[0].primary in LOWER_IS_BETTER
    assert scores == sorted(scores, reverse=not lower)
    assert not [m for m in found if isinstance(m, Failed)]


def test_unknown_model_is_refused(shop: dx.Dataset) -> None:
    with pytest.raises(dx.ExplorerError):
        shop.ml("bought").compare(["nope"])


def test_time_series_choices_and_analyses(sales: dx.Dataset) -> None:
    series = sales.timeseries()
    setup = series.setup()
    assert setup.time == "date" and setup.period in ("day", "week", "month")
    assert series.stationarity().methods
    assert series.decomposition() is not None


def test_kpis_are_suggested_then_computed(sales: dx.Dataset) -> None:
    suggestions = sales.kpi_suggestions()
    assert suggestions.kpis
    result = sales.kpi(suggestions.kpis[0].definition)
    assert result is not None


def test_synthetic_rows_and_their_report(tmp_path: Path) -> None:
    ds = dx.open(SAMPLES / "typing_traps.csv", cache_dir=tmp_path)
    synth = ds.synthetic(use_gpu=False)
    plan = synth.plan()
    assert any(c.treatment == "model" for c in plan.columns)
    made = synth.generate(rows=200)
    assert len(made.data) == 200
    assert made.report.scores.quality is not None and made.report.scores.quality > 0.7
    copy = made.to_dataset()
    assert len(copy) == 200
    assert copy.column("age").type == ds.column("age").type
    written = made.save(tmp_path / "synthetic.csv")
    assert len(pd.read_csv(written)) == 200
