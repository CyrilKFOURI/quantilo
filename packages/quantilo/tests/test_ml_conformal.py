"""057: conformal prediction intervals of a number (split conformal, CQR), their coverage checked
on held-out rows that did not set the width."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import quantilo as dx
from quantilo.ml import conformal


def test_the_conformal_quantile_is_the_rank_ceil_n_plus_1_level() -> None:
    scores = np.arange(1.0, 11.0)  # 1 … 10
    assert conformal.quantile(scores, 0.9) == 10.0  # ⌈11 x 0.9⌉ = 10
    assert conformal.quantile(scores, 0.5) == 6.0  # ⌈5.5⌉ = 6
    assert conformal.quantile(scores, 0.95) == float("inf")  # ⌈10.45⌉ = 11 > 10


def test_rows_needed_grow_with_the_level() -> None:
    assert conformal.needed(0.9) == 20
    assert conformal.needed(0.95) == 38
    assert conformal.needed(0.99) == 198


def test_too_few_rows_give_no_interval() -> None:
    actual = np.arange(10.0)
    assert conformal.split(actual, actual + 1) is None


def test_split_conformal_covers_its_level_on_exchangeable_rows() -> None:
    rng = np.random.default_rng(0)
    predicted = rng.normal(0, 1, 4_000)
    actual = predicted + rng.normal(0, 2, 4_000)
    found = conformal.split(actual, predicted, 0.9)
    assert found is not None and found.method == "split"
    assert found.calibration_rows == 2_000 and found.check_rows == 2_000
    assert found.coverage == pytest.approx(0.9, abs=0.03)
    assert not found.below
    # the half-width of a normal error of sd 2 at 90 %: 1.645 x 2
    assert found.half_width == pytest.approx(3.29, rel=0.08)
    assert any(r.startswith("R-ML-CONF-01") for r in found.rules)


def test_a_drift_shows_as_a_lower_coverage_in_time_order() -> None:
    """R-ML-CONF-02: errors grow with time; the earlier half sets a width too narrow for the later
    half, and the page warns (R-ML-CONF-03)."""
    rng = np.random.default_rng(1)
    n = 2_000
    spread = np.linspace(1, 4, n)
    predicted = np.zeros(n)
    actual = rng.normal(0, spread)
    found = conformal.split(actual, predicted, 0.9, chronological=True)
    assert found is not None and found.order == "time"
    assert found.coverage is not None and found.coverage < 0.8
    assert found.below


def test_cqr_widens_where_the_target_spreads() -> None:
    rng = np.random.default_rng(2)
    x = rng.uniform(0, 10, (3_000, 1))
    y = x[:, 0] + rng.normal(0, 0.2 + 0.5 * x[:, 0])
    lower, upper = conformal.fit_bands(x[:2_000], y[:2_000], 0.9)
    found = conformal.cqr(lower, upper, x[2_000:], y[2_000:], 0.9)
    assert found is not None
    result, bands = found
    assert result.method == "cqr"
    assert result.coverage == pytest.approx(0.9, abs=0.05)
    low, high = bands.interval(np.array([[1.0], [9.0]]))
    assert (high - low)[1] > 2 * (high - low)[0]  # wider where the noise is larger


def frame(n: int = 1_200, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    area = rng.uniform(20, 200, n)
    rooms = rng.integers(1, 6, n)
    price = 3 * area + 10 * rooms + rng.normal(0, 0.2 * area, n)
    return pd.DataFrame({"area": area, "rooms": rooms, "price": price})


def test_a_trained_number_shows_its_interval_and_coverage(tmp_path: Path) -> None:
    data = dx.from_pandas(frame(), "houses", cache_dir=tmp_path)
    model = data.ml("price").train("linear_regression")
    interval = model.result.interval
    assert interval is not None and interval.method == "split" and interval.level == 0.9
    assert interval.rows == model.result.test_rows == 240
    assert interval.coverage is not None and 0.8 <= interval.coverage <= 1.0
    card = model.card.interval
    assert card is not None and card.half_width == interval.half_width
    assert card.coverage == interval.coverage


def test_the_level_is_chosen_in_the_validation(tmp_path: Path) -> None:
    data = dx.from_pandas(frame(), "houses-80", cache_dir=tmp_path)
    narrow = data.ml("price", validation={"interval_level": 0.8}).train("linear_regression")
    wide = data.ml("price", validation={"interval_level": 0.95}).train("linear_regression")
    assert narrow.result.interval is not None and wide.result.interval is not None
    assert narrow.result.interval.half_width < wide.result.interval.half_width


def test_cqr_is_kept_on_the_model_for_new_rows(tmp_path: Path) -> None:
    data = dx.from_pandas(frame(), "houses-cqr", cache_dir=tmp_path)
    model = data.ml("price", validation={"interval_method": "cqr"}).train("linear_regression")
    interval = model.result.interval
    assert interval is not None and interval.method == "cqr"
    assert model.card.interval is not None and model.card.interval.method == "cqr"
    fresh = dx.from_pandas(frame(400, seed=9).drop(columns="price"), "new", cache_dir=tmp_path)
    rows = model.predict(fresh)
    widths = (rows["high"] - rows["low"]).to_numpy()
    assert widths.std() > 0  # adaptive: not the same width everywhere
    assert bool((rows["low"].to_numpy() <= rows["high"].to_numpy()).all())


def test_cqr_under_cross_validation_falls_back_to_split(tmp_path: Path) -> None:
    data = dx.from_pandas(frame(), "houses-cv", cache_dir=tmp_path)
    model = data.ml("price", validation={"method": "kfold", "interval_method": "cqr"}).train(
        "linear_regression"
    )
    interval = model.result.interval
    assert interval is not None and interval.method == "split"
    assert interval.rows == 1_200  # every out-of-fold row
    assert any(r.startswith("R-ML-CONF-04") for r in interval.rules)
