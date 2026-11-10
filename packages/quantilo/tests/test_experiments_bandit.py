"""049 Experiments, bandit view: posteriors, P(best) and expected loss, the replay of a Thompson
sampling on a randomized test, and the Plan simulation "fixed split vs bandit"."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import quantilo as dx
from quantilo.experiments import bandit as bd
from quantilo.experiments.plan import SampleSizeRequest


def ones(n: int, hits: int) -> np.ndarray:
    return np.r_[np.ones(hits), np.zeros(n - hits)]


def load(frame: pd.DataFrame, tmp: Path, name: str = "exp") -> dx.Dataset:
    return dx.from_pandas(frame, name, cache_dir=tmp)


@pytest.fixture(scope="module")
def conversion(tmp_path_factory: pytest.TempPathFactory) -> dx.Dataset:
    """10,000 rows per variant, 11.1 % vs 12.4 % (034's first scenario)."""
    frame = pd.DataFrame(
        {
            "group": ["A"] * 10_000 + ["B"] * 10_000,
            "converted": np.r_[ones(10_000, 1_110), ones(10_000, 1_240)].astype(bool),
        }
    )
    return load(frame, tmp_path_factory.mktemp("conversion"))


def dated(tmp: Path, days: int = 20, per_day: int = 400, rates=(0.10, 0.16)) -> dx.Dataset:
    """A randomized test over `days` days: B converts clearly more than A."""
    rng = np.random.default_rng(7)
    n = days * per_day
    group = rng.choice(["A", "B"], size=n)
    day = np.repeat(np.arange(days), per_day)
    p = np.where(group == "A", rates[0], rates[1])
    frame = pd.DataFrame(
        {
            "group": group,
            "date": pd.Timestamp("2026-01-01") + pd.to_timedelta(day, unit="D"),
            "converted": (rng.random(n) < p).astype(int),
            "basket": rng.normal(50, 10, n) + np.where(group == "B", 1.0, 0.0),
        }
    )
    return load(frame, tmp, "dated")


# --- P(best) and expected loss ------------------------------------------------------------------


def test_rate_posterior_gives_p_best_and_loss(conversion: dx.Dataset) -> None:
    view = conversion.experiment(variant="group", metrics=["converted"]).bandit()

    assert view is not None and view.kind == "rate" and view.prior.startswith("Beta(1, 1)")
    a, b = view.arms
    assert a.variant == "A" and a.control and b.variant == "B"
    # normal approximation of the two Beta posteriors
    se = np.hypot(np.sqrt(0.111 * 0.889 / 10_000), np.sqrt(0.124 * 0.876 / 10_000))
    assert b.p_best == pytest.approx(stats.norm.cdf(0.013 / se), abs=0.005)
    assert a.p_best + b.p_best == pytest.approx(1.0)
    assert b.expected_loss < 1e-4
    assert a.expected_loss == pytest.approx(0.013, abs=0.001)
    assert b.mean == pytest.approx(1241 / 10_002)
    assert b.low < b.mean < b.high
    assert view.leader == "B"
    assert view.sentence.startswith("P(B is best) = 99.") or view.sentence.startswith(
        "P(B is best) > 99.9 %"
    )
    assert "expected loss if you ship it: < 0.01 pt" in view.sentence
    assert view.guardrail == bd.GUARDRAIL
    assert view.replay is None and "date" in (view.replay_reason or "")


def test_the_view_is_deterministic(conversion: dx.Dataset) -> None:
    first = bd.view(
        "m", "mean", ["A", "B"], [np.arange(50.0), np.arange(50.0) + 1], [0.5, 0.5], None
    )
    second = bd.view(
        "m", "mean", ["A", "B"], [np.arange(50.0), np.arange(50.0) + 1], [0.5, 0.5], None
    )
    assert dx.to_dict(first) == dx.to_dict(second)


def test_equal_means_split_p_best() -> None:
    values = np.random.default_rng(3).normal(10, 2, 400)
    view = bd.view("basket", "mean", ["A", "B"], [values, values.copy()], [0.5, 0.5], None)

    assert view.prior.startswith("Normal")
    assert view.arms[0].p_best == pytest.approx(0.5, abs=0.02)
    se = values.std(ddof=1) / np.sqrt(400) * np.sqrt(2)
    # E[max(X, Y) - X] for X - Y ~ N(0, s^2): s / sqrt(2 pi)
    assert view.arms[0].expected_loss == pytest.approx(se / np.sqrt(2 * np.pi), rel=0.05)


def test_three_variants_sum_to_one() -> None:
    rng = np.random.default_rng(4)
    values = [rng.normal(m, 1, 300) for m in (0.0, 0.1, 0.5)]
    view = bd.view("x", "mean", ["A", "B", "C"], values, [1 / 3] * 3, None)

    assert sum(arm.p_best for arm in view.arms) == pytest.approx(1.0)
    assert view.leader == "C" and view.arms[2].p_best > 0.99


def test_a_ratio_gets_no_view(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {"group": ["A", "B"] * 50, "revenue": np.arange(100.0), "orders": np.arange(100) % 3 + 1.0}
    )
    exp = load(frame, tmp_path).experiment(
        "group", [{"column": "revenue", "denominator": "orders"}]
    )
    result = exp.analyze()

    assert result.bandit is None and "ratio" in (result.bandit_reason or "")
    assert exp.bandit() is None


# --- The replay ---------------------------------------------------------------------------------


def test_replay_moves_traffic_to_the_better_variant(tmp_path: Path) -> None:
    ds = dated(tmp_path)
    view = ds.experiment("group", ["converted"], time="date").bandit()

    assert view is not None and view.replay is not None
    replay = view.replay
    assert replay.grain == "day" and len(replay.periods) == 20 and replay.column == "date"
    a_share, b_share = (s.shares for s in replay.shares)
    assert a_share[0] == pytest.approx(0.5) and b_share[0] == pytest.approx(0.5)
    assert b_share[-1] > 0.9 and np.mean(b_share) > 0.75
    assert replay.winner == "B" and replay.fixed == [0.5, 0.5]
    assert replay.gained > 0 and replay.gained == pytest.approx(
        replay.value_bandit - replay.value_fixed
    )
    assert replay.regret_bandit[-1] < replay.regret_fixed[-1]
    assert replay.regret_fixed[-1] - replay.regret_bandit[-1] == pytest.approx(replay.gained)
    assert sum(replay.units) == 8_000
    assert "a bandit would have sent" in replay.sentence and "converted" in replay.sentence
    assert "randomized" in replay.note


def test_replay_of_a_mean_counts_the_metric(tmp_path: Path) -> None:
    ds = dated(tmp_path)
    view = ds.experiment("group", ["basket"], time="date").bandit()

    assert view is not None and view.kind == "mean" and view.replay is not None
    assert "basket" in view.replay.sentence


def test_replay_with_a_single_period_is_skipped() -> None:
    values = [ones(30, 3), ones(30, 9)]
    timeline = bd.Timeline("date", "day", ["2026-01-01"], values, [np.zeros(30), np.zeros(30)])
    view = bd.view("converted", "rate", ["A", "B"], values, [0.5, 0.5], timeline)

    assert view.replay is None and "one period" in (view.replay_reason or "")


# --- Plan: fixed split vs bandit ----------------------------------------------------------------


def test_simulation_earns_with_a_bandit_and_is_deterministic() -> None:
    request = SampleSizeRequest(baseline=0.10, effect=0.20, daily=1_000)
    sim = dx.bandit_simulation(baseline=0.10, effect=0.20, daily=1_000)

    assert sim.best == "B" and sim.labels == ["control", "B"]
    assert sim.grain == "day" and sim.units == dx.sample_size(0.10, 0.20, daily=1_000).total
    assert sim.gained > 0 and sim.gained_low <= sim.gained <= sim.gained_high
    assert sim.best_share_bandit > sim.best_share_fixed == pytest.approx(0.5)
    assert sim.regret_bandit[-1] < sim.regret_fixed[-1]
    assert 0 <= sim.sure_bandit <= 1 and 0 <= sim.sure_fixed <= 1
    assert sim.runs == bd.SIM_RUNS and sim.seed == bd.SEED
    assert "conversions" in sim.sentence and sim.guardrail == bd.GUARDRAIL
    assert dx.to_dict(bd.simulate(request)) == dx.to_dict(sim)


def test_simulation_of_a_negative_effect_favours_the_control() -> None:
    sim = dx.bandit_simulation(baseline=0.10, effect=-0.20, daily=2_000)

    assert sim.best == "control" and sim.best_share_bandit > 0.5


def test_simulation_without_daily_traffic_uses_batches_and_means() -> None:
    sim = dx.bandit_simulation(baseline=50, sd=10, effect=0.04, kind="mean", variants=3)

    assert sim.grain == "batch" and len(sim.per_period) == bd.SIM_BATCHES
    assert sim.labels == ["control", "B", "C"] and sim.best == "B"
    assert sim.best_share_fixed == pytest.approx(1 / 3)
    assert sum(sim.per_period) == sim.units


def test_long_horizons_are_grouped() -> None:
    sim = dx.bandit_simulation(baseline=0.10, effect=0.02, daily=100)

    assert len(sim.per_period) == bd.SIM_MAX_PERIODS and sim.grain == "batch"
    assert sum(sim.per_period) == sim.units


def test_results_as_tables(conversion: dx.Dataset) -> None:
    view = conversion.experiment(variant="group", metrics=["converted"]).bandit()
    table = dx.to_pandas(view)

    assert list(table["variant"]) == ["A", "B"] and "p_best" in table
    curve = dx.to_pandas(dx.bandit_simulation(baseline=0.10, effect=0.20, daily=1_000))
    assert {"units", "regret_fixed", "regret_bandit"} <= set(curve.columns)
