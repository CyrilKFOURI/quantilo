"""062: Kaplan-Meier, log-rank, Markov chains and their detection rules, against known values."""

from datetime import datetime, timedelta

import duckdb
import numpy as np
import pyarrow as pa
import pytest

from quantilo.compare import durations as du
from quantilo.compare import survival as sv
from quantilo.compare import transitions as tr
from quantilo.typing.types import Parser

RAW = Parser(kind="raw")

# Freireich et al. (1963), 6-MP against placebo, remission in weeks: the textbook example.
DRUG = [6, 6, 6, 6, 7, 9, 10, 10, 11, 13, 16, 17, 19, 20, 22, 23, 25, 32, 32, 34, 35]
DRUG_EVENT = [1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
PLACEBO = [1, 1, 2, 2, 3, 4, 4, 5, 5, 8, 8, 8, 8, 11, 11, 12, 12, 15, 17, 22, 23]


def test_kaplan_meier_matches_the_textbook_values() -> None:
    km = sv.kaplan_meier(np.array(DRUG), np.array(DRUG_EVENT, dtype=bool))
    expected = {6: 0.857, 7: 0.807, 10: 0.753, 13: 0.690, 16: 0.627, 22: 0.538, 23: 0.448}
    for time, value in expected.items():
        assert sv.survival_at(km, time) == pytest.approx(value, abs=1e-3)
    assert km.times[0] == 0 and km.survival[0] == 1
    assert km.median == 23
    assert km.n == 21 and km.total_events == 9
    # Greenwood at t = 6: S^2 * 3 / (21 * 18)
    s6 = 18 / 21
    se = np.sqrt(3 / (21 * 18))
    low = s6 ** np.exp(1.959964 * se / abs(np.log(s6)))  # R, conf.type = "log-log": 0.620
    high = s6 ** np.exp(-1.959964 * se / abs(np.log(s6)))  # 0.952
    i = km.times.index(6)
    assert km.lower[i] == pytest.approx(low, rel=1e-6)
    assert km.upper[i] == pytest.approx(high, rel=1e-6)
    assert km.lower[i] < km.survival[i] < km.upper[i]
    assert km.median_lower == 13  # R: 0.95LCL 13 (the lower band crosses 0.5)
    assert km.median_upper is None  # R: UCL NA (the upper band never does)


def test_kaplan_meier_without_censoring_is_the_empirical_survival() -> None:
    km = sv.kaplan_meier(np.array(PLACEBO), np.ones(21, dtype=bool))
    for time in [1, 5, 8, 12, 23]:
        assert sv.survival_at(km, time) == pytest.approx(np.mean(np.array(PLACEBO) > time))
    assert km.median == 8


def test_log_rank_matches_survdiff_on_freireich() -> None:
    durations = np.array(DRUG + PLACEBO, dtype=float)
    events = np.array(DRUG_EVENT + [1] * 21, dtype=bool)
    groups = np.array(["6-MP"] * 21 + ["placebo"] * 21)
    test = sv.log_rank(durations, events, groups)
    assert test is not None
    assert test.statistic == pytest.approx(16.79, abs=0.01)  # R: survdiff, Chisq = 16.8
    assert test.df == 1
    assert test.p_value == pytest.approx(4.17e-5, rel=0.01)
    assert test.observed == [9, 21]
    assert test.expected[0] == pytest.approx(19.25, abs=0.01)


def test_log_rank_three_groups_and_same_curve() -> None:
    rng = np.random.default_rng(0)
    d = rng.exponential(10, 900)
    groups = np.repeat(["a", "b", "c"], 300)
    test = sv.log_rank(d, np.ones(900, dtype=bool), groups)
    assert test is not None and test.df == 2 and test.p_value > 0.01
    assert sv.log_rank(d, np.ones(900, dtype=bool), np.repeat("a", 900)) is None


def test_absorbing_chain_drunkards_walk() -> None:
    # 0 and 4 absorbing, 1-3 step left or right with probability 1/2: 3, 4, 3 steps on average
    counts = np.array(
        [
            [5, 0, 0, 0, 0],
            [10, 0, 10, 0, 0],
            [0, 10, 0, 10, 0],
            [0, 0, 10, 0, 10],
            [0, 0, 0, 0, 0],  # never left: absorbing too (R-MKV-03)
        ]
    )
    chain = tr.chain(counts, ["0", "1", "2", "3", "4"])
    assert chain.absorbing == ["0", "4"]
    assert chain.never_left == ["4"]
    assert chain.steps_to_absorption == pytest.approx({"1": 3.0, "2": 4.0, "3": 3.0})
    assert chain.absorption["1"] == pytest.approx({"0": 0.75, "4": 0.25})
    assert chain.stationary is None


def test_stationary_distribution_of_a_regular_chain() -> None:
    p = np.array([[0.9, 0.1], [0.5, 0.5]])
    counts = (p * 1000).astype(int)
    chain = tr.chain(counts, ["a", "b"])
    assert chain.stationary == pytest.approx([5 / 6, 1 / 6], abs=1e-9)
    assert chain.stay_steps == pytest.approx([10.0, 2.0])
    pi = np.array(chain.stationary)
    assert pi @ np.array(chain.probabilities) == pytest.approx(pi)


def test_count_transitions_pairs_rows_of_the_same_entity() -> None:
    ids = np.array(["x", "x", "x", "y", "y"])
    states = np.array(["a", "b", "b", "b", "a"])
    counts = tr.count_transitions(ids, states, ["a", "b"])
    assert counts.tolist() == [[0, 1], [1, 1]]  # y's first row is not x's next


def column(name: str, kind: str, **extra) -> du.DurationColumn:
    return du.DurationColumn(name, kind, RAW, **extra)  # type: ignore[arg-type]


def test_detection_rules_and_their_reasons() -> None:
    telco = [
        column("customerID", "identifier", distinct=100, rows=100),
        column("tenure", "numeric_discrete", role="duration", distinct=73, minimum=0),
        column("MonthlyCharges", "numeric_continuous"),
        column("Contract", "categorical_nominal", distinct=3),
        column("Churn", "boolean", distinct=2),
    ]
    found = du.detect_survival(telco)
    assert found is not None and found.kind == "duration"
    assert (found.duration, found.event) == ("tenure", "Churn")
    assert "R-SRV-01" in found.reason and "Check that your data" in found.reason
    dates = [
        column("SignupDate", "datetime"),
        column("cancel_date", "datetime"),
        column("plan", "categorical_nominal", distinct=3),
    ]
    found = du.detect_survival(dates)
    assert found is not None and (found.start, found.end) == ("SignupDate", "cancel_date")
    assert "R-SRV-02" in found.reason
    # a number and a yes/no, but nothing says duration or event: no survival
    assert (
        du.detect_survival([column("Age", "numeric_continuous"), column("Exited", "boolean")])
        is None
    )
    assert (
        du.detect_survival([column("price", "numeric_continuous"), column("flag", "boolean")])
        is None
    )


def test_transition_detection_needs_a_repeating_key_and_a_status() -> None:
    columns = [
        column("customer_id", "categorical_nominal", role="customer_key", distinct=50, rows=400),
        column("month", "datetime"),
        column("status", "categorical_nominal", role="status", distinct=4),
    ]
    found = du.detect_transitions(columns)
    assert found is not None and "R-MKV-01" in found.reason
    unique = [column("customer_id", "identifier", distinct=400, rows=400), *columns[1:]]
    assert du.detect_transitions(unique) is None
    geography = [*columns[:2], column("State", "categorical_nominal", role="region", distinct=4)]
    assert du.detect_transitions(geography) is None


@pytest.fixture()
def con() -> duckdb.DuckDBPyConnection:
    connection = duckdb.connect()
    durations = DRUG + PLACEBO
    table = pa.table(
        {
            "weeks": [float(d) for d in durations],
            "relapse": ["Yes" if e else "No" for e in DRUG_EVENT + [1] * 21],
            "arm": ["6-MP"] * 21 + ["placebo"] * 21,
            "start_date": [datetime(2024, 1, 1)] * 42,
            "end_date": [
                datetime(2024, 1, 1) + timedelta(days=d) if e else None
                for d, e in zip(durations, DRUG_EVENT + [1] * 21, strict=True)
            ],
        }
    )
    connection.register("t", table)
    connection.execute("CREATE TABLE data AS SELECT * FROM t")
    return connection


def by_name(*columns: du.DurationColumn) -> dict[str, du.DurationColumn]:
    return {c.name: c for c in columns}


def test_survival_from_a_duration_and_an_event(con) -> None:
    columns = by_name(
        column("weeks", "numeric_continuous"),
        column("relapse", "categorical_nominal", distinct=2),
        column("arm", "categorical_nominal", distinct=2),
    )
    setup = du.SurvivalSetup(kind="duration", duration="weeks", event="relapse")
    result = du.survival(con, "data", setup, columns, group="arm")
    assert setup.event_value == "Yes" and "R-SRV-03" in (setup.event_reason or "")
    assert [c.label for c in result.curves] == ["6-MP", "placebo"]
    assert result.curves[0].median == 23 and result.curves[1].median == 8
    assert result.log_rank is not None and result.log_rank.significant
    assert result.rows == 42 and result.events == 30 and result.censored == 12
    assert "differ" in (result.sentence or "")
    single = du.survival(con, "data", setup, columns)
    assert len(single.curves) == 1 and single.sentence.startswith("Half reach the event")


def test_survival_from_two_dates_censors_open_ends(con) -> None:
    columns = by_name(column("start_date", "datetime"), column("end_date", "datetime"))
    setup = du.SurvivalSetup(kind="dates", start="start_date", end="end_date")
    result = du.survival(con, "data", setup, columns)
    assert result.events == 30 and result.censored == 12
    assert setup.unit == "days" and "R-SRV-02" in (setup.event_reason or "")


def test_event_level_rules() -> None:
    col = column("Churn", "categorical_nominal", distinct=2)
    assert du._event_level(col, {"No": 70, "Yes": 30}, None)[0] == "Yes"
    censor = column("censored", "categorical_nominal", distinct=2)
    assert du._event_level(censor, {"0": 30, "1": 70}, None)[0] == "0"
    odd = column("result", "categorical_nominal", distinct=2)
    level, why = du._event_level(odd, {"A": 80, "B": 20}, None)
    assert level == "B" and "rarer" in why
    assert du._event_level(odd, {"A": 80, "B": 20}, "A")[0] == "A"


def test_transitions_end_to_end() -> None:
    rng = np.random.default_rng(1)
    rows = []
    p = {"trial": [0.5, 0.3, 0.2], "paid": [0.0, 0.9, 0.1]}
    names = ["trial", "paid", "churned"]
    for customer in range(300):
        state = "trial"
        for month in range(12):
            rows.append((f"c{customer}", datetime(2024, 1, 1) + timedelta(days=30 * month), state))
            if state == "churned":
                break
            state = names[rng.choice(3, p=p[state])]
    connection = duckdb.connect()
    table = pa.table(
        {
            "customer_id": [r[0] for r in rows],
            "month": [r[1] for r in rows],
            "status": [r[2] for r in rows],
        }
    )
    connection.register("t", table)
    columns = by_name(
        column("customer_id", "categorical_nominal", distinct=300, rows=len(rows)),
        column("month", "datetime"),
        column("status", "categorical_nominal", distinct=3),
    )
    setup = du.TransitionSetup("customer_id", "month", "status")
    result = du.transitions(connection, "t", setup, columns)
    assert result.chain is not None
    assert result.chain.absorbing == ["churned"]
    assert result.step_days == pytest.approx(30)
    # theory: from trial, N = (I - Q)^-1 with Q = [[.5, .3], [0, .9]]: 2 + 0.6 * 10 = 8 steps
    assert result.chain.steps_to_absorption["trial"] == pytest.approx(8, rel=0.15)
    assert "churned is never left" in (result.sentence or "")


def test_fixed_attribute_is_not_a_status() -> None:
    connection = duckdb.connect()
    table = pa.table(
        {
            "id": ["a", "a", "b", "b"],
            "t": [1.0, 2.0, 1.0, 2.0],
            "s": ["x", "x", "y", "y"],
        }
    )
    connection.register("t", table)
    columns = by_name(
        column("id", "categorical_nominal", distinct=2, rows=4),
        column("t", "numeric_continuous"),
        column("s", "categorical_nominal", distinct=2),
    )
    result = du.transitions(connection, "t", du.TransitionSetup("id", "t", "s"), columns)
    assert result.chain is None and "fixed attribute" in (result.message or "")


def test_detected_on_a_real_import(tmp_path) -> None:
    """Through the whole pipeline: import, column types and roles, then the rules."""
    import pandas as pd

    import quantilo as dx

    rng = np.random.default_rng(4)
    n = 400
    contract = rng.choice(["Month-to-month", "Two year"], n)
    tenure = np.where(contract == "Two year", rng.exponential(60, n), rng.exponential(12, n))
    frame = pd.DataFrame(
        {
            "tenure": np.minimum(tenure, 72).round().astype(int),
            "Contract": contract,
            "Churn": np.where(tenure < 72, "Yes", "No"),
        }
    )
    data = dx.from_pandas(frame, "telco", cache_dir=tmp_path)
    found = data.survival()
    assert found.setup is not None and found.setup.duration == "tenure"
    assert found.setup.event_value in {"Yes", "True"}
    grouped = data.survival("tenure", "Churn", group="Contract")
    assert grouped.log_rank is not None and grouped.log_rank.significant
    assert data.transitions().setup is None
