import math
from typing import Any

import duckdb
import numpy as np
import pyarrow as pa
import pytest
from scipy import stats

from quantilo.correlations.insights import insights_for
from quantilo.correlations.matrix import (
    MatrixColumn,
    association_matrix,
    cramers_v,
    eta,
)
from quantilo.typing.types import AnalysisType, Parser

RAW = Parser(kind="raw")
N = 2_000


@pytest.fixture(scope="module")
def connection() -> duckdb.DuckDBPyConnection:
    rng = np.random.default_rng(11)
    x = rng.normal(0, 1, N)
    group = rng.choice(["a", "b", "c"], size=N, p=[0.8, 0.15, 0.05])
    shift = {"a": 0.0, "b": 2.0, "c": 4.0}
    income = rng.lognormal(10, 1.2, N)
    missing = np.where(rng.random(N) < 0.3, np.nan, rng.normal(0, 1, N))
    table = pa.table(
        {
            "file_row_number": pa.array(range(N)),
            "x": x,
            "y": x + rng.normal(0, 0.8, N),  # r ≈ 0.78: strong, not redundant
            "twin": 2 * x + 1,
            "noise": rng.normal(0, 1, N),
            "group": group,
            "score": [rng.normal(shift[g], 1) for g in group],
            "linked": [g if rng.random() < 0.8 else "a" for g in group],
            "income": income,
            "sparse": pa.array(missing, from_pandas=True),
            "label": [f"id{i}" for i in range(N)],
        }
    )
    con = duckdb.connect()
    con.register("t", table)
    con.execute("CREATE TABLE data AS SELECT * FROM t")
    return con


TYPES: dict[str, AnalysisType] = {
    "x": "numeric_continuous",
    "y": "numeric_continuous",
    "twin": "numeric_continuous",
    "noise": "numeric_continuous",
    "group": "categorical_nominal",
    "score": "numeric_continuous",
    "linked": "categorical_nominal",
    "income": "numeric_continuous",
    "sparse": "numeric_continuous",
    "label": "identifier",
}
COLUMNS = [MatrixColumn(name, kind, RAW) for name, kind in TYPES.items()]


def cell(matrix, a: str, b: str):
    names = [c.name for c in matrix.columns]
    return matrix.values[names.index(a)][names.index(b)]


def test_metrics_follow_column_types(connection: duckdb.DuckDBPyConnection) -> None:
    matrix, _ = association_matrix(connection, "data", COLUMNS)
    data = connection.execute("SELECT x, y, noise FROM data").fetchnumpy()

    assert [e.name for e in matrix.excluded] == ["label"]
    assert cell(matrix, "x", "y") == pytest.approx(stats.pearsonr(data["x"], data["y"])[0])
    names = [c.name for c in matrix.columns]
    i, j = names.index("x"), names.index("group")
    assert matrix.metrics[i][j] == "eta"
    assert matrix.metrics[names.index("group")][names.index("linked")] == "cramers_v"
    assert all(matrix.values[k][k] == 1.0 for k in range(len(names)))
    assert matrix.values == [list(row) for row in zip(*matrix.values, strict=True)]  # symmetric


def test_spearman_uses_ranks(connection: duckdb.DuckDBPyConnection) -> None:
    matrix, _ = association_matrix(connection, "data", COLUMNS, method="spearman")
    data = connection.execute("SELECT x, income FROM data").fetchnumpy()

    rho: Any = stats.spearmanr(data["x"], data["income"])
    assert cell(matrix, "x", "income") == pytest.approx(rho.statistic, abs=1e-9)


def test_pairs_skip_missing_values(connection: duckdb.DuckDBPyConnection) -> None:
    matrix, _ = association_matrix(connection, "data", COLUMNS)
    names = [c.name for c in matrix.columns]

    n = matrix.n[names.index("x")][names.index("sparse")]
    assert 0.6 * N < n < 0.8 * N


def test_eta_matches_its_definition() -> None:
    rng = np.random.default_rng(1)
    groups = rng.integers(0, 3, 500)
    values = groups * 1.5 + rng.normal(0, 1, 500)

    value, p, n = eta(values, groups)

    mean = values.mean()
    between = sum((groups == g).sum() * (values[groups == g].mean() - mean) ** 2 for g in range(3))
    total = ((values - mean) ** 2).sum()
    anova: Any = stats.f_oneway(*(values[groups == g] for g in range(3)))
    assert value == pytest.approx(math.sqrt(between / total))
    assert p == pytest.approx(anova.pvalue, rel=1e-6)
    assert n == 500


def test_cramers_v_is_bias_corrected() -> None:
    rng = np.random.default_rng(2)
    independent_a = rng.integers(0, 4, 5_000)
    independent_b = rng.integers(0, 4, 5_000)

    unrelated, p, _ = cramers_v(independent_a, independent_b)
    same, _, _ = cramers_v(independent_a, independent_a)

    assert unrelated is not None and unrelated < 0.03  # uncorrected V would be biased upward
    assert p is not None and p > 0.01
    assert same == pytest.approx(1.0)


def test_clustering_groups_related_variables(connection: duckdb.DuckDBPyConnection) -> None:
    matrix, _ = association_matrix(connection, "data", COLUMNS)
    names = [c.name for c in matrix.columns]
    order = [names[i] for i in matrix.clustered]

    positions = sorted(order.index(name) for name in ("x", "y", "twin"))
    assert positions[2] - positions[0] == 2  # the three linked variables sit together
    assert sorted(matrix.clustered) == list(range(len(names)))


def test_insights(connection: duckdb.DuckDBPyConnection) -> None:
    columns = [c for c in COLUMNS if c.type != "identifier"]
    matrix, loaded = association_matrix(connection, "data", columns)

    found = insights_for(columns, matrix, loaded)

    kinds = {(i.kind, tuple(i.columns)) for i in found}
    assert ("redundant", ("x", "twin")) in kinds
    assert ("correlated", ("x", "y")) in kinds
    assert ("missing", ("sparse",)) in kinds
    assert ("skewed", ("income",)) in kinds
    assert ("imbalanced", ("group",)) in kinds
    assert any(kind == "group_difference" and "score" in cols for kind, cols in kinds)
    assert any(kind == "associated" for kind, _ in kinds)
    assert found == sorted(found, key=lambda insight: -insight.score)
    assert len({i.id for i in found}) == len(found)
    assert all(i.evidence for i in found if i.kind in ("correlated", "group_difference"))
    assert all(i.evidence is None for i in found if i.kind in ("missing", "skewed"))
    assert all(
        pair.q_value is not None and pair.q_value >= (pair.p_value or 0) for pair in matrix.pairs
    )
    redundant = next(i for i in found if i.kind == "redundant")
    assert redundant.link.page == "compare" and redundant.link.y == "twin"


def test_column_limit_and_choice(connection: duckdb.DuckDBPyConnection) -> None:
    matrix, _ = association_matrix(connection, "data", COLUMNS, chosen=["x", "y", "label"])

    assert [c.name for c in matrix.columns] == ["x", "y"]
    assert [e.name for e in matrix.excluded] == ["label"]


def test_large_tables_are_sampled() -> None:
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE big AS SELECT range AS file_row_number, random() AS a, random() AS b "
        "FROM range(250000)"
    )
    columns = [
        MatrixColumn("a", "numeric_continuous", RAW),
        MatrixColumn("b", "numeric_continuous", RAW),
    ]

    matrix, _ = association_matrix(con, "big", columns)

    assert matrix.sampled and matrix.rows == 200_000 and matrix.total == 250_000


def test_near_unique_categories_are_left_out(connection: duckdb.DuckDBPyConnection) -> None:
    columns = [
        MatrixColumn("x", "numeric_continuous", RAW),
        MatrixColumn("label", "categorical_nominal", RAW),  # every value distinct
    ]

    matrix, _ = association_matrix(connection, "data", columns)

    assert [c.name for c in matrix.columns] == ["x"]
    assert "identifier" in matrix.excluded[0].reason


def test_near_unique_categories_are_left_out_with_spearman(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    columns = [
        MatrixColumn("x", "numeric_continuous", RAW),
        MatrixColumn("y", "numeric_continuous", RAW),
        MatrixColumn("label", "categorical_nominal", RAW),  # every value distinct
    ]

    matrix, _ = association_matrix(connection, "data", columns, method="spearman")

    assert [c.name for c in matrix.columns] == ["x", "y"]
    assert [e.name for e in matrix.excluded] == ["label"]


def graded(a: str, b: str, metric, value: float, p: float):
    from quantilo.correlations.evidence import effect_size, grade
    from quantilo.correlations.matrix import Pair

    return Pair(a, b, metric, value, p, 600, p, grade(effect_size(metric, value), p))


def test_effect_sizes_follow_cohen() -> None:
    from quantilo.correlations.evidence import effect_size

    assert [effect_size("pearson", v) for v in (0.05, -0.2, 0.35, -0.6)] == [
        "negligible",
        "small",
        "medium",
        "large",
    ]
    assert effect_size("eta", 0.25) == "medium" and effect_size("eta", 0.37) == "large"


def test_evidence_crosses_size_and_adjusted_p() -> None:
    from quantilo.correlations.evidence import grade

    assert grade("large", 0.0001) == "strong"
    assert grade("large", 0.01) == "moderate"  # big, real, but not beyond doubt
    assert grade("medium", 0.01) == "moderate"
    assert grade("medium", 0.4) == "weak"  # notable, could be chance
    assert grade("small", 0.001) == "weak"  # real but minor
    assert grade("small", 0.4) == "none"
    assert grade("negligible", 0.0) == "none"
    assert grade("large", None) == "none"


def test_p_values_are_adjusted_for_testing_every_pair() -> None:
    from quantilo.correlations.evidence import adjusted

    q = adjusted([0.01, 0.04, None, 0.03, 0.5])

    assert q[2] is None
    assert q[0] == pytest.approx(0.04)  # 0.01 * 4 tests / rank 1
    assert q[1] == q[3] == pytest.approx(0.04 * 4 / 3)  # monotone: 0.03 shares 0.04's bound
    assert q[4] == pytest.approx(0.5)


def test_link_wording_follows_the_evidence() -> None:
    from quantilo.correlations.insights import pair_insight

    roles = {"price": "num", "city": "cat", "x": "num", "y": "num"}
    chance = pair_insight(graded("price", "city", "eta", 0.57, 0.426), roles)  # many small groups
    real = pair_insight(graded("price", "city", "eta", 0.57, 0.0001), roles)
    small = pair_insight(graded("x", "y", "pearson", -0.15, 0.001), roles)

    assert chance is not None and chance.evidence == "weak"
    assert chance.text.startswith("price seems to differ across the groups of city")
    assert "could be chance" in chance.text and "adj. p = 0.426" in chance.text
    assert real is not None and real.evidence == "strong"
    assert real.text.startswith("price differs strongly across the groups of city")
    assert small is not None and small.evidence == "weak"
    assert small.text.startswith("y falls slightly as x rises")
    assert pair_insight(graded("x", "y", "pearson", 0.02, 0.9), roles) is None
    redundant = pair_insight(graded("x", "y", "pearson", 0.99, 0.0001), roles)
    assert redundant is not None and redundant.kind == "redundant"
