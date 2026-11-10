"""Structure of the numeric columns (078): PCA with parallel analysis, factor analysis, KMO."""

import numpy as np
import pytest

from quantilo.correlations.matrix import Matrix, MatrixMember, Pair
from quantilo.correlations.structure import (
    MIN_NUMERIC,
    bartlett,
    component_label,
    correlation,
    factor_analysis,
    kmo,
    kmo_label,
    ledermann,
    pca_structure,
    prepare,
    structure_rule,
)

N = 2_000


def two_factors(n: int = N, noise: float = 0.5, seed: int = 3) -> dict[str, np.ndarray]:
    """Six columns on two latent factors, three each."""
    rng = np.random.default_rng(seed)
    f, g = rng.normal(size=n), rng.normal(size=n)
    columns = {f"a{i}": f + rng.normal(0, noise, n) for i in range(1, 4)}
    columns.update({f"b{i}": g + rng.normal(0, noise, n) for i in range(1, 4)})
    return columns


def independent(n: int = N, p: int = 8, seed: int = 4) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {f"x{i}": rng.normal(size=n) for i in range(p)}


def test_two_latent_factors_give_two_components() -> None:
    found = pca_structure(two_factors())
    assert found.reason is None
    assert found.retained == 2
    assert found.components[1].cumulative > 0.7
    loadings = np.array(found.loadings)
    assert found.shown == 2
    first = {name for name, row in zip(found.columns, loadings, strict=True) if abs(row[0]) >= 0.5}
    second = {name for name, row in zip(found.columns, loadings, strict=True) if abs(row[1]) >= 0.5}
    assert {frozenset(first), frozenset(second)} == {
        frozenset({"a1", "a2", "a3"}),
        frozenset({"b1", "b2", "b3"}),
    }
    assert found.conclusion.startswith("2 components explain")
    assert "PC1 = " in found.conclusion and "PC2 = " in found.conclusion
    assert "PC3" not in found.conclusion  # only the retained components are named
    assert found.parallel_rows == N
    assert all(c.threshold is not None for c in found.components)


def test_independent_noise_keeps_no_component() -> None:
    found = pca_structure(independent())
    assert found.retained == 0
    assert found.conclusion.startswith("No component stands out from chance")
    assert found.shown == 2  # the first two are still drawn


def test_same_data_same_result() -> None:
    first, second = pca_structure(two_factors()), pca_structure(two_factors())
    assert first.loadings == second.loadings
    assert first.components == second.components
    assert first.biplot == second.biplot


def test_units_do_not_matter() -> None:
    columns = two_factors()
    scaled = {**columns, "a1": columns["a1"] * 1e6}
    assert np.array(pca_structure(scaled).loadings) == pytest.approx(
        np.array(pca_structure(columns).loadings), abs=1e-3
    )


def test_signs_are_fixed_largest_loading_positive() -> None:
    columns = {name: -values for name, values in two_factors().items()}
    for j in range(2):
        loadings = np.array(pca_structure(columns).loadings)[:, j]
        assert loadings[np.abs(loadings).argmax()] > 0


def test_biplot_is_sampled_with_arrows() -> None:
    found = pca_structure(two_factors(n=6_000))
    assert found.biplot is not None
    assert found.biplot.points == 2_000 and found.biplot.of == 6_000
    assert len(found.biplot.x) == 2_000
    assert {a.column for a in found.biplot.arrows} == set(found.columns)
    assert found.parallel_rows == 5_000


def test_constant_and_mostly_missing_columns_are_left_out() -> None:
    columns = two_factors()
    columns["flat"] = np.ones(N)
    holes = columns["a1"].copy()
    holes[: N // 2] = np.nan
    columns["holes"] = holes
    found = pca_structure(columns)
    reasons = {item.name: item.reason for item in found.left_out}
    assert "flat" in reasons and "Constant" in reasons["flat"]
    assert "holes" in reasons and "50% missing" in reasons["holes"]
    assert "flat" not in found.columns


def test_complete_rows_or_medians() -> None:
    columns = two_factors()
    columns["a1"][:100] = np.nan  # 5 % incomplete: rows dropped
    found = prepare(columns)
    assert found.dropped == 100 and found.filled == 0 and len(found.z) == N - 100
    rng = np.random.default_rng(0)
    scattered = two_factors()
    for name in scattered:  # 10 % per column, scattered: about 47 % of rows complete
        scattered[name][rng.random(N) < 0.1] = np.nan
    filled = prepare(scattered)
    assert filled.dropped == 0 and filled.filled > 0 and len(filled.z) == N
    assert "median" in filled.notes[0]


def test_too_few_columns_or_rows_are_refused() -> None:
    columns = two_factors()
    few = {name: columns[name] for name in ("a1", "a2")}
    assert pca_structure(few).reason == "Needs at least 3 usable numeric columns (2 here)."
    short = {name: values[:20] for name, values in columns.items()}
    found = pca_structure(short)
    assert found.reason is not None and "complete rows" in found.reason


def test_label_names_strong_columns_and_contrasts() -> None:
    names = ["a", "b", "c", "d", "e"]
    assert component_label(np.array([0.9, 0.8, -0.7, 0.1, 0.2]), names) == "a, b against c"
    assert component_label(np.array([0.9, 0.8, 0.7, 0.6, 0.1]), names) == "a, b, c…"
    assert component_label(np.array([0.3, 0.1, 0.2, 0.0, 0.0]), names) == "a"


def test_kaiser_is_noted_when_it_differs() -> None:
    found = pca_structure(independent(p=20))
    assert found.kaiser > found.retained
    assert any("Kaiser" in note for note in found.notes)


def test_rule_needs_five_numbers_and_three_strong_pairs() -> None:
    def matrix(numeric: int, strong: int) -> Matrix:
        members = [MatrixMember(f"c{i}", "numeric_continuous", "num") for i in range(numeric)]
        members.append(MatrixMember("city", "categorical_nominal", "cat"))
        pairs = [Pair("c0", f"c{i}", "pearson", 0.8, 0.0, 100) for i in range(strong)]
        pairs.append(Pair("c0", "city", "eta", 0.9, 0.0, 100))  # not a pair of numbers
        pairs.append(Pair("c1", "c2", "pearson", -0.3, 0.0, 100))
        return Matrix("pearson", members, [], [], [], [], 100, 100, False, pairs=pairs)

    assert structure_rule(matrix(MIN_NUMERIC, 3)).proposed
    rule = structure_rule(matrix(MIN_NUMERIC, 3))
    assert rule.reason.startswith("5 numeric columns, 3 pairs with |r| ≥ 0.5")
    assert not structure_rule(matrix(4, 3)).proposed
    assert not structure_rule(matrix(8, 2)).proposed
    assert "3 pairs" in structure_rule(matrix(8, 2)).reason


def test_factor_analysis_recovers_two_factors_after_varimax() -> None:
    found = factor_analysis(two_factors())
    assert found.reason is None
    assert found.k == 2 and found.rotation == "varimax"
    loadings = np.array(found.loadings)
    groups = {"a": [0, 1, 2], "b": [3, 4, 5]}
    for factor in range(2):
        column = np.abs(loadings[:, factor])
        strong = {i for i in range(6) if column[i] >= 0.5}
        assert strong in (set(groups["a"]), set(groups["b"]))
        assert all(column[i] < 0.3 for i in range(6) if i not in strong)
    assert found.kmo is not None and found.kmo > 0.6
    assert found.bartlett_p is not None and found.bartlett_p < 0.001
    assert found.bartlett_df == 15
    assert all(0 < h < 1 for h in found.communalities)
    assert found.communalities == pytest.approx(
        [1 - u for u in found.uniqueness], abs=0.05
    )  # standardised: h² + ψ ≈ 1
    assert "the columns share enough to be factored" in found.conclusion
    assert "2 factors after varimax rotation: F1 = " in found.conclusion


def test_factor_analysis_of_noise_is_discouraged() -> None:
    found = factor_analysis(independent())
    assert found.k == 1 and found.rotation is None
    assert found.kmo is not None and found.kmo < 0.6
    assert "One factor" in found.conclusion
    assert found.bartlett_p is not None and found.bartlett_p > 0.001


def test_factor_count_is_capped_by_ledermann() -> None:
    assert ledermann(3) == 1
    assert ledermann(5) == 2
    assert ledermann(6) == 3
    found = factor_analysis(two_factors(), k=5)
    assert found.k == 3
    assert any("Ledermann" in note for note in found.notes)


def test_kmo_and_bartlett_against_known_values() -> None:
    r = np.array([[1, 0.5, 0.4], [0.5, 1, 0.3], [0.4, 0.3, 1.0]])
    overall, per_column, singular = kmo(r)
    assert not singular
    inverse = np.linalg.inv(r)
    partial = -inverse[0, 1] / np.sqrt(inverse[0, 0] * inverse[1, 1])
    assert partial == pytest.approx(0.38 / np.sqrt(0.84 * 0.91), abs=1e-9)
    r2 = 0.5**2 + 0.4**2 + 0.3**2
    pairs = ((0, 1), (0, 2), (1, 2))
    q = [-inverse[i, j] / np.sqrt(inverse[i, i] * inverse[j, j]) for i, j in pairs]
    expected = r2 / (r2 + sum(v * v for v in q))
    assert overall == pytest.approx(expected, abs=1e-4)
    assert len(per_column) == 3
    chi2, df, p = bartlett(r, 100)
    assert df == 3
    assert chi2 == pytest.approx(-(99 - 11 / 6) * np.log(np.linalg.det(r)), abs=0.01)
    assert p is not None and p < 0.001
    assert kmo_label(0.95) == "marvelous" and kmo_label(0.65) == "mediocre"
    assert kmo_label(0.3) == "unacceptable"


def test_exactly_redundant_columns_do_not_break() -> None:
    columns = two_factors()
    columns["sum"] = columns["a1"] + columns["a2"]
    found = factor_analysis(columns)
    assert found.reason is None and found.kmo is not None
    assert found.bartlett_chi2 is None
    assert any("redundant" in note for note in found.notes)
    assert pca_structure(columns).reason is None


def test_large_input_is_bounded() -> None:
    columns = two_factors(n=60_000)
    found = factor_analysis(columns, retained=2)
    assert found.rows == 20_000 and found.sampled
    assert any("Bartlett's test rejects almost any" in note for note in found.notes)


def test_correlation_of_z_scores_is_pearson() -> None:
    columns = two_factors()
    z = prepare(columns).z
    expected = np.corrcoef(np.column_stack(list(columns.values())), rowvar=False)
    assert correlation(z) == pytest.approx(expected, abs=1e-9)


def test_library_structure_and_factors(tmp_path) -> None:
    import pandas as pd

    import quantilo as dx

    frame = pd.DataFrame(two_factors(n=600))
    frame["city"] = np.random.default_rng(1).choice(["Paris", "Lyon"], 600)
    ds = dx.from_pandas(frame, "latent", cache_dir=tmp_path)
    rule = ds.correlations().structure
    assert rule is not None and rule.proposed
    found = ds.structure()
    assert found.retained == 2 and "city" not in found.columns
    table = dx.to_pandas(found)
    assert list(table.columns) == ["PC1", "PC2"] and len(table) == 6
    factors = ds.factors()
    assert factors.k == 2
    assert {"F1", "F2", "communality", "uniqueness", "kmo"} <= set(dx.to_pandas(factors).columns)
    assert ds.factors(k=1).k == 1
    narrow = ds.structure(columns=["a1", "a2", "b1"])
    assert narrow.columns == ["a1", "a2", "b1"]
    assert dx.to_dict(found)["conclusion"] == found.conclusion
