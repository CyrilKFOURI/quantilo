"""032 — Segments: column choices, k-means and the choice of k, stability, k-prototypes, the other
methods, portraits, map, and the library's front door."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score

import quantilo as dx
from quantilo.segments import kprototypes as kp
from quantilo.segments import methods
from quantilo.segments.engine import SegmentsError
from quantilo.segments.portrait import number, percent

SAMPLES = Path(__file__).resolve().parents[3] / "samples"


def blobs(n: int = 600, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """Three well-separated gaussian blobs in three columns of different scales."""
    rng = np.random.default_rng(seed)
    truth = rng.integers(0, 3, n)
    centres = np.array([[0, 0, 0], [8, 8, 0], [0, 8, 8]], dtype=float)
    points = centres[truth] + rng.normal(0, 1, (n, 3))
    frame = pd.DataFrame({"a": points[:, 0] * 1_000, "b": points[:, 1], "c": points[:, 2] / 100})
    return frame, truth


def as_dataset(frame: pd.DataFrame, tmp_path: Path, name: str = "frame") -> dx.Dataset:
    return dx.from_pandas(frame, name, cache_dir=tmp_path / name)


def truth_of(found: dx.Dataset, fitted: object, truth: np.ndarray) -> float:
    labels = fitted.labels.sort_values("row")["segment"].to_numpy()  # type: ignore[attr-defined]
    return float(adjusted_rand_score(truth, labels))


# --- choosing k, stability -----------------------------------------------------------------------


def test_three_blobs_give_three_stable_segments(tmp_path: Path) -> None:
    frame, truth = blobs()
    ds = as_dataset(frame, tmp_path)
    found = ds.segments().fit()

    assert found.method == "kmeans" and found.k == 3 and not found.choice.forced
    assert found.choice.why.startswith("3 segments: best silhouette")
    assert truth_of(ds, found, truth) >= 0.9
    assert found.stability is not None and found.stability.ari >= 0.9 and found.stability.stable
    assert [s.k for s in found.scores] == list(range(2, 11))
    assert sum(s.size for s in found.segments) == 600
    assert found.segments[0].size >= found.segments[1].size >= found.segments[2].size
    assert not any("not stable" in w for w in found.warnings)


def test_uniform_rows_are_not_stable_and_say_so(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.uniform(size=(1_000, 6)), columns=[f"x{i}" for i in range(6)])
    found = as_dataset(frame, tmp_path).segments().fit()

    assert found.stability is not None and found.stability.ari < 0.6
    assert not found.stability.stable
    assert methods.NOT_STABLE in found.warnings


def test_forced_k_is_kept_with_its_reason(tmp_path: Path) -> None:
    frame, _ = blobs(300)
    found = as_dataset(frame, tmp_path).segments(k=5).fit()

    assert found.k == 5 and found.choice.forced and found.choice.why == "5 segments: your choice."
    assert len(found.segments) == 5


def test_choose_k_sentence_names_the_criteria() -> None:
    scores = [
        methods.SegmentScore(2, 0.4, 100, 0.9, 100),
        methods.SegmentScore(3, 0.6, 300, 0.5, 40),
        methods.SegmentScore(4, 0.5, 250, 0.6, 35),
        methods.SegmentScore(5, 0.45, 200, 0.7, 32),
    ]
    choice = methods.choose_k(scores, "kmeans", None)

    assert choice.k == 3
    assert choice.why == (
        "3 segments: best silhouette, best Calinski–Harabasz and best Davies–Bouldin, "
        "and the elbow is at 3."
    )
    gmm = methods.choose_k(
        [
            methods.SegmentScore(2, None, None, None, None, 50),
            methods.SegmentScore(3, None, None, None, None, 20),
        ],
        "gmm",
        None,
    )
    assert gmm.k == 3 and gmm.why == "3 segments: the lowest BIC."


# --- columns and preparation ---------------------------------------------------------------------


def test_identifiers_and_dates_are_left_out_with_their_reason(tmp_path: Path) -> None:
    traps = dx.open(SAMPLES / "typing_traps.csv", cache_dir=tmp_path)
    plan = traps.segments().plan()
    by_name = {d.column: d for d in plan.preparation.decisions}

    assert not by_name["customer_id"].use and by_name["customer_id"].why[0].startswith("Identifier")
    assert not by_name["signup_date"].use and by_name["signup_date"].rules == ["R-DIST-03"]
    assert not by_name["comment"].use
    used = [d for d in plan.preparation.decisions if d.use]
    assert all(d.scaling == "robust" for d in used if d.type.startswith("numeric"))
    assert plan.rows == 600 and not plan.sampled


def test_columns_given_by_name_are_the_only_ones_used(tmp_path: Path) -> None:
    frame, _ = blobs(200)
    frame["noise"] = np.random.default_rng(1).normal(size=200)
    plan = as_dataset(frame, tmp_path).segments(["a", "b"]).plan()

    assert [d.column for d in plan.preparation.decisions if d.use] == ["a", "b"]
    with pytest.raises(SegmentsError, match="Unknown"):
        as_dataset(frame, tmp_path, "again").segments(preparation={"columns": {"z": {}}}).plan()


def test_mostly_categories_choose_k_prototypes_and_say_so(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    n = 400
    group = rng.integers(0, 2, n)
    frame = pd.DataFrame(
        {
            "income": np.where(group == 0, 30_000, 80_000) + rng.normal(0, 4_000, n),
            "age": np.where(group == 0, 25, 55) + rng.normal(0, 4, n),
            "spend": rng.normal(100, 10, n),
            "plan": np.where(group == 0, "Basic", "Premium"),
            "channel": np.where(
                rng.uniform(size=n) < 0.8, np.where(group == 0, "web", "shop"), "phone"
            ),
        }
    )
    ds = as_dataset(frame, tmp_path)
    plan = ds.segments().plan()
    assert plan.categorical_share == pytest.approx(0.4)
    assert plan.method == "kprototypes" and "40%" in plan.method_why

    found = ds.segments(k=2).fit()
    assert found.method == "kprototypes"
    assert adjusted_rand_score(group, found.labels.sort_values("row")["segment"]) > 0.9
    premium = next(s for s in found.segments if any(t.value == "Premium" for t in s.traits))
    income = next(t for t in premium.traits if t.column == "income")
    assert income.text.startswith("income: 1.") and "× the average" in income.text
    assert "Premium" in premium.name


# --- k-prototypes --------------------------------------------------------------------------------


def reference_kprototypes(
    numbers: np.ndarray, codes: np.ndarray, centres: np.ndarray, modes: np.ndarray, gamma: float
) -> np.ndarray:
    """Huang's iteration written plainly, row by row, to check the vectorised one."""
    centres, modes = centres.copy(), modes.copy()
    labels = [-1] * len(numbers)
    for _ in range(100):
        moved = []
        for row in range(len(numbers)):
            costs = []
            for j in range(len(centres)):
                numeric = sum(
                    (numbers[row][a] - centres[j][a]) ** 2 for a in range(numbers.shape[1])
                )
                mismatch = sum(1 for a in range(codes.shape[1]) if codes[row][a] != modes[j][a])
                costs.append(numeric + gamma * mismatch)
            moved.append(int(np.argmin(costs)))
        if moved == labels:
            break
        labels = moved
        for j in range(len(centres)):
            members = [r for r in range(len(numbers)) if labels[r] == j]
            if not members:
                continue
            centres[j] = numbers[members].mean(axis=0)
            for a in range(codes.shape[1]):
                values = [int(codes[r][a]) for r in members]
                modes[j][a] = max(sorted(set(values)), key=values.count)
    return np.array(labels)


def test_k_prototypes_matches_the_plain_reference() -> None:
    rng = np.random.default_rng(7)
    numbers = np.vstack(
        [rng.normal(0, 1, (30, 2)), rng.normal(4, 1, (30, 2)), rng.normal([0, 6], 1, (30, 2))]
    )
    codes = np.column_stack([np.repeat([0, 1, 2], 30), rng.integers(0, 3, 90)])
    codes[::7, 0] = (codes[::7, 0] + 1) % 3  # some noise in the categories
    gamma = kp.default_gamma(numbers)
    start = (numbers[[0, 30, 60]].copy(), codes[[0, 30, 60]].copy())

    fast = kp.fit_once(numbers, codes, 3, gamma, np.random.default_rng(0), start)
    plain = reference_kprototypes(numbers, codes, start[0], start[1], gamma)

    assert np.array_equal(fast.labels, plain)
    assert adjusted_rand_score(np.repeat([0, 1, 2], 30), fast.labels) > 0.9


def test_k_modes_on_categories_alone_recovers_the_groups() -> None:
    """Huang's soybean-like case: categories only (γ has nothing to balance)."""
    rng = np.random.default_rng(1)
    truth = np.repeat([0, 1, 2, 3], 12)
    profiles = rng.integers(0, 4, (4, 10))
    codes = profiles[truth].copy()
    flip = rng.uniform(size=codes.shape) < 0.1
    codes[flip] = rng.integers(0, 4, int(flip.sum()))
    found = kp.fit(np.zeros((48, 0)), codes, 4, seed=0, n_init=8)

    assert found.gamma == 1.0
    assert adjusted_rand_score(truth, found.labels) > 0.8


def test_k_prototypes_is_reproducible() -> None:
    rng = np.random.default_rng(2)
    numbers, codes = rng.normal(size=(200, 3)), rng.integers(0, 4, (200, 2))
    first = kp.fit(numbers, codes, 4, seed=5)
    second = kp.fit(numbers, codes, 4, seed=5)
    assert np.array_equal(first.labels, second.labels) and first.cost == second.cost


# --- other methods -------------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["gmm", "ward", "hdbscan"])
def test_other_methods_find_the_blobs(tmp_path: Path, method: str) -> None:
    frame, truth = blobs(450)
    found = as_dataset(frame, tmp_path).segments(method=method).fit()  # type: ignore[arg-type]

    assert found.method == method and found.k == 3
    grouped = found.labels.sort_values("row")["segment"].to_numpy()
    assert adjusted_rand_score(truth, grouped) >= 0.9
    if method == "ward":
        assert found.dendrogram is not None and found.dendrogram.leaves
    if method == "gmm":
        assert all(s.bic is not None for s in found.scores)
        assert found.segments[0].membership is not None and found.segments[0].membership > 0.9
    if method == "hdbscan":
        assert "found by HDBSCAN" in found.choice.why


def test_hdbscan_noise_is_its_own_segment(tmp_path: Path) -> None:
    frame, _ = blobs(450)
    rng = np.random.default_rng(9)
    noise = pd.DataFrame(
        {
            "a": rng.uniform(-20_000, 30_000, 40),
            "b": rng.uniform(-20, 30, 40),
            "c": rng.uniform(-0.2, 0.3, 40),
        }
    )
    found = (
        as_dataset(pd.concat([frame, noise], ignore_index=True), tmp_path)
        .segments(method="hdbscan")
        .fit()
    )

    last = found.segments[-1]
    assert last.id == -1 and last.name == "Not grouped" and not last.grouped and last.size > 0


# --- portrait, map -------------------------------------------------------------------------------


def test_portrait_speaks_in_original_units(tmp_path: Path) -> None:
    rng = np.random.default_rng(4)
    n = 500
    rich = rng.uniform(size=n) < 0.3
    frame = pd.DataFrame(
        {
            "revenue": np.where(rich, 6_000, 1_500) + rng.normal(0, 300, n),
            "age": np.where(rich, 52, 30) + rng.normal(0, 3, n),
        }
    )
    found = as_dataset(frame, tmp_path).segments(k=2).fit()
    small = found.segments[1]  # the rich ones are the smaller segment
    revenue = next(t for t in small.traits if t.column == "revenue")

    mean = frame["revenue"].mean()
    expected = frame.loc[rich, "revenue"].mean() / mean
    assert revenue.text == f"revenue: {expected:.1f}× the average"
    assert revenue.segment == pytest.approx(frame.loc[rich, "revenue"].mean(), rel=0.02)
    assert small.name == "High revenue · High age"
    assert small.rules is not None and small.rules.recall > 0.9
    assert small.rules.rules[0].parts[0].column in ("revenue", "age")
    assert "of the segment" in small.rules.text
    table = found.to_pandas()
    assert list(table.columns) == ["overall", found.segments[0].name, small.name]
    assert table.loc["revenue", small.name] == pytest.approx(revenue.segment)
    revenue_row = next(r for r in found.profiles if r.column == "revenue")
    assert revenue_row.cells[1].strong


def test_map_projects_every_segment(tmp_path: Path) -> None:
    frame, _ = blobs(300)
    found = as_dataset(frame, tmp_path).segments().fit()

    shown = found.map
    assert shown.projection == "pca" and len(shown.x) == len(shown.y) == len(shown.rows) == 300
    assert {c.segment for c in shown.centres} == {0, 1, 2}
    assert shown.explained is not None and sum(shown.explained) > 0.9
    tsne = found.tsne()
    assert tsne.projection == "tsne" and len(tsne.x) == 300 and found.tsne is not None


def test_filters_apply_and_save_as_dataset_adds_a_segment_column(tmp_path: Path) -> None:
    frame, _ = blobs(300)
    frame["country"] = np.where(np.arange(300) % 3 == 0, "FR", "BE")
    ds = as_dataset(frame, tmp_path)
    french = ds.where("country", "in", ["FR"])
    found = french.segments(["a", "b", "c"]).fit()

    assert found.rows == 100
    made = found.to_dataset()
    copy = made.to_pandas()
    assert len(copy) == 100 and "segment" in copy.columns
    assert set(copy["segment"]) == {s.name for s in found.segments}
    assert made.column("segment").type == "categorical_nominal"
    assert made.compare("a", "segment").tests


def test_beyond_the_fit_rows_every_row_goes_to_its_nearest_segment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from quantilo.segments import prepare

    monkeypatch.setattr(prepare, "FIT_ROWS", 200)
    monkeypatch.setattr(prepare, "CHUNK", 150)
    frame, truth = blobs(600)
    found = as_dataset(frame, tmp_path).segments().fit()

    assert found.sampled and found.fit_rows == 200 and found.rows == 600
    labels = found.labels.sort_values("row")
    assert len(labels) == 600 and labels["row"].tolist() == list(range(600))
    assert adjusted_rand_score(truth, labels["segment"]) >= 0.9
    assert sum(s.size for s in found.segments) == 600


def test_number_and_percent_formats() -> None:
    assert number(1234.5) == "1,234" or number(1234.5) == "1,235"
    assert number(0.5) == "0.5" and number(3012) == "3,012"
    assert percent(0.78) == "78 %" and percent(0.002) == "0.2 %" and percent(0.00001) == "< 0.01 %"


def test_results_as_plain_data(tmp_path: Path) -> None:
    frame, _ = blobs(200)
    found = as_dataset(frame, tmp_path).segments().fit()
    plain = dx.to_dict(found)

    assert plain["k"] == 3 and len(plain["segments"]) == 3
    assert dx.to_pandas(found).shape == (3, 4)


def test_a_column_with_no_spread_in_its_middle_does_not_decide_the_segments() -> None:
    # ccdata: TENURE is 12 for 85 % of clients, its IQR is 0; scaled by 1 after Yeo-Johnson it
    # split every segment and named them all (040)
    from quantilo.ml.steps import RobustScale

    rng = np.random.default_rng(0)
    tenure = np.where(rng.random(2_000) < 0.85, 12.0, rng.integers(6, 12, 2_000))
    scaled = RobustScale().fit_transform(np.column_stack([tenure, rng.normal(0, 1, 2_000)]))
    assert scaled[:, 0].std() == pytest.approx(1.0, rel=0.05)
