"""033 — Unusual rows: rows normal on each column but not together, the threshold at the break,
the reasons per row, the distance to its segment, and expected rows left out."""

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest

import quantilo as dx
from quantilo.segments import unusual
from quantilo.segments.engine import SegmentsEngine, SegmentsError, UnusualRequest


def correlated(seed: int = 0) -> tuple[pd.DataFrame, list[int]]:
    """1,000 rows along y = x, and 10 rows off the line but inside each column's range."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, 1_000)
    frame = pd.DataFrame({"x": x, "y": x + rng.normal(0, 0.15, 1_000)})
    off = pd.DataFrame({"x": np.linspace(-1.2, 1.2, 10), "y": -np.linspace(-1.2, 1.2, 10) + 0.05})
    off.loc[off["x"].abs() < 0.3, "y"] = off["x"] + 1.4  # the middle ones: above the line
    return pd.concat([frame, off], ignore_index=True), list(range(1_000, 1_010))


def iqr_outliers(series: pd.Series) -> pd.Series:
    q1, q3 = series.quantile([0.25, 0.75])
    return (series < q1 - 1.5 * (q3 - q1)) | (series > q3 + 1.5 * (q3 - q1))


def as_dataset(frame: pd.DataFrame, tmp_path: Path, name: str = "frame") -> dx.Dataset:
    return dx.from_pandas(frame, name, cache_dir=tmp_path / name)


# Isolation Forest cuts along axes (and the principal components): 8 of 10 here; Auto takes the
# robust Mahalanobis distance on these two numbers, which the spec's scenario is about.
@pytest.mark.parametrize(
    ("method", "found_at_least"), [("auto", 9), ("lof", 9), ("isolation_forest", 8)]
)
def test_rows_off_the_correlation_are_found(
    tmp_path: Path, method: str, found_at_least: int
) -> None:
    frame, planted = correlated()
    off = frame.loc[planted]
    for column in ("x", "y"):
        assert not iqr_outliers(cast(pd.Series, frame[column]))[planted].any()

    found = as_dataset(frame, tmp_path).unusual_rows(method=method)  # type: ignore[arg-type]
    top = found.scores.sort_values("score", ascending=False).head(int(0.02 * len(frame)))

    assert len(set(top["row"]) & set(planted)) >= found_at_least, off
    if method == "auto":
        assert found.method == "mahalanobis" and "every column is a number" in found.method_why
    assert found.threshold.rows >= 1 and found.listed[0].rank == 1
    assert len(found.scores) == len(frame)


def test_the_two_columns_that_clash_are_the_first_reasons(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    n = 2_000
    orders = rng.integers(1, 30, n).astype(float)
    frame = pd.DataFrame(
        {
            "orders": orders,
            "revenue": orders * 100 + rng.normal(0, 40, n),
            "age": rng.normal(40, 10, n),
            "plan": rng.choice(["basic", "plus"], n),
        }
    )
    frame.loc[0, ["orders", "revenue"]] = [0, 3_012]
    found = as_dataset(frame, tmp_path).unusual_rows()
    first = next(r for r in found.listed if r.row == 0)

    assert found.method == "isolation_forest"
    assert first.rank <= 3
    assert {r.column for r in first.reasons[:2]} == {"orders", "revenue"}
    assert first.combination is not None
    assert "orders = 0" in first.combination and "revenue = 3,012" in first.combination
    assert "this combination appears in" in first.combination


def test_reasons_name_the_responsible_column() -> None:
    rng = np.random.default_rng(2)
    frame = pd.DataFrame({c: rng.normal(0, 1, 500) for c in "abcd"})
    frame.loc[0, "c"] = 9.0
    from quantilo.ml import preparation as prep
    from quantilo.ml.data import Field
    from quantilo.segments import prepare as preparing

    fields = [Field(c, "numeric_continuous", None) for c in "abcd"]  # type: ignore[arg-type]
    decided = prep.decide_distance(frame, fields)
    prepared = preparing.matrix(decided, frame)
    scorer = unusual.isolation_forest(prepared.x, 42)
    found = unusual.reasons(
        frame.head(1),
        scorer.fit_scores[:1],
        None,
        decided.used,
        frame,
        None,
        {},
        prepared.transform,
        scorer.score,
    )
    reasons, _ = found[0]
    assert reasons[0].column == "c" and reasons[0].text.startswith("c is in the top")


def test_threshold_at_the_break_else_one_percent() -> None:
    rng = np.random.default_rng(3)
    clear = np.concatenate([rng.normal(0, 0.05, 990), np.linspace(5, 6, 10)])
    found = unusual.propose(clear)
    assert found.rule == "break" and found.rows == 10

    smooth = rng.uniform(0, 1, 1_000)
    flat = unusual.propose(smooth)
    assert flat.rule == "top_share" and flat.rows == 10
    shape = unusual.curve(smooth)
    assert shape.ranks[:3] == [1, 2, 3] and shape.ranks[-1] == 1_000
    assert shape.scores == sorted(shape.scores, reverse=True)


def test_expected_rows_are_hidden_and_counted(tmp_path: Path) -> None:
    frame, _ = correlated()
    ds = as_dataset(frame, tmp_path)
    engine = SegmentsEngine(ds._state)
    first = engine.unusual(ds._id, [], UnusualRequest()).result
    top = first.listed[0]

    again = engine.unusual(ds._id, [], UnusualRequest(), hidden={top.key}).result
    assert again.hidden_expected == 1
    assert top.row not in {r.row for r in again.listed}
    assert top.key.startswith("row:") and len(top.key) == 36


def test_identifier_is_the_key_when_there_is_one(tmp_path: Path) -> None:
    frame, _ = correlated()
    frame.insert(0, "customer_id", [f"C{i:05d}" for i in range(len(frame))])
    ds = as_dataset(frame, tmp_path)
    found = ds.unusual_rows()
    assert ds.column("customer_id").type == "identifier"
    assert found.listed[0].key == f"customer_id={found.listed[0].values['customer_id']}"
    assert "customer_id" not in found.used


def test_distance_to_its_segment(tmp_path: Path) -> None:
    """A luxury car is not unusual among luxury cars, but it is among city cars."""
    rng = np.random.default_rng(4)
    city = pd.DataFrame({"price": rng.normal(15_000, 2_000, 300), "power": rng.normal(80, 8, 300)})
    luxury = pd.DataFrame(
        {"price": rng.normal(90_000, 8_000, 100), "power": rng.normal(300, 25, 100)}
    )
    frame = pd.concat([city, luxury], ignore_index=True)
    ds = as_dataset(frame, tmp_path)
    segments = ds.segments(k=2).fit()
    found = ds.unusual_rows(method="segment", segments=segments)
    scores = found.scores.set_index("row")["score"]

    assert found.method == "segment"
    assert float(np.median(scores.loc[300:399])) < 2  # ordinary within their own segment
    labels = segments.labels.set_index("row")["segment"]
    luxury_segment = int(labels.loc[300])
    city_segment = 1 - luxury_segment
    moved = segments.labels.copy()
    moved.loc[moved["row"] == 305, "segment"] = city_segment  # the same car, filed as a city car
    from quantilo.segments.engine import SegmentLabels

    engine = SegmentsEngine(ds._state)
    shifted = engine.unusual(
        ds._id,
        [],
        UnusualRequest(method="segment"),
        SegmentLabels(moved, segments.names(), segments.request),
    )
    ranked = shifted.scores.sort_values("score", ascending=False)["row"].tolist()
    assert ranked[0] == 305
    row = next(r for r in shifted.result.listed if r.row == 305)
    assert row.segment == segments.names()[city_segment]
    assert "for segment =" in row.reasons[0].text


def test_every_row_is_scored_beyond_the_fit_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from quantilo.segments import prepare

    monkeypatch.setattr(prepare, "FIT_ROWS", 300)
    monkeypatch.setattr(prepare, "CHUNK", 400)
    frame, planted = correlated()
    found = as_dataset(frame, tmp_path).unusual_rows()

    assert found.sampled and found.fit_rows == 300 and len(found.scores) == len(frame)
    top = found.scores.sort_values("score", ascending=False).head(30)
    assert len(set(top["row"]) & set(planted)) >= 8


def test_segment_method_needs_a_segmentation(tmp_path: Path) -> None:
    frame, _ = correlated()
    with pytest.raises(SegmentsError, match="needs a segmentation"):
        as_dataset(frame, tmp_path).unusual_rows(method="segment")


def test_row_card_places_each_value(tmp_path: Path) -> None:
    frame, planted = correlated()
    frame["kind"] = np.where(np.arange(len(frame)) % 4 == 0, "a", "b")
    ds = as_dataset(frame, tmp_path)
    card = SegmentsEngine(ds._state).row_card(ds._id, [], planted[0])
    x = next(v for v in card if v.column == "x")
    kind = next(v for v in card if v.column == "kind")
    assert x.percentile is not None and 0 < x.percentile < 0.5
    assert kind.value == "a" and kind.share == pytest.approx(0.25, abs=0.01)


def test_table_and_plain_data(tmp_path: Path) -> None:
    frame, _ = correlated()
    found = as_dataset(frame, tmp_path).unusual_rows()
    table = dx.to_pandas(found)
    assert list(table.columns[:4]) == ["rank", "row", "score", "reasons"]
    assert len(table) == len(found.listed) <= unusual.LIST_ROWS
    assert dx.to_dict(found)["threshold"]["rows"] == found.threshold.rows
    assert all(r.score >= found.threshold.value for r in found.above())


def mixed(numbers: int, seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 1_500
    base = rng.normal(0, 1, n)
    frame = pd.DataFrame(
        {f"n{i}": base + rng.normal(0, 0.3, n) for i in range(numbers)}
        | {"city": rng.choice(["Paris", "Lyon", "Nice"], n)}
    )
    frame.loc[0, [f"n{i}" for i in range(numbers)]] = [2.5, -2.5, 2.5][:numbers]
    return frame


def test_mahalanobis_on_a_mix_reads_its_numbers(tmp_path: Path) -> None:
    """056, R-UNU-MCD-1: 3 numbers and a category → the distance on the 3 numbers."""
    found = as_dataset(mixed(3), tmp_path).unusual_rows(method="mahalanobis")

    assert found.method == "mahalanobis"
    assert found.used == ["n0", "n1", "n2"]
    assert "city left out" in found.method_why
    city = next(d for d in found.preparation.decisions if d.column == "city")
    assert not city.use and "R-UNU-MCD-1" in city.rules
    assert found.listed[0].row == 0  # off the correlation of the three numbers


def test_mahalanobis_on_two_numbers_and_a_category_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SegmentsError) as error:
        as_dataset(mixed(2), tmp_path).unusual_rows(method="mahalanobis")
    assert error.value.code == "method_unavailable"


def test_auto_on_a_mix_stays_isolation_forest(tmp_path: Path) -> None:
    found = as_dataset(mixed(3), tmp_path).unusual_rows()
    assert found.method == "isolation_forest"
