"""Using a model on new data (037): the check before predicting, predictions with their
reliability and interval, scores when the file has the answer, export and reload of the model."""

import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import quantilo as dx
from quantilo.ml import predict as use
from quantilo.ml.facade import TrainedModel

REGIONS = ["North", "South", "East", "West"]


def month(seed: int, n: int = 800) -> pd.DataFrame:
    """Clients of one month: churn depends on the region and on the income."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "client_id": [f"C{seed}-{i:04d}" for i in range(n)],
            "region": rng.choice(REGIONS, n),
            "income": rng.normal(40_000, 8_000, n).round(2),
            "age": rng.integers(18, 70, n),
        }
    )
    lifts = {"North": 2.0, "South": -2.0, "East": 0.5, "West": -0.5}
    lift = np.array([lifts[r] for r in frame["region"]])
    score = lift + (frame["income"] - 40_000) / 4_000 + rng.normal(0, 0.5, n)
    frame["churn"] = np.where(score > 0, "leave", "stay")
    frame["spend"] = (frame["income"] * 0.02 + lift * 50 + rng.normal(0, 20, n)).round(2)
    return frame


@pytest.fixture(scope="module")
def cache(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("cache")


@pytest.fixture(scope="module")
def churn(cache: Path) -> TrainedModel:
    october = dx.from_pandas(month(10).drop(columns="spend"), "october", cache_dir=cache)
    return october.ml("churn").train()


@pytest.fixture(scope="module")
def spend(cache: Path) -> TrainedModel:
    october = dx.from_pandas(month(10).drop(columns="churn"), "october-spend", cache_dir=cache)
    return october.ml("spend").train()


def november(cache: Path, change=None, name: str = "november") -> dx.Dataset:
    frame = month(11).drop(columns=["churn", "spend"])
    if change is not None:
        frame = change(frame)
    return dx.from_pandas(frame, name, cache_dir=cache)


def test_same_columns_are_ready_and_predicted(churn: TrainedModel, cache: Path) -> None:
    data = november(cache)

    checked = churn.check(data)
    predicted = churn.predict(data, keep=["client_id"])

    assert checked.level == "ready", checked.message
    assert checked.rows == 800 and not checked.missing and not checked.target_present
    assert "client_id" in checked.ignored
    assert list(predicted.columns[:2]) == ["client_id", "predicted"]
    assert {"p(leave)", "p(stay)", "less_reliable", "why"} <= set(predicted.columns)
    assert len(predicted) == 800 and predicted["client_id"].iloc[0] == "C11-0000"
    # a few incomes beyond October's lowest and highest
    assert predicted["less_reliable"].to_numpy(dtype=bool).mean() < 0.01
    # the same predictions as the fitted pipeline on the same rows
    raw = month(11)
    raw["age"] = raw["age"].astype(float)
    expected = churn.pipeline.predict_proba(raw[churn.inputs])[:, churn.card.labels.index("leave")]
    assert np.allclose(predicted["p(leave)"].to_numpy(), expected)


def test_a_missing_column_blocks_and_is_named(churn: TrainedModel, cache: Path) -> None:
    data = november(cache, lambda f: f.drop(columns=["income"]), "no-income")

    checked = churn.check(data)

    assert checked.level == "blocked"
    assert checked.missing == ["income"] and "income" in checked.message
    with pytest.raises(dx.ExplorerError, match="income"):
        churn.predict(data)


def test_unknown_categories_are_counted_predicted_and_less_reliable(
    churn: TrainedModel, cache: Path
) -> None:
    def corsica(frame: pd.DataFrame) -> pd.DataFrame:
        frame.loc[frame.index[:24], "region"] = "Corsica"  # 3 % of 800 rows
        return frame

    data = november(cache, corsica, "corsica")

    checked = churn.check(data)
    region = next(c for c in checked.columns if c.column == "region")
    predicted = churn.predict(data)

    assert checked.level == "check" and "region" in checked.message
    assert region.unknown_rows == 24 and region.unknown_share == pytest.approx(0.03)
    assert region.unknown_values == ["Corsica"] and region.unknown_rule
    assert region.important and region.note and "Corsica" in region.note
    assert bool(predicted["predicted"].notna().all())
    assert predicted["less_reliable"].iloc[:24].all()
    assert predicted["why"].iloc[:24].str.contains("region: unknown category").all()
    assert not predicted["why"].iloc[24:].str.contains("region").any()


def test_text_where_a_number_was_expected_is_converted_or_blocks(
    churn: TrainedModel, cache: Path
) -> None:
    def some_text(frame: pd.DataFrame) -> pd.DataFrame:
        frame["income"] = frame["income"].astype(str)
        frame.loc[frame.index[:80], "income"] = "unknown"
        return frame

    def mostly_text(frame: pd.DataFrame) -> pd.DataFrame:
        frame["income"] = ["unknown"] * 600 + [str(v) for v in frame["income"].iloc[600:]]
        return frame

    partly = churn.check(november(cache, some_text, "some-text"))
    income = next(c for c in partly.columns if c.column == "income")
    blocked = churn.check(november(cache, mostly_text, "mostly-text"))

    assert income.unreadable == pytest.approx(0.1)
    assert partly.level == "check"
    assert blocked.level == "blocked" and "income" in blocked.message


def test_with_the_answer_the_file_is_scored(churn: TrainedModel, cache: Path) -> None:
    data = dx.from_pandas(month(11).drop(columns="spend"), "november-answered", cache_dir=cache)

    found = churn.prediction(data)

    assert found.check.target_present
    assert "actual" in found.rows.columns
    assert found.summary.scored_rows == 800 and found.summary.scores is not None
    primary = found.summary.primary
    assert found.summary.scores[primary] is not None
    assert found.summary.validation_scores[primary] == churn.scores[primary]
    assert (found.summary.scores["accuracy"] or 0) > 0.75
    shares = found.summary.distribution
    assert shares.kind == "classes" and sum(shares.new) == pytest.approx(1)


def test_a_number_gets_a_conformal_interval(spend: TrainedModel, cache: Path) -> None:
    data = dx.from_pandas(month(11).drop(columns="churn"), "november-spend", cache_dir=cache)

    found = spend.prediction(data)
    rows = found.rows
    actual = rows["actual"].to_numpy(dtype=float)
    inside = np.mean((actual >= rows["low"].to_numpy()) & (actual <= rows["high"].to_numpy()))

    interval = spend.card.interval
    assert interval is not None and interval.level == 0.9
    assert np.allclose(rows["high"] - rows["low"], 2 * interval.half_width)
    assert 0.8 <= float(inside) <= 0.98
    assert found.summary.distribution.kind == "number"
    assert sum(found.summary.distribution.validation) == pytest.approx(1, abs=1e-6)


def test_one_predicted_row_is_explained(churn: TrainedModel, cache: Path) -> None:
    found = churn.explain_row(november(cache), 5)

    assert found.unavailable is None and found.row == 5
    assert found.steps and any("region" in s.label for s in found.steps)


def test_the_card_holds_no_rows(churn: TrainedModel) -> None:
    card = churn.card

    region = next(c for c in card.columns if c.name == "region")
    income = next(c for c in card.columns if c.name == "income")
    assert sorted(region.categories or []) == sorted(REGIONS)
    assert income.minimum is not None and income.maximum is not None
    assert card.versions["scikit-learn"] and card.versions["python"]
    assert card.validation.primary == churn.primary and card.labels == ["leave", "stay"]
    text = card.model_dump_json()
    assert "C10-0001" not in text  # no identifier, no row


def test_export_then_load_model_predicts_the_same(
    churn: TrainedModel, cache: Path, tmp_path: Path
) -> None:
    path = churn.export(tmp_path / "churn_logistic.zip")

    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
    loaded = dx.load_model(path)
    data = november(cache)

    assert names == {"model.joblib", "model_card.json", "predict.py"}
    assert loaded.card.target.name == "churn" and loaded.card.pipeline_sha256
    assert loaded.check(data).level == "ready"
    pd.testing.assert_frame_equal(loaded.predict(data), churn.predict(data))


def _rewrite(source: Path, target: Path, change) -> Path:
    with zipfile.ZipFile(source) as archive:
        files = {name: archive.read(name) for name in archive.namelist()}
    change(files)
    with zipfile.ZipFile(target, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return target


def test_a_model_exported_before_the_rename_still_loads(
    churn: TrainedModel, tmp_path: Path
) -> None:
    path = churn.export(tmp_path / "model.zip")

    def old_format(files: dict[str, bytes]) -> None:
        card = json.loads(files["model_card.json"])
        assert card["format"] == "quantilo/model-card"
        card["format"] = "cyfouri-data-explorer/model-card"
        files["model_card.json"] = json.dumps(card).encode()

    loaded = dx.load_model(_rewrite(path, tmp_path / "old.zip", old_format))

    assert loaded.card.format == "quantilo/model-card"


def test_load_model_refuses_a_zip_without_card_or_with_another_pipeline(
    churn: TrainedModel, tmp_path: Path
) -> None:
    path = churn.export(tmp_path / "model.zip")
    bare = _rewrite(path, tmp_path / "bare.zip", lambda f: f.pop("model_card.json"))
    swapped = _rewrite(
        path, tmp_path / "swapped.zip", lambda f: f.update({"model.joblib": b"not the pipeline"})
    )

    with pytest.raises(dx.ExplorerError, match=r"no model_card.json") as refused:
        dx.load_model(bare)
    assert refused.value.code == "no_model_card"
    with pytest.raises(dx.ExplorerError, match="does not match its card"):
        dx.load_model(swapped)
    with pytest.raises(dx.ExplorerError, match="not a model zip"):
        dx.load_model(tmp_path / "missing.zip")


def test_a_version_difference_is_named_when_loading(churn: TrainedModel, tmp_path: Path) -> None:
    path = churn.export(tmp_path / "model.zip")

    def older(files: dict[str, bytes]) -> None:
        card = json.loads(files["model_card.json"])
        card["versions"]["scikit-learn"] = "0.24.2"
        files["model_card.json"] = json.dumps(card).encode()

    with pytest.warns(UserWarning, match=r"scikit-learn 0\.24\.2 trained this model; .* installed"):
        dx.load_model(_rewrite(path, tmp_path / "older.zip", older))


def test_the_exported_predict_script_gives_the_same_predictions(
    churn: TrainedModel, cache: Path, tmp_path: Path
) -> None:
    folder = tmp_path / "elsewhere"
    folder.mkdir()
    path = churn.export(folder / "churn_logistic.zip")
    with zipfile.ZipFile(path) as archive:
        (folder / "predict.py").write_bytes(archive.read("predict.py"))
    month(11).drop(columns=["churn", "spend"]).to_csv(folder / "november.csv", index=False)

    done = subprocess.run(
        [sys.executable, "predict.py", "november.csv", "out.csv"],
        cwd=folder,
        capture_output=True,
        text=True,
        env={**os.environ, "QUANTILO_CACHE": str(tmp_path / "script-cache")},
        timeout=300,
    )

    assert done.returncode == 0, done.stderr
    assert "Ready" in done.stdout
    out = pd.read_csv(folder / "out.csv")
    expected = churn.predict(november(cache))
    assert list(out["predicted"]) == list(expected["predicted"])
    assert np.allclose(out["p(leave)"], expected["p(leave)"])


def test_conformal_needs_enough_errors() -> None:
    assert use.conformal(np.arange(5.0), np.arange(5.0)) is None
    found = use.conformal(np.zeros(99), np.linspace(0, 1, 99))
    assert found is not None and found.half_width == pytest.approx(0.9, abs=0.02)
