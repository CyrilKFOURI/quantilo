"""The library's entry point: open a file, read its types, run the analyses, filter, report."""

import re
import subprocess
import sys
from pathlib import Path

import duckdb
import pandas as pd
import pytest

import quantilo as dx
from quantilo.ingestion.sql import lit
from quantilo.typing.engine import detect_columns

SAMPLES = Path(__file__).resolve().parents[3] / "samples"
TRAPS = SAMPLES / "typing_traps.csv"
PACKAGE = Path(dx.__file__).parent


@pytest.fixture(scope="module")
def cache(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("cache")


@pytest.fixture
def traps(cache: Path) -> dx.Dataset:
    return dx.open(TRAPS, cache_dir=cache)


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def people(tmp_path: Path) -> dx.Dataset:
    rows = ["age,sex,score,group"]
    for i in range(120):
        sex = "female" if i % 3 else "male"
        rows.append(f"{18 + i % 50},{sex},{(i * 7) % 23 + (5 if sex == 'female' else 0)},{i % 3}")
    return dx.open(write(tmp_path / "people.csv", "\n".join(rows)), cache_dir=tmp_path / "cache")


def test_the_package_never_imports_the_app() -> None:
    code = (
        "import sys, quantilo\n"
        "print(sorted(m for m in ('fastapi', 'starlette', 'uvicorn', 'sqlite3', 'app') "
        "if m in sys.modules))"
    )
    loaded = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert loaded.returncode == 0, loaded.stderr
    assert loaded.stdout.strip() == "[]"
    for source in PACKAGE.rglob("*.py"):
        text = source.read_text(encoding="utf-8")
        assert not re.search(r"^\s*(from|import) (app|fastapi|sqlite3)\b", text, re.M), source


def test_types_are_the_engine_detection(traps: dx.Dataset) -> None:
    connection = duckdb.connect()
    expected = detect_columns(connection, f"read_parquet({lit(str(traps.path))})")
    by_name = {column.name: column for column in traps.types}
    assert list(by_name) == [d.name for d in expected]
    for detection in expected:
        column = by_name[detection.name]
        assert (column.type, column.role, column.parser) == (
            detection.type,
            detection.role,
            detection.parser,
        )
        assert column.summary is not None
    assert by_name["customer_id"].type == "identifier"
    assert by_name["price_eur"].type == "numeric_continuous"  # "79,50": French decimals


def test_opening_again_reuses_the_cached_copy(cache: Path) -> None:
    first = dx.open(TRAPS, cache_dir=cache)
    stamp = first.path.stat().st_mtime_ns
    again = dx.open(TRAPS, cache_dir=cache)
    assert again.path == first.path
    assert again.path.stat().st_mtime_ns == stamp
    assert [c.type for c in again.types] == [c.type for c in first.types]


def test_an_unknown_column_suggests_the_closest(traps: dx.Dataset) -> None:
    with pytest.raises(dx.ColumnNotFound) as error:
        traps.variable("agee")
    assert error.value.suggestions[0] == "age"
    assert "did you mean 'age'" in str(error.value)


def test_analyses_run_on_the_dataset(traps: dx.Dataset) -> None:
    overview = traps.overview()
    assert overview.rows == 600
    profile = traps.variable("age")
    assert profile.type == "numeric_continuous"
    assert traps.compare("age", "segment").tests
    correlations = traps.correlations()
    assert "age" in [member.name for member in correlations.matrix.columns]
    assert traps.fit("age").laws
    assert traps.outliers("income").rules
    tail = traps.tail("income")  # 600 rows: analysed when its tail is heavy enough
    assert tail.n == 551 and tail.rule.fires and tail.xi is not None  # 49 empty
    page = traps.rows(limit=3)
    assert len(page.rows) == 3 and page.total == 600


def test_retype_changes_how_the_column_is_read(tmp_path: Path) -> None:
    ds = people(tmp_path)
    detected = ds.column("age").type
    assert detected.startswith("numeric")
    before = ds.compare("score", "age")
    assert before.roles["age"] == "num"
    ds.retype("age", "categorical_ordinal")
    assert ds.column("age").type == "categorical_ordinal"
    assert ds.column("age").overridden
    after = ds.compare("score", "age")
    assert after.roles["age"] == "cat"
    assert after.kind != before.kind
    ds.reset_types("age")
    assert ds.column("age").type == detected


def test_where_filters_a_view_and_leaves_the_dataset_alone(tmp_path: Path) -> None:
    ds = people(tmp_path)
    women = ds.where("sex", "in", ["female"])
    assert len(women) == 80
    assert len(ds) == 120
    assert women.overview().rows == 80
    assert ds.overview().rows == 120
    older = women.where("age", "between", (40, 67))
    assert len(older) < 80
    numeric = women.variable("score").numeric
    assert numeric is not None and numeric.describe.count == 80
    with pytest.raises(dx.ColumnNotFound):
        ds.where("sexe", "in", ["female"])


def test_report_is_one_self_contained_file(tmp_path: Path) -> None:
    ds = people(tmp_path)
    target = tmp_path / "out" / "people.html"
    content = ds.report(target)
    assert target.read_text(encoding="utf-8") == content
    assert "<script" not in content and "http://" not in content
    assert "people" in content


def test_export_follows_the_extension(tmp_path: Path) -> None:
    ds = people(tmp_path).where("sex", "in", ["male"])
    written = ds.export(tmp_path / "men.csv")
    assert len(pd.read_csv(written)) == 40
    with pytest.raises(dx.ExplorerError):
        ds.export(tmp_path / "men.json")


def test_from_pandas(tmp_path: Path) -> None:
    frame = pd.DataFrame({"x": range(50), "label": ["a", "b"] * 25})
    ds = dx.from_pandas(frame, "frame", cache_dir=tmp_path)
    assert ds.columns == ["x", "label"]
    assert len(ds) == 50
    assert ds.column("label").type in ("categorical_nominal", "boolean")


def test_results_as_plain_data(traps: dx.Dataset) -> None:
    import json

    profile = dx.to_dict(traps.variable("age"))
    assert profile["name"] == "age" and profile["numeric"]["describe"]["count"] > 0
    json.dumps(dx.to_dict(traps.correlations()), allow_nan=False)
    matrix = dx.to_pandas(traps.correlations())
    assert matrix.shape[0] == matrix.shape[1] and "age" in matrix.index
    assert list(dx.to_pandas(traps.types)["name"]) == traps.columns
    assert len(dx.to_pandas(traps.rows(limit=5))) == 5
    with pytest.raises(TypeError, match="No table"):
        dx.to_pandas(traps.variable("age"))


def test_to_pandas_reads_every_column_in_its_type(traps: dx.Dataset) -> None:
    frame = traps.to_pandas()
    assert len(frame) == 600
    assert frame["price_eur"].dtype.kind == "f"  # "79,50" read as 79.5
    assert frame["is_active"].dtype.kind == "b" or str(frame["is_active"].dtype) == "boolean"


def test_jupyter_shows_the_columns_and_the_first_rows(tmp_path: Path) -> None:
    frame = pd.DataFrame({"city": ["Lyon", "<b>Nice</b>", None], "sales": [1.5, 2.0, 3.25]})
    ds = dx.from_pandas(frame, "shops", cache_dir=tmp_path)
    html = ds._repr_html_()
    assert "<strong>shops</strong>: 3 rows" in html and "2 columns" in html
    assert "<code>sales</code>" in html and "First 3 rows" in html
    assert "&lt;b&gt;Nice&lt;/b&gt;" in html and "<b>Nice</b>" not in html  # escaped
    assert "3.25" in html
