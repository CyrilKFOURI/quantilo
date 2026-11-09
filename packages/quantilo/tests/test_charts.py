import duckdb
import pytest

from quantilo.charts.prepare import ChartParams, ColumnInput, prepare_chart
from quantilo.typing.dates import auto_step, classify_step, part_label
from quantilo.typing.types import Parser

RAW = Parser(kind="raw")
TEXT = Parser(kind="text")


@pytest.fixture
def connection() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(
        """
        CREATE TABLE t AS
        SELECT
            range AS id,
            (range % 7) AS small_int,
            CAST(range AS DOUBLE) * 1.5 AS measure,
            CASE WHEN range % 10 = 0 THEN NULL ELSE 'cat_' || (range % 15) END AS category,
            ['low', 'medium', 'high'][1 + (range % 3)] AS level,
            TIMESTAMP '2024-01-01 00:00:00' + INTERVAL (range * 7) HOUR AS moment,
            DATE '2020-01-01' + CAST(range AS INTEGER) AS day
        FROM range(300)
        """
    )
    return con


def chart(con, name, analysis_type, parser=RAW, order=None, **params):
    column = ColumnInput(name, analysis_type, parser, order)
    return prepare_chart(con, "t", column, ChartParams(**params))


def test_histogram_bins_cover_all_values(connection: duckdb.DuckDBPyConnection) -> None:
    result = chart(connection, "measure", "numeric_continuous")

    assert result.kind == "histogram"
    assert 10 <= len(result.bins) <= 60
    assert sum(b["count"] for b in result.bins) == 300
    assert result.stats is not None and result.stats["median"] == pytest.approx(224.25)


def test_bin_count_can_be_forced(connection: duckdb.DuckDBPyConnection) -> None:
    result = chart(connection, "measure", "numeric_continuous", bins=80)

    assert len(result.bins) == 80
    assert sum(b["count"] for b in result.bins) == 300


def test_box_view_has_quartiles(connection: duckdb.DuckDBPyConnection) -> None:
    result = chart(connection, "measure", "numeric_continuous", view="box")

    assert result.view == "box"
    assert result.box is not None and result.box["q1"] < result.box["median"] < result.box["q3"]


def test_top_n_with_others_and_missing(connection: duckdb.DuckDBPyConnection) -> None:
    result = chart(connection, "category", "categorical_nominal", TEXT, top_n=5)

    assert len(result.bars) == 5
    assert result.missing == 30
    assert result.others == {
        "count": 270 - sum(b["count"] for b in result.bars),
        "categories": 10,
    }
    assert result.views == ["bars"]  # 15 categories: no pie


def test_full_view_returns_every_category(connection: duckdb.DuckDBPyConnection) -> None:
    result = chart(connection, "category", "categorical_nominal", TEXT, full=True)

    assert len(result.bars) == 15
    assert result.others is None


def test_ordinal_keeps_its_order(connection: duckdb.DuckDBPyConnection) -> None:
    result = chart(
        connection, "level", "categorical_ordinal", TEXT, order=["low", "medium", "high"]
    )

    assert [b["label"] for b in result.bars] == ["low", "medium", "high"]
    assert result.ordered


def test_discrete_values_in_numeric_order(connection: duckdb.DuckDBPyConnection) -> None:
    result = chart(connection, "small_int", "numeric_discrete")

    assert [b["label"] for b in result.bars] == [str(i) for i in range(7)]


def test_unique_text_is_flagged_instead_of_drawn(connection: duckdb.DuckDBPyConnection) -> None:
    result = chart(connection, "id", "categorical_nominal", TEXT)

    assert result.kind == "high_cardinality"
    assert result.message


def test_timeline_picks_a_readable_step(connection: duckdb.DuckDBPyConnection) -> None:
    result = chart(connection, "day", "datetime")

    assert result.kind == "timeline"
    assert result.step == "week"  # 300 days: weeks keep it under 120 periods
    assert sum(b["count"] for b in result.bars) == 300
    assert "hour" not in result.views


def test_date_parts_views(connection: duckdb.DuckDBPyConnection) -> None:
    weekdays = chart(connection, "moment", "datetime", view="day_of_week")
    hours = chart(connection, "moment", "datetime", view="hour")

    assert [b["label"] for b in weekdays.bars] == ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    assert len(hours.bars) == 24
    assert sum(b["count"] for b in hours.bars) == 300


def test_identifier_counts_duplicates(connection: duckdb.DuckDBPyConnection) -> None:
    result = chart(connection, "small_int", "identifier")

    assert result.kind == "identifier"
    assert result.duplicates == 7


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(None, "single"), (86_400, "daily"), (7 * 86_400, "weekly"), (31 * 86_400, "monthly")],
)
def test_step_classification(seconds: float | None, expected: str) -> None:
    assert classify_step(seconds) == expected


def test_auto_step_and_labels() -> None:
    assert auto_step(3 * 86_400, has_time=True) == "hour"
    assert auto_step(10 * 365 * 86_400, has_time=False) == "month"
    assert part_label("month", 3) == "Mar"
    assert part_label("day_of_week", 7) == "Sun"
