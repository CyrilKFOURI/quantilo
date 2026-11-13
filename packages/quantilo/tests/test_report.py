from quantilo.charts.prepare import Chart
from quantilo.correlations.matrix import Matrix, MatrixMember
from quantilo.filters.model import Condition
from quantilo.report import svg
from quantilo.report.build import anchor, describe_condition, file_name, fmt
from quantilo.report.model import Missing


def test_numbers_are_written_for_people() -> None:
    assert fmt(None) == "—"
    assert fmt(1234567) == "1,234,567"
    assert fmt(12.0) == "12"
    assert fmt(0.123456) == "0.1235"
    assert fmt(1234.5678) == "1,235"
    assert fmt(float("nan")) == "—"
    assert fmt(1.5e9) == "1,500,000,000"
    assert fmt(2.5e-5) == "2.5e-05"


def test_filters_in_words() -> None:
    def said(**fields) -> str:
        return describe_condition(Condition(column="Age", **fields))

    assert said(op="between", low=20.0, high=40.0) == "Age between 20 and 40"
    assert said(op="in", values=["female"]) == "Age is female"
    assert said(op="gte", value=18.0) == "Age ≥ 18"
    assert said(op="is_null", negate=True) == "Age is not empty"
    assert said(op="part_in", part="day_of_week", values=["1"]) == "day of week of Age is 1"
    assert said(op="unique", keep="last") == "one row per Age (the last)"
    assert (
        said(op="last_days", value=30.0, high="2024-12-30")
        == "Age in the last 30 days up to 2024-12-30"
    )


def test_file_names_and_anchors_are_safe() -> None:
    assert file_name('sales: 2024/Q1 "final"') == "sales_ 2024_Q1 _final_-report.html"
    used: set[str] = set()
    assert anchor("Ticket price (€)", used) == "v-ticket-price"
    assert anchor("ticket PRICE", used) == "v-ticket-price-2"
    assert anchor("€€€", used) == "v-column"


def test_charts_escape_the_data_and_carry_a_tooltip_per_mark() -> None:
    chart = Chart(
        "bars",
        "bars",
        ["bars"],
        bars=[{"label": "<script>", "count": 3}, {"label": "b", "count": 1}],
        others={"count": 2, "categories": 4},
    )

    drawn = svg.chart_svg(chart, "Distribution of <x>")

    assert "<script>" not in drawn and "&lt;script&gt;" in drawn
    assert drawn.count("<title>") == 3  # two categories and "Others"
    assert "Others (4)" in drawn and 'aria-label="Distribution of &lt;x&gt;"' in drawn


def test_histogram_and_missing_bars() -> None:
    chart = Chart(
        "histogram",
        "histogram",
        ["histogram"],
        bins=[{"start": 0.0, "end": 10.0, "count": 4}, {"start": 10.0, "end": 20.0, "count": 2}],
    )

    drawn = svg.chart_svg(chart, "Age")

    assert drawn.count('class="bar"') == 2 and "0 \u2013 10: 4 rows" in drawn
    assert svg.chart_svg(Chart("identifier", "identifier", []), "id") == ""
    assert "50.0 %" in svg.missing_bars([Missing("Cabin", 50.0)])


def test_heatmap_colours_follow_the_sign() -> None:
    matrix = Matrix(
        method="pearson",
        columns=[MatrixMember("a", "numeric_continuous", "num"), MatrixMember("b", "x", "num")],
        values=[[1.0, -0.5], [-0.5, 1.0]],
        metrics=[[None, "pearson"], ["pearson", None]],
        n=[[10, 10], [10, 10]],
        clustered=[1, 0],
        rows=10,
        total=10,
        sampled=False,
    )

    drawn = svg.heatmap(matrix)

    assert "var(--chart-8) 50%" in drawn  # negative: red pole
    assert "var(--chart-1) 100%" in drawn  # the diagonal
    assert drawn.index(">b</text>") < drawn.index(">a</text>")  # clustered order


def test_print_sheet_and_running_header_escape_the_name() -> None:
    """025, R-REP-16: A4, sections on new pages, cards never cut; the name in CSS is data."""
    from quantilo.report.html import CSS, _css_string, _running_header

    assert "size: A4" in CSS and "break-before: page" in CSS and "print-color-adjust: exact" in CSS
    header = _running_header('a "b"</style><script>', "2026-11-12")
    assert "</style>" not in header and "<script>" not in header
    assert header.startswith("@page { @top-left { content: ")
    assert _css_string('x"y') == r'"x\22 y"'
