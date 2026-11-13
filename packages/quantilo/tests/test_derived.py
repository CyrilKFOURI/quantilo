# ruff: noqa: RUF001  (derived names use their separator, as dates do)
"""058: derived columns of text and lists of choices."""

import random
from collections.abc import Sequence

import duckdb
import pandas as pd
import pytest

import quantilo
from quantilo.errors import ExplorerError
from quantilo.filters.model import Condition
from quantilo.filters.scope import scoped_source, where_for
from quantilo.typing.derived import (
    MAX_CHOICE_COLUMNS,
    derived_sql,
    names_in,
    resolve,
    with_derived,
)
from quantilo.typing.types import DerivedSpec, TopicRule

SENTENCES = [
    "The dress is lovely and the fabric is soft.",
    "I love it, the color is great!",
    "Is it true to size? I am not sure about the fit.",
    "Terrible quality, I want a refund!!",
    "Check https://example.com for the size chart.",
    "It was too small for me and the zip broke.",
    "Write to help@example.com if the order is late.",
]
LANGUAGES = ["Python", "SQL", "JavaScript", "Rust", "Go", "Java", "C", "R"]


def frame(rows: int = 300) -> pd.DataFrame:
    rng = random.Random(1)
    reviews, skills = [], []
    for i in range(rows):
        reviews.append(f"{rng.choice(SENTENCES)} Review {i} of the shop." if i % 25 else None)
        picked = rng.sample(LANGUAGES, rng.randint(1, 3))
        if i % 3 == 0:
            picked = ["Python", *[p for p in picked if p != "Python"]]
        skills.append("; ".join(picked) if i % 40 else "Fortran")
    return pd.DataFrame(
        {
            "review": reviews,
            "rating": [1 + i % 5 for i in range(rows)],
            "age": [20 + (i * 7) % 50 for i in range(rows)],
            "skills": skills,
        }
    )


@pytest.fixture(scope="module")
def ds(tmp_path_factory: pytest.TempPathFactory) -> quantilo.Dataset:
    return quantilo.from_pandas(frame(), "reviews", cache_dir=tmp_path_factory.mktemp("cache"))


def values(spec: DerivedSpec, texts: Sequence[str | None]) -> list:
    con = duckdb.connect()
    con.execute("CREATE TABLE t (i INTEGER, src VARCHAR)")
    con.executemany("INSERT INTO t VALUES (?, ?)", list(enumerate(texts)))
    return [
        row[0] for row in con.execute(f"SELECT {derived_sql(spec)} FROM t ORDER BY i").fetchall()
    ]


def test_text_parts_sql() -> None:
    texts = ["Is it OK? Yes!!", "  ", None, "Mail me at a@b.org", "see www.x.com 42"]
    of = lambda part: values(DerivedSpec(source="src", part=part), texts)  # noqa: E731
    assert of("length") == [15, None, None, 18, 16]
    assert of("words") == [4, None, None, 4, 3]
    assert of("empty") == [False, True, True, False, False]  # a blank text is empty, never NULL
    assert of("has_question") == [True, None, None, False, False]
    assert of("exclamations") == [2, None, None, 0, 0]
    assert of("uppercase_share") == pytest.approx([4 / 9, None, None, 1 / 13, 0.0], nan_ok=True)
    assert of("contains_url") == [False, None, None, False, True]
    assert of("contains_email") == [False, None, None, True, False]
    assert of("contains_number") == [False, None, None, False, True]


def test_choice_parts_sql() -> None:
    texts = ["Python; SQL;R", "SQL", "", "Pythonic"]
    has = values(DerivedSpec(source="src", part="has", option="Python"), texts)
    assert has == [True, False, None, False]  # whole options only: "Pythonic" is not Python
    assert values(DerivedSpec(source="src", part="count"), texts) == [3, 1, None, 1]


def test_topic_rule_sql() -> None:
    topics = [
        TopicRule(label="1 · fit, size", words={"fit": 0.6, "size": 0.4}),
        TopicRule(label="2 · refund", words={"refund": 0.7, "money": 0.3}),
    ]
    texts = [
        "Great FIT, wrong size",
        "I want my money and a refund",
        "Nothing here",
        None,
        "Fit, refund",
    ]
    found = values(DerivedSpec(source="src", part="topic", topics=topics), texts)
    # R-TOP-05: the highest weighted score, "No topic" without any word, empty stays empty
    assert found == ["1 · fit, size", "2 · refund", "No topic", None, "2 · refund"]


def test_offered_columns(ds: quantilo.Dataset) -> None:
    by_name = {c.name: c for c in ds.types}
    assert by_name["review"].type == "text" and by_name["review"].role in (None, "free_text")
    assert by_name["skills"].role == "choices"
    offered = {c.name: c for c in ds.derived}
    for part in ("Length", "Words", "Empty", "Has question", "Exclamations", "Uppercase share",
                 "Contains URL", "Contains e-mail", "Contains number"):  # fmt: skip
        assert f"review › {part}" in offered
    length = offered["review › Length"]
    assert length.type == "numeric_discrete" and length.parser.kind == "derived"
    assert "(R-DER-01)" in length.reasons[0] and "'review'" in length.reasons[0]
    # R-DER-05: the options picked in at least 1% of the answers, most picked first
    has = [name for name in offered if name.startswith("skills › has ")]
    assert has[0] == "skills › has Python"
    assert len(has) <= MAX_CHOICE_COLUMNS
    assert offered["skills › Count"].type == "numeric_discrete"
    assert not any(name.startswith("rating") for name in offered)  # only text columns


def test_rare_options_are_left_out_but_resolved(tmp_path) -> None:
    data = frame(400)
    data.loc[0, "skills"] = "Python; Cobol"
    ds = quantilo.from_pandas(data, "rare", cache_dir=tmp_path)
    offered = {c.name: c for c in ds.derived}
    assert "skills › has Cobol" not in offered  # 1 answer in 400 (R-DER-05)
    assert any("not offered" in reason for reason in offered["skills › Count"].reasons)
    assert len(ds.where("skills › has Cobol", "in", [True])) == 1  # R-DER-06


def test_label_columns_get_fewer_parts() -> None:
    label = type("V", (), {"name": "item", "type": "text", "role": "label", "excluded": False,
                           "position": 0, "summary": None})()  # fmt: skip
    columns = {"item": label}
    assert resolve(columns, "item › Length") is not None
    assert resolve(columns, "item › Contains number") is not None
    assert resolve(columns, "item › Has question") is None  # R-DER-02: sentences only
    assert resolve(columns, "item › Nonsense") is None
    assert resolve(columns, "other › Length") is None


def test_with_derived_adds_only_what_is_named(ds: quantilo.Dataset) -> None:
    views = ds.types
    extended = with_derived(views, ["review › Length", "rating", None, "nope › Length"])
    assert [v.name for v in extended[len(views) :]] == ["review › Length"]
    assert names_in({"a": ["review › Words", {"b": "x"}]}) == ["review › Words"]


def test_filters_on_derived(ds: quantilo.Dataset) -> None:
    questions = ds.where("review › Has question", "in", [True])
    expected = sum(isinstance(t, str) and "?" in t for t in frame()["review"].to_numpy())
    assert len(questions) == expected > 0
    empty = ds.where("review › Empty", "in", [True])
    assert len(empty) == int(frame()["review"].isna().to_numpy().sum())
    where = where_for(ds.types, [Condition(column="review › Length", op="gt", value=40)])
    con = duckdb.connect()
    count = con.execute(f"SELECT count(*) FROM {scoped_source(str(ds.path), where)}").fetchone()
    assert count and count[0] > 0


def test_unknown_derived_fails(ds: quantilo.Dataset) -> None:
    with pytest.raises(ExplorerError):
        ds.where("review › Nonsense", "gt", 1)


def test_compare_with_derived(ds: quantilo.Dataset) -> None:
    result = ds.compare("rating", "review › Length")
    assert result.kind == "groups"
    colored = ds.compare("rating", "age", color="skills › has Python")
    assert colored.kind == "groups" and colored.color == "skills › has Python"


def test_correlations_with_derived(ds: quantilo.Dataset) -> None:
    names = ["rating", "age", "review › Words", "skills › has Python"]
    result = ds.correlations(columns=names)
    assert {c.name for c in result.matrix.columns} >= {"review › Words", "rating"}
    default = ds.correlations()
    assert not any("›" in c.name for c in default.matrix.columns)  # R-DER-07


def test_kpi_with_derived(ds: quantilo.Dataset) -> None:
    result = ds.kpi(
        {
            "measures": [{"id": "m1", "agg": "mean", "column": "review › Length"}],
            "groups": [{"column": "skills › has Python"}],
        }
    )
    assert result.value is not None and result.value > 20
    assert result.splits and result.splits[0].column == "skills › has Python"


def test_segments_with_derived(ds: quantilo.Dataset) -> None:
    plan = ds.segments(["age", "review › Length", "skills › has Python"]).plan()
    used = {d.column for d in plan.preparation.used}
    assert {"review › Length", "skills › has Python"} <= used


def test_chart_of_derived(ds: quantilo.Dataset) -> None:
    chart = ds.chart("skills › has Python")
    assert chart is not None
