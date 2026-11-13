# ruff: noqa: RUF001  (derived names use their separator, as dates do)
"""059: distinctive words by group and topics."""

import random

import duckdb
import pandas as pd
import pytest

import quantilo
from quantilo.errors import ExplorerError
from quantilo.text.distinctive import MIN_TEXTS, distinctive
from quantilo.text.service import TextError, TextService, words_between
from quantilo.text.tokens import stop_words, words
from quantilo.text.topics import TopicsError, auto_k, fit_topics, rule_topic, topic_words
from quantilo.typing.derived import NO_TOPIC, topic_sql
from quantilo.typing.types import TopicRule

THEMES = {
    "fit": ["size", "small", "large", "fit", "tight", "loose", "ordered"],
    "fabric": ["fabric", "soft", "cotton", "material", "thin", "quality", "wash"],
    "refund": ["refund", "return", "money", "late", "shipping", "order", "broken"],
}
FILLER = ["the", "dress", "was", "and", "really", "it", "is", "very", "my", "for"]


def corpus(rows: int = 600, seed: int = 3) -> pd.DataFrame:
    rng = random.Random(seed)
    texts, themes, ratings = [], [], []
    for i in range(rows):
        theme = list(THEMES)[i % 3]
        picked = rng.sample(THEMES[theme], 3) + rng.sample(FILLER, 4)
        rng.shuffle(picked)
        texts.append(" ".join(picked).capitalize() + ".")
        themes.append(theme)
        ratings.append(1 if theme == "refund" else rng.choice([4, 5]))
    return pd.DataFrame({"review": texts, "theme": themes, "rating": ratings})


@pytest.fixture(scope="module")
def ds(tmp_path_factory: pytest.TempPathFactory) -> quantilo.Dataset:
    return quantilo.from_pandas(corpus(), "corpus", cache_dir=tmp_path_factory.mktemp("cache"))


def test_tokens_drop_stop_words_accents_and_numbers() -> None:
    stop = stop_words("en")
    assert words("The DRESS didn't fit: 2 sizes, Été 42!", stop) == ["dress", "fit", "sizes", "ete"]
    assert "didn" in stop and "the" in stop


def test_distinctive_finds_the_group_words() -> None:
    data = corpus()
    found = {g.group: g for g in distinctive(list(data.review), list(data.theme))}
    assert {w.word for w in found["refund"].words[:5]} <= set(THEMES["refund"])
    top = found["refund"].words[0]
    assert top.share > 0.3 and top.elsewhere == 0 and top.score >= 3
    assert not {w.word for w in found["fit"].words} & set(THEMES["refund"])


def test_small_groups_get_no_words() -> None:
    texts = ["refund money now"] * 10 + ["soft fabric"] * 100
    groups = ["tiny"] * 10 + ["big"] * 100
    found = {g.group: g for g in distinctive(texts, groups)}
    assert found["tiny"].words == [] and f"fewer than {MIN_TEXTS}" in (found["tiny"].note or "")


def test_compare_text_with_category(ds: quantilo.Dataset) -> None:
    result = ds.compare("rating", "review")
    assert result.kind == "words" and result.words is not None
    groups = {g.group: g for g in result.words.groups}
    assert set(groups) == {"1", "4", "5"}
    assert groups["1"].words[0].word in THEMES["refund"]
    assert [g.group for g in result.words.groups] == ["1", "4", "5"]  # ordered levels
    assert result.message and "marks rating = 1" in result.message
    assert ds.distinctive_words("review", "theme").groups


def test_compare_text_with_a_number_fails(tmp_path) -> None:
    data = corpus()
    data["score"] = [i * 0.37 for i in range(len(data))]
    ds = quantilo.from_pandas(data, "num", cache_dir=tmp_path)
    with pytest.raises(ExplorerError, match="R-WRD-06"):
        ds.compare("score", "review")


def test_auto_k() -> None:
    assert auto_k(200) == 3 and auto_k(4_900) == 7 and auto_k(1_000_000) == 8


def test_topics_find_the_themes(ds: quantilo.Dataset) -> None:
    topics = ds.topics("review", k=3)
    assert topics.k == 3 and len(topics.topics) == 3
    found = [set(w.word for w in t.words[:4]) for t in topics.topics]
    for theme in THEMES.values():
        assert any(len(words & set(theme)) >= 3 for words in found)
    assert topics.agreement > 0.9
    assert sum(t.share for t in topics.topics) + topics.no_topic == pytest.approx(1)
    assert "R-TOP-05" in " ".join(topics.reasons)


def test_topic_rule_sql_matches_python() -> None:
    data = corpus(400)
    stop = stop_words("en")
    topics = fit_topics("review", list(data.review), stop, 3)
    con = duckdb.connect()
    con.register("t", data)
    sql = topic_sql("review", topics.rules)
    found = [row[0] for row in con.execute(f"SELECT {sql} FROM t").fetchall()]
    expected = []
    for text in data.review:
        index = rule_topic(topic_words(text, topics.rules), topics.rules)
        expected.append(NO_TOPIC if index is None else topics.rules[index].label)
    assert found == expected


def test_topic_rule_reads_whole_words_in_any_script() -> None:
    rules = [
        TopicRule(label="1 · groß", words={"groß": 1.0, "œuvre": 1.0, "книга": 1.0}),
        TopicRule(label="2 · dress", words={"dress": 1.0, "fabric": 0.5}),
    ]
    texts = [
        "Sehr GROß, wirklich groß!",  # ß at the end: an ASCII \b would miss it
        "Une œuvre; a dress",  # œ at the start
        "Хорошая книга",  # Cyrillic
        "dress fabric fabric",  # consecutive words all count
        "dresses and fabrics",  # longer words do not
        "Dress2fabric_dress",  # digits and _ cut words, as R-WRD-01
        "",
        None,
    ]
    con = duckdb.connect()
    con.register("t", pd.DataFrame({"review": texts}))
    found = [
        row[0] for row in con.execute(f"SELECT {topic_sql('review', rules)} FROM t").fetchall()
    ]
    assert found == [
        "1 · groß",
        "1 · groß",
        "1 · groß",
        "2 · dress",
        NO_TOPIC,
        "2 · dress",
        "No topic",
        None,
    ]
    assert topic_words("Dress2fabric_dress", rules) == {"dress": 2, "fabric": 1}
    for text, label in zip(texts[:6], found[:6], strict=True):
        index = rule_topic(topic_words(text, rules), rules)
        assert (NO_TOPIC if index is None else rules[index].label) == label


def test_topics_need_enough_texts() -> None:
    with pytest.raises(TopicsError, match="at least 200"):
        fit_topics("review", ["a b c"] * 50, frozenset())
    with pytest.raises(TopicsError):
        fit_topics("review", list(corpus().review), frozenset(), k=20)


def test_keep_topics_makes_a_derived_column(tmp_path) -> None:
    ds = quantilo.from_pandas(corpus(), "keep", cache_dir=tmp_path)
    assert "review › Topic" not in {c.name for c in ds.derived}
    topics = ds.topics("review", k=3)
    ds.keep_topics(topics)
    assert ds.topics("review", k=3).kept
    assert "review › Topic" in {c.name for c in ds.derived}
    labels = [t.label for t in topics.topics]
    chart = ds.compare("review › Topic", "rating")
    assert chart.kind == "contingency"
    one = ds.where("review › Topic", "in", [labels[0]])
    assert len(one) == topics.topics[0].texts  # R-TOP-07: the shares are the rule's
    ds.retype("review", "categorical_nominal")
    assert "review › Topic" not in {c.name for c in ds.derived}  # R-TOP-06


def test_words_by_rows_and_between_versions(ds: quantilo.Dataset, tmp_path) -> None:
    service = ds._state.service(TextService)
    data = corpus()
    found = service.words_by_rows(ds._id, "review", list(range(len(data))), list(data.theme))
    assert {g.group for g in found.groups} == set(THEMES)
    with pytest.raises(TextError):
        service.words_by_rows(ds._id, "theme", [0], ["a"])
    older = corpus(seed=9)
    older["review"] = [t.replace("refund", "parcel") for t in older.review]
    before = quantilo.from_pandas(older, "old", cache_dir=tmp_path)
    changes = words_between(
        "review",
        (str(before.path), before.column("review")),
        (str(ds.path), ds.column("review")),
    )
    groups = {g.group: g for g in changes.groups}
    assert groups["this version"].words[0].word == "refund"
    assert groups["previous version"].words[0].word == "parcel"
