"""052 — free text as a model input: TF-IDF + SVD (R-TEXT-01 to 05), the words that move the
predictions, the vocabulary the check of new data compares with."""

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest

import quantilo as dx
from quantilo.ml import explain as explainer
from quantilo.ml import models as catalogue
from quantilo.ml import predict as use
from quantilo.ml import preparation as prep
from quantilo.ml import text as textual
from quantilo.ml.data import Field
from quantilo.ml.facade import TrainedModel
from quantilo.ml.train import train
from quantilo.typing.types import Parser

RAW = Parser(kind="raw")
N = 1_000
FILLERS = [
    "the parcel came on monday",
    "ordered for my sister",
    "colour as on the picture",
    "second order this year",
    "box was a little dented",
    "delivery took three days",
    "bought it for the kitchen",
    "size runs a bit small",
]
BAD = ["asked for a refund", "arrived broken", "refund please", "stopped working, broken"]
GOOD = ["really great", "love it", "great value", "my kids love it"]


def make_reviews(n: int = N, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    unhappy = rng.random(n) < 0.4
    reviews = []
    for sad in unhappy:
        words = [str(rng.choice(FILLERS))]
        if rng.random() < 0.85:  # most reviews say it; some do not
            words.append(str(rng.choice(BAD if sad else GOOD)))
        words.append(str(rng.choice(FILLERS)))
        reviews.append(", ".join(words))
    review = pd.Series(reviews, dtype=object)
    review[rng.random(n) < 0.05] = None
    price = rng.lognormal(3, 0.5, n)
    return pd.DataFrame(
        {
            "review": review,
            "price": price,
            "returned": np.where(unhappy, "yes", "no"),
            "__row": np.arange(n),
        }
    )


REVIEWS = make_reviews()
FIELDS = [
    Field("review", "text", RAW, role="free_text"),
    Field("price", "numeric_continuous", RAW),
    Field("returned", "categorical_nominal", RAW),
]
TARGET = FIELDS[2]


def decision(preparation: prep.Preparation, column: str) -> prep.Decision:
    return next(d for d in preparation.decisions if d.column == column)


# --- the step itself -----------------------------------------------------------------------------


def test_text_features_tfidf_then_svd() -> None:
    step = textual.TextFeatures("words").fit(REVIEWS[["review"]])

    out = step.transform(REVIEWS[["review"]])
    assert out.shape == (N, min(textual.COMPONENTS, len(step.vocabulary) - 1))
    assert np.isfinite(out).all()
    names = step.get_feature_names_out(["review"])
    assert names[0] == "review_text1" and len(names) == out.shape[1]
    assert "refund" in step.vocabulary and "arrived broken" in step.vocabulary  # one and two words
    assert 0 <= step.unknown_share_ < 1


def test_vocabulary_and_svd_are_learnt_on_a_bounded_sample() -> None:
    step = textual.TextFeatures("words", fit_rows=300).fit(REVIEWS[["review"]])

    assert step.fit_rows_ == 300
    assert step.transform(REVIEWS[["review"]]).shape[0] == N  # applied to every row


def test_characters_and_tiny_vocabularies() -> None:
    chars = textual.TextFeatures("chars").fit(REVIEWS[["review"]])
    assert all(3 <= len(t) <= 5 for t in chars.vocabulary)

    tiny = pd.DataFrame({"t": ["a b", "a b", "a", None]})
    step = textual.TextFeatures("words").fit(tiny)  # single letters: no token at all
    assert step.transform(tiny).shape == (4, 1)


def test_the_gap_counts_unknown_terms_and_rows_without_a_known_term() -> None:
    cut = textual.cutter("words")

    found = textual.gap(["love it", "zebra quartz", ""], cut, {"love", "it", "love it"})

    assert found.unknown == pytest.approx(3 / 6)  # love, it, love it | zebra, quartz, zebra quartz
    assert found.unmatched == pytest.approx(1 / 2)  # the empty text does not count
    assert found.rows.tolist() == [False, True, False]


# --- the rules -----------------------------------------------------------------------------------


def test_free_text_is_used_by_rule_with_its_reasons() -> None:
    preparation = prep.decide(REVIEWS, FIELDS, TARGET, "binary", catalogue.LINEAR)

    review = decision(preparation, "review")
    assert review.use
    assert review.rules == ["R-TEXT-01", "R-TEXT-02", "R-TEXT-03", "R-TEXT-04", "R-TEXT-05"]
    assert review.encoding == "words" and review.transform == "tfidf_svd"
    assert review.scaling == "standard"
    assert all(review.why)
    assert [o.value for o in review.options["encoding"]] == ["words", "chars", "embeddings"]


def test_short_labels_are_cut_into_characters() -> None:
    frame = REVIEWS.copy()
    frame["review"] = np.random.default_rng(1).choice(["red mug", "blue tin box", "mug"], N)

    preparation = prep.decide(frame, FIELDS, TARGET, "binary", catalogue.LINEAR)

    review = decision(preparation, "review")
    assert review.encoding == "chars" and "short labels" in review.why[1]


def test_the_user_can_leave_text_out_or_change_its_cut() -> None:
    off = prep.Choices(columns={"review": prep.ColumnChoice(use=False)})
    chars = prep.Choices(columns={"review": prep.ColumnChoice(encoding="chars")})

    left = decision(
        prep.decide(REVIEWS, FIELDS, TARGET, "binary", catalogue.LINEAR, None, off), "review"
    )
    cut = decision(
        prep.decide(REVIEWS, FIELDS, TARGET, "binary", catalogue.LINEAR, None, chars), "review"
    )

    assert not left.use and left.why[-1] == "Left out: your choice."
    assert cut.encoding == "chars" and cut.manual == ["encoding"] and cut.scaling == "standard"


def test_too_few_texts_and_distances_leave_text_out() -> None:
    frame = REVIEWS.copy()
    frame["review"] = None
    frame.loc[:9, "review"] = "a few words here"

    sparse = decision(prep.decide(frame, FIELDS, TARGET, "binary", catalogue.LINEAR), "review")
    distance = decision(prep.decide_distance(REVIEWS, FIELDS[:2]), "review")

    assert not sparse.use and sparse.rules[0] in ("R-SEL-03", "R-SEL-05")
    assert not distance.use and distance.rules == ["R-SEL-03"]


def test_trees_keep_the_themes_as_they_are() -> None:
    trees = prep.ModelProfile(tree_based=True, handles_missing=True)

    review = decision(prep.decide(REVIEWS, FIELDS, TARGET, "binary", trees), "review")

    assert review.use and review.scaling is None and "R-TEXT-05" in review.rules


# --- training and explaining ---------------------------------------------------------------------


def run(model: str, use_text: bool = True):
    choices = None if use_text else prep.Choices(columns={"review": prep.ColumnChoice(use=False)})
    return train(REVIEWS, FIELDS, TARGET, "binary", model, None, N, False, choices=choices)


def test_text_improves_the_model() -> None:
    with_text, without = run("logistic_regression"), run("logistic_regression", False)

    assert with_text.score is not None and without.score is not None
    assert with_text.score > 0.85 and with_text.score > without.score + 0.2
    assert with_text.preparation.inputs > 10  # price + the themes of the text


def test_words_of_a_linear_model_are_exact() -> None:
    trained = run("logistic_regression")
    x = cast(pd.DataFrame, REVIEWS[["review", "price"]])

    words = explainer.words(trained.pipeline, x, "binary", "yes")

    assert len(words) == 1 and words[0].column == "review" and words[0].method == "linear"
    up = [w.term for w in words[0].up]
    down = [w.term for w in words[0].down]
    assert {"refund", "broken"} & set(up[:5]) and {"love", "great"} & set(down[:5])
    assert all(0 < w.rows <= 1 for w in words[0].up)
    assert words[0].sentence


def test_words_of_a_forest_compare_rows_with_and_without() -> None:
    trained = run("random_forest")
    x = cast(pd.DataFrame, REVIEWS[["review", "price"]])

    words = explainer.words(trained.pipeline, x, "binary", "yes")

    assert words[0].method == "with_without"
    assert {"refund", "broken"} & {w.term for w in words[0].up[:5]}


def test_explanation_carries_words_and_skips_text_in_dependence_and_pairs() -> None:
    trained = run("logistic_regression")
    x = cast(pd.DataFrame, REVIEWS[["review", "price"]]).reset_index(drop=True)
    y = cast(pd.Series, REVIEWS["returned"])

    found = explainer.explain(
        trained.pipeline, x, y, REVIEWS["__row"].to_numpy(), "binary", "roc_auc", [], "yes"
    )

    assert found.words and found.words[0].column == "review"
    assert all(d.column != "review" for d in found.dependence)
    assert all("review" not in (p.first, p.second) for p in found.pairs)
    assert found.shap is not None and "review" in found.shap.columns
    text_feature = next(f for f in found.shap.features if f.column == "review")
    assert text_feature.reading is None


# --- using the model on new data (037) -----------------------------------------------------------


@pytest.fixture(scope="module")
def cache(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("cache")


def dataset(frame: pd.DataFrame, name: str, cache: Path) -> dx.Dataset:
    return dx.from_pandas(frame.drop(columns="__row"), name, cache_dir=cache).retype(
        "review", "text"
    )


@pytest.fixture(scope="module")
def returns(cache: Path) -> TrainedModel:
    data = dataset(REVIEWS.drop(columns="price"), "reviews", cache)
    return data.ml("returned").train(model="logistic_regression")


def test_the_card_keeps_the_vocabulary(returns: TrainedModel) -> None:
    card = returns.card
    review = next(c for c in card.columns if c.name == "review")

    assert review.text is not None and review.categories is None
    assert review.text.analyzer == "words" and "refund" in review.text.terms
    assert len(review.text.terms) <= textual.TERMS and 0 <= review.text.unknown_share < 1


def test_the_same_kind_of_text_is_ready(returns: TrainedModel, cache: Path) -> None:
    fresh = make_reviews(400, seed=9).drop(columns="price")

    checked = returns.check(dataset(fresh, "fresh", cache))

    review = next(c for c in checked.columns if c.column == "review")
    assert checked.level == "ready", checked.message
    assert review.unknown_terms is not None and review.training_unknown_terms is not None
    assert abs(review.unknown_terms - review.training_unknown_terms) < use.TEXT_GAP


def test_unknown_vocabulary_asks_for_a_check(
    returns: TrainedModel, cache: Path, tmp_path: Path
) -> None:
    rng = np.random.default_rng(5)
    other = [f"zorglub{i} quintar wexel" for i in rng.integers(0, 50, 300)]
    fresh = make_reviews(400, seed=9).drop(columns="price")
    fresh.loc[:299, "review"] = other
    data = dataset(fresh, "foreign", cache)

    checked = returns.check(data)
    predicted = returns.predict(data)

    review = next(c for c in checked.columns if c.column == "review")
    assert checked.level == "check" and "unknown to the model" in checked.message
    assert review.note and "no known term" in review.note
    flagged = predicted["less_reliable"].to_numpy(dtype=bool)
    assert flagged[:300].all() and "review: no known word" in predicted["why"].iloc[0]
    # the exported zip reloads with its vocabulary and checks the same way
    loaded = dx.load_model(returns.export(tmp_path / "returns.zip"))
    again = loaded.check(data)
    assert again.level == "check" and again.message == checked.message


def test_words_of_several_classes_name_the_class_they_raise() -> None:
    frame = REVIEWS.copy()
    review = frame["review"].fillna("")
    frame["mood"] = np.where(
        review.str.contains("refund|broken"),
        "angry",
        np.where(review.str.contains("love|great"), "happy", "quiet"),
    )
    fields = [*FIELDS[:2], Field("mood", "categorical_nominal", RAW)]
    trained = train(frame, fields, fields[2], "multiclass", "random_forest", None, N, False)

    words = explainer.words(
        trained.pipeline, cast(pd.DataFrame, frame[["review", "price"]]), "multiclass", None
    )

    found = {w.term: w.towards for w in words[0].up}
    assert words[0].method == "with_without" and found.get("refund") == "angry"
    assert (
        "probability of angry" in words[0].sentence or "probability of happy" in words[0].sentence
    )
