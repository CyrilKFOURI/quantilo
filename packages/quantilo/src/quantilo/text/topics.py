# ruff: noqa: RUF002  (topic labels use "·")
"""Topics of a text column (059): TF-IDF then NMF, and the written rule behind `› Topic`.

R-TOP-02: TF-IDF on the words of R-WRD-01 (in at least 5 texts and at most half of them, 5,000
at most, sub-linear TF), then NMF (`init="nndsvd"`: deterministic), on at most 20,000 texts.
R-TOP-05: each topic keeps its 30 heaviest words, weighted by H[t, w] x idf(w) / |H[t]|^2 (a big
topic does not win on its size alone); a text goes to the topic whose words it uses most, each
time it uses one, "No topic" without any of them. The rule's agreement with the model's own
choice (argmax of W) is measured, not assumed.
"""

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from quantilo.errors import ExplorerError
from quantilo.text.tokens import words
from quantilo.typing.derived import NO_TOPIC
from quantilo.typing.types import TopicRule

MAX_TEXTS = 20_000  # R-TOP-02
MIN_TEXTS = 200  # R-TOP-01
MIN_DF = 5
MAX_DF = 0.5
MAX_FEATURES = 5_000
SHOWN_WORDS = 10  # R-TOP-04
RULE_WORDS = 30  # R-TOP-05
LOWEST_K, HIGHEST_AUTO_K, HIGHEST_K = 3, 8, 12  # R-TOP-03
EXAMPLES = 2
EXAMPLE_CHARS = 300


class TopicsError(ExplorerError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="topics", status_code=422)


@dataclass
class TopicWord:
    word: str
    weight: float  # share of the topic's weight among its shown words


@dataclass
class Topic:
    number: int
    label: str  # "1 · dress, fabric, color" (R-TOP-04)
    words: list[TopicWord]
    texts: int  # texts the rule gives it (R-TOP-07)
    share: float
    examples: list[str] = field(default_factory=list)


@dataclass
class Topics:
    column: str
    texts: int  # texts the topics were learnt on
    sampled: bool  # at most MAX_TEXTS texts (R-TOP-02)
    k: int
    k_rule: str  # why this number of topics (R-TOP-03)
    topics: list[Topic]
    no_topic: float  # share of the texts with none of the rule's words
    agreement: float  # share of the texts where the rule and the model agree (R-TOP-05)
    sentence: str
    rules: list[TopicRule]  # what Keep as column stores (R-TOP-06)
    reasons: list[str]
    kept: bool = False  # these topics are the ones kept on the column


def auto_k(texts: int) -> int:
    """R-TOP-03: round(sqrt(texts / 100)), from 3 to 8."""
    return int(min(HIGHEST_AUTO_K, max(LOWEST_K, round(math.sqrt(texts / 100)))))


def topic_words(text: str, rules: Sequence[TopicRule]) -> Counter[str]:
    """The topics' words a text uses, each use counted, read as the SQL rule reads them
    (`typing.derived.topic_tokens_sql`: whole words in any script, lower case, no accents)."""
    known = {word for rule in rules for word in rule.words}
    return Counter(word for word in words(text) if word in known)


def rule_topic(used: Counter[str], rules: Sequence[TopicRule]) -> int | None:
    """R-TOP-05 on one text's words (with their counts): its topic's index, None for none."""
    scores = [sum(weight * used[word] for word, weight in rule.words.items()) for rule in rules]
    best = max(scores, default=0.0)
    return scores.index(best) if best > 0 else None


def fit_topics(
    column: str,
    texts: Sequence[str],
    stop: frozenset[str],
    k: int | None = None,
    sampled: bool = False,
) -> Topics:
    """Topics of `texts` (already sampled, R-TOP-02), `k` of them or as many as R-TOP-03 says."""
    from sklearn.decomposition import NMF
    from sklearn.feature_extraction.text import TfidfVectorizer

    if len(texts) < MIN_TEXTS:
        raise TopicsError(f"{len(texts)} texts: topics need at least {MIN_TEXTS} (R-TOP-01).")
    if k is not None and not LOWEST_K <= k <= HIGHEST_K:
        raise TopicsError(f"Pick from {LOWEST_K} to {HIGHEST_K} topics.")
    chosen = k if k is not None else auto_k(len(texts))
    k_rule = (
        f"{chosen} topics: your choice"
        if k is not None
        else f"{chosen} topics for {len(texts):,} texts: round(√(texts / 100)), "
        f"from {LOWEST_K} to {HIGHEST_AUTO_K} (R-TOP-03)"
    )
    vectorizer = TfidfVectorizer(
        analyzer=lambda text: words(text, stop),  # type: ignore[arg-type]
        min_df=MIN_DF,
        max_df=MAX_DF,
        max_features=MAX_FEATURES,
        sublinear_tf=True,
    )
    try:
        matrix = vectorizer.fit_transform(texts)
    except ValueError:
        raise TopicsError("Too few words repeat across the texts to find topics.") from None
    vocabulary = vectorizer.get_feature_names_out()
    if len(vocabulary) < chosen * 3:
        raise TopicsError(
            f"{len(vocabulary)} words repeat across the texts: too few for {chosen} topics."
        )
    model = NMF(n_components=chosen, init="nndsvd", random_state=0, max_iter=400)  # type: ignore[arg-type]
    weights = model.fit_transform(matrix)
    components = model.components_
    idf = vectorizer.idf_

    rules: list[TopicRule] = []
    shown: list[list[TopicWord]] = []
    for index, row in enumerate(components):
        top = np.argsort(-row, kind="stable")[:SHOWN_WORDS]
        total = float(row[top].sum()) or 1.0
        shown.append([TopicWord(str(vocabulary[i]), float(row[i] / total)) for i in top])
        heavy = row * idf
        best = np.argsort(-heavy, kind="stable")[:RULE_WORDS]
        size = float((row**2).sum()) or 1.0
        label = f"{index + 1} · " + ", ".join(str(vocabulary[i]) for i in top[:3])
        rules.append(
            TopicRule(
                label=label,
                words={str(vocabulary[i]): float(f"{heavy[i] / size:.6e}") for i in best},
            )
        )

    by_rule = [rule_topic(topic_words(text, rules), rules) for text in texts]
    by_model = [int(np.argmax(w)) if w.max() > 0 else None for w in weights]
    agree = sum(1 for r, m in zip(by_rule, by_model, strict=True) if r == m)
    agreement = agree / len(texts)
    counts = [sum(1 for r in by_rule if r == index) for index in range(chosen)]
    none = sum(1 for r in by_rule if r is None)

    topics = []
    for index, rule in enumerate(rules):
        mine = [i for i, r in enumerate(by_rule) if r == index]
        mine.sort(key=lambda i: -float(weights[i, index]))
        examples = [texts[i][:EXAMPLE_CHARS] for i in mine[:EXAMPLES]]
        topics.append(
            Topic(
                index + 1,
                rule.label,
                shown[index],
                counts[index],
                counts[index] / len(texts),
                examples,
            )
        )
    largest = max(topics, key=lambda t: t.texts)
    sentence = (
        f"{chosen} topics in {len(texts):,} texts; the largest, “{largest.label}”, "
        f"covers {largest.share:.0%} of them."
    )
    reasons = [
        k_rule,
        "R-TOP-02: TF-IDF on words in at least 5 texts and at most half of them, then NMF",
        f"R-TOP-05: the column's rule agrees with the model on {agreement:.0%} of the texts; "
        f"{none / len(texts):.0%} use none of the topics' words ({NO_TOPIC})",
    ]
    return Topics(
        column, len(texts), sampled, chosen, k_rule, topics, none / len(texts), agreement,
        sentence, rules, reasons,
    )  # fmt: skip
