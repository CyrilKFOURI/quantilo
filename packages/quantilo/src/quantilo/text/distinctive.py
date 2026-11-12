"""Words over-represented in a group of texts (059, R-WRD-02 to R-WRD-05).

Each word counts once per text. A group is compared with every other text by the log-odds ratio
with an informative Dirichlet prior (Monroe, Colaresi and Quinn, 2008): the prior is the word's
share over all texts, worth `PRIOR_TEXTS` texts; the score is that log-odds divided by its
standard error (a z). Kept: z >= `MIN_Z`, used by at least `MIN_GROUP_TEXTS` texts of the group.
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from quantilo.text.tokens import distinct_words

PRIOR_TEXTS = 10.0  # R-WRD-03: the prior is worth 10 texts
MIN_Z = 3.0  # R-WRD-04: thousands of words tested at once
MIN_GROUP_TEXTS = 5  # R-WRD-04: a word used by fewer texts of the group is one author's
TOP_WORDS = 10  # R-WRD-04
MAX_TEXTS = 20_000  # R-WRD-05
MAX_GROUPS = 12  # R-WRD-05
MIN_TEXTS = 20  # R-WRD-05: under 20 texts no vocabulary stands out (as R-SEL-03)


@dataclass
class DistinctiveWord:
    word: str
    score: float  # z of the smoothed log-odds ratio (R-WRD-03)
    share: float  # share of the group's texts that use it
    elsewhere: float  # share of the other texts that use it
    texts: int  # texts of the group that use it


@dataclass
class GroupWords:
    group: str
    texts: int  # texts of the group read
    words: list[DistinctiveWord] = field(default_factory=list)
    note: str | None = None  # why it has no words


@dataclass
class DistinctiveWords:
    """The words of a text column that mark each group (Compare, Segments, Changes)."""

    column: str
    by: str  # what the groups are (a column, "segment", "version")
    texts: int  # texts read
    sampled: bool  # at most MAX_TEXTS texts were read (R-WRD-05)
    groups: list[GroupWords]
    groups_left: int  # smaller groups not read (R-WRD-05)
    sentence: str
    rules: list[str]


RULES = [
    "R-WRD-01: words of 2 letters or more, stop words and numbers left out",
    "R-WRD-02: a word counts once per text",
    f"R-WRD-03/04: smoothed log-odds against every other text, kept when z >= {MIN_Z:g} "
    f"and used by at least {MIN_GROUP_TEXTS} texts of the group",
]


def _z(in_group: np.ndarray, group_size: int, total_used: np.ndarray, total: int) -> np.ndarray:
    """R-WRD-03 for every word at once."""
    prior = PRIOR_TEXTS * total_used / total
    prior_not = PRIOR_TEXTS - prior
    rest = total_used - in_group
    rest_size = total - group_size
    used_g, unused_g = in_group + prior, group_size - in_group + prior_not
    used_r, unused_r = rest + prior, rest_size - rest + prior_not
    delta = np.log(used_g / unused_g) - np.log(used_r / unused_r)
    variance = 1 / used_g + 1 / unused_g + 1 / used_r + 1 / unused_r
    return delta / np.sqrt(variance)


def distinctive(
    texts: Sequence[str], groups: Sequence[str], stop: frozenset[str] = frozenset()
) -> list[GroupWords]:
    """For each group (in the order of their first text), its over-represented words."""
    bags = [distinct_words(text, stop) for text in texts]
    order = list(dict.fromkeys(groups))
    sizes = Counter(groups)
    overall: Counter[str] = Counter()
    by_group: dict[str, Counter[str]] = {group: Counter() for group in order}
    for bag, group in zip(bags, groups, strict=True):
        overall.update(bag)
        by_group[group].update(bag)
    vocabulary = [word for word, used in overall.items() if used >= MIN_GROUP_TEXTS]
    total = len(bags)
    found: list[GroupWords] = []
    used_all = np.array([overall[word] for word in vocabulary], dtype=float)
    for group in order:
        size = sizes[group]
        if size < MIN_TEXTS:
            found.append(GroupWords(group, size, note=f"{size} texts: fewer than {MIN_TEXTS}"))
            continue
        if size == total or not vocabulary:
            found.append(GroupWords(group, size, note="nothing to compare it with"))
            continue
        in_group = np.array([by_group[group][word] for word in vocabulary], dtype=float)
        scores = _z(in_group, size, used_all, total)
        kept = np.flatnonzero((scores >= MIN_Z) & (in_group >= MIN_GROUP_TEXTS))
        best = kept[np.argsort(-scores[kept], kind="stable")][:TOP_WORDS]
        rest = total - size
        words = [
            DistinctiveWord(
                vocabulary[i],
                round(float(scores[i]), 2),
                float(in_group[i] / size),
                float((used_all[i] - in_group[i]) / rest) if rest else 0.0,
                int(in_group[i]),
            )
            for i in best
        ]
        note = None if words else f"no word stands out (z >= {MIN_Z:g})"
        found.append(GroupWords(group, size, words, note))
    return found


def sentence(column: str, by: str, groups: list[GroupWords]) -> str:
    """The conclusion above the cards: the strongest word of the strongest group."""
    best = max(
        ((g, g.words[0]) for g in groups if g.words), key=lambda pair: pair[1].score, default=None
    )
    if best is None:
        return f"No word of {column} stands out in any group of {by}."
    group, word = best
    return (
        f"“{word.word}” marks {by} = {group.group}: {word.share:.0%} of its texts use "
        f"it, {word.elsewhere:.0%} of the others."
    )
