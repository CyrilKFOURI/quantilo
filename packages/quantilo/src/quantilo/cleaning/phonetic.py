"""Keys that two spellings of one value share (097, A-01 to A-03), after OpenRefine's clustering.

- `fingerprint`: case, accents, punctuation and word order ignored ('Cruise, Tom' = 'Tom Cruise').
- `ngram_fingerprint`: the same on letter pairs, spaces dropped ('New York' = 'NewYork'); `ordered`
  keeps the pairs in their order (N14: 'Pierre-Philippe' and 'Philippe Pierre' stay apart).
- `items` / `items_key`: the items of a list value ('vegan, no gluten'), keyed one by one (N13).
- `phonetic`: a sound key for English and French (no dependency, Pyodide-friendly): the rules both
  languages share, then the vowels after the first letter dropped ('Keyboard' = 'Kiboard').
"""

import re
import unicodedata
from collections.abc import Callable

PUNCTUATION = re.compile(r"[^\w\s]|_")
NOT_LETTER = re.compile(r"[^a-z ]+")
DIGITS = re.compile(r"\d+")
ITEM_SEPARATOR = re.compile(r"[,;|]")
# Applied in order; each pattern sees the result of the previous ones.
SOUNDS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"eaux?|aux?"), "o"),
    (re.compile(r"x(?!$)"), "ks"),
    (re.compile(r"ph"), "f"),
    (re.compile(r"sch|ch|sh"), "x"),
    (re.compile(r"ck|qu|q"), "k"),
    (re.compile(r"c(?=[eiy])"), "s"),
    (re.compile(r"c"), "k"),
    (re.compile(r"g(?=[eiy])"), "j"),
    (re.compile(r"gn"), "n"),
    (re.compile(r"th"), "t"),
    (re.compile(r"^k(?=n)|^w(?=r)"), ""),
    (re.compile(r"z"), "s"),
    (re.compile(r"w"), "v"),
    (re.compile(r"y"), "i"),
    (re.compile(r"h"), ""),
]
VOWELS = re.compile(r"(?<!^)[aeiou]")
DOUBLED = re.compile(r"(.)\1+")


def fold(value: str) -> str:
    """Lowercase, accents removed, spaces normalized."""
    if value.isascii():  # no accent to remove: the fast way
        return " ".join(value.lower().split())
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    plain = "".join(char for char in decomposed if not unicodedata.combining(char))
    return " ".join(plain.split())


def fingerprint(value: str, ordered: bool = False) -> str:
    """OpenRefine's key-collision fingerprint, punctuation read as a word break. `ordered` keeps
    the word order (names of people: 'Jean-Pierre' and 'Pierre-Jean' are two first names)."""
    words = PUNCTUATION.sub(" ", fold(value)).split()
    return " ".join(words if ordered else sorted(set(words)))


def ngram_fingerprint(value: str, size: int = 2, ordered: bool = False) -> str:
    """The sorted letter pairs of the value without spaces or punctuation. Unlike OpenRefine, a
    pair that repeats counts twice: with a set, 'Berge' and 'Berger' (be, er, rg, ge) collide.
    `ordered` keeps the pairs in the order of the text: only spaces and separators differ
    ('VANDAM ST' = 'VAN DAM ST'), letters moved around do not ('Pierre-Philippe' ≠ 'Philippe
    Pierre')."""
    text = PUNCTUATION.sub("", fold(value)).replace(" ", "")
    if len(text) < size:
        return text
    pairs = [text[i : i + size] for i in range(len(text) - size + 1)]
    return " ".join(pairs if ordered else sorted(pairs))


def items(value: str) -> list[str]:
    """The items of a list value, split on `,` `;` `|`, blanks left out."""
    return [item.strip() for item in ITEM_SEPARATOR.split(value) if item.strip()]


def items_key(value: str, key: Callable[[str], str]) -> str:
    """N13: a list's key is the set of its items' keys, in any order: 'No gluten, No GMOs' =
    'No GMOs; no gluten', but 'en:vegetarian' ≠ 'en:non-vegetarian' (the words of two items are
    never pooled, as one fingerprint of the whole value would)."""
    return " | ".join(sorted({k for k in (key(item) for item in items(value)) if k}))


def _word(word: str) -> str:
    for pattern, replacement in SOUNDS:
        word = pattern.sub(replacement, word)
    word = DOUBLED.sub(r"\1", word)
    return VOWELS.sub("", word)


def phonetic(value: str) -> str:
    """A sound key, word by word: 'Pariss' and 'Paris' → 'prs'; 'Parisot' → 'prst'."""
    words = NOT_LETTER.sub(" ", PUNCTUATION.sub(" ", fold(value))).split()
    return " ".join(_word(word) for word in words if word)


def digits(value: str) -> tuple[str, ...]:
    """The numbers a value carries: two values with other numbers are two things (C-22)."""
    return tuple(DIGITS.findall(value))


def damerau(a: str, b: str, limit: int) -> int:
    """Optimal string alignment distance (a swap of two neighbours counts one), or `limit` + 1
    as soon as it is sure to exceed `limit`."""
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    before: list[int] = []
    previous = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        current = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                current[j] = min(current[j], before[j - 2] + 1)
        if min(current) > limit:
            return limit + 1
        before, previous = previous, current
    return previous[-1]
