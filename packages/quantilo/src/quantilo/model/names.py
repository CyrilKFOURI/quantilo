"""Column names compared the way a person reads them (088, R-REL-02, R-KEY-04).

`customer_id`, `Customer ID`, `CustomerKey` and `customerID` all name the same thing: split the
words, drop a key suffix, put each word in the singular. A bare `id` takes the table's name
(`customers.id` reads as `customer`)."""

import re
import unicodedata

# Words that only say "this is a key" (R-KEY-04, R-REL-02).
KEY_SUFFIXES = frozenset(
    {
        "id",
        "ids",
        "key",
        "code",
        "no",
        "nr",
        "num",
        "nbr",
        "number",
        "ref",
        "reference",
        "identifier",
        "uuid",
        "guid",
        "sk",
        "fk",
        "pk",
    }
)
# Short shared starts below this length are chance ("id", "cat"), not a common word.
MIN_PREFIX = 4

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_SPLIT = re.compile(r"[^a-z0-9]+")


def words(name: str) -> list[str]:
    """Lower-case words of a name: camel case, `_`, `-`, spaces and accents handled."""
    plain = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    spaced = _CAMEL.sub(" ", plain)
    return [word for word in _SPLIT.split(spaced.lower()) if word]


def singular(word: str) -> str:
    """English plural to singular, for the endings that matter in table names."""
    if len(word) <= 3 or word.endswith(("ss", "us", "is")):
        return word
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith(("ches", "shes", "xes", "sses")):
        return word[:-2]
    if word.endswith("s"):
        return word[:-1]
    return word


def has_key_word(name: str) -> bool:
    """`orderID`, `customer_key`, `id`: the name says it is a key."""
    found = words(name)
    return bool(found) and (found[-1] in KEY_SUFFIXES or found[0] in KEY_SUFFIXES)


def base(name: str, table: str) -> str:
    """What the column names, without its key suffix; a bare key takes the table's name."""
    found = words(name)
    while found and found[-1] in KEY_SUFFIXES:
        found.pop()
    while found and found[0] in KEY_SUFFIXES:
        found.pop(0)
    if not found:
        found = words(table)
    return " ".join(singular(word) for word in found)


def names_table(name: str, table: str) -> bool:
    """`customers.customerID`, `products.id`: the column names its own table (R-KEY-04)."""
    return base(name, table) == " ".join(singular(word) for word in words(table))


def similarity(child: str, child_table: str, parent: str, parent_table: str) -> float:
    """R-REL-02: 1 same thing, 0.8 one inside the other (whole words), 0.6 same start of at least
    4 letters (`shipVia` / `shipperID`), 0 otherwise."""
    a, b = base(child, child_table), base(parent, parent_table)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    a_words, b_words = a.split(), b.split()
    if _contains(a_words, b_words) or _contains(b_words, a_words):
        return 0.8
    joined_a, joined_b = a.replace(" ", ""), b.replace(" ", "")
    shared = 0
    for left, right in zip(joined_a, joined_b, strict=False):
        if left != right:
            break
        shared += 1
    return 0.6 if shared >= MIN_PREFIX else 0.0


def _contains(outer: list[str], inner: list[str]) -> bool:
    size = len(inner)
    return size < len(outer) and any(
        outer[start : start + size] == inner for start in range(len(outer) - size + 1)
    )


def prefix(name: str) -> str:
    """A table or column name as a column prefix: `Sample - Superstore` → `Sample_Superstore`."""
    cleaned = re.sub(r"\W+", "_", name, flags=re.UNICODE).strip("_")
    return cleaned or "table"
