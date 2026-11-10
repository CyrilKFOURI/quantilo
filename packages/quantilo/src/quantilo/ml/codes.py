# ruff: noqa: RUF002, RUF003  (−, ρ, · are the app's symbols)
"""What a column's name and values say about it, beyond its type (094): a code or an identifier
still typed as a number, a sequence number, a quality flag, a value one predicts.

The typing decides a column's type (docs/COLUMN_DETECTION.md). These rules keep the analyses that
read numbers (models, distances between rows, correlations, the structure) robust when a code is
still typed as a number: each analysis that uses them says so, with its rule and the way back.
They read whole words of the name, never a substring: "patient_nbr" and "bin__" are codes,
"number_of_persons_injured" and "num_lab_procedures" are counts, "cabin" is neither.
"""

import re
import unicodedata

import numpy as np

# A word that makes a name a code wherever it stands ("Code du département", "customerID").
CODE_WORDS = frozenset(
    {
        "id",
        "ids",
        "code",
        "codes",
        "key",
        "pid",
        "bbl",
        "bin",
        "zip",
        "zipcode",
        "postcode",
        "insee",
        "siren",
        "siret",
        "fips",
        "uuid",
        "guid",
        "sku",
        "ean",
        "isbn",
    }
)
# A word that makes a name a code only as its last word: "permit_si_no", "patient_nbr",
# "adresse_numero"; at the start it counts ("number_of_…", "num_lab_procedures").
LAST_CODE_WORDS = frozenset({"no", "nr", "nbr", "num", "number", "numero", "ref"})
FLAG_WORDS = frozenset(
    {"attributes", "attribute", "attr", "attrs", "flag", "flags", "qc", "mflag", "qflag", "sflag"}
)
# A number one predicts, said by its name (when no role says it).
VALUE_WORDS = frozenset(
    {
        "price",
        "prix",
        "value",
        "valeur",
        "amount",
        "montant",
        "cost",
        "charges",
        "salary",
        "salaire",
        "revenue",
        "sales",
        "income",
        "profit",
        "score",
        "target",
        "outcome",
    }
)
SEQUENCE_MIN_ROWS = 100
SEQUENCE_SHARE = 0.3  # distinct values per row, at least
SEQUENCE_DENSITY = 1.5  # max − min + 1 at most this many times the distinct values
SEQUENCE_RANK = 0.99  # Spearman ρ with the row order, at least


def words(name: str) -> list[str]:
    """The words of a name, lower case, accents removed: "customerID" → ["customer", "id"],
    "Code du département" → ["code", "du", "departement"], "bin__" → ["bin"]."""
    decomposed = unicodedata.normalize("NFKD", name)
    plain = "".join(c for c in decomposed if not unicodedata.combining(c))
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", plain)
    return [w for w in re.split(r"[^a-z0-9]+", spaced.casefold()) if w and not w.isdigit()]


def code_word(name: str) -> str | None:
    """The word that makes the name a code or an identifier, or None."""
    found = words(name)
    for word in found:
        if word in CODE_WORDS:
            return word
    if len(found) > 1 and found[-1] in LAST_CODE_WORDS:
        return found[-1]
    return None


def flag_word(name: str) -> str | None:
    """The word that makes the name a quality flag or an attribute of another column."""
    return next((w for w in words(name) if w in FLAG_WORDS), None)


def value_word(name: str) -> str | None:
    """The word that makes the name a value one predicts (price, valeur, charges…)."""
    return next((w for w in words(name) if w in VALUE_WORDS), None)


def whole(values: np.ndarray) -> bool:
    """Every present value is a whole number (at least one present)."""
    present = values[~np.isnan(values)]
    return bool(len(present)) and bool(np.all(np.mod(present, 1) == 0))


def numeric_code(name: str, values: np.ndarray) -> str | None:
    """R-CODE-01: a number whose name says code or identifier and whose values are whole
    numbers; returns the word, or None. Reason: its order and its distances mean nothing."""
    word = code_word(name)
    return word if word is not None and whole(values) else None


def sequence(values: np.ndarray, order: np.ndarray) -> float | None:
    """R-CODE-02: whole numbers, nearly one per row, over a dense range, rising with the row
    order: a row or sequence number (gameorder). Returns Spearman's ρ with the order, or None."""
    present = ~np.isnan(values)
    x, rows = values[present], order[present]
    if len(x) < SEQUENCE_MIN_ROWS or not whole(x):
        return None
    distinct = len(np.unique(x))
    if distinct < SEQUENCE_SHARE * len(x) or (x.max() - x.min() + 1) > SEQUENCE_DENSITY * distinct:
        return None
    from scipy import stats

    rho = float(np.corrcoef(stats.rankdata(x), stats.rankdata(rows))[0, 1])  # Spearman's ρ
    return rho if rho == rho and abs(rho) >= SEQUENCE_RANK else None
