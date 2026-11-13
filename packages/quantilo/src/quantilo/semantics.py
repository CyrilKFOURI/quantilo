"""What a column's name (and, when known, its values) says about how it may be used (093).

Shared by cleaning, KPI suggestions, the time-series defaults and the tail rule, so that all of them
agree that `patient_nbr` is a number that names a patient, not a quantity. Rules S-01 to S-06 of
docs/features/093-corrections-multidomaine-a/spec.md; every answer carries its reason.
"""

import re
from dataclasses import dataclass

from quantilo.typing.knowledge_base import load, name_tokens

# S-01: a code word anywhere in the name.
CODE_WORDS = frozenset(
    [
        *("id", "ids", "uuid", "guid", "code", "codes", "key", "ref", "zip", "zipcode"),
        *("postcode", "insee", "siren", "siret", "fips", "sku", "ean", "isbn", "pid"),
        "matricule",
        *("iban", "gtin", "upc"),  # N21: identifiers with a check digit (cleaning/rules/formats.py)
    ]
)
# S-01: a number word last ("patient_nbr"); first it is a count ("number_of_persons").
NUMBER_WORDS = frozenset(["no", "nr", "num", "nbr", "nb", "number", "numero", "nro"])
COORDINATE_WORDS = frozenset(["coordinate", "coordinates", "coord", "coords"])
COORDINATE_ROLES = frozenset(["latitude", "longitude"])
KEY_ROLES = frozenset(["identifier", "postal_code", "phone", "customer_key", "order_key"])
RATIO_WORDS = frozenset(["per"])  # S-04
# S-01: with one of these, "code" is an activity ("YearsCode": years of coding), not a code.
DURATION_WORDS = frozenset(["years", "year", "months", "days", "hours", "minutes", "time"])
FIXED_DIGITS = 5  # S-02: whole numbers all this long or longer
SEQUENCE_MIN = 100  # S-02 bis: at least this many distinct consecutive whole numbers
SEQUENCE_SHARE = 0.9  # S-02 bis: distinct values cover this share of the range
DATE_PARTS = frozenset(
    [
        *("day", "jour", "month", "mois", "year", "an", "annee", "week", "semaine", "quarter"),
        *("trimestre", "hour", "heure", "minute", "weekday", "dow", "hrmn", "yr", "mo"),
    ]
)
# S-05: the amount role's words that say money; "total" alone does not.
GENERIC_AMOUNT = frozenset(["total"])
# S-06: dates ranked, the event first, a characteristic (birth, construction) last.
EVENT_DATES = frozenset(
    [
        *("order", "ordered", "transaction", "created", "creation", "create", "invoice"),
        *("purchase", "purchased", "sale", "sold", "booking", "booked", "signup"),
        *("registration", "registered", "commande", "facture", "vente", "mutation"),
    ]
)
SHIPPING_DATES = frozenset(
    ["ship", "shipped", "shipping", "shipment", "delivery", "delivered", "dispatch"]
)
UPDATE_DATES = frozenset(
    ["updated", "update", "modified", "modification", "last", "changed", "edited"]
)
CHARACTERISTIC_DATES = frozenset(
    [
        *("birth", "birthday", "dob", "born", "naissance", "built", "blt", "construction"),
        *("constructed", "founded", "established", "build"),
    ]
)


@dataclass(frozen=True)
class Code:
    """A column that names or codes things: never summed, averaged or used as a measure."""

    kind: str  # "identifier" (names one thing: patient_nbr) or "code" (classifies: dep code)
    reason: str  # "its name ends in 'nbr'"


def tokens(name: str) -> tuple[str, ...]:
    return name_tokens(name)


def code_by_name(name: str, role: str | None = None, type: str | None = None) -> Code | None:
    """S-01: the name (or the detected type or role) says an identifier or a code."""
    if type == "identifier":
        return Code("identifier", "it is typed as an identifier")
    if role in KEY_ROLES:
        return Code("identifier", f"its role is {role.replace('_', ' ')}")
    words = tokens(name)
    if not words:
        return None
    if name.rstrip().endswith("#") or re.search(r"[A-Za-z0-9]__$", name):
        return Code("identifier", "its name ends in '#' (a number sign)")
    found = sorted(set(words) & CODE_WORDS)
    if found and set(found) <= {"code", "codes"} and set(words) & DURATION_WORDS:
        found = []
    if found:
        kind = "code" if found[0] in ("code", "codes") else "identifier"
        return Code(kind, f"its name says '{found[0]}'")
    if words[-1] in NUMBER_WORDS and len(words) > 1:
        return Code("identifier", f"its name ends in '{words[-1]}'")
    if len(words) == 1 and words[0] in NUMBER_WORDS - {"nb", "nbr"}:
        return Code("identifier", f"its name is '{words[0]}'")
    # 092: the detection's head-word rule (R-TYP-04), so that both read the name alike
    # ("council_district", "Mother's occupation", "MS SubClass": category codes)
    if (named := load().name_code(name)) is not None:
        kind, word = named
        if kind == "category":
            return Code("code", f"its name says '{word}' (a category)")
        kind = "code" if word in ("code", "codes") else "identifier"
        return Code(kind, f"its name says '{word}'")
    return None


def code_by_values(low: float | None, high: float | None, integer: bool) -> Code | None:
    """S-02: whole positive numbers all with the same number of digits (5 or more)."""
    if not integer or low is None or high is None or low < 0:
        return None
    digits = len(str(int(low)))
    if digits >= FIXED_DIGITS and len(str(int(high))) == digits:
        return Code("code", f"its values are whole numbers of {digits} digits each")
    return None


def sequence(low: float | None, high: float | None, integer: bool, distinct: int) -> Code | None:
    """S-02 bis: consecutive whole numbers (1, 2, 3 … 62,541): a row or sequence number."""
    if not integer or low is None or high is None or distinct < SEQUENCE_MIN:
        return None
    if distinct >= SEQUENCE_SHARE * (high - low + 1):
        return Code("identifier", "its values are consecutive whole numbers (a sequence number)")
    return None


def is_date_part(name: str) -> bool:
    """A number named after a part of a date ('jour', 'mois', 'an', 'hour'): no measure."""
    words = set(tokens(name))
    return bool(words) and words <= DATE_PARTS


def is_coordinate(name: str, role: str | None = None) -> bool:
    """S-03: a latitude, a longitude or a map coordinate."""
    if role in COORDINATE_ROLES:
        return True
    words = set(tokens(name))
    if words & COORDINATE_WORDS:
        return True
    match = load().match_role(name)
    return match is not None and match.id in COORDINATE_ROLES


def is_ratio(name: str) -> bool:
    """S-04: an intensity ('co2_per_capita', 'cost per unit'), never summed."""
    return bool(set(tokens(name)) & RATIO_WORDS)


def money_words(name: str) -> list[str]:
    """S-05: the words of the amount role in the name, 'total' left out."""
    amount = next((r for r in load().roles if r.id == "amount"), None)
    if amount is None:
        return []
    words = tokens(name)
    found = []
    for phrase in amount.names:
        if set(phrase) <= GENERIC_AMOUNT:
            continue
        size = len(phrase)
        if any(words[i : i + size] == phrase for i in range(len(words) - size + 1)):
            found.append(" ".join(phrase))
    return found


def is_money(name: str) -> bool:
    """S-05: an amount role that really says money: a money word, or 'total' alone."""
    words = tokens(name)
    if money_words(name):
        return True
    return bool(words) and set(words) <= GENERIC_AMOUNT


def date_tier(name: str) -> int:
    """S-06: 0 event, 1 other, 2 shipping, 3 update, 4 characteristic (birth, construction)."""
    words = set(tokens(name))
    if words & CHARACTERISTIC_DATES:
        return 4
    if words & EVENT_DATES:
        return 0
    if words & SHIPPING_DATES:
        return 2
    if words & UPDATE_DATES:
        return 3
    return 1
