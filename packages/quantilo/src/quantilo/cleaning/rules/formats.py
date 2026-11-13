"""Advanced rules on business formats (audit 2026-11-12, § 4.3, rules F-xx):

- F-02 (N21): identifiers with a check digit. A column named or shaped like SIREN or SIRET numbers
  (Luhn, La Poste's exception), French VAT numbers (key `(12 + 3 × (SIREN mod 97)) mod 97`), IBANs
  (length of the country, mod 97), EAN / UPC / GTIN barcodes (mod 10) or ISBNs (mod 11, or mod 10
  for ISBN-13). The values whose check digit fails are flagged in a new column (filter or pin the
  rows on it), never corrected: a wrong check digit says there is an error, not which one (audit
  § 5). IBANs written in groups or in small letters are also put in their electronic form
  (spaces removed, capitals), which loses nothing.
- F-01 (N20): phone numbers written several ways, put in the international form E.164
  (+33612345678) when their country is known: written with a code, the row's country column, or
  the one country of the column's numbers that carry a code. A coded table of the plans of the
  main French, European, North American and British numbers (calling code, national length,
  trunk prefix); a number without a known country, or of another country, stays as it is, and a
  length no number of its country has is listed.
- F-04 (N24): countries written several ways in one column (France, FR, FRA, france; USA among
  names), read through a coded table (`typing/knowledge_base/countries.toml`: ISO 3166 codes,
  English and French names, usual variants) and written the way the column mostly does. Values
  the table does not know (World, England, West Germany) are never merged into a country.

Every reading is a coded rule whose reason is shown (no model guesses a kind); a column whose
name says nothing must prove its kind by its values (90 % pass a check that a random number passes
once in ten).
"""

from __future__ import annotations

import re
import tomllib
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cache
from typing import TYPE_CHECKING

from quantilo import semantics
from quantilo.cleaning import model as m
from quantilo.cleaning.compile import INTEGERS, as_text, is_text, literal_list
from quantilo.cleaning.rules import basic
from quantilo.cleaning.rules.basic import Column, TextStages, n, q
from quantilo.ingestion.sql import ident
from quantilo.typing import knowledge_base

if TYPE_CHECKING:
    from quantilo.cleaning.rules.advanced import Advanced

NAMED_SHAPE = 0.8  # F-02: a column named after a kind has its shape on this share of the rows
VALUE_SHAPE = 0.9  # … a column whose name says nothing, on this share
VALUE_PASS = 0.9  # … and its check passes on this share of them (a random number: 10 %)
VALUE_DISTINCT = 30  # … over this many distinct values at least
MOSTLY_FAILING = 0.5  # more failures than this: not this kind of number, nothing is flagged
LOST_ZEROS = 2  # leading zeros a column of whole numbers may have lost
MAX_READ = basic.MAX_DISTINCT  # distinct values read in a column; more: a sample, information
SHOWN = 5
LOOK = 200  # distinct values a column is first looked at
LOOK_SHARE = 0.5  # … half of them with a kind's shape to read it all
LOOK_PASS = 0.7  # … and this share of those passing its check
FLAG = "check digit fails"
LA_POSTE = "356000000"  # La Poste's SIREN breaks Luhn's rule; its SIRETs may sum to 5 × k
SEPARATORS = re.compile(r"[\s.\-_/]+")
NULLS = basic.SURE_TOKENS | basic.WEAK_TOKENS

# IBAN length by country (SWIFT registry, ISO 13616), the two letters first.
IBAN_LENGTHS = {
    "AD": 24, "AE": 23, "AL": 28, "AT": 20, "AZ": 28, "BA": 20, "BE": 16, "BG": 22, "BH": 22,
    "BR": 29, "BY": 28, "CH": 21, "CR": 22, "CY": 28, "CZ": 24, "DE": 22, "DK": 18, "DO": 28,
    "EE": 20, "EG": 29, "ES": 24, "FI": 18, "FO": 18, "FR": 27, "GB": 22, "GE": 22, "GI": 23,
    "GL": 18, "GR": 27, "GT": 28, "HR": 21, "HU": 28, "IE": 22, "IL": 23, "IQ": 23, "IS": 26,
    "IT": 27, "JO": 30, "KW": 30, "KZ": 20, "LB": 28, "LC": 32, "LI": 21, "LT": 20, "LU": 20,
    "LV": 21, "MC": 27, "MD": 24, "ME": 22, "MK": 19, "MR": 27, "MT": 31, "MU": 30, "NL": 18,
    "NO": 15, "PK": 24, "PL": 28, "PS": 29, "PT": 25, "QA": 29, "RO": 24, "RS": 22, "SA": 24,
    "SC": 31, "SE": 24, "SI": 19, "SK": 24, "SM": 27, "ST": 25, "SV": 28, "TL": 23, "TN": 24,
    "TR": 26, "UA": 29, "VA": 22, "VG": 24, "XK": 20,
}  # fmt: skip


# --- Check digits --------------------------------------------------------------------------------


def luhn(digits: str) -> bool:
    """Luhn's rule (SIREN, SIRET): every second digit from the right doubled, the sum ends in 0."""
    if not digits.isdigit():
        return False
    total = 0
    for index, char in enumerate(reversed(digits)):
        digit = ord(char) - 48
        if index % 2:
            digit = digit * 2 - 9 if digit > 4 else digit * 2
        total += digit
    return total % 10 == 0


def siren(value: str) -> bool:
    return len(value) == 9 and (luhn(value) or value == LA_POSTE)


def siret(value: str) -> bool:
    if len(value) != 14 or not value.isdigit():
        return False
    if luhn(value):
        return True
    # La Poste's establishments outnumber Luhn's keys: their digits sum to a multiple of 5 instead
    return value.startswith(LA_POSTE) and sum(ord(c) - 48 for c in value) % 5 == 0


def vat_fr(value: str) -> bool:
    """FR + a 2-digit key + the SIREN: key = (12 + 3 × (SIREN mod 97)) mod 97."""
    key, number = value[2:4], value[4:]
    if not (value.startswith("FR") and key.isdigit() and len(number) == 9 and number.isdigit()):
        return False
    return int(key) == (12 + 3 * (int(number) % 97)) % 97


def iban(value: str) -> bool:
    """The length of its country, then the first 4 characters moved to the end, letters as
    numbers (A = 10 … Z = 35): the number mod 97 is 1 (ISO 13616)."""
    if IBAN_LENGTHS.get(value[:2]) != len(value) or not value[2:4].isdigit():
        return False
    moved = value[4:] + value[:4]
    if not moved.isalnum() or not moved.isascii():
        return False
    return int("".join(str(int(c, 36)) for c in moved)) % 97 == 1


def gtin(value: str) -> bool:
    """EAN-8, UPC-A (12), EAN-13, GTIN-14: digits weighted 3 and 1 from the right of the key."""
    if len(value) not in (8, 12, 13, 14) or not value.isdigit():
        return False
    total = sum(
        (ord(c) - 48) * (3 if i % 2 == 0 else 1) for i, c in enumerate(reversed(value[:-1]))
    )
    return (10 - total % 10) % 10 == ord(value[-1]) - 48


def upc_e(value: str) -> bool:
    """A UPC-E (8 digits, number system 0 or 1): its key is the one of the UPC-A it shortens."""
    if len(value) != 8 or value[0] not in "01" or not value.isdigit():
        return False
    a, b, c, d, e, last = value[1:7]
    if last in "012":
        body = a + b + last + "0000" + c + d + e
    elif last == "3":
        body = a + b + c + "00000" + d + e
    elif last == "4":
        body = a + b + c + d + "00000" + e
    else:
        body = a + b + c + d + e + "0000" + last
    return gtin(value[0] + body + value[7])


def barcode(value: str) -> bool:
    """EAN / UPC / GTIN, or a UPC-E for 8 digits starting with 0 or 1."""
    return gtin(value) or upc_e(value)


def isbn(value: str) -> bool:
    """ISBN-10 (weights 10 to 1, mod 11, X = 10) or ISBN-13 (an EAN-13 starting 978 or 979)."""
    if len(value) == 13:
        return value[:3] in ("978", "979") and gtin(value)
    if len(value) != 10 or not value[:9].isdigit() or not (value[9].isdigit() or value[9] == "X"):
        return False
    digits = [ord(c) - 48 for c in value[:9]] + [10 if value[9] == "X" else ord(value[9]) - 48]
    return sum((10 - i) * d for i, d in enumerate(digits)) % 11 == 0


@dataclass(frozen=True)
class Kind:
    id: str
    label: str  # "SIREN", in a sentence
    plural: str  # "SIREN numbers"
    words: frozenset[str]  # one of them in the name says the kind
    shape: re.Pattern[str]  # the compact value (separators out, capitals)
    check: Callable[[str], bool]
    how: str  # the check, in the reason
    by_values: bool = True  # can be read from the values alone
    width: int | None = None  # digits, when an integer column lost its leading zeros
    phrases: tuple[tuple[str, ...], ...] = ()


KINDS = (
    Kind(
        "siren", "SIREN", "SIREN numbers", frozenset({"siren"}), re.compile(r"\d{9}"), siren,
        "Luhn key", width=9,
    ),
    Kind(
        "siret", "SIRET", "SIRET numbers", frozenset({"siret"}), re.compile(r"\d{14}"), siret,
        "Luhn key", width=14,
    ),
    Kind(
        "vat_fr", "French VAT number", "French VAT numbers",
        frozenset({"tva", "vat", "tvaintra", "tvaintracom"}), re.compile(r"FR[0-9A-Z]{2}\d{9}"),
        vat_fr, "key (12 + 3 × (SIREN mod 97)) mod 97",
    ),
    Kind(
        "iban", "IBAN", "IBANs", frozenset({"iban"}), re.compile(r"[A-Z]{2}\d{2}[A-Z0-9]{11,30}"),
        iban, "country length and mod 97",
    ),
    Kind(
        "isbn", "ISBN", "ISBNs", frozenset({"isbn", "isbn10", "isbn13"}),
        re.compile(r"\d{9}[\dX]|97[89]\d{10}"), isbn, "ISBN key", by_values=False,
    ),
    Kind(
        "gtin", "EAN / UPC barcode", "EAN / UPC barcodes",
        frozenset({"ean", "ean8", "ean13", "gtin", "gtin8", "gtin12", "gtin13", "gtin14", "upc",
                   "barcode", "barcodes"}),
        re.compile(r"\d{8}|\d{12,14}"), barcode, "mod 10 key", width=14,
        phrases=(("code", "barre"), ("codes", "barres"), ("bar", "code")),
    ),
)  # fmt: skip
# A VAT word alone is often an amount ('TVA', 'vat_amount'): the values decide (`by_values`).
AMOUNT_WORDS = {"amount", "montant", "rate", "taux", "total", "ht", "ttc", "price", "prix"}


def compact(value: str) -> str:
    """'FR76 3000 6000 0112 3456 7890 189' → 'FR7630006000011234567890189'."""
    return SEPARATORS.sub("", value.strip()).upper()


def named_kind(column: Column) -> Kind | None:
    words = semantics.tokens(column.name)
    found = set(words)
    for kind in KINDS:
        if kind.id == "vat_fr" and found & AMOUNT_WORDS:
            continue
        if found & kind.words or any(
            words[i : i + len(p)] == p for p in kind.phrases for i in range(len(words))
        ):
            return kind
    return None


# --- Reading a column ----------------------------------------------------------------------------


@dataclass
class Reading:
    """A column read as one kind: its values (as stored, as text) by their compact form."""

    kind: Kind
    named: bool
    total: int  # rows with a value
    shaped: int = 0  # rows whose value has the kind's shape
    passing: int = 0
    distinct: int = 0  # distinct values with the shape
    passing_distinct: int = 0  # … of them that pass the check
    failing: dict[str, int] = field(default_factory=dict)  # compact → rows
    spellings: dict[str, set[str]] = field(default_factory=dict)  # compact → stored values

    @property
    def fails(self) -> int:
        return sum(self.failing.values())


def _values(advanced: Advanced, column: Column) -> tuple[list[tuple[str, int]], bool]:
    """The column's distinct values as text with their rows, and whether it is a sample."""
    stages = advanced.context.stages.get(column.name)
    if stages is not None:
        return stages.raw, False
    text = as_text(column.name, column.dtype)
    found = advanced.context.connection.execute(
        f"SELECT {text}, count(*) FROM {advanced.context.source} WHERE {ident(column.name)} "
        f"IS NOT NULL GROUP BY 1 ORDER BY hash({text}) LIMIT {MAX_READ + 1}"
    ).fetchall()
    return [(str(v), int(c)) for v, c in found[:MAX_READ]], len(found) > MAX_READ


def _likely(advanced: Advanced, column: Column, integer: bool) -> bool:
    """Half of a few distinct values have the shape of a kind read by its values, and most of
    those pass its check (a random number: one in ten)."""
    stages = advanced.context.stages.get(column.name)
    if stages is not None:
        sample = [v for v, _ in stages.raw[:LOOK]]
    else:
        text = as_text(column.name, column.dtype)
        sample = [
            str(row[0])
            for row in advanced.context.connection.execute(
                f"SELECT DISTINCT {text} FROM {advanced.context.source} "
                f"WHERE {ident(column.name)} IS NOT NULL LIMIT {LOOK}"
            ).fetchall()
        ]
    for kind in KINDS:
        if not kind.by_values:
            continue
        forms = [f for f in (_forms(kind, v, integer) for v in sample) if kind.shape.fullmatch(f)]
        passing = sum(1 for f in forms if kind.check(f))
        if len(forms) >= LOOK_SHARE * len(sample) and passing >= LOOK_PASS * len(forms):
            return True
    return False


def _forms(kind: Kind, value: str, integer: bool) -> str:
    """The compact value; in a column of whole numbers, with the leading zeros it lost (two at
    most: a smaller number is no such code), which changes neither key."""
    form = compact(value)
    if not (integer and kind.width is not None and form.isdigit()):
        return form
    if kind.id == "gtin":
        width = next((w for w in (8, 13) if w - LOST_ZEROS <= len(form) <= w), len(form))
    else:
        width = kind.width if len(form) >= kind.width - LOST_ZEROS else len(form)
    return form.zfill(width)


def _read(kind: Kind, named: bool, values: list[tuple[str, int]], integer: bool) -> Reading:
    reading = Reading(kind, named, sum(c for _, c in values))
    seen: set[str] = set()
    for value, count in values:
        if not value.strip() or value.strip().lower() in NULLS:
            reading.total -= count
            continue
        form = _forms(kind, value, integer)
        if not kind.shape.fullmatch(form) or len(set(form[-9:])) == 1:
            continue  # '000000000', '999999999': a placeholder (B-09's), not a number to check
        reading.shaped += count
        new = form not in seen
        seen.add(form)
        reading.distinct += new
        reading.spellings.setdefault(form, set()).add(value)
        if kind.check(form):
            reading.passing += count
            reading.passing_distinct += new
        else:
            reading.failing[form] = reading.failing.get(form, 0) + count
    return reading


def read_column(advanced: Advanced, column: Column) -> tuple[Reading, bool] | None:
    """The kind of identifier a column holds (its name, or its values proving it), or None."""
    integer = column.dtype in INTEGERS
    if not (is_text(column.dtype) or integer) or not column.nonnull:
        return None
    if basic._person_names(column):
        return None
    kind = named_kind(column)
    if kind is None and column.distinct < VALUE_DISTINCT:
        return None
    if kind is None and not _likely(advanced, column, integer):
        return None  # a first look at a few values: no kind's shape, the column is not read
    values, sampled = _values(advanced, column)
    if kind is not None:
        reading = _read(kind, True, values, integer)
        if reading.total and reading.shaped >= NAMED_SHAPE * reading.total:
            return reading, sampled
        if kind.id not in ("vat_fr", "gtin"):
            return None  # named but not shaped so: another kind of value
    best: Reading | None = None
    for candidate in KINDS:
        if not candidate.by_values:
            continue
        reading = _read(candidate, False, values, integer)
        if (
            reading.total
            and reading.shaped >= VALUE_SHAPE * reading.total
            and reading.distinct >= VALUE_DISTINCT
            # counted on distinct values: a frequent one (a code repeated) proves nothing
            and reading.passing_distinct >= VALUE_PASS * reading.distinct
            and (best is None or reading.passing > best.passing)
        ):
            best = reading
    return (best, sampled) if best is not None else None


# --- F-02 ----------------------------------------------------------------------------------------


def _rows(advanced: Advanced, column: Column, stored: list[str]) -> list[tuple[int, str]]:
    """The first rows holding one of these values, with the value."""
    if not stored:
        return []
    text = as_text(column.name, column.dtype)
    found = advanced.context.connection.execute(
        f"SELECT file_row_number, {text} FROM {advanced.rows_source} WHERE {text} IN "
        f"{literal_list(stored[: m.MAX_VALUES])} ORDER BY 1 LIMIT {SHOWN}"
    ).fetchall()
    return [(int(row), str(value)) for row, value in found]


def _finals(advanced: Advanced, column: Column) -> dict[str, str]:
    """Each stored value → the value after the Basic text fixes (spaces, invisible characters)."""
    stages = advanced.context.stages.get(column.name)
    if stages is None:
        return {}
    from quantilo.cleaning.rules.advanced import _final_of

    return {raw: final for final, raws in _final_of(stages).items() for raw in raws}


def _flag_values(reading: Reading, advanced: Advanced, column: Column) -> list[str]:
    """Every writing of a failing value a step before may leave: as stored, after the Basic text
    fixes, and in its compact form (C-23), so the flag finds them whatever was ticked."""
    finals = _finals(advanced, column)
    found: set[str] = set()
    for form in reading.failing:
        for stored in reading.spellings.get(form, set()):
            found |= {stored, finals.get(stored, stored)}
        found.add(form)
    return sorted(found)


def check_digits(advanced: Advanced) -> None:
    from quantilo.cleaning.rules.advanced import row_label, rows_list

    for column in advanced.context.columns:
        read = read_column(advanced, column)
        if read is None:
            continue
        reading, sampled = read
        kind = reading.kind
        if kind.id == "iban":
            _iban_form(advanced, column, reading)
        if not reading.failing:
            continue
        fails = reading.fails
        how = (
            f"its name says {kind.plural}"
            if reading.named
            else f"{reading.passing_distinct / reading.distinct:.1%} of its distinct values pass "
            f"the {kind.label} check ({kind.how}), which a random number passes once in ten"
        )
        on = f" (on a sample of {n(MAX_READ)} distinct values)" if sampled else ""
        stored = sorted({s for f in reading.failing for s in reading.spellings.get(f, set())})
        rows = _rows(advanced, column, stored)
        shown = ", ".join(q(f) for f in list(reading.failing)[:SHOWN])
        if fails > MOSTLY_FAILING * reading.shaped:
            advanced.add(
                "F-02",
                "I",
                [column],
                f"Most {kind.plural} of {column.name} fail their check digit",
                f"{n(fails)} of {n(reading.shaped)} values of {column.name}{on} fail the "
                f"{kind.label} check ({kind.how}): more than half, so they are probably not "
                f"real {kind.plural} (test or internal numbers, or another kind of code), and "
                "nothing is flagged. Information only. Check your data.",
                cells=fails,
                examples=[(f, FLAG) for f in list(reading.failing)[:SHOWN]],
                key="check",
            )
            continue
        listed = f" ({rows_list([r for r, _ in rows], fails)})" if rows else ""
        noun = kind.plural if fails > 1 else kind.label
        verb = "fail their" if fails > 1 else "fails its"
        reason = (
            f"{n(fails)} {noun} of {column.name}{on} {verb} check digit{listed}: mistyped? "
            f"{shown}. The column holds {kind.plural} ({how}). A wrong check digit says there is "
            "an error, not which digit: nothing is corrected. "
        )
        values = _flag_values(reading, advanced, column)
        examples = [(f"{row_label(r)}: {value}", FLAG) for r, value in rows]
        if sampled or len(values) > m.MAX_VALUES:
            advanced.add(
                "F-02",
                "I",
                [column],
                f"{n(fails)} {noun} of {column.name} {verb} check digit",
                reason
                + (
                    "Read on a sample of the values, so not flagged: "
                    if sampled
                    else "Too many different values to flag in one step: "
                )
                + "filter the column in Data to see them. Information only. Check your data.",
                cells=fails,
                examples=examples,
                key="check",
            )
            continue
        flag = f"{column.name}_check"
        taken = {c.name for c in advanced.context.columns}
        while flag in taken:
            flag += "_"
        advanced.add(
            "F-02",
            "S",
            [column],
            f"Flag {n(fails)} {noun} of {column.name} whose check digit fails",
            reason + f"Ticked, a column {flag} says {q(FLAG)} on their rows: filter or pin them "
            "there. Check your data.",
            m.ConditionalColumn(
                id="s0",
                to=flag,
                cases=[
                    m.Case(
                        conditions=[m.RowCondition(column=column.name, op="in", values=values)],
                        value=FLAG,
                    )
                ],
                otherwise=None,
            ),
            cells=fails,
            examples=examples,
            key="check",
            impact="columns",
        )


def _iban_form(advanced: Advanced, column: Column, reading: Reading) -> None:
    """IBANs in groups of 4 or in small letters → the electronic form (nothing lost)."""
    mapping: dict[str, str] = {}
    finals = _finals(advanced, column)
    for form, stored in reading.spellings.items():
        for value in stored:
            for writing in (value, finals.get(value, value)):
                if writing != form:
                    mapping[writing] = form
    stages = advanced.context.stages.get(column.name)
    if not mapping or stages is None:
        return  # more than 50,000 distinct IBANs: one step could not list them (C-05)
    counts = dict(stages.raw)
    cells = sum(c for v, c in counts.items() if v in mapping)
    grouped = sum(c for v, c in counts.items() if v in mapping and " " in v.strip())
    before, after = next(iter(mapping.items()))
    advanced.add(
        "F-02",
        "S",
        [column],
        f"Write {n(cells)} IBANs of {column.name} without spaces",
        f"{n(cells)} IBANs of {column.name} are written in groups or with small letters "
        f"({q(before)} → {q(after)}"
        + (f", {n(grouped)} with spaces" if grouped else "")
        + "): the electronic form, one block of capitals, is how banks and files exchange them; "
        "the printed groups are only for reading. Nothing else changes. Check your data.",
        m.MapValues(id="s0", column=column.name, map=basic._capped(mapping, counts)),
        cells=cells,
        examples=list(mapping.items())[:SHOWN],
        key="format",
        impact="format",
    )


# --- Countries (F-04, N24) -----------------------------------------------------------------------

COUNTRY_FILE = knowledge_base.DIRECTORY / "countries.toml"
COUNTRY_WORDS = {"country", "countries", "pays", "nation", "nations"}
COUNTRY_SHARE = 0.8  # F-04: the column's values found in the table, on this share of the rows
COUNTRY_ROWS = 20
STYLES = ("en", "fr", "a2", "a3")  # the writings a column can be put in, preferred in this order
STYLE_LABELS = {
    "en": "English names",
    "fr": "French names",
    "a2": "two-letter ISO codes",
    "a3": "three-letter ISO codes",
}
CODE_STYLES = frozenset({"a2", "a3", "abbr"})
ABBREVIATION = re.compile(r"[A-Z]{2,5}")


@dataclass(frozen=True)
class Country:
    a2: str
    a3: str
    en: str
    fr: str

    def written(self, style: str) -> str:
        return {"en": self.en, "fr": self.fr, "a2": self.a2, "a3": self.a3}[style]


@dataclass(frozen=True)
class CountryTable:
    countries: dict[str, Country]  # alpha-2 → its writings
    codes: dict[str, tuple[str, str]]  # 'FR', 'FRA' → (alpha-2, 'a2' or 'a3')
    names: dict[str, tuple[str, frozenset[str]]]  # folded name → (alpha-2, its styles)
    abbreviations: dict[str, str]  # 'UK', 'UAE', 'EIRE' → alpha-2
    ambiguous: frozenset[str]  # folded writings two countries share: left out


def fold_country(value: str) -> str:
    """'  United-States ' → 'united states', 'Côte d'Ivoire' → 'cote divoire', 'The Gambia' →
    'gambia', 'St Lucia' → 'saint lucia'."""
    text = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode().lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[.'`()]", "", text)
    text = " ".join(re.sub(r"[-_/,]+", " ", text).split())
    if text.startswith("the "):
        text = text[4:]
    if text.startswith("st "):
        text = "saint " + text[3:]
    return text


@cache
def country_table() -> CountryTable:
    raw = tomllib.loads(COUNTRY_FILE.read_text(encoding="utf-8"))["countries"]
    countries: dict[str, Country] = {}
    codes: dict[str, tuple[str, str]] = {}
    names: dict[str, dict[str, set[str]]] = {}
    abbreviations: dict[str, str] = {}
    for a2, (a3, en, fr, *others) in raw.items():
        countries[a2] = Country(a2, a3, en, fr)
        codes[a2] = (a2, "a2")
        codes[a3] = (a2, "a3")
        for style, writing in (("en", en), ("fr", fr), *(("other", o) for o in others)):
            letters = writing.replace(".", "")
            if style == "other" and ABBREVIATION.fullmatch(letters):
                abbreviations[letters] = a2
                continue
            names.setdefault(fold_country(writing), {}).setdefault(a2, set()).add(style)
    ambiguous = frozenset(key for key, found in names.items() if len(found) > 1)
    named = {
        key: (a2, frozenset(styles))
        for key, found in names.items()
        if key not in ambiguous
        for a2, styles in found.items()
    }
    return CountryTable(countries, codes, named, abbreviations, ambiguous)


def read_country(value: str) -> tuple[str, frozenset[str]] | None:
    """The country a value names (its alpha-2) and how it is written: {'a2'}, {'a3'}, {'abbr'}
    (UK, UAE), or the names it is ({'en', 'fr'} for 'France'); None: no country of the table."""
    text = value.strip()
    if not text or text.lower() in NULLS:
        return None
    table = country_table()
    letters = text.replace(".", "").replace(" ", "")
    if letters.isascii() and letters.isalpha() and len(letters) in (2, 3):
        code = table.codes.get(letters.upper())
        if code is not None:
            return code[0], frozenset({code[1]})
    if letters.isascii() and letters.isupper() and letters in table.abbreviations:
        return table.abbreviations[letters], frozenset({"abbr"})
    return table.names.get(fold_country(text))


def _writing_key(value: str, styles: frozenset[str]) -> str:
    """Two writings that differ only by case, accents or spacing are one (B-03's): 'France' and
    'france', 'FR' and 'fr'."""
    if styles & CODE_STYLES:
        return value.replace(".", "").replace(" ", "").upper()
    return fold_country(value)


def is_country_column(column: Column) -> bool:
    role = column.view.role if column.view is not None else None
    if role == "country" or (column.role is not None and column.role.id == "country"):
        return True
    return bool(set(semantics.tokens(column.name)) & COUNTRY_WORDS)


@dataclass
class CountryReading:
    """A country column read through the table."""

    total: int  # rows with a value
    read: dict[str, tuple[str, frozenset[str]]]  # value → (alpha-2, styles)
    rows: dict[str, int]
    unread: list[tuple[str, int]]  # the values the table does not know, the most frequent first

    @property
    def share(self) -> float:
        return sum(self.rows[v] for v in self.read) / self.total if self.total else 0.0


def read_countries(values: list[tuple[str, int]]) -> CountryReading:
    real = [(v, c) for v, c in values if v.strip() and v.strip().lower() not in NULLS]
    rows: dict[str, int] = {}
    for value, count in real:
        rows[value] = rows.get(value, 0) + count
    read: dict[str, tuple[str, frozenset[str]]] = {}
    unread: list[tuple[str, int]] = []
    for value, count in rows.items():
        found = read_country(value)
        if found is None:
            unread.append((value, count))
        else:
            read[value] = found
    unread.sort(key=lambda item: -item[1])
    return CountryReading(sum(rows.values()), read, rows, unread)


def countries_of(n_countries: int) -> str:
    return f"{n(n_countries)} {'country' if n_countries == 1 else 'countries'}"


def countries(advanced: Advanced) -> None:
    """F-04: a country column that writes one country several ways (France, FR, FRA), or mixes
    codes and names, written the way the column mostly does (its distinct writings vote)."""
    from quantilo.cleaning.rules.shapes import _map_suggestion

    table = country_table()
    for column, stages in advanced.context.texts():
        if not is_country_column(column) or basic._converted(column) or not stages.final:
            continue
        reading = read_countries(stages.final)
        if reading.total < COUNTRY_ROWS or reading.share < COUNTRY_SHARE:
            continue
        groups: dict[str, dict[str, list[str]]] = {}
        for value, (a2, styles) in reading.read.items():
            groups.setdefault(a2, {}).setdefault(_writing_key(value, styles), []).append(value)
        votes: dict[str, int] = dict.fromkeys(STYLES, 0)
        for writings in groups.values():
            for spellings in writings.values():
                for style in reading.read[spellings[0]][1] & set(STYLES):
                    votes[style] += 1
        style = max(STYLES, key=lambda s: (votes[s], -STYLES.index(s)))
        if not votes[style]:
            continue
        coded = style in CODE_STYLES
        mapping: dict[str, str] = {}
        merged: list[str] = []  # countries written several ways
        recoded: list[str] = []  # … written once, a code among names (or a name among codes)
        ways = 0
        for a2, writings in groups.items():
            values = [v for spellings in writings.values() for v in spellings]
            kinds = {bool(reading.read[v][1] & CODE_STYLES) for v in values}
            if len(writings) < 2 and kinds == {coded}:
                continue  # one writing of this country, of the column's kind: left as it is
            form = table.countries[a2].written(style)
            fitting = [v for v in values if style in reading.read[v][1]]
            target = (
                form
                if form in fitting or not fitting
                else max(fitting, key=reading.rows.__getitem__)
            )
            moved = {v: target for v in values if v != target}
            if not moved:
                continue
            mapping |= moved
            if len(writings) > 1:
                merged.append(a2)
                ways += len(writings)
            else:
                recoded.append(a2)
        if not mapping:
            continue
        distinct = sum(len(w) for w in groups.values())
        changed = len(merged) + len(recoded)
        parts = []
        if merged:
            shown = [f"{q(v)} → {q(t)}" for v, t in mapping.items() if reading.read[v][0] in merged]
            parts.append(
                f"{ways} ways to write {countries_of(len(merged))} ({', '.join(shown[:SHOWN])})"
            )
        if recoded:
            shown = [
                f"{q(v)} → {q(t)}" for v, t in mapping.items() if reading.read[v][0] in recoded
            ]
            kind = "names among codes" if coded else "codes among names"
            parts.append(
                f"{countries_of(len(recoded))} written as {kind} ({', '.join(shown[:SHOWN])})"
            )
        reason = (
            f"In {column.name}, {'; '.join(parts)}: written as the column mostly does, in "
            f"{STYLE_LABELS[style]} ({votes[style]} of its {distinct} distinct writings). "
            f"{reading.share:.1%} of its rows are countries of a coded table (ISO 3166 codes, "
            "English and French names, usual variants such as UK or USA). "
        )
        lists = [v for v, _ in reading.unread if "," in v or ";" in v]
        others = [v for v, _ in reading.unread if v not in lists]
        if others:
            reason += (
                f"{len(others)} other values are no country of the table and stay as they are "
                f"({', '.join(q(v) for v in others[:SHOWN])}): a region, a group, a former "
                "country or a typo is never merged into a country. "
            )
        if lists:
            reason += f"{len(lists)} values list several countries and stay as they are. "
        _map_suggestion(
            advanced,
            "F-04",
            column,
            stages,
            mapping,
            f"Write {countries_of(changed)} of {column.name} like the others",
            reason + "Check your data.",
            "countries",
        )


# --- Phone numbers (F-01, N20) -------------------------------------------------------------------

PHONE_CHARS = re.compile(r"^[\d\s().\-/+]+$")
PHONE_ROWS = 20
AGREE = 0.9  # F-01: a country column is the numbers' own when it agrees with their country codes
ONE_CODE = 0.9  # … without one, the numbers with a code share it on this share of their rows


@dataclass(frozen=True)
class Plan:
    country: str  # alpha-2
    code: str  # the calling code, '33'
    lengths: frozenset[int]  # digits of the national significant number
    trunk: str  # '0' in France, '' in Spain, '1' in North America
    first: str = ""  # the digits the number may start with ('' : any but the trunk prefix)

    def significant(self, digits: str) -> str | None:
        """The national significant number of digits dialled inside the country, or None when no
        number of this plan has that length ('0612345678' → '612345678' in France)."""
        if self.trunk and digits.startswith(self.trunk):
            rest = digits[len(self.trunk) :]
            if len(rest) in self.lengths and self.fits(rest):
                return rest
        if len(digits) in self.lengths and self.fits(digits):
            return digits
        return None

    def fits(self, number: str) -> bool:
        if self.first and number[:1] not in self.first:
            return False  # a North American area code never starts with 0 or 1
        return not (self.trunk and number.startswith(self.trunk))


@cache
def phone_plans() -> dict[str, Plan]:
    raw = tomllib.loads(COUNTRY_FILE.read_text(encoding="utf-8"))["phones"]
    return {
        a2: Plan(
            a2,
            str(p["code"]),
            frozenset(int(x) for x in p["lengths"]),
            str(p["trunk"]),
            str(p.get("first", "")),
        )
        for a2, p in raw.items()
    }


@cache
def plans_by_code() -> dict[str, Plan]:
    """One plan per calling code (North America's '1' is the United States' and Canada's)."""
    found: dict[str, Plan] = {}
    for plan in phone_plans().values():
        found.setdefault(plan.code, plan)
    return found


@dataclass(frozen=True)
class Dialled:
    """A phone number as read: its plan when known, its national significant number when its
    length fits that plan."""

    international: bool  # written with a country code (+33, 0033)
    plan: Plan | None
    number: str | None
    code: str | None = None  # the code as written, known or not

    @property
    def e164(self) -> str | None:
        if self.plan is None or self.number is None:
            return None
        return f"+{self.plan.code}{self.number}"


def dial(value: str, plan: Plan | None = None) -> Dialled:
    """'+33 (0)6 12 34 56 78', '0033 6…', '06.12.34.56.78' (with France's plan), '(503) 555-9573'
    (with the United States')."""
    text = value.strip()
    international = text.startswith(("+", "00"))
    if international:
        rest = re.sub(r"\(0\)", "", text[1:] if text.startswith("+") else text[2:])
        digits = re.sub(r"\D", "", rest)
        known = plans_by_code()
        code = next((digits[:k] for k in (1, 2, 3) if digits[:k] in known), None)
        if code is None:
            return Dialled(True, None, None, digits[:3] or None)
        found = known[code]
        rest_digits = digits[len(code) :]
        # '+33 06 12 34 56 78': the trunk prefix kept after the country code by mistake
        number = (
            rest_digits
            if len(rest_digits) in found.lengths and found.fits(rest_digits)
            else found.significant(rest_digits)
        )
        return Dialled(True, found, number, code)
    if plan is None:
        return Dialled(False, None, None)
    if text.startswith("(0)"):  # '(0)2-953010': the trunk prefix, dialled inside the country only
        digits = re.sub(r"\D", "", text[3:])
        return Dialled(
            False, plan, digits if len(digits) in plan.lengths and plan.fits(digits) else None
        )
    return Dialled(False, plan, plan.significant(re.sub(r"\D", "", text)))


def is_phone_column(column: Column) -> bool:
    from quantilo.cleaning.rules.shapes import _role

    return _role(column) == "phone"


@dataclass
class PhoneReading:
    """A phone column read number by number (values after the Basic text fixes)."""

    how: str = ""  # how the country of the numbers without a code is known, for the reason
    mapping: dict[str, str] = field(default_factory=dict)  # value → its E.164 form
    impossible: dict[str, Plan] = field(default_factory=dict)  # a length no number of it has
    no_country: list[str] = field(default_factory=list)
    outside: list[str] = field(default_factory=list)  # a country outside the coded plans
    two_countries: list[str] = field(default_factory=list)


def _row_countries(
    advanced: Advanced, column: Column, country: Column, finals: dict[str, str]
) -> dict[str, set[str]]:
    """Each phone value (after the Basic text fixes) → the countries of its rows."""
    phone_text = as_text(column.name, column.dtype)
    country_text = as_text(country.name, country.dtype)
    found = advanced.context.connection.execute(
        f"SELECT {phone_text}, {country_text} FROM {advanced.context.source} WHERE "
        f"{ident(column.name)} IS NOT NULL GROUP BY 1, 2 LIMIT {MAX_READ + 1}"
    ).fetchall()
    countries_of_value: dict[str, set[str]] = {}
    for phone, place in found:
        read = read_country(str(place)) if place is not None else None
        value = finals.get(str(phone), str(phone))
        countries_of_value.setdefault(value, set())
        if read is not None:
            countries_of_value[value].add(read[0])
    return countries_of_value


def _country_column(advanced: Advanced, column: Column) -> Column | None:
    """The table's country column, when its values are countries of the coded table."""
    for other, stages in advanced.context.texts():
        if other.name == column.name or not is_country_column(other):
            continue
        if read_countries(stages.final).share >= COUNTRY_SHARE:
            return other
    return None


def _agreement(dialled: dict[str, Dialled], countries: dict[str, set[str]]) -> float | None:
    """Among the numbers written with a known country code, the share whose row's country has that
    code (the country column is the numbers' own, not a delivery country)."""
    plans = phone_plans()
    checked = agreed = 0
    for value, number in dialled.items():
        places = countries.get(value, set())
        if not number.international or number.plan is None or len(places) != 1:
            continue
        checked += 1
        place = plans.get(next(iter(places)))
        agreed += place is not None and place.code == number.plan.code
    return agreed / checked if checked else None


def read_phones(advanced: Advanced, column: Column, values: list[tuple[str, int]]) -> PhoneReading:
    reading = PhoneReading()
    rows = dict(values)
    dialled = {value: dial(value) for value, _ in values}
    plans = phone_plans()
    names = country_table().countries
    country = _country_column(advanced, column)
    by_row: dict[str, set[str]] | None = None
    if country is not None:
        by_row = _row_countries(advanced, column, country, _finals(advanced, column))
        agreement = _agreement(dialled, by_row)
        if agreement is not None and agreement < AGREE:
            reading.how = (
                f"The country column {country.name} is not the numbers' own: only "
                f"{agreement:.0%} of the numbers written with a country code agree with it. "
            )
            by_row = None
        else:
            reading.how = f"A number without a country code is read with its row's {country.name}"
            reading.how += (
                f" ({agreement:.0%} of those written with a code agree with it). "
                if agreement is not None
                else ". "
            )
    column_plan: Plan | None = None
    if by_row is None:
        codes: dict[str, int] = {}
        for value, number in dialled.items():
            if number.international and number.plan is not None:
                codes[number.plan.code] = codes.get(number.plan.code, 0) + rows[value]
        written = sum(codes.values())
        if written:
            code, count = max(codes.items(), key=lambda item: item[1])
            national = sum(rows[v] for v, d in dialled.items() if not d.international)
            if count >= ONE_CODE * written and national:
                column_plan = plans_by_code()[code]
                place = names[column_plan.country].en if code != "1" else "North America"
                reading.how += (
                    f"The {n(written)} numbers written with a country code are "
                    f"{count / written:.0%} of {place} (+{code}): the {n(national)} written "
                    f"without one are read as numbers of {place}, when their digits fit its plan. "
                )
    for value in _first_of_each_layout(list(rows)):
        number = dialled[value]
        if not number.international:
            plan = column_plan
            if by_row is not None:
                places = by_row.get(value, set())
                if len(places) > 1:
                    reading.two_countries.append(value)
                    continue
                place = next(iter(places), None)
                plan = plans.get(place) if place is not None else None
                if place is not None and plan is None:
                    reading.outside.append(value)
                    continue
            if plan is None:
                reading.no_country.append(value)
                continue
            number = dial(value, plan)
        if number.plan is None:
            reading.outside.append(value)
        elif number.e164 is None:
            reading.impossible[value] = number.plan
        elif number.e164 != value:
            reading.mapping[value] = number.e164
    return reading


def phones(advanced: Advanced) -> None:
    """F-01: phone numbers written several ways, each in the international E.164 form when its
    country is known: written with a code (+33, 0033), or the row's country column, or the one
    country of the column's numbers that carry a code. Never a guessed country."""
    from quantilo.cleaning.rules.shapes import _map_suggestion, shape

    for column, stages in advanced.context.texts():
        if not is_phone_column(column) or not stages.final:
            continue
        values = [
            (v, c)
            for v, c in stages.final
            if v.strip() and v.strip().lower() not in NULLS and PHONE_CHARS.match(v.strip())
        ]
        rows = dict(values)
        if sum(rows.values()) < PHONE_ROWS:
            continue
        layouts = {shape(v.strip()) for v in rows}
        if len(layouts) < 2:
            continue  # one way of writing them already (nyc_dob's 10 bare digits)
        reading = read_phones(advanced, column, values)
        left = _phones_left(reading, rows)
        if reading.mapping:
            cells = sum(rows[v] for v in reading.mapping)
            before, after = next(iter(reading.mapping.items()))
            finals = _final_of_raw(stages)
            spelled = sum(1 + len(finals.get(v, ())) for v in reading.mapping)
            reason = (
                f"{n(cells)} phone numbers of {column.name} are written {len(layouts)} ways "
                f"({', '.join(q(v) for v in _one_per_layout(rows))}): written one way, the "
                f"international form E.164 ({q(before)} → {q(after)}), a + and the country code "
                "then the number without its trunk prefix, which every phone system reads and "
                f"which tells numbers of two countries apart. {reading.how}{left}"
            )
            if spelled > m.MAX_MAP:
                advanced.add(
                    "F-01",
                    "I",
                    [column],
                    f"{n(cells)} phone numbers of {column.name} could be written one way",
                    reason + "Too many different numbers for one step: information only. "
                    "Check your data.",
                    cells=cells,
                    examples=list(reading.mapping.items())[:SHOWN],
                    key="e164",
                )
            else:
                _map_suggestion(
                    advanced,
                    "F-01",
                    column,
                    stages,
                    reading.mapping,
                    f"Write {n(cells)} phone numbers of {column.name} in the international form",
                    reason + "Check your data.",
                    "e164",
                )
        if reading.impossible:
            bad = sum(rows[v] for v in reading.impossible)
            plans = list(dict.fromkeys(reading.impossible.values()))
            told = "; ".join(_plan_text(p) for p in plans[:3]) + ("…" if len(plans) > 3 else "")
            listed = ", ".join(q(v) for v in list(reading.impossible)[:SHOWN])
            advanced.add(
                "F-01",
                "I",
                [column],
                f"{n(bad)} phone numbers of {column.name} cannot be numbers of their country",
                f"{n(bad)} phone numbers of {column.name} have a number of digits, or a first "
                f"digit, no number of their country has ({listed}; "
                f"{told}): a digit missing or too many? They stay as they are. Information only. "
                "Check your data.",
                cells=bad,
                examples=[(v, None) for v in list(reading.impossible)[:SHOWN]],
                key="length",
            )


def _plan_text(plan: Plan) -> str:
    """'France: 9 digits after the trunk prefix 0', 'Norway: 8 digits, starting 2 to 9'."""
    name = country_table().countries[plan.country].en if plan.code != "1" else "North America"
    lengths = " or ".join(str(x) for x in sorted(plan.lengths))
    text = f"{name}: {lengths} digits"
    if plan.trunk:
        text += f" after the trunk prefix {plan.trunk}"
    never = [d for d in "0123456789" if plan.first and d not in plan.first]
    if never and not (plan.trunk and never == [plan.trunk]):
        text += (
            f", never starting with {' or '.join(never)}"
            if len(never) <= 3
            else f", starting with {', '.join(plan.first[:-1])} or {plan.first[-1]}"
        )
    return text


def _first_of_each_layout(values: list[str]) -> list[str]:
    """The values with the first of each layout first, so that the examples show each way."""
    from quantilo.cleaning.rules.shapes import shape

    seen: set[str] = set()
    firsts: list[str] = []
    others: list[str] = []
    for value in values:
        layout = shape(value.strip())
        (others if layout in seen else firsts).append(value)
        seen.add(layout)
    return firsts + others


def _final_of_raw(stages: TextStages) -> dict[str, list[str]]:
    from quantilo.cleaning.rules.advanced import _final_of

    return _final_of(stages)


def _one_per_layout(rows: dict[str, int]) -> list[str]:
    from quantilo.cleaning.rules.shapes import shape

    seen: dict[str, str] = {}
    for value, _count in sorted(rows.items(), key=lambda item: -item[1]):
        seen.setdefault(shape(value.strip()), value)
    return list(seen.values())[:4]


def _phones_left(reading: PhoneReading, rows: dict[str, int]) -> str:
    """What stays as it is, and why (nothing silent)."""
    parts = []
    for values, one, many, why in (
        (reading.no_country, "has", "have", "no country code and no country to read it with"),
        (reading.outside, "is", "are", "of a country whose plan the coded table does not hold "
         "(it has France and most of Europe, North America, the United Kingdom, the Maghreb)"),
        (reading.two_countries, "appears", "appear", "on rows of two countries"),
        (list(reading.impossible), "cannot be", "cannot be",
         "a number of its country (listed apart)"),
    ):  # fmt: skip
        if values:
            count = sum(rows[v] for v in values)
            verb = one if count == 1 else many
            parts.append(f"{n(count)} {'number' if count == 1 else 'numbers'} {verb} {why}")
    if not parts:
        return ""
    return "Left as they are: " + "; ".join(parts) + ". "


# --- Quantities with a unit (F-05, N23) ----------------------------------------------------------

QUANTITY_SHARE = 0.8  # F-05: the column's values that are a number and a known unit
QUANTITY_ROWS = 20
KINDS_OF_QUANTITY = ("grams", "litres", "metres")  # the base unit of each kind, its new column
SUFFIXES = {"grams": "g", "litres": "l", "metres": "m"}
# unit (small letters, without spaces or dots) → (kind, its value in the base unit); and how each
# unit may be written, when not as its key
UNITS: dict[str, tuple[str, float]] = {
    "mg": ("grams", 0.001), "g": ("grams", 1.0), "gr": ("grams", 1.0), "gram": ("grams", 1.0),
    "grams": ("grams", 1.0), "gramme": ("grams", 1.0), "grammes": ("grams", 1.0),
    "kg": ("grams", 1000.0), "kgs": ("grams", 1000.0), "kilo": ("grams", 1000.0),
    "kilos": ("grams", 1000.0), "kilogram": ("grams", 1000.0), "kilograms": ("grams", 1000.0),
    "kilogramme": ("grams", 1000.0), "kilogrammes": ("grams", 1000.0),
    "lb": ("grams", 453.59237), "lbs": ("grams", 453.59237), "oz": ("grams", 28.349523125),
    "ml": ("litres", 0.001), "cl": ("litres", 0.01), "dl": ("litres", 0.1), "l": ("litres", 1.0),
    "litre": ("litres", 1.0), "litres": ("litres", 1.0), "liter": ("litres", 1.0),
    "liters": ("litres", 1.0), "floz": ("litres", 0.0295735295625),
    "mm": ("metres", 0.001), "cm": ("metres", 0.01), "m": ("metres", 1.0),
    "km": ("metres", 1000.0),
}  # fmt: skip
UNIT_WRITINGS = {"floz": r"fl\.? ?oz"}
SPACE = "[ \t  ]*"
# one pattern for the rule (Python) and the step (DuckDB, RE2): only what both read the same way
QUANTITY_PATTERN = (
    f"(?i)^{SPACE}(?:([0-9]+){SPACE}[x×*]{SPACE})?([0-9]+(?:[.,][0-9]+)?){SPACE}("
    + "|".join(UNIT_WRITINGS.get(u, u) for u in sorted(UNITS, key=len, reverse=True))
    + rf")\.?(?:{SPACE}[e℮])?{SPACE}$"  # '150 g e': the EU sign of an estimated quantity
)
QUANTITY = re.compile(QUANTITY_PATTERN)


def unit_key(unit: str) -> str:
    return re.sub(r"[ .]", "", unit.lower())


def read_quantity(value: str) -> tuple[str, float, str] | None:
    """'1,5 kg' → ('grams', 1500.0, 'kg'); '12 x 33 cl' → ('litres', 3.96, 'cl'); None when the
    value is not a number and a known unit."""
    found = QUANTITY.match(value)
    if found is None:
        return None
    times, number, unit = found.groups()
    kind, factor = UNITS[unit_key(unit)]
    amount = float(number.replace(",", ".")) * (int(times) if times else 1) * factor
    return kind, round(amount, 9), unit


def _new_name(base: str, taken: set[str]) -> str:
    name = base
    while name in taken:
        name += "_"
    taken.add(name)
    return name


def quantities(advanced: Advanced) -> None:
    """F-05: a text column of quantities with their unit, read as numbers in the base unit of
    their kind (grams, litres, metres), in new columns: the column itself stays."""
    taken = {c.name for c in advanced.context.columns}
    for column, stages in advanced.context.texts():
        if basic._converted(column) or basic._person_names(column):
            continue
        values = [(v, c) for v, c in stages.final if v.strip() and v.strip().lower() not in NULLS]
        total = sum(c for _, c in values)
        if total < QUANTITY_ROWS or len(values) < 2:
            continue
        read = {v: r for v, _ in values if (r := read_quantity(v)) is not None}
        rows = dict(values)
        cells = sum(rows[v] for v in read)
        if cells < QUANTITY_SHARE * total:
            continue
        kinds = [k for k in KINDS_OF_QUANTITY if any(r[0] == k for r in read.values())]
        names = {k: _new_name(f"{column.name}_{SUFFIXES[k]}", taken) for k in kinds}
        unit_to = _new_name(f"{column.name}_unit", taken)
        units: dict[str, int] = {}
        for value, (_, _, unit) in read.items():
            units[unit_key(unit)] = units.get(unit_key(unit), 0) + rows[value]
        shown_units = ", ".join(u for u, _ in sorted(units.items(), key=lambda i: -i[1])[:8])
        unread = [(v, c) for v, c in values if v not in read]
        unread_rows = sum(c for _, c in unread)
        into = " and ".join(f"{k} in {q(names[k])}" for k in kinds)
        multiplied = [v for v in read if "x" in v.lower() or "×" in v or "*" in v]
        reason = (
            f"{n(cells)} values of {column.name} ({cells / total:.1%} of its rows) are a number "
            f"and a known unit ({shown_units}): converted to {into}, the unit as written in "
            f"{q(unit_to)}. The column itself stays. A comma is read as the decimal mark "
            "(1,5 kg = 1,500 g)"
            + (
                f"; a multiplier before the quantity is read ({q(multiplied[0])} = "
                f"{_amount(read[multiplied[0]])})"
                if multiplied
                else ""
            )
            + ". "
        )
        if len(kinds) > 1:
            reason += "Each value goes to the column of its kind; the other columns stay empty. "
        if unread:
            reason += (
                f"{n(unread_rows)} values have no known unit or are not a single quantity and "
                f"stay empty in the new columns ({', '.join(q(v) for v, _ in unread[:SHOWN])}). "
            )
        step = m.ParseQuantity(
            id="s0",
            column=column.name,
            to_grams=names.get("grams"),
            to_litres=names.get("litres"),
            to_metres=names.get("metres"),
            unit_to=unit_to,
        )
        examples = [(v, _amount(r)) for v, r in list(read.items())[:SHOWN]]
        advanced.add(
            "F-05",
            "S",
            [column],
            f"Read {n(cells)} quantities of {column.name} as numbers",
            reason + "Check your data.",
            step,
            cells=cells,
            examples=examples,
            key="units",
        )


def _amount(reading: tuple[str, float, str]) -> str:
    kind, amount, _unit = reading
    return f"{amount:,.9g} {SUFFIXES[kind] if kind != 'litres' else 'L'}"
