"""N02: the catalogue of cleaning rules, the one place that says which fixes are reliable.

Each rule, or sub-rule when one rule does things of different reliability (`B-04:sure` and
`B-04:na`), has a mode:
- `automatic`: measured right on 99 % or more of the proposals judged on real files (or lossless by
  construction); its suggestions in mode A are `auto`, applied in the draft by default (N03) and
  listed under "Applied automatically" (N04), each undone in one click;
- `suggested`: found in mode A but not measured enough to be applied on its own: left unticked
  (N03, the owner's decision of 2026-11-12), its step's origin `suggested`;
- `decision`: never ticked, the user decides (mode S, risk review);
- `information`: a finding without a step.

The measure is the judged proposals of the audit (docs/etudes/2026-11-12-audit-nettoyage.md, § 2.3),
re-run with `scripts/clean_bench.py --real`; the committed corpus (`tests/clean_corpus/`) keeps
every automatic rule at 99 % or more in CI (N01). A rule falls back to `suggested` as soon as a
measure puts it under the threshold.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

from quantilo.cleaning import text as tx
from quantilo.cleaning.suggestion import CleanSuggestion
from quantilo.typing.knowledge_base import load

AUTO_PRECISION = 0.99  # the owner's threshold (2026-11-12): below it, never applied on its own
MEASURED_ON = "2026-11-13"  # the real files re-run with the audit's judgements (31 files)
AUDIT = "audit 2026-11-12 (§ 2.3), 31 real files"
PROBE = "the audit's French probe, extended (synthetic, seeded): no real file has the case"
PHONE_FILES = (
    "the real files with phone numbers in several layouts (Northwind customers, suppliers)"
)
QUANTITY_FILES = "the corpora's text columns of quantities (Open Food Facts) and the French probe"
COUNTRY_FILES = "the 11 real files with a country column (Northwind, IMDB, OWID, online-retail…)"

CatalogueMode = Literal["automatic", "suggested", "decision", "information"]

# B-04:file (N08): weak tokens that are the file's own code for "missing" when they sit in several
# columns; B-09:repeated (N09): the same sentinel in this many numeric columns.
FILE_TOKENS = frozenset({"?", "-", "--", "unknown"})
REPEATED_COLUMNS = 3
BARE_NA = "na"


@dataclass(frozen=True)
class Measure:
    """Proposals judged right or wrong, and the cells (or rows) they change."""

    judged: int
    correct: int
    cells: int | None = None
    cells_correct: int | None = None
    source: str = AUDIT
    measured_on: str = MEASURED_ON

    def __post_init__(self) -> None:
        if not 0 <= self.correct <= self.judged:
            raise ValueError(f"{self.correct} right out of {self.judged} judged")
        if (self.cells is None) != (self.cells_correct is None):
            raise ValueError("cells and cells_correct go together")
        if self.cells is not None and not 0 <= (self.cells_correct or 0) <= self.cells:
            raise ValueError(f"{self.cells_correct} right cells out of {self.cells}")

    @property
    def precision(self) -> float | None:
        return self.correct / self.judged if self.judged else None

    @property
    def cell_precision(self) -> float | None:
        if self.cells is None or not self.cells:
            return None
        return (self.cells_correct or 0) / self.cells


@dataclass(frozen=True)
class Entry:
    id: str  # "B-04:sure": the rule, and what part of it
    rule: str  # "B-04"
    title: str
    mode: CatalogueMode
    measure: Measure | None
    evidence: str  # how it was judged, what goes wrong when it does
    lossless: bool = False  # cannot change a value's meaning (B-02): automatic without real cases
    # the milestone its reliability rests on (B-13: the country guard, N12): automatic only once
    # that milestone is in the engine (`GUARDS`), suggested until then
    needs: str | None = None

    def __post_init__(self) -> None:
        if self.mode == "automatic" and not self.reliable:
            raise ValueError(
                f"{self.id}: automatic needs a measure at {AUTO_PRECISION:.0%} or more "
                "(or a lossless rule)"
            )

    @property
    def reliable(self) -> bool:
        if self.lossless:
            return True
        precision = None if self.measure is None else self.measure.precision
        cells = None if self.measure is None else self.measure.cell_precision
        return (
            precision is not None
            and precision >= AUTO_PRECISION
            and (cells is None or cells >= AUTO_PRECISION)
        )

    @property
    def automatic(self) -> bool:
        return self.mode == "automatic" and (self.needs is None or GUARDS[self.needs]())

    @property
    def effective_mode(self) -> CatalogueMode:
        return "suggested" if self.mode == "automatic" and not self.automatic else self.mode

    def note(self) -> str:
        """The sentence an automatic suggestion's reason ends with."""
        judged = 0 if self.measure is None else self.measure.judged
        if self.lossless and not judged:
            return (
                "A reliable fix: it only removes characters that cannot be seen, so no value "
                "changes meaning. Check your data."
            )
        return (
            f"A reliable fix: this rule was right on {AUTO_PRECISION:.0%} or more of the cases "
            f"measured (n = {judged:,}). Check your data."
        )


def _m(judged: int, correct: int, cells: int | None = None, right: int | None = None) -> Measure:
    if cells is not None and right is None:
        right = cells if judged == correct else None
    return Measure(judged, correct, cells, right)


def _country_guard() -> bool:
    """N12: B-13 pads a postal code only for the countries whose codes are that long (the knowledge
    base's postal lengths, which the rule reads)."""
    return bool(getattr(load(), "postal_lengths", ()))


GUARDS = {"N12": _country_guard}
NOT_MEASURED = None

ENTRIES: tuple[Entry, ...] = (
    # --- Basic ---------------------------------------------------------------------------------
    Entry(
        "B-01",
        "B-01",
        "Spaces at the ends, doubled or unusual",
        "automatic",
        _m(50, 50, 372_603),
        "Every change is blanks only; line breaks looked for by hand (one layout field).",
    ),
    Entry(
        "B-02",
        "B-02",
        "Invisible characters",
        "automatic",
        _m(0, 0),
        "No proposal on the real files; removes zero-width characters only (lossless), checked "
        "on the corpus.",
        lossless=True,
    ),
    Entry(
        "B-03",
        "B-03",
        "Same value up to case or accents",
        "automatic",
        _m(20, 20, 1_539),
        "Each group read; the target is sometimes lower case ('kroger'), cosmetic.",
    ),
    Entry(
        "B-03:person",
        "B-03",
        "Case of people's names",
        "decision",
        NOT_MEASURED,
        "Two spellings of a name may be two people (CLN-5).",
    ),
    Entry(
        "B-04:sure",
        "B-04",
        "Sure words for 'missing' ('N/A', 'NULL', 'missing', blank)",
        "automatic",
        _m(21, 21, 4_937),
        "'N/A', 'null', 'NULL', 'missing', 'nan', blank: always empty in the files read.",
    ),
    Entry(
        "B-04:na",
        "B-04",
        "A bare 'NA' read as missing",
        "suggested",
        _m(18, 9, 3_415, 2_390),
        "Ames: 'NA' means 'no garage' or 'no basement' (the data's documentation); right in "
        "nyc_dob and openfoodfacts. N06 makes it a decision.",
    ),
    Entry(
        "B-04:file",
        "B-04",
        "A weak token that is the file's code for 'missing' (several columns)",
        "automatic",
        _m(10, 10, 197_111),
        "'?' in 3 columns of uci_adult and 7 of diabetes; automatic once N08 ticks it.",
    ),
    Entry(
        "B-04:weak",
        "B-04",
        "Words for 'missing' that can be a real category",
        "decision",
        NOT_MEASURED,
        "'Unknown', 'None', 'Unspecified': a real answer in nyc_311, diabetes (decision 8).",
    ),
    Entry(
        "B-04:none",
        "B-04",
        "'NA' that means 'none of it', written 'None' (N06)",
        "decision",
        NOT_MEASURED,
        "Ames Garage and Bsmt columns.",
    ),
    Entry(
        "B-05",
        "B-05",
        "Text saved in one encoding and read in another",
        "automatic",
        _m(1, 1, 1),
        "The cp1252 -> UTF-8 round trip is checked value by value; n is small (fr_baac).",
    ),
    Entry(
        "B-05:wide",
        "B-05",
        "Text read wrong twice, as Latin-1 or as Mac Roman (N28)",
        "suggested",
        NOT_MEASURED,
        "Each value's bytes read back until valid UTF-8 with no marker left; no real file had "
        "one (the synthetic corpus does): ticked, never applied on its own.",
    ),
    Entry(
        "B-05:lost",
        "B-05",
        "Characters lost before the file was saved ('�', N28)",
        "information",
        NOT_MEASURED,
        "U+FFFD replaced the character for good: counted, nothing can repair it.",
    ),
    Entry(
        "B-06",
        "B-06",
        "HTML codes ('&amp;')",
        "automatic",
        _m(7, 7, 17),
        "'&gt;', '&quot;' in openfoodfacts.",
    ),
    Entry(
        "B-07",
        "B-07",
        "Numbers stored as text",
        "automatic",
        _m(42, 42, 44_335),
        "'69,55%', '03', blanks; every value read.",
    ),
    Entry(
        "B-07:unreadable",
        "B-07",
        "Numbers stored as text, some unreadable",
        "decision",
        NOT_MEASURED,
        "The unreadable values become empty.",
    ),
    Entry(
        "B-07:currency",
        "B-07",
        "Several currencies in one column",
        "decision",
        NOT_MEASURED,
        "N18: the currency goes to its own column; nothing is converted.",
    ),
    Entry(
        "B-07:bounds",
        "B-07",
        "Bounds in words ('Less than 1 year')",
        "decision",
        NOT_MEASURED,
        "Read as their bound: a choice.",
    ),
    Entry(
        "B-09",
        "B-09",
        "A sentinel value (-999, a round isolated peak)",
        "suggested",
        NOT_MEASURED,
        "No automatic proposal on the real files: not measured.",
    ),
    Entry(
        "B-09:repeated",
        "B-09",
        "The same sentinel in 3 numeric columns or more",
        "automatic",
        _m(16, 16, 22_995),
        "-200 in 13 columns of air_quality, -100000 in 3 of chicago_beach; automatic once N09 "
        "ticks it.",
    ),
    Entry(
        "B-09:zero",
        "B-09",
        "Positions at (0, 0)",
        "decision",
        _m(2, 2, 988),
        "Right on nyc_collisions, but one file only.",
    ),
    Entry(
        "B-10",
        "B-10",
        "Outside the role's range, negatives",
        "decision",
        NOT_MEASURED,
        "Negatives are often real (cancellations, owid). N10: also the range a physical measure's "
        "name gives (water temperature, relative humidity, battery level); chicago_beach: 2 water "
        "temperatures above 40 °C, right (2026-11-13).",
    ),
    Entry(
        "B-10:year",
        "B-10",
        "An impossible year in a year column (N11)",
        "decision",
        _m(1, 1, 1),
        "melbourne YearBuilt 1196 among 1830 to 2018 (1916 or 1996 said, never written); owid "
        "year from 1750: nothing (2026-11-13).",
    ),
    Entry(
        "B-22",
        "B-22",
        "0 for 'never' in a year column (N10)",
        "decision",
        _m(1, 1, 20_699),
        "kc_house yr_renovated: 0 on 20,699 rows means never renovated (the data's "
        "documentation); the only year column with zeros among the 31 files (2026-11-13).",
    ),
    Entry(
        "B-22:rare",
        "B-22",
        "A count of rooms or a built area at 0 on a few rows (N10)",
        "information",
        _m(5, 5, 90),
        "kc_house bedrooms (13) and bathrooms (10), melbourne Bedroom2 (16), Bathroom (34) and "
        "BuildingArea (17): values worth a look, shown and never changed (2026-11-13).",
    ),
    Entry(
        "B-11",
        "B-11",
        "Dates stored as text",
        "automatic",
        _m(5, 5, 35_100),
        "Day / month order decided on the column, checked on dates after the 12th.",
    ),
    Entry(
        "B-11:review",
        "B-11",
        "Dates stored as text, to review",
        "decision",
        NOT_MEASURED,
        "Some values unreadable or ambiguous.",
    ),
    Entry(
        "B-17",
        "B-17",
        "Excel serial numbers as dates",
        "suggested",
        NOT_MEASURED,
        "Not seen on the real files.",
    ),
    Entry(
        "B-12",
        "B-12",
        "Identifiers with a trailing '.0'",
        "automatic",
        _m(1, 1, 148_889),
        "'17669.0' -> '17669' (online_retail); n is small.",
    ),
    Entry(
        "B-13:postal",
        "B-13",
        "Postal codes that lost their leading zero",
        "automatic",
        _m(1, 1, 438),
        "Superstore, US states checked one by one; Belgian or Swiss codes have 4 digits by "
        "nature: automatic only with the country guard (N12).",
        needs="N12",
    ),
    Entry(
        "B-13:code",
        "B-13",
        "Codes that lost their leading zero",
        "decision",
        NOT_MEASURED,
        "A code of digits may be a counter.",
    ),
    Entry(
        "B-13:text",
        "B-13",
        "Codes kept as text (leading zeros)",
        "decision",
        NOT_MEASURED,
        "CLN-6.",
    ),
    Entry(
        "B-14:id",
        "B-14",
        "Exact copies of a row, identifier included",
        "automatic",
        _m(1, 1, 45),
        "imdb: the same movie_imdb_link and every other value; n is small.",
    ),
    Entry(
        "B-14",
        "B-14",
        "Exact copies of a row, no identifier",
        "decision",
        NOT_MEASURED,
        "uci_adult, fr_dvf: identical rows can be real (C-07).",
    ),
    Entry(
        "B-15",
        "B-15",
        "Empty rows and repeated header rows",
        "automatic",
        _m(1, 1, 114),
        "';;;' at the end of air_quality; n is small.",
    ),
    Entry(
        "B-16",
        "B-16",
        "Placeholder dates (1900-01-01, 9999-12-31)",
        "suggested",
        NOT_MEASURED,
        "No proposal on the real files.",
    ),
    Entry(
        "B-18:mixed",
        "B-18",
        "Yes / no written several ways",
        "suggested",
        NOT_MEASURED,
        "No automatic proposal on the real files.",
    ),
    Entry(
        "B-18:pair",
        "B-18",
        "Two values that mean yes and no",
        "decision",
        _m(4, 4, 23_402),
        "Right on 'Y' / 'N', but the display changes with no gain (decision 7 of 065).",
    ),
    Entry(
        "B-18:checkbox",
        "B-18",
        "A checkbox: one mark and blanks (N27)",
        "decision",
        _m(1, 1, 6_395),
        "nyc_dob residential: 'YES' or blank, read true and empty at import; a blank means not "
        "residential (the data's documentation). No longer called a constant column (B-19) "
        "(2026-11-13).",
    ),
    Entry(
        "B-18:mark",
        "B-18",
        "Yes / no words with a box mark ('X') among them (N27)",
        "decision",
        NOT_MEASURED,
        "French probe ('Oui', 'Non', 'X'); not seen on the real files.",
    ),
    Entry("B-19", "B-19", "Empty or constant columns", "decision", NOT_MEASURED, "C-07."),
    Entry("B-20", "B-20", "A date assembled from its parts", "decision", NOT_MEASURED, "A choice."),
    Entry("B-21", "B-21", "Rows that sum up other rows", "decision", NOT_MEASURED, "owid: right."),
    Entry(
        "B-23",
        "B-23",
        "A key that repeats with differences (N22)",
        "decision",
        _m(79, 79),
        "IMDB: 79 rows over 76 links, same title, director and year (2026-11-13).",
    ),
    Entry(
        "B-24:header", "B-24", "Header under a note (N47)", "suggested", _m(1, 1), "kiwi: right."
    ),
    # --- Advanced (097): never automatic (C-17); estimates from reading the groups --------------
    Entry(
        "A-01",
        "A-01",
        "Spellings with the same fingerprint",
        "decision",
        _m(28, 28),
        "Columns of lists compared item by item (N13): 35/22 before, 28/28 after (2026-11-13).",
    ),
    Entry(
        "A-02",
        "A-02",
        "Spellings with the same 2-grams",
        "decision",
        _m(11, 11),  # N14: no A-02 on names of people (17/11 before)
        "People's names: distinct surnames ('LE ROUX' / 'LEROUX', N14).",
    ),
    Entry(
        "A-03",
        "A-03",
        "A rare spelling near a frequent one",
        "decision",
        _m(7, 6),
        "'William' / 'Williams', two agencies.",
    ),
    Entry(
        "A-21",
        "A-21",
        "Business synonyms ('Not Rated' / 'Unrated', N30)",
        "decision",
        _m(1, 1),
        "imdb: 'Unrated' (62 rows) is 'Not Rated' (116); coded lists only (synonyms.toml), "
        "nothing on superstore (2026-11-13).",
    ),
    Entry(
        "F-02",
        "F-02",
        "Identifiers with a check digit (N21)",
        "decision",
        _m(17, 17),
        "French probe: the 14 corrupted SIRENs; Open Food Facts: 3 EANs (2026-11-13).",
    ),
    Entry(
        "F-01",
        "F-01",
        "Phone numbers in the international form (N20)",
        "decision",
        Measure(115, 115, 115, 115, source=PHONE_FILES),
        "Each number read with its row's country: Northwind customers (phone, fax) and suppliers, "
        "same digits, the right code, the trunk prefix dropped (kept in Italy); Norway's "
        "'07-98 92 35' and '(0)2-953010' listed, not rewritten. French probe: 600 / 600 numbers in "
        "4 layouts, the country column rejected (63 % agree with the codes); nyc_dob: one layout, "
        "nothing.",
    ),
    Entry(
        "F-04",
        "F-04",
        "Countries written several ways (N24)",
        "decision",
        Measure(15, 15, 7_892, 7_892, source=COUNTRY_FILES),
        "Each country group read: Northwind (3 tables) and IMDB USA / UK among names, "
        "online-retail EIRE, USA, RSA, Open Food Facts États-Unis, Deutschland…, Stack Overflow "
        "'Republic of Korea' beside 'South Korea'; OWID's World and income groups, IMDB's West "
        "Germany never merged. French probe: 2 groups, 5 writings.",
    ),
    Entry(
        "F-05",
        "F-05",
        "Quantities with a unit read as numbers (N23)",
        "decision",
        Measure(2, 2, 4_161, 4_161, source=QUANTITY_FILES),
        "Open Food Facts 'quantity' (40 %): 3,561 values read, each unit spelling (g, ml, oz, "
        "fl oz, kg, l, lb, cl, gr, grams, liter) and 40 values drawn at random checked; "
        "'2 x 125 g' = 250 g, '1,155 kg' = 1,155 g; '7 oz (198 g)', '12 pcs' left empty and "
        "listed. French probe 'poids': 600 / 600 grams. No other text column of the corpora "
        "reads as quantities (Northwind 'quantityPerUnit' is packings).",
    ),
    Entry(
        "A-04",
        "A-04",
        "Shapes of codes",
        "decision",
        _m(14, 8),  # N17: 18 judged, 8 right before
        "N17: Corsican and overseas INSEE codes ('2A', '2B004', '971') and intervals ('[0-10)') "
        "are no longer unusual shapes: the 4 wrong informations of fr_baac, fr_rne_maires (2) "
        "and diabetes are gone (2026-11-13).",
    ),
    Entry(
        "A-05",
        "A-05",
        "Units x100 or x1,000",
        "decision",
        NOT_MEASURED,
        "N17: a percentage's low tail with no gap (0.04 % among scores) is not rescaled; the 2 "
        "proposals of the audit (législatives, openfoodfacts, 0 right) are gone, none is left on "
        "the 31 files; messy_orders' discounts written 0.05 are still found (2026-11-13).",
    ),
    Entry(
        "A-04:domain",
        "A-04",
        "Mistyped e-mail domains (N26)",
        "decision",
        Measure(4, 4, 22, 22, source=PROBE, measured_on="2026-11-13"),
        "No real file has enough e-mails (audit § 4.3); the extended French probe: the 4 planted "
        "slips (gmial.com, hotmial.com, '.con', gmail.fr), no name before '@' rewritten.",
    ),
    Entry("A-06", "A-06", "Unusual values", "information", NOT_MEASURED, "Marked, not changed."),
    Entry("A-07", "A-07", "Floating-point noise", "decision", NOT_MEASURED, "Not seen."),
    Entry("A-08", "A-08", "Benford's law", "information", NOT_MEASURED, "A hint (C-51)."),
    Entry("A-09", "A-09", "Last digits", "information", NOT_MEASURED, "A hint (C-51)."),
    Entry(
        "A-10",
        "A-10",
        "A value that breaks a dependency",
        "decision",
        _m(5, 3),  # N15: exceptions kept as information (12/2 before)
        "Repeated real exceptions: Marble Hill, postcodes on a border, epicene names (N15).",
    ),
    Entry(
        "A-11",
        "A-11",
        "Dates out of order",
        "decision",
        _m(2, 2),  # N16: 1 proposal, a wrong swap, before
        "nyc_311: the 9 closed_date of the same minute written without seconds, said and not "
        "swapped; a 2026 creation against a 2024 update listed, not swapped (60 days or more "
        "apart); swaps only under 60 days (N16, 2026-11-13).",
    ),
    Entry(
        "A-12",
        "A-12",
        "Total unequal to its parts",
        "decision",
        NOT_MEASURED,
        "N17: a percentage is never added to counts: législatives' 'Inscrits = Votants + % "
        "Votants + Abstentions' is gone; eco2mix's 'eolien = terrestre + offshore' (79 rows) is "
        "left, not judged (2 proposals, 0 right before; 2026-11-13).",
    ),
    Entry("A-13", "A-13", "Age against birth date", "decision", NOT_MEASURED, "Not seen."),
    Entry("A-14", "A-14", "Dates written another way", "decision", NOT_MEASURED, "Not measured."),
    Entry("A-15", "A-15", "Near-duplicate rows", "decision", NOT_MEASURED, "Not measured."),
    Entry(
        "A-16",
        "A-16",
        "Records of one entity (Fellegi-Sunter)",
        "decision",
        NOT_MEASURED,
        "N17: a pair is one record only when it differs on one column at most, all columns read, "
        "and never on a code, a repeated identifier or a time of day: air quality's hours, "
        "nyc_dob's permits and online_retail's stock codes are gone (3 proposals, 0 right "
        "before); dvf keeps 45 rows of one parcel and crop whose land differs by 1 m², not "
        "judged (2026-11-13).",
    ),
    Entry(
        "A-17",
        "A-17",
        "Almost empty rows",
        "decision",
        _m(2, 1),  # N17: eco2mix's forecasts left out (3/1 before)
        "N17: the rows after the last full one are data to come (eco2mix's 144 forecasts), "
        "kept; chicago_beach, imdb unchanged (2026-11-13).",
    ),
    Entry(
        "A-18",
        "A-18",
        "A value of a group repeated on its rows (N33)",
        "information",
        _m(1, 1),
        "dvf: valeur_fonciere on every line of its id_mutation (226 lines for 2024-3574); the only "
        "one on the 31 files (superstore's Sales vary by line) (2026-11-13).",
    ),
    Entry(
        "A-19:copy",
        "A-19",
        "A column that is another written again (N34)",
        "decision",
        _m(5, 5),
        "A copy (nyc_311 park_borough), a date written as text (chicago_beach "
        "measurement_timestamp_label), seconds since 1970 (openfoodfacts created_t and 2 more), "
        "checked on every sampled row; removed only if ticked (C-07) (2026-11-13).",
    ),
    Entry(
        "A-19",
        "A-19",
        "Two columns one to one (N34)",
        "information",
        _m(5, 5),
        "uci_adult education / education-num, nyc_311 agency / agency_name, fr_rne code / label "
        "of the CSP, openfoodfacts countries and states (tags / names); a key with its name "
        "(superstore Customer ID) is left out (2026-11-13).",
    ),
    Entry(
        "A-20",
        "A-20",
        "One entity written several ways, numbered (N36)",
        "decision",
        _m(618, 617),  # clusters judged one by one on nyc_dob, the only real file with a case
        "Judged cluster by cluster on nyc_dob (10,000 permits): 530 permittees by their phone, 37 "
        "superintendents' companies and 51 owners by their postcode, all one company each "
        "(N.Y.C. HOUSING AUTHORITY / NYC HOUSING AUTHORTY, LE HARVE / LE HAVRE OWNERS CORP.) but "
        "KALABI / KALARI REALTY CO., not sure. 'n/a' / 'na', A.T.A. / AIA and north / south "
        "(gis_nta_name, a place) were wrong before the guards. Nothing on the 30 other files "
        "(superstore's and Northwind's names are written one way). A new column, the names "
        "unchanged: never ticked (2026-11-13).",
    ),
    # --- Tables of a project (N32, rules/tables.py): Advanced, never ticked -----------------------
    Entry(
        "T-01",
        "T-01",
        "Rows whose key is missing from the table it points to",
        "information",
        NOT_MEASURED,
        "Counted exactly on every row (R-JOIN-06); kept, a join keeps them with empty columns.",
    ),
    Entry(
        "T-01:near",
        "T-01",
        "A missing key one letter away from exactly one key (R-REL-15)",
        "decision",
        NOT_MEASURED,
        "'90200D' / '90200B' may be two parts (C-49): shown as a fix to tick, never joined.",
    ),
    Entry(
        "T-02",
        "T-02",
        "A key that repeats in the table others point to",
        "information",
        NOT_MEASURED,
        "Its rows would match twice: the build refuses the relation (R-BLD-03).",
    ),
    # --- Expert (101): run one by one, never ticked (C-30) --------------------------------------
    Entry("E-01", "E-01", "Cell errors learned from marks", "decision", NOT_MEASURED, "C-30."),
    Entry("E-03", "E-03", "Suspicious labels", "decision", NOT_MEASURED, "C-30."),
    Entry("E-06", "E-06", "Unusual rows (Isolation Forest)", "decision", NOT_MEASURED, "C-30."),
    Entry("E-07", "E-07", "Filling empty cells (KNN, MICE)", "decision", NOT_MEASURED, "C-36."),
)

CATALOGUE: dict[str, Entry] = {entry.id: entry for entry in ENTRIES}


def automatic() -> list[Entry]:
    return [entry for entry in ENTRIES if entry.automatic]


def _values(suggestion: CleanSuggestion) -> set[str]:
    step = suggestion.step
    values = getattr(step, "values", None) or []
    return {str(v).strip().lower() for v in values}


def _sentinel(suggestion: CleanSuggestion) -> str | None:
    values = _values(suggestion)
    return next(iter(values)) if len(values) == 1 else None


def classify(suggestion: CleanSuggestion, found: Iterable[CleanSuggestion] = ()) -> str:
    """The catalogue entry a suggestion falls under, from its rule, mode and step (and, for the
    rules that look across columns, the other suggestions of the same file)."""
    rule, mode = suggestion.rule, suggestion.mode
    op = getattr(suggestion.step, "op", None)
    if rule == "B-03":
        return "B-03" if mode == "A" else "B-03:person"
    if rule == "B-04":
        return _classify_tokens(suggestion, found)
    if rule == "B-05":
        return _classify_encoding(suggestion)
    if rule == "T-01":
        return "T-01" if suggestion.step is None else "T-01:near"
    if rule == "B-07":
        if op == "map_values":
            return "B-07:bounds"
        if op == "extract_text":
            return "B-07:currency"
        return "B-07" if mode == "A" else "B-07:unreadable"
    if rule == "B-09":
        return _classify_sentinel(suggestion, found)
    if rule == "B-11":
        return "B-11" if mode == "A" else "B-11:review"
    if rule == "B-13":
        if op == "pad_zeros":
            return "B-13:postal" if mode == "A" else "B-13:code"
        return "B-13:text"
    if rule == "B-14":
        return "B-14:id" if mode == "A" else "B-14"
    if rule == "B-18":
        if suggestion.id.endswith((":checkbox", ":mark")):
            return "B-18:" + suggestion.id.rsplit(":", 1)[1]
        return "B-18:mixed" if mode == "A" else "B-18:pair"
    if rule == "A-04" and suggestion.id.endswith(":domain"):
        return "A-04:domain"
    if rule == "B-10" and suggestion.id.endswith(":year"):
        return "B-10:year"
    if rule == "A-19":
        return "A-19:copy" if suggestion.step is not None else "A-19"
    if rule == "B-22":
        return "B-22" if suggestion.step is not None else "B-22:rare"
    return rule


def _classify_encoding(suggestion: CleanSuggestion) -> str:
    """N28: B-05 is automatic for UTF-8 read once as Windows-1252 (measured); the other readings
    (twice, Latin-1, Mac Roman) are suggested; '�' is an information."""
    if suggestion.step is None:
        return "B-05:lost"
    mapping = getattr(suggestion.step, "map", None) or {}
    for value in mapping:
        repaired = tx.repair_encoding(str(value))
        if repaired is not None and repaired[1] != "windows-1252":
            return "B-05:wide"
    return "B-05"


def _classify_tokens(suggestion: CleanSuggestion, found: Iterable[CleanSuggestion]) -> str:
    if suggestion.id.endswith(":none") or getattr(suggestion.step, "op", None) == "map_values":
        return "B-04:none"
    values = _values(suggestion)
    if suggestion.mode == "A" and BARE_NA in values:
        return "B-04:na"
    if values and values <= FILE_TOKENS:
        column = suggestion.columns[0] if suggestion.columns else None
        elsewhere = any(
            other.rule == "B-04"
            and other.columns
            and other.columns[0] != column
            and _values(other) & values
            for other in found
        )
        if elsewhere or len(suggestion.columns) >= 2:
            return "B-04:file"
    return "B-04:sure" if suggestion.mode == "A" else "B-04:weak"


def _classify_sentinel(suggestion: CleanSuggestion, found: Iterable[CleanSuggestion]) -> str:
    if suggestion.id.endswith(":0,0"):
        return "B-09:zero"
    if len(suggestion.columns) >= REPEATED_COLUMNS:
        return "B-09:repeated"
    value = _sentinel(suggestion)
    if value is not None:
        columns = {
            column
            for other in found
            if other.rule == "B-09" and _sentinel(other) == value
            for column in other.columns
        }
        if len(columns) >= REPEATED_COLUMNS:
            return "B-09:repeated"
    return "B-09"


def entry_of(suggestion: CleanSuggestion, found: Iterable[CleanSuggestion] = ()) -> Entry | None:
    return CATALOGUE.get(classify(suggestion, found))


def annotate(found: list[CleanSuggestion]) -> list[CleanSuggestion]:
    """Each suggestion's catalogue entry; `auto` when that entry is automatic and the rule ticked
    it (mode A, risk safe), with the reason saying why it is reliable. In place, and returned."""
    for suggestion in found:
        entry = entry_of(suggestion, found)
        suggestion.catalogue = None if entry is None else entry.id
        auto = (
            entry is not None
            and entry.automatic
            and suggestion.layer == "basic"
            and suggestion.mode == "A"
            and suggestion.risk == "safe"
            and suggestion.step is not None
        )
        if auto and not suggestion.auto and entry is not None:
            suggestion.reason = _with_note(suggestion.reason, entry.note())
        suggestion.auto = auto
        suggestion.measured = (
            (0 if entry.measure is None else entry.measure.judged) if auto and entry else None
        )
        # N03: a step is "automatic" (applied without asking, said so in Versions and Changes)
        # only when its rule is reliable; a ticked rule that is not stays a suggestion
        if suggestion.step is not None and suggestion.step.origin == "auto" and not auto:
            suggestion.step.origin = "suggested"
    return found


def _with_note(reason: str, note: str) -> str:
    if note in reason:
        return reason
    tail = " Check your data."
    base = reason[: -len(tail)] if reason.endswith(tail) else reason
    return f"{base.rstrip()} {note}"


def table() -> list[dict[str, Any]]:
    """The catalogue as rows (the bench, the docs)."""
    rows = []
    for entry in ENTRIES:
        measure = entry.measure
        rows.append(
            {
                "id": entry.id,
                "rule": entry.rule,
                "title": entry.title,
                "mode": entry.effective_mode,
                "judged": None if measure is None else measure.judged,
                "correct": None if measure is None else measure.correct,
                "precision": None if measure is None else measure.precision,
                "cell_precision": None if measure is None else measure.cell_precision,
                "measured_on": None if measure is None else measure.measured_on,
                "lossless": entry.lossless,
                "needs": entry.needs,
                "evidence": entry.evidence,
            }
        )
    return rows
