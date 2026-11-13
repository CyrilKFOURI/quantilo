"""Ready-made KPIs, ranked and explained, and the best columns to split them by.

Spec 022, section 1 (rules R-SUG-01 to R-SUG-08), revised by spec 048 (R-SUG-09 to R-SUG-19)
and spec 053 (R-SUG-20 to R-SUG-24: margin, profit margin, share of a status, weighted mean, delay):
the column roles give the candidates, a score ranks them, and every suggestion carries its reasons.
What only the rows can tell (a column that fixes another, two measures that move together) comes
in `Facts`, measured on a sample by `kpi.probe`; without them the rules still work, on names alone.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from quantilo import semantics
from quantilo.kpi.definition import Definition, Goal, Group, Measure, Ratio, Time
from quantilo.kpi.rules import measure_label, ratio_label
from quantilo.reasons import say
from quantilo.typing.knowledge_base import load, name_tokens
from quantilo.typing.types import AnalysisType

MAX_SUGGESTIONS = 12  # the page shows 6 and lists the rest under "More (n)" (048)
MAX_SPLITS = 3
MAX_HIERARCHIES = 2
PROBED_SPLITS = 8  # split candidates whose dependencies are measured
REPEATED = 0.9  # an identifier with fewer distinct values than this share of rows repeats
SPLIT_ROLES = ("country", "region", "city", "status", "gender", "currency")
LEVEL_ROLES = ("rating", "percentage", "duration", "age")
NUMBERS: frozenset[str] = frozenset({"numeric_continuous", "numeric_discrete"})
SPLITS: frozenset[str] = frozenset({"categorical_nominal", "categorical_ordinal", "boolean"})
KEY_TYPES: frozenset[str] = frozenset({"identifier", "categorical_nominal"})
MIN_LEVELS, MAX_LEVELS = 2, 50
# P1: an amount whose name says price or level is averaged, never summed (R-SUG-09).
LEVEL_WORDS: frozenset[str] = frozenset(
    [
        "price",
        "prices",
        "prix",
        "tarif",
        "unit",
        "salary",
        "salaries",
        "salaire",
        "wage",
        "wages",
        "avg",
        "average",
        "mean",
        "moyenne",
        "moyen",
        "rate",
        "balance",
    ]
)
# P16a: the date a KPI is read at (R-SUG-17); lower tier first, birth dates never.
DATE_TIERS: tuple[frozenset[str], ...] = (
    frozenset(
        [
            "order",
            "ordered",
            "transaction",
            "created",
            "creation",
            "create",
            "invoice",
            "purchase",
            "purchased",
            "sale",
            "sold",
            "booking",
            "booked",
            "signup",
            "registration",
            "registered",
            "commande",
            "facture",
        ]
    ),
    frozenset(),  # every other name ("date", "timestamp", "week")
    frozenset(
        [
            *("ship", "shipped", "shipping", "shipment", "delivery", "delivered", "dispatch"),
            # R-SUG-33: an end (contract, subscription) is filled for some rows only and dates an
            # exit, not the activity: read after the other dates
            *("end", "ended", "fin", "closed", "churn", "churned", "cancelled", "canceled"),
            *("termination", "terminated", "resiliation", "expiry", "expiration", "expires"),
        ]
    ),
    frozenset({"updated", "update", "modified", "modification", "last", "changed", "edited"}),
)
NEVER_DATES: frozenset[str] = frozenset({"birth", "birthday", "dob", "born", "naissance"})
# P16b: the dates a delay runs between (R-SUG-24).
DELAY_STARTS: frozenset[str] = DATE_TIERS[0] | frozenset(
    [
        *("start", "started", "opened", "open", "submitted", "received", "requested", "request"),
        *("admission", "admitted", "checkin", "reported", "logged"),
    ]
)
DELAY_ENDS: frozenset[str] = DATE_TIERS[2] | frozenset(
    [
        *("resolved", "resolution", "closed", "close", "completed", "completion", "end", "ended"),
        *("finished", "solved", "fulfilled", "discharge", "discharged", "checkout", "arrival"),
        *("arrived", "paid", "payment", "returned"),
    ]
)
# R-SUG-21: a flow whose name says it is what is left after the costs.
PROFIT_WORDS: frozenset[str] = frozenset(
    ["profit", "profits", "margin", "marge", "benefice", "earnings", "ebit", "ebitda"]
)
WEIGHTED_ROLES = ("percentage",)  # R-SUG-23: rates weighted by a quantity, like unit prices
# P8: a column fixes another on at least this share of the rows (R-SUG-18, R-SUG-08 bis).
DEPENDS = 0.98
REDUNDANT = 0.95  # |r| above which a measure repeats a better-ranked one (R-SUG-16)
CONFIDENCE = {"high": 1.0, "medium": 0.9, "low": 0.75}
NO_DATE = 0.9
WEIGHTS = {
    "R-SUG-02": 90,
    "R-SUG-20": 82,
    "R-SUG-21": 80,
    "R-SUG-22": 72,
    "R-SUG-24": 64,
    "R-SUG-12": 85,
    "R-SUG-10": 80,
    "R-SUG-11": 78,
    "R-SUG-15": 76,
    "R-SUG-03": 70,
    "R-SUG-13": 68,
    "R-SUG-14": 66,
    "R-SUG-09": 62,
    "R-SUG-01": 60,
    "R-SUG-04": 58,
    "R-SUG-05": 56,
    "R-SUG-06": 55,
    "R-SUG-07": 50,
}
SECOND_AMOUNT, COST = 84, 75
WEIGHTS["R-SUG-28"] = 52  # 093: a measure without a role, averaged
WEIGHTS["R-SUG-31"] = 83  # 061: an amount against its target column
WEIGHTS["R-SUG-32"] = 80  # 093 bis: a price per square foot (a price over a surface)
PRICE_WORDS: frozenset[str] = frozenset(["price", "prix", "value", "valeur", "saleprice"])
# R-SUG-32: the area a price is read against, the living one first, land and annexes last
LIVING_WORDS: frozenset[str] = frozenset(
    ["living", "liv", "habitable", "bati", "built", "above", "floor", "flr", "interior", "reelle"]
)
ANNEX_WORDS: frozenset[str] = frozenset(
    ["lot", "land", "terrain", "garage", "bsmt", "basement", "porch", "deck", "pool", "mas", "vnr"]
)
FEET_WORDS: frozenset[str] = frozenset(["sqft", "sf", "feet", "foot", "footage"])
METRE_WORDS: frozenset[str] = frozenset(
    ["m2", "sqm", "meters", "metres", "surface", "surfaces", "superficie"]
)
MEASURES = 2  # R-SUG-28: measures without a role suggested at most
MIN_MEASURE_LEVELS = 20  # R-SUG-28: fewer distinct values look like levels or codes
MEASURE_ROLES = frozenset(["amount", "cost", "quantity", "unit_price", "target", *LEVEL_ROLES])
NOT_MEASURES = frozenset(["year", "month", "weekday", "hour", "latitude", "longitude", "angle"])
PERCENT_WORDS: frozenset[str] = frozenset(
    ["pct", "percent", "percentage", "pourcentage", "rate", "taux", "ratio"]
)
COUNT_WORDS: frozenset[str] = frozenset(["number", "nb", "nbr", "num", "count", "nombre"])


class Column(Protocol):
    name: str
    type: AnalysisType
    role: str | None
    summary: dict[str, Any] | None


@dataclass
class Facts:
    """What a sample of the rows says (048): A → B dependencies and correlations of measures."""

    # (a, b) → share of the rows (both filled) where the value of a fixes the value of b
    dependency: dict[tuple[str, str], float] = field(default_factory=dict)
    # frozenset({a, b}) → Pearson r of two numeric columns
    correlation: dict[frozenset[str], float] = field(default_factory=dict)
    # 093: name → (min, max, whole numbers only) on the sample (S-02 codes and sequences)
    ranges: dict[str, tuple[float | None, float | None, bool]] = field(default_factory=dict)
    rows: int = 0  # rows in the sample

    def fixes(self, a: str, b: str) -> float | None:
        return self.dependency.get((a, b))

    def twins(self, a: str, b: str) -> bool:
        """Near-bijective: each fixes the other (Customer ID and Customer Name)."""
        there, back = self.fixes(a, b), self.fixes(b, a)
        return there is not None and back is not None and there >= DEPENDS and back >= DEPENDS

    def corr(self, a: str, b: str) -> float | None:
        return self.correlation.get(frozenset((a, b)))


@dataclass
class Suggestion:
    rule: str
    name: str
    definition: Definition
    score: int = 0
    reasons: list[str] = field(default_factory=list)


@dataclass
class Hierarchy:
    """Split by the parent, then the child: the child belongs to one parent (R-SUG-18)."""

    parent: str
    child: str
    share: float
    reason: str


def _summary(column: Column) -> dict[str, Any]:
    return column.summary or {}


def _distinct(column: Column) -> int:
    return int(_summary(column).get("distinct_count") or 0)


def _rows(column: Column) -> int:
    return int(_summary(column).get("rows") or 0)


def _valid(column: Column) -> int:
    return _rows(column) - int(_summary(column).get("null_count") or 0)


def _missing(column: Column) -> float:
    rows = _rows(column)
    return int(_summary(column).get("null_count") or 0) / rows if rows else 1.0


def _filled(column: Column) -> float:
    """Share of filled values; a column not summarised yet counts as full."""
    return 1.0 if column.summary is None else 1.0 - _missing(column)


def _tokens(column: Column) -> set[str]:
    return set(name_tokens(column.name))


def code_of(column: Column) -> semantics.Code | None:
    """R-SUG-25 (093): an identifier or a code by its name (`patient_nbr`), whatever its role."""
    if column.role in ("customer_key", "order_key"):
        return None  # keys have their own rules (R-SUG-10 to 15)
    return semantics.code_by_name(column.name, None, column.type)


def repeated(column: Column) -> bool:
    """An identifier whose values repeat (customer_id), unlike a row id (R-SUG-04); 093: a number
    given a measure role whose name says it names something (`patient_nbr`, R-SUG-25) too."""
    valid = _valid(column)
    code = code_of(column)
    named = (
        column.type in NUMBERS
        and column.role in MEASURE_ROLES
        and code is not None
        and code.kind == "identifier"
    )
    return (
        (column.type == "identifier" or named)
        and valid > 0
        and _distinct(column) < (REPEATED * valid)
    )


def is_ratio(column: Column) -> bool:
    """R-SUG-26 (093): 'co2_per_capita', 'cost per unit': an intensity, averaged, never summed."""
    return column.role != "unit_price" and semantics.is_ratio(column.name)


def is_money(column: Column, columns: Sequence[Column] = ()) -> bool:
    """R-SUG-27 (093): an amount whose name says money; 'Total Bsmt SF' is a total of a surface.
    With a currency column, a 'total' is money (`CompTotal` beside `Currency`)."""
    if column.role != "amount" or semantics.is_money(column.name):
        return True
    return any(c.role == "currency" and _distinct(c) >= 1 for c in columns)


def is_count_flag(column: Column) -> bool:
    """R-SUG-06 (093): 'number_of_cyclist_killed' is a count even when its values are 0 and 1."""
    words = semantics.tokens(column.name)
    return column.type == "boolean" and len(words) > 1 and words[0] in COUNT_WORDS


def is_level(column: Column) -> bool:
    """A price, a salary, an average: averaged, never summed (P1, R-SUG-09)."""
    return column.role == "unit_price" or (
        column.role == "amount" and bool(_tokens(column) & (LEVEL_WORDS | PERCENT_WORDS))
    )


def readable(column: Column) -> bool:
    """A column with at least one value readable in its type (a summary of 0 distinct: none)."""
    return column.summary is None or _distinct(column) > 0


# --- Splits and hierarchies (R-SUG-08, R-SUG-18) -------------------------------------------------


def _ranked_splits(columns: Sequence[Column]) -> list[Column]:
    def distance(levels: int) -> int:
        return 0 if 3 <= levels <= 12 else (3 - levels if levels < 3 else levels - 12)

    fitting = [c for c in columns if c.type in SPLITS and MIN_LEVELS <= _distinct(c) <= MAX_LEVELS]
    fitting.sort(key=lambda c: (c.role not in SPLIT_ROLES, distance(_distinct(c)), _missing(c)))
    return fitting


def split_candidates(columns: Sequence[Column], facts: Facts | None = None) -> list[str]:
    """Columns worth splitting by, best first (R-SUG-08); a twin of a better one is left out."""
    kept: list[Column] = []
    for column in _ranked_splits(columns):
        if facts and any(facts.twins(column.name, better.name) for better in kept):
            continue
        kept.append(column)
    return [c.name for c in kept[:MAX_SPLITS]]


def hierarchies(columns: Sequence[Column], facts: Facts | None) -> list[Hierarchy]:
    """B > A when A fixes B on 98 % of the rows and has more levels (P8, R-SUG-18)."""
    if not facts:
        return []
    ranked = _ranked_splits(columns)[:PROBED_SPLITS]
    found: list[Hierarchy] = []
    for parent in ranked:
        for child in ranked:
            share = facts.fixes(child.name, parent.name)
            if (
                share is None
                or share < DEPENDS
                or _distinct(child) <= _distinct(parent)
                or _distinct(child) < 3
                or facts.twins(child.name, parent.name)
            ):
                continue
            reason = (
                f"R-SUG-18 · Each {child.name} belongs to one {parent.name} ({share:.0%} of rows): "
                f"split by {parent.name}, then {child.name}"
            )
            found.append(Hierarchy(parent.name, child.name, round(share, 4), reason))
    return found[:MAX_HIERARCHIES]


# --- Date (R-SUG-17) -----------------------------------------------------------------------------


def _date_tier(column: Column) -> int | None:
    tokens = _tokens(column)
    if tokens & NEVER_DATES:
        return None
    for tier, words in enumerate(DATE_TIERS):
        if tokens & words:
            return tier
    return 1


@dataclass
class DateChoice:
    """A date column of the Builder's *Over time* list, in the order of R-SUG-17 (048 bis)."""

    name: str
    tier: int | None  # 0 the event's date … 3 an update; None a birth date, never a time axis
    reason: str


TIER_WORDS = (
    "the date of the event itself (order, transaction, creation…)",
    "a date column",
    "a later step (shipping, delivery, an end)",
    "an update date, read last",
)


def rank_dates(columns: Sequence[Column]) -> list[DateChoice]:
    """R-SUG-17 bis (048): every date column in the order the rule prefers, each with its reason;
    birth dates come last, marked as never a time axis. The Builder lists them in this order."""
    dates = [c for c in columns if c.type == "datetime"]
    ranked = sorted(
        (
            (tier, -_filled(c), i, c)
            for i, c in enumerate(dates)
            if (tier := _date_tier(c)) is not None
        ),
        key=lambda item: item[:3],
    )
    found: list[DateChoice] = []
    for position, (tier, filled, _, column) in enumerate(ranked):
        best = "R-SUG-17 · Preferred: " if position == 0 else "R-SUG-17 · "
        found.append(
            DateChoice(
                column.name,
                tier,
                f"{best}{TIER_WORDS[tier]}, {-filled:.0%} filled",
            )
        )
    for column in dates:
        if _date_tier(column) is None:
            found.append(
                DateChoice(
                    column.name,
                    None,
                    "R-SUG-17 · A birth date: an attribute of the person, never a time axis",
                )
            )
    return found


def choose_date(columns: Sequence[Column]) -> tuple[Column | None, str | None]:
    """The date a KPI is read at: the order's, then any other, then shipping, then updates."""
    dates = [c for c in columns if c.type == "datetime"]
    ranked = sorted(
        (
            (tier, -_filled(c), i, c)
            for i, c in enumerate(dates)
            if (tier := _date_tier(c)) is not None
        ),
        key=lambda item: item[:3],
    )
    never = [c.name for c in dates if _date_tier(c) is None]
    if not ranked:
        if never:
            return None, say(
                "R-SUG-17.none",
                "R-SUG-17 · No time axis: {never} is a birth date",
                never=", ".join(never),
            )
        return None, None
    tier, _, _, chosen = ranked[0]
    others = " and ".join(item[3].name for item in ranked[1:3])
    said = say("R-SUG-17", "R-SUG-17 · Over {date}: ", date=chosen.name)
    if not others:
        said += say("R-SUG-17.only", "the only date column")
    elif tier == 0:
        said += say(
            "R-SUG-17.event", "the date of the event itself, preferred to {dates}", dates=others
        )
    elif ranked[1][0] > tier:
        said += say("R-SUG-17.later", "preferred to {dates}, a later step", dates=others)
    else:
        said += say("R-SUG-17.filled", "the best-filled date, before {dates}", dates=others)
    if never:
        said += say("R-SUG-17.never", "; {never} is never a time axis", never=", ".join(never))
    return chosen, said


# --- Suggestions ---------------------------------------------------------------------------------


@dataclass
class _Candidate:
    rule: str
    name: str
    measures: list[Measure]
    reason: str
    columns: list[Column]
    weight: int
    ratio: Ratio | None = None
    money: bool = False  # an amount: split by currency when there are several
    single: Column | None = None  # the numeric column of a one-measure KPI (redundancy check)
    more: list[str] = field(default_factory=list)  # reasons after the first one (053)
    goal: Goal | None = None  # 061: read against a target (R-SUG-31)


def _count(value: int) -> str:
    return f"{value:,}"


def _keys(columns: Sequence[Column], role: str) -> list[Column]:
    found = [c for c in columns if c.role == role and c.type in KEY_TYPES and _distinct(c) >= 2]
    return sorted(found, key=_missing)


def _numbers(columns: Sequence[Column], roles: tuple[str, ...]) -> list[Column]:
    """Numbers of these roles that are measures: never a code (R-SUG-25), a coordinate, a ratio
    summed (R-SUG-26) or a 'total' that is no money (R-SUG-27)."""
    return [
        c
        for c in columns
        if c.type in NUMBERS
        and c.role in roles
        and code_of(c) is None
        and not semantics.is_coordinate(c.name, c.role)
        and (c.role not in ("amount", "cost", "quantity", "target") or not is_ratio(c))
        and is_money(c, columns)
    ]


def _by_values(column: Column, facts: Facts | None) -> semantics.Code | None:
    """S-02 on the probe's sample: a fixed-length code (`bbl`) or a sequence number."""
    found = facts.ranges.get(column.name) if facts else None
    if found is None:
        return None
    low, high, integer = found
    return semantics.code_by_values(low, high, integer) or semantics.sequence(
        low, high, integer, _distinct(column)
    )


def _plain_measures(
    columns: Sequence[Column], used: set[str], facts: Facts | None = None, size: int = MEASURES
) -> list[Column]:
    """R-SUG-28 (093): numbers without a measure role (or a 'total' that is no money), which are
    neither codes, coordinates nor parts of a date: averaged."""
    found = [
        c
        for c in columns
        if c.type in NUMBERS
        and c.name not in used
        and (
            c.role is None
            or c.role == "surface"  # 093 bis: an area is averaged too (R-SUG-28)
            or (c.role == "amount" and not is_money(c, columns))
        )
        and c.role not in NOT_MEASURES
        and not semantics.is_date_part(c.name)
        and code_of(c) is None
        and not semantics.is_coordinate(c.name, c.role)
        and (c.summary is None or _distinct(c) >= MIN_MEASURE_LEVELS)
        and _by_values(c, facts) is None
    ]
    found.sort(key=lambda c: c.type != "numeric_continuous")
    return found[:size]


def notes(columns: Sequence[Column]) -> list[str]:
    """What the rules read differently from the column's role (093, owner rule: no silent
    correction), shown under the suggestions with a link to the Columns page."""
    said: list[str] = []
    check = "Check that your data was read correctly on the Columns page."
    for column in columns:
        if column.type in NUMBERS and column.role in MEASURE_ROLES:
            code = code_of(column)
            if code is not None:
                said.append(
                    f"R-SUG-25 · {column.name} is never summed or averaged: {code.reason}, so it "
                    f"names or codes things rather than measuring them. {check}"
                )
                continue
            if column.role in ("amount", "cost", "quantity", "target") and is_ratio(column):
                said.append(
                    f"R-SUG-26 · {column.name} is a ratio (its name says 'per'): averaged, "
                    f"never summed. {check}"
                )
                continue
            if not is_money(column, columns):
                said.append(
                    f"R-SUG-27 · {column.name} says 'total' but no money word: read as a measure, "
                    f"not an amount (no basket, no unit price, no split by currency). {check}"
                )
        if is_count_flag(column):
            said.append(
                f"R-SUG-25 · {column.name} is a count read as yes or no (only 0 and 1 here): no "
                f"share of yes is suggested. {check}"
            )
    return said


def probe_targets(columns: Sequence[Column]) -> tuple[list[str], list[str]]:
    """Columns whose rows are worth reading (048): split candidates and keys (dependencies),
    the measures of one-column KPIs (correlations)."""
    columns = [c for c in columns if readable(c)]
    labels = [c.name for c in _ranked_splits(columns)[:PROBED_SPLITS]]
    for role in ("customer_key", "order_key"):
        labels += [c.name for c in _keys(columns, role)[:1]]
    numbers = [c.name for c in _numbers(columns, ("amount", "cost", "unit_price", "quantity"))]
    numbers += [c.name for c in _plain_measures(columns, set(numbers), size=8)]
    return labels, numbers[:8]


def repeated_keys(columns: Sequence[Column]) -> list[Column]:
    """R-SUG-29 (093): identifiers that repeat (`id_mutation`): by type, or a category whose name
    says an identifier."""
    found = []
    for column in columns:
        code = code_of(column)
        named = column.type in KEY_TYPES and code is not None and code.kind == "identifier"
        valid = _valid(column)
        if (repeated(column) or named) and valid and 2 <= _distinct(column) < REPEATED * valid:
            found.append(column)
    return found[:2]


def probe_keys(columns: Sequence[Column]) -> list[str]:
    """R-SUG-29 (093): repeated identifiers whose dependency on each measure is measured."""
    return [c.name for c in repeated_keys([c for c in columns if readable(c)])]


def _candidates(columns: Sequence[Column], facts: Facts | None) -> list[_Candidate]:
    found: list[_Candidate] = []
    used: set[str] = set()

    def sums(column: Column) -> Measure:
        return Measure(id="m1", agg="sum", column=column.name)

    found.append(
        _Candidate(
            "R-SUG-01",
            "Rows",
            [Measure(id="m1", agg="count")],
            say("R-SUG-01", "R-SUG-01 · Counts the rows: the volume of records, always available"),
            [],
            WEIGHTS["R-SUG-01"],
        )
    )
    amounts = [c for c in _numbers(columns, ("amount",)) if not is_level(c)][:2]
    costs = _numbers(columns, ("cost",))[:1]
    for index, column in enumerate(amounts):
        reason = say(
            "R-SUG-02",
            "R-SUG-02 · {column} is an amount that adds up over rows: summed",
            column=column.name,
        )
        weight = WEIGHTS["R-SUG-02"] if index == 0 else SECOND_AMOUNT
        found.append(
            _Candidate("R-SUG-02", "", [sums(column)], reason, [column], weight, None, True, column)
        )
    for column in costs:
        reason = say(
            "R-SUG-02.cost",
            "R-SUG-02 · {column} is a cost: costs add up over rows, so it is summed",
            column=column.name,
        )
        found.append(
            _Candidate("R-SUG-02", "", [sums(column)], reason, [column], COST, None, True, column)
        )
    targets = [
        c
        for c in _numbers(columns, ("target",))
        if c.summary is None or _distinct(c) >= MIN_MEASURE_LEVELS
    ][:1]
    for amount, target in zip(amounts[:1], targets, strict=False):
        reason = (
            f"R-SUG-31 · {target.name} is a target (its name says so) beside the amount "
            f"{amount.name}: Σ {amount.name} read against Σ {target.name}, in % of target"
        )
        found.append(
            _Candidate(
                "R-SUG-31",
                f"{amount.name} vs {target.name}",
                [sums(amount)],
                reason,
                [amount, target],
                WEIGHTS["R-SUG-31"],
                money=True,
                goal=Goal(column=target.name),
            )
        )
    quantities = _numbers(columns, ("quantity",))[:1]
    for column in [c for c in _numbers(columns, ("amount", "unit_price")) if is_level(c)][:1]:
        percent = sorted(_tokens(column) & PERCENT_WORDS)
        if column.role == "unit_price":
            reason = say(
                "R-SUG-09.unit",
                "R-SUG-09 · {column} is a unit price: averaged, as a sum of prices means nothing",
                column=column.name,
            )
        elif percent:  # KPI-7
            reason = say(
                "R-SUG-09.percent",
                "R-SUG-09 · {column} is a percentage (its name says {word}): averaged, as a sum "
                "of prices means nothing",
                column=column.name,
                word=percent[0],
            )
        else:
            reason = say(
                "R-SUG-09",
                "R-SUG-09 · {column} is a price or level (its name says {word}): averaged, as a "
                "sum of prices means nothing",
                column=column.name,
                word=sorted(_tokens(column) & LEVEL_WORDS)[0],
            )
        measure, more = _averaged(column, quantities)
        found.append(
            _Candidate(
                "R-SUG-09",
                "",
                [measure],
                reason,
                [column],
                WEIGHTS["R-SUG-09"],
                None,
                True,
                column,
                more,
            )
        )
    for column in quantities:
        reason = say(
            "R-SUG-03",
            "R-SUG-03 · {column} is a quantity: units add up, so it is summed",
            column=column.name,
        )
        found.append(
            _Candidate(
                "R-SUG-03", "", [sums(column)], reason, [column], WEIGHTS["R-SUG-03"], single=column
            )
        )
    used.update(c.name for c in [*amounts, *costs, *quantities])

    flow = (amounts or costs or [None])[0]
    quantity = quantities[0] if quantities else None
    found += _margins(columns)
    found += _per_area(columns)
    orders = _keys(columns, "order_key")[:1]
    customers = _keys(columns, "customer_key")[:1]
    if orders and customers and facts and facts.twins(orders[0].name, customers[0].name):
        customers = []  # one order per customer: the customer KPIs would repeat the order ones

    def ratio_of(numerator: Measure, denominator: Measure) -> tuple[list[Measure], Ratio]:
        return [numerator, denominator], Ratio(numerator="m1", denominator="m2")

    for order in orders:
        distinct = Measure(id="m2", agg="distinct", column=order.name)
        lines = _valid(order)
        if _distinct(order) < REPEATED * lines:
            reason = say(
                "R-SUG-11",
                "R-SUG-11 · {column} names the order: {orders} orders over {rows} rows",
                column=order.name,
                orders=_count(_distinct(order)),
                rows=_count(lines),
            )
            found.append(
                _Candidate(
                    "R-SUG-11",
                    "",
                    [Measure(id="m1", agg="distinct", column=order.name)],
                    reason,
                    [order],
                    WEIGHTS["R-SUG-11"],
                )
            )
        if flow is not None:
            measures, ratio = ratio_of(sums(flow), distinct)
            reason = say(
                "R-SUG-12",
                "R-SUG-12 · {flow} ÷ distinct {order}: the average value of an order ({orders} "
                "orders)",
                flow=flow.name,
                order=order.name,
                orders=_count(_distinct(order)),
            )
            found.append(
                _Candidate(
                    "R-SUG-12",
                    "Average basket",
                    measures,
                    reason,
                    [flow, order],
                    WEIGHTS["R-SUG-12"],
                    ratio,
                    True,
                )
            )
        if quantity is not None:
            measures, ratio = ratio_of(sums(quantity), distinct)
            reason = say(
                "R-SUG-13",
                "R-SUG-13 · {quantity} ÷ distinct {order}: how many units an order holds",
                quantity=quantity.name,
                order=order.name,
            )
            found.append(
                _Candidate(
                    "R-SUG-13",
                    "Units per order",
                    measures,
                    reason,
                    [quantity, order],
                    WEIGHTS["R-SUG-13"],
                    ratio,
                )
            )
    if flow is not None and quantity is not None:
        measures, ratio = ratio_of(sums(flow), Measure(id="m2", agg="sum", column=quantity.name))
        reason = say(
            "R-SUG-14",
            "R-SUG-14 · Σ {flow} ÷ Σ {quantity}: the average price paid per unit, weighted by "
            "the units (a ratio of sums, not a mean of prices)",
            flow=flow.name,
            quantity=quantity.name,
        )
        found.append(
            _Candidate(
                "R-SUG-14",
                "Average unit price",
                measures,
                reason,
                [flow, quantity],
                WEIGHTS["R-SUG-14"],
                ratio,
                True,
            )
        )
    for customer in customers:
        reason = say(
            "R-SUG-10",
            "R-SUG-10 · {column} names the customer: {customers} distinct customers over {rows} "
            "rows",
            column=customer.name,
            customers=_count(_distinct(customer)),
            rows=_count(_valid(customer)),
        )
        found.append(
            _Candidate(
                "R-SUG-10",
                "",
                [Measure(id="m1", agg="distinct", column=customer.name)],
                reason,
                [customer],
                WEIGHTS["R-SUG-10"],
            )
        )
        if flow is not None:
            measures, ratio = ratio_of(
                sums(flow), Measure(id="m2", agg="distinct", column=customer.name)
            )
            reason = say(
                "R-SUG-15",
                "R-SUG-15 · {flow} ÷ distinct {customer}: what one customer brings ({customers} "
                "customers)",
                flow=flow.name,
                customer=customer.name,
                customers=_count(_distinct(customer)),
            )
            found.append(
                _Candidate(
                    "R-SUG-15",
                    f"{flow.name} per customer",
                    measures,
                    reason,
                    [flow, customer],
                    WEIGHTS["R-SUG-15"],
                    ratio,
                    True,
                )
            )
    keys = {c.name for c in [*orders, *customers]}
    ids = [c for c in columns if repeated(c) and c.name not in keys]
    ids = [c for c in ids if c.role not in ("customer_key", "order_key")]
    if ids:
        column = ids[0]
        reason = (
            f"R-SUG-04 · {column.name} repeats ({_count(_distinct(column))} values over "
            f"{_count(_valid(column))} rows): counting its distinct values"
        )
        found.append(
            _Candidate(
                "R-SUG-04",
                "",
                [Measure(id="m1", agg="distinct", column=column.name)],
                reason,
                [column],
                WEIGHTS["R-SUG-04"],
            )
        )
        if amounts:
            measures, ratio = ratio_of(
                sums(amounts[0]), Measure(id="m2", agg="distinct", column=column.name)
            )
            reason = f"R-SUG-05 · {amounts[0].name} ÷ distinct {column.name}: the amount per value"
            found.append(
                _Candidate(
                    "R-SUG-05",
                    "",
                    measures,
                    reason,
                    [amounts[0], column],
                    WEIGHTS["R-SUG-05"],
                    ratio,
                    True,
                )
            )
    found += _shares(columns, orders)
    found += _delays(columns)
    booleans = [
        c for c in columns if c.type == "boolean" and c.name not in used and not is_count_flag(c)
    ]
    outcomes = load().outcome_names
    booleans.sort(key=lambda c: not (_tokens(c) & outcomes))  # 053: "Churn" before "Partner"
    if booleans:
        column = booleans[0]
        reason = say(
            "R-SUG-06", "R-SUG-06 · {column} is yes or no: the share of yes", column=column.name
        )
        if _tokens(column) & outcomes:
            reason = say(
                "R-SUG-06.outcome",
                "R-SUG-06 · {column} is yes or no and its name says an outcome: the share of yes",
                column=column.name,
            )
        found.append(
            _Candidate(
                "R-SUG-06",
                "",
                [Measure(id="m1", agg="true_share", column=column.name)],
                reason,
                [column],
                WEIGHTS["R-SUG-06"],
            )
        )
    ratios = [
        c
        for c in columns
        if c.type in NUMBERS
        and c.role in (None, "amount", "cost", "quantity", "target")
        and is_ratio(c)
        and code_of(c) is None
    ]
    levels = [c for c in [*_numbers(columns, LEVEL_ROLES), *ratios] if c.name not in used][:1]
    for column in levels:
        what = "a ratio (its name says 'per')" if is_ratio(column) else _article(column.role)
        reason = (
            say(
                "R-SUG-07.any",
                "R-SUG-07 · {column}: averaged, as for any {role}",
                column=column.name,
                role=column.role,
            )
            if _tokens(column) == {column.role}
            else say(
                "R-SUG-07", "R-SUG-07 · {column} is {what}: averaged", column=column.name, what=what
            )
        )
        measure, more = _averaged(column, quantities if column.role in WEIGHTED_ROLES else [])
        found.append(
            _Candidate(
                "R-SUG-07",
                "",
                [measure],
                reason,
                [column],
                WEIGHTS["R-SUG-07"],
                single=column,
                more=more,
            )
        )
    used.update(c.name for c in levels)
    found += _plain(columns, used, facts)
    return found


def _article(word: str | None) -> str:
    """'an age', 'a duration' (KPI-6)."""
    text = (word or "level").replace("_", " ")
    return f"{'an' if text[:1] in 'aeiou' else 'a'} {text}"


def _plain(columns: Sequence[Column], used: set[str], facts: Facts | None) -> list[_Candidate]:
    """R-SUG-28 and R-SUG-29 (093): measures without a role, averaged; the median when a repeated
    identifier fixes the value (the same value on every row of one key)."""
    found: list[_Candidate] = []
    keys = repeated_keys(columns)
    for column in _plain_measures(columns, used, facts):
        what = (
            "a surface"
            if column.role == "surface"
            else "a measure without a role (no amount, quantity or price)"
        )
        reason = (
            say(
                "R-SUG-28.surface",
                "R-SUG-28 · {column} is a surface: averaged, as its sum rarely means anything",
                column=column.name,
            )
            if column.role == "surface"
            else say(
                "R-SUG-28",
                "R-SUG-28 · {column} is a measure without a role (no amount, quantity or price): "
                "averaged, as its sum rarely means anything",
                column=column.name,
            )
        )
        agg = "mean"
        more: list[str] = []
        key = next(
            (
                k
                for k in keys
                if facts is not None and (facts.fixes(k.name, column.name) or 0.0) >= DEPENDS
            ),
            None,
        )
        if key is not None and facts is not None:
            agg = "median"
            reason = (
                f"R-SUG-28 · {column.name} is {what}: its median, as its sum rarely means anything"
            )
            share = facts.fixes(key.name, column.name) or 0.0
            more.append(
                f"R-SUG-29 · {column.name} repeats on every row of one {key.name} ({share:.0%} of "
                "rows): a sum would count it several times, the median reads each value as it is"
            )
        found.append(
            _Candidate(
                "R-SUG-28",
                "",
                [Measure(id="m1", agg=agg, column=column.name)],
                reason,
                [column],
                WEIGHTS["R-SUG-28"],
                single=column,
                more=more,
            )
        )
    return found


def _averaged(column: Column, quantities: list[Column]) -> tuple[Measure, list[str]]:
    """A mean, weighted by the first quantity when there is one (R-SUG-23, P18)."""
    if not quantities:
        return Measure(id="m1", agg="mean", column=column.name), []
    weight = quantities[0]
    reason = say(
        "R-SUG-23",
        "R-SUG-23 · Weighted by {weight}: Σ({column} x {weight}) ÷ Σ {weight}, so a value on 100 "
        "units counts 100 times (rows without a positive {weight} are left out)",
        weight=weight.name,
        column=column.name,
    )
    measure = Measure(id="m1", agg="weighted_mean", column=column.name, weight=weight.name)
    return measure, [reason]


def _margins(columns: Sequence[Column]) -> list[_Candidate]:
    """R-SUG-20 (revenue and cost) and R-SUG-21 (profit and revenue), P4."""
    found: list[_Candidate] = []
    flows = [c for c in _numbers(columns, ("amount",)) if not is_level(c)]
    profits = [c for c in flows if _tokens(c) & PROFIT_WORDS]
    revenues = [c for c in flows if c not in profits]
    costs = _numbers(columns, ("cost",))[:1]
    if revenues and costs:
        revenue, cost = revenues[0], costs[0]
        measures = [
            Measure(id="m1", agg="sum", column=revenue.name),
            Measure(id="m2", agg="sum", column=cost.name),
        ]
        reason = say(
            "R-SUG-20",
            "R-SUG-20 · {revenue} and the cost {cost}: the margin (Σ {revenue} - Σ {cost}) ÷ "
            "Σ {revenue}, in percent",
            revenue=revenue.name,
            cost=cost.name,
        )
        found.append(
            _Candidate(
                "R-SUG-20",
                "Margin",
                measures,
                reason,
                [revenue, cost],
                WEIGHTS["R-SUG-20"],
                Ratio(numerator="m1", denominator="m2", kind="margin"),
                True,
            )
        )
    if profits and revenues:
        profit, revenue = profits[0], revenues[0]
        measures = [
            Measure(id="m1", agg="sum", column=profit.name),
            Measure(id="m2", agg="sum", column=revenue.name),
        ]
        reason = say(
            "R-SUG-21",
            "R-SUG-21 · {profit} is what is left of {revenue}: Σ {profit} ÷ Σ {revenue} is the "
            "margin, in percent",
            profit=profit.name,
            revenue=revenue.name,
        )
        found.append(
            _Candidate(
                "R-SUG-21",
                "Profit margin",
                measures,
                reason,
                [profit, revenue],
                WEIGHTS["R-SUG-21"],
                Ratio(numerator="m1", denominator="m2", percent=True),
                True,
            )
        )
    return found


def _surfaces(columns: Sequence[Column]) -> list[Column]:
    """R-SUG-32: surfaces (role `surface`), the living area first, then any, land and annexes last;
    at equal rank the best-filled, then the first."""
    found = [
        (1 - bool(_tokens(c) & LIVING_WORDS) + bool(_tokens(c) & ANNEX_WORDS), -_filled(c), i, c)
        for i, c in enumerate(columns)
        if c.type in NUMBERS and c.role == "surface" and code_of(c) is None
    ]
    return [item[3] for item in sorted(found, key=lambda item: item[:3])]


def area_unit(column: Column) -> str:
    """'sq ft' for a name in feet (sqft, SF), 'm²' for metres or a French surface, else the name."""
    tokens = _tokens(column)
    if tokens & FEET_WORDS:
        return "sq ft"
    if tokens & METRE_WORDS:
        return "m²"
    return column.name


def _per_area(columns: Sequence[Column]) -> list[_Candidate]:
    """R-SUG-32 (093 bis, KPI-3): a price beside a surface gives the price per square foot (or m²),
    Σ price ÷ Σ surface: each sale weighs by its size, so a studio does not count like a house."""
    surfaces = _surfaces(columns)
    prices = [c for c in _numbers(columns, ("amount",)) if _tokens(c) & PRICE_WORDS]
    if not surfaces or not prices:
        return []
    price, surface = prices[0], surfaces[0]
    unit = area_unit(surface)
    others = [c.name for c in surfaces[1:3]]
    chosen = f" ({surface.name} preferred to {' and '.join(others)})" if others else ""
    reason = (
        f"R-SUG-32 · {price.name} is a price and {surface.name} a surface{chosen}: "
        f"Σ {price.name} ÷ Σ {surface.name}, the price of one {unit} weighted by size "
        "(an average of row ratios would let a small one weigh as much as a large one)"
    )
    measures = [
        Measure(id="m1", agg="sum", column=price.name),
        Measure(id="m2", agg="sum", column=surface.name),
    ]
    name = f"{price.name} per {unit}"
    return [
        _Candidate(
            "R-SUG-32",
            name,
            measures,
            reason,
            [price, surface],
            WEIGHTS["R-SUG-32"],
            Ratio(numerator="m1", denominator="m2", label=name[:80]),
            True,
        )
    ]


def _levels(column: Column) -> list[str]:
    """The most frequent values of a column, from its summary (8 at most)."""
    return [str(bar.get("label")) for bar in _summary(column).get("histogram") or []]


def _shares(columns: Sequence[Column], orders: list[Column]) -> list[_Candidate]:
    """R-SUG-22 (P5): the share of a status's tracked outcome (% paid, % churned)."""
    statuses = [
        c
        for c in columns
        if c.type in ("categorical_nominal", "categorical_ordinal")
        and MIN_LEVELS <= _distinct(c) <= MAX_LEVELS
    ]
    statuses.sort(key=lambda c: c.role != "status")
    key = next((o for o in orders if _distinct(o) < REPEATED * _valid(o)), None)
    found: list[_Candidate] = []
    for column in statuses:
        levels = _levels(column)
        lowered = {level.lower().strip(): level for level in levels}
        for share, other in load().outcomes:
            tracked = [lowered[word] for word in lowered if word in share]
            opposite = [lowered[word] for word in lowered if word in other]
            if not tracked or not opposite:
                continue
            reason = (
                say(
                    "R-SUG-22.key",
                    "R-SUG-22 · {column} is a status ({tracked} against {opposite}): the share of "
                    "distinct {key} that are {states}",
                    column=column.name,
                    tracked=", ".join(tracked),
                    opposite=", ".join(opposite),
                    key=key.name,
                    states=" or ".join(tracked),
                )
                if key
                else say(
                    "R-SUG-22",
                    "R-SUG-22 · {column} is a status ({tracked} against {opposite}): the share of "
                    "rows that are {states}",
                    column=column.name,
                    tracked=", ".join(tracked),
                    opposite=", ".join(opposite),
                    states=" or ".join(tracked),
                )
            )
            measure = Measure(
                id="m1",
                agg="share",
                column=column.name,
                values=tracked,
                key=key.name if key else None,
            )
            used = [column, key] if key else [column]
            found.append(_Candidate("R-SUG-22", "", [measure], reason, used, WEIGHTS["R-SUG-22"]))
            break
        if len(found) >= 2:
            break
    return found


def _delays(columns: Sequence[Column]) -> list[_Candidate]:
    """R-SUG-24 (P16b): the average delay from a start date to an end date."""
    dates = [c for c in columns if c.type == "datetime" and not _tokens(c) & NEVER_DATES]
    starts = [c for c in dates if _tokens(c) & DELAY_STARTS and not _tokens(c) & DELAY_ENDS]
    ends = [c for c in dates if _tokens(c) & DELAY_ENDS and not _tokens(c) & DELAY_STARTS]
    if not starts or not ends:
        return []
    start, end = starts[0], ends[0]
    reason = say(
        "R-SUG-24",
        "R-SUG-24 · {start} then {end}: the average number of days between the two",
        start=start.name,
        end=end.name,
    )
    measure = Measure(id="m1", agg="mean", column=start.name, until=end.name)
    return [_Candidate("R-SUG-24", "", [measure], reason, [start, end], WEIGHTS["R-SUG-24"])]


def _score(candidate: _Candidate, has_date: bool) -> tuple[int, list[str]]:
    # A key is known by its whole name; its type's confidence only says it has many levels.
    confidence = min(
        (
            str(getattr(c, "confidence", "high"))
            for c in candidate.columns
            if c.role not in ("customer_key", "order_key")
        ),
        key=lambda level: CONFIDENCE.get(level, 1.0),
        default="high",
    )
    filled = min((_filled(c) for c in candidate.columns), default=1.0)
    factor = CONFIDENCE.get(confidence, 1.0) * filled * (1.0 if has_date else NO_DATE)
    score = round(candidate.weight * factor)
    parts: list[str] = [
        say("R-SUG-16.weight", "weight {weight} for this kind of KPI", weight=candidate.weight)
    ]
    if confidence != "high":
        parts.append(
            say(
                "R-SUG-16.confidence",
                "{confidence}-confidence type (x{factor})",
                confidence=confidence,
                factor=f"{CONFIDENCE.get(confidence, 1.0):g}",
            )
        )
    if filled < 1:
        parts.append(say("R-SUG-16.filled", "{share} of values filled", share=f"{filled:.0%}"))
    if not has_date:
        parts.append(say("R-SUG-16.no_date", "no date column (x{penalty})", penalty=f"{NO_DATE:g}"))
    return score, parts


def _score_reason(score: int, parts: list[str]) -> str:
    """R-SUG-16: the score and what made it, parts separated by commas."""
    said: str = say("R-SUG-16", "R-SUG-16 · Score {score}/100: ", score=score)
    for index, part in enumerate(parts):
        if index:
            said += say("SEP.comma", ", ")
        said += part
    return said


def suggest(columns: Sequence[Column], facts: Facts | None = None) -> list[Suggestion]:
    """Up to 12 KPIs, best score first, each with its reasons (R-SUG-01 to 19)."""
    columns = [c for c in columns if readable(c)]
    date, date_reason = choose_date(columns)
    time = Time(column=date.name) if date else None
    splits = split_candidates(columns, facts)
    groups = [] if time or not splits else [Group(column=splits[0])]
    currencies = [c for c in columns if c.role == "currency" and _distinct(c) >= 2][:1]

    scored: list[tuple[int, int, _Candidate, list[str]]] = []
    score_parts: dict[int, list[str]] = {}  # R-SUG-16: what made each score, to halve it
    for order, candidate in enumerate(_candidates(columns, facts)):
        score, parts = _score(candidate, time is not None)
        score_parts[order] = parts
        score_reason = _score_reason(score, parts)
        reasons = [candidate.reason, score_reason, *candidate.more]
        if date_reason:
            reasons.append(date_reason)
        scored.append((score, order, candidate, reasons))
    scored.sort(key=lambda item: (-item[0], item[1]))

    # R-SUG-16: a measure that moves with a better-ranked one loses half its score.
    if facts:
        kept: list[Column] = []
        for index, (score, order, candidate, reasons) in enumerate(scored):
            single = candidate.single
            if single is None:
                continue
            twin = next(
                (
                    better
                    for better in kept
                    if abs(facts.corr(single.name, better.name) or 0.0) > REDUNDANT
                ),
                None,
            )
            if twin is not None:
                r = facts.corr(single.name, twin.name) or 0.0
                halved = score // 2
                reasons[1] = _score_reason(halved, score_parts[order])
                reasons.append(
                    say(
                        "R-SUG-16.twin",
                        "R-SUG-16 · Moves with {column} (r = {r}): score halved",
                        column=twin.name,
                        r=f"{r:.2f}",
                    )
                )

                score = halved
                scored[index] = (score, order, candidate, reasons)
            kept.append(single)
        scored.sort(key=lambda item: (-item[0], item[1]))

    found: list[Suggestion] = []
    names: set[str] = set()
    for score, _, candidate, reasons in scored:
        kpi_groups = groups
        if candidate.money and currencies:
            currency = currencies[0]
            kpi_groups = [Group(column=currency.name)]
            reasons = [
                *reasons,
                say(
                    "R-SUG-19",
                    "R-SUG-19 · Split by {column}: amounts in {count} currencies are never added "
                    "together",
                    column=currency.name,
                    count=_distinct(currency),
                ),
            ]
        definition = Definition(
            measures=candidate.measures,
            ratio=candidate.ratio,
            time=time,
            groups=kpi_groups,
            goal=candidate.goal,
        )
        name = candidate.name or (
            ratio_label(candidate.ratio, candidate.measures)
            if candidate.ratio
            else measure_label(candidate.measures[0])
        )
        name = name[:1].upper() + name[1:]
        if name in names:
            continue
        names.add(name)
        found.append(Suggestion(candidate.rule, name, definition, score, reasons))
    return found[:MAX_SUGGESTIONS]
