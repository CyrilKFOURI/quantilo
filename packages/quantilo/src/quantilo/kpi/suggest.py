"""Ready-made KPIs, ranked and explained, and the best columns to split them by.

Spec 022, section 1 (rules R-SUG-01 to R-SUG-08), revised by spec 048 (R-SUG-09 to R-SUG-19):
the column roles give the candidates, a score ranks them, and every suggestion carries its reasons.
What only the rows can tell (a column that fixes another, two measures that move together) comes
in `Facts`, measured on a sample by `kpi.probe`; without them the rules still work, on names alone.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from quantilo.kpi.definition import Definition, Group, Measure, Ratio, Time
from quantilo.kpi.rules import measure_label, ratio_label
from quantilo.typing.knowledge_base import name_tokens
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
    frozenset({"ship", "shipped", "shipping", "shipment", "delivery", "delivered", "dispatch"}),
    frozenset({"updated", "update", "modified", "modification", "last", "changed", "edited"}),
)
NEVER_DATES: frozenset[str] = frozenset({"birth", "birthday", "dob", "born", "naissance"})
# P8: a column fixes another on at least this share of the rows (R-SUG-18, R-SUG-08 bis).
DEPENDS = 0.98
REDUNDANT = 0.95  # |r| above which a measure repeats a better-ranked one (R-SUG-16)
CONFIDENCE = {"high": 1.0, "medium": 0.9, "low": 0.75}
NO_DATE = 0.9
WEIGHTS = {
    "R-SUG-02": 90,
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


def repeated(column: Column) -> bool:
    """An identifier whose values repeat (customer_id), unlike a row id (R-SUG-04)."""
    valid = _valid(column)
    return column.type == "identifier" and valid > 0 and _distinct(column) < REPEATED * valid


def is_level(column: Column) -> bool:
    """A price, a salary, an average: averaged, never summed (P1, R-SUG-09)."""
    return column.role == "unit_price" or (
        column.role == "amount" and bool(_tokens(column) & LEVEL_WORDS)
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
            return None, f"R-SUG-17 · No time axis: {', '.join(never)} is a birth date"
        return None, None
    tier, _, _, chosen = ranked[0]
    others = [item[3].name for item in ranked[1:3]]
    if not others:
        why = "the only date column"
    elif tier == 0:
        why = f"the date of the event itself, preferred to {' and '.join(others)}"
    elif ranked[1][0] > tier:
        why = f"preferred to {' and '.join(others)}, a later step"
    else:
        why = f"the best-filled date, before {' and '.join(others)}"
    if never:
        why += f"; {', '.join(never)} is never a time axis"
    return chosen, f"R-SUG-17 · Over {chosen.name}: {why}"


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


def _count(value: int) -> str:
    return f"{value:,}"


def _keys(columns: Sequence[Column], role: str) -> list[Column]:
    found = [c for c in columns if c.role == role and c.type in KEY_TYPES and _distinct(c) >= 2]
    return sorted(found, key=_missing)


def _numbers(columns: Sequence[Column], roles: tuple[str, ...]) -> list[Column]:
    return [c for c in columns if c.type in NUMBERS and c.role in roles]


def probe_targets(columns: Sequence[Column]) -> tuple[list[str], list[str]]:
    """Columns whose rows are worth reading (048): split candidates and keys (dependencies),
    the measures of one-column KPIs (correlations)."""
    columns = [c for c in columns if readable(c)]
    labels = [c.name for c in _ranked_splits(columns)[:PROBED_SPLITS]]
    for role in ("customer_key", "order_key"):
        labels += [c.name for c in _keys(columns, role)[:1]]
    numbers = [c.name for c in _numbers(columns, ("amount", "cost", "unit_price", "quantity"))]
    return labels, numbers[:8]


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
            "R-SUG-01 · Counts the rows: the volume of records, always available",
            [],
            WEIGHTS["R-SUG-01"],
        )
    )
    amounts = [c for c in _numbers(columns, ("amount",)) if not is_level(c)][:2]
    costs = _numbers(columns, ("cost",))[:1]
    for index, column in enumerate(amounts):
        reason = f"R-SUG-02 · {column.name} is an amount that adds up over rows: summed"
        weight = WEIGHTS["R-SUG-02"] if index == 0 else SECOND_AMOUNT
        found.append(
            _Candidate("R-SUG-02", "", [sums(column)], reason, [column], weight, None, True, column)
        )
    for column in costs:
        reason = f"R-SUG-02 · {column.name} is a cost: costs add up over rows, so it is summed"
        found.append(
            _Candidate("R-SUG-02", "", [sums(column)], reason, [column], COST, None, True, column)
        )
    for column in [c for c in _numbers(columns, ("amount", "unit_price")) if is_level(c)][:1]:
        what = (
            "a unit price"
            if column.role == "unit_price"
            else f"a price or level (its name says {sorted(_tokens(column) & LEVEL_WORDS)[0]})"
        )
        reason = f"R-SUG-09 · {column.name} is {what}: averaged, as a sum of prices means nothing"
        measure = Measure(id="m1", agg="mean", column=column.name)
        found.append(
            _Candidate(
                "R-SUG-09", "", [measure], reason, [column], WEIGHTS["R-SUG-09"], None, True, column
            )
        )
    quantities = _numbers(columns, ("quantity",))[:1]
    for column in quantities:
        reason = f"R-SUG-03 · {column.name} is a quantity: units add up, so it is summed"
        found.append(
            _Candidate(
                "R-SUG-03", "", [sums(column)], reason, [column], WEIGHTS["R-SUG-03"], single=column
            )
        )
    used.update(c.name for c in [*amounts, *costs, *quantities])

    flow = (amounts or costs or [None])[0]
    quantity = quantities[0] if quantities else None
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
            reason = (
                f"R-SUG-11 · {order.name} names the order: {_count(_distinct(order))} orders "
                f"over {_count(lines)} rows"
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
            reason = (
                f"R-SUG-12 · {flow.name} ÷ distinct {order.name}: the average value of an order "
                f"({_count(_distinct(order))} orders)"
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
            reason = (
                f"R-SUG-13 · {quantity.name} ÷ distinct {order.name}: how many units an order holds"
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
        reason = (
            f"R-SUG-14 · Σ {flow.name} ÷ Σ {quantity.name}: the average price paid per unit, "
            "weighted by the units (a ratio of sums, not a mean of prices)"
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
        reason = (
            f"R-SUG-10 · {customer.name} names the customer: {_count(_distinct(customer))} "
            f"distinct customers over {_count(_valid(customer))} rows"
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
            reason = (
                f"R-SUG-15 · {flow.name} ÷ distinct {customer.name}: what one customer brings "
                f"({_count(_distinct(customer))} customers)"
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
    booleans = [c for c in columns if c.type == "boolean" and c.name not in used]
    if booleans:
        column = booleans[0]
        reason = f"R-SUG-06 · {column.name} is yes or no: the share of yes"
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
    levels = [c for c in _numbers(columns, LEVEL_ROLES) if c.name not in used][:1]
    for column in levels:
        reason = f"R-SUG-07 · {column.name} is a {column.role}: averaged"
        found.append(
            _Candidate(
                "R-SUG-07",
                "",
                [Measure(id="m1", agg="mean", column=column.name)],
                reason,
                [column],
                WEIGHTS["R-SUG-07"],
                single=column,
            )
        )
    return found


def _score(candidate: _Candidate, has_date: bool) -> tuple[int, str]:
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
    parts = [f"weight {candidate.weight} for this kind of KPI"]
    if confidence != "high":
        parts.append(f"{confidence}-confidence type (x{CONFIDENCE.get(confidence, 1.0):g})")
    if filled < 1:
        parts.append(f"{filled:.0%} of values filled")
    if not has_date:
        parts.append(f"no date column (x{NO_DATE:g})")
    return score, f"R-SUG-16 · Score {score}/100: {', '.join(parts)}"


def suggest(columns: Sequence[Column], facts: Facts | None = None) -> list[Suggestion]:
    """Up to 12 KPIs, best score first, each with its reasons (R-SUG-01 to 19)."""
    columns = [c for c in columns if readable(c)]
    date, date_reason = choose_date(columns)
    time = Time(column=date.name) if date else None
    splits = split_candidates(columns, facts)
    groups = [] if time or not splits else [Group(column=splits[0])]
    currencies = [c for c in columns if c.role == "currency" and _distinct(c) >= 2][:1]

    scored: list[tuple[int, int, _Candidate, list[str]]] = []
    for order, candidate in enumerate(_candidates(columns, facts)):
        score, score_reason = _score(candidate, time is not None)
        reasons = [candidate.reason, score_reason]
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
                reasons[1] = reasons[1].replace(f"Score {score}/100", f"Score {halved}/100", 1)
                reasons.append(f"R-SUG-16 · Moves with {twin.name} (r = {r:.2f}): score halved")
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
                f"R-SUG-19 · Split by {currency.name}: amounts in {_distinct(currency)} "
                "currencies are never added together",
            ]
        definition = Definition(
            measures=candidate.measures, ratio=candidate.ratio, time=time, groups=kpi_groups
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
