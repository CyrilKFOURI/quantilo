"""Advanced rules on how values are written (B1, study § 2.2), on each column's distinct values:

- A-04: the shape of codes, phone numbers and e-mails (`A` capital, `a` small, `9` digit, other
  characters as they are; Auto-Detect, SIGMOD 2018): a minority in another case is put back in the
  usual case, a phone number laid out another way is laid out like the others, and what follows no
  usual shape is listed. E-mails (N26): only the domain is put in small letters (the name before
  '@' keeps its case), and the domains of a coded list of slips are put right (gmial.com).
- A-14: dates written another way than the rest of their column, which B-11 would leave empty:
  month first in a day-first column, a two-digit year, a time zone (C-46: their steps come before
  B-11's reading).
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from quantilo.cleaning import formula as fx
from quantilo.cleaning import model as m
from quantilo.cleaning.compile import is_temporal, is_text
from quantilo.cleaning.rules import basic
from quantilo.cleaning.rules.basic import Column, TextStages, n, q, rows_of
from quantilo.ingestion.sql import ident
from quantilo.typing.parsers import VALUE_PATTERNS, english_months

if TYPE_CHECKING:
    from quantilo.cleaning.rules.advanced import Advanced

SHAPE_SHARE = 0.8  # A-04: the usual shape covers this share of the rows
CASE_RATIO = 10  # A-04: the other case is at least ten times rarer
ODD_SHARE = 0.05  # A-04: what follows no usual shape is listed up to this share of the rows
SHAPES_SHOWN = 6
EMAIL = re.compile(VALUE_PATTERNS["email"])
PHONE_SEPARATORS = re.compile(r"^[\d\s().\-/+]+$")
ZONE = re.compile(
    r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})$", re.I
)
NAIVE = re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?$")
ZONE_SHARE = 0.5  # A-14: values with a time zone are at least half the column
# N26: slips in the domains of the big mail providers, each to the domain it misses. Only domains
# that belong to no one are listed ('hotmail.fr', 'yahoo.fr', 'outlook.fr' are real).
DOMAIN_TYPOS = {
    **dict.fromkeys(
        ("gmial.com", "gmal.com", "gmai.com", "gamil.com", "gnail.com", "gmaill.com", "gmali.com",
         "gmail.co", "gmail.cm", "gmail.om", "gmail.fr", "gmail.cmo",
         "g-mail.com", "gmil.com", "gmaik.com", "gmsil.com"),
        "gmail.com",
    ),
    **dict.fromkeys(
        ("hotmial.com", "hotmal.com", "hotmai.com", "hotmil.com", "homail.com", "hotnail.com",
         "hotmaill.com", "hotamil.com", "htomail.com", "hotmail.co", "hotmail.cm"),
        "hotmail.com",
    ),
    **dict.fromkeys(("hotmial.fr", "hotmal.fr", "hotmai.fr", "homail.fr", "hotnail.fr"),
                    "hotmail.fr"),
    **dict.fromkeys(("yaho.com", "yahooo.com", "yhoo.com", "yahou.com", "yahoo.co", "yahoo.cm",
                     "yaoo.com"), "yahoo.com"),
    **dict.fromkeys(("yaho.fr", "yahooo.fr", "yhoo.fr", "yahou.fr"), "yahoo.fr"),
    **dict.fromkeys(("outlok.com", "outllok.com", "outloo.com", "outlook.co", "otlook.com"),
                    "outlook.com"),
    **dict.fromkeys(("outlok.fr", "outllok.fr", "outloo.fr"), "outlook.fr"),
    **dict.fromkeys(("iclod.com", "icoud.com", "icloud.co", "iclould.com"), "icloud.com"),
    **dict.fromkeys(("wanado.fr", "wanadoo.f", "wanadoo.fe", "wandoo.fr"), "wanadoo.fr"),
    **dict.fromkeys(("orange.f", "oragne.fr", "ornage.fr"), "orange.fr"),
    **dict.fromkeys(("laposte.ne", "lapost.net", "laposte.nte"), "laposte.net"),
    "free.f": "free.fr",
    "sfr.f": "sfr.fr",
}  # fmt: skip
# … and endings that are no top-level domain, read as '.com'
TLD_TYPOS = {"con": "com", "cmo": "com", "ocm": "com", "vom": "com", "xom": "com", "comm": "com",
             "coom": "com", "cpm": "com", "c0m": "com"}  # fmt: skip


_ASCII_SHAPE = str.maketrans(
    {
        **{chr(c): "A" for c in range(ord("A"), ord("Z") + 1)},
        **{chr(c): "a" for c in range(ord("a"), ord("z") + 1)},
        **{chr(c): "9" for c in range(ord("0"), ord("9") + 1)},
    }
)
_LOWER = re.compile("[a-z]")
_UPPER = re.compile("[A-Z]")


def shape(value: str) -> str:
    """'ORD-100023' → 'AAA-999999', '85123a' → '99999a'."""
    if value.isascii():  # N44: the same, in one call (the only digits are 0 to 9)
        return value.translate(_ASCII_SHAPE)
    out = []
    for char in value:
        if "A" <= char <= "Z":
            out.append("A")
        elif "a" <= char <= "z":
            out.append("a")
        elif char.isdigit():
            out.append("9")
        else:
            out.append(char)
    return "".join(out)


def _role(column: Column) -> str | None:
    view = column.view
    if view is not None and view.role in ("email", "phone"):
        return view.role
    if column.role is not None and column.role.id in ("email", "phone"):
        return column.role.id
    return None


def _finals(stages: TextStages) -> dict[str, list[str]]:
    from quantilo.cleaning.rules.advanced import _final_of

    return _final_of(stages)


def _map_suggestion(
    advanced: Advanced,
    rule: str,
    column: Column,
    stages: TextStages,
    mapping: dict[str, str],
    title: str,
    reason: str,
    key: str,
) -> None:
    """A `map_values` on the values after the Basic text fixes and on their raw spellings
    (C-23)."""
    raws = _finals(stages)
    full: dict[str, str] = {}
    weights: dict[str, int] = {}
    rows = dict(stages.final)
    for value, new in mapping.items():
        for spelling in {value, *raws.get(value, [])}:
            if spelling != new:
                full[spelling] = new
                weights[spelling] = rows.get(value, 0)
    if not full:
        return
    cells = sum(rows.get(value, 0) for value in mapping)
    advanced.add(
        rule,
        "S",
        [column],
        title,
        reason,
        m.MapValues(id="s0", column=column.name, map=basic._capped(full, weights)),
        cells=cells,
        examples=list(mapping.items())[:10],
        key=key,
    )


# --- A-04 ----------------------------------------------------------------------------------------


def codes(advanced: Advanced) -> None:
    for column, stages in advanced.context.texts():
        if basic._converted(column) or basic._person_names(column) or not stages.final:
            continue
        role = _role(column)
        if role == "email":
            _emails(advanced, column, stages)
        elif role == "phone":
            _phones(advanced, column, stages)
        elif basic._is_code(stages):
            _codes(advanced, column, stages)


def _real(stages: TextStages) -> list[tuple[str, int]]:
    return [
        (v, c)
        for v, c in stages.final
        if v.strip() and v.strip().lower() not in basic.SURE_TOKENS | basic.WEAK_TOKENS
    ]


def _codes(advanced: Advanced, column: Column, stages: TextStages) -> None:
    values = _real(stages)
    total = sum(c for _, c in values)
    if total < 20:
        return
    shapes: Counter[str] = Counter()
    for value, count in values:
        shapes[shape(value)] += count
    lower = sum(c for v, c in values if _LOWER.search(v))
    upper = sum(c for v, c in values if _UPPER.search(v))
    mapping: dict[str, str] = {}
    for case, rare, often in (("upper", lower, upper), ("lower", upper, lower)):
        if not rare or often < CASE_RATIO * rare:
            continue
        for value, _count in values:
            changed = value.upper() if case == "upper" else value.lower()
            if changed != value and shapes[shape(changed)] >= CASE_RATIO * shapes[shape(value)]:
                mapping[value] = changed
        if mapping:
            word = "small letters" if case == "upper" else "capitals"
            cells = sum(c for v, c in values if v in mapping)
            merged = sum(1 for new in mapping.values() if new in dict(values))
            reason = (
                f"{n(cells)} values of {column.name} are written in {word}, while "
                f"{n(often)} rows use the other case ({q(next(iter(mapping)))} → "
                f"{q(next(iter(mapping.values())))}). "
                + (f"{n(merged)} of them then equal an existing code. " if merged else "")
                + "Case can carry meaning in some codes: check before ticking."
            )
            _map_suggestion(
                advanced,
                "A-04",
                column,
                stages,
                mapping,
                f"Write {n(cells)} codes of {column.name} in {case} case",
                reason,
                "case",
            )
            break
    if advanced.basic_found and any(
        f.rule in ("B-12", "B-13") and f.columns == [column.name] for f in advanced.basic_found
    ):
        return  # Basic already reads these codes (a trailing '.0', a lost zero)
    usual, rows = shapes.most_common(1)[0]
    if rows < SHAPE_SHARE * total:
        return
    odd = [
        (v, c)
        for v, c in values
        if shape(v) != usual and v not in mapping and not _known_code(v, usual)
    ]
    odd_rows = sum(c for _, c in odd)
    if not odd or odd_rows > ODD_SHARE * total:
        return
    _odd_information(advanced, column, odd, odd_rows, usual, rows / total, "codes")


# N17: what a code column holds by nature beside its usual shape: Corsica's département
# ('2A', '2B') and its communes ('2A004'), overseas départements ('971'); intervals ('[0-10)'
# among '[70-80)', diabetes' ages)
INSEE = {"99": re.compile(r"2[AB]|9[78][0-9]"), "99999": re.compile(r"2[AB][0-9]{3}")}
INTERVAL = re.compile(r"[\[(] ?-?[0-9]+(?:[.,][0-9]+)? ?[-–;,] ?-?[0-9]+(?:[.,][0-9]+)? ?[\])]")


def _known_code(value: str, usual: str) -> bool:
    insee = INSEE.get(usual)
    if insee is not None and insee.fullmatch(value):
        return True
    return bool(INTERVAL.fullmatch(value)) and bool(INTERVAL.fullmatch(usual.replace("9", "1")))


def _odd_information(
    advanced: Advanced,
    column: Column,
    odd: list[tuple[str, int]],
    odd_rows: int,
    usual: str,
    share: float,
    noun: str,
) -> None:
    by_shape: dict[str, int] = defaultdict(int)
    for value, count in odd:
        by_shape[shape(value)] += count
    shown = ", ".join(q(v) for v, _ in odd[:SHAPES_SHOWN])
    reason = (
        f"{share:.1%} of {column.name} has the shape {q(usual)} (A = capital, a = small "
        f"letter, 9 = digit); {n(odd_rows)} rows do not ({shown}, {len(by_shape)} other "
        "shapes): typing errors, or another kind of value? Information only: check these values."
    )
    advanced.add(
        "A-04",
        "I",
        [column],
        f"{n(odd_rows)} {noun} of {column.name} have an unusual shape",
        reason,
        cells=odd_rows,
        examples=[(v, shape(v)) for v, _ in odd[:10]],
        key="shape",
    )


def _small_domain(address: str) -> str:
    """'Jean.Dupont@Gmail.COM' → 'Jean.Dupont@gmail.com': the domain only (N26); the name before
    '@' keeps its case, which a mail server may read."""
    local, _, domain = address.rpartition("@")
    return f"{local}@{domain.lower()}"


def _fixed_domain(domain: str) -> str | None:
    """A domain of the coded list of slips (N26), or a '.con' for '.com', put right; None when the
    domain is not one of them."""
    domain = domain.lower()
    if domain in DOMAIN_TYPOS:
        return DOMAIN_TYPOS[domain]
    head, dot, tld = domain.rpartition(".")
    if dot and head and tld in TLD_TYPOS:
        return f"{head}.{TLD_TYPOS[tld]}"
    return None


def _emails(advanced: Advanced, column: Column, stages: TextStages) -> None:
    values = _real(stages)
    total = sum(c for _, c in values)
    if total < 20:
        return
    invalid = [(v, c) for v, c in values if not EMAIL.match(v.strip())]
    valid = [(v, c) for v, c in values if EMAIL.match(v.strip())]
    capitals = {v: _small_domain(v) for v, _ in valid if _small_domain(v) != v}
    small = sum(c for v, c in values if "@" in v and _small_domain(v) == v)
    if capitals and small >= SHAPE_SHARE * total:
        cells = sum(c for v, c in values if v in capitals)
        _map_suggestion(
            advanced,
            "A-04",
            column,
            stages,
            capitals,
            f"Write the domain of {n(cells)} e-mail addresses of {column.name} in small letters",
            f"{n(cells)} addresses have capitals in their domain while {small / total:.1%} are "
            f"written in small letters ({q(next(iter(capitals)))} → "
            f"{q(next(iter(capitals.values())))}): a domain is read the same in any case, so two "
            "spellings would count as two addresses. The name before '@' keeps its case (a mail "
            "server may tell it apart). Check your data.",
            "case",
        )
    addresses = [(v, c) for v, c in values if v.count("@") == 1 and not v.startswith("@")]
    _domain_typos(advanced, column, stages, addresses)
    invalid_rows = sum(c for _, c in invalid)
    if invalid and invalid_rows <= ODD_SHARE * total:
        advanced.add(
            "A-04",
            "I",
            [column],
            f"{n(invalid_rows)} e-mail addresses of {column.name} look invalid",
            f"{n(invalid_rows)} values are not written name@domain.tld "
            f"({', '.join(q(v) for v, _ in invalid[:SHAPES_SHOWN])}). Information only: check "
            "these values.",
            cells=invalid_rows,
            examples=[(v, None) for v, _ in invalid[:10]],
            key="shape",
        )


def _domain_typos(
    advanced: Advanced, column: Column, stages: TextStages, addresses: list[tuple[str, int]]
) -> None:
    """N26: domains of the coded list of slips (gmial.com → gmail.com, '.con' → '.com'). The
    address is mapped as stored and with its domain in small letters, so the step finds it whether
    or not the case step came first."""
    mapping: dict[str, str] = {}
    domains: dict[str, str] = {}
    for value, _count in addresses:
        local, _, domain = value.strip().rpartition("@")
        fixed = _fixed_domain(domain)
        if fixed is None:
            continue
        domains[domain.lower()] = fixed
        right = f"{local}@{fixed}"
        mapping[value] = right
        if _small_domain(value) != value:
            mapping[_small_domain(value)] = right
    if not mapping:
        return
    rows = dict(stages.final)
    cells = sum(rows.get(v, 0) for v in mapping)
    shown = ", ".join(f"{d} → {f}" for d, f in list(domains.items())[:SHAPES_SHOWN])
    count = len(domains)
    _map_suggestion(
        advanced,
        "A-04",
        column,
        stages,
        mapping,
        f"Correct {rows_of(count, 'mistyped e-mail domain')} of {column.name}",
        f"{count} e-mail domain{'s look' if count > 1 else ' looks'} mistyped in {column.name} "
        f"({shown}), on {n(cells)} addresses: a coded list of common slips of the big mail "
        "providers (letters swapped or missing, gmail.fr), and '.con' or '.cmo' for '.com', which "
        "is no domain. Only the domain changes; the name before '@' is kept as it is. Check your "
        "data.",
        "domain",
    )


def _phones(advanced: Advanced, column: Column, stages: TextStages) -> None:
    values = [(v, c) for v, c in _real(stages) if PHONE_SEPARATORS.match(v.strip())]
    total = sum(c for _, c in values)
    if total < 20:
        return
    digits = Counter[int]()
    for value, count in values:
        digits[len(re.sub(r"\D", "", value))] += count
    length, same = digits.most_common(1)[0]
    layouts = Counter[str]()
    for value, count in values:
        if len(re.sub(r"\D", "", value)) == length:
            layouts[shape(value.strip())] += count
    layout, laid = layouts.most_common(1)[0]
    mapping: dict[str, str] = {}
    if laid >= SHAPE_SHARE * same:
        for value, _count in values:
            numbers = re.sub(r"\D", "", value)
            if len(numbers) != length or shape(value.strip()) == layout:
                continue
            if value.strip().startswith("+") != layout.startswith("+"):
                continue  # an international prefix cannot be guessed
            filled = iter(numbers)
            mapping[value] = "".join(next(filled) if ch == "9" else ch for ch in layout)
    if mapping:
        cells = sum(c for v, c in values if v in mapping)
        _map_suggestion(
            advanced,
            "A-04",
            column,
            stages,
            mapping,
            f"Lay out {n(cells)} phone numbers of {column.name} like the others",
            f"{laid / same:.1%} of the {length}-digit numbers are written {q(layout)}; "
            f"{n(cells)} are laid out another way ({q(next(iter(mapping)))}): same digits, "
            "written alike.",
            "layout",
        )
    other = [(v, c) for v, c in values if len(re.sub(r"\D", "", v)) != length]
    other_rows = sum(c for _, c in other)
    if other and other_rows <= ODD_SHARE * total and same >= SHAPE_SHARE * total:
        advanced.add(
            "A-04",
            "I",
            [column],
            f"{n(other_rows)} phone numbers of {column.name} have another number of digits",
            f"{same / total:.1%} of the numbers have {length} digits; {n(other_rows)} do not "
            f"({', '.join(q(v) for v, _ in other[:SHAPES_SHOWN])}): a digit missing or too "
            "many? Information only: check these values.",
            cells=other_rows,
            examples=[(v, None) for v, _ in other[:10]],
            key="shape",
        )


# --- A-14 ----------------------------------------------------------------------------------------


def _date_step(advanced: Advanced, column: Column) -> Any:
    for found in advanced.basic_found:
        if found.rule == "B-11" and found.columns == [column.name] and found.step is not None:
            return found.step
    return None


def _reads(value: str, formats: list[str]) -> datetime | None:
    for date_format in formats:
        text = value
        if "%b" in date_format or "%B" in date_format:  # N19: French month names, as in SQL
            text, date_format = english_months(value), date_format.replace("%B", "%b")
        try:
            return datetime.strptime(text, date_format)
        except ValueError:
            continue
    return None


def dates(advanced: Advanced) -> None:
    for column, stages in advanced.context.texts():
        if not is_text(column.dtype):
            continue
        step = _date_step(advanced, column)
        if step is not None and step.formats:
            _other_order(advanced, column, stages, list(step.formats))
        _time_zones(advanced, column, stages)
    for column in advanced.context.columns:
        if is_temporal(column.dtype) and column.nonnull:
            _years_read_short(advanced, column)


def _century(year: int, middle: int) -> int:
    return min((1900, 2000), key=lambda c: abs(c + year - middle))


def _years_read_short(advanced: Advanced, column: Column) -> None:
    """A date column where a few dates fall in the years 0 to 99 while the others are recent: a
    two-digit year read as written ('15/06/24' → 0024). Each such year gets the century that
    brings it closest to the other dates (C-45: a conditional step on the year)."""
    context = advanced.context
    name = ident(column.name)
    years = context.connection.execute(
        f"SELECT year({name}), count(*) FROM {context.source} WHERE {name} IS NOT NULL "
        "GROUP BY 1 ORDER BY 1"
    ).fetchall()
    short = [(int(y), int(c)) for y, c in years if 0 <= y < 100]
    full = [(int(y), int(c)) for y, c in years if y >= 1000]
    total = sum(c for _, c in years)
    if not short or sum(c for _, c in full) < 0.9 * total:
        return
    weighted = sorted(y for y, c in full for _ in range(min(c, 1_000)))
    middle = weighted[len(weighted) // 2]
    quoted = fx.quoted(column.name)
    cases = [
        m.Case(
            when=f"year({quoted}) = {year}",
            formula=f"date_add({quoted}, to_years({_century(year, middle)}))",
        )
        for year, _count in short
    ]
    cells = sum(c for _, c in short)
    first = short[0][0]
    advanced.add(
        "A-14",
        "S",
        [column],
        f"Write the century of {n(cells)} dates of {column.name}",
        f"{n(cells)} dates of {column.name} fall in the years {first:04d} to {short[-1][0]:04d}, "
        f"while the others center on {middle}: a two-digit year read as written (24 → 0024). "
        f"Each gets the century closest to the other dates ({first:04d} → "
        f"{_century(first, middle) + first}). Check dates far in the past, such as birth dates.",
        m.SetWhere(id="s0", column=column.name, cases=cases),
        cells=cells,
        examples=[(f"{y:04d}", str(_century(y, middle) + y)) for y, _ in short[:3]],
        key="date:century",
    )


def _other_order(
    advanced: Advanced, column: Column, stages: TextStages, formats: list[str]
) -> None:
    """Values the column's formats cannot read but their day/month twin (06/15/2024 in a
    day-first column), or their two-digit-year twin (05/03/24), can."""
    swapped_formats = [basic._swapped(f) for f in formats if basic._swapped(f) != f]
    short_formats = [f.replace("%Y", "%y") for f in formats if "%Y" in f]
    years: list[int] = []
    swapped: dict[str, str] = {}
    short: dict[str, tuple[datetime, str]] = {}
    for value, count in stages.final:
        text = value.strip()
        if not text:
            continue
        read = _reads(text, formats)
        if read is not None:
            years += [read.year] * min(count, 50)
            continue
        for twin in swapped_formats:
            other = _reads(text, [twin])
            if other is not None:
                swapped[value] = other.strftime(basic._swapped(twin))
                break
        else:
            for twin in short_formats:
                other = _reads(text, [twin])
                if other is not None:
                    short[value] = (other, twin.replace("%y", "%Y"))
                    break
    rows = dict(stages.final)
    if swapped:
        cells = sum(rows.get(v, 0) for v in swapped)
        first = next(iter(swapped))
        _map_suggestion(
            advanced,
            "A-14",
            column,
            stages,
            swapped,
            f"Rewrite {n(cells)} month-first dates of {column.name}",
            f"{n(cells)} dates of {column.name} can only be read month first ({q(first)}), "
            f"while the column is written {formats[0]}: from another source? Rewritten in the "
            f"column's order ({q(first)} → {q(swapped[first])}), B-11 then reads them instead of "
            "leaving them empty. A date such as 03/04/2024 cannot be told apart and keeps the "
            "column's order.",
            "date:order",
        )
    if short and years:
        middle = sorted(years)[len(years) // 2]
        mapping: dict[str, str] = {}
        for value, (read, target) in short.items():
            yy = read.year % 100
            year = min((1900 + yy, 2000 + yy), key=lambda y: abs(y - middle))
            mapping[value] = read.replace(year=year).strftime(target)
        cells = sum(rows.get(v, 0) for v in mapping)
        first = next(iter(mapping))
        _map_suggestion(
            advanced,
            "A-14",
            column,
            stages,
            mapping,
            f"Write the century of {n(cells)} dates of {column.name}",
            f"{n(cells)} dates of {column.name} have a two-digit year ({q(first)}); the other "
            f"dates center on {middle}: written with the century closest to it "
            f"({q(first)} → {q(mapping[first])}), B-11 then reads them. Check dates far in the "
            "past, such as birth dates.",
            "date:century",
        )


def _time_zones(advanced: Advanced, column: Column, stages: TextStages) -> None:
    """ISO times with mixed zones ('…+02:00', '…Z'): each one written in UTC."""
    values = [(v, c) for v, c in stages.final if v.strip()]
    total = sum(c for _, c in values)
    zoned = [(v, c) for v, c in values if ZONE.match(v.strip())]
    zoned_rows = sum(c for _, c in zoned)
    if not total or zoned_rows < ZONE_SHARE * total:
        return
    offsets: set[Any] = set()
    mapping: dict[str, str] = {}
    for value, _count in zoned:
        try:
            read = datetime.fromisoformat(value.strip().replace("z", "Z"))
        except ValueError:
            continue
        offsets.add(read.utcoffset())
        mapping[value] = read.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S")
    naive = sum(c for v, c in values if NAIVE.match(v.strip()))
    if not mapping or (len(offsets) < 2 and not naive):
        return
    cells = sum(c for v, c in zoned if v in mapping)
    first = next(iter(mapping))
    reason = (
        f"{n(cells)} times of {column.name} carry {len(offsets)} different time zones "
        f"({q(first)}): written in UTC ({q(first)} → {q(mapping[first])}) they compare and sort "
        "as one clock. Then read the column as dates (Add step › Change type)."
    )
    if naive:
        reason += f" {n(naive)} values have no zone: they are left as they are."
    if len(mapping) > m.MAX_MAP:
        advanced.add(
            "A-14",
            "I",
            [column],
            f"Mixed time zones in {column.name}",
            reason + " Too many distinct values for one step: information only.",
            cells=cells,
            examples=list(mapping.items())[:3],
            key="zone",
        )
        return
    _map_suggestion(
        advanced,
        "A-14",
        column,
        stages,
        mapping,
        f"Write {n(cells)} times of {column.name} in UTC",
        reason,
        "zone",
    )
