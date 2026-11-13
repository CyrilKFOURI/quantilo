"""A-20 (audit 2026-11-12, N36): rows that describe one entity (a customer, a company, a product)
written several ways, numbered instead of merged.

Splink's and dedupe's way, in plain code: a *record* is a name (case, accents and spaces folded)
with an *anchor*, a column that tells the entity apart (its postcode, e-mail, phone or key); the
records are blocked on the anchor, every pair of a block compared on its name at four levels
(the same once punctuation, spacing, legal forms and word order are set aside; one slip; two
slips on a long name; different), and scored by Fellegi–Sunter: *u* (how often two records
taken at random agree at that level, and share their anchor) read on the data, *m* coded. Pairs
at a probability ≥ 0.95 are chained into clusters (transitive closure, at most 12 records each).
The step is a new column `<name>_entity`, one number per entity: the names stay as they are,
nothing is merged or removed (Informatica's and Talend's survivorship left to the user), and the
clusters are reviewed one by one (N35's panel: untick a cluster, split one). Never ticked (C-17).
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from quantilo import semantics
from quantilo.cleaning import model as m
from quantilo.cleaning import phonetic as ph
from quantilo.cleaning.compile import as_text, entity_anchor_sql, entity_name_sql, is_text
from quantilo.cleaning.rules import basic
from quantilo.cleaning.rules.basic import Column, n, q

if TYPE_CHECKING:
    from quantilo.cleaning.rules.advanced import Advanced

NAME_COLUMNS = 6  # name columns tried at most
ANCHORS = 2  # anchors tried per name column, the first giving clusters kept
MIN_ROWS = 20
MIN_DISTINCT = 10
ANCHOR_DISTINCT = 20  # an anchor with fewer values is a category, not what tells entities apart
ANCHOR_SHARED = 0.9  # … and its values are shared by rows (an order number names one row)
RECORDS = 60_000  # records (a name and its anchor) read at most
CLOSE_BLOCK = 60  # records of one anchor compared two by two for two slips (larger: keys only)
RANDOM_PAIRS = 20_000
SEED = 97
CLUSTER = 12  # records of one cluster at most (more: a chain of close names, left out)
MATCH = 0.95
SHORT_WORD = 5  # two slips never in a word shorter than this
SLIP_LENGTH = 6  # a name of this many letters at least for one slip
CLOSE_LENGTH = 12  # … and for two
# Fellegi–Sunter: how often two records of one entity agree on the name at each level (coded,
# Splink's defaults for a name), and share their anchor
M_NAME = {3: 0.6, 2: 0.3, 1: 0.07, 0: 0.03}
M_ANCHOR = 0.95

# a name column names a party (customer_name, owner_s_business_name, companyName, product) or is
# a name alone (name, full_name, raison sociale); gis_nta_name (a place) or owner_s_house__ (a
# number) are not
PARTY_WORDS = {"customer", "client", "company", "business", "owner", "supplier", "vendor"}
PARTY_WORDS |= {"contact", "firm", "entreprise", "societe", "merchant", "seller", "employer"}
PARTY_WORDS |= {"organisation", "organization", "payee", "beneficiary", "product", "produit"}
PARTY_WORDS |= {"fournisseur", "titulaire", "proprietaire", "consignee", "recipient", "buyer"}
PARTY_WORDS |= {"destinataire", "acheteur", "person", "personne", "patient", "member"}
PARTY_WORDS |= {"employee", "permittee", "superintendent", "contractor", "landlord", "tenant"}
PARTY_WORDS |= {"applicant", "licensee", "manufacturer", "brand"}
NAME_ALONE = {"name", "names", "nom", "full", "fullname", "raison", "sociale", "s"}
NAME_WORDS = {"name", "names", "nom", "fullname", "raison"}
COMPANY_WORDS = {"company", "business", "firm", "entreprise", "societe", "supplier", "vendor"}
COMPANY_WORDS |= {"merchant", "organisation", "organization", "employer", "fournisseur"}
COMPANY_WORDS |= {"raison", "agency", "bank"}
PRODUCT_WORDS = {"product", "produit", "item", "article"}
CUSTOMER_WORDS = {"customer", "client", "buyer", "acheteur"}
FIRST_WORDS = {"first", "given", "prenom", "middle", "forename"}
LAST_WORDS = {"last", "family", "surname"}
# words of a column name that say what it is, not whose (left out when matching a name column to
# its anchor: owner_s_business_name goes with owner_s_zip_code, not permittee_s_phone__)
GENERIC_WORDS = {"name", "names", "nom", "s", "business", "company", "code", "zip", "postal"}
GENERIC_WORDS |= {"postcode", "zipcode", "phone", "telephone", "tel", "mobile", "email", "mail"}
GENERIC_WORDS |= {"e", "id", "key", "number", "no", "nbr", "num", "first", "last", "full"}
GENERIC_WORDS |= {"cp", "address", "contact", "of", "the", "de", "du"}
ANCHOR_ROLES = {"customer_key": "key", "email": "email", "phone": "phone"}
ANCHOR_ROLES |= {"postal_code": "postal_code"}
ADDRESS_WORDS = {"house", "street", "building", "bldg", "apt", "suite", "unit", "floor", "lot"}
ADDRESS_WORDS |= {"block", "bin", "parcel", "parcelle", "room"}
KIND_ORDER = {"key": 0, "email": 1, "phone": 2, "postal_code": 3, "code": 4}
NOT_NAMES = {"city", "region", "country", "address", "status", "free_text", "url", "email"}
NOT_NAMES |= {"phone", "postal_code", "gender", "currency"}

# legal forms (companies): set aside to compare names, but two different ones (LP, LTD) are two
# companies
LEGAL = {
    "inc": "inc", "incorporated": "inc", "corp": "corp", "corporation": "corp", "co": "co",
    "company": "co", "llc": "llc", "ltd": "ltd", "limited": "ltd", "lp": "lp", "llp": "llp",
    "plc": "plc", "gmbh": "gmbh", "ag": "ag", "sa": "sa", "sas": "sas", "sarl": "sarl",
    "eurl": "eurl", "bv": "bv", "nv": "nv", "pllc": "pllc", "pc": "pc",
}  # fmt: skip
SMALL_WORDS = {"the", "and", "et", "und", "y"}
# person names: an ending that makes another name (Paul / Paula, Louis / Louise, Jean / Jeanne,
# William / Williams), never a slip
NAME_ENDINGS = ("a", "e", "o", "s", "ne", "ie", "y", "ette", "ine")
VOWELS = set("aeiouy")
OPPOSITES = {"north", "south", "east", "west", "upper", "lower", "new", "old", "nord", "sud"}
OPPOSITES |= {"est", "ouest", "i", "ii", "iii", "iv", "v", "vi", "jr", "sr", "junior", "senior"}
OPPOSITES |= {"northeast", "northwest", "southeast", "southwest", "inner", "outer", "haut", "bas"}
NULL_TOKENS = basic.SURE_TOKENS | basic.WEAK_TOKENS | {"na", "nd", "nc"}
WORD = re.compile(r"[^\W_]+")
DIGITS = re.compile(r"\d+")


@dataclass
class Record:
    anchor: str
    key: str  # the folded name (`entity_name_sql`)
    display: str  # its most frequent spelling
    rows: int
    first: int  # its first row in the file
    compact: str = ""  # letters and digits only, legal forms and small words set aside
    ordered: str = ""  # … with its words in order (companies and products)
    legal: frozenset[str] = frozenset()
    digits: tuple[str, ...] = ()
    words: tuple[str, ...] = ()
    grams: frozenset[str] = frozenset()  # pairs of letters of `compact`


@dataclass
class Found:
    column: Column
    anchor: Column
    kind: str
    clusters: list[list[Record]]
    probabilities: list[float]
    left_out: int  # clusters too large
    records: int
    noun: str
    levels: set[int] = field(default_factory=set)


def _role(column: Column) -> str | None:
    view = column.view
    if view is not None and view.role:
        return view.role
    return column.role.id if column.role is not None else None


def _words(column: Column) -> set[str]:
    return set(semantics.tokens(column.name))


def kind_of(column: Column) -> str:
    """How the name is compared: a person's words in order, a company's or a product's in any."""
    words = _words(column)
    if words & (COMPANY_WORDS | PRODUCT_WORDS):
        return "company"
    return "person"


def _noun(column: Column) -> str:
    words = _words(column)
    if words & CUSTOMER_WORDS:
        return "customers"
    if words & PRODUCT_WORDS:
        return "products"
    if words & COMPANY_WORDS:
        return "companies"
    return "names"


def _name_columns(advanced: Advanced) -> list[Column]:
    from quantilo.cleaning.rules.advanced import _trivial

    context = advanced.context
    found = []
    for column in context.columns:
        words = _words(column)
        if not is_text(column.dtype):
            continue
        party = words & PARTY_WORDS and (words & NAME_WORDS or words <= PARTY_WORDS | {"s"})
        alone = words & NAME_WORDS and words <= NAME_ALONE | FIRST_WORDS | LAST_WORDS
        if not (party or alone):
            continue
        if words & FIRST_WORDS and not words & LAST_WORDS:
            continue  # a part of a name (owner_s_first_name): too short to tell people apart
        if words & LAST_WORDS and not words & FIRST_WORDS:
            continue
        if _role(column) in NOT_NAMES or _trivial(column, context.rows):
            continue
        if column.nonnull < MIN_ROWS or column.distinct < MIN_DISTINCT:
            continue
        found.append(column)
    return sorted(found, key=lambda c: -c.distinct)[:NAME_COLUMNS]


def _anchor_kind(column: Column) -> str | None:
    role = _role(column)
    if role in ANCHOR_ROLES:
        return ANCHOR_ROLES[role]
    if role == "address" or _words(column) & ADDRESS_WORDS:
        return None  # a house number names a place, many entities' (owner_s_house__)
    view = column.view
    code = semantics.code_by_name(column.name)
    if (view is not None and view.type == "identifier") or (
        code is not None and code.kind == "identifier"
    ):
        return "code"
    return None


def _anchors(advanced: Advanced, name: Column) -> list[tuple[Column, str]]:
    """The columns that can tell the entities of `name` apart, best first: one of the same party
    (owner's name, owner's postcode), then a key, an e-mail, a phone, a postcode; a product only
    by a key (its code), never by a place."""
    own = _words(name) - GENERIC_WORDS
    product = bool(_words(name) & PRODUCT_WORDS)
    ranked = []
    for column in advanced.context.columns:
        if column is name:
            continue
        kind = _anchor_kind(column)
        if kind is None or column.distinct < ANCHOR_DISTINCT:
            continue
        if product and kind not in ("key", "code"):
            continue
        if column.distinct > ANCHOR_SHARED * max(column.nonnull, 1):
            continue
        if column.nonnull < 0.5 * name.nonnull:
            continue
        party = _words(column) - GENERIC_WORDS
        if party and own and not party & own:
            continue  # another party's (the permittee's phone for the owner's name)
        ranked.append((0 if party & own else 1, KIND_ORDER[kind], column.position, column, kind))
    ranked.sort(key=lambda r: r[:3])
    return [(column, "key" if kind == "code" else kind) for *_, column, kind in ranked][:ANCHORS]


# --- Names compared ------------------------------------------------------------------------------


def _words_of(key: str) -> list[str]:
    """The words of a folded name, its initials joined (N.Y.C. → nyc, H P D → hpd)."""
    joined: list[str] = []
    initials = False  # the last word is initials being joined
    for word in WORD.findall(key):
        if len(word) == 1 and word.isalpha():
            if joined and initials:
                joined[-1] += word
            else:
                joined.append(word)
            initials = True
        else:
            joined.append(word)
            initials = False
    return joined


def _prepare(record: Record, company: bool) -> None:
    words = _words_of(record.key)
    legal = {LEGAL[w] for w in words if w in LEGAL} if company else set()
    core = [w for w in words if w not in SMALL_WORDS and not (company and w in LEGAL)]
    record.words = tuple(core)
    record.compact = "".join(core)
    record.ordered = "".join(sorted(core)) if company else record.compact
    record.legal = frozenset(legal)
    record.digits = tuple(DIGITS.findall(record.key))
    record.grams = frozenset(record.compact[k : k + 2] for k in range(len(record.compact) - 1))


def _ending(a: str, b: str) -> bool:
    """Paul / Paula, Louis / Louise, Mario / Maria: one is the other with another ending."""
    short, long = sorted((a, b), key=len)
    if long.startswith(short) and long[len(short) :] in NAME_ENDINGS and len(short) >= 3:
        return True
    return len(a) == len(b) >= 4 and a[:-1] == b[:-1] and a[-1] in VOWELS and b[-1] in VOWELS


def _other_word(a: Record, b: Record, company: bool) -> bool:
    """Two names one word apart, that word another one, not a slip: initials or an acronym
    (A.I.G. / A.E.G., NYC / NYU), an opposite (north / south, I / II), a person's other name
    (Paul / Paula, William / Williams)."""
    if len(a.words) != len(b.words):
        return False
    differ = [(x, y) for x, y in zip(a.words, b.words, strict=True) if x != y]
    if len(differ) != 1:
        return False
    x, y = differ[0]
    if len(x) <= 3 and len(y) <= 3:
        return True
    if x in OPPOSITES and y in OPPOSITES:
        return True
    return not company and _ending(x, y)


def level(a: Record, b: Record, company: bool) -> int:
    """3 the same once punctuation, spacing, legal forms and word order are set aside; 2 one
    slip; 1 two slips on a long name; 0 different (another number, another legal form, another
    first name)."""
    if not a.compact or not b.compact or a.digits != b.digits:
        return 0
    if company and a.legal and b.legal and not a.legal & b.legal:
        return 0
    if a.compact == b.compact or (company and a.ordered == b.ordered):
        return 3
    if _other_word(a, b, company):
        return 0
    shortest = min(len(a.compact), len(b.compact))
    if shortest < SLIP_LENGTH or abs(len(a.compact) - len(b.compact)) > 2:
        return 0
    # two slips change 6 pairs of letters at most: a cheap test before the distance
    if len(a.grams & b.grams) < max(len(a.grams), len(b.grams)) - 6:
        return 0
    distance = ph.damerau(a.compact, b.compact, 2)
    if distance == 1:
        return 2
    if distance == 2 and shortest >= CLOSE_LENGTH and not _short_change(a, b):
        return 1
    return 0


def _short_change(a: Record, b: Record) -> bool:
    """Two slips that touch a short word (A.T.A. / AIA): an acronym changed, not a typo."""
    if len(a.words) != len(b.words):
        return False
    return any(
        x != y and min(len(x), len(y)) < SHORT_WORD for x, y in zip(a.words, b.words, strict=True)
    )


def _candidates(block: list[Record], close: bool) -> set[tuple[int, int]]:
    """Pairs of a block that may be one slip apart (a deletion index) or the same; every pair when
    two slips are looked for in a small block."""
    if close and len(block) <= CLOSE_BLOCK:
        return {(i, j) for i in range(len(block)) for j in range(i + 1, len(block))}
    index: dict[str, set[int]] = defaultdict(set)
    for at, record in enumerate(block):
        keys = {record.compact, record.ordered}
        if len(record.compact) >= SLIP_LENGTH - 1:
            keys |= {
                record.compact[:k] + record.compact[k + 1 :] for k in range(len(record.compact))
            }
        for key in keys:
            index[key].add(at)
    pairs: set[tuple[int, int]] = set()
    for members in index.values():
        if 1 < len(members) <= CLOSE_BLOCK:
            ordered = sorted(members)
            pairs.update(
                (ordered[i], ordered[j])
                for i in range(len(ordered))
                for j in range(i + 1, len(ordered))
            )
    return pairs


def _clusters(pairs: list[tuple[int, int]], size: int) -> list[list[int]]:
    parent = list(range(size))

    def find(at: int) -> int:
        while parent[at] != at:
            parent[at] = parent[parent[at]]
            at = parent[at]
        return at

    for a, b in pairs:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)
    groups: dict[int, list[int]] = defaultdict(list)
    for at in {x for pair in pairs for x in pair}:
        groups[find(at)].append(at)
    return [sorted(g) for g in groups.values()]


# --- The rule ------------------------------------------------------------------------------------


def _records(advanced: Advanced, name: Column, anchor: Column, kind: str) -> list[Record]:
    from quantilo.cleaning.rules.advanced import ROW, SAMPLE

    key = entity_name_sql(as_text(name.name, name.dtype))
    tied = entity_anchor_sql(as_text(anchor.name, anchor.dtype), kind)
    raw = as_text(name.name, name.dtype)
    found = advanced.context.connection.execute(
        f"SELECT a, k, mode(v), count(*), min(r) FROM (SELECT {tied} AS a, {key} AS k, "
        f"{raw} AS v, {ROW} AS r FROM {SAMPLE}) WHERE a IS NOT NULL AND k IS NOT NULL "
        f"GROUP BY a, k ORDER BY a, k LIMIT {RECORDS + 1}"
    ).fetchall()
    return [Record(str(a), str(k), str(v), int(c), int(r)) for a, k, v, c, r in found]


def _odds(records: list[Record], company: bool) -> tuple[float, dict[int, float]]:
    """Fellegi–Sunter: the prior odds of a pair (one in as many as records) times the weight of
    sharing an anchor (`M_ANCHOR` / how often two records taken at random do), and the weight of
    each name level (`M_NAME` / how often two records taken at random agree at it)."""
    count = len(records)
    shares = np.array(list(_counts(records).values()), dtype=float) / count
    u_anchor = float((shares**2).sum())
    rng = np.random.default_rng(SEED)
    left = rng.integers(0, count, RANDOM_PAIRS)
    right = rng.integers(0, count, RANDOM_PAIRS)
    seen = np.zeros(4)
    for a, b in zip(left.tolist(), right.tolist(), strict=True):
        if a != b:
            seen[level(records[a], records[b], company)] += 1
    u_name = (seen + 1) / (seen.sum() + 4)
    prior = 1 / max(count - 1, 1)
    base = prior / (1 - prior) * M_ANCHOR / max(u_anchor, 1e-12)
    return base, {lv: M_NAME[lv] / float(u_name[lv]) for lv in range(4)}


def _counts(records: list[Record]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        counts[record.anchor] += 1
    return counts


def _find(advanced: Advanced, name: Column, anchor: Column, kind: str) -> Found | None:
    records = _records(advanced, name, anchor, kind)
    if len(records) < 2 or len(records) > RECORDS:
        return None
    company = kind_of(name) == "company"
    for record in records:
        _prepare(record, company)
    blocks: dict[str, list[int]] = defaultdict(list)
    for at, record in enumerate(records):
        if record.compact and record.key not in NULL_TOKENS:  # 'n/a' names no one
            blocks[record.anchor].append(at)
    # the pairs of a block whose names are close, before any weight (most files have none)
    levels: dict[tuple[int, int], int] = {}
    for members in blocks.values():
        if len(members) < 2:
            continue
        block = [records[at] for at in members]
        for i, j in _candidates(block, True):
            found = level(block[i], block[j], company)
            if found:
                levels[(members[i], members[j])] = found
    if not levels:
        return None
    base, weights = _odds(records, company)
    probability = {lv: base * w / (1 + base * w) for lv, w in weights.items()}
    passing = {lv for lv in (1, 2, 3) if probability[lv] >= MATCH}
    pairs = [pair for pair, found in levels.items() if found in passing]
    chances = {pair: probability[levels[pair]] for pair in pairs}
    if not pairs:
        return None
    clusters, left_out = [], 0
    lowest: list[float] = []
    for group in _clusters(pairs, len(records)):
        if len(group) > CLUSTER:
            left_out += 1
            continue
        members = set(group)
        clusters.append(sorted((records[at] for at in group), key=lambda r: (-r.rows, r.first)))
        lowest.append(min(p for pair, p in chances.items() if pair[0] in members))
    if not clusters:
        return None
    order = sorted(range(len(clusters)), key=lambda i: clusters[i][0].first)
    return Found(
        name,
        anchor,
        kind,
        [clusters[i] for i in order],
        [lowest[i] for i in order],
        left_out,
        len(records),
        _noun(name),
        {lv for lv in passing},
    )


def entities(advanced: Advanced) -> None:
    for name in _name_columns(advanced):
        for anchor, kind in _anchors(advanced, name):
            if advanced.out_of_time():
                return
            found = _find(advanced, name, anchor, kind)
            if found is not None:
                _suggest(advanced, found)
                break


def _target(advanced: Advanced, name: Column) -> str:
    taken = {c.name for c in advanced.context.columns}
    target, extra = f"{name.name}_entity", 2
    while target in taken:
        target, extra = f"{name.name}_entity_{extra}", extra + 1
    return target


def _suggest(advanced: Advanced, found: Found) -> None:
    clusters, kept = [], 0
    for cluster in sorted(found.clusters, key=lambda c: -sum(r.rows for r in c)):
        if kept + len(cluster) > m.MAX_ENTITY_RECORDS:
            break
        clusters.append(cluster)
        kept += len(cluster)
    clusters.sort(key=lambda c: min(r.first for r in c))
    name, anchor = found.column, found.anchor
    to = _target(advanced, name)
    rows = sum(r.rows for cluster in clusters for r in cluster)
    widest = max(len(cluster) for cluster in clusters)
    ways = "2" if widest == 2 else f"2 to {widest}"
    chance = min(found.probabilities)
    company = kind_of(name) == "company"
    how = ["spacing, punctuation, case or accents"]
    if company:
        how.append("a legal form (Inc., LLC) or the order of the words")
    if found.levels & {1, 2}:
        how.append(
            "one slip on a name of 6 letters or more"
            + (" (two on a name of 12 or more)" if 1 in found.levels else "")
        )
    never = (
        "another number, another legal form, other initials or an opposite (north / south)"
        if company
        else "another number, other initials or another first name (Paul / Paula)"
    )
    reason = (
        f"{n(len(clusters))} {found.noun} are written {ways} ways with the same {anchor.name} "
        f"({n(rows)} rows): an entity number groups them in a new column {to}; the names are not "
        f"changed. Two spellings are one entity when they differ only by {', '.join(how[:-1])}"
        + (f" or {how[-1]}" if len(how) > 1 else how[0])
        + f", with a probability of {'over 99.9 %' if chance > 0.999 else f'at least {chance:.1%}'}"
        f" (Fellegi–Sunter: sharing {anchor.name}, weighted by how often two of the "
        f"{n(found.records)} name and {anchor.name} pairs taken at random do; {never} is never a "
        f"slip). Spellings are chained into clusters of {CLUSTER} at most"
        + (f" ({n(found.left_out)} longer chains left out)" if found.left_out else "")
        + ". Review the clusters before ticking: two neighbours can have close names."
    )
    if advanced.sampled:
        reason += f" Found on a sample of {n(advanced.sample_rows)} rows."
    reason += " Check your data."
    step = m.EntityColumn(
        id="s0",
        column=name.name,
        anchor=anchor.name,
        anchor_kind=found.kind,  # type: ignore[arg-type]
        to=to,
        groups=[[m.EntityRecord(anchor=r.anchor, name=r.key) for r in c] for c in clusters],
    )
    examples = [
        (f"{q(c[1].display)} ({anchor.name} {q(c[0].anchor)})", f"same entity as {q(c[0].display)}")
        for c in clusters
    ]
    advanced.add(
        "A-20",
        "S",
        [name],
        f"Number the {n(len(clusters))} {found.noun} written several ways (new column {to})",
        reason,
        step,
        rows=rows,
        examples=examples,
        key="entity",
    )
