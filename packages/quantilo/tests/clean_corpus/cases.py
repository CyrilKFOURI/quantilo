"""The committed cleaning corpus (N01): small synthetic files, written in code, that reproduce what
the audit of 2026-11-12 found on real files (§ 2.2), each with its ground truth.

A truth says what a rule must propose (`correct`: right if applied), must not propose (`wrong`:
the false positive the audit saw) or must leave to the user (`decision`: proposed in mode S, not
counted in the precision). `after` names the milestone that makes it true when it is not merged
yet: the test is then an expected failure that turns into a pass once the milestone lands.
Every suggestion the catalogue marks `auto` must be judged by a truth of its case (closed world).
"""

from __future__ import annotations

import csv
import io
import json
import random
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

SAMPLES = Path(__file__).resolve().parents[4] / "samples"

Verdict = Literal["correct", "wrong", "decision"]


@dataclass(frozen=True)
class Truth:
    rule: str
    column: str | None  # None: a rule on rows ("B-14:rows")
    verdict: Verdict
    why: str
    sub: str | None = None  # the catalogue entry it must fall under ("B-04:file")
    mode: str | None = None  # the mode it must have when found
    auto: bool | None = None  # whether the catalogue must mark it automatic
    after: str | None = None  # the milestone that makes it true (xfail until merged)
    merges: tuple[tuple[str, str], ...] = ()  # map rules: a pair the map puts together
    id: str | None = None  # an exact suggestion id instead of rule + column
    step: bool | None = None  # whether it carries a step (an information has none)
    op: str | None = None  # the op of its step (real files: what the audit judged)

    @property
    def label(self) -> str:
        what = self.id or f"{self.rule}:{self.column or 'rows'}"
        if self.sub and self.sub != self.rule:
            what += f"[{self.sub}]"
        if self.merges:
            what += "[" + "|".join(f"{a}~{b}" for a, b in self.merges) + "]"
        if self.mode:
            what += f"[{self.mode}]"
        if self.step is not None:
            what += "[step]" if self.step else "[no step]"
        if self.auto is not None:
            what += "[auto]" if self.auto else "[not auto]"
        return f"{what}:{self.verdict}"


@dataclass(frozen=True)
class Case:
    name: str
    about: str  # the real file and what the audit saw on it
    write: Callable[[Path], Path]  # writes the file into a folder, returns its path
    truth: tuple[Truth, ...]
    advanced: bool = False  # also compute the Advanced layer (slower)
    open_options: dict[str, object] = field(default_factory=dict)


def _csv(rows: Iterable[Sequence[object]], header: Sequence[str], delimiter: str = ",") -> str:
    out = io.StringIO()
    writer = csv.writer(out, delimiter=delimiter, lineterminator="\n")
    writer.writerow(header)
    for row in rows:
        writer.writerow(["" if v is None else v for v in row])
    return out.getvalue()


def _writer(name: str, text: Callable[[], str], suffix: str = ".csv") -> Callable[[Path], Path]:
    def write(folder: Path) -> Path:
        path = folder / f"{name}{suffix}"
        path.write_text(text(), encoding="utf-8", newline="")
        return path

    return write


def _copy(source: Path) -> Callable[[Path], Path]:
    def write(folder: Path) -> Path:
        return source

    return write


# --- The cases ---------------------------------------------------------------------------------


def _ames() -> str:
    """Ames: 'NA' means 'no garage' where Garage Area is 0 (the data's documentation)."""
    rng = random.Random(1)
    rows = []
    for i in range(300):
        garage = rng.random() >= 0.06
        basement = rng.random() >= 0.03
        rows.append(
            [
                i + 1,
                rng.randint(5_000, 20_000),
                rng.choice(["Attchd", "Detchd", "BuiltIn"]) if garage else "NA",
                rng.choice(["Fin", "RFn", "Unf"]) if garage else "NA",
                rng.randint(250, 900) if garage else 0,
                rng.randint(1, 3) if garage else 0,
                rng.choice(["Ex", "Gd", "TA", "Fa"]) if basement else "NA",
                rng.randint(400, 2_000) if basement else 0,
                rng.choice(["Grvl", "Pave"]) if rng.random() < 0.07 else "NA",
                rng.randint(100_000, 450_000),
            ]
        )
    header = [
        "Order",
        "Lot Area",
        "Garage Type",
        "Garage Finish",
        "Garage Area",
        "Garage Cars",
        "Bsmt Qual",
        "Total Bsmt SF",
        "Alley",
        "SalePrice",
    ]
    return _csv(rows, header, "\t")


def _na_names() -> str:
    """nyc_dob: 'N/A' and a bare 'NA' in business names, both for 'not available'."""
    rng = random.Random(2)
    companies = ["ACME BUILDERS", "BLUE ROCK LLC", "CITY CONSTRUCTION", "DELTA GROUP", "EAST CORP"]
    rows = []
    for i in range(200):
        r = rng.random()
        name = "N/A" if r < 0.12 else "NA" if r < 0.18 else rng.choice(companies) + f" {i % 37}"
        rows.append([f"P{100_000 + i}", name, rng.choice(["NB", "A1", "A2", "DM"])])
    return _csv(rows, ["permit_id", "owner_s_business_name", "job_type"])


def _adult() -> str:
    """uci_adult: '?' for 'missing' in 3 columns, values written after ', ' (a leading space)."""
    rng = random.Random(3)
    work = ["Private", "Self-emp-not-inc", "State-gov", "Local-gov", "Federal-gov"]
    jobs = ["Adm-clerical", "Exec-managerial", "Craft-repair", "Sales", "Tech-support"]
    lands = ["United-States", "Mexico", "Philippines", "Germany", "Canada"]
    rows: list[list[object]] = []
    for _ in range(400):
        missing = rng.random() < 0.06
        rows.append(
            [
                rng.randint(17, 80),
                "?" if missing else rng.choice(work),
                rng.choice(["Bachelors", "HS-grad", "Masters", "Some-college"]),
                "?" if missing or rng.random() < 0.01 else rng.choice(jobs),
                rng.choice(["Male", "Female"]),
                "?" if rng.random() < 0.02 else rng.choice(lands),
                rng.choice(["<=50K", ">50K"]),
            ]
        )
    rows += [list(r) for r in rows[:4]]  # identical rows, no identifier: may be two people
    header = ["age", "workclass", "education", "occupation", "sex", "native_country", "income"]
    return ", ".join(header) + "\n" + "".join(", ".join(str(v) for v in r) + "\n" for r in rows)


def _retail() -> str:
    """online_retail: '?' in one column only (stays a decision), 'missing' (sure), ids in '.0'."""
    rng = random.Random(4)
    items = ["WHITE HANGING HEART", "REGENCY CAKESTAND", "JUMBO BAG RED", "LUNCH BAG CARS"]
    rows = []
    for i in range(400):
        r = rng.random()
        description = "?" if r < 0.02 else "missing" if r < 0.03 else rng.choice(items)
        customer = f"{rng.randint(12_346, 18_287)}.0" if rng.random() > 0.05 else ""
        rows.append([536_365 + i // 4, description, rng.randint(1, 24), customer])
    return _csv(rows, ["Invoice", "Description", "Quantity", "Customer ID"])


def _air() -> str:
    """air_quality: -200 for 'no measure' in every sensor column; ';;;' rows at the end."""
    rng = random.Random(5)
    rows: list[list[object]] = []
    for i in range(400):
        out = rng.random() < 0.08
        hour = i % 24

        def value(low: float, high: float, digits: int = 1, out: bool = out) -> object:
            if out or rng.random() < 0.01:
                return -200
            return f"{rng.uniform(low, high):.{digits}f}".replace(".", ",")

        rows.append(
            [
                f"{10 + i // 24:02d}/03/2004",
                f"{hour:02d}.00.00",
                value(0.3, 9),
                value(500, 2_000, 0),
                value(20, 900, 0),
                value(10, 250, 0),
                value(2, 40),
                value(10, 90),
            ]
        )
    text = _csv(
        rows, ["Date", "Time", "CO(GT)", "PT08.S1(CO)", "NOx(GT)", "NO2(GT)", "T", "RH"], ";"
    )
    return text + ";;;;;;;\n" * 6


def _beach() -> str:
    """chicago_beach: -100000 for 'no measure' in 3 columns; an hourly timestamp label."""
    rng = random.Random(6)
    rows = []
    for i in range(300):
        broken = rng.random() < 0.07
        rows.append(
            [
                rng.choice(["Ohio Street Beach", "Calumet Beach", "Montrose Beach"]),
                f"2026-09-{1 + i // 24:02d}T{i % 24:02d}:00:00.000",
                -100000 if broken else round(rng.uniform(12, 26), 1),
                -100000 if broken else round(rng.uniform(0, 4), 2),
                -100000 if broken or rng.random() < 0.02 else round(rng.uniform(1, 9), 1),
                round(rng.uniform(4, 12), 1),
            ]
        )
    header = [
        "beach_name",
        "measurement_timestamp",
        "water_temperature",
        "turbidity",
        "wave_period",
        "battery_life",
    ]
    return _csv(rows, header)


def _income() -> str:
    """A single column with -999: one sentinel, one column: left to the user."""
    rng = random.Random(7)
    rows = []
    for i in range(300):
        score = -999 if rng.random() < 0.04 else rng.randint(0, 100)
        rows.append([i + 1, rng.randint(18, 70), score])
    return _csv(rows, ["respondent", "age", "score"])


def _collisions() -> str:
    """nyc_collisions: positions at (0, 0), the geocoders' default point."""
    rng = random.Random(8)
    rows = []
    for i in range(400):
        zero = rng.random() < 0.03
        latitude = 0 if zero else round(rng.uniform(40.55, 40.9), 6)
        longitude = 0 if zero else round(rng.uniform(-74.2, -73.75), 6)
        rows.append([4_000_000 + i, latitude, longitude, rng.randint(0, 3)])
    return _csv(rows, ["collision_id", "latitude", "longitude", "persons_injured"])


US_STATES = [("Massachusetts", "0"), ("New Jersey", "0"), ("Kentucky", "4"), ("Texas", "7")]
US_WEIGHTS = [3, 2, 50, 45]  # superstore: 4.4 % of codes in New England


def _superstore() -> str:
    """superstore: 4-digit postal codes of US states (MA, NJ) that lost their leading zero; a
    non-breaking space in some names."""
    rng = random.Random(9)
    rows = []
    for i in range(400):
        state, first = rng.choices(US_STATES, US_WEIGHTS)[0]
        code = int(first + f"{rng.randint(1_000, 9_999)}")
        name = rng.choice(["Claire Gute", "Darrin Van Huff", "Sean O'Donnell", "Brosina Hoffman"])
        if rng.random() < 0.03:
            name = name.replace(" ", "\u00a0", 1)
        rows.append([i + 1, f"CG-{12_520 + i}", name, "United States", state, code])
    return _csv(rows, ["Row ID", "Customer ID", "Customer Name", "Country", "State", "Postal Code"])


def _postcodes_europe() -> str:
    """Northwind-like: 4-digit codes of Belgium and Switzerland among French 5-digit ones: not
    zeros lost, their length by nature."""
    rng = random.Random(10)
    rows = []
    for i in range(300):
        r = rng.random()
        if r < 0.04:
            country, code = "Belgium", rng.choice([1000, 1180, 6000, 8000])
        elif r < 0.07:
            country, code = "Switzerland", rng.choice([1204, 3012, 8001])
        else:
            country, code = "France", rng.randint(10_000, 95_999)
        rows.append([f"C{i:04d}", country, code])
    return _csv(rows, ["customerID", "country", "postalCode"])


def _labels() -> str:
    """openfoodfacts: columns of lists, where a fingerprint of the words merges 'vegetarian' with
    'non-vegetarian'; the same labels in another order are one value."""
    rng = random.Random(11)
    lists = [
        "en:vegetarian,en:non-vegan",
        "en:vegan,en:vegetarian",
        "en:organic,en:eu-organic",
        "en:no-gluten,en:no-gmos",
        "en:palm-oil-free",
        "en:green-dot",
    ]
    rare = ["en:non-vegetarian,en:vegan", "en:no-gmos,en:no-gluten"]
    rows = []
    for i in range(400):
        value = rng.choice(rare) if rng.random() < 0.04 else rng.choice(lists)
        rows.append([f"00{30_000_000 + i}", rng.choice(["Kroger", "Dynasty", "Walmart"]), value])
    return _csv(rows, ["code", "brands", "labels_tags"])


def _mayors() -> str:
    """fr_rne_maires: distinct surnames that a 2-gram fingerprint merges ('LE ROUX' / 'LEROUX');
    a street column where it is right ('VANDAM ST' / 'VAN DAM ST')."""
    rng = random.Random(12)
    surnames = ["MARTIN", "BERNARD", "DUBOIS", "THOMAS", "ROBERT", "RICHARD", "PETIT", "DURAND"]
    streets = ["VAN DAM ST", "BROADWAY", "LINDEN BLVD", "OCEAN AVE", "ATLANTIC AVE"]
    rows = []
    for i in range(400):
        r = rng.random()
        name = "LE ROUX" if r < 0.12 else "LEROUX" if r < 0.15 else rng.choice(surnames)
        street = "VANDAM ST" if rng.random() < 0.02 else rng.choice(streets)
        rows.append([f"{i:05d}", name, rng.choice(["Jean", "Marie", "Claude"]), street])
    return _csv(rows, ["Code de la commune", "Nom de l'élu", "Prénom de l'élu", "on_street"])


def _boroughs() -> str:
    """nyc_dob: zip -> borough holds except Marble Hill (10463, in Manhattan) on several rows:
    a real, repeated exception, not an error. As in nyc_dob, those permits share their block
    (02215): the third column that explains the exception (N15)."""
    rng = random.Random(13)
    zips = {
        "10001": "MANHATTAN",
        "10002": "MANHATTAN",
        "10003": "MANHATTAN",
        "10451": "BRONX",
        "10452": "BRONX",
        "10453": "BRONX",
        "10463": "BRONX",
        "11201": "BROOKLYN",
        "11203": "BROOKLYN",
        "11204": "BROOKLYN",
        "11354": "QUEENS",
        "11355": "QUEENS",
        "10301": "STATEN ISLAND",
    }
    keys = list(zips)
    rows = []
    marble = 0
    for i in range(600):
        code = rng.choice(keys)
        borough = zips[code]
        block = f"{rng.randrange(1, 9000):05d}"
        if code == "10463" and marble < 4:
            marble += 1
            borough = "MANHATTAN"  # Marble Hill, on 4 rows
            block = "02215"
        rows.append([f"J{i:05d}", code, borough, block])
    return _csv(rows, ["job_id", "zip_code", "borough", "block"])


def _amounts() -> str:
    """French accounting exports: '1 234,56 €' with thin spaces, negatives in brackets."""
    rng = random.Random(14)
    rows = []
    for i in range(300):
        value = rng.uniform(5, 20_000)
        whole, cents = f"{value:,.2f}".split(".")
        text = whole.replace(",", "\u202f") + "," + cents
        r = rng.random()
        if r < 0.1:
            text = f"({text})"
        elif r < 0.2:
            text = f"EUR {text}"
        else:
            text = f"{text} €"
        rows.append([f"F{i:05d}", text])
    return _csv(rows, ["facture", "montant_ttc"], ";")


def _orders() -> str:
    """Northwind orders: dates with milliseconds ('1998-04-10 00:00:00.000'), 'NULL' for none."""
    rng = random.Random(15)
    rows = []
    for i in range(300):
        day = rng.randint(1, 28)
        month = rng.randint(1, 12)
        shipped = (
            "NULL"
            if rng.random() < 0.03
            else f"1997-{month:02d}-{min(day + 3, 28):02d} 00:00:00.000"
        )
        region = "NULL" if rng.random() < 0.6 else rng.choice(["RJ", "SP", "WA", "OR"])
        rows.append(
            [
                10_248 + i,
                f"1997-{month:02d}-{day:02d} 00:00:00.000",
                shipped,
                region,
                rng.choice(["France", "Brazil", "USA", "Germany"]),
            ]
        )
    return _csv(rows, ["orderID", "orderDate", "shippedDate", "shipRegion", "shipCountry"])


def _socrata() -> str:
    """Socrata exports: '01/05/2024 12:00:00 AM', and 'N/A' for none (without it, DuckDB's CSV
    reader stores the column as a TIMESTAMP itself)."""
    rng = random.Random(16)
    rows = []
    for i in range(300):
        hour = rng.randint(1, 12)
        created = (
            "N/A"
            if i % 50 == 7
            else f"{rng.randint(1, 12):02d}/{rng.randint(1, 28):02d}/2024 "
            f"{hour:02d}:{rng.randint(0, 59):02d}:00 {rng.choice(['AM', 'PM'])}"
        )
        rows.append([f"R{i:05d}", created])
    return _csv(rows, ["unique_key", "created_date"])


def _tokens_fr() -> str:
    """kiwi and the French probe: 'n/d' (sure), 'néant' and '—' (can be a real answer)."""
    rng = random.Random(17)
    rows = []
    for i in range(200):
        revenue = "n/d" if rng.random() < 0.05 else rng.randint(1, 900)
        staff = rng.choice(["néant", "—"]) if rng.random() < 0.06 else rng.randint(1, 50)
        sector = "n/d" if rng.random() < 0.05 else rng.choice(["Commerce", "Industrie", "Services"])
        rows.append([f"S{i:04d}", rng.choice(["Paris", "Lyon", "Nantes"]), sector, revenue, staff])
    return _csv(rows, ["siren", "ville", "secteur", "ca_millions", "effectif"], ";")


def _votes() -> str:
    """législatives 2024: percentages with a decimal comma and a sign ('69,55%')."""
    rng = random.Random(18)
    rows = []
    for i in range(107):
        rows.append(
            [
                f"{i + 1:02d}",
                f"{rng.uniform(55, 75):.2f}%".replace(".", ","),
                rng.randint(100_000, 900_000),
            ]
        )
    return _csv(rows, ["Code département", "% Votants", "Inscrits"], ";")


def _houses() -> str:
    """kc_house: dates as '20140623T000000'."""
    rng = random.Random(19)
    rows = []
    for i in range(300):
        rows.append(
            [
                7_129_300_520 + i,
                f"2014{rng.randint(5, 12):02d}{rng.randint(1, 28):02d}T000000",
                rng.randint(150_000, 900_000),
                rng.randint(1, 6),
            ]
        )
    return _csv(rows, ["id", "date", "price", "bedrooms"])


def _movies() -> str:
    """imdb: exact copies of a row, identifier included; a non-breaking space ending titles."""
    rng = random.Random(20)
    rows = []
    for i in range(250):
        rows.append(
            [
                f"http://www.imdb.com/title/tt{499_549 + i}/",
                f"Movie {i}\u00a0",
                rng.choice(["Color", "Black and White"]),
                rng.randint(1_000, 900_000),
            ]
        )
    rows += [list(r) for r in rows[10:14]]  # entered twice
    return _csv(rows, ["movie_imdb_link", "movie_title", "color", "num_voted_users"])


def _clean() -> str:
    """insurance: a clean file: no automatic fix."""
    rng = random.Random(21)
    rows = []
    for _ in range(300):
        rows.append(
            [
                rng.randint(18, 64),
                rng.choice(["male", "female"]),
                round(rng.uniform(16, 48), 2),
                rng.randint(0, 5),
                rng.choice(["southwest", "southeast", "northwest", "northeast"]),
                round(rng.uniform(1_100, 60_000), 2),
            ]
        )
    return _csv(rows, ["age", "sex", "bmi", "children", "region", "charges"])


def _ratings() -> str:
    """imdb: content_rating holds 'Not Rated' (116 rows) and 'Unrated' (62), one rating (N30)."""
    rng = random.Random(30)
    ratings = ["R"] * 40 + ["PG-13"] * 30 + ["PG"] * 15 + ["Not Rated"] * 6 + ["Unrated"] * 3
    ratings += ["G", "Approved", "TV-14"]
    rows = [[f"Movie {i}", rng.choice(ratings), rng.randint(1990, 2016)] for i in range(400)]
    return _csv(rows, ["movie_title", "content_rating", "title_year"])


def _sexes() -> str:
    """N30: 'H' / 'F' beside 'Homme' / 'Femme' (French), and a Spanish 'H' / 'M' column, where
    'M' is 'mujer': only the French one is a synonym group."""
    rng = random.Random(31)
    rows = []
    for i in range(300):
        french = rng.choice(["H", "F", "H", "F", "Homme", "Femme"])
        spanish = rng.choice(["H", "M"])
        rows.append([i + 1, french, spanish, rng.choice(["Standard Class", "First Class"])])
    return _csv(rows, ["id", "sexe", "sexo", "ship_mode"])


def _segments() -> str:
    """superstore: Segment, Ship Mode, Category and a clean 'Male' / 'Female': no synonym group."""
    rng = random.Random(32)
    rows = [
        [
            rng.choice(["Consumer", "Corporate", "Home Office"]),
            rng.choice(["Standard Class", "Second Class", "First Class", "Same Day"]),
            rng.choice(["Furniture", "Office Supplies", "Technology"]),
            rng.choice(["Male", "Female"]),
        ]
        for _ in range(300)
    ]
    return _csv(rows, ["Segment", "Ship Mode", "Category", "gender"])


def _twice(value: str) -> str:
    """N28: a value read wrong twice (UTF-8 → Windows-1252, saved, read so again)."""
    return value.encode("utf-8").decode("cp1252").encode("utf-8").decode("cp1252")


def _mojibake() -> str:
    """N28 (the audit's fr_baac 'NÂ°', and what an enterprise export does): product names read
    wrong twice ('CafÃƒÂ©'), UTF-8 read as Mac Roman ('Caf√©'), and comments holding '�'
    (U+FFFD: the character was lost before the file was saved)."""
    rng = random.Random(28)
    names = ["Café noir", "Crème brûlée", "Thé vert", "Pâte à tartiner", "Soupe"]
    rows = []
    for i in range(240):
        name = rng.choice(names)
        if i % 40 == 0:
            name = _twice(name if "é" in name else "Café noir")
        elif i % 40 == 1:
            name = "Thé glacé".encode().decode("mac_roman")
        comment = (
            "Livr" + chr(0xFFFD) + " en retard" if i % 50 == 3 else rng.choice(["ok", "à revoir"])
        )
        rows.append([i + 1, name, comment])
    return _csv(rows, ["ref", "produit", "commentaire"])


def _single_byte(name: str, encoding: str) -> Callable[[Path], Path]:
    """N28: a file written by an old Mac (Mac Roman) or a DOS tool (CP850)."""

    def write(folder: Path) -> Path:
        rows = [["ville", "habitants"]] + [
            [city, str(1_000 + i)]
            for i, city in enumerate(
                ["Orléans", "Besançon", "Nîmes", "Sète", "Évry", "Vénissieux"] * 20
            )
        ]
        path = folder / f"{name}.csv"
        text = "".join(",".join(row) + chr(10) for row in rows)
        path.write_bytes(text.encode(encoding))
        return path

    return write


def _emails() -> str:
    """French probe, extended (N26): domains in capitals, four mistyped domains, names before '@'
    in capitals that stay."""
    rng = random.Random(26)
    slips = ("gmial.com", "hotmial.com", "orange.con", "gmail.fr")
    rows = []
    for i in range(300):
        domain = rng.choice(("exemple.fr", "gmail.com", "hotmail.com", "orange.com"))
        name = f"client{i}"
        if i % 50 == 7:
            domain = slips[(i // 50) % 4]
        elif i % 40 == 3:
            domain = "EXEMPLE.FR"
        if i % 60 == 11:
            name = name.upper()
        rows.append([i + 1, f"{name}@{domain}", rng.randint(1, 9)])
    rows.append([301, "jean@exemple.fr", 1])
    rows.append([302, "JEAN@EXEMPLE.FR", 1])
    return _csv(rows, ["id_client", "email", "commandes"], ";")


def _countries_fr() -> str:
    """French probe (N24): 'France', 'FR', 'FRA', 'france', 'Belgique', 'BE' in one column."""
    rng = random.Random(24)
    writings = ("France", "FR", "FRA", "france", "Belgique", "BE")
    rows = [[i + 1, rng.choice(writings), rng.randint(1, 500)] for i in range(300)]
    return _csv(rows, ["id_client", "pays", "montant"], ";")


def _countries_panel() -> str:
    """OWID (N24): countries beside groups ('World', 'High-income countries') and former countries,
    each written once: nothing to merge."""
    places = (
        "France",
        "Germany",
        "United States",
        "India",
        "Brazil",
        "World",
        "Europe",
        "High-income countries",
        "European Union (27)",
        "West Germany",
        "Kosovo",
        "Congo",
    )
    rows = [[place, 1990 + year, round(1.5 * year + i, 1)] for i, place in enumerate(places)
            for year in range(30)]  # fmt: skip
    return _csv(rows, ["country", "year", "co2"])


def _phones_fr() -> str:
    """French probe (N20): mobile numbers in four layouts, some with +33; no country column."""
    rng = random.Random(20)
    rows = [[1, "06 12 34 56 78"], [2, "+33 6 12 34 56 78"], [3, "06.12.34.56.78"]]
    for i in range(4, 301):
        digits = "06" + "".join(rng.choice("0123456789") for _ in range(8))
        pairs = [digits[k : k + 2] for k in range(0, 10, 2)]
        style = rng.random()
        if style < 0.5:
            phone = " ".join(pairs)
        elif style < 0.7:
            phone = digits
        elif style < 0.85:
            phone = "+33 " + digits[1] + " " + " ".join(pairs[1:])
        else:
            phone = ".".join(pairs)
        rows.append([i, phone])
    return _csv(rows, ["id_client", "telephone"], ";")


def _phones_us() -> str:
    """nyc_dob (N20): ten bare digits, one layout, no country: nothing to rewrite."""
    rng = random.Random(21)
    rows = [[f"P{i}", f"{rng.randint(200, 999)}{rng.randint(200, 999)}{rng.randint(0, 9999):04d}"]
            for i in range(300)]  # fmt: skip
    return _csv(rows, ["permit", "permittee_phone"])


def _quantities_fr() -> str:
    """French probe (N23): weights with their unit; Open Food Facts' '12 x 33 cl'."""
    rng = random.Random(23)
    weights = ("12 kg", "500 g", "1,5 kg", "750 g", "2 kg")
    packs = ("33 cl", "1 L", "12 x 33 cl", "500 ml", "250 g", "6 x 25 cl")
    rows = [[i + 1, rng.choice(weights), rng.choice(packs)] for i in range(300)]
    return _csv(rows, ["id_produit", "poids", "contenance"], ";")


def _packs() -> str:
    """Northwind (N23): 'quantityPerUnit' says how a product is packed, not one quantity."""
    rng = random.Random(24)
    packs = (
        "24 - 12 oz bottles",
        "10 boxes x 20 bags",
        "12 - 550 ml bottles",
        "5 kg pkg.",
        "48 - 6 oz jars",
        "36 boxes",
        "16 - 500 g tins",
        "1 kg pkg.",
    )
    rows = [[i + 1, rng.choice(packs), rng.randint(5, 80)] for i in range(77)]  # fmt: skip
    return _csv(rows, ["productID", "quantityPerUnit", "unitPrice"])


def _correct(rule: str, column: str | None, why: str, **more: object) -> Truth:
    return Truth(rule, column, "correct", why, **more)  # type: ignore[arg-type]


def _wrong(rule: str, column: str | None, why: str, **more: object) -> Truth:
    return Truth(rule, column, "wrong", why, **more)  # type: ignore[arg-type]


def _decision(rule: str, column: str | None, why: str, **more: object) -> Truth:
    return Truth(rule, column, "decision", why, **more)  # type: ignore[arg-type]


NA_NONE = "'NA' means 'none of it' here (Garage Area is 0): not a missing value"
AIR_200 = "-200 is the file's code for 'no measure' in every sensor column"
BEACH = "-100000 is the sensors' code for 'no measure'"
QUESTION = "'?' is the file's code for 'missing' (3 columns)"

CASES: tuple[Case, ...] = (
    Case(
        "ames_na",
        "ames: B-04 read 'NA' as missing in 9 Garage / Bsmt columns (1,025 cells), where it means "
        "'no garage' or 'no basement'",
        _writer("ames", _ames, ".tsv"),
        (
            _wrong("B-04", "Garage Type", NA_NONE, sub="B-04:na", after="N06"),
            _wrong("B-04", "Garage Finish", NA_NONE, sub="B-04:na", after="N06"),
            _wrong("B-04", "Bsmt Qual", NA_NONE, sub="B-04:na", after="N06"),
            _correct("B-04", "Garage Type", NA_NONE, sub="B-04:none", mode="S", after="N06"),
            _correct("B-04", "Garage Finish", NA_NONE, sub="B-04:none", mode="S", after="N06"),
            _correct("B-04", "Bsmt Qual", NA_NONE, sub="B-04:none", mode="S", after="N06"),
            _decision("B-04", "Alley", "'NA' is 'no alley' on most rows: the user decides"),
        ),
    ),
    Case(
        "synonyms_ratings",
        "imdb (N30): 'Unrated' and 'Not Rated' are one rating; merged only if ticked (A-21)",
        _writer("imdb_ratings", _ratings),
        (
            _decision(
                "A-21",
                "content_rating",
                "the same rating under two names",
                merges=(("Unrated", "Not Rated"),),
                mode="S",
            ),
        ),
        advanced=True,
    ),
    Case(
        "synonyms_sexes",
        "N30: 'H' / 'Homme' and 'F' / 'Femme' (French); a Spanish H / M left alone",
        _writer("personnes", _sexes),
        (
            _decision(
                "A-21",
                "sexe",
                "'Homme' is 'H', 'Femme' is 'F'",
                merges=(("Homme", "H"), ("Femme", "F")),
                mode="S",
            ),
            _wrong("A-21", "sexo", "'M' is 'mujer' beside 'H': not male"),
        ),
        advanced=True,
    ),
    Case(
        "synonyms_none",
        "superstore (N30): no business synonym in Segment, Ship Mode, Category, gender",
        _writer("superstore_segments", _segments),
        (_wrong("A-21", None, "no two values of one group in any column"),),
        advanced=True,
    ),
    Case(
        "encodings_wide",
        "N28: values read wrong twice, UTF-8 read as Mac Roman, characters lost ('�')",
        _writer("produits", _mojibake),
        (
            _correct(
                "B-05",
                "produit",
                "read back twice and from Mac Roman, checked value by value",
                sub="B-05:wide",
                auto=False,
                merges=((_twice("Café noir"), "Café noir"), ("Th√© glac√©", "Thé glacé")),
            ),
            _correct(
                "B-05",
                "commentaire",
                "'�' cannot be repaired: counted, nothing changed",
                sub="B-05:lost",
                mode="I",
                step=False,
            ),
        ),
    ),
    Case(
        "encoding_mac_roman",
        "N28: a Mac Roman file read as Mac Roman (R-ING-08), so nothing is left to repair",
        _single_byte("villes_mac", "mac_roman"),
        (_wrong("B-05", "ville", "the file is read in its own encoding: nothing to repair"),),
    ),
    Case(
        "encoding_cp850",
        "N28: a DOS (CP850) file read as CP850 (R-ING-08), so nothing is left to repair",
        _single_byte("villes_dos", "cp850"),
        (_wrong("B-05", "ville", "the file is read in its own encoding: nothing to repair"),),
    ),
    Case(
        "na_names",
        "nyc_dob: 'N/A' and a bare 'NA' in business names, both 'not available' (right)",
        _writer("nyc_dob", _na_names),
        (
            _correct(
                "B-04",
                None,
                "'N/A' (and, before N06, 'NA') is 'not available'",
                mode="A",
                id="B-04:owner_s_business_name",
            ),
            _correct(
                "B-04",
                None,
                "a bare 'NA' is 'not available' here too, left to the user since N06",
                mode="S",
                id="B-04:owner_s_business_name:maybe",
                after="N06",
            ),
        ),
    ),
    Case(
        "question_mark",
        "uci_adult: '?' in 3 columns (10 suggestions in mode S, all right, with diabetes); values "
        "with a leading space; 4 identical rows without an identifier",
        _writer("adult", _adult),
        (
            _correct(
                "B-04", "workclass", QUESTION, sub="B-04:file", mode="A", auto=True, after="N08"
            ),
            _correct(
                "B-04", "occupation", QUESTION, sub="B-04:file", mode="A", auto=True, after="N08"
            ),
            _correct(
                "B-04",
                "native_country",
                QUESTION,
                sub="B-04:file",
                mode="A",
                auto=True,
                after="N08",
            ),
            _correct("B-01", "workclass", "a leading space after ', '", auto=True),
            _correct("B-01", "education", "a leading space after ', '", auto=True),
            _correct("B-01", "occupation", "a leading space after ', '", auto=True),
            _correct("B-01", "sex", "a leading space after ', '", auto=True),
            _correct("B-01", "native_country", "a leading space after ', '", auto=True),
            _correct("B-01", "income", "a leading space after ', '", auto=True),
            _decision("B-14", None, "identical rows without an identifier can be two people"),
        ),
    ),
    Case(
        "question_mark_once",
        "online_retail: '?' in one column only stays a decision; 'missing' is sure; customer ids "
        "stored with '.0'",
        _writer("retail", _retail),
        (
            _decision(
                "B-04", "Description", "'?' in one column: maybe a real value", sub="B-04:weak"
            ),
            _correct(
                "B-04", "Description", "'missing' is a word for missing", sub="B-04:sure", auto=True
            ),
            _correct("B-12", "Customer ID", "'17669.0' -> '17669'", auto=True),
        ),
    ),
    Case(
        "sentinel_repeated",
        "air_quality: -200 in 13 sensor columns (all right, suggested one by one); 114 empty rows",
        _writer("air_quality", _air),
        (
            *(
                _correct("B-09", c, AIR_200, sub="B-09:repeated", mode="A", auto=True, after="N09")
                for c in ("CO(GT)", "PT08.S1(CO)", "NOx(GT)", "NO2(GT)", "T", "RH")
            ),
            _correct("B-15", None, "rows of ';' only at the end", auto=True),
        ),
    ),
    Case(
        "sentinel_beach",
        "chicago_beach: -100000 in 3 sensor columns (right)",
        _writer("chicago_beach", _beach),
        tuple(
            _correct("B-09", c, BEACH, sub="B-09:repeated", mode="A", auto=True, after="N09")
            for c in ("water_temperature", "turbidity", "wave_period")
        ),
    ),
    Case(
        "sentinel_once",
        "a single column with -999: a sentinel, left to the user",
        _writer("survey", _income),
        (_correct("B-09", "score", "-999 is a code for 'not given'", sub="B-09", mode="S"),),
    ),
    Case(
        "null_island",
        "nyc_collisions: 494 rows at (0, 0) (right, mode S)",
        _writer("collisions", _collisions),
        (
            _correct(
                "B-09",
                "latitude",
                "(0, 0) is the geocoders' default point",
                sub="B-09:zero",
                mode="S",
            ),
            _correct(
                "B-09",
                "longitude",
                "(0, 0) is the geocoders' default point",
                sub="B-09:zero",
                mode="S",
            ),
        ),
    ),
    Case(
        "postcodes_us",
        "superstore: 438 postal codes of MA, NJ... with 4 digits (right); non-breaking spaces",
        _writer("superstore", _superstore),
        (
            _correct(
                "B-13",
                "Postal Code",
                "US codes lost their leading zero",
                sub="B-13:postal",
                mode="A",
            ),
            _correct(
                "B-13",
                "Postal Code",
                "automatic once the country guard is in (N12)",
                sub="B-13:postal",
                auto=True,
                after="N12",
            ),
            _correct("B-01", "Customer Name", "a non-breaking space", auto=True),
            _decision("B-19", "Country", "always United States: the user decides"),
        ),
    ),
    Case(
        "postcodes_europe",
        "northwind: Belgian and Swiss codes have 4 digits by nature (the risk the audit names)",
        _writer("customers", _postcodes_europe),
        (
            _wrong(
                "B-13",
                "postalCode",
                "Belgian and Swiss codes are not padded",
                sub="B-13:postal",
                after="N12",
            ),
        ),
    ),
    Case(
        "list_columns",
        "openfoodfacts: A-01 merged 'en:vegetarian' with 'en:non-vegetarian' (13 of 22 wrong)",
        _writer("off", _labels),
        (
            _wrong(
                "A-01",
                "labels_tags",
                "a vegetarian product is not a non-vegetarian one",
                merges=(("en:non-vegetarian,en:vegan", "en:vegetarian,en:non-vegan"),),
                after="N13",
            ),
            _correct(
                "A-01",
                "labels_tags",
                "the same labels in another order",
                merges=(("en:no-gmos,en:no-gluten", "en:no-gluten,en:no-gmos"),),
            ),
        ),
        advanced=True,
    ),
    Case(
        "surnames",
        "fr_rne_maires: A-02 merged 'LE ROUX' into 'LEROUX' (55 groups of distinct surnames)",
        _writer("maires", _mayors),
        (
            _wrong("A-02", "Nom de l'élu", "two surnames, maybe two people", after="N14"),
            _correct(
                "A-02",
                "on_street",
                "'VANDAM ST' is 'VAN DAM ST'",
                merges=(("VANDAM ST", "VAN DAM ST"),),
            ),
        ),
        advanced=True,
    ),
    Case(
        "dependency_exceptions",
        "nyc_dob: A-10 '10463 -> MANHATTAN' is Marble Hill, a real exception on several rows",
        _writer("boroughs", _boroughs),
        (
            _wrong(
                "A-10",
                "zip_code",
                "Marble Hill is in Manhattan with a Bronx code: not to be changed",
                step=True,
                after="N15",
            ),
            _decision(
                "A-10",
                "zip_code",
                "a repeated exception, shown without a step",
                step=False,
                after="N15",
            ),
        ),
        advanced=True,
    ),
    Case(
        "amounts_fr",
        "French probe: '1 234,56 €', thin spaces, negatives in brackets (0 of 600 read)",
        _writer("factures", _amounts),
        (
            _correct(
                "B-07",
                "montant_ttc",
                "amounts stored as text, (1 234,56) is -1234.56",
                mode="A",
                auto=True,
                after="N18",
            ),
            _correct("B-01", "montant_ttc", "thin spaces written as plain spaces", auto=True),
        ),
    ),
    Case(
        "dates_ms",
        "northwind: 809 dates '1998-04-10 00:00:00.000' never read; 'NULL' for none",
        _writer("orders", _orders),
        (
            # orderDate has no 'NULL': DuckDB's CSV reader already stores it as a TIMESTAMP (as on
            # the real northwind), so there is nothing for B-11 to read
            _correct(
                "B-11", "shippedDate", "dates with milliseconds", mode="A", auto=True, after="N19"
            ),
            # N19: shippedDate is read as dates, so B-11 reads its 'NULL' as empty (B-04 leaves
            # the null tokens of a converted column to the conversion, `_converted`)
            _correct("B-04", "shipRegion", "'NULL' for none", sub="B-04:sure", auto=True),
        ),
    ),
    Case(
        "dates_ampm",
        "Socrata exports: '01/05/2024 12:00:00 AM'",
        _writer("requests", _socrata),
        (_correct("B-11", "created_date", "dates with AM / PM", mode="A", auto=True, after="N19"),),
    ),
    Case(
        "tokens_fr",
        "kiwi and the French probe: 'n/d' (8 rows), 'néant', '—'",
        _writer("societes", _tokens_fr),
        (
            _correct(
                "B-04",
                "secteur",
                "'n/d' is 'non disponible'",
                sub="B-04:sure",
                auto=True,
                after="N07",
            ),
            _decision("B-04", "effectif", "'néant' can be a real 'none'", after="N07"),
            _correct("B-07", "ca_millions", "numbers; 'n/d' becomes empty", mode="S"),
        ),
    ),
    Case(
        "percent_comma",
        "législatives 2024: '69,55%' (38 columns, right)",
        _writer("legislatives", _votes),
        (_correct("B-07", "% Votants", "numbers with a decimal comma", mode="A", auto=True),),
    ),
    Case(
        "dates_compact",
        "kc_house: '20140623T000000' (right)",
        _writer("kc_house", _houses),
        (_correct("B-11", "date", "compact ISO dates", mode="A", auto=True),),
    ),
    Case(
        "copies_with_id",
        "imdb: 45 exact copies, identifier included; a non-breaking space ending every title",
        _writer("imdb", _movies),
        (
            _correct("B-14", None, "the same movie entered twice", sub="B-14:id", auto=True),
            _correct("B-01", "movie_title", "a non-breaking space at the end", auto=True),
        ),
    ),
    Case(
        "emails_fr",
        "French probe, extended (N26): 4 mistyped domains, domains in capitals; no real file has "
        "enough e-mails",
        _writer("clients", _emails),
        (
            _correct(
                "A-04",
                "email",
                "gmial.com is gmail.com",
                id="A-04:email:domain",
                sub="A-04:domain",
                mode="S",
                merges=(("client7@gmial.com", "client7@gmail.com"),),
            ),
            _correct(
                "A-04",
                "email",
                "a domain in capitals, the name kept",
                id="A-04:email:case",
                merges=(("client3@EXEMPLE.FR", "client3@exemple.fr"),),
            ),
            _wrong(
                "A-04",
                "email",
                "the name before '@' keeps its case: JEAN and jean may be two mailboxes",
                id="A-04:email:case",
                merges=(("JEAN@EXEMPLE.FR", "jean@exemple.fr"),),
            ),
        ),
        advanced=True,
    ),
    Case(
        "countries_fr",
        "French probe (N24): six writings of France and Belgium in one column",
        _writer("clients_pays", _countries_fr),
        (
            _correct(
                "F-04",
                "pays",
                "FR, FRA and france are France; BE is Belgique",
                mode="S",
                merges=(("FR", "France"), ("FRA", "France"), ("BE", "Belgique")),
            ),
            _correct("B-03", "pays", "'france' is 'France' (case only)", auto=True),
        ),
        advanced=True,
    ),
    Case(
        "countries_panel",
        "owid: countries and groups (World, income groups), each written once",
        _writer("panel", _countries_panel),
        (_wrong("F-04", "country", "every country is written once; a group is no country"),),
        advanced=True,
    ),
    Case(
        "phones_fr",
        "French probe (N20): 600 mobile numbers in 4 layouts ('+33 6 …' has 11 digits and escapes "
        "A-04)",
        _writer("clients_tel", _phones_fr),
        (
            _correct(
                "F-01",
                "telephone",
                "the same number, in the international form",
                mode="S",
                merges=(
                    ("06 12 34 56 78", "+33 6 12 34 56 78"),
                    ("06.12.34.56.78", "06 12 34 56 78"),
                ),
            ),
        ),
        advanced=True,
    ),
    Case(
        "phones_us",
        "nyc_dob (N20): ten bare digits in one layout and no country: left as they are",
        _writer("permits", _phones_us),
        (_wrong("F-01", "permittee_phone", "one layout already; no country to read them with"),),
        advanced=True,
    ),
    Case(
        "quantities_fr",
        "French probe and Open Food Facts (N23): '12 kg', '1,5 kg', '12 x 33 cl'",
        _writer("produits", _quantities_fr),
        (
            _correct("F-05", "poids", "weights read in grams", mode="S", op="parse_quantity"),
            _correct("F-05", "contenance", "volumes in litres, '250 g' in grams", mode="S"),
        ),
        advanced=True,
    ),
    Case(
        "packs",
        "Northwind products: 'quantityPerUnit' is a packing, not a quantity",
        _writer("products", _packs),
        (_wrong("F-05", "quantityPerUnit", "'24 - 12 oz bottles' is no single quantity"),),
        advanced=True,
    ),
    Case(
        "clean_file",
        "insurance: a clean file: nothing automatic",
        _writer("insurance", _clean),
        (),
    ),
)


def _sample_truth(name: str, layer: str = "basic") -> tuple[Truth, ...]:
    expected = json.loads((SAMPLES / f"{name}.expected.json").read_text(encoding="utf-8"))
    items = expected["expected"] if "expected" in expected else expected[layer]
    return tuple(
        Truth(e["id"].split(":")[0], None, "correct", e["trap"], mode=e["mode"], id=e["id"])
        for e in items
    )


SAMPLE_CASES: tuple[Case, ...] = (
    Case(
        "dirty_traps",
        "samples/dirty_traps.csv (065): every Basic trap, 100 % recall",
        _copy(SAMPLES / "dirty_traps.csv"),
        _sample_truth("dirty_traps"),
    ),
    Case(
        "messy_orders",
        "samples/messy_orders.csv (097): its Basic traps",
        _copy(SAMPLES / "messy_orders.csv"),
        _sample_truth("messy_orders"),
    ),
)

# --- Audit milestones N10 to N34 (feat/clean-adv): years, booleans, text, columns ---------------


def _years() -> str:
    """kc_house: `yr_renovated` = 0 on most rows (never renovated); 0 bedrooms on 1 house; a water
    temperature above 40 °C (chicago_beach)."""
    rng = random.Random(31)
    rows = []
    for i in range(300):
        built = rng.randint(1900, 2015)
        renovated = rng.randint(max(built, 1950), 2015) if rng.random() < 0.1 else 0
        rows.append(
            [
                i + 1,
                built,
                renovated,
                0 if i == 17 else rng.randint(1, 6),
                0 if rng.random() < 0.05 else rng.randint(1, 4),
                rng.randint(0, 30),
                46.6 if i in (40, 41) else round(rng.uniform(8, 26), 1),
                rng.randint(150_000, 900_000),
            ]
        )
    header = [
        "id",
        "yr_built",
        "yr_renovated",
        "bedrooms",
        "bathrooms",
        "years_at_company",
        "water_temperature",
        "price",
    ]
    return _csv(rows, header)


def _built() -> str:
    """melbourne: YearBuilt = 1196 among 1830 to 2018 (1996 mistyped); owid: years from 1750."""
    rng = random.Random(32)
    rows = []
    for i in range(1_200):
        year = 1196 if i == 7 else rng.randint(1830, 2018)
        rows.append([i + 1, year, rng.randint(1750, 2023), rng.randint(300_000, 2_000_000)])
    return _csv(rows, ["id", "YearBuilt", "year", "Price"])


def _boxes() -> str:
    """French probe: a checkbox ('X' and blanks), 'Oui' / 'Non' with an 'X', 'V' / 'F'; nyc_dob:
    'YES' or blank (read as a boolean at import), 'Y' / 'N'."""
    rng = random.Random(33)
    rows = []
    for i in range(300):
        rows.append(
            [
                i + 1,
                "X" if rng.random() < 0.4 else "",
                rng.choice(["Oui", "Non", "oui", "X", ""]),
                rng.choice(["V", "F"]),
                "YES" if rng.random() < 0.35 else "",
                rng.choice(["Y", "N"]),
            ]
        )
    return _csv(rows, ["id", "newsletter", "abonne", "valide", "residential", "self_cert"])


def _remarks() -> str:
    """A column of comments in paragraphs, with doubled spaces (N29: B-01 keeps the breaks)."""
    rows = []
    for i in range(150):
        note = (
            f"  Called  back on day {i % 9}.\n\nWaiting  for the  part {i % 7}. " if i % 3 else ""
        )
        rows.append([i + 1, note, f"Store {i % 4}"])
    return _csv(rows, ["ticket", "comments", "site"])


def _requests() -> str:
    """nyc_311: closed_date at :00 seconds, a few seconds 'before' created_date on 9 rows; orders
    shipped a few days before they were ordered (swapped) and one two years before (wrong)."""
    import datetime as dt

    rng = random.Random(34)
    rows = []
    start = dt.datetime(2024, 1, 1)
    for i in range(400):
        created = start + dt.timedelta(minutes=rng.randint(0, 500_000), seconds=rng.randint(1, 59))
        closed = created.replace(second=0) if i < 9 else created + dt.timedelta(hours=5)
        ordered = (start + dt.timedelta(days=rng.randint(0, 300))).date()
        shipped = ordered + dt.timedelta(days=rng.randint(1, 9))
        if 10 <= i < 16:
            ordered, shipped = shipped, ordered
        if i == 20:
            shipped = ordered - dt.timedelta(days=700)
        rows.append(
            [i + 1, f"{created:%Y-%m-%d %H:%M:%S}", f"{closed:%Y-%m-%d %H:%M:%S}", ordered, shipped]
        )
    return _csv(rows, ["unique_key", "created_date", "closed_date", "order_date", "ship_date"])


def _counts() -> str:
    """législatives: '% Votants' beside counts (A-12 added them), small scores of 0.04 % (A-05
    rescaled them); fr_rne: départements '2A', '2B', '971' (A-04); diabetes: ages '[0-10)'."""
    rng = random.Random(35)
    rows = []
    departements = [f"{d:02d}" for d in range(1, 96) if d != 20] + ["2A", "2B", "971", "974"]
    ages = ["[50-60)", "[60-70)", "[70-80)", "[80-90)", "[0-10)", "[90-100)"]
    for i in range(400):
        voters = rng.randint(20_000, 900_000)
        registered = round(voters / rng.uniform(0.55, 0.75))
        abstentions = registered - voters
        if i == 7:
            registered += 200_000  # one wrong total
        # scores from 0.01 % to 80 %, the small ones (a fifth) going on into the others
        small = 10 ** rng.uniform(-2, -0.1) if rng.random() < 0.2 else 10 ** rng.uniform(0, 1.9)
        rows.append(
            [
                departements[i % len(departements)],
                registered,
                voters,
                f"{100 * voters / registered:.2f}",
                abstentions,
                f"{small:.2f}",
                ages[i % 4] if i % 30 else ages[4 + i % 2],
            ]
        )
    header = ["Code du département", "Inscrits", "Votants", "% Votants", "Abstentions"]
    return _csv(rows, [*header, "% Voix/exprimés 2", "age"])


def _hours() -> str:
    """air quality: readings of one date at two hours look alike (A-16); eco2mix: the last rows
    are forecasts with most cells still empty (A-17)."""
    rng = random.Random(36)
    rows = []
    for day in range(1, 61):
        for hour in range(0, 24, 3):
            co = round(rng.uniform(0.5, 4), 1)
            rows.append(
                [
                    f"2024-03-{day % 28 + 1:02d}",
                    f"{hour:02d}.00.00",
                    co,
                    round(co * 47),
                    round(co * 39),
                    12,
                    40,
                ]
            )
    for hour in range(24):  # forecasts: only the date, the hour and one column
        rows.append(["2024-04-01", f"{hour:02d}.00.00", None, None, None, None, 41])
    header = ["Date", "Time", "CO(GT)", "NOx(GT)", "NO2(GT)", "T", "prevision"]
    return _csv(rows, header, ";")


def _sales() -> str:
    """dvf: valeur_fonciere repeated on each line of its id_mutation (lots, parcels); superstore:
    Sales vary on the lines of one Order ID."""
    rng = random.Random(37)
    rows = []
    for sale in range(120):
        price = rng.randint(50_000, 900_000)
        for line in range(rng.randint(1, 6)):
            rows.append(
                [
                    f"2024-{sale + 1}",
                    price,
                    rng.randint(100, 9_000),
                    f"CA-{1000 + sale // 2}",
                    round(rng.uniform(5, 500), 2),
                    line + 1,
                ]
            )
    header = ["id_mutation", "valeur_fonciere", "surface_terrain", "Order ID", "Sales", "lot"]
    return _csv(rows, header)


def _twins() -> str:
    """adult: education and education-num one to one; chicago_beach: a timestamp and its text
    label; nyc_311: park_borough a copy of borough; superstore: Customer ID and Customer Name."""
    import datetime as dt

    rng = random.Random(38)
    levels = ["Bachelors", "HS-grad", "Masters", "Doctorate", "Some-college", "Assoc-voc"]
    boroughs = ["BRONX", "BROOKLYN", "MANHATTAN", "QUEENS", "STATEN ISLAND"]
    rows = []
    start = dt.datetime(2024, 5, 1)
    for _ in range(400):
        level = rng.randrange(len(levels))
        stamp = start + dt.timedelta(hours=rng.randint(0, 300))
        borough = rng.choice(boroughs)
        customer = rng.randint(1, 60)
        rows.append(
            [
                levels[level],
                [13, 9, 14, 16, 10, 11][level],
                f"{stamp:%Y-%m-%d %H:%M:%S}",
                f"{stamp:%m/%d/%Y %I:%M %p}",
                borough,
                borough,
                f"CU-{customer:04d}",
                f"Customer {customer}",
                rng.randint(1, 900),
            ]
        )
    header = ["education", "education-num", "measurement_timestamp", "measurement_timestamp_label"]
    header += ["borough", "park_borough", "Customer ID", "Customer Name", "amount"]
    return _csv(rows, header)


# owners written several ways (N36): the spelling a row uses, and the owner it is
OWNER_SPELLINGS = {
    "NYC HOUSING AUTHORITY": ["N.Y.C. HOUSING AUTHORITY", "NYC HOUSING AUTHORTY"],
    "VORNADO OFFICE MANAGEMENT": ["VORNADO OFFICE MANAGEMENT, LLC", "Vornado Office Managment"],
    "LE HAVRE OWNERS CORP.": ["LE HARVE OWNERS CORP."],
    "TRUSTEES OF COLUMBIA UNIVERSITY": ["TRUSTEE OF COLUMBIA UNIVERSITY"],
    "SL GREEN REALTY CORP": ["SL GREEN REALTY CORP."],
}
# … and owners alike that are two: another number, another legal form, other initials
OWNER_PAIRS = (
    ("509 MADISON AVENUE ASSOCIATES LP", "590 MADISON AVENUE ASSOCIATES LP"),
    ("1285 ASSOCIATES LP", "1285 ASSOCIATES LTD"),
    ("A.I.G. REALTY INC.", "A.E.G. REALTY INC."),
)


def _owners() -> str:
    """nyc_dob: owners written several ways with the same postcode (N36, A-20), owners alike that
    are two, neighbourhoods north and south of one postcode (gis_nta_name), people one first name
    apart (Paul / Paula Martin)."""
    rng = random.Random(36)
    zips = [f"10{n:03d}" for n in range(1, 41)]
    owners = [
        (f"OWNER {chr(65 + i // 26)}{chr(65 + i % 26)} REALTY CORP", zips[i % 40])
        for i in range(60)
    ]
    owners += [(name, zips[i]) for i, name in enumerate(OWNER_SPELLINGS)]
    owners += [(name, zips[10 + i]) for i, pair in enumerate(OWNER_PAIRS) for name in pair]
    places = ["BUSHWICK NORTH", "BUSHWICK SOUTH", "EAST HARLEM NORTH", "EAST HARLEM SOUTH"]
    people = ["Paul Martin", "Paula Martin", "Louis Petit", "Louise Petit", "Marie Durand"]
    rows = []
    for i in range(900):
        name, zip_code = owners[i % len(owners)]
        spellings = OWNER_SPELLINGS.get(name)
        if spellings and rng.random() < 0.4:
            name = rng.choice(spellings)
        site = zips[i % 7]
        rows.append(
            [
                f"J{100_000 + i}",
                name,
                zip_code,
                rng.choice(places) if site in zips[:2] else f"PLACE {i % 7}",
                site,
                people[i % len(people)],
                rng.randint(1, 900),
            ]
        )
    header = ["job__", "owner_s_business_name", "owner_s_zip_code", "gis_nta_name", "zip_code"]
    return _csv(rows, [*header, "contact_name", "units"])


ADV_CASES: tuple[Case, ...] = (
    Case(
        "years_zero",
        "kc_house: yr_renovated = 0 on 20,699 rows means 'never'; 13 houses with 0 bedrooms; "
        "chicago_beach: 2 water temperatures above 40 °C",
        _writer("houses_years", _years),
        (
            _correct("B-22", "yr_renovated", "0 means never renovated", sub="B-22", mode="S"),
            _correct(
                "B-22",
                "bedrooms",
                "a house with 0 bedrooms is worth a look",
                sub="B-22:rare",
                mode="I",
                step=False,
            ),
            _wrong("B-22", "bathrooms", "0 on 5 % of the rows: too common to single out"),
            _wrong("B-22", "yr_built", "no 0 among the years"),
            _wrong("B-22", "years_at_company", "years of service: 0 is a real value"),
            _correct("B-10", "water_temperature", "46.6 °C is no lake temperature", mode="S"),
        ),
    ),
    Case(
        "years_impossible",
        "melbourne: YearBuilt 1196 among 1830 to 2018 (1996 mistyped); owid: years from 1750",
        _writer("built", _built),
        (
            _correct("B-10", "YearBuilt", "1196 is a typing error", sub="B-10:year", mode="S"),
            _wrong("B-10", "year", "1750 to 2023: a long panel, no impossible year"),
        ),
    ),
    Case(
        "checkboxes",
        "French probe: a checkbox 'X' and 'Oui' / 'Non' / 'X'; nyc_dob: 'YES' or blank, read as a "
        "boolean, called a constant column",
        _writer("formulaire", _boxes),
        (
            _correct("B-18", "newsletter", "'X' ticked, blank not", sub="B-18:checkbox", mode="S"),
            _correct("B-18", "residential", "a blank means no", sub="B-18:checkbox", mode="S"),
            _wrong("B-19", "newsletter", "a checkbox is not a constant column"),
            _wrong("B-19", "residential", "a checkbox is not a constant column"),
            _decision("B-18", "abonne", "'X' among 'Oui' / 'Non': a ticked box", sub="B-18:mark"),
            _decision("B-18", "valide", "'V' / 'F': vrai / faux", sub="B-18:pair"),
            _decision("B-18", "self_cert", "'Y' / 'N' stays a decision", sub="B-18:pair"),
            _correct("B-03", "abonne", "'oui' is 'Oui'", auto=True),
        ),
    ),
    Case(
        "multiline_text",
        "B-01 turned the line breaks of a comment column into spaces (N29)",
        _writer("tickets", _remarks),
        (_correct("B-01", "comments", "spaces fixed, paragraphs kept", mode="A", auto=True),),
    ),
    Case(
        "dates_order",
        "nyc_311: 9 closed_date a few seconds before created_date (written to the minute); A-11 "
        "swapped a 2026 creation with a 2024 update",
        _writer("requests_order", _requests),
        (
            Truth(
                "A-11",
                None,
                "correct",
                "the same minute written two ways: said, not swapped",
                id="A-11:created_date+closed_date:minute",
                mode="I",
                step=False,
            ),
            Truth(
                "A-11",
                None,
                "wrong",
                "a swap of the same minute",
                id="A-11:created_date+closed_date",
            ),
            _correct("A-11", "ship_date", "shipped days before ordered: swapped", step=True),
        ),
        advanced=True,
    ),
    Case(
        "guards_counts",
        "législatives: A-12 'Inscrits = Votants + % Votants + Abstentions', A-05 on small scores; "
        "fr_rne, diabetes: A-04 on '2A' and '[0-10)'",
        _writer("votes_guards", _counts),
        (
            _wrong("A-12", "% Votants", "a percentage is not added to counts"),
            _correct("A-12", "Inscrits", "Inscrits = Votants + Abstentions, one wrong total"),
            _wrong("A-05", "% Voix/exprimés 2", "the low tail of real small scores"),
            _wrong("A-04", "Code du département", "Corsica and overseas départements"),
            _wrong("A-04", "age", "intervals of ages"),
        ),
        advanced=True,
    ),
    Case(
        "guards_rows",
        "air quality: A-16 on readings of two hours; eco2mix: A-17 on the forecast rows at the end",
        _writer("air_guards", _hours),
        (
            _wrong("A-16", None, "two hours of one date are two readings"),
            _wrong("A-17", None, "the last rows are forecasts still to come"),
        ),
        advanced=True,
    ),
    Case(
        "header_values",
        "dvf: valeur_fonciere repeated on every line of its id_mutation: summed, it counts 226 "
        "times; superstore: Sales vary by line",
        _writer("mutations", _sales),
        (
            _correct("A-18", "valeur_fonciere", "a value of the sale, once per id_mutation"),
            _wrong("A-18", "Sales", "a value of the line"),
            _wrong("A-18", "surface_terrain", "a value of the line (each parcel)"),
        ),
        advanced=True,
    ),
    Case(
        "redundant_columns",
        "adult: education / education-num; chicago_beach: measurement_timestamp_label, a copy as "
        "text; superstore: Customer ID / Customer Name (a key and its name)",
        _writer("twins", _twins),
        (
            _correct("A-19", "education-num", "the same information as education", sub="A-19"),
            _correct(
                "A-19",
                "measurement_timestamp_label",
                "the timestamp written as text",
                sub="A-19:copy",
                mode="S",
            ),
            _correct("A-19", "park_borough", "a copy of borough", sub="A-19:copy", mode="S"),
            _wrong("A-19", "Customer Name", "a key and its name: not a redundancy"),
            _correct("B-11", "measurement_timestamp_label", "dates as text", auto=True),
        ),
        advanced=True,
    ),
    Case(
        "entity_clusters",
        "nyc_dob: owners written several ways with the same postcode (N36), numbered as one "
        "entity; neighbourhoods north and south of one postcode and Paul / Paula are not",
        _writer("owners", _owners),
        (
            _correct(
                "A-20", "owner_s_business_name", "spellings of one owner, one postcode", mode="S"
            ),
            _wrong("A-20", "gis_nta_name", "a place, north and south are two"),
            _wrong("A-20", "contact_name", "Paul and Paula Martin are two people"),
        ),
        advanced=True,
    ),
)

ALL_CASES: tuple[Case, ...] = (*CASES, *ADV_CASES, *SAMPLE_CASES)
