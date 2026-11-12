"""Generate samples/messy_orders.csv: synthetic orders packed with what Clean finds (feature 097).

Every problem is planted on purpose, at a known rate, so that samples/messy_orders.expected.json can
list the suggestions each layer must make. Three control columns (channel, payment_method,
loyalty_points) stay clean: no rule may touch them. Deterministic (seed 97), stdlib only.

Usage: python scripts/make_messy_sample.py
"""

import csv
import random
from datetime import date, timedelta
from pathlib import Path

SAMPLES = Path(__file__).resolve().parents[1] / "samples"
ROWS = 3_000
SEED = 97

# city, country, postal codes (None: abroad, no French code), weight
CITIES = [
    ("Paris", "France", [f"750{n:02d}" for n in range(1, 21)], 20),
    ("Lyon", "France", [f"6900{n}" for n in range(1, 10)], 12),
    ("Marseille", "France", [f"130{n:02d}" for n in range(1, 17)], 10),
    ("Toulouse", "France", ["31000", "31100", "31200"], 8),
    ("Nice", "France", ["06000", "06100", "06200"], 6),
    ("Lille", "France", ["59000", "59800"], 6),
    ("Bordeaux", "France", ["33000", "33800"], 6),
    ("Saint-Étienne", "France", ["42000", "42100"], 5),
    ("Annecy", "France", ["74000"], 4),
    ("Bourg-en-Bresse", "France", ["01000"], 4),
    ("Parisot", "France", ["81310"], 4),  # a real village: never to be merged with Paris
    ("Bruxelles", "Belgium", None, 5),
    ("Liège", "Belgium", None, 3),
    ("Genève", "Switzerland", None, 4),
    ("Zürich", "Switzerland", None, 3),
]
# product, category, unit price (EUR), weight (kg), brand, weight
PRODUCTS = [
    ("Wireless Mouse", "Accessories", 24.90, 0.10, "Logitech", 14),
    ("Keyboard", "Accessories", 79.00, 0.90, "Logitech", 10),
    ("USB-C Cable", "Accessories", 11.50, 0.05, "Anker", 14),
    ("Laptop Stand", "Accessories", 39.00, 1.20, "Rain Design", 7),
    ("Monitor 27 inch", "Displays", 249.00, 5.50, "Dell", 6),
    ("Monitor 24 inch", "Displays", 179.00, 4.10, "Dell", 6),  # one digit from the 27: kept
    ("Office Chair", "Furniture", 199.00, 14.00, "Steelcase", 6),
    ("Standing Desk", "Furniture", 449.00, 28.00, "IKEA", 4),
    ("Desk Lamp", "Furniture", 45.00, 1.50, "IKEA", 8),
    ("Noise Cancelling Headphones", "Audio", 179.00, 0.30, "Bang & Olufsen", 6),
    ("Bluetooth Speaker", "Audio", 59.00, 0.60, "JBL", 9),
    ("Webcam HD", "Video", 69.00, 0.20, "Logitech", 10),
]
FIRST = ["Paul", "Marie", "Lucas", "Emma", "Hugo", "Léa", "Louis", "Chloé", "Jules", "Inès", "Noah",
         "Camille", "Adam", "Manon", "Arthur", "Sarah", "Gabriel", "Zoé", "Raphaël", "Alice"]  # fmt: skip
LAST = ["Martin", "Bernard", "Dubois", "Thomas", "Robert", "Richard", "Petit", "Durand", "Leroy",
        "Moreau", "Simon", "Laurent", "Lefèvre", "Michel", "Garcia", "David", "Bertrand", "Roux"]  # fmt: skip
STATUS = [("Delivered", 70), ("Shipped", 15), ("Cancelled", 5), ("Returned", 4)]
MISSING_STATUS = ["N/A", "", "unknown", "-"]
NOTES = ["Leave at the door", "Gift wrap &amp; card", "Livraison trÃ¨s rapide",
         "Call before delivery​", "  Fragile   handle with care ", "Office reception"]  # fmt: skip
COLUMNS = [
    "order_id", "order_date", "ship_date", "customer_id", "customer_name", "birth_date",
    "customer_age", "city", "zip_code", "country", "product", "category", "brand", "quantity",
    "unit_price", "discount_pct", "shipping", "total", "weight_kg", "rating", "status",
    "lifetime_value", "notes", "channel", "payment_method", "loyalty_points", "currency",
    "legacy_field",
]  # fmt: skip


def pick(rng: random.Random, items: list, weights: list[int]):
    return rng.choices(items, weights=weights, k=1)[0]


def euros(value: float) -> str:
    return f"{value:.2f}"


def order_date_text(rng: random.Random, day: date) -> str:
    """Three formats mixed (B-11): ISO, day/month/year, '05 Mar 2024'."""
    roll = rng.random()
    if roll < 0.6:
        return day.isoformat()
    if roll < 0.9:
        return day.strftime("%d/%m/%Y")
    return day.strftime("%d %b %Y")


def city_spelling(rng: random.Random, city: str) -> str:
    """Case, spacing and accent variants (B-01, B-03), mojibake (B-05), word order and
    separators (A-01, A-02), rare typos (A-03)."""
    roll = rng.random()
    variants = {
        "Paris": [("PARIS", 0.03), ("paris", 0.03), (" Paris", 0.02), ("Paris.", 0.01)],
        "Marseille": [("MARSEILLE", 0.04)],
        "Bourg-en-Bresse": [("Bourg en Bresse", 0.12), ("BourgenBresse", 0.04)],
        "Saint-Étienne": [("Saint-Etienne", 0.10), ("SaintEtienne", 0.03)],
        "Genève": [("GenÃ¨ve", 0.06), ("Geneve", 0.05)],
        "Zürich": [("Zurich", 0.08)],
        "Lyon": [("Lyon ", 0.03), ("LYON", 0.03)],
    }
    cumulative = 0.0
    for spelling, share in variants.get(city, []):
        cumulative += share
        if roll < cumulative:
            return spelling
    return city


def product_spelling(rng: random.Random, product: str) -> str:
    roll = rng.random()
    variants = {
        "Wireless Mouse": [("wireless mouse", 0.03), ("Mouse, Wireless", 0.025)],
        "USB-C Cable": [("USB C Cable", 0.03), ("USBC Cable", 0.012)],
        "Desk Lamp": [(" Desk Lamp ", 0.03)],
        "Standing Desk": [("Standing Desk​", 0.05)],
    }
    cumulative = 0.0
    for spelling, share in variants.get(product, []):
        cumulative += share
        if roll < cumulative:
            return spelling
    return product


def customers(rng: random.Random, count: int = 1_500) -> list[tuple[int, str, date]]:
    people = []
    for index in range(count):
        name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        born = date(1950, 1, 1) + timedelta(days=rng.randrange(0, 365 * 55))
        people.append((10_001 + index, name, born))
    return people


def age_on(born: date, day: date) -> int:
    return day.year - born.year - ((day.month, day.day) < (born.month, born.day))


def order(rng: random.Random, number: int, people: list[tuple[int, str, date]]) -> dict[str, str]:
    city, country, codes, _ = pick(rng, CITIES, [c[3] for c in CITIES])
    product, category, price, weight, brand, _ = pick(rng, PRODUCTS, [p[5] for p in PRODUCTS])
    customer_id, name, born = rng.choice(people)
    day = date(2024, 1, 1) + timedelta(days=rng.randrange(0, 547))
    shipped = day + timedelta(days=rng.randint(1, 7))
    quantity = pick(rng, [1, 2, 3, 4], [60, 25, 10, 5])
    shipping = pick(rng, [0.0, 4.99, 9.99], [40, 40, 20])
    discount = pick(rng, [0, 5, 10, 15, 20], [50, 20, 15, 10, 5])
    total = round(quantity * price + shipping, 2)
    zip_code = rng.choice(codes) if codes else ""
    return {
        "order_id": f"ORD-{100_000 + number}",
        "order_date": order_date_text(rng, day),
        "ship_date": shipped.isoformat(),
        "customer_id": str(customer_id),
        "customer_name": name.upper() if rng.random() < 0.03 else name,
        "birth_date": born.isoformat(),
        "customer_age": str(age_on(born, day)),
        "city": city_spelling(rng, city),
        "zip_code": zip_code,
        "country": country,
        "product": product_spelling(rng, product),
        "category": category,
        "brand": brand.replace("&", "&amp;") if "&" in brand and rng.random() < 0.15 else brand,
        "quantity": str(quantity),
        "unit_price": euros(price),
        "discount_pct": str(discount),
        "shipping": euros(shipping),
        "total": euros(total),
        "weight_kg": f"{weight:.2f}",  # of one item
        "rating": str(rng.randint(1, 5)) if rng.random() < 0.9 else "",
        "status": pick(rng, [s for s, _ in STATUS], [w for _, w in STATUS]),
        "lifetime_value": f"€{rng.uniform(50, 20_000):,.2f}",
        "notes": rng.choice(NOTES) if rng.random() < 0.15 else "",
        "channel": pick(rng, ["Web", "Store", "Phone"], [50, 30, 20]),
        "payment_method": pick(rng, ["Card", "PayPal", "Transfer"], [60, 25, 15]),
        "loyalty_points": str(rng.randint(100, 5_000)),
        "currency": "EUR",
        "legacy_field": "",
    }


def plant(rng: random.Random, rows: list[dict[str, str]]) -> None:
    """Problems at known rates, each on rows chosen at random (seeded)."""

    def some(share: float | int, where=lambda r: True) -> list[dict[str, str]]:
        pool = [r for r in rows if where(r)]
        count = share if isinstance(share, int) else round(share * len(pool))
        return rng.sample(pool, min(count, len(pool)))

    def is_city(name: str):
        return lambda r: r["city"] == name

    for row in some(4, is_city("Paris")):
        row["city"] = "Pariss"  # A-03: one letter from Paris
    for row in some(3, is_city("Marseille")):
        row["city"] = "Marseile"
    for row in some(3, is_city("Toulouse")):
        row["city"] = "Tolouse"
    for row in some(4, lambda r: r["product"] == "Wireless Mouse"):
        row["product"] = "Wirless Mouse"
    for row in some(4, lambda r: r["product"] == "Bluetooth Speaker"):
        row["product"] = "Blutooth Speeker"  # A-03: two letters away, the same sound
    # A-10: dependencies broken on a few rows
    for row in some(8, is_city("Paris")):
        row["country"] = "Belgium"
    for row in some(5, is_city("Lyon")):
        row["country"] = "Switzerland"
    for row in some(6, lambda r: r["product"] == "Office Chair"):
        row["category"] = "Accessories"
    for row in some(4, lambda r: r["city"] == "Lyon" and r["country"] == "France"):
        row["zip_code"] = "75011"  # a Paris code on a Lyon row
    # B-13: codes that lost their leading zero
    for row in some(0.3, lambda r: r["zip_code"].startswith("0")):
        row["zip_code"] = row["zip_code"][1:]
    # B-03: country in lowercase
    for row in some(0.02, lambda r: r["country"] == "France"):
        row["country"] = "france"
    # A-05: cents among euros, grams among kilograms, fractions among percentages
    for row in some(0.025):
        row["unit_price"] = str(round(float(row["unit_price"]) * 100))
    for row in some(0.04):
        row["weight_kg"] = str(round(float(row["weight_kg"]) * 1000))
    for row in some(0.08, lambda r: r["discount_pct"] != "0"):
        row["discount_pct"] = f"{int(row['discount_pct']) / 100:g}"
    # A-12: totals that do not match quantity × unit price + shipping
    for row in some(0.01):
        row["total"] = euros(float(row["total"]) + rng.choice([10, 100, -5]))
    # A-11: shipped before ordered
    for row in some(0.02):
        ordered = date.fromisoformat(row["ship_date"]) + timedelta(days=rng.randint(9, 20))
        row["order_date"] = ordered.isoformat()
        born = date.fromisoformat(row["birth_date"])
        row["customer_age"] = str(age_on(born, ordered))
    # A-13: ages that contradict the birth date
    for row in some(0.02):
        row["customer_age"] = str(int(row["customer_age"]) + rng.choice([-12, -9, 8, 11]))
    # B-09: 99 for "no rating"
    for row in some(0.02, lambda r: r["rating"] != ""):
        row["rating"] = "99"
    # B-04 null tokens and B-03 case in status
    for row in some(0.055):
        row["status"] = rng.choice(MISSING_STATUS)
    for row in some(0.02, lambda r: r["status"] == "Delivered"):
        row["status"] = "delivered"


def main() -> None:
    rng = random.Random(SEED)
    people = customers(rng)
    rows = [order(rng, number, people) for number in range(1, ROWS + 1)]
    plant(rng, rows)
    # A-15: the same order entered again under a new number, with another spelling
    for number, source in enumerate(rng.sample(rows, 20), start=ROWS + 1):
        copy = dict(source, order_id=f"ORD-{100_000 + number}")
        copy["city"] = copy["city"].upper() if rng.random() < 0.5 else f"  {copy['city']}"
        rows.insert(rng.randrange(len(rows)), copy)
    # B-14: exact copies; B-15: empty rows (no repeated header: it would make every column text)
    for source in rng.sample(rows, 30):
        rows.insert(rng.randrange(len(rows)), dict(source))
    for _ in range(6):
        rows.insert(rng.randrange(len(rows)), dict.fromkeys(COLUMNS, ""))
    path = SAMPLES / "messy_orders.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{path.name}: {len(rows):,} rows, {len(COLUMNS)} columns")


if __name__ == "__main__":
    main()
