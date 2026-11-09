"""Write samples/ab_test.csv: a checkout A/B test, one row per session (spec 034).

2,000 users split 50/50 between `control` and `new_checkout`, 1 to 3 sessions each over three
weeks. The new checkout converts a little more; `spend_before` (the user's spend in the month
before the test) is correlated with what they spend during it, for CUPED. Seeded: the same file
every time.

    uv run --project backend python scripts/make_ab_sample.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

USERS = 2_000
SEED = 34
TARGET = Path(__file__).resolve().parents[1] / "samples" / "ab_test.csv"


def main() -> None:
    rng = np.random.default_rng(SEED)
    users = np.arange(10_001, 10_001 + USERS)
    variant = np.where(rng.random(USERS) < 0.5, "control", "new_checkout")
    device = rng.choice(["mobile", "desktop", "tablet"], USERS, p=[0.55, 0.37, 0.08])
    appetite = rng.lognormal(3.4, 0.6, USERS)  # what a user tends to spend
    spend_before = np.round(appetite * rng.lognormal(0, 0.35, USERS), 2)
    sessions = rng.integers(1, 4, USERS)

    rows = []
    start = pd.Timestamp("2026-03-02")
    for i in range(USERS):
        first = int(rng.integers(0, 21))
        for s in range(int(sessions[i])):
            day = start + pd.Timedelta(days=min(20, first + 2 * s))
            rate = 0.10 if variant[i] == "control" else 0.125
            rate *= 1.25 if device[i] == "desktop" else 1.0
            converted = rng.random() < rate
            orders = int(rng.integers(1, 3)) if converted else 0
            revenue = round(float(orders * appetite[i] * rng.lognormal(0, 0.3)), 2) * converted
            rows.append(
                (
                    users[i],
                    day.date().isoformat(),
                    variant[i],
                    device[i],
                    int(converted),
                    orders,
                    revenue,
                    spend_before[i],
                )
            )
    frame = pd.DataFrame(
        rows,
        columns=[
            "user_id",
            "date",
            "variant",
            "device",
            "converted",
            "orders",
            "revenue",
            "spend_before",
        ],
    )
    frame.to_csv(TARGET, index=False)
    print(f"{TARGET}: {len(frame):,} rows")


if __name__ == "__main__":
    main()
