"""Clean, B1: the remaining Advanced rules (A-04, A-06 to A-09, A-14, A-16, A-17), the
conditional step (C-45) and the M5 transformations (split, merge, extract, format, numbers,
formulas). Each rule on a small table: what it finds, what it must leave alone, and that its step
fixes what it found."""

import csv
import math
import random
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import duckdb
import pytest

import quantilo as dx
from quantilo.cleaning import compile_steps, model
from quantilo.cleaning import formula as fx
from quantilo.cleaning.replay import renamed_step, step_columns
from quantilo.cleaning.rules import advanced as adv
from quantilo.cleaning.suggestion import CleanSuggestion


def write(tmp_path: Path, rows: list[dict[str, Any]], name: str = "data") -> dx.Dataset:
    path = tmp_path / f"{name}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return dx.open(path, cache_dir=tmp_path / f"cache-{name}")


def advanced(ds: dx.Dataset) -> dict[str, CleanSuggestion]:
    return {s.id: s for s in ds.clean().advanced()}


def run(ds: dx.Dataset, steps: list[Any], sql: str) -> list[tuple[Any, ...]]:
    """`sql` on the cleaned rows (`t`), the steps compiled strictly."""
    connection = duckdb.connect()
    try:
        compiled = compile_steps(connection, str(ds.path), steps, strict=True)
        return connection.execute(sql.replace("{t}", f"({compiled.sql()})")).fetchall()
    finally:
        connection.close()


def steps(*raw: dict[str, Any]) -> list[Any]:
    return model.parse_steps([{"id": f"s{i}", **r} for i, r in enumerate(raw, 1)]).steps


# --- Formulas ------------------------------------------------------------------------------------


def test_formula_reads_columns_words_and_allowed_functions_only() -> None:
    sql = fx.to_sql(
        "round(quantity * \"unit price\", 2) > 10 and city = 'Paris'",
        ["quantity", "unit price", "city"],
    )
    assert '"unit price"' in sql and "'Paris'" in sql and "AND" in sql
    for bad, why in (
        ("read_csv('x')", "not one of the functions"),
        ("price * 2", "no column 'price'"),
        ("quantity; DROP TABLE t", "cannot read"),
        ("(quantity", "not closed"),
        ("quantity)", "closes nothing"),
    ):
        with pytest.raises(model.CleanError, match=why):
            fx.to_sql(bad, ["quantity"])
    assert fx.columns_in('a + "b c" * abs(d)') == ["a", "b c", "d"]
    assert fx.renamed('a + "b c"', {"b c": "B C"}) == 'a + "B C"'


def test_replay_follows_columns_inside_formulas_and_cases() -> None:
    [step] = steps(
        {
            "op": "set_where",
            "column": "total",
            "cases": [
                {
                    "conditions": [{"column": "city", "op": "eq", "value": "Paris"}],
                    "when": "qty > 1",
                    "formula": "qty * price",
                }
            ],
        }
    )
    assert set(step_columns(step)) == {"total", "city", "qty", "price"}
    moved = renamed_step(step, {"qty": "Quantity", "city": "City"})
    assert moved.cases[0].conditions[0].column == "City"
    assert moved.cases[0].when == "Quantity > 1" and "Quantity" in moved.cases[0].formula


# --- M5 transformations and the conditional step --------------------------------------------------


def test_m5_steps_split_merge_extract_format_and_compute(tmp_path: Path) -> None:
    rows = [
        {"code": "AB-12-x", "size": "12kg", "first": "Ada", "last": "Lovelace", "x": "1.26"},
        {"code": "CD-7", "size": "3m", "first": "Alan", "last": "", "x": "-2.5"},
    ]
    ds = write(tmp_path, rows)
    found = run(
        ds,
        steps(
            {"op": "split_column", "column": "code", "to": ["head", "rest"], "delimiter": "-"},
            {"op": "split_column", "column": "size", "by": "digits", "to": ["amount", "unit"]},
            {"op": "merge_columns", "columns": ["first", "last"], "to": "name"},
            {
                "op": "extract_text",
                "column": "code",
                "how": "between",
                "start": "-",
                "end": "-",
                "to": "mid",
            },
            {"op": "format_text", "column": "head", "prefix": "#", "pad_to": 3, "pad_with": "_"},
            {"op": "transform_number", "column": "x", "how": "round", "decimals": 1},
            {"op": "computed_column", "to": "double", "formula": "x * 2"},
            {"op": "transform_number", "column": "x", "how": "clip", "min": -1, "max": 1},
        ),
        "SELECT head, rest, amount, unit, name, mid, x, double FROM {t}",
    )
    assert found[0] == ("#_AB", "12-x", "12", "kg", "Ada Lovelace", "12", 1.0, pytest.approx(2.6))
    assert found[1][:6] == ("#_CD", "7", "3", "m", "Alan", None)
    assert found[1][6] == -1.0 and found[1][7] == pytest.approx(-5.0)


def test_conditional_step_sets_a_value_where_its_cases_hold(tmp_path: Path) -> None:
    rows = [
        {"city": c, "country": k}
        for c, k in [("Paris", "Belgium"), ("Paris", "France"), ("Lyon", "Belgium")]
    ]
    ds = write(tmp_path, rows)
    found = run(
        ds,
        steps(
            {
                "op": "set_where",
                "column": "country",
                "cases": [
                    {
                        "conditions": [
                            {"column": "city", "op": "eq", "value": "Paris"},
                            {"column": "country", "op": "eq", "value": "Belgium"},
                        ],
                        "value": "France",
                    }
                ],
            },
            {
                "op": "conditional_column",
                "to": "is_paris",
                "cases": [
                    {
                        "conditions": [{"column": "city", "op": "eq", "value": "Paris"}],
                        "value": "yes",
                    }
                ],
                "otherwise": "no",
            },
        ),
        "SELECT country, is_paris FROM {t}",
    )
    assert found == [("France", "yes"), ("France", "yes"), ("Belgium", "no")]  # Lyon untouched
    with pytest.raises(model.CleanError, match="condition or a formula"):
        model.parse_steps(
            [{"id": "s1", "op": "set_where", "column": "c", "cases": [{"value": "x"}]}]
        )


def test_a_formula_that_fails_on_values_is_reported_not_raised(tmp_path: Path) -> None:
    ds = write(tmp_path, [{"amount": "€1,200.00"}, {"amount": "€15.00"}])
    connection = duckdb.connect()
    try:
        compiled = compile_steps(
            connection,
            str(ds.path),
            steps({"op": "computed_column", "to": "y", "formula": "amount * 2"}),
        )
    finally:
        connection.close()
    assert "s1" in compiled.problems


def test_drop_empty_rows_by_share(tmp_path: Path) -> None:
    rows = [
        {"a": "1", "b": "2", "c": "3", "d": "4"},
        {"a": "1", "b": "", "c": "", "d": ""},
        {"a": "", "b": "", "c": "", "d": ""},
    ]
    ds = write(tmp_path, rows)
    assert run(ds, steps({"op": "drop_empty_rows", "share": 0.5}), "SELECT count(*) FROM {t}") == [
        (1,)
    ]


# --- A-04 ----------------------------------------------------------------------------------------


def test_codes_in_the_other_case_phones_and_emails(tmp_path: Path) -> None:
    rng = random.Random(1)
    rows = []
    for i in range(400):
        digits = f"06{rng.randrange(10**8):08d}"
        rows.append(
            {
                "sku": f"AB-{1000 + i % 150}" if i % 40 else f"ab-{1000 + i % 150}",
                "phone": " ".join(digits[k : k + 2] for k in range(0, 10, 2)) if i % 25 else digits,
                "email": f"user{i}@example.com" if i % 50 else f"User{i}@Example.com",
                "amount": rng.randint(1, 50),
            }
        )
    rows[7]["email"] = "not an address"
    found = advanced(write(tmp_path, rows))
    case = found["A-04:sku:case"]
    assert case.mode == "S" and case.risk == "review" and case.step is not None
    assert case.step.map["ab-1000"] == "AB-1000"  # type: ignore[union-attr]
    layout = found["A-04:phone:layout"]
    assert all(" " in new for new in layout.step.map.values())  # type: ignore[union-attr]
    assert found["A-04:email:case"].cells == 8
    assert found["A-04:email:shape"].mode == "I" and found["A-04:email:shape"].cells == 1
    assert not any(s.columns == ["amount"] for s in found.values())


# --- A-14 ----------------------------------------------------------------------------------------


def _orders_dates(rng: random.Random, odd: str) -> list[dict[str, Any]]:
    rows = []
    for i in range(400):
        day = date(2023, 1, 1) + timedelta(days=rng.randrange(700))
        text = day.strftime("%d/%m/%Y")
        if i % 40 == 0:
            if odd == "order" and day.day > 12:
                text = day.strftime("%m/%d/%Y")
            elif odd == "century":
                text = day.strftime("%d/%m/%y")
        rows.append({"ordered": text, "units": rng.randint(1, 9)})
    return rows


@pytest.mark.parametrize("odd, key", [("order", "date:order"), ("century", "date:century")])
def test_dates_written_another_way_are_rewritten_before_b11(
    tmp_path: Path, odd: str, key: str
) -> None:
    ds = write(tmp_path, _orders_dates(random.Random(2), odd), odd)
    cleaning = ds.clean()
    found = {s.id: s for s in cleaning.advanced()}
    suggestion = found[f"A-14:ordered:{key}"]
    assert suggestion.mode == "S" and suggestion.order < 110 * 10_000  # C-46: before B-11
    # month first: the column stays text, B-11 reads it once rewritten; two-digit years: the
    # import read them as 0023 and 0024, the conditional step gives them their century
    reading = [s.step for s in cleaning.suggestions if s.rule == "B-11"]
    fixed = run(
        ds,
        [suggestion.step, *reading],
        "SELECT count(*) FILTER (WHERE ordered IS NULL), min(ordered) FROM {t}",
    )
    assert fixed[0][0] == 0 and fixed[0][1].year >= 2023  # every date read, none in 1923


def test_mixed_time_zones_are_written_in_utc(tmp_path: Path) -> None:
    rows = [  # two values that are not times keep the column as text
        {"at": f"2024-03-0{1 + i % 9}T10:00:00{'+02:00' if i % 2 else 'Z'}", "n": i}
        for i in range(60)
    ]
    rows[5]["at"] = rows[9]["at"] = "see the delivery note"
    found = advanced(write(tmp_path, rows))
    zone = found["A-14:at:zone"]
    assert zone.step.map["2024-03-02T10:00:00+02:00"] == "2024-03-02 08:00:00"  # type: ignore[union-attr]


# --- A-06 to A-09 --------------------------------------------------------------------------------


def test_outliers_are_flagged_in_a_new_column(tmp_path: Path) -> None:
    rng = random.Random(3)
    rows = [{"price": round(rng.gauss(100, 10), 2), "units": rng.randint(1, 9)} for _ in range(800)]
    for row in rows[:4]:
        row["price"] = 10_000
    ds = write(tmp_path, rows)
    found = advanced(ds)
    flag = found["A-06:price"]
    assert flag.cells == 4 and flag.impact == "columns" and flag.step.op == "conditional_column"  # type: ignore[union-attr]
    flagged = run(ds, [flag.step], "SELECT count(*) FROM {t} WHERE price_unusual = 'unusual'")
    assert flagged == [(4,)]
    assert "A-06:units" not in found


def test_floating_point_noise_is_rounded(tmp_path: Path) -> None:
    rows = [{"fee": 4.99 if i % 100 else 4.990000000000001, "k": i % 7} for i in range(300)]
    rows += [{"fee": 9.99, "k": 1}] * 50
    ds = write(tmp_path, rows)
    noise = advanced(ds)["A-07:fee"]
    assert noise.impact == "format" and noise.step.decimals == 2  # type: ignore[union-attr]
    assert run(ds, [noise.step], "SELECT count(DISTINCT fee) FROM {t}") == [(2,)]


def test_benford_flags_generated_amounts_not_natural_ones(tmp_path: Path) -> None:
    rng = random.Random(4)
    natural = [{"amount": round(10 ** rng.uniform(0, 4), 2)} for _ in range(2_000)]
    made_up = [  # amounts that start with 5 to 9: made up, or capped
        {"amount": round(rng.uniform(500, 999) * 10 ** rng.randint(0, 2), 2)} for _ in range(2_000)
    ]
    assert "A-08:amount" not in advanced(write(tmp_path, natural, "natural"))
    found = advanced(write(tmp_path, made_up, "made_up"))["A-08:amount"]
    assert found.mode == "I" and found.step is None and "Benford" in found.reason


def test_last_digits_heaped_on_zero_and_five(tmp_path: Path) -> None:
    rng = random.Random(5)
    heaped = [
        {
            "weight": rng.randrange(40, 120) // 5 * 5
            if rng.random() < 0.6
            else rng.randrange(40, 120)
        }
        for _ in range(600)
    ]
    plain = [{"weight": rng.randrange(40, 120)} for _ in range(600)]
    assert advanced(write(tmp_path, heaped, "heaped"))["A-09:weight"].mode == "I"
    assert "A-09:weight" not in advanced(write(tmp_path, plain, "plain"))


# --- A-16, A-17 ----------------------------------------------------------------------------------


def _orders(rng: random.Random) -> list[dict[str, Any]]:
    names = [
        f"{first} {last}"
        for first in ("Ada", "Alan", "Grace", "Linus", "Barbara")
        for last in ("Smith", "Jones", "Martin", "Dubois")
    ]
    rows = []
    for i in range(600):
        customer = rng.randrange(150)
        rows.append(
            {
                "order_id": f"O-{i:05d}",
                "customer_id": f"C{customer:04d}",
                "name": names[customer % len(names)],
                "day": (date(2024, 1, 1) + timedelta(days=rng.randrange(300))).isoformat(),
                "product": rng.choice(["Mouse", "Keyboard", "Screen", "Cable", "Lamp"]),
                "quantity": rng.randint(1, 5),
                "total": round(rng.uniform(5, 500), 2),
            }
        )
    return rows


def test_fellegi_sunter_finds_orders_typed_again_with_one_slip(tmp_path: Path) -> None:
    rng = random.Random(6)
    rows = _orders(rng)
    assert "A-16:rows:linkage" not in advanced(write(tmp_path, rows, "clean"))
    for index, source in enumerate(rows[100:108]):
        rows.append(dict(source, order_id=f"O-9{index:04d}", name=source["name"][:-1]))
    ds = write(tmp_path, rows, "typed")
    found = advanced(ds)["A-16:rows:linkage"]
    assert found.rows == 8 and found.impact == "rows" and found.step.op == "drop_duplicates"  # type: ignore[union-attr]
    assert run(ds, [found.step], "SELECT count(*) FROM {t}") == [(600,)]


def test_mostly_empty_rows(tmp_path: Path) -> None:
    rows = [{"a": i, "b": i * 2, "c": "x", "d": "y", "e": "z"} for i in range(200)]
    for row in rows[:6]:
        row.update(b="", c="", d="", e="")
    found = advanced(write(tmp_path, rows))["A-17:rows:sparse"]
    assert found.rows == 6 and found.step.share == 0.5  # type: ignore[union-attr]


# --- Catalogue -----------------------------------------------------------------------------------


def test_every_advanced_rule_has_its_effect_and_order() -> None:
    assert set(adv.ORDER) == set(adv.IMPACT)
    for (rule, op), impact in adv.STEP_IMPACT.items():
        assert rule in adv.ORDER and op in model.OPS and impact in model.IMPACTS
    assert set(model.OP_IMPACT) == set(model.OPS)
    assert not math.isnan(adv.ORDER["A-14"]) and adv.ORDER["A-14"] < 110  # C-46
