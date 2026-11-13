"""fix3: roles read from the word a name is about (R-TYP-13, R-TYP-14), relevé de 040."""

from pathlib import Path

import pytest

from quantilo.typing.knowledge_base import load, name_note
from tests.test_typing_092 import coded, table


@pytest.mark.parametrize(
    ("name", "role"),
    [
        ("Spending Score (1-100)", "rating"),  # Mall customers: a score, not a spending
        ("Spending Score", "rating"),
        ("Annual Income (k$)", "amount"),  # the note gives the unit, the role is unchanged
        ("Price [EUR]", "amount"),
        ("PercentSalaryHike", "percentage"),  # R-TYP-08 still holds without a note
        ("long_description", "free_text"),
    ],
)
def test_a_closing_note_in_brackets_is_not_the_head_word(name: str, role: str) -> None:
    found = load().match_role(name)
    assert found is not None and found.id == role


def test_the_note_is_what_closes_the_name() -> None:
    assert name_note("Spending Score (1-100)") == "(1-100)"
    assert name_note("Income (k$) [2024]") == "(k$) [2024]"
    assert name_note("total ads") is None
    assert name_note("(x)") is None  # a name made of a note only is a name


@pytest.mark.parametrize(
    ("name", "only"),
    [
        ("total ads", True),
        ("Total Bags", True),
        ("total_rooms", True),
        ("Total", False),  # alone, 'total' is the amount (S-05)
        ("Total Revenue", False),  # a money word says money
        ("total_price", False),
        ("Sales", False),
    ],
)
def test_total_is_the_only_amount_word(name: str, only: bool) -> None:
    assert load().only_total(name) is only


def test_the_mall_score_and_the_ads_count_get_their_role(tmp_path: Path) -> None:
    rows = [[(i * 37) % 99 + 1, (i * 13) % 300 + 1, 15 + i % 120, round(18.5 + i * 0.73, 2)]
            for i in range(400)]  # fmt: skip
    detections = table(
        tmp_path, ["Spending Score (1-100)", "total ads", "Annual Income (k$)", "Total Bill"], rows
    )

    score = detections["Spending Score (1-100)"]
    assert score.role == "rating" and coded(score, "R-TYP-13")
    ads = detections["total ads"]
    assert ads.role == "quantity" and coded(ads, "R-TYP-14")
    assert ads.type in ("numeric_continuous", "numeric_discrete")
    income = detections["Annual Income (k$)"]
    assert income.role == "amount" and not coded(income, "R-TYP-13")  # the note changed nothing
    bill = detections["Total Bill"]  # decimals: a total in cents stays money
    assert bill.role == "amount" and not coded(bill, "R-TYP-14")
