"""Which columns can be predicted, and what kind of problem each one is."""

from dataclasses import dataclass
from typing import Literal

from quantilo.ml import codes
from quantilo.ml.data import CATEGORICAL, NUMERIC

Task = Literal["regression", "binary", "multiclass"]
MAX_CLASSES = 20  # beyond, the 19 most frequent classes plus "Other"
SUGGESTED_ROLES = {
    "amount",
    "quantity",
    "rating",
    "status",
    "percentage",
    "duration",
    "cost",  # 048: once amounts
    "unit_price",
}
# R-TGT-02 (094): roles that describe who, where or when the row is, rarely what one predicts
DESCRIPTOR_ROLES = {
    "gender",
    "age",
    "region",
    "country",
    "city",
    "postal_code",
    "latitude",
    "longitude",
    "identifier",
    "entity",
    "customer_key",
    "order_key",
    "year",
    "month",
    "weekday",
    "hour",
    "address",
    "email",
    "phone",
    "url",
    "currency",
}
DOMINANT = 0.95  # R-TGT-04: a number whose most frequent value covers this share of the rows
CLASS_VALUES = (3, 10)  # R-TGT-05: whole numbers with this many values, suggested as classes


@dataclass
class Candidate:
    name: str
    type: str
    role: str | None
    task: Task | None  # None: cannot be a target
    reason: str | None  # why it cannot, or a warning
    suggested: bool
    classes: int | None  # distinct values (categories)
    missing_share: float


def task_for(analysis_type: str, distinct: int) -> Task | None:
    if analysis_type in NUMERIC:
        return "regression"
    if analysis_type == "boolean" or (analysis_type in CATEGORICAL and distinct == 2):
        return "binary"
    if analysis_type in CATEGORICAL and distinct >= 3:
        return "multiclass"
    return None


def candidate(
    name: str,
    analysis_type: str,
    role: str | None,
    distinct: int,
    missing_share: float,
    present: int = 0,  # rows with a value (0: unknown)
    dominant: float | None = None,  # numbers: share of the most frequent value (None: unknown)
) -> Candidate:
    """The problem a column defines as a target, or why it cannot be one (016 spec); whether it
    is suggested, and why not (R-TGT-01 to 06, 094)."""
    if analysis_type in ("identifier", "text", "datetime"):
        why = {"identifier": "an identifier", "text": "free text", "datetime": "a date"}[
            analysis_type
        ]
        return Candidate(
            name,
            analysis_type,
            role,
            None,
            f"Cannot be predicted: {why}.",
            False,
            None,
            missing_share,
        )
    if missing_share > 0.5:
        return Candidate(
            name,
            analysis_type,
            role,
            None,
            f"{missing_share:.0%} empty.",
            False,
            distinct,
            missing_share,
        )
    if analysis_type in CATEGORICAL and present and distinct > 20 and distinct > 0.5 * present:
        return Candidate(
            name,
            analysis_type,
            role,
            None,
            f"Behaves like an identifier ({distinct:,} distinct values for {present:,} rows).",
            False,
            distinct,
            missing_share,
        )
    task = task_for(analysis_type, distinct)
    if task is None:
        return Candidate(
            name,
            analysis_type,
            role,
            None,
            "Not enough distinct values.",
            False,
            distinct,
            missing_share,
        )
    reason = None
    if task == "multiclass" and distinct > MAX_CLASSES:
        reason = (
            f"{distinct} classes: the {MAX_CLASSES - 1} most frequent are kept, "
            "the rest becomes Other."
        )
    elif missing_share > 0.1:
        reason = f"{missing_share:.0%} of rows have no value and are left out."
    elif task == "regression" and analysis_type == "numeric_discrete" and distinct <= 10:
        reason = "Few distinct values: predicting them as classes may suit better."
    suggested = role in SUGGESTED_ROLES or (task != "regression" and distinct <= 10)
    suggested, why = _suggestion(name, analysis_type, role, task, distinct, dominant, suggested)
    if why is not None:
        reason = why if reason is None or not suggested else f"{why} {reason}"
    classes = distinct if task != "regression" or analysis_type == "numeric_discrete" else None
    return Candidate(name, analysis_type, role, task, reason, suggested, classes, missing_share)


def _suggestion(
    name: str,
    analysis_type: str,
    role: str | None,
    task: Task,
    distinct: int,
    dominant: float | None,
    suggested: bool,
) -> tuple[bool, str | None]:
    """R-TGT-01 to 06 (094): what keeps a possible target from being suggested, or what makes
    one suggested; with the sentence that says so."""
    if analysis_type in NUMERIC and codes.code_word(name):
        return (
            False,
            f"Its name says a code or an identifier ({codes.code_word(name)}): not suggested.",
        )
    if codes.flag_word(name):
        return False, "A quality flag or an attribute of another column: not suggested."
    if role in DESCRIPTOR_ROLES:
        return False, f"Describes the row ({role.replace('_', ' ')}): rarely what one predicts."
    if task == "regression" and dominant is not None and dominant >= DOMINANT:
        return False, f"Almost always the same value ({dominant:.1%} of rows): not suggested."
    low, high = CLASS_VALUES
    if task == "regression" and analysis_type == "numeric_discrete" and low <= distinct <= high:
        return True, (
            f"{distinct} distinct values: suggested as classes (Predict as classes, one per value)."
        )
    if task == "regression" and not suggested and role is None and codes.value_word(name):
        return True, f"Its name says a value to predict ({codes.value_word(name)})."
    return suggested, None
