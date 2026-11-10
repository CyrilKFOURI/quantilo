# ruff: noqa: RUF001, RUF003  (−, ρ, · are the app's symbols)
"""094 — ML on real-world quirks: `__` in names, answers hidden in sums and differences,
repeated identifiers, codes typed as numbers, plausible targets, the "did you mean" of a name."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import quantilo as dx
from quantilo.errors import ColumnNotFound
from quantilo.ml import codes
from quantilo.ml import models as catalogue
from quantilo.ml import preparation as prep
from quantilo.ml.data import Field
from quantilo.ml.target import candidate
from quantilo.ml.train import preview, train
from quantilo.typing.types import Parser

RAW = Parser(kind="raw")


def decision(preparation: prep.Preparation, column: str) -> prep.Decision:
    return next(d for d in preparation.decisions if d.column == column)


def numbers(frame: pd.DataFrame, *, discrete: tuple[str, ...] = ()) -> list[Field]:
    return [
        Field(c, "numeric_discrete" if c in discrete else "numeric_continuous", RAW)
        for c in frame.columns
        if c != "__row"
    ]


# --- R-CODE: names and values --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "word"),
    [
        ("patient_nbr", "nbr"),
        ("bin__", "bin"),
        ("PID", "pid"),
        ("customerID", "id"),
        ("Code du département", "code"),
        ("permit_si_no", "no"),
        ("number_of_persons_injured", None),
        ("num_lab_procedures", None),
        ("cabin", None),
        ("Paid", None),
    ],
)
def test_code_words_are_whole_words(name: str, word: str | None) -> None:
    assert codes.code_word(name) == word


def test_a_sequence_number_rises_with_the_rows() -> None:
    order = np.arange(400)
    assert codes.sequence(np.repeat(np.arange(1, 201), 2).astype(float), order) is not None
    noisy = np.random.default_rng(0).normal(50, 10, 400).round()
    assert codes.sequence(noisy, order) is None


# --- ML-1: `__` in a column name -----------------------------------------------------------------


def test_double_underscores_in_names_train_segment_and_score(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    n = 300
    frame = pd.DataFrame(
        {
            "bin__": rng.normal(0, 1, n),
            "job__": rng.normal(0, 1, n),
            "verbose": rng.normal(0, 1, n),  # a ColumnTransformer parameter name
            "kind": rng.choice(["a", "b", "c"], n),
        }
    )
    frame["price"] = 2 * frame["bin__"] - frame["job__"] + rng.normal(0, 0.5, n)
    ds = dx.from_pandas(frame, "underscores", cache_dir=tmp_path / "u")

    model = ds.ml("price").train()
    assert model.score is not None and model.score > 0.5
    named = " ".join(c.feature for c in model.coefficients)
    assert "bin__" in named and "job__" in named and "verbose" in named
    assert len(ds.segments(k=2).fit().labels) == n
    assert len(ds.unusual_rows().scores) == n


# --- ML-2: the answer hidden in a sum or a difference --------------------------------------------


def _injuries(n: int = 2_000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    a, b, c = (rng.poisson(lam, n).astype(float) for lam in (0.3, 0.1, 0.6))
    total = a + b + c
    other = rng.random(n) < 0.05  # 5 % of the rows count someone else as well
    total[other] += 1
    return pd.DataFrame(
        {
            "pedestrians": a,
            "cyclists": b,
            "motorists": c,
            "hour": rng.integers(0, 24, n).astype(float),
            "injured": total,
            "__row": np.arange(n),
        }
    )


def test_a_target_equal_to_a_sum_of_columns_is_a_leak() -> None:
    frame = _injuries()
    fields = numbers(frame)
    target = next(f for f in fields if f.name == "injured")

    preparation = prep.decide(frame, fields, target, "regression", catalogue.LINEAR)

    for column in ("pedestrians", "cyclists", "motorists"):
        d = decision(preparation, column)
        assert not d.use and d.leak and d.reactivable and d.rules == ["R-SEL-06"]
        assert "injured = " in d.why[0] and "of the rows" in d.why[0]
    assert "pedestrians + cyclists + motorists" in "".join(decision(preparation, "cyclists").why)
    assert decision(preparation, "hour").use


def test_a_difference_that_separates_the_classes_is_a_leak() -> None:
    rng = np.random.default_rng(1)
    n = 1_500
    pts = rng.normal(100, 12, n).round()
    opp = rng.normal(100, 12, n).round()
    opp[opp == pts] += 1
    frame = pd.DataFrame(
        {
            "pts": pts,
            "opp_pts": opp,
            "forecast": rng.random(n),
            "result": np.where(pts > opp, "W", "L"),
            "__row": np.arange(n),
        }
    )
    fields = [*numbers(frame.drop(columns="result")), Field("result", "categorical_nominal", RAW)]

    preparation = prep.decide(frame, fields, fields[-1], "binary", catalogue.LINEAR)

    for column in ("pts", "opp_pts"):
        d = decision(preparation, column)
        assert not d.use and d.leak and "AUC" in d.why[0]
    assert "pts − opp_pts" in decision(preparation, "pts").why[0]
    assert decision(preparation, "forecast").use


def test_unrelated_strong_predictors_are_not_leaks() -> None:
    rng = np.random.default_rng(2)
    n = 1_500
    frame = pd.DataFrame({"x1": rng.normal(0, 1, n), "x2": rng.normal(0, 1, n)})
    frame["y"] = frame["x1"] + frame["x2"] + rng.normal(0, 0.3, n)
    frame["__row"] = np.arange(n)
    fields = numbers(frame)

    preparation = prep.decide(frame, fields, fields[-1], "regression", catalogue.LINEAR)

    assert decision(preparation, "x1").use and decision(preparation, "x2").use


def test_a_near_perfect_score_is_flagged_on_the_result() -> None:
    rng = np.random.default_rng(4)
    n = 600
    frame = pd.DataFrame({"x": rng.normal(0, 1, n), "z": rng.normal(0, 1, n)})
    frame["y"] = 3 * frame["x"] + rng.normal(0, 0.02, n)  # |r| ≈ 0.99998
    frame["__row"] = np.arange(n)
    fields = numbers(frame)
    target = fields[2]
    # the leak rule would leave x out: force it back, as a user may
    choices = prep.Choices(columns={"x": prep.ColumnChoice(use=True)})

    trained = train(
        frame, fields, target, "regression", "linear_regression", None, n, False, choices=choices
    )

    assert trained.score is not None and trained.score > 0.995
    assert any("R-LEAK-01" in w and "x" in w for w in trained.preparation.warnings)


# --- ML-3: a repeated identifier -----------------------------------------------------------------


def test_a_repeated_identifier_suggests_group_k_fold() -> None:
    rng = np.random.default_rng(5)
    n = 400
    frame = pd.DataFrame(
        {
            "game_id": np.repeat([f"G{i}" for i in range(n // 2)], 2),
            "x": rng.normal(0, 1, n),
        }
    )
    frame["y"] = frame["x"] + rng.normal(0, 1, n)
    frame["__row"] = np.arange(n)
    fields = [
        Field("game_id", "identifier", RAW),
        Field("x", "numeric_continuous", RAW),
        Field("y", "numeric_continuous", RAW),
    ]

    shown = preview(frame, fields, fields[2], "regression", "linear_regression", None)

    warning = next(w for w in shown.preparation.warnings if "R-VAL-01" in w)
    assert "game_id" in warning and "Group k-fold" in warning


# --- ML-4: codes and sequence numbers typed as numbers -------------------------------------------


def test_codes_and_sequence_numbers_are_left_out_with_their_reason() -> None:
    rng = np.random.default_rng(6)
    n = 600
    frame = pd.DataFrame(
        {
            "PID": rng.integers(526_301_000, 999_999_999, n).astype(float),
            "gameorder": np.repeat(np.arange(1, n // 2 + 1), 2).astype(float),
            "area": rng.normal(100, 20, n),
        }
    )
    frame["price"] = 3 * frame["area"] + rng.normal(0, 40, n)
    frame["__row"] = np.arange(n)
    fields = numbers(frame)

    preparation = prep.decide(frame, fields, fields[-1], "regression", catalogue.LINEAR)

    pid, order = decision(preparation, "PID"), decision(preparation, "gameorder")
    assert not pid.use and pid.reactivable and pid.rules == ["R-SEL-10"] and "code" in pid.why[0]
    assert not order.use and order.reactivable and order.rules == ["R-SEL-11"]
    assert decision(preparation, "area").use


# --- ML-5: target suggestions --------------------------------------------------------------------


def test_implausible_targets_are_not_suggested_with_their_reason() -> None:
    patient = candidate("patient_nbr", "numeric_continuous", "quantity", 70_000, 0)
    flag = candidate("SNOW_ATTRIBUTES", "categorical_nominal", None, 6, 0)
    sex = candidate("sex", "categorical_nominal", "gender", 2, 0)
    killed = candidate("killed", "numeric_discrete", None, 3, 0, dominant=0.995)
    for found, words in ((patient, "code"), (flag, "flag"), (sex, "gender"), (killed, "99")):
        assert not found.suggested and words in (found.reason or ""), found


def test_few_values_are_suggested_as_classes_and_value_names_as_targets() -> None:
    stage = candidate("column13", "numeric_discrete", None, 5, 0)
    assert stage.suggested and "classes" in (stage.reason or "")
    value = candidate("valeur_fonciere", "numeric_continuous", None, 9_000, 0)
    assert value.suggested and "valeur" in (value.reason or "")
    plain = candidate("bmi", "numeric_continuous", None, 500, 0)
    assert not plain.suggested


# --- ML-7: did you mean --------------------------------------------------------------------------


def test_did_you_mean_compares_normalized_names() -> None:
    error = ColumnNotFound("Recommended IND", ["recommended_ind", "rating", "Clothing ID"])
    assert error.suggestions == ["recommended_ind"] and "recommended_ind" in error.message
    assert ColumnNotFound("Taget", ["target"]).suggestions == ["target"]


# --- ML-6: a first model that stays fast ---------------------------------------------------------


def test_only_the_richest_free_texts_are_used_by_default() -> None:
    rng = np.random.default_rng(8)
    n = 300
    words = ["red", "blue", "green", "fast", "slow", "big", "small", "new", "old", "cheap"]
    frame = pd.DataFrame(
        {f"t{k}": [" ".join(rng.choice(words, k + 1)) for _ in range(n)] for k in range(5)}
    )
    frame["y"] = rng.normal(0, 1, n)
    frame["__row"] = np.arange(n)
    fields = [
        *(Field(f"t{k}", "text", RAW) for k in range(5)),
        Field("y", "numeric_continuous", RAW),
    ]

    preparation = prep.decide(frame, fields, fields[-1], "regression", catalogue.LINEAR)

    used = [d.column for d in preparation.used if d.type == "text"]
    assert used == ["t2", "t3", "t4"]
    left = decision(preparation, "t0")
    assert not left.use and left.reactivable and left.rules[-1] == "R-TEXT-06"


def test_normal_equations_give_the_statsmodels_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    from quantilo.ml import train as trainer

    rng = np.random.default_rng(9)
    n = 400
    frame = pd.DataFrame({"a": rng.normal(0, 1, n), "b": rng.normal(0, 1, n)})
    frame["y"] = 2 * frame["a"] - frame["b"] + rng.normal(0, 1, n)
    frame["__row"] = np.arange(n)
    fields = numbers(frame)
    settings = {"regularization": "none"}

    def run() -> dict[str, tuple[float | None, float | None]]:
        trained = train(
            frame, fields, fields[-1], "regression", "linear_regression", settings, n, False
        )
        return {c.feature: (c.std_error, c.p_value) for c in trained.coefficients}

    reference = run()
    monkeypatch.setattr(trainer, "NORMAL_EQUATIONS_ABOVE", 0)
    fast = run()

    for name, (error, _p) in reference.items():
        assert fast[name][0] == pytest.approx(error, rel=1e-6)


def test_a_column_the_identity_does_not_need_stays() -> None:
    rng = np.random.default_rng(10)
    n = 2_000
    pts = rng.normal(100, 12, n).round()
    opp = rng.normal(100, 12, n).round()
    opp[opp == pts] += 1
    # a pre-match forecast: informative, taken first by the forward search
    forecast = 1 / (1 + np.exp(-(pts - opp) / 8 + rng.normal(0, 1.2, n)))
    frame = pd.DataFrame(
        {
            "forecast": forecast,
            "pts": pts,
            "opp_pts": opp,
            "result": np.where(pts > opp, "W", "L"),
            "__row": np.arange(n),
        }
    )
    fields = [*numbers(frame.drop(columns="result")), Field("result", "categorical_nominal", RAW)]

    preparation = prep.decide(frame, fields, fields[-1], "binary", catalogue.LINEAR)

    assert not decision(preparation, "pts").use and not decision(preparation, "opp_pts").use
    assert decision(preparation, "forecast").use
