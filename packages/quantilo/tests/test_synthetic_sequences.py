"""Sequences and TimeGAN (023c): the plan rules, windows, held-out histories, the measures."""

import numpy as np
import pandas as pd
import pytest

from quantilo.synthetic import diagnostic, encode
from quantilo.synthetic import sequences as seq
from quantilo.synthetic.generate import Spec, synthesize
from quantilo.synthetic.methods import Cancelled, Options, SequenceSettings


def stores(days: int = 120, count: int = 4, seed: int = 0) -> pd.DataFrame:
    """Daily sales of a few stores: a weekly cycle plus noise."""
    rng = np.random.default_rng(seed)
    frames = []
    for k in range(count):
        t = np.arange(days)
        frames.append(
            pd.DataFrame(
                {
                    "date": pd.Timestamp("2022-01-01") + pd.to_timedelta(t, unit="D"),
                    "store": f"S{k}",
                    "units": np.round(
                        50 + 10 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 2, days), 1
                    ),
                    "promo": np.where(rng.random(days) < 0.2, "True", "False").astype(object),
                }
            )
        )
    return pd.concat(frames, ignore_index=True).astype({"store": object})


def test_the_plan_finds_the_date_the_entity_the_step_and_the_length() -> None:
    found = seq.plan_sequences(stores(), ["date"], ["promo", "store"])
    assert (found.time, found.entity, found.step, found.length, found.sequences) == (
        "date",
        "store",
        "daily",
        24,
        4,
    )
    assert [c.name for c in found.candidates] == ["store"]  # promo repeats dates within a value
    single = seq.plan_sequences(stores(count=1), ["date"], [])
    assert single.entity is None and single.sequences == 5  # 120 days cut into windows of 24
    assert seq.plan_sequences(stores(), [], ["store"]).error == "Needs a date column."
    short = seq.plan_sequences(stores(days=12), ["date"], ["store"])
    assert short.length == 12


def test_windows_go_back_to_the_same_values() -> None:
    frame = stores(days=40, count=2)
    frame.loc[3, "units"] = np.nan
    learned = [
        encode.learn_column(n, encode.column_of(frame, n), t)  # type: ignore[arg-type]
        for n, t in (("units", "numeric_continuous"), ("promo", "categorical_nominal"))
    ]
    coded = encode.encode(frame, learned)
    layout = seq.layout_for(coded, learned)
    assert layout.width == 2 + 2  # units + its empty flag, promo's two levels
    found = seq.windows(frame, coded, layout, "date", "store", 10, seed=1)
    assert found.data.shape == (2 * 31, 10, layout.width)
    assert found.data.min() >= 0 and found.data.max() <= 1
    steps = seq.encode_steps(coded, layout)
    back = seq.decode_steps(steps, layout, np.random.default_rng(0))
    present = ~np.isnan(coded["units"])
    assert np.allclose(back["units"][present], coded["units"][present])
    assert np.array_equal(back["promo"], coded["promo"])


def test_held_out_rows_are_whole_histories() -> None:
    frame = stores(count=5)
    train, holdout = seq.split_histories(frame, "date", "store", seed=3)
    assert set(train["store"]).isdisjoint(set(holdout["store"]))
    assert bool(holdout["store"].nunique() == 1) and len(train) + len(holdout) == len(frame)
    series_train, series_holdout = seq.split_histories(stores(count=1), "date", None, seed=3)
    assert bool(series_train["date"].max() < series_holdout["date"].min())


def test_autocorrelation_sees_a_cycle_and_its_similarity_is_one_for_the_same_windows() -> None:
    t = np.arange(24)
    windows = np.stack([np.sin(2 * np.pi * (t + k) / 12) for k in range(30)])
    acf = seq.autocorrelation(windows, 10)
    assert acf[0] > 0.5 and acf[5] < -0.5
    frame = stores(days=60, count=2)
    learned = [encode.learn_column("units", encode.column_of(frame, "units"), "numeric_continuous")]
    coded = encode.encode(frame, learned)
    layout = seq.layout_for(coded, learned)
    real = seq.windows(frame, coded, layout, "date", "store", 14, seed=1)
    same = seq.autocorrelations(real, real)
    assert same[0].similarity == pytest.approx(1.0)


torch = pytest.importorskip("torch")

from quantilo.synthetic.deep.sequence_report import evaluate_sequences  # noqa: E402

SPECS = [
    Spec("date", "datetime", "model"),
    Spec("store", "categorical_nominal", "unique"),
    Spec("units", "numeric_continuous", "model"),
    Spec("promo", "categorical_nominal", "model"),
]


def settings(steps: int = 100, sequences: int = 6) -> SequenceSettings:
    return SequenceSettings("date", "store", 14, sequences, steps)


def test_timegan_makes_new_dated_histories_of_the_planned_length() -> None:
    frame = stores()
    messages: list[str] = []
    result = synthesize(
        frame,
        SPECS,
        "timegan",
        Options(
            rows=0, device="cpu", sequence=settings(), progress=lambda _s, m: messages.append(m)
        ),
    )
    out = result.frame
    assert len(out) == 6 * 14 and bool(out["store"].nunique() == 6)
    assert not encode.column_of(out, "store").isin(encode.column_of(frame, "store")).any()
    for _, history in out.groupby("store"):
        gaps = history["date"].diff().dropna()
        assert len(history) == 14 and bool((gaps == pd.Timedelta(days=1)).all())
    assert messages[-1] == "Joint training · step 100 / 100"
    checks = diagnostic.diagnose(out, result.learned, [], result.bounds, [s.name for s in SPECS])
    assert checks.score == 1.0
    assert result.fitted.device == "CPU" and result.fitted.steps == 100


def test_the_same_seed_on_the_cpu_gives_the_same_histories() -> None:
    frame = stores(days=40, count=3)
    options = Options(rows=0, device="cpu", seed=4, sequence=settings(steps=100, sequences=3))
    first = synthesize(frame, SPECS, "timegan", options).frame
    second = synthesize(frame, SPECS, "timegan", options).frame
    pd.testing.assert_frame_equal(first, second)


def test_a_stopped_training_raises_cancelled() -> None:
    with pytest.raises(Cancelled):
        synthesize(
            stores(),
            SPECS,
            "timegan",
            Options(rows=0, device="cpu", sequence=settings()),
            should_stop=lambda: True,
        )


def test_real_windows_copied_cannot_be_told_apart() -> None:
    frame = stores(days=200, count=8)
    train, holdout = seq.split_histories(frame, "date", "store", seed=1)
    rng = np.random.default_rng(0)
    copies = []
    for k in range(200):  # real windows at random starts, written like TimeGAN's histories
        names = sorted(set(encode.column_of(train, "store")))
        store = train[encode.column_of(train, "store") == names[int(rng.integers(len(names)))]]
        begin = int(rng.integers(0, len(store) - 14))
        copies.append(store.iloc[begin : begin + 14].assign(store=f"N{k}"))
    synthetic = pd.concat(copies, ignore_index=True)
    learned = [
        encode.learn_column(n, encode.column_of(train, n), t)  # type: ignore[arg-type]
        for n, t in (
            ("date", "datetime"),
            ("units", "numeric_continuous"),
            ("promo", "categorical_nominal"),
        )
    ]
    report = evaluate_sequences(
        train, holdout, synthetic, learned, "date", "store", 14, "daily", seed=2, allow_gpu=False
    )
    assert report.discriminative is not None and report.discriminative > 0.7
    assert report.autocorrelation is not None and report.autocorrelation > 0.95
    assert report.predictive_ratio is not None and report.samples[0].column == "units"


@pytest.mark.slow
def test_timegan_on_three_years_of_three_stores_within_three_minutes_on_the_gpu() -> None:
    import time

    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    frame = stores(days=1_095, count=3)
    started = time.perf_counter()
    result = synthesize(
        frame,
        SPECS,
        "timegan",
        Options(rows=0, device="gpu", sequence=SequenceSettings("date", "store", 24, 3)),
    )
    assert result.fitted.steps == 1_000 and "GPU" in (result.fitted.device or "")
    assert time.perf_counter() - started < 180


def test_dates_that_repeat_without_an_entity_are_flagged() -> None:
    found = seq.plan_sequences(stores(count=3), ["date"], [])
    assert (
        found.entity is None and found.warning is not None and "3.0 rows per date" in found.warning
    )
    assert seq.plan_sequences(stores(count=1), ["date"], []).warning is None
