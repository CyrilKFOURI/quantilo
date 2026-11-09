"""Deep generators (023b): CTGAN and TVAE, the column transform, the sampler, the device."""

import numpy as np
import pandas as pd
import pytest

from quantilo.synthetic import diagnostic, encode
from quantilo.synthetic import quality as q
from quantilo.synthetic.deep import auto_epochs
from quantilo.synthetic.generate import Spec, synthesize
from quantilo.synthetic.methods import Cancelled, Options
from quantilo.synthetic.plan import DEEP_MISSING, PlanColumn, make_plan

torch = pytest.importorskip("torch")

from quantilo.synthetic.deep import transformer  # noqa: E402
from quantilo.synthetic.deep.device import Device, pick, with_fallback  # noqa: E402
from quantilo.synthetic.deep.sampler import Sampler  # noqa: E402


def bimodal(n: int = 2_000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    low = rng.random(n) < 0.5
    value = np.where(low, rng.normal(10, 1, n), rng.normal(40, 2, n)).round(2)
    value[rng.random(n) < 0.1] = np.nan
    level = np.where(low, "a", "b").astype(object)
    level[rng.random(n) < 0.02] = "rare"
    return pd.DataFrame(
        {"value": value, "level": level, "count": rng.integers(0, 6, n).astype(float)}
    )


SPECS = [
    Spec("value", "numeric_continuous", "model"),
    Spec("level", "categorical_nominal", "model"),
    Spec("count", "numeric_discrete", "model"),
]


def learned_and_coded(frame: pd.DataFrame):
    learned = [encode.learn_column(s.name, encode.column_of(frame, s.name), s.type) for s in SPECS]
    return learned, encode.encode(frame, learned)


def test_automatic_epochs_follow_a_budget_of_rows_seen() -> None:
    assert auto_epochs(5_000, gpu=True) == 300
    assert auto_epochs(50_000, gpu=True) == 120
    assert auto_epochs(5_000, gpu=False) == 300
    assert auto_epochs(50_000, gpu=False) == 30
    assert auto_epochs(20_000, gpu=False) == 75


def test_deep_methods_say_why_they_cannot_run_without_pytorch() -> None:
    columns = [PlanColumn("v", "numeric_continuous", None, 100, 100, 50)]
    missing = {m.key: m for m in make_plan(columns, 100, deep_installed=False).methods}
    assert not missing["ctgan"].available and missing["ctgan"].unavailable == DEEP_MISSING
    ready = {m.key: m for m in make_plan(columns, 100, deep_installed=True, gpu="RTX").methods}
    assert ready["tvae"].available and ready["tvae"].badge == "Deep · GPU"
    assert ready["cart"].badge is None


def test_the_transform_finds_the_modes_and_goes_back_exactly() -> None:
    frame = bimodal()
    learned, coded = learned_and_coded(frame)
    fitted = transformer.fit(coded, learned, seed=1)
    value = fitted.columns[0]
    assert len(value.means) == 2 and value.empty
    assert sorted(np.round(value.means)) == [10, 40]
    data = transformer.transform(fitted, coded, seed=1)
    assert data.shape == (len(frame), fitted.width)
    back = transformer.inverse(fitted, data)
    original = coded["value"]
    both = ~np.isnan(original)
    assert np.array_equal(np.isnan(back["value"]), np.isnan(original))
    assert np.allclose(back["value"][both], original[both], atol=1e-4)
    assert np.array_equal(np.isnan(back["level"]), np.isnan(coded["level"]))


def test_the_sampler_conditions_on_rare_levels_and_generates_at_real_frequency() -> None:
    frame = bimodal(5_000)
    learned, coded = learned_and_coded(frame)
    fitted = transformer.fit(coded, learned, seed=1)
    data = transformer.transform(fitted, coded, seed=1)
    sampler = Sampler(data, fitted, np.random.default_rng(0))
    assert sampler.width == fitted.columns[1].width
    condition = sampler.train_condition(20_000)
    assert condition is not None
    _vector, _, columns, levels = condition
    rare = learned[1].levels.index("rare")
    training_share = (levels == rare).mean()
    assert training_share > 0.15  # log-frequency: far above the real 2 %
    real = sampler.real_rows(columns[:200], levels[:200], 200)
    start = fitted.columns[0].width
    assert np.array_equal(real[:, start : start + sampler.width].argmax(axis=1), levels[:200])
    generated = sampler.generation_condition(20_000)
    assert generated is not None
    assert abs(generated[:, rare].mean() - 0.02) < 0.01


@pytest.mark.parametrize("method", ["ctgan", "tvae"])
def test_a_short_training_gives_valid_data_with_the_rare_level(method: str) -> None:
    frame = bimodal(2_000)
    messages: list[str] = []
    result = synthesize(
        frame,
        SPECS,
        method,  # type: ignore[arg-type]
        Options(rows=1_500, epochs=60, device="cpu", progress=lambda _s, m: messages.append(m)),
    )
    found = diagnostic.diagnose(
        result.frame, result.learned, [], result.bounds, [s.name for s in SPECS]
    )
    assert found.score == 1.0
    assert result.fitted.device == "CPU" and result.fitted.epochs == 60
    assert messages[-1].startswith("Epoch 60 / 60")
    assert "rare" in set(result.frame["level"].dropna())
    missing = result.frame["value"].isna().to_numpy().mean()
    assert abs(missing - frame["value"].isna().to_numpy().mean()) <= 0.08


def test_tvae_keeps_both_peaks() -> None:
    frame = bimodal(3_000)
    result = synthesize(frame, SPECS, "tvae", Options(rows=3_000, epochs=150, device="cpu"))
    real = encode.numbers(encode.column_of(frame, "value"))
    synthetic = encode.numbers(encode.column_of(result.frame, "value"))
    assert (q.ks_complement(real, synthetic) or 0) >= 0.8
    low = synthetic[~np.isnan(synthetic)] < 25
    assert 0.3 < low.mean() < 0.7


def test_the_same_seed_on_the_cpu_gives_the_same_data() -> None:
    frame = bimodal(600)
    options = Options(rows=300, epochs=15, device="cpu", seed=5)
    first = synthesize(frame, SPECS, "ctgan", options).frame
    second = synthesize(frame, SPECS, "ctgan", options).frame
    pd.testing.assert_frame_equal(first, second)


def test_a_stopped_training_raises_cancelled() -> None:
    with pytest.raises(Cancelled):
        synthesize(
            bimodal(600),
            SPECS,
            "tvae",
            Options(rows=100, epochs=50, device="cpu"),
            should_stop=lambda: True,
        )


def test_the_gpu_setting_decides_auto_and_memory_errors_fall_back_to_the_cpu() -> None:
    assert pick("auto", allow_gpu=False).label == "CPU"
    assert pick("cpu", allow_gpu=True).label == "CPU"

    def fit(device: object) -> str:
        if getattr(device, "type", "cpu") != "cpu":
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return "trained"

    result, used = with_fallback(fit, Device(torch.device("cuda"), "Some GPU"))
    assert result == "trained" and used.label == "CPU"
    assert used.note == "GPU ran out of memory: finished on the CPU."
    with pytest.raises(ValueError):
        with_fallback(
            lambda _d: (_ for _ in ()).throw(ValueError("other")),
            Device(torch.device("cuda"), "GPU"),
        )


@pytest.mark.slow
def test_ctgan_on_fifty_thousand_rows_trains_within_five_minutes_on_the_gpu() -> None:
    import time

    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    rng = np.random.default_rng(0)
    rows = 50_000
    frame = pd.DataFrame({f"n{i}": rng.normal(size=rows).round(3) for i in range(14)})
    for i in range(5):
        frame[f"c{i}"] = rng.choice(list("abcdefgh"), rows).astype(object)
    frame["day"] = pd.Timestamp("2020-01-01") + pd.to_timedelta(
        rng.integers(0, 1_500, rows), unit="D"
    )
    specs = [Spec(f"n{i}", "numeric_continuous", "model") for i in range(14)]
    specs += [Spec(f"c{i}", "categorical_nominal", "model") for i in range(5)]
    specs += [Spec("day", "datetime", "model")]
    started = time.perf_counter()
    result = synthesize(frame, specs, "ctgan", Options(rows=rows, device="gpu"))
    assert result.fitted.epochs == 120 and "GPU" in (result.fitted.device or "")
    assert time.perf_counter() - started < 300
