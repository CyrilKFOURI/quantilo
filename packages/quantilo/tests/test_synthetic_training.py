"""066: very advanced training of the deep generators (optimizer, schedule, clipping, EMA)."""

import math

import pytest

from quantilo.synthetic.deep import training as net

# --- the schedule: no torch needed ---------------------------------------------------------------


def test_constant_without_warmup_keeps_the_rate() -> None:
    assert [net.factor(s, 100, "constant", 0.0) for s in (0, 50, 99)] == [1.0, 1.0, 1.0]


def test_a_warmup_ramps_linearly_then_holds() -> None:
    assert net.factor(0, 100, "constant", 0.1) == pytest.approx(0.1)
    assert net.factor(4, 100, "constant", 0.1) == pytest.approx(0.5)
    assert net.factor(10, 100, "constant", 0.1) == 1.0


def test_cosine_decays_to_zero_after_its_warmup() -> None:
    assert net.factor(10, 110, "cosine", 10 / 110) == pytest.approx(1.0)
    assert net.factor(60, 110, "cosine", 10 / 110) == pytest.approx(0.5)
    assert net.factor(110, 110, "cosine", 10 / 110) == pytest.approx(0.0, abs=1e-9)


def test_one_cycle_rises_over_30_percent_by_default() -> None:
    """R-SYN-NET-03."""
    assert net.factor(0, 100, "one_cycle", 0.0) == pytest.approx(net.ONE_CYCLE_START)
    assert net.factor(30, 100, "one_cycle", 0.0) == pytest.approx(1.0)
    assert net.factor(65, 100, "one_cycle", 0.0) == pytest.approx(0.5)
    assert net.factor(10, 100, "one_cycle", 0.1) == pytest.approx(1.0)


def test_the_run_says_how_the_networks_were_trained() -> None:
    assert net.describe(None) is None
    text = net.describe(net.Training(schedule="cosine", warmup=0.05, clip_norm=1.0, ema=0.999))
    assert text is not None
    assert "AdamW" in text and "cosine" in text and "clipped at 1" in text and "EMA" in text


def test_settings_are_bounded() -> None:
    with pytest.raises(ValueError):
        net.Training(ema=0.5)
    with pytest.raises(ValueError):
        net.Training(warmup=0.9)


# --- with torch ----------------------------------------------------------------------------------


def test_each_optimizer_is_the_one_asked_for() -> None:
    torch = pytest.importorskip("torch")
    layer = torch.nn.Linear(3, 1)
    paper = net.Paper(1e-3, weight_decay=1e-5)
    params = list(layer.parameters())
    assert isinstance(net.make_optimizer(params, None, paper), torch.optim.Adam)
    adamw = net.make_optimizer(params, net.Training(), paper)
    assert isinstance(adamw, torch.optim.AdamW)
    assert adamw.param_groups[0]["weight_decay"] == net.ADAMW_DECAY  # R-SYN-NET-02
    sgd = net.make_optimizer(params, net.Training(optimizer="sgd_nesterov"), paper)
    assert isinstance(sgd, torch.optim.SGD) and sgd.param_groups[0]["nesterov"]
    assert sgd.param_groups[0]["lr"] == pytest.approx(1e-2)  # R-SYN-NET-01
    radam = net.make_optimizer(params, net.Training(optimizer="radam"), paper)
    assert isinstance(radam, torch.optim.RAdam)


def test_the_stepper_clips_and_follows_the_schedule() -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    layer = torch.nn.Linear(4, 1)
    chosen = net.Training(optimizer="adam", schedule="cosine", clip_norm=0.01)
    stepper = net.Stepper(list(layer.parameters()), chosen, net.Paper(0.1), total_steps=10)
    for _ in range(5):
        stepper.zero_grad()
        (layer(torch.randn(8, 4) * 100).pow(2).mean()).backward()
        stepper.step()
        norm = math.sqrt(sum(float(p.grad.norm() ** 2) for p in layer.parameters()))
        assert norm <= 0.01 + 1e-6
    assert stepper.optimizer.param_groups[0]["lr"] == pytest.approx(0.05, rel=1e-6)


def test_the_moving_average_lags_the_weights() -> None:
    torch = pytest.importorskip("torch")
    layer = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        layer.weight.fill_(0.0)
    ema = net.Ema(layer, 0.9)
    with torch.no_grad():
        layer.weight.fill_(1.0)
    ema.update(layer)
    ema.copy_to(layer)
    assert float(layer.weight[0, 0].detach()) == pytest.approx(0.1)


@pytest.mark.parametrize("method", ["tvae", "ctgan"])
def test_deep_generators_train_with_the_settings_and_say_so(method: str) -> None:
    pytest.importorskip("torch")
    import numpy as np
    import pandas as pd

    from quantilo.synthetic.generate import Spec, synthesize
    from quantilo.synthetic.methods import Options

    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"x": rng.normal(0, 1, 600), "g": rng.choice(["a", "b", "c"], 600)})
    specs = [Spec("x", "numeric_continuous", "model"), Spec("g", "categorical_nominal", "model")]
    chosen = net.Training(schedule="one_cycle", clip_norm=1.0, ema=0.99)
    result = synthesize(
        frame,
        specs,
        method,  # type: ignore[arg-type]
        Options(rows=300, epochs=10, device="cpu", training=chosen),
    )
    assert len(result.frame) == 300
    assert any(n.startswith("Networks trained with AdamW") for n in result.fitted.notes)
