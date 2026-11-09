"""TimeGAN as a generation method: histories in, new histories out (023c)."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantilo.synthetic import sequences as seq
from quantilo.synthetic.deep import TIMEGAN_CPU_STEPS, TIMEGAN_GPU_STEPS
from quantilo.synthetic.deep.device import pick, with_fallback
from quantilo.synthetic.encode import Coded, Learned, encode
from quantilo.synthetic.methods import Cancelled, Fitted, Options, StopCheck

PHASES = ["Autoencoder", "Supervisor", "Joint training"]


@dataclass
class Generated:
    coded: Coded  # feature columns, rows = sequences * length
    times: np.ndarray  # seconds
    sequence: np.ndarray  # index of the sequence of each row
    fitted: Fitted


def auto_steps(gpu: bool) -> int:
    return TIMEGAN_GPU_STEPS if gpu else TIMEGAN_CPU_STEPS


def generate(
    frame: pd.DataFrame,
    learned: list[Learned],
    options: Options,
    should_stop: StopCheck,
) -> Generated:
    import torch

    from quantilo.synthetic.deep.timegan import TimeGAN

    plan = options.sequence
    if plan is None or plan.time is None:
        raise ValueError("TimeGAN needs a date column.")
    time, entity, length = plan.time, plan.entity, plan.length
    features = [i for i in learned if i.name not in (time, entity)]
    if not features:
        raise ValueError("TimeGAN needs at least one column besides the date and the entity.")
    coded = encode(frame, features)
    layout = seq.layout_for(coded, features)
    windows = seq.windows(frame, coded, layout, time, entity, length, options.seed)
    if len(windows.data) < 2:
        raise ValueError(f"Too few histories of {length} dates to learn from.")
    chosen = pick(options.device, options.allow_gpu)
    steps = plan.steps or auto_steps(chosen.device.type != "cpu")

    def report(phase: str, step: int, total: int) -> None:
        if options.progress is not None:
            done = (PHASES.index(phase) * total + step) / (len(PHASES) * total)
            options.progress(done, f"{phase} · step {step:,} / {total:,}")

    def fit(device: torch.device) -> np.ndarray:
        model = TimeGAN(layout.width, device, options.seed)
        if not model.fit(windows.data, steps, report, should_stop):
            raise Cancelled
        return model.sample(plan.sequences, length)

    if should_stop():
        raise Cancelled
    sampled, used = with_fallback(fit, chosen)
    rng = np.random.default_rng(options.seed)
    flat = sampled.reshape(-1, layout.width)
    decoded = seq.decode_steps(flat, layout, rng)

    stamps, groups = seq.order_rows(frame, time, entity)
    present = np.concatenate([stamps[g] for g in groups]) if groups else np.zeros(0)
    histories = [np.sort(stamps[g]) for g in groups]
    gaps = np.concatenate([np.diff(h) for h in histories if len(h) > 1] or [np.zeros(1)])
    median_gap = float(np.median(gaps[gaps > 0])) if (gaps > 0).any() else 86_400.0
    step = plan.step_seconds or median_gap
    latest = float(present.max()) - (length - 1) * step if present.size else 0.0
    starts_pool = present[present <= latest] if present.size else np.zeros(1)
    if starts_pool.size == 0:
        starts_pool = present[:1] if present.size else np.zeros(1)
    if entity is None:
        # one long series, as the original: the windows follow each other, and start again from
        # the first date when they would go past the real period (dates that repeat in the data)
        first = float(present.min())
        room = max(latest - first, 0.0) + step
        starts = first + np.mod(np.arange(plan.sequences) * length * step, room)
    else:
        starts = rng.choice(starts_pool, plan.sequences)
    times = (starts[:, None] + np.arange(length)[None, :] * step).reshape(-1)
    notes = [n for n in (chosen.note, used.note) if n]
    if used.device.type == "cpu" and plan.steps is None:
        notes.append("Trained on the CPU: a short training, the dynamics may be rough.")
    notes.append(
        f"{len(windows.data):,} real windows of {length} steps learned, {steps:,} steps per phase."
    )
    fitted = Fitted("timegan", notes=notes, device=used.label, steps=steps)
    return Generated(decoded, times, np.repeat(np.arange(plan.sequences), length), fitted)
