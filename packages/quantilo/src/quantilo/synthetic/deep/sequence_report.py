"""The sequence measures of the TimeGAN paper, on real and synthetic histories (023c)."""

import numpy as np
import pandas as pd

from quantilo.synthetic import sequences as seq
from quantilo.synthetic.deep.device import pick
from quantilo.synthetic.encode import Learned, encode


def evaluate_sequences(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    synthetic: pd.DataFrame,
    learned: list[Learned],
    time: str,
    entity: str | None,
    length: int,
    step: str,
    seed: int,
    allow_gpu: bool = True,
) -> seq.SequenceReport:
    """`entity` is the column that names a history in all three frames (the synthetic ones have
    new names)."""
    features = [i for i in learned if i.name not in (time, entity)]
    coded_train = encode(train, features)
    layout = seq.layout_for(coded_train, features)
    real = seq.windows(train, coded_train, layout, time, entity, length, seed)
    held = seq.windows(holdout, encode(holdout, features), layout, time, entity, length, seed)
    # synthetic histories are written one after the other, `length` rows each
    fake = seq.windows_in_blocks(encode(synthetic, features), layout, length)
    columns = seq.autocorrelations(real, fake)
    report = seq.SequenceReport(
        time,
        entity,
        length,
        step,
        len(fake.data),
        len(real.data),
        float(np.mean([c.similarity for c in columns])) if columns else None,
        columns,
        samples=seq.samples(real, fake, seed),
    )
    if len(real.data) >= 10 and len(fake.data) >= 10:
        from quantilo.synthetic.deep.timegan import (
            discriminative_accuracy,
            predictive_error,
        )

        device = pick("auto", allow_gpu).device
        accuracy = discriminative_accuracy(real.data, fake.data, device, seed)
        report.discriminative_accuracy = accuracy
        report.discriminative = float(1 - 2 * abs(accuracy - 0.5))
        numeric = [s.start for s in layout.slots if s.kind == "number"]
        if numeric and len(held.data):
            report.predictive_real = predictive_error(real.data, held.data, numeric, device, seed)
            report.predictive_synthetic = predictive_error(
                fake.data, held.data, numeric, device, seed
            )
            if report.predictive_synthetic > 0:
                report.predictive_ratio = min(
                    1.5, report.predictive_real / report.predictive_synthetic
                )
    report.sentences = _sentences(report)
    return report


def _sentences(report: seq.SequenceReport) -> list[str]:
    out = []
    if report.autocorrelation is not None:
        quality = (
            "the same"
            if report.autocorrelation >= 0.9
            else "close"
            if report.autocorrelation >= 0.75
            else "a different"
        )
        out.append(
            f"The synthetic sequences have {quality} dynamics "
            f"(autocorrelation similarity {report.autocorrelation:.0%})."
        )
    if report.discriminative is not None:
        out.append(
            "A recurrent classifier tells real from synthetic windows "
            f"{report.discriminative_accuracy:.0%} of the time "
            "(50 % = impossible to tell)."
        )
    if report.predictive_ratio is not None:
        out.append(
            f"The synthetic sequences keep {min(report.predictive_ratio, 1):.0%} "
            "of the real predictive power."
        )
    out.append(f"TimeGAN learns windows of {report.length} steps: longer trends are not modelled.")
    return out
