"""CTGAN and TVAE as generation methods: coded columns in, coded columns out."""

from typing import Literal

import numpy as np

from quantilo.synthetic.deep import auto_epochs, transformer
from quantilo.synthetic.deep.device import pick, with_fallback
from quantilo.synthetic.encode import Coded, Learned
from quantilo.synthetic.methods import Cancelled, Fitted, Options, StopCheck


def generate(
    method: Literal["ctgan", "tvae"],
    coded: Coded,
    learned: list[Learned],
    options: Options,
    should_stop: StopCheck,
) -> tuple[Coded, Fitted]:
    import torch

    from quantilo.synthetic.deep.ctgan import CTGAN
    from quantilo.synthetic.deep.sampler import Sampler
    from quantilo.synthetic.deep.tvae import TVAE

    fitted_transform = transformer.fit(coded, learned, options.seed)
    data = transformer.transform(fitted_transform, coded, options.seed)
    rows = len(data)
    chosen = pick(options.device, options.allow_gpu)
    epochs = options.epochs or auto_epochs(rows, chosen.device.type != "cpu")
    spans = fitted_transform.spans

    def report(epoch: int, first: float, second: float) -> None:
        if options.progress is not None:
            losses = (
                f"loss G {first:.3f} · D {second:.3f}" if method == "ctgan" else f"loss {first:.3f}"
            )
            options.progress(epoch / epochs, f"Epoch {epoch} / {epochs} · {losses}")

    def fit(device: torch.device) -> np.ndarray:
        if method == "ctgan":
            sampler = Sampler(data, fitted_transform, np.random.default_rng(options.seed))
            model: CTGAN | TVAE = CTGAN(spans, sampler, device, options.seed)
        else:
            model = TVAE(spans, data, device, options.seed)
        model.fit(epochs, report, should_stop)
        if should_stop():
            raise Cancelled
        return model.sample(options.rows)

    if should_stop():
        raise Cancelled
    sampled, used = with_fallback(fit, chosen)
    notes = [n for n in (chosen.note, used.note) if n]
    if used.device.type != "cpu":
        notes.append("Trained on the GPU: the same seed may give slightly different data.")
    fitted = Fitted(method, notes=list(dict.fromkeys(notes)), device=used.label, epochs=epochs)
    return transformer.inverse(fitted_transform, sampled), fitted
