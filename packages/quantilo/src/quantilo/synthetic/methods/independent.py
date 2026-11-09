"""Independent columns: each column drawn alone from its real values (reference method)."""

import numpy as np

from quantilo.synthetic.encode import Coded, Learned
from quantilo.synthetic.methods import Cancelled, Fitted, Options, StopCheck, marginal


def generate(
    coded: Coded, learned: list[Learned], options: Options, should_stop: StopCheck
) -> tuple[Coded, Fitted]:
    rng = np.random.default_rng(options.seed)
    result: Coded = {}
    for item in learned:
        if should_stop():
            raise Cancelled
        result[item.name] = marginal(coded[item.name], options.rows, rng)
    return result, Fitted("independent", notes=["Every link between columns is lost."])
