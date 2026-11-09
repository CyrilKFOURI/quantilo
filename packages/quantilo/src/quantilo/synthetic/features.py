"""Typed columns as model features: numbers and dates as floats, categories as shared codes."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantilo.synthetic.encode import Learned, column_of, labels
from quantilo.synthetic.quality import as_numbers

MAX_LEVELS = 250  # histogram gradient boosting handles at most 255 categories per feature


@dataclass
class Features:
    names: list[str]
    categorical: list[bool]
    frames: list[np.ndarray]  # one matrix per input frame, same columns


def features(frames: list[pd.DataFrame], learned: list[Learned]) -> Features:
    """The same columns for every frame; levels coded on all frames together (frequent first),
    the rarest grouped beyond MAX_LEVELS."""
    columns: list[list[np.ndarray]] = [[] for _ in frames]
    for item in learned:
        if item.categorical:
            texts = [labels(column_of(f, item.name)) for f in frames]
            counts = pd.concat(texts).value_counts()
            index = {level: min(i, MAX_LEVELS) for i, level in enumerate(counts.index)}
            for slot, text in enumerate(texts):
                columns[slot].append(
                    np.array([np.nan if v is None else index[v] for v in text], dtype=float)
                )
        else:
            for slot, frame in enumerate(frames):
                columns[slot].append(as_numbers(column_of(frame, item.name), item))
    return Features(
        [i.name for i in learned],
        [i.categorical for i in learned],
        [
            np.column_stack(c) if c else np.empty((len(f), 0))
            for c, f in zip(columns, frames, strict=True)
        ],
    )
