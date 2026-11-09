"""New identifiers in the same format as the real ones (R-SYN-05), and fake values that keep
only the shape of real ones (R-SYN-06 option). Never a real value."""

import re
import string
from collections import Counter
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

PATTERN = re.compile(r"^(\D*?)(\d+)(\D*)$")
LOWER = list(string.ascii_lowercase)
UPPER = list(string.ascii_uppercase)
DIGITS = list(string.digits)


@dataclass
class IdFormat:
    """How the identifiers are written: "C" + 5 digits, plain numbers, or free shapes."""

    kind: str  # "numbered" | "number" | "shape"
    prefix: str = ""
    width: int = 0
    suffix: str = ""
    example: str = ""


def id_format(values: pd.Series) -> IdFormat:
    present = values.dropna()
    if present.empty:
        return IdFormat("number")
    if pd.api.types.is_numeric_dtype(present):
        return IdFormat("number", example=str(present.iloc[0]))
    text = present.astype(str)
    parts = [PATTERN.match(v) for v in text]
    shapes = Counter((m.group(1), len(m.group(2)), m.group(3)) for m in parts if m)
    if shapes:
        (prefix, width, suffix), count = shapes.most_common(1)[0]
        if count >= 0.9 * len(text):
            return IdFormat("numbered", prefix, width, suffix, text.iloc[0])
    return IdFormat("shape", example=text.iloc[0])


def _shape(value: str, rng: np.random.Generator) -> str:
    """Same characters classes, new characters: "AB-12" → "QF-80"."""
    draws = rng.integers(0, 260, size=len(value)).tolist()
    out = []
    for char, draw in zip(value, draws, strict=True):
        if char.isdigit():
            out.append(DIGITS[draw % 10])
        elif char.islower():
            out.append(LOWER[draw % 26])
        elif char.isupper():
            out.append(UPPER[draw % 26])
        else:
            out.append(char)
    return "".join(out)


def new_identifiers(existing: pd.Series, count: int, seed: int) -> pd.Series:
    """`count` unique values in the format of `existing`, none of them in `existing`."""
    rng = np.random.default_rng(seed)
    found = id_format(existing)
    present = existing.dropna()
    if found.kind == "number":
        taken = set(
            cast(pd.Series, pd.to_numeric(present, errors="coerce"))
            .dropna()
            .astype("int64")
            .tolist()
        )
        top = max([*taken, 0]) + 1
        space = max(top * 2, count * 4, 1_000)
        picked = _pick(space, taken, count, rng)
        return pd.Series(picked, dtype="int64")
    if found.kind == "numbered":
        text = present.astype(str)
        taken = {int(m.group(2)) for v in text if (m := PATTERN.match(v)) is not None}
        width = found.width
        while 10**width - len(taken) < count * 2:
            width += 1
        picked = _pick(10**width, taken, count, rng)
        return pd.Series([f"{found.prefix}{n:0{width}d}{found.suffix}" for n in picked])
    taken_text = set(present.astype(str))
    samples = present.astype(str).to_numpy()
    result: list[str] = []
    seen: set[str] = set()
    attempts = 0
    while len(result) < count:
        value = _shape(str(samples[rng.integers(len(samples))]), rng)
        attempts += 1
        if value not in taken_text and value not in seen:
            seen.add(value)
            result.append(value)
        elif attempts > count * 20:
            value = f"{value}-{len(result)}"
            if value not in taken_text and value not in seen:
                seen.add(value)
                result.append(value)
    return pd.Series(result)


def _pick(space: int, taken: set[int], count: int, rng: np.random.Generator) -> list[int]:
    """`count` distinct numbers in [0, space) outside `taken`."""
    picked: list[int] = []
    chosen: set[int] = set()
    while len(picked) < count:
        draw = rng.integers(0, space, size=(count - len(picked)) * 2 + 16)
        for n in draw.tolist():
            if n not in taken and n not in chosen:
                chosen.add(n)
                picked.append(n)
                if len(picked) == count:
                    break
    return picked


def fake_values(existing: pd.Series, count: int, seed: int) -> pd.Series:
    """Values with the shape of real ones (e-mails, phones, URLs) and the real empty rate."""
    rng = np.random.default_rng(seed)
    present = existing.dropna().astype(str).to_numpy()
    if present.size == 0:
        return pd.Series([None] * count, dtype=object)
    missing = float(existing.isna().mean())
    taken = set(present.tolist())
    values: list[str | None] = []
    for _ in range(count):
        if rng.random() < missing:
            values.append(None)
            continue
        value = _shape(str(present[rng.integers(present.size)]), rng)
        values.append(value if value not in taken else _shape(value, rng))
    return pd.Series(values, dtype=object)
