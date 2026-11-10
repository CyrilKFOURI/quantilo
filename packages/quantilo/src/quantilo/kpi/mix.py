"""Mix effect and Simpson's paradox of a rate split by a group (054, P14).

A total rate R = Σ s_g r_g, with s_g the group's share of the denominator (rows of a mean, the
denominator of a ratio) and r_g its own rate. Between two periods, the symmetric shift-share
decomposition splits R₁ - R₀ exactly into a rate effect Σ ½(s₀ + s₁)(r₁ - r₀) (the groups' own
rates moved) and a mix effect Σ ½(r₀ + r₁)(s₁ - s₀) (their weights moved). Simpson's paradox
(R-SIG-08): the total moves one way while groups weighing at least 2/3 move the other.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from quantilo.kpi.signals import Signal, SignalCell, number

SIMPSON = 2 / 3
TOLERANCE = 0.01  # rate + mix must give the total's change within 1 % (shared keys break it)


@dataclass
class Part:
    label: str
    r0: float | None  # the group's rate in the earlier and the later period
    r1: float | None
    w0: float  # its weight (the denominator's size) in each period
    w1: float


@dataclass
class Decomposition:
    change: float  # R₁ - R₀
    rate: float
    mix: float
    opposite: float  # weight share of the groups moving against the total


def decompose(r0: float, r1: float, parts: Sequence[Part]) -> Decomposition | None:
    total0 = sum(p.w0 for p in parts if p.w0 > 0)
    total1 = sum(p.w1 for p in parts if p.w1 > 0)
    if total0 <= 0 or total1 <= 0:
        return None
    rate = mix = 0.0
    against = 0.0
    change = r1 - r0
    for part in parts:
        s0 = part.w0 / total0 if part.w0 > 0 else 0.0
        s1 = part.w1 / total1 if part.w1 > 0 else 0.0
        a = part.r0 if s0 else None
        b = part.r1 if s1 else None
        if a is None and b is None:
            continue
        a = b if a is None else a  # a group entering or leaving moves the mix only
        b = a if b is None else b
        assert a is not None and b is not None
        rate += 0.5 * (s0 + s1) * (b - a)
        mix += 0.5 * (a + b) * (s1 - s0)
        if s0 and s1 and change and (b - a) * change < 0:
            against += 0.5 * (s0 + s1)
    scale = max(abs(change), 1e-9 * max(abs(r0), abs(r1), 1.0))
    if abs(rate + mix - change) > TOLERANCE * scale:
        return None
    present = sum(0.5 * (p.w0 / total0 + p.w1 / total1) for p in parts if p.w0 > 0 and p.w1 > 0)
    return Decomposition(change, rate, mix, against / present if present else 0.0)


def _signed(value: float, percent: bool) -> str:
    if percent:
        return "0.0 pts" if abs(value) < 0.0005 else f"{value * 100:+.1f} pts"
    text = number(value)
    return text if text.startswith("-") else f"+{text}"


def mix_signal(
    measure: str,
    label: str,
    percent: bool,
    split: str,
    before: str,
    after_start: str,
    r0: float | None,
    r1: float | None,
    parts: Sequence[Part],
) -> Signal | None:
    """R-SIG-07 (or R-SIG-08 when the groups mostly move against the total) between the two
    last complete periods; `before` is the earlier period's label."""
    if r0 is None or r1 is None or r0 == r1:
        return None
    found = decompose(r0, r1, parts)
    if found is None:
        return None
    sentence = (
        f"{label} {_signed(found.change, percent)} vs {before}: rate effect "
        f"{_signed(found.rate, percent)}, mix effect {_signed(found.mix, percent)}"
    )
    cells = [SignalCell(after_start, None, None, found.change)]
    if found.opposite >= SIMPSON:
        way, other = ("falls", "rise") if found.change < 0 else ("rises", "fall")
        return Signal(
            "simpson",
            "warning",
            "change",
            "R-SIG-08",
            measure,
            f"Simpson's paradox: {label} {way} while groups weighing "
            f"{found.opposite * 100:.0f} % of {split} {other}; {sentence} (R-SIG-08).",
            cells,
            round(found.opposite, 4),
        )
    return Signal(
        "mix",
        "info",
        "change",
        "R-SIG-07",
        measure,
        f"{sentence} (R-SIG-07).",
        cells,
        round(found.mix, 6),
    )
