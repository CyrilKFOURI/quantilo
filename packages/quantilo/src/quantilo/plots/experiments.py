"""The Experiments page's figures (frontend/src/modules/experiments/figures.ts): the interval of
each variant's difference against zero, the cumulative difference over time, and the sample size
curve of the Plan section.

A variant keeps its colour in every chart: the i-th variant after the control takes the i-th
colour of the categorical palette (the control is the reference, never a series)."""

from typing import Any

from quantilo.plots.theme import LEGEND, Tokens, axis_title, with_alpha

Built = dict[str, Any]


def _scale(kind: str) -> float:
    """Rates are drawn in percentage points."""
    return 100.0 if kind == "rate" else 1.0


def _unit(kind: str) -> str:
    return " pts" if kind == "rate" else ""


def _axis_unit(kind: str) -> str:
    return " (pts)" if kind == "rate" else ""


def with_zero(values: list[float]) -> list[float]:
    """An axis range that always shows zero, the reference, with some air on both sides."""
    low, high = min([0.0, *values]), max([0.0, *values])
    pad = (high - low or 1) * 0.08
    return [low - pad, high + pad]


def _signed(value: float, digits: int = 1) -> str:
    text = f"{abs(value):,.{digits}f}"
    return f"+{text}" if value >= 0 else f"−{text}"


def variant_color(tokens: Tokens, index: int) -> str:
    """The colour of the index-th variant after the control."""
    return tokens.palette[index % len(tokens.palette)]


def intervals_figure(metric: dict[str, Any], tokens: Tokens) -> Built:
    """Each variant's difference to the control with its interval, against a zero line; the
    lift written at the end of the interval."""
    kind, scale = metric["kind"], _scale(metric["kind"])
    data: list[Built] = []
    labels: list[str] = []
    ends: list[float] = []
    for i, c in enumerate(metric["comparisons"]):
        if c["diff"] is None:
            continue
        color = variant_color(tokens, i)
        label = c["variant"]
        labels.append(label)
        low = c["diff_low"] * scale if c["diff_low"] is not None else None
        high = c["diff_high"] * scale if c["diff_high"] is not None else None
        diff = c["diff"] * scale
        ends += [v for v in (diff, low, high) if v is not None]
        lift = f" ({_signed(c['lift'] * 100)} %)" if c["lift"] is not None else ""
        text = f"{_signed(diff, 2 if abs(diff) < 10 else 1)}{_unit(kind)}{lift}"
        if low is not None and high is not None:
            data.append(
                {
                    "type": "scatter",
                    "mode": "lines",
                    "x": [low, high],
                    "y": [label, label],
                    "line": {"color": color, "width": 2},
                    "hoverinfo": "skip",
                }
            )
        p = c["p_adjusted"] if c["p_adjusted"] is not None else c["p_value"]
        data.append(
            {
                "type": "scatter",
                "mode": "markers+text",
                "x": [diff],
                "y": [label],
                "text": [text],
                "textposition": "top center",
                "textfont": {"size": 11, "color": tokens.ink2},
                "marker": {
                    "color": color,
                    "size": 10,
                    "line": {"color": tokens.surface, "width": 2},
                },
                "customdata": [
                    [
                        c["control"],
                        "—" if low is None else f"{low:,.3g}",
                        "—" if high is None else f"{high:,.3g}",
                        "—" if p is None else f"{p:.3g}",
                    ]
                ],
                "hovertemplate": (
                    f"{label} vs %{{customdata[0]}}<br>difference %{{x:,.3g}}{_unit(kind)}"
                    "<br>CI [%{customdata[1]} ; %{customdata[2]}]<br>p = %{customdata[3]}"
                    "<extra></extra>"
                ),
            }
        )
    return {
        "data": data,
        "layout": {
            "xaxis": {
                **axis_title(f"Difference to the control{_axis_unit(kind)}"),
                "range": with_zero(ends),
                "showgrid": True,
                "zeroline": True,
                "zerolinecolor": tokens.ink3,
                "zerolinewidth": 1,
            },
            "yaxis": {
                "type": "category",
                "showgrid": False,
                "autorange": "reversed",
                "categoryorder": "array",
                "categoryarray": labels,
            },
            "margin": {"t": 12},
        },
    }


def cumulative_figure(cumulative: dict[str, Any], tokens: Tokens) -> Built:
    """The difference on every unit seen so far, period by period, with its interval."""
    kind, scale = cumulative["kind"], _scale(cumulative["kind"])
    data: list[Built] = []

    def scaled(values: list[float | None]) -> list[float | None]:
        return [None if v is None else v * scale for v in values]

    for i, line in enumerate(cumulative["lines"]):
        color = variant_color(tokens, i)
        x = line["periods"]
        data.append(
            {
                "type": "scatter",
                "mode": "lines",
                "x": x,
                "y": scaled(line["low"]),
                "line": {"width": 0},
                "hoverinfo": "skip",
                "showlegend": False,
            }
        )
        data.append(
            {
                "type": "scatter",
                "mode": "lines",
                "x": x,
                "y": scaled(line["high"]),
                "fill": "tonexty",
                "fillcolor": with_alpha(color, 0.15),
                "line": {"width": 0},
                "hoverinfo": "skip",
                "showlegend": False,
            }
        )
        data.append(
            {
                "type": "scatter",
                "mode": "lines",
                "name": f"{line['variant']} − control",
                "x": x,
                "y": scaled(line["diff"]),
                "line": {"color": color, "width": 2},
                "hovertemplate": f"%{{x}}<br>{line['variant']}: %{{y:,.3g}}{_unit(kind)}"
                "<extra></extra>",
            }
        )
    layout: Built = {
        "xaxis": {"type": "date", "nticks": 6, "tickangle": 0},
        "yaxis": {
            **axis_title(f"Cumulative difference{_axis_unit(kind)}"),
            "zeroline": True,
            "zerolinecolor": tokens.ink3,
            "zerolinewidth": 1,
        },
        "hovermode": "x unified",
    }
    if len(cumulative["lines"]) > 1:
        layout = {**layout, **LEGEND}
    return {"data": data, "layout": layout}


def sample_size_figure(plan: dict[str, Any], tokens: Tokens) -> Built:
    """Units per group (log scale) against the smallest lift they detect, the plan marked."""
    points = [p for p in plan["curve"] if p["lift"] is not None]
    relative = plan["effect_relative"] is not None
    y = [abs(p["lift"] if relative else p["absolute"]) * (100 if relative else 1) for p in points]
    unit = " %" if relative else ""
    color = tokens.palette[0]
    target = abs(plan["effect_relative"] if relative else plan["effect_absolute"])
    target_y = target * (100 if relative else 1)
    return {
        "data": [
            {
                "type": "scatter",
                "mode": "lines",
                "x": [p["per_group"] for p in points],
                "y": y,
                "line": {"color": color, "width": 2},
                "hovertemplate": f"%{{x:,}} units per group<br>detects ±%{{y:.3g}}{unit}"
                "<extra></extra>",
            },
            {
                "type": "scatter",
                "mode": "markers+text",
                "x": [plan["control"]],
                "y": [target_y],
                "text": [f"Your plan: {plan['control']:,} per group"],
                "textposition": "top right",
                "textfont": {"size": 11, "color": tokens.ink1},
                "marker": {
                    "color": color,
                    "size": 10,
                    "line": {"color": tokens.surface, "width": 2},
                },
                "hovertemplate": f"%{{x:,}} units per group<br>detects ±%{{y:.3g}}{unit}"
                "<extra>plan</extra>",
            },
        ],
        "layout": {
            "xaxis": {**axis_title("Units per group (control)"), "type": "log", "showgrid": True},
            "yaxis": {
                **axis_title("Smallest detectable lift" + (" (%)" if relative else "")),
                "rangemode": "tozero",
            },
            "margin": {"t": 12},
        },
    }
