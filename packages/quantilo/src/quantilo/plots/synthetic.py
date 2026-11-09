"""The Synthetic data page's figures (frontend/src/modules/synthetic/figures.ts): real in solid
dark grey, synthetic as an accent outline, so the two never depend on colour alone."""

import datetime as dt
import math
from typing import Any

from quantilo.plots.theme import LEGEND, Tokens, axis_title, with_alpha

Built = dict[str, Any]
PERCENT_AXIS = {"tickformat": ".0%", "rangemode": "tozero"}


def looks(tokens: Tokens) -> dict[str, dict[str, str]]:
    return {
        "real": {"name": "Real", "color": tokens.ink2, "fill": with_alpha(tokens.ink2, 0.45)},
        "synthetic": {
            "name": "Synthetic",
            "color": tokens.accent,
            "fill": with_alpha(tokens.accent, 0.12),
        },
    }


def _iso(seconds: float) -> str:
    return (dt.datetime(1970, 1, 1) + dt.timedelta(seconds=seconds)).isoformat()


def _bars(
    tokens: Tokens,
    x: list[Any],
    real: list[float],
    synthetic: list[float],
    width: list[float] | None = None,
) -> list[Built]:
    look = looks(tokens)
    common = {"width": width} if width is not None else {}
    return [
        {
            "type": "bar",
            "name": look["real"]["name"],
            "x": x,
            "y": real,
            **common,
            "marker": {"color": look["real"]["fill"], "line": {"width": 0}},
            "hovertemplate": "%{x}<br>real %{y:.1%}<extra></extra>",
        },
        {
            "type": "bar",
            "name": look["synthetic"]["name"],
            "x": x,
            "y": synthetic,
            **common,
            "marker": {
                "color": look["synthetic"]["fill"],
                "line": {"color": look["synthetic"]["color"], "width": 1.5},
            },
            "hovertemplate": "%{x}<br>synthetic %{y:.1%}<extra></extra>",
        },
    ]


def histogram_figure(detail: dict[str, Any], tokens: Tokens) -> Built:
    """Both sets on the same intervals (computed on the real data), as shares of each set."""
    found = detail.get("histogram") or {"edges": [], "real": [], "synthetic": []}
    dates = detail.get("type") == "datetime"
    edges = found["edges"]
    centres = [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)]
    widths = [edges[i + 1] - edges[i] for i in range(len(edges) - 1)]
    x = [_iso(c) for c in centres] if dates else centres
    width = [w * 1000 for w in widths] if dates else widths
    return {
        "data": _bars(tokens, x, found["real"], found["synthetic"], width),
        "layout": {
            **LEGEND,
            "barmode": "overlay",
            "bargap": 0,
            "yaxis": {**PERCENT_AXIS, **axis_title("Share of rows")},
        },
    }


def ecdf_figure(detail: dict[str, Any], tokens: Tokens) -> Built:
    """Cumulative curves: a cut tail shows as one curve reaching 100 % early."""
    look = looks(tokens)
    found = detail.get("ecdf") or {"real_x": [], "synthetic_x": [], "levels": []}
    dates = detail.get("type") == "datetime"

    def x(values: list[float]) -> list[Any]:
        return [_iso(v) for v in values] if dates else values

    return {
        "data": [
            {
                "type": "scatter",
                "mode": "lines",
                "name": look["real"]["name"],
                "x": x(found["real_x"]),
                "y": found["levels"],
                "line": {"color": look["real"]["color"], "width": 2, "shape": "hv"},
                "hovertemplate": "%{x}<br>real %{y:.0%} below<extra></extra>",
            },
            {
                "type": "scatter",
                "mode": "lines",
                "name": look["synthetic"]["name"],
                "x": x(found["synthetic_x"]),
                "y": found["levels"],
                "line": {
                    "color": look["synthetic"]["color"],
                    "width": 2,
                    "dash": "dash",
                    "shape": "hv",
                },
                "hovertemplate": "%{x}<br>synthetic %{y:.0%} below<extra></extra>",
            },
        ],
        "layout": {**LEGEND, "yaxis": {**PERCENT_AXIS, **axis_title("Share of rows below")}},
    }


def levels_figure(detail: dict[str, Any], tokens: Tokens) -> Built:
    """Levels side by side, horizontal, the most frequent in the real data on top."""
    look = looks(tokens)
    rows = list(reversed(detail["levels"]))
    y = [r["value"] for r in rows]
    return {
        "data": [
            {
                "type": "bar",
                "orientation": "h",
                "name": look["real"]["name"],
                "y": y,
                "x": [r["real_share"] for r in rows],
                "marker": {"color": look["real"]["fill"]},
                "hovertemplate": "%{y}<br>real %{x:.1%}<extra></extra>",
            },
            {
                "type": "bar",
                "orientation": "h",
                "name": look["synthetic"]["name"],
                "y": y,
                "x": [r["synthetic_share"] for r in rows],
                "marker": {
                    "color": look["synthetic"]["fill"],
                    "line": {"color": look["synthetic"]["color"], "width": 1.5},
                },
                "hovertemplate": "%{y}<br>synthetic %{x:.1%}<extra></extra>",
            },
        ],
        "layout": {
            **LEGEND,
            "barmode": "group",
            "bargap": 0.25,
            "xaxis": {**PERCENT_AXIS, "showgrid": True},
            "yaxis": {"showgrid": False},
        },
    }


def target_figure(target: dict[str, Any], tokens: Tokens) -> Built:
    """The target's mean (or share) per interval or level, in both sets: is the link kept?"""
    look = looks(tokens)
    return {
        "data": [
            {
                "type": "scatter",
                "mode": "lines+markers",
                "name": look["real"]["name"],
                "x": target["groups"],
                "y": target["real"],
                "line": {"color": look["real"]["color"], "width": 2},
                "marker": {"size": 8},
                "connectgaps": True,
            },
            {
                "type": "scatter",
                "mode": "lines+markers",
                "name": look["synthetic"]["name"],
                "x": target["groups"],
                "y": target["synthetic"],
                "line": {"color": look["synthetic"]["color"], "width": 2, "dash": "dash"},
                "marker": {"size": 8, "symbol": "circle-open"},
                "connectgaps": True,
            },
        ],
        "layout": {**LEGEND, "yaxis": axis_title(target["measure"]), "xaxis": {"type": "category"}},
    }


def column_figure(detail: dict[str, Any], tokens: Tokens, view: str = "histogram") -> Built:
    """A column, real against synthetic: levels for a category, histogram or ECDF otherwise."""
    if detail.get("levels") and not detail.get("histogram"):
        return levels_figure(detail, tokens)
    return ecdf_figure(detail, tokens) if view == "ecdf" else histogram_figure(detail, tokens)


def links_figure(links: dict[str, Any], tokens: Tokens, which: str = "gap") -> Built:
    """An association matrix: signed blue–red for real and synthetic, |gap| capped at 0.3."""
    gap = which == "gap"
    return {
        "data": [
            {
                "type": "heatmap",
                "x": links["columns"],
                "y": links["columns"],
                "z": links[which],
                "zmin": 0 if gap else -1,
                "zmax": 0.3 if gap else 1,
                "colorscale": [[0, tokens.surface], [1, tokens.accent]]
                if gap
                else [[0, tokens.palette[1]], [0.5, tokens.surface], [1, tokens.palette[0]]],
                "xgap": 1,
                "ygap": 1,
                "showscale": False,
                "hovertemplate": "%{y} · %{x}<br>%{z:.2f}<extra></extra>",
            }
        ],
        "layout": {
            "xaxis": {"showgrid": False, "tickangle": -45, "tickfont": {"size": 10}},
            "yaxis": {"showgrid": False, "autorange": "reversed", "tickfont": {"size": 10}},
        },
    }


def pair_figure(pair: dict[str, Any], tokens: Tokens) -> Built:
    """A pair in both sets: scatter (two numbers), boxes per level (mixed)."""
    look = looks(tokens)
    if pair["kind"] == "numbers":
        return {
            "data": [
                {
                    "type": "scattergl",
                    "mode": "markers",
                    "name": look["real"]["name"],
                    "x": [p[0] for p in pair["real_points"]],
                    "y": [p[1] for p in pair["real_points"]],
                    "marker": {"color": look["real"]["fill"], "size": 5},
                },
                {
                    "type": "scattergl",
                    "mode": "markers",
                    "name": look["synthetic"]["name"],
                    "x": [p[0] for p in pair["synthetic_points"]],
                    "y": [p[1] for p in pair["synthetic_points"]],
                    "marker": {
                        "color": "rgba(0,0,0,0)",
                        "size": 6,
                        "line": {"color": look["synthetic"]["color"], "width": 1},
                    },
                },
            ],
            "layout": {**LEGEND, "xaxis": axis_title(pair["a"]), "yaxis": axis_title(pair["b"])},
        }

    def box(side: str) -> Built:
        boxes = pair["boxes"]
        return {
            "type": "box",
            "name": look[side]["name"],
            "x": [b["group"] for b in boxes],
            "lowerfence": [b[side][0] for b in boxes],
            "q1": [b[side][1] for b in boxes],
            "median": [b[side][2] for b in boxes],
            "q3": [b[side][3] for b in boxes],
            "upperfence": [b[side][4] for b in boxes],
            "fillcolor": look[side]["fill"],
            "line": {"color": look[side]["color"], "width": 1.5},
        }

    return {
        "data": [box("real"), box("synthetic")],
        "layout": {
            **LEGEND,
            "boxmode": "group",
            "yaxis": axis_title(pair["b"]),
            "xaxis": {"type": "category"},
        },
    }


def dcr_figure(histogram: dict[str, Any], tokens: Tokens) -> Built:
    """Distances to the closest training row: synthetic rows against held-out real rows."""
    look = looks(tokens)
    edges = histogram["edges"]
    x = [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)]
    return {
        "data": [
            {
                "type": "bar",
                "name": "Held-out real rows",
                "x": x,
                "y": histogram["holdout"],
                "marker": {"color": look["real"]["fill"]},
                "hovertemplate": "distance %{x:.2f}<br>held-out %{y:.1%}<extra></extra>",
            },
            {
                "type": "bar",
                "name": look["synthetic"]["name"],
                "x": x,
                "y": histogram["synthetic"],
                "marker": {
                    "color": look["synthetic"]["fill"],
                    "line": {"color": look["synthetic"]["color"], "width": 1.5},
                },
                "hovertemplate": "distance %{x:.2f}<br>synthetic %{y:.1%}<extra></extra>",
            },
        ],
        "layout": {
            **LEGEND,
            "barmode": "overlay",
            "bargap": 0,
            "xaxis": axis_title("Distance to the closest training row"),
            "yaxis": {**PERCENT_AXIS, **axis_title("Share of rows")},
        },
    }


def mean_spread_figure(points: list[dict[str, Any]], tokens: Tokens, which: str = "mean") -> Built:
    """One point per numeric column: real on x, synthetic on y, log scales, the diagonal."""
    look = looks(tokens)
    real = [abs(p["real_mean"] if which == "mean" else p["real_std"]) for p in points]
    synthetic = [
        abs(p["synthetic_mean"] if which == "mean" else p["synthetic_std"]) for p in points
    ]
    positive = [v for v in [*real, *synthetic] if v > 0]
    low, high = (min(positive), max(positive)) if positive else (0.1, 1)
    far = [
        r > 0 and s > 0 and abs(math.log10(s / r)) > 0.05
        for r, s in zip(real, synthetic, strict=False)
    ]
    label = "|mean|" if which == "mean" else "std. deviation"
    return {
        "data": [
            {
                "type": "scatter",
                "mode": "lines",
                "x": [low, high],
                "y": [low, high],
                "line": {"color": tokens.ink3, "dash": "dot", "width": 1},
                "hoverinfo": "skip",
                "showlegend": False,
            },
            {
                "type": "scatter",
                "mode": "text+markers",
                "x": real,
                "y": synthetic,
                "text": [p["column"] if far[i] else "" for i, p in enumerate(points)],
                "textposition": "top center",
                "textfont": {"size": 10, "color": tokens.ink2},
                "customdata": [p["column"] for p in points],
                "marker": {
                    "color": look["synthetic"]["fill"],
                    "size": 9,
                    "line": {"color": look["synthetic"]["color"], "width": 1.5},
                },
                "hovertemplate": f"%{{customdata}}<br>real {which} %{{x:,.4g}}<br>synthetic "
                f"{which} %{{y:,.4g}}<extra></extra>",
                "showlegend": False,
            },
        ],
        "layout": {
            "xaxis": {"type": "log", **axis_title(f"Real {label}")},
            "yaxis": {"type": "log", **axis_title(f"Synthetic {label}")},
        },
    }


def pca_figure(pca: dict[str, Any], tokens: Tokens) -> Built:
    """Two components fitted on the real data; both sets projected in that plane."""
    look = looks(tokens)
    explained = pca["explained"]

    def percent(i: int) -> str:
        return f"{round((explained[i] if i < len(explained) else 0) * 100)} %"

    return {
        "data": [
            {
                "type": "scattergl",
                "mode": "markers",
                "name": look["real"]["name"],
                "x": [p[0] for p in pca["real"]],
                "y": [p[1] for p in pca["real"]],
                "marker": {"color": look["real"]["color"], "size": 5, "opacity": 0.6},
                "hoverinfo": "skip",
            },
            {
                "type": "scattergl",
                "mode": "markers",
                "name": look["synthetic"]["name"],
                "x": [p[0] for p in pca["synthetic"]],
                "y": [p[1] for p in pca["synthetic"]],
                "marker": {
                    "color": "rgba(0,0,0,0)",
                    "size": 5,
                    "line": {"color": look["synthetic"]["color"], "width": 1},
                },
                "hoverinfo": "skip",
            },
        ],
        "layout": {
            **LEGEND,
            "xaxis": axis_title(f"Component 1 ({percent(0)})"),
            "yaxis": axis_title(f"Component 2 ({percent(1)})"),
        },
    }


SCORES = ["diagnostic", "quality", "shapes", "trends", "detection", "privacy", "utility"]


def scores_figure(runs: list[tuple[str, dict[str, Any]]], tokens: Tokens) -> Built:
    """Scores per run, grouped bars (one colour per run, fixed palette order)."""
    return {
        "data": [
            {
                "type": "bar",
                "name": name,
                "x": [key.capitalize() for key in SCORES],
                "y": [scores.get(key) for key in SCORES],
                "marker": {"color": tokens.palette[i % 8]},
                "hovertemplate": "%{x}: %{y:.0%}<extra>%{fullData.name}</extra>",
            }
            for i, (name, scores) in enumerate(runs)
        ],
        "layout": {**LEGEND, "barmode": "group", "yaxis": {**PERCENT_AXIS, "range": [0, 1]}},
    }


def acf_figure(column: dict[str, Any], tokens: Tokens) -> Built:
    """Mean autocorrelation by lag, real and synthetic (TimeGAN)."""
    look = looks(tokens)
    lags = list(range(1, len(column["real"]) + 1))
    return {
        "data": [
            {
                "type": "scatter",
                "mode": "lines+markers",
                "name": look["real"]["name"],
                "x": lags,
                "y": column["real"],
                "line": {"color": look["real"]["color"], "width": 2},
                "marker": {"size": 5},
            },
            {
                "type": "scatter",
                "mode": "lines+markers",
                "name": look["synthetic"]["name"],
                "x": lags,
                "y": column["synthetic"],
                "line": {"color": look["synthetic"]["color"], "width": 2, "dash": "dash"},
                "marker": {"size": 6, "symbol": "circle-open"},
            },
        ],
        "layout": {
            **LEGEND,
            "xaxis": {**axis_title("Lag (steps)"), "dtick": 1},
            "yaxis": {
                "range": [-1, 1],
                "zeroline": True,
                "zerolinecolor": tokens.ink3,
                **axis_title("Autocorrelation"),
            },
        },
    }


def samples_figure(sample: dict[str, Any], tokens: Tokens) -> Built:
    """A few real and synthetic histories of one column, step by step (TimeGAN)."""
    look = looks(tokens)
    data = []
    for side in ("real", "synthetic"):
        for i, values in enumerate(sample[side]):
            data.append(
                {
                    "type": "scatter",
                    "mode": "lines",
                    "name": look[side]["name"],
                    "legendgroup": side,
                    "showlegend": i == 0,
                    "x": list(range(1, len(values) + 1)),
                    "y": values,
                    "line": {
                        "color": look[side]["color"],
                        "width": 1.5,
                        "dash": "solid" if side == "real" else "dash",
                    },
                    "opacity": 0.7 if side == "real" else 0.9,
                }
            )
    return {
        "data": data,
        "layout": {**LEGEND, "xaxis": axis_title("Step"), "yaxis": axis_title(sample["column"])},
    }
