"""The Variable page's figures (frontend/src/features/variable/figures.ts) and the correlation
matrix (frontend/src/features/correlations/figures.ts)."""

import datetime as dt
from typing import Any

from quantilo.plots.theme import LEGEND, Tokens, axis_title, format_number, with_alpha

Built = dict[str, Any]

METRIC_LABELS = {
    "pearson": "Pearson r",
    "spearman": "Spearman ρ",
    "cramers_v": "Cramér's V",
    "eta": "η (correlation ratio)",
}


def kde_figure(numeric: dict[str, Any], tokens: Tokens) -> Built:
    curve = numeric.get("kde")
    if not curve:
        return {"data": []}
    return {
        "data": [
            {
                "type": "scatter",
                "mode": "lines",
                "x": curve["x"],
                "y": curve["y"],
                "line": {"color": tokens.numeric, "width": 2, "shape": "spline"},
                "fill": "tozeroy",
                "fillcolor": with_alpha(tokens.numeric, 0.15),
                "hovertemplate": "%{x:,.4g}<br>density %{y:.3g}<extra></extra>",
            }
        ],
        "layout": {"yaxis": axis_title("Density")},
    }


def ecdf_figure(numeric: dict[str, Any], tokens: Tokens) -> Built:
    curve = numeric.get("ecdf")
    if not curve:
        return {"data": []}
    return {
        "data": [
            {
                "type": "scatter",
                "mode": "lines",
                "x": curve["x"],
                "y": curve["y"],
                "line": {"color": tokens.numeric, "width": 2, "shape": "hv"},
                "hovertemplate": "%{y:.0%} of values ≤ %{x:,.4g}<extra></extra>",
            }
        ],
        "layout": {"yaxis": {"tickformat": ".0%", "range": [0, 1.02]}},
    }


def qq_figure(
    qq: dict[str, Any] | None, tokens: Tokens, axis_label: str = "Normal quantiles"
) -> Built:
    """Sample quantiles against theoretical ones, with the reference line."""
    if not qq:
        return {"data": []}
    theoretical = qq["theoretical"]
    ends = [theoretical[0] if theoretical else 0, theoretical[-1] if theoretical else 0]
    return {
        "data": [
            {
                "type": "scatter",
                "mode": "lines",
                "x": ends,
                "y": [qq["intercept"] + qq["slope"] * x for x in ends],
                "line": {"color": tokens.ink3, "width": 1.5, "dash": "dash"},
                "hoverinfo": "skip",
            },
            {
                "type": "scatter",
                "mode": "markers",
                "x": theoretical,
                "y": qq["sample"],
                "marker": {
                    "color": tokens.numeric,
                    "size": 8,
                    "line": {"color": tokens.surface, "width": 1},
                },
                "hovertemplate": "theoretical %{x:.3g}<br>observed %{y:,.4g}<extra></extra>",
            },
        ],
        "layout": {
            "xaxis": {**axis_title(axis_label), "showgrid": True},
            "yaxis": axis_title("Observed quantiles"),
        },
    }


def box_from_profile(numeric: dict[str, Any], tokens: Tokens) -> Built:
    """Box plot from the exact quartiles; points beyond the whiskers come from the sample."""
    d = numeric["describe"]
    if any(d.get(key) is None for key in ("q1", "q3", "median", "min", "max")):
        return {"data": []}
    spread = d["q3"] - d["q1"]
    low = max(d["min"], d["q1"] - 1.5 * spread)
    high = min(d["max"], d["q3"] + 1.5 * spread)
    outliers = [v for v in numeric.get("sample", []) if v < low or v > high]
    return {
        "data": [
            {
                "type": "box",
                "orientation": "h",
                "y": ["values"],
                "q1": [d["q1"]],
                "median": [d["median"]],
                "q3": [d["q3"]],
                "lowerfence": [low],
                "upperfence": [high],
                "mean": [d["mean"] if d.get("mean") is not None else d["median"]],
                "fillcolor": tokens.surface,
                "line": {"color": tokens.numeric, "width": 2},
                "hoverinfo": "x",
            },
            {
                "type": "scatter",
                "mode": "markers",
                "x": outliers,
                "y": ["values"] * len(outliers),
                "marker": {"color": tokens.numeric, "size": 6, "opacity": 0.5},
                "hovertemplate": "%{x:,}<extra>outlier</extra>",
            },
        ],
        "layout": {"yaxis": {"visible": False}, "xaxis": {"showgrid": True}},
    }


def violin_figure(numeric: dict[str, Any], tokens: Tokens) -> Built:
    return {
        "data": [
            {
                "type": "violin",
                "orientation": "h",
                "x": numeric.get("sample", []),
                "y0": "values",
                "line": {"color": tokens.numeric, "width": 2},
                "fillcolor": with_alpha(tokens.numeric, 0.2),
                "box": {
                    "visible": True,
                    "fillcolor": tokens.surface,
                    "line": {"color": tokens.numeric},
                },
                "meanline": {"visible": True, "color": tokens.ink2},
                "points": False,
                "hoveron": "violins",
            }
        ],
        "layout": {"yaxis": {"visible": False}, "xaxis": {"showgrid": True}},
    }


def laws_figure(
    bins: list[dict[str, Any]],
    grid: list[float],
    laws: list[dict[str, Any]],
    value_count: int,
    tokens: Tokens,
) -> Built:
    """Histogram with the best laws' densities, scaled to counts per bin."""
    width = bins[0]["end"] - bins[0]["start"] if bins else 1
    bars = {
        "type": "bar",
        "name": "Data",
        "x": [(b["start"] + b["end"]) / 2 for b in bins],
        "y": [b["count"] for b in bins],
        "width": [b["end"] - b["start"] for b in bins],
        "marker": {
            "color": with_alpha(tokens.ink3, 0.35),
            "line": {"color": tokens.surface, "width": 1},
        },
        "customdata": [[format_number(b["start"]), format_number(b["end"])] for b in bins],
        "hovertemplate": "%{customdata[0]} – %{customdata[1]}<br>%{y:,} rows<extra></extra>",
    }
    curves = []
    for index, law in enumerate(laws):
        scale = max(1, round(width)) if law["discrete"] else width
        curves.append(
            {
                "type": "scatter",
                "mode": "lines",
                "name": law["label"],
                "x": grid,
                "y": [d * value_count * scale for d in law["density"]],
                "line": {
                    "color": tokens.palette[index % 8],
                    "width": 2,
                    "shape": "hvh" if law["discrete"] else "spline",
                },
                "hovertemplate": f"{law['label']}<br>%{{x:,.4g}}: %{{y:,.0f}} expected"
                "<extra></extra>",
            }
        )
    return {"data": [bars, *curves], "layout": {**LEGEND, "bargap": 0}}


def horizontal_bars(bars: list[dict[str, Any]], color: str) -> Built:
    return {
        "data": [
            {
                "type": "bar",
                "orientation": "h",
                "x": [bar["count"] for bar in bars],
                "y": [bar["label"] for bar in bars],
                "marker": {"color": color},
                "hovertemplate": "%{y}<br>%{x:,}<extra></extra>",
            }
        ],
        "layout": {
            "yaxis": {"autorange": "reversed", "type": "category", "showgrid": False},
            "xaxis": {"showgrid": True},
            "barcornerradius": 3,
        },
    }


def calendar_figure(days: list[dict[str, Any]], tokens: Tokens) -> Built:
    """Rows per day as a calendar: one column per week, Monday on top."""
    if not days:
        return {"data": []}
    first = dt.date.fromisoformat(days[0]["label"][:10])
    origin = first - dt.timedelta(days=first.weekday())
    x, y, z, text = [], [], [], []
    for day in days:
        date = dt.date.fromisoformat(day["label"][:10])
        week = (date - dt.timedelta(days=date.weekday()) - origin).days // 7
        x.append((origin + dt.timedelta(weeks=week)).isoformat())
        y.append(date.weekday())
        z.append(day["count"])
        text.append(day["label"])
    return {
        "data": [
            {
                "type": "heatmap",
                "x": x,
                "y": y,
                "z": z,
                "text": text,
                "xgap": 2,
                "ygap": 2,
                "colorscale": [[0, with_alpha(tokens.date, 0.12)], [1, tokens.date]],
                "showscale": False,
                "hovertemplate": "%{text}<br>%{z:,} rows<extra></extra>",
            }
        ],
        "layout": {
            "xaxis": {"type": "date", "showgrid": False, "nticks": 8},
            "yaxis": {
                "tickvals": [0, 2, 4, 6],
                "ticktext": ["Mon", "Wed", "Fri", "Sun"],
                "autorange": "reversed",
                "showgrid": False,
            },
        },
    }


def matrix_figure(matrix: dict[str, Any], tokens: Tokens, clustered: bool = True) -> Built:
    """Heatmap of the association matrix in its clustered order; values shown when readable."""
    order = (
        matrix["clustered"]
        if clustered and matrix.get("clustered")
        else list(range(len(matrix["columns"])))
    )
    names = [matrix["columns"][i]["name"] for i in order]
    z = [[matrix["values"][i][j] for j in order] for i in order]
    custom = [
        [
            [METRIC_LABELS.get(matrix["metrics"][i][j] or "", ""), f"{matrix['n'][i][j]:,}"]
            for j in order
        ]
        for i in order
    ]
    return {
        "data": [
            {
                "type": "heatmap",
                "x": names,
                "y": names,
                "z": z,
                "zmin": -1,
                "zmax": 1,
                "colorscale": [[0, tokens.palette[7]], [0.5, tokens.line], [1, tokens.palette[0]]],
                "customdata": custom,
                "text": [["" if v is None else f"{v:.2f}" for v in row] for row in z],
                "texttemplate": "%{text}" if len(names) <= 16 else "",
                "textfont": {"size": 10, "color": tokens.ink1},
                "xgap": 1,
                "ygap": 1,
                "colorbar": {
                    "thickness": 10,
                    "outlinewidth": 0,
                    "tickfont": {"size": 10, "color": tokens.ink3},
                    "len": 0.8,
                },
                "hovertemplate": "%{y} × %{x}<br>%{customdata[0]} = %{z:.3f}<br>"
                "n = %{customdata[1]}<extra></extra>",
                "hoverongaps": False,
            }
        ],
        "layout": {
            "margin": {"l": 8, "r": 8, "t": 8, "b": 8},
            "xaxis": {"type": "category", "showgrid": False, "tickangle": -45, "side": "bottom"},
            "yaxis": {"type": "category", "showgrid": False, "autorange": "reversed"},
        },
    }
