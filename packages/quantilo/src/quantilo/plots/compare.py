"""The Compare page's figures (frontend/src/features/compare/figures.ts): the visual follows the
types, as in the app."""

from typing import Any

from quantilo.plots.theme import Tokens, axis_title, format_number, with_alpha

Built = dict[str, Any]
OTHERS = "Others"
LEGEND = {
    "showlegend": True,
    "legend": {"orientation": "h", "x": 0, "y": 1.1, "font": {"size": 11}},
}


def color_of(labels: list[str], tokens: Tokens) -> dict[str, str]:
    """Categorical colours in fixed order; "Others" always muted grey."""
    named = [label for label in labels if label != OTHERS]
    colors = {label: tokens.palette[i % 8] for i, label in enumerate(dict.fromkeys(named))}
    colors[OTHERS] = tokens.ink3
    return colors


def unique_in_order(values: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return sorted(counts, key=lambda v: (v == OTHERS, -counts[v]))


# --- num × num -----------------------------------------------------------------------------------


def scatter_figure(result: dict[str, Any], tokens: Tokens, regression: bool = True) -> Built:
    points = result.get("points")
    if not points:
        return {"data": []}
    size = 4 if len(points["x"]) > 5000 else 7
    traces: list[dict[str, Any]] = []
    if points.get("color"):
        labels = unique_in_order(points["color"])
        colors = color_of(labels, tokens)
        for label in labels:
            index = [i for i, value in enumerate(points["color"]) if value == label]
            traces.append(
                {
                    "type": "scattergl",
                    "mode": "markers",
                    "name": label,
                    "x": [points["x"][i] for i in index],
                    "y": [points["y"][i] for i in index],
                    "marker": {
                        "color": colors[label],
                        "size": size,
                        "opacity": 0.7,
                        "line": {"color": tokens.surface, "width": 0.5},
                    },
                    "hovertemplate": f"{label}<br>%{{x:,.4g}}, %{{y:,.4g}}<extra></extra>",
                }
            )
    else:
        traces.append(
            {
                "type": "scattergl",
                "mode": "markers",
                "name": "rows",
                "x": points["x"],
                "y": points["y"],
                "marker": {
                    "color": tokens.numeric,
                    "size": size,
                    "opacity": 0.55,
                    "line": {"color": tokens.surface, "width": 0.5},
                },
                "hovertemplate": "%{x:,.4g}, %{y:,.4g}<extra></extra>",
            }
        )
    line = result.get("regression")
    if regression and line and points["x"]:
        low, high = min(points["x"]), max(points["x"])
        slope, intercept = line["slope"], line["intercept"]
        traces.append(
            {
                "type": "scatter",
                "mode": "lines",
                "name": "Linear fit",
                "x": [low, high],
                "y": [intercept + slope * low, intercept + slope * high],
                "line": {"color": tokens.ink1, "width": 2, "dash": "dash"},
                "hovertemplate": f"y = {format_number(slope)}·x + {format_number(intercept)}"
                "<extra></extra>",
            }
        )
    return {
        "data": traces,
        "layout": {
            **(LEGEND if points.get("color") else {}),
            "xaxis": {**axis_title(result["x"]), "showgrid": True},
            "yaxis": axis_title(result["y"]),
        },
    }


def density_figure(result: dict[str, Any], tokens: Tokens) -> Built:
    density = result.get("density")
    if not density:
        return {"data": []}

    def centers(edges: list[float]) -> list[float]:
        return [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)]

    return {
        "data": [
            {
                "type": "heatmap",
                "x": centers(density["x_edges"]),
                "y": centers(density["y_edges"]),
                "z": [[None if c == 0 else c for c in row] for row in density["counts"]],
                "colorscale": [[0, with_alpha(tokens.numeric, 0.12)], [1, tokens.numeric]],
                "showscale": True,
                "colorbar": {
                    "thickness": 10,
                    "outlinewidth": 0,
                    "tickfont": {"size": 10, "color": tokens.ink3},
                },
                "hovertemplate": "%{x:,.3g}, %{y:,.3g}<br>%{z:,} rows<extra></extra>",
            }
        ],
        "layout": {
            "xaxis": {**axis_title(result["x"]), "showgrid": False},
            "yaxis": {**axis_title(result["y"]), "showgrid": False},
        },
    }


def scatter3d_figure(result: dict[str, Any], tokens: Tokens) -> Built:
    points = result.get("points")
    if not points or not points.get("z"):
        return {"data": []}

    def axis(title: str) -> dict[str, Any]:
        return {
            "title": {"text": title, "font": {"size": 11, "color": tokens.ink2}},
            "color": tokens.ink3,
            "gridcolor": tokens.line,
            "zerolinecolor": tokens.line,
            "backgroundcolor": "rgba(0,0,0,0)",
        }

    colored = bool(points.get("color"))
    labels: list[str | None] = list(unique_in_order(points["color"])) if colored else [None]
    colors = color_of([label for label in labels if label], tokens) if colored else {}
    traces = []
    for label in labels:
        index = [i for i in range(len(points["x"])) if label is None or points["color"][i] == label]
        traces.append(
            {
                "type": "scatter3d",
                "mode": "markers",
                "name": label or "rows",
                "x": [points["x"][i] for i in index],
                "y": [points["y"][i] for i in index],
                "z": [points["z"][i] for i in index],
                "marker": {
                    "size": 3,
                    "color": tokens.numeric if label is None else colors.get(label),
                    "opacity": 0.75,
                },
                "hovertemplate": f"{label + '<br>' if label else ''}%{{x:,.3g}}, %{{y:,.3g}}, "
                "%{z:,.3g}<extra></extra>",
            }
        )
    return {
        "data": traces,
        "layout": {
            **(LEGEND if colored else {}),
            "margin": {"l": 0, "r": 0, "t": 0, "b": 0},
            "scene": {
                "xaxis": axis(result["x"]),
                "yaxis": axis(result["y"]),
                "zaxis": axis(result.get("z") or ""),
                "bgcolor": "rgba(0,0,0,0)",
            },
        },
    }


# --- num × groups --------------------------------------------------------------------------------


def _means(groups: list[dict[str, Any]], vertical: bool) -> dict[str, Any]:
    return {
        "type": "data",
        "symmetric": False,
        "array": [(g["ci_high"] or 0) - (g["mean"] or 0) for g in groups],
        "arrayminus": [(g["mean"] or 0) - (g["ci_low"] or 0) for g in groups],
        "thickness": 1.5,
        "width": 6,
    }


def _colored_groups(
    result: dict[str, Any], view: str, tokens: Tokens, vertical: bool
) -> list[dict[str, Any]]:
    """Groups split by a colour: one box (violin, mean) per colour level, side by side."""
    colors = color_of(result["color_levels"], tokens)
    axis, value = ("x", "y") if vertical else ("y", "x")
    traces = []
    for level in result["color_levels"]:
        groups = [g for g in result["groups"] if g.get("color") == level]
        labels = [g["label"] for g in groups]
        color = colors.get(level, tokens.ink3)
        shared = {"name": level, "legendgroup": level, "offsetgroup": level}
        if view == "means":
            error = {**_means(groups, vertical), "color": color}
            traces.append(
                {
                    "type": "scatter",
                    "mode": "markers",
                    **shared,
                    axis: labels,
                    value: [g["mean"] for g in groups],
                    "marker": {"size": 10, "color": color},
                    f"error_{value}": error,
                    "customdata": [
                        [g["n"], format_number(g["ci_low"]), format_number(g["ci_high"])]
                        for g in groups
                    ],
                    "hovertemplate": f"{level} · %{{{axis}}}<br>mean %{{{value}:,.4g}}<br>95% CI "
                    "%{customdata[1]} – %{customdata[2]}<br>n = %{customdata[0]:,}<extra></extra>",
                }
            )
        elif view == "violin":
            traces.append(
                {
                    "type": "violin",
                    **shared,
                    axis: [g["label"] for g in groups for _ in g["sample"]],
                    value: [v for g in groups for v in g["sample"]],
                    "orientation": "v" if vertical else "h",
                    "line": {"color": color, "width": 1.5},
                    "fillcolor": with_alpha(color, 0.25),
                    "box": {"visible": True, "width": 0.15, "fillcolor": tokens.surface},
                    "meanline": {"visible": True},
                    "points": False,
                    "spanmode": "hard",
                }
            )
        else:
            traces.append(
                {
                    "type": "box",
                    **shared,
                    axis: labels,
                    "q1": [g["q1"] for g in groups],
                    "median": [g["median"] for g in groups],
                    "q3": [g["q3"] for g in groups],
                    "lowerfence": [g["low"] for g in groups],
                    "upperfence": [g["high"] for g in groups],
                    "mean": [g["mean"] for g in groups],
                    "orientation": "v" if vertical else "h",
                    "fillcolor": with_alpha(color, 0.25),
                    "line": {"color": color, "width": 1.5},
                }
            )
    return traces


def groups_figure(result: dict[str, Any], tokens: Tokens, view: str = "box") -> Built:
    """A number across groups: box, violin or means ± 95 % CI."""
    vertical = result.get("group_axis") == "x"
    numeric = result["y"] if vertical else result["x"]
    category = (result.get("layout") or {}).get("axis") or (
        result["x"] if vertical else result["y"]
    )
    groups = result["groups"]
    colors = color_of([g["label"] for g in groups], tokens)
    axis, value = ("x", "y") if vertical else ("y", "x")
    if result.get("color_levels"):
        data = _colored_groups(result, view, tokens, vertical)
    elif view == "means":
        data = [
            {
                "type": "scatter",
                "mode": "markers",
                axis: [g["label"] for g in groups],
                value: [g["mean"] for g in groups],
                "marker": {
                    "size": 10,
                    "color": [colors.get(g["label"], tokens.ink3) for g in groups],
                },
                f"error_{value}": {**_means(groups, vertical), "color": tokens.ink2},
                "customdata": [
                    [g["n"], format_number(g["ci_low"]), format_number(g["ci_high"])]
                    for g in groups
                ],
                "hovertemplate": f"%{{{axis}}}<br>mean %{{{value}:,.4g}}<br>95% CI "
                "%{customdata[1]} – %{customdata[2]}<br>n = %{customdata[0]:,}<extra></extra>",
            }
        ]
    elif view == "violin":
        data = [
            {
                "type": "violin",
                "name": g["label"],
                value: g["sample"],
                f"{axis}0": g["label"],
                "orientation": "v" if vertical else "h",
                "line": {"color": colors.get(g["label"]), "width": 1.5},
                "fillcolor": with_alpha(colors.get(g["label"], tokens.ink3), 0.25),
                "box": {"visible": True, "width": 0.15, "fillcolor": tokens.surface},
                "meanline": {"visible": True},
                "points": False,
                "spanmode": "hard",
            }
            for g in groups
        ]
    else:
        data = [
            {
                "type": "box",
                "name": g["label"],
                axis: [g["label"]],
                "q1": [g["q1"]],
                "median": [g["median"]],
                "q3": [g["q3"]],
                "lowerfence": [g["low"]],
                "upperfence": [g["high"]],
                "mean": [g["mean"]],
                "orientation": "v" if vertical else "h",
                "fillcolor": with_alpha(colors.get(g["label"], tokens.ink3), 0.25),
                "line": {"color": colors.get(g["label"]), "width": 1.5},
                "hoverinfo": "y" if vertical else "x",
            }
            for g in groups
        ]
    category_axis = {
        **axis_title(category),
        "type": "category",
        "showgrid": False,
        **({} if vertical else {"autorange": "reversed"}),
    }
    value_axis = {**axis_title(numeric), "showgrid": True}
    legend_title = (result.get("layout") or {}).get("color") or result.get("color") or ""
    return {
        "data": data,
        "layout": {
            "showlegend": bool(result.get("color_levels")),
            "legend": {"title": {"text": legend_title}, "orientation": "h", "y": 1.08},
            "boxmode": "group",
            "violinmode": "group",
            "scattermode": "group",
            "xaxis": category_axis if vertical else value_axis,
            "yaxis": value_axis if vertical else category_axis,
        },
    }


# --- cat × cat -----------------------------------------------------------------------------------


def contingency_figure(result: dict[str, Any], tokens: Tokens, view: str = "heatmap") -> Built:
    table = result.get("contingency")
    if not table:
        return {"data": []}
    rows, columns, counts = table["rows"], table["columns"], table["counts"]
    if view == "heatmap":
        return {
            "data": [
                {
                    "type": "heatmap",
                    "x": columns,
                    "y": rows,
                    "z": counts,
                    "text": [[f"{c:,}" if c else "" for c in row] for row in counts],
                    "texttemplate": "%{text}" if len(columns) * len(rows) <= 120 else "",
                    "textfont": {"size": 11, "color": tokens.ink1},
                    "colorscale": [
                        [0, with_alpha(tokens.categorical, 0.06)],
                        [1, with_alpha(tokens.categorical, 0.7)],
                    ],
                    "showscale": False,
                    "xgap": 2,
                    "ygap": 2,
                    "hovertemplate": f"{result['x']}: %{{x}}<br>{result['y']}: %{{y}}<br>"
                    "%{z:,} rows<extra></extra>",
                }
            ],
            "layout": {
                "xaxis": {**axis_title(result["x"]), "type": "category", "showgrid": False},
                "yaxis": {
                    **axis_title(result["y"]),
                    "type": "category",
                    "showgrid": False,
                    "autorange": "reversed",
                },
            },
        }
    totals = [sum(row[j] for row in counts) for j in range(len(columns))]
    colors = color_of(rows, tokens)
    return {
        "data": [
            {
                "type": "bar",
                "name": row,
                "x": columns,
                "y": [counts[i][j] / (totals[j] or 1) * 100 for j in range(len(columns))],
                "customdata": counts[i],
                "marker": {"color": colors.get(row), "line": {"color": tokens.surface, "width": 1}},
                "hovertemplate": f"{row}<br>%{{x}}: %{{y:.1f}}% (%{{customdata:,}} rows)"
                "<extra></extra>",
            }
            for i, row in enumerate(rows)
        ],
        "layout": {
            **LEGEND,
            "barmode": "stack",
            "xaxis": {**axis_title(result["x"]), "type": "category"},
            "yaxis": {
                **axis_title(f"{result['y']} (% of each {result['x']})"),
                "ticksuffix": "%",
                "range": [0, 100],
            },
        },
    }


# --- dates ---------------------------------------------------------------------------------------


def _edge(mean: list[float | None], std: list[float | None], sign: int) -> list[float | None]:
    return [
        None if m is None or s is None else m + sign * s for m, s in zip(mean, std, strict=False)
    ]


def timeline_figure(
    result: dict[str, Any], tokens: Tokens, layers: tuple[str, ...] = ("values", "mean", "band")
) -> Built:
    """Aggregated values, rolling mean and (for a single series) a ± 1σ band."""
    numeric = result["y"] if result["roles"].get(result["x"]) == "date" else result["x"]
    series_list = result["series"]
    single = len(series_list) == 1
    colors = color_of([s.get("label") or "" for s in series_list], tokens)
    time = result.get("time")
    what = f"{time['aggregate']} {numeric}" if time else numeric
    rolled = (
        f"{time['window']}-{time['period']} rolling mean"
        if time and time.get("window")
        else ("rolling mean")
    )
    data: list[dict[str, Any]] = []
    for series in series_list:
        label = series.get("label")
        color = tokens.palette[0] if single else colors.get(label or "", tokens.ink3)
        name = label or what
        mean, std = series.get("rolling_mean", []), series.get("rolling_std", [])
        rolling = len(mean) > 0
        if single and rolling and "band" in layers:
            data += [
                {
                    "type": "scatter",
                    "mode": "lines",
                    "x": series["periods"],
                    "y": _edge(mean, std, -1),
                    "line": {"width": 0},
                    "hoverinfo": "skip",
                    "showlegend": False,
                    "name": "lower",
                },
                {
                    "type": "scatter",
                    "mode": "lines",
                    "x": series["periods"],
                    "y": _edge(mean, std, 1),
                    "fill": "tonexty",
                    "fillcolor": with_alpha(color, 0.15),
                    "line": {"width": 0},
                    "hoverinfo": "skip",
                    "name": "± 1σ band",
                },
            ]
        if "values" in layers or not rolling:
            data.append(
                {
                    "type": "scatter",
                    "mode": "lines" if len(series["periods"]) > 60 else "lines+markers",
                    "name": what if single else name,
                    "legendgroup": name,
                    "x": series["periods"],
                    "y": series["values"],
                    "customdata": series.get("counts"),
                    "connectgaps": False,
                    "line": {
                        "color": with_alpha(color, 0.55) if rolling and "mean" in layers else color,
                        "width": 1.25,
                    },
                    "marker": {"size": 4},
                    "hovertemplate": f"{label + '<br>' if label else ''}%{{x}}<br>{what}: "
                    "%{y:,.4g}<br>%{customdata:,} rows<extra></extra>",
                }
            )
        if rolling and "mean" in layers:
            data.append(
                {
                    "type": "scatter",
                    "mode": "lines",
                    "name": rolled if single else f"{name} · mean",
                    "legendgroup": name,
                    "showlegend": single or "values" not in layers,
                    "x": series["periods"],
                    "y": mean,
                    "customdata": std,
                    "line": {"color": color, "width": 2.5},
                    "hovertemplate": f"{label + '<br>' if label else ''}%{{x}}<br>{rolled}: "
                    "%{y:,.4g}<br>σ %{customdata:,.3g}<extra></extra>",
                }
            )
    return {
        "data": data,
        "layout": {
            **LEGEND,
            "hovermode": "x unified",
            "xaxis": {"type": "date", "nticks": 8, "tickangle": 0},
            "yaxis": axis_title(what),
        },
    }


def date_counts_figure(result: dict[str, Any], tokens: Tokens) -> Built:
    colors = color_of([s.get("label") or "" for s in result["series"]], tokens)
    return {
        "data": [
            {
                "type": "scatter",
                "mode": "lines",
                "stackgroup": "rows",
                "name": s.get("label"),
                "x": s["periods"],
                "y": s["values"],
                "line": {"color": colors.get(s.get("label") or ""), "width": 1},
                "hovertemplate": f"{s.get('label')}<br>%{{x}}: %{{y:,}} rows<extra></extra>",
            }
            for s in result["series"]
        ],
        "layout": {
            **LEGEND,
            "xaxis": {"type": "date", "nticks": 8, "tickangle": 0},
            "yaxis": axis_title("rows"),
        },
    }


def views_for(result: dict[str, Any]) -> list[str]:
    kind = result["kind"]
    if kind == "scatter":
        return ["3d", "points", "density"] if result.get("z") else ["points", "density"]
    if kind == "groups":
        return ["box", "violin", "means"]
    if kind == "contingency":
        return ["heatmap", "stacked"]
    return ["series"]


def comparison_figure(
    result: dict[str, Any], tokens: Tokens, view: str | None = None, regression: bool = True
) -> Built:
    """The Compare page's chart: `view` among `views_for(result)` (the first by default)."""
    views = views_for(result)
    if view is not None and view not in views:
        raise ValueError(f"View {view!r} does not fit this comparison: choose {', '.join(views)}.")
    chosen = view or views[0]
    if chosen == "3d":
        return scatter3d_figure(result, tokens)
    if chosen == "points":
        return scatter_figure(result, tokens, regression)
    if chosen == "density":
        return density_figure(result, tokens)
    if chosen in ("box", "violin", "means"):
        return groups_figure(result, tokens, chosen)
    if chosen in ("heatmap", "stacked"):
        return contingency_figure(result, tokens, chosen)
    if result["kind"] == "date_counts":
        return date_counts_figure(result, tokens)
    return timeline_figure(result, tokens)
