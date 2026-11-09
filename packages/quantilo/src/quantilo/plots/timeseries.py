"""The Time series page's figures (frontend/src/modules/timeseries: figures.ts, Forecast.tsx,
Memory.tsx, Shape.tsx)."""

from typing import Any

from quantilo.plots.theme import LEGEND, Tokens, axis_title, with_alpha
from quantilo.plots.variable import qq_figure

Built = dict[str, Any]


def value_label(setup: dict[str, Any]) -> str:
    """ "sum units", as the page's axis title."""
    return f"{setup['aggregate']} {setup['value'] or 'rows'}"


def _band(
    x: list[str], mean: list[float | None], std: list[float | None], color: str
) -> list[Built]:
    def edge(sign: int) -> list[float | None]:
        return [
            None if m is None or s is None else m + sign * s
            for m, s in zip(mean, std, strict=False)
        ]

    return [
        {
            "type": "scatter",
            "mode": "lines",
            "x": x,
            "y": edge(-1),
            "line": {"width": 0},
            "hoverinfo": "skip",
            "showlegend": False,
        },
        {
            "type": "scatter",
            "mode": "lines",
            "name": "± 1σ band",
            "x": x,
            "y": edge(1),
            "fill": "tonexty",
            "fillcolor": with_alpha(color, 0.15),
            "line": {"width": 0},
            "hoverinfo": "skip",
        },
    ]


def overview_figure(
    overview: dict[str, Any],
    tokens: Tokens,
    layers: tuple[str, ...] = ("values", "mean", "band"),
    all_groups: bool = False,
) -> Built:
    """The analysed series with its rolling mean and ± 1σ band; other groups faintly behind."""
    periods, analyzed = overview["periods"], overview["analyzed"]
    what = value_label(overview["setup"])
    window = overview.get("window")
    centered = " centered" if overview.get("align") == "centered" else ""
    rolled = f"{window}-{overview['setup']['period']}{centered} rolling mean" if window else ""
    color = tokens.palette[0]
    others = (
        [line for line in overview["lines"] if line["label"] != analyzed["label"]]
        if all_groups
        else []
    )
    data: list[Built] = [
        {
            "type": "scatter",
            "mode": "lines",
            "name": line["label"] or "",
            "x": periods,
            "y": line["values"],
            "line": {"color": with_alpha(tokens.palette[(i + 1) % 8], 0.6), "width": 1},
            "hovertemplate": f"{line['label']}<br>%{{x}}: %{{y:,.4g}}<extra></extra>",
        }
        for i, line in enumerate(others)
    ]
    rolling = len(analyzed.get("rolling_mean", [])) > 0
    if rolling and "band" in layers:
        data += _band(periods, analyzed["rolling_mean"], analyzed["rolling_std"], color)
    if "values" in layers or not rolling:
        data.append(
            {
                "type": "scatter",
                "mode": "lines" if len(periods) > 60 else "lines+markers",
                "name": f"{analyzed['label']} · {what}" if analyzed.get("label") else what,
                "x": periods,
                "y": analyzed["values"],
                "customdata": analyzed["counts"],
                "line": {
                    "color": with_alpha(color, 0.55) if rolling and "mean" in layers else color,
                    "width": 1.25,
                },
                "marker": {"size": 4},
                "hovertemplate": f"%{{x}}<br>{what}: %{{y:,.4g}}<br>%{{customdata:,}} rows"
                "<extra></extra>",
            }
        )
    if rolling and "mean" in layers:
        data.append(
            {
                "type": "scatter",
                "mode": "lines",
                "name": rolled,
                "x": periods,
                "y": analyzed["rolling_mean"],
                "line": {"color": color, "width": 2.5},
                "hovertemplate": f"%{{x}}<br>{rolled}: %{{y:,.4g}}<extra></extra>",
            }
        )
    return {
        "data": data,
        "layout": {
            **LEGEND,
            "hovermode": "x unified",
            "xaxis": {
                "type": "date",
                "rangeslider": {"visible": len(periods) > 30, "thickness": 0.08},
            },
            "yaxis": axis_title(what),
        },
    }


def profile_figure(profile: dict[str, Any], what: str, tokens: Tokens) -> Built:
    """Box per calendar label (Mon…Sun, Jan…Dec), the strongest label highlighted."""
    data = []
    for i, label in enumerate(profile["labels"]):
        deviation = profile["deviation"][i]
        shift = (
            ""
            if deviation is None
            else f"{'+' if deviation >= 0 else ''}{deviation:.1f} % vs average"
        )
        data.append(
            {
                "type": "box",
                "name": label,
                "y": profile["values"][i],
                "boxpoints": False,
                "marker": {
                    "color": tokens.palette[0] if label == profile.get("strongest") else tokens.ink3
                },
                "line": {"width": 1.25},
                "hovertemplate": f"{label}<br>%{{y:,.4g}}<br>{shift}<extra></extra>",
            }
        )
    return {
        "data": data,
        "layout": {"showlegend": False, "xaxis": {"type": "category"}, "yaxis": axis_title(what)},
    }


def overlay_figure(overlay: dict[str, Any], what: str, tokens: Tokens) -> Built:
    """One line per year, the latest in the accent colour."""
    years = overlay["years"]
    latest = years[-1]["year"] if years else None
    return {
        "data": [
            {
                "type": "scatter",
                "mode": "lines",
                "name": str(line["year"]),
                "x": line["x"],
                "y": line["y"],
                "line": {
                    "color": tokens.palette[0]
                    if line["year"] == latest
                    else with_alpha(tokens.ink3, 0.55),
                    "width": 2.5 if line["year"] == latest else 1.25,
                },
                "hovertemplate": f"{line['year']}<br>{overlay['unit']} %{{x}}: %{{y:,.4g}}"
                "<extra></extra>",
            }
            for line in years
        ],
        "layout": {**LEGEND, "xaxis": axis_title(overlay["unit"]), "yaxis": axis_title(what)},
    }


def decomposition_figure(decomposition: dict[str, Any], tokens: Tokens) -> Built:
    """Observed, trend, seasonal component(s) and remainder, stacked on one time axis."""
    parts = decomposition.get("result")
    if not parts:
        return {"data": []}
    periods = decomposition["periods"]
    seasonal = parts["seasonal"]
    rows = [
        {"name": "Observed", "y": decomposition["observed"], "color": tokens.ink2, "bars": False},
        {"name": "Trend", "y": parts["trend"], "color": tokens.palette[0], "bars": False},
        *[
            {
                "name": f"Seasonal ({parts['seasons'][i]})" if len(seasonal) > 1 else "Seasonal",
                "y": y,
                "color": tokens.palette[(1 + i) % 8],
                "bars": False,
            }
            for i, y in enumerate(seasonal)
        ],
        {"name": "Remainder", "y": parts["resid"], "color": tokens.ink3, "bars": True},
    ]
    gap = 0.04
    size = (1 - gap * (len(rows) - 1)) / len(rows)
    layout: dict[str, Any] = {"showlegend": False, "hovermode": "x unified"}
    data = []
    for i, row in enumerate(rows):
        axis = "" if i == 0 else str(i + 1)
        top = 1 - i * (size + gap)
        layout[f"yaxis{axis}"] = {
            "domain": [max(0.0, round(top - size, 6)), min(1.0, round(top, 6))],
            **axis_title(row["name"]),
            "zeroline": row["bars"],
        }
        data.append(
            {
                "type": "bar" if row["bars"] else "scatter",
                **({} if row["bars"] else {"mode": "lines"}),
                "name": row["name"],
                "x": periods,
                "y": row["y"],
                "xaxis": "x",
                "yaxis": f"y{axis}",
                **({"marker": {"color": row["color"]}} if row["bars"] else {}),
                **({} if row["bars"] else {"line": {"color": row["color"], "width": 1.5}}),
                "hovertemplate": f"{row['name']}: %{{y:,.4g}}<extra></extra>",
            }
        )
    layout["xaxis"] = {"type": "date", "anchor": f"y{len(rows)}"}
    return {"data": data, "layout": layout}


def transformed_figure(stationarity: dict[str, Any], tokens: Tokens) -> Built:
    return {
        "data": [
            {
                "type": "scatter",
                "mode": "lines",
                "x": stationarity["periods"],
                "y": stationarity["transformed"],
                "line": {"color": tokens.palette[0], "width": 1.5},
                "hovertemplate": "%{x}: %{y:,.4g}<extra></extra>",
            }
        ],
        "layout": {
            "showlegend": False,
            "xaxis": {"type": "date"},
            "yaxis": {"zeroline": True, "zerolinecolor": tokens.line},
        },
    }


def correlogram_figure(lags: list[int], values: list[float], band: float, tokens: Tokens) -> Built:
    """ACF or PACF bars; the bars beyond the ± 1.96/√n band stand out."""
    end = (max(lags) if lags else 0) + 1
    return {
        "data": [
            {
                "type": "bar",
                "x": lags,
                "y": values,
                "marker": {
                    "color": [tokens.palette[0] if abs(v) > band else tokens.ink3 for v in values]
                },
                "hovertemplate": "lag %{x}: %{y:.3f}<extra></extra>",
            },
            {
                "type": "scatter",
                "mode": "lines",
                "x": [0, end, None, 0, end],
                "y": [band, band, None, -band, -band],
                "line": {"color": tokens.ink3, "width": 1, "dash": "dot"},
                "hoverinfo": "skip",
            },
        ],
        "layout": {
            "showlegend": False,
            "bargap": 0.35,
            "xaxis": axis_title("lag"),
            "yaxis": {"range": [-1, 1]},
        },
    }


def volatility_figure(volatility: dict[str, Any], tokens: Tokens) -> Built:
    return {
        "data": [
            {
                "type": "bar",
                "x": volatility["periods"],
                "y": volatility["residuals"],
                "marker": {"color": with_alpha(tokens.ink3, 0.6)},
                "hovertemplate": "Residual: %{y:,.4g}<extra></extra>",
            },
            {
                "type": "scatter",
                "mode": "lines",
                "x": volatility["periods"],
                "y": volatility["rolling_std"],
                "yaxis": "y2",
                "line": {"color": tokens.palette[0], "width": 2},
                "hovertemplate": f"{volatility['window']}-period σ: %{{y:,.4g}}<extra></extra>",
            },
        ],
        "layout": {
            "showlegend": False,
            "hovermode": "x unified",
            "xaxis": {"type": "date", "anchor": "y2"},
            "yaxis": {
                "domain": [0.45, 1],
                **axis_title("Residuals"),
                "zeroline": True,
                "zerolinecolor": tokens.line,
            },
            "yaxis2": {"domain": [0, 0.38], **axis_title("Rolling σ"), "rangemode": "tozero"},
        },
    }


def residuals_figure(residuals: dict[str, Any], tokens: Tokens) -> Built:
    shape = residuals.get("distribution")
    return qq_figure(shape.get("qq") if shape else None, tokens, "Normal quantiles")


def breaks_figure(found: dict[str, Any], tokens: Tokens) -> Built:
    periods, values = found["periods"], found["values"]
    breaks = found.get("breaks") or {}
    trend = breaks.get("cost") == "trend"
    means: list[float | None] = [None] * len(periods)
    for segment in breaks.get("segments", []):
        length = max(segment["end"] - segment["start"] - 1, 1)
        for i in range(segment["start"], segment["end"]):
            means[i] = (
                segment["first"]
                + (segment["last"] - segment["first"]) * (i - segment["start"]) / length
                if trend
                else segment["mean"]
            )
    shapes = [
        {
            "type": "line",
            "x0": point["period"],
            "x1": point["period"],
            "yref": "paper",
            "y0": 0,
            "y1": 1,
            "line": {"color": tokens.palette[1], "width": 1.5, "dash": "dash"},
        }
        for point in found["points"]
    ]
    return {
        "data": [
            {
                "type": "scatter",
                "mode": "lines",
                "name": "Series without seasonality" if found.get("deseasonalized") else "Series",
                "x": periods,
                "y": values,
                "line": {"color": with_alpha(tokens.palette[0], 0.6), "width": 1.25},
                "hovertemplate": "%{x}: %{y:,.4g}<extra></extra>",
            },
            {
                "type": "scatter",
                "mode": "lines",
                "name": "Segment trend" if trend else "Segment mean",
                "x": periods,
                "y": means,
                "line": {"color": tokens.palette[0], "width": 2.5, "shape": "hv"},
                "hovertemplate": "Segment mean: %{y:,.4g}<extra></extra>",
            },
            {
                "type": "scatter",
                "mode": "markers",
                "name": "Anomalies",
                "x": [a["period"] for a in found["anomalies"]],
                "y": [values[a["index"]] for a in found["anomalies"]],
                "marker": {
                    "color": tokens.palette[3],
                    "size": 9,
                    "symbol": "circle-open",
                    "line": {"width": 2},
                },
                "hovertemplate": "%{x}: anomaly<extra></extra>",
            },
        ],
        "layout": {**LEGEND, "shapes": shapes, "xaxis": {"type": "date"}},
    }


def forecast_figure(
    forecast: dict[str, Any],
    tokens: Tokens,
    model: str | None = None,
    overlay: bool = False,
    what: str = "",
) -> Built:
    """History, the chosen model's forecast with its 80 % and 95 % intervals, and optionally
    every other model dotted."""
    results = [r for r in forecast["results"] if r.get("scores")]
    shown = next((r for r in results if r["key"] == (model or forecast.get("best"))), None)
    if shown is None:
        return {"data": []}
    future = forecast["future"]
    color = tokens.palette[0]

    def band(low: list[float], high: list[float], alpha: float, name: str) -> list[Built]:
        return [
            {
                "type": "scatter",
                "mode": "lines",
                "x": future,
                "y": low,
                "line": {"width": 0},
                "hoverinfo": "skip",
                "showlegend": False,
            },
            {
                "type": "scatter",
                "mode": "lines",
                "name": name,
                "x": future,
                "y": high,
                "fill": "tonexty",
                "fillcolor": with_alpha(color, alpha),
                "line": {"width": 0},
                "hoverinfo": "skip",
            },
        ]

    others = [r for r in results if r["key"] != shown["key"]] if overlay else []
    return {
        "data": [
            {
                "type": "scatter",
                "mode": "lines",
                "name": what or "history",
                "x": forecast["periods"],
                "y": forecast["history"],
                "line": {"color": tokens.ink2, "width": 1.5},
                "hovertemplate": "%{x}: %{y:,.4g}<extra></extra>",
            },
            *band(shown["low95"], shown["high95"], 0.12, "95 % interval"),
            *band(shown["low80"], shown["high80"], 0.22, "80 % interval"),
            *[
                {
                    "type": "scatter",
                    "mode": "lines",
                    "name": other["info"] or other["key"],
                    "x": future,
                    "y": other["forecast"],
                    "line": {"color": tokens.palette[(i + 1) % 8], "width": 1.25, "dash": "dot"},
                    "hovertemplate": f"{other['info'] or other['key']}: %{{y:,.4g}}<extra></extra>",
                }
                for i, other in enumerate(others)
            ],
            {
                "type": "scatter",
                "mode": "lines+markers",
                "name": shown["info"] or shown["key"],
                "x": future,
                "y": shown["forecast"],
                "line": {"color": color, "width": 2.5},
                "marker": {"size": 5},
                "hovertemplate": "%{x}: %{y:,.4g}<extra></extra>",
            },
        ],
        "layout": {
            **LEGEND,
            "hovermode": "x unified",
            "xaxis": {
                "type": "date",
                "rangeslider": {"visible": len(forecast["periods"]) > 30, "thickness": 0.08},
            },
        },
    }
