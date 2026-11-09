"""The Changes page's figures (frontend/src/features/changes/figures.ts): the previous version in
solid grey, the new one as an accent outline, so the two never depend on colour alone."""

from typing import Any

from quantilo.plots.theme import LEGEND, Tokens, axis_title, with_alpha

Built = dict[str, Any]
PERCENT_AXIS = {"tickformat": ".0%", "rangemode": "tozero"}


def looks(tokens: Tokens) -> dict[str, dict[str, str]]:
    return {
        "before": {"name": "Before", "color": tokens.ink2, "fill": with_alpha(tokens.ink2, 0.45)},
        "after": {"name": "After", "color": tokens.accent, "fill": with_alpha(tokens.accent, 0.12)},
    }


def _pair(
    look: dict[str, dict[str, str]], common: dict[str, Any], before: Any, after: Any, axis: str
):
    other = "y" if axis == "x" else "x"
    return [
        {
            **common,
            "name": look["before"]["name"],
            other: before,
            "marker": {"color": look["before"]["fill"], "line": {"width": 0}},
            "hovertemplate": f"%{{{axis}}}<br>before %{{{other}:.1%}}<extra></extra>",
        },
        {
            **common,
            "name": look["after"]["name"],
            other: after,
            "marker": {
                "color": look["after"]["fill"],
                "line": {"color": look["after"]["color"], "width": 1.5},
            },
            "hovertemplate": f"%{{{axis}}}<br>after %{{{other}:.1%}}<extra></extra>",
        },
    ]


def shift_figure(shift: dict[str, Any], tokens: Tokens) -> Built:
    """Both versions on the same intervals (numbers) or values (categories), as shares."""
    look = looks(tokens)
    levels = shift.get("levels") or []
    histogram = shift.get("histogram")
    if histogram and not (shift["kind"] == "numeric" and levels):
        edges = histogram["edges"]
        centres = [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)]
        widths = [edges[i + 1] - edges[i] for i in range(len(edges) - 1)]
        common = {"type": "bar", "x": centres, "width": widths}
        return {
            "data": _pair(look, common, histogram["before"], histogram["after"], "x"),
            "layout": {
                **LEGEND,
                "barmode": "overlay",
                "bargap": 0,
                "yaxis": {**PERCENT_AXIS, **axis_title("Share of values")},
            },
        }
    rows = list(reversed(levels))
    common = {"type": "bar", "orientation": "h", "y": [r["value"] for r in rows]}
    return {
        "data": _pair(look, common, [r["before"] for r in rows], [r["after"] for r in rows], "y"),
        "layout": {
            **LEGEND,
            "barmode": "group",
            "bargap": 0.25,
            "xaxis": {**PERCENT_AXIS, "showgrid": True},
            "yaxis": {"showgrid": False, "type": "category"},
        },
    }


def periods_figure(dates: dict[str, Any], tokens: Tokens) -> Built:
    """Rows per period in both versions: a hole or a new last period shows at once."""
    look = looks(tokens)
    common = {"type": "bar", "x": dates["periods"]}
    return {
        "data": [
            {
                **common,
                "name": look["before"]["name"],
                "y": dates["before_counts"],
                "marker": {"color": look["before"]["fill"]},
                "hovertemplate": "%{x}<br>before %{y:,} rows<extra></extra>",
            },
            {
                **common,
                "name": look["after"]["name"],
                "y": dates["after_counts"],
                "marker": {
                    "color": look["after"]["fill"],
                    "line": {"color": look["after"]["color"], "width": 1.5},
                },
                "hovertemplate": "%{x}<br>after %{y:,} rows<extra></extra>",
            },
        ],
        "layout": {
            **LEGEND,
            "barmode": "group",
            "bargap": 0.2,
            "xaxis": {"type": "category"},
            "yaxis": {**axis_title("Rows"), "rangemode": "tozero"},
        },
    }
