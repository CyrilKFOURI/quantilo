"""The app's chart theme: the colour tokens of frontend/src/styles/tokens.css (light and dark) and
the base layout every figure is drawn on (frontend/src/charts/theme.ts)."""

import re
from dataclasses import dataclass
from typing import Any, Literal

Theme = Literal["light", "dark"]


@dataclass(frozen=True)
class Tokens:
    ink1: str
    ink2: str
    ink3: str
    line: str
    surface: str
    accent: str
    numeric: str
    categorical: str
    date: str
    bool: str
    text: str
    id: str
    palette: tuple[str, ...]  # categorical palette, fixed order (--chart-1…8)
    font: str = '"IBM Plex Sans", system-ui, sans-serif'


LIGHT = Tokens(
    ink1="#24201b",
    ink2="#5c544a",
    ink3="#71695f",
    line="#e2dcd3",
    surface="#fffdfa",
    accent="#ad4a27",
    numeric="#3b6ea8",
    categorical="#8a5a9e",
    date="#2f7d6d",
    bool="#8c6d1f",
    text="#6b6259",
    id="#77706a",
    palette=(
        "#2a78d6",
        "#eb6834",
        "#1baf7a",
        "#eda100",
        "#e87ba4",
        "#008300",
        "#4a3aa7",
        "#e34948",
    ),
)

DARK = Tokens(
    ink1="#ede8e1",
    ink2="#b5aca0",
    ink3="#988f83",
    line="#332e28",
    surface="#1d1a16",
    accent="#e07a50",
    numeric="#7fa7d6",
    categorical="#c39ad3",
    date="#6cc0ad",
    bool="#d4b25e",
    text="#b0a69b",
    id="#a39b93",
    palette=(
        "#3987e5",
        "#d95926",
        "#199e70",
        "#c98500",
        "#d55181",
        "#008300",
        "#9085e9",
        "#e66767",
    ),
)


def tokens_for(theme: Theme) -> Tokens:
    return DARK if theme == "dark" else LIGHT


def base_layout(tokens: Tokens) -> dict[str, Any]:
    axis = {
        "color": tokens.ink3,
        "gridcolor": tokens.line,
        "linecolor": tokens.line,
        "zerolinecolor": tokens.line,
        "tickfont": {"size": 11, "color": tokens.ink3},
        "automargin": True,
    }
    return {
        # a notebook cell has no page background to show through: the surface colour instead
        "paper_bgcolor": tokens.surface,
        "plot_bgcolor": tokens.surface,
        "font": {"family": tokens.font, "size": 12, "color": tokens.ink2},
        "margin": {"l": 8, "r": 8, "t": 4, "b": 8},
        "showlegend": False,
        "bargap": 0.12,
        "hoverlabel": {
            "bgcolor": tokens.surface,
            "bordercolor": tokens.line,
            "font": {"family": tokens.font, "size": 12, "color": tokens.ink1},
        },
        "xaxis": {**axis, "showgrid": False},
        "yaxis": {**axis, "showgrid": True},
    }


def compose(
    built: dict[str, Any], tokens: Tokens, height: int, title: str | None
) -> dict[str, Any]:
    """The figure's layout over the base one, axes merged key by key (as the app's Plot)."""
    base = base_layout(tokens)
    layout = dict(built.get("layout") or {})
    merged = {
        **base,
        **layout,
        "xaxis": {**base["xaxis"], **layout.get("xaxis", {})},
        "yaxis": {**base["yaxis"], **layout.get("yaxis", {})},
        # a figure that widens one side keeps the base's other sides (not plotly's 80-100 px)
        "margin": {**base["margin"], **layout.get("margin", {})},
        "height": height,
    }
    if title:
        margin = dict(merged.get("margin", {}))
        margin["t"] = max(margin.get("t", 0), 36)
        merged["margin"] = margin
        merged["title"] = {
            "text": title,
            "x": 0,
            "xanchor": "left",
            "font": {"size": 14, "color": tokens.ink1},
        }
    return {"data": built.get("data", []), "layout": merged}


def with_alpha(hex_color: str, alpha: float) -> str:
    """ "#rrggbb" + alpha → "rgba(r, g, b, a)"."""
    match = re.fullmatch(r"#?([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})", hex_color.strip(), re.I)
    if not match:
        return hex_color
    r, g, b = (int(part, 16) for part in match.groups())
    return f"rgba({r}, {g}, {b}, {alpha})"


def format_number(value: float | None) -> str:
    """As the app's formatNumber: exponents for very large or small, at most 2 decimals."""
    if value is None:
        return "—"
    magnitude = abs(value)
    if magnitude != 0 and (magnitude >= 1e6 or magnitude < 0.01):
        return f"{value:.2e}".replace("e+0", "e+").replace("e-0", "e-")
    return f"{value:,.0f}" if magnitude >= 100 else f"{value:,.2f}".rstrip("0").rstrip(".")


def count(value: int) -> str:
    return f"{value:,}"


def axis_title(text: str, size: int = 11) -> dict[str, Any]:
    return {"title": {"text": text, "font": {"size": size}}}


LEGEND = {
    "showlegend": True,
    "legend": {"orientation": "h", "x": 0, "y": 1.12, "font": {"size": 11}},
}
