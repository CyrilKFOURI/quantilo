import type { Data, Layout } from "plotly.js-dist-min";

import type { Regression } from "../../api/client";
import type { ChartTokens } from "../../charts/theme";

type Built = { data: Data[]; layout?: Partial<Layout> };
type Plots = NonNullable<Regression["plots"]>;

const axis = (text: string) => ({ title: { text, font: { size: 11 } }, zeroline: false });

/** Residuals against fitted values: a flat band means straight lines and an even spread. */
export function residualsFigure(plots: Plots): (tokens: ChartTokens) => Built {
  return (tokens) => {
    const x = plots.fitted;
    const low = Math.min(...x);
    const high = Math.max(...x);
    return {
      data: [
        {
          type: "scatter",
          mode: "lines",
          x: [low, high],
          y: [0, 0],
          line: { color: tokens.ink3, width: 1, dash: "dash" },
          hoverinfo: "skip",
        } as Data,
        {
          type: "scatter",
          mode: "markers",
          x,
          y: plots.residuals,
          customdata: plots.rows,
          marker: { color: tokens.numeric, size: 4, opacity: 0.45 },
          hovertemplate: "row %{customdata}<br>fitted %{x:,.4g}<br>residual %{y:,.4g}<extra></extra>",
        } as Data,
      ],
      layout: { showlegend: false, xaxis: axis("Fitted values"), yaxis: axis("Residuals") },
    };
  };
}

/** Leverage against Cook's distance, with the 4/n line: rows above it pull the coefficients. */
export function influenceFigure(plots: Plots): (tokens: ChartTokens) => Built {
  return (tokens) => {
    const high = Math.max(...plots.leverage, 0);
    const above = plots.cook.map((cook) => cook > plots.cook_threshold);
    return {
      data: [
        {
          type: "scatter",
          mode: "lines",
          x: [0, high],
          y: [plots.cook_threshold, plots.cook_threshold],
          line: { color: tokens.ink3, width: 1, dash: "dash" },
          hoverinfo: "skip",
        } as Data,
        {
          type: "scatter",
          mode: "markers",
          x: plots.leverage,
          y: plots.cook,
          customdata: plots.rows,
          marker: {
            color: above.map((flag) => (flag ? tokens.accent : tokens.numeric)),
            size: 5,
            opacity: 0.55,
          },
          hovertemplate: "row %{customdata}<br>leverage %{x:.3g}<br>Cook %{y:.3g}<extra></extra>",
        } as Data,
      ],
      layout: {
        showlegend: false,
        xaxis: axis("Leverage"),
        yaxis: axis("Cook's distance"),
        annotations: [
          {
            x: high,
            y: plots.cook_threshold,
            xanchor: "right",
            yanchor: "bottom",
            text: "4/n",
            showarrow: false,
            font: { size: 10, color: tokens.ink2 },
          },
        ],
      },
    };
  };
}
