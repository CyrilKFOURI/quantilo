import type { Data, Layout } from "plotly.js-dist-min";

import type { Regression, RegressionCurve, RegressionCusum, RegressionPartial, RegressionQuantileCurve, RegressionRolling } from "../../api/client";
import { withAlpha } from "../../charts/figures";
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

const plain = (text: string) => text.replaceAll("`", "");

/** 070 (R-REG-TERM-10): the fitted target along a column, with its 95 % band; one line per
 *  level of the partner of an interaction. */
export function effectCurveFigure(curve: RegressionCurve): (tokens: ChartTokens) => Built {
  return (tokens) => {
    const data: Data[] = [];
    const several = curve.lines.length > 1;
    curve.lines.forEach((line, i) => {
      const color = several ? (tokens.palette[i % tokens.palette.length] ?? tokens.numeric) : tokens.numeric;
      data.push(
        { type: "scatter", mode: "lines", x: curve.x, y: line.high, line: { width: 0, color }, hoverinfo: "skip", showlegend: false } as Data,
        {
          type: "scatter",
          mode: "lines",
          x: curve.x,
          y: line.low,
          fill: "tonexty",
          fillcolor: withAlpha(color, 0.12),
          line: { width: 0, color },
          hoverinfo: "skip",
          showlegend: false,
        } as Data,
        {
          type: "scatter",
          mode: "lines",
          name: plain(line.label),
          x: curve.x,
          y: line.y,
          line: { color, width: 2 },
          showlegend: several,
          hovertemplate: `${plain(curve.column)} %{x:,.4g}<br>fitted %{y:,.4g}<extra>${plain(line.label)}</extra>`,
        } as Data,
      );
    });
    return {
      data,
      layout: {
        xaxis: axis(plain(curve.column)),
        yaxis: axis(`Fitted ${plain(curve.scale)}`),
        showlegend: several,
        legend: { orientation: "h", x: 0, y: 1.02, yanchor: "bottom", font: { size: 11 } },
      },
    };
  };
}

/** 070 (R-REG-TERM-11): residuals of the target against residuals of the column once the
 *  other terms are taken out; the slope of the line is the coefficient. */
export function addedVariableFigure(plot: RegressionPartial): (tokens: ChartTokens) => Built {
  return (tokens) => {
    const low = Math.min(...plot.x_resid);
    const high = Math.max(...plot.x_resid);
    const slope = plot.slope ?? 0;
    return {
      data: [
        {
          type: "scatter",
          mode: "markers",
          x: plot.x_resid,
          y: plot.y_resid,
          customdata: plot.rows,
          marker: { color: tokens.numeric, size: 4, opacity: 0.4 },
          hovertemplate: "row %{customdata}<br>%{x:,.4g} ; %{y:,.4g}<extra></extra>",
        } as Data,
        { type: "scatter", mode: "lines", x: [low, high], y: [slope * low, slope * high], line: { color: tokens.accent, width: 2 }, hoverinfo: "skip" } as Data,
      ],
      layout: { showlegend: false, xaxis: axis(`${plain(plot.column)} | other terms`), yaxis: axis("Target | other terms") },
    };
  };
}

/** 070: the estimated contribution of the column (its curve included) plus the residuals. */
export function componentFigure(plot: RegressionPartial): (tokens: ChartTokens) => Built {
  return (tokens) => ({
    data: [
      {
        type: "scatter",
        mode: "markers",
        x: plot.x,
        y: plot.component_resid,
        customdata: plot.rows,
        marker: { color: tokens.numeric, size: 4, opacity: 0.4 },
        hovertemplate: "row %{customdata}<br>%{x:,.4g} ; %{y:,.4g}<extra></extra>",
      } as Data,
      { type: "scatter", mode: "lines", x: plot.line_x, y: plot.line_y, line: { color: tokens.accent, width: 2 }, hoverinfo: "skip" } as Data,
    ],
    layout: { showlegend: false, xaxis: axis(plain(plot.column)), yaxis: axis("Component + residual") },
  });
}

/** 072 (R-REG-QR-02): a coefficient along τ with its band, least squares dashed. */
export function quantileFigure(curve: RegressionQuantileCurve): (tokens: ChartTokens) => Built {
  return (tokens) => ({
    data: [
      { type: "scatter", mode: "lines", x: curve.taus, y: curve.high, line: { width: 0, color: tokens.numeric }, hoverinfo: "skip" } as Data,
      {
        type: "scatter",
        mode: "lines",
        x: curve.taus,
        y: curve.low,
        fill: "tonexty",
        fillcolor: withAlpha(tokens.numeric, 0.15),
        line: { width: 0, color: tokens.numeric },
        hoverinfo: "skip",
      } as Data,
      {
        type: "scatter",
        mode: "lines+markers",
        x: curve.taus,
        y: curve.estimates,
        line: { color: tokens.numeric, width: 2 },
        marker: { size: 5 },
        hovertemplate: "τ = %{x}<br>%{y:,.4g}<extra></extra>",
      } as Data,
      {
        type: "scatter",
        mode: "lines",
        x: [curve.taus[0], curve.taus[curve.taus.length - 1]],
        y: [curve.ols, curve.ols],
        line: { color: tokens.ink3, width: 1.5, dash: "dash" },
        hovertemplate: "least squares %{y:,.4g}<extra></extra>",
      } as Data,
    ],
    layout: { showlegend: false, xaxis: { ...axis("Quantile τ"), range: [0.05, 0.95] }, yaxis: axis("Effect") },
  });
}

/** 073: a CUSUM path inside its 5 % band; leaving the band says the coefficients drift. */
export function cusumFigure(path: RegressionCusum, label: string): (tokens: ChartTokens) => Built {
  return (tokens) => ({
    data: [
      { type: "scatter", mode: "lines", x: path.dates, y: path.upper, line: { color: tokens.ink3, width: 1, dash: "dash" }, hoverinfo: "skip" } as Data,
      { type: "scatter", mode: "lines", x: path.dates, y: path.lower, line: { color: tokens.ink3, width: 1, dash: "dash" }, hoverinfo: "skip" } as Data,
      {
        type: "scatter",
        mode: "lines",
        x: path.dates,
        y: path.values,
        line: { color: tokens.numeric, width: 2 },
        hovertemplate: "%{x}<br>%{y:,.3g}<extra></extra>",
      } as Data,
    ],
    layout: { showlegend: false, xaxis: { type: "date", zeroline: false }, yaxis: axis(label) },
  });
}

/** 073: one coefficient over rolling windows, its 95 % band, and the whole-series value. */
export function rollingFigure(curve: RegressionRolling): (tokens: ChartTokens) => Built {
  return (tokens) => ({
    data: [
      { type: "scatter", mode: "lines", x: curve.dates, y: curve.high, line: { width: 0, color: tokens.numeric }, hoverinfo: "skip" } as Data,
      {
        type: "scatter",
        mode: "lines",
        x: curve.dates,
        y: curve.low,
        fill: "tonexty",
        fillcolor: withAlpha(tokens.numeric, 0.15),
        line: { width: 0, color: tokens.numeric },
        hoverinfo: "skip",
      } as Data,
      {
        type: "scatter",
        mode: "lines",
        x: curve.dates,
        y: curve.estimates,
        line: { color: tokens.numeric, width: 2 },
        hovertemplate: "%{x}<br>%{y:,.4g}<extra></extra>",
      } as Data,
      {
        type: "scatter",
        mode: "lines",
        x: [curve.dates[0], curve.dates[curve.dates.length - 1]],
        y: [curve.full, curve.full],
        line: { color: tokens.ink3, width: 1.5, dash: "dash" },
        hovertemplate: "whole series %{y:,.4g}<extra></extra>",
      } as Data,
    ],
    layout: { showlegend: false, xaxis: { type: "date", zeroline: false }, yaxis: axis("Effect") },
  });
}
