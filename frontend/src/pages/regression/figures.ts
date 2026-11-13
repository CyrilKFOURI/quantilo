import type { Data, Layout } from "plotly.js-dist-min";

import type {
  Regression,
  RegressionCurve,
  RegressionCurves,
  RegressionCusum,
  RegressionDriftCurve,
  RegressionPartial,
  RegressionQuantileCurve,
  RegressionRegimes,
  RegressionRolling,
  RegressionSurvivalCurve,
} from "../../api/client";
import { withAlpha } from "../../charts/figures";
import type { ChartTokens } from "../../charts/theme";
import { t } from "../../lib/i18n";

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
          hovertemplate: t("fig.reg.hoverFitted"),
        } as Data,
      ],
      layout: { showlegend: false, xaxis: axis(t("fig.reg.fittedValues")), yaxis: axis(t("fig.ts.residuals")) },
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
          hovertemplate: t("fig.reg.hoverLeverage"),
        } as Data,
      ],
      layout: {
        showlegend: false,
        xaxis: axis(t("fig.reg.leverage")),
        yaxis: axis(t("fig.reg.cook")),
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
          hovertemplate: `${plain(curve.column)} %{x:,.4g}<br>${t("fig.reg.fitted")} %{y:,.4g}<extra>${plain(line.label)}</extra>`,
        } as Data,
      );
    });
    return {
      data,
      layout: {
        xaxis: axis(plain(curve.column)),
        yaxis: axis(t("fig.reg.fittedScale", { scale: plain(curve.scale) })),
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
          hovertemplate: t("fig.reg.hoverRowPoint"),
        } as Data,
        { type: "scatter", mode: "lines", x: [low, high], y: [slope * low, slope * high], line: { color: tokens.accent, width: 2 }, hoverinfo: "skip" } as Data,
      ],
      layout: { showlegend: false, xaxis: axis(t("fig.reg.otherTerms", { column: plain(plot.column) })), yaxis: axis(t("fig.reg.targetOtherTerms")) },
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
        hovertemplate: t("fig.reg.hoverRowPoint"),
      } as Data,
      { type: "scatter", mode: "lines", x: plot.line_x, y: plot.line_y, line: { color: tokens.accent, width: 2 }, hoverinfo: "skip" } as Data,
    ],
    layout: { showlegend: false, xaxis: axis(plain(plot.column)), yaxis: axis(t("fig.reg.componentResidual")) },
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
        hovertemplate: t("fig.reg.hoverLeastSquares"),
      } as Data,
    ],
    layout: { showlegend: false, xaxis: { ...axis(t("fig.reg.quantile")), range: [0.05, 0.95] }, yaxis: axis(t("fig.reg.effect")) },
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
export function rollingFigure(curve: Pick<RegressionRolling, "dates" | "estimates" | "low" | "high" | "full">, reference = t("fig.reg.wholeSeries")): (tokens: ChartTokens) => Built {
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
        hovertemplate: `${reference} %{y:,.4g}<extra></extra>`,
      } as Data,
    ],
    layout: { showlegend: false, xaxis: { type: "date", zeroline: false }, yaxis: axis("Effect") },
  });
}

/** 075: the points and the fitted shapes, the best first in the first palette colour; the
 *  straight line dashed as the reference. */
export function curvesFigure(table: RegressionCurves, shown: number, xLabel: string, yLabel: string): (tokens: ChartTokens) => Built {
  return (tokens) => {
    const fits = table.fits.filter((fit) => fit.xs.length > 0 && fit.error === null);
    const drawn = fits.filter((fit) => fit.kind !== "line").slice(0, shown);
    const line = fits.find((fit) => fit.kind === "line");
    return {
      data: [
        {
          type: "scatter",
          mode: "markers",
          x: table.data_x,
          y: table.data_y,
          marker: { color: withAlpha(tokens.ink3, 0.5), size: 5 },
          name: t("fig.Rows"),
          hovertemplate: "%{x:,.4g}, %{y:,.4g}<extra></extra>",
        } as Data,
        ...(line
          ? [
              {
                type: "scatter",
                mode: "lines",
                x: line.xs,
                y: line.ys,
                name: line.label,
                line: { color: tokens.ink3, width: 1.5, dash: "dash" },
                hoverinfo: "skip",
              } as Data,
            ]
          : []),
        ...drawn.map(
          (fit, i) =>
            ({
              type: "scatter",
              mode: "lines",
              x: fit.xs,
              y: fit.ys,
              name: fit.label,
              line: { color: tokens.palette[i % tokens.palette.length] ?? tokens.numeric, width: i === 0 ? 2.5 : 1.5 },
              hovertemplate: `${fit.label}<br>%{x:,.4g}, %{y:,.4g}<extra></extra>`,
            }) as Data,
        ),
      ],
      layout: { showlegend: true, legend: { orientation: "h", y: -0.25 }, xaxis: axis(xLabel), yaxis: axis(yLabel) },
    };
  };
}

/** 073: one coefficient let free to drift (Kalman smoother), its 95 % band, and the fixed value. */
export function driftFigure(curve: RegressionDriftCurve): (tokens: ChartTokens) => Built {
  return rollingFigure({ ...curve, full: curve.fixed }, t("fig.reg.fixed"));
}

/** 073: the smoothed probability of each regime along the rows, stacked to 1. */
export function regimesFigure(table: RegressionRegimes): (tokens: ChartTokens) => Built {
  return (tokens) => ({
    data: table.regimes.map(
      (regime, i) =>
        ({
          type: "scatter",
          mode: "lines",
          x: table.dates,
          y: table.probabilities[i] ?? [],
          name: regime.name,
          stackgroup: "regimes",
          line: { width: 0.5, color: tokens.palette[i % tokens.palette.length] ?? tokens.numeric },
          hovertemplate: `${regime.name}<br>%{x}<br>%{y:.0%}<extra></extra>`,
        }) as Data,
    ),
    layout: { showlegend: true, legend: { orientation: "h", y: -0.25 }, xaxis: { type: "date", zeroline: false }, yaxis: { ...axis("Probability"), range: [0, 1], tickformat: ".0%" } },
  });
}

/** 076: adjusted survival curves (steps), one per profile or stratum. */
export function survivalFigure(curves: RegressionSurvivalCurve[], timeLabel: string): (tokens: ChartTokens) => Built {
  return (tokens) => ({
    data: curves.map(
      (curve, i) =>
        ({
          type: "scatter",
          mode: "lines",
          x: curve.times,
          y: curve.survival,
          name: curve.label,
          line: { shape: "hv", width: 2, color: tokens.palette[i % tokens.palette.length] ?? tokens.numeric },
          hovertemplate: `${curve.label}<br>%{x:,.4g}: %{y:.0%}<extra></extra>`,
        }) as Data,
    ),
    layout: { showlegend: true, legend: { orientation: "h", y: -0.25 }, xaxis: axis(timeLabel), yaxis: { ...axis(t("fig.reg.stillWithout")), range: [0, 1.02], tickformat: ".0%" } },
  });
}
