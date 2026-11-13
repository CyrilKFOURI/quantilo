import type { Data, Layout } from "plotly.js-dist-min";

import type { Tail } from "../../api/client";
import { withAlpha } from "../../charts/figures";
import type { ChartTokens } from "../../charts/theme";
import { t } from "../../lib/i18n";

type Built = { data: Data[]; layout?: Partial<Layout> };
type Figure = (tokens: ChartTokens) => Built;

const small = { font: { size: 11 } };

/** Return levels: the GPD curve, the largest observed values at their empirical period, and the
 *  three levels with their 95 % intervals. Both axes in log: a heavy tail reads as a straight line. */
export function returnLevelFigure(tail: Tail): Figure {
  return (tokens) => {
    if (!tail.curve || !tail.observed) return { data: [] };
    const levels = tail.return_levels;
    const plus = levels.map((r) => (r.high === null ? 0 : r.high - r.value));
    const minus = levels.map((r) => (r.low === null ? 0 : r.value - r.low));
    return {
      data: [
        {
          type: "scatter",
          mode: "markers",
          name: t("fig.tail.observed"),
          x: tail.observed.x,
          y: tail.observed.y,
          marker: { color: tokens.ink3, size: 5 },
          hovertemplate: t("fig.tail.hoverObserved"),
        } as Data,
        {
          type: "scatter",
          mode: "lines",
          name: "GPD",
          x: tail.curve.x,
          y: tail.curve.y,
          line: { color: tokens.numeric, width: 2 },
          hovertemplate: t("fig.tail.hoverGpd"),
        } as Data,
        {
          type: "scatter",
          mode: "markers",
          name: t("fig.tail.interval95"),
          x: levels.map((r) => r.period),
          y: levels.map((r) => r.value),
          error_y: { type: "data", symmetric: false, array: plus, arrayminus: minus, color: tokens.accent, thickness: 1.5 },
          marker: { color: tokens.accent, size: 8, line: { color: tokens.surface, width: 1 } },
          hovertemplate: t("fig.tail.hoverInterval"),
        } as Data,
      ],
      layout: {
        showlegend: true,
        legend: { orientation: "h", x: 0, y: 1.14, font: { size: 11 } },
        xaxis: { type: "log", title: { text: t("fig.tail.onceEvery"), ...small } },
        yaxis: { type: "log", title: { text: t("fig.tail.value"), ...small } },
      },
    };
  };
}

/** Mean excess e(u) = mean(X − u | X > u): roughly a straight line above a good threshold. */
export function meanExcessFigure(tail: Tail): Figure {
  return (tokens) => {
    const curve = tail.mean_excess;
    if (!curve || !curve.x.length) return { data: [] };
    const threshold = tail.threshold?.value;
    return {
      data: [
        {
          type: "scatter",
          mode: "lines+markers",
          x: curve.x,
          y: curve.y,
          customdata: curve.count,
          line: { color: tokens.numeric, width: 2 },
          marker: { color: tokens.numeric, size: 5 },
          hovertemplate: t("fig.tail.hoverExcess"),
        } as Data,
      ],
      layout: {
        xaxis: { title: { text: t("fig.tail.threshold"), ...small } },
        yaxis: { title: { text: t("fig.tail.meanExcess"), ...small } },
        shapes:
          threshold === undefined
            ? []
            : [
                {
                  type: "line",
                  xref: "x",
                  yref: "paper",
                  x0: threshold,
                  x1: threshold,
                  y0: 0,
                  y1: 1,
                  line: { color: tokens.accent, width: 1.5, dash: "dash" },
                },
              ],
      },
    };
  };
}

/** Hill estimates against the number k of largest values, with their band, and the GPD's ξ. */
export function hillFigure(tail: Tail): Figure {
  return (tokens) => {
    const hill = tail.hill;
    if (!hill || !hill.x.length) return { data: [] };
    const xi = tail.xi;
    return {
      data: [
        {
          type: "scatter",
          mode: "lines",
          x: hill.x,
          y: hill.high,
          line: { width: 0, color: tokens.numeric },
          hoverinfo: "skip",
          showlegend: false,
        } as Data,
        {
          type: "scatter",
          mode: "lines",
          x: hill.x,
          y: hill.low,
          fill: "tonexty",
          fillcolor: withAlpha(tokens.numeric, 0.15),
          line: { width: 0, color: tokens.numeric },
          hoverinfo: "skip",
          showlegend: false,
        } as Data,
        {
          type: "scatter",
          mode: "lines",
          x: hill.x,
          y: hill.y,
          line: { color: tokens.numeric, width: 2 },
          hovertemplate: "k = %{x:,}<br>Hill ξ %{y:.3f}<extra></extra>",
          showlegend: false,
        } as Data,
      ],
      layout: {
        xaxis: { type: "log", title: { text: t("fig.tail.kLargest"), ...small } },
        yaxis: { title: { text: "Hill ξ", ...small } },
        shapes:
          xi === null
            ? []
            : [
                {
                  type: "line",
                  xref: "paper",
                  yref: "y",
                  x0: 0,
                  x1: 1,
                  y0: xi,
                  y1: xi,
                  line: { color: tokens.accent, width: 1.5, dash: "dash" },
                },
              ],
      },
    };
  };
}

/** The Hill estimate at the k closest to the number of values above the threshold. */
export function hillAt(tail: Tail): { k: number; xi: number } | null {
  const hill = tail.hill;
  const wanted = tail.threshold?.exceedances;
  if (!hill || !hill.x.length || wanted === undefined) return null;
  let best = 0;
  hill.x.forEach((k, index) => {
    if (Math.abs(k - wanted) < Math.abs((hill.x[best] ?? 0) - wanted)) best = index;
  });
  return { k: hill.x[best] ?? 0, xi: hill.y[best] ?? 0 };
}
