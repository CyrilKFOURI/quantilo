import type { Data, Layout } from "plotly.js-dist-min";

import type { Figure } from "../../charts/Plot";
import { withAlpha } from "../../charts/figures";
import type { ChartTokens } from "../../charts/theme";
import type { Curve } from "./scores";
import { t } from "../../lib/i18n";

/** 113: the Results charts, on `charts/Plot` and the theme tokens (047). Series take the
 *  categorical palette in the fixed order of the run's labels, never by rank. */

export type NamedCurve = { name: string; curve: Curve };

const colour = (tokens: ChartTokens, slot: number) => tokens.palette[slot % tokens.palette.length] ?? tokens.numeric;

const unitAxes = (x: string, y: string, tokens: ChartTokens): Partial<Layout> => ({
  xaxis: { title: { text: x, font: { size: 11, color: tokens.ink3 } }, range: [-0.01, 1.01], showgrid: true, gridcolor: tokens.line, zeroline: false },
  yaxis: { title: { text: y, font: { size: 11, color: tokens.ink3 } }, range: [-0.01, 1.03], zeroline: false },
  showlegend: false,
  margin: { l: 8, r: 8, t: 8, b: 8 },
});

/** ROC (or PR) curves, one line per class (2 px, palette order), the chance line in muted ink. The
 *  legend (swatch, class, area) is written under the chart in HTML: labels on the lines collide
 *  where good curves crowd the corner. */
export function curvesFigure(curves: NamedCurve[], kind: "roc" | "pr", baseRate: number | null = null): Figure {
  return (tokens) => {
    const chance: Data =
      kind === "roc"
        ? { type: "scatter", mode: "lines", x: [0, 1], y: [0, 1], line: { color: tokens.ink3, width: 1, dash: "dot" }, hoverinfo: "skip", name: t("fig.vision.chance") }
        : { type: "scatter", mode: "lines", x: [0, 1], y: [baseRate ?? 0, baseRate ?? 0], line: { color: tokens.ink3, width: 1, dash: "dot" }, hoverinfo: "skip", name: t("fig.vision.chance") };
    const lines: Data[] = curves.map(({ name, curve }, slot) => ({
      type: "scatter",
      mode: "lines",
      name,
      x: curve.x,
      y: curve.y,
      line: { color: colour(tokens, slot), width: 2, shape: kind === "pr" ? "hv" : "linear" },
      hovertemplate:
        kind === "roc"
          ? `${name}<br>${t("fig.vision.hoverRoc")}<extra>AUC ${curve.area.toFixed(3)}</extra>`
          : `${name}<br>${t("fig.vision.hoverPr")}<extra>AP ${curve.area.toFixed(3)}</extra>`,
    }));
    const axes = kind === "roc" ? unitAxes(t("fig.vision.falseAlarmRate"), t("fig.vision.recallTpr"), tokens) : unitAxes(t("fig.ml.recall"), t("fig.ml.precision"), tokens);
    return { data: [chance, ...lines], layout: { ...axes, hovermode: "closest" } };
  };
}

export type ReliabilityBins = { edges: number[]; confidence: (number | null)[]; accuracy: (number | null)[]; counts: number[] };

/** Reliability diagram: accuracy against confidence per bin (dots sized by images, a 2 px line),
 *  the diagonal is perfect calibration; the gap to it is what the ECE averages. */
export function reliabilityFigure(bins: ReliabilityBins): Figure {
  return (tokens) => {
    const kept = bins.counts.map((count, k) => ({ count, k })).filter(({ count, k }) => count > 0 && bins.confidence[k] !== null);
    const total = bins.counts.reduce((a, b) => a + b, 0) || 1;
    const x = kept.map(({ k }) => bins.confidence[k] ?? 0);
    const y = kept.map(({ k }) => bins.accuracy[k] ?? 0);
    const gaps: Data[] = kept.map(({ k }) => ({
      type: "scatter",
      mode: "lines",
      x: [bins.confidence[k] ?? 0, bins.confidence[k] ?? 0],
      y: [bins.confidence[k] ?? 0, bins.accuracy[k] ?? 0],
      line: { color: withAlpha(tokens.palette[1] ?? tokens.accent, 0.55), width: 2 },
      hoverinfo: "skip",
    }));
    return {
      data: [
        { type: "scatter", mode: "lines", x: [0, 1], y: [0, 1], line: { color: tokens.ink3, width: 1, dash: "dot" }, hoverinfo: "skip" },
        ...gaps,
        {
          type: "scatter",
          mode: "lines+markers",
          x,
          y,
          customdata: kept.map(({ count }) => [count, count / total]),
          line: { color: colour(tokens, 0), width: 2 },
          marker: { color: colour(tokens, 0), size: kept.map(({ count }) => 8 + 10 * Math.sqrt(count / total)), line: { color: tokens.surface, width: 2 } },
          hovertemplate: t("fig.vision.hoverConfidence"),
        } as Data,
      ],
      layout: unitAxes(t("fig.vision.confidence"), t("fig.vision.shareRight"), tokens),
    };
  };
}

/** One bar per item (AUROC per defect type, silhouette per k), horizontal, value at the bar end. */
export function barsFigure(items: { name: string; value: number; highlight?: boolean }[], options: { range?: [number, number]; format?: string } = {}): Figure {
  return (tokens) => ({
    data: [
      {
        type: "bar",
        orientation: "h",
        y: items.map((item) => item.name),
        x: items.map((item) => item.value),
        marker: { color: items.map((item) => (item.highlight === false ? withAlpha(colour(tokens, 0), 0.45) : colour(tokens, 0))), line: { color: tokens.surface, width: 2 } },
        text: items.map((item) => item.value.toFixed(3)),
        textposition: "outside",
        textfont: { size: 11, color: tokens.ink2 },
        cliponaxis: false,
        hovertemplate: `%{y}: %{x:${options.format ?? ".3f"}}<extra></extra>`,
      } as Data,
    ],
    layout: {
      xaxis: { range: options.range, showgrid: true, gridcolor: tokens.line, zeroline: false },
      yaxis: { autorange: "reversed", showgrid: false, automargin: true },
      bargap: 0.35,
      margin: { l: 8, r: 40, t: 4, b: 8 },
    },
  });
}
