import type { Data } from "plotly.js-dist-min";

import type { Figure } from "../../charts/Plot";
import { withAlpha } from "../../charts/figures";
import type { ChartTokens } from "../../charts/theme";
import type { SegmentDendrogram, SegmentMap, SegmentScore, Segmentation, UnusualResult } from "./api";

const title = (text: string) => ({ title: { text, font: { size: 11 } } });

/** A segment keeps its palette colour everywhere (fixed order, never cycled past 8: names say
 * which is which); "Not grouped" is muted grey. */
export function segmentColor(segment: number, tokens: ChartTokens): string {
  if (segment < 0) return tokens.ink3;
  return tokens.palette[segment % tokens.palette.length] ?? tokens.ink2;
}

/** The rows in 2D, one trace per segment (legend + names at the centres: never colour alone). */
export function mapFigure(shown: SegmentMap, names: Record<number, string>): Figure {
  return (tokens) => {
    const segments = [...new Set(shown.segments)].sort((a, b) => (a < 0 ? 1 : b < 0 ? -1 : a - b));
    const traces: Data[] = segments.map((segment) => {
      const index = shown.segments.flatMap((s, i) => (s === segment ? [i] : []));
      const name = names[segment] ?? String(segment);
      return {
        type: "scattergl",
        mode: "markers",
        name,
        x: index.map((i) => shown.x[i]),
        y: index.map((i) => shown.y[i]),
        customdata: index.map((i) => shown.rows[i]),
        marker: { color: segmentColor(segment, tokens), size: 6, opacity: segment < 0 ? 0.35 : 0.6 },
        hovertemplate: `${name}<br>row %{customdata}<extra></extra>`,
      } as Data;
    });
    const centres = shown.centres.filter((c) => c.segment >= 0);
    traces.push({
      type: "scatter",
      mode: "markers+text",
      name: "Centres",
      showlegend: false,
      x: centres.map((c) => c.x),
      y: centres.map((c) => c.y),
      text: centres.map((c) => names[c.segment] ?? String(c.segment)),
      textposition: "top center",
      textfont: { color: tokens.ink1, size: 12 },
      marker: {
        color: centres.map((c) => segmentColor(c.segment, tokens)),
        size: 12,
        symbol: "diamond",
        line: { color: tokens.surface, width: 2 },
      },
      hoverinfo: "skip",
    } as Data);
    const share = (i: number) => `${Math.round((shown.explained?.[i] ?? 0) * 100)} %`;
    const pca = shown.projection === "pca";
    return {
      data: traces,
      layout: {
        showlegend: true,
        legend: { orientation: "h", x: 0, y: 1.12, font: { size: 11 } },
        xaxis: { ...title(pca ? `Component 1 (${share(0)})` : "t-SNE 1"), zeroline: false },
        yaxis: { ...title(pca ? `Component 2 (${share(1)})` : "t-SNE 2"), zeroline: false },
      },
    };
  };
}

export const SCORES: { key: keyof SegmentScore; label: string }[] = [
  { key: "silhouette", label: "Silhouette (higher is better)" },
  { key: "calinski_harabasz", label: "Calinski–Harabasz (higher is better)" },
  { key: "davies_bouldin", label: "Davies–Bouldin (lower is better)" },
  { key: "inertia", label: "Inertia (look for the elbow)" },
  { key: "bic", label: "BIC (lower is better)" },
];

/** The scores this run has (BIC only for a Gaussian mixture). */
export function shownScores(scores: SegmentScore[]) {
  return SCORES.filter((s) => scores.some((score) => score[s.key] !== null));
}

/** One score against k, the chosen k marked: the page shows one small chart per score side by
 * side, never two y-scales on one chart. */
export function scoreFigure(scores: SegmentScore[], key: keyof SegmentScore, chosen: number): Figure {
  return (tokens) => {
    const points = scores.filter((score) => score[key] !== null);
    const label = SCORES.find((s) => s.key === key)?.label.split(" (")[0] ?? key;
    return {
      data: [
        {
          type: "scatter",
          mode: "lines+markers",
          x: points.map((p) => p.k),
          y: points.map((p) => p[key]),
          line: { color: tokens.palette[0], width: 2 },
          marker: {
            size: points.map((p) => (p.k === chosen ? 10 : 7)),
            color: points.map((p) => (p.k === chosen ? tokens.accent : tokens.palette[0])),
            line: { color: tokens.surface, width: 2 },
          },
          hovertemplate: `k = %{x}<br>${label} %{y:.4g}<extra></extra>`,
        } as Data,
      ],
      layout: {
        showlegend: false,
        xaxis: { dtick: 1, ...title("k") },
        shapes: [{ type: "line", xref: "x", yref: "paper", x0: chosen, x1: chosen, y0: 0, y1: 1, line: { color: tokens.ink3, width: 1, dash: "dot" } }],
      },
    };
  };
}

/** The agreement of each subsample with the segmentation; the 0.6 line is the bar. */
export function stabilityFigure(runs: number[]): Figure {
  return (tokens) => ({
    data: [
      {
        type: "bar",
        x: runs.map((_, i) => `#${i + 1}`),
        y: runs,
        marker: { color: withAlpha(tokens.palette[0] ?? tokens.numeric, 0.8) },
        hovertemplate: "subsample %{x}<br>adjusted Rand %{y:.2f}<extra></extra>",
      } as Data,
    ],
    layout: {
      yaxis: { ...title("Agreement (adjusted Rand)"), range: [0, 1.05] },
      xaxis: title("Subsample (80 % of the rows)"),
      barcornerradius: 3,
      shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: 0.6, y1: 0.6, line: { color: tokens.ink3, width: 1, dash: "dash" } }],
    },
  });
}

export function dendrogramFigure(tree: SegmentDendrogram): Figure {
  return (tokens) => ({
    data: tree.icoord.map(
      (xs, i) =>
        ({ type: "scatter", mode: "lines", x: xs, y: tree.dcoord[i], line: { color: tokens.ink2, width: 1.5 }, hoverinfo: "skip" }) as Data,
    ),
    layout: {
      showlegend: false,
      xaxis: { tickvals: tree.leaves.map((_, i) => 5 + 10 * i), ticktext: tree.leaves, tickangle: -60, tickfont: { size: 9 } },
      yaxis: title("Merge height (Ward)"),
      shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: tree.cut, y1: tree.cut, line: { color: tokens.accent, width: 1.5, dash: "dash" } }],
    },
  });
}

/** Scores sorted from the most unusual row down (log ranks): the rows kept above the threshold
 * in the accent, the threshold as a dashed line. */
export function unusualFigure(result: UnusualResult, threshold: number): Figure {
  return (tokens) => {
    const { ranks, scores } = result.curve;
    const above = scores.flatMap((s, i) => (s >= threshold ? [i] : []));
    return {
      data: [
        {
          type: "scatter",
          mode: "lines",
          x: ranks,
          y: scores,
          line: { color: tokens.palette[0], width: 2 },
          hovertemplate: "rank %{x:,}<br>score %{y:.4g}<extra></extra>",
        } as Data,
        {
          type: "scatter",
          mode: "markers",
          x: above.map((i) => ranks[i]),
          y: above.map((i) => scores[i]),
          marker: { color: tokens.accent, size: 7 },
          hoverinfo: "skip",
        } as Data,
      ],
      layout: {
        showlegend: false,
        xaxis: { ...title("Rows, most unusual first"), type: "log" },
        yaxis: title("Unusual score"),
        shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: threshold, y1: threshold, line: { color: tokens.accent, width: 1.5, dash: "dash" } }],
      },
    };
  };
}

/** Rows at or above a threshold among the listed ones (the list is the most unusual first). */
export function rowsAbove(result: UnusualResult, threshold: number): number {
  return result.listed.filter((row) => row.score >= threshold).length;
}

export function namesOf(result: Segmentation): Record<number, string> {
  return Object.fromEntries(result.segments.map((s) => [s.id, s.name]));
}
