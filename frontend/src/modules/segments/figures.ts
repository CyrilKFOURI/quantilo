import type { Data } from "plotly.js-dist-min";

import type { Figure } from "../../charts/Plot";
import { withAlpha } from "../../charts/figures";
import type { ChartTokens } from "../../charts/theme";
import type { SegmentDendrogram, SegmentMap, SegmentScore, Segmentation, UnusualResult } from "./api";
import { t, type MessageKey } from "../../lib/i18n";

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
        hovertemplate: `${name}<br>${t("fig.seg.row")} %{customdata}<extra></extra>`,
      } as Data;
    });
    const centres = shown.centres.filter((c) => c.segment >= 0);
    traces.push({
      type: "scatter",
      mode: "markers+text",
      name: t("fig.seg.centres"),
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
        xaxis: { ...title(pca ? t("fig.seg.component", { n: "1", share: share(0) }) : "t-SNE 1"), zeroline: false },
        yaxis: { ...title(pca ? t("fig.seg.component", { n: "2", share: share(1) }) : "t-SNE 2"), zeroline: false },
      },
    };
  };
}

/** Each score with its label and its short name, translated when shown (127). */
const SCORES: { key: keyof SegmentScore; label: MessageKey; name: MessageKey }[] = [
  { key: "silhouette", label: "fig.seg.silhouette", name: "fig.seg.silhouetteName" },
  { key: "calinski_harabasz", label: "fig.seg.calinski", name: "fig.seg.calinskiName" },
  { key: "davies_bouldin", label: "fig.seg.davies", name: "fig.seg.daviesName" },
  { key: "inertia", label: "fig.seg.inertia", name: "fig.seg.inertiaName" },
  { key: "bic", label: "fig.seg.bic", name: "fig.seg.bicName" },
];

/** The scores this run has (BIC only for a Gaussian mixture). */
export function shownScores(scores: SegmentScore[]): { key: keyof SegmentScore; label: string }[] {
  return SCORES.filter((s) => scores.some((score) => score[s.key] !== null)).map((s) => ({ key: s.key, label: t(s.label) }));
}

/** One score against k, the chosen k marked: the page shows one small chart per score side by
 * side, never two y-scales on one chart. */
export function scoreFigure(scores: SegmentScore[], key: keyof SegmentScore, chosen: number): Figure {
  return (tokens) => {
    const points = scores.filter((score) => score[key] !== null);
    const found = SCORES.find((s) => s.key === key);
    const label = found ? t(found.name) : key;
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
        hovertemplate: t("fig.seg.hoverSubsample"),
      } as Data,
    ],
    layout: {
      yaxis: { ...title(t("fig.seg.agreement")), range: [0, 1.05] },
      xaxis: title(t("fig.seg.subsample")),
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
      yaxis: title(t("fig.seg.mergeHeight")),
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
          hovertemplate: t("fig.seg.hoverRank"),
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
        xaxis: { ...title(t("fig.seg.rowsUnusualFirst")), type: "log" },
        yaxis: title(t("fig.seg.unusualScore")),
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
