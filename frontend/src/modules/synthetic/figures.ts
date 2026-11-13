import type { Data, Layout } from "plotly.js-dist-min";

import type { SyntheticColumnDetail, SyntheticPair, SyntheticReport, SyntheticRunSummary } from "../../api/client";
import type { Figure } from "../../charts/Plot";
import { withAlpha } from "../../charts/figures";
import type { ChartTokens } from "../../charts/theme";
import { t } from "../../lib/i18n";

/** One fixed look per set on the whole page: real = solid dark grey, synthetic = accent outline
 *  (and dashed lines), so the two never depend on colour alone (spec 023, section 7). */
export function looks(tokens: ChartTokens) {
  return {
    real: { name: t("fig.syn.real"), color: tokens.ink2, fill: withAlpha(tokens.ink2, 0.45) },
    synthetic: { name: t("fig.syn.synthetic"), color: tokens.accent, fill: withAlpha(tokens.accent, 0.12) },
  };
}

const legend: Partial<Layout> = {
  showlegend: true,
  legend: { orientation: "h", x: 0, y: 1.12, font: { size: 11 } },
};
const axisTitle = (text: string) => ({ title: { text, font: { size: 11 } } });
const percentAxis = { tickformat: ".0%", rangemode: "tozero" as const };

function bars(tokens: ChartTokens, x: (string | number)[], real: number[], synthetic: number[], width?: number[]): Data[] {
  const look = looks(tokens);
  return [
    {
      type: "bar",
      name: look.real.name,
      x,
      y: real,
      width,
      marker: { color: look.real.fill, line: { width: 0 } },
      hovertemplate: `%{x}<br>${t("fig.syn.realLower")} %{y:.1%}<extra></extra>`,
    } as Data,
    {
      type: "bar",
      name: look.synthetic.name,
      x,
      y: synthetic,
      width,
      marker: { color: look.synthetic.fill, line: { color: look.synthetic.color, width: 1.5 } },
      hovertemplate: `%{x}<br>${t("fig.syn.syntheticLower")} %{y:.1%}<extra></extra>`,
    } as Data,
  ];
}

/** Both sets on the same intervals (computed on the real data), as shares of each set. */
export function histogramFigure(detail: SyntheticColumnDetail, dates: boolean): Figure {
  return (tokens) => {
    const edges = detail.histogram?.edges ?? [];
    const centres = edges.slice(0, -1).map((e, i) => (e + (edges[i + 1] ?? e)) / 2);
    const widths = edges.slice(0, -1).map((e, i) => (edges[i + 1] ?? e) - e);
    const x = dates ? centres.map((c) => new Date(c * 1000).toISOString()) : centres;
    const width = dates ? widths.map((w) => w * 1000) : widths;
    return {
      data: bars(tokens, x, detail.histogram?.real ?? [], detail.histogram?.synthetic ?? [], width),
      layout: { ...legend, barmode: "overlay", bargap: 0, yaxis: { ...percentAxis, ...axisTitle(t("fig.shareOfRows")) } },
    };
  };
}

/** Cumulative curves: a cut tail shows as one curve reaching 100 % early. */
export function ecdfFigure(detail: SyntheticColumnDetail, dates: boolean): Figure {
  return (tokens) => {
    const look = looks(tokens);
    const ecdf = detail.ecdf;
    const x = (values: number[]) => (dates ? values.map((v) => new Date(v * 1000).toISOString()) : values);
    return {
      data: [
        {
          type: "scatter",
          mode: "lines",
          name: look.real.name,
          x: x(ecdf?.real_x ?? []),
          y: ecdf?.levels ?? [],
          line: { color: look.real.color, width: 2, shape: "hv" },
          hovertemplate: t("fig.syn.hoverRealBelow"),
        } as Data,
        {
          type: "scatter",
          mode: "lines",
          name: look.synthetic.name,
          x: x(ecdf?.synthetic_x ?? []),
          y: ecdf?.levels ?? [],
          line: { color: look.synthetic.color, width: 2, dash: "dash", shape: "hv" },
          hovertemplate: t("fig.syn.hoverSyntheticBelow"),
        } as Data,
      ],
      layout: { ...legend, yaxis: { ...percentAxis, ...axisTitle(t("fig.syn.shareBelow")) } },
    };
  };
}

/** Levels side by side, horizontal, most frequent in the real data on top. */
export function levelsFigure(detail: SyntheticColumnDetail): Figure {
  return (tokens) => {
    const look = looks(tokens);
    const rows = [...detail.levels].reverse();
    const y = rows.map((r) => r.value);
    return {
      data: [
        {
          type: "bar",
          orientation: "h",
          name: look.real.name,
          y,
          x: rows.map((r) => r.real_share),
          marker: { color: look.real.fill },
          hovertemplate: `%{y}<br>${t("fig.syn.realLower")} %{x:.1%}<extra></extra>`,
        } as Data,
        {
          type: "bar",
          orientation: "h",
          name: look.synthetic.name,
          y,
          x: rows.map((r) => r.synthetic_share),
          marker: { color: look.synthetic.fill, line: { color: look.synthetic.color, width: 1.5 } },
          hovertemplate: `%{y}<br>${t("fig.syn.syntheticLower")} %{x:.1%}<extra></extra>`,
        } as Data,
      ],
      layout: { ...legend, barmode: "group", bargap: 0.25, xaxis: { ...percentAxis, showgrid: true }, yaxis: { showgrid: false } },
    };
  };
}

/** The target's mean (or share) per interval or level, in both sets: is the link kept? */
export function targetFigure(target: NonNullable<SyntheticColumnDetail["target"]>): Figure {
  return (tokens) => {
    const look = looks(tokens);
    return {
      data: [
        {
          type: "scatter",
          mode: "lines+markers",
          name: look.real.name,
          x: target.groups,
          y: target.real,
          line: { color: look.real.color, width: 2 },
          marker: { size: 8 },
          connectgaps: true,
        } as Data,
        {
          type: "scatter",
          mode: "lines+markers",
          name: look.synthetic.name,
          x: target.groups,
          y: target.synthetic,
          line: { color: look.synthetic.color, width: 2, dash: "dash" },
          marker: { size: 8, symbol: "circle-open" },
          connectgaps: true,
        } as Data,
      ],
      layout: { ...legend, yaxis: axisTitle(target.measure), xaxis: { type: "category" } },
    };
  };
}

/** Tiny overlaid histogram of a column line (no axes). */
export function miniFigure(real: number[], synthetic: number[]): Figure {
  return (tokens) => ({
    data: bars(tokens, real.map((_, i) => i), real, synthetic),
    layout: {
      barmode: "overlay",
      bargap: 0.1,
      margin: { l: 0, r: 0, t: 0, b: 0 },
      xaxis: { visible: false },
      yaxis: { visible: false },
      hovermode: false,
    },
  });
}

type Links = SyntheticReport["links"];

/** One association matrix: signed blue–red for real and synthetic, sequential |gap| capped at
 *  0.3 for the difference (as table-evaluator). */
export function linksFigure(links: Links, which: "real" | "synthetic" | "gap"): Figure {
  return (tokens) => {
    const values = links[which];
    const gap = which === "gap";
    return {
      data: [
        {
          type: "heatmap",
          x: links.columns,
          y: links.columns,
          z: values,
          zmin: gap ? 0 : -1,
          zmax: gap ? 0.3 : 1,
          colorscale: gap
            ? [
                [0, tokens.surface],
                [1, tokens.accent],
              ]
            : [
                [0, tokens.palette[1] ?? "#c0392b"],
                [0.5, tokens.surface],
                [1, tokens.palette[0] ?? "#2a78d6"],
              ],
          xgap: 1,
          ygap: 1,
          showscale: false,
          hovertemplate: "%{y} · %{x}<br>%{z:.2f}<extra></extra>",
        } as Data,
      ],
      layout: {
        xaxis: { showgrid: false, tickangle: -45, tickfont: { size: 10 } },
        yaxis: { showgrid: false, autorange: "reversed", tickfont: { size: 10 } },
      },
    };
  };
}

/** A pair in both sets: scatter (two numbers), boxes per level (mixed). */
export function pairFigure(pair: SyntheticPair): Figure {
  return (tokens) => {
    const look = looks(tokens);
    if (pair.kind === "numbers") {
      return {
        data: [
          {
            type: "scattergl",
            mode: "markers",
            name: look.real.name,
            x: pair.real_points.map((p) => p[0]),
            y: pair.real_points.map((p) => p[1]),
            marker: { color: look.real.fill, size: 5 },
          } as Data,
          {
            type: "scattergl",
            mode: "markers",
            name: look.synthetic.name,
            x: pair.synthetic_points.map((p) => p[0]),
            y: pair.synthetic_points.map((p) => p[1]),
            marker: { color: "rgba(0,0,0,0)", size: 6, line: { color: look.synthetic.color, width: 1 } },
          } as Data,
        ],
        layout: { ...legend, xaxis: axisTitle(pair.a), yaxis: axisTitle(pair.b) },
      };
    }
    const box = (side: "real" | "synthetic"): Data =>
      ({
        type: "box",
        name: look[side].name,
        x: pair.boxes.map((b) => b.group),
        lowerfence: pair.boxes.map((b) => b[side][0]),
        q1: pair.boxes.map((b) => b[side][1]),
        median: pair.boxes.map((b) => b[side][2]),
        q3: pair.boxes.map((b) => b[side][3]),
        upperfence: pair.boxes.map((b) => b[side][4]),
        fillcolor: look[side].fill,
        line: { color: look[side].color, width: 1.5 },
      }) as Data;
    return {
      data: [box("real"), box("synthetic")],
      layout: { ...legend, boxmode: "group", yaxis: axisTitle(pair.b), xaxis: { type: "category" } },
    };
  };
}

/** Distances to the closest training row: synthetic rows vs new (held-out) real rows. */
export function dcrFigure(histogram: NonNullable<SyntheticReport["privacy"]["histogram"]>): Figure {
  return (tokens) => {
    const look = looks(tokens);
    const edges = histogram.edges;
    const x = edges.slice(0, -1).map((e, i) => (e + (edges[i + 1] ?? e)) / 2);
    return {
      data: [
        {
          type: "bar",
          name: t("fig.syn.heldOut"),
          x,
          y: histogram.holdout,
          marker: { color: look.real.fill },
          hovertemplate: t("fig.syn.hoverHeldOut"),
        } as Data,
        {
          type: "bar",
          name: look.synthetic.name,
          x,
          y: histogram.synthetic,
          marker: { color: look.synthetic.fill, line: { color: look.synthetic.color, width: 1.5 } },
          hovertemplate: t("fig.syn.hoverDcrSynthetic"),
        } as Data,
      ],
      layout: {
        ...legend,
        barmode: "overlay",
        bargap: 0,
        xaxis: axisTitle(t("fig.syn.distance")),
        yaxis: { ...percentAxis, ...axisTitle(t("fig.shareOfRows")) },
      },
    };
  };
}

/** One point per numeric column: real on x, synthetic on y, log scales, the diagonal. */
export function meanSpreadFigure(points: SyntheticReport["overview"]["mean_spread"], which: "mean" | "std"): Figure {
  return (tokens) => {
    const look = looks(tokens);
    const real = points.map((p) => Math.abs(which === "mean" ? p.real_mean : p.real_std));
    const synthetic = points.map((p) => Math.abs(which === "mean" ? p.synthetic_mean : p.synthetic_std));
    const positive = [...real, ...synthetic].filter((v) => v > 0);
    const low = positive.length ? Math.min(...positive) : 0.1;
    const high = positive.length ? Math.max(...positive) : 1;
    const far = points.map((_, i) => {
      const r = real[i] ?? 0;
      const s = synthetic[i] ?? 0;
      return r > 0 && s > 0 && Math.abs(Math.log10(s / r)) > 0.05;
    });
    return {
      data: [
        {
          type: "scatter",
          mode: "lines",
          x: [low, high],
          y: [low, high],
          line: { color: tokens.ink3, dash: "dot", width: 1 },
          hoverinfo: "skip",
          showlegend: false,
        } as Data,
        {
          type: "scatter",
          mode: "text+markers",
          x: real,
          y: synthetic,
          text: points.map((p, i) => (far[i] ? p.column : "")),
          textposition: "top center",
          textfont: { size: 10, color: tokens.ink2 },
          customdata: points.map((p) => p.column),
          marker: { color: look.synthetic.fill, size: 9, line: { color: look.synthetic.color, width: 1.5 } },
          hovertemplate: which === "mean" ? t("fig.syn.hoverMean") : t("fig.syn.hoverStd"),
          showlegend: false,
        } as Data,
      ],
      layout: {
        xaxis: { type: "log", ...axisTitle(`Real ${which === "mean" ? "|mean|" : "std. deviation"}`) },
        yaxis: { type: "log", ...axisTitle(`Synthetic ${which === "mean" ? "|mean|" : "std. deviation"}`) },
      },
    };
  };
}

/** Two components fitted on the real data; both sets projected in that plane. */
export function pcaFigure(pca: NonNullable<SyntheticReport["overview"]["pca"]>): Figure {
  return (tokens) => {
    const look = looks(tokens);
    const percent = (i: number) => `${Math.round((pca.explained[i] ?? 0) * 100)} %`;
    return {
      data: [
        {
          type: "scattergl",
          mode: "markers",
          name: look.real.name,
          x: pca.real.map((p) => p[0]),
          y: pca.real.map((p) => p[1]),
          marker: { color: look.real.color, size: 5, opacity: 0.6 },
          hoverinfo: "skip",
        } as Data,
        {
          type: "scattergl",
          mode: "markers",
          name: look.synthetic.name,
          x: pca.synthetic.map((p) => p[0]),
          y: pca.synthetic.map((p) => p[1]),
          marker: { color: "rgba(0,0,0,0)", size: 5, line: { color: look.synthetic.color, width: 1 } },
          hoverinfo: "skip",
        } as Data,
      ],
      layout: { ...legend, xaxis: axisTitle(t("fig.seg.component", { n: "1", share: percent(0) })), yaxis: axisTitle(t("fig.seg.component", { n: "2", share: percent(1) })) },
    };
  };
}

export const COMPARED = ["diagnostic", "quality", "shapes", "trends", "detection", "privacy", "utility"] as const;

/** Properties per run, grouped bars (one colour per run, in the fixed palette order). */
export function compareFigure(runs: SyntheticRunSummary[], labels: Record<string, string>): Figure {
  return (tokens) => ({
    data: runs.map(
      (run, i) =>
        ({
          type: "bar",
          name: `${run.method_label} · ${run.id.slice(0, 4)}`,
          x: COMPARED.map((key) => labels[key] ?? key),
          y: COMPARED.map((key) => run.scores[key] ?? null),
          marker: { color: tokens.palette[i % tokens.palette.length] },
          hovertemplate: "%{x}: %{y:.0%}<extra>%{fullData.name}</extra>",
        }) as Data,
    ),
    layout: { ...legend, barmode: "group", yaxis: { ...percentAxis, range: [0, 1] } },
  });
}

type SequenceReport = NonNullable<SyntheticReport["sequences"]>;

/** Mean autocorrelation by lag, real and synthetic. */
export function acfFigure(column: SequenceReport["columns"][number]): Figure {
  return (tokens) => {
    const look = looks(tokens);
    const lags = column.real.map((_, i) => i + 1);
    return {
      data: [
        { type: "scatter", mode: "lines+markers", name: look.real.name, x: lags, y: column.real, line: { color: look.real.color, width: 2 }, marker: { size: 5 } } as Data,
        { type: "scatter", mode: "lines+markers", name: look.synthetic.name, x: lags, y: column.synthetic, line: { color: look.synthetic.color, width: 2, dash: "dash" }, marker: { size: 6, symbol: "circle-open" } } as Data,
      ],
      layout: {
        ...legend,
        xaxis: { ...axisTitle(t("fig.syn.lagSteps")), dtick: 1 },
        yaxis: { range: [-1, 1], zeroline: true, zerolinecolor: tokens.ink3, ...axisTitle(t("fig.syn.autocorrelation")) },
      },
    };
  };
}

/** A few real and synthetic histories of one column, step by step. */
export function samplesFigure(sample: SequenceReport["samples"][number]): Figure {
  return (tokens) => {
    const look = looks(tokens);
    const lines = (series: number[][], side: "real" | "synthetic"): Data[] =>
      series.map(
        (values, i) =>
          ({
            type: "scatter",
            mode: "lines",
            name: look[side].name,
            legendgroup: side,
            showlegend: i === 0,
            x: values.map((_, k) => k + 1),
            y: values,
            line: { color: side === "real" ? look.real.color : look.synthetic.color, width: 1.5, dash: side === "real" ? "solid" : "dash" },
            opacity: side === "real" ? 0.7 : 0.9,
          }) as Data,
      );
    return {
      data: [...lines(sample.real, "real"), ...lines(sample.synthetic, "synthetic")],
      layout: { ...legend, xaxis: axisTitle(t("fig.syn.step")), yaxis: axisTitle(sample.column) },
    };
  };
}
