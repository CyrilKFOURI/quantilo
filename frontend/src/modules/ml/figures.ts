import type { Data } from "plotly.js-dist-min";

import type { MlAle, MlExplanation, MlRun } from "../../api/client";
import type { Figure } from "../../charts/Plot";
import { withAlpha } from "../../charts/figures";

type Regression = NonNullable<MlRun["regression"]>;
type Classification = NonNullable<MlRun["classification"]>;
type Coefficient = MlRun["coefficients"][number];

const title = (text: string) => ({ title: { text, font: { size: 11 } } });

/** Predicted against actual, with the line of perfect predictions. */
export function predictedFigure(result: Regression, target: string): Figure {
  return (tokens) => {
    const values = [...result.actual, ...result.predicted];
    const low = Math.min(...values);
    const high = Math.max(...values);
    return {
      data: [
        {
          type: "scattergl",
          mode: "markers",
          x: result.actual,
          y: result.predicted,
          marker: { color: tokens.palette[0] ?? tokens.numeric, size: 5, opacity: 0.55 },
          hovertemplate: "actual %{x:,.4g}<br>predicted %{y:,.4g}<extra></extra>",
        } as Data,
        {
          type: "scatter",
          mode: "lines",
          x: [low, high],
          y: [low, high],
          line: { color: tokens.ink3, dash: "dash", width: 1 },
          hoverinfo: "skip",
        } as Data,
      ],
      layout: { showlegend: false, xaxis: title(`Actual ${target}`), yaxis: title(`Predicted ${target}`) },
    };
  };
}

/** Residuals against predicted values: a funnel shape means the error grows with the value. */
export function residualsFigure(result: Regression): Figure {
  return (tokens) => ({
    data: [
      {
        type: "scattergl",
        mode: "markers",
        x: result.predicted,
        y: result.actual.map((a, i) => a - (result.predicted[i] ?? 0)),
        marker: { color: tokens.palette[0] ?? tokens.numeric, size: 5, opacity: 0.55 },
        hovertemplate: "predicted %{x:,.4g}<br>residual %{y:,.4g}<extra></extra>",
      } as Data,
    ],
    layout: {
      showlegend: false,
      xaxis: title("Predicted"),
      yaxis: { ...title("Actual − predicted"), zeroline: true, zerolinecolor: tokens.ink3 },
    },
  });
}

export function residualHistogram(result: Regression): Figure {
  return (tokens) => ({
    data: [
      {
        type: "bar",
        x: result.residual_bins.map((b) => ((b.start ?? 0) + (b.end ?? 0)) / 2),
        y: result.residual_bins.map((b) => b.count),
        width: result.residual_bins.map((b) => (b.end ?? 0) - (b.start ?? 0)),
        marker: { color: withAlpha(tokens.ink3, 0.5), line: { color: tokens.surface, width: 1 } },
        hovertemplate: "%{x:,.4g}: %{y} rows<extra></extra>",
      } as Data,
    ],
    layout: { showlegend: false, bargap: 0, xaxis: title("Residual"), yaxis: title("Rows") },
  });
}

/** Confusion matrix as a heatmap: counts or % of each actual class (row). */
export function confusionFigure(labels: string[], matrix: number[][], percent: boolean): Figure {
  return (tokens) => {
    const values = matrix.map((row) => {
      const total = row.reduce((a, b) => a + b, 0) || 1;
      return row.map((v) => (percent ? (v / total) * 100 : v));
    });
    return {
      data: [
        {
          type: "heatmap",
          x: labels,
          y: labels,
          z: values,
          text: values.map((row) => row.map((v) => (percent ? `${v.toFixed(0)}%` : String(v)))),
          texttemplate: "%{text}",
          colorscale: [
            [0, tokens.surface],
            [1, tokens.palette[0] ?? tokens.numeric],
          ],
          showscale: false,
          hovertemplate: "actual %{y}<br>predicted %{x}<br>%{text}<extra></extra>",
        } as Data,
      ],
      layout: {
        xaxis: { ...title("Predicted"), type: "category" },
        yaxis: { ...title("Actual"), type: "category", autorange: "reversed" },
      },
    };
  };
}

/** Several curves on one chart, one colour per class in the fixed palette order, with a legend
 *  (multiclass ROC, precision-recall and calibration: each class against the others). */
export function classCurvesFigure(
  curves: { name: string; x: number[]; y: number[] }[],
  xTitle: string,
  yTitle: string,
  diagonal: boolean,
): Figure {
  return (tokens) => {
    const data: Data[] = curves.map(
      (curve, i) =>
        ({
          type: "scatter",
          mode: "lines",
          name: curve.name,
          x: curve.x,
          y: curve.y,
          line: { color: tokens.palette[i % tokens.palette.length] ?? tokens.numeric, width: 2 },
          hovertemplate: `${curve.name}<br>${xTitle} %{x:.3f}<br>${yTitle} %{y:.3f}<extra></extra>`,
        }) as Data,
    );
    if (diagonal) {
      data.push({
        type: "scatter",
        mode: "lines",
        x: [0, 1],
        y: [0, 1],
        line: { color: tokens.ink3, dash: "dash", width: 1 },
        hoverinfo: "skip",
        showlegend: false,
      } as Data);
    }
    return {
      data,
      layout: {
        showlegend: true,
        legend: { orientation: "h", x: 0, y: -0.28, font: { size: 11, color: tokens.ink2 } },
        margin: { l: 8, r: 8, t: 4, b: 8 },
        xaxis: { ...title(xTitle), range: [0, 1] },
        yaxis: { ...title(yTitle), range: [0, 1.02] },
      },
    };
  };
}

export function curveFigure(
  x: number[],
  y: number[],
  xTitle: string,
  yTitle: string,
  reference: "diagonal" | "flat" | null,
  flat = 0,
): Figure {
  return (tokens) => {
    const data: Data[] = [
      {
        type: "scatter",
        mode: "lines+markers",
        x,
        y,
        line: { color: tokens.palette[0] ?? tokens.numeric, width: 2 },
        marker: { size: 3 },
        hovertemplate: `${xTitle} %{x:.3f}<br>${yTitle} %{y:.3f}<extra></extra>`,
      } as Data,
    ];
    if (reference) {
      data.push({
        type: "scatter",
        mode: "lines",
        x: [0, 1],
        y: reference === "diagonal" ? [0, 1] : [flat, flat],
        line: { color: tokens.ink3, dash: "dash", width: 1 },
        hoverinfo: "skip",
      } as Data);
    }
    return {
      data,
      layout: {
        showlegend: false,
        xaxis: { ...title(xTitle), range: [0, 1] },
        yaxis: { ...title(yTitle), range: [0, 1.02] },
      },
    };
  };
}

/** The largest coefficients, with their 95 % interval when there is one. */
export function coefficientsFigure(coefficients: Coefficient[]): Figure {
  return (tokens) => {
    const top = coefficients.slice(0, 15).reverse();
    return {
      data: [
        {
          type: "bar",
          orientation: "h",
          y: top.map((c) => c.feature),
          x: top.map((c) => c.estimate),
          error_x:
            top.every((c) => c.low !== null && c.high !== null)
              ? {
                  type: "data",
                  symmetric: false,
                  array: top.map((c) => (c.high ?? c.estimate) - c.estimate),
                  arrayminus: top.map((c) => c.estimate - (c.low ?? c.estimate)),
                  color: tokens.ink3,
                }
              : undefined,
          marker: { color: top.map((c) => (c.estimate >= 0 ? (tokens.palette[0] ?? tokens.numeric) : (tokens.palette[1] ?? tokens.ink2))) },
          hovertemplate: "%{y}: %{x:,.4g}<extra></extra>",
        } as Data,
      ],
      layout: { showlegend: false, margin: { l: 160 }, xaxis: { ...title("Coefficient"), zeroline: true, zerolinecolor: tokens.ink3 } },
    };
  };
}

export type { Classification };

export function importancesFigure(importances: { feature: string; importance: number }[]): Figure {
  return (tokens) => {
    const top = importances.slice(0, 15).reverse();
    return {
      data: [
        {
          type: "bar",
          orientation: "h",
          y: top.map((i) => i.feature),
          x: top.map((i) => i.importance),
          marker: { color: tokens.palette[0] ?? tokens.numeric },
          hovertemplate: "%{y}: %{x:.1%}<extra></extra>",
        } as Data,
      ],
      layout: { showlegend: false, margin: { l: 160 }, xaxis: { ...title("Share of importance"), tickformat: ".0%" } },
    };
  };
}

type Explanation = NonNullable<MlExplanation>;
type Shap = NonNullable<Explanation["shap"]>;
type Dependence = Explanation["dependence"][number];
type PermutationItem = NonNullable<Explanation["permutation"]>["items"][number];

/** How much the score drops when each column is shuffled (mean ± sd over 5 repeats). */
export function permutationFigure(items: PermutationItem[], metric: string): Figure {
  return (tokens) => {
    const top = items.slice(0, 15).reverse();
    return {
      data: [
        {
          type: "bar",
          orientation: "h",
          y: top.map((i) => i.column),
          x: top.map((i) => i.mean),
          error_x: { type: "data", array: top.map((i) => i.std), color: tokens.ink3 },
          marker: { color: top.map((i) => (i.mean > 0 ? (tokens.palette[0] ?? tokens.numeric) : tokens.ink3)) },
          hovertemplate: "%{y}: %{x:.4g}<extra></extra>",
        } as Data,
      ],
      layout: {
        showlegend: false,
        margin: { l: 150 },
        xaxis: { ...title(`Drop in ${metric} when shuffled`), zeroline: true, zerolinecolor: tokens.ink3 },
      },
    };
  };
}

/** One line per column, one point per row: where its SHAP value falls, colored by its value. */
export function beeswarmFigure(shap: Shap): Figure {
  return (tokens) => {
    const features = [...shap.features].reverse();
    const low = tokens.palette[0] ?? tokens.numeric;
    const high = tokens.palette[1] ?? tokens.categorical;
    const data: Data[] = features.map((feature, index) => {
      // a fixed jitter, so the swarm does not move between renders
      const jitter = feature.values.map((_, i) => ((((i * 2654435761) >>> 0) % 1000) / 1000 - 0.5) * 0.6);
      const numeric = feature.colors.some((c) => c !== null);
      return {
        type: "scattergl",
        mode: "markers",
        name: feature.column,
        x: feature.values,
        y: jitter.map((j) => index + j),
        marker: numeric
          ? {
              size: 4,
              opacity: 0.75,
              color: feature.colors.map((c) => (c === null ? 0.5 : c)),
              colorscale: [
                [0, low],
                [0.5, tokens.ink3],
                [1, high],
              ],
              cmin: 0,
              cmax: 1,
              showscale: index === features.length - 1,
              colorbar: {
                title: { text: "value", side: "right", font: { size: 10 } },
                tickvals: [0, 1],
                ticktext: ["low", "high"],
                thickness: 8,
                len: 0.5,
              },
            }
          : { size: 4, opacity: 0.6, color: tokens.ink3 },
        hovertemplate: `${feature.column}: %{x:.4g}<extra></extra>`,
      } as Data;
    });
    return {
      data,
      layout: {
        showlegend: false,
        margin: { l: 160 },
        xaxis: { ...title(`SHAP value (${shap.scale})`), zeroline: true, zerolinecolor: tokens.ink3 },
        yaxis: { tickvals: features.map((_, i) => i), ticktext: features.map((f) => f.column), range: [-0.6, features.length - 0.4] },
      },
    };
  };
}

/** The average prediction as one column varies, with how many rows sit at each value below. */
export function dependenceFigure(found: Dependence): Figure {
  return (tokens) => {
    const number = found.kind === "number";
    const curves = Object.entries(found.curves);
    const data: Data[] = curves.map(
      ([name, values], index) =>
        ({
          type: number ? "scatter" : "bar",
          mode: number ? "lines+markers" : undefined,
          name,
          x: found.grid,
          y: values,
          marker: { color: tokens.palette[index % 8], size: 4 },
          line: { color: tokens.palette[index % 8], width: 2 },
          hovertemplate: `${name}: %{y:.4g}<extra>%{x}</extra>`,
        }) as Data,
    );
    data.push({
      type: "bar",
      x: found.grid,
      y: found.counts,
      yaxis: "y2",
      marker: { color: tokens.line },
      hovertemplate: "%{y} rows<extra>%{x}</extra>",
      showlegend: false,
    } as Data);
    return {
      data,
      layout: {
        showlegend: curves.length > 1,
        barmode: "group",
        xaxis: { ...title(found.column), type: number ? "linear" : "category" },
        yaxis: { ...title(found.output), domain: [0.3, 1] },
        yaxis2: { domain: [0, 0.2], showticklabels: false, title: { text: "rows", font: { size: 10 } } },
      },
    };
  };
}

/** Accumulated local effects (125): the centred effect at each edge of an interval (a number)
 * or each category in order, the rows of each interval or category below. */
export function aleFigure(found: MlAle): Figure {
  return (tokens) => {
    const number = found.kind === "number";
    const curves = Object.entries(found.curves);
    const edges = found.grid.map(Number);
    const data: Data[] = curves.map(
      ([name, values], index) =>
        ({
          type: "scatter",
          mode: number ? "lines+markers" : "lines+markers",
          line: { color: tokens.palette[index % 8], width: 2, shape: number ? "linear" : "hv" },
          marker: { color: tokens.palette[index % 8], size: 4 },
          name,
          x: found.grid,
          y: values,
          hovertemplate: `${name}: %{y:+.4g}<extra>%{x}</extra>`,
        }) as Data,
    );
    data.push({
      type: "bar",
      // a number: one bar per interval, at its middle; categories: one per category
      x: number ? edges.slice(1).map((high, i) => ((edges[i] ?? high) + high) / 2) : found.grid,
      width: number ? edges.slice(1).map((high, i) => high - (edges[i] ?? high)) : undefined,
      y: found.counts,
      yaxis: "y2",
      marker: { color: tokens.line },
      hovertemplate: "%{y} rows<extra></extra>",
      showlegend: false,
    } as Data);
    return {
      data,
      layout: {
        showlegend: curves.length > 1,
        xaxis: { ...title(found.column), type: number ? "linear" : "category" },
        yaxis: { ...title(`Effect on ${found.output.toLowerCase()}`), domain: [0.3, 1], zeroline: true, zerolinecolor: tokens.ink3 },
        yaxis2: { domain: [0, 0.2], showticklabels: false, title: { text: "rows", font: { size: 10 } } },
      },
    };
  };
}

/** From the average prediction to this row's, one contribution at a time. */
export function waterfallFigure(base: number, steps: { label: string; value: number }[], scale: string): Figure {
  return (tokens) => ({
    data: [
      {
        type: "waterfall",
        orientation: "h",
        measure: ["absolute", ...steps.map(() => "relative"), "total"],
        y: ["average", ...steps.map((s) => s.label), "this row"],
        x: [base, ...steps.map((s) => s.value), 0],
        increasing: { marker: { color: tokens.palette[1] ?? tokens.categorical } },
        decreasing: { marker: { color: tokens.palette[0] ?? tokens.numeric } },
        totals: { marker: { color: tokens.ink3 } },
        connector: { line: { color: tokens.line } },
        hovertemplate: "%{y}: %{x:+.4g}<extra></extra>",
      } as Data,
    ],
    layout: {
      showlegend: false,
      margin: { l: 220 },
      yaxis: { autorange: "reversed" },
      xaxis: title(scale === "prediction" ? "Prediction" : `Prediction (${scale})`),
    },
  });
}

type TrialPoint = { number: number; value: number | null; state: string };

/** Score of every trial and the best so far. */
export function trialsFigure(trials: TrialPoint[], higher: boolean, metric: string): Figure {
  return (tokens) => {
    const done = trials.filter((t) => t.value !== null);
    let best: number | null = null;
    const running = done.map((t) => {
      const v = t.value as number;
      best = best === null ? v : higher ? Math.max(best, v) : Math.min(best, v);
      return best;
    });
    return {
      data: [
        {
          type: "scatter",
          mode: "markers",
          name: "trial",
          x: done.map((t) => t.number),
          y: done.map((t) => t.value),
          marker: { color: done.map((t) => (t.state === "complete" ? (tokens.palette[0] ?? tokens.numeric) : tokens.ink3)), size: 6 },
          hovertemplate: "trial %{x}: %{y:.4g}<extra></extra>",
        } as Data,
        {
          type: "scatter",
          mode: "lines",
          name: "best so far",
          x: done.map((t) => t.number),
          y: running,
          line: { color: tokens.palette[1] ?? tokens.categorical, width: 2, shape: "hv" },
          hoverinfo: "skip",
        } as Data,
      ],
      layout: { showlegend: true, xaxis: title("Trial"), yaxis: title(metric) },
    };
  };
}

/** How much each setting mattered in the search (fANOVA). */
export function importanceBars(importances: Record<string, number>, labels: Record<string, string>): Figure {
  return (tokens) => {
    const entries = Object.entries(importances).sort((a, b) => a[1] - b[1]);
    return {
      data: [
        {
          type: "bar",
          orientation: "h",
          y: entries.map(([k]) => labels[k] ?? k),
          x: entries.map(([, v]) => v),
          marker: { color: tokens.palette[0] ?? tokens.numeric },
          hovertemplate: "%{y}: %{x:.0%}<extra></extra>",
        } as Data,
      ],
      layout: { showlegend: false, margin: { l: 160 }, xaxis: { ...title("Share of the score's variation"), tickformat: ".0%" } },
    };
  };
}

/** Predicted against actual for several runs, one color each. */
export function overlayPredictedFigure(runs: { name: string; run: MlRun }[]): Figure {
  return (tokens) => {
    const all = runs.flatMap(({ run }) => [...(run.regression?.actual ?? []), ...(run.regression?.predicted ?? [])]);
    const low = Math.min(...all);
    const high = Math.max(...all);
    return {
      data: [
        ...runs.map(
          ({ name, run }, i) =>
            ({
              type: "scattergl",
              mode: "markers",
              name,
              x: run.regression?.actual ?? [],
              y: run.regression?.predicted ?? [],
              marker: { color: tokens.palette[i % 8], size: 4, opacity: 0.5 },
            }) as Data,
        ),
        { type: "scatter", mode: "lines", x: [low, high], y: [low, high], line: { color: tokens.ink3, dash: "dash", width: 1 }, hoverinfo: "skip", showlegend: false } as Data,
      ],
      layout: { showlegend: true, xaxis: title("Actual"), yaxis: title("Predicted") },
    };
  };
}

/** ROC curves of several binary runs. */
export function overlayRocFigure(runs: { name: string; run: MlRun }[]): Figure {
  return (tokens) => ({
    data: [
      ...runs.map(
        ({ name, run }, i) =>
          ({
            type: "scatter",
            mode: "lines",
            name: `${name} (AUC ${run.classification?.roc_auc?.toFixed(3) ?? "—"})`,
            x: run.classification?.roc?.x ?? [],
            y: run.classification?.roc?.y ?? [],
            line: { color: tokens.palette[i % 8], width: 2 },
          }) as Data,
      ),
      { type: "scatter", mode: "lines", x: [0, 1], y: [0, 1], line: { color: tokens.ink3, dash: "dash", width: 1 }, hoverinfo: "skip", showlegend: false } as Data,
    ],
    layout: { showlegend: true, xaxis: title("False positive rate"), yaxis: title("True positive rate") },
  });
}

/** Training and held-out scores iteration after iteration; the iteration kept is marked. */
export function learningCurveFigure(curve: NonNullable<MlRun["learning_curve"]>): Figure {
  return (tokens) => {
    const data: Data[] = [];
    if (curve.train)
      data.push({ type: "scatter", mode: "lines", name: "training", x: curve.iterations, y: curve.train, line: { color: tokens.palette[0] ?? tokens.numeric, width: 2 } } as Data);
    if (curve.validation)
      data.push({ type: "scatter", mode: "lines", name: "held out", x: curve.iterations, y: curve.validation, line: { color: tokens.palette[1] ?? tokens.categorical, width: 2 } } as Data);
    const shapes =
      curve.best !== null
        ? [{ type: "line" as const, x0: curve.best, x1: curve.best, yref: "paper" as const, y0: 0, y1: 1, line: { color: tokens.ink3, dash: "dot" as const, width: 1 } }]
        : [];
    return { data, layout: { showlegend: data.length > 1, shapes, xaxis: title("Iteration"), yaxis: title(curve.label) } };
  };
}

// --- Panel data (035) -------------------------------------------------------------------------

type Panel = NonNullable<MlRun["panel"]>;

/** A period on the good side of the threshold (always, when there is none). */
export function goodPeriod(summary: Panel["summary"], value: number | null): boolean {
  if (value === null || summary.threshold === null) return true;
  return summary.lower_is_better ? value < summary.threshold : value > summary.threshold;
}

/** The score of every test period; periods on the wrong side of the threshold are muted. */
export function periodScoresFigure(panel: Pick<Panel, "periods" | "summary" | "period_column">): Figure {
  return (tokens) => {
    const threshold = panel.summary.threshold;
    const shapes =
      threshold !== null
        ? [{ type: "line" as const, xref: "paper" as const, x0: 0, x1: 1, y0: threshold, y1: threshold, line: { color: tokens.ink3, dash: "dash" as const, width: 1 } }]
        : [];
    return {
      data: [
        {
          type: "bar",
          x: panel.periods.map((p) => p.period),
          y: panel.periods.map((p) => p.score),
          customdata: panel.periods.map((p) => p.fold),
          marker: { color: panel.periods.map((p) => (goodPeriod(panel.summary, p.score) ? (tokens.palette[0] ?? tokens.numeric) : withAlpha(tokens.ink3, 0.55))) },
          hovertemplate: "%{x}: %{y:,.4g} (fold %{customdata})<extra></extra>",
        } as Data,
      ],
      layout: { showlegend: false, bargap: 0.15, shapes, xaxis: { ...title(panel.period_column), type: "category" }, yaxis: title(panel.summary.metric) },
    };
  };
}

/** The scores of the periods added up in time: a steady climb is a regular model. */
export function cumulativeFigure(panel: Pick<Panel, "periods" | "summary" | "period_column">): Figure {
  return (tokens) => {
    let total = 0;
    const running = panel.periods.map((p) => (total += p.score ?? 0));
    return {
      data: [
        {
          type: "scatter",
          mode: "lines",
          x: panel.periods.map((p) => p.period),
          y: running,
          line: { color: tokens.palette[0] ?? tokens.numeric, width: 2 },
          hovertemplate: "%{x}: %{y:,.4g}<extra></extra>",
        } as Data,
      ],
      layout: { showlegend: false, xaxis: { ...title(panel.period_column), type: "category" }, yaxis: { ...title(`Cumulative ${panel.summary.metric}`), zeroline: true } },
    };
  };
}

/** Each fold on the time axis: the periods it trained on, the embargo, the periods tested. */
export function foldsFigure(panel: Panel): Figure {
  return (tokens) => {
    const folds = panel.folds;
    const names = folds.map((f) => `Fold ${f.fold}`);
    const plural = (n: number) => `${n} period${n === 1 ? "" : "s"}`;
    const bar = (name: string, color: string, base: number[], width: number[], text: string[]) =>
      ({
        type: "bar",
        orientation: "h",
        name,
        y: names,
        x: width,
        base,
        marker: { color, line: { color: tokens.surface, width: 2 } },
        text,
        textposition: "none",
        hovertemplate: `%{y} · ${name}: %{text}<extra></extra>`,
      }) as Data;
    const ticks = new Map<number, string>(panel.periods.map((p) => [p.rank, p.period]));
    for (const f of folds) if (!ticks.has(f.train_start)) ticks.set(f.train_start, f.train_from);
    const ranks = [...ticks.keys()].sort((a, b) => a - b);
    const chosen = ranks.filter((_, i) => i % Math.max(1, Math.floor(ranks.length / 6)) === 0);
    return {
      data: [
        bar(
          "trained on",
          tokens.palette[0] ?? tokens.numeric,
          folds.map((f) => f.train_start),
          folds.map((f) => f.train_end - f.train_start + 1),
          folds.map((f) => `${f.train_from} to ${f.train_to} · ${plural(f.train_periods)}, ${f.train_rows.toLocaleString("en-US")} rows`),
        ),
        bar(
          "embargo",
          withAlpha(tokens.ink3, 0.35),
          folds.map((f) => f.train_end + 1),
          folds.map((f) => f.test_start - f.train_end - 1),
          folds.map((f) => `${plural(f.test_start - f.train_end - 1)} left out`),
        ),
        bar(
          "tested on",
          tokens.palette[1] ?? tokens.categorical,
          folds.map((f) => f.test_start),
          folds.map((f) => f.test_end - f.test_start + 1),
          folds.map((f) => `${f.test_from} to ${f.test_to} · ${plural(f.test_periods)}, ${f.test_rows.toLocaleString("en-US")} rows`),
        ),
      ],
      layout: {
        barmode: "overlay",
        showlegend: true,
        legend: { orientation: "h", x: 0, y: 1.02, yanchor: "bottom", entrywidth: 96, font: { size: 11, color: tokens.ink2 } },
        yaxis: { autorange: "reversed" },
        xaxis: { ...title(panel.period_column), tickvals: chosen, ticktext: chosen.map((r) => ticks.get(r) ?? String(r)) },
      },
    };
  };
}
