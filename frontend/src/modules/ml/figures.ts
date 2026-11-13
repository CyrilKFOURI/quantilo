import type { Data } from "plotly.js-dist-min";

import type { MlAle, MlAle2d, MlExplanation, MlRun } from "../../api/client";
import type { Figure } from "../../charts/Plot";
import { withAlpha } from "../../charts/figures";
import { t } from "../../lib/i18n";

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
          hovertemplate: t("fig.ml.hoverActualPredicted"),
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
      layout: { showlegend: false, xaxis: title(t("fig.ml.actualOf", { target })), yaxis: title(t("fig.ml.predictedOf", { target })) },
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
        hovertemplate: t("fig.ml.hoverResidual"),
      } as Data,
    ],
    layout: {
      showlegend: false,
      xaxis: title(t("fig.ml.predicted")),
      yaxis: { ...title(t("fig.ml.actualMinusPredicted")), zeroline: true, zerolinecolor: tokens.ink3 },
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
        hovertemplate: `%{x:,.4g}: %{y} ${t("fig.rows")}<extra></extra>`,
      } as Data,
    ],
    layout: { showlegend: false, bargap: 0, xaxis: title(t("fig.ml.residual")), yaxis: title(t("fig.Rows")) },
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
          hovertemplate: t("fig.ml.hoverConfusion"),
        } as Data,
      ],
      layout: {
        xaxis: { ...title(t("fig.ml.predicted")), type: "category" },
        yaxis: { ...title(t("fig.ml.actual")), type: "category", autorange: "reversed" },
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
      layout: { showlegend: false, margin: { l: 160 }, xaxis: { ...title(t("fig.ml.coefficient")), zeroline: true, zerolinecolor: tokens.ink3 } },
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
      layout: { showlegend: false, margin: { l: 160 }, xaxis: { ...title(t("fig.ml.importanceShare")), tickformat: ".0%" } },
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
        xaxis: { ...title(t("fig.ml.dropWhenShuffled", { metric })), zeroline: true, zerolinecolor: tokens.ink3 },
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
                title: { text: t("fig.ml.value"), side: "right", font: { size: 10 } },
                tickvals: [0, 1],
                ticktext: [t("fig.ml.low"), t("fig.ml.high")],
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
        xaxis: { ...title(t("fig.ml.shapValue", { scale: shap.scale })), zeroline: true, zerolinecolor: tokens.ink3 },
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
      hovertemplate: `%{y} ${t("fig.rows")}<extra>%{x}</extra>`,
      showlegend: false,
    } as Data);
    return {
      data,
      layout: {
        showlegend: curves.length > 1,
        barmode: "group",
        xaxis: { ...title(found.column), type: number ? "linear" : "category" },
        yaxis: { ...title(found.output), domain: [0.3, 1] },
        yaxis2: { domain: [0, 0.2], showticklabels: false, title: { text: t("fig.rows"), font: { size: 10 } } },
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
      hovertemplate: `%{y} ${t("fig.rows")}<extra></extra>`,
      showlegend: false,
    } as Data);
    return {
      data,
      layout: {
        showlegend: curves.length > 1,
        xaxis: { ...title(found.column), type: number ? "linear" : "category" },
        yaxis: { ...title(t("fig.ml.effectOn", { output: found.output.toLowerCase() })), domain: [0.3, 1], zeroline: true, zerolinecolor: tokens.ink3 },
        yaxis2: { domain: [0, 0.2], showticklabels: false, title: { text: t("fig.rows"), font: { size: 10 } } },
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
        y: [t("fig.ml.average"), ...steps.map((s) => s.label), t("fig.ml.thisRow")],
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
      xaxis: title(scale === "prediction" ? t("fig.ml.prediction") : t("fig.ml.predictionScale", { scale })),
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
          name: t("fig.ml.trial"),
          x: done.map((t) => t.number),
          y: done.map((t) => t.value),
          marker: { color: done.map((t) => (t.state === "complete" ? (tokens.palette[0] ?? tokens.numeric) : tokens.ink3)), size: 6 },
          hovertemplate: t("fig.ml.hoverTrial"),
        } as Data,
        {
          type: "scatter",
          mode: "lines",
          name: t("fig.ml.bestSoFar"),
          x: done.map((t) => t.number),
          y: running,
          line: { color: tokens.palette[1] ?? tokens.categorical, width: 2, shape: "hv" },
          hoverinfo: "skip",
        } as Data,
      ],
      layout: { showlegend: true, xaxis: title(t("fig.ml.trialAxis")), yaxis: title(metric) },
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
      layout: { showlegend: false, margin: { l: 160 }, xaxis: { ...title(t("fig.ml.scoreVariation")), tickformat: ".0%" } },
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
      layout: { showlegend: true, xaxis: title(t("fig.ml.actual")), yaxis: title(t("fig.ml.predicted")) },
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
    layout: { showlegend: true, xaxis: title(t("fig.ml.fpr")), yaxis: title(t("fig.ml.tpr")) },
  });
}

/** Training and held-out scores iteration after iteration; the iteration kept is marked. */
export function learningCurveFigure(curve: NonNullable<MlRun["learning_curve"]>): Figure {
  return (tokens) => {
    const data: Data[] = [];
    if (curve.train)
      data.push({ type: "scatter", mode: "lines", name: t("fig.ml.training"), x: curve.iterations, y: curve.train, line: { color: tokens.palette[0] ?? tokens.numeric, width: 2 } } as Data);
    if (curve.validation)
      data.push({ type: "scatter", mode: "lines", name: t("fig.ml.heldOut"), x: curve.iterations, y: curve.validation, line: { color: tokens.palette[1] ?? tokens.categorical, width: 2 } } as Data);
    const shapes =
      curve.best !== null
        ? [{ type: "line" as const, x0: curve.best, x1: curve.best, yref: "paper" as const, y0: 0, y1: 1, line: { color: tokens.ink3, dash: "dot" as const, width: 1 } }]
        : [];
    return { data, layout: { showlegend: data.length > 1, shapes, xaxis: title(t("fig.ml.iteration")), yaxis: title(curve.label) } };
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
          hovertemplate: t("fig.ml.hoverFold"),
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
      layout: { showlegend: false, xaxis: { ...title(panel.period_column), type: "category" }, yaxis: { ...title(t("fig.ml.cumulative", { metric: panel.summary.metric })), zeroline: true } },
    };
  };
}

/** Each fold on the time axis: the periods it trained on, the embargo, the periods tested. */
export function foldsFigure(panel: Panel): Figure {
  return (tokens) => {
    const folds = panel.folds;
    const names = folds.map((f) => t("fig.ml.fold", { fold: String(f.fold) }));
    const plural = (n: number) => (n === 1 ? t("fig.ml.onePeriod") : t("fig.ml.periods", { count: String(n) }));
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
          t("fig.ml.trainedOn"),
          tokens.palette[0] ?? tokens.numeric,
          folds.map((f) => f.train_start),
          folds.map((f) => f.train_end - f.train_start + 1),
          folds.map((f) => t("fig.ml.foldSpan", { from: f.train_from, to: f.train_to, periods: plural(f.train_periods), rows: f.train_rows.toLocaleString("en-US") })),
        ),
        bar(
          t("fig.ml.embargo"),
          withAlpha(tokens.ink3, 0.35),
          folds.map((f) => f.train_end + 1),
          folds.map((f) => f.test_start - f.train_end - 1),
          folds.map((f) => t("fig.ml.leftOut", { periods: plural(f.test_start - f.train_end - 1) })),
        ),
        bar(
          t("fig.ml.testedOn"),
          tokens.palette[1] ?? tokens.categorical,
          folds.map((f) => f.test_start),
          folds.map((f) => f.test_end - f.test_start + 1),
          folds.map((f) => t("fig.ml.foldSpan", { from: f.test_from, to: f.test_to, periods: plural(f.test_periods), rows: f.test_rows.toLocaleString("en-US") })),
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

/** 125 bis (R-XAI-20): the two-column ALE as a heatmap, two hues around a neutral 0 (what the pair
 *  adds on top of each column's own effect), at the middle of each cell. */
export function ale2dFigure(found: MlAle2d): Figure {
  return (tokens) => {
    const middles = (edges: number[]) => edges.slice(1).map((high, i) => ((edges[i] ?? high) + high) / 2);
    const corners = found.values;
    // the value of a cell: the mean of its four corners
    const z = found.counts.map((row, i) =>
      row.map((_, j) => {
        const a = corners[i]?.[j] ?? 0;
        const b = corners[i + 1]?.[j] ?? 0;
        const c = corners[i]?.[j + 1] ?? 0;
        const d = corners[i + 1]?.[j + 1] ?? 0;
        return (a + b + c + d) / 4;
      }),
    );
    const extent = Math.max(1e-12, ...z.flat().map((v) => Math.abs(v)));
    const [blue = tokens.ink2, orange = tokens.ink3] = tokens.palette;
    return {
      data: [
        {
          type: "heatmap",
          x: middles(found.second_edges),
          y: middles(found.first_edges),
          z,
          customdata: found.counts as unknown as string[],
          colorscale: [
            [0, orange],
            [0.5, tokens.line],
            [1, blue],
          ],
          zmid: 0,
          zmin: -extent,
          zmax: extent,
          showscale: false,
          xgap: 1,
          ygap: 1,
          hovertemplate: `${found.second} %{x:.4g} · ${found.first} %{y:.4g}: %{z:+.3g} (%{customdata} ${t("fig.rows")})<extra></extra>`,
        } as Data,
      ],
      layout: {
        xaxis: { ...title(found.second), type: "linear" },
        yaxis: { ...title(found.first), type: "linear" },
      },
    };
  };
}
