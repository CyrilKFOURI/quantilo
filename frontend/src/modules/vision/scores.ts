import type { Prediction, VisionModel } from "./modelApi";

/** 113: what the Results tab computes in the page from a run's predictions (R-113-08, R-113-09):
 *  ROC and PR curves, the counts at a threshold the user moves, histograms and galleries. Pure
 *  functions, mirrored on `quantilo.vision.metrics` (111) so a number read here matches the run's. */

export type Curve = { x: number[]; y: number[]; area: number };

/** Sort by score, highest first, and walk the distinct thresholds (ties move together). */
function walk(truth: readonly boolean[], scores: readonly number[]) {
  const order = scores
    .map((score, i) => ({ score, positive: truth[i] === true }))
    .filter((point) => Number.isFinite(point.score))
    .sort((a, b) => b.score - a.score);
  const positives = order.filter((point) => point.positive).length;
  const negatives = order.length - positives;
  const steps: { tp: number; fp: number; threshold: number }[] = [];
  let tp = 0;
  let fp = 0;
  for (let i = 0; i < order.length; i += 1) {
    const point = order[i];
    if (!point) continue;
    if (point.positive) tp += 1;
    else fp += 1;
    const next = order[i + 1];
    if (!next || next.score !== point.score) steps.push({ tp, fp, threshold: point.score });
  }
  return { steps, positives, negatives };
}

/** ROC curve (false alarm rate → recall) and its area by trapezoids, as `metrics.roc_auc`. */
export function rocCurve(truth: readonly boolean[], scores: readonly number[]): Curve | null {
  const { steps, positives, negatives } = walk(truth, scores);
  if (positives === 0 || negatives === 0) return null;
  const x = [0, ...steps.map((s) => s.fp / negatives)];
  const y = [0, ...steps.map((s) => s.tp / positives)];
  let area = 0;
  for (let i = 1; i < x.length; i += 1) area += ((x[i] ?? 0) - (x[i - 1] ?? 0)) * (((y[i] ?? 0) + (y[i - 1] ?? 0)) / 2);
  return { x, y, area };
}

/** Precision–recall curve and the average precision (step sum, as `metrics.average_precision`). */
export function prCurve(truth: readonly boolean[], scores: readonly number[]): Curve | null {
  const { steps, positives } = walk(truth, scores);
  if (positives === 0 || steps.length === 0) return null;
  const x = [0];
  const y = [1];
  let area = 0;
  let previous = 0;
  for (const step of steps) {
    const recall = step.tp / positives;
    const precision = step.tp / (step.tp + step.fp);
    area += (recall - previous) * precision;
    previous = recall;
    x.push(recall);
    y.push(precision);
  }
  return { x, y, area };
}

export type Counts = {
  threshold: number;
  tp: number;
  fp: number;
  fn: number;
  tn: number;
  precision: number;
  recall: number;
  f1: number;
  falseAlarm: number;
};

const ratio = (num: number, den: number) => (den > 0 ? num / den : 0);

/** Counts at a threshold (an image is flagged when its score is at or above it), as
 *  `metrics.at_threshold`. */
export function countsAt(isDefect: readonly boolean[], scores: readonly number[], threshold: number): Counts {
  let tp = 0;
  let fp = 0;
  let fn = 0;
  let tn = 0;
  scores.forEach((score, i) => {
    const flagged = score >= threshold;
    const defect = isDefect[i] === true;
    if (flagged && defect) tp += 1;
    else if (flagged) fp += 1;
    else if (defect) fn += 1;
    else tn += 1;
  });
  const precision = ratio(tp, tp + fp);
  const recall = ratio(tp, tp + fn);
  return { threshold, tp, fp, fn, tn, precision, recall, f1: ratio(2 * precision * recall, precision + recall), falseAlarm: ratio(fp, fp + tn) };
}

export type Histogram = { edges: number[]; normal: number[]; defect: number[] };

/** Normal and defect scores on shared equal-width bins (as `metrics.score_histogram`). */
export function histogram(isDefect: readonly boolean[], scores: readonly number[], bins = 30): Histogram {
  const finite = scores.filter(Number.isFinite);
  const low = Math.min(...finite);
  const high = Math.max(...finite);
  const span = high > low ? high - low : 1;
  const edges = Array.from({ length: bins + 1 }, (_, k) => low + (span * k) / bins);
  const normal = new Array<number>(bins).fill(0);
  const defect = new Array<number>(bins).fill(0);
  scores.forEach((score, i) => {
    if (!Number.isFinite(score)) return;
    const k = Math.min(bins - 1, Math.max(0, Math.floor(((score - low) / span) * bins)));
    if (isDefect[i]) defect[k] = (defect[k] ?? 0) + 1;
    else normal[k] = (normal[k] ?? 0) + 1;
  });
  return { edges, normal, defect };
}

/** The threshold that gives the best F1 on these scores (halfway to the next score), the fallback
 *  when a run kept none. */
export function bestF1Threshold(isDefect: readonly boolean[], scores: readonly number[]): number {
  const sorted = [...new Set(scores.filter(Number.isFinite))].sort((a, b) => b - a);
  let best = sorted[0] ?? 0;
  let bestF1 = -1;
  sorted.forEach((value, i) => {
    const f1 = countsAt(isDefect, scores, value).f1;
    if (f1 > bestF1) {
      bestF1 = f1;
      const next = sorted[i + 1];
      best = next === undefined ? value : (value + next) / 2;
    }
  });
  return best;
}

// ---------------------------------------------------------------------------------------------
// Reading the predictions

export const num = (value: unknown): number => (typeof value === "number" && Number.isFinite(value) ? value : Number.NaN);
export const text = (value: unknown): string => (value === null || value === undefined ? "" : String(value));

/** Anomalies: the rows of the test (held-out normals left out), in the order the run scored them,
 *  with the chosen model's score. */
export function anomalyRows(predictions: readonly Prediction[], key: string) {
  return predictions
    .filter((p) => p.holdout !== true)
    .map((p) => ({ row: p.row, truth: text(p.truth), defect: p.is_defect === true, score: num(p[`${key}_score`]) }));
}

/** Classification: each class's one-vs-rest curve from the kept probabilities (R-VCL-08). */
export function classCurves(predictions: readonly Prediction[], key: string, labels: readonly string[]) {
  return labels.map((label, k) => {
    const truth = predictions.map((p) => text(p.truth) === label);
    const scores = predictions.map((p) => {
      const probs = p[`${key}_probs`];
      return Array.isArray(probs) ? num(probs[k]) : Number.NaN;
    });
    return { label, roc: rocCurve(truth, scores), pr: prCurve(truth, scores) };
  });
}

/** Classification: the rows behind one cell of the confusion matrix (truth → prediction). */
export function cellRows(predictions: readonly Prediction[], key: string, truth: string, predicted: string): number[] {
  return predictions.filter((p) => text(p.truth) === truth && text(p[`${key}_pred`]) === predicted).map((p) => p.row);
}

/** The main number of a model, for ranking: balanced accuracy or accuracy (its report says which),
 *  image AUROC for anomalies. */
export function mainScore(model: VisionModel): { metric: string; value: number | null; low: number | null; high: number | null } {
  if (model.classification) {
    const report = model.classification;
    const metric = report.main.metric;
    const interval = report.intervals[metric];
    return { metric, value: report[metric], low: interval?.low ?? null, high: interval?.high ?? null };
  }
  if (model.anomaly) {
    const interval = model.anomaly.auroc_interval;
    return { metric: "auroc", value: model.anomaly.auroc, low: interval?.low ?? null, high: interval?.high ?? null };
  }
  if (model.regression) {
    const interval = model.regression.intervals.r2;
    return { metric: "r2", value: model.regression.r2, low: interval?.low ?? null, high: interval?.high ?? null };
  }
  if (model.multilabel) {
    const interval = model.multilabel.intervals.macro_f1;
    return { metric: "macro_f1", value: model.multilabel.macro_f1, low: interval?.low ?? null, high: interval?.high ?? null };
  }
  return { metric: "flagged", value: model.flagged ?? null, low: null, high: null };
}

/** The models of a run, the default first, then by their main number (best first). */
export function rankModels(models: readonly VisionModel[]): VisionModel[] {
  return [...models].sort((a, b) => {
    if (a.default !== b.default) return a.default ? -1 : 1;
    return (mainScore(b).value ?? -1) - (mainScore(a).value ?? -1);
  });
}

/** A metric as text: 3 decimals, "—" when the run could not compute it (R-113-07). */
export function metric(value: number | null | undefined, digits = 3): string {
  return value === null || value === undefined || !Number.isFinite(value) ? "—" : value.toFixed(digits);
}

export function percent(value: number | null | undefined, digits = 1): string {
  return value === null || value === undefined || !Number.isFinite(value) ? "—" : `${(value * 100).toFixed(digits)} %`;
}
