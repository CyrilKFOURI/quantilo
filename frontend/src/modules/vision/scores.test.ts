import { describe, expect, it } from "vitest";

import { bestF1Threshold, cellRows, classCurves, countsAt, histogram, metric, prCurve, rankModels, rocCurve } from "./scores";
import type { Prediction, VisionModel } from "./modelApi";

describe("curves (R-113-09)", () => {
  it("gives the AUROC of sklearn on a small case, ties moving together", () => {
    // sklearn.metrics.roc_auc_score([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) = 0.75
    expect(rocCurve([false, false, true, true], [0.1, 0.4, 0.35, 0.8])?.area).toBeCloseTo(0.75, 6);
    // a tie between a positive and a negative counts half
    expect(rocCurve([false, true], [0.5, 0.5])?.area).toBeCloseTo(0.5, 6);
    expect(rocCurve([true, true], [0.2, 0.9])).toBeNull();
  });

  it("gives the average precision of sklearn", () => {
    // sklearn.metrics.average_precision_score([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) = 0.8333
    expect(prCurve([false, false, true, true], [0.1, 0.4, 0.35, 0.8])?.area).toBeCloseTo(0.8333, 4);
  });
});

describe("threshold (R-VTH-03)", () => {
  const defect = [false, false, false, true, true, true];
  const scores = [0.1, 0.2, 0.6, 0.5, 0.7, 0.9];

  it("counts the images at or above the threshold as flagged", () => {
    const at = countsAt(defect, scores, 0.5);
    expect(at).toMatchObject({ tp: 3, fp: 1, fn: 0, tn: 2 });
    expect(at.precision).toBeCloseTo(0.75);
    expect(at.recall).toBe(1);
    expect(at.falseAlarm).toBeCloseTo(1 / 3);
    expect(countsAt(defect, scores, 2)).toMatchObject({ tp: 0, f1: 0, precision: 0 });
  });

  it("finds the best F1 halfway to the next score", () => {
    expect(bestF1Threshold(defect, scores)).toBeCloseTo(0.35);
  });

  it("bins normals and defects on shared edges", () => {
    const h = histogram(defect, scores, 4);
    expect(h.edges).toHaveLength(5);
    expect(h.normal.reduce((a, b) => a + b)).toBe(3);
    expect(h.defect.reduce((a, b) => a + b)).toBe(3);
    expect(h.defect[3]).toBeGreaterThanOrEqual(1); // 0.9 in the last bin
    expect(h.normal[0]).toBe(2); // 0.1 and 0.2 in the first
  });
});

describe("predictions", () => {
  const predictions: Prediction[] = [
    { row: 0, image_id: "a", truth: "cat", logistic_pred: "cat", logistic_probs: [0.9, 0.1] },
    { row: 1, image_id: "b", truth: "dog", logistic_pred: "cat", logistic_probs: [0.6, 0.4] },
    { row: 2, image_id: "c", truth: "dog", logistic_pred: "dog", logistic_probs: [0.2, 0.8] },
  ];

  it("finds the images of a confusion cell", () => {
    expect(cellRows(predictions, "logistic", "dog", "cat")).toEqual([1]);
  });

  it("draws one curve per class from the kept probabilities", () => {
    const curves = classCurves(predictions, "logistic", ["cat", "dog"]);
    expect(curves.map((c) => c.label)).toEqual(["cat", "dog"]);
    expect(curves[1]?.roc?.area).toBeCloseTo(1);
  });

  it("puts the default model first, then the best", () => {
    const model = (key: string, value: number, isDefault = false) =>
      ({ key, default: isDefault, anomaly: { auroc: value, auroc_interval: null } }) as unknown as VisionModel;
    expect(rankModels([model("a", 0.7), model("b", 0.9), model("c", 0.8, true)]).map((m) => m.key)).toEqual(["c", "b", "a"]);
  });

  it("writes a metric that could not be computed as a dash", () => {
    expect(metric(null)).toBe("—");
    expect(metric(0.91234)).toBe("0.912");
  });
});
