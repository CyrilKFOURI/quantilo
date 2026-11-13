import type { Prediction, TaskOut, VisionRun } from "./modelApi";

/** 113: a task, runs and predictions for the tests of Model and Results, on the six images of
 *  `testData.ts` (rows 0 to 5). */

const weight = (key: string, title: string, size: string, downloaded: boolean) => ({
  key,
  title,
  size,
  licence: "BSD-3-Clause",
  note: "trained on ImageNet: ImageNet terms apply (non-commercial research use)",
  downloaded,
  download: "idle" as const,
  progress: 0,
  error: null,
});

const choiceBase = {
  chosen_by: "rule" as const,
  rule_task: null,
  rule_code: "",
  rule_message: "",
  n_images: 6,
  n_usable: 6,
  classes: {},
  left_out: {},
  n_normal: 0,
  n_defect: 0,
  evaluable: false,
  pixel: false,
  alternatives: [],
  blocked: null,
  notes: [],
  target: null,
  n_target: 0,
  label_names: [],
  label_source: null,
  numeric_targets: [],
  binary_targets: [],
};

export const anomalyTask: TaskOut = {
  choice: {
    ...choiceBase,
    task: "anomaly",
    code: "R-VML-02",
    message: "Anomaly detection: the model learns from 4 normal images (R-VML-02).",
    reason: "only normals in train",
    n_normal: 4,
    n_defect: 2,
    evaluable: true,
    alternatives: ["clusters"],
  },
  split: { kind: "producer", code: "R-VSP-04", reason: "The producer's split (R-VSP-04).", notes: [], folds: 1, train: 4, scored: 2, holdout: 0, leaks: 0 },
  deep: { used: false, backbone: "dinov2_s14", reason: "Download the weights first." },
  models: ["PCA of patches (reconstruction error)", "Nearest normal images (classic descriptors)", "Patch PCA + nearest normals (average rank)"],
  detectors: [
    { key: "patchcore", title: "PatchCore", level: "first", available: false, reason: "Download the weights first: Wide ResNet-50-2, 276 MB (R-VDP-02).", note: "", weights: ["wide_resnet50_2", "resnet18"] },
    { key: "padim", title: "PaDiM", level: "advanced", available: false, reason: "Download the weights first: ResNet-18, 47 MB.", note: "", weights: ["resnet18"] },
    { key: "autoencoder", title: "Convolutional autoencoder", level: "advanced", available: true, reason: null, note: "Trained here.", weights: [] },
    { key: "vae", title: "Variational autoencoder", level: "advanced", available: true, reason: null, note: "Trained here.", weights: [] },
  ],
  default_detector: null,
  weights: [weight("wide_resnet50_2", "Wide ResNet-50-2", "276 MB", false), weight("resnet18", "ResNet-18", "47 MB", false)],
  installed: true,
  install: 'pip install "quantilo[vision]"',
  more: [],
  concepts: [],
};

const interval = (low: number, high: number) => ({ low, high, level: 0.95, draws: 1000, dropped: 0, by_group: true });

const split = { kind: "producer", code: "R-VSP-01", reason: "producer", folds: [{ train: [0, 1, 2], test: [3, 4, 5] }], holdout: [], leaks: 0, seed: 0, notes: [] };

function run(result: Record<string, unknown>): VisionRun {
  return {
    id: "aaaaaaaaaaaa",
    dataset_id: "d1",
    created_at: "2026-11-12T10:00:00Z",
    request: { task: null, deep: null, threshold: null, confirm_long: false, seed: 0, detector: "auto" },
    result: { split, deep: { used: false, backbone: "dinov2_s14", reason: null }, features: [], default: null, clusters: null, seconds: 12, seed: 0, notes: [], ...result },
  } as unknown as VisionRun;
}

const model = (key: string, title: string, extra: Record<string, unknown>) => ({
  key,
  title,
  family: "classic",
  setting: "",
  default: false,
  selection: null,
  classification: null,
  anomaly: null,
  threshold: null,
  flagged: null,
  galleries: {},
  seconds: 1,
  notes: [],
  map_range: null,
  map_source: null,
  reconstructions: false,
  ...extra,
});

export const anomalyRun = run({
  task: { ...anomalyTask.choice },
  sentence: "Patch PCA + nearest normals (average rank): AUROC 1.000.",
  default: "ensemble",
  models: [
    model("ensemble", "Patch PCA + nearest normals (average rank)", {
      default: true,
      map_source: "patch_pca",
      map_range: [0, 1],
      anomaly: {
        n_normal: 2,
        n_defect: 1,
        auroc: 1,
        ap: 1,
        auroc_interval: interval(1, 1),
        auroc_by_type: { blot: 1 },
        threshold: { code: "R-VTH-02", threshold: 0.6, reason: "q99", message: "Threshold 0.6: about 1 false alarm in 100 normal images (R-VTH-02).", calibration: [], evaluation: [] },
        at_threshold: null,
        histogram: { edges: [0, 1], normal: [2], defect: [1], threshold: 0.6 },
        pixel: null,
        notes: [],
      },
    }),
    model("knn", "Nearest normal images (classic descriptors)", { anomaly: { n_normal: 2, n_defect: 1, auroc: 0.5, ap: 0.5, auroc_interval: null, auroc_by_type: {}, threshold: null, at_threshold: null, histogram: { edges: [0, 1], normal: [2], defect: [1] }, pixel: null, notes: [] } }),
  ],
});

export const anomalyPredictions: Prediction[] = [
  { row: 3, image_id: "x", fold: 0, truth: "good", is_defect: false, holdout: false, ensemble_score: 0.2, knn_score: 0.5 },
  { row: 4, image_id: "y", fold: 0, truth: "good", is_defect: false, holdout: false, ensemble_score: 0.5, knn_score: 0.9 },
  { row: 5, image_id: "z", fold: 0, truth: "blot", is_defect: true, holdout: false, ensemble_score: 0.9, knn_score: 0.6 },
];

const report = {
  labels: ["cat", "dog"],
  n: 3,
  main: { metric: "accuracy", code: "R-VQ-13", reason: "Classes are balanced: accuracy." },
  accuracy: 0.667,
  balanced_accuracy: 0.75,
  f1_macro: 0.667,
  kappa: 0.4,
  per_class: [
    { label: "cat", precision: 0.5, recall: 1, f1: 0.667, support: 1 },
    { label: "dog", precision: 1, recall: 0.5, f1: 0.667, support: 2 },
  ],
  confusion: [
    [1, 0],
    [1, 1],
  ],
  confusion_normalized: [
    [1, 0],
    [0.5, 0.5],
  ],
  baseline: { label: "dog", accuracy: 0.667, balanced_accuracy: 0.5, reason: "R-VM-01" },
  roc_auc: 1,
  pr_auc: 1,
  roc_auc_by_class: { cat: 1, dog: 1 },
  pr_auc_by_class: { cat: 1, dog: 1 },
  log_loss: 0.4,
  brier: 0.2,
  calibration: {
    reliability: { edges: [0, 0.5, 1], confidence: [null, 0.8], accuracy: [null, 0.667], counts: [0, 3], ece: 0.13, mce: 0.13 },
    suggest_temperature: true,
    temperature: 1.6,
    ece_after: 0.04,
    reason: "ECE above 0.05: temperature scaling is proposed.",
  },
  top_k: {},
  intervals: { accuracy: interval(0.33, 1) },
  notes: [],
};

export const classificationRun = run({
  task: { ...choiceBase, task: "classification", code: "R-VML-01", message: "Classification: 2 classes (R-VML-01).", reason: "labels", classes: { cat: 3, dog: 3 } },
  sentence: "Logistic regression on classic descriptors: accuracy 0.667.",
  default: "logistic",
  models: [model("logistic", "Logistic regression on classic descriptors", { default: true, classification: report, galleries: { confident_errors: [4], uncertain: [3, 4, 5] } })],
});

export const classificationPredictions: Prediction[] = [
  { row: 3, image_id: "x", fold: 0, truth: "cat", logistic_pred: "cat", logistic_proba: 0.9, logistic_probs: [0.9, 0.1] },
  { row: 4, image_id: "y", fold: 0, truth: "dog", logistic_pred: "cat", logistic_proba: 0.7, logistic_probs: [0.7, 0.3] },
  { row: 5, image_id: "z", fold: 0, truth: "dog", logistic_pred: "dog", logistic_proba: 0.8, logistic_probs: [0.2, 0.8] },
];

export const runSummary = (task: "anomaly" | "classification") => ({
  id: "aaaaaaaaaaaa",
  created_at: "2026-11-12T10:00:00Z",
  task,
  chosen_by: "rule",
  default: task === "anomaly" ? "ensemble" : "logistic",
  default_title: task === "anomaly" ? "Patch PCA + nearest normals (average rank)" : "Logistic regression on classic descriptors",
  metric: task === "anomaly" ? "auroc" : "accuracy",
  score: task === "anomaly" ? 1 : 0.667,
  models: [],
  deep: false,
  scored: 3,
  seconds: 12,
  sentence: "",
});

// c3: a regression run (R-VML-05, R-VRG) and a multi-label run (R-VML-06, R-VMLB)
export const regressionRun = run({
  task: { ...choiceBase, task: "regression", code: "R-VML-05", message: "Predict a number: weight_g (R-VML-05).", reason: "numeric", target: "weight_g", n_target: 3, numeric_targets: ["weight_g"] },
  sentence: "Ridge regression on classic descriptors (weight_g): R² 0.810, MAE 4.2 against 11 for the training mean (-62 %).",
  default: "ridge",
  models: [
    model("ridge", "Ridge regression on classic descriptors", {
      default: true,
      regression: { n: 3, r2: 0.81, mae: 4.2, rmse: 5.1, median_ae: 3.9, spearman: 0.9, baseline_mae: 11, baseline_rmse: 12, baseline_r2: 0, train_mean: 50, intervals: { r2: interval(0.5, 0.95) }, sentence: "R² 0.810." },
      galleries: { over: [4], under: [5] },
    }),
  ],
});

export const regressionPredictions: Prediction[] = [
  { row: 3, image_id: "x", fold: 0, truth: 40, ridge_pred: 42 },
  { row: 4, image_id: "y", fold: 0, truth: 55, ridge_pred: 61 },
  { row: 5, image_id: "z", fold: 0, truth: 70, ridge_pred: 66 },
];

export const multilabelRun = run({
  task: { ...choiceBase, task: "multilabel", code: "R-VML-06", message: "Multi-label: 2 labels (R-VML-06).", reason: "pipes", label_names: ["spot", "stripe"], label_source: "pipe" },
  sentence: "One logistic regression per label: Macro F1 0.750 over 2 labels.",
  default: "ovr_logistic",
  models: [
    model("ovr_logistic", "One logistic regression per label (classic descriptors)", {
      default: true,
      multilabel: {
        n: 3,
        labels: ["spot", "stripe"],
        exact_match: 0.667,
        hamming_loss: 0.167,
        micro_f1: 0.8,
        macro_f1: 0.75,
        per_label: [
          { label: "spot", support: 1, prevalence: 0.4, precision: 1, recall: 1, f1: 1, auroc: 1 },
          { label: "stripe", support: 2, prevalence: 0.5, precision: 1, recall: 0.5, f1: 0.5, auroc: 0.75 },
        ],
        baseline_exact_match: 0.333,
        baseline_hamming_loss: 0.5,
        baseline_macro_f1: 0.4,
        intervals: {},
        threshold: 0.5,
        sentence: "Macro F1 0.750 over 2 labels (each label at its training frequency: 0.400); exact match 0.667.",
      },
      galleries: { most_wrong: [5] },
    }),
  ],
});

export const multilabelPredictions: Prediction[] = [
  { row: 3, image_id: "x", fold: 0, truth: ["spot"], ovr_logistic_pred: ["spot"], ovr_logistic_probs: [0.9, 0.1] },
  { row: 4, image_id: "y", fold: 0, truth: ["stripe"], ovr_logistic_pred: ["stripe"], ovr_logistic_probs: [0.1, 0.8] },
  { row: 5, image_id: "z", fold: 0, truth: ["stripe"], ovr_logistic_pred: [], ovr_logistic_probs: [0.2, 0.3] },
];
