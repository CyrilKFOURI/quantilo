import { fireEvent, screen, within } from "@testing-library/react";
import { forwardRef } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { column, roles } from "../../test/columns";
import { dataset, health, mockApi } from "../../test/mockApi";
import { renderAt } from "../../test/render";
import { errorByColumnFigure, shownBins } from "./errorFigures";

vi.mock("../../charts/Plot", () => ({
  Plot: forwardRef<HTMLDivElement, { label: string }>(function Plot({ label }, ref) {
    return <div ref={ref} role="img" aria-label={label} />;
  }),
}));

afterEach(() => vi.unstubAllGlobals());

const validation = {
  method: "split",
  test_size: 0.2,
  folds: 5,
  repeats: 3,
  group: null,
  time_column: null,
  metric: null,
  seed: 42,
  max_rows: 200000,
  period_column: null,
  entity_column: null,
  embargo: 0,
  window: "expanding",
  window_periods: null,
  sample_periods: null,
  criterion: "mean",
  period_threshold: null,
};

const decision = (name: string, type = "numeric_continuous", use = true) => ({
  column: name,
  type,
  use,
  rules: [],
  why: [],
  impute: null,
  indicator: false,
  transform: null,
  encoding: null,
  scaling: null,
  order: null,
});

const run = {
  id: "r1",
  dataset_id: "d1",
  target: "price",
  created_at: "2026-11-07T10:00:00Z",
  task: "regression",
  model: "linear_regression",
  settings: {},
  preparation: { decisions: [decision("price", "numeric_continuous", false), decision("region", "categorical_nominal"), decision("age"), decision("vin", "identifier", false)], inputs: 2, excluded: 1 },
  train_rows: 800,
  test_rows: 200,
  rows: 1000,
  sampled: false,
  seconds: 0.3,
  regression: { r2: 0.8, mae: 10, rmse: 12, mape: null, baseline_mae: 17, baseline_rmse: 20, skill: 0.41, actual: [1], predicted: [1], residual_bins: [], heteroscedasticity_p: null },
  classification: null,
  coefficients: [],
  predictions: [],
  primary: "r2",
  score: 0.8,
  classes: null,
  scores: { r2: 0.8 },
  cross_validation: null,
  importances: [],
  validation,
  evaluation_rows: [],
  learning_curve: null,
  panel: null,
};

const southYoung = {
  where: "region = South and age < 25",
  conditions: [
    { column: "region", op: "in", values: ["South"], negate: false, value: null, low: null, high: null, part: null, columns: [], keep: "first", order_by: null },
    { column: "age", op: "lt", value: 25, negate: false, values: [], low: null, high: null, part: null, columns: [], keep: "first", order_by: null },
  ],
  columns: ["region", "age"],
  rows: 42,
  share: 0.042,
  error: 31,
  ratio: 2.6,
  excess: 800,
  p_value: 0.00001,
  q_value: 0.0004,
  found_by: "tree",
  sentence: "When region = South and age < 25 (4.2 % of rows), the model is off by 31 on average vs 12 overall.",
  examples: [{ row: 9, actual: "100", predicted: "180", error: 80, probability: null }],
};

const errors = (overrides: Record<string, unknown> = {}) => ({
  task: "regression",
  measure: "absolute_error",
  measures: ["absolute_error"],
  label: "absolute error",
  source: "test rows",
  rows: 200,
  overall: 12,
  columns: 2,
  tested: 64,
  slices: [southYoung],
  message: null,
  by_column: [
    { column: "region", kind: "category", bins: [{ label: "South", rows: 50, error: 20 }, { label: "North", rows: 150, error: 9 }], spread: 11, worst: "South", excluded: false },
    { column: "age", kind: "number", bins: [{ label: "< 25", rows: 40, error: 18 }, { label: "≥ 25", rows: 160, error: 10 }], spread: 8, worst: "< 25", excluded: false },
  ],
  worst: [
    { row: 6, actual: "100", predicted: "190", error: 90, probability: null },
    { row: 9, actual: "100", predicted: "180", error: 80, probability: null },
  ],
  hints: [{ kind: "few_rows", text: "Few examples where region = South and age < 25 (42 rows): more data for this group would help.", column: null }],
  conditions: [],
  ...overrides,
});

function setup(extra: Record<string, unknown> = {}) {
  return mockApi({
    "GET /health": health,
    "GET /datasets": [dataset()],
    "GET /datasets/d1": dataset(),
    "GET /datasets/d1/columns": [column({ name: "price" })],
    "GET /column-roles": roles,
    "GET /modules/ml/datasets/d1/targets": {
      candidates: [{ name: "price", type: "numeric_continuous", role: null, task: "regression", reason: null, suggested: true, classes: null, missing_share: 0 }],
      rows: 1000,
    },
    "GET /modules/ml/catalog": { families: [], models: [], recommended: [], metrics: ["r2"], default_metric: "r2", compare: [] },
    "GET /modules/ml/datasets/d1/runs": [],
    "GET /modules/ml/datasets/d1/runs/r1": run,
    "GET /modules/ml/datasets/d1/runs/r1/errors": errors(),
    ...extra,
  });
}

describe("Errors tab", () => {
  it("lists the failing subgroups in sentences and opens one on the other pages", async () => {
    setup();
    renderAt("/d/d1/ml?target=price&run=r1");

    fireEvent.click(await screen.findByRole("radio", { name: "Errors" }));
    const sentence = await screen.findByText(southYoung.sentence);
    expect(screen.getByText(/On 200 test rows · absolute error: 12 on average · 64 subgroups tested in 2 columns/)).toBeInTheDocument();
    expect(screen.getByText(/Few examples where region = South/)).toBeInTheDocument();

    fireEvent.click(sentence);
    const data = screen.getByRole("link", { name: /Open in Data/ });
    const compare = screen.getByRole("link", { name: /Open in Compare/ });
    for (const link of [data, compare]) {
      const href = decodeURIComponent(link.getAttribute("href") ?? "");
      expect(href).toMatch(/^\/d\/d1\/(data|compare)\?/);
      expect(href).toContain('"column":"region"');
      expect(href).toContain('"values":["South"]');
      expect(href).toContain('"op":"lt"');
    }
    expect(screen.getByText(/q < 0.001 after Benjamini–Hochberg · found by the tree on the error/)).toBeInTheDocument();
    expect(within(screen.getByRole("table", { name: "Worst predictions of this group" })).getByText("180")).toBeInTheDocument();
  });

  it("says when no subgroup is notably worse", async () => {
    setup({ "GET /modules/ml/datasets/d1/runs/r1/errors": errors({ slices: [], hints: [], message: "No subgroup is notably worse than average." }) });
    renderAt("/d/d1/ml?target=price&run=r1");

    fireEvent.click(await screen.findByRole("radio", { name: "Errors" }));
    expect(await screen.findByText("No subgroup is notably worse than average.")).toBeInTheDocument();
  });

  it("shows the error by column and explains a worst prediction", async () => {
    setup({
      "GET /modules/ml/datasets/d1/runs/r1/explanation": {
        status: "done",
        error: null,
        explanation: {
          permutation: null,
          shap: { method: "Exact (linear model)", scale: "prediction", base: 100, positive: null, rows: [6], features: [], columns: ["age", "region"], matrix: [[70, 20]], labels: [["19", "South"]] },
          shap_unavailable: null,
          dependence: [],
          tree: null,
          pairs: [],
          warnings: [],
        },
      },
    });
    renderAt("/d/d1/ml?target=price&run=r1");

    fireEvent.click(await screen.findByRole("radio", { name: "Errors" }));
    expect(await screen.findByRole("img", { name: "Error by region" })).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "Error by age" })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Explain row 7" }));
    expect(await screen.findByText("Row 7: the strongest pushes come from age (+70), region (+20).")).toBeInTheDocument();
  });

  it("switches a classifier between wrong predictions and log-loss", async () => {
    const calls = setup({
      "GET /modules/ml/datasets/d1/runs/r1": { ...run, task: "binary", regression: null, classification: null },
      "GET /modules/ml/datasets/d1/runs/r1/errors": errors({ task: "binary", measure: "misclassified", measures: ["misclassified", "log_loss"], label: "share of wrong predictions", overall: 0.12 }),
    });
    renderAt("/d/d1/ml?target=price&run=r1");

    fireEvent.click(await screen.findByRole("radio", { name: "Errors" }));
    expect(await screen.findByText(/share of wrong predictions: 12 % on average/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("radio", { name: "Log-loss" }));
    await screen.findByRole("radio", { name: "Log-loss" });
    expect(calls.some((c) => c.key === "GET /modules/ml/datasets/d1/runs/r1/errors" && c.params.get("measure") === "log_loss")).toBe(true);
  });

  it("puts the metrics of each group of a sensitive column side by side", async () => {
    const calls = setup({
      "GET /modules/ml/datasets/d1/runs/r1/errors/fairness": {
        column: "region",
        metrics: ["mae", "mean_predicted", "mean_actual"],
        labels: { mae: "Mean absolute error", mean_predicted: "Mean prediction", mean_actual: "Mean actual value" },
        groups: [
          { group: "North", rows: 150, values: { mae: 9, mean_predicted: 100, mean_actual: 101 }, few: false },
          { group: "South", rows: 50, values: { mae: 20, mean_predicted: 98, mean_actual: 110 }, few: false },
        ],
        gaps: { mae: 11, mean_predicted: 2, mean_actual: 9 },
        positive: null,
      },
    });
    renderAt("/d/d1/ml?target=price&run=r1");

    fireEvent.click(await screen.findByRole("radio", { name: "Errors" }));
    const select = await screen.findByRole("combobox", { name: "Sensitive column" });
    expect(within(select).queryByRole("option", { name: "vin" })).toBeNull(); // an identifier
    fireEvent.change(select, { target: { value: "region" } });
    const table = await screen.findByRole("table", { name: "Metrics by group" });
    expect(within(table).getByText("Largest gap")).toBeInTheDocument();
    expect(within(table).getByText("11")).toBeInTheDocument();
    expect(calls.find((c) => c.key.endsWith("/errors/fairness"))?.params.get("column")).toBe("region");
    expect(screen.getByText(/Display only: nothing is corrected/)).toBeInTheDocument();
  });

  it("compares the errors of two runs", async () => {
    const summary = (id: string) => ({ id, target: "price", task: "regression", model: "linear_regression", label: "Linear regression", settings: {}, metric: "r2", score: 0.8, seconds: 0.2, created_at: "2026-11-07T10:00:00Z", classes: null, greater_is_better: true, validation });
    setup({
      "GET /modules/ml/datasets/d1/runs": [summary("r2"), summary("r1")],
      "GET /modules/ml/datasets/d1/runs/r2": { ...run, id: "r2" },
      "GET /modules/ml/datasets/d1/runs/r2/errors/compare/r1": {
        measure: "absolute_error",
        label: "absolute error",
        rows: 200,
        first: 12,
        second: 10,
        better: [{ where: "region = South", conditions: [], rows: 50, share: 0.25, first: 20, second: 11, p_value: 0.001, q_value: 0.01, sentence: "When region = South (25.0 % of rows), the absolute error goes from 20 to 11 (-45 %)." }],
        worse: [],
        message: null,
      },
    });
    renderAt("/d/d1/ml?target=price&run=r1");

    fireEvent.click(await screen.findByRole("checkbox", { name: "Compare run #2" }));
    fireEvent.click(screen.getByRole("checkbox", { name: "Compare run #1" }));
    const section = await screen.findByRole("region", { name: "Where the errors change" });
    expect(await within(section).findByText(/the absolute error goes from 20 to 11/)).toBeInTheDocument();
    expect(within(section).getByText(/Where #2 Linear regression fixes errors/)).toBeInTheDocument();
  });
});

describe("Error by column figure", () => {
  it("keeps the 12 most populated categories and mutes the small ones", () => {
    const bins = Array.from({ length: 15 }, (_, i) => ({ label: `c${i}`, rows: i < 3 ? 5 : 100 + i, error: i }));
    const kept = shownBins({ column: "city", kind: "category", bins, spread: 1, worst: null, excluded: false });
    expect(kept).toHaveLength(12);
    expect(kept.map((b) => b.label)).not.toContain("c0");
    const tokens = { ink1: "#000", ink2: "#111", ink3: "#222", line: "#333", surface: "#fff", accent: "#f00", numeric: "#00f", palette: ["#123456"] };
    const figure = errorByColumnFigure({ column: "age", kind: "number", bins: [{ label: "< 25", rows: 10, error: 3 }, { label: "≥ 25", rows: 90, error: 1 }], spread: 2, worst: "< 25", excluded: false }, 1.2, "absolute error")(tokens as never);
    const colors = (figure.data[0] as { marker: { color: string[] } }).marker.color;
    expect(colors[1]).toBe("#123456");
    expect(colors[0]).not.toBe("#123456");
  });
});
