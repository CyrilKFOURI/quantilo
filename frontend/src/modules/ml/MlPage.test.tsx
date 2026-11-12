import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { forwardRef } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { MlRun } from "../../api/client";
import { column, roles } from "../../test/columns";
import { dataset, health, mockApi } from "../../test/mockApi";
import { renderAt } from "../../test/render";
import { cumulativeFigure, periodScoresFigure } from "./figures";
import { verdict } from "./Results";

vi.mock("../../charts/Plot", () => ({
  Plot: forwardRef<HTMLDivElement, { label: string }>(function Plot({ label }, ref) {
    return <div ref={ref} role="img" aria-label={label} />;
  }),
}));

const candidate = (name: string, task: string | null, overrides: Record<string, unknown> = {}) => ({
  name,
  type: "numeric_continuous",
  role: null,
  task,
  reason: task ? null : "Cannot be predicted: an identifier.",
  suggested: false,
  classes: null,
  missing_share: 0,
  ...overrides,
});

type Model = { key: string; label: string; tasks: string[]; settings?: Record<string, unknown>[]; family?: string; rules?: unknown[]; installed?: boolean; badges?: string[]; desktop_only?: boolean; space?: unknown[] };
type Limits = { max_rows: number; auto_minutes: number; tuning: boolean };

/** The catalogue as the API serves it. */
function catalogue(models: Model[], recommended: { key: string; why: string }[] = [], browser: Limits | null = null) {
  return {
    families: [...new Set(models.map((m) => m.family ?? "Linear models"))],
    models: models.map((m) => ({
      family: "Linear models",
      description: `${m.label} description.`,
      rules: [],
      badges: [],
      installed: true,
      unavailable: null,
      probabilities: true,
      space: [],
      desktop_only: false,
      ...m,
      settings: (m.settings ?? []).map((s) => ({
        kind: "select",
        level: "essential",
        options: [],
        minimum: null,
        maximum: null,
        log: false,
        why: "",
        shown_when: null,
        group: "Main",
        param: "",
        option_labels: {},
        ...s,
      })),
    })),
    recommended,
    metrics: ["r2", "mae"],
    default_metric: "r2",
    compare: models.filter((m) => !m.desktop_only).map((m) => m.key),
    browser,
  };
}

const regressionRun = {
  id: "r1",
  dataset_id: "d1",
  target: "price",
  created_at: "2026-10-18T10:00:00Z",
  task: "regression",
  model: "linear_regression",
  settings: { regularization: "none", alpha: 1 },
  preparation: {
    decisions: [
      { column: "price", type: "numeric_continuous", use: false, rules: ["R-SEL-01"], why: ["This is the target."], impute: null, indicator: false, transform: null, encoding: null, scaling: null, order: null },
      { column: "size", type: "numeric_continuous", use: true, rules: ["R-NUM-08"], why: ["Standardised."], impute: "mean", indicator: false, transform: null, encoding: null, scaling: "standard", order: null },
    ],
    inputs: 1,
    excluded: 1,
  },
  train_rows: 80,
  test_rows: 20,
  rows: 100,
  sampled: false,
  seconds: 0.3,
  regression: {
    r2: 0.82,
    mae: 10,
    rmse: 12,
    mape: null,
    baseline_mae: 17,
    baseline_rmse: 20,
    skill: 0.41,
    actual: [1, 2, 3],
    predicted: [1.1, 2.1, 2.9],
    residual_bins: [{ start: -1, end: 1, count: 3 }],
    heteroscedasticity_p: 0.4,
    inference: "R-REG-SE-05: HC3 standard errors, as in the Regression tab.",
  },
  classification: null,
  coefficients: [{ feature: "size", estimate: 3, std_error: 0.1, low: 2.8, high: 3.2, p_value: 0.0001, q_value: 0.0002, vif: 1, per_unit: 0.5, odds_ratio: null }],
  predictions: [{ row: 4, actual: "10", predicted: "11", error: 1, probability: null, correct: null }],
  primary: "r2",
  score: 0.82,
  classes: null,
  scores: { r2: 0.82, mae: 10 },
  cross_validation: null,
  importances: [],
  validation: {
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
  },
};

function setup() {
  return mockApi({
    "GET /health": health,
    "GET /datasets": [dataset()],
    "GET /datasets/d1": dataset(),
    "GET /datasets/d1/columns": [column({ name: "price" })],
    "GET /column-roles": roles,
    "GET /modules/ml/datasets/d1/targets": {
      candidates: [candidate("price", "regression", { suggested: true }), candidate("vin", null)],
      rows: 100,
    },
    "GET /modules/ml/catalog": catalogue([
      {
        key: "linear_regression",
        label: "Linear regression",
        tasks: ["regression"],
        settings: [{ key: "regularization", label: "Regularization", default: "none", options: ["none", "ridge", "lasso"] }],
      },
    ]),
    "GET /modules/ml/datasets/d1/runs": [],
    "POST /modules/ml/datasets/d1/runs": { id: "job1", status: "running", stage: "loading", run_id: null, error: null },
    "GET /modules/ml/jobs/job1": { id: "job1", status: "done", stage: "saving", run_id: "r1", error: null },
    "GET /modules/ml/datasets/d1/runs/r1": regressionRun,
  });
}

afterEach(() => vi.unstubAllGlobals());

describe("Machine learning page", () => {
  it("asks what to predict, with the columns that cannot be predicted greyed out", async () => {
    setup();
    renderAt("/d/d1/ml");

    expect(await screen.findByRole("heading", { name: "What do you want to predict?" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /vin/ })).toBeDisabled();
    expect(screen.getByRole("button", { name: /price.*Suggested/ })).toBeEnabled();
  });

  it("trains the reference model as soon as a target is chosen, then shows the result", async () => {
    const calls = setup();
    const router = renderAt("/d/d1/ml");

    fireEvent.click(await screen.findByRole("button", { name: /price/ }));

    expect(
      await screen.findByText("The model explains 82 % of the variation in price — 41 % better than always predicting the average."),
    ).toBeInTheDocument();
    const posted = calls.find((c) => c.key === "POST /modules/ml/datasets/d1/runs");
    expect(posted?.params.get("target")).toBe("price");
    expect(posted?.params.get("settings")).toBeNull(); // defaults resolved on the server
    await waitFor(() => expect(router.state.location.search).toMatchObject({ target: "price", run: "r1" }));
    expect(screen.getByText("1 columns → 1 model inputs · 1 excluded")).toBeInTheDocument();
  });

  it("reads the linear coefficients through the Regression engine and links to Inference (079)", async () => {
    setup();
    renderAt("/d/d1/ml");
    fireEvent.click(await screen.findByRole("button", { name: /price/ }));
    fireEvent.click(await screen.findByRole("radio", { name: "Features" }));
    expect(await screen.findByText(/R-REG-SE-05: HC3 standard errors/)).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "adj. p" })).toBeInTheDocument();
    const link = screen.getByRole("link", { name: "Inference →" });
    expect(link.getAttribute("href")).toContain("/d/d1/regression");
    expect(link.getAttribute("href")).toContain("target=price");
    expect(link.getAttribute("href")).toContain("columns=size");
  });
});

describe("Preparation editor", () => {
  const size = (scaling: string, manual: string[]) => ({
    column: "size",
    type: "numeric_continuous",
    use: true,
    rules: ["R-NUM-02", "R-NUM-08"],
    why: ["Missing values → mean.", "Standardised (mean 0, standard deviation 1)."],
    impute: "mean",
    indicator: false,
    transform: null,
    encoding: null,
    scaling,
    order: null,
    auto: { use: "yes", impute: "mean", indicator: "no", transform: "none", encoding: "none", scaling: "standard" },
    manual,
    options: {
      use: [{ value: "yes", reason: null }, { value: "no", reason: null }],
      impute: [
        { value: "mean", reason: null },
        { value: "none", reason: "This model cannot handle missing values: they must be filled." },
      ],
      scaling: ["none", "standard", "robust", "minmax"].map((value) => ({ value, reason: null })),
    },
    reactivable: true,
    leak: false,
  });
  const preparation = (scaling: string, manual: string[]) => ({
    decisions: [
      { ...regressionRun.preparation.decisions[0], auto: { use: "no" }, manual: [], options: {}, reactivable: false, leak: false },
      size(scaling, manual),
      { column: "copy", type: "numeric_continuous", use: false, rules: ["R-SEL-06"], why: ["Looks like it contains the answer (association 0.999 with the target)."], impute: null, indicator: false, transform: null, encoding: null, scaling: null, order: null, auto: { use: "no" }, manual: [], options: {}, reactivable: true, leak: true },
    ],
    inputs: 1,
    excluded: 2,
    choices: { columns: {}, numeric_impute: "auto", indicator_share: 0.05, rare_share: 0.01, rare_rows: 10, one_hot_limit: 15, target_smoothing: 10, outliers: "none", imbalance: "auto", selection: [] },
    custom: manual.length > 0,
    warnings: [],
    selection: [],
    balance: null,
    dropped_rows: 0,
  });

  it("changes a column's scaling, previews it and sends it with the next training", async () => {
    const calls = mockApi({
      "GET /health": health,
      "GET /datasets": [dataset()],
      "GET /datasets/d1": dataset(),
      "GET /datasets/d1/columns": [column({ name: "price" })],
      "GET /column-roles": roles,
      "GET /modules/ml/datasets/d1/targets": { candidates: [candidate("price", "regression")], rows: 100 },
      "GET /modules/ml/catalog": catalogue([{ key: "linear_regression", label: "Linear regression", tasks: ["regression"] }]),
      "GET /modules/ml/datasets/d1/runs": [
        { id: "r1", target: "price", task: "regression", model: "linear_regression", settings: {}, metric: "r2", score: 0.82, seconds: 0.3, created_at: "2026-10-18T10:00:00Z" },
      ],
      "GET /modules/ml/datasets/d1/runs/r1": { ...regressionRun, preparation: preparation("standard", []) },
      "POST /modules/ml/datasets/d1/preparation": (body: unknown) => {
        const chosen = (body as { columns?: Record<string, { scaling?: string }> }).columns?.size?.scaling;
        return {
          preparation: preparation(chosen ?? "standard", chosen ? ["scaling"] : []),
          columns: ["size"],
          rows: [[0.5], [-1.2]],
          row_numbers: [3, 8],
          train_rows: 80,
        };
      },
      "POST /modules/ml/datasets/d1/runs": { id: "job1", status: "running", stage: "loading", run_id: null, error: null },
      "GET /modules/ml/jobs/job1": { id: "job1", status: "running", stage: "loading", run_id: null, error: null },
    });
    renderAt("/d/d1/ml?target=price");

    const edit = await screen.findByRole("button", { name: "Edit" });
    await waitFor(() => expect(edit).toBeEnabled()); // once the model is known
    fireEvent.click(edit);
    const scaling = await screen.findByRole("combobox", { name: "Scaling — size" });
    expect(scaling).toHaveValue(""); // automatic
    expect(screen.getByRole("option", { name: "Auto: Standard" })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: /Leave empty — This model cannot handle missing values/ })).toBeDisabled();
    expect(screen.getByText(/excluded · Looks like it contains the answer/)).toBeInTheDocument();

    fireEvent.change(scaling, { target: { value: "robust" } });
    await waitFor(() => expect(screen.getByRole("combobox", { name: "Scaling — size" })).toHaveValue("robust"));
    expect(screen.getByRole("button", { name: "Back to automatic — size" })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("checkbox", { name: "Use copy" }));
    expect(screen.getByText(/seems to contain the answer/)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Done" }));
    fireEvent.click(screen.getByRole("button", { name: "Train" }));
    await waitFor(() => expect(calls.some((c) => c.key === "POST /modules/ml/datasets/d1/runs")).toBe(true));
    const trained = calls.find((c) => c.key === "POST /modules/ml/datasets/d1/runs");
    expect(trained?.body).toMatchObject({ columns: { size: { scaling: "robust" } } });
  });
});

describe("Predict a number as classes", () => {
  it("cuts the target live and trains a classification 600 ms after the last change", async () => {
    const values = Array.from({ length: 300 }, (_, i) => i + 1);
    const at = (level: number) => values[Math.max(0, Math.ceil(level * 300) - 1)] as number;
    const calls = mockApi({
      "GET /health": health,
      "GET /datasets": [dataset()],
      "GET /datasets/d1": dataset(),
      "GET /datasets/d1/columns": [column({ name: "price" })],
      "GET /column-roles": roles,
      "GET /modules/ml/datasets/d1/targets": { candidates: [candidate("price", "regression")], rows: 300 },
      "GET /modules/ml/catalog": catalogue([
        { key: "linear_regression", label: "Linear regression", tasks: ["regression"] },
        { key: "logistic_regression", label: "Logistic regression", tasks: ["binary", "multiclass"] },
      ]),
      "GET /modules/ml/datasets/d1/runs": [
        { id: "r1", target: "price", task: "regression", model: "linear_regression", settings: {}, metric: "r2", score: 0.82, seconds: 0.3, created_at: "2026-10-18T10:00:00Z", classes: null },
      ],
      "GET /modules/ml/datasets/d1/runs/r1": regressionRun,
      "GET /modules/ml/datasets/d1/targets/price/distribution": {
        column: "price",
        rows: 300,
        distinct: 300,
        min: 1,
        max: 300,
        p1: 3,
        p99: 297,
        quantiles: Array.from({ length: 1001 }, (_, i) => at(i / 1000)),
        equal_counts: Array.from({ length: 9 }, (_, j) => Array.from({ length: j + 1 }, (_, i) => at((i + 1) / (j + 2)))),
        histogram_start: 1,
        histogram_width: 299 / 80,
        histogram: Array.from({ length: 80 }, () => 4),
        sample: values,
        values: null,
      },
      "POST /modules/ml/datasets/d1/runs": { id: "job1", status: "running", stage: "loading", run_id: null, error: null },
      "GET /modules/ml/jobs/job1": { id: "job1", status: "running", stage: "loading", run_id: null, error: null },
    });
    renderAt("/d/d1/ml?target=price");

    fireEvent.click(await screen.findByRole("radio", { name: "Classes" }));
    expect(await screen.findByRole("textbox", { name: "Name of class 3" })).toHaveValue("High");
    expect(screen.getByText("≤ 100")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "More classes" }));
    expect(await screen.findByRole("textbox", { name: "Name of class 4" })).toHaveValue("High");

    await waitFor(() => expect(calls.some((c) => c.key === "POST /modules/ml/datasets/d1/runs")).toBe(true), { timeout: 3000 });
    const posted = calls.filter((c) => c.key === "POST /modules/ml/datasets/d1/runs");
    expect(posted).toHaveLength(1); // one training for the last setup only
    expect(JSON.parse(posted[0]?.params.get("classes") ?? "{}")).toMatchObject({
      method: "quantile",
      edges: [75, 150, 230],
      names: ["Low", "Medium-low", "Medium-high", "High"],
    });
  });
});

describe("Model catalogue", () => {
  it("picks a model by family, shows its essentials, and opens every setting in the drawer", async () => {
    const calls = mockApi({
      "GET /health": health,
      "GET /datasets": [dataset()],
      "GET /datasets/d1": dataset(),
      "GET /datasets/d1/columns": [column({ name: "price" })],
      "GET /column-roles": roles,
      "GET /modules/ml/datasets/d1/targets": { candidates: [candidate("price", "regression")], rows: 100 },
      "GET /modules/ml/catalog": catalogue(
        [
          { key: "linear_regression", label: "Linear regression", tasks: ["regression"] },
          {
            key: "random_forest",
            label: "Random forest",
            family: "Bagging ensembles",
            tasks: ["regression"],
            settings: [
              { key: "n_estimators", label: "Number of trees", kind: "int", default: 300, minimum: 10, maximum: 5000, param: "n_estimators", group: "Structure" },
              { key: "max_features", label: "Features tried per split", default: "1.0", options: ["sqrt", "1.0"], level: "advanced", group: "Sampling", param: "max_features" },
              { key: "criterion", label: "Split quality", default: "squared_error", options: ["squared_error", "absolute_error"], level: "expert", group: "Optimizer", param: "criterion" },
            ],
          },
          { key: "xgboost", label: "XGBoost", family: "Boosting ensembles", tasks: ["regression"], installed: false },
        ],
        [{ key: "random_forest", why: "Robust without tuning." }],
      ),
      "GET /modules/ml/datasets/d1/runs": [
        { id: "r1", target: "price", task: "regression", model: "linear_regression", settings: {}, metric: "r2", score: 0.82, seconds: 0.3, created_at: "2026-10-18T10:00:00Z", classes: null, greater_is_better: true },
      ],
      "GET /modules/ml/datasets/d1/runs/r1": regressionRun,
      "POST /modules/ml/datasets/d1/runs": { id: "job1", status: "running", stage: "loading", run_id: null, error: null },
      "GET /modules/ml/jobs/job1": { id: "job1", status: "running", stage: "loading", run_id: null, error: null },
    });
    renderAt("/d/d1/ml?target=price");

    fireEvent.click(await screen.findByRole("button", { name: /Linear regression.*Linear models/ }));
    expect(await screen.findByText("Recommended for this data")).toBeInTheDocument();
    expect(screen.getByText("Robust without tuning.")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /Boosting ensembles/ }));
    expect(screen.getByRole("button", { name: /XGBoost/ })).toBeDisabled();
    expect(screen.getByText("Not installed in this build.")).toBeInTheDocument();
    fireEvent.click(screen.getAllByRole("button", { name: /Random forest/ })[0] as HTMLElement);

    const trees = await screen.findByRole("spinbutton", { name: "Number of trees" });
    expect(trees).toHaveValue(300);
    expect(screen.queryByRole("combobox", { name: "Features tried per split" })).not.toBeInTheDocument(); // advanced

    fireEvent.click(screen.getByRole("button", { name: "All settings" }));
    expect(await screen.findByRole("combobox", { name: "Features tried per split" })).toBeInTheDocument();
    expect(screen.queryByRole("tab", { name: "Optimizer" })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("checkbox", { name: "Expert" }));
    fireEvent.click(screen.getByRole("tab", { name: "Optimizer" }));
    expect(screen.getByRole("combobox", { name: "Split quality" })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("tab", { name: "Validation" }));
    fireEvent.change(screen.getByRole("combobox", { name: "Method" }), { target: { value: "kfold" } });
    const train = screen.getAllByRole("button", { name: "Train" });
    fireEvent.click(train[train.length - 1] as HTMLElement);

    await waitFor(() => expect(calls.some((c) => c.key === "POST /modules/ml/datasets/d1/runs")).toBe(true));
    const posted = calls.find((c) => c.key === "POST /modules/ml/datasets/d1/runs");
    expect(posted?.params.get("model")).toBe("random_forest");
    expect(JSON.parse(posted?.params.get("validation") ?? "{}")).toMatchObject({ method: "kfold", folds: 5 });
  });
});

describe("Browser version (102)", () => {
  const run = { id: "r1", target: "price", task: "regression", model: "linear_regression", settings: {}, metric: "r2", score: 0.82, seconds: 0.3, created_at: "2026-10-18T10:00:00Z", classes: null, greater_is_better: true };
  const api = (served: ReturnType<typeof catalogue>) =>
    mockApi({
      "GET /health": health,
      "GET /datasets": [dataset()],
      "GET /datasets/d1": dataset(),
      "GET /datasets/d1/columns": [column({ name: "price" })],
      "GET /column-roles": roles,
      "GET /modules/ml/datasets/d1/targets": { candidates: [candidate("price", "regression")], rows: 100 },
      "GET /modules/ml/catalog": served,
      "GET /modules/ml/datasets/d1/runs": [run],
      "GET /modules/ml/datasets/d1/runs/r1": regressionRun,
    });

  it("offers the scikit-learn models only and says once what is left to the desktop app", async () => {
    api(
      catalogue(
        [
          { key: "linear_regression", label: "Linear regression", tasks: ["regression"], space: [{ key: "alpha" }] },
          { key: "hist_gradient_boosting", label: "Histogram gradient boosting", family: "Boosting ensembles", tasks: ["regression"] },
          { key: "xgboost", label: "XGBoost", family: "Boosting ensembles", tasks: ["regression"], installed: false, desktop_only: true },
          { key: "lightgbm", label: "LightGBM", family: "Boosting ensembles", tasks: ["regression"], installed: false, desktop_only: true },
          { key: "catboost", label: "CatBoost", family: "Boosting ensembles", tasks: ["regression"], installed: false, desktop_only: true },
        ],
        [],
        { max_rows: 50_000, auto_minutes: 5, tuning: false },
      ),
    );
    renderAt("/d/d1/ml?target=price");

    expect(
      await screen.findByText(
        "Browser version: scikit-learn models, trained on one core on at most 50,000 rows (R-WEB-01, R-WEB-02). XGBoost, LightGBM, CatBoost and tuning are available in the desktop app (R-WEB-04).",
      ),
    ).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Tune" })).not.toBeInTheDocument(); // no dead button
    expect(screen.getByRole("button", { name: "Auto" })).toBeEnabled();

    fireEvent.click(screen.getByRole("button", { name: /Linear regression.*Linear models/ }));
    const picker = await screen.findByRole("dialog");
    expect(within(picker).getByText("2 models for this problem, by family")).toBeInTheDocument();
    fireEvent.click(within(picker).getByRole("button", { name: /Boosting ensembles/ }));
    expect(within(picker).getByRole("button", { name: /Histogram gradient boosting/ })).toBeEnabled();
    expect(within(picker).queryByRole("button", { name: /XGBoost/ })).not.toBeInTheDocument();
    expect(within(picker).getByText("XGBoost, LightGBM and CatBoost: available in the desktop app (R-WEB-04).")).toBeInTheDocument();
  });

  it("keeps the Tune button and says nothing of the browser on a computer", async () => {
    api(catalogue([{ key: "linear_regression", label: "Linear regression", tasks: ["regression"], space: [{ key: "alpha" }] }]));
    renderAt("/d/d1/ml?target=price");

    expect(await screen.findByRole("button", { name: "Tune" })).toBeInTheDocument();
    expect(screen.queryByText(/R-WEB-04/)).not.toBeInTheDocument();
  });
});

describe("Explanations", () => {
  it("shows what the model relies on, how columns move predictions, and explains one row", async () => {
    let calls = 0;
    mockApi({
      "GET /health": health,
      "GET /datasets": [dataset()],
      "GET /datasets/d1": dataset(),
      "GET /datasets/d1/columns": [column({ name: "price" })],
      "GET /column-roles": roles,
      "GET /modules/ml/datasets/d1/targets": { candidates: [candidate("price", "regression")], rows: 100 },
      "GET /modules/ml/catalog": catalogue([{ key: "linear_regression", label: "Linear regression", tasks: ["regression"] }]),
      "GET /modules/ml/datasets/d1/runs": [
        { id: "r1", target: "price", task: "regression", model: "linear_regression", settings: {}, metric: "r2", score: 0.82, seconds: 0.3, created_at: "2026-10-18T10:00:00Z", classes: null, greater_is_better: true },
      ],
      "GET /modules/ml/datasets/d1/runs/r1": regressionRun,
      "GET /modules/ml/datasets/d1/runs/r1/explanation": () =>
        ++calls < 2
          ? { status: "running", error: null, explanation: null }
          : {
              status: "done",
              error: null,
              explanation: {
                permutation: { metric: "r2", rows: 20, sampled: false, items: [{ column: "size", mean: 0.6, std: 0.02 }] },
                shap: {
                  method: "Exact (linear model)",
                  scale: "prediction",
                  base: 100,
                  positive: null,
                  rows: [4],
                  features: [{ column: "size", mean_abs: 12, values: [15], colors: [0.9], reading: "Higher size pushes the prediction up." }],
                  columns: ["size", "city"],
                  matrix: [[15, -4]],
                  labels: [["120", "Nice"]],
                },
                shap_unavailable: null,
                dependence: [{ column: "size", kind: "number", grid: [1, 2], curves: { prediction: [90, 110] }, counts: [5, 5], output: "Average prediction" }],
                tree: null,
                pairs: [{ first: "size", second: "rooms", strength: 0.91, measure: "|r|" }],
                warnings: [],
                words: [
                  {
                    column: "notes",
                    method: "linear",
                    output: "prediction",
                    terms: 812,
                    up: [{ term: "sea view", effect: 21.5, rows: 0.12, towards: null }],
                    down: [],
                    sentence: "In notes, “sea view” pushes the prediction up most (+21.5), in prediction: exact, through the model's coefficients.",
                  },
                ],
              },
            },
    });
    renderAt("/d/d1/ml?target=price&run=r1");

    fireEvent.click(await screen.findByRole("radio", { name: "Explain" }));
    expect(await screen.findByText(/Computing the explanation/)).toBeInTheDocument();
    expect(await screen.findByText("Higher size pushes the prediction up.", {}, { timeout: 3000 })).toBeInTheDocument();
    expect(screen.getByText(/size and rooms are strongly linked/)).toBeInTheDocument();
    expect(screen.getByRole("combobox", { name: "Column" })).toHaveValue("size");
    expect(screen.getByText(/“sea view” pushes the prediction up most/)).toBeInTheDocument();
    expect(within(screen.getByRole("table", { name: "Pushes up" })).getByText("sea view")).toBeInTheDocument();
    expect(screen.getByText("No term in this direction.")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("radio", { name: "Predictions" }));
    fireEvent.click(screen.getByRole("button", { name: "Explain row 5" }));
    expect(await screen.findByText("Row 5: the strongest pushes come from size (+15), city (-4).")).toBeInTheDocument();
  });
});

describe("Comparing and tuning", () => {
  const summary = (id: string, extra: Record<string, unknown> = {}) => ({
    id,
    target: "price",
    task: "regression",
    model: "linear_regression",
    label: "Linear regression",
    settings: {},
    metric: "r2",
    score: 0.8,
    seconds: 0.2,
    created_at: "2026-10-19T10:00:00Z",
    classes: null,
    greater_is_better: true,
    validation: regressionRun.validation,
    ...extra,
  });
  const routes = (extra: Record<string, unknown>) => ({
    "GET /health": health,
    "GET /datasets": [dataset()],
    "GET /datasets/d1": dataset(),
    "GET /datasets/d1/columns": [column({ name: "price" })],
    "GET /column-roles": roles,
    "GET /modules/ml/datasets/d1/targets": { candidates: [candidate("price", "regression")], rows: 100 },
    "GET /modules/ml/catalog": catalogue([
      {
        key: "linear_regression",
        label: "Linear regression",
        tasks: ["regression"],
        settings: [{ key: "alpha", label: "Strength", kind: "number", default: 1, minimum: 0.000001, maximum: 10000, level: "advanced" }],
        space: [{ key: "alpha", kind: "float", low: 0.0001, high: 100, log: true, choices: [] }],
      } as Model,
      { key: "random_forest", label: "Random forest", family: "Bagging ensembles", tasks: ["regression"] },
    ]),
    "GET /modules/ml/datasets/d1/runs/r1": regressionRun,
    ...extra,
  });

  it("compares models in one job and ranks them", async () => {
    const calls = mockApi(
      routes({
        "GET /modules/ml/datasets/d1/runs": [summary("r1")],
        "POST /modules/ml/datasets/d1/compare": { id: "c1", status: "running", metric: "r2", items: [] },
        "GET /modules/ml/compare-jobs/c1": {
          id: "c1",
          status: "done",
          metric: "r2",
          items: [
            { key: "linear_regression", label: "Linear regression", status: "done", run_id: "r2", score: 0.6, seconds: 0.1, error: null },
            { key: "random_forest", label: "Random forest", status: "done", run_id: "r3", score: 0.9, seconds: 1.2, error: null },
          ],
        },
      }),
    );
    renderAt("/d/d1/ml?target=price");

    fireEvent.click(await screen.findByRole("button", { name: "Compare models" }));
    fireEvent.click(await screen.findByRole("button", { name: "Compare" }));
    expect(await screen.findByText(/Done: best first/)).toBeInTheDocument();
    const ranked = within(screen.getByRole("dialog")).getAllByRole("row").map((r) => r.textContent ?? "");
    expect(ranked.join(" | ")).toMatch(/Random forest ★.*Linear regression/);
    const posted = calls.find((c) => c.key === "POST /modules/ml/datasets/d1/compare");
    expect(posted?.body).toMatchObject({ models: ["linear_regression", "random_forest"] });
  });

  it("runs the Auto within a budget: steps, journal and leaderboard, the kept model as a run", async () => {
    const entry = (id: string, label: string, score: number | null, extra: Record<string, unknown> = {}) => ({
      id,
      model: id.split(":")[0],
      label,
      family: "Linear models",
      step: "families",
      settings: {},
      status: "done",
      score,
      std: 0.01,
      seconds: 1.5,
      rows: 80,
      share: 1,
      note: null,
      estimate: null,
      ...extra,
    });
    const step = (key: string, status: string, note = "") => ({ key, title: key, status, seconds: 2, note });
    const progress = {
      budget_seconds: 600,
      metric: "r2",
      greater_is_better: true,
      criterion: null,
      compared_on: "Your validation is a train / test split: every model is compared on 5 folds of the training rows.",
      step: null,
      steps: [step("baselines", "done"), step("families", "done"), step("reduction", "done"), step("tuning", "done"), step("ensemble", "done", "kept")],
      entries: [
        entry("constant", "Constant", -0.01, { family: "Baseline" }),
        entry("linear_regression", "Linear regression", 0.6),
        entry("random_forest", "Random forest", 0.8, { family: "Bagging ensembles" }),
        entry("knn", "k-nearest neighbours", null, { status: "skipped", note: "Too slow for the budget." }),
        entry("ensemble", "Ensemble", 0.85, { family: "Ensemble" }),
      ],
      log: [{ at: 0.2, step: "baselines", message: "Constant: r2 -0.01." }],
      ensemble: {
        method: "greedy",
        members: [
          { entry: "random_forest", model: "random_forest", label: "Random forest", weight: 3, settings: {} },
          { entry: "linear_regression", model: "linear_regression", label: "Linear regression", weight: 1, settings: {} },
        ],
        score: 0.85,
        std: 0.01,
        best_single: "random_forest",
        best_score: 0.8,
        best_std: 0.02,
        kept: true,
        reason: "The ensemble scores 0.85 against 0.8 for Random forest alone: kept.",
        stacking_score: null,
      },
      best: "ensemble",
      gain: "63% lower error than the linear model",
      test_score: 0.84,
      seconds: 590,
      trials: 40,
      stopped: false,
      finished: true,
    };
    const calls = mockApi(
      routes({
        "GET /modules/ml/datasets/d1/runs": [summary("r1")],
        "POST /modules/ml/datasets/d1/auto": { id: "a1", status: "running", target: "price", progress: null, run_id: null, error: null },
        "GET /modules/ml/auto-jobs/a1": { id: "a1", status: "done", target: "price", progress, run_id: "r9", error: null },
        "GET /modules/ml/datasets/d1/runs/r9": { ...regressionRun, id: "r9", model: "ensemble", settings: { method: "greedy" } },
      }),
    );
    const router = renderAt("/d/d1/ml?target=price");

    fireEvent.click(await screen.findByRole("button", { name: "Auto" }));
    expect(screen.getByText(/compared on 5 folds of the training rows/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("radio", { name: "10 min" }));
    fireEvent.click(screen.getByRole("button", { name: "Start the Auto" }));

    expect(await screen.findByRole("heading", { name: "Kept: Ensemble" })).toBeInTheDocument();
    const posted = calls.find((c) => c.key === "POST /modules/ml/datasets/d1/auto");
    expect(posted?.body).toMatchObject({ minutes: 10, stacking: false });
    expect(JSON.parse(posted?.params.get("validation") ?? "{}")).toMatchObject({ metric: "r2" });
    const dialog = within(screen.getByRole("dialog"));
    expect(dialog.getByText("63% lower error than the linear model.")).toBeInTheDocument();
    expect(dialog.getByText("On the test rows: R² 0.84")).toBeInTheDocument();
    expect(dialog.getByText("Constant: r2 -0.01.")).toBeInTheDocument();
    const rows = dialog.getAllByRole("row").map((r) => r.textContent ?? "");
    expect(rows.join(" | ")).toMatch(/Ensemblekept.*Random forest.*Linear regression.*Constant.*k-nearest neighbours.*Too slow/);
    expect(dialog.getByText("Random forest ×3, Linear regression ×1")).toBeInTheDocument();
    fireEvent.click(dialog.getByRole("button", { name: "Open the run" }));
    await waitFor(() => expect(router.state.location.search).toMatchObject({ run: "r9" }));
  });

  it("only lets comparable runs be ticked", async () => {
    mockApi(
      routes({
        "GET /modules/ml/datasets/d1/runs": [
          summary("r1"),
          summary("r2", { validation: { ...regressionRun.validation, method: "kfold" } }),
          summary("r3", { model: "random_forest", label: "Random forest", score: 0.9 }),
        ],
        "GET /modules/ml/datasets/d1/runs/r3": { ...regressionRun, id: "r3", model: "random_forest", settings: { n_estimators: 300 } },
      }),
    );
    renderAt("/d/d1/ml?target=price&run=r1");

    fireEvent.click(await screen.findByRole("checkbox", { name: "Compare run #3" }));
    expect(screen.getByRole("checkbox", { name: "Compare run #2" })).toBeDisabled();
    fireEvent.click(screen.getByRole("checkbox", { name: "Compare run #1" }));
    expect(await screen.findByRole("region", { name: "Runs side by side" })).toBeInTheDocument();
    expect(screen.getByText("n_estimators")).toBeInTheDocument();
  });

  it("tunes the model within a budget and uses the settings found", async () => {
    const calls = mockApi(
      routes({
        "GET /modules/ml/datasets/d1/runs": [summary("r1")],
        "POST /modules/ml/datasets/d1/tune": { id: "t1", status: "running", model: "linear_regression", label: "Linear regression", progress: null, error: null, space: [] },
        "GET /modules/ml/tune-jobs/t1": {
          id: "t1",
          status: "done",
          model: "linear_regression",
          label: "Linear regression",
          error: null,
          space: [],
          progress: {
            metric: "r2",
            greater_is_better: true,
            trials: [
              { number: 0, value: 0.7, state: "complete", params: { alpha: 3 }, seconds: 0.1 },
              { number: 1, value: 0.85, state: "complete", params: { alpha: 0.2 }, seconds: 0.1 },
            ],
            best_value: 0.85,
            best_params: { alpha: 0.2 },
            default_value: 0.8,
            importances: {},
            stopped: false,
            seconds: 2,
          },
        },
        "POST /modules/ml/datasets/d1/runs": { id: "job9", status: "running", stage: "loading", run_id: null, error: null },
        "GET /modules/ml/jobs/job9": { id: "job9", status: "running", stage: "loading", run_id: null, error: null },
      }),
    );
    renderAt("/d/d1/ml?target=price");

    fireEvent.click(await screen.findByRole("button", { name: "Tune" }));
    fireEvent.click(await screen.findByRole("radio", { name: "5 min" }));
    fireEvent.click(screen.getByRole("button", { name: "Start the search" }));
    expect(await screen.findByText("+0.05 R² over the defaults")).toBeInTheDocument();
    const tuned = calls.find((c) => c.key === "POST /modules/ml/datasets/d1/tune");
    expect(tuned?.body).toMatchObject({ model: "linear_regression", budget_seconds: 300, sampler: "tpe" });

    fireEvent.click(screen.getByRole("button", { name: "Use these settings" }));
    await waitFor(() => expect(calls.some((c) => c.key === "POST /modules/ml/datasets/d1/runs")).toBe(true));
    const trained = calls.find((c) => c.key === "POST /modules/ml/datasets/d1/runs");
    expect(JSON.parse(trained?.params.get("settings") ?? "{}")).toMatchObject({ alpha: 0.2 });
  });
});

describe("Long trainings", () => {
  it("estimates the duration of a slow model and cancels a training", async () => {
    const calls = mockApi({
      "GET /health": health,
      "GET /datasets": [dataset()],
      "GET /datasets/d1": dataset(),
      "GET /datasets/d1/columns": [column({ name: "price" })],
      "GET /column-roles": roles,
      "GET /modules/ml/datasets/d1/targets": { candidates: [candidate("price", "regression")], rows: 100 },
      "GET /modules/ml/catalog": catalogue([
        { key: "linear_regression", label: "Linear regression", tasks: ["regression"] },
        { key: "kernel_svm", label: "Kernel SVM", family: "Support vector machines", tasks: ["regression"], badges: ["Slow on 20,000 rows"] } as Model,
      ]),
      "GET /modules/ml/datasets/d1/runs": [
        { id: "r1", target: "price", task: "regression", model: "kernel_svm", label: "Kernel SVM", settings: {}, metric: "r2", score: 0.8, seconds: 40, created_at: "2026-10-19T10:00:00Z", classes: null, greater_is_better: true, validation: regressionRun.validation },
      ],
      "GET /modules/ml/datasets/d1/runs/r1": { ...regressionRun, model: "kernel_svm" },
      "POST /modules/ml/datasets/d1/estimate": { seconds: 75, fits: 1, rows: 16000 },
      "POST /modules/ml/datasets/d1/runs": { id: "job5", status: "running", stage: "training", run_id: null, error: null },
      "GET /modules/ml/jobs/job5": { id: "job5", status: "running", stage: "training", run_id: null, error: null },
      "DELETE /modules/ml/jobs/job5": { id: "job5", status: "cancelled", stage: "training", run_id: null, error: "Stopped: nothing was saved." },
    });
    renderAt("/d/d1/ml?target=price");

    fireEvent.click(await screen.findByRole("button", { name: "How long will it take?" }));
    expect(await screen.findByText(/About 1 min 15 s \(1 fit\)/)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Train" }));
    fireEvent.click(await screen.findByRole("button", { name: "Cancel" }));
    expect(await screen.findByText("Stopped: nothing was saved.")).toBeInTheDocument();
    expect(calls.some((c) => c.key === "DELETE /modules/ml/jobs/job5")).toBe(true);
  });
});

describe("ML verdicts", () => {
  const run = regressionRun as unknown as MlRun;

  it("says when a model is no better than the naive reference", () => {
    const flat = { ...run, regression: { ...run.regression!, skill: -0.2 } } as MlRun;
    expect(verdict(flat)).toContain("does no better than always predicting the average");
    const negative = { ...run, regression: { ...run.regression!, r2: -0.01, skill: 0.004 } } as MlRun;
    expect(verdict(negative)).toContain("does no better than always predicting the average");

    const classification = {
      ...run,
      regression: null,
      classification: {
        labels: ["no", "yes"],
        positive: "yes",
        accuracy: 0.6,
        balanced_accuracy: 0.52,
        f1_macro: 0.5,
        roc_auc: 0.55,
        log_loss: 0.7,
        baseline_accuracy: 0.64,
        majority: "no",
        confusion: [],
        report: [],
        roc: null,
        pr: null,
        calibration: null,
        truth: [],
        probability: [],
      },
    } as unknown as MlRun;
    expect(verdict(classification)).toBe(
      "The model does no better than always answering “no”: the features may not carry information about the classes. (ROC AUC 0.55: 0.5 is chance.)",
    );
  });
});

describe("Panel data", () => {
  const check = {
    panel: true,
    period: "era",
    entity: "id",
    periods: 574,
    per_period: 5000,
    entities: 5000,
    duplicates: 12,
    embargo: 4,
    embargo_reason: "The target of an id stays correlated with its next ones for 3 periods (r = 0.25 at lag 3): it likely looks 4 periods ahead.",
    period_candidates: ["era"],
    entity_candidates: ["id"],
    message: "This looks like panel data: 574 eras × ~5,000 ids. Random splits would overestimate the score.",
  };
  const summary = {
    metric: "spearman",
    periods: 3,
    mean: 0.02,
    std: 0.01,
    sharpe: 2,
    threshold: 0,
    share_good: 2 / 3,
    worst: -0.01,
    worst_period: "0575",
    longest_bad_streak: 1,
    lower_is_better: false,
  };
  const panelValidationSaved = {
    ...regressionRun.validation,
    method: "panel_walk_forward",
    period_column: "era",
    entity_column: "id",
    embargo: 4,
    window: "expanding",
    window_periods: null,
    sample_periods: null,
    criterion: "mean",
    period_threshold: null,
    threshold: "default",
  };
  const panelRun = {
    ...regressionRun,
    id: "r2",
    primary: "spearman",
    score: 0.02,
    validation: panelValidationSaved,
    panel: {
      period_column: "era",
      entity_column: "id",
      window: "expanding",
      embargo: 4,
      step: 4,
      criterion: "mean",
      value: 0.02,
      periods: [
        { period: "0575", rank: 574, fold: 1, rows: 5000, score: -0.01, pearson: -0.01, spearman: -0.01 },
        { period: "0576", rank: 575, fold: 2, rows: 5000, score: 0.03, pearson: 0.02, spearman: 0.03 },
        { period: "0577", rank: 576, fold: 3, rows: 5000, score: 0.04, pearson: 0.05, spearman: 0.04 },
      ],
      summary,
      folds: [
        { fold: 1, train_start: 0, train_end: 569, test_start: 574, test_end: 574, train_periods: 143, test_periods: 1, train_rows: 715000, test_rows: 5000, train_from: "0001", train_to: "0570", test_from: "0575", test_to: "0575" },
      ],
      exposures: [{ column: "feature_a", mean: 0.42 }],
      max_exposure: 0.42,
      exposure_columns: 2,
      alerts: [{ kind: "future", message: "late_column looks like a column from the future: it follows the next era's target.", column: "late_column" }],
      suggested_embargo: 4,
      embargo_reason: null,
    },
  };
  const summaryRun = {
    id: "r1",
    target: "price",
    task: "regression",
    model: "linear_regression",
    settings: {},
    metric: "r2",
    score: 0.82,
    seconds: 0.3,
    created_at: "2026-10-18T10:00:00Z",
    classes: null,
    greater_is_better: true,
    label: "",
    validation: regressionRun.validation,
  };

  function panelSetup(runs: unknown[]) {
    return mockApi({
      "GET /health": health,
      "GET /datasets": [dataset()],
      "GET /datasets/d1": dataset(),
      "GET /datasets/d1/columns": [column({ name: "price" })],
      "GET /column-roles": roles,
      "GET /modules/ml/datasets/d1/targets": {
        candidates: [candidate("price", "regression"), candidate("era", "regression", { type: "numeric_discrete" }), candidate("id", null, { type: "identifier" })],
        rows: 100,
      },
      "GET /modules/ml/catalog": catalogue([{ key: "linear_regression", label: "Linear regression", tasks: ["regression"] }]),
      "GET /modules/ml/datasets/d1/panel": check,
      "GET /modules/ml/datasets/d1/runs": runs,
      "GET /modules/ml/datasets/d1/runs/r1": regressionRun,
      "GET /modules/ml/datasets/d1/runs/r2": panelRun,
      "POST /modules/ml/datasets/d1/runs": { id: "job1", status: "running", stage: "loading", run_id: null, error: null },
      "GET /modules/ml/jobs/job1": { id: "job1", status: "running", stage: "loading", run_id: null, error: null },
      "POST /modules/ml/datasets/d1/runs/r2/neutralize": {
        columns: ["feature_a"],
        proportion: 1,
        value: 0.025,
        periods: panelRun.panel.periods,
        summary: { ...summary, mean: 0.025, sharpe: 3.1 },
        before: [{ column: "feature_a", mean: 0.42 }],
        after: [{ column: "feature_a", mean: 0.001 }],
      },
    });
  }

  it("says the rows look like a panel and switches to walk-forward with the embargo found", async () => {
    const calls = panelSetup([summaryRun]);
    renderAt("/d/d1/ml?target=price&run=r1");

    expect(await screen.findByText(check.message)).toBeInTheDocument();
    expect(screen.getByText(/same ids are in training and test rows/)).toBeInTheDocument();
    expect(screen.getByText(/12 rows repeat an \(id, era\) pair/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Use panel validation" }));
    expect(await screen.findByText(/Panel walk-forward · 5 folds · era · embargo 4/)).toBeInTheDocument();
    expect(screen.queryByText(check.message)).not.toBeInTheDocument();

    const train = screen.getAllByRole("button", { name: "Train" });
    fireEvent.click(train[train.length - 1] as HTMLElement);
    await waitFor(() => expect(calls.some((c) => c.key === "POST /modules/ml/datasets/d1/runs")).toBe(true));
    const posted = calls.filter((c) => c.key === "POST /modules/ml/datasets/d1/runs").pop();
    expect(JSON.parse(posted?.params.get("validation") ?? "{}")).toMatchObject({
      method: "panel_walk_forward",
      period_column: "era",
      entity_column: "id",
      embargo: 4,
    });
  });

  it("proposes the embargo the target calls for, with the reason", async () => {
    panelSetup([summaryRun]);
    renderAt("/d/d1/ml?target=price&run=r1");

    fireEvent.click(await screen.findByRole("button", { name: "Edit validation" }));
    fireEvent.click(await screen.findByRole("tab", { name: "Validation" }));
    fireEvent.change(screen.getByRole("combobox", { name: "Method" }), { target: { value: "panel_walk_forward" } });
    const embargo = await screen.findByRole("spinbutton", { name: "Embargo (periods)" });
    expect(embargo).toHaveValue(4);
    fireEvent.blur(embargo, { target: { value: "0" } });
    expect(await screen.findByText(check.embargo_reason)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Use 4" }));
    expect(await screen.findByRole("spinbutton", { name: "Embargo (periods)" })).toHaveValue(4);
    expect(screen.queryByText(check.embargo_reason)).not.toBeInTheDocument();
    expect(screen.getByRole("combobox", { name: "Primary metric" })).toHaveTextContent("Auto (Spearman ρ)");
    expect(screen.getByRole("combobox", { name: "Score over the periods" })).toHaveValue("mean");
  });

  it("shows every period, the folds, the panel checks, and neutralizes on a column", async () => {
    const calls = panelSetup([{ ...summaryRun, id: "r2", metric: "spearman", score: 0.02, validation: panelValidationSaved }]);
    renderAt("/d/d1/ml?target=price&run=r2");

    expect(await screen.findByText(/Spearman ρ per era: 0.02 on average over 3 test periods/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("radio", { name: "Periods" }));
    expect(screen.getAllByText("Mean / std").length).toBeGreaterThan(0);
    expect(screen.getByText("67 %")).toBeInTheDocument();
    expect(screen.getByText(/late_column looks like a column from the future/)).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "Spearman ρ per era" })).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "Cumulative Spearman ρ" })).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "Folds on the time axis" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Export the periods (CSV)" })).toHaveAttribute("href", expect.stringContaining("/runs/r2/periods.csv"));

    expect(screen.getByRole("checkbox", { name: "feature_a" })).toBeChecked();
    fireEvent.click(screen.getByRole("button", { name: "Neutralize" }));
    expect(await screen.findByText(/After neutralizing: Spearman ρ 0.03 on average \(was 0.02\), mean \/ std 3.1 \(was 2\)/)).toBeInTheDocument();
    const posted = calls.find((c) => c.key === "POST /modules/ml/datasets/d1/runs/r2/neutralize");
    expect(posted?.body).toEqual({ columns: ["feature_a"], proportion: 1 });
    expect(screen.getByText("Exposure to feature_a")).toBeInTheDocument();
  });
});

describe("Panel figures", () => {
  const tokens = { ink1: "#000", ink2: "#111", ink3: "#222", line: "#333", surface: "#fff", accent: "#a00", numeric: "#00f", categorical: "#f0f", date: "#0f0", bool: "#ff0", text: "#444", id: "#555", font: "sans", palette: ["#2a78d6", "#eb6834"] };
  const summary = { metric: "spearman", periods: 2, mean: 0, std: 0.1, sharpe: 0, threshold: 0, share_good: 0.5, worst: -0.1, worst_period: "2", longest_bad_streak: 1, lower_is_better: false };
  const periods = [
    { period: "1", rank: 0, fold: 1, rows: 3, score: 0.1, pearson: null, spearman: null },
    { period: "2", rank: 1, fold: 1, rows: 3, score: -0.1, pearson: null, spearman: null },
  ];

  it("mutes the periods below the threshold and adds the scores up in time", () => {
    const bars = periodScoresFigure({ periods, summary, period_column: "era" })(tokens);
    const colors = (bars.data[0] as { marker: { color: string[] } }).marker.color;
    expect(colors[0]).toBe("#2a78d6");
    expect(colors[1]).not.toBe("#2a78d6");
    const line = cumulativeFigure({ periods, summary, period_column: "era" })(tokens);
    expect((line.data[0] as { y: number[] }).y).toEqual([0.1, 0]);
  });
});
