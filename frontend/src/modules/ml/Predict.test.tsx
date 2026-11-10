import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { createMemoryHistory, createRootRoute, createRouter, RouterProvider } from "@tanstack/react-router";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { forwardRef, useState } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { MlRun } from "../../api/client";
import { column } from "../../test/columns";
import { dataset, mockApi } from "../../test/mockApi";
import { distributionFigure } from "./predictFigures";
import { PredictSection } from "./Predict";

vi.mock("../../charts/Plot", () => ({
  Plot: forwardRef<HTMLDivElement, { label: string }>(function Plot({ label }, ref) {
    return <div ref={ref} role="img" aria-label={label} />;
  }),
}));

const run = { id: "r1", target: "churn", task: "binary", classification: { threshold: null, positive: "leave" } } as unknown as MlRun;

const columnCheck = (name: string, overrides: Record<string, unknown> = {}) => ({
  column: name,
  expected: "categorical_nominal",
  found: "categorical_nominal",
  importance: 0.4,
  important: true,
  converted: false,
  unreadable: 0,
  missing: 0,
  unknown_rows: 0,
  unknown_share: 0,
  unknown_values: [],
  unknown_rule: "grouped with the rare categories (Other)",
  out_of_range: 0,
  range: null,
  note: null,
  unknown_terms: null,
  training_unknown_terms: null,
  unmatched_texts: null,
  ...overrides,
});

const ready = {
  level: "ready",
  message: "Ready: the 600 rows have the columns, types and values the model was trained on.",
  rows: 600,
  columns: [columnCheck("region"), columnCheck("income", { expected: "numeric_continuous", found: "numeric_continuous", importance: 0.6 })],
  missing: [],
  ignored: ["client_id"],
  target_present: true,
  problems: [],
  drift: null,
};

const blocked = {
  ...ready,
  level: "blocked",
  message: "Blocked: Missing column the model needs: income.",
  columns: [columnCheck("income", { found: null, note: "Missing: the model needs this column." }), columnCheck("region")],
  missing: ["income"],
};

const prediction = {
  id: "p1",
  run_id: "r1",
  source_id: "d2",
  source_name: "november",
  rows: 600,
  less_reliable: 18,
  level: "check",
  created_at: "2026-11-07T10:00:00Z",
  created_dataset_id: null,
  keep: ["client_id"],
  check: { ...ready, level: "check", message: "Check: the model can predict, but region: 3% of rows have an unknown category." },
  summary: {
    rows: 600,
    less_reliable: 18,
    columns: ["client_id", "predicted", "p(leave)", "p(stay)", "less_reliable", "why", "actual"],
    distribution: { kind: "classes", edges: [], labels: ["leave", "stay"], new: [0.4, 0.6], validation: [0.45, 0.55] },
    scores: { roc_auc: 0.91, accuracy: 0.85 },
    scored_rows: 600,
    validation_scores: { roc_auc: 0.95, accuracy: 0.88 },
    primary: "roc_auc",
    interval: null,
  },
};

const rows = {
  columns: ["row", "client_id", "predicted", "p(leave)", "p(stay)", "less_reliable", "why", "actual"],
  rows: [
    [1, "C11-0000", "leave", 0.8, 0.2, true, "region: unknown category", "leave"],
    [2, "C11-0001", "stay", 0.1, 0.9, false, "", "stay"],
  ],
  total: 600,
  offset: 0,
};

function Section() {
  const [shown, setShown] = useState<string | undefined>(undefined);
  return <PredictSection run={run} datasetId="d1" predictionId={shown} onPrediction={setShown} />;
}

function renderSection() {
  const root = createRootRoute({ component: Section });
  const router = createRouter({ routeTree: root, history: createMemoryHistory({ initialEntries: ["/"] }) });
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <RouterProvider router={router} />
    </QueryClientProvider>,
  );
}

function routes(check: unknown, made: { done: boolean }) {
  return {
    "GET /datasets": [dataset(), dataset({ id: "d2", name: "november" })],
    "GET /datasets/d2/columns": [
      column({ name: "client_id", type: "identifier" }),
      column({ name: "region", type: "categorical_nominal" }),
      column({ name: "income" }),
    ],
    "GET /modules/ml/datasets/d1/runs/r1/check": check,
    "POST /modules/ml/datasets/d1/runs/r1/predictions": () => {
      made.done = true;
      return { id: "j1", status: "running", rows: 0, total: 600, prediction_id: null, error: null };
    },
    "GET /modules/ml/predict-jobs/j1": { id: "j1", status: "done", rows: 600, total: 600, prediction_id: "p1", error: null },
    "GET /modules/ml/datasets/d1/runs/r1/predictions": () => (made.done ? [prediction] : []),
    "GET /modules/ml/datasets/d1/predictions/p1": prediction,
    "GET /modules/ml/datasets/d1/predictions/p1/rows": rows,
  };
}

describe("Predict on new data", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("shows the share of unknown terms of a free-text column", async () => {
    const note = "48 % of its terms are unknown to the model (21 % in training): they are ignored.";
    const texts = {
      ...ready,
      level: "check",
      message: "Check: the model can predict, but review: 48% of its terms are unknown to the model (21% in training).",
      columns: [columnCheck("review", { expected: "text", found: "text", unknown_terms: 0.48, training_unknown_terms: 0.21, unmatched_texts: 0.1, note })],
    };
    mockApi(routes(texts, { done: false }));
    renderSection();

    fireEvent.click(await screen.findByRole("button", { name: "Predict on new data" }));
    const dialog = await screen.findByRole("dialog");
    expect(await within(dialog).findByText("48 % of terms")).toHaveAttribute("title", expect.stringContaining("21 % in training"));
    expect(within(dialog).getByText(note)).toBeInTheDocument();
  });

  it("checks the other dataset, predicts it with the identifiers kept, then shows the result", async () => {
    const calls = mockApi(routes(ready, { done: false }));
    renderSection();

    expect(await screen.findByRole("link", { name: "Export model (.zip)" })).toHaveAttribute("href", "/api/v1/modules/ml/datasets/d1/runs/r1/model.zip");
    expect(screen.getByText(/Importing a model file is not offered/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Predict on new data" }));
    const dialog = await screen.findByRole("dialog");
    expect(await within(dialog).findByText(/The 600 rows have the columns/)).toBeInTheDocument();
    expect(calls.find((c) => c.key.endsWith("/check"))?.params.get("source")).toBe("d2");
    expect(within(dialog).getByText(/This data has the answer/)).toBeInTheDocument();
    expect(within(dialog).getByText("Not read by the model: client_id.")).toBeInTheDocument();
    await waitFor(() => expect(within(dialog).getByRole("checkbox", { name: "client_id" })).toBeChecked());

    fireEvent.click(within(dialog).getByRole("button", { name: "Predict" }));

    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    expect(calls.find((c) => c.key.startsWith("POST"))?.body).toEqual({ source: "d2", keep: ["client_id"] });
    expect(await screen.findByText(/600 rows of november predicted/)).toBeInTheDocument();
    expect(screen.getByText(/3 % are less reliable/)).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "Predictions on this data against the validation" })).toBeInTheDocument();
    expect(screen.getByText("Scores on this data (600 rows with the answer)")).toBeInTheDocument();
    expect(screen.getByText(/The model holds/)).toBeInTheDocument();
    expect(await screen.findByText("C11-0000")).toBeInTheDocument();
    expect(screen.getByText("region: unknown category")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Export CSV" })).toHaveAttribute("href", "/api/v1/modules/ml/datasets/d1/predictions/p1/predictions.csv");
    expect(screen.getByRole("button", { name: "Save as dataset" })).toBeEnabled();
  });

  it("names the missing column and does not let a blocked check predict", async () => {
    mockApi(routes(blocked, { done: false }));
    renderSection();

    fireEvent.click(await screen.findByRole("button", { name: "Predict on new data" }));
    const dialog = await screen.findByRole("dialog");

    expect(await within(dialog).findByText("Missing column the model needs: income.")).toBeInTheDocument();
    expect(within(dialog).getByText("Missing: the model needs this column.")).toBeInTheDocument();
    expect(within(dialog).getByRole("button", { name: "Predict" })).toBeDisabled();
  });
});

describe("prediction distribution figure", () => {
  const tokens = { palette: ["#111111"], ink2: "#222222", ink3: "#333333", surface: "#ffffff", numeric: "#444444" } as never;

  it("draws classes as grouped bars and a number as bars with the validation as a step line", () => {
    const classes = distributionFigure(prediction.summary.distribution as never)(tokens);
    expect(classes.data.map((d) => d.type)).toEqual(["bar", "bar"]);
    expect(classes.layout?.barmode).toBe("group");

    const number = distributionFigure({ kind: "number", edges: [0, 1, 2], labels: [], new: [0.5, 0.5], validation: [0.4, 0.6] } as never)(tokens);
    expect(number.data.map((d) => d.type)).toEqual(["bar", "scatter"]);
    expect((number.data[0] as { x: number[] }).x).toEqual([0.5, 1.5]);
  });
});
