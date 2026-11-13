import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { forwardRef } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { dataset, health, mockApi } from "../../test/mockApi";
import { renderAt } from "../../test/render";
import { stepRoute } from "./parts";

vi.mock("../../charts/Plot", () => ({
  Plot: forwardRef<HTMLDivElement, { label: string }>(function Plot({ label }, ref) {
    return <div ref={ref} role="img" aria-label={label} />;
  }),
}));

const BASE = "/modules/recipes";
const figure = (id: string, kind: string, label: string, value: number) => ({
  id,
  step: id,
  kind,
  label,
  value,
  detail: kind === "model" ? "roc_auc" : null,
  higher_is_better: kind === "model" ? true : null,
});
const runSummary = (id: string, overrides: Record<string, unknown> = {}) => ({
  id,
  recipe_id: "r1",
  version: 1,
  dataset_id: "d1",
  dataset_name: "ventes_novembre",
  status: "done",
  created_at: "2026-11-07T10:00:00Z",
  finished_at: "2026-11-07T10:01:00Z",
  figures: [figure("s4", "kpi", "Revenue", 1200), figure("s7", "model", "LightGBM → churn", 0.91)],
  kind: "run",
  ...overrides,
});
const steps = [
  { id: "s1", kind: "column", sentence: "Treat code_postal as categorical", enabled: true },
  { id: "s3", kind: "filter", sentence: "Filter “FR+BE”: keep rows where pays is FR or BE", enabled: true },
  { id: "s4", kind: "kpi", sentence: "KPI “Revenue”: sum of montant", enabled: true },
  { id: "s7", kind: "ml", sentence: "Train LightGBM to predict churn", enabled: true },
];
const detail = {
  id: "r1",
  name: "Ventes mensuelles",
  version: 1,
  steps: 4,
  source_dataset_id: "d0",
  source: "ventes_octobre",
  created_at: "2026-11-01T10:00:00Z",
  updated_at: "2026-11-01T10:00:00Z",
  last_run: runSummary("run2"),
  recipe: {
    format: "quantilo/recipe",
    schema: 1,
    name: "Ventes mensuelles",
    version: 1,
    created_at: null,
    source: "ventes_octobre",
    columns: [{ name: "montant", type: "numeric_continuous", categories: null }],
    steps: [],
  },
  step_views: steps,
  runs: [runSummary("run2"), runSummary("run1", { kind: "baseline", dataset_name: "ventes_octobre", dataset_id: "d0" })],
  history: [
    {
      id: "s4",
      label: "Revenue",
      kind: "kpi",
      detail: null,
      points: [
        { run_id: "run1", created_at: "2026-11-01T10:00:00Z", dataset_name: "ventes_octobre", value: 1000 },
        { run_id: "run2", created_at: "2026-11-07T10:00:00Z", dataset_name: "ventes_novembre", value: 1200 },
      ],
    },
  ],
};
const stepResult = (id: string, kind: string, status: string, extra: Record<string, unknown> = {}) => ({
  id,
  kind,
  sentence: steps.find((s) => s.id === id)?.sentence ?? id,
  status,
  message: null,
  warnings: [],
  link: null,
  files: [],
  ...extra,
});
const runDetail = (status = "warning", stepsOut = [
  stepResult("s1", "column", "warning", { message: "Not applied: 40% of the values of date can't be read as a date." }),
  stepResult("s3", "filter", "done"),
  stepResult("s4", "kpi", "done", { link: { page: "kpi", params: { kpi: "k1" }, label: "Open the KPI" } }),
  stepResult("s7", "ml", "skipped", { message: "Uses churn, missing from the file." }),
]) => ({
  ...runSummary("run2", { status }),
  report: {
    recipe: "Ventes mensuelles",
    version: 1,
    dataset: "ventes_novembre",
    status,
    steps: stepsOut,
    matching: { columns: [], new_columns: [] },
    mapping: {},
    figures: [],
    new_columns: ["remise"],
    started_at: "",
    finished_at: null,
  },
  changes: { level: "check", level_label: "Check", headline: "Compared with ventes_octobre: +3.1% rows.", vs: "d0" },
  compared: [
    {
      id: "s4", label: "Revenue", kind: "kpi", value: 1200, previous: 1000, change: 200, change_pct: 0.2, detail: null, higher_is_better: null,
      attained: 1.04, previous_attained: 0.92, crossed: "reached",
      goal: "◎ Revenue reached its target: 104% of target in Nov 2025, 92% at the previous run (R-GOAL-04)",
    },
  ],
  previous_run_id: "run1",
  recipe_name: "Ventes mensuelles",
  error: null,
});

afterEach(() => vi.unstubAllGlobals());

describe("RecipePage", () => {
  it("shows the key figures over time, the runs and a report with each step", async () => {
    mockApi({
      "GET /health": health,
      "GET /datasets": [dataset()],
      [`GET ${BASE}/recipes/r1`]: detail,
      [`GET ${BASE}/recipe-runs/run2`]: runDetail(),
    });
    renderAt("/recipes/r1");

    expect(await screen.findByRole("heading", { name: "Ventes mensuelles" })).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "Revenue over the runs of the recipe" })).toBeInTheDocument();
    const report = await screen.findByRole("region", { name: "Run report" });
    expect(within(report).getByText("Compared with ventes_octobre: +3.1% rows.")).toBeInTheDocument();
    expect(within(report).getByText(/\+200 \(\+20\.0%\)/)).toBeInTheDocument();
    // 061 bis: a target crossed since the previous run is said in words (R-GOAL-04)
    expect(within(report).getByText(/Revenue reached its target: 104% of target/)).toBeInTheDocument();
    expect(within(report).getByText(/40% of the values/)).toBeInTheDocument();
    expect(within(report).getAllByText("Done with a warning").length).toBeGreaterThan(0);
    expect(within(report).getByText("Uses churn, missing from the file.")).toBeInTheDocument();
    expect(within(report).getByRole("link", { name: "Open the KPI" })).toHaveAttribute("href", "/d/d1/kpi?kpi=k1");
    expect(within(report).getByText("remise")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /Export as Python/ })).toHaveAttribute(
      "href",
      "/api/v1/modules/recipes/recipes/r1/python",
    );
  });

  it("a running report can be cancelled", async () => {
    const calls = mockApi({
      "GET /health": health,
      "GET /datasets": [dataset()],
      [`GET ${BASE}/recipes/r1`]: { ...detail, runs: [runSummary("run2", { status: "running" })] },
      [`GET ${BASE}/recipe-runs/run2`]: runDetail("running", [
        stepResult("s1", "column", "done"),
        stepResult("s3", "filter", "running"),
        stepResult("s4", "kpi", "waiting"),
        stepResult("s7", "ml", "waiting"),
      ]),
      [`DELETE ${BASE}/recipe-runs/run2`]: runSummary("run2", { status: "running" }),
    });
    renderAt("/recipes/r1");

    expect(await screen.findByText("1 of 4 steps")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));

    await waitFor(() => expect(calls.some((c) => c.key === `DELETE ${BASE}/recipe-runs/run2`)).toBe(true));
  });

  it("Run on… pairs the columns and lists the steps that will be skipped", async () => {
    const calls = mockApi({
      "GET /health": health,
      "GET /datasets": [dataset({ id: "d2", name: "ventes_novembre" })],
      [`GET ${BASE}/recipes/r1`]: { ...detail, runs: [] },
      [`POST ${BASE}/recipes/r1/plan`]: (body: unknown) => {
        const mapping = (body as { mapping: Record<string, string | null> | null }).mapping;
        return {
          matching: {
            columns: [
              { recipe: "montant", type: "numeric_continuous", column: "Montant", status: "matched", distance: null, file_type: "numeric_continuous" },
              { recipe: "pays", type: "categorical_nominal", column: "pyas", status: "proposed", distance: 2, file_type: "categorical_nominal" },
            ],
            new_columns: ["remise"],
          },
          mapping: mapping ?? { montant: "Montant", pays: null },
          steps: [
            { id: "s3", kind: "filter", sentence: "Filter “FR+BE”", enabled: true, skip: mapping?.pays ? null : "Uses pays, missing from the file." },
          ],
        };
      },
      [`POST ${BASE}/recipes/r1/runs`]: runSummary("run3", { status: "running" }),
      [`GET ${BASE}/recipe-runs/run3`]: runDetail("running"),
    });
    renderAt("/recipes/r1?run=d2");

    expect(await screen.findByText("Matched (case, accents, _)")).toBeInTheDocument();
    expect(screen.getByText("Uses pays, missing from the file.")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("checkbox", { name: /Proposed/ }));
    await waitFor(() => expect(screen.getByText("Every step can run")).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: "Run" }));

    await waitFor(() => {
      const run = calls.find((c) => c.key === `POST ${BASE}/recipes/r1/runs`);
      expect(run?.body).toEqual({ dataset_id: "d2", mapping: { montant: "Montant", pays: "pyas" } });
    });
  });
});

describe("stepRoute", () => {
  it("sends each step to the page it filled", () => {
    expect(stepRoute("d1", { page: "ml", params: { target: "churn", run: "x1" }, label: "" })).toEqual({
      to: "/d/$datasetId/ml",
      params: { datasetId: "d1" },
      search: { target: "churn", run: "x1" },
    });
    expect(stepRoute("d1", { page: "unknown", params: {}, label: "" })).toBeNull();
  });
});
