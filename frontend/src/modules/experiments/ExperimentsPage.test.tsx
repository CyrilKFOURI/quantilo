import { fireEvent, screen, within } from "@testing-library/react";
import { forwardRef } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { readTokens } from "../../charts/theme";
import { column, roles } from "../../test/columns";
import fixtures from "../../test/experiments.json";
import { dataset, health, mockApi } from "../../test/mockApi";
import { renderAt } from "../../test/render";
import { linkTarget } from "../agent/links";
import { modules } from "../registry";
import type { ExperimentMetricResult, SampleSize } from "./api";
import { intervalsFigure, sampleSizeFigure } from "./figures";
import { validateExperimentsSearch } from "./search";

vi.mock("../../charts/Plot", () => ({
  Plot: forwardRef<HTMLDivElement, { label: string }>(function Plot({ label }, ref) {
    return <div ref={ref} role="img" aria-label={label} />;
  }),
}));

/* Responses written by the real engine on samples/ab_test.csv (src/test/experiments.json). */
const { candidates, loose, byUser, srm, plan } = fixtures;

function setup(analyze: (body: { unit?: string | null; expected?: unknown }) => unknown = (body) => (body.unit ? byUser : loose)) {
  return mockApi({
    "GET /health": health,
    "GET /datasets": [dataset()],
    "GET /datasets/d1": dataset(),
    "GET /datasets/d1/columns": [column({ name: "revenue" })],
    "GET /column-roles": roles,
    "GET /modules/experiments/datasets/d1/candidates": candidates,
    "POST /modules/experiments/datasets/d1/analyze": analyze,
    "GET /modules/experiments/datasets/d1/baseline": { column: "converted", kind: "rate", value: 0.1, sd: 0.3, rows: 4023 },
    "POST /modules/experiments/sample-size": plan,
  });
}

afterEach(() => vi.unstubAllGlobals());

describe("Experiments page", () => {
  it("opens on the suggested experiment: by user, with its sentence, chart and Holm line", async () => {
    const calls = setup();
    renderAt("/d/d1/experiments");

    expect(await screen.findByRole("heading", { name: "Experiments" })).toBeInTheDocument();
    expect(await screen.findByText(/new_checkout reaches 25.0 % vs 19.8 % for control/)).toBeInTheDocument();
    expect(screen.getAllByText("Significant · higher").length).toBeGreaterThan(0);
    expect(screen.getByText("p adjusted for 2 comparisons (Holm)")).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "Difference of converted to the control, with its interval" })).toBeInTheDocument();
    const table = screen.getByRole("table", { name: "converted by variant" });
    expect(within(table).getByText("+5.2 pts")).toBeInTheDocument();
    expect(screen.getByRole("combobox", { name: "Unit" })).toHaveValue("user_id");
    expect(screen.getByRole("heading", { name: /converted over time, cumulative by day/ })).toBeInTheDocument();
    expect(screen.getByText(/Subgroups are hypotheses to confirm/)).toBeInTheDocument();
    const sent = calls.find((c) => c.key === "POST /modules/experiments/datasets/d1/analyze");
    expect(sent?.body).toMatchObject({ variant: "variant", unit: "user_id", metrics: [{ column: "converted" }] });
  });

  it("warns when an identifier repeats and switches to it as the unit", async () => {
    const calls = setup();
    renderAt("/d/d1/experiments?variant=variant&metric=converted");

    expect(await screen.findByText(/user_id repeats \(2.0 rows per value\)/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Use user_id as the unit" }));

    expect(await screen.findByText(/new_checkout reaches 25.0 %/)).toBeInTheDocument();
    const last = calls.filter((c) => c.key === "POST /modules/experiments/datasets/d1/analyze").at(-1);
    expect(last?.body).toMatchObject({ unit: "user_id" });
  });

  it("puts the sample ratio alarm before any result", async () => {
    setup(() => srm);
    renderAt("/d/d1/experiments");

    const alarm = await screen.findByRole("alert");
    expect(alarm).toHaveTextContent("The groups are not the size you planned: the results may be biased.");
  });

  it("plans the sample size: per group, total and the curve", async () => {
    const calls = setup();
    renderAt("/d/d1/experiments?tab=plan");

    expect(await screen.findByText("57,756")).toBeInTheDocument();
    expect(screen.getByText(plan.sentence)).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "Units per group against the smallest detectable lift" })).toBeInTheDocument();
    const sent = calls.find((c) => c.key === "POST /modules/experiments/sample-size");
    expect(sent?.body).toMatchObject({ kind: "rate", baseline: 0.1, effect: 0.05, relative: true, alpha: 0.05, power: 0.8 });

    fireEvent.change(screen.getByRole("combobox", { name: "Read from" }), { target: { value: "converted" } });
    expect(await screen.findByDisplayValue("10")).toBeInTheDocument();
  });
});

describe("Experiments module", () => {
  it("is available in the browser-only build too (not locked)", () => {
    const found = modules.find((m) => m.id === "experiments");
    expect(found?.path).toBe("/d/$datasetId/experiments");
    expect(found?.locked).toBeFalsy();
  });

  it("keeps a well-formed experiment from the URL and drops the rest", () => {
    expect(validateExperimentsSearch({ tab: "plan", design: { variant: "g", metrics: [{ column: "y" }] } })).toEqual({
      tab: "plan",
      design: { variant: "g", metrics: [{ column: "y" }] },
    });
    expect(validateExperimentsSearch({ tab: "x", design: { variant: "g", metrics: [] }, metric: 3 })).toEqual({});
  });

  it("opens from an assistant link", () => {
    expect(linkTarget("d1", { page: "experiments", params: { variant: "variant", metric: "converted", unit: "user_id" }, label: "" })).toEqual({
      to: "/d/$datasetId/experiments",
      params: { datasetId: "d1" },
      search: { variant: "variant", metric: "converted", unit: "user_id" },
    });
  });

  it("draws each variant in its palette colour with the lift written, and the plan on the curve", () => {
    const tokens = { ...readTokens(), palette: ["#111111", "#222222"], surface: "#ffffff", ink2: "#555555", ink3: "#777777", ink1: "#000000" };
    const metric = byUser.metrics[0] as unknown as ExperimentMetricResult;
    const intervals = intervalsFigure(metric)(tokens);
    const [line, point] = intervals.data as { x: number[]; line?: { color: string }; text?: string[] }[];
    expect(line?.line?.color).toBe("#111111");
    expect(line?.x[0]).toBeCloseTo((metric.comparisons[0]?.diff_low ?? 0) * 100);
    expect(point?.text?.[0]).toBe("+5.22 pts (+26.4 %)");

    const curve = sampleSizeFigure(plan as unknown as SampleSize)(tokens);
    const marker = curve.data[1] as { x: number[]; y: number[] };
    expect(marker.x[0]).toBe(57_756);
    expect(marker.y[0]).toBeCloseTo(5);
  });
});
