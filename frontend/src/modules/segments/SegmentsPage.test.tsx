import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { forwardRef } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { column, roles } from "../../test/columns";
import { dataset, health, mockApi } from "../../test/mockApi";
import { renderAt } from "../../test/render";
import { plan, run, summary, unusualRun, unusualSummary } from "../../test/segments";

vi.mock("../../charts/Plot", () => ({
  Plot: forwardRef<HTMLDivElement, { label: string }>(function Plot({ label }, ref) {
    return <div ref={ref} role="img" aria-label={label} />;
  }),
}));

const BASE = "/modules/segments/datasets/d1";

function setup(runs: unknown[] = [summary], unusual: unknown[] = [unusualSummary]) {
  return mockApi({
    "GET /health": health,
    "GET /datasets": [dataset()],
    "GET /datasets/d1": dataset(),
    "GET /datasets/d1/columns": [column({ name: "income" })],
    "GET /column-roles": roles,
    [`POST ${BASE}/plan`]: plan,
    [`GET ${BASE}/runs`]: runs,
    [`POST ${BASE}/runs`]: { id: "job1", kind: "segments", status: "running", stage: "loading", progress: null, message: null, run_id: null, error: null },
    "GET /modules/segments/jobs/job1": { id: "job1", kind: "segments", status: "done", stage: "saving", progress: null, message: null, run_id: run.id, error: null },
    [`GET ${BASE}/runs/${run.id}`]: run,
    [`PATCH ${BASE}/runs/${run.id}/names`]: (body: unknown) => {
      const names = (body as { names: Record<string, string> }).names;
      return { ...run, result: { ...run.result, segments: run.result.segments.map((s) => ({ ...s, name: names[String(s.id)] ?? s.name })) } };
    },
    [`POST ${BASE}/runs/${run.id}/dataset`]: { dataset_id: "d2" },
    [`GET ${BASE}/unusual`]: unusual,
    [`POST ${BASE}/unusual`]: { id: "job2", kind: "unusual", status: "running", stage: "loading", progress: null, message: null, run_id: null, error: null },
    "GET /modules/segments/jobs/job2": { id: "job2", kind: "unusual", status: "done", stage: "saving", progress: null, message: null, run_id: unusualRun.id, error: null },
    [`GET ${BASE}/unusual/${unusualRun.id}`]: unusualRun,
    [`GET ${BASE}/unusual/${unusualRun.id}/rows/10`]: [
      { column: "orders", type: "numeric_discrete", value: 1, percentile: 0.001, share: null, median: 15 },
      { column: "customer_id", type: "identifier", value: "C1", percentile: null, share: null, median: null },
    ],
    "PUT /modules/segments/datasets/d1/marks": [{ key: "customer_id=C1", mark: "expected", created_at: "2026-11-07T10:06:00Z" }],
  });
}

afterEach(() => vi.unstubAllGlobals());

describe("Segments page", () => {
  it("segments on its own the first time, with the columns left out and why", async () => {
    const calls = setup([]);
    renderAt("/d/d1/segments");

    expect(await screen.findByRole("heading", { name: "Segments", level: 1 })).toBeInTheDocument();
    expect(await screen.findByText("2 columns used, 2 left out")).toBeInTheDocument();
    expect(screen.getByText("customer_id")).toBeInTheDocument();
    expect(document.body).toHaveTextContent("customer_id excluded: Identifier (300 distinct values).");
    await waitFor(() => expect(calls.some((c) => c.key === `POST ${BASE}/runs`)).toBe(true));
    expect(calls.find((c) => c.key === `POST ${BASE}/runs`)?.body).toEqual({});
    expect(screen.getByRole("combobox", { name: "Method" })).toHaveValue("auto");
    expect(screen.getByRole("option", { name: "Auto (k-means)" })).toBeInTheDocument();
  });

  it("shows the chosen k, stability, portraits in original units and the profile table", async () => {
    setup();
    renderAt("/d/d1/segments");

    expect(await screen.findByRole("heading", { name: "3 segments: best silhouette, and the elbow is at 3." })).toBeInTheDocument();
    expect(screen.getByText("Stable on 10 subsamples")).toBeInTheDocument();
    expect(screen.getByText("income: 2.1× the average")).toBeInTheDocument();
    expect(screen.getByRole("textbox", { name: "Name of segment 1" })).toHaveValue("High income · Low age");
    expect(screen.getAllByText(/These rules describe 91 %/)).toHaveLength(3);
    const table = screen.getByRole("table", { name: "Profiles" });
    expect(within(table).getByText("52,000")).toBeInTheDocument();
    expect(within(table).getAllByText("(strong gap)")).toHaveLength(2);
    expect(screen.getByRole("img", { name: "Map" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Export profiles (CSV)" })).toHaveAttribute(
      "href",
      `/api/v1${BASE}/runs/${run.id}/profiles.csv`,
    );
  });

  it("renames a segment, saves the segments as a dataset, forces k with the slider", async () => {
    const calls = setup();
    renderAt("/d/d1/segments");

    const name = await screen.findByRole("textbox", { name: "Name of segment 1" });
    fireEvent.change(name, { target: { value: "Wealthy young" } });
    fireEvent.blur(name);
    await waitFor(() => expect(calls.find((c) => c.key === `PATCH ${BASE}/runs/${run.id}/names`)?.body).toEqual({ names: { "0": "Wealthy young" } }));

    fireEvent.click(screen.getByRole("button", { name: "Save as dataset" }));
    expect(await screen.findByRole("link", { name: "open it" })).toHaveAttribute("href", "/d/d2");

    fireEvent.click(screen.getByRole("radio", { name: "Set" }));
    const slider = screen.getByRole("slider", { name: "Number of segments" });
    fireEvent.change(slider, { target: { value: "5" } });
    fireEvent.pointerUp(slider);
    await waitFor(() => expect(calls.find((c) => c.key === `POST ${BASE}/runs`)?.body).toMatchObject({ k: 5, method: "auto" }));
  });
});

describe("Unusual rows section", () => {
  it("moves the threshold without scoring again, explains a row and marks it", async () => {
    const calls = setup();
    renderAt("/d/d1/segments?section=unusual");

    expect(await screen.findByRole("heading", { name: "2 unusual rows of 2,000" })).toBeInTheDocument();
    expect(screen.getByText("1 row marked expected hidden")).toBeInTheDocument();
    const table = screen.getByRole("table", { name: "Unusual rows" });
    expect(within(table).getAllByRole("row")).toHaveLength(3); // header + 2 rows
    expect(within(table).getByText(/orders = 0 while revenue = 3,012/)).toBeInTheDocument();

    const before = calls.length;
    fireEvent.change(screen.getByRole("slider", { name: "Threshold" }), { target: { value: "0.46" } });
    expect(await screen.findByRole("heading", { name: "3 unusual rows of 2,000" })).toBeInTheDocument();
    expect(within(screen.getByRole("table", { name: "Unusual rows" })).getAllByRole("row")).toHaveLength(4);
    expect(calls.slice(before).some((c) => c.key.startsWith("POST"))).toBe(false);
    expect(screen.getByRole("link", { name: "Export rows (CSV)" }).getAttribute("href")).toContain("threshold=0.46");

    fireEvent.click(screen.getByRole("button", { name: "Open row 10" }));
    expect(await screen.findByText("0.1 % of rows are at or below (median 15)")).toBeInTheDocument();
    fireEvent.keyDown(document.activeElement ?? document.body, { key: "Escape" });

    fireEvent.click(within(screen.getByRole("table", { name: "Unusual rows" })).getAllByRole("button", { name: "Expected" })[0]!);
    await waitFor(() =>
      expect(calls.find((c) => c.key === "PUT /modules/segments/datasets/d1/marks")?.body).toEqual({ key: "customer_id=C1", mark: "expected" }),
    );
  });

  it("finds unusual rows on its own the first time", async () => {
    const calls = setup([summary], []);
    renderAt("/d/d1/segments?section=unusual");

    await waitFor(() => expect(calls.some((c) => c.key === `POST ${BASE}/unusual`)).toBe(true));
    expect(await screen.findByRole("heading", { name: "2 unusual rows of 2,000" })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "Distance to its segment" })).toBeEnabled();
  });
});
