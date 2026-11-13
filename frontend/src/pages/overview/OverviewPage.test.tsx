import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { forwardRef } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { column, roles } from "../../test/columns";
import { dataset, health, mockApi } from "../../test/mockApi";
import { renderAt } from "../../test/render";
import { categoricalProfile } from "../../test/variable";

// jsdom can't draw Plotly: render a labelled placeholder instead.
vi.mock("../../charts/Plot", () => ({
  Plot: forwardRef<HTMLDivElement, { label: string; height: number }>(function Plot({ label }, ref) {
    return <div ref={ref} role="img" aria-label={label} />;
  }),
}));

const columns = [
  column({ name: "price", position: 0 }),
  column({ name: "country", position: 1, type: "categorical_nominal", detected_type: "categorical_nominal" }),
  column({ name: "signup", position: 2, type: "datetime", detected_type: "datetime" }),
  column({ name: "hidden", position: 3, excluded: true }),
];

const overview = {
  rows: 1200,
  columns: 4,
  used_columns: 3,
  missing_cells: 40,
  missing_pct: 1.11,
  duplicate_rows: 3,
  duplicate_columns: [],
  size_bytes: 20480,
  types: { numeric_continuous: 1, categorical_nominal: 1, datetime: 1 },
  alerts: [
    { kind: "duplicate_rows", severity: "warning", message: "3 duplicate rows (0.25%)", columns: [] },
    { kind: "skewed", severity: "info", message: "1 strongly skewed numeric column", columns: ["price"] },
  ],
  column_stats: {
    price: { missing_pct: 0, distinct: 900, skewness: 3.1 },
    country: { missing_pct: 2, distinct: 8, skewness: null },
    signup: { missing_pct: 0, distinct: 300, skewness: null },
  },
};

const bars = (overrides: Record<string, unknown> = {}) => ({
  kind: "bars",
  view: "bars",
  views: ["bars"],
  rows: 1200,
  missing: 10,
  distinct: 8,
  bars: [
    { label: "FR", count: 500 },
    { label: "BE", count: 300 },
  ],
  others: { count: 390, categories: 6 },
  ordered: false,
  bins: [],
  stats: null,
  box: null,
  step: null,
  steps: [],
  examples: [],
  duplicates: 0,
  message: null,
  ...overrides,
});

function setup() {
  return mockApi({
    "GET /health": health,
    "GET /datasets/d1": dataset({ types_reviewed_at: "2026-10-13T10:00:00Z" }),
    "POST /datasets/d1/open": dataset(),
    "GET /datasets/d1/columns": columns,
    "GET /datasets/d1/overview": overview,
    "GET /column-roles": roles,
    "GET /datasets/d1/columns/chart": bars(),
    "GET /datasets/d1/variable": { ...categoricalProfile, name: "country" },
  });
}

const chartCalls = (calls: { key: string }[]) => calls.filter((call) => call.key === "GET /datasets/d1/columns/chart");

beforeEach(() => {
  localStorage.clear();
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("Overview page", () => {
  it("shows the summary strip, alerts and one card per used column", async () => {
    setup();
    renderAt("/d/d1");

    expect(await screen.findByRole("heading", { name: "Overview" })).toBeInTheDocument();
    const summary = await screen.findByRole("region", { name: "Dataset summary" });
    expect(within(summary).getByText("1,200")).toBeInTheDocument();
    expect(within(summary).getByText("1.11%")).toBeInTheDocument();
    expect(screen.getByText("3 duplicate rows (0.25%)")).toBeInTheDocument();
    expect(screen.getAllByRole("article")).toHaveLength(3); // the excluded column has no card
  });

  it("loads each card's chart and draws it", async () => {
    const calls = setup();
    renderAt("/d/d1");

    expect(await screen.findByRole("img", { name: "Distribution of country" })).toBeInTheDocument();
    expect(chartCalls(calls)).toHaveLength(3);
  });

  it("changes Top N for every chart and remembers it", async () => {
    const calls = setup();
    renderAt("/d/d1");
    await screen.findByRole("img", { name: "Distribution of country" });

    fireEvent.change(screen.getByLabelText("Top"), { target: { value: "20" } });

    await waitFor(() => expect(chartCalls(calls).length).toBeGreaterThan(3));
    expect(localStorage.getItem("de.overview.d1.topN")).toBe("20");
  });

  it("exports the HTML report and downloads it once done", async () => {
    const job = { id: "j1", status: "running", progress: 0.5, message: "Variables 2 / 3", error: null, file_name: null };
    let polls = 0;
    const calls = mockApi({
      "GET /health": health,
      "GET /datasets/d1": dataset({ types_reviewed_at: "2026-10-13T10:00:00Z" }),
      "POST /datasets/d1/open": dataset(),
      "GET /datasets/d1/columns": columns,
      "GET /datasets/d1/overview": overview,
      "GET /column-roles": roles,
      "GET /datasets/d1/columns/chart": bars(),
      "POST /datasets/d1/report": job,
      "GET /reports/jobs/j1": () =>
        ++polls < 2 ? job : { ...job, status: "done", progress: 1, message: null, file_name: "sales-report.html" },
    });
    const click = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});
    renderAt("/d/d1");

    fireEvent.click(await screen.findByRole("button", { name: "Export report" }));

    expect(await screen.findByText("Report · Variables 2 / 3")).toBeInTheDocument();
    await waitFor(() => expect(click).toHaveBeenCalledTimes(1), { timeout: 3000 });
    expect(calls.some((c) => c.key === "POST /datasets/d1/report")).toBe(true);
    expect(screen.getByRole("button", { name: "Export report" })).toBeEnabled();
    click.mockRestore();
  });

  it("opens the report in a print preview without downloading it (025, R-REP-17)", async () => {
    const job = { id: "j2", status: "running", progress: 0.5, message: "Variables 2 / 3", error: null, file_name: null };
    let polls = 0;
    mockApi({
      "GET /health": health,
      "GET /datasets/d1": dataset({ types_reviewed_at: "2026-10-13T10:00:00Z" }),
      "POST /datasets/d1/open": dataset(),
      "GET /datasets/d1/columns": columns,
      "GET /datasets/d1/overview": overview,
      "GET /column-roles": roles,
      "GET /datasets/d1/columns/chart": bars(),
      "POST /datasets/d1/report": job,
      "GET /reports/jobs/j2": () =>
        ++polls < 2 ? job : { ...job, status: "done", progress: 1, message: null, file_name: "sales-report.html" },
      "GET /reports/jobs/j2/file": () => new Response("<!doctype html><title>r</title><h1>Data report</h1>", { headers: { "content-type": "text/html" } }),
    });
    const click = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});
    renderAt("/d/d1");

    fireEvent.click(await screen.findByRole("button", { name: "Print / PDF" }));
    const dialog = await screen.findByRole("dialog", { name: "Report, print preview" });
    const frame = await within(dialog).findByTitle("Report, print preview", {}, { timeout: 3000 });
    expect(frame.getAttribute("sandbox")).toBe("allow-same-origin allow-modals");
    expect(frame.getAttribute("srcdoc")).toContain("Data report");
    expect(within(dialog).getByRole("button", { name: "Print or save as PDF" })).toBeEnabled();
    expect(click).not.toHaveBeenCalled();
    click.mockRestore();
  });

  it("focuses a column from an alert", async () => {
    setup();
    renderAt("/d/d1");
    await screen.findByText("1 strongly skewed numeric column");

    fireEvent.click(screen.getByRole("button", { name: "price" }));

    expect(screen.getByLabelText("Search variables")).toHaveValue("price");
    expect(screen.getAllByRole("article")).toHaveLength(1);
  });

  it("paginates the grid", async () => {
    setup();
    renderAt("/d/d1");
    await screen.findByRole("heading", { name: "Overview" });

    fireEvent.change(screen.getByLabelText("Per page"), { target: { value: "6" } });

    expect(screen.getAllByText("1–3 of 3")).toHaveLength(1);
  });

  it("opens a category in full with its table", async () => {
    setup();
    renderAt("/d/d1");
    await screen.findByRole("img", { name: "Distribution of country" });

    fireEvent.click(screen.getByRole("button", { name: "Show country in full" }));

    const dialog = await screen.findByRole("dialog");
    expect(await within(dialog).findByText("1,200 rows · 8 distinct values")).toBeInTheDocument();
    expect(await within(dialog).findByRole("cell", { name: "FR" })).toBeInTheDocument();
    expect(await within(dialog).findByText("0.52 bits")).toBeInTheDocument();
    expect(within(dialog).getByRole("link", { name: "Open variable page" })).toHaveAttribute(
      "href",
      "/d/d1/variable/country",
    );
  });

  it("links each card to its variable page", async () => {
    setup();
    renderAt("/d/d1");

    const links = await screen.findAllByRole("link", { name: "Open variable details" });
    expect(links[0]).toHaveAttribute("href", "/d/d1/variable/price");
  });
});
