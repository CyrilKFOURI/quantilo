import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { forwardRef } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { column, roles } from "../../test/columns";
import { dataset, health, mockApi } from "../../test/mockApi";
import { renderAt } from "../../test/render";

vi.mock("../../charts/Plot", () => ({
  Plot: forwardRef<HTMLDivElement, { label: string }>(function Plot({ label }, ref) {
    return <div ref={ref} role="img" aria-label={label} />;
  }),
}));

const columns = [
  column({ name: "price", position: 0 }),
  column({ name: "country", position: 1, type: "categorical_nominal", detected_type: "categorical_nominal" }),
];

const overview = (rows: number) => ({
  rows,
  total_rows: 1200,
  columns: 2,
  used_columns: 2,
  missing_cells: 0,
  missing_pct: 0,
  duplicate_rows: 0,
  duplicate_columns: [],
  size_bytes: 1024,
  types: { numeric_continuous: 1, categorical_nominal: 1 },
  alerts: [],
  column_stats: {},
});

const countries = {
  kind: "bars",
  view: "bars",
  views: ["bars"],
  rows: 1200,
  missing: 0,
  distinct: 2,
  bars: [
    { label: "FR", count: 700 },
    { label: "BE", count: 500 },
  ],
  others: null,
  ordered: false,
  bins: [],
  stats: null,
  box: null,
  step: null,
  steps: [],
  examples: [],
  duplicates: 0,
  message: null,
};

function setup() {
  return mockApi({
    "GET /health": health,
    "GET /datasets/d1": dataset({ types_reviewed_at: "2026-10-13T10:00:00Z" }),
    "POST /datasets/d1/open": dataset(),
    "GET /datasets/d1/columns": columns,
    "GET /column-roles": roles,
    "GET /datasets/d1/overview": () => overview(700),
    "GET /datasets/d1/columns/chart": countries,
    "GET /datasets/d1/columns/summaries": {},
    "GET /datasets/d1/saved-filters": [],
  });
}

const FR = '[{"column":"country","op":"in","negate":false,"values":["FR"]}]';

beforeEach(() => localStorage.clear());
afterEach(() => vi.unstubAllGlobals());

describe("Global filters", () => {
  it("adds a category filter and applies it to every request", async () => {
    const calls = setup();
    renderAt("/d/d1");
    fireEvent.click(await screen.findByRole("button", { name: /^Filter/ }));

    const panel = await screen.findByRole("dialog", { name: "Filters" });
    fireEvent.change(within(panel).getByLabelText("Column"), { target: { value: "country" } });
    // a category is picked from its values (R-FLT-03): "is" first, a list of the data's values
    expect(within(panel).getByLabelText("Condition")).toHaveValue("is");
    await within(panel).findByRole("option", { name: "FR (700)" });
    fireEvent.change(within(panel).getByLabelText("Value"), { target: { value: "FR" } });
    fireEvent.click(within(panel).getByRole("button", { name: "Apply" }));

    expect(await within(panel).findByText("country ∈ FR")).toBeInTheDocument();
    await waitFor(() =>
      expect(calls.some((c) => c.key === "GET /datasets/d1/overview" && c.params.get("filters") === FR)).toBe(true),
    );
    expect(await screen.findAllByText("700 / 1,200 rows")).not.toHaveLength(0);
    expect(screen.getByRole("button", { name: /^Filter/ })).toHaveTextContent("1");
  });

  it("keeps one row per value of one column", async () => {
    const calls = setup();
    renderAt("/d/d1");
    fireEvent.click(await screen.findByRole("button", { name: /^Filter/ }));

    const panel = await screen.findByRole("dialog", { name: "Filters" });
    fireEvent.change(within(panel).getByLabelText("Column"), { target: { value: "country" } });
    fireEvent.change(within(panel).getByLabelText("Condition"), { target: { value: "unique" } });
    expect(within(panel).getByRole("radio", { name: "This column" })).toHaveAttribute("aria-checked", "true");
    fireEvent.click(within(panel).getByRole("button", { name: "Apply" }));

    expect(await within(panel).findByText("country · unique (first)")).toBeInTheDocument();
    await waitFor(() => {
      const sent = calls.find((c) => c.key === "GET /datasets/d1/overview" && c.params.get("filters"));
      expect(JSON.parse(sent?.params.get("filters") ?? "[]")[0]).toMatchObject({ op: "unique", columns: [] });
    });
  });

  it("keeps one row per value of a combination of columns", async () => {
    const calls = setup();
    renderAt("/d/d1");
    fireEvent.click(await screen.findByRole("button", { name: /^Filter/ }));

    const panel = await screen.findByRole("dialog", { name: "Filters" });
    fireEvent.change(within(panel).getByLabelText("Column"), { target: { value: "country" } });
    fireEvent.change(within(panel).getByLabelText("Condition"), { target: { value: "unique" } });
    fireEvent.click(within(panel).getByRole("radio", { name: "Several columns (key)" }));
    fireEvent.click(within(panel).getByRole("button", { name: "Apply" }));
    expect(await within(panel).findByText("Tick at least one more column for the key.")).toBeInTheDocument();
    fireEvent.click(within(within(panel).getByRole("list", { name: "Columns of the key" })).getByRole("checkbox", { name: "price" }));
    fireEvent.click(within(panel).getByRole("radio", { name: "Last row" }));
    fireEvent.change(within(panel).getByLabelText("Ordered by"), { target: { value: "price" } });
    fireEvent.click(within(panel).getByRole("button", { name: "Apply" }));

    expect(await within(panel).findByText("country + price · unique (last by price)")).toBeInTheDocument();
    await waitFor(() => {
      const sent = calls.find((c) => c.key === "GET /datasets/d1/overview" && c.params.get("filters"));
      expect(JSON.parse(sent?.params.get("filters") ?? "[]")).toEqual([
        { column: "country", op: "unique", negate: false, columns: ["price"], keep: "last", order_by: "price" },
      ]);
    });
  });

  it("keeps the filters when moving to another page", async () => {
    const calls = setup();
    const filters = encodeURIComponent(JSON.stringify({ dataset: "d1", conditions: JSON.parse(FR) }));
    const router = renderAt(`/d/d1?filters=${filters}`);
    await screen.findByRole("heading", { name: "Overview" });

    fireEvent.click(screen.getByRole("link", { name: "Columns" }));

    await waitFor(() => expect(router.state.location.pathname).toBe("/d/d1/columns"));
    await waitFor(() =>
      expect(calls.some((c) => c.key === "GET /datasets/d1/columns/summaries" && c.params.get("filters") === FR)).toBe(
        true,
      ),
    );
  });

  it("ignores filters that belong to another dataset", async () => {
    const calls = setup();
    const filters = encodeURIComponent(JSON.stringify({ dataset: "other", conditions: JSON.parse(FR) }));
    renderAt(`/d/d1?filters=${filters}`);
    await screen.findByRole("heading", { name: "Overview" });

    await waitFor(() => expect(calls.some((c) => c.key === "GET /datasets/d1/overview")).toBe(true));
    expect(calls.filter((c) => c.key === "GET /datasets/d1/overview").every((c) => !c.params.has("filters"))).toBe(true);
    expect(screen.getByRole("button", { name: /^Filter/ })).not.toHaveTextContent("1");
  });

  it("removes a filter and clears them all", async () => {
    setup();
    const filters = encodeURIComponent(JSON.stringify({ dataset: "d1", conditions: JSON.parse(FR) }));
    const router = renderAt(`/d/d1?filters=${filters}`);
    fireEvent.click(await screen.findByRole("button", { name: /^Filter/ }));

    const panel = await screen.findByRole("dialog", { name: "Filters" });
    fireEvent.click(within(panel).getByRole("button", { name: "Remove country ∈ FR" }));

    await waitFor(() => expect(router.state.location.search).toEqual({}));
  });
});

const guided = [
  column({
    name: "revenue",
    position: 0,
    summary: { ...column().summary, numeric: { min: 0, max: 1250, median: 48.5 } },
  }),
  column({
    name: "date",
    position: 1,
    type: "datetime",
    detected_type: "datetime",
    summary: {
      ...column().summary,
      date: { has_time: false, min: "2022-01-01 00:00:00", max: "2024-12-30 00:00:00", step: "day", parts: [] },
    },
  }),
  column({ name: "user_id", position: 2, type: "identifier", detected_type: "identifier" }),
  column({ name: "device", position: 3, type: "categorical_nominal", detected_type: "categorical_nominal" }),
  column({ name: "comment", position: 4, type: "text", detected_type: "text" }),
];

function guidedSetup() {
  return mockApi({
    "GET /health": health,
    "GET /datasets/d1": dataset({ types_reviewed_at: "2026-10-13T10:00:00Z" }),
    "POST /datasets/d1/open": dataset(),
    "GET /datasets/d1/columns": guided,
    "GET /column-roles": roles,
    "GET /datasets/d1/overview": () => overview(700),
    "GET /datasets/d1/columns/chart": countries,
    "GET /datasets/d1/columns/summaries": {},
    "GET /datasets/d1/saved-filters": [],
  });
}

async function openEditor(path = "/d/d1") {
  renderAt(path);
  fireEvent.click(await screen.findByRole("button", { name: /^Filter/ }));
  return screen.findByRole("dialog", { name: "Filters" });
}

const sentFilters = (calls: ReturnType<typeof mockApi>) =>
  JSON.parse(calls.find((c) => c.key === "GET /datasets/d1/overview" && c.params.get("filters"))?.params.get("filters") ?? "[]");

const offered = (panel: HTMLElement) =>
  [...within(panel).getByLabelText("Condition").querySelectorAll("option")].map((o) => o.textContent);

describe("Filters guided by the data (100)", () => {
  it("shows a number's range and warns when a threshold keeps no row", async () => {
    guidedSetup();
    const panel = await openEditor();
    fireEvent.change(within(panel).getByLabelText("Condition"), { target: { value: "gt" } });

    expect(within(panel).getByText("In the data: 0 – 1,250 · median 48.5")).toBeInTheDocument();
    expect(within(panel).getByLabelText("Value")).toHaveAttribute("placeholder", "e.g. 48.5");
    fireEvent.change(within(panel).getByLabelText("Value"), { target: { value: "5000" } });
    expect(within(panel).getByRole("status")).toHaveTextContent("this filter keeps no row");
  });

  it("refuses a range whose start is after its end", async () => {
    const calls = guidedSetup();
    const panel = await openEditor();
    fireEvent.change(within(panel).getByLabelText("Condition"), { target: { value: "between" } });
    expect(within(panel).getByLabelText("From")).toHaveAttribute("placeholder", "0");
    fireEvent.change(within(panel).getByLabelText("From"), { target: { value: "10" } });
    fireEvent.change(within(panel).getByLabelText("To"), { target: { value: "5" } });
    fireEvent.click(within(panel).getByRole("button", { name: "Apply" }));

    expect(await within(panel).findByText("From must not be after To.")).toBeInTheDocument();
    expect(calls.some((c) => c.key === "GET /datasets/d1/overview" && c.params.get("filters"))).toBe(false);
  });

  it("opens a date inside the data, filled with its first and last dates", async () => {
    guidedSetup();
    const panel = await openEditor();
    fireEvent.change(within(panel).getByLabelText("Column"), { target: { value: "date" } });
    fireEvent.change(within(panel).getByLabelText("Condition"), { target: { value: "between" } });

    const from = within(panel).getByLabelText("From");
    expect(from).toHaveValue("2022-01-01");
    expect(from).toHaveAttribute("min", "2022-01-01");
    expect(within(panel).getByLabelText("To")).toHaveValue("2024-12-30");
    expect(within(panel).getByText("Data runs from 2022-01-01 to 2024-12-30.")).toBeInTheDocument();

    fireEvent.change(within(panel).getByLabelText("Condition"), { target: { value: "lt" } });
    expect(within(panel).getByLabelText("Value")).toHaveValue("2024-12-30");
  });

  it("keeps the last N days of the data, anchored on its last date", async () => {
    const calls = guidedSetup();
    const panel = await openEditor();
    fireEvent.change(within(panel).getByLabelText("Column"), { target: { value: "date" } });
    fireEvent.change(within(panel).getByLabelText("Condition"), { target: { value: "last_days" } });
    fireEvent.change(within(panel).getByLabelText("Number of days"), { target: { value: "7" } });
    expect(within(panel).getByText("From 2024-12-24 to 2024-12-30, the last date in the data.")).toBeInTheDocument();
    fireEvent.click(within(panel).getByRole("button", { name: "Apply" }));

    expect(await within(panel).findByText("date · last 7 days (to 2024-12-30)")).toBeInTheDocument();
    await waitFor(() =>
      expect(sentFilters(calls)).toEqual([{ column: "date", op: "last_days", negate: false, value: 7, high: "2024-12-30" }]),
    );
  });

  it("takes a pasted list of identifiers", async () => {
    const calls = guidedSetup();
    const panel = await openEditor();
    fireEvent.change(within(panel).getByLabelText("Column"), { target: { value: "user_id" } });
    expect(offered(panel)).not.toContain("contains");
    fireEvent.change(within(panel).getByLabelText("Condition"), { target: { value: "in" } });
    fireEvent.change(within(panel).getByLabelText("Values, one per line"), { target: { value: "A1\nA2, A3;A1" } });
    expect(within(panel).getByText(/3 values\./)).toBeInTheDocument();
    fireEvent.click(within(panel).getByRole("button", { name: "Apply" }));

    await waitFor(() => expect(sentFilters(calls)[0]).toMatchObject({ op: "in", values: ["A1", "A2", "A3"] }));
  });

  it("offers each type its own conditions", async () => {
    guidedSetup();
    const panel = await openEditor();
    fireEvent.change(within(panel).getByLabelText("Column"), { target: { value: "device" } });
    expect(offered(panel).slice(0, 6)).toEqual(["is", "is not", "is any of", "is none of", "is missing", "is not missing"]);
    fireEvent.change(within(panel).getByLabelText("Column"), { target: { value: "comment" } });
    expect(offered(panel).slice(0, 5)).toEqual(["contains", "doesn't contain", "starts with", "is missing", "is not missing"]);
  });

  it("opens older conditions as they were saved", async () => {
    guidedSetup();
    const conditions = [
      { column: "device", op: "eq", value: "mobile", negate: false },
      { column: "device", op: "contains", value: "mob", negate: false },
    ];
    const filters = encodeURIComponent(JSON.stringify({ dataset: "d1", conditions }));
    const panel = await openEditor(`/d/d1?filters=${filters}`);

    fireEvent.click(within(panel).getByRole("button", { name: "Edit device = mobile" }));
    expect(within(panel).getByLabelText("Condition")).toHaveValue("is");
    await waitFor(() => expect(within(panel).getByLabelText("Value")).toHaveValue("mobile"));
    fireEvent.click(within(panel).getByRole("button", { name: "Cancel" }));

    fireEvent.click(within(panel).getByRole("button", { name: 'Edit device contains "mob"' }));
    expect(offered(panel)).toContain("contains (older condition)");
    expect(within(panel).getByLabelText("Value")).toHaveValue("mob");
  });
});
