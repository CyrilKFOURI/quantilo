import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { column, roles } from "../../test/columns";
import { dataset, health, mockApi } from "../../test/mockApi";
import { renderAt } from "../../test/render";

const columns = [
  column({ name: "price", position: 0 }),
  column({
    name: "store",
    position: 1,
    type: "categorical_nominal",
    detected_type: "categorical_nominal",
    role: null,
    role_label: null,
    detected_role: null,
    confidence: "medium",
    reasons: ["3 distinct values"],
  }),
  column({
    name: "zip",
    position: 2,
    type: "numeric_discrete",
    detected_type: "categorical_nominal",
    overridden: true,
    summary: { ...column().summary, invalid_count: 4 },
  }),
];

type Change = { name: string; [key: string]: unknown };

function setup() {
  const calls = mockApi({
    "GET /health": health,
    "GET /datasets/d1": dataset({ types_reviewed_at: null }),
    "GET /datasets/d1/columns": columns,
    "GET /column-roles": roles,
    "PATCH /datasets/d1/columns": (body: unknown) => {
      const { changes } = body as { changes: Change[] };
      return columns.map((c) => {
        const change = changes.find((item) => item.name === c.name);
        return change ? { ...c, ...change, overridden: true } : c;
      });
    },
    "POST /datasets/d1/columns/review": dataset({ types_reviewed_at: "2026-10-13T10:00:00Z" }),
    "POST /datasets/d1/open": dataset(),
    "GET /datasets/d1/overview": {
      rows: 100, columns: 3, used_columns: 3, missing_cells: 0, missing_pct: 0, duplicate_rows: 0,
      duplicate_columns: [], size_bytes: 100, types: {}, alerts: [], column_stats: {},
    },
  });
  renderAt("/d/d1/columns");
  return calls;
}

const patches = (calls: { key: string; body: unknown }[]) =>
  calls.filter((call) => call.key === "PATCH /datasets/d1/columns").map((call) => call.body);

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("Columns page", () => {
  it("lists columns, least confident first, with the review banner", async () => {
    setup();

    await screen.findByRole("heading", { name: "Columns" });
    const rows = screen.getAllByRole("row").slice(1);
    expect(within(rows[0] as HTMLElement).getByText("store")).toBeInTheDocument();
    expect(screen.getByText("3 columns · 1 to review")).toBeInTheDocument();
    expect(screen.getByText(/Check the columns marked/)).toBeInTheDocument();
    expect(screen.getByText(/4 values \(4%\) can't be read/)).toBeInTheDocument();
  });

  it("changes a role, then undoes it", async () => {
    const calls = setup();
    await screen.findByRole("heading", { name: "Columns" });

    fireEvent.pointerDown(screen.getByRole("button", { name: "Role of store" }), { button: 0, pointerType: "mouse" });
    fireEvent.click(await screen.findByRole("menuitemradio", { name: "City" }));
    await waitFor(() => expect(patches(calls)).toHaveLength(1));
    expect(patches(calls)[0]).toEqual({ changes: [{ name: "store", role: "city" }] });

    await waitFor(() => expect(screen.getByRole("button", { name: "Undo" })).toBeEnabled());
    fireEvent.click(screen.getByRole("button", { name: "Undo" }));
    await waitFor(() => expect(patches(calls)).toHaveLength(2));
    expect(patches(calls)[1]).toEqual({ changes: [{ name: "store", role: "" }] });
  });

  it("changes a type from the type menu", async () => {
    const calls = setup();
    await screen.findByRole("heading", { name: "Columns" });
    const storeRow = screen.getByText("store").closest("tr") as HTMLElement;

    const trigger = within(storeRow).getByRole("button", { name: "Change type" });
    fireEvent.pointerDown(trigger, { button: 0, ctrlKey: false, pointerType: "mouse" });
    fireEvent.click(await screen.findByRole("menuitemradio", { name: /Ordered category/ }));

    await waitFor(() => expect(patches(calls)).toHaveLength(1));
    expect(patches(calls)[0]).toEqual({ changes: [{ name: "store", type: "categorical_ordinal" }] });
  });

  it("excludes several columns at once", async () => {
    const calls = setup();
    await screen.findByRole("heading", { name: "Columns" });

    fireEvent.click(screen.getByLabelText("Select price"));
    fireEvent.click(screen.getByLabelText("Select store"));
    fireEvent.click(screen.getByRole("button", { name: "Exclude" }));

    await waitFor(() => expect(patches(calls)).toHaveLength(1));
    expect(patches(calls)[0]).toEqual({
      changes: [
        { name: "price", excluded: true },
        { name: "store", excluded: true },
      ],
    });
  });

  it("filters to the columns to review and by type", async () => {
    setup();
    await screen.findByRole("heading", { name: "Columns" });

    fireEvent.click(screen.getByRole("radio", { name: "To review 1" }));
    expect(screen.queryByText("price")).not.toBeInTheDocument();
    expect(screen.getByText("store")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("radio", { name: "All 3" }));
    fireEvent.click(screen.getByRole("button", { name: /Discrete/ }));
    expect(screen.getByText("zip")).toBeInTheDocument();
    expect(screen.queryByText("store")).not.toBeInTheDocument();
  });

  it("validates the types and goes to the dataset", async () => {
    const calls = setup();
    await screen.findByRole("heading", { name: "Columns" });

    fireEvent.click(screen.getByRole("button", { name: "Validate types" }));

    await waitFor(() =>
      expect(calls.some((call) => call.key === "POST /datasets/d1/columns/review")).toBe(true),
    );
    expect(await screen.findByRole("heading", { name: "Overview" })).toBeInTheDocument();
  });

  it("tells free text, short text and address apart, with the language (050)", async () => {
    const text = (name: string, role: string, language: Record<string, unknown>) =>
      column({
        name,
        type: "text",
        detected_type: "text",
        role,
        detected_role: role,
        reasons: [`reasons of ${name}`],
        summary: { ...column().summary, text: { median_words: 4, ...language } },
      });
    mockApi({
      "GET /health": health,
      "GET /datasets/d1": dataset(),
      "GET /column-roles": roles,
      "GET /datasets/d1/columns": [
        text("review", "free_text", { language: "fr", language_label: "French", language_share: 0.9, median_words: 14 }),
        text("item", "label", { language: null, language_label: null, language_share: null }),
        text("ship_to", "address", { language: "mixed", language_label: "Mixed", language_share: 0.6 }),
      ],
    });
    renderAt("/d/d1/columns");

    const row = async (name: string) => (await screen.findByText(name)).closest("tr") as HTMLElement;
    const badge = (tr: HTMLElement) => within(tr).getByRole("button", { name: "Change type" });
    expect(badge(await row("review"))).toHaveTextContent("Free text");
    expect(within(await row("review")).getByText("French · median 14 words")).toBeInTheDocument();
    expect(badge(await row("item"))).toHaveTextContent("Short text");
    expect(within(await row("item")).getByText("Median 4 words")).toBeInTheDocument();
    expect(badge(await row("ship_to"))).toHaveTextContent("Address");
    expect(within(await row("ship_to")).getByText("Mixed languages · median 4 words")).toBeInTheDocument();
  });

  it("lists the columns a reading rule changed, with their reason (092)", async () => {
    mockApi({
      "GET /health": health,
      "GET /datasets/d1": dataset({ types_reviewed_at: null }),
      "GET /datasets/d1/columns": [
        ...columns,
        column({
          name: "dep",
          position: 3,
          type: "categorical_nominal",
          detected_type: "categorical_nominal",
          reasons: ["1% of values, such as 2A, 2B, cannot be read as numbers: kept as text so that no value is lost (R-TYP-01)"],
        }),
      ],
      "GET /column-roles": roles,
    });
    renderAt("/d/d1/columns");

    const notes = await screen.findByRole("region", { name: "Columns read by a rule" });
    expect(within(notes).getByText(/^dep: 1% of values, such as 2A/)).toBeInTheDocument();
    expect(within(notes).getByText(/Check that your data was read correctly/)).toBeInTheDocument();
  });
});

