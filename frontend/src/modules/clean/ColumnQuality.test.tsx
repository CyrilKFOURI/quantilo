import { fireEvent, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { column, roles } from "../../test/columns";
import { dataset, health, mockApi } from "../../test/mockApi";
import { renderAt } from "../../test/render";

afterEach(() => vi.unstubAllGlobals());

const BASE = "/modules/clean";
const step = { id: "s1", op: "null_values", column: "amount", values: ["n.c."], match: "token", enabled: true, origin: "suggested", rule: "B-04", suggestion: "B-04:amount" };
const found = [
  { id: "B-04:amount", rule: "B-04", layer: "basic", mode: "S", impact: "values", risk: "review", order: 600001, title: "Read 'n.c.' as missing in amount", reason: "10 values are words for 'missing'.", columns: ["amount"], cells: 10, rows: 0, examples: [], step },
];
const quality = [
  { column: "amount", rows: 100, missing: 20, unreadable: 5, unreadable_values: [{ value: "12 kg", rows: 3 }, { value: "about 5", rows: 2 }], reading: "number" },
  { column: "city", rows: 100, missing: 0, unreadable: 0, unreadable_values: [], reading: null },
];

function setup(steps: unknown[] = []) {
  return mockApi({
    "GET /health": health,
    "GET /datasets": [dataset()],
    "GET /datasets/d1": dataset(),
    "GET /datasets/d1/columns": [column({ name: "amount" }), column({ name: "city" })],
    "GET /column-roles": roles,
    [`GET ${BASE}/lineage`]: [],
    [`GET ${BASE}/settings`]: { mode: "suggest" },
    [`GET ${BASE}/datasets/d1/suggestions`]: { dataset_id: "d1", rows: 100, seconds: 0.2, suggestions: found, quality },
    [`GET ${BASE}/datasets/d1/draft`]: { dataset_id: "d1", steps, automatic: steps.length === 0, updated_at: null },
    [`PUT ${BASE}/datasets/d1/draft`]: (body: unknown) => ({ dataset_id: "d1", steps: (body as { steps: unknown[] }).steps, automatic: false, updated_at: "now" }),
    [`POST ${BASE}/datasets/d1/preview`]: { rows_before: 100, rows_after: 100, rows_removed: 0, cells_changed: 0, columns: [], added: [], dropped: [], shown: [], sample: [], problems: {} },
    [`GET ${BASE}/datasets/d1/versions`]: { dataset_id: "d1", original_id: "d1", versions: [] },
  });
}

describe("Column quality bars (N37)", () => {
  it("shows a bar per column at the head of By column, with its parts named and a table view", async () => {
    setup();
    renderAt("/d/d1/clean");
    fireEvent.click(await screen.findByRole("radio", { name: "By column" }));
    const overview = screen.getByRole("region", { name: "Column quality" });
    expect(within(overview).getByText("2 columns · 1 with cells to fix, empty or unreadable")).toBeInTheDocument();
    // never colour alone: the bar is named with every part
    expect(within(overview).getByRole("img", { name: "amount: 65% valid, 10% to fix, 20% empty, 5% unreadable" })).toBeInTheDocument();
    // the section of the column carries its bar too
    expect(screen.getAllByRole("img", { name: /^amount:/ })).toHaveLength(2);
    // a line opens the values not read and the column's fixes
    fireEvent.click(within(overview).getByRole("button", { name: "Details of amount" }));
    expect(within(overview).getByText("Values not read as numbers, most rows first:")).toBeInTheDocument();
    expect(within(overview).getByText("12 kg")).toBeInTheDocument();
    expect(within(overview).getByRole("button", { name: "Show its 1 fix" })).toBeInTheDocument();
    // the same numbers as a table
    fireEvent.click(within(overview).getByRole("radio", { name: "Table" }));
    const table = within(overview).getByRole("table");
    const row = within(table).getByRole("row", { name: /amount/ });
    expect(within(row).getAllByRole("cell").map((cell) => cell.textContent)).toEqual(["65 (65%)", "10 (10%)", "20 (20%)", "5 (5%)"]);
  });

  it("follows the draft: ticking the fix moves its cells out of 'to fix'", async () => {
    setup();
    renderAt("/d/d1/clean");
    fireEvent.click(await screen.findByRole("radio", { name: "By column" }));
    fireEvent.click(screen.getByRole("checkbox", { name: "Read 'n.c.' as missing in amount" }));
    const overview = screen.getByRole("region", { name: "Column quality" });
    expect(await within(overview).findByRole("img", { name: "amount: 75% valid, 0% to fix, 20% empty, 5% unreadable" })).toBeInTheDocument();
  });
});
