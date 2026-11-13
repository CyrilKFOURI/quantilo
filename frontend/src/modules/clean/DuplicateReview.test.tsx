import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { column, roles } from "../../test/columns";
import { dataset, health, mockApi } from "../../test/mockApi";
import { renderAt } from "../../test/render";

afterEach(() => vi.unstubAllGlobals());

const BASE = "/modules/clean";
const trim = { id: "s1", op: "normalize_spaces", columns: ["city"], enabled: true, origin: "auto", rule: "B-01", suggestion: "B-01:city" };
const dedupe = { id: "s0", op: "drop_duplicates", keys: ["id"], keep: "latest", order_by: "updated", groups: null, survivors: null, not_duplicates: null, merge: false, enabled: true, origin: "suggested", rule: "B-23", suggestion: "B-23:id" };
const b23 = { id: "B-23:id", rule: "B-23", layer: "basic", mode: "S", impact: "rows", risk: "review", order: 161001, title: "Keep one row per id (3 rows with differences)", reason: "2 id values repeat… Check your data.", columns: ["id"], cells: 0, rows: 3, examples: [], step: dedupe };
const groups = {
  step_id: "s2",
  columns: ["id", "city", "updated"],
  dates: ["updated"],
  keep: "latest",
  order_by: "updated",
  total_groups: 2,
  total_rows: 5,
  removed: 3,
  offset: 0,
  groups: [
    { rows: [{ row: 0, filled: 3, cells: ["A1", "Paris", "2024-01-01"] }, { row: 1, filled: 2, cells: ["A1", null, "2024-03-01"] }], size: 2, order: [1, 0], first: 0, most_complete: 0, latest: { updated: 1 }, differing: [1, 2] },
    { rows: [{ row: 3, filled: 1, cells: ["C3", null, null] }, { row: 4, filled: 3, cells: ["C3", "Nice", "2024-05-01"] }, { row: 5, filled: 3, cells: ["C3", "Nice", "2024-04-01"] }], size: 3, order: [4, 5, 3], first: 3, most_complete: 4, latest: { updated: 4 }, differing: [1, 2] },
  ],
};

function setup() {
  return mockApi({
    "GET /health": health,
    "GET /datasets": [dataset()],
    "GET /datasets/d1": dataset(),
    "GET /datasets/d1/columns": [column({ name: "id" }), column({ name: "city" }), column({ name: "updated" })],
    "GET /column-roles": roles,
    [`GET ${BASE}/lineage`]: [],
    [`GET ${BASE}/settings`]: { mode: "suggest" },
    [`GET ${BASE}/datasets/d1/suggestions`]: { dataset_id: "d1", rows: 7, seconds: 0.1, suggestions: [b23] },
    [`GET ${BASE}/datasets/d1/draft`]: { dataset_id: "d1", steps: [trim], automatic: true, updated_at: null },
    [`PUT ${BASE}/datasets/d1/draft`]: (body: unknown) => ({ dataset_id: "d1", steps: (body as { steps: unknown[] }).steps, automatic: false, updated_at: "now" }),
    [`POST ${BASE}/datasets/d1/preview`]: { rows_before: 7, rows_after: 7, rows_removed: 0, cells_changed: 0, columns: [], added: [], dropped: [], shown: [], sample: [], problems: {} },
    [`POST ${BASE}/datasets/d1/duplicates`]: groups,
    [`GET ${BASE}/datasets/d1/versions`]: { dataset_id: "d1", original_id: "d1", versions: [] },
    [`GET ${BASE}/datasets/d1/suggestions/advanced`]: { dataset_id: "d1", rows: 7, seconds: 0.1, suggestions: [] },
  });
}

const puts = (calls: ReturnType<typeof setup>) => calls.filter((call) => call.key === `PUT ${BASE}/datasets/d1/draft`);
type Body = { steps: { suggestion: string | null; survivors?: number[] | null; not_duplicates?: number[][] | null; merge?: boolean }[] };
const lastStep = (calls: ReturnType<typeof setup>) => (puts(calls).at(-1)!.body as Body).steps.at(-1)!;

describe("Review of duplicates by group (N35)", () => {
  it("lists each group side by side, the row kept marked, and writes what the review decides", async () => {
    const calls = setup();
    renderAt("/d/d1/clean");
    const fix = await screen.findByRole("checkbox", { name: "Keep one row per id (3 rows with differences)" });
    expect(fix).not.toBeChecked();
    fireEvent.click(screen.getByRole("button", { name: "Review the groups" }));
    const review = screen.getByRole("region", { name: "Review of the duplicates, group by group" });
    expect(await within(review).findByText("2 groups · rows removed: 0 of 5")).toBeInTheDocument();
    // the groups of the rule, as the draft leaves the rows before it
    const asked = calls.find((call) => call.key === `POST ${BASE}/datasets/d1/duplicates`)!.body as { steps: { suggestion: string | null }[]; step_id: string };
    expect(asked.steps.map((step) => step.suggestion)).toEqual(["B-01:city", "B-23:id"]);
    expect(within(review).getByRole("radio", { name: "Keep row 2 in group 1" })).toBeChecked(); // the latest
    expect(within(review).getAllByText("empty").length).toBeGreaterThan(0);

    // another row kept, by hand: the step names it and is ticked
    fireEvent.click(within(review).getByRole("radio", { name: "Keep row 1 in group 1" }));
    await waitFor(() => expect(puts(calls)).toHaveLength(1));
    expect(lastStep(calls)).toMatchObject({ suggestion: "B-23:id", survivors: [0] });
    expect(fix).toBeChecked();
    expect(within(review).getByRole("combobox", { name: "Row kept in group 1" })).toHaveValue("first");

    // a group that is not duplicates: all its rows stay, the fix is partly kept
    fireEvent.click(within(review).getByRole("checkbox", { name: "Group 2: its rows are duplicates" }));
    await waitFor(() => expect(puts(calls)).toHaveLength(2));
    expect(lastStep(calls)).toMatchObject({ survivors: [0], not_duplicates: [[3, 4, 5]] });
    expect(fix).toBePartiallyChecked();
    expect(within(review).getByText("2 groups · rows removed: 1 of 5 · 1 kept whole · 1 with another row kept")).toBeInTheDocument();
    expect(screen.getByText("partly kept")).toBeInTheDocument();

    // merge only when asked: the kept row's empty city comes from its group, said so
    fireEvent.click(within(review).getByRole("checkbox", { name: "Fill the kept row's empty cells from its group" }));
    await waitFor(() => expect(puts(calls)).toHaveLength(3));
    expect(lastStep(calls).merge).toBe(true);
    // the rule's choice again, from the list: no row named
    fireEvent.change(within(review).getByRole("combobox", { name: "Row kept in group 1" }), { target: { value: "rule" } });
    await waitFor(() => expect(puts(calls)).toHaveLength(4));
    expect(lastStep(calls).survivors).toBeNull();
    // the whole fix again, as found
    fireEvent.click(fix);
    await waitFor(() => expect(puts(calls)).toHaveLength(5));
    expect(lastStep(calls)).toMatchObject({ survivors: null, not_duplicates: null, merge: false });
  });
});
