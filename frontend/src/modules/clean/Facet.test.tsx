import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { column, roles } from "../../test/columns";
import { dataset, health, mockApi } from "../../test/mockApi";
import { renderAt } from "../../test/render";

afterEach(() => vi.unstubAllGlobals());

const BASE = "/modules/clean";
const facet = (body: unknown) => {
  const picked = (body as { selection?: { values?: string[] } | null }).selection?.values ?? [];
  return {
    column: "content_rating",
    kind: "text",
    dtype: "VARCHAR",
    rows: 4932,
    distinct: 3,
    blank: 12,
    unreadable: null,
    unreadable_values: [],
    reading: null,
    values: [
      { value: "R", rows: 2100 },
      { value: "Not Rated", rows: 30 },
      { value: "Unrated", rows: 62 },
    ],
    listed_from: 3,
    low: null,
    high: null,
    bins: [],
    matched: picked.length ? 62 : null,
    shown: ["content_rating", "title"],
    sample: picked.length ? [{ row: 9, values: ["Unrated", "Old film"] }] : [],
  };
};

function setup() {
  return mockApi({
    "GET /health": health,
    "GET /datasets": [dataset()],
    "GET /datasets/d1": dataset(),
    "GET /datasets/d1/columns": [column({ name: "content_rating" }), column({ name: "title" })],
    "GET /column-roles": roles,
    [`GET ${BASE}/lineage`]: [],
    [`GET ${BASE}/settings`]: { mode: "suggest" },
    [`GET ${BASE}/datasets/d1/suggestions`]: { dataset_id: "d1", rows: 4932, seconds: 0.2, suggestions: [], quality: null },
    [`GET ${BASE}/datasets/d1/draft`]: { dataset_id: "d1", steps: [], automatic: true, updated_at: null },
    [`PUT ${BASE}/datasets/d1/draft`]: (body: unknown) => ({ dataset_id: "d1", steps: (body as { steps: unknown[] }).steps, automatic: false, updated_at: "now" }),
    [`POST ${BASE}/datasets/d1/preview`]: { rows_before: 4932, rows_after: 4932, rows_removed: 0, cells_changed: 0, columns: [], added: [], dropped: [], shown: [], sample: [], problems: {} },
    [`POST ${BASE}/datasets/d1/facet`]: facet,
    [`GET ${BASE}/datasets/d1/versions`]: { dataset_id: "d1", original_id: "d1", versions: [] },
  });
}

const puts = (calls: ReturnType<typeof setup>) => calls.filter((call) => call.key === `PUT ${BASE}/datasets/d1/draft`);

describe("Facets in Clean (N40)", () => {
  it("is folded under Steps and asks nothing before it opens", async () => {
    const calls = setup();
    renderAt("/d/d1/clean?tab=steps");
    expect(await screen.findByText("Facets")).toBeInTheDocument();
    expect(calls.some((call) => call.key.endsWith("/facet"))).toBe(false);
  });

  it("merges Unrated into Not Rated: values with their rows, the rows picked, then one step after a click", async () => {
    const calls = setup();
    renderAt("/d/d1/clean?tab=steps");
    fireEvent.click(await screen.findByText("Facets"));
    const list = await screen.findByRole("list", { name: "Values of content_rating" });
    expect(within(list).getByRole("listitem", { name: "Unrated" })).toHaveTextContent("62");
    expect(screen.getByText("(blank)")).toBeInTheDocument();
    fireEvent.click(within(list).getByRole("checkbox", { name: "Unrated" }));
    expect(await screen.findByText("62 of 4,932 rows of the draft picked")).toBeInTheDocument();
    expect(await screen.findByText("Old film")).toBeInTheDocument(); // the draft's rows, filtered
    const asked = calls.filter((call) => call.key.endsWith("/facet")).at(-1)!.body as { selection: { values: string[] } };
    expect(asked.selection.values).toEqual(["Unrated"]);
    fireEvent.click(screen.getByRole("button", { name: "Merge into…" }));
    fireEvent.change(screen.getByRole("combobox", { name: "Merge into" }), { target: { value: "Not Rated" } });
    expect(screen.getByText('Merge "Unrated" into "Not Rated" in content_rating: 62 rows change.')).toBeInTheDocument();
    expect(puts(calls)).toHaveLength(0); // nothing before the click
    fireEvent.click(screen.getByRole("button", { name: "Add this step" }));
    await waitFor(() => expect(puts(calls)).toHaveLength(1));
    const put = puts(calls)[0]!.body as { steps: Record<string, unknown>[] };
    expect(put.steps).toEqual([expect.objectContaining({ op: "map_values", column: "content_rating", map: { Unrated: "Not Rated" }, origin: "manual" })]);
    expect(screen.getByText(/Step s1 added to the draft\./)).toBeInTheDocument();
  });

  it("picks values with the keyboard: j / k move, Space picks", async () => {
    setup();
    renderAt("/d/d1/clean?tab=steps");
    fireEvent.click(await screen.findByText("Facets"));
    const list = await screen.findByRole("list", { name: "Values of content_rating" });
    fireEvent.keyDown(window, { key: "j" });
    expect(within(list).getByRole("listitem", { name: "R" })).toHaveFocus();
    fireEvent.keyDown(window, { key: "j" });
    const second = within(list).getByRole("listitem", { name: "Not Rated" });
    expect(second).toHaveFocus();
    fireEvent.keyDown(second, { key: " " });
    expect(within(list).getByRole("checkbox", { name: "Not Rated" })).toBeChecked();
    // on past the values, to the blank cells
    fireEvent.keyDown(window, { key: "j" });
    fireEvent.keyDown(window, { key: "j" });
    expect(screen.getByRole("listitem", { name: "(blank)" })).toHaveFocus();
  });
});
