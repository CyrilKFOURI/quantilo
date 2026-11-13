import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { column, roles } from "../../test/columns";
import { dataset, health, mockApi } from "../../test/mockApi";
import { renderAt } from "../../test/render";

afterEach(() => vi.unstubAllGlobals());

const BASE = "/modules/clean";
const cluster = (anchor: string, ...names: string[]) => names.map((name) => ({ anchor, name }));
const entity = {
  id: "s0",
  op: "entity_column",
  column: "owner",
  anchor: "zip",
  anchor_kind: "postal_code",
  to: "owner_entity",
  groups: [cluster("10007", "nyc housing authority", "n.y.c. housing authority", "nyc housing authorty"), cluster("10019", "vornado office management", "vornado office management, llc")],
  not_entities: null,
  split: null,
  enabled: true,
  origin: "suggested",
  rule: "A-20",
  suggestion: "A-20:owner:entity",
};
const a20 = {
  id: "A-20:owner:entity",
  rule: "A-20",
  layer: "advanced",
  mode: "S",
  impact: "columns",
  risk: "review",
  order: 2_470_000,
  title: "Number the 2 companies written several ways (new column owner_entity)",
  reason: "2 companies are written 2 to 3 ways with the same zip… Check your data.",
  columns: ["owner"],
  cells: 0,
  rows: 9,
  examples: [],
  step: entity,
};
const records = (anchor: string, spellings: [string, string, number, number][]) =>
  spellings.map(([name, spelling, rows, first], index) => ({ index, anchor, name, spelling, rows, first, cells: [spelling, anchor, `row ${first + 1}`] }));
const clusters = {
  step_id: "s0",
  column: "owner",
  anchor: "zip",
  to: "owner_entity",
  columns: ["owner", "zip", "note"],
  total_clusters: 2,
  total_records: 5,
  total_rows: 9,
  offset: 0,
  clusters: [
    {
      index: 0,
      rows: 6,
      records: records("10007", [
        ["nyc housing authority", "NYC HOUSING AUTHORITY", 4, 0],
        ["n.y.c. housing authority", "N.Y.C. HOUSING AUTHORITY", 1, 3],
        ["nyc housing authorty", "NYC HOUSING AUTHORTY", 1, 7],
      ]),
      differing: [0, 2],
    },
    {
      index: 1,
      rows: 3,
      records: records("10019", [
        ["vornado office management", "VORNADO OFFICE MANAGEMENT", 2, 1],
        ["vornado office management, llc", "VORNADO OFFICE MANAGEMENT, LLC", 1, 5],
      ]),
      differing: [0, 2],
    },
  ],
};
const suggestions = { dataset_id: "d1", rows: 10, seconds: 0.1, suggestions: [] };

function setup() {
  return mockApi({
    "GET /health": health,
    "GET /datasets": [dataset()],
    "GET /datasets/d1": dataset(),
    "GET /datasets/d1/columns": [column({ name: "owner" }), column({ name: "zip" }), column({ name: "note" })],
    "GET /column-roles": roles,
    [`GET ${BASE}/lineage`]: [],
    [`GET ${BASE}/settings`]: { mode: "suggest" },
    [`GET ${BASE}/datasets/d1/suggestions`]: suggestions,
    [`GET ${BASE}/datasets/d1/suggestions/advanced`]: { ...suggestions, suggestions: [a20] },
    [`GET ${BASE}/datasets/d1/draft`]: { dataset_id: "d1", steps: [], automatic: true, updated_at: null },
    [`PUT ${BASE}/datasets/d1/draft`]: (body: unknown) => ({ dataset_id: "d1", steps: (body as { steps: unknown[] }).steps, automatic: false, updated_at: "now" }),
    [`POST ${BASE}/datasets/d1/preview`]: { rows_before: 10, rows_after: 10, rows_removed: 0, cells_changed: 0, columns: [], added: [], dropped: [], shown: [], sample: [], problems: {} },
    [`POST ${BASE}/datasets/d1/entities`]: clusters,
    [`GET ${BASE}/datasets/d1/versions`]: { dataset_id: "d1", original_id: "d1", versions: [] },
  });
}

const puts = (calls: ReturnType<typeof setup>) => calls.filter((call) => call.key === `PUT ${BASE}/datasets/d1/draft`);
type Body = { steps: { suggestion: string | null; not_entities?: number[] | null; split?: Record<string, number[]> | null }[] };
const lastStep = (calls: ReturnType<typeof setup>) => (puts(calls).at(-1)!.body as Body).steps.at(-1)!;

describe("Review of entity clusters (N36)", () => {
  it("lists each cluster's spellings side by side and writes what the review decides", async () => {
    const calls = setup();
    renderAt("/d/d1/clean");
    fireEvent.click(await screen.findByRole("button", { name: /Advanced/ }));
    const fix = await screen.findByRole("checkbox", { name: a20.title });
    expect(fix).not.toBeChecked();
    fireEvent.click(screen.getByRole("button", { name: "Review the clusters" }));
    const review = screen.getByRole("region", { name: "Review of the entities, cluster by cluster" });
    expect(await within(review).findByText("2 clusters · 9 rows · 2 entity numbers")).toBeInTheDocument();
    const asked = calls.find((call) => call.key === `POST ${BASE}/datasets/d1/entities`)!.body as { steps: { suggestion: string | null }[] };
    expect(asked.steps.map((step) => step.suggestion)).toEqual(["A-20:owner:entity"]);
    expect(within(review).getAllByText("NYC HOUSING AUTHORTY")).toHaveLength(2); // its name, and its cell
    expect(within(review).getByText("3 spellings, 6 rows: one number")).toBeInTheDocument();
    expect(within(review).getByText("4 rows, first row 1")).toBeInTheDocument();
    expect(within(review).getByText("1 row, row 8")).toBeInTheDocument(); // the singular

    // a spelling split off: another entity, the fix ticked
    fireEvent.click(within(review).getByRole("checkbox", { name: "Split 'NYC HOUSING AUTHORTY' off cluster 1" }));
    await waitFor(() => expect(puts(calls)).toHaveLength(1));
    expect(lastStep(calls)).toMatchObject({ suggestion: "A-20:owner:entity", split: { "0": [2] }, not_entities: null });
    expect(fix).toBeChecked();
    expect(within(review).getByText("1 of 3 spellings split off: two numbers")).toBeInTheDocument();

    // a cluster that is not one entity: each spelling its own number, the fix partly kept
    fireEvent.click(within(review).getByRole("checkbox", { name: "Cluster 2: its spellings are one entity" }));
    await waitFor(() => expect(puts(calls)).toHaveLength(2));
    expect(lastStep(calls)).toMatchObject({ not_entities: [1], split: { "0": [2] } });
    expect(fix).toBePartiallyChecked();
    expect(within(review).getByText("2 clusters · 9 rows · 4 entity numbers · 1 kept apart · 1 split")).toBeInTheDocument();
    expect(screen.getByText("partly kept")).toBeInTheDocument();

    // the whole fix again, as found
    fireEvent.click(fix);
    await waitFor(() => expect(puts(calls)).toHaveLength(3));
    expect(lastStep(calls)).toMatchObject({ not_entities: null, split: null });
  });
});
