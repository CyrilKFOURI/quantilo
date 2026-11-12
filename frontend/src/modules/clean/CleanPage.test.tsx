import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { column, roles } from "../../test/columns";
import { dataset, health, mockApi } from "../../test/mockApi";
import { renderAt } from "../../test/render";

afterEach(() => vi.unstubAllGlobals());

const BASE = "/modules/clean";
const trim = { id: "s1", op: "normalize_spaces", columns: ["city"], enabled: true, origin: "auto", rule: "B-01", suggestion: "B-01:city" };
const unknown = { id: "s2", op: "null_values", column: "region", values: ["unknown"], match: "token", enabled: true, origin: "suggested", rule: "B-04", suggestion: "B-04:region:maybe" };
const na = { id: "s3", op: "null_values", column: "status", values: ["n/a"], match: "token", enabled: true, origin: "auto", rule: "B-04", suggestion: "B-04:status" };
const merge = { id: "s4", op: "map_values", column: "city", map: { PARIS: "Paris" }, enabled: true, origin: "auto", rule: "B-03", suggestion: "B-03:city" };
const basic = [
  { id: "B-01:city", rule: "B-01", layer: "basic", mode: "A", impact: "format", risk: "safe", order: 300001, title: "Trim extra spaces in city", reason: "52 values had extra spaces.", columns: ["city"], cells: 52, rows: 0, examples: [{ before: " Paris", after: "Paris" }], step: trim },
  { id: "B-04:region:maybe", rule: "B-04", layer: "basic", mode: "S", impact: "values", risk: "review", order: 600002, title: "Read 'unknown' as missing in region", reason: "22 values are words for 'missing'. Check they are not a real category.", columns: ["region"], cells: 22, rows: 0, examples: [], step: unknown },
];
const suggestions = { dataset_id: "d1", rows: 1200, seconds: 0.4, suggestions: basic };
// Two value changes, not ticked yet, and nothing of the Advanced layer.
const values = [
  { id: "B-04:status", rule: "B-04", layer: "basic", mode: "A", impact: "values", risk: "safe", order: 600003, title: "Read 'n/a' as missing in status", reason: "3 values are words for 'missing'.", columns: ["status"], cells: 3, rows: 0, examples: [{ before: "n/a", after: null }], step: na },
  { id: "B-03:city", rule: "B-03", layer: "basic", mode: "A", impact: "values", risk: "safe", order: 700001, title: "Merge 2 spellings of 'Paris'", reason: "'PARIS' → 'Paris'.", columns: ["city"], cells: 4, rows: 0, examples: [{ before: "PARIS", after: "Paris" }], step: merge },
];
const preview = (cells: number) => ({
  rows_before: 1200,
  rows_after: 1200,
  rows_removed: 0,
  cells_changed: cells,
  columns: [{ column: "city", before: "city", cells, dtype_before: "VARCHAR", dtype_after: "VARCHAR" }],
  added: [],
  dropped: [],
  shown: ["city"],
  sample: [{ row: 3, removed: false, before: [" Paris"], after: ["Paris"], changed: [true] }],
  problems: {},
});
const job = (status: string) => ({ id: "job1", status, stage: status === "done" ? "done" : "cleaning", dataset_id: "d1", progress: null, created_dataset_id: status === "done" ? "d2" : null, error: null });

function setup(lineage: unknown[] = [], found: unknown[] = basic) {
  const versions = lineage.map((item) => ({ ...(item as object), name: "sales · Cleaned v1", status: "ready", steps: [], report: null, source_id: null, source_seq: null }));
  return mockApi({
    "GET /health": health,
    "GET /datasets": [dataset(), ...(lineage.length ? [dataset({ id: "d2", name: "sales · Cleaned v1" })] : [])],
    "GET /datasets/d1": dataset(),
    "GET /datasets/d2": dataset({ id: "d2", name: "sales · Cleaned v1" }),
    "GET /datasets/d1/columns": [column({ name: "city" }), column({ name: "region" })],
    "GET /column-roles": roles,
    [`GET ${BASE}/lineage`]: lineage,
    [`GET ${BASE}/settings`]: { mode: "suggest" },
    [`GET ${BASE}/datasets/d1/suggestions`]: { ...suggestions, suggestions: found },
    [`GET ${BASE}/datasets/d1/draft`]: { dataset_id: "d1", steps: [trim], automatic: true, updated_at: null },
    [`PUT ${BASE}/datasets/d1/draft`]: (body: unknown) => ({ dataset_id: "d1", steps: (body as { steps: unknown[] }).steps, automatic: false, updated_at: "now" }),
    [`POST ${BASE}/datasets/d1/preview`]: (body: unknown) => preview((body as { steps: unknown[] }).steps.length * 10),
    [`GET ${BASE}/datasets/d1/versions`]: { dataset_id: "d1", original_id: "d1", versions: [] },
    [`POST ${BASE}/datasets/d1/versions`]: job("running"),
    [`GET ${BASE}/jobs/job1`]: job("done"),
    // a cleaned version, cleaned again (096)
    "GET /datasets/d2/columns": [column({ name: "city" })],
    [`GET ${BASE}/datasets/d2/suggestions`]: { ...suggestions, dataset_id: "d2", suggestions: [] },
    [`GET ${BASE}/datasets/d2/draft`]: { dataset_id: "d2", steps: [], automatic: true, updated_at: null },
    [`POST ${BASE}/datasets/d2/preview`]: preview(0),
    [`GET ${BASE}/datasets/d2/versions`]: { dataset_id: "d2", original_id: "d1", versions },
    [`POST ${BASE}/datasets/d2/versions`]: job("running"),
  });
}

const puts = (calls: ReturnType<typeof setup>) => calls.filter((call) => call.key === `PUT ${BASE}/datasets/d1/draft`);

describe("Clean page (065)", () => {
  it("is an entry of the Dataset section, under Overview", async () => {
    setup();
    renderAt("/d/d1/clean");
    const nav = await screen.findByRole("navigation", { name: "Main navigation" });
    const links = within(nav).getAllByRole("link").map((link) => link.textContent);
    expect(links.indexOf("Clean")).toBe(links.indexOf("Overview") + 1);
    expect(within(nav).getByRole("link", { name: "Clean" })).toHaveAttribute("href", "/d/d1/clean");
  });

  it("groups the suggestions by their effect, the safe ones ticked, the decisions not, with their reason (096)", async () => {
    setup();
    renderAt("/d/d1/clean");
    const trimBox = await screen.findByRole("checkbox", { name: "Trim extra spaces in city" });
    expect(trimBox).toBeChecked();
    expect(screen.getByRole("checkbox", { name: "Read 'unknown' as missing in region" })).not.toBeChecked();
    expect(screen.getByText("2 fixes")).toBeInTheDocument();
    expect(screen.getByText(/1 safe formatting · 1 needs your decision/)).toBeInTheDocument();
    const headings = screen.getAllByRole("heading", { level: 3 }).map((heading) => heading.textContent);
    expect(headings).toEqual(["Safe formatting1", "Needs your decision1"]); // empty groups are not rendered
    const decision = screen.getByRole("region", { name: /Needs your decision/ });
    expect(within(decision).getAllByText(/Check they are not a real category/)).toHaveLength(2); // the reason, written and in the tip
    expect(within(decision).getByText("0 of 1 ticked")).toBeInTheDocument();
    expect(screen.getAllByText(/52 values had extra spaces/).length).toBeGreaterThan(0); // behind the tip
    expect(screen.getByText("·Paris → Paris")).toBeInTheDocument();
    // nothing in the Advanced layer: it is not rendered (CL-05)
    expect(screen.queryByRole("button", { name: /Advanced/ })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /More checks/ })).not.toBeInTheDocument();
    expect(await screen.findByText(/10 cells changed/)).toBeInTheDocument();
    for (const move of ["Review", "Apply", "Save version"]) expect(screen.getByText(move)).toBeInTheDocument();
    expect(screen.getByText("2 fixes found, grouped by what they change in the data.")).toBeInTheDocument();
  });

  it("ticks a whole group in one change, and unticks it (096)", async () => {
    const calls = setup([], [...basic, ...values]);
    renderAt("/d/d1/clean");
    const group = await screen.findByRole("region", { name: /^Values/ });
    expect(within(group).getByText("0 of 2 ticked")).toBeInTheDocument();
    fireEvent.click(within(group).getByRole("button", { name: "Tick all in group: Values" }));
    await waitFor(() => expect(puts(calls)).toHaveLength(1));
    const steps = (puts(calls)[0]!.body as { steps: { suggestion: string }[] }).steps.map((step) => step.suggestion);
    expect(steps).toEqual(["B-01:city", "B-04:status", "B-03:city"]); // at their place (C-06)
    expect(within(group).getByText("2 of 2 ticked")).toBeInTheDocument();
    fireEvent.click(within(group).getByRole("button", { name: "Untick group: Values" }));
    await waitFor(() => expect(puts(calls)).toHaveLength(2));
    expect((puts(calls)[1]!.body as { steps: unknown[] }).steps).toHaveLength(1);
  });

  it("shows the Advanced layer only when it has something, More only inside it (096)", async () => {
    const advanced = { ...values[0]!, id: "A-01:status", rule: "A-01", layer: "advanced", title: "Cluster 3 spellings in status" };
    const expert = { ...values[1]!, id: "E-01:city", rule: "E-01", layer: "expert", title: "Suspicious label in city" };
    setup([], [...basic, advanced, expert]);
    renderAt("/d/d1/clean");
    const layer = await screen.findByRole("button", { name: /Advanced/ });
    expect(layer).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByText("Cluster 3 spellings in status")).not.toBeInTheDocument();
    fireEvent.click(layer);
    expect(screen.getByText("Cluster 3 spellings in status")).toBeInTheDocument();
    const more = screen.getByRole("button", { name: /More checks/ });
    expect(more).toHaveAttribute("aria-expanded", "false");
    fireEvent.click(more);
    expect(screen.getByText("Suspicious label in city")).toBeInTheDocument();
  });

  it("ticks a suggestion into the draft, saves the draft, undoes it", async () => {
    const calls = setup();
    renderAt("/d/d1/clean");
    fireEvent.click(await screen.findByRole("checkbox", { name: "Read 'unknown' as missing in region" }));
    await waitFor(() => expect(puts(calls)).toHaveLength(1));
    expect((puts(calls)[0]!.body as { steps: { id: string }[] }).steps.map((step) => step.id)).toEqual(["s1", "s2"]);
    expect(await screen.findByText(/20 cells changed/)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Undo" }));
    await waitFor(() => expect(puts(calls)).toHaveLength(2));
    expect((puts(calls)[1]!.body as { steps: unknown[] }).steps).toHaveLength(1);
    expect(screen.getByRole("checkbox", { name: "Read 'unknown' as missing in region" })).not.toBeChecked();
  });

  it("adds a step from the catalogue and shows what it changes", async () => {
    const calls = setup();
    renderAt("/d/d1/clean?tab=steps");
    expect(await screen.findByText("Trim extra spaces in city")).toBeInTheDocument(); // its suggestion's sentence
    expect(await screen.findByText("The first 1 rows touched, before and after")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Add step" }));
    const dialog = await screen.findByRole("dialog", { name: "Add a step" });
    expect(within(dialog).getByRole("textbox", { name: "Search steps" })).not.toHaveFocus(); // no keyboard popping up (096)
    fireEvent.change(within(dialog).getByRole("textbox", { name: "Search steps" }), { target: { value: "rename" } });
    fireEvent.click(within(dialog).getByRole("button", { name: /Rename column/ }));
    await waitFor(() => expect(puts(calls)).toHaveLength(1));
    const steps = (puts(calls)[0]!.body as { steps: { op: string; origin: string }[] }).steps;
    expect(steps.at(-1)).toMatchObject({ op: "rename_column", origin: "manual" });
    expect(screen.getByRole("textbox", { name: "New name" })).toHaveValue("city 2");
  });

  it("saves the draft as Cleaned v1", async () => {
    const calls = setup();
    renderAt("/d/d1/clean");
    fireEvent.click(await screen.findByRole("button", { name: "Save as Cleaned v1" }));
    expect(await screen.findByText("Cleaned v1 saved.")).toBeInTheDocument();
    expect(calls.some((call) => call.key === `POST ${BASE}/datasets/d1/versions`)).toBe(true);
    expect(screen.getByRole("link", { name: "Open" })).toHaveAttribute("href", "/d/d2");
  });

  it("on a cleaned version, cleans that version and saves the next one (096)", async () => {
    const calls = setup([{ dataset_id: "d2", parent_id: "d1", seq: 1, origin: "user", steps: 1, created_at: "2026-11-10T10:00:00Z", out_of_date: false, source_id: null, source_seq: null }]);
    renderAt("/d/d2/clean");
    expect(await screen.findByText("Cleaning Cleaned v1")).toBeInTheDocument();
    expect(await screen.findByText("will save as Cleaned v2")).toBeInTheDocument();
    expect(await screen.findByText("Nothing to fix here: no rule found a problem.")).toBeInTheDocument();
    expect(calls.some((call) => call.key === `GET ${BASE}/datasets/d2/suggestions`)).toBe(true); // computed on v1
    expect(screen.getByRole("button", { name: "Save as Cleaned v2" })).toBeDisabled(); // nothing ticked yet
    const family = await screen.findByRole("combobox", { name: "Version of the dataset" });
    expect(within(family).getAllByRole("option").map((option) => option.textContent)).toEqual(["Original", "Cleaned v1"]);
  });

  it("groups the versions under their original on the home page", async () => {
    setup([{ dataset_id: "d2", parent_id: "d1", seq: 1, origin: "auto", steps: 1, created_at: "2026-11-10T10:00:00Z", out_of_date: false }]);
    renderAt("/");
    expect(await screen.findByText("Cleaned v1 · auto · latest")).toBeInTheDocument();
    const names = screen.getAllByRole("link").map((link) => link.textContent).filter((text) => text?.startsWith("sales"));
    expect(names).toEqual(["sales", "sales · Cleaned v1"]);
    fireEvent.click(screen.getAllByRole("button", { name: "Delete" })[0]!);
    expect(await screen.findByRole("checkbox", { name: "Also delete its cleaned version" })).not.toBeChecked();
  });
});
