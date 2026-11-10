import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { column, roles } from "../../test/columns";
import { dataset, health, mockApi } from "../../test/mockApi";
import { renderAt } from "../../test/render";

afterEach(() => vi.unstubAllGlobals());

const BASE = "/modules/clean";
const trim = { id: "s1", op: "normalize_spaces", columns: ["city"], enabled: true, origin: "auto", rule: "B-01", suggestion: "B-01:city" };
const unknown = { id: "s2", op: "null_values", column: "region", values: ["unknown"], match: "token", enabled: true, origin: "suggested", rule: "B-04", suggestion: "B-04:region:maybe" };
const suggestions = {
  dataset_id: "d1",
  rows: 1200,
  seconds: 0.4,
  suggestions: [
    { id: "B-01:city", rule: "B-01", layer: "basic", mode: "A", order: 300001, title: "Trim extra spaces in city", reason: "52 values had extra spaces.", columns: ["city"], cells: 52, rows: 0, examples: [{ before: " Paris", after: "Paris" }], step: trim },
    { id: "B-04:region:maybe", rule: "B-04", layer: "basic", mode: "S", order: 600002, title: "Read 'unknown' as missing in region", reason: "22 values are words for 'missing'.", columns: ["region"], cells: 22, rows: 0, examples: [], step: unknown },
  ],
};
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

function setup(lineage: unknown[] = []) {
  return mockApi({
    "GET /health": health,
    "GET /datasets": [dataset(), ...(lineage.length ? [dataset({ id: "d2", name: "sales · Cleaned v1" })] : [])],
    "GET /datasets/d1": dataset(),
    "GET /datasets/d2": dataset({ id: "d2", name: "sales · Cleaned v1" }),
    "GET /datasets/d1/columns": [column({ name: "city" }), column({ name: "region" })],
    "GET /column-roles": roles,
    [`GET ${BASE}/lineage`]: lineage,
    [`GET ${BASE}/settings`]: { mode: "suggest" },
    [`GET ${BASE}/datasets/d1/suggestions`]: suggestions,
    [`GET ${BASE}/datasets/d1/draft`]: { dataset_id: "d1", steps: [trim], automatic: true, updated_at: null },
    [`PUT ${BASE}/datasets/d1/draft`]: (body: unknown) => ({ dataset_id: "d1", steps: (body as { steps: unknown[] }).steps, automatic: false, updated_at: "now" }),
    [`POST ${BASE}/datasets/d1/preview`]: (body: unknown) => preview((body as { steps: unknown[] }).steps.length * 10),
    [`GET ${BASE}/datasets/d1/versions`]: { dataset_id: "d1", original_id: "d1", versions: [] },
    [`POST ${BASE}/datasets/d1/versions`]: job("running"),
    [`GET ${BASE}/jobs/job1`]: job("done"),
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

  it("shows the Basic suggestions with their reasons, the automatic ones ticked, the other layers folded", async () => {
    setup();
    renderAt("/d/d1/clean");
    const trimBox = await screen.findByRole("checkbox", { name: "Trim extra spaces in city" });
    expect(trimBox).toBeChecked();
    expect(screen.getByRole("checkbox", { name: "Read 'unknown' as missing in region" })).not.toBeChecked();
    expect(screen.getByText("52 values had extra spaces.")).toBeInTheDocument();
    const advanced = screen.getByRole("button", { name: /Advanced/ });
    expect(advanced).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByRole("button", { name: /More checks/ })).not.toBeInTheDocument(); // folded inside Advanced
    fireEvent.click(advanced);
    expect(screen.getByRole("button", { name: /More checks/ })).toHaveAttribute("aria-expanded", "false");
    expect(await screen.findByText(/10 cells changed/)).toBeInTheDocument();
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

  it("on a cleaned version, leads to the original", async () => {
    setup([{ dataset_id: "d2", parent_id: "d1", seq: 1, origin: "user", steps: 1, created_at: "2026-11-10T10:00:00Z", out_of_date: false }]);
    renderAt("/d/d2/clean");
    expect(await screen.findByText(/This is Cleaned v1, a cleaned version/)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Clean the original" })).toHaveAttribute("href", "/d/d1/clean");
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
