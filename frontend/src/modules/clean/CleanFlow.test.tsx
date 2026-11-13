import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { column, roles } from "../../test/columns";
import { dataset, health, mockApi } from "../../test/mockApi";
import { renderAt } from "../../test/render";

/** Audit of 2026-11-12: N05 (don't suggest again), N31 (declared checks), N45 (the cleaning as SQL
 *  and Python), N46 (the cleaning journal). */

afterEach(() => vi.unstubAllGlobals());

const BASE = "/modules/clean";
const trim = { id: "s1", op: "normalize_spaces", columns: ["city"], enabled: true, origin: "auto", rule: "B-01", suggestion: "B-01:city" };
const trimBrand = { ...trim, id: "s2", columns: ["brand"], suggestion: "B-01:brand" };
const fix = (id: string, columns: string[], step: unknown, extra: Record<string, unknown> = {}) => ({
  id,
  rule: id.split(":")[0],
  layer: "basic",
  mode: "A",
  impact: "format",
  risk: "safe",
  order: 300001,
  title: `Trim extra spaces in ${columns[0]}`,
  reason: "Values had extra spaces.",
  columns,
  cells: 5,
  rows: 0,
  examples: [],
  step,
  ...extra,
});
const city = fix("B-01:city", ["city"], trim);
const brand = fix("B-01:brand", ["brand"], trimBrand);
const empty = fix("B-19:notes", ["notes"], null, { title: "Remove the empty column notes", impact: "columns" });
const ruleOff = { id: "x1", dataset_id: "d1", scope: "rule", rule: "B-19", column: null, suggestion: null, title: "Remove the empty column notes", note: "always empty", created_at: "2026-11-13T10:00:00Z", reason: "You turned off rule B-19 for this dataset. always empty" };
const preview = { rows_before: 30, rows_after: 30, rows_removed: 0, cells_changed: 5, columns: [], added: [], dropped: [], shown: [], sample: [], problems: {} };

function setup(extra: Record<string, unknown> = {}) {
  return mockApi({
    "GET /health": health,
    "GET /datasets": [dataset()],
    "GET /datasets/d1": dataset(),
    "GET /datasets/d1/columns": [column({ name: "city" }), column({ name: "brand" })],
    "GET /column-roles": roles,
    [`GET ${BASE}/lineage`]: [],
    [`GET ${BASE}/settings`]: { mode: "suggest" },
    [`GET ${BASE}/datasets/d1/suggestions`]: { dataset_id: "d1", rows: 30, seconds: 0.1, suggestions: [city, brand], hidden: [{ suggestion: empty, dismissal: "x1" }], dismissals: [ruleOff] },
    [`GET ${BASE}/datasets/d1/draft`]: { dataset_id: "d1", steps: [trim, trimBrand], automatic: true, updated_at: null },
    [`PUT ${BASE}/datasets/d1/draft`]: (body: unknown) => ({ dataset_id: "d1", steps: (body as { steps: unknown[] }).steps, automatic: false, updated_at: "now" }),
    [`POST ${BASE}/datasets/d1/preview`]: preview,
    [`GET ${BASE}/datasets/d1/versions`]: { dataset_id: "d1", original_id: "d1", versions: [] },
    [`POST ${BASE}/datasets/d1/dismissed`]: (body: unknown) => ({ dismissal: { ...ruleOff, id: "x2", ...(body as object) }, hidden: ["B-01:city", "B-01:brand"] }),
    [`DELETE ${BASE}/dismissed/x1`]: new Response(null, { status: 204 }),
    ...extra,
  });
}

const puts = (calls: ReturnType<typeof setup>) => calls.filter((call) => call.key === `PUT ${BASE}/datasets/d1/draft`);

describe("Don't suggest again (N05)", () => {
  it("dismisses a rule here: its steps leave the draft in one change", async () => {
    const calls = setup();
    renderAt("/d/d1/clean");
    fireEvent.click(await screen.findByRole("button", { name: "Don't suggest again: Trim extra spaces in city" }));
    const dialog = await screen.findByRole("dialog", { name: "Don't suggest this again" });
    fireEvent.click(within(dialog).getByRole("radio", { name: "Rule B-01 on every column of this dataset" }));
    fireEvent.change(within(dialog).getByRole("textbox", { name: /Why/ }), { target: { value: "spaces matter" } });
    fireEvent.click(within(dialog).getByRole("button", { name: "Don't suggest again" }));
    await waitFor(() => expect(puts(calls)).toHaveLength(1));
    const posted = calls.find((call) => call.key === `POST ${BASE}/datasets/d1/dismissed`)!.body;
    expect(posted).toMatchObject({ scope: "rule", rule: "B-01", column: "city", suggestion: "B-01:city", note: "spaces matter" });
    expect((puts(calls)[0]!.body as { steps: unknown[] }).steps).toEqual([]); // both trims gone
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
  });

  it("lists what is dismissed, folded last, and suggests it again", async () => {
    const calls = setup();
    renderAt("/d/d1/clean");
    const fold = await screen.findByRole("button", { name: /^Dismissed/ });
    expect(fold).toHaveTextContent("1 hidden");
    expect(fold).toHaveAttribute("aria-expanded", "false");
    fireEvent.click(fold);
    expect(screen.getByText("Remove the empty column notes")).toBeInTheDocument();
    expect(screen.getByText(/rule turned off for this dataset · hides 1 here/)).toBeInTheDocument();
    expect(screen.getByText("Your reason: always empty")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Suggest again: Remove the empty column notes" }));
    await waitFor(() => expect(calls.some((call) => call.key === `DELETE ${BASE}/dismissed/x1`)).toBe(true));
    expect(puts(calls)).toHaveLength(0); // nothing ticked again
  });
});

const notesCheck = { id: "c1", kind: "not_empty", name: null, enabled: true, columns: ["city"], values: [], min: null, max: null, pattern: null, op: "le", tolerance: 0, conditions: [], combine: "all", expect: "all", origin: "user", rule: null };
const orderCheck = { ...notesCheck, id: "c2", kind: "compare", columns: ["created", "closed"] };
const report = {
  rows: 30,
  results: [
    { id: "c1", status: "fail", failing: 2, rows: 30, message: null, examples: [3, 7], description: "city is never empty" },
    { id: "c2", status: "pass", failing: 0, rows: 30, message: null, examples: [], description: "created ≤ closed" },
  ],
  passed: 1,
  failed: 1,
  errors: 0,
  failing_rows: 2,
  seconds: 0.1,
};
const declared = { dataset_id: "d1", original_id: "d1", checks: [notesCheck, orderCheck], report, ran_at: "2026-11-13T10:00:00Z", trigger: "import", history: [] };

describe("Declared checks (N31)", () => {
  it("shows what fails at each import, its rows, and never changes the data", async () => {
    const calls = setup({
      [`GET ${BASE}/datasets/d1/checks`]: declared,
      [`GET ${BASE}/datasets/d1/checks/c1/rows`]: { check: "c1", failing: 2, columns: ["city", "brand"], rows: [{ row: 3, values: [null, "Acme"] }, { row: 7, values: ["", "Acme"] }] },
    });
    renderAt("/d/d1/clean");
    const notice = await screen.findByText(/1 declared check fails: 2 rows/);
    fireEvent.click(within(notice.parentElement!).getByRole("button", { name: "See the checks" }));
    const panel = await screen.findByRole("region", { name: "Declared checks" });
    expect(within(panel).getByText(/2 checks/)).toBeInTheDocument();
    expect(within(panel).getByText(/run .+, after a reimport/)).toBeInTheDocument();
    expect(within(panel).getByText("2 rows fail, of 30 (rows 4, 8)")).toBeInTheDocument();
    expect(within(panel).getByText("created ≤ closed")).toBeInTheDocument();
    fireEvent.click(within(panel).getByRole("button", { name: "See the rows: city is never empty" }));
    const dialog = await screen.findByRole("dialog", { name: "Rows that fail the check" });
    expect(await within(dialog).findAllByText("Acme")).toHaveLength(2); // the failing rows, all columns
    expect(within(dialog).getByRole("link", { name: /Open these rows in Data/ })).toHaveAttribute("href", expect.stringContaining("is_null"));
    expect(calls.some((call) => call.key.startsWith(`PUT ${BASE}/datasets/d1/draft`))).toBe(false);
  });

  it("adds a check from the form, and makes one of a filter step", async () => {
    const filter = { id: "s3", op: "filter_rows", conditions: [{ column: "city", op: "eq", value: "Atlantis", values: null }], combine: "all", action: "remove", enabled: true, origin: "manual", rule: null, suggestion: null };
    const calls = setup({
      [`GET ${BASE}/datasets/d1/checks`]: { ...declared, checks: [notesCheck], report: null },
      [`PUT ${BASE}/datasets/d1/checks`]: (body: unknown) => ({ ...declared, checks: (body as { checks: unknown[] }).checks, report: null }),
      [`GET ${BASE}/datasets/d1/draft`]: { dataset_id: "d1", steps: [filter], automatic: false, updated_at: "now" },
    });
    renderAt("/d/d1/clean?tab=checks");
    const panel = await screen.findByRole("region", { name: "Declared checks" });
    fireEvent.click(within(panel).getByRole("button", { name: "Add a check" }));
    const form = within(panel).getByRole("form", { name: "Check" });
    fireEvent.change(within(form).getByRole("combobox", { name: "Check that" }), { target: { value: "unique" } });
    fireEvent.click(within(form).getByRole("button", { name: "Save the check" }));
    await waitFor(() => expect(calls.filter((call) => call.key === `PUT ${BASE}/datasets/d1/checks`)).toHaveLength(1));
    const saved = (calls.find((call) => call.key === `PUT ${BASE}/datasets/d1/checks`)!.body as { checks: { id: string; kind: string; columns: string[] }[] }).checks;
    expect(saved.map((check) => [check.id, check.kind, check.columns])).toEqual([["c1", "not_empty", ["city"]], ["c2", "unique", ["city"]]]);

    fireEvent.click(screen.getByRole("tab", { name: /Steps/ }));
    fireEvent.click(await screen.findByRole("button", { name: /Make it a check: No row: city = Atlantis/ }));
    await waitFor(() => expect(calls.filter((call) => call.key === `PUT ${BASE}/datasets/d1/checks`)).toHaveLength(2));
    const made = (calls.filter((call) => call.key === `PUT ${BASE}/datasets/d1/checks`)[1]!.body as { checks: { kind: string; expect: string; origin: string }[] }).checks.at(-1);
    expect(made).toMatchObject({ kind: "condition", expect: "none", origin: "step" });
    expect(await screen.findByText(/Declared as check c2/)).toBeInTheDocument();
  });
});

const version = { dataset_id: "d2", parent_id: "d1", seq: 1, name: "sales · Cleaned v1", origin: "user", status: "ready", created_at: "2026-11-13T10:00:00Z", out_of_date: false, steps: [trim], report: null, source_id: null, source_seq: null, checks: { ran_at: "2026-11-13T10:00:00Z", trigger: "version", rows: 30, passed: 1, failed: 1, errors: 0, failing_rows: 2 } };

describe("The cleaning as SQL and Python (N45)", () => {
  it("downloads a version as a DuckDB script or a Python script, beside its recipe", async () => {
    setup({ [`GET ${BASE}/datasets/d1/versions`]: { dataset_id: "d1", original_id: "d1", versions: [version] } });
    renderAt("/d/d1/clean?tab=versions");
    expect(await screen.findByText(/1 check fails: 2 rows/)).toBeInTheDocument(); // N31 on the version
    fireEvent.click(screen.getByRole("button", { name: "Download: Cleaned v1" }));
    expect(screen.getByRole("link", { name: /SQL script \(\.sql\)/ })).toHaveAttribute("href", "/api/v1/modules/clean/versions/d2/sql");
    expect(screen.getByRole("link", { name: /Python script \(\.py\)/ })).toHaveAttribute("href", "/api/v1/modules/clean/versions/d2/python");
    expect(screen.getByRole("button", { name: /Download recipe/ })).toBeInTheDocument();
  });
});

const entry = (action: string, extra: Record<string, unknown> = {}) => ({ at: "2026-11-13T10:00:00Z", actor: "user", action, step_id: null, op: null, rule: null, origin: null, columns: [], cells: null, rows: null, detail: null, version_id: null, ...extra });

describe("The cleaning journal (N46)", () => {
  it("lists the draft's changes and a version's log, Markdown and HTML to keep", async () => {
    setup({
      [`GET ${BASE}/datasets/d1/versions`]: { dataset_id: "d1", original_id: "d1", versions: [version] },
      [`GET ${BASE}/datasets/d1/journal`]: [entry("applied", { actor: "auto", step_id: "s1", op: "normalize_spaces", rule: "B-01", columns: ["city"], cells: 52, detail: "Trim extra spaces in city" }), entry("removed", { step_id: "s1", op: "normalize_spaces", rule: "B-01", columns: ["city"] })],
      [`GET ${BASE}/versions/d2/journal`]: {
        name: "sales · Cleaned v1",
        dataset_id: "d2",
        made_from: "sales",
        made_at: "2026-11-13T10:00:00Z",
        origin: "user",
        steps: [{ id: "s1", op: "normalize_spaces", rule: "B-01", origin: "auto", enabled: true, title: "Trim extra spaces in city", columns: ["city"], cells: 52, rows_removed: 0, example: [" Paris", "Paris"] }],
        report: { rows_before: 30, rows_after: 30, rows_removed: 0, cells_changed: 52, columns_before: 3, columns_after: 3, cells_by_column: { city: 52 }, steps: [], added: [], dropped: [], type_changes: [], seconds: 0.1, reshaped_by: null },
        entries: [entry("saved", { detail: "Cleaned v1, 1 steps: 52 cells changed", version_id: "d2" })],
        applied: { auto: 1, suggested: 0, manual: 0 },
      },
    });
    renderAt("/d/d1/clean?tab=versions");
    const draft = await screen.findByText("2 changes of the draft since the last version");
    fireEvent.click(draft);
    expect(screen.getByText("Applied s1 (B-01, normalize_spaces · city): Trim extra spaces in city · 52 cells · Clean, automatically")).toBeInTheDocument();
    expect(screen.getByText("Removed s1 (B-01, normalize_spaces · city) · you")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Cleaning log: Cleaned v1" }));
    const dialog = await screen.findByRole("dialog", { name: "Cleaning log of Cleaned v1" });
    expect(await within(dialog).findByText(/1 steps applied automatically · 0 suggestions ticked · 0 added by hand/)).toBeInTheDocument();
    expect(within(dialog).getByText("Saved: Cleaned v1, 1 steps: 52 cells changed · you")).toBeInTheDocument();
    expect(within(dialog).getByRole("link", { name: "Cleaning log (.md)" })).toHaveAttribute("href", "/api/v1/modules/clean/versions/d2/journal/markdown");
    expect(within(dialog).getByRole("link", { name: "Cleaning log (.html)" })).toHaveAttribute("href", "/api/v1/modules/clean/versions/d2/journal/html");
  });
});
