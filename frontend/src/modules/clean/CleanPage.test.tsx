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

// 101: the Advanced layer on demand, More checks and replay
const unusualStep = { id: "s5001", op: "drop_rows", rows: [4, 9], enabled: true, origin: "suggested", rule: "E-06", suggestion: "E-06:remove" };
const unusual = { id: "E-06:remove", rule: "E-06", layer: "expert", mode: "S", impact: "rows", risk: "review", order: 4000000, title: "Remove 2 unusual rows", reason: "Isolation Forest… An unusual row may be right.", columns: ["city"], cells: 0, rows: 2, examples: [{ before: "row 5: amount 9000", after: "removed" }], step: unusualStep };
const methods = [
  { id: "unusual_rows", rule: "E-06", available: true, why_not: null, columns: [] },
  { id: "suspicious_labels", rule: "E-03", available: false, why_not: "Needs a category of 2 to 50 values (the labels) and other columns to learn from.", columns: [] },
  { id: "fill_empty", rule: "E-07", available: false, why_not: "Needs a column of numbers with empty cells (60 % of them at most).", columns: [] },
  { id: "cell_errors", rule: "E-01", available: true, why_not: null, columns: ["city"] },
];
const result = (method: string, extra: Record<string, unknown> = {}) => ({ method, rule: "E-06", suggestions: [], note: null, to_mark: [], rows: 1200, sampled: false, seconds: 0.8, ...extra });
const expertJob = (status: string, method = "unusual_rows", extra: Record<string, unknown> = {}) => ({ id: "x1", status, stage: status, dataset_id: "d1", method, error: status === "cancelled" ? "Stopped: nothing was found." : null, result: null, ...extra });

function setup(lineage: unknown[] = [], found: unknown[] = basic, extra: Record<string, unknown> = {}) {
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
    [`GET ${BASE}/datasets/d1/suggestions/advanced`]: { ...suggestions, suggestions: [] },
    [`GET ${BASE}/datasets/d1/expert`]: methods,
    [`POST ${BASE}/datasets/d1/expert/unusual_rows`]: expertJob("running"),
    [`GET ${BASE}/expert-jobs/x1`]: expertJob("done", "unusual_rows", { result: result("unusual_rows", { suggestions: [unusual] }) }),
    ...extra,
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
    // Advanced is folded and not asked for until opened (101, decision 1); More checks is inside it
    expect(screen.getByRole("button", { name: /Advanced/ })).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByRole("button", { name: /More checks/ })).not.toBeInTheDocument();
    expect(await screen.findByText(/10 cells changed/)).toBeInTheDocument();
    for (const move of ["Review", "Apply", "Save version"]) expect(screen.getByText(move)).toBeInTheDocument();
    expect(screen.getByText("2 fixes found, grouped by what they change in the data.")).toBeInTheDocument();
  });

  it("applies the reliable fixes first, folded, each undone on its own or all at once (N03, N04)", async () => {
    const reliable = { ...basic[0]!, auto: true, measured: 50 };
    const calls = setup([], [reliable, basic[1]!, ...values], { [`GET ${BASE}/settings`]: { mode: "apply" } });
    renderAt("/d/d1/clean");
    const group = await screen.findByRole("region", { name: /^Applied automatically/ });
    expect(screen.getByText(/1 applied automatically · 2 value changes · 1 needs your decision/)).toBeInTheDocument();
    expect(within(group).getByText("1 of 1 applied · 52 cells")).toBeInTheDocument();
    const fold = within(group).getByRole("button", { name: /Applied automatically/ });
    expect(fold).toHaveAttribute("aria-expanded", "false");
    expect(within(group).queryByText("Trim extra spaces in city")).not.toBeInTheDocument(); // folded
    expect(screen.queryByRole("checkbox", { name: "Trim extra spaces in city" })).not.toBeInTheDocument(); // listed once
    fireEvent.click(fold);
    expect(within(group).getByText(/A reliable fix: right on 99% or more of the cases measured \(n = 50\)\./)).toBeInTheDocument();

    fireEvent.click(within(group).getByRole("button", { name: "Undo: Trim extra spaces in city" }));
    await waitFor(() => expect(puts(calls)).toHaveLength(1));
    expect((puts(calls)[0]!.body as { steps: unknown[] }).steps).toEqual([]);
    expect(within(group).getByText("Undone")).toBeInTheDocument();
    fireEvent.keyDown(window, { key: "z", ctrlKey: true }); // Ctrl+Z puts it back
    await waitFor(() => expect(puts(calls)).toHaveLength(2));
    expect((puts(calls)[1]!.body as { steps: { suggestion: string }[] }).steps.map((step) => step.suggestion)).toEqual(["B-01:city"]);

    fireEvent.click(within(group).getByRole("button", { name: "Undo all automatic fixes" }));
    await waitFor(() => expect(puts(calls)).toHaveLength(3));
    expect((puts(calls)[2]!.body as { steps: unknown[] }).steps).toEqual([]);
    expect(within(group).getByRole("button", { name: "Apply them all again" })).toBeInTheDocument();
    expect(calls.some((call) => call.key === `POST ${BASE}/datasets/d1/versions`)).toBe(false); // nothing saved by itself
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

  it("shows the same fix on several columns as one line, ticked for all or one by one (N39)", async () => {
    const trimOf = (column: string, order: number) => ({
      ...basic[0]!,
      id: `B-01:${column}`,
      order,
      title: `Trim extra spaces in ${column}`,
      columns: [column],
      cells: 10,
      step: { ...trim, id: `s${order}`, columns: [column], origin: "suggested", suggestion: `B-01:${column}` },
    });
    const calls = setup([], [{ ...basic[0]!, cells: 10 }, trimOf("region", 300002), trimOf("state", 300003), basic[1]!]);
    renderAt("/d/d1/clean");
    const group = await screen.findByRole("region", { name: /^Safe formatting/ });
    const line = within(group).getByRole("checkbox", { name: "Trim extra spaces in 3 columns" });
    expect(line).toBePartiallyChecked(); // city is in the draft, region and state are not
    expect(within(group).getAllByText("1 of 3 ticked")).toHaveLength(2); // the group's, the line's
    expect(within(group).getByText("30 cells")).toBeInTheDocument(); // summed
    expect(within(group).queryByRole("checkbox", { name: "Trim extra spaces in region" })).not.toBeInTheDocument();
    fireEvent.click(line);
    await waitFor(() => expect(puts(calls)).toHaveLength(1));
    expect((puts(calls)[0]!.body as { steps: { suggestion: string }[] }).steps.map((step) => step.suggestion)).toEqual(["B-01:city", "B-01:region", "B-01:state"]);
    expect(line).toBeChecked();
    fireEvent.click(within(group).getByRole("button", { name: "3 columns" }));
    const region = within(group).getByRole("checkbox", { name: "Trim extra spaces in region" }); // each column still reachable
    fireEvent.click(region);
    await waitFor(() => expect(puts(calls)).toHaveLength(2));
    expect((puts(calls)[1]!.body as { steps: { suggestion: string }[] }).steps.map((step) => step.suggestion)).toEqual(["B-01:city", "B-01:state"]);
    expect(line).toBePartiallyChecked();
  });

  it("gathers a reliable fix applied on several columns, undone in one gesture (N39, N04)", async () => {
    const trimOf = (column: string, order: number) => ({ ...basic[0]!, id: `B-01:${column}`, order, auto: true, measured: 50, title: `Trim extra spaces in ${column}`, columns: [column], step: { ...trim, id: `s${order}`, columns: [column], suggestion: `B-01:${column}` } });
    const found = [{ ...basic[0]!, auto: true, measured: 50 }, trimOf("region", 300002), trimOf("state", 300003)];
    const draft = { dataset_id: "d1", steps: found.map((item) => item.step), automatic: true, updated_at: null };
    const calls = setup([], found, { [`GET ${BASE}/settings`]: { mode: "apply" }, [`GET ${BASE}/datasets/d1/draft`]: draft });
    renderAt("/d/d1/clean");
    const group = await screen.findByRole("region", { name: /^Applied automatically/ });
    fireEvent.click(within(group).getByRole("button", { name: /Applied automatically/ }));
    expect(within(group).getByText("Trim extra spaces in 3 columns")).toBeInTheDocument();
    fireEvent.click(within(group).getByRole("button", { name: "Undo: Trim extra spaces in 3 columns" }));
    await waitFor(() => expect(puts(calls)).toHaveLength(1));
    expect((puts(calls)[0]!.body as { steps: unknown[] }).steps).toEqual([]);
    expect(within(group).getByRole("button", { name: "Apply again: Trim extra spaces in 3 columns" })).toBeInTheDocument();
  });

  it("asks for Advanced only when it is opened, and for the checks only when More checks is (097, 101)", async () => {
    const advanced = { ...values[0]!, id: "A-01:status", rule: "A-01", layer: "advanced", mode: "S", risk: "review", title: "Cluster 3 spellings in status" };
    const calls = setup([], basic, { [`GET ${BASE}/datasets/d1/suggestions/advanced`]: { ...suggestions, suggestions: [advanced] } });
    renderAt("/d/d1/clean");
    const layer = await screen.findByRole("button", { name: /Advanced/ });
    expect(layer).toHaveAttribute("aria-expanded", "false");
    expect(calls.some((call) => call.key.includes("/suggestions/advanced") || call.key.includes("/expert"))).toBe(false); // C-29
    fireEvent.click(layer);
    expect(await screen.findByText("Cluster 3 spellings in status")).toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: "Cluster 3 spellings in status" })).not.toBeChecked();
    expect(calls.some((call) => call.key.endsWith("/expert"))).toBe(false);
    const more = screen.getByRole("button", { name: /More checks/ });
    expect(more).toHaveAttribute("aria-expanded", "false");
    fireEvent.click(more);
    expect(await screen.findByRole("heading", { name: /Unusual rows/ })).toBeInTheDocument();
    expect(screen.getByText(/Not for these data: Needs a category/)).toBeInTheDocument(); // says why
    expect(calls.some((call) => call.key.includes("/expert/"))).toBe(false); // nothing run before Run
  });

  it("reviews a merge value by value: untick a value or a group, change a target, the step keeps what is ticked (N38)", async () => {
    const map = { "FOREST HILL": "FOREST HILLS", "FOREST HILS": "FOREST HILLS", BROKLYN: "BROOKLYN" };
    const step = { id: "s9", op: "map_values", column: "city", map, enabled: true, origin: "suggested", rule: "A-03", suggestion: "A-03:city" };
    const clusters = [
      { target: "FOREST HILLS", rows: 900, values: [{ value: "FOREST HILL", rows: 1, spellings: ["FOREST HILL"] }, { value: "FOREST HILS", rows: 3, spellings: ["FOREST HILS"] }] },
      { target: "BROOKLYN", rows: 5000, values: [{ value: "BROKLYN", rows: 2, spellings: ["BROKLYN"] }] },
    ];
    const a03 = { ...values[0]!, id: "A-03:city", rule: "A-03", layer: "advanced", mode: "S", risk: "review", title: "Merge near spellings in city (2 groups)", reason: "…", columns: ["city"], cells: 6, examples: [], step, clusters };
    const calls = setup([], basic, { [`GET ${BASE}/datasets/d1/suggestions/advanced`]: { ...suggestions, suggestions: [a03] } });
    renderAt("/d/d1/clean");
    fireEvent.click(await screen.findByRole("button", { name: /Advanced/ }));
    const merge = await screen.findByRole("checkbox", { name: "Merge near spellings in city (2 groups)" });
    expect(merge).not.toBeChecked();
    fireEvent.click(screen.getByRole("button", { name: "Review 2 groups" }));
    const review = screen.getByRole("region", { name: "Review of the merge, value by value" });
    expect(within(review).getByText("Groups kept: 0 of 2 · values: 0 of 3 · 0 cells")).toBeInTheDocument();
    fireEvent.click(within(review).getByRole("button", { name: "Tick all" }));
    await waitFor(() => expect(puts(calls)).toHaveLength(1));
    expect((puts(calls)[0]!.body as { steps: { map?: unknown }[] }).steps.at(-1)!.map).toEqual(map);
    expect(merge).toBeChecked();

    fireEvent.click(within(review).getByRole("checkbox", { name: "Merge 'FOREST HILL' into 'FOREST HILLS'" }));
    await waitFor(() => expect(puts(calls)).toHaveLength(2));
    expect((puts(calls)[1]!.body as { steps: { map?: unknown }[] }).steps.at(-1)!.map).toEqual({ "FOREST HILS": "FOREST HILLS", BROKLYN: "BROOKLYN" });
    expect(merge).toBePartiallyChecked();
    expect(screen.getByText("partly kept")).toBeInTheDocument();

    const target = within(review).getByRole("textbox", { name: "Value the group of 'BROOKLYN' becomes" });
    fireEvent.change(target, { target: { value: "Brooklyn" } });
    fireEvent.keyDown(target, { key: "Enter" });
    await waitFor(() => expect(puts(calls)).toHaveLength(3));
    expect((puts(calls)[2]!.body as { steps: { map?: unknown }[] }).steps.at(-1)!.map).toEqual({ "FOREST HILS": "FOREST HILLS", BROKLYN: "Brooklyn", BROOKLYN: "Brooklyn" });
    expect(within(review).getByText("5,000 rows of 'BROOKLYN' become it too")).toBeInTheDocument();

    const forest = within(review).getByRole("checkbox", { name: "Merge the group of 'FOREST HILLS'" });
    expect(forest).toBePartiallyChecked();
    fireEvent.click(forest); // mixed → the whole group
    await waitFor(() => expect(puts(calls)).toHaveLength(4));
    expect(Object.keys((puts(calls)[3]!.body as { steps: { map?: object }[] }).steps.at(-1)!.map!)).toEqual(["FOREST HILL", "FOREST HILS", "BROKLYN", "BROOKLYN"]);
    fireEvent.click(forest);
    fireEvent.click(within(review).getByRole("checkbox", { name: "Merge the group of 'BROOKLYN'" }));
    await waitFor(() => expect(puts(calls)).toHaveLength(6));
    expect((puts(calls)[5]!.body as { steps: { suggestion: string }[] }).steps.map((one) => one.suggestion)).toEqual(["B-01:city"]); // nothing ticked: out of the draft
    expect(merge).not.toBeChecked();
  });

  it("shows the rows a fix touches, before and after, and opens them in Data (N41)", async () => {
    const calls = setup([], [...basic, ...values], { "GET /datasets/d1/columns": [column({ name: "city", type: "categorical_nominal" }), column({ name: "region" })] });
    renderAt("/d/d1/clean");
    fireEvent.click(await screen.findByRole("button", { name: "See the rows: Merge 2 spellings of 'Paris'" }));
    const dialog = await screen.findByRole("dialog", { name: "Rows this fix touches" });
    expect(within(dialog).getByText(/Not ticked: shown as if it were/)).toBeInTheDocument();
    expect(await within(dialog).findByText("The first 1 row touched, before and after")).toBeInTheDocument();
    const asked = calls.filter((call) => call.key === `POST ${BASE}/datasets/d1/preview`).map((call) => call.body as { steps: { suggestion: string; id: string }[]; step_id: string });
    const one = asked.find((body) => body.step_id !== null)!;
    expect(one.steps.map((step) => step.suggestion)).toEqual(["B-01:city", "B-03:city"]); // the draft, then this fix at its place
    expect(one.step_id).toBe(one.steps[1]!.id);
    const open = await within(dialog).findByRole("link", { name: "Open these rows in Data" });
    expect(decodeURIComponent(open.getAttribute("href") ?? "")).toContain('"conditions":[{"column":"city","op":"in","values":["PARIS"],"negate":false}]');
    expect(puts(calls)).toHaveLength(0); // nothing added to the draft
    fireEvent.keyDown(dialog, { key: "Escape" });
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    // a token step cannot be an exact filter: it says so
    fireEvent.click(screen.getByRole("button", { name: "See the rows: Read 'unknown' as missing in region" }));
    expect(await screen.findByText(/Only shown here/)).toBeInTheDocument();
  });

  it("reviews without a mouse: j / k move, Space ticks, Enter opens, u undoes, listed under ? (N42)", async () => {
    const calls = setup([], [...basic, ...values]);
    renderAt("/d/d1/clean");
    await screen.findByRole("checkbox", { name: "Trim extra spaces in city" });
    const focused = () => document.activeElement?.getAttribute("aria-label");
    fireEvent.keyDown(window, { key: "j" });
    expect(focused()).toBe("Trim extra spaces in city"); // the first fix, wherever the focus was
    fireEvent.keyDown(window, { key: "j" });
    expect(focused()).toBe("Read 'n/a' as missing in status");
    fireEvent.keyDown(window, { key: "k" });
    fireEvent.keyDown(window, { key: "j" });
    fireEvent.keyDown(document.activeElement!, { key: " " });
    await waitFor(() => expect(puts(calls)).toHaveLength(1));
    expect((puts(calls)[0]!.body as { steps: { suggestion: string }[] }).steps.map((step) => step.suggestion)).toEqual(["B-01:city", "B-04:status"]);
    expect(screen.getByRole("checkbox", { name: "Read 'n/a' as missing in status" })).toBeChecked();

    fireEvent.keyDown(document.activeElement!, { key: "Enter" }); // its rows
    expect(await screen.findByRole("dialog", { name: "Rows this fix touches" })).toBeInTheDocument();
    fireEvent.keyDown(window, { key: "j" }); // under a dialog: nothing moves
    expect(focused()).not.toBe("Merge 2 spellings of 'Paris'");
    fireEvent.keyDown(screen.getByRole("dialog"), { key: "Escape" });
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());

    fireEvent.keyDown(window, { key: "u" });
    await waitFor(() => expect(puts(calls)).toHaveLength(2));
    expect((puts(calls)[1]!.body as { steps: unknown[] }).steps).toHaveLength(1);
    expect(screen.getByRole("checkbox", { name: "Read 'n/a' as missing in status" })).not.toBeChecked();
    fireEvent.keyDown(window, { key: "u", ctrlKey: true }); // never with a modifier
    fireEvent.keyDown(screen.getByRole("checkbox", { name: "Trim extra spaces in city" }), { key: "j" }); // a checkbox is not a field
    expect(focused()).toBe("Trim extra spaces in city");

    fireEvent.keyDown(window, { key: "?" });
    const help = await screen.findByRole("dialog", { name: "Keyboard shortcuts" });
    expect(within(help).getByText("Clean › Suggestions")).toBeInTheDocument();
    expect(within(help).getByText("Next or previous fix")).toBeInTheDocument();
    expect(puts(calls)).toHaveLength(2);
  });

  it("runs a check, shows what it found unticked, and ticks it into the draft (101)", async () => {
    const calls = setup();
    renderAt("/d/d1/clean");
    fireEvent.click(await screen.findByRole("button", { name: /Advanced/ }));
    fireEvent.click(screen.getByRole("button", { name: /More checks/ }));
    const check = await screen.findByRole("region", { name: /Unusual rows/ });
    fireEvent.click(within(check).getByRole("button", { name: "Run" }));
    const fix = await within(check).findByRole("checkbox", { name: "Remove 2 unusual rows" });
    expect(fix).not.toBeChecked(); // C-30
    expect(within(check).getByText(/Needs your decision/)).toBeInTheDocument();
    expect(within(check).getByText("1,200 rows read in 0.8 s")).toBeInTheDocument();
    fireEvent.click(fix);
    await waitFor(() => expect(puts(calls)).toHaveLength(1));
    const steps = (puts(calls)[0]!.body as { steps: { op: string; suggestion: string }[] }).steps;
    expect(steps.at(-1)).toMatchObject({ op: "drop_rows", suggestion: "E-06:remove" });
    fireEvent.click(within(check).getByRole("button", { name: "Clear" }));
    await waitFor(() => expect(within(check).queryByRole("checkbox", { name: "Remove 2 unusual rows" })).not.toBeInTheDocument());
    expect(within(check).getByRole("button", { name: "Run" })).toBeInTheDocument();
  });

  it("stops a running check (101, C-31)", async () => {
    let status = "running";
    const calls = setup([], basic, {
      [`GET ${BASE}/expert-jobs/x1`]: () => expertJob(status),
      [`DELETE ${BASE}/expert-jobs/x1`]: () => {
        status = "cancelled";
        return expertJob("running");
      },
    });
    renderAt("/d/d1/clean");
    fireEvent.click(await screen.findByRole("button", { name: /Advanced/ }));
    fireEvent.click(screen.getByRole("button", { name: /More checks/ }));
    const check = await screen.findByRole("region", { name: /Unusual rows/ });
    fireEvent.click(within(check).getByRole("button", { name: "Run" }));
    fireEvent.click(await within(check).findByRole("button", { name: "Stop" }));
    expect(await within(check).findByText("Stopped: nothing was found.")).toBeInTheDocument();
    expect(calls.some((call) => call.key === `DELETE ${BASE}/expert-jobs/x1`)).toBe(true);
  });

  it("asks for marks, then learns from them (101, S7)", async () => {
    const toMark = [
      { row: 4, value: "Pariss", hint: "close to a much more frequent value", alike: 3 },
      { row: 0, value: "Paris", hint: "a typical cell", alike: 900 },
    ];
    const calls = setup([], basic, {
      [`POST ${BASE}/datasets/d1/expert/cell_errors`]: expertJob("running", "cell_errors"),
      [`GET ${BASE}/expert-jobs/x1`]: expertJob("done", "cell_errors", { result: result("cell_errors", { rule: "E-01", to_mark: toMark }) }),
    });
    renderAt("/d/d1/clean");
    fireEvent.click(await screen.findByRole("button", { name: /Advanced/ }));
    fireEvent.click(screen.getByRole("button", { name: /More checks/ }));
    const check = await screen.findByRole("region", { name: /Cell errors from your marks/ });
    fireEvent.click(within(check).getByRole("button", { name: "Run" }));
    const learn = await within(check).findByRole("button", { name: /Learn from/ });
    expect(learn).toBeDisabled();
    fireEvent.click(within(within(check).getByRole("radiogroup", { name: "Mark of row 5" })).getByRole("radio", { name: "Error" }));
    expect(learn).toBeDisabled(); // an OK is needed too
    fireEvent.click(within(within(check).getByRole("radiogroup", { name: "Mark of row 1" })).getByRole("radio", { name: "OK" }));
    fireEvent.click(within(check).getByRole("button", { name: "Learn from 2 marks" }));
    await waitFor(() => expect(calls.filter((call) => call.key === `POST ${BASE}/datasets/d1/expert/cell_errors`)).toHaveLength(2));
    const body = calls.filter((call) => call.key === `POST ${BASE}/datasets/d1/expert/cell_errors`)[1]!.body;
    expect(body).toEqual({ column: "city", marks: [{ row: 0, error: false }, { row: 4, error: true }] });
  });

  it("replays a cleaning made elsewhere and uses it as the draft (101, M5)", async () => {
    const report = {
      recipe: "sales · Cleaned v1",
      made_on: "sales",
      steps: [
        { id: "s1", op: "normalize_spaces", rule: "B-01", status: "applies", message: null, columns: ["city"], cells: 12, rows: 0, new_values: [], new_count: 0 },
        { id: "s9", op: "drop_rows", rule: "E-06", status: "skipped", message: "Holds row numbers of the file it was made on: it does not apply to another file.", columns: [], cells: 0, rows: 0, new_values: [], new_count: 0 },
      ],
      columns: [{ recipe: "region", column: null, status: "missing", distance: null }],
      new_columns: ["channel"],
      new_problems: [values[0]],
      adapted: [trim, { ...unusualStep, id: "s9", enabled: false }],
      applies: 1,
      skipped: 1,
      seconds: 0.3,
    };
    const lineage = [{ dataset_id: "d2", parent_id: "d1", seq: 1, origin: "user", steps: 2, created_at: "2026-11-10T10:00:00Z", out_of_date: false, source_id: null, source_seq: null }];
    const calls = setup(lineage, basic, { [`POST ${BASE}/datasets/d1/replay`]: report });
    renderAt("/d/d1/clean?tab=versions");
    fireEvent.click(await screen.findByRole("button", { name: "Replay a cleaning" }));
    const dialog = await screen.findByRole("dialog", { name: "Replay a cleaning on this dataset" });
    expect(within(dialog).getByRole("button", { name: "Use as draft" })).toBeDisabled();
    fireEvent.change(within(dialog).getByRole("combobox", { name: "Cleaning to replay" }), { target: { value: "d2" } });
    fireEvent.click(within(dialog).getByRole("button", { name: "Check" }));
    expect(await within(dialog).findByText("1 steps apply · 1 no longer apply")).toBeInTheDocument();
    expect(within(dialog).getByText(/Holds row numbers of the file it was made on/)).toBeInTheDocument();
    expect(within(dialog).getByText("region is missing from this file")).toBeInTheDocument();
    expect(within(dialog).getByText(/Read 'n\/a' as missing in status/)).toBeInTheDocument(); // new, not applied
    expect(calls.find((call) => call.key === `POST ${BASE}/datasets/d1/replay`)!.body).toEqual({ version_id: "d2", accept_proposals: false });
    fireEvent.click(within(dialog).getByRole("button", { name: "Use as draft" }));
    await waitFor(() => expect(puts(calls)).toHaveLength(1));
    const steps = (puts(calls)[0]!.body as { steps: { id: string; enabled: boolean }[] }).steps;
    expect(steps.map((step) => [step.id, step.enabled])).toEqual([["s1", true], ["s9", false]]);
    expect(await screen.findByRole("button", { name: "Add step" })).toBeInTheDocument(); // on the Steps tab
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
    expect(await screen.findByText("The first 1 row touched, before and after")).toBeInTheDocument();
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
