import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { column } from "../../test/columns";
import { dataset, health, mockApi } from "../../test/mockApi";
import { renderAt } from "../../test/render";

/* 095: Changes knows the cleaned versions of a file (065). */

vi.mock("../../charts/Plot", () => ({ Plot: () => null }));

const changes = {
  before: { label: "sales", taken_at: "2026-10-01T08:00:00Z", rows: 407, duplicates: 7 },
  after: { label: "sales · Cleaned v1", taken_at: "2026-11-01T08:00:00Z", rows: 400, duplicates: 0 },
  level: "check",
  level_label: "Check",
  headline: "Compared with sales: -7 rows (-1.7%).",
  reasons: [],
  rows_change: -7,
  rows_change_pct: -1.7,
  counts: { added: 0, removed: 0, renamed: 0, retyped: 0, shifted: 0, new_categories: 0 },
  schema: [],
  quality: [],
  dates: [],
  distributions: [],
  thresholds: "PSI < 0.1 stable",
};

const step = (id: string, op: string, extra: Record<string, unknown> = {}) => ({ id, op, enabled: true, origin: "rule", rule: null, suggestion: null, ...extra });
const steps = [
  step("s1", "normalize_spaces", { columns: ["city"] }),
  step("s2", "drop_duplicates", { keys: null, keep: "first" }),
  step("s3", "change_case", { columns: ["brand"], case: "title" }),
  step("s4", "remove_invisible", { columns: ["notes"] }),
  step("s5", "drop_empty_rows"),
  step("s6", "fill_empty", { column: "region", value: "Unknown", enabled: false }),
];

const version = (seq: number, overrides: Record<string, unknown> = {}) => ({
  dataset_id: `v${seq}`,
  parent_id: "d1",
  seq,
  name: `sales · Cleaned v${seq}`,
  origin: "user",
  status: "ready",
  created_at: "2026-11-01T08:00:00Z",
  out_of_date: false,
  steps: steps.slice(0, seq === 1 ? 6 : 1),
  report: null,
  ...overrides,
});

const family = (id: string) => ({ dataset_id: id, original_id: "d1", versions: [version(1), version(2)] });
const current = (version: number) => ({ version, imported_at: "2026-11-01T08:00:00Z", rows: 400, kept: true, kept_bytes: 0, current: true });
const earlier = (version: number, day: string) => ({ version, imported_at: `${day}T08:00:00Z`, rows: 407, kept: version === 1, kept_bytes: 10, current: false });

function setup(extra: Record<string, unknown> = {}) {
  const datasets = [
    dataset(),
    dataset({ id: "v1", name: "sales · Cleaned v1" }),
    dataset({ id: "v2", name: "sales · Cleaned v2" }),
    dataset({ id: "d2", name: "october" }),
  ];
  const routes: Record<string, unknown> = { "GET /health": health, "GET /datasets": datasets };
  for (const one of datasets.slice(0, 3)) {
    routes[`GET /datasets/${one.id}`] = one;
    routes[`GET /datasets/${one.id}/columns`] = [column({ name: "city", position: 0 })];
    routes[`GET /datasets/${one.id}/versions`] = [current(1)];
    routes[`GET /datasets/${one.id}/changes`] = changes;
    routes[`GET /modules/clean/datasets/${one.id}/versions`] = family(one.id);
  }
  return mockApi({ ...routes, ...extra });
}

const asked = (calls: ReturnType<typeof setup>, id: string) => calls.filter((call) => call.key === `GET /datasets/${id}/changes`);
const labels = (group: string) => within(screen.getByRole("group", { name: group })).getAllByRole("option").map((option) => option.textContent);

describe("Changes of a cleaned version (095)", () => {
  beforeEach(() => localStorage.clear());
  afterEach(() => vi.unstubAllGlobals());

  it("compares with its original by default and lists the steps that explain it", async () => {
    const calls = setup();
    renderAt("/d/v1/changes");
    expect(await screen.findByText("Changes made by cleaning: Cleaned v1 vs the original.")).toBeInTheDocument();
    await screen.findByText(changes.headline);
    expect(asked(calls, "v1").every((call) => call.params.get("vs") === "d1")).toBe(true);
    expect(screen.getByText("Cleaning steps of Cleaned v1")).toBeInTheDocument();
    expect(screen.getByText("Trim extra spaces · city")).toBeInTheDocument();
    expect(screen.getByText("Remove duplicate rows")).toBeInTheDocument();
    expect(screen.getByText("… and 1 more")).toBeInTheDocument(); // 5 active steps, 4 listed
    expect((screen.getByRole("combobox") as HTMLSelectElement).value).toBe("d1");
  });

  it("groups the picker: this file's versions first, newest first, then the other datasets", async () => {
    setup();
    renderAt("/d/v1/changes");
    await screen.findByText(changes.headline);
    expect(labels("This file's versions")).toEqual(["Cleaned v2", "Original"]);
    expect(labels("Other datasets")).toEqual(["october"]);
  });

  it("compares two cleanings of one original", async () => {
    const calls = setup();
    renderAt("/d/v2/changes");
    await screen.findByText("Changes made by cleaning: Cleaned v2 vs the original.");
    fireEvent.change(screen.getByRole("combobox"), { target: { value: "v1" } });
    expect(await screen.findByText("Cleaned v2 vs Cleaned v1: two cleanings of the same original.")).toBeInTheDocument();
    await waitFor(() => expect(asked(calls, "v2").at(-1)?.params.get("vs")).toBe("v1"));
    expect(screen.queryByText(/Cleaning steps of/)).toBeNull();
  });

  it("an original never reimported compares with its latest cleaned version", async () => {
    const calls = setup();
    renderAt("/d/d1/changes");
    expect(await screen.findByText("The original vs Cleaned v2: what cleaning changed, read from the original's side.")).toBeInTheDocument();
    await screen.findByText(changes.headline);
    expect(asked(calls, "d1").every((call) => call.params.get("vs") === "v2")).toBe(true);
    expect(labels("This file's versions")).toEqual(["Cleaned v2", "Cleaned v1"]);
  });

  it("from the original: the previous import by default, any cleaned version on demand", async () => {
    const calls = setup({ "GET /datasets/d1/versions": [current(3), earlier(2, "2026-10-01"), earlier(1, "2026-09-01")] });
    renderAt("/d/d1/changes");
    expect(await screen.findByText("What changed since the previous import (2026-10-01).")).toBeInTheDocument();
    expect(labels("This file's versions")).toEqual(["Cleaned v2", "Cleaned v1", "Previous import · 2026-10-01", "Import of 2026-09-01"]);
    const picker = screen.getByRole("combobox") as HTMLSelectElement;
    expect(picker.value).toBe("@2");
    expect(asked(calls, "d1").at(-1)?.params.get("vs")).toBeNull();

    fireEvent.change(picker, { target: { value: "v1" } });
    expect(await screen.findByText("The original vs Cleaned v1: what cleaning changed, read from the original's side.")).toBeInTheDocument();
    expect(screen.getByText("Cleaning steps of Cleaned v1")).toBeInTheDocument();
    await waitFor(() => expect(asked(calls, "d1").at(-1)?.params.get("vs")).toBe("v1"));

    fireEvent.change(picker, { target: { value: "@1" } });
    expect(await screen.findByText("What changed since the import of 2026-09-01.")).toBeInTheDocument();
    await waitFor(() => expect(asked(calls, "d1").at(-1)?.params.get("version")).toBe("1"));
    expect(asked(calls, "d1").at(-1)?.params.get("vs")).toBeNull();
  });

  it("says the rows are paired by the cleaning's row map", async () => {
    setup({
      "GET /datasets/v1/changes/rows": {
        available: true,
        reason: null,
        candidates: [],
        alignment: "rowmap",
        rowmap: true,
        alignment_note: null,
        result: {
          key: "row",
          aligned: true,
          rows_before: 407,
          rows_after: 400,
          added: 0,
          removed: 7,
          modified: 120,
          unchanged: 280,
          columns: [{ column: "city", modified: 120 }],
          examples: [{ key: "2", column: "city", before: "  Paris", after: "Paris" }],
          added_keys: [],
          removed_keys: ["13"],
          compared: ["city"],
        },
      },
    });
    renderAt("/d/v1/changes?tab=rows");
    expect(await screen.findByText(/Rows are paired by the row map saved with each cleaned version/)).toBeInTheDocument();
    const key = screen.getByRole("combobox", { name: /Rows matched on/ }) as HTMLSelectElement;
    expect(key.value).toBe("__rowmap");
    expect(screen.getByRole("link", { name: /Download the differences/ })).toHaveAttribute("href", "/api/v1/datasets/v1/changes/rows/export?vs=d1");
  });

  it("without the Clean module, behaves as before", async () => {
    const missing = () => Response.json({ error: { message: "Not found", code: "no_previous_version" } }, { status: 404 });
    const calls = setup({ "GET /modules/clean/datasets/v1/versions": missing, "GET /datasets/v1/changes": missing });
    renderAt("/d/v1/changes");
    expect(await screen.findByText("No previous version yet")).toBeInTheDocument();
    expect(asked(calls, "v1").every((call) => call.params.get("vs") === null)).toBe(true);
  });
});
