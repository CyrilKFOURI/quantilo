import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { forwardRef } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { column } from "../../test/columns";
import { dataset, health, mockApi } from "../../test/mockApi";
import { renderAt } from "../../test/render";

vi.mock("../../charts/Plot", () => ({
  Plot: forwardRef<HTMLDivElement, { label: string }>(function Plot({ label }, ref) {
    return <div ref={ref} role="img" aria-label={label} />;
  }),
}));

const columns = [
  column({ name: "prix", position: 0 }),
  column({ name: "region", position: 1, type: "categorical_nominal", detected_type: "categorical_nominal" }),
];

const shift = (overrides: Record<string, unknown> = {}) => ({
  column: "prix",
  type: "numeric_continuous",
  kind: "numeric",
  psi: 0.62,
  drift: "strong",
  rows_before: 2000,
  rows_after: 2000,
  wasserstein: 0.8,
  ks: 0.31,
  ks_p: 1e-9,
  chi2: null,
  chi2_p: null,
  dof: null,
  before: { mean: 20, median: 19, std: 8 },
  after: { mean: 26, median: 24.7, std: 10.4 },
  histogram: { edges: [0, 10, 20, 30], before: [0.3, 0.5, 0.2], after: [0.1, 0.4, 0.5] },
  levels: [],
  moves: [],
  appeared: [],
  disappeared: [],
  used_by: [],
  focus: false,
  ...overrides,
});

const changes = {
  before: { label: "2026-10-01", taken_at: "2026-10-01T08:00:00Z", rows: 2000, duplicates: 0 },
  after: { label: "this version", taken_at: "2026-11-01T08:00:00Z", rows: 2100, duplicates: 4 },
  level: "check",
  level_label: "Check",
  headline: "Compared with 2026-10-01: +100 rows (+5.0%), 1 column probably renamed, 2 distributions shifted, 1 new category.",
  reasons: ["1 column probably renamed.", "2 distributions shifted (PSI ≥ 0.1)."],
  rows_change: 100,
  rows_change_pct: 5,
  counts: { added: 0, removed: 0, renamed: 1, retyped: 0, shifted: 2, new_categories: 1 },
  schema: [
    {
      kind: "renamed",
      column: "ca",
      before: "chiffre_affaires",
      after: "ca",
      detail: "Probably renamed (same values): not applied, the old name is gone.",
      used_by: [],
      breaking: false,
    },
  ],
  quality: [
    { column: "prix", missing_before: 0, missing_after: 0, outliers_before: 0.5, outliers_after: 0.6, notable: false },
    { column: "region", missing_before: 0, missing_after: 8, outliers_before: null, outliers_after: null, notable: true },
  ],
  dates: [],
  distributions: [
    shift(),
    shift({
      column: "region",
      type: "categorical_nominal",
      kind: "categorical",
      psi: 0.18,
      drift: "moderate",
      wasserstein: null,
      ks: null,
      ks_p: null,
      chi2: 120.5,
      chi2_p: 1e-6,
      dof: 4,
      before: null,
      after: null,
      histogram: null,
      levels: [
        { value: "North", before: 0.4, after: 0.36, status: "kept", certain: true },
        { value: "Outre-mer", before: 0, after: 0.05, status: "appeared", certain: true },
      ],
      appeared: [{ value: "Outre-mer", before: 0, after: 0.05, status: "appeared", certain: true }],
    }),
  ],
  thresholds: "PSI < 0.1 stable · 0.1–0.25 moderate · > 0.25 strong (Siddiqi, 2006)",
};

const versions = [
  { version: 2, imported_at: "2026-11-01T08:00:00Z", rows: 2100, kept: true, kept_bytes: 0, current: true },
  { version: 1, imported_at: "2026-10-01T08:00:00Z", rows: 2000, kept: true, kept_bytes: 2048, current: false },
];

const rows = {
  available: true,
  reason: null,
  candidates: ["order_id"],
  result: {
    key: "order_id",
    rows_before: 2000,
    rows_after: 2100,
    added: 100,
    removed: 0,
    modified: 12,
    unchanged: 1988,
    columns: [{ column: "prix", modified: 12 }],
    examples: [{ key: "o00001", column: "prix", before: "10.5", after: "11.5" }],
    added_keys: ["o02000"],
    removed_keys: [],
    compared: ["prix", "region"],
  },
};

function setup(extra: Record<string, unknown> = {}) {
  return mockApi({
    "GET /health": health,
    "GET /datasets": [dataset(), dataset({ id: "d2", name: "october" })],
    "GET /datasets/d1": dataset(),
    "GET /datasets/d1/columns": columns,
    "GET /datasets/d1/versions": versions,
    "GET /datasets/d1/changes": changes,
    ...extra,
  });
}

describe("Changes page", () => {
  beforeEach(() => localStorage.clear());
  afterEach(() => vi.unstubAllGlobals());

  it("opens on the summary and the strongest shift", async () => {
    setup();
    renderAt("/d/d1/changes");
    expect(await screen.findByText(changes.headline)).toBeInTheDocument();
    expect(screen.getAllByText("Check").length).toBeGreaterThan(0);
    const tab = screen.getByRole("tab", { name: /Distributions/ });
    expect(tab).toHaveAttribute("aria-selected", "true");
    expect(screen.getByText(changes.thresholds)).toBeInTheDocument();
    expect(screen.getByText("0.620")).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "prix before and after" })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "region" }));
    expect(await screen.findByRole("img", { name: "region before and after" })).toBeInTheDocument();
    expect(screen.getAllByText("Outre-mer").length).toBeGreaterThan(0);
  });

  it("lists probable renames without applying them", async () => {
    setup();
    renderAt("/d/d1/changes?tab=schema");
    expect(await screen.findByText("Probably renamed")).toBeInTheDocument();
    expect(screen.getByText("chiffre_affaires")).toBeInTheDocument();
    expect(screen.getByText(/not applied/)).toBeInTheDocument();
  });

  it("shows rows added, removed and modified with the CSV link", async () => {
    setup({ "GET /datasets/d1/changes/rows": rows });
    renderAt("/d/d1/changes?tab=rows");
    expect(await screen.findByText("Unchanged")).toBeInTheDocument();
    const link = screen.getByRole("link", { name: /Download the differences/ });
    expect(link.getAttribute("href")).toBe("/api/v1/datasets/d1/changes/rows/export?key=order_id");
    expect(screen.getByText("o00001")).toBeInTheDocument();
  });

  it("explains the Rows tab when the previous version was not kept", async () => {
    setup({
      "GET /datasets/d1/changes/rows": {
        available: false,
        reason: "The previous version's rows were not kept, only its statistics.",
        candidates: [],
        result: null,
      },
    });
    renderAt("/d/d1/changes?tab=rows");
    expect(await screen.findByText(/were not kept/)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Open Settings" })).toBeInTheDocument();
  });

  it("compares with another dataset through vs", async () => {
    const calls = setup();
    renderAt("/d/d1/changes?vs=d2");
    await screen.findByText(changes.headline);
    const asked = calls.filter((call) => call.key === "GET /datasets/d1/changes");
    expect(asked.at(-1)?.params.get("vs")).toBe("d2");
    expect((screen.getByRole("combobox") as HTMLSelectElement).value).toBe("d2");
  });

  it("says when there is no previous version", async () => {
    mockApi({
      "GET /health": health,
      "GET /datasets": [dataset()],
      "GET /datasets/d1": dataset(),
      "GET /datasets/d1/columns": columns,
      "GET /datasets/d1/versions": versions.slice(0, 1),
    });
    renderAt("/d/d1/changes");
    expect(await screen.findByText("No previous version yet")).toBeInTheDocument();
  });
});

describe("After a reimport", () => {
  beforeEach(() => localStorage.clear());
  afterEach(() => vi.unstubAllGlobals());

  it("the overview shows the summary until it is closed", async () => {
    setup();
    renderAt("/d/d1");
    const banner = await screen.findByRole("region", { name: "Changes since the previous version" });
    expect(within(banner).getByText(changes.headline)).toBeInTheDocument();
    expect(within(banner).getByRole("link", { name: "See what changed" })).toHaveAttribute("href", "/d/d1/changes");
    fireEvent.click(within(banner).getByRole("button", { name: "Hide until the next reimport" }));
    await waitFor(() => expect(screen.queryByRole("region", { name: "Changes since the previous version" })).toBeNull());
  });
});

describe("Home page", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("compares two ticked datasets, the newest against the oldest", async () => {
    mockApi({
      "GET /health": health,
      "GET /datasets": [
        dataset({ id: "d2", name: "november", imported_at: "2026-11-02T08:00:00Z" }),
        dataset({ id: "d1", name: "october", imported_at: "2026-10-02T08:00:00Z" }),
      ],
      "GET /samples": [],
    });
    const router = renderAt("/");
    fireEvent.click(await screen.findByRole("checkbox", { name: "Select october to compare" }));
    expect(screen.getByText("Tick one more dataset to compare")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("checkbox", { name: "Select november to compare" }));
    fireEvent.click(screen.getByRole("button", { name: "Compare versions" }));
    await waitFor(() => expect(router.state.location.pathname).toBe("/d/d2/changes"));
    expect(router.state.location.search).toEqual({ vs: "d1" });
  });
});
