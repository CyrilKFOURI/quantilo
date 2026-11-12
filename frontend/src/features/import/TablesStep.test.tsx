import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { column, roles } from "../../test/columns";
import { dataset, health, mockApi } from "../../test/mockApi";
import { renderAt } from "../../test/render";
import { importStepFor } from "./stepTypes";

const table = (id: string, overrides: Record<string, unknown> = {}) => ({
  id,
  name: `stock - ${id}`,
  location: id,
  rows: 8,
  columns: 3,
  level: "high",
  confidence: 0.9,
  summary: `Table ${id.split("!")[1]}, header row 1, 8 rows.`,
  selected: true,
  kind: "table",
  sheet: "Two",
  parent: null,
  area: { top: 1, left: 1, bottom: 9, right: 3, header_rows: [1], first_data_row: 2, dropped_rows: [] },
  reasons: [{ rule: "C3", weight: 0.1, message: "Styled header above a typed body." }],
  warnings: [],
  proposals: [],
  ...overrides,
});

const workbook = {
  kind: "file",
  format: "excel",
  files: [{ name: "stock.xlsx", format: "excel", size: 1000 }],
  columns: [{ name: "Product", dtype: "VARCHAR" }],
  rows: [["P0"]],
  row_estimate: 8,
  csv: null,
  sheets: ["Two"],
  schema_mismatch: false,
  suggested_name: "stock",
  warnings: [],
  header_row: null,
  detected: { kind: "table", format: "excel", label: "Excel workbook (.xlsx)", rule: "R-DET-01", reason: "", mismatch: false },
  tables: [
    table("Two!A1:C9"),
    table("Two!E1:F4", {
      rows: 3,
      level: "review",
      selected: false,
      area: { top: 1, left: 5, bottom: 4, right: 6, header_rows: [1], first_data_row: 2, dropped_rows: [] },
    }),
  ],
  table: null,
  links: [],
};

const grid = {
  kind: "sheet",
  sheet: "Two",
  cells: [
    ["Product", "Units", "Price", null, "Region", "Target"],
    ["P0", "0", "0", null, "R0", "100"],
  ],
  merges: [],
  text: null,
  truncated: false,
};

afterEach(() => vi.unstubAllGlobals());

describe("import step choice (082, 080b)", () => {
  it("keeps the one-table dialog for a single sure table and a plain CSV", () => {
    const sure = { ...workbook, tables: [table("Two!A1:C9")] } as never;
    expect(importStepFor(sure, "auto")).toBe("default");
    expect(importStepFor(sure, "tables")).toBe("tables");
    expect(importStepFor(workbook as never, "auto")).toBe("tables");
    expect(importStepFor(workbook as never, "simple")).toBe("default");
    expect(importStepFor({ ...workbook, format: "csv", tables: null } as never, "auto")).toBe("default");
    expect(importStepFor({ ...workbook, detected: { kind: "image" } } as never, "auto")).toBe("image");
  });
});

describe("tables step", () => {
  it("shows the original and the result, then imports the ticked tables as one source", async () => {
    const imported = dataset({ id: "d7", name: "stock - Two!A1:C9", types_reviewed_at: null });
    const calls = mockApi({
      "GET /health": health,
      "GET /datasets": [],
      "GET /samples": [{ name: "stock", path: "/s/stock.xlsx", format: "excel", size: 1 }],
      "POST /sources/preview": (body: unknown) => {
        const options = (body as { options: { table?: string | null } }).options;
        if (!options.table) return workbook;
        return {
          ...workbook,
          table: options.table,
          columns: [{ name: "Region", dtype: "VARCHAR" }],
          rows: [["R0"]],
          row_estimate: 3,
        };
      },
      "POST /sources/original": grid,
      "POST /sources/import": {
        source: { id: "s1", kind: "workbook", name: "stock", source_path: "/s/stock.xlsx", size_bytes: 1, created_at: "", dataset_ids: ["d7", "d8"] },
        imports: [{ dataset: imported, job: { status: "done" }, reused: false }],
      },
      "GET /datasets/d7": imported,
      "POST /datasets/d7/open": imported,
      "GET /datasets/d7/schema": [{ name: "Product", dtype: "VARCHAR" }],
      "GET /datasets/d7/head": { columns: ["Product"], rows: [["P0"]] },
      "GET /datasets/d7/columns": [column({ name: "Product" })],
      "GET /column-roles": roles,
    });
    renderAt("/");

    fireEvent.click(await screen.findByRole("button", { name: "stock" }));

    expect(await screen.findByText(/2 tables found in stock.xlsx/)).toBeInTheDocument();
    const original = await screen.findByRole("region", { name: "Original" });
    expect(within(original).getByText("Region")).toBeInTheDocument();
    // the second table is not ticked (review); ticking it and showing it gives its result
    const second = screen.getByRole("checkbox", { name: "stock - Two!E1:F4" });
    expect(second).not.toBeChecked();
    fireEvent.click(second);
    fireEvent.click(screen.getByText("Two!E1:F4"));
    expect(await screen.findByText("Table E1:F4, header row 1, 8 rows.")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Import selected (2)" }));

    expect(await screen.findByRole("heading", { name: "Columns" })).toBeInTheDocument();
    const request = calls.find((call) => call.key === "POST /sources/import")?.body as {
      tables: { id: string; name: string }[];
    };
    expect(request.tables.map((item) => item.id)).toEqual(["Two!A1:C9", "Two!E1:F4"]);
  });

  it("goes back to whole sheets on request", async () => {
    mockApi({
      "GET /health": health,
      "GET /datasets": [],
      "GET /samples": [{ name: "stock", path: "/s/stock.xlsx", format: "excel", size: 1 }],
      "POST /sources/preview": workbook,
      "POST /sources/original": grid,
    });
    renderAt("/");

    fireEvent.click(await screen.findByRole("button", { name: "stock" }));
    fireEvent.click(await screen.findByRole("button", { name: "Import whole sheets instead" }));

    await waitFor(() => expect(screen.getByRole("button", { name: "Choose tables…" })).toBeInTheDocument());
    expect(screen.getByRole("button", { name: "Import" })).toBeInTheDocument();
  });
});
