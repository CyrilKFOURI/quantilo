import { fireEvent, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { renderAt } from "../../test/render";
import { column, roles } from "../../test/columns";
import { dataset, health, mockApi } from "../../test/mockApi";

const preview = {
  kind: "file",
  format: "csv",
  files: [{ name: "typing_traps.csv", format: "csv", size: 1000 }],
  columns: [
    { name: "customer_id", dtype: "VARCHAR" },
    { name: "age", dtype: "BIGINT" },
  ],
  rows: [
    ["C00001", "49"],
    ["C00002", null],
  ],
  row_estimate: 600,
  csv: { delimiter: ";", encoding: "utf-8", header: true, decimal: "." },
  sheets: null,
  schema_mismatch: false,
  suggested_name: "typing_traps",
  warnings: [],
};

const imported = dataset({ id: "d9", name: "traps", row_count: 600, col_count: 1 });

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("import flow", () => {
  it("previews a sample, lets the user tweak options, imports and opens the dataset", async () => {
    const calls = mockApi({
      "GET /health": health,
      "GET /datasets": [],
      "GET /samples": [{ name: "typing_traps", path: "/s/typing_traps.csv", format: "csv", size: 1 }],
      "POST /sources/preview": preview,
      "POST /datasets": { dataset: imported, job: { status: "done" }, reused: false },
      "GET /datasets/d9": imported,
      "POST /datasets/d9/open": imported,
      "GET /datasets/d9/schema": [{ name: "customer_id", dtype: "VARCHAR" }],
      "GET /datasets/d9/head": { columns: ["customer_id"], rows: [["C00001"]] },
      "GET /datasets/d9/columns": [column({ name: "customer_id" })],
      "GET /column-roles": roles,
    });
    renderAt("/");

    fireEvent.click(await screen.findByRole("button", { name: "typing_traps" }));

    expect(await screen.findByText("CSV · ~600 rows · 2 columns")).toBeInTheDocument();
    expect(screen.getByLabelText("Delimiter")).toHaveValue(";");

    fireEvent.click(screen.getByRole("button", { name: /^age/ }));
    expect(screen.getByText("1 of 2 columns selected")).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Name"), { target: { value: "traps" } });
    fireEvent.click(screen.getByRole("button", { name: "Import" }));

    // A fresh import opens on the Columns page to review the detected types.
    expect(await screen.findByRole("heading", { name: "Columns" })).toBeInTheDocument();
    const request = calls.find((call) => call.key === "POST /datasets")?.body as {
      name: string;
      options: { columns: string[]; sample_pct: number };
    };
    expect(request.name).toBe("traps");
    expect(request.options.columns).toEqual(["customer_id"]);
    expect(request.options.sample_pct).toBe(100);
  });

  it("shows a readable preview error", async () => {
    mockApi({
      "GET /health": health,
      "GET /datasets": [],
      "GET /samples": [{ name: "bad", path: "/s/bad.csv", format: "csv", size: 1 }],
    });
    renderAt("/");

    fireEvent.click(await screen.findByRole("button", { name: "bad" }));

    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent("No mock"));
  });

  it("says how the file is read before importing it (092)", async () => {
    mockApi({
      "GET /health": health,
      "GET /datasets": [],
      "GET /samples": [{ name: "german", path: "/s/german.data", format: "csv", size: 1 }],
      "POST /sources/preview": {
        ...preview,
        warnings: ["Columns separated by spaces (21 fields on every line) (R-ING-02)."],
      },
    });
    renderAt("/");

    fireEvent.click(await screen.findByRole("button", { name: "german" }));

    const notes = await screen.findByRole("region", { name: "How the file is read" });
    expect(notes).toHaveTextContent("Columns separated by spaces (21 fields on every line) (R-ING-02).");
    expect(notes).toHaveTextContent("Check that your data was read correctly.");
  });
});

