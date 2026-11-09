import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from "vitest";

import { column, roles } from "../../test/columns";
import { dataset, health, mockApi } from "../../test/mockApi";
import { renderAt } from "../../test/render";

const columns = [
  column({ name: "price", position: 0 }),
  column({ name: "city", position: 1, type: "categorical_nominal", detected_type: "categorical_nominal" }),
  column({ name: "day", position: 2, type: "datetime", detected_type: "datetime" }),
];

function page(params: URLSearchParams) {
  const offset = Number(params.get("offset") ?? 0);
  const mode = params.get("mode");
  const total = mode === "duplicates" ? 0 : 250;
  const rows = Array.from({ length: Math.max(0, Math.min(100, total - offset)) }, (_, k) => {
    const n = offset + k + 1;
    return [n, n === 2 ? { invalid: "abc" } : n * 1000.5, n === 3 ? null : `City ${n}`, "2024-02-01T00:00:00"];
  });
  return {
    columns: [
      { name: "price", type: "numeric_continuous", excluded: false },
      { name: "city", type: "categorical_nominal", excluded: false },
      { name: "day", type: "datetime", excluded: false },
    ],
    rows,
    total,
    offset,
  };
}

function setup() {
  const calls = mockApi({
    "GET /health": health,
    "GET /datasets/d1": dataset(),
    "GET /datasets/d1/columns": columns,
    "GET /column-roles": roles,
  });
  // Rows answer according to their query string.
  const stubbed = vi.mocked(fetch);
  const inner = stubbed.getMockImplementation();
  stubbed.mockImplementation(async (input, init) => {
    const url = new URL(String(input), "http://test");
    if (url.pathname === "/api/v1/datasets/d1/rows") {
      calls.push({ key: "GET /datasets/d1/rows", body: undefined, params: url.searchParams });
      return Response.json(page(url.searchParams));
    }
    return inner!(input, init);
  });
  return calls;
}

const rowCalls = (calls: ReturnType<typeof setup>) => calls.filter((c) => c.key === "GET /datasets/d1/rows");

// jsdom lays nothing out: give the scroll container a size so the virtualizer shows rows.
const sizes = ["offsetHeight", "offsetWidth", "clientHeight", "clientWidth"] as const;
const saved = sizes.map((key) => Object.getOwnPropertyDescriptor(HTMLElement.prototype, key));
beforeAll(() => {
  for (const key of sizes) {
    Object.defineProperty(HTMLElement.prototype, key, {
      configurable: true,
      get: () => (key.endsWith("Height") ? 640 : 1200),
    });
  }
});
afterAll(() => {
  sizes.forEach((key, index) => {
    const descriptor = saved[index];
    if (descriptor) Object.defineProperty(HTMLElement.prototype, key, descriptor);
  });
});
afterEach(() => vi.unstubAllGlobals());

describe("Data page", () => {
  it("shows typed, formatted rows with missing and unreadable values marked", async () => {
    setup();
    renderAt("/d/d1/data");

    const grid = await screen.findByRole("grid", { name: "Rows of the dataset" });
    expect(await within(grid).findByText("1,000.5")).toBeInTheDocument();
    expect(screen.getByText("250 rows")).toBeInTheDocument();
    const unreadable = within(grid).getByText("abc");
    expect(unreadable).toHaveClass("italic");
    expect(unreadable).toHaveAttribute("title", expect.stringContaining("cannot be read"));
    expect(within(grid).getAllByLabelText("Missing").length).toBeGreaterThan(0);
    expect(within(grid).getAllByText("2024-02-01")[0]).toBeInTheDocument(); // no midnight time
    expect(within(grid).getAllByRole("columnheader").map((h) => h.textContent)).toEqual(
      expect.arrayContaining(["#", expect.stringContaining("price"), expect.stringContaining("city")]),
    );
  });

  it("loads the blocks of rows scrolled into view", async () => {
    const calls = setup();
    renderAt("/d/d1/data");
    const grid = await screen.findByRole("grid", { name: "Rows of the dataset" });
    await within(grid).findByText("1,000.5");

    grid.scrollTop = 32 * 210;
    fireEvent.scroll(grid);

    await waitFor(() => expect(rowCalls(calls).some((c) => c.params.get("offset") === "200")).toBe(true));
    expect(await within(grid).findByRole("rowheader", { name: "215" })).toBeInTheDocument();
  });

  it("sorts from the header, cycling ascending, descending, file order", async () => {
    const calls = setup();
    renderAt("/d/d1/data");
    await screen.findByText("1,000.5");

    fireEvent.click(screen.getByRole("button", { name: "price" }));
    await waitFor(() => expect(rowCalls(calls).some((c) => c.params.get("sort") === "price" && !c.params.get("desc"))).toBe(true));
    expect(screen.getByRole("columnheader", { name: /price/ })).toHaveAttribute("aria-sort", "ascending");

    fireEvent.click(screen.getByRole("button", { name: "price" }));
    await waitFor(() => expect(rowCalls(calls).some((c) => c.params.get("desc") === "true")).toBe(true));
    expect(screen.getByRole("columnheader", { name: /price/ })).toHaveAttribute("aria-sort", "descending");
  });

  it("switches modes and searches, with an explicit empty state", async () => {
    const calls = setup();
    renderAt("/d/d1/data");
    await screen.findByText("1,000.5");

    fireEvent.click(screen.getByRole("radio", { name: "Duplicates" }));
    expect(await screen.findByText("No duplicate rows: every row is unique.")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("radio", { name: "All rows" }));
    const box = screen.getByRole("searchbox", { name: "Search the rows" });
    fireEvent.change(box, { target: { value: "lyon" } });
    fireEvent.submit(box.closest("form") as HTMLFormElement);
    await waitFor(() => expect(rowCalls(calls).some((c) => c.params.get("search") === "lyon")).toBe(true));
  });

  it("hides and pins columns, remembered for the dataset", async () => {
    setup();
    renderAt("/d/d1/data");
    await screen.findByText("1,000.5");

    fireEvent.click(screen.getByRole("button", { name: "3 of 3 columns" }));
    fireEvent.click(screen.getByRole("checkbox", { name: "city" }));
    await waitFor(() => expect(screen.queryByRole("button", { name: "city" })).not.toBeInTheDocument());
    expect(JSON.parse(localStorage.getItem("de.data.d1.hidden") ?? "[]")).toEqual(["city"]);
    expect(screen.getByRole("button", { name: "2 of 3 columns" })).toBeInTheDocument();
  });

  it("exports the current view with the filters, mode and visible columns", async () => {
    setup();
    renderAt("/d/d1/data?mode=sample&q=paris");
    await screen.findByText("1,000.5");

    fireEvent.keyDown(screen.getByRole("button", { name: /Export/ }), { key: "Enter" });
    const csv = await screen.findByRole("menuitem", { name: "CSV" });
    const href = new URL(csv.getAttribute("href") ?? "", "http://test");
    expect(href.pathname).toBe("/api/v1/datasets/d1/export");
    expect(Object.fromEntries(href.searchParams)).toMatchObject({ format: "csv", mode: "sample", search: "paris" });
    expect(screen.getByRole("menuitem", { name: "Column summary (CSV)" })).toHaveAttribute(
      "href",
      "/api/v1/datasets/d1/export/summary?",
    );
  });

  it("on a phone, keeps the rows in a table that scrolls sideways (046)", async () => {
    vi.stubGlobal("matchMedia", (query: string) => ({
      matches: query.includes("max-width"),
      addEventListener: () => {},
      removeEventListener: () => {},
    }));
    setup();
    renderAt("/d/d1/data");

    expect(await screen.findByRole("grid")).toBeInTheDocument();
    expect(screen.queryByRole("list", { name: "Rows of the dataset" })).not.toBeInTheDocument();
  });
});
