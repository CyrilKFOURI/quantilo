import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { dataset, health, mockApi } from "../../test/mockApi";
import { renderAt } from "../../test/render";
import type { Model, ModelTable, Relation } from "./api";
import { reachedFrom } from "./BuildDialog";
import { summary } from "./labels";
import { CARD_H, CARD_W, edgeBetween, planeSize, starLayout } from "./layout";

const column = (name: string, category: string, extra: Record<string, unknown> = {}) => ({
  name,
  type: category === "measure" ? "numeric_continuous" : "categorical_nominal",
  role: null,
  category,
  linkable: category === "key",
  excluded_by: null,
  default_pick: category === "attribute" || category === "measure",
  ...extra,
});

const table = (id: string, name: string, role: ModelTable["role"], depth: number | null, columns: ReturnType<typeof column>[], key: string[]): ModelTable =>
  ({
    id,
    root_id: id,
    name,
    rows: id === "sales" ? 2155 : 77,
    columns,
    keys: [{ columns: key, primary: true, reasons: [{ code: "R-KEY-01", text: "unique" }] }],
    measures: columns.filter((item) => item.category === "measure").length,
    attributes: columns.filter((item) => item.category === "attribute").length,
    dates: 0,
    role,
    depth,
    primary: role === "fact",
    role_reasons: [{ code: "R-ROLE-01", text: "refers to customers" }],
    versions: [{ id, seq: 0 }],
  }) as ModelTable;

const check = (orphans = 0) => ({
  child_rows: 2155,
  child_empty: 0,
  matched_rows: 2155 - orphans,
  orphan_rows: orphans,
  orphans: orphans ? [{ value: "C999", rows: orphans }] : [],
  child_distinct: 89,
  parent_rows: 91,
  parent_distinct: 91,
  duplicate_keys: 0,
  duplicates: [],
  unused_keys: 2,
  cardinality: "n:1" as const,
  recommended: "left" as const,
  reasons: [{ code: "R-JOIN-01", text: "LEFT JOIN keeps all 2 155 sales rows" }],
  warnings: [],
});

const relation = (id: string, parent: string, columns: [string, string], status: Relation["status"], orphans = 0): Relation => ({
  id,
  status,
  origin: "detected",
  child_id: "sales",
  child_columns: [columns[0]],
  parent_id: parent,
  parent_columns: [columns[1]],
  cardinality: "n:1",
  join_type: "left",
  recommended: "left",
  confidence: status === "suggested" ? "high" : null,
  score: 1,
  inclusion: 1,
  reasons: [{ code: "R-REL-02", text: `${columns[0]} and ${columns[1]} name the same thing` }],
  check: check(orphans),
  problem: null,
});

function makeModel(relations: Relation[]): Model {
  return {
    project_id: "p1",
    tables: [
      table("sales", "sales", "fact", 0, [column("customer_id", "key"), column("product_id", "key"), column("amount", "measure")], ["sale_id"]),
      table("customers", "customers", "dimension", 1, [column("customer_id", "key"), column("country", "attribute"), column("phone", "attribute", { default_pick: false })], ["customer_id"]),
      table("products", "products", "dimension", 1, [column("product_id", "key"), column("family", "attribute")], ["product_id"]),
    ],
    relations,
    facts: ["sales"],
    shape: "star",
    positions: {},
    builds: [],
    notes: [],
    elapsed_ms: 12,
  };
}

const project = { id: "p1", name: "Shop", created_at: "", updated_at: "", dataset_ids: ["sales", "customers", "products"] };

function setup(model: Model, extra: Record<string, unknown> = {}) {
  return mockApi({
    "GET /health": health,
    "GET /datasets": [dataset({ id: "sales", name: "sales" }), dataset({ id: "customers", name: "customers" }), dataset({ id: "products", name: "products" })],
    "GET /projects": [project],
    "GET /modules/clean/lineage": [],
    "GET /projects/p1/model": model,
    ...extra,
  });
}

afterEach(() => {
  vi.unstubAllGlobals();
  localStorage.clear();
});

describe("layout (088)", () => {
  it("puts the fact in the centre and its dimensions around it", () => {
    const tables = [
      { id: "f", root_id: "f", role: "fact" as const, depth: 0, primary: true, name: "sales" },
      { id: "a", root_id: "a", role: "dimension" as const, depth: 1, primary: false, name: "customers" },
      { id: "b", root_id: "b", role: "dimension" as const, depth: 1, primary: false, name: "products" },
      { id: "c", root_id: "c", role: "dimension" as const, depth: 2, primary: false, name: "categories" },
      { id: "u", root_id: "u", role: "unrelated" as const, depth: null, primary: false, name: "notes" },
    ];
    const positions = starLayout(tables, [
      { child_id: "f", parent_id: "a" },
      { child_id: "f", parent_id: "b" },
      { child_id: "b", parent_id: "c" },
    ], ["f"]);
    const centre = (id: string) => ({ x: positions[id]!.x + CARD_W / 2, y: positions[id]!.y + CARD_H / 2 });
    // two dimensions: left and right of the fact, the snowflake further out on its side
    expect(centre("a").x).toBeLessThan(centre("f").x);
    expect(centre("b").x).toBeGreaterThan(centre("f").x);
    expect(centre("c").x).toBeGreaterThan(centre("b").x);
    // the unrelated table under everything
    expect(positions.u!.y).toBeGreaterThan(Math.max(positions.a!.y, positions.b!.y, positions.c!.y));
    // nothing overlaps, everything starts inside the plane
    const boxes = Object.values(positions);
    for (const [index, box] of boxes.entries()) {
      expect(box.x).toBeGreaterThanOrEqual(0);
      for (const other of boxes.slice(index + 1)) {
        const apart = Math.abs(box.x - other.x) >= CARD_W || Math.abs(box.y - other.y) >= CARD_H;
        expect(apart).toBe(true);
      }
    }
    expect(planeSize(boxes).width).toBeGreaterThan(3 * CARD_W);
  });

  it("clips a line at the cards' edges", () => {
    const edge = edgeBetween({ x: 0, y: 0 }, { x: 400, y: 0 });
    expect(edge.start).toEqual({ x: CARD_W, y: CARD_H / 2 });
    expect(edge.end).toEqual({ x: 400, y: CARD_H / 2 });
    expect(edgeBetween({ x: 0, y: 0 }, { x: 400, y: 0 }, 16).start.y).toBe(CARD_H / 2 + 16);
  });
});

describe("words (088)", () => {
  it("says what the diagram shows", () => {
    const model = makeModel([relation("s-1", "customers", ["customer_id", "customer_id"], "suggested")]);
    expect(summary(model)).toBe("sales is the fact table: 1 relation in a star of 2 dimensions. 1 to review.");
  });

  it("follows accepted relations from the fact to pick dimensions", () => {
    const model = makeModel([
      relation("r1", "customers", ["customer_id", "customer_id"], "accepted"),
      relation("s-2", "products", ["product_id", "product_id"], "suggested"),
    ]);
    expect(reachedFrom(model, "sales").map((item) => item.table.id)).toEqual(["customers"]);
  });
});

describe("Model page (088)", () => {
  it("draws the tables and relations, and explains a relation", async () => {
    setup(makeModel([relation("s-1", "customers", ["customer_id", "customer_id"], "suggested", 61)]));
    renderAt("/projects/p1/model");
    expect(await screen.findByText("sales is the fact table: 1 relation in a star of 2 dimensions. 1 to review.")).toBeInTheDocument();
    const diagram = screen.getByRole("group", { name: "Diagram of the tables and their relations" });
    expect(within(diagram).getByRole("button", { name: "Show sales" })).toBeInTheDocument();
    fireEvent.click(within(diagram).getByRole("button", { name: /sales\.customer_id to customers\.customer_id/ }));
    expect(await screen.findByRole("heading", { name: "sales.customer_id to customers.customer_id" })).toBeInTheDocument();
    expect(screen.getByText("61")).toBeInTheDocument(); // rows without a match
    expect(screen.getByText("C999")).toBeInTheDocument();
    expect(screen.getByText("LEFT JOIN keeps all 2 155 sales rows")).toBeInTheDocument();
  });

  it("accepts every proposal in one click", async () => {
    const suggested = [
      relation("s-1", "customers", ["customer_id", "customer_id"], "suggested"),
      relation("s-2", "products", ["product_id", "product_id"], "suggested"),
    ];
    const accepted = makeModel(suggested.map((item, index) => ({ ...item, id: `r${index}`, status: "accepted" as const, confidence: null })));
    const calls = setup(makeModel(suggested), { "POST /projects/p1/model/relations": accepted });
    renderAt("/projects/p1/model");
    fireEvent.click(await screen.findByRole("button", { name: "Accept all (2)" }));
    await waitFor(() => expect(calls.filter((call) => call.key === "POST /projects/p1/model/relations")).toHaveLength(2));
    expect(calls.find((call) => call.key === "POST /projects/p1/model/relations")?.body).toMatchObject({
      child_id: "sales",
      child_columns: ["customer_id"],
      status: "accepted",
    });
    expect(await screen.findByRole("button", { name: "Build analysis dataset" })).toBeEnabled();
  });

  it("builds the analysis dataset with the default columns and shows the counts", async () => {
    const model = makeModel([relation("r1", "customers", ["customer_id", "customer_id"], "accepted")]);
    const preview = {
      fact_id: "sales",
      rows_before: 2155,
      rows_after: 2155,
      columns: [{ name: "customers_country", table_id: "customers", column: "country" }],
      steps: [],
      refused: [],
      skipped: [],
      name: "sales · analysis",
    };
    const calls = setup(model, {
      "POST /projects/p1/model/preview": preview,
      "POST /projects/p1/model/build": { dataset_id: "built" },
    });
    renderAt("/projects/p1/model");
    fireEvent.click(await screen.findByRole("button", { name: "Build analysis dataset" }));
    const dialog = await screen.findByRole("dialog");
    expect(await within(dialog).findByText("Every fact row is kept, none is repeated.")).toBeInTheDocument();
    // R-BLD-02: country picked, phone (contact) not, the key never offered
    expect(within(dialog).getByRole("checkbox", { name: /country/ })).toBeChecked();
    expect(within(dialog).getByRole("checkbox", { name: /phone/ })).not.toBeChecked();
    expect(within(dialog).queryByRole("checkbox", { name: /customer_id/ })).not.toBeInTheDocument();
    expect(calls.find((call) => call.key === "POST /projects/p1/model/preview")?.body).toEqual({
      fact_id: "sales",
      picks: { customers: ["country"] },
    });
    fireEvent.click(within(dialog).getByRole("button", { name: "Build" }));
    expect(await within(dialog).findByRole("link", { name: "Machine learning" })).toHaveAttribute("href", "/d/built/ml");
    expect(calls.find((call) => call.key === "POST /projects/p1/model/build")?.body).toMatchObject({ name: "sales · analysis" });
  });

  it("offers the model from the project on the home page", async () => {
    setup(makeModel([]));
    renderAt("/");
    const link = await screen.findByRole("link", { name: "Model" });
    expect(link).toHaveAttribute("href", "/projects/p1/model");
  });
});
