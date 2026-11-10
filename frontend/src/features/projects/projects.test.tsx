import { createEvent, fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { dataset, health, mockApi } from "../../test/mockApi";
import { renderAt } from "../../test/render";
import { groupFamilies } from "../../modules/clean/lineage";
import type { Project } from "./api";
import { moveLocally } from "./api";
import { familyIds, groupByProject } from "./grouping";

const project = (id: string, name: string, ids: string[]): Project => ({
  id,
  name,
  created_at: "2026-11-10T10:00:00Z",
  updated_at: "2026-11-10T10:00:00Z",
  dataset_ids: ids,
});
const version = (child: string, parent: string, seq: number) => ({
  dataset_id: child,
  parent_id: parent,
  seq,
  origin: "user",
  steps: 1,
  created_at: "2026-11-10T10:00:00Z",
  out_of_date: false,
});

const datasets = [
  dataset({ id: "d1", name: "sales" }),
  dataset({ id: "d2", name: "sales · Cleaned v1" }),
  dataset({ id: "d3", name: "churn" }),
  dataset({ id: "d4", name: "web logs" }),
];
const lineage = [version("d2", "d1", 1)];

function setup(projects: Project[], extra: Record<string, unknown> = {}) {
  return mockApi({
    "GET /health": health,
    "GET /samples": [],
    "GET /datasets": datasets,
    "GET /modules/clean/lineage": lineage,
    "GET /projects": projects,
    "POST /projects/move": (body: unknown) => {
      const { dataset_ids, project_id } = body as { dataset_ids: string[]; project_id: string | null };
      return moveLocally(projects, dataset_ids, project_id);
    },
    ...extra,
  });
}

afterEach(() => {
  vi.unstubAllGlobals();
  localStorage.clear();
});

describe("grouping (087)", () => {
  it("puts a cleaned version in its original's group, whatever the members say", () => {
    const rows = groupFamilies(datasets as never, lineage);
    const groups = groupByProject(rows, [project("p1", "Churn", ["d3"]), project("p2", "Sales", ["d1"])]);
    expect(groups.map((group) => [group.project?.name ?? null, group.rows.map((row) => row.dataset.id)])).toEqual([
      ["Churn", ["d3"]],
      ["Sales", ["d1", "d2"]],
      [null, ["d4"]],
    ]);
    // moving a version moves its whole family (R-PRJ-04)
    expect(familyIds(rows[1]!, rows)).toEqual(["d1", "d2"]);
  });
});

describe("home page with projects (087)", () => {
  it("shows the plain list while there is no project", async () => {
    setup([]);
    renderAt("/");
    expect(await screen.findByRole("link", { name: "churn" })).toBeInTheDocument();
    expect(screen.queryByText("Not in a project")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "New project" })).toBeInTheDocument();
  });

  it("groups datasets by project, with counts, an empty project and Not in a project last", async () => {
    setup([project("p1", "Churn", ["d3"]), project("p2", "Empty one", [])]);
    renderAt("/");
    const groups = await screen.findAllByRole("rowgroup");
    const headers = groups.map((group) => group.querySelector("th[scope=rowgroup]")?.textContent ?? "");
    expect(headers.filter(Boolean)).toEqual(["Churn1 dataset", "Empty one0 datasets", "Not in a project3 datasets"]);
    expect(screen.getByText(/No datasets yet\. Drag one here/)).toBeInTheDocument();
    // folding hides the rows, not the header
    fireEvent.click(screen.getByRole("button", { name: "Hide the datasets of Churn" }));
    expect(screen.queryByRole("link", { name: "churn" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Show the datasets of Churn" })).toHaveAttribute("aria-expanded", "false");
  });

  it("moves a dataset with its Move to… menu, its cleaned version with it", async () => {
    const calls = setup([project("p1", "Churn", ["d3"])]);
    renderAt("/");
    const trigger = await screen.findByRole("button", { name: "Move sales to a project" });
    fireEvent.keyDown(trigger, { key: "Enter" });
    const menu = await screen.findByRole("menu");
    expect(within(menu).getByRole("menuitem", { name: /Not in a project/ })).toHaveAttribute("data-disabled");
    fireEvent.click(within(menu).getByRole("menuitem", { name: "Churn" }));
    await waitFor(() => expect(calls.some((call) => call.key === "POST /projects/move")).toBe(true));
    expect(calls.find((call) => call.key === "POST /projects/move")?.body).toEqual({
      dataset_ids: ["d1", "d2"],
      project_id: "p1",
    });
    expect(await screen.findByText("3 datasets")).toBeInTheDocument();
  });

  it("moves a dataset by drag and drop onto another group", async () => {
    const calls = setup([project("p1", "Churn", ["d3"])]);
    renderAt("/");
    const row = (await screen.findByRole("link", { name: "web logs" })).closest("tr")!;
    expect(row).toHaveAttribute("draggable", "true");
    const store = new Map<string, string>();
    const dataTransfer = {
      types: ["application/x-quantilo-dataset"],
      setData: (type: string, value: string) => store.set(type, value),
      getData: (type: string) => store.get(type) ?? "",
      effectAllowed: "",
      dropEffect: "",
    };
    fireEvent.dragStart(row, { dataTransfer });
    const target = screen.getByRole("link", { name: "Churn" }).closest("tbody")!;
    const over = createEvent.dragOver(target, { dataTransfer });
    fireEvent(target, over);
    expect(over.defaultPrevented).toBe(true);
    expect(target).toHaveAttribute("data-drop", "true");
    expect(within(target).getByText("Drop to move here")).toBeInTheDocument();
    fireEvent.drop(target, { dataTransfer });
    await waitFor(() =>
      expect(calls.find((call) => call.key === "POST /projects/move")?.body).toEqual({
        dataset_ids: ["d4"],
        project_id: "p1",
      }),
    );
  });

  it("says that deleting a project keeps its datasets", async () => {
    const calls = setup([project("p1", "Churn", ["d3"])], { "DELETE /projects/p1": null });
    renderAt("/");
    fireEvent.keyDown(await screen.findByRole("button", { name: "Actions for Churn" }), { key: "Enter" });
    fireEvent.click(await screen.findByRole("menuitem", { name: "Delete project" }));
    expect(await screen.findByText("The dataset in it is kept and moves to Not in a project.")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Delete project" }));
    await waitFor(() => expect(calls.some((call) => call.key === "DELETE /projects/p1")).toBe(true));
  });

  it("creates a project inline", async () => {
    const calls = setup([], { "POST /projects": project("p9", "Q3", []) });
    renderAt("/");
    fireEvent.click(await screen.findByRole("button", { name: "New project" }));
    const input = screen.getByRole("textbox", { name: "New project name" });
    fireEvent.change(input, { target: { value: "  Q3 " } });
    fireEvent.submit(input.closest("form")!);
    await waitFor(() => expect(calls.find((call) => call.key === "POST /projects")?.body).toEqual({ name: "Q3" }));
  });

  it("opens one project: breadcrumb, its datasets only", async () => {
    setup([project("p1", "Churn", ["d3"])]);
    renderAt("/?project=p1");
    expect(await screen.findByRole("heading", { name: "Churn" })).toBeInTheDocument();
    expect(screen.getByRole("navigation", { name: "Breadcrumb" })).toHaveTextContent("All datasets");
    expect(await screen.findByRole("link", { name: "churn" })).toBeInTheDocument();
    expect(screen.queryByRole("link", { name: "web logs" })).not.toBeInTheDocument();
  });
});

describe("top bar (087)", () => {
  it("shows the project before the dataset and lists its other datasets", async () => {
    setup([project("p1", "Churn", ["d1", "d2", "d3"])], {
      "GET /datasets/d3": datasets[2],
      "POST /datasets/d3/open": datasets[2],
    });
    renderAt("/d/d3/columns");
    const crumb = await screen.findByRole("link", { name: "Churn" });
    expect(crumb).toHaveAttribute("href", "/?project=p1");
    const switcher = await screen.findByRole("combobox", { name: "Switch dataset" });
    const group = within(switcher).getByRole("group", { name: "In Churn" });
    // the original only: its cleaned version is reached from it
    expect(within(group).getAllByRole("option").map((option) => option.textContent)).toEqual(["sales"]);
  });
});

describe("import from an open project (087, R-PRJ-05)", () => {
  it("offers the project, ticked; the request carries it", async () => {
    const preview = {
      kind: "file",
      format: "csv",
      files: [{ name: "q3.csv", format: "csv", size: 10 }],
      columns: [{ name: "id", dtype: "BIGINT" }],
      rows: [["1"]],
      row_estimate: 1,
      csv: { delimiter: ",", encoding: "utf-8", header: true, decimal: "." },
      sheets: null,
      schema_mismatch: false,
      suggested_name: "q3",
      warnings: [],
    };
    const calls = mockApi({
      "GET /health": health,
      "GET /datasets": [],
      "GET /samples": [{ name: "q3", path: "/s/q3.csv", format: "csv", size: 1 }],
      "GET /projects": [project("p1", "Churn", [])],
      "POST /sources/preview": preview,
      "POST /datasets": { dataset: dataset({ id: "d9" }), job: null, reused: false },
    });
    renderAt("/?project=p1");
    fireEvent.click(await screen.findByRole("button", { name: "q3" }));
    const box = await screen.findByRole("checkbox", { name: "Add to Churn" });
    expect(box).toBeChecked();
    fireEvent.click(screen.getByRole("button", { name: "Import" }));
    await waitFor(() =>
      expect((calls.find((call) => call.key === "POST /datasets")?.body as { project_id?: string }).project_id).toBe("p1"),
    );
  });
});
