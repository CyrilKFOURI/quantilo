import { screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { column, roles } from "../../test/columns";
import { dataset, health, mockApi } from "../../test/mockApi";
import { renderAt } from "../../test/render";
import { groupEntry, groupOf } from "./PageGroups";

afterEach(() => vi.unstubAllGlobals());

describe("Grouped pages (046)", () => {
  it("knows which group a page of the dataset belongs to", () => {
    expect(groupOf("/d/d1")).toBe("overview");
    expect(groupOf("/d/d1/columns")).toBe("overview");
    expect(groupOf("/d/d1/data")).toBe("overview");
    expect(groupOf("/d/d1/changes")).toBe("overview");
    expect(groupOf("/d/d1/compare")).toBe("relationships");
    expect(groupOf("/d/d1/correlations")).toBe("relationships");
    expect(groupOf("/d/d1/regression")).toBe("relationships");
    expect(groupOf("/d/d1/ml")).toBe("ml");
    expect(groupOf("/d/d1/segments")).toBe("ml");
    expect(groupOf("/d/d1/kpi")).toBeNull();
    expect(groupOf("/recipes")).toBeNull();
    expect(groupEntry("relationships")).toBe("/d/$datasetId/compare");
  });

  it("lists two dataset entries and the analyses, and shows the group's tabs on a page", async () => {
    mockApi({
      "GET /health": health,
      "GET /datasets": [dataset()],
      "GET /datasets/d1": dataset(),
      "GET /datasets/d1/columns": [column({ name: "income" })],
      "GET /column-roles": roles,
    });
    renderAt("/d/d1/columns");

    const nav = await screen.findByRole("navigation", { name: "Main navigation" });
    expect(within(nav).getByRole("link", { name: "Overview" })).toHaveAttribute("aria-current", "page");
    expect(within(nav).getByRole("link", { name: "Relationships" })).toBeInTheDocument();
    for (const gone of ["Columns", "Data", "Changes", "Compare", "Correlations", "Regression", "Segments"]) {
      expect(within(nav).queryByRole("link", { name: gone })).not.toBeInTheDocument();
    }

    const tabs = screen.getByRole("navigation", { name: "Sections of this page" });
    expect(within(tabs).getAllByRole("link").map((link) => link.textContent)).toEqual(["Summary", "Columns", "Data", "Changes"]);
    expect(within(tabs).getByRole("link", { name: "Columns" })).toHaveAttribute("aria-current", "page");
  });
});
