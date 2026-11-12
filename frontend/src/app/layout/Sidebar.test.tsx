import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { CURRENT_DATASET_KEY } from "../../features/datasets/current";
import { column, roles } from "../../test/columns";
import { dataset, health, mockApi } from "../../test/mockApi";
import { renderAt } from "../../test/render";

afterEach(() => vi.unstubAllGlobals());

function setup(datasets = [dataset()]) {
  return mockApi({
    "GET /health": health,
    "GET /datasets": datasets,
    "GET /datasets/d1": dataset(),
    "GET /datasets/d1/columns": [
      column({ name: "day", type: "datetime", detected_type: "datetime", role: null }),
      column({ name: "units", position: 1 }),
    ],
    "GET /column-roles": roles,
    "GET /samples": [],
  });
}

const nav = () => screen.findByRole("navigation", { name: "Main navigation" });

describe("Main navigation (098)", () => {
  it("keeps the main pages, then the Toolbox last; Time series and Experiments are tools", async () => {
    setup();
    renderAt("/d/d1/columns");

    const sidebar = await nav();
    const names = within(sidebar)
      .getAllByRole("link")
      .slice(1) // the logo
      .map((link) => link.textContent);
    expect(names).toEqual([
      "Home",
      "Recipes",
      "Overview",
      "Clean",
      "Relationships",
      "KPI",
      "Machine learning",
      "Synthetic data",
      "Guide",
      "Settings",
    ]);
    expect(within(sidebar).queryByRole("link", { name: "Time series" })).not.toBeInTheDocument();
    expect(within(sidebar).queryByRole("link", { name: "Experiments" })).not.toBeInTheDocument();
  });

  it("marks the Toolbox while a tool without an entry of its own is open", async () => {
    setup();
    renderAt("/d/d1/experiments?tab=plan");

    const sidebar = await nav();
    expect(within(sidebar).getByRole("button", { name: "Toolbox" })).toHaveAttribute("aria-current", "page");
  });

  it("keeps the dataset open while browsing Home, until it is closed", async () => {
    setup();
    const router = renderAt("/d/d1/columns");
    const sidebar = await nav();
    await screen.findAllByText("sales");

    fireEvent.click(within(sidebar).getByRole("link", { name: "Home" }));
    await waitFor(() => expect(router.state.location.pathname).toBe("/"));
    // the top bar still names it, the dataset entries still lead back to it
    expect(await screen.findByRole("button", { name: "Close the dataset" })).toBeInTheDocument();
    expect(within(sidebar).getByRole("link", { name: "Overview" })).toHaveAttribute("href", "/d/d1");
    fireEvent.click(within(sidebar).getByRole("button", { name: "Toolbox" }));
    const menu = within(sidebar).getByRole("dialog", { name: "Toolbox" });
    expect(within(menu).getAllByRole("link").map((link) => link.getAttribute("href"))).toContain("/d/d1/compare");
    fireEvent.keyDown(document, { key: "Escape" });
    expect(localStorage.getItem(CURRENT_DATASET_KEY)).toBe("d1");

    fireEvent.click(screen.getByRole("button", { name: "Close the dataset" }));
    expect(await screen.findByText("No dataset open")).toBeInTheDocument();
    expect(within(sidebar).queryByRole("link", { name: "Overview" })).not.toBeInTheDocument();
    expect(localStorage.getItem(CURRENT_DATASET_KEY)).toBeNull();
  });

  it("reopens on the dataset kept in this browser", async () => {
    localStorage.setItem(CURRENT_DATASET_KEY, "d1");
    setup();
    renderAt("/recipes");
    const sidebar = await nav();
    await waitFor(() => expect(within(sidebar).getByRole("link", { name: "Overview" })).toHaveAttribute("href", "/d/d1"));
  });

  it("forgets a dataset that no longer exists", async () => {
    localStorage.setItem(CURRENT_DATASET_KEY, "gone");
    setup();
    renderAt("/");
    expect(await screen.findByText("No dataset open")).toBeInTheDocument();
    await waitFor(() => expect(localStorage.getItem(CURRENT_DATASET_KEY)).toBeNull());
  });
});
