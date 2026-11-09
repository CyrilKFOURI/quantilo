import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { dataset, health, mockApi } from "../test/mockApi";
import { renderAt } from "../test/render";

const baseRoutes = {
  "GET /health": health,
  "GET /datasets": [],
  "GET /samples": [],
};

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("App shell", () => {
  it("renders the home page, silent about a working backend", async () => {
    mockApi(baseRoutes);
    renderAt("/");

    expect(
      await screen.findByRole("heading", { name: "Datasets" }),
    ).toBeInTheDocument();
    expect(
      screen.getByText("Drop a CSV, Parquet or Excel file"),
    ).toBeInTheDocument();
    await waitFor(() => expect(screen.queryByRole("status")).not.toBeInTheDocument());
    expect(screen.queryByText(/Backend connected/)).not.toBeInTheDocument();
  });

  it("explains when the backend is unreachable", async () => {
    mockApi({ ...baseRoutes, "GET /health": new TypeError("Failed to fetch") });
    renderAt("/");

    expect(
      await screen.findByText(
        "Backend unreachable. Start it with scripts/dev.ps1.",
      ),
    ).toBeInTheDocument();
  });

  it("lists recent datasets with their status", async () => {
    mockApi({
      ...baseRoutes,
      "GET /datasets": [
        dataset(),
        dataset({ id: "d2", name: "old", source_state: "changed" }),
        dataset({
          id: "d3",
          name: "broken",
          status: "error",
          error: "Bad file",
        }),
      ],
    });
    renderAt("/");

    expect(
      await screen.findByRole("link", { name: "sales" }),
    ).toBeInTheDocument();
    expect(screen.getByText("Source changed")).toBeInTheDocument();
    expect(screen.getByText("Failed")).toHaveAttribute("title", "Bad file");
    expect(screen.getAllByText("1,200 × 3")).toHaveLength(3);
  });

  it("renders the settings page", async () => {
    mockApi(baseRoutes);
    renderAt("/settings");

    expect(
      await screen.findByRole("heading", { name: "Settings" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("radiogroup", { name: "Density" }),
    ).toBeInTheDocument();
  });

  it("renders a not-found page for unknown routes", async () => {
    mockApi(baseRoutes);
    renderAt("/nope");

    expect(
      await screen.findByRole("heading", { name: "Page not found" }),
    ).toBeInTheDocument();
  });

  it("lists dataset modules in the sidebar, greyed out without a dataset", async () => {
    mockApi(baseRoutes);
    renderAt("/");

    await screen.findByRole("heading", { name: "Datasets" });
    const item = screen.getByText("Time series").closest("[aria-disabled]");
    expect(item).toHaveAttribute("title", "Open a dataset first");
  });

  it("drops a module that is no longer registered", async () => {
    mockApi(baseRoutes);
    renderAt("/", []);

    await screen.findByRole("heading", { name: "Datasets" });
    expect(screen.queryByText("Time series")).not.toBeInTheDocument();
  });

  it("on a phone, opens the navigation in a drawer that a link closes (041)", async () => {
    vi.stubGlobal("matchMedia", (query: string) => ({
      matches: query.includes("max-width"),
      addEventListener: () => {},
      removeEventListener: () => {},
    }));
    mockApi(baseRoutes);
    renderAt("/");

    await screen.findByRole("heading", { name: "Datasets" });
    expect(screen.queryByRole("navigation", { name: "Main navigation" })).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Open the menu" }));
    const nav = await screen.findByRole("navigation", { name: "Main navigation" });
    fireEvent.click(within(nav).getByRole("link", { name: "Recipes" }));

    await waitFor(() =>
      expect(screen.queryByRole("navigation", { name: "Main navigation" })).not.toBeInTheDocument(),
    );
  });
});
