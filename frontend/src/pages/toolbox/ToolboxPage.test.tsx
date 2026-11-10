import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { availableTools, inToolbox, THEMES, TOOLS } from "../../features/toolbox/tools";
import { modules } from "../../modules/registry";
import { column, roles } from "../../test/columns";
import { dataset, health, mockApi } from "../../test/mockApi";
import { renderAt } from "../../test/render";

afterEach(() => vi.unstubAllGlobals());

function setup(columns = [column({ name: "day", type: "datetime", detected_type: "datetime", role: null }), column({ name: "units", position: 1 })]) {
  return mockApi({
    "GET /health": health,
    "GET /datasets": [dataset()],
    "GET /datasets/d1": dataset(),
    "GET /datasets/d1/columns": columns,
    "GET /column-roles": roles,
  });
}

describe("Toolbox tools (098)", () => {
  it("lists each tool once, under a known theme, on a registered route", () => {
    expect(new Set(TOOLS.map((tool) => tool.id)).size).toBe(TOOLS.length);
    for (const tool of TOOLS) {
      expect(THEMES).toContain(tool.theme);
      expect(tool.to.startsWith("/d/$datasetId/")).toBe(true);
      if (tool.module) expect(modules.some((module) => module.id === tool.module)).toBe(true);
    }
  });

  it("drops the tools of a missing module and locks those of a locked one", () => {
    const tools = availableTools([{ id: "timeseries", locked: true }]);
    expect(tools.find((tool) => tool.id === "timeseries")?.locked).toBe(true);
    expect(tools.some((tool) => tool.id === "abtest")).toBe(false);
    expect(tools.find((tool) => tool.id === "compare")?.locked).toBe(false);
  });

  it("knows the pages reached through the Toolbox only", () => {
    expect(inToolbox("/d/d1/toolbox")).toBe(true);
    expect(inToolbox("/d/d1/timeseries")).toBe(true);
    expect(inToolbox("/d/d1/experiments")).toBe(true);
    expect(inToolbox("/d/d1/compare")).toBe(false);
    expect(inToolbox("/d/d1/kpi")).toBe(false);
  });
});

describe("Toolbox page (098)", () => {
  it("classifies the tools under titles, each card saying what it answers and when", async () => {
    setup();
    renderAt("/d/d1/toolbox");

    expect(await screen.findByRole("heading", { level: 1, name: "Toolbox" })).toBeInTheDocument();
    const titles = within(screen.getByRole("main")).getAllByRole("heading", { level: 2 }).map((heading) => heading.textContent);
    expect(titles).toEqual(["Time & sequences", "Experiments & decisions", "Statistics", "Rows & segments"]);

    const experiments = screen.getByRole("region", { name: "Experiments & decisions" });
    const cards = within(experiments).getAllByRole("link");
    expect(cards.map((card) => card.querySelector(".font-medium")?.textContent)).toEqual(["A/B test", "Sample size", "Policies"]);
    expect(within(experiments).getByText("Did a variant change a metric, beyond chance?")).toBeInTheDocument();
    expect(cards[1]).toHaveAttribute("href", "/d/d1/experiments?tab=plan");

    const rows = screen.getByRole("region", { name: "Rows & segments" });
    expect(within(rows).getByRole("link", { name: /^Unusual rows/ })).toHaveAttribute("href", "/d/d1/segments?section=unusual");
  });

  it("greys out Time series without a date column, with the reason", async () => {
    setup([column({ name: "units" })]);
    renderAt("/d/d1/toolbox");

    const time = await screen.findByRole("region", { name: "Time & sequences" });
    await waitFor(() => expect(within(time).queryByRole("link")).not.toBeInTheDocument());
    expect(within(time).getByText("Needs a date column")).toBeInTheDocument();
  });

  it("marks a tool locked in the browser version", async () => {
    setup();
    renderAt(
      "/d/d1/toolbox",
      modules.map((module) => (module.id === "segments" ? { ...module, locked: true } : module)),
    );

    const rows = await screen.findByRole("region", { name: "Rows & segments" });
    expect(within(rows).getAllByLabelText("Needs the full app")).toHaveLength(2);
  });

  it("opens a tool empty from its card", async () => {
    setup();
    const router = renderAt("/d/d1/toolbox");

    fireEvent.click(await screen.findByRole("link", { name: /^Time series/ }));
    await waitFor(() => expect(router.state.location.pathname).toBe("/d/d1/timeseries"));
    expect(await screen.findByRole("heading", { name: "Pick a series" })).toBeInTheDocument();
  });

  it("finds every tool and the Toolbox from the command palette", async () => {
    setup();
    const router = renderAt("/d/d1/toolbox");
    await screen.findByRole("heading", { level: 1, name: "Toolbox" });

    fireEvent.keyDown(window, { key: "k", ctrlKey: true });
    const dialog = await screen.findByRole("dialog", { name: "Command palette" });
    const box = within(dialog).getByRole("combobox");
    for (const name of ["Time series", "A/B test", "Sample size", "Unusual rows", "Structure", "Toolbox", "KPI", "Synthetic data"]) {
      fireEvent.change(box, { target: { value: name } });
      expect(within(dialog).getAllByRole("option", { name: new RegExp(name) }).length).toBeGreaterThan(0);
    }
    fireEvent.change(box, { target: { value: "unusual" } });
    fireEvent.keyDown(box, { key: "Enter" });
    await waitFor(() => expect(router.state.location.pathname).toBe("/d/d1/segments"));
    expect(router.state.location.search).toMatchObject({ section: "unusual" });
  });
});
