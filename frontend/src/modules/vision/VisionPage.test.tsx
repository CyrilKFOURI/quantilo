import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from "vitest";

import { dataset, health, mockApi } from "../../test/mockApi";
import { renderAt } from "../../test/render";
import { imageIndex, imageReport, qualityRules } from "./testData";

// jsdom lays nothing out: give elements a size so the virtualized grid shows its rows.
const sizes = ["offsetHeight", "offsetWidth", "clientHeight", "clientWidth"] as const;
const saved = sizes.map((key) => Object.getOwnPropertyDescriptor(HTMLElement.prototype, key));
beforeAll(() => {
  for (const key of sizes) {
    Object.defineProperty(HTMLElement.prototype, key, { configurable: true, get: () => (key.endsWith("Height") ? 900 : 1200) });
  }
});
afterAll(() => {
  sizes.forEach((key, index) => {
    const descriptor = saved[index];
    if (descriptor) Object.defineProperty(HTMLElement.prototype, key, descriptor);
  });
});
afterEach(() => vi.unstubAllGlobals());

function setup(decisions = { draft: {}, applied: {}, pending: 0, version: 1 }) {
  return mockApi({
    "GET /health": health,
    "GET /datasets": [dataset({ format: "images", name: "pets" })],
    "GET /datasets/d1": dataset({ format: "images", name: "pets" }),
    "GET /datasets/d1/columns": [],
    "GET /datasets/d1/images/index": imageIndex(),
    "GET /datasets/d1/images/report": imageReport,
    "GET /images/rules": qualityRules,
    "GET /datasets/d1/images/decisions": decisions,
    "PUT /datasets/d1/images/decisions": (body: unknown) => {
      const patch = (body as { decisions: Record<string, string | null> }).decisions;
      const draft = Object.fromEntries(Object.entries(patch).filter(([, value]) => value !== null));
      return { draft, applied: {}, pending: Object.keys(draft).length, version: 1 };
    },
    "POST /datasets/d1/images/decisions/apply": { version: 2, excluded: 3, decided: 1 },
  });
}

describe("Vision page (107, 108)", () => {
  it("shows the set's figures, the tabs to come and every image", async () => {
    setup();
    renderAt("/d/d1/vision?stats=false");
    expect(await screen.findByRole("heading", { name: "Vision" })).toBeInTheDocument();
    const grid = await screen.findByRole("listbox", { name: "6 images" });
    expect(within(grid).getAllByRole("option")).toHaveLength(6);
    expect(within(grid).getByRole("option", { name: "blur.png, cat, R-VQ-04" })).toBeInTheDocument();
    // the second exact copy is left out by its rule, said on the tile
    expect(within(grid).getByText("Excluded · R-VQ-09")).toBeInTheDocument();
    const map = screen.getByRole("tab", { name: /Map/ });
    expect(map).toHaveAttribute("aria-disabled", "true");
    expect(map).toHaveTextContent("Coming");
    expect(screen.getAllByText("Left out of models").length).toBeGreaterThan(0);
    // the sidebar lists Vision for an image set
    expect(screen.getAllByRole("link", { name: "Vision" }).length).toBeGreaterThan(0);
  });

  it("filters by a rule's chip and selects with Shift", async () => {
    setup();
    const router = renderAt("/d/d1/vision?stats=false");
    fireEvent.click(await screen.findByRole("button", { name: /^Blurry/ }));
    await waitFor(() => expect(screen.getByRole("listbox", { name: "1 images" })).toBeInTheDocument());
    expect(router.state.location.search).toMatchObject({ quality: "R-VQ-04" });
    fireEvent.click(screen.getByRole("button", { name: "Clear all" }));
    const grid = await screen.findByRole("listbox", { name: "6 images" });
    const options = within(grid).getAllByRole("option");
    fireEvent.click(options[0] as HTMLElement);
    fireEvent.click(options[2] as HTMLElement, { shiftKey: true });
    expect(screen.getByText("3 selected")).toBeInTheDocument();
    expect(options.filter((option) => option.getAttribute("aria-selected") === "true")).toHaveLength(3);
  });

  it("opens an image's sheet with its measures, rules and copies, and goes to the next", async () => {
    setup();
    const router = renderAt("/d/d1/vision?stats=false");
    const grid = await screen.findByRole("listbox", { name: "6 images" });
    fireEvent.doubleClick(within(grid).getByRole("option", { name: "0.png, cat, R-VQ-09" }));
    const sheet = await screen.findByRole("dialog");
    expect(within(sheet).getByText("200 × 200 PNG, 7.8 KB")).toBeInTheDocument();
    expect(within(sheet).getByText("Exact copies")).toBeInTheDocument();
    expect(within(sheet).getByText("Exact duplicate")).toBeInTheDocument();
    expect(within(sheet).getByText("Used by models.")).toBeInTheDocument();
    expect(within(sheet).getByText("1 of 6")).toBeInTheDocument();
    expect(router.state.location.search).toMatchObject({ image: "cat/0.png" });
    fireEvent.keyDown(window, { key: "ArrowRight" });
    await waitFor(() => expect(router.state.location.search).toMatchObject({ image: "cat/copy.png" }));
    expect(await within(screen.getByRole("dialog")).findByText(/Left out of models by R-VQ-09/)).toBeInTheDocument();
  });

  it("never shows a location, only that it is not read", async () => {
    setup();
    renderAt("/d/d1/vision?stats=false&image=%22dog%2F1.png%22");
    const sheet = await screen.findByRole("dialog");
    expect(within(sheet).getByText(/never read nor shown \(R-IMG-05\)/)).toBeInTheDocument();
  });

  it("lists the quality findings and saves decisions as a version (108)", async () => {
    const calls = setup();
    renderAt("/d/d1/vision?tab=quality");
    expect(await screen.findByText("4 images with an issue (R-VQ-01, R-VQ-04, R-VQ-09).")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: /R-VQ-04\s*Blurry/ })).toBeInTheDocument();
    expect(screen.getByText(/blur_sharpness 100/)).toBeInTheDocument();
    expect(screen.getByText("Rules that found nothing (1)")).toBeInTheDocument();
    // keep the copy, leave the first out: one request
    const strips = screen.getAllByRole("button", { name: "Keep this one" });
    fireEvent.click(strips[1] as HTMLElement);
    await waitFor(() => expect(calls.some((call) => call.key === "PUT /datasets/d1/images/decisions")).toBe(true));
    const put = calls.find((call) => call.key === "PUT /datasets/d1/images/decisions")?.body as { decisions: Record<string, string> };
    expect(put.decisions).toEqual({ "cat/0.png": "exclude", "cat/copy.png": "keep" });
    fireEvent.click(await screen.findByRole("button", { name: "Save as version 2" }));
    expect(await screen.findByText(/Saved as version 2: 3 images left out of models/)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "See what changed" })).toBeInTheDocument();
  });

  it("is not listed for a table", async () => {
    mockApi({
      "GET /health": health,
      "GET /datasets": [dataset()],
      "GET /datasets/d1": dataset(),
      "GET /datasets/d1/columns": [],
      "GET /datasets/d1/overview": { error: "skip" },
    });
    renderAt("/d/d1/vision");
    expect(await screen.findByText(/This dataset is a table, not an image set/)).toBeInTheDocument();
    expect(screen.queryByRole("link", { name: "Vision" })).not.toBeInTheDocument();
  });
});
