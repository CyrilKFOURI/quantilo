import { screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { column, roles } from "../../test/columns";
import fixtures from "../../test/experiments.json";
import { dataset, health, mockApi } from "../../test/mockApi";
import { renderAt } from "../../test/render";

function setup(decisions: unknown) {
  return mockApi({
    "GET /health": health,
    "GET /datasets": [dataset()],
    "GET /datasets/d1": dataset(),
    "GET /datasets/d1/columns": [column({ name: "offer" })],
    "GET /column-roles": roles,
    "GET /modules/ml/datasets/d1/targets": { candidates: [], rows: 100 },
    "GET /modules/ml/catalog": { models: [], recommendations: [] },
    "GET /modules/experiments/datasets/d1/decisions": decisions,
  });
}

afterEach(() => vi.unstubAllGlobals());

describe("Logged decisions banner (055)", () => {
  it("points the ML page to the Policies tab when the data looks like logged decisions", async () => {
    setup(fixtures.decisions);
    renderAt("/d/d1/ml");

    expect(await screen.findByText("This looks like logged decisions (offer, converted) —")).toBeInTheDocument();
    const link = screen.getByRole("link", { name: "evaluate a policy →" });
    expect(link.getAttribute("href")).toContain("/d/d1/experiments?tab=policies");
  });

  it("says nothing otherwise", async () => {
    setup(fixtures.noDecisions);
    renderAt("/d/d1/ml");

    expect(await screen.findByRole("heading", { name: "Machine learning" })).toBeInTheDocument();
    expect(screen.queryByText(/logged decisions/)).toBeNull();
  });
});
