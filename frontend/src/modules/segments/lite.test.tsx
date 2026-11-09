import { screen } from "@testing-library/react";
import { forwardRef } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { column, roles } from "../../test/columns";
import { dataset, health, mockApi } from "../../test/mockApi";
import { renderAt } from "../../test/render";
import { plan, run, summary } from "../../test/segments";

vi.mock("../../lite/flag", () => ({ LITE: true, FULL_APP_URL: "https://example.test" }));
vi.mock("../../charts/Plot", () => ({
  Plot: forwardRef<HTMLDivElement, { label: string }>(function Plot({ label }, ref) {
    return <div ref={ref} role="img" aria-label={label} />;
  }),
}));

afterEach(() => vi.unstubAllGlobals());

describe("Segments in the browser version", () => {
  it("stays available, with t-SNE locked", async () => {
    mockApi({
      "GET /health": health,
      "GET /datasets": [dataset()],
      "GET /datasets/d1": dataset(),
      "GET /datasets/d1/columns": [column({ name: "income" })],
      "GET /column-roles": roles,
      "POST /modules/segments/datasets/d1/plan": plan,
      "GET /modules/segments/datasets/d1/runs": [summary],
      [`GET /modules/segments/datasets/d1/runs/${run.id}`]: run,
    });
    renderAt("/d/d1/segments");

    expect(await screen.findByRole("heading", { name: /3 segments/ })).toBeInTheDocument();
    expect(screen.getByText("t-SNE needs the full app")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Draw with t-SNE" })).not.toBeInTheDocument();
  });
});
