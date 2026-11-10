import { fireEvent, screen } from "@testing-library/react";
import { forwardRef } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { column, roles } from "../../test/columns";
import { dataset, health, mockApi } from "../../test/mockApi";
import { renderAt } from "../../test/render";

vi.mock("../../charts/Plot", () => ({
  Plot: forwardRef<HTMLDivElement, { label: string }>(function Plot({ label }, ref) {
    return <div ref={ref} role="img" aria-label={label} />;
  }),
}));

afterEach(() => vi.unstubAllGlobals());

const columns = [column({ name: "units", position: 0 }), column({ name: "revenue", position: 1 })];
const base = {
  "GET /health": health,
  "GET /datasets": [dataset()],
  "GET /datasets/d1": dataset(),
  "GET /datasets/d1/columns": columns,
  "GET /column-roles": roles,
};

describe("Ways into the Regression tab (068)", () => {
  it("Correlations › Focus models the variable on its strongest links", async () => {
    mockApi({
      ...base,
      "GET /datasets/d1/correlations": {
        matrix: {
          method: "pearson",
          columns: [
            { name: "units", type: "numeric_continuous", role: "num" },
            { name: "revenue", type: "numeric_continuous", role: "num" },
          ],
          values: [
            [1, 0.8],
            [0.8, 1],
          ],
          metrics: [
            [null, "pearson"],
            ["pearson", null],
          ],
          n: [
            [100, 100],
            [100, 100],
          ],
          clustered: [0, 1],
          rows: 100,
          total: 100,
          sampled: false,
          excluded: [],
          truncated: 0,
          pairs: [{ a: "units", b: "revenue", metric: "pearson", value: 0.8, p_value: 0.0001, n: 100, q_value: 0.0001, evidence: "strong" }],
          multiple: null,
        },
        insights: [],
      },
    });
    renderAt("/d/d1/correlations");
    fireEvent.click(await screen.findByRole("button", { name: "revenue" }));
    expect(screen.getByRole("link", { name: /Model revenue on its 5 strongest links/ })).toHaveAttribute(
      "href",
      "/d/d1/regression?target=revenue",
    );
  });

  it("Two variables adds controls to Y on X", async () => {
    mockApi({
      ...base,
      "GET /datasets/d1/compare": {
        kind: "scatter",
        x: "units",
        y: "revenue",
        z: null,
        color: null,
        roles: { units: "num", revenue: "num" },
        rows: 100,
        total: 100,
        sample: null,
        group_axis: null,
        points: { x: [1, 2, 3], y: [2, 4, 6], z: null, color: null },
        regression: { slope: 2, intercept: 0, r2: 0.25 },
        density: null,
        groups: [],
        contingency: null,
        series: [],
        step: null,
        tests: [],
        message: null,
        intervals: null,
        differences: [],
        multiple: null,
      },
    });
    renderAt("/d/d1/compare?x=units&y=revenue");
    expect(await screen.findByRole("link", { name: "Add controls →" })).toHaveAttribute(
      "href",
      "/d/d1/regression?target=revenue&columns=units&focus=units",
    );
  });
});
