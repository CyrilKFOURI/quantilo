import { fireEvent, screen, waitFor, within } from "@testing-library/react";
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

const columns = [
  column({ name: "Sales", position: 0 }),
  column({ name: "Price", position: 1, role: null }),
  column({ name: "Cost", position: 2, role: null }),
  column({ name: "Store", position: 3, type: "numeric_discrete", detected_type: "numeric_discrete", role: null }),
];

const coefficient = (term: string, estimate: number) => ({
  term,
  column: term,
  level: null,
  estimate,
  std_error: 0.1,
  ci_low: estimate - 0.2,
  ci_high: estimate + 0.2,
  statistic: 20,
  p_value: 1e-6,
  q_value: term === "(Intercept)" ? null : 1e-6,
  evidence: term === "(Intercept)" ? "none" : "strong",
  standardized: term === "(Intercept)" ? null : 0.5,
  std_low: null,
  std_high: null,
  odds_ratio: null,
  marginal: null,
  marginal_low: null,
  marginal_high: null,
  sentence: `Each extra unit of \`${term}\` adds ${estimate}.`,
});

const regression = {
  target: "Sales",
  positive: null,
  model: "ols",
  model_label: "Least squares (OLS)",
  model_reason: "A number to explain.",
  columns: ["Price"],
  focus: null,
  terms: [coefficient("(Intercept)", 1), coefficient("Price", 2.6)],
  se: { kind: "cluster", label: "Clustered", rule: "R-REG-SE-01", reason: "clustered by `Store`.", lag: null, groups: ["Store"], clusters: 40, warning: null },
  weights: null,
  quality: { rows: 800, loaded: 800, total: 800, sampled: false, missing: [], r2: 0.6, adj_r2: 0.6, oos_r2: 0.59, oos_how: "5 folds by `Store`", aic: 1, bic: 1, f: 10, f_p: 0, df_model: 1, df_resid: 798, pseudo_r2: null },
  multiple: { method: "none", label: "None", tests: 1, significant: 1, kept: 1, alpha: 0.05, auto: true, reason: "", sentence: "One test." },
  sentence: "Each extra unit of `Price` adds 2.6. One test.",
  checks: [
    {
      key: "panel",
      status: "info",
      title: "Panel data",
      sentence: "`Price` changes within a `Store`: fixed effects compare each `Store` with itself.",
      rule: "R-REG-14",
      action: "panel",
      column: null,
    },
  ],
  references: [],
  left_out: [],
  structure: { kind: "panel", date: null, entity: "Store", period: "Date", periods: 20, entities: 40, message: "Panel data: 40 `Store` seen over 20 `Date`." },
  hints: [],
  plots: null,
};

const pc = (estimate: number | null) => ({ estimate, std_error: estimate === null ? null : 0.1, ci_low: estimate === null ? null : estimate - 0.2, ci_high: estimate === null ? null : estimate + 0.2, p_value: estimate === null ? null : 0.001 });
const estimator = (key: string, label: string, values: (number | null)[], available = true) => ({
  key,
  label,
  available,
  reason: available ? `${label}: what it compares.` : "Several rows share a `Store` and a `Date`.",
  coefficients: available ? values.map(pc) : [],
  dropped: [],
  rows: 800,
  groups: 40,
  r2: 0.3,
  r2_kind: "within R²",
  se_label: "Clustered by entity",
  theta: key === "re" ? 0.9 : null,
});

const panel = {
  available: true,
  reason: "",
  entity: "Store",
  period: "Date",
  entities: 40,
  periods: 20,
  balanced: true,
  terms: ["(Intercept)", "Price"],
  varying: ["Price"],
  fixed: [],
  estimators: [
    estimator("pooled", "Pooled OLS", [1, 2.6]),
    estimator("entity", "Fixed effects (entity)", [null, 2.01]),
    estimator("twoway", "Fixed effects (two-way)", [null, 2.02]),
    estimator("fd", "First differences", [], false),
    estimator("re", "Random effects (GLS)", [1, 2.3]),
  ],
  recommended: "entity",
  recommended_rule: "R-REG-PAN-01",
  recommended_reason: "Random effects would be biased: the `Store` effect is tied to the columns (Mundlak p < 0.001). Fixed effects by `Store`.",
  tests: [{ key: "mundlak", title: "Mundlak (robust Hausman)", statistic: 30, df: 1, df2: 39, p_value: 0.0001, sentence: "The `Store` effect is tied to the columns (Mundlak p < 0.001).", rule: "R-REG-PAN-01" }],
  sentence: "Fixed effects (entity) recommended.",
  notes: [],
};

const iv = {
  target: "Sales",
  endogenous: ["Price"],
  instruments: ["Cost"],
  instrument_terms: ["Cost"],
  rows: 800,
  se: { kind: "cluster", label: "Clustered", rule: "R-REG-SE-01", reason: "clustered by `Store`.", lag: null, groups: ["Store"], clusters: 40, warning: null },
  coefficients: [
    { term: "(Intercept)", column: "(Intercept)", endogenous: false, estimate: 1, std_error: 0.1, ci_low: 0.8, ci_high: 1.2, p_value: 0.001, ols: 1.1, ols_low: 0.9, ols_high: 1.3 },
    { term: "Price", column: "Price", endogenous: true, estimate: 2.05, std_error: 0.1, ci_low: 1.85, ci_high: 2.25, p_value: 0.001, ols: 2.6, ols_low: 2.5, ols_high: 2.7 },
  ],
  first_stages: [{ term: "Price", f: 80, classic_f: 85, effective_f: 80, partial_r2: 0.2, df1: 1, df2: 39, p_value: 0, status: "ok", sentence: "Instruments for `Price`: effective F = 80 (strong)." }],
  tests: [
    { key: "endogeneity", title: "Endogeneity (Wu–Hausman)", statistic: 20, df: 1, df2: 39, p_value: 0.0001, status: "warn", sentence: "`Price` is endogenous: least squares is biased; keep IV.", rule: "R-REG-IV-05" },
  ],
  reduced_form: [],
  left_out: [],
  missing: [],
  sentence: "Instrumenting `Price` with `Cost`: each extra unit of `Price` moves `Sales` by 2.05.",
  notes: [],
};

const more = { white: null, white_sentence: "White's test.", removed: 0, shifts: [], shift_sentence: "No influential row to leave out." };

function routes() {
  return {
    "GET /health": health,
    "GET /datasets": [dataset()],
    "GET /datasets/d1": dataset(),
    "GET /datasets/d1/columns": columns,
    "GET /column-roles": roles,
    "GET /datasets/d1/regression": regression,
    "GET /datasets/d1/regression/panel": panel,
    "GET /datasets/d1/regression/more": more,
    "GET /datasets/d1/regression/instruments": iv,
  };
}

describe("Regression: panel and instruments (071)", () => {
  it("opens Panel from the R-REG-14 check, the recommended estimator preselected with its reason", async () => {
    const calls = mockApi(routes());
    renderAt("/d/d1/regression?target=Sales&columns=Price");
    await screen.findByText(/compare each/);
    expect(calls.some((call) => call.key === "GET /datasets/d1/regression/panel")).toBe(false);
    window.HTMLElement.prototype.scrollIntoView = vi.fn();

    fireEvent.click(screen.getByRole("button", { name: "Try…" }));
    const select = await screen.findByRole("combobox", { name: "Estimator" });
    expect(select).toHaveValue("entity");
    expect(within(select).getByRole("option", { name: "Fixed effects (entity) (recommended)" })).toBeInTheDocument();
    expect(within(select).queryByRole("option", { name: /First differences/ })).toBeNull();
    expect(screen.getAllByText(/Random effects would be biased/).length).toBeGreaterThan(0);
    expect(screen.getByText(/Several rows share a/)).toBeInTheDocument();

    fireEvent.change(select, { target: { value: "re" } });
    expect(screen.getByRole("button", { name: "Reset to auto: Estimator" })).toBeInTheDocument();
    expect(screen.getByText(/θ = 0.90/)).toBeInTheDocument();
  });

  it("asks for IV only once the endogenous column and an instrument are named", async () => {
    const calls = mockApi(routes());
    renderAt("/d/d1/regression?target=Sales&columns=Price");
    await screen.findByText(/compare each/);
    fireEvent.click(screen.getByRole("button", { name: /^Advanced/ }));
    fireEvent.click(screen.getByRole("button", { name: /^More/ }));
    expect(await screen.findByText(/nothing is computed before/)).toBeInTheDocument();
    expect(calls.some((call) => call.key === "GET /datasets/d1/regression/instruments")).toBe(false);

    fireEvent.change(screen.getByRole("combobox", { name: "Endogenous column" }), { target: { value: "Price" } });
    expect(calls.some((call) => call.key === "GET /datasets/d1/regression/instruments")).toBe(false);
    const add = screen.getByRole("combobox", { name: "Add an instrument…" });
    expect(within(add).queryByRole("option", { name: "Price" })).toBeNull(); // already in the model
    fireEvent.change(add, { target: { value: "Cost" } });

    expect(await screen.findByText(/Instrumenting/)).toBeInTheDocument();
    const call = calls.find((c) => c.key === "GET /datasets/d1/regression/instruments");
    expect(call?.params.get("endogenous")).toBe("Price");
    expect(call?.params.get("instruments")).toBe("Cost");
    expect(screen.getByText(/effective F = 80/)).toBeInTheDocument();
    await waitFor(() => expect(screen.getByRole("button", { name: "Remove Cost" })).toBeInTheDocument());
  });
});
